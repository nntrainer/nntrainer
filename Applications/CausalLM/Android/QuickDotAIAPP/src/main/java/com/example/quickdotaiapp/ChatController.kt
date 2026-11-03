// SPDX-License-Identifier: Apache-2.0
/**
 * @file  ChatController.kt
 * @brief All engine logic behind the Compose screens: one StateFlow, one
 *        engine at a time, every engine call on [engineDispatcher].
 *        Pure Kotlin so the JVM tests drive it with a fake engine.
 */
package com.example.quickdotaiapp

import com.example.quickdotai.BackendResult
import com.example.quickdotai.BackendType
import com.example.quickdotai.Capability
import com.example.quickdotai.LoadModelRequest
import com.example.quickdotai.ModelDescriptor
import com.example.quickdotai.PerformanceMetrics
import com.example.quickdotai.PromptPart
import com.example.quickdotai.QuantizationType
import com.example.quickdotai.QuickAiChatMessage
import com.example.quickdotai.QuickAiChatRole
import com.example.quickdotai.QuickAiChatSamplingConfig
import com.example.quickdotai.QuickAiChatSessionConfig
import com.example.quickdotai.QuickAiChatTemplateKwargs
import com.example.quickdotai.QuickAiError
import com.example.quickdotai.QuickDotAI
import com.example.quickdotai.RuntimeKind
import com.example.quickdotai.StreamSink
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong

class ChatController(
    private val scope: CoroutineScope,
    private val engineDispatcher: CoroutineDispatcher,
    private val catalogProvider: suspend () -> List<ModelDescriptor>,
    private val engineFactory: (ModelDescriptor, String) -> QuickDotAI,
    private val nativeLibDir: String?,
    private val examples: Examples,
    private val pathExists: (String) -> Boolean = { File(it).exists() },
    private val now: () -> Long = System::currentTimeMillis,
    initialTheme: ThemeMode = ThemeMode.SYSTEM,
    private val publishIntervalMs: Long = 50,
) {
    private val _state = MutableStateFlow(UiState(theme = initialTheme))
    val state: StateFlow<UiState> = _state.asStateFlow()

    /** Written only on [engineDispatcher]; read from the main thread by cancel(). */
    @Volatile private var engine: QuickDotAI? = null
    @Volatile private var cancelRequested = false
    // Turn ids are LazyColumn keys: unique across threads, never minted inside update {}.
    private val nextTurnId = AtomicLong(1)
    private val runIndex = AtomicInteger(0)

    init {
        scope.launch {
            val catalog = withContext(engineDispatcher) { catalogProvider() } // dlopen off the main thread
            val initial = catalog.firstOrNull()
            update {
                copy(
                    catalog = catalog,
                    selection = initial?.let {
                        selection.copy(family = it.family, runtime = it.runtime, backend = it.backends.first())
                    } ?: selection,
                )
            }
            seedDefaults()
        }
    }

    // ----- selection -------------------------------------------------------

    fun selectFamily(family: String) {
        val s = state.value
        val runtime = s.runtimesFor(family).firstOrNull() ?: s.selection.runtime
        val backend = s.backendsFor(family, runtime).firstOrNull() ?: s.selection.backend
        applySelection(s.selection.copy(family = family, runtime = runtime, backend = backend, sdEnabled = false))
    }

    fun selectRuntime(runtime: RuntimeKind) {
        val s = state.value
        val backend = s.backendsFor(s.selection.family, runtime).firstOrNull() ?: s.selection.backend
        applySelection(s.selection.copy(runtime = runtime, backend = backend, sdEnabled = false))
    }

    fun selectBackend(backend: BackendType) =
        applySelection(state.value.selection.copy(backend = backend, sdEnabled = false))

    fun setSdEnabled(enabled: Boolean) = update { copy(selection = selection.copy(sdEnabled = enabled)) }
    fun setBasePath(path: String) = update { copy(selection = selection.copy(basePath = path)) }

    private fun applySelection(sel: ModelSelection) {
        update { copy(selection = sel) }
        seedDefaults()
    }

    /** Like the View sample: a model pick resets the Raw example and leaves the draft alone. */
    private fun seedDefaults() {
        update { copy(rawJson = examples.rawJsonFor(state.value.descriptor), rawRevision = rawRevision + 1) }
    }

    // ----- plain UI state --------------------------------------------------

    fun setDraft(text: String) = update { copy(draft = text) }
    fun addDraftImages(images: List<ByteArray>) = update { copy(draftImages = draftImages + images) }
    fun removeDraftImage(index: Int) = update { copy(draftImages = draftImages.filterIndexed { i, _ -> i != index }) }
    fun setRawJson(text: String) = update { copy(rawJson = text) }
    fun resetRawJson() = update { copy(rawJson = examples.rawJsonFor(descriptor), rawRevision = rawRevision + 1) }
    fun clearRuns() = update { copy(runs = emptyList(), lastMetrics = null) }
    fun setScreen(screen: Screen) = update { copy(screen = screen) }
    fun setRawMode(on: Boolean) = update { copy(rawMode = on) }
    fun setTheme(theme: ThemeMode) = update { copy(theme = theme) }
    fun splashDone() = update { copy(splashShown = true) }
    fun consumeNotice() = update { copy(notice = null) }
    fun notify(text: String) = update { copy(notice = text) }

    // ----- model lifecycle -------------------------------------------------

    fun load(force: Boolean = false) {
        scope.launch { loadInternal(buildRequest(), force) }
    }

    fun unload() {
        scope.launch {
            val e = engine ?: run { setStatus("Nothing to unload."); return@launch }
            if (state.value.streaming) { cancelRequested = true; e.cancel(); e.chatCancel() }
            setStatus("Unloading…")
            withContext(engineDispatcher) {
                runCatching { if (e.chatSessionId != null) e.closeChatSession() }
                runCatching { e.unload() }
                runCatching { e.close() }
                engine = null
            }
            update { copy(load = LoadState.Idle, status = "Unloaded.") }
        }
    }

    /** For the ViewModel's onCleared: hands the engine back so the caller can close it on its executor. */
    fun shutdown(): QuickDotAI? {
        val e = engine
        engine = null
        e?.cancel()
        return e
    }

    internal fun buildRequest(): LoadModelRequest {
        val s = state.value
        val d = s.descriptor
        val base = s.selection.basePath.trim().ifEmpty { DEFAULT_BASE_PATH }
        val id = if (s.selection.sdEnabled) (d?.sdVariantId ?: d?.id) else d?.id
        return LoadModelRequest(
            backend = s.selection.backend,
            modelId = id ?: s.selection.family,
            quantization = QuantizationType.W4A32,
            modelPath = d?.let { Examples.modelPath(it, base) },
            visionBackend = d?.takeIf { Capability.MULTIMODAL in it.capabilities }?.let { s.selection.backend },
            nativeLibDir = nativeLibDir,
            modelBasePath = base,
            useSpeculativeDecoding = s.selection.sdEnabled,
        )
    }

    /** Returns the engine for [req], loading or replacing as needed; null and a status on failure. */
    internal suspend fun loadInternal(req: LoadModelRequest, force: Boolean): QuickDotAI? {
        val d = state.value.descriptor ?: run { setStatus("Load failed: no model selected"); return null }
        val current = engine
        if (current != null && !force && state.value.loadedRequest == req) return current
        val required = Examples.requiredPath(d, req.modelBasePath ?: DEFAULT_BASE_PATH)
        if (!pathExists(required)) {
            // Checked before anything is torn down: a typo must not cost the loaded model.
            update { copy(status = "Load failed: not found: $required", notice = "Not found: $required") }
            return null
        }
        if (current != null && state.value.streaming) {
            // Spec §5: cancel first; the run finishes (its partial turn marked stopped) before the swap.
            cancelRequested = true
            current.cancel()
            current.chatCancel()
        }
        update { copy(load = LoadState.Loading(req.modelKey), status = "Loading ${req.modelKey}…") }
        return withContext(engineDispatcher) {
            current?.let { runCatching { it.close() } }
            engine = null
            val e = engineFactory(d, req.modelBasePath ?: DEFAULT_BASE_PATH)
            when (val r = e.load(req)) {
                is BackendResult.Ok -> {
                    engine = e
                    update {
                        copy(
                            load = LoadState.Loaded(req, e.kind, e.architecture),
                            status = "Loaded ${req.modelKey} (${e.kind}, arch=${e.architecture ?: "?"})",
                            turns = if (current != null) turns + ChatTurn(nextId(), TurnRole.NOTE, "Loaded ${req.modelKey}") else turns,
                        )
                    }
                    e
                }
                is BackendResult.Err -> {
                    runCatching { e.close() }
                    update { copy(load = LoadState.Idle, status = "Load failed: [${r.error.name}] ${r.message ?: ""}") }
                    null
                }
            }
        }
    }

    // ----- generation ------------------------------------------------------

    fun send() {
        val s = state.value
        val text = s.draft.trim()
        if (text.isEmpty() && s.draftImages.isEmpty()) { setStatus("Message is empty."); return }
        if (s.busy) { setStatus("Busy."); return }
        val images = s.draftImages
        val sent = s.draft
        val onStarted = {
            update {
                copy(
                    draft = if (draft == sent) "" else draft,
                    draftRevision = if (draft == sent) draftRevision + 1 else draftRevision,
                    draftImages = if (draftImages == images) emptyList() else draftImages,
                )
            }
        }
        runGeneration(ChatTurn(nextId(), TurnRole.USER, text, images), mode = "chat", onStarted) { e, d, sink ->
            if (d.runtime == RuntimeKind.LITERT) sendLiteRt(e, text, images, sink)
            else sendNative(e, sink)
        }
    }

    fun sendRaw() {
        val s = state.value
        val text = s.rawJson.trim()
        if (text.isEmpty()) { setStatus("Messages JSON is empty."); return }
        val parsed = RawRequest.parse(text)
        if (parsed is RawRequest.Parsed.Invalid) { setStatus("Invalid JSON: ${parsed.reason}"); return }
        if (s.busy) { setStatus("Busy."); return }
        val images = s.draftImages
        val onStarted = { update { copy(draftImages = if (draftImages == images) emptyList() else draftImages) } }
        runGeneration(ChatTurn(nextId(), TurnRole.USER, text, images, raw = true), mode = "raw", onStarted) { e, d, sink ->
            // The flat calls below need LiteRT's own conversation, which an open chat session replaces.
            if (e.chatSessionId != null) e.closeChatSession()
            when {
                images.isNotEmpty() && parsed is RawRequest.Parsed.Messages ->
                    e.runMultimodalHandleWithMessagesStreaming(RawRequest.withImages(parsed.messages, images), sink)
                parsed is RawRequest.Parsed.Messages &&
                    (Capability.MESSAGES_API in d.capabilities || d.runtime == RuntimeKind.LITERT) ->
                    e.runModelHandleWithMessagesStreaming(parsed.messages, sink)
                else -> e.runModelHandleWithJsonStreaming(text, sink)   // the only path that forwards tools
            }
        }
    }

    /** From the main thread: never waits for the engine. */
    fun cancel() {
        val e = engine ?: return
        if (!state.value.streaming) return
        cancelRequested = true
        // Status first: once the engine is signalled it may finish and write "Done." at once.
        // LiteRT-LM only flags its callback: tokens stop showing, but the engine runs the answer
        // to its end and an Unload or Reload queues behind it, so say so.
        setStatus(if (e.chatSessionId != null) "Stopping… LiteRT finishes the current answer first" else "Stopping…")
        e.cancel()
        e.chatCancel()
    }

    fun clearConversation() {
        update { copy(turns = emptyList()) }
        val e = engine ?: return
        scope.launch {
            withContext(engineDispatcher) { if (e.chatSessionId != null) runCatching { e.chatRebuild(emptyList()) } }
        }
    }

    /** LiteRT reads the session config when the session opens, so a change closes the current one. */
    fun setSession(settings: SessionSettings) {
        update { copy(session = settings) }
        val e = engine ?: return
        scope.launch {
            withContext(engineDispatcher) { if (e.chatSessionId != null) runCatching { e.closeChatSession() } }
        }
    }

    fun insertTextFile(name: String, text: String) {
        val body = text.removePrefix("﻿")
        update {
            copy(
                draft = if (draft.isEmpty()) body else draft + body,
                draftRevision = draftRevision + 1,
                notice = "$name inserted into the message (${body.length} chars)",
            )
        }
    }

    fun loadJsonFile(name: String, text: String) {
        when (val p = RawRequest.parse(text)) {
            is RawRequest.Parsed.Messages ->
                update {
                    copy(rawJson = text, rawRevision = rawRevision + 1, rawMode = true, screen = Screen.CHAT,
                        notice = "$name loaded as JSON (${p.messages.size} messages)")
                }
            is RawRequest.Parsed.Other -> insertTextFile(name, text)
            is RawRequest.Parsed.Invalid -> update { copy(notice = "$name is not valid JSON: ${p.reason}") }
        }
    }

    private fun runGeneration(
        userTurn: ChatTurn,
        mode: String,
        onStarted: () -> Unit,
        invoke: (QuickDotAI, ModelDescriptor, StreamSink) -> BackendResult<*>,
    ) {
        scope.launch {
            // A failed load returns here with the draft and images untouched.
            val e = loadInternal(buildRequest(), force = false) ?: return@launch
            val d = state.value.descriptor ?: return@launch
            onStarted()
            val assistantId = nextId()
            cancelRequested = false
            update {
                copy(
                    turns = turns + userTurn + ChatTurn(assistantId, TurnRole.ASSISTANT, "", streaming = true, raw = userTurn.raw),
                    streaming = true, status = "Generating…", liveTokens = 0, liveElapsedMs = 0,
                )
            }
            val startedMs = now()
            val collector = StreamCollector(assistantId, startedMs)
            val result: BackendResult<*> = withContext(engineDispatcher) {
                try { invoke(e, d, collector) } catch (t: Throwable) { BackendResult.Err(QuickAiError.INFERENCE_FAILED, t.message) }
            }
            collector.flush()
            val metrics = withContext(engineDispatcher) { (e.metrics() as? BackendResult.Ok)?.value }
            finishTurn(e, assistantId, result, collector.error, metrics, mode, startedMs, d)
        }
    }

    private fun sendNative(e: QuickDotAI, sink: StreamSink): BackendResult<Unit> {
        val history = state.value.turns.filter { !it.raw && !it.streaming && it.role != TurnRole.NOTE }
        val lastUser = history.indexOfLast { it.role == TurnRole.USER }
        val messages = history.mapIndexed { i, t ->
            val role = if (t.role == TurnRole.USER) QuickAiChatRole.USER else QuickAiChatRole.ASSISTANT
            val text = if (t.role == TurnRole.ASSISTANT) t.text.removeSuffix("<turn|>").trimEnd() else t.text
            // Earlier turns' images are not resent: the vision path takes the current turn's only.
            val images = if (i == lastUser) t.images.map { PromptPart.ImageBytes(it) } else emptyList()
            QuickAiChatMessage(role, images + listOf(PromptPart.Text(text)))
        }
        val hasImages = messages.lastOrNull()?.parts?.any { it is PromptPart.ImageBytes } == true
        return if (hasImages) e.runMultimodalHandleWithMessagesStreaming(messages, sink)
        else e.runModelHandleWithMessagesStreaming(messages, sink)
    }

    private fun sendLiteRt(e: QuickDotAI, text: String, images: List<ByteArray>, sink: StreamSink): BackendResult<*> {
        if (e.chatSessionId == null) {
            val opened = e.openChatSession(sessionConfig())
            if (opened is BackendResult.Err) return opened
        }
        return if (images.isEmpty()) e.runChatModelHandleStreaming(text, sink)
        else e.runChatMultimodalHandleStreaming(images.map { PromptPart.ImageBytes(it) } + PromptPart.Text(text), sink)
    }

    private fun sessionConfig(): QuickAiChatSessionConfig? {
        val s = state.value.session
        val system = s.systemPrompt.trim().ifEmpty { null }
        val temperature = s.temperature.trim().toDoubleOrNull()
        val topK = s.topK.trim().toIntOrNull()
        val topP = s.topP.trim().toDoubleOrNull()
        val seed = s.seed.trim().toIntOrNull()
        val sampling = if (temperature != null || topK != null || topP != null || seed != null)
            QuickAiChatSamplingConfig(temperature = temperature, topK = topK, topP = topP, seed = seed) else null
        val kwargs = when (s.thinking) {
            ThinkingChoice.ON -> QuickAiChatTemplateKwargs(enableThinking = true)
            ThinkingChoice.OFF -> QuickAiChatTemplateKwargs(enableThinking = false)
            ThinkingChoice.DEFAULT -> null
        }
        return if (system != null || sampling != null || kwargs != null)
            QuickAiChatSessionConfig(systemInstruction = system, sampling = sampling, chatTemplateKwargs = kwargs) else null
    }

    private fun finishTurn(
        ran: QuickDotAI, turnId: Long, result: BackendResult<*>, sinkError: String?, metrics: PerformanceMetrics?,
        mode: String, startedMs: Long, d: ModelDescriptor,
    ) {
        val stopped = cancelRequested
        cancelRequested = false
        val error = (result as? BackendResult.Err)?.let { "[${it.error.name}] ${it.message ?: ""}" } ?: sinkError
        val wallMs = (now() - startedMs).toDouble()
        val turnMetrics = metrics?.let {
            TurnMetrics(
                ttftMs = it.ttftMs,
                tokensPerSec = if (it.generationDurationMs > 0) it.generationTokens * 1000.0 / it.generationDurationMs else 0.0,
                generationTokens = it.generationTokens,
                prefillTokens = it.prefillTokens,
                totalMs = if (it.totalDurationMs > 0) it.totalDurationMs else wallMs,
                engineLabel = Examples.runtimeLabel(d.runtime),
            )
        }
        val note = error?.let { ChatTurn(nextId(), TurnRole.NOTE, "Error: $it") }
        val index = runIndex.incrementAndGet()
        val replaced = engine !== ran
        update {
            val text = turns.firstOrNull { it.id == turnId }?.text ?: ""
            copy(
                turns = turns.map { if (it.id == turnId) it.copy(streaming = false, stopped = stopped, metrics = turnMetrics) else it } + listOfNotNull(note),
                streaming = false,
                // An unload or reload that raced the run already wrote its status; keep it.
                status = when { error != null -> "Failed: $error"; replaced -> status; else -> "Done." },
                lastMetrics = metrics ?: lastMetrics,
                runs = (runs + RunRecord(
                    index = index, modelId = d.id, mode = mode,
                    prefillTokens = metrics?.prefillTokens ?: 0, generationTokens = metrics?.generationTokens ?: 0,
                    ttftMs = metrics?.ttftMs ?: 0.0, tokensPerSec = turnMetrics?.tokensPerSec ?: 0.0,
                    totalMs = wallMs, output = text,
                )).takeLast(20),
            )
        }
    }

    /** Accumulates deltas on the engine thread and publishes at most every [publishIntervalMs]. */
    private inner class StreamCollector(private val turnId: Long, private val startedMs: Long) : StreamSink {
        private val buffer = StringBuilder()
        private var tokens = 0
        private var lastPublishMs = startedMs
        @Volatile var error: String? = null

        @Synchronized override fun onDelta(text: String) {
            buffer.append(text)
            tokens++
            val t = now()
            if (t - lastPublishMs >= publishIntervalMs) publish(t)
        }
        override fun onDone() {}
        override fun onError(error: QuickAiError, message: String?) { this.error = "[${error.name}] ${message ?: ""}" }
        @Synchronized fun flush() = publish(now())

        private fun publish(t: Long) {
            lastPublishMs = t
            val snapshot = buffer.toString()
            val n = tokens
            update {
                copy(turns = turns.map { if (it.id == turnId) it.copy(text = snapshot) else it },
                    liveTokens = n, liveElapsedMs = t - startedMs)
            }
        }
    }

    // ----- helpers ---------------------------------------------------------

    internal fun nextId(): Long = nextTurnId.getAndIncrement()
    internal fun setStatus(text: String) = update { copy(status = text) }
    private fun update(block: UiState.() -> UiState) = _state.update(block)
}
