// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    LiteRTLmChatSession.kt
 * @brief   Chat session helper backed by a LiteRT-LM Conversation.
 *
 * Each session owns a LiteRT-LM [Conversation] and an [ImageStore] that
 * caches image bytes keyed by SHA-256 hash. When the same image arrives
 * via a different temporary file path, the hash matches and the
 * conversation history stays consistent.
 *
 * History ownership:
 * This wrapper does NOT track conversation history itself. The
 * underlying LiteRT-LM [Conversation] owns the full KV cache, so prior
 * turns are implicitly retained across calls as long as the same
 * [Conversation] is reused. Callers are expected to pass only *new*
 * messages on each turn — not the whole transcript.
 *
 * Role handling:
 * OpenAI-style role-interleaved inputs (including multiple SYSTEM turns,
 * e.g. `[SYSTEM, USER, ASSISTANT, SYSTEM, USER]`) are forwarded to
 * LiteRT-LM with roles preserved. When such input arrives, the session
 * rebuilds the underlying [Conversation] with prior turns passed
 * through [ConversationConfig.initialMessages] as a mix of
 * [Message.system], [Message.user], and [Message.model] — the model's
 * embedded chat template then renders the full role-annotated array
 * natively.
 *
 * Fast path:
 * When the caller passes a single trailing USER turn (the common
 * "continue the dialogue" case), the session reuses the existing
 * [Conversation] and simply calls `sendMessage(user)` — no close / no
 * re-prefill. LiteRT-LM keeps the prior history in its KV cache, so
 * this stays O(new tokens) instead of O(all tokens) per turn.
 */
package com.example.quickdotai

import android.util.Log
import com.google.ai.edge.litertlm.Channel
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.MessageCallback
import com.google.ai.edge.litertlm.SamplerConfig
import java.io.File
import java.util.UUID
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

/**
 * @brief Internal helper that holds the state of a single LiteRT-LM
 *        chat session. Not an interface implementation — all public
 *        chat API goes through [QuickDotAI] / [LiteRTLm].
 *
 * @param engine      the parent LiteRT-LM engine (kept for rebuild)
 * @param config      session-level sampling + template config
 * @param visionEnabled whether the engine was loaded with a vision backend
 * @param onSessionClosed callback fired once when [close] is invoked.
 *        Used by [LiteRTLm] to clear its `activeSession` and restore
 *        the flat-API Conversation. Skipped when the parent engine is
 *        tearing down (it nulls out `activeSession` before calling
 *        close so the callback sees nothing to do).
 */
internal class LiteRTLmChatSession(
    private val engine: Engine,
    private val config: QuickAiChatSessionConfig?,
    private val visionEnabled: Boolean,
    private val onSessionClosed: (() -> Unit)? = null,
    val sessionId: String = UUID.randomUUID().toString()
) {

    private var conversation: Conversation? = null
    internal val imageStore = ImageStore()

    /** Signals an in-flight cancel request. */
    private val cancelRequested = AtomicBoolean(false)

    /**
     * Session-level `extraContext` map forwarded to every
     * [Conversation.sendMessage] / [Conversation.sendMessageAsync] call.
     *
     * LiteRT-LM's [ConversationConfig] does not expose template kwargs,
     * so we materialize them per-call instead. Built once from
     * [QuickAiChatSessionConfig.chatTemplateKwargs]:
     *  - `enableThinking` → `"enable_thinking" → Boolean`
     *
     * Empty when no template kwargs are configured.
     */
    private val extraContext: Map<String, Any> =
        buildExtraContext(config?.chatTemplateKwargs).also {
            if (it.isNotEmpty()) {
                Log.i(TAG, "LiteRTLmChatSession($sessionId): extraContext=$it")
            }
        }

    @Volatile
    private var closed = false

    private var lastRunDurationMs: Double = 0.0

    // ----- run (blocking) ------------------------------------------------

    fun run(
        messages: List<QuickAiChatMessage>
    ): BackendResult<QuickAiChatResult> {
        if (closed) return errClosed()
        cancelRequested.set(false)

        val prep = prepareTurn(messages) ?: return lastPrepError
            ?: BackendResult.Err(QuickAiError.INVALID_PARAMETER, "invalid chat input")

        return try {
            val c = acquireConversationForTurn(prep, messages)
                ?: return BackendResult.Err(
                    QuickAiError.INFERENCE_FAILED,
                    "Failed to build Conversation for this turn"
                )

            val startNs = System.nanoTime()
            val response = if (hasImages(prep.lastUser)) {
                val contents = toChatContents(prep.lastUser)
                c.sendMessage(contents, extraContext)
            } else {
                val text = extractText(prep.lastUser)
                c.sendMessage(text, extraContext)
            }
            lastRunDurationMs = (System.nanoTime() - startNs) / 1_000_000.0

            val output = response.toString()
            val reasoning = response.channels[THOUGHT_CHANNEL_NAME]?.takeIf { it.isNotBlank() }
            Log.i(TAG, "run($sessionId): completed in ${lastRunDurationMs.toLong()} ms")

            BackendResult.Ok(
                QuickAiChatResult(
                    content = output,
                    reasoning = reasoning,
                    metrics = PerformanceMetrics(totalDurationMs = lastRunDurationMs)
                )
            )
        } catch (t: Throwable) {
            Log.e(TAG, "run($sessionId): inference failed", t)
            BackendResult.Err(
                QuickAiError.INFERENCE_FAILED,
                t.message ?: "chat inference failed"
            )
        }
    }

    // ----- runStreaming ---------------------------------------------------

    fun runStreaming(
        messages: List<QuickAiChatMessage>,
        sink: StreamSink
    ): BackendResult<QuickAiChatResult> {
        if (closed) {
            val err = errClosed()
            sink.onError(err.error, err.message)
            return err
        }
        cancelRequested.set(false)

        val prep = prepareTurn(messages)
        if (prep == null) {
            val err = lastPrepError ?: BackendResult.Err(
                QuickAiError.INVALID_PARAMETER,
                "invalid chat input"
            )
            sink.onError(err.error, err.message)
            return err
        }

        val c = acquireConversationForTurn(prep, messages)
        if (c == null) {
            val err = BackendResult.Err(
                QuickAiError.INFERENCE_FAILED,
                "Failed to build Conversation for this turn"
            )
            sink.onError(err.error, err.message)
            return err
        }

        val latch = CountDownLatch(1)
        val accumulated = StringBuilder()
        val reasoningAccumulated = StringBuilder()
        var terminalError: BackendResult.Err? = null
        val startNs = System.nanoTime()

        val callback = object : MessageCallback {
            override fun onMessage(message: Message) {
                if (cancelRequested.get()) return
                try {
                    val full = message.toString()
                    val delta = if (full.startsWith(accumulated.toString())) {
                        full.substring(accumulated.length)
                    } else {
                        full
                    }
                    if (delta.isNotEmpty()) {
                        accumulated.append(delta)
                        sink.onDelta(delta)
                    }
                    val reasoning = message.channels[THOUGHT_CHANNEL_NAME].orEmpty()
                    val reasoningDelta = if (reasoning.startsWith(reasoningAccumulated.toString())) {
                        reasoning.substring(reasoningAccumulated.length)
                    } else {
                        reasoning
                    }
                    if (reasoningDelta.isNotEmpty()) {
                        reasoningAccumulated.append(reasoningDelta)
                        sink.onReasoningDelta(reasoningDelta)
                    }
                } catch (t: Throwable) {
                    Log.w(TAG, "runStreaming($sessionId): onMessage threw", t)
                }
            }

            override fun onDone() {
                lastRunDurationMs = (System.nanoTime() - startNs) / 1_000_000.0
                Log.i(
                    TAG,
                    "runStreaming($sessionId): onDone after " +
                        "${lastRunDurationMs.toLong()} ms"
                )
                try { sink.onDone() } finally { latch.countDown() }
            }

            override fun onError(throwable: Throwable) {
                Log.e(TAG, "runStreaming($sessionId): onError", throwable)
                val err = BackendResult.Err(
                    QuickAiError.INFERENCE_FAILED,
                    throwable.message ?: "chat streaming failed"
                )
                terminalError = err
                try { sink.onError(err.error, err.message) } finally { latch.countDown() }
            }
        }

        return try {
            if (hasImages(prep.lastUser)) {
                val contents = toChatContents(prep.lastUser)
                c.sendMessageAsync(contents, callback, extraContext)
            } else {
                val text = extractText(prep.lastUser)
                c.sendMessageAsync(text, callback, extraContext)
            }

            val finished = latch.await(5, TimeUnit.MINUTES)
            if (!finished) {
                Log.e(TAG, "runStreaming($sessionId): timed out")
                val err = BackendResult.Err(
                    QuickAiError.INFERENCE_FAILED,
                    "chat streaming timeout"
                )
                sink.onError(err.error, err.message)
                return err
            }

            if (terminalError != null) {
                return terminalError!!
            }

            val output = accumulated.toString()
            BackendResult.Ok(
                QuickAiChatResult(
                    content = output,
                    reasoning = reasoningAccumulated.toString().takeIf { it.isNotEmpty() },
                    metrics = PerformanceMetrics(totalDurationMs = lastRunDurationMs)
                )
            )
        } catch (t: Throwable) {
            Log.e(TAG, "runStreaming($sessionId): threw", t)
            val err = BackendResult.Err(
                QuickAiError.INFERENCE_FAILED,
                t.message ?: "chat streaming failed"
            )
            sink.onError(err.error, err.message)
            err
        }
    }

    // ----- cancel --------------------------------------------------------

    fun cancel() {
        cancelRequested.set(true)
        Log.i(TAG, "cancel($sessionId): cancel requested")
    }

    // ----- rebuild -------------------------------------------------------

    fun rebuild(
        messages: List<QuickAiChatMessage>
    ): BackendResult<Unit> {
        if (closed) return errClosed()

        Log.i(
            TAG,
            "rebuild($sessionId): reset Conversation and pre-seed with " +
                "${messages.size} message(s)"
        )

        // Drop the KV cache carried by the current Conversation so the
        // rebuild is a true reset.
        try {
            conversation?.close()
        } catch (t: Throwable) {
            Log.w(TAG, "rebuild($sessionId): conversation.close() threw", t)
        }
        conversation = null

        // Prune images not referenced by the new seed.
        val referencedHashes = collectImageHashes(messages)
        imageStore.retainOnly(referencedHashes)

        // When the caller provides seed messages, eagerly create a new
        // Conversation with them as initialMessages so LiteRT-LM can
        // pre-fill its KV cache. If no seed is given, we leave the
        // Conversation null and the next run() will lazily create it.
        if (messages.isNotEmpty()) {
            val initial = try {
                messages.map { toLiteRtMessage(it) }
            } catch (t: Throwable) {
                Log.e(TAG, "rebuild($sessionId): mapping seed failed", t)
                return BackendResult.Err(
                    QuickAiError.INFERENCE_FAILED,
                    t.message ?: "failed to map seed messages"
                )
            }
            try {
                conversation = createConversationFromConfig(engine, config, initial)
            } catch (t: Throwable) {
                Log.e(TAG, "rebuild($sessionId): createConversation threw", t)
                return BackendResult.Err(
                    QuickAiError.INFERENCE_FAILED,
                    t.message ?: "failed to create seeded Conversation"
                )
            }
        }

        return BackendResult.Ok(Unit)
    }

    // ----- close ---------------------------------------------------------

    fun close() {
        if (closed) return
        closed = true
        Log.i(TAG, "close($sessionId)")
        try {
            conversation?.close()
        } catch (t: Throwable) {
            Log.w(TAG, "close($sessionId): conversation.close() threw", t)
        }
        conversation = null
        imageStore.clear()
        try {
            onSessionClosed?.invoke()
        } catch (t: Throwable) {
            Log.w(TAG, "close($sessionId): onSessionClosed threw", t)
        }
    }

    // ----- helpers -------------------------------------------------------

    /** Per-turn data: trailing USER and any prior turns in this call. */
    private data class TurnPrep(
        val priorTurns: List<QuickAiChatMessage>,
        val lastUser: QuickAiChatMessage,
    )

    /**
     * Surface for [run] / [runStreaming] to learn WHY [prepareTurn]
     * returned null without resorting to exceptions. Mutated only from
     * the caller thread right before prepareTurn is invoked.
     */
    @Volatile
    private var lastPrepError: BackendResult.Err? = null

    /**
     * Validate [messages] (non-empty, trailing USER, vision gating) and
     * split off the trailing USER turn that will drive inference from
     * any leading turns that need to feed the chat template.
     *
     * Returns null on validation failure; the concrete error is stashed
     * in [lastPrepError] for the caller to surface (and optionally
     * forward to a StreamSink).
     */
    private fun prepareTurn(messages: List<QuickAiChatMessage>): TurnPrep? {
        lastPrepError = null
        if (messages.isEmpty()) {
            lastPrepError = BackendResult.Err(
                QuickAiError.INVALID_PARAMETER,
                "messages list is empty"
            )
            return null
        }
        if (messages.last().role != QuickAiChatRole.USER) {
            lastPrepError = BackendResult.Err(
                QuickAiError.INVALID_PARAMETER,
                "last message must have role USER to trigger inference " +
                    "(got ${messages.last().role})"
            )
            return null
        }

        val lastUser = messages.last()
        val priorTurns = messages.subList(0, messages.size - 1)

        if (!visionEnabled && messages.any { hasImages(it) }) {
            lastPrepError = BackendResult.Err(
                QuickAiError.UNSUPPORTED,
                "Engine loaded in text-only mode — cannot process images"
            )
            return null
        }

        return TurnPrep(priorTurns = priorTurns, lastUser = lastUser)
    }

    /**
     * Close any previously-held Conversation and build a fresh one for
     * this turn, passing [priorTurns] through
     * [ConversationConfig.initialMessages] so LiteRT-LM's native chat
     * template renders the full role-annotated array.
     *
     * Returns null on construction failure.
     */
    private fun rebuildConversationForTurn(
        priorTurns: List<QuickAiChatMessage>
    ): Conversation? {
        try {
            conversation?.close()
        } catch (t: Throwable) {
            Log.w(TAG, "rebuildConversationForTurn($sessionId): close() threw", t)
        }
        conversation = null

        val initial = try {
            priorTurns.map { toLiteRtMessage(it) }
        } catch (t: Throwable) {
            Log.e(TAG, "rebuildConversationForTurn($sessionId): mapping failed", t)
            return null
        }

        return try {
            createConversationFromConfig(engine, config, initial).also {
                conversation = it
                if (initial.isNotEmpty()) {
                    Log.i(
                        TAG,
                        "rebuildConversationForTurn($sessionId): seeded with " +
                            "${initial.size} initialMessage(s) " +
                            priorTurns.joinToString(prefix = "[", postfix = "]") {
                                it.role.name
                            }
                    )
                }
            }
        } catch (t: Throwable) {
            Log.e(TAG, "rebuildConversationForTurn($sessionId): createConversation threw", t)
            null
        }
    }

    /**
     * Fast path wrapper around [rebuildConversationForTurn].
     *
     * Source of truth: LiteRT-LM's [Conversation] owns the KV cache and
     * therefore the full conversation history. As long as we keep
     * handing the same [Conversation] out, `sendMessage(user)` extends
     * the dialogue correctly — no replay needed on our side.
     *
     * When the caller passes only a single trailing USER turn we just
     * reuse the existing [Conversation]. This keeps each follow-up turn
     * O(new user tokens) instead of O(all history tokens).
     *
     * Falls back to [rebuildConversationForTurn] whenever:
     *  - no existing Conversation is held yet (first turn, or post-rebuild), or
     *  - the caller injects anything other than exactly one USER turn
     *    (e.g. SYSTEM/ASSISTANT turns, role-interleaved bundles, or
     *    multi-USER batches) — those require the full role-annotated
     *    initialMessages replay, which drops the KV cache.
     */
    private fun acquireConversationForTurn(
        prep: TurnPrep,
        newMessages: List<QuickAiChatMessage>
    ): Conversation? {
        val existing = conversation
        if (existing != null &&
            newMessages.size == 1 &&
            newMessages[0].role == QuickAiChatRole.USER
        ) {
            Log.i(
                TAG,
                "acquireConversationForTurn($sessionId): fast path — " +
                    "reusing existing Conversation (KV cache retained)"
            )
            return existing
        }
        return rebuildConversationForTurn(prep.priorTurns)
    }

    /**
     * Map a [QuickAiChatMessage] to a LiteRT-LM [Message], preserving
     * the original role:
     *  - [QuickAiChatRole.SYSTEM]    → [Message.system]
     *  - [QuickAiChatRole.USER]      → [Message.user]
     *  - [QuickAiChatRole.ASSISTANT] → [Message.model]
     *
     * Image parts are also stored in the session's [imageStore] so the
     * hash-based identity stays stable across rebuilds.
     */
    private fun toLiteRtMessage(msg: QuickAiChatMessage): Message {
        val contents = toChatContents(msg)
        return when (msg.role) {
            QuickAiChatRole.SYSTEM -> Message.system(contents)
            QuickAiChatRole.USER -> Message.user(contents)
            QuickAiChatRole.ASSISTANT -> Message.model(contents = contents)
        }
    }

    private fun hasImages(msg: QuickAiChatMessage): Boolean =
        msg.parts.any { it is PromptPart.ImageFile || it is PromptPart.ImageBytes }

    private fun extractText(msg: QuickAiChatMessage): String =
        msg.parts.filterIsInstance<PromptPart.Text>().joinToString("") { it.text }

    /**
     * Convert a user message's parts into a LiteRT-LM [Contents].
     * Images are stored in [imageStore] along the way.
     */
    private fun toChatContents(msg: QuickAiChatMessage): Contents {
        val mapped: List<Content> = msg.parts.map { p ->
            when (p) {
                is PromptPart.Text -> Content.Text(p.text)
                is PromptPart.ImageFile -> {
                    val f = File(p.absolutePath)
                    require(f.exists() && f.canRead()) {
                        "Image file not readable: ${p.absolutePath}"
                    }
                    // Store in ImageStore for stable hash-based identity
                    imageStore.store(p.absolutePath)
                    Content.ImageFile(p.absolutePath)
                }
                is PromptPart.ImageBytes -> {
                    require(p.bytes.isNotEmpty()) {
                        "Image bytes are empty"
                    }
                    imageStore.store(p.bytes)
                    Content.ImageBytes(p.bytes)
                }
                is PromptPart.PreprocessedPixels -> {
                    throw UnsupportedOperationException(
                        "PromptPart.PreprocessedPixels is not supported by LiteRTLm. " +
                        "Use NativeQuickDotAI for V-JEPA multi-image inference."
                    )
                }
            }
        }
        return Contents.of(mapped)
    }

    /**
     * Collect all image hashes referenced in a list of messages.
     * Used by [rebuild] to determine which cached images to keep.
     */
    private fun collectImageHashes(messages: List<QuickAiChatMessage>): Set<String> {
        val hashes = mutableSetOf<String>()
        for (msg in messages) {
            for (part in msg.parts) {
                when (part) {
                    is PromptPart.ImageFile -> {
                        val f = File(part.absolutePath)
                        if (f.exists() && f.canRead()) {
                            hashes.add(ImageStore.sha256Hex(f.readBytes()))
                        }
                    }
                    is PromptPart.ImageBytes -> {
                        if (part.bytes.isNotEmpty()) {
                            hashes.add(ImageStore.sha256Hex(part.bytes))
                        }
                    }
                    is PromptPart.PreprocessedPixels -> { /* not supported by LiteRTLm */ }
                    is PromptPart.Text -> { /* no image */ }
                }
            }
        }
        return hashes
    }

    private fun errClosed(): BackendResult.Err = BackendResult.Err(
        QuickAiError.NOT_INITIALIZED,
        "Chat session $sessionId is closed"
    )

    companion object {
        private const val TAG = "LiteRTLmChatSession"

        /**
         * Translate a [QuickAiChatTemplateKwargs] into the
         * `extraContext: Map<String, Any>` that LiteRT-LM's
         * [Conversation.sendMessage] / `sendMessageAsync` accepts.
         *
         * The native chat template reads these keys directly
         * (e.g. Jinja `{% if enable_thinking %}`), so the JSON-style
         * snake_case names must be preserved exactly:
         *  - [QuickAiChatTemplateKwargs.enableThinking] →
         *    `"enable_thinking" → Boolean`
         *
         * Null fields are omitted from the map so the chat template's
         * own default kicks in. Returns an empty map when [kwargs] is
         * null.
         */
        private fun buildExtraContext(
            kwargs: QuickAiChatTemplateKwargs?
        ): Map<String, Any> {
            if (kwargs == null) return emptyMap()
            val out = mutableMapOf<String, Any>()
            kwargs.enableThinking?.let { out["enable_thinking"] = it }
            return out
        }

        // Neutral fallback values used only when the caller specifies
        // *some* sampling fields but not all three required ones.
        // LiteRT-LM's SamplerConfig(topK, topP, temperature) are all
        // non-nullable with no defaults, so we must supply something
        // whenever we construct a SamplerConfig at all.
        private const val FALLBACK_TEMPERATURE = 1.0
        private const val FALLBACK_TOP_K = 40
        private const val FALLBACK_TOP_P = 0.95
        private const val THOUGHT_CHANNEL_NAME = "thought"
        private const val THOUGHT_CHANNEL_START = "<|channel>thought"
        private const val THOUGHT_CHANNEL_END = "<channel|>"

        /**
         * Build a LiteRT-LM [Conversation] from a [QuickAiChatSessionConfig]
         * and a list of prior turns that will be forwarded to the native
         * chat template via [ConversationConfig.initialMessages].
         *
         * Maps:
         *  - [QuickAiChatSessionConfig.systemInstruction] →
         *    [ConversationConfig.systemInstruction]
         *  - [QuickAiChatSamplingConfig] → [SamplerConfig]
         *    (see [buildSamplerConfig] for per-field behavior)
         *  - [initialMessages] → [ConversationConfig.initialMessages]
         *    (role-preserving: SYSTEM/USER/ASSISTANT → system/user/model)
         *
         * Falls back to the bare `engine.createConversation()` overload
         * only when nothing is configured AND there are no prior turns,
         * so LiteRT-LM uses its own engine-level defaults.
         */
        private fun createConversationFromConfig(
            engine: Engine,
            config: QuickAiChatSessionConfig?,
            initialMessages: List<Message> = emptyList(),
        ): Conversation {
            val sysInstruction = config?.systemInstruction?.takeIf { it.isNotBlank() }
            val samplerConfig = buildSamplerConfig(config?.sampling)
            val channels = buildConversationChannels(config?.chatTemplateKwargs)

            // Skip ConversationConfig entirely when nothing is configured
            // so LiteRT-LM uses its own engine/model defaults.
            if (sysInstruction == null &&
                samplerConfig == null &&
                channels == null &&
                initialMessages.isEmpty()
            ) {
                return engine.createConversation()
            }

            val convConfig = ConversationConfig(
                systemInstruction =
                    sysInstruction?.let { Contents.of(it) },
                initialMessages = initialMessages,
                samplerConfig = samplerConfig,
                channels = channels,
            )
            Log.i(
                TAG,
                "createConversationFromConfig: " +
                    "sysInstruction=${sysInstruction?.take(60)}, " +
                    "samplerConfig=$samplerConfig, " +
                    "initialMessages=${initialMessages.size}, " +
                    "channels=${channels?.joinToString { it.channelName } ?: "none"}"
            )
            return engine.createConversation(convConfig)
        }

        private fun buildConversationChannels(
            kwargs: QuickAiChatTemplateKwargs?
        ): List<Channel>? {
            if (kwargs?.enableThinking != true) {
                return null
            }
            return listOf(
                Channel(
                    channelName = THOUGHT_CHANNEL_NAME,
                    start = THOUGHT_CHANNEL_START,
                    end = THOUGHT_CHANNEL_END,
                )
            )
        }

        /**
         * Map [QuickAiChatSamplingConfig] to LiteRT-LM [SamplerConfig].
         *
         * LiteRT-LM's `SamplerConfig(topK: Int, topP: Double,
         * temperature: Double, seed: Int = 0)` has three non-nullable
         * core fields. That means we cannot express "set only
         * temperature, leave topK/topP to the engine default" through
         * a partially-populated SamplerConfig — any SamplerConfig we
         * construct MUST carry all three.
         *
         * Behavior:
         *  - Returns `null` when [sampling] is null or all relevant
         *    fields are null. The caller then passes no samplerConfig
         *    to ConversationConfig, and LiteRT-LM uses its own
         *    engine-level defaults (preferred path for best quality).
         *  - When any of temperature/topK/topP/seed is specified,
         *    constructs a full SamplerConfig, filling the remaining
         *    core fields from [FALLBACK_TEMPERATURE]/[FALLBACK_TOP_K]/
         *    [FALLBACK_TOP_P]. A warning is logged so partial
         *    specification is visible in logcat.
         *  - [QuickAiChatSamplingConfig.minP] and
         *    [QuickAiChatSamplingConfig.maxTokens] are not supported by
         *    LiteRT-LM's SamplerConfig; values are ignored and a
         *    warning is logged.
         *
         * LiteRT-LM validates ranges in `SamplerConfig.init`
         * (topK > 0, topP in [0,1], temperature >= 0) and throws
         * [IllegalArgumentException] on violation. That throw
         * propagates up through [createConversationFromConfig] and is
         * caught in [LiteRTLm.openChatSession], where it is converted
         * to a BackendResult.Err.
         */
        private fun buildSamplerConfig(
            sampling: QuickAiChatSamplingConfig?
        ): SamplerConfig? {
            if (sampling == null) return null

            val anyCoreSet = sampling.temperature != null ||
                sampling.topK != null ||
                sampling.topP != null ||
                sampling.seed != null

            // Warn about QuickAi fields that LiteRT-LM's SamplerConfig
            // does not expose. Doing it up front means the warning
            // fires even if no core field is set (i.e. even if we end
            // up returning null below).
            if (sampling.minP != null) {
                Log.w(
                    TAG,
                    "buildSamplerConfig: minP=${sampling.minP} is not " +
                        "supported by LiteRT-LM SamplerConfig — ignored"
                )
            }
            if (sampling.maxTokens != null) {
                Log.w(
                    TAG,
                    "buildSamplerConfig: maxTokens=${sampling.maxTokens} " +
                        "is not supported by LiteRT-LM SamplerConfig — ignored"
                )
            }

            if (!anyCoreSet) return null

            val missing = buildList {
                if (sampling.temperature == null) add("temperature")
                if (sampling.topK == null) add("topK")
                if (sampling.topP == null) add("topP")
            }
            if (missing.isNotEmpty()) {
                Log.w(
                    TAG,
                    "buildSamplerConfig: partial sampling config — " +
                        "LiteRT-LM SamplerConfig requires all core fields; " +
                        "filling ${missing.joinToString()} with fallback " +
                        "defaults (temperature=$FALLBACK_TEMPERATURE, " +
                        "topK=$FALLBACK_TOP_K, topP=$FALLBACK_TOP_P). " +
                        "Specify all three together to avoid this."
                )
            }

            return SamplerConfig(
                topK = sampling.topK ?: FALLBACK_TOP_K,
                topP = sampling.topP ?: FALLBACK_TOP_P,
                temperature = sampling.temperature ?: FALLBACK_TEMPERATURE,
                seed = sampling.seed ?: 0,
            )
        }
    }
}
