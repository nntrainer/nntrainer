// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    LiteRTLm.kt
 * @brief   QuickDotAI implementation backed by the LiteRT-LM Kotlin API
 *          (https://github.com/google-ai-edge/LiteRT-LM).
 *
 * LiteRTLm is the QuickDotAI-level routing target for ModelId.GEMMA4
 * and is typically selected inside the host app's registry. It
 * implements the same [QuickDotAI] contract as [NativeQuickDotAI], so
 * consumers never need to branch on the concrete implementation.
 *
 * See how-to-use-litert-lm-guide.md at the repo root for the canonical
 * LiteRT-LM Kotlin API surface this code is written against.
 */
package com.example.quickdotai

import android.content.Context
import android.util.Log
import com.google.ai.edge.litertlm.Backend as LlmBackend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.MessageCallback
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

/**
 * @brief LiteRT-LM-backed QuickDotAI implementation for Gemma-family
 *        models.
 *
 * Non-thread-safe — the host app must drive a single instance from a
 * single worker thread.
 *
 * @param appContext application context, used only to resolve the
 *        test-mode fallback model path via [Context.getExternalFilesDir]
 *        when [LoadModelRequest.modelPath] is null. Third-party apps
 *        are encouraged to always pass an explicit [LoadModelRequest.modelPath]
 *        and therefore never hit the fallback.
 */
class LiteRTLm(
    private val appContext: Context,
    /** Base directory for model files, used by [testModelFile] fallback.
     *  Defaults to `/sdcard/Download/aistudio-mobile/models/`. */
    private val defaultModelBasePath: String = "/sdcard/Download/aistudio-mobile/models/"
) : QuickDotAI {

    override val kind: String = "litert-lm"

    override var architecture: String? = "Gemma4ForCausalLM"
        private set

    // LiteRT-LM's Engine is AutoCloseable — we hold on to it (and a
    // single reusable Conversation) for the entire lifetime of the
    // LiteRTLm instance. Closed in [close].
    private var engine: Engine? = null
    private var conversation: Conversation? = null

    // True if the engine was loaded with a non-null visionBackend. We
    // gate runMultimodal / runMultimodalStreaming on this so callers
    // that loaded in text-only mode get a clear UNSUPPORTED error
    // instead of a cryptic native failure deep inside LiteRT-LM.
    private var visionEnabled: Boolean = false

    /** Signals an in-flight cancel request for one-shot run(). */
    private val cancelRequested = AtomicBoolean(false)

    // Simple wall-clock metrics. LiteRT-LM's Kotlin API does not expose
    // token-level prefill/generation timings in the release we target,
    // so we record initialization and last-run durations ourselves and
    // leave the token counts at 0 for now.
    private var initializationDurationMs: Double = 0.0
    private var lastRunDurationMs: Double = 0.0

    // LiteRT-LM allows only one Conversation per Engine at a time.
    // A new session cannot be opened until the active one is closed.
    private var activeSession: LiteRTLmChatSession? = null

    override val chatSessionId: String?
        get() = activeSession?.sessionId

    override fun load(req: LoadModelRequest): BackendResult<Unit> {
        Log.i(
            TAG,
            "load() entered: modelId=${req.modelId} backend=${req.backend} " +
                "quant=${req.quantization} modelPath=${req.modelPath}"
        )

        // TEST-MODE fallback: during bring-up we want `load(GEMMA4)` to
        // Just Work even if the caller forgets to pass model_path. Fall
        // back to a known-good on-device path inside the host app's
        // external files dir so the pipeline can be end-to-end verified.
        //
        // /data/local/tmp is NOT app-readable on user builds: it carries
        // the shell_data_file SELinux context and the untrusted_app
        // domain is denied read access. getExternalFilesDir() (a)
        // requires no runtime permissions, (b) is always writable by
        // adb, and (c) is already created by the framework for us.
        val modelPath = req.modelPath?.takeIf { it.isNotBlank() }
            ?: run {
                val fallback = testModelFile().absolutePath
                Log.w(
                    TAG,
                    "load(): model_path not provided, falling back to " +
                        "test path: $fallback"
                )
                fallback
            }

        Log.i(TAG, "load(): resolved modelPath=$modelPath")

        val modelFile = File(modelPath)
        if (!modelFile.exists()) {
            val parentDir = testModelFile().parentFile?.absolutePath ?: "<unknown>"
            val hint = "push it with: adb push $TEST_GEMMA4_FILE_NAME $parentDir/"
            Log.e(TAG, "load(): model file not found at $modelPath — $hint")
            return BackendResult.Err(
                QuickAiError.MODEL_LOAD_FAILED,
                "model file not found at $modelPath. $hint"
            )
        }
        Log.i(
            TAG,
            "load(): model file exists, size=${modelFile.length()} bytes, " +
                "canRead=${modelFile.canRead()}"
        )

        val llmBackend: LlmBackend = mapBackend(req.backend)
        // Null visionBackend leaves the engine in text-only mode. A
        // non-null value enables the multimodal code path and unblocks
        // [runMultimodal] / [runMultimodalStreaming] at call time.
        val visionLlmBackend: LlmBackend? = req.visionBackend?.let(::mapBackend)

        Log.i(
            TAG,
            "load(): mapped compute backend ${req.backend} -> " +
                "${llmBackend::class.java.simpleName}, vision=${req.visionBackend} -> " +
                (visionLlmBackend?.let { it::class.java.simpleName } ?: "<none>")
        )

        val engineConfig = EngineConfig(
            modelPath = modelPath,
            backend = llmBackend,
            visionBackend = visionLlmBackend,
            cacheDir = req.cacheDir,
            maxNumTokens = req.maxNumTokens,
        )
        Log.i(
            TAG,
            "load(): EngineConfig built (cacheDir=${req.cacheDir}, " +
                "maxNumTokens=${req.maxNumTokens}), constructing Engine…"
        )

        return try {
            val startNs = System.nanoTime()
            val e = Engine(engineConfig)
            Log.i(TAG, "load(): Engine() constructed, calling initialize()…")
            e.initialize()
            Log.i(
                TAG,
                "load(): Engine.initialize() returned after " +
                    "${(System.nanoTime() - startNs) / 1_000_000} ms"
            )

            val c = e.createConversation()
            Log.i(TAG, "load(): Engine.createConversation() returned")

            initializationDurationMs = (System.nanoTime() - startNs) / 1_000_000.0
            engine = e
            conversation = c
            visionEnabled = (visionLlmBackend != null)
            Log.i(
                TAG,
                "load(): SUCCESS, total init duration=${initializationDurationMs} ms, " +
                    "visionEnabled=$visionEnabled"
            )
            BackendResult.Ok(Unit)
        } catch (t: Throwable) {
            Log.e(TAG, "load(): LiteRT-LM engine load failed", t)
            // On partial success, make sure we don't leak a half-initialised
            // engine into the caller's registry.
            closeQuietly()
            BackendResult.Err(
                QuickAiError.MODEL_LOAD_FAILED,
                t.message ?: "LiteRT-LM engine initialization failed"
            )
        }
    }

    /**
     * @brief Multimodal inference — blocking.
     *
     * Builds a LiteRT-LM [Contents] from [parts] and hands it to
     * `conversation.sendMessage(contents)`. Returns the decoded text
     * of the model's reply on success, or a [BackendResult.Err] on
     * failure. Gated on [visionEnabled] so callers that forgot to set
     * [LoadModelRequest.visionBackend] get a clear UNSUPPORTED error
     * rather than a cryptic native crash.
     */
    override fun runMultimodalHandle(parts: List<PromptPart>): BackendResult<String> {
        val c = conversation
            ?: run {
                Log.e(TAG, "runMultimodal(): called before load() — conversation is null")
                return BackendResult.Err(
                    QuickAiError.NOT_INITIALIZED,
                    "LiteRTLm has not been loaded yet"
                )
            }
        if (!visionEnabled) {
            Log.e(
                TAG,
                "runMultimodal(): engine loaded in text-only mode — " +
                    "reload with LoadModelRequest.visionBackend set"
            )
            return BackendResult.Err(
                QuickAiError.UNSUPPORTED,
                "LiteRTLm was loaded without a visionBackend — reload with " +
                    "LoadModelRequest.visionBackend set to a non-null value."
            )
        }
        if (parts.isEmpty()) {
            return BackendResult.Err(
                QuickAiError.INVALID_PARAMETER,
                "runMultimodal(): parts list is empty"
            )
        }

        val contents = try {
            toLiteRtContents(parts)
        } catch (t: Throwable) {
            Log.e(TAG, "runMultimodal(): failed to build Contents", t)
            return BackendResult.Err(
                QuickAiError.INVALID_PARAMETER,
                t.message ?: "failed to build LiteRT-LM Contents from parts"
            )
        }

        Log.i(TAG, "runMultimodal(): sending ${parts.size} parts")
        return try {
            val startNs = System.nanoTime()
            val message = c.sendMessage(contents)
            lastRunDurationMs = (System.nanoTime() - startNs) / 1_000_000.0
            val output = message.toString()
            Log.i(
                TAG,
                "runMultimodal(): sendMessage returned in ${lastRunDurationMs.toLong()} ms, " +
                    "output length=${output.length}"
            )
            BackendResult.Ok(output)
        } catch (t: Throwable) {
            Log.e(TAG, "runMultimodal(): LiteRT-LM sendMessage failed", t)
            BackendResult.Err(
                QuickAiError.INFERENCE_FAILED,
                t.message ?: "LiteRT-LM multimodal inference failed"
            )
        }
    }

    /**
     * @brief Multimodal inference — streaming.
     *
     * Same shape as [runStreaming]: drives LiteRT-LM's
     * `sendMessageAsync(contents, callback)` and forwards incremental
     * deltas to [sink], blocking the caller thread on a
     * [CountDownLatch] until the callback signals `onDone` or
     * `onError`. The delta-extraction logic is shared with the
     * text-only path (see [runStreaming] for the rationale behind
     * the `accumulated` StringBuilder defensive handling).
     */
    override fun runMultimodalHandleStreaming(
        parts: List<PromptPart>,
        sink: StreamSink
    ): BackendResult<Unit> {
        val c = conversation
            ?: run {
                Log.e(TAG, "runMultimodalStreaming(): called before load()")
                val err = BackendResult.Err(
                    QuickAiError.NOT_INITIALIZED,
                    "LiteRTLm has not been loaded yet"
                )
                sink.onError(err.error, err.message)
                return err
            }
        if (!visionEnabled) {
            Log.e(TAG, "runMultimodalStreaming(): engine loaded in text-only mode")
            val err = BackendResult.Err(
                QuickAiError.UNSUPPORTED,
                "LiteRTLm was loaded without a visionBackend — reload with " +
                    "LoadModelRequest.visionBackend set to a non-null value."
            )
            sink.onError(err.error, err.message)
            return err
        }
        if (parts.isEmpty()) {
            val err = BackendResult.Err(
                QuickAiError.INVALID_PARAMETER,
                "runMultimodalStreaming(): parts list is empty"
            )
            sink.onError(err.error, err.message)
            return err
        }

        val contents = try {
            toLiteRtContents(parts)
        } catch (t: Throwable) {
            Log.e(TAG, "runMultimodalStreaming(): failed to build Contents", t)
            val err = BackendResult.Err(
                QuickAiError.INVALID_PARAMETER,
                t.message ?: "failed to build LiteRT-LM Contents from parts"
            )
            sink.onError(err.error, err.message)
            return err
        }

        Log.i(TAG, "runMultimodalStreaming(): ${parts.size} parts")

        val latch = CountDownLatch(1)
        val accumulated = StringBuilder()
        var terminalError: BackendResult.Err? = null
        val startNs = System.nanoTime()

        val callback = object : MessageCallback {
            override fun onMessage(message: Message) {
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
                } catch (t: Throwable) {
                    Log.w(TAG, "runMultimodalStreaming(): onMessage threw", t)
                }
            }

            override fun onDone() {
                lastRunDurationMs = (System.nanoTime() - startNs) / 1_000_000.0
                Log.i(
                    TAG,
                    "runMultimodalStreaming(): onDone after ${lastRunDurationMs.toLong()} ms, " +
                        "total chars=${accumulated.length}"
                )
                try {
                    sink.onDone()
                } finally {
                    latch.countDown()
                }
            }

            override fun onError(throwable: Throwable) {
                Log.e(TAG, "runMultimodalStreaming(): onError from LiteRT-LM", throwable)
                val err = BackendResult.Err(
                    QuickAiError.INFERENCE_FAILED,
                    throwable.message ?: "LiteRT-LM multimodal streaming inference failed"
                )
                terminalError = err
                try {
                    sink.onError(err.error, err.message)
                } finally {
                    latch.countDown()
                }
            }
        }

        return try {
            c.sendMessageAsync(contents, callback)
            val finished = latch.await(5, TimeUnit.MINUTES)
            if (!finished) {
                Log.e(TAG, "runMultimodalStreaming(): timed out waiting for onDone/onError")
                val err = BackendResult.Err(
                    QuickAiError.INFERENCE_FAILED,
                    "LiteRT-LM multimodal streaming timeout"
                )
                sink.onError(err.error, err.message)
                return err
            }
            terminalError ?: BackendResult.Ok(Unit)
        } catch (t: Throwable) {
            Log.e(TAG, "runMultimodalStreaming(): sendMessageAsync threw", t)
            val err = BackendResult.Err(
                QuickAiError.INFERENCE_FAILED,
                t.message ?: "LiteRT-LM multimodal streaming inference failed"
            )
            sink.onError(err.error, err.message)
            err
        }
    }

    // --- chat session management -----------------------------------------

    override fun openChatSession(
        config: QuickAiChatSessionConfig?
    ): BackendResult<String> {
        val e = engine
            ?: return BackendResult.Err(
                QuickAiError.NOT_INITIALIZED,
                "LiteRTLm has not been loaded yet"
            )

        // LiteRT-LM supports only one Conversation per Engine. Reject
        // if a session is already active.
        if (activeSession != null) {
            Log.w(
                TAG,
                "openChatSession(): rejected — session ${activeSession!!.sessionId} " +
                    "is still active. Close it first."
            )
            return BackendResult.Err(
                QuickAiError.BAD_REQUEST,
                "A chat session is already active (${activeSession!!.sessionId}). " +
                    "Close it before opening a new one."
            )
        }

        // LiteRT-LM allows only one Conversation per Engine. The flat
        // run()/runStreaming() API keeps its own Conversation in
        // `this.conversation` — close it first so the chat session can
        // create a fresh one. It will be recreated when the session is
        // closed (see closeActiveSession / closeChatSession).
        try {
            conversation?.close()
        } catch (t: Throwable) {
            Log.w(TAG, "openChatSession(): conversation.close() threw", t)
        }
        conversation = null

        return try {
            val session = LiteRTLmChatSession(
                engine = e,
                config = config,
                visionEnabled = visionEnabled,
                onSessionClosed = {
                    // Called when session.close() fires (from any caller).
                    // Only act if we still own this session — closeActiveSession()
                    // nulls activeSession first so teardown skips restoration.
                    if (activeSession != null) {
                        activeSession = null
                        restoreConversation()
                    }
                }
            )
            activeSession = session
            Log.i(TAG, "openChatSession(): created session ${session.sessionId}")
            BackendResult.Ok(session.sessionId)
        } catch (t: Throwable) {
            Log.e(TAG, "openChatSession(): failed", t)
            // Session creation failed — restore the flat-API Conversation
            // so run()/runStreaming() remain usable.
            restoreConversation()
            BackendResult.Err(
                QuickAiError.INFERENCE_FAILED,
                t.message ?: "failed to create chat session"
            )
        }
    }

    override fun closeChatSession(): BackendResult<Unit> {
        val session = activeSession
        if (session == null) {
            Log.w(TAG, "closeChatSession(): no active session")
            return BackendResult.Err(
                QuickAiError.BAD_REQUEST,
                "No active chat session to close"
            )
        }
        // session.close() fires the onSessionClosed callback which
        // nulls activeSession and calls restoreConversation().
        session.close()
        Log.i(TAG, "closeChatSession(${session.sessionId}): closed")
        return BackendResult.Ok(Unit)
    }

    override fun runChatModelHandleStreaming(
        text: String,
        sink: StreamSink
    ): BackendResult<QuickAiChatResult> {
        val session = activeSession
        if (session == null) {
            val err = BackendResult.Err(
                QuickAiError.BAD_REQUEST,
                "No active chat session — call openChatSession() first"
            )
            sink.onError(err.error, err.message)
            return err
        }
        return try {
            val messages = listOf(
                QuickAiChatMessage(role = QuickAiChatRole.USER, parts = listOf(PromptPart.Text(text)))
            )
            session.runStreaming(messages, sink)
        } catch (t: Throwable) {
            Log.e(TAG, "runChatModelHandleStreaming(): threw", t)
            val err = BackendResult.Err(
                QuickAiError.INFERENCE_FAILED,
                t.message ?: "chat streaming failed"
            )
            sink.onError(err.error, err.message)
            err
        }
    }

    override fun runChatMultimodalHandleStreaming(
        parts: List<PromptPart>,
        sink: StreamSink
    ): BackendResult<QuickAiChatResult> {
        val session = activeSession
        if (session == null) {
            val err = BackendResult.Err(
                QuickAiError.BAD_REQUEST,
                "No active chat session — call openChatSession() first"
            )
            sink.onError(err.error, err.message)
            return err
        }
        if (!visionEnabled) {
            val err = BackendResult.Err(
                QuickAiError.UNSUPPORTED,
                "LiteRTLm was loaded without a visionBackend"
            )
            sink.onError(err.error, err.message)
            return err
        }
        return try {
            val messages = listOf(
                QuickAiChatMessage(role = QuickAiChatRole.USER, parts = parts)
            )
            session.runStreaming(messages, sink)
        } catch (t: Throwable) {
            Log.e(TAG, "runChatMultimodalHandleStreaming(): threw", t)
            val err = BackendResult.Err(
                QuickAiError.INFERENCE_FAILED,
                t.message ?: "chat multimodal streaming failed"
            )
            sink.onError(err.error, err.message)
            err
        }
    }

    // ----- OpenAI messages API (handle-based) --------------------------------

    /**
     * @brief Streaming inference with OpenAI message format.
     *
     * Accumulates deltas into a single response, then emits it through [sink].
     * LiteRT-LM does not currently support true token-by-token streaming for
     * handle-based messages, so this is implemented as blocking + chunk.
     */
    override fun runModelHandleWithMessagesStreaming(
        messages: List<QuickAiChatMessage>,
        sink: StreamSink
    ): BackendResult<Unit> {
        val c = conversation
            ?: run {
                val err = BackendResult.Err(QuickAiError.NOT_INITIALIZED)
                sink.onError(err.error, err.message)
                return err
            }

        val prompt = messages.joinToString("\n") { msg ->
            "${msg.role}: ${msg.parts.filterIsInstance<PromptPart.Text>().joinToString("") { it.text }}"
        }

        return try {
            val message = c.sendMessage(prompt)
            val text = message.toString()
            if (text.isNotEmpty()) sink.onDelta(text)
            sink.onDone()
            BackendResult.Ok(Unit)
        } catch (t: Throwable) {
            val err = BackendResult.Err(QuickAiError.INFERENCE_FAILED, t.message)
            sink.onError(err.error, err.message)
            err
        }
    }

    /**
     * @brief Streaming multimodal inference with OpenAI message format.
     *
     * Accumulates deltas into a single response, then emits it through [sink].
     */
    override fun runMultimodalHandleWithMessagesStreaming(
        messages: List<QuickAiChatMessage>,
        sink: StreamSink
    ): BackendResult<Unit> {
        val c = conversation
            ?: run {
                val err = BackendResult.Err(QuickAiError.NOT_INITIALIZED)
                sink.onError(err.error, err.message)
                return err
            }
        if (!visionEnabled) {
            val err = BackendResult.Err(QuickAiError.UNSUPPORTED, "Vision not enabled")
            sink.onError(err.error, err.message)
            return err
        }

        // Validate image count (1 only)
        val imageCount = messages.sumOf { msg ->
            msg.parts.count { it is PromptPart.ImageBytes || it is PromptPart.ImageFile }
        }
        if (imageCount == 0) {
            val err = BackendResult.Err(QuickAiError.INVALID_PARAMETER, "No image found")
            sink.onError(err.error, err.message)
            return err
        }
        if (imageCount > 1) {
            val err = BackendResult.Err(QuickAiError.INVALID_PARAMETER, "Only 1 image is allowed")
            sink.onError(err.error, err.message)
            return err
        }

        val contents = try {
            toLiteRtContentsFromMessages(messages)
        } catch (t: Throwable) {
            val err = BackendResult.Err(QuickAiError.INVALID_PARAMETER, t.message)
            sink.onError(err.error, err.message)
            return err
        }

        return try {
            val message = c.sendMessage(contents)
            val text = message.toString()
            if (text.isNotEmpty()) sink.onDelta(text)
            sink.onDone()
            BackendResult.Ok(Unit)
        } catch (t: Throwable) {
            val err = BackendResult.Err(QuickAiError.INFERENCE_FAILED, t.message)
            sink.onError(err.error, err.message)
            err
        }
    }

    private fun toLiteRtContentsFromMessages(messages: List<QuickAiChatMessage>): Contents {
        val mapped: List<Content> = messages.flatMap { msg ->
            msg.parts.map { part ->
                when (part) {
                    is PromptPart.Text -> Content.Text(part.text)
                    is PromptPart.ImageFile -> {
                        val f = File(part.absolutePath)
                        require(f.exists() && f.canRead()) {
                            "PromptPart.ImageFile not readable: ${part.absolutePath}"
                        }
                        Content.ImageFile(part.absolutePath)
                    }
                    is PromptPart.ImageBytes -> {
                        require(part.bytes.isNotEmpty()) {
                            "PromptPart.ImageBytes has empty byte array"
                        }
                        Content.ImageBytes(part.bytes)
                    }
                    is PromptPart.PreprocessedPixels -> {
                        throw UnsupportedOperationException(
                            "PromptPart.PreprocessedPixels is not supported by LiteRTLm. " +
                            "Use NativeQuickDotAI for V-JEPA multi-image inference."
                        )
                    }
                }
            }
        }
        return Contents.of(mapped)
    }

    override fun cancel() {
        cancelRequested.set(true)
        Log.i(TAG, "cancel(): one-shot run cancel requested")
    }

    override fun chatCancel() {
        activeSession?.cancel()
            ?: Log.w(TAG, "chatCancel(): no active session")
    }

    override fun chatRebuild(
        messages: List<QuickAiChatMessage>
    ): BackendResult<Unit> {
        val session = activeSession
            ?: return BackendResult.Err(
                QuickAiError.BAD_REQUEST,
                "No active chat session — call openChatSession() first"
            )
        return try {
            session.rebuild(messages)
        } catch (t: Throwable) {
            Log.e(TAG, "chatRebuild(): threw", t)
            BackendResult.Err(QuickAiError.UNKNOWN, t.message)
        }
    }

    /**
     * Close the active chat session if any. Nulls [activeSession] first
     * so the session's onSessionClosed callback sees nothing to restore
     * — [unload] / [close] call [closeQuietly] right after, which tears
     * down the entire Engine.
     */
    private fun closeActiveSession() {
        val session = activeSession ?: return
        Log.i(TAG, "closeActiveSession(): closing ${session.sessionId}")
        activeSession = null   // detach first → callback skips restore
        session.close()
    }

    /**
     * Recreate `this.conversation` from the engine so the flat run()/
     * runStreaming() API is usable again after a chat session closes.
     */
    private fun restoreConversation() {
        if (conversation != null || engine == null) return
        try {
            conversation = engine!!.createConversation()
            Log.i(TAG, "restoreConversation(): flat-API Conversation recreated")
        } catch (t: Throwable) {
            Log.e(TAG, "restoreConversation(): failed to recreate Conversation", t)
        }
    }

    override fun unload(): BackendResult<Unit> {
        Log.i(TAG, "unload() invoked")
        // Cancel any in-flight inference before unloading
        cancel()

        closeActiveSession()
        closeQuietly()
        return BackendResult.Ok(Unit)
    }

    override fun metrics(): BackendResult<PerformanceMetrics> {
        if (engine == null) {
            return BackendResult.Err(
                QuickAiError.NOT_INITIALIZED,
                "LiteRTLm has not been loaded yet"
            )
        }
        // LiteRT-LM does not currently expose token-level counters
        // through its Kotlin API, so most fields stay at 0. We still
        // publish the wall-clock timings we measured ourselves so
        // callers can at least see the load + last-run durations.
        return BackendResult.Ok(
            PerformanceMetrics(
                initializationDurationMs = initializationDurationMs,
                totalDurationMs = lastRunDurationMs
            )
        )
    }

    override fun close() {
        Log.i(TAG, "close() invoked")
        closeActiveSession()
        closeQuietly()
    }

    private fun closeQuietly() {
        try {
            conversation?.close()
        } catch (t: Throwable) {
            Log.w(TAG, "conversation.close() threw", t)
        }
        conversation = null
        try {
            engine?.close()
        } catch (t: Throwable) {
            Log.w(TAG, "engine.close() threw", t)
        }
        engine = null
        visionEnabled = false
    }

    /**
     * @brief Map a QuickDotAI [BackendType] to a LiteRT-LM [LlmBackend].
     *
     * Extracted as a helper so both the compute and vision backends
     * use exactly the same mapping (including the NPU → CPU fallback)
     * and we never drift between the two.
     */
    private fun mapBackend(b: BackendType): LlmBackend = when (b) {
        BackendType.CPU -> LlmBackend.CPU()
        BackendType.GPU -> LlmBackend.GPU()
        // LiteRT-LM's NPU backend wants the dir holding the vendor
        // native .so files. For an app-bundled setup that is simply
        // the APK's nativeLibraryDir, but we don't have a Context
        // here — fall back to CPU until the caller wires one in.
        BackendType.NPU -> LlmBackend.CPU()
    }

    /**
     * @brief Convert a list of [PromptPart]s into a LiteRT-LM [Contents]
     * object ready to hand to `sendMessage` / `sendMessageAsync`.
     *
     * This is where the AAR-level public types cross the boundary into
     * the LiteRT-LM package. We also validate each part eagerly so the
     * caller gets a crisp error BEFORE we reach the native layer:
     *  - ImageFile: file must exist and be readable
     *  - ImageBytes: bytes must be non-empty
     *
     * Throws [IllegalArgumentException] on validation failure; the
     * calling runMultimodal* wrappers translate that into a
     * [QuickAiError.INVALID_PARAMETER] BackendResult.
     */
    private fun toLiteRtContents(parts: List<PromptPart>): Contents {
        val mapped: List<Content> = parts.map { p ->
            when (p) {
                is PromptPart.Text -> Content.Text(p.text)
                is PromptPart.ImageFile -> {
                    val f = File(p.absolutePath)
                    require(f.exists() && f.canRead()) {
                        "PromptPart.ImageFile not readable: ${p.absolutePath}"
                    }
                    Content.ImageFile(p.absolutePath)
                }
                is PromptPart.ImageBytes -> {
                    require(p.bytes.isNotEmpty()) {
                        "PromptPart.ImageBytes has empty byte array"
                    }
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
     * @brief Build the test-mode fallback model file handle from the shared
     * model directory at `/sdcard/Download/aistudio-mobile/models/`.
     * This ensures all team apps (AI Studio Mobile, SampleTestAPP, etc.)
     * can share the same model file without duplication.
     */
    private fun testModelFile(): File {
        val baseDir = File(defaultModelBasePath.trimEnd('/'))
        val dir = File(baseDir, TEST_GEMMA4_REL_DIR)
        if (!dir.exists()) dir.mkdirs()
        return File(dir, TEST_GEMMA4_FILE_NAME)
    }

    companion object {
        private const val TAG = "LiteRTLm"

        /**
         * @brief TEST ONLY — path components of the Gemma-4 E2B-IT
         * `.litertlm` model, relative to the shared model directory.
         * The absolute path is resolved at runtime via [testModelFile].
         *
         * Push the file with adb before running:
         *   adb push gemma-4-E2B-it.litertlm \
         *       /sdcard/Download/aistudio-mobile/models/gemma-4-E2B-it/
         */
        const val TEST_GEMMA4_REL_DIR: String = "gemma-4-E2B-it"
        const val TEST_GEMMA4_FILE_NAME: String = "gemma-4-E2B-it.litertlm"
    }
}
