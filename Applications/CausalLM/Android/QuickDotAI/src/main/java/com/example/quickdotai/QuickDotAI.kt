// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    QuickDotAI.kt
 * @brief   Public surface of the QuickDotAI AAR.
 *
 * QuickDotAI is a thin abstraction over a single loaded on-device
 * language model. Two concrete implementations are shipped in this AAR:
 *
 *  - [NativeQuickDotAI] — routes non-Gemma models through JNI to
 *    libquick_dot_ai_api.so, the handle-based C API built from
 *    Applications/CausalLM.
 *  - [LiteRTLm]         — routes Gemma-family models through the
 *    LiteRT-LM Kotlin API.
 *
 * Both implementations satisfy the same [QuickDotAI] contract so a host
 * app can pick an engine once at load time and then drive it through a
 * interface for handle-based inference (OpenAI tab), session-based
 * chat (Chat tab), and lifecycle management (load / unload / close).
 *
 * Threading: a [QuickDotAI] instance is NOT internally thread-safe. The
 * expectation is that the host app owns exactly one instance per loaded
 * model and drives it from a single worker thread — the same contract
 * that QuickAIService's ModelWorker implements, and the one the sample
 * app (SampleTestAPP) follows from its background dispatcher.
 */
package com.example.quickdotai

import android.content.Context

/**
 * @brief Outcome of a QuickDotAI call.
 *
 * Every public method returns a [BackendResult] so errors never
 * propagate out as exceptions across the AAR boundary. [Ok] carries the
 * successful value; [Err] carries a [QuickAiError] code and an optional
 * human-readable message.
 */
sealed class BackendResult<out T> {
    data class Ok<T>(val value: T) : BackendResult<T>()
    data class Err(
        val error: QuickAiError,
        val message: String? = null
    ) : BackendResult<Nothing>()
}

/**
 * @brief Where a [QuickDotAI] implementation pushes streamed output
 *        during [QuickDotAI.runStreaming].
 *
 * The contract is:
 *  - zero or more [onDelta] calls carrying newly-generated text,
 *    followed by
 *  - exactly one terminal call — either [onDone] on success or
 *    [onError] on failure.
 *
 * Implementations may be invoked from an implementation-internal
 * thread (LiteRT-LM for example dispatches MessageCallback on its own
 * worker thread). Host code that wants to marshal events back to the UI
 * thread must do that bridging itself — the AAR does not assume any
 * particular threading model on the consumer side.
 */
interface StreamSink {
    fun onDelta(text: String)
    fun onReasoningDelta(text: String) {
    }
    fun onDone()
    fun onError(error: QuickAiError, message: String?)
}

/**
 * @brief Common interface implemented by every QuickDotAI engine.
 *
 * Lifecycle: [load] exactly once, then inference calls, then [close]
 * exactly once. Calling any inference method before [load] returns a
 * [BackendResult.Err] with [QuickAiError.NOT_INITIALIZED].
 *
 * **Chat session lifecycle:** [openChatSession] → [runChatModelHandleStreaming] /
 * [runChatMultimodalHandleStreaming] / [chatCancel] / [chatRebuild] → [closeChatSession].
 * Only one session may be active at a time.
 */
interface QuickDotAI {
    /** @return a short identifier like "native" or "litert-lm". */
    val kind: String

    /** @return the architecture string reported by the engine, if any. */
    val architecture: String?

    /**
     * @return the sessionId of the currently active chat session, or
     * null if no session is open.
     */
    val chatSessionId: String?
        get() = null

    /**
     * @brief Load the model described by [req]. Must be called exactly
     * once before any inference call.
     */
    fun load(req: LoadModelRequest): BackendResult<Unit>

    /**
     * @brief Blocking multimodal inference — accepts a sequence of
     * [PromptPart]s that may interleave text and image inputs.
     *
     * The default implementation returns [QuickAiError.UNSUPPORTED]
     * because not every engine can handle non-text inputs. Concrete
     * implementations backed by multimodal-capable models (currently
     * [LiteRTLm] with a multimodal Gemma loaded through a non-null
     * [LoadModelRequest.visionBackend]) override this to do the real
     * work. [NativeQuickDotAI] inherits the UNSUPPORTED default, so
     * consumers get a clear error message instead of a silent failure
     * when they aim an image prompt at the text-only native engine.
     *
     * Contract:
     *  - [parts] must be non-empty; an empty list returns
     *    [QuickAiError.INVALID_PARAMETER].
     *  - Parts may appear in any order. The canonical Gemma-4 /
     *    Gemma3n convention is one or more image parts followed by a
     *    single trailing text instruction.
     *  - Must be called only after a successful [load]; calling it
     *    before [load] returns [QuickAiError.NOT_INITIALIZED].
     *
     * Example:
     * ```
     * val reply = engine.runMultimodalHandle(listOf(
     *     PromptPart.ImageFile("/sdcard/photo.jpg"),
     *     PromptPart.Text("What is happening in this picture?"),
     * ))
     * ```
     */
    fun runMultimodalHandle(parts: List<PromptPart>): BackendResult<String> =
        BackendResult.Err(
            QuickAiError.UNSUPPORTED,
            "runMultimodalHandle is not supported by engine '$kind'. " +
                "Load a multimodal-capable model (e.g. GEMMA4) with " +
                "LoadModelRequest.visionBackend set to a non-null value."
        )

    /**
     * @brief Streaming variant of [runMultimodalHandle].
     *
     * The default implementation returns [QuickAiError.UNSUPPORTED]
     * and delivers a single terminal [StreamSink.onError] before
     * returning, so callers can rely on the same StreamSink contract
     * as text-only streaming regardless of which engine they targeted.
     */
    fun runMultimodalHandleStreaming(
        parts: List<PromptPart>,
        sink: StreamSink
    ): BackendResult<Unit> {
        val err = BackendResult.Err(
            QuickAiError.UNSUPPORTED,
            "runMultimodalHandleStreaming is not supported by engine '$kind'. " +
                "Load a multimodal-capable model (e.g. GEMMA4) with " +
                "LoadModelRequest.visionBackend set to a non-null value."
        )
        sink.onError(err.error, err.message)
        return err
    }

    /**
     * @brief Unload the model weights without destroying the engine.
     *
     * After a successful unload the engine is in a "not initialized" state
     * — subsequent [run] / [runStreaming] / [metrics] calls will return
     * [QuickAiError.NOT_INITIALIZED]. The instance can still be [close]d
     * normally (and must be, to release any remaining resources).
     *
     * Implementations that do not support partial unload may treat this as
     * a full [close] or return [BackendResult.Ok] as a no-op.
     */
    fun unload(): BackendResult<Unit>

    /**
     * @brief Fetch performance metrics for the most recent run.
     */
    fun metrics(): BackendResult<PerformanceMetrics>

    /**
     * Encode [text] into a sentence-embedding vector. Only embedding models
     * (e.g. the Ouro family) support this; other engines return
     * [QuickAiError.INFERENCE_FAILED] by default.
     *
     * @return [BackendResult.Ok] with the embedding FloatArray, or
     *         [BackendResult.Err] on failure.
     */
    fun encode(text: String): BackendResult<FloatArray> =
        BackendResult.Err(
            QuickAiError.INFERENCE_FAILED,
            "encode() is not supported by this engine"
        )

    // ----- Chat session API ------------------------------------------------
    // All chat operations go through this interface so the app never needs
    // to interact with chat session classes directly.

    /**
     * @brief Open a new structured chat session on this engine.
     *
     * Only **one** session may be active at a time (LiteRT-LM allows a
     * single Conversation per Engine). If a session is already open,
     * this method returns [QuickAiError.BAD_REQUEST]. Returns the
     * session ID on success.
     */
    fun openChatSession(
        config: QuickAiChatSessionConfig? = null
    ): BackendResult<String> =
        BackendResult.Err(
            QuickAiError.UNSUPPORTED,
            "openChatSession is not supported by engine '$kind'."
        )

    /**
     * @brief Close the active chat session, releasing its resources
     * (conversation handle, cached images, etc.). After closing, the
     * flat [run] / [runStreaming] APIs become usable again.
     */
    fun closeChatSession(): BackendResult<Unit> =
        BackendResult.Err(
            QuickAiError.UNSUPPORTED,
            "closeChatSession is not supported by engine '$kind'."
        )

    /**
     * @brief Send a chat message in a session with streaming response.
     *
     * Requires an active session opened via [openChatSession].
     * The text message is converted internally to a structured format
     * and sent to the native engine.
     *
     * @param text Raw text input from the user
     * @param sink StreamSink to receive streaming output
     * @return BackendResult containing the chat result or an error
     */
    fun runChatModelHandleStreaming(
        text: String,
        sink: StreamSink
    ): BackendResult<QuickAiChatResult> {
        val err = BackendResult.Err(
            QuickAiError.UNSUPPORTED,
            "runChatModelHandleStreaming is not supported by engine '$kind'."
        )
        sink.onError(err.error, err.message)
        return err
    }

    /**
     * @brief Send a multimodal chat message (with image) in a session
     * with streaming response.
     *
     * Requires an active session opened via [openChatSession].
     *
     * @param parts List of PromptPart containing text and/or images
     * @param sink StreamSink to receive streaming output
     * @return BackendResult containing the chat result or an error
     */
    fun runChatMultimodalHandleStreaming(
        parts: List<PromptPart>,
        sink: StreamSink
    ): BackendResult<QuickAiChatResult> {
        val err = BackendResult.Err(
            QuickAiError.UNSUPPORTED,
            "runChatMultimodalHandleStreaming is not supported by engine '$kind'."
        )
        sink.onError(err.error, err.message)
        return err
    }

    /**
     * @brief Cancel an in-flight [runStreaming] or [runMultimodalStreaming].
     * Safe to call from any thread. No-op if no generation is running.
     */
    fun cancel() { /* no-op by default */ }

    /**
     * @brief Cancel an in-flight [chatRun] or [chatRunStreaming].
     * Safe to call from any thread. No-op if no generation is running.
     */
    fun chatCancel() { /* no-op by default */ }

    /**
     * @brief Reset the active session: drop the backend's KV cache and
     * optionally pre-seed a fresh conversation with [messages] as
     * initial turns. Pass `emptyList()` to simply clear the session.
     * Use this after history edits, sampling changes, or to recover
     * from a failed/cancelled turn.
     */
    fun chatRebuild(
        messages: List<QuickAiChatMessage>
    ): BackendResult<Unit> =
        BackendResult.Err(
            QuickAiError.UNSUPPORTED,
            "chatRebuild is not supported by engine '$kind'."
        )

    // ----- Handle-based OpenAI messages API (streaming only) ----------

    /**
     * @brief Streaming inference with OpenAI message format on a specific handle.
     *
     * @param messages List of chat messages with role (system/user/assistant) and content
     * @param sink StreamSink to receive streaming output
     * @return BackendResult<Unit> on completion
     */
    fun runModelHandleWithMessagesStreaming(
        messages: List<QuickAiChatMessage>,
        sink: StreamSink
    ): BackendResult<Unit> {
        val err = BackendResult.Err(
            QuickAiError.UNSUPPORTED,
            "runModelHandleWithMessagesStreaming is not supported by engine '$kind'."
        )
        sink.onError(err.error, err.message)
        return err
    }

    /**
     * @brief Streaming multimodal inference with OpenAI message format on a specific handle.
     *
     * @param messages List of chat messages. Image should be included as ImageBytes part.
     * @param sink StreamSink to receive streaming output
     * @return BackendResult<Unit> on completion
     */
    fun runMultimodalHandleWithMessagesStreaming(
        messages: List<QuickAiChatMessage>,
        sink: StreamSink
    ): BackendResult<Unit> {
        val err = BackendResult.Err(
            QuickAiError.UNSUPPORTED,
            "runMultimodalHandleWithMessagesStreaming is not supported by engine '$kind'."
        )
        sink.onError(err.error, err.message)
        return err
    }

    /**
     * @brief Streaming inference with OpenAI JSON format.
     *
     * Accepts a JSON string in OpenAI format and processes it through the
     * chat template. Supports messages, tools, functions, and all other
     * fields recognized by minja chat template renderer.
     *
     * Example JSON input:
     * ```
     * {
     *   "messages": [
     *     {"role": "developer", "content": "..."},
     *     {"role": "user", "content": "..."}
     *   ],
     *   "tools": [
     *     {"type": "function", "function": {"name": "call", "description": "..."}}
     *   ]
     * }
     * ```
     *
     * @param jsonRequest OpenAI format JSON string
     * @param sink StreamSink to receive streaming output
     * @return BackendResult<Unit> on completion
     */
    fun runModelHandleWithJsonStreaming(
        jsonRequest: String,
        sink: StreamSink
    ): BackendResult<Unit> {
        val err = BackendResult.Err(
            QuickAiError.UNSUPPORTED,
            "runModelHandleWithJsonStreaming is not supported by engine '$kind'."
        )
        sink.onError(err.error, err.message)
        return err
    }

    /**
     * @brief Release all resources. Idempotent — safe to call more
     * than once.
     */
    fun close()
}

/**
 * Factory: create the right engine for a [ModelDescriptor].
 */
fun createEngine(
    context: Context,
    descriptor: ModelDescriptor,
    modelBasePath: String? = null
): QuickDotAI =
    when (descriptor.runtime) {
        RuntimeKind.LITERT -> LiteRTLm(
            context,
            defaultModelBasePath = modelBasePath ?: "/sdcard/Download/aistudio-mobile/models/"
        )
        RuntimeKind.NATIVE -> NativeQuickDotAI(context)
    }
