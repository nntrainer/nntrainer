// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    NativeQuickDotAI.kt
 * @brief   QuickDotAI implementation backed by the handle-based
 *          quick_dot_ai_api.h (routed through libquickai_jni.so → JNI →
 *          libquick_dot_ai_api.so).
 */
package com.example.quickdotai

import android.content.Context
import android.graphics.BitmapFactory
import android.util.Log
import java.io.File

/**
 * @brief Kotlin wrapper around a single `CausalLmHandle` in native code.
 *
 * Non-thread-safe by design — the host app must drive a single instance
 * from a single worker thread.
 *
 * @param appContext Application context required for multimodal image processing.
 *                   Must be non-null to enable runMultimodal/runMultimodalStreaming.
 */
class NativeQuickDotAI(
    private val appContext: Context
) : QuickDotAI {

    override val kind: String = "native"

    override var architecture: String? = null
        private set

    private var handle: Long = 0L
    private var loaded: Boolean = false

    // Image processor for multimodal inference
    private var imageProcessor: LlavaNextImageProcessor? = null

    // Vision backend type (null = text-only mode)
    private var visionBackend: BackendType? = null

    // Currently loaded model ID — used to route multi-image vs single-image paths
    private var currentModelId: String? = null

    override fun load(req: LoadModelRequest): BackendResult<Unit> {
        Log.i(
            TAG,
            "load() entered: modelId=${req.modelId} backend=${req.backend} " +
                "quant=${req.quantization}"
        )
        if (loaded) {
            Log.i(TAG, "load(): already loaded, returning Ok")
            return BackendResult.Ok(Unit)
        }

        if (!req.htpBackendConfigPath.isNullOrBlank()) {
            Log.w(TAG, "load(): htpBackendConfigPath='${req.htpBackendConfigPath}' " +
                "is not forwarded by the byName load path; " +
                "C layer will derive HTP config from modelBasePath.")
        }

        if (!NativeCausalLm.ensureLoaded()) {
            Log.e(TAG, "load(): native libs unavailable on this device")
            return BackendResult.Err(
                QuickAiError.MODEL_LOAD_FAILED,
                "libquickai_jni.so / libquick_dot_ai_api.so not available on this device"
            )
        }

        // modelBasePath is passed directly from the caller. The C API uses
        // this as the base directory for resolving model directories
        // (e.g. "<model_base_path>/qwen3-0.6b").
        val modelBasePath = req.modelBasePath
        if (modelBasePath == null || modelBasePath.isBlank()) {
            Log.w(
                TAG,
                "load(): modelBasePath is null/blank — C API will use its default " +
                    "fallback path. Specify modelBasePath for shared model access."
            )
        } else {
            Log.i(TAG, "load(): modelBasePath=$modelBasePath")
        }

        return try {
            Log.i(TAG, "load(): calling loadModelHandleByNameNative(backend=${req.backend.ordinal}, " +
                "modelId=${req.modelId}, quant=${req.quantization.ordinal}, " +
                "nativeLibDir=${req.nativeLibDir}, modelBasePath=$modelBasePath)")
            val h = NativeCausalLm.loadModelHandleByNameNative(
                backend = mapBackend(req.backend),
                modelId = req.modelId,
                quant = mapQuant(req.quantization),
                nativeLibDir = req.nativeLibDir,
                modelBasePath = modelBasePath,
            )
            if (h == 0L) {
                Log.e(TAG, "load(): loadModelHandleByNameNative returned 0 for '${req.modelId}'")
                BackendResult.Err(
                    QuickAiError.MODEL_LOAD_FAILED,
                    "loadModelHandleByName failed for '${req.modelId}'"
                )
            } else {
                val sdResult = NativeCausalLm.configureSpeculativeDecodingNative(h, req.useSpeculativeDecoding)
                if (sdResult != 0) {
                    Log.e(TAG, "load(): configureSpeculativeDecoding failed (code=$sdResult) for '${req.modelId}'")
                    NativeCausalLm.destroyModelHandleNative(h)
                    return BackendResult.Err(
                        QuickAiError.MODEL_LOAD_FAILED,
                        "Speculative decoding not supported for '${req.modelId}'"
                    )
                }
                handle = h
                loaded = true
                architecture = req.modelId
                currentModelId = req.modelId
                visionBackend = req.visionBackend
                if (req.visionBackend != null) {
                    imageProcessor = LlavaNextImageProcessor(appContext)
                    Log.i(TAG, "load(): visionBackend=${req.visionBackend}, image processor initialized")
                }
                Log.i(TAG, "load(): SUCCESS, handle=0x${h.toString(16)}, sd=${req.useSpeculativeDecoding}")
                BackendResult.Ok(Unit)
            }
        } catch (t: Throwable) {
            Log.e(TAG, "load(): loadModelHandleByNameNative threw", t)
            BackendResult.Err(QuickAiError.MODEL_LOAD_FAILED, t.message)
        }
    }

    override fun metrics(): BackendResult<PerformanceMetrics> {
        if (!loaded || handle == 0L) {
            return BackendResult.Err(QuickAiError.NOT_INITIALIZED)
        }
        return try {
            val m = NativeCausalLm.getPerformanceMetricsHandleNative(handle)
            if (m.errorCode != 0) {
                BackendResult.Err(QuickAiError.fromNativeCode(m.errorCode))
            } else {
                BackendResult.Ok(
                    PerformanceMetrics(
                        prefillTokens = m.prefillTokens,
                        prefillDurationMs = m.prefillDurationMs,
                        generationTokens = m.generationTokens,
                        generationDurationMs = m.generationDurationMs,
                        totalDurationMs = m.totalDurationMs,
                        initializationDurationMs = m.initializationDurationMs,
                        peakMemoryKb = m.peakMemoryKb
                    )
                )
            }
        } catch (t: Throwable) {
            Log.e(TAG, "getPerformanceMetricsHandleNative threw", t)
            BackendResult.Err(QuickAiError.UNKNOWN, t.message)
        }
    }

    override fun encode(text: String): BackendResult<FloatArray> {
        if (handle == 0L) {
            return BackendResult.Err(
                QuickAiError.NOT_INITIALIZED,
                "encode(): no model loaded"
            )
        }
        return try {
            val vec = NativeCausalLm.encodeModelHandleNative(handle, text)
            if (vec == null || vec.isEmpty()) {
                BackendResult.Err(
                    QuickAiError.INFERENCE_FAILED,
                    "encode() failed for current model '$currentModelId'"
                )
            } else {
                BackendResult.Ok(vec)
            }
        } catch (t: Throwable) {
            Log.e(TAG, "encode() threw", t)
            BackendResult.Err(QuickAiError.INFERENCE_FAILED, t.message)
        }
    }

    override fun unload(): BackendResult<Unit> {
        // Cancel any in-flight inference before unloading
        cancel()
        activeSession?.close()
        activeSession = null

        if (!loaded || handle == 0L) {
            return BackendResult.Ok(Unit)
        }
        return try {
            val ec = NativeCausalLm.unloadModelHandleNative(handle)
            loaded = false
            if (ec != 0) {
                BackendResult.Err(QuickAiError.fromNativeCode(ec))
            } else {
                BackendResult.Ok(Unit)
            }
        } catch (t: Throwable) {
            Log.w(TAG, "unloadModelHandleNative threw", t)
            BackendResult.Err(QuickAiError.UNKNOWN, t.message)
        }
    }

    // --- chat session (dummy) --------------------------------------------

    private var activeSession: NativeChatSession? = null

    override val chatSessionId: String?
        get() = activeSession?.sessionId

    override fun openChatSession(
        config: QuickAiChatSessionConfig?
    ): BackendResult<String> {
        if (!loaded || handle == 0L) {
            return BackendResult.Err(
                QuickAiError.NOT_INITIALIZED,
                "NativeQuickDotAI has not been loaded yet"
            )
        }
        if (activeSession != null) {
            return BackendResult.Err(
                QuickAiError.BAD_REQUEST,
                "A chat session is already active (${activeSession!!.sessionId}). " +
                    "Close it before opening a new one."
            )
        }
        val session = NativeChatSession(
            handleProvider = { handle },
            config = config
        )
        activeSession = session
        Log.i(TAG, "openChatSession(): created session ${session.sessionId} with handle=0x${handle.toString(16)}")
        return BackendResult.Ok(session.sessionId)
    }

    override fun closeChatSession(): BackendResult<Unit> {
        val session = activeSession
        if (session == null) {
            return BackendResult.Err(
                QuickAiError.BAD_REQUEST,
                "No active chat session to close"
            )
        }
        session.close()
        activeSession = null
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
        return session.runStreaming(text, sink)
    }

    override fun runChatMultimodalHandleStreaming(
        parts: List<PromptPart>,
        sink: StreamSink
    ): BackendResult<QuickAiChatResult> {
        if (activeSession == null) {
            val err = BackendResult.Err(
                QuickAiError.BAD_REQUEST,
                "No active chat session — call openChatSession() first"
            )
            sink.onError(err.error, err.message)
            return err
        }
        val accumulated = StringBuilder()
        val forwardingSink = object : StreamSink {
            override fun onDelta(text: String) {
                accumulated.append(text)
                sink.onDelta(text)
            }

            override fun onReasoningDelta(text: String) {
                sink.onReasoningDelta(text)
            }

            override fun onDone() {
                sink.onDone()
            }

            override fun onError(error: QuickAiError, message: String?) {
                sink.onError(error, message)
            }
        }
        val messages = listOf(
            QuickAiChatMessage(role = QuickAiChatRole.USER, parts = parts)
        )
        return when (val r = runMultimodalHandleWithMessagesStreaming(messages, forwardingSink)) {
            is BackendResult.Ok -> {
                val metrics = when (val m = metrics()) {
                    is BackendResult.Ok -> m.value
                    is BackendResult.Err -> null
                }
                BackendResult.Ok(
                    QuickAiChatResult(
                        content = accumulated.toString(),
                        metrics = metrics
                    )
                )
            }
            is BackendResult.Err -> BackendResult.Err(r.error, r.message)
        }
    }

    override fun cancel() {
        Log.d(TAG, "cancel(): START, handle=0x${handle.toString(16)}")
        if (handle != 0L) {
            Log.d(TAG, "cancel(): calling NativeCausalLm.cancelModelHandleNative(handle=0x${handle.toString(16)})")
            val result = NativeCausalLm.cancelModelHandleNative(handle)
            Log.d(TAG, "cancel(): cancelModelHandleNative returned $result")
        } else {
            Log.w(TAG, "cancel(): no valid handle to cancel")
        }
    }

    override fun chatCancel() {
        activeSession?.cancel()
    }

    override fun chatRebuild(
        messages: List<QuickAiChatMessage>
    ): BackendResult<Unit> {
        val session = activeSession
            ?: return BackendResult.Err(
                QuickAiError.BAD_REQUEST,
                "No active chat session — call openChatSession() first"
            )
        return session.rebuild(messages)
    }

    // --- OpenAI messages API (handle-based) --------------------------------

    override fun runModelHandleWithMessagesStreaming(
        messages: List<QuickAiChatMessage>,
        sink: StreamSink
    ): BackendResult<Unit> {
        if (!loaded || handle == 0L) {
            val err = BackendResult.Err(
                QuickAiError.NOT_INITIALIZED,
                "NativeQuickDotAI has not been loaded yet"
            )
            sink.onError(err.error, err.message)
            return err
        }

        return try {
            val errorCode = NativeCausalLm.runModelHandleWithMessagesStreamingNative(
                handle = handle,
                messages = messages.toTypedArray(),
                addGenerationPrompt = true,
                listener = object : NativeCausalLm.NativeStreamListener {
                    override fun onDelta(text: String) {
                        sink.onDelta(text)
                    }
                }
            )
            if (errorCode != 0) {
                val err = QuickAiError.fromNativeCode(errorCode)
                sink.onError(err, "runModelHandleWithMessagesStreaming failed (errorCode=$errorCode)")
                BackendResult.Err(err, "runModelHandleWithMessagesStreaming failed (errorCode=$errorCode)")
            } else {
                sink.onDone()
                BackendResult.Ok(Unit)
            }
        } catch (t: Throwable) {
            sink.onError(QuickAiError.INFERENCE_FAILED, t.message)
            BackendResult.Err(QuickAiError.INFERENCE_FAILED, t.message)
        }
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
     * @param sink StreamSink for receiving streaming output
     * @return BackendResult<Unit>
     */
    override fun runModelHandleWithJsonStreaming(
        jsonRequest: String,
        sink: StreamSink
    ): BackendResult<Unit> {
        if (!loaded || handle == 0L) {
            val err = BackendResult.Err(
                QuickAiError.NOT_INITIALIZED,
                "NativeQuickDotAI has not been loaded yet"
            )
            sink.onError(err.error, err.message)
            return err
        }

        return try {
            val errorCode = NativeCausalLm.runModelHandleWithJsonStreamingNative(
                handle = handle,
                jsonRequest = jsonRequest,
                listener = object : NativeCausalLm.NativeStreamListener {
                    override fun onDelta(text: String) {
                        sink.onDelta(text)
                    }
                }
            )
            if (errorCode != 0) {
                val err = QuickAiError.fromNativeCode(errorCode)
                sink.onError(err, "runModelHandleWithJsonStreaming failed (errorCode=$errorCode)")
                BackendResult.Err(err, "runModelHandleWithJsonStreaming failed (errorCode=$errorCode)")
            } else {
                sink.onDone()
                BackendResult.Ok(Unit)
            }
        } catch (t: Throwable) {
            sink.onError(QuickAiError.INFERENCE_FAILED, t.message)
            BackendResult.Err(QuickAiError.INFERENCE_FAILED, t.message)
        }
    }

    /**
     * @brief Streaming multimodal inference with OpenAI message format on a specific handle.
     */
    override fun runMultimodalHandleWithMessagesStreaming(
        messages: List<QuickAiChatMessage>,
        sink: StreamSink
    ): BackendResult<Unit> {
        if (!loaded || handle == 0L) {
            val err = BackendResult.Err(
                QuickAiError.NOT_INITIALIZED,
                "NativeQuickDotAI has not been loaded yet"
            )
            sink.onError(err.error, err.message)
            return err
        }

        val processor = imageProcessor
        if (processor == null) {
            val err = BackendResult.Err(
                QuickAiError.UNSUPPORTED,
                "Vision model not loaded"
            )
            sink.onError(err.error, err.message)
            return err
        }

        // Extract image from messages
        val allParts = messages.flatMap { it.parts }
        val imageParts = allParts.filter { it is PromptPart.ImageBytes || it is PromptPart.ImageFile || it is PromptPart.PreprocessedPixels }

        if (imageParts.isEmpty()) {
            val err = BackendResult.Err(
                QuickAiError.INVALID_PARAMETER,
                "No image found. Expected parts: [Text, ImageBytes]"
            )
            sink.onError(err.error, err.message)
            return err
        }

        val multimodalInput = prepareMultimodalInput(allParts, processor)
        if (multimodalInput == null) {
            val err = BackendResult.Err(
                QuickAiError.INVALID_PARAMETER,
                "Image preprocessing failed"
            )
            sink.onError(err.error, err.message)
            return err
        }

        return try {
            val errorCode = if (multimodalInput.numImages > 1 && multimodalInput.patchesPerImage != null) {
                // Multi-image path (V-JEPA)
                Log.i(TAG, "runMultimodalHandleWithMessagesStreaming(): using multi-image path, numImages=${multimodalInput.numImages}")
                NativeCausalLm.runMultimodalMultiImageWithMessagesStreamingNative(
                    handle = handle,
                    messages = messages.toTypedArray(),
                    addGenerationPrompt = true,
                    pixelValues = multimodalInput.pixelValues,
                    numPatches = multimodalInput.numPatches,
                    numImages = multimodalInput.numImages,
                    patchesPerImage = multimodalInput.patchesPerImage,
                    originalHeights = multimodalInput.originalHeights ?: IntArray(multimodalInput.numImages) { multimodalInput.originalHeight },
                    originalWidths = multimodalInput.originalWidths ?: IntArray(multimodalInput.numImages) { multimodalInput.originalWidth },
                    listener = object : NativeCausalLm.NativeStreamListener {
                        override fun onDelta(text: String) {
                            sink.onDelta(text)
                        }
                    }
                )
            } else {
                // Single-image path (legacy)
                NativeCausalLm.runMultimodalHandleWithMessagesStreamingNative(
                    handle = handle,
                    messages = messages.toTypedArray(),
                    addGenerationPrompt = true,
                    pixelValues = multimodalInput.pixelValues,
                    numPatches = multimodalInput.numPatches,
                    originalHeight = multimodalInput.originalHeight,
                    originalWidth = multimodalInput.originalWidth,
                    listener = object : NativeCausalLm.NativeStreamListener {
                        override fun onDelta(text: String) {
                            sink.onDelta(text)
                        }
                    }
                )
            }
            if (errorCode != 0) {
                val err = QuickAiError.fromNativeCode(errorCode)
                sink.onError(err, "runMultimodalHandleWithMessagesStreaming failed (errorCode=$errorCode)")
                BackendResult.Err(err, "runMultimodalHandleWithMessagesStreaming failed (errorCode=$errorCode)")
            } else {
                sink.onDone()
                BackendResult.Ok(Unit)
            }
        } catch (t: Throwable) {
            sink.onError(QuickAiError.INFERENCE_FAILED, t.message)
            BackendResult.Err(QuickAiError.INFERENCE_FAILED, t.message)
        }
    }

    override fun close() {
        activeSession?.close()
        activeSession = null
        if (handle != 0L) {
            try {
                NativeCausalLm.destroyModelHandleNative(handle)
            } catch (t: Throwable) {
                Log.w(TAG, "destroyModelHandleNative threw", t)
            }
            handle = 0L
        }
        loaded = false
    }

    // --- multimodal -------------------------------------------------------

    /**
     * @brief Blocking multimodal inference.
     *
     * Preprocesses images from [parts], combines with text prompt, and
     * runs inference through the native engine.
     *
     * @param parts List of PromptPart containing text and/or images
     * @return BackendResult with generated text on success
     */
    override fun runMultimodalHandle(parts: List<PromptPart>): BackendResult<String> {
        if (!loaded || handle == 0L) {
            return BackendResult.Err(
                QuickAiError.NOT_INITIALIZED,
                "NativeQuickDotAI has not been loaded yet"
            )
        }

        val processor = imageProcessor
        if (processor == null) {
            return BackendResult.Err(
                QuickAiError.UNSUPPORTED,
                "Multimodal not enabled — reload with LoadModelRequest.visionBackend set"
            )
        }

        // Extract image and text from parts
        val multimodalInput = prepareMultimodalInput(parts, processor)
            ?: return BackendResult.Err(
                QuickAiError.INVALID_PARAMETER,
                "No valid image found in parts"
            )

        val textPrompt = extractTextPrompt(parts)

        Log.i(
            TAG,
            "runMultimodal(): numPatches=${multimodalInput.numPatches}, " +
                "originalSize=${multimodalInput.originalHeight}x${multimodalInput.originalWidth}, " +
                "prompt length=${textPrompt.length}"
        )

        return try {
            val accumulated = StringBuilder()
            val errorCode = NativeCausalLm.runMultimodalHandleStreamingNative(
                handle,
                textPrompt,
                multimodalInput.pixelValues,
                multimodalInput.numPatches,
                multimodalInput.originalHeight,
                multimodalInput.originalWidth,
                object : NativeCausalLm.NativeStreamListener {
                    override fun onDelta(text: String) {
                        accumulated.append(text)
                    }
                }
            )
            if (errorCode != 0) {
                val err = QuickAiError.fromNativeCode(errorCode)
                Log.e(TAG, "runMultimodal(): failed with errorCode=$errorCode")
                BackendResult.Err(err, "runMultimodalHandle failed (errorCode=$errorCode)")
            } else {
                val output = accumulated.toString()
                Log.i(TAG, "runMultimodal(): success, output length=${output.length}")
                BackendResult.Ok(output)
            }
        } catch (t: Throwable) {
            Log.e(TAG, "runMultimodal(): threw exception", t)
            BackendResult.Err(QuickAiError.INFERENCE_FAILED, t.message)
        }
    }

    /**
     * @brief Streaming multimodal inference.
     *
     * Preprocesses images from [parts], combines with text prompt, and
     * runs streaming inference through the native engine. Deltas are
     * forwarded to [sink] as they are generated.
     *
     * @param parts List of PromptPart containing text and/or images
     * @param sink StreamSink to receive streaming output
     * @return BackendResult<Unit> on completion
     */
    override fun runMultimodalHandleStreaming(
        parts: List<PromptPart>,
        sink: StreamSink
    ): BackendResult<Unit> {
        if (!loaded || handle == 0L) {
            val err = BackendResult.Err(
                QuickAiError.NOT_INITIALIZED,
                "NativeQuickDotAI has not been loaded yet"
            )
            sink.onError(err.error, err.message)
            return err
        }

        val processor = imageProcessor
        if (processor == null) {
            val err = BackendResult.Err(
                QuickAiError.UNSUPPORTED,
                "MultimodalStreaming not enabled — reload with LoadModelRequest.visionBackend set"
            )
            sink.onError(err.error, err.message)
            return err
        }

        // Extract image and text from parts
        val multimodalInput = prepareMultimodalInput(parts, processor)
        if (multimodalInput == null) {
            val err = BackendResult.Err(
                QuickAiError.INVALID_PARAMETER,
                "No valid image found in parts"
            )
            sink.onError(err.error, err.message)
            return err
        }

        val textPrompt = extractTextPrompt(parts)

        Log.i(
            TAG,
            "runMultimodalStreaming(): numPatches=${multimodalInput.numPatches}, " +
                "originalSize=${multimodalInput.originalHeight}x${multimodalInput.originalWidth}, " +
                "prompt length=${textPrompt.length}"
        )

        return try {
            val errorCode = if (multimodalInput.numImages > 1 && multimodalInput.patchesPerImage != null) {
                // Multi-image path (V-JEPA)
                Log.i(TAG, "runMultimodalStreaming(): using multi-image path, numImages=${multimodalInput.numImages}")
                NativeCausalLm.runMultimodalMultiImageStreamingNative(
                    handle,
                    textPrompt,
                    multimodalInput.pixelValues,
                    multimodalInput.numPatches,
                    multimodalInput.numImages,
                    multimodalInput.patchesPerImage,
                    multimodalInput.originalHeights ?: IntArray(multimodalInput.numImages) { multimodalInput.originalHeight },
                    multimodalInput.originalWidths ?: IntArray(multimodalInput.numImages) { multimodalInput.originalWidth },
                ) { delta ->
                    sink.onDelta(delta)
                }
            } else {
                // Single-image path (legacy)
                NativeCausalLm.runMultimodalHandleStreamingNative(
                    handle,
                    textPrompt,
                    multimodalInput.pixelValues,
                    multimodalInput.numPatches,
                    multimodalInput.originalHeight,
                    multimodalInput.originalWidth
                ) { delta ->
                    sink.onDelta(delta)
                }
            }

            if (errorCode != 0) {
                val err = QuickAiError.fromNativeCode(errorCode)
                Log.e(TAG, "runMultimodalStreaming(): failed with errorCode=$errorCode")
                sink.onError(err, "runMultimodalHandleStreaming failed (errorCode=$errorCode)")
                BackendResult.Err(err, "runMultimodalHandleStreaming failed (errorCode=$errorCode)")
            } else {
                Log.i(TAG, "runMultimodalStreaming(): success")
                sink.onDone()
                BackendResult.Ok(Unit)
            }
        } catch (t: Throwable) {
            Log.e(TAG, "runMultimodalStreaming(): threw exception", t)
            sink.onError(QuickAiError.INFERENCE_FAILED, t.message)
            BackendResult.Err(QuickAiError.INFERENCE_FAILED, t.message)
        }
    }

    /**
     * @brief Prepare multimodal input from PromptPart list.
     *
     * Extracts images from parts and preprocesses them using
     * LlavaNextImageProcessor. Supports both single-image and
     * multi-image (V-JEPA) scenarios:
     * - Single image: returns a MultimodalInput with numImages=1 (default)
     * - Multiple ImageBytes: preprocesses each image, concatenates pixel
     *   values, and returns a multi-image MultimodalInput
     * - PreprocessedPixels: passed through directly
     *
     * @return MultimodalInput with preprocessed pixel values, or null if no image found
     */
    private fun prepareMultimodalInput(
        parts: List<PromptPart>,
        processor: LlavaNextImageProcessor
    ): NativeCausalLm.MultimodalInput? {
        // Collect all image parts first
        val imageParts = mutableListOf<PromptPart>()
        for (part in parts) {
            when (part) {
                is PromptPart.ImageFile -> imageParts.add(part)
                is PromptPart.ImageBytes -> imageParts.add(part)
                is PromptPart.PreprocessedPixels -> {
                    // PreprocessedPixels bypass the image processor entirely
                    return NativeCausalLm.MultimodalInput(
                        pixelValues = part.pixelValues,
                        numPatches = part.numPatches,
                        originalHeight = part.imageHeights.firstOrNull() ?: 0,
                        originalWidth = part.imageWidths.firstOrNull() ?: 0,
                        numImages = part.numImages,
                        patchesPerImage = part.patchesPerImage,
                        originalHeights = part.imageHeights,
                        originalWidths = part.imageWidths
                    )
                }
                is PromptPart.Text -> { /* skip text parts */ }
            }
        }

        if (imageParts.isEmpty()) return null

        // Single image: use the original single-image path
        if (imageParts.size == 1) {
            return preprocessSingleImage(imageParts[0], processor)
        }

        // Multiple images: preprocess each and concatenate
        val allPixelValues = mutableListOf<Float>()
        val patchesPerImageList = mutableListOf<Int>()
        val heightsList = mutableListOf<Int>()
        val widthsList = mutableListOf<Int>()
        var totalPatches = 0
        val cropSize = processor.getCropSize()
        val patchSize = cropSize * cropSize * 3

        for (imgPart in imageParts) {
            val bitmap = when (imgPart) {
                is PromptPart.ImageFile -> {
                    val file = File(imgPart.absolutePath)
                    if (!file.exists() || !file.canRead()) {
                        Log.w(TAG, "Image file not readable: ${imgPart.absolutePath}")
                        continue
                    }
                    BitmapFactory.decodeFile(imgPart.absolutePath)
                }
                is PromptPart.ImageBytes -> {
                    if (imgPart.bytes.isEmpty()) {
                        Log.w(TAG, "Image bytes are empty")
                        continue
                    }
                    BitmapFactory.decodeByteArray(imgPart.bytes, 0, imgPart.bytes.size)
                }
                else -> null
            }
            if (bitmap == null) {
                Log.w(TAG, "Failed to decode image in multi-image batch")
                continue
            }
            val modelInput = processor.preprocess(bitmap)
            val numPatches = modelInput.pixelValues.size / patchSize
            allPixelValues.addAll(modelInput.pixelValues.toList())
            patchesPerImageList.add(numPatches)
            heightsList.add(modelInput.originalSize.first)
            widthsList.add(modelInput.originalSize.second)
            totalPatches += numPatches
        }

        if (allPixelValues.isEmpty()) return null

        val numImages = patchesPerImageList.size
        Log.i(TAG, "prepareMultimodalInput(): multi-image mode, numImages=$numImages, " +
            "totalPatches=$totalPatches, patchesPerImage=$patchesPerImageList")

        return NativeCausalLm.MultimodalInput(
            pixelValues = allPixelValues.toFloatArray(),
            numPatches = totalPatches,
            originalHeight = heightsList.firstOrNull() ?: 0,
            originalWidth = widthsList.firstOrNull() ?: 0,
            numImages = numImages,
            patchesPerImage = patchesPerImageList.toIntArray(),
            originalHeights = heightsList.toIntArray(),
            originalWidths = widthsList.toIntArray()
        )
    }

    /**
     * @brief Preprocess a single image part into a MultimodalInput.
     */
    private fun preprocessSingleImage(
        part: PromptPart,
        processor: LlavaNextImageProcessor
    ): NativeCausalLm.MultimodalInput? {
        when (part) {
            is PromptPart.ImageFile -> {
                val file = File(part.absolutePath)
                if (!file.exists() || !file.canRead()) {
                    Log.w(TAG, "Image file not readable: ${part.absolutePath}")
                    return null
                }
                val bitmap = BitmapFactory.decodeFile(part.absolutePath)
                if (bitmap == null) {
                    Log.w(TAG, "Failed to decode image: ${part.absolutePath}")
                    return null
                }
                val modelInput = processor.preprocess(bitmap)
                return NativeCausalLm.MultimodalInput(
                    pixelValues = modelInput.pixelValues,
                    numPatches = modelInput.pixelValues.size / (processor.getCropSize() * processor.getCropSize() * 3),
                    originalHeight = modelInput.originalSize.first,
                    originalWidth = modelInput.originalSize.second
                )
            }
            is PromptPart.ImageBytes -> {
                if (part.bytes.isEmpty()) {
                    Log.w(TAG, "Image bytes are empty")
                    return null
                }
                val bitmap = BitmapFactory.decodeByteArray(part.bytes, 0, part.bytes.size)
                if (bitmap == null) {
                    Log.w(TAG, "Failed to decode image from bytes")
                    return null
                }
                val modelInput = processor.preprocess(bitmap)
                return NativeCausalLm.MultimodalInput(
                    pixelValues = modelInput.pixelValues,
                    numPatches = modelInput.pixelValues.size / (processor.getCropSize() * processor.getCropSize() * 3),
                    originalHeight = modelInput.originalSize.first,
                    originalWidth = modelInput.originalSize.second
                )
            }
            else -> return null
        }
    }

    /**
     * @brief Extract text prompt from PromptPart list.
     *
     * Concatenates all Text parts into a single prompt string.
     */
    private fun extractTextPrompt(parts: List<PromptPart>): String {
        return parts.filterIsInstance<PromptPart.Text>()
            .joinToString(" ") { it.text }
            .ifEmpty { "Describe this image." }
    }

    private fun mapBackend(b: BackendType): Int = when (b) {
        BackendType.CPU -> 0
        BackendType.GPU -> 1
        BackendType.NPU -> 2
    }

    private fun mapQuant(q: QuantizationType): Int = when (q) {
        QuantizationType.UNKNOWN -> 0
        QuantizationType.W4A32 -> 1
        QuantizationType.W16A16 -> 2
        QuantizationType.W8A16 -> 3
        QuantizationType.W32A32 -> 4
    }

    companion object {
        private const val TAG = "NativeQuickDotAI"
    }
}
