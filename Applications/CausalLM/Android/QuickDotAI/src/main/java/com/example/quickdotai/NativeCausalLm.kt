// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    NativeCausalLm.kt
 * @brief   JNI bindings for the handle-based QuickDotAI API.
 */
package com.example.quickdotai

/**
 * @brief Low-level JNI bridge for QuickDotAI.
 *
 * Public to preserve JNI symbol names; use [NativeQuickDotAI] instead.
 * @hide
 */
object NativeCausalLm {

    @Volatile
    private var loaded: Boolean = false

    /** @brief Load the required native libraries once. */
    @Synchronized
    fun ensureLoaded(): Boolean {
        if (loaded) return true
        return try {
            // CPU-only packages omit the optional QNN backend.
            try {
                System.loadLibrary("qnn_context")
                android.util.Log.i(TAG, "Loaded optional QNN backend (qnn_context)")
            } catch (t: UnsatisfiedLinkError) {
                android.util.Log.d(TAG, "No optional QNN backend present (CPU build)")
            }
            System.loadLibrary("quickai_jni")

            // Optional model-extension plugin. A downstream project may ship a
            // self-registering plugin (libqai_ext_model.so) that adds extra models to
            // the catalog; the public build ships without it and runs public models
            // only. Loaded after quickai_jni so libcausallm.so/libquick_dot_ai_api.so
            // (and their model/descriptor registries) are already present for the
            // plugin's constructors to register into. Absence is normal -> ignore.
            try {
                System.loadLibrary("qai_ext_model")
                android.util.Log.i(TAG, "Loaded optional model-extension plugin (qai_ext_model)")
            } catch (t: UnsatisfiedLinkError) {
                android.util.Log.d(TAG, "No optional model-extension plugin present (public build)")
            }

            loaded = true
            true
        } catch (t: UnsatisfiedLinkError) {
            android.util.Log.e(TAG, "Failed to load libquickai_jni.so: ${t.message}")
            false
        }
    }

    /**
     * @brief Result of a loadModel call. [handle] is an opaque pointer
     * (packed in a long) that must be passed back to [runModelHandleNative],
     * [getPerformanceMetricsHandleNative] and [destroyModelHandleNative].
     */
    data class LoadResult(val errorCode: Int, val handle: Long)

    /**
     * @brief Result of a runModel call.
     */
    data class RunResult(val errorCode: Int, val output: String?)

    /**
     * @brief Result of a metrics call.
     */
    data class MetricsResult(
        val errorCode: Int,
        val prefillTokens: Int,
        val prefillDurationMs: Double,
        val generationTokens: Int,
        val generationDurationMs: Double,
        val totalDurationMs: Double,
        val initializationDurationMs: Double,
        val peakMemoryKb: Long
    )


    /**
     * @brief Result of a multimodal run call.
     */
    data class MultimodalRunResult(val errorCode: Int, val output: String?)

    /**
     * @brief Multimodal input data for vision encoder.
     *
     * Supports both single-image (legacy) and multi-image (e.g. V-JEPA
     * video frames) scenarios.
     *
     * @param pixelValues      Preprocessed image patches in CHW format.
     *                         Shape: [numPatches * 3 * 512 * 512] (patch size is fixed at 512)
     *                         For multi-image, all images' patches are concatenated.
     * @param numPatches       Total number of image patches across all images
     * @param originalHeight   Original image height before preprocessing (first image)
     * @param originalWidth    Original image width before preprocessing (first image)
     * @param numImages        Number of images (e.g. 16 for V-JEPA video frames).
     *                         Defaults to 1 for backward compatibility.
     * @param patchesPerImage  Number of patches per image. Null for single-image.
     * @param originalHeights  Original height of each image. Null for single-image.
     * @param originalWidths   Original width of each image. Null for single-image.
     */
    data class MultimodalInput(
        val pixelValues: FloatArray,
        val numPatches: Int,
        val originalHeight: Int,
        val originalWidth: Int,
        val numImages: Int = 1,
        val patchesPerImage: IntArray? = null,
        val originalHeights: IntArray? = null,
        val originalWidths: IntArray? = null
    )


    /** Forwards to `setOptions` in quick_dot_ai_api.h. */
    external fun setOptionsNative(
        useChatTemplate: Boolean,
        debugMode: Boolean,
        verbose: Boolean
    ): Int

    /**
     * @brief Thin wrapper around POSIX `chdir(2)`.
     *
     * The native C API in quick_dot_ai_api.cpp builds its model paths as
     * `./models/<name>-<quant>` (see `resolve_model_path`), so the
     * loader's behaviour depends on the process's current working
     * directory. Android apps launch with cwd="/" which is not writable,
     * so the host code must chdir the process to an app-owned directory
     * (typically `Context.getExternalFilesDir(null)`) before calling
     * [loadModelHandleNative]. [NativeQuickDotAI] does this
     * automatically when the caller supplies a [LoadModelRequest.modelPath].
     *
     * @return 0 on success, or the POSIX errno value on failure.
     */
    external fun chdirNative(path: String): Int

    /**
     * @brief Forwards to `loadModelHandle` in quick_dot_ai_api.h.
     *
     * @param nativeLibDir Native library directory path from
     *        ApplicationInfo.nativeLibraryDir. May be null.
     * @param modelBasePath Base directory for model files
     *        (e.g. "/sdcard/Download/aistudio-mobile/models/").
     *        May be null (uses C API default).
     */
    external fun loadModelHandleNative(
        backendOrdinal: Int,
        modelOrdinal: Int,
        quantOrdinal: Int,
        nativeLibDir: String?,
        modelBasePath: String?,
        htpBackendConfigPath: String?
    ): LoadResult

    /**
     * @brief Loads model by string catalog id (T4 path).
     * @return Handle as Long, or 0 on failure.
     */
    external fun loadModelHandleByNameNative(
        backend: Int,
        modelId: String,
        quant: Int,
        nativeLibDir: String?,
        modelBasePath: String?,
    ): Long

    /** @brief Returns the registered model catalog as a JSON array string. */
    external fun nativeQueryCatalog(): String

    /**
     * Encode [text] into a sentence-embedding vector using an embedding handle
     * (models[0] must be a SentenceTransformer, e.g. "ouro").
     *
     * @return the embedding FloatArray on success, or null on any native error
     *         (unsupported model, not initialized, inference failure).
     */
    external fun encodeModelHandleNative(handle: Long, text: String): FloatArray?

    /**
     * @brief Listener invoked by the JNI trampoline once per decoded
     * delta during [runModelHandleStreamingNative].
     *
     * The method is called **on the same thread that invoked
     * runModelHandleStreamingNative** — the JNI bridge does NOT attach
     * any new thread to the JVM — so implementations must be
     * non-blocking (deltas arrive back-to-back at decode speed).
     */
    fun interface NativeStreamListener {
        fun onDelta(text: String)
    }

    /**
     * @brief Forwards to `runModelHandleStreaming` in quick_dot_ai_api.h.
     *
     * Blocking: returns only when generation finishes, EOS is emitted,
     * NUM_TO_GENERATE is reached, the listener throws, or an error
     * occurs. [listener] is invoked synchronously from the same thread
     * for every decoded delta; if it throws, the JNI bridge catches
     * the exception, asks the native runner to cancel at the next
     * token boundary, and propagates a non-zero ErrorCode back here.
     * Terminal events (onDone / onError) are synthesized on the Kotlin
     * side from the return value — see [NativeQuickDotAI.runStreaming].
     *
     * @return An `ErrorCode` int; 0 on clean completion.
     */
    external fun runModelHandleStreamingNative(
        handle: Long,
        prompt: String,
        listener: NativeStreamListener
    ): Int

    /**
     * @brief Forwards to `runModelHandleWithMessagesStreaming` in quick_dot_ai_api.h.
     *
     * Streaming inference with OpenAI message format on a specific handle.
     *
     * @param handle              Handle returned by loadModelHandleNative
     * @param messages            Array of chat messages
     * @param addGenerationPrompt Whether to append generation prompt at end
     * @param listener            Callback for streaming output
     * @return An `ErrorCode` int; 0 on clean completion.
     */
    external fun runModelHandleWithMessagesStreamingNative(
        handle: Long,
        messages: Array< QuickAiChatMessage>,
        addGenerationPrompt: Boolean,
        listener: NativeStreamListener
    ): Int

    /** Forwards to `getPerformanceMetricsHandle` in quick_dot_ai_api.h. */
    external fun getPerformanceMetricsHandleNative(handle: Long): MetricsResult

    /** Forwards to `unloadModelHandle` in quick_dot_ai_api.h. */
    external fun unloadModelHandleNative(handle: Long): Int

    /** Forwards to `destroyModelHandle` in quick_dot_ai_api.h. */
    external fun destroyModelHandleNative(handle: Long): Int

    /**
     * @brief Forwards to `cancelModelHandle` in quick_dot_ai_api.h.
     *
     * Requests cancellation of an in-progress streaming run. Thread-safe:
     * can be called from any thread (e.g., UI cancel button handler).
     *
     * @param handle Handle returned by loadModelHandleNative
     * @return An `ErrorCode` int; 0 on success.
     */
    external fun cancelModelHandleNative(handle: Long): Int

    /**
     * @brief Forwards to `runMultimodalHandleStreaming` in quick_dot_ai_api.h.
     *
     * Multimodal streaming inference that accepts preprocessed image patches
     * and a text prompt. The pixel values are passed as a FloatArray and
     * converted to native float* in JNI layer.
     *
     * @param handle         Handle returned by loadModelHandleNative
     * @param prompt         Text prompt
     * @param pixelValues    Preprocessed image patches (CHW format)
     * @param numPatches     Number of image patches
     * @param originalHeight Original image height before preprocessing
     * @param originalWidth  Original image width before preprocessing
     * @param listener       Callback for streaming output
     * @return An `ErrorCode` int; 0 on clean completion.
     */
    external fun runMultimodalHandleStreamingNative(
        handle: Long,
        prompt: String,
        pixelValues: FloatArray,
        numPatches: Int,
        originalHeight: Int,
        originalWidth: Int,
        listener: NativeStreamListener
    ): Int

    /**
     * @brief Forwards to `runMultimodalHandleWithMessagesStreaming` in quick_dot_ai_api.h.
     *
     * Streaming multimodal inference with OpenAI message format on a specific handle.
     *
     * @param handle              Handle returned by loadModelHandleNative
     * @param messages            Array of chat messages (text-only, image via pixelValues)
     * @param addGenerationPrompt Whether to append generation prompt at end
     * @param pixelValues         Preprocessed image patches (CHW format)
     * @param numPatches          Number of image patches
     * @param originalHeight      Original image height before preprocessing
     * @param originalWidth       Original image width before preprocessing
     * @param listener            Callback for streaming output
     * @return An `ErrorCode` int; 0 on clean completion.
     */
    external fun runMultimodalHandleWithMessagesStreamingNative(
        handle: Long,
        messages: Array<QuickAiChatMessage>,
        addGenerationPrompt: Boolean,
        pixelValues: FloatArray,
        numPatches: Int,
        originalHeight: Int,
        originalWidth: Int,
        listener: NativeStreamListener
    ): Int

    /**
     * @brief Forwards to `runModelHandleWithJsonStreaming` in quick_dot_ai_api.h.
     *
     * Streaming inference with OpenAI JSON format on a specific handle.
     * Accepts a JSON string containing messages, tools, functions, etc.
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
     * @param handle       Handle returned by loadModelHandleNative
     * @param jsonRequest  OpenAI format JSON string
     * @param listener     Callback for streaming output
     * @return An `ErrorCode` int; 0 on clean completion.
     */
    external fun runModelHandleWithJsonStreamingNative(
        handle: Long,
        jsonRequest: String,
        listener: NativeStreamListener
    ): Int

    /**
     * @brief Multimodal streaming inference with multi-image support (V-JEPA).
     *
     * @param handle              Handle returned by loadModelHandleNative
     * @param prompt              Text prompt
     * @param pixelValues         Preprocessed image patches (CHW format, all images concatenated)
     * @param numPatches          Total number of image patches
     * @param numImages           Number of images (e.g. 16 for V-JEPA)
     * @param patchesPerImage     Number of patches per image
     * @param originalHeights     Original height of each image
     * @param originalWidths      Original width of each image
     * @param listener            Callback for streaming output
     * @return An `ErrorCode` int; 0 on clean completion.
     */
    external fun runMultimodalMultiImageStreamingNative(
        handle: Long,
        prompt: String,
        pixelValues: FloatArray,
        numPatches: Int,
        numImages: Int,
        patchesPerImage: IntArray,
        originalHeights: IntArray,
        originalWidths: IntArray,
        listener: NativeStreamListener
    ): Int

    /**
     * @brief Multimodal streaming inference with multi-image + messages (V-JEPA).
     *
     * @param handle              Handle returned by loadModelHandleNative
     * @param messages            Array of chat messages
     * @param addGenerationPrompt Whether to append generation prompt at end
     * @param pixelValues         Preprocessed image patches (CHW format, all images concatenated)
     * @param numPatches          Total number of image patches
     * @param numImages           Number of images (e.g. 16 for V-JEPA)
     * @param patchesPerImage     Number of patches per image
     * @param originalHeights     Original height of each image
     * @param originalWidths      Original width of each image
     * @param listener            Callback for streaming output
     * @return An `ErrorCode` int; 0 on clean completion.
     */
    external fun runMultimodalMultiImageWithMessagesStreamingNative(
        handle: Long,
        messages: Array<QuickAiChatMessage>,
        addGenerationPrompt: Boolean,
        pixelValues: FloatArray,
        numPatches: Int,
        numImages: Int,
        patchesPerImage: IntArray,
        originalHeights: IntArray,
        originalWidths: IntArray,
        listener: NativeStreamListener
    ): Int

    @JvmStatic
    external fun configureSpeculativeDecodingNative(handle: Long, on: Boolean): Int

    private const val TAG = "NativeCausalLm"
}
