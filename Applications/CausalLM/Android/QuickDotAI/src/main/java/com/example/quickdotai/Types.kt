// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    Types.kt
 * @brief   Value types shared by the QuickDotAI interface and its
 *          implementations.
 *
 * The enums mirror the C enums in Applications/CausalLM/api/quick_dot_ai_api.h.
 * Model identifiers are plain Strings (see [ModelIds] in ModelCatalog.kt)
 * so the AAR is not re-compiled whenever the model list changes.
 *
 * Every public class in this file carries `@Serializable` so host apps
 * that want to JSON-ify requests/responses (for example QuickAIService's
 * REST layer) can do so without redefining the types — the AAR exposes
 * kotlinx-serialization-json as an `api` dependency for that purpose.
 */
package com.example.quickdotai

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * @brief Compute backend. Mirrors BackendType in quick_dot_ai_api.h.
 */
@Serializable
enum class BackendType {
    CPU,
    GPU,
    NPU
}


/**
 * @brief Quantization type. Mirrors ModelQuantizationType in
 *        quick_dot_ai_api.h.
 */
@Serializable
enum class QuantizationType {
    UNKNOWN,
    W4A32,
    W16A16,
    W8A16,
    W32A32
}

/**
 * @brief Error code. Mirrors ErrorCode in quick_dot_ai_api.h plus a few
 *        Kotlin-level additions for out-of-band conditions.
 */
@Serializable
enum class QuickAiError(val code: Int) {
    NONE(0),
    INVALID_PARAMETER(1),
    MODEL_LOAD_FAILED(2),
    INFERENCE_FAILED(3),
    NOT_INITIALIZED(4),
    INFERENCE_NOT_RUN(5),
    UNKNOWN(99),

    // Kotlin-only conditions returned by higher layers (QuickAIService
    // worker / dispatcher). The AAR itself only surfaces the native
    // codes and NOT_INITIALIZED, but it is convenient for the host app
    // to have the full enum in one place.
    QUEUE_FULL(100),
    MODEL_NOT_FOUND(101),
    UNSUPPORTED(102),
    BAD_REQUEST(103);

    companion object {
        fun fromNativeCode(code: Int): QuickAiError =
            entries.firstOrNull { it.code == code } ?: UNKNOWN
    }
}

/**
 * @brief Descriptor passed to [QuickDotAI.load].
 *
 * [modelPath] is required by [LiteRTLm] (which takes an explicit path
 * to a `.litertlm` file) and ignored by [NativeQuickDotAI] (which
 * discovers its model assets through the native C API's internal
 * model-directory resolution).
 *
 * [visionBackend] and [cacheDir] are optional knobs used only by
 * multimodal-capable engines ([LiteRTLm] today). They are ignored
 * by [NativeQuickDotAI].
 */
@Serializable
data class LoadModelRequest(
    val backend: BackendType = BackendType.GPU,
    @SerialName("model_id") val modelId: String,
    val quantization: QuantizationType = QuantizationType.W4A32,
    @SerialName("model_path") val modelPath: String? = null,

    /**
     * Compute backend for the model's vision encoder when loading a
     * multimodal-capable model (e.g. Gemma-4 / Gemma3n). Null means
     * the engine is loaded in text-only mode — in that case
     * [QuickDotAI.runMultimodal] returns [QuickAiError.UNSUPPORTED]
     * even on backends that would otherwise support images.
     *
     * Only honored by [LiteRTLm]; [NativeQuickDotAI] ignores it.
     */
    @SerialName("vision_backend") val visionBackend: BackendType? = null,

    /**
     * Writable directory for engine on-disk caches. Populating this
     * field materially speeds up the second and subsequent loads of
     * the same model. Maps to LiteRT-LM's EngineConfig.cacheDir.
     * Null = engine default.
     *
     * Only honored by [LiteRTLm]; [NativeQuickDotAI] ignores it.
     *
     * (Note: the LiteRT-LM 0.10.x EngineConfig surface we compile
     * against does not yet expose a per-prompt `maxNumImages` cap —
     * that is a 1.0+ feature. Once we roll forward past 1.0 we can
     * add the corresponding field back to this request.)
     */
    @SerialName("cache_dir") val cacheDir: String? = null,

    /**
     * Maximum number of tokens the engine should allocate for the KV
     * cache / context window. This is a load-sensitive setting — it must
     * be known at engine-construction time and cannot be changed per
     * request. Null = engine default.
     *
     * Honored by [LiteRTLm] (maps to EngineConfig.maxNumTokens) and
     * ignored by [NativeQuickDotAI].
     */
    @SerialName("max_num_tokens") val maxNumTokens: Int? = null,

    /**
     * Native library directory path from ApplicationInfo.nativeLibraryDir.
     * Used by the native engine to locate shared libraries for loading.
     *
     * Only honored by [NativeQuickDotAI]; [LiteRTLm] ignores it.
     */
    @SerialName("native_lib_dir") val nativeLibDir: String? = null,

    /**
     * Base directory path for model files. The native C API uses this
     * as the prefix when resolving model directories (e.g.
     * "$modelBasePath/qwen3-0.6b"). When null, the C API falls back
     * to its built-in default path.
     *
     * Only honored by [NativeQuickDotAI]; [LiteRTLm] ignores it.
     */
    @SerialName("model_base_path") val modelBasePath: String? = null,

    /**
     * Path to the HTP backend extension config JSON file used by QNN
     * models. Absolute paths are used as-is. Relative paths are resolved
     * from `<externalFilesDir>`. When null, the native engine falls back to
     * `<externalFilesDir>/htp_backend_ext_config.json`.
     *
     * Only honored by [NativeQuickDotAI]; [LiteRTLm] ignores it.
     */
    @SerialName("htp_backend_config_path") val htpBackendConfigPath: String? = null,
    val useSpeculativeDecoding: Boolean = false,
) {
    /**
     * Canonical key shared across the stack: one worker/handle per
     * (model, quantization) pair.
     */
    val modelKey: String get() = "$modelId:${quantization.name}:sd=$useSpeculativeDecoding"
}

/**
 * @brief One part of a multimodal prompt passed to
 *        [QuickDotAI.runMultimodal] / [QuickDotAI.runMultimodalStreaming].
 *
 * The concrete backend (currently [LiteRTLm] for Gemma-family models
 * loaded with a non-null [LoadModelRequest.visionBackend]) translates
 * each part to its native content representation. The ordering in the
 * list is preserved — the canonical Gemma-4 / Gemma3n convention is
 * one or more image parts followed by a single trailing text
 * instruction, e.g.
 * ```
 * runMultimodal(listOf(
 *     PromptPart.ImageFile("/sdcard/.../photo.jpg"),
 *     PromptPart.Text("Describe this picture in one sentence."),
 * ))
 * ```
 *
 * PromptPart is intentionally NOT @Serializable: `ImageBytes.bytes`
 * would serialize as a JSON array of ints, which is the wrong wire
 * format for a REST layer. Consumers that need to carry multimodal
 * prompts over the wire (e.g. LauncherApp's HTTP server) should
 * define their own Base64-flavored DTO and convert at the boundary.
 */
sealed class PromptPart {
    /** A chunk of text — typically the user's question or instruction. */
    data class Text(val text: String) : PromptPart()

    /**
     * A local image file. [absolutePath] must point to a readable file
     * on the device — the engine opens it directly from the native
     * layer, so relative paths are NOT supported. Mirrors LiteRT-LM's
     * parameter naming for clarity.
     *
     * Supported formats depend on the underlying engine but generally
     * include JPEG and PNG.
     */
    data class ImageFile(val absolutePath: String) : PromptPart()

    /**
     * Image bytes already held in memory. Useful when the image comes
     * from an in-process source (camera buffer, bundled asset, HTTP
     * download) and the caller does not want to materialize it to a
     * temporary file first.
     *
     * The byte layout must be the raw file contents of an encoded
     * image (JPEG / PNG / …), NOT a decoded pixel array.
     */
    data class ImageBytes(val bytes: ByteArray) : PromptPart() {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is ImageBytes) return false
            return bytes.contentEquals(other.bytes)
        }
        override fun hashCode(): Int = bytes.contentHashCode()
    }

    /**
     * Pre-processed pixel values already in CHW float format.
     * Used by models like V-JEPA where the caller has already performed
     * image preprocessing externally and wants to pass the result
     * directly to the native vision encoder without any further
     * transformation on the Kotlin side.
     *
     * @param pixelValues  Flattened pixel values in CHW format
     *                     (all images concatenated)
     * @param numPatches   Total number of patches across all images
     * @param numImages    Number of images (e.g. video frames)
     * @param patchesPerImage Number of patches per image
     * @param imageHeights Original height of each image before preprocessing
     * @param imageWidths  Original width of each image before preprocessing
     */
    data class PreprocessedPixels(
        val pixelValues: FloatArray,
        val numPatches: Int,
        val numImages: Int,
        val patchesPerImage: IntArray,
        val imageHeights: IntArray,
        val imageWidths: IntArray
    ) : PromptPart() {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is PreprocessedPixels) return false
            return pixelValues.contentEquals(other.pixelValues)
        }
        override fun hashCode(): Int = pixelValues.contentHashCode()
    }
}

/**
 * @brief Performance metrics for the most recent run.
 *
 * Not every engine fills every field:
 *  - [NativeQuickDotAI] fills prefill_* / generation_* / peak_memory_kb
 *    from the C API's PerformanceMetrics struct.
 *  - [LiteRTLm] currently only fills [initializationDurationMs] and
 *    [totalDurationMs] because the LiteRT-LM Kotlin API does not expose
 *    token-level counters in the release we target.
 */
/* -------------------------------------------------------------------- */
/* Structured Chat / Session types (request-mail1 §1–§5, mail2 §6)      */
/* -------------------------------------------------------------------- */

/**
 * @brief Role within a structured chat conversation.
 */
enum class QuickAiChatRole {
    SYSTEM,
    USER,
    ASSISTANT
}

/**
 * @brief Sampling configuration applied to a chat session.
 *
 * Field support by backend:
 *  - [LiteRTLm] maps [temperature], [topK], [topP], and [seed] to
 *    LiteRT-LM's `SamplerConfig`. [minP] and [maxTokens] are NOT
 *    supported by LiteRT-LM's SamplerConfig and are silently ignored
 *    (a warning is logged).
 *
 * Partial specification:
 *  - Leaving every field null is equivalent to passing no sampling
 *    config at all — LiteRT-LM uses its engine/model default.
 *  - Specifying any of [temperature] / [topK] / [topP] requires the
 *    wrapper to build a full `SamplerConfig`, which in LiteRT-LM has
 *    non-nullable core fields. Unspecified core fields fall back to
 *    temperature=1.0, topK=40, topP=0.95, and a warning is logged.
 *    To avoid surprises, specify all three together.
 *
 * Validation:
 *  - LiteRT-LM requires topK > 0, topP in [0, 1], temperature >= 0.
 *    Violations throw from the underlying engine and surface as a
 *    [BackendResult.Err].
 */
@Serializable
data class QuickAiChatSamplingConfig(
    val temperature: Double? = null,
    @SerialName("top_k") val topK: Int? = null,
    @SerialName("top_p") val topP: Double? = null,
    @SerialName("min_p") val minP: Double? = null,
    @SerialName("max_tokens") val maxTokens: Int? = null,
    val seed: Int? = null
)

/**
 * @brief Template keyword arguments forwarded to the chat template
 * renderer. [enableThinking] controls whether the model's "thinking"
 * prompt preamble is activated.
 */
@Serializable
data class QuickAiChatTemplateKwargs(
    @SerialName("enable_thinking") val enableThinking: Boolean? = null
)

/**
 * @brief Configuration for a new chat session, passed to
 * [QuickDotAI.openChatSession].
 *
 * [systemInstruction] maps to LiteRT-LM's
 * `ConversationConfig.systemInstruction` and is applied once when the
 * conversation is created — equivalent to the `"system"` role in
 * OpenAI-style message lists.
 */
@Serializable
data class QuickAiChatSessionConfig(
    @SerialName("system_instruction") val systemInstruction: String? = null,
    val sampling: QuickAiChatSamplingConfig? = null,
    @SerialName("chat_template_kwargs") val chatTemplateKwargs: QuickAiChatTemplateKwargs? = null
)

/**
 * @brief One message in a structured chat conversation.
 *
 * The [parts] list may contain text, image files, or raw image bytes
 * in any order — the backend preserves insertion order. For text-only
 * turns, a single [PromptPart.Text] suffices.
 */
data class QuickAiChatMessage(
    val role: QuickAiChatRole,
    val parts: List<PromptPart>
)

/**
 * @brief Result returned by [QuickDotAI.chatRun] /
 * [QuickDotAI.chatRunStreaming].
 */
data class QuickAiChatResult(
    val content: String,
    val reasoning: String? = null,
    val metrics: PerformanceMetrics? = null
)

/* -------------------------------------------------------------------- */
/* Performance metrics                                                  */
/* -------------------------------------------------------------------- */

@Serializable
data class PerformanceMetrics(
    @SerialName("prefill_tokens") val prefillTokens: Int = 0,
    @SerialName("prefill_duration_ms") val prefillDurationMs: Double = 0.0,
    @SerialName("generation_tokens") val generationTokens: Int = 0,
    @SerialName("generation_duration_ms") val generationDurationMs: Double = 0.0,
    @SerialName("total_duration_ms") val totalDurationMs: Double = 0.0,
    @SerialName("initialization_duration_ms") val initializationDurationMs: Double = 0.0,
    @SerialName("peak_memory_kb") val peakMemoryKb: Long = 0
)
