// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    quick_dot_ai_api.h
 * @date    20 Mar 2026
 * @brief   C API for src (extension of CausalLM)
 *
 *          This header is self-contained: if causal_lm_api.h has already
 *          been included its types are reused; otherwise fallback
 *          definitions are provided so that this single header is
 *          sufficient for application code.
 *
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Eunju Yang <ej.yang@samsung.com>
 * @bug     No known bugs except for NYI items
 */
#ifndef __QUICK_DOT_AI_API_H__
#define __QUICK_DOT_AI_API_H__

#ifndef __cplusplus
#include <stdbool.h>
#endif

/* ── Extended model types (src additions) ────────────────────── */
#ifdef __CAUSAL_LM_API_H__
/* Model types already defined from causal_lm_api.h */
#else /* causal_lm_api.h not included — provide full definitions */

#define __CAUSAL_LM_API_H__

#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include "callback_streamer.h"
#include "streamer.h"

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

typedef enum {
  CAUSAL_LM_ERROR_NONE = 0,
  CAUSAL_LM_ERROR_INVALID_PARAMETER = 1,
  CAUSAL_LM_ERROR_MODEL_LOAD_FAILED = 2,
  CAUSAL_LM_ERROR_INFERENCE_FAILED = 3,
  CAUSAL_LM_ERROR_NOT_INITIALIZED = 4,
  CAUSAL_LM_ERROR_INFERENCE_NOT_RUN = 5,
  CAUSAL_LM_ERROR_UNSUPPORTED = 6,
  CAUSAL_LM_ERROR_UNKNOWN = 99
} ErrorCode;

typedef enum {
  CAUSAL_LM_BACKEND_CPU = 0,
  CAUSAL_LM_BACKEND_GPU = 1,
  CAUSAL_LM_BACKEND_NPU = 2,
} BackendType;

/** causallm::transformer.h defines enum class ModelType at global scope.
 * Suppress our deprecated compat shim when that header is already included
 * to prevent an ambiguous-name error in translation units that include both. */
#ifndef __TRANSFORMER_H__
/**
 * @deprecated T4: 모델 식별의 정본은 문자열 id (loadModelHandleByName).
 *             이 enum은 기존 호출자 호환용 public-only compat shim.
 *             모델은 카탈로그로 자동 등록.
 */
typedef enum {
  CAUSAL_LM_MODEL_QWEN3_0_6B = 0,
  CAUSAL_LM_MODEL_QWEN3_1_7B_Q40 = 4,  /* original ordinal preserved */
  CAUSAL_LM_MODEL_TINY_BERT = 8,       /* original */
  CAUSAL_LM_MODEL_FUNCTION_GEMMA = 9,  /* original */
  CAUSAL_LM_MODEL_GEMMA4_CPU = 11,     /* original */
  CAUSAL_LM_MODEL_GEMMA4_E2B_QNN = 12, /* original */
  CAUSAL_LM_MODEL_VJEPA2_QNN = 13,
  CAUSAL_LM_MODEL_OURO_EMBEDDING = 14,
} ModelType;
#endif /* __TRANSFORMER_H__ */

typedef struct {
  // Add configuration options here as needed
  bool use_chat_template; /// < @brief Whether to apply chat template to input
  bool debug_mode; /// < @brief Check model file validity during initialization
  bool verbose;    /// < @brief Whether to print output during generation
  const char
    *chat_template_name; /// < @brief Template name to select from array
                         ///  (e.g., "default", "tool_use"). NULL for
                         ///  "default".
} Config;

WIN_EXPORT ErrorCode setOptions(Config config);

typedef enum {
  CAUSAL_LM_QUANTIZATION_UNKNOWN = 0,
  CAUSAL_LM_QUANTIZATION_W4A32 = 1,
  CAUSAL_LM_QUANTIZATION_W16A16 = 2,
  CAUSAL_LM_QUANTIZATION_W8A16 = 3,
  CAUSAL_LM_QUANTIZATION_W32A32 = 4,
} ModelQuantizationType;

/**
 * @brief Chat message structure for chat template formatting
 * @note  Compatible with HuggingFace apply_chat_template() format
 */
typedef struct {
  const char *role;    /**< Message role: "system", "user", or "assistant" */
  const char *content; /**< Message content text */
} CausalLMChatMessage;

/**
 * @brief Load a model
 * @param compute Backend compute type
 * @param modeltype Model type
 * @param quant_type Model quantization type
 * @return ErrorCode
 */
#ifndef __TRANSFORMER_H__
WIN_EXPORT ErrorCode loadModel(BackendType compute, ModelType modeltype,
                               ModelQuantizationType quant_type,
                               const char *model_base_path);
#endif /* __TRANSFORMER_H__ */

typedef struct {
  unsigned int prefill_tokens;
  double prefill_duration_ms;
  unsigned int generation_tokens;
  double generation_duration_ms;
  double total_duration_ms;
  double initialization_duration_ms;
  size_t peak_memory_kb;
} PerformanceMetrics;

WIN_EXPORT ErrorCode getPerformanceMetrics(PerformanceMetrics *metrics);

WIN_EXPORT ErrorCode saveQnnKvCache(const char *cache_path);
WIN_EXPORT ErrorCode loadQnnKvCache(const char *cache_path);
WIN_EXPORT ErrorCode resetQnnKvCache(void);

/**
 * @brief Apply chat template to messages without running inference
 * @param messages Array of chat messages with role and content
 * @param num_messages Number of messages in the array
 * @param add_generation_prompt Whether to append generation prompt at end
 * @param formattedText Buffer to store formatted text (owned by the library)
 * @return ErrorCode
 */
WIN_EXPORT ErrorCode applyChatTemplate(const CausalLMChatMessage *messages,
                                       size_t num_messages,
                                       bool add_generation_prompt,
                                       const char **formattedText);
/**============================================================================
 * Handle-based API (for parallel multi-model execution)
 *
 * The non-handle API above operates on a single process-wide model instance
 * protected by one global mutex, which serializes every call and prevents
 * loading more than one model at a time. The handle-based API below lets a
 * caller load several models simultaneously and run them in parallel from
 * different threads, with per-handle state so that different handles never
 * block each other. Each handle owns its own model, its own last-output
 * buffer, and its own mutex.
 *
 * A single handle may internally carry multiple sub-models (e.g. vision
 * encoder + LLM) when loaded from a top-level nntr_config.json that
 * specifies "architectures" and "model_dirs" arrays. The single-model
 * run API (runModelHandleWithMessages / runModelHandleStreaming) drives
 *models[0] only; the multimodal API (runMultimodalHandle*) drives the full set.
 *
 * Typical usage:
 *   CausalLmHandle h = NULL;
 *   loadModelHandle(CAUSAL_LM_BACKEND_CPU, CAUSAL_LM_MODEL_QWEN3_0_6B,
 *                   CAUSAL_LM_QUANTIZATION_W4A32, NULL, &h);
 *   const char *out = NULL;
 *   CausalLMChatMessage msg;
 *   msg.role = "user";
 *   msg.content = "Hello";
 *   runModelHandleWithMessages(h, &msg, 1, true, &out);
 *   // ... use out (owned by h, valid until the next run or destroy) ...
 *   destroyModelHandle(h);
 *============================================================================*/

/**
 * @brief Opaque handle to a loaded CausalLM model instance.
 */
typedef struct CausalLmModel *CausalLmHandle;

/**
 * @brief Load a model and return a newly-allocated handle.
 *
 * Calling this multiple times with different parameters returns independent
 * handles, each with its own model state. The caller must eventually call
 * destroyModelHandle on the returned handle to release resources.
 *
 * @param compute         Backend compute type
 * @param modeltype       Model type enum
 * @param quant_type      Quantization type
 * @param native_lib_dir  Native library directory path (from Android
 *                        ApplicationInfo.nativeLibraryDir). May be NULL.
 * @param out_handle      Out-parameter that receives the new handle on success
 * @return ErrorCode
 */
#ifndef __TRANSFORMER_H__
WIN_EXPORT ErrorCode loadModelHandle(BackendType compute, ModelType modeltype,
                                     ModelQuantizationType quant_type,
                                     const char *native_lib_dir,
                                     const char *model_base_path,
                                     CausalLmHandle *out_handle);
#endif /* __TRANSFORMER_H__ */

/**
 * @brief Load model by string id (T4 catalog path).
 *
 * Looks up the descriptor from the registry by @p model_id, validates the
 * backend, then loads via the same internal path as loadModelHandle.
 * Returns CAUSAL_LM_ERROR_INVALID_PARAMETER if the id is unknown, the
 * descriptor has no config_name, or the backend is not in backend_mask.
 *
 * @param compute         Backend compute type
 * @param model_id        Catalog string id e.g. "Qwen3-0.6B"
 * @param quant_type      Quantization type
 * @param native_lib_dir  Native library directory path. May be NULL.
 * @param model_base_path Base path for model files. May be NULL.
 * @param out_handle      Out-parameter receiving the new handle on success
 * @return ErrorCode
 */
WIN_EXPORT ErrorCode loadModelHandleByName(BackendType compute,
                                           const char *model_id,
                                           ModelQuantizationType quant_type,
                                           const char *native_lib_dir,
                                           const char *model_base_path,
                                           CausalLmHandle *out_handle);

/**
 * @brief SD 모델 로드 후 DDTree speculative decoding 활성화 여부를 설정한다.
 *        use_sd=false는 모든 모델에서 no-op (항상 CAUSAL_LM_ERROR_NONE).
 *        use_sd=true는 handle의 architecture가 configure_speculative_decoding
 *        콜백을 등록했고(ModelCallbackRegistry) draft 그래프가 있을 때만 성공.
 * @return CAUSAL_LM_ERROR_NONE on success,
 *         CAUSAL_LM_ERROR_MODEL_LOAD_FAILED if not supported.
 */
WIN_EXPORT ErrorCode configureSpeculativeDecoding(CausalLmHandle h,
                                                  bool use_sd);

/**
 * @brief Load a vision-encoder model and an LLM model as one multimodal handle.
 *
 * Lets the user freely pair an embedding (vision) model with an LLM, e.g.
 * in future ("vjepa", "lfm") / ("siglip", "lfm"). The resulting handle has
 * models[0] = embedding producer (vision) and models[1] = consumer (LLM);
 * the multimodal run path drives the pair through the generic composer.
 *
 * @param compute             Backend compute type
 * @param embedding_model_id  Catalog id of the vision encoder
 * @param llm_model_id        Catalog id of the LLM
 * @param quant_type          Quantization type
 * @param native_lib_dir      Native library directory path. May be NULL.
 * @param model_base_path     Base path for model files. May be NULL.
 * @param out_handle          Out-parameter receiving the new handle on success
 * @return ErrorCode. CAUSAL_LM_ERROR_UNSUPPORTED if the pair is incompatible
 *         (e.g. the chosen LLM exposes no embedding table).
 */
WIN_EXPORT ErrorCode loadMultimodalHandleByName(
  BackendType compute, const char *embedding_model_id, const char *llm_model_id,
  ModelQuantizationType quant_type, const char *native_lib_dir,
  const char *model_base_path, CausalLmHandle *out_handle);

/**
 * @brief Run inference on a specific handle.
 *
 * The returned outputText pointer is owned by the handle and remains valid
 * until the next runModelHandleWithMessages call on the same handle or until
 * the handle is destroyed. Different handles are safe to call concurrently from
 * different threads; the same handle is serialized by its own internal
 * mutex.
 *
 * Single-model API: drives models[0] only even when the handle was
 * populated with multiple sub-models. Use runMultimodalHandleWithMessages for
 * compositions such as vision-encoder + LLM.
 *
 * @param handle          Handle returned by loadModelHandle
 * @param messages        Array of chat messages with role and content
 * @param num_messages    Number of messages in the array
 * @param add_generation_prompt Whether to append generation prompt at end
 * @param outputText      Out-parameter that receives a pointer to the output
 * @return ErrorCode
 */
WIN_EXPORT ErrorCode runModelHandleWithMessages(
  CausalLmHandle handle, const CausalLMChatMessage *messages,
  size_t num_messages, bool add_generation_prompt, const char **outputText);

/**
 * @brief Streaming inference with OpenAI message format on a specific handle.
 *
 * Format the messages array through the chat template, then drive
 * generation token-by-token, invoking @p callback for each delta.
 * Blocks on the invoking thread until generation finishes or an error
 * occurs. Semantics are otherwise identical to runModelHandleStreaming.
 *
 * @param handle              Handle returned by loadModelHandle
 * @param messages            Array of chat messages with role and content
 * @param num_messages        Number of messages in the array
 * @param add_generation_prompt Whether to append generation prompt at end
 * @param callback            Token delta callback. Must be non-NULL.
 * @param user_data           Opaque pointer forwarded to callback
 * @return ErrorCode
 */
WIN_EXPORT ErrorCode runModelHandleWithMessagesStreaming(
  CausalLmHandle handle, const CausalLMChatMessage *messages,
  size_t num_messages, bool add_generation_prompt,
  CausalLmTokenCallback callback, void *user_data);

WIN_EXPORT ErrorCode saveQnnKvCacheHandle(CausalLmHandle handle,
                                          const char *cache_path);
WIN_EXPORT ErrorCode loadQnnKvCacheHandle(CausalLmHandle handle,
                                          const char *cache_path);
WIN_EXPORT ErrorCode resetQnnKvCacheHandle(CausalLmHandle handle);

/**
 * @brief Retrieve performance metrics of the last run for a given handle.
 * @param handle  Handle returned by loadModelHandle
 * @param metrics Pointer to a PerformanceMetrics struct to be filled
 * @return ErrorCode
 */
WIN_EXPORT ErrorCode getPerformanceMetricsHandle(CausalLmHandle handle,
                                                 PerformanceMetrics *metrics);

/**
 * @brief Release all resources owned by a handle.
 *
 * Passing a NULL handle is a no-op and returns CAUSAL_LM_ERROR_NONE.
 *
 * @param handle Handle returned by loadModelHandle
 * @return ErrorCode
 */
WIN_EXPORT ErrorCode destroyModelHandle(CausalLmHandle handle);

/**
 * @brief Request cancellation of an in-progress run on a handle.
 *
 * Sets the stop flag on the model, causing the token generation loop
 * to exit at the next token boundary. Thread-safe: can be called from
 * any thread (e.g., from a UI cancel button handler).
 *
 * If no run is in progress, this function is a no-op.
 *
 * @param handle Handle returned by loadModelHandle
 * @return ErrorCode
 */
WIN_EXPORT ErrorCode cancelModelHandle(CausalLmHandle handle);

/**
 * @brief Unload the model from a handle without destroying the handle.
 *
 * Releases the model weights and internal state but keeps the handle
 * struct alive. After a successful unload, the handle's initialized flag
 * is cleared and subsequent run / metrics calls will return
 * CAUSAL_LM_ERROR_NOT_INITIALIZED. The handle can be destroyed later
 * with destroyModelHandle, or (in future) re-loaded.
 *
 * Passing a NULL handle is a no-op and returns CAUSAL_LM_ERROR_NONE.
 *
 * @param handle Handle returned by loadModelHandle
 * @return ErrorCode
 */
WIN_EXPORT ErrorCode unloadModelHandle(CausalLmHandle handle);

/**
 * @brief Streaming counterpart of runModelHandle.
 *
 * Synchronously drives inference on @p handle and invokes @p callback
 * once per decoded-token boundary with a UTF-8 delta string. The call
 * blocks on the invoking thread until generation finishes, hits an EOS
 * token, reaches NUM_TO_GENERATE, the callback returns non-zero (which
 * requests cancellation at the next token boundary), or an error
 * occurs.
 *
 * The @p delta pointer passed into the callback is owned by the
 * streaming runtime and is only valid for the duration of the callback
 * invocation. Callers that need to retain the text must copy it.
 *
 * After a successful return the handle's "last output" buffer holds
 * the full concatenated generation (or the partial output on a
 * cancelled run), so a subsequent getPerformanceMetricsHandle() call
 * returns valid metrics and the same handle can be reused for another
 * run — identical semantics to runModelHandleWithMessages.
 *
 * Streaming is currently only supported on models whose underlying
 * C++ implementation derives from causallm::CausalLM (all the Qwen
 * variants and Llama do; non-CausalLM models return
 * CAUSAL_LM_ERROR_UNKNOWN). See AsyncAndStreaming.md §3.4 at the repo
 * root for the full design.
 *
 * @param handle          Handle returned by loadModelHandle.
 * @param inputTextPrompt Input prompt (UTF-8, NUL-terminated).
 * @param callback        Token delta callback. Must be non-NULL.
 * @param user_data       Opaque pointer forwarded verbatim to the
 *                        callback on every invocation. May be NULL.
 * @return ErrorCode
 */
WIN_EXPORT ErrorCode runModelHandleStreaming(CausalLmHandle handle,
                                             const char *inputTextPrompt,
                                             CausalLmTokenCallback callback,
                                             void *user_data);

/**
 * @brief Encode a single text prompt into a sentence-embedding vector using a
 *        handle whose models[0] is an embedding model (e.g. Ouro / "ouro").
 *
 * On success, *out_embedding points to a freshly allocated array of *out_dim
 * floats (the batch-0 embedding). The caller OWNS this buffer and MUST release
 * it with freeEmbedding(). On any error, *out_embedding is set to NULL and
 * *out_dim to 0.
 *
 * @param handle         Handle from loadModelHandle / loadModelHandleByName
 * @param text           UTF-8 input text (NUL-terminated)
 * @param out_embedding  [out] receives a newly allocated float[*out_dim]
 * @param out_dim        [out] receives the embedding dimension
 * @return ErrorCode. CAUSAL_LM_ERROR_UNSUPPORTED if models[0] is not an
 *         embedding (SentenceTransformer) model.
 */
WIN_EXPORT ErrorCode encodeModelHandle(CausalLmHandle handle, const char *text,
                                       float **out_embedding, int *out_dim);

/**
 * @brief Release a buffer returned by encodeModelHandle().
 * @param embedding  Pointer previously returned via out_embedding (may be NULL)
 */
WIN_EXPORT void freeEmbedding(float *embedding);

/**
 * @brief Run a standalone vision/video encoder (e.g. V-JEPA2 QNN) on raw
 *        pixel values and return its raw (quantized) embedding bytes.
 *
 * Unlike the multimodal path, this does NOT require an LLM consumer: the
 * encoder runs alone (no set_quant_param), so the output is the encoder's
 * native quantized buffer (uint16 for V-JEPA2), copied out verbatim.
 *
 * On success *out_embedding points to a freshly malloc'd buffer of *out_bytes
 * bytes. The caller OWNS it and MUST release it with freeImageEmbedding()
 * (NOT freeEmbedding(), which uses delete[]). On error *out_embedding is NULL
 * and *out_bytes is 0.
 *
 * @param handle         Handle whose models[0] is a vision encoder
 * @param pixelValues    Pointer to float buffer of raw pixels
 * @param numFloats      Number of floats in pixelValues
 *                       (V-JEPA2 raw layout: 1*24*3*256*256 = 4718592)
 * @param height         Original frame height (V-JEPA2: 256)
 * @param width          Original frame width  (V-JEPA2: 256)
 * @param out_embedding  [out] receives a newly malloc'd byte buffer
 * @param out_bytes      [out] receives the buffer length in bytes
 * @return CAUSAL_LM_ERROR_NONE on success; CAUSAL_LM_ERROR_UNSUPPORTED if
 *         built without QNN support.
 */
WIN_EXPORT ErrorCode encodeImageModelHandle(CausalLmHandle handle,
                                            const float *pixelValues,
                                            size_t numFloats, int height,
                                            int width, void **out_embedding,
                                            int *out_bytes);

/**
 * @brief Release a buffer returned by encodeImageModelHandle().
 * @param embedding  Pointer previously returned via out_embedding (may be NULL)
 */
WIN_EXPORT void freeImageEmbedding(void *embedding);

/**
 * @brief Run inference on a handle with a tool schema for constrained
 * generation.
 *
 * @param handle          Handle returned by loadModelHandle
 * @param inputTextPrompt Input prompt text
 * @param outputText      Buffer to store output text (owned by the handle)
 * @param tool_name       Name of the tool (used to cache the grammar)
 * @param tool_schema     JSON schema for the tool output format
 * @return ErrorCode
 */
WIN_EXPORT ErrorCode runModelHandleWithTool(CausalLmHandle handle,
                                            const char *inputTextPrompt,
                                            const char **outputText,
                                            const char *tool_name,
                                            const char *tool_schema);

/**============================================================================
 * Multimodal API
 *
 * These functions extend the handle-based API to support image+text inputs.
 * The pixel values are passed as preprocessed FloatArray (CHW format) from
 * the Kotlin image processor (LlavaNextImageProcessor).
 *
 * The handle must have been loaded from a multi-model nntr_config.json
 * (architectures[] + model_dirs[]) with at least [vision_encoder, llm];
 * a single-model handle returns CAUSAL_LM_ERROR_UNSUPPORTED.
 *
 * Vision Encoder integration is planned for future implementation.
 * Currently these functions return CAUSAL_LM_ERROR_UNSUPPORTED as stubs
 * once the multi-model precondition is satisfied.
 *============================================================================*/

/**
 * @brief Streaming multimodal inference on a specific handle.
 *
 * @param handle         Handle returned by loadModelHandle
 * @param prompt         Text prompt (UTF-8, NUL-terminated)
 * @param pixelValues    Preprocessed image patches in CHW format
 * @param numPatches     Number of image patches
 * @param originalHeight Original image height before preprocessing
 * @param originalWidth  Original image width before preprocessing
 * @param callback       Token delta callback. Must be non-NULL.
 * @param user_data      Opaque pointer forwarded to callback
 * @return ErrorCode (CAUSAL_LM_ERROR_UNSUPPORTED until Vision Encoder
 * implemented)
 */
WIN_EXPORT ErrorCode runMultimodalHandleStreaming(
  CausalLmHandle handle, const char *prompt, const float *pixelValues,
  int numPatches, int originalHeight, int originalWidth,
  CausalLmTokenCallback callback, void *user_data);

/**
 * @brief Blocking multimodal inference with OpenAI message format on a specific
 * handle.
 *
 * @param handle         Handle returned by loadModelHandle
 * @param messages       Array of chat messages with role and content
 * (text-only, image via pixelValues)
 * @param num_messages   Number of messages in the array
 * @param add_generation_prompt Whether to append generation prompt at end
 * @param pixelValues    Preprocessed image patches in CHW format
 * @param numPatches     Number of image patches
 * @param originalHeight Original image height before preprocessing
 * @param originalWidth  Original image width before preprocessing
 * @param outputText     Out-parameter that receives a pointer to the output
 * @return ErrorCode (CAUSAL_LM_ERROR_UNSUPPORTED until Vision Encoder
 * implemented)
 */
WIN_EXPORT ErrorCode runMultimodalHandleWithMessages(
  CausalLmHandle handle, const CausalLMChatMessage *messages,
  size_t num_messages, bool add_generation_prompt, const float *pixelValues,
  int numPatches, int originalHeight, int originalWidth,
  const char **outputText);

/**
 * @brief Streaming multimodal inference with OpenAI message format on a
 * specific handle.
 *
 * Format the messages array through the chat template, run the vision
 * encoder if needed, then drive LLM generation token-by-token invoking
 * @p callback for each delta. Blocks on the invoking thread until
 * generation finishes or an error occurs.
 *
 * @param handle         Handle returned by loadModelHandle
 * @param messages       Array of chat messages with role and content
 * (text-only, image via pixelValues)
 * @param num_messages   Number of messages in the array
 * @param add_generation_prompt Whether to append generation prompt at end
 * @param pixelValues    Preprocessed image patches in CHW format
 * @param numPatches     Number of image patches
 * @param originalHeight Original image height before preprocessing
 * @param originalWidth  Original image width before preprocessing
 * @param callback       Token delta callback. Must be non-NULL.
 * @param user_data      Opaque pointer forwarded to callback
 * @return ErrorCode
 */
WIN_EXPORT ErrorCode runMultimodalHandleWithMessagesStreaming(
  CausalLmHandle handle, const CausalLMChatMessage *messages,
  size_t num_messages, bool add_generation_prompt, const float *pixelValues,
  int numPatches, int originalHeight, int originalWidth,
  CausalLmTokenCallback callback, void *user_data);

/**============================================================================
 * Multi-image Multimodal API (V-JEPA)
 *
 * These functions extend the multimodal API to support multiple images
 * (e.g. video frames for V-JEPA). The pixel values for all images are
 * concatenated into a single flat array, with per-image metadata
 * (patches per image, heights, widths) passed as separate arrays.
 *
 * The handle must have been loaded with CAUSAL_LM_MODEL_VJEPA_QNN or
 * another multi-image-capable model type.
 *============================================================================*/

/**
 * @brief Streaming multi-image multimodal inference on a specific handle.
 *
 * Designed for models like V-JEPA that accept multiple preprocessed
 * image frames (e.g. 16 video frames) as input.
 *
 * @param handle            Handle returned by loadModelHandle
 * @param prompt            Text prompt (UTF-8, NUL-terminated)
 * @param pixelValues       Preprocessed image patches in CHW format
 *                          (all images concatenated)
 * @param numPatches        Total number of image patches across all images
 * @param numImages         Number of images (e.g. 16 for V-JEPA)
 * @param patchesPerImage   Array of numImages ints: patches per image
 * @param originalHeights   Array of numImages ints: original height per image
 * @param originalWidths    Array of numImages ints: original width per image
 * @param callback          Token delta callback. Must be non-NULL.
 * @param user_data         Opaque pointer forwarded to callback
 * @return ErrorCode
 */
WIN_EXPORT ErrorCode runMultimodalMultiImageHandleStreaming(
  CausalLmHandle handle, const char *prompt, const float *pixelValues,
  int numPatches, int numImages, const int *patchesPerImage,
  const int *originalHeights, const int *originalWidths,
  CausalLmTokenCallback callback, void *user_data);

/**
 * @brief Streaming multi-image multimodal inference with OpenAI message
 * format on a specific handle.
 *
 * @param handle              Handle returned by loadModelHandle
 * @param messages            Array of chat messages with role and content
 * @param num_messages        Number of messages in the array
 * @param add_generation_prompt Whether to append generation prompt at end
 * @param pixelValues         Preprocessed image patches in CHW format
 *                            (all images concatenated)
 * @param numPatches          Total number of image patches across all images
 * @param numImages           Number of images (e.g. 16 for V-JEPA)
 * @param patchesPerImage     Array of numImages ints: patches per image
 * @param originalHeights     Array of numImages ints: original height per image
 * @param originalWidths       Array of numImages ints: original width per image
 * @param callback            Token delta callback. Must be non-NULL.
 * @param user_data           Opaque pointer forwarded to callback
 * @return ErrorCode
 */
WIN_EXPORT ErrorCode runMultimodalMultiImageHandleWithMessagesStreaming(
  CausalLmHandle handle, const CausalLMChatMessage *messages,
  size_t num_messages, bool add_generation_prompt, const float *pixelValues,
  int numPatches, int numImages, const int *patchesPerImage,
  const int *originalHeights, const int *originalWidths,
  CausalLmTokenCallback callback, void *user_data);

/**============================================================================
 * OpenAI JSON streaming API
 *
 * Accepts a JSON string in OpenAI format and processes it through the
 * chat template. Supports messages, tools, functions, and all other
 * fields recognized by minja chat template renderer.
 *
 * Example JSON input:
 * {
 *   "messages": [
 *     {"role": "developer", "content": "..."},
 *     {"role": "user", "content": "..."}
 *   ],
 *   "tools": [
 *     {"type": "function", "function": {"name": "call", "description": "..."}}
 *   ]
 * }
 *============================================================================*/

/**
 * @brief Streaming inference with OpenAI JSON format.
 *
 * Parses the JSON request and applies the chat template, then drives
 * generation token-by-token invoking @p callback for each delta.
 *
 * @param handle       Handle returned by loadModelHandle
 * @param jsonRequest  OpenAI format JSON string (UTF-8, NUL-terminated)
 * @param callback     Token delta callback. Must be non-NULL.
 * @param user_data    Opaque pointer forwarded to callback
 * @return ErrorCode
 */
WIN_EXPORT ErrorCode runModelHandleWithJsonStreaming(
  CausalLmHandle handle, const char *jsonRequest,
  CausalLmTokenCallback callback, void *user_data);

/**
 * @brief Return a JSON array of all registered ModelDescriptors.
 *
 * Returns a NUL-terminated UTF-8 string like:
 *   [{"id":"...","family":"...","display_name":"...","runtime":0,
 *     "backend_mask":0,"capabilities":0}, ...]
 *
 * The registry is empty until tasks that call
 * quick_dot_ai::register_model_descriptor() are linked in.
 * The returned pointer is valid until the next call to getModelCatalogJson().
 *
 * @return const char* JSON array string (never NULL)
 */
WIN_EXPORT const char *getModelCatalogJson(void);

#ifdef __cplusplus
}
#endif

#endif /* __CAUSAL_LM_API_H__ */

#endif /* __QUICK_DOT_AI_API_H__ */
