// SPDX-License-Identifier: Apache-2.0
/**
 * @file   model_descriptor.h
 * @brief  T4 string-id model catalog schema (pluggable, self-registering).
 * @author jayden0701 <jrock.oh@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#ifndef __QUICK_DOT_AI_MODEL_DESCRIPTOR_H__
#define __QUICK_DOT_AI_MODEL_DESCRIPTOR_H__

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  QDA_RUNTIME_NATIVE = 0, /**< nntrainer (NativeQuickDotAI) */
  QDA_RUNTIME_LITERT = 1, /**< LiteRT-LM - Kotlin only, not registered in C */
} RuntimeKind;

typedef enum {
  QDA_CAP_STREAMING = 1u << 0,
  QDA_CAP_MESSAGES_API = 1u << 1, /**< requires messages-based API */
  QDA_CAP_MULTIMODAL = 1u << 2,
  QDA_CAP_TOOL_USE = 1u << 3,
  QDA_CAP_EMBEDDING = 1u << 4,
  QDA_CAP_MULTI_IMAGE = 1u << 5, /**< supports multiple images (e.g. V-JEPA) */
  QDA_CAP_VISION_ENCODER =
    1u << 6, /**< standalone vision embedding producer; pair with an LLM */
  QDA_CAP_SPECULATIVE =
    1u
    << 7, /**< supports speculative decoding; load sd_variant_id when enabled */
} CapabilityFlag;

/**
 * All `const char*` fields must point to storage with lifetime at least as
 * long as the process (e.g. string literals or static storage).  The registry
 * stores pointers, not copies.
 */
typedef struct {
  const char *id;            /**< "Qwen3-0.6B"  (catalog key) */
  const char *family;        /**< "qwen3-0.6b" */
  const char *display_name;  /**< "Qwen3 0.6B" */
  RuntimeKind runtime;       /**< QDA_RUNTIME_NATIVE */
  unsigned int backend_mask; /**< bit i set => BackendType i supported */
  unsigned int capabilities; /**< CapabilityFlag OR */
  const char
    *config_name; /**< g_model_registry lookup key e.g. "Qwen3-0.6B-W4A32" */
  const char *arch_string; /**< causallm::Factory key e.g. "Qwen3ForCausalLM" */
  const char *sd_variant_id; /**< catalog id of the speculative-decoding
                                  variant to load when SD is enabled; NULL if
                                  none */
} ModelDescriptor;

#ifdef __cplusplus
} // extern "C"
#endif

#ifdef __cplusplus
namespace quick_dot_ai {
/**
 * @brief Register a descriptor into g_descriptor_registry.
 *        Duplicate id overwrites the existing entry (logs a warning).
 */
void register_model_descriptor(const ModelDescriptor *desc);
} // namespace quick_dot_ai
#endif

#endif /* __QUICK_DOT_AI_MODEL_DESCRIPTOR_H__ */
