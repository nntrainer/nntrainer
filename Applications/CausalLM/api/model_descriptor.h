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
#include <cstddef>
#else
#include <stddef.h>
#endif

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
  QDA_CAP_PLANAR_CHW =
    1u << 8, /**< the vision encoder consumes planar CHW (NCHW) pixels, not the
                  interleaved HWC layout the other vision models take. Appended
                  after QDA_CAP_SPECULATIVE: bit values never move. */
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

/*
 * ModelDescriptor is copied BY VALUE into the host's registry, and a model
 * extension (libqai_ext_model.so) is compiled separately from the host. A
 * plugin built against a header with fewer fields would be over-read, one with
 * more would be truncated. These asserts pin the layout; a change here is an
 * ABI change and must bump QDA_EXT_ABI_VERSION (ext_model_abi.h), which the
 * host checks when it loads a plugin.
 */
#ifdef __cplusplus
static_assert(offsetof(ModelDescriptor, sd_variant_id) + sizeof(const char *) ==
                sizeof(ModelDescriptor),
              "sd_variant_id must stay the last ModelDescriptor field; append "
              "new fields after it and bump QDA_EXT_ABI_VERSION");
#if defined(__LP64__) || defined(_WIN64)
static_assert(sizeof(ModelDescriptor) == 64,
              "ModelDescriptor ABI changed: bump QDA_EXT_ABI_VERSION");
#endif
#endif

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

/**
 * @brief Map a descriptor's config_name to the directory name its model pack
 *        lives in, for packs whose directory is not simply config_name
 *        lower-cased (which is what an unmapped name resolves to). The key is
 *        compared case-insensitively. Used by model extensions to place their
 *        packs without the host knowing their names.
 */
void register_model_path(const char *config_name, const char *directory);
} // namespace quick_dot_ai
#endif

#endif /* __QUICK_DOT_AI_MODEL_DESCRIPTOR_H__ */
