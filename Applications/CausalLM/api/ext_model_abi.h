// SPDX-License-Identifier: Apache-2.0
/**
 * @file   ext_model_abi.h
 * @brief  Contract between the host API and an optional model-extension plugin
 *         (libqai_ext_model.so).
 *
 * The host library (libquick_dot_ai_api.so + libcausallm.so) knows nothing of
 * the models a plugin adds. A plugin, built from a separate source tree, adds
 * them through the registries the host already owns:
 *
 *   - causallm::Factory::Instance().registerModel()   (architecture -> class)
 *   - quick_dot_ai::register_model_descriptor()       (catalog entry)
 *   - ModelCallbackRegistry::instance().register_for() (per-architecture hooks:
 *     prompt format, KV length, speculative decoding, multimodal)
 *
 * Those registries are single instances owned by the host libraries; a plugin
 * must IMPORT them, never define its own copy.
 *
 * Loading. The host dlopen()s "libqai_ext_model.so" (or the path in the
 * QAI_EXT_MODEL_PATH environment variable) once, on first use of the model
 * registry, and does not run any plugin code that registers before it has
 * checked the plugin's ABI:
 *
 *   1. dlopen. A plugin must not register from a static constructor: loading it
 *      alone (System.loadLibrary from Kotlin, a DT_NEEDED edge) has to be
 *      harmless.
 *   2. dlsym QDA_EXT_ABI_INFO_SYMBOL, compare the version AND the struct sizes
 *      the plugin was compiled with against the host's. A mismatch refuses the
 *      plugin with a message naming both sides; nothing is registered.
 *   3. dlsym + call QDA_EXT_REGISTER_SYMBOL, which registers the models.
 *
 * A host without the plugin on its library path is the normal public build and
 * is reported as "absent"; every other failure is reported as "failed" with the
 * reason (see getExtensionModelStatus() in quick_dot_ai_api.h).
 *
 * Bump QDA_EXT_ABI_VERSION whenever ModelDescriptor, ModelCallbacks (or what
 * they contain), or the semantics of a registry hook change.
 */
#ifndef __QUICK_DOT_AI_EXT_MODEL_ABI_H__
#define __QUICK_DOT_AI_EXT_MODEL_ABI_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Version of the host <-> plugin contract. 1: nine-field ModelDescriptor
 *  (+ QDA_CAP_PLANAR_CHW), ModelCallbacks with MultimodalCallbacks and the
 *  speculative-decoding hook, register_model_path, deferred registration. */
#define QDA_EXT_ABI_VERSION 1u

/** What the plugin was compiled against. Returned by the plugin's exported
 *  QDA_EXT_ABI_INFO_SYMBOL function; the plugin builds it from ITS headers. */
typedef struct {
  uint32_t abi_version; /**< QDA_EXT_ABI_VERSION of the plugin */
  uint32_t
    sizeof_model_descriptor;       /**< sizeof(ModelDescriptor) of the plugin */
  uint32_t sizeof_model_callbacks; /**< sizeof(ModelCallbacks) of the plugin */
  uint32_t reserved;               /**< 0 */
} QdaExtAbiInfo;

/** extern "C" const QdaExtAbiInfo *qai_ext_model_abi_info(void); */
#define QDA_EXT_ABI_INFO_SYMBOL "qai_ext_model_abi_info"
/** extern "C" int qai_ext_model_register(void); returns the number of
 *  registrars that ran, or a negative value on failure. */
#define QDA_EXT_REGISTER_SYMBOL "qai_ext_model_register"
/** File name the host looks for. */
#define QDA_EXT_LIBRARY_NAME "libqai_ext_model.so"
/** Environment variable that overrides the dlopen() name with an explicit path.
 */
#define QDA_EXT_PATH_ENV "QAI_EXT_MODEL_PATH"

typedef const QdaExtAbiInfo *(*QdaExtAbiInfoFn)(void);
typedef int (*QdaExtRegisterFn)(void);

#ifdef __cplusplus
} // extern "C"

#include "model_callbacks.h"
#include "model_descriptor.h"

namespace quick_dot_ai {

/** The ABI this translation unit's headers describe. */
inline QdaExtAbiInfo currentExtAbiInfo() {
  QdaExtAbiInfo info;
  info.abi_version = QDA_EXT_ABI_VERSION;
  info.sizeof_model_descriptor = static_cast<uint32_t>(sizeof(ModelDescriptor));
  info.sizeof_model_callbacks = static_cast<uint32_t>(sizeof(ModelCallbacks));
  info.reserved = 0;
  return info;
}

} // namespace quick_dot_ai

/**
 * Defines the plugin's exported ABI query. Use once in the plugin, in a TU that
 * includes this header, so the sizes are the plugin's own.
 */
#if defined(_WIN32)
#define QDA_EXT_EXPORT __declspec(dllexport)
#else
#define QDA_EXT_EXPORT __attribute__((visibility("default")))
#endif

#define QDA_EXT_DEFINE_ABI_INFO()                                              \
  extern "C" QDA_EXT_EXPORT const QdaExtAbiInfo *qai_ext_model_abi_info(       \
    void) {                                                                    \
    static const QdaExtAbiInfo info = ::quick_dot_ai::currentExtAbiInfo();     \
    return &info;                                                              \
  }

#endif /* __cplusplus */

#endif /* __QUICK_DOT_AI_EXT_MODEL_ABI_H__ */
