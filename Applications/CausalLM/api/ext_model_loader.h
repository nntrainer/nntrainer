// SPDX-License-Identifier: Apache-2.0
/**
 * @file   ext_model_loader.h
 * @brief  Host-side loader for the optional model-extension plugin. See
 *         ext_model_abi.h for the contract.
 */
#ifndef __QUICK_DOT_AI_EXT_MODEL_LOADER_H__
#define __QUICK_DOT_AI_EXT_MODEL_LOADER_H__

#include <string>

#include "ext_model_abi.h"

namespace quick_dot_ai {

/** Outcome of looking for the model-extension plugin. */
enum class ExtModelState {
  NOT_TRIED, /**< ensureExtensionModels() has not run yet */
  ABSENT,    /**< no plugin on the library path: a normal public build */
  LOADED,    /**< plugin found, ABI accepted, models registered */
  FAILED,    /**< plugin present but refused or broken; see message */
};

struct ExtModelStatus {
  ExtModelState state = ExtModelState::NOT_TRIED;
  std::string message; /**< one line, "absent: ...", "loaded: ..." or
                            "failed: ..." */
  int registrars = 0;  /**< registrars the plugin ran (LOADED only) */
};

/**
 * @brief True when a dlopen() error means the plugin file itself is not there,
 *        as opposed to being found and failing (a missing dependency, an
 *        unresolved symbol, a bad ELF). Handles the bionic and glibc wordings.
 */
bool extDlopenErrorMeansAbsent(const std::string &error,
                               const std::string &library);

/**
 * @brief Compare a plugin's ABI with the host's.
 * @param why receives a message naming both sides on a mismatch
 */
bool checkExtAbi(const QdaExtAbiInfo *plugin, const QdaExtAbiInfo &host,
                 std::string *why);

/**
 * @brief Load one plugin: dlopen, ABI check, register. Does not cache; each
 *        call that finds the library registers again, so callers use
 *        ensureExtensionModels() unless they are a test.
 * @param library dlopen() name or path
 * @param explicit_path true when the caller named the file on purpose, so a
 *        missing file is a failure rather than "absent"
 */
ExtModelStatus loadExtensionModelsFrom(const char *library, bool explicit_path);

/**
 * @brief Process-wide, once: look for the plugin (QAI_EXT_MODEL_PATH if set,
 *        else "libqai_ext_model.so"), load it, remember the outcome. Thread
 *        safe. Call it before reading the model registries and NOT while
 *        holding a registry lock: the plugin registers into them.
 */
const ExtModelStatus &ensureExtensionModels();

} // namespace quick_dot_ai

#endif /* __QUICK_DOT_AI_EXT_MODEL_LOADER_H__ */
