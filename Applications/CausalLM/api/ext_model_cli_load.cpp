// SPDX-License-Identifier: Apache-2.0
/**
 * @file   ext_model_cli_load.cpp
 * @brief  Lets a path-based CLI (main.cpp) pick up the optional model-extension
 *         plugin: look for libqai_ext_model.so before main() runs, exactly as
 *         the C API does on first use. Nothing happens when there is none.
 */
#include "ext_model_loader.h"

#ifdef NNTR_STATIC_MODEL_EXTENSION
namespace quick_dot_ai {
// The path-based runner selects Factory architectures, without a C API catalog.
void register_model_descriptor(const ModelDescriptor *) {}
void register_model_path(const char *, const char *) {}
} // namespace quick_dot_ai
#else
namespace {
struct LoadModelExtension {
  LoadModelExtension() { quick_dot_ai::ensureExtensionModels(); }
} load_model_extension;
} // namespace
#endif
