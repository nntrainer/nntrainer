// SPDX-License-Identifier: Apache-2.0
/**
 * @file   ext_plugin_fixture.cpp
 * @brief  Stand-in model-extension plugin for unittest_ext_model. Built three
 *         ways: a plugin that matches the host's headers, one built against an
 *         older ABI, and one without the ABI query at all.
 */
#include "ext_model_abi.h"

#include <cstdint>

namespace {
const ModelDescriptor kDescriptor = {"fixture-model",
                                     "fixture",
                                     "Fixture Model",
                                     QDA_RUNTIME_NATIVE,
                                     1u,
                                     QDA_CAP_STREAMING,
                                     "fixture-model",
                                     "FixtureForCausalLM",
                                     nullptr};
} // namespace

#if !defined(FIXTURE_NO_ABI_INFO)
extern "C" __attribute__((visibility("default"))) const QdaExtAbiInfo *
qai_ext_model_abi_info(void) {
#if defined(FIXTURE_STALE_ABI)
  // What a plugin compiled against the 8-field descriptor reports.
  static const QdaExtAbiInfo info = {0u, 56u, 200u, 0u};
#else
  static const QdaExtAbiInfo info = quick_dot_ai::currentExtAbiInfo();
#endif
  return &info;
}
#endif

extern "C" __attribute__((visibility("default"))) int
qai_ext_model_register(void) {
  quick_dot_ai::register_model_descriptor(&kDescriptor);
  return 1;
}
