// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 j2z0.lee <j2z0.lee@ax.samsung.com>
 *
 * @file   htp_context.cpp
 * @date   19 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author j2z0.lee <j2z0.lee@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  HTP compute Context implementation.
 */

#ifdef ENABLE_HEXKL

#include <htp_context.h>

#include <app_context.h>
#include <compute_ops.h>
#include <context_data.h>
#include <htp_backend.h>

namespace nntrainer {

std::unique_ptr<nntrainer::Layer>
HtpContext::createLayerObject(const std::string &type,
                              const std::vector<std::string> &properties) {
  // HTP has no custom layers; reuse the CPU app context's factories.
  return AppContext::Global().createLayerObject(type, properties);
}

std::unique_ptr<nntrainer::Layer>
HtpContext::createLayerObject(const int int_key,
                              const std::vector<std::string> &properties) {
  return AppContext::Global().createLayerObject(int_key, properties);
}

void HtpContext::initialize() noexcept {
  // Ensure the CPU backend ops table exists — it is both the fallback
  // and the base class of HtpComputeOps. Go through ensureComputeOps()
  // rather than init_backend() directly: init_backend() is not
  // idempotent (__ggml_init, __openblas_set_num_threads), and
  // ensureComputeOps() is the call_once funnel every other Context
  // routes through.
  ensureComputeOps();

  auto cd = getContextData();
  if (!cd)
    return;

  // Bind the ops table lazily. A CPU-only API handle must not open the DSP
  // before a later HTP load supplies its library directory and fixed options.
  // The accelerated operations themselves check HtpBackend::enabled().
  try {
    cd->setComputeOps(get_htp_ops());
  } catch (...) {
    cd->setComputeOps(get_cpu_ops());
  }
}

} // namespace nntrainer

#endif // ENABLE_HEXKL
