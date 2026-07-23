// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   hexagon_compute_ops.cpp
 * @date   23 July 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @brief  Hexagon cDSP ComputeOps subclass. Same shape as ClComputeOps
 * (cl_operations/cl_compute_ops.cpp): only the accelerator-only Q4_0 GEMM is
 * overridden, gated by its supports_*() predicate, so float_tensor.cpp's
 * existing dispatch (dotQnK) picks this up automatically once a layer's
 * weight Tensor carries this ComputeOps via its ContextData.
 *
 * Stage 2: gemm_q4_0_accel_fp32() dlopen/dlsyms
 * nntr_htp_bridge_gemm_q4_0() out of libggml-hexagon.so (see
 * ggml/src/ggml-hexagon/nntr-htp-bridge.cpp in the ggml-hexagon repo) and
 * calls straight into it. That bridge function reuses ggml-hexagon's own
 * cDSP session machinery directly (ggml_hexagon_session::enqueue_op/flush)
 * without going through ggml's graph/backend scheduler, since nntrainer has
 * no ggml_cgraph of its own to hand it. matAdata is already q4x4x2-tiled by
 * repack_q4_0_to_htp_q4x4x2 at quantize time, so no repack happens here or
 * on the DSP side.
 *
 * UNVERIFIED: no Hexagon hardware was available while writing this, so
 * dlopen/dlsym wiring and the ggml-hexagon-side bridge are only
 * compile-checked (Android arm64 cross-build), not run. The M/N/K -> tensor
 * shape mapping in nntr-htp-bridge.cpp is my best-effort read of
 * ggml_mul_mat's convention, not confirmed against real hardware or a
 * reference GEMM - see the UNVERIFIED note in that file before trusting
 * output values.
 */

#include <compute_ops.h>
#include <hexagon_compute_ops.h>

#include <dlfcn.h>

#include <mutex>
#include <sstream>
#include <stdexcept>

namespace nntrainer {

namespace {

using nntr_htp_bridge_gemm_q4_0_fn = int (*)(const void *, const float *,
                                              float *, unsigned int,
                                              unsigned int, unsigned int);

/**
 * @brief Lazily dlopen libggml-hexagon.so and dlsym the bridge entry point.
 * Cached for the process lifetime - dlopen/dlsym cost is not worth paying
 * per GEMM call, and ggml-hexagon's own session (kept alive across calls on
 * its side) already assumes one long-lived library handle.
 */
nntr_htp_bridge_gemm_q4_0_fn get_bridge_fn() {
  static nntr_htp_bridge_gemm_q4_0_fn fn = [] {
    void *handle = dlopen("libggml-hexagon.so", RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
      std::ostringstream oss;
      oss << "HexagonComputeOps: dlopen(libggml-hexagon.so) failed: "
          << dlerror();
      throw std::runtime_error(oss.str());
    }

    void *sym = dlsym(handle, "nntr_htp_bridge_gemm_q4_0");
    if (!sym) {
      std::ostringstream oss;
      oss << "HexagonComputeOps: dlsym(nntr_htp_bridge_gemm_q4_0) failed: "
          << dlerror();
      throw std::runtime_error(oss.str());
    }

    return reinterpret_cast<nntr_htp_bridge_gemm_q4_0_fn>(sym);
  }();

  return fn;
}

} // namespace

class HexagonComputeOps : public ComputeOps {
public:
  bool supports_gemm_q4_0_accel_fp32() const override { return true; }
  void gemm_q4_0_accel_fp32(void *matAdata, float *matBdata, float *matCdata,
                            unsigned int M, unsigned int N,
                            unsigned int K) override {
    static std::mutex bridge_init_mutex;
    nntr_htp_bridge_gemm_q4_0_fn fn;
    {
      // get_bridge_fn()'s static-local init is already thread-safe (C++11
      // magic statics); the mutex here only serializes the dlopen/dlsym
      // path itself against concurrent first-callers so a failed attempt
      // from one thread can't race a second attempt from another.
      std::lock_guard<std::mutex> lock(bridge_init_mutex);
      fn = get_bridge_fn();
    }

    int rc = fn(matAdata, matBdata, matCdata, M, N, K);
    if (rc != 0) {
      throw std::runtime_error(
        "HexagonComputeOps::gemm_q4_0_accel_fp32: "
        "nntr_htp_bridge_gemm_q4_0 failed (see log for details)");
    }
  }
};

ComputeOps *get_hexagon_ops() {
  static HexagonComputeOps instance;
  return &instance;
}

} // namespace nntrainer
