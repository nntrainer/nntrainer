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
 * Stage 1: gemm_q4_0_accel_fp32() is a stub that throws - it documents the
 * seam without depending on the Hexagon SDK / FastRPC bridge, which are not
 * available in this environment yet. The real implementation routes through
 * ggml-hexagon's cDSP session (see docs/nntrainer-htp-bridge.md in the
 * ggml-hexagon repo): copy activations into rpcmem, enqueue HTP_OP_MUL_MAT
 * against the weight buffer (already q4x4x2-tiled by
 * repack_q4_0_to_htp_q4x4x2 at quantize time - no repack needed here),
 * flush, copy the result back.
 */

#include <compute_ops.h>
#include <hexagon_compute_ops.h>

#include <stdexcept>

namespace nntrainer {

class HexagonComputeOps : public ComputeOps {
public:
  bool supports_gemm_q4_0_accel_fp32() const override { return true; }
  void gemm_q4_0_accel_fp32(void *matAdata, float *matBdata, float *matCdata,
                            unsigned int M, unsigned int N,
                            unsigned int K) override {
    (void)matAdata;
    (void)matBdata;
    (void)matCdata;
    (void)M;
    (void)N;
    (void)K;
    throw std::runtime_error(
      "HexagonComputeOps::gemm_q4_0_accel_fp32: not implemented yet - "
      "the cDSP FastRPC bridge lands once the Hexagon SDK is available to "
      "build/test against (see docs/nntrainer-htp-bridge.md)");
  }
};

ComputeOps *get_hexagon_ops() {
  static HexagonComputeOps instance;
  return &instance;
}

} // namespace nntrainer
