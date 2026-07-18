// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_compute_ops.h
 * @date    29 Jul 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   CUDA ComputeOps subclass: the op table the cuda context installs
 *          via ContextData::setComputeOps(), so the backend-neutral layers
 *          dispatch device work through getOps() instead of calling CUDA
 *          kernels directly. Inherits CpuComputeOps (not the abstract
 *          ComputeOps base): engine=cuda tensors are Unified Memory
 *          (host-coherent), so every op the CUDA backend does not accelerate
 *          runs correctly via the CPU implementation over the managed
 *          buffers. This class overrides only the element-wise decode ops
 *          (swiglu / scalar_mul / softcap); each override falls back to the
 *          inherited host body when its device contract is not met.
 */

#ifndef __CUDA_COMPUTE_OPS_H__
#define __CUDA_COMPUTE_OPS_H__

#include <cpu_ops_table.h>

namespace nntrainer {

/**
 * @brief CUDA op table: CpuComputeOps plus device overrides for the
 *        element-wise decode kernels. The host bodies stay correct on the
 *        UVM buffers, so an override is only added where a device kernel
 *        exists; everything else inherits.
 */
class CudaComputeOps : public CpuComputeOps {
public:
  /**
   * @brief SwiGLU whole-op: device-resident fp16 one-kernel fast path
   *        (cuda_swiglu_fp16) under the residency gates, else the inherited
   *        host body.
   */
  void swiglu(const Tensor &in1, const Tensor &in2, Tensor &out,
              unsigned int active_rows, unsigned int row_offset) override;

  /**
   * @brief Scalar multiply whole-op: opt-in (NNTR_CUDA_ELTWISE) fp16 device
   *        kernel (cuda_scalar_mul_fp16), else drain-then-host fallback.
   */
  void scalar_mul(const Tensor &in, Tensor &out, float scale) override;

  /**
   * @brief Logit soft-capping whole-op: fp16 device kernel
   *        (cuda_softcap_fp16) on device-accessible logits, else the
   *        inherited host body. Carries the terminal pipeline drain for the
   *        selective-sync path (first host-read point of the logits).
   */
  void softcap(const Tensor &in, Tensor &out, float cap, int act_type) override;

  /**
   * @brief RMSNorm whole-op: block-per-row fp16 device kernel
   *        (cuda_rmsnorm_fp16, FP32 sum-of-squares) for decode-sized row
   *        counts on device-accessible tensors, else this backend's own
   *        fused host fallback (also FP32-accumulated) after the async
   *        coherence drain. Deliberately does NOT delegate to the inherited
   *        CpuComputeOps::rms_norm: the fallback here is the fused
   *        normalize*gamma loop this backend has always run, kept
   *        bit-for-bit.
   */
  void rms_norm(const Tensor &in, Tensor &out, const Tensor &gamma,
                float epsilon, unsigned int active_rows,
                unsigned int row_offset) override;

  /**
   * @brief FC GEMM whole-op: output = input * weight. QS4CX weight -> fused
   *        dequant-GEMM on device, consuming the PLAIN nibble payload in
   *        place (single weight copy, no UVM duplicate), else the inherited
   *        host dot after the async coherence drain. QINT4 never reaches
   *        here: layer_context coerces it to QS4CX at init.
   */
  void fc(Tensor &input, Tensor &weight, Tensor &output) override;

  // ── Copy ops (device-only aware) ───────────────────────────────────────
  // Under the device-only activation pool (NNTR_CUDA_DEV_ACT) an activation is
  // real device memory; Tensor::copy() -> the CpuComputeOps host loop would
  // fault on it. Route contiguous device-only copies through a stream-ordered
  // cudaMemcpyAsync; host / host-coherent UVM keep the CPU path.
  void scopy_fp32(const unsigned int N, const float *X, const unsigned int incX,
                  float *Y, const unsigned int incY) override;
#ifdef ENABLE_FP16
  void scopy_fp16(const unsigned int N, const _FP16 *X, const unsigned int incX,
                  _FP16 *Y, const unsigned int incY) override;
  // Converting copies with a device-only endpoint: stage through host temps
  // (synchronous; these do not occur inside graph capture today).
  void scopy_fp32_to_fp16(const unsigned int N, const float *X,
                          const unsigned int incX, _FP16 *Y,
                          const unsigned int incY) override;
  void scopy_fp16_to_fp32(const unsigned int N, const _FP16 *X,
                          const unsigned int incX, float *Y,
                          const unsigned int incY) override;
#endif
};

} // namespace nntrainer

#endif // __CUDA_COMPUTE_OPS_H__
