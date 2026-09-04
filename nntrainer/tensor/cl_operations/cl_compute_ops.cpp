// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   cl_compute_ops.cpp
 * @date   25 April 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  OpenCL ComputeOps subclass — provides accelerated quantized
 *         GEMM/GEMV variants on top of the existing nntrainer
 *         OpenCL kernels in cl_operations/blas_kernels.cpp.
 *
 * Two families live here. The accelerator-specific ops (Q4_0 batch / accel,
 * INT4 batch / accel) are overridden with their supports_*() predicates
 * returning true; callers rely on supports_*() to decide whether to use this
 * path or fall back to a CPU ops table — exactly the contract
 * float_tensor.cpp's dispatch sites already follow. The whole-ops a
 * backend-neutral layer calls unconditionally (fc, apply_activation) have no
 * supports_*() escape hatch, so every op a layer registered on the gpu engine
 * reaches must resolve here: the base ComputeOps default throws, and a layer
 * has nowhere to catch that. Those are implemented below.
 *
 * This file is what unblocks GPU dispatch end-to-end:
 *   ClContext (Engine-registered) -> ContextData -> ClComputeOps
 *   -> nntrainer::gemm_q4_0_async_cl(...) -> OpenCL kernel queue.
 */

#include <blas_kernel_interface.h>
#include <blas_kernels.h>
#include <compute_ops.h>
#include <tensor.h>

namespace nntrainer {

class ClComputeOps : public ComputeOps {
public:
  /**
   * @brief Fully connected matmul on the OpenCL GEMM.
   *
   * This is the call FullyConnectedLayerCl made directly before the layer
   * became backend-neutral, so a gpu-engine graph keeps the same kernel and
   * the same numerics. dotCl() writes the result (it does not accumulate),
   * which is why no zero-fill precedes it.
   */
  void fc(Tensor &input, Tensor &weight, Tensor &output) override {
    dotCl(input, weight, output);
  }

  /**
   * @brief Activation epilogue, run on the host.
   *
   * A gpu-engine Tensor is host-allocated (the kernels stage it into device
   * buffers per call), so the CPU table operates on exactly the same memory
   * and yields the same values a standalone ActivationLayer would. There is no
   * whole-op activation kernel yet; this exists so a fused epilogue on the gpu
   * engine computes rather than throwing, and it can be replaced by a kernel
   * without touching a caller.
   */
  void apply_activation(Tensor &out, int act_type) override {
    get_cpu_ops()->apply_activation(out, act_type);
  }

  // ── Accelerator-only Q4_0 / INT4 GEMM/GEMV ────────────────
  bool supports_gemm_q4_0_batch_fp32() const override { return true; }
  void gemm_q4_0_batch_fp32(std::vector<void *> matAdata, float *matBdata,
                            std::vector<float *> matCdata, unsigned int M,
                            std::vector<unsigned int> N,
                            unsigned int K) override {
    nntrainer::gemm_q4_0_async_cl(matAdata, matBdata, matCdata, M, N, K);
  }

  bool supports_gemm_q4_0_accel_fp32() const override { return true; }
  void gemm_q4_0_accel_fp32(void *matAdata, float *matBdata, float *matCdata,
                            unsigned int M, unsigned int N,
                            unsigned int K) override {
    nntrainer::gemm_q4_0_cl(matAdata, matBdata, matCdata, M, N, K);
  }

  bool supports_gemv_int4_batch_fp32() const override { return true; }
  void gemv_int4_batch_fp32(std::vector<void *> weights,
                            std::vector<uint16_t *> scales, float *input,
                            std::vector<float *> outputs, unsigned int K,
                            std::vector<unsigned int> Ns,
                            unsigned int group_size) override {
    nntrainer::gemv_int4_async_cl(weights, scales, input, outputs, K, Ns,
                                  group_size);
  }

  bool supports_gemm_int4_batch_fp32() const override { return true; }
  void gemm_int4_batch_fp32(float *input, std::vector<void *> weights,
                            std::vector<uint16_t *> scales,
                            std::vector<float *> matCdata, unsigned int M,
                            std::vector<unsigned int> Ns, unsigned int K,
                            unsigned int group_size) override {
    nntrainer::gemm_int4_async_cl(input, weights, scales, matCdata, M, Ns, K,
                                  group_size);
  }

  bool supports_gemv_int4_accel_fp32() const override { return true; }
  void gemv_int4_accel_fp32(char *weight, uint16_t *scale, float *input,
                            float *output, unsigned int K, unsigned int N,
                            unsigned int group_size) override {
    nntrainer::gemv_int4_cl(weight, scale, input, output, K, N, group_size);
  }

  bool supports_sgemm_int4_accel_fp32() const override { return true; }
  void sgemm_int4_accel_fp32(float *input, char *weight, uint16_t *scale,
                             float *output, unsigned int M, unsigned int N,
                             unsigned int K, unsigned int group_size) override {
    nntrainer::sgemm_int4_cl(input, weight, scale, output, M, N, K, group_size);
  }
};

ComputeOps *get_cl_ops() {
  static ClComputeOps instance;
  return &instance;
}

} // namespace nntrainer
