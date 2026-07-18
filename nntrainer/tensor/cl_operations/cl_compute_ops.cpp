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
 * Only the accelerator-specific ops (Q4_0 batch / accel,
 * INT4 batch / accel) are overridden, with their supports_*()
 * predicates returning true. All other ops fall through to the
 * base ComputeOps default (which throws), so callers rely on
 * supports_*() to decide whether to use this path or fall back
 * to a CPU ops table — exactly the contract float_tensor.cpp's
 * dispatch sites already follow.
 *
 * This file is what unblocks GPU dispatch end-to-end:
 *   ClContext (Engine-registered) -> ContextData -> ClComputeOps
 *   -> nntrainer::gemm_q4_0_async_cl(...) -> OpenCL kernel queue.
 */

#include <cstdlib>
#include <cstring>

#include <acti_func.h>
#include <attention_kernels.h> // gpu_copy_f16_cl
#include <blas_kernel_interface.h>
#include <blas_kernels.h>
#include <compute_ops.h>
#include <geglu_cl_op.h>
#include <swiglu_cl_op.h>
#include <tensor.h>

namespace nntrainer {

/**
 * @brief OpenCL ComputeOps table: the accelerator-backed subset of the
 *        ComputeOps interface, dispatched through ClContext's ContextData.
 */
class ClComputeOps : public ComputeOps {
public:
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

  // Plain elementwise copy (Y = X). Tensor::copy() calls this unconditionally
  // (no supports_*() guard), so the GPU backend must provide it or copy()
  // throws "not implemented" -- which blocks any GPU layer that copies a
  // tensor (e.g. the residual first-input copy). A host copy is correct for
  // host and (host-coherent) SVM pointers.
  void scopy_fp32(const unsigned int N, const float *X, const unsigned int incX,
                  float *Y, const unsigned int incY) override {
    for (unsigned int i = 0; i < N; ++i)
      Y[i * incY] = X[i * incX];
  }

#ifdef ENABLE_FP16
  // fp16 counterpart (FP16-activation graphs).
  void scopy_fp16(const unsigned int N, const _FP16 *X, const unsigned int incX,
                  _FP16 *Y, const unsigned int incY) override {
    for (unsigned int i = 0; i < N; ++i)
      Y[i * incY] = X[i * incX];
  }
  // Mixed-precision host copies: Tensor::copy()/copyData() across dtypes
  // routes here on the GPU backend.
  void scopy_fp32_to_fp16(const unsigned int N, const float *X,
                          const unsigned int incX, _FP16 *Y,
                          const unsigned int incY) override {
    for (unsigned int i = 0; i < N; ++i)
      Y[i * incY] = static_cast<_FP16>(X[i * incX]);
  }
  void scopy_fp16_to_fp32(const unsigned int N, const _FP16 *X,
                          const unsigned int incX, float *Y,
                          const unsigned int incY) override {
    for (unsigned int i = 0; i < N; ++i)
      Y[i * incY] = static_cast<float>(X[i * incX]);
  }
#endif

  // Residual-add operand on the GPU residency path: an FP32 host fast-path,
  // else the FP16 cl_mem/SVM-resident copy/add.
  void residual_op(Tensor &hidden, const Tensor &input,
                   bool accumulate) override {
    const bool fp32_fast =
      hidden.getDataType() == ml::train::TensorDim::DataType::FP32;
    if (fp32_fast && hidden.size() == input.size() &&
        input.getDataType() == ml::train::TensorDim::DataType::FP32) {
      const size_t n = hidden.size();
      if (!accumulate) {
        std::memcpy(hidden.getData<uint8_t>(), input.getData<uint8_t>(),
                    n * sizeof(float));
      } else {
        float *out = hidden.getData<float>();
        const float *in = input.getData<float>();
        for (size_t k = 0; k < n; ++k)
          out[k] += in[k];
      }
    } else if (!accumulate) {
      // First residual operand: copy input -> hidden.
#ifdef ENABLE_FP16
      if (nntrainer::clmem_residual_op_cl(hidden, input,
                                          /** accumulate */ false))
        return;
      const bool svm16 =
        hidden.getDataType() == ml::train::TensorDim::DataType::FP16 &&
        input.getDataType() == ml::train::TensorDim::DataType::FP16 &&
        hidden.getMemoryData() && hidden.getMemoryData()->isSVM() &&
        input.getMemoryData() && input.getMemoryData()->isSVM() &&
        hidden.size() == input.size();
      static const bool add_drain = std::getenv("NNTR_ADD_DRAIN") != nullptr;
      if (svm16 && nntrainer::gpu_copy_f16_cl(
                     reinterpret_cast<const uint16_t *>(input.getData<_FP16>()),
                     reinterpret_cast<uint16_t *>(hidden.getData<_FP16>()),
                     (unsigned int)hidden.size(), /** svm */ true,
                     /** in_clmem */ nullptr, /** out_clmem */ nullptr,
                     /** drain */ add_drain)) {
        // GPU copy done.
      } else
#endif
        hidden.copy(input);
    } else {
#ifdef ENABLE_FP16
      if (nntrainer::clmem_residual_op_cl(hidden, input,
                                          /** accumulate */ true))
        return;
#endif
      nntrainer::add_i_cl(hidden, input);
    }
  }

  // Whole-op (Tensor-level) GLU dispatches. The neutral GeGLU/SwiGLU layers
  // call in1.getOps()->geglu/swiglu(...), which lands here on a CL-attached
  // tensor and forwards to the OpenCL kernel dispatch.
  void geglu(const Tensor &in1, const Tensor &in2, Tensor &out,
             unsigned int active_rows, unsigned int row_offset) override {
    nntrainer::geglu_cl_op(in1, in2, out, active_rows, row_offset);
  }
  void swiglu(const Tensor &in1, const Tensor &in2, Tensor &out,
              unsigned int active_rows, unsigned int row_offset) override {
    nntrainer::swiglu_cl_op(in1, in2, out, active_rows, row_offset);
  }

  /**
   * @brief Whole-op FC dispatch: try the v8c int8xint4 GPU GEMM first; on
   * rejection (env off / non-int4 weight / unsupported shape) fall back to
   * the generic quantized host dot or the CLBlast-backed dotCl.
   */
  void fc(Tensor &input, Tensor &weight, Tensor &output) override {
    // Pre-zero the output plane: redundant on the v8c path (it overwrites the
    // full MxN region) but kept for the fallbacks. Default OFF;
    // NNTR_FC_OUTZERO=1 restores it.
    static const bool fc_out_zero = []() {
      const char *e = std::getenv("NNTR_FC_OUTZERO");
      return e && e[0] == '1';
    }();
    if (fc_out_zero)
      output.setZero();
    if (!nntrainer::dotCl_v8c(input, weight, output)) {
      if (!fc_out_zero)
        output.setZero();
      // Static GPU_CLMEM residency: the host/SVM fallbacks below read input
      // and write output through host pointers only, so bridge a resident
      // input down first and raise a resident output afterwards -- a v8c
      // rejection must not leave a GPU_CLMEM tensor's planes inconsistent.
      nntrainer::clmem_lower_cl(input, 0);
      auto wt = weight.getDataType();
      if (wt == ml::train::TensorDim::DataType::QINT4 ||
          wt == ml::train::TensorDim::DataType::QS4CX ||
          wt == ml::train::TensorDim::DataType::Q4_0 ||
          wt == ml::train::TensorDim::DataType::Q4_K ||
          wt == ml::train::TensorDim::DataType::Q6_K) {
        input.dot(weight, output, false, false);
      } else {
        nntrainer::dotCl(input, weight, output);
      }
      nntrainer::clmem_raise_cl(output, 0);
    }
  }

  /**
   * @brief Eager v8c GPU weight build at load (so the first prefill does not
   * pay the one-time transform).
   */
  void fc_prebuild_weight(Tensor &weight) override {
    nntrainer::dotCl_v8c_prebuild_weight(weight);
  }

  /**
   * @brief Fused activation epilogue on the GPU FC. For now this runs the
   * same host ActiFunc as CpuComputeOps (value-identical, correct on
   * SVM-resident output); a GPU activation kernel fused into the GEMM
   * epilogue is a perf follow-up. The LLM GPU stack never reaches here (no
   * fc+activation), so it is inert until a GPU CNN/MLP sets
   * fused_activation.
   */
  void apply_activation(Tensor &out, int act_type) override {
    const auto at = static_cast<ActivationType>(act_type);
    if (at == ActivationType::ACT_NONE)
      return;
    ActiFunc f;
    if (out.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
      f.setActiFunc<_FP16>(at);
#else
      throw std::invalid_argument("apply_activation: fp16 needs enable-fp16");
#endif
    } else {
      f.setActiFunc<float>(at);
    }
    if (f.supportInPlace()) {
      f.run_fn(out, out);
    } else {
      Tensor in_copy = out.clone();
      f.run_fn(in_copy, out);
    }
  }
};

ComputeOps *get_cl_ops() {
  static ClComputeOps instance;
  return &instance;
}

} // namespace nntrainer
