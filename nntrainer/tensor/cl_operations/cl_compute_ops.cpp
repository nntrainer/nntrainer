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
 * float_tensor.cpp's dispatch sites already follow.
 *
 * The second family is the whole-op table a backend-neutral layer calls
 * without asking a predicate first: fc, apply_activation, layer_norm,
 * activation, residual_op, geglu, swiglu and the scopy family. These have no
 * supports_*() escape hatch, so every op a layer registered on the gpu engine
 * can reach has to resolve here — the base ComputeOps default throws, and a
 * layer has nowhere to catch that. The list is closed against the layers this
 * context registers: FullyConnectedLayerCl (fc, apply_activation,
 * fc_prebuild_weight), AdditionLayer (residual_op), SwiGLULayer (swiglu),
 * LayerNormalizationLayer (layer_norm), ActivationLayer (activation), plus
 * Tensor::copy (scopy). ActivationLayer is the one entry that can still
 * throw, and only for a mode with no OpenCL kernel: supports_activation() is
 * the query, and the message names the fix.
 *
 * This file is what unblocks GPU dispatch end-to-end:
 *   ClContext (Engine-registered) -> ContextData -> ClComputeOps
 *   -> nntrainer::gemm_q4_0_async_cl(...) -> OpenCL kernel queue.
 */

#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include <blas_kernel_interface.h> // add_i_cl, dotCl, dotCl_v8c
#include <blas_kernels.h>
#include <common_properties.h> // ActivationType, the act_type int encoding
#include <compute_ops.h>
#include <embedding_pool_cl_op.h>
#include <geglu_cl_op.h>
#include <gelu_cl_op.h>
#include <layernorm_cl_op.h>
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

  // ── Whole-ops (Tensor level) ──────────────────────────────────
  // Fully-connected matmul, for the neutral fully-connected layer: the layer
  // keeps the weight and bias binding and the shape logic, and only the GEMM
  // comes here. dotCl picks the dot, GEMV or GEMM kernel from the operand
  // shapes, exactly as the layer used to do inline.
  //
  // dotCl is contracted to PRODUCE its output, not accumulate into it, which
  // is why no zero-fill precedes this call. One shape does not honour that
  // yet: the FP32 general-GEMM branch still dispatches through the CLBlast
  // wrapper with beta = 1.0, so it reads the destination it is about to
  // overwrite. That branch is deleted by the CLBlast removal this PR depends
  // on, which routes the same shape to sgemm_cl -- which stores, like the
  // other three shapes and like the whole FP16 branch. Do not add a zero-fill
  // here to paper over it: that would cost a full-output memset on every
  // forward for one transitional branch.
  //
  // A quantized weight goes to the v8c int8xint4 GEMM first. dotCl_v8c
  // returns false rather than throwing when it declines the call (the weight
  // is not int4, the shape is outside what the kernel covers, or the path is
  // switched off), so the fallbacks below stay reachable: a quantized weight
  // the kernel declined is dotted on the host, and everything else keeps the
  // dotCl route this layer has always used.
  void fc(Tensor &input, Tensor &weight, Tensor &output) override {
    if (nntrainer::dotCl_v8c(input, weight, output))
      return;

    // The v8c GEMM produces its output; the fallbacks below accumulate into
    // theirs, so zero it here and only here.
    output.setZero();

    switch (weight.getDataType()) {
    case ml::train::TensorDim::DataType::QINT4:
    case ml::train::TensorDim::DataType::QS4CX:
    case ml::train::TensorDim::DataType::Q4_0:
    case ml::train::TensorDim::DataType::Q4_K:
    case ml::train::TensorDim::DataType::Q6_K:
      // A quantized weight has no GPU kernel once v8c declines it: dispatch
      // the host dot, which knows every quantization the CPU backend does.
      input.dot(weight, output, false, false);
      break;
    default:
      nntrainer::dotCl(input, weight, output);
      break;
    }
  }

  // Build the device-side v8c backing for this weight at load time, so the
  // first prefill does not pay the one-time repack. A no-op when the weight
  // is not one the v8c GEMM can take.
  void fc_prebuild_weight(Tensor &weight) override {
    nntrainer::dotCl_v8c_prebuild_weight(weight);
  }

  // The fused activation epilogue, which the fully-connected layer applies to
  // its own output in place. It runs on the host table: the operand is the
  // tensor the GEMM just read back, so the CPU table works on exactly the
  // memory the kernels staged and yields the values a standalone
  // ActivationLayer would. Unlike ::activation below, this one accepts every
  // mode, because a fused epilogue has no way to decline one. Replacing it
  // with a kernel is a residency refinement and touches no caller.
  void apply_activation(Tensor &out, int act_type) override {
    get_cpu_ops()->apply_activation(out, act_type);
  }

  // The gated pairs: (gate, up) -> out, element-wise, one kernel each.
  void geglu(const Tensor &in1, const Tensor &in2, Tensor &out,
             unsigned int active_rows, unsigned int row_offset) override {
    nntrainer::geglu_cl_op(in1, in2, out, active_rows, row_offset);
  }
  void swiglu(const Tensor &in1, const Tensor &in2, Tensor &out,
              unsigned int active_rows, unsigned int row_offset) override {
    nntrainer::swiglu_cl_op(in1, in2, out, active_rows, row_offset);
  }

  // LayerNorm over the last axis. The neutral LayerNormalizationLayer owns
  // the axis contract and only dispatches here when its property matches, so
  // this op never sees a property.
  void layer_norm(const Tensor &in, Tensor &out, const Tensor &gamma,
                  const Tensor &beta, float epsilon, unsigned int active_rows,
                  unsigned int row_offset) override {
    nntrainer::layernorm_cl_op(in, out, gamma, beta, epsilon, active_rows,
                               row_offset);
  }

  // Element-wise activation. Only gelu and tanh_gelu have OpenCL kernels;
  // every other mode throws rather than quietly running a host loop, because a
  // tensor on this context may live in device memory the host has unmapped,
  // where a host loop is not merely slower but wrong. Which mode a backend can
  // serve is a backend question, so the mapping lives here and not in a Layer.
  void activation(const Tensor &in, Tensor &out, int act_type,
                  unsigned int active_rows, unsigned int row_offset) override {
    switch (static_cast<ActivationType>(act_type)) {
    case ActivationType::ACT_GELU:
      nntrainer::gelu_cl_op(in, out, /*mode=*/0, active_rows, row_offset);
      return;
    case ActivationType::ACT_TANH_GELU:
      nntrainer::gelu_cl_op(in, out, /*mode=*/1, active_rows, row_offset);
      return;
    default:
      throw std::invalid_argument(
        "ClComputeOps::activation: only gelu and tanh_gelu are accelerated on "
        "this backend; use the cpu engine for the other activations");
    }
  }

  bool supports_activation(int act_type) const override {
    const auto type = static_cast<ActivationType>(act_type);
    return type == ActivationType::ACT_GELU ||
           type == ActivationType::ACT_TANH_GELU;
  }

  // One residual-add operand, for the neutral AdditionLayer. FP32 same-size
  // operands take a host copy/add: the FP32 addition kernel reads its result
  // back into the caller's pointer, which is the very read into shared memory
  // that does not land (see the FP32 GEMM read-back), and both operands are
  // host-addressable here anyway.
  void residual_op(Tensor &hidden, const Tensor &input,
                   bool accumulate) override {
    const auto fp32 = ml::train::TensorDim::DataType::FP32;
    if (hidden.getDataType() == fp32 && input.getDataType() == fp32 &&
        hidden.size() == input.size()) {
      const size_t n = hidden.size();
      float *out = hidden.getData<float>();
      const float *in = input.getData<float>();
      if (!accumulate) {
        std::memcpy(out, in, n * sizeof(float));
      } else {
        for (size_t i = 0; i < n; ++i)
          out[i] += in[i];
      }
      return;
    }

    if (!accumulate) {
      hidden.copy(input);
    } else {
      nntrainer::add_i_cl(hidden, input);
    }
  }

  // Tensor::copy() reaches the table with no supports_*() guard, so without
  // these a copy of a tensor on this context would throw "not implemented" --
  // which the residual copy above does on the FP16 path. A host loop is
  // correct for host pointers and for host-coherent shared memory; moving the
  // copy onto a kernel is a residency refinement, not a correctness one.
  void scopy_fp32(const unsigned int N, const float *X, const unsigned int incX,
                  float *Y, const unsigned int incY) override {
    for (unsigned int i = 0; i < N; ++i)
      Y[i * incY] = X[i * incX];
  }

  // The two row reductions of the sentence-embedding tail. Their callers are
  // backend-neutral layers reaching them through in.getOps(), and before these
  // overrides the same maths arrived as Tensor::average(2) -> sgemv_fp32 and
  // normalization_i(3) -> snrm2_fp32 + sscal_fp32, none of which this table
  // implements: the tail threw on a gpu-context tensor rather than running.
  void mean_rows(const Tensor &in, Tensor &out, unsigned int active_rows,
                 unsigned int row_offset) override {
    mean_rows_cl_op(in, out, active_rows, row_offset);
  }
  void l2_normalize_rows(const Tensor &in, Tensor &out,
                         float epsilon) override {
    l2_normalize_rows_cl_op(in, out, epsilon);
  }

#ifdef ENABLE_FP16
  void scopy_fp16(const unsigned int N, const _FP16 *X, const unsigned int incX,
                  _FP16 *Y, const unsigned int incY) override {
    for (unsigned int i = 0; i < N; ++i)
      Y[i * incY] = X[i * incX];
  }
  // Mixed precision: an FP32 source feeding an FP16 graph, or an FP16 result
  // read back as FP32, both route here on this backend.
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
};

ComputeOps *get_cl_ops() {
  static ClComputeOps instance;
  return &instance;
}

} // namespace nntrainer
