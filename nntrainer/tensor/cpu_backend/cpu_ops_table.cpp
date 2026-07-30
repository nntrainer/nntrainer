// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   cpu_ops_table.cpp
 * @date   04 April 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Unified CPU backend ComputeOps subclass.
 *
 * Single concrete ComputeOps subclass for ALL CPU targets (ARM /
 * x86 / fallback). The nntrainer::sgemm etc. functions are arch-
 * specialized — each arch_compute_backend.cpp defines its own body
 * — so a single forwarding wrapper is enough; build-time arch
 * dispatch picks the right symbol at link time.
 */

#include "cpu_ops_table.h"

#include <cmath>
#include <stdexcept>

#include <acti_func.h>
#include <tensor.h>

namespace nntrainer {

ComputeOps *get_cpu_ops() {
  static CpuComputeOps instance;
  return &instance;
}

namespace {
// gelu (tanh approximation, gelu_pytorch_tanh) -- same constants as the OpenCL
// geglu_cl / CUDA geglu kernels, so the host path is numerically consistent.
inline float gelu_tanh(float x) {
  const float k = 0.7978845608028654f; // sqrt(2/pi)
  return 0.5f * x * (1.0f + std::tanh(k * (x + 0.044715f * x * x * x)));
}
// silu (numerically stable: x/(1+exp(-x)) == x*sigmoid(x)) -- matches the
// OpenCL swiglu_cl kernel exactly (avoids the x*exp(x)/(1+exp(x)) overflow).
inline float silu(float x) { return x / (1.0f + std::exp(-x)); }
// sigmoid -- matches the OpenCL sigmoid_glu/sigmoid_add kernels and the CUDA
// ELTWISE_SRC form (1/(1+exp(-x))) so the three backends agree token-for-token.
inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }
} // namespace

// out = gelu_tanh(in1) * in2 over rows [row_offset, row_offset+active_rows).
// row_offset is 0 on every current caller (the live token is at the buffer
// base for the host/SVM/UVM paths); the offset is honored for generality.
void CpuComputeOps::geglu(const Tensor &in1, const Tensor &in2, Tensor &out,
                          unsigned int active_rows, unsigned int row_offset) {
  const unsigned int dim2 = in1.width();
  const size_t elem_off = (size_t)row_offset * dim2;
  const size_t n = (size_t)active_rows * dim2;
  const auto dt = in1.getDataType();

  if (dt == ml::train::TensorDim::DataType::FP32) {
    const float *a = in1.getData<float>() + elem_off;
    const float *b = in2.getData<float>() + elem_off;
    float *o = out.getData<float>() + elem_off;
    for (size_t i = 0; i < n; ++i)
      o[i] = gelu_tanh(a[i]) * b[i];
#ifdef ENABLE_FP16
  } else if (dt == ml::train::TensorDim::DataType::FP16) {
    const _FP16 *a = in1.getData<_FP16>() + elem_off;
    const _FP16 *b = in2.getData<_FP16>() + elem_off;
    _FP16 *o = out.getData<_FP16>() + elem_off;
    for (size_t i = 0; i < n; ++i)
      o[i] = static_cast<_FP16>(gelu_tanh((float)a[i]) * (float)b[i]);
#endif
  } else {
    throw std::invalid_argument("CpuComputeOps::geglu: unsupported data type");
  }
}

// out = silu(in1) * in2 over rows [row_offset, row_offset+active_rows).
void CpuComputeOps::swiglu(const Tensor &in1, const Tensor &in2, Tensor &out,
                           unsigned int active_rows, unsigned int row_offset) {
  const unsigned int dim2 = in1.width();
  const size_t elem_off = (size_t)row_offset * dim2;
  const size_t n = (size_t)active_rows * dim2;
  const auto dt = in1.getDataType();

  if (dt == ml::train::TensorDim::DataType::FP32) {
    const float *a = in1.getData<float>() + elem_off;
    const float *b = in2.getData<float>() + elem_off;
    float *o = out.getData<float>() + elem_off;
    for (size_t i = 0; i < n; ++i)
      o[i] = silu(a[i]) * b[i];
#ifdef ENABLE_FP16
  } else if (dt == ml::train::TensorDim::DataType::FP16) {
    const _FP16 *a = in1.getData<_FP16>() + elem_off;
    const _FP16 *b = in2.getData<_FP16>() + elem_off;
    _FP16 *o = out.getData<_FP16>() + elem_off;
    for (size_t i = 0; i < n; ++i)
      o[i] = static_cast<_FP16>(silu((float)a[i]) * (float)b[i]);
#endif
  } else {
    throw std::invalid_argument("CpuComputeOps::swiglu: unsupported data type");
  }
}

// out = sigmoid(in1) * in2 over rows [row_offset, row_offset+active_rows).
// A sigmoid-gated attention output gate is one example. FP32 accumulation
// (upcast fp16 -> float) so the LRA-MLP intermediates do not overflow fp16.
void CpuComputeOps::sigmoid_glu(const Tensor &in1, const Tensor &in2,
                                Tensor &out, unsigned int active_rows,
                                unsigned int row_offset) {
  const unsigned int dim2 = in1.width();
  const size_t elem_off = (size_t)row_offset * dim2;
  const size_t n = (size_t)active_rows * dim2;
  const auto dt = in1.getDataType();

  if (dt == ml::train::TensorDim::DataType::FP32) {
    const float *a = in1.getData<float>() + elem_off;
    const float *b = in2.getData<float>() + elem_off;
    float *o = out.getData<float>() + elem_off;
    for (size_t i = 0; i < n; ++i)
      o[i] = sigmoidf(a[i]) * b[i];
#ifdef ENABLE_FP16
  } else if (dt == ml::train::TensorDim::DataType::FP16) {
    const _FP16 *a = in1.getData<_FP16>() + elem_off;
    const _FP16 *b = in2.getData<_FP16>() + elem_off;
    _FP16 *o = out.getData<_FP16>() + elem_off;
    for (size_t i = 0; i < n; ++i)
      o[i] = static_cast<_FP16>(sigmoidf((float)a[i]) * (float)b[i]);
#endif
  } else {
    throw std::invalid_argument(
      "CpuComputeOps::sigmoid_glu: unsupported data type");
  }
}

// out = sigmoid(in1) + in2 over rows [row_offset, row_offset+active_rows).
// A per-layer-embedding (PLE) mix path (method=1) is one example. FP32
// accumulation as above.
void CpuComputeOps::sigmoid_add(const Tensor &in1, const Tensor &in2,
                                Tensor &out, unsigned int active_rows,
                                unsigned int row_offset) {
  const unsigned int dim2 = in1.width();
  const size_t elem_off = (size_t)row_offset * dim2;
  const size_t n = (size_t)active_rows * dim2;
  const auto dt = in1.getDataType();

  if (dt == ml::train::TensorDim::DataType::FP32) {
    const float *a = in1.getData<float>() + elem_off;
    const float *b = in2.getData<float>() + elem_off;
    float *o = out.getData<float>() + elem_off;
    for (size_t i = 0; i < n; ++i)
      o[i] = sigmoidf(a[i]) + b[i];
#ifdef ENABLE_FP16
  } else if (dt == ml::train::TensorDim::DataType::FP16) {
    const _FP16 *a = in1.getData<_FP16>() + elem_off;
    const _FP16 *b = in2.getData<_FP16>() + elem_off;
    _FP16 *o = out.getData<_FP16>() + elem_off;
    for (size_t i = 0; i < n; ++i)
      o[i] = static_cast<_FP16>(sigmoidf((float)a[i]) + (float)b[i]);
#endif
  } else {
    throw std::invalid_argument(
      "CpuComputeOps::sigmoid_add: unsupported data type");
  }
}

// hidden = input (copy) or hidden += input (add) on the host buffer. Mirrors
// the core AdditionLayer's per-input copy()/add_i() (correct for host and UVM).
void CpuComputeOps::residual_op(Tensor &hidden, const Tensor &input,
                                bool accumulate) {
  if (accumulate)
    hidden.add_i(input);
  else
    hidden.copy(input);
}

// output = input * weight. Host Tensor::dot (CPU/UVM FC matmul). The CL/CUDA
// quantized GEMM paths override this in their ComputeOps subclasses.
void CpuComputeOps::fc(Tensor &input, Tensor &weight, Tensor &output) {
  input.dot(weight, output, false, false);
}

// Fused activation epilogue on the host: build the SAME ActiFunc the standalone
// ActivationLayer would (so the fused result is value-identical), and run it in
// place when the activation supports it (relu/sigmoid/tanh) or via a temp input
// copy otherwise — mirroring ActivationLayer::run_fn(input, output) exactly.
void CpuComputeOps::apply_activation(Tensor &out, int act_type) {
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

// out = in * scale on the host. The whole-op half of the neutral
// scalar-multiply layer: the layer keeps the chunk/step bookkeeping and this
// runs one chunk (the layer's former open-coded host body, unchanged).
void CpuComputeOps::scalar_mul(const Tensor &in, Tensor &out, float scale) {
  in.multiply(scale, out);
}

// out = cap * act(in / cap) on the host -- the neutral logit-softcapping
// layer's former open-coded chunk body, statement for statement: copy, scale
// down, activation, scale back up. The ActiFunc is rebuilt from act_type keyed
// on the chunk dtype (the apply_activation convention), which matches the
// activation the layer configured at finalize (the chunk dtype IS the
// activation dtype).
void CpuComputeOps::softcap(const Tensor &in, Tensor &out, float cap,
                            int act_type) {
  const auto at = static_cast<ActivationType>(act_type);
  ActiFunc f;
  if (out.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    f.setActiFunc<_FP16>(at);
#else
    throw std::invalid_argument("softcap: fp16 needs enable-fp16");
#endif
  } else {
    f.setActiFunc<float>(at);
  }
  out.copyData(in);
  in.multiply(1.0f / cap, out);
  f.run_fn(out, out);
  out.multiply(cap, out);
}

// out = in * rsqrt(mean(in^2)+eps) * gamma over rows
// [row_offset, row_offset+active_rows). The normalize half runs through the
// arch-dispatched width-wise intrinsics; both dtypes accumulate the sum of
// squares in FP32 — the FP16 kernel converts each lane up and FMAs in float
// ("squared accumulation across a 1024-wide row would overflow FP16's 65504
// ceiling", its own rationale) — so the whole-op contract holds on every CPU
// arch. The gamma half is the broadcast per-row multiply on the live window
// only, with gamma cloned to the activation dtype when the bin stores it at a
// different one (unquantized gamma is FP32 in FP32-weight bins).
void CpuComputeOps::rms_norm(const Tensor &in, Tensor &out, const Tensor &gamma,
                             float epsilon, unsigned int active_rows,
                             unsigned int row_offset) {
  const unsigned int width = in.width();
  const size_t elem_off = (size_t)row_offset * width;
  const auto dt = in.getDataType();

  if (dt == ml::train::TensorDim::DataType::FP32) {
    nntrainer::rms_norm_wrt_width_fp32_intrinsic(
      in.getData<float>() + elem_off, out.getData<float>() + elem_off,
      active_rows, width, epsilon);
#ifdef ENABLE_FP16
  } else if (dt == ml::train::TensorDim::DataType::FP16) {
    nntrainer::rms_norm_wrt_width_fp16_intrinsic(
      in.getData<_FP16>() + elem_off, out.getData<_FP16>() + elem_off,
      active_rows, width, epsilon);
#endif
  } else {
    throw std::invalid_argument(
      "CpuComputeOps::rms_norm: unsupported data type");
  }

  // gamma multiply over the live window only (rows outside it stay untouched).
  Tensor out_win = out.getSharedDataTensor(
    TensorDim(1, 1, active_rows, width, out.getDim().getTensorType()), elem_off,
    true);
  if (gamma.getDataType() != out.getDataType()) {
    Tensor gamma_cast = gamma.clone(out.getDataType());
    out_win.multiply_i(gamma_cast);
  } else {
    out_win.multiply_i(gamma);
  }
}

} // namespace nntrainer
