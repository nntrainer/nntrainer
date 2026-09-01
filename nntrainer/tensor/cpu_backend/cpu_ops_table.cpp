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

#include <cpu_ops_table.h>

#include <cmath>
#include <cstring>
#include <stdexcept>

#include <acti_func.h>
#include <tensor.h>

namespace nntrainer {

ComputeOps *get_cpu_ops() {
  static CpuComputeOps instance;
  return &instance;
}

namespace {

/**
 * @todo These scalar helpers are the reference form of the activation, and the
 * baseline every other backend table is checked against. A vectorized form
 * (NEON/AVX, riding the link-time arch dispatch that nntrainer::sgemm already
 * uses) is intended but not written yet.
 */

/** gelu, tanh approximation (gelu_pytorch_tanh) */
inline float gelu_tanh_f(float x) {
  constexpr float k = 0.7978845608028654f; // sqrt(2/pi)
  return 0.5f * x * (1.0f + std::tanh(k * (x + 0.044715f * x * x * x)));
}
/** silu, in the numerically stable x / (1 + exp(-x)) form */
inline float silu_f(float x) { return x / (1.0f + std::exp(-x)); }

/**
 * @brief Run a binary element-wise op over the row window, dispatching on the
 *        activation dtype once instead of per element. FP16 operands are
 *        computed in float and rounded back, so the two dtypes agree.
 * @todo This is a scalar reference implementation, and also the cross-backend
 * correctness baseline. NEON/AVX specializations can ride the existing
 * link-time arch dispatch (the nntrainer::sgemm pattern), and row-level
 * threading applies to prefill windows; neither is done yet.
 */
template <typename Op>
void elementwise2_rows(const Tensor &in1, const Tensor &in2, Tensor &out,
                       unsigned int active_rows, unsigned int row_offset,
                       const char *op_name, Op op) {
  const unsigned int width = in1.width();
  const size_t elem_off = (size_t)row_offset * width;
  const size_t n = (size_t)active_rows * width;

  // These are the reference implementations every backend table is checked
  // against, and they index in2/out with in1's dtype and stride, so a
  // disagreeing operand would be silently reinterpreted rather than rejected.
  if (in2.getDim() != in1.getDim() || out.getDim() != in1.getDim())
    throw std::invalid_argument(std::string("CpuComputeOps::") + op_name +
                                ": operands must share shape and dtype");

  if (n == 0)
    return;

  switch (in1.getDataType()) {
  case ml::train::TensorDim::DataType::FP32: {
    const float *a = in1.getData<float>() + elem_off;
    const float *b = in2.getData<float>() + elem_off;
    float *o = out.getData<float>() + elem_off;
    for (size_t i = 0; i < n; ++i)
      o[i] = op(a[i], b[i]);
    break;
  }
#ifdef ENABLE_FP16
  case ml::train::TensorDim::DataType::FP16: {
    const _FP16 *a = in1.getData<_FP16>() + elem_off;
    const _FP16 *b = in2.getData<_FP16>() + elem_off;
    _FP16 *o = out.getData<_FP16>() + elem_off;
    for (size_t i = 0; i < n; ++i)
      o[i] = static_cast<_FP16>(op((float)a[i], (float)b[i]));
    break;
  }
#endif
  default:
    throw std::invalid_argument(std::string("CpuComputeOps::") + op_name +
                                ": unsupported data type");
  }
}

/**
 * @brief (x - mean) * rsqrt(var + eps) * gamma + beta over `rows` rows of
 *        `width` elements. Mean and variance accumulate in FP32 regardless of
 *        the operand dtype.
 * @param T activation dtype
 * @param G weight dtype of gamma/beta, which need not equal T -- the core
 *          LayerNormalizationLayer requests them at the weight dtype.
 * @todo Scalar reference implementation (also the cross-backend correctness
 * baseline). An arch-specialized version can follow the existing
 * rms_norm_wrt_width_*_intrinsic pattern (x86/avx2_impl.h, arm/neon_impl.h) --
 * LayerNorm adds mean subtraction and gamma/beta but the row-wise reduction
 * structure is identical; row-level threading applies for prefill windows.
 * Not yet implemented.
 */
template <typename T, typename G>
void layernorm_rows(const T *x, const G *g, const G *b, T *y, unsigned int rows,
                    unsigned int width, float eps) {
  for (unsigned int r = 0; r < rows; ++r) {
    const T *xr = x + (size_t)r * width;
    T *yr = y + (size_t)r * width;

    float mean = 0.0f;
    for (unsigned int k = 0; k < width; ++k)
      mean += (float)xr[k];
    mean /= (float)width;

    float ssd = 0.0f;
    for (unsigned int k = 0; k < width; ++k) {
      const float d = (float)xr[k] - mean;
      ssd += d * d;
    }
    const float inv = 1.0f / std::sqrt(ssd / (float)width + eps);

    for (unsigned int k = 0; k < width; ++k)
      yr[k] = (T)((((float)xr[k] - mean) * inv) * (float)g[k] + (float)b[k]);
  }
}

/**
 * @brief GELU computed in float. The constants match __fallback_gelu_v2 /
 *        __fallback_tanh_gelu so the FP16 path agrees with the FP32 one.
 * @param tanh_mode false = erf-exact GELU, true = tanh approximation
 * @todo This is a scalar reference implementation, and also the cross-backend
 * correctness baseline. A NEON/AVX specialization can ride the existing
 * link-time arch dispatch (the nntrainer::sgemm pattern) the way the FP32 path
 * already reaches nntrainer::gelu_v2; it is not done yet.
 */
template <typename T>
void gelu_elems(const T *x, T *y, size_t n, bool tanh_mode) {
  for (size_t i = 0; i < n; ++i) {
    const float v = (float)x[i];
    const float r =
      tanh_mode
        ? 0.5f * v *
            (1.0f + std::tanh(0.7978845608f * (v + 0.044715f * v * v * v)))
        : 0.5f * v * (1.0f + std::erf(v * 0.7071067811f));
    y[i] = (T)r;
  }
}

} // namespace

void CpuComputeOps::geglu(const Tensor &in1, const Tensor &in2, Tensor &out,
                          unsigned int active_rows, unsigned int row_offset) {
  elementwise2_rows(in1, in2, out, active_rows, row_offset, "geglu",
                    [](float a, float b) { return gelu_tanh_f(a) * b; });
}

void CpuComputeOps::swiglu(const Tensor &in1, const Tensor &in2, Tensor &out,
                           unsigned int active_rows, unsigned int row_offset) {
  elementwise2_rows(in1, in2, out, active_rows, row_offset, "swiglu",
                    [](float a, float b) { return silu_f(a) * b; });
}

void CpuComputeOps::layer_norm(const Tensor &in, Tensor &out,
                               const Tensor &gamma, const Tensor &beta,
                               float epsilon, unsigned int active_rows,
                               unsigned int row_offset) {
  using DT = ml::train::TensorDim::DataType;
  const unsigned int width = in.width();
  const size_t elem_off = (size_t)row_offset * width;
  const DT dt = in.getDataType();
  const DT gt = gamma.getDataType(); // gamma and beta always share one dtype

  if (active_rows == 0 || width == 0)
    return;

  if (dt == DT::FP32 && gt == DT::FP32) {
    layernorm_rows(in.getData<float>() + elem_off, gamma.getData<float>(),
                   beta.getData<float>(), out.getData<float>() + elem_off,
                   active_rows, width, epsilon);
#ifdef ENABLE_FP16
  } else if (dt == DT::FP16 && gt == DT::FP16) {
    layernorm_rows(in.getData<_FP16>() + elem_off, gamma.getData<_FP16>(),
                   beta.getData<_FP16>(), out.getData<_FP16>() + elem_off,
                   active_rows, width, epsilon);
  } else if (dt == DT::FP16 && gt == DT::FP32) {
    layernorm_rows(in.getData<_FP16>() + elem_off, gamma.getData<float>(),
                   beta.getData<float>(), out.getData<_FP16>() + elem_off,
                   active_rows, width, epsilon);
  } else if (dt == DT::FP32 && gt == DT::FP16) {
    layernorm_rows(in.getData<float>() + elem_off, gamma.getData<_FP16>(),
                   beta.getData<_FP16>(), out.getData<float>() + elem_off,
                   active_rows, width, epsilon);
#endif
  } else {
    throw std::invalid_argument(
      "CpuComputeOps::layer_norm: unsupported data type");
  }
}

/**
 * GELU and tanh-GELU get an explicit dtype switch instead of going through
 * ActiFunc, because ActiFunc::gelu/tanhGelu call nntrainer::gelu_v2/tanh_gelu
 * with t_in.getData<float>() unconditionally: on an FP16 activation tensor
 * that reinterprets half data as float and reads size() * 4 bytes out of a
 * size() * 2 byte buffer. The FP32 path below calls the very same backend
 * functions ActiFunc does, so FP32 output is unchanged.
 *
 * Every other mode is delegated to ActiFunc so there is ONE host
 * implementation of the activation family and every existing mode stays
 * value-identical. A partial row window is expressed as a shared-data view,
 * which is safe here because this is by definition the host path.
 */
void CpuComputeOps::activation(const Tensor &in, Tensor &out, int act_type,
                               unsigned int active_rows,
                               unsigned int row_offset) {
  using DT = ml::train::TensorDim::DataType;
  const auto at = static_cast<ActivationType>(act_type);
  const unsigned int width = in.width();
  const size_t elem_off = (size_t)row_offset * width;
  const size_t n = (size_t)active_rows * width;
  const DT dt = in.getDataType();

  if (n == 0)
    return;

  if (at == ActivationType::ACT_GELU || at == ActivationType::ACT_TANH_GELU) {
    const bool tanh_mode = (at == ActivationType::ACT_TANH_GELU);
    if (dt == DT::FP32) {
      const float *x = in.getData<float>() + elem_off;
      float *y = out.getData<float>() + elem_off;
      if (tanh_mode)
        nntrainer::tanh_gelu((unsigned int)n, x, y);
      else
        nntrainer::gelu_v2((unsigned int)n, x, y);
      return;
    }
#ifdef ENABLE_FP16
    if (dt == DT::FP16) {
      gelu_elems(in.getData<_FP16>() + elem_off,
                 out.getData<_FP16>() + elem_off, n, tanh_mode);
      return;
    }
#endif
    throw std::invalid_argument(
      "CpuComputeOps::activation: unsupported data type for gelu");
  }

  if (at == ActivationType::ACT_NONE) {
    if (in.getData<uint8_t>() == out.getData<uint8_t>())
      return;
    size_t esz;
    if (dt == DT::FP32)
      esz = sizeof(float);
    else if (dt == DT::FP16)
      esz = 2u;
    else
      throw std::invalid_argument(
        "CpuComputeOps::activation: unsupported data type");
    std::memcpy(out.getData<uint8_t>() + elem_off * esz,
                in.getData<uint8_t>() + elem_off * esz, n * esz);
    return;
  }

  ActiFunc f;
  if (dt == DT::FP16) {
#ifdef ENABLE_FP16
    f.setActiFunc<_FP16>(at);
#else
    throw std::invalid_argument(
      "CpuComputeOps::activation: fp16 needs enable-fp16");
#endif
  } else {
    f.setActiFunc<float>(at);
  }

  const TensorDim d = in.getDim();
  const unsigned int total_rows = d.batch() * d.channel() * d.height();
  const bool aliased = (in.getData<uint8_t>() == out.getData<uint8_t>());

  if (row_offset == 0 && active_rows == total_rows) {
    if (aliased && !f.supportInPlace()) {
      Tensor in_copy = out.clone();
      f.run_fn(in_copy, out);
    } else {
      f.run_fn(in, out);
    }
    return;
  }

  TensorDim wd = d;
  wd.batch(1);
  wd.channel(1);
  wd.height(active_rows);
  Tensor in_view = in.getSharedDataTensor(wd, elem_off, true);
  Tensor out_view = out.getSharedDataTensor(wd, elem_off, true);
  if (aliased && !f.supportInPlace()) {
    Tensor in_copy = in_view.clone();
    f.run_fn(in_copy, out_view);
  } else {
    f.run_fn(in_view, out_view);
  }
}

void CpuComputeOps::residual_op(Tensor &hidden, const Tensor &input,
                                bool accumulate) {
  if (accumulate)
    hidden.add_i(input);
  else
    hidden.copy(input);
}

void CpuComputeOps::fc(Tensor &input, Tensor &weight, Tensor &output) {
  input.dot(weight, output, false, false);
}

void CpuComputeOps::apply_activation(Tensor &out, int act_type) {
  if (static_cast<ActivationType>(act_type) == ActivationType::ACT_NONE)
    return;
  const TensorDim d = out.getDim();
  activation(out, out, act_type, d.batch() * d.channel() * d.height(),
             /*row_offset=*/0);
}

void CpuComputeOps::mean_rows(const Tensor &in, Tensor &out,
                              unsigned int active_rows,
                              unsigned int row_offset) {
  // Reduce through a [1, 1, active_rows, W] window and average over axis 2 --
  // exactly the shape and call a mean-pooling caller built by hand before this
  // op existed, so the reduction order (and therefore the result) is unchanged.
  const unsigned int W = in.width();
  Tensor rows =
    in.getSharedDataTensor({1, 1, active_rows, W}, (size_t)row_offset * W);
  out.copyData(rows.average(2));
}

void CpuComputeOps::l2_normalize_rows(const Tensor &in, Tensor &out,
                                      float epsilon) {
  // Same rationale as mean_rows: this is the copyData + normalization_i(3)
  // pair the callers ran before, so host numerics are unchanged.
  if (out.getData<uint8_t>() != in.getData<uint8_t>())
    out.copyData(in);
  out.normalization_i(3, 2.0f, epsilon);
}

} // namespace nntrainer
