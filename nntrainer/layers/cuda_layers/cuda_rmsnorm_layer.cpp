// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   cuda_rmsnorm_layer.cpp
 * @date   22 Jun 2026
 * @brief  RMS normalization for the CUDA backend (FP32-safe sum-of-squares).
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <cuda_rmsnorm_layer.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <cuda_context_manager.h>
#include <cuda_fc_qs4cx.h> // cuda_fc_qs4cx_rmsnorm_prequant_fp16 (fused norm+quant)
#include <cuda_rmsnorm.h>
#include <cuda_runtime.h>
#include <cuda_stream_manager.h>

#include <layer_context.h>
#include <nntrainer_error.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;
enum RMSParams { gamma };

void CudaRMSNormLayer::finalize(InitLayerContext &context) {
  std::vector<TensorDim> dim = context.getInputDimensions();
  context.setOutputDimensions(dim);

  if (!std::get<props::SkipPrefill>(rms_props).empty())
    skip_prefill = std::get<props::SkipPrefill>(rms_props).get();

  TensorDim gamma_dim(
    1, 1, 1, dim[0].width(),
    TensorDim::TensorType(context.getFormat(), context.getWeightDataType()));
  wt_idx[RMSParams::gamma] =
    context.requestWeight(gamma_dim, props::InitializerInfo::Enum::NONE,
                          WeightRegularizer::NONE, 1.0f, 0.0f, "gamma", false);
}

namespace {
// x * rsqrt(mean(x^2)+eps) * gamma, sum-of-squares accumulated in FP32 (no
// FP16 overflow). rows = leading dims folded, width = feature size.
template <typename T, typename G>
void rmsnorm_rows(const T *x, const G *g, T *y, unsigned int rows,
                  unsigned int width, float eps) {
  for (unsigned int r = 0; r < rows; ++r) {
    const T *xr = x + (size_t)r * width;
    T *yr = y + (size_t)r * width;
    float ss = 0.f;
    for (unsigned int k = 0; k < width; ++k) {
      float v = (float)xr[k];
      ss += v * v;
    }
    float inv = 1.0f / std::sqrt(ss / (float)width + eps);
    for (unsigned int k = 0; k < width; ++k)
      yr[k] = (T)(((float)xr[k] * inv) * (float)g[k]);
  }
}

#ifdef ENABLE_FP16
bool dev_ok(const void *p) { return nntrainer::cuda::dev_accessible(p); }
#endif

void rmsnorm_dispatch(const Tensor &in, const Tensor &gamma, Tensor &out,
                      unsigned int rows, unsigned int width, float eps) {
  if (std::getenv("NNTR_CUDA_DBG")) {
    static int _n = 0;
    if (_n++ < 3)
      std::fprintf(stderr,
                   "[CUDA-DBG] CudaRMSNormLayer USED rows=%u width=%u\n", rows,
                   width);
  }
  using DT = ml::train::TensorDim::DataType;
  const DT dt = in.getDataType();
  const DT gt = gamma.getDataType();
#ifdef ENABLE_FP16
  // GPU path: fp16 in/out/gamma all device-resident (UVM). Block-per-row, FP32
  // sum-of-squares. Used only for small row counts (decode, rows~1): the kernel
  // syncs per call, so for the wide prefill norm (rows=seq_len) the
  // multi-thread host norm wins -- gating by rows gives the decode speedup
  // (+~13%) without a prefill regression. NNTR_RMSNORM_CUDA_OFF disables; =all
  // forces all rows.
  static const int gpu_max_rows = []() {
    const char *e = std::getenv("NNTR_RMSNORM_CUDA_OFF");
    if (e && e[0] == 'a')
      return 1 << 30; // "all"
    if (e)
      return 0; // off
    return 32;  // decode-only default
  }();
  if (dt == DT::FP16 && gt == DT::FP16 && out.getDataType() == DT::FP16 &&
      (int)rows <= gpu_max_rows) {
    const unsigned short *xi =
      reinterpret_cast<const unsigned short *>(in.getData<_FP16>());
    const unsigned short *gi =
      reinterpret_cast<const unsigned short *>(gamma.getData<_FP16>());
    unsigned short *yi =
      reinterpret_cast<unsigned short *>(out.getData<_FP16>());
    if (dev_ok(xi) && dev_ok(gi) && dev_ok(yi)) {
      // Decode fast path: the norm output feeds an int4 FC group whose first
      // act is to quantize it to int8. Folding that quant into this launch
      // (and letting the group's siblings consume the staging) removes two
      // single-block kernels per norm from the decode graph. Falls through to
      // the plain norm when the lever is off or the staging cannot be
      // prepared.
      if (cuda::cuda_fc_qs4cx_rmsnorm_prequant_fp16(xi, gi, yi, eps, rows,
                                                    width))
        return;
      if (cuda::cuda_rmsnorm_fp16(xi, gi, yi, eps, rows, width))
        return;
    }
  }
#endif
  // Host rmsnorm fallback: sync first so the host read of GPU-produced input is
  // coherent under NNTR_CUDA_ASYNC (no-op in sync mode).
  cuda::StreamManager::Global().finishIfAsync();
  if (dt == DT::FP32 && gt == DT::FP32) {
    rmsnorm_rows(in.getData<float>(), gamma.getData<float>(),
                 out.getData<float>(), rows, width, eps);
#ifdef ENABLE_FP16
  } else if (dt == DT::FP16 && gt == DT::FP16) {
    rmsnorm_rows(in.getData<_FP16>(), gamma.getData<_FP16>(),
                 out.getData<_FP16>(), rows, width, eps);
  } else if (dt == DT::FP16 && gt == DT::FP32) {
    rmsnorm_rows(in.getData<_FP16>(), gamma.getData<float>(),
                 out.getData<_FP16>(), rows, width, eps);
  } else if (dt == DT::FP32 && gt == DT::FP16) {
    rmsnorm_rows(in.getData<float>(), gamma.getData<_FP16>(),
                 out.getData<float>(), rows, width, eps);
#endif
  } else {
    throw std::invalid_argument("CudaRMSNormLayer: unsupported data type");
  }
}
} // namespace

void CudaRMSNormLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  incremental_forwarding(context, 0, in.getDim().height(), training);
}

void CudaRMSNormLayer::incremental_forwarding(RunLayerContext &context,
                                              unsigned int from,
                                              unsigned int to, bool training) {
  if (skip_prefill && from == 0)
    return;

  auto &epsilon = std::get<props::Epsilon>(rms_props).get();
  Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &gamma = context.getWeight(wt_idx[RMSParams::gamma]);

  TensorDim in_dim = in.getDim();
  TensorDim out_dim = out.getDim();
  TensorDim in_step_dim = in_dim;
  TensorDim out_step_dim = out_dim;
  in_step_dim.batch(1);
  in_step_dim.height(to - from);
  out_step_dim.batch(1);
  out_step_dim.height(to - from);

  const unsigned int width = in_dim.width();
  const unsigned int rows_per_b = in_step_dim.channel() * (to - from);

  for (unsigned int b = 0; b < in_dim.batch(); ++b) {
    Tensor in_step =
      in.getSharedDataTensor(in_step_dim, b * in_dim.getFeatureLen(), true);
    Tensor out_step =
      out.getSharedDataTensor(out_step_dim, b * out_dim.getFeatureLen(), true);
    rmsnorm_dispatch(in_step, gamma, out_step, rows_per_b, width, epsilon);
  }
}

} // namespace nntrainer
