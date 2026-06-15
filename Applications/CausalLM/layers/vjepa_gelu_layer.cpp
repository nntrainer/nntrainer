// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   vjepa_gelu_layer.cpp
 * @date   22 May 2026
 * @brief  Token-parallel GELU activation (NEON gelu_v2 split over the pool).
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "vjepa_gelu_layer.h"

#include <cmath>
#include <cpu_backend.h>
#include <nntrainer_error.h>
#include <stdexcept>
#include <thread_manager.h>

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void VjepaGeluLayer::finalize(nntrainer::InitLayerContext &context) {
  context.setOutputDimensions(context.getInputDimensions());
}

// gelu over [X, X+n) split across the ThreadManager pool (elementwise, so any
// split is valid). Matches the core activation's gelu_v2 exactly.
static void gelu_parallel(const float *X, float *Y, size_t n) {
  if (n == 0)
    return;
  auto &tm = nntrainer::ThreadManager::Global();
  const unsigned int nt = tm.getComputeThreadCount();
  if (nt <= 1 || n < 4096) {
    nntrainer::gelu_v2(static_cast<unsigned int>(n), X, Y);
    return;
  }
  tm.parallel_for(0, static_cast<size_t>(nt), [=](size_t t) {
    size_t s = (n * t) / nt;
    size_t e = (n * (t + 1)) / nt;
    if (e > s)
      nntrainer::gelu_v2(static_cast<unsigned int>(e - s), X + s, Y + s);
  });
}

#ifdef ENABLE_FP16
// Exact erf GELU for FP16 activation: reads/writes FP16 but evaluates each
// element in FP32 (no FP32 buffer conversion). gelu_v2 is FP32-only, so this is
// a self-contained scalar erf that numerically matches it (exact erf GELU),
// split across the pool the same way as the FP32 path.
static void gelu_parallel(const _FP16 *X, _FP16 *Y, size_t n) {
  if (n == 0)
    return;
  constexpr float kInvSqrt2 = 0.70710678118654752440f;
  auto chunk = [=](size_t s, size_t e) {
    for (size_t i = s; i < e; ++i) {
      const float x = static_cast<float>(X[i]);
      Y[i] = static_cast<_FP16>(0.5f * x * (1.0f + std::erf(x * kInvSqrt2)));
    }
  };
  auto &tm = nntrainer::ThreadManager::Global();
  const unsigned int nt = tm.getComputeThreadCount();
  if (nt <= 1 || n < 4096) {
    chunk(0, n);
    return;
  }
  tm.parallel_for(0, static_cast<size_t>(nt), [=](size_t t) {
    chunk((n * t) / nt, (n * (t + 1)) / nt);
  });
}
#endif

void VjepaGeluLayer::forwarding(nntrainer::RunLayerContext &context,
                                bool training) {
  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  if (in.getDataType() == ml::train::TensorDim::DataType::FP32) {
    gelu_parallel(in.getData<float>(), out.getData<float>(), in.size());
  } else if (in.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    gelu_parallel(in.getData<_FP16>(), out.getData<_FP16>(), in.size());
#else
    throw std::invalid_argument("[vjepa_gelu] enable-fp16 is not set!");
#endif
  } else {
    throw std::invalid_argument("[vjepa_gelu] unsupported data type");
  }
}

void VjepaGeluLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                            unsigned int from, unsigned int to,
                                            bool training) {
  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);

  const nntrainer::TensorDim dim = in.getDim();
  const size_t width = dim.width();
  const size_t feature_len = dim.getFeatureLen();
  const size_t off = static_cast<size_t>(from) * width;
  const size_t n = static_cast<size_t>(to - from) * width;
  const bool is_fp16 = in.getDataType() == ml::train::TensorDim::DataType::FP16;
  if (!is_fp16 && in.getDataType() != ml::train::TensorDim::DataType::FP32)
    throw std::invalid_argument("[vjepa_gelu] unsupported data type");
  for (unsigned int b = 0; b < dim.batch(); ++b) {
    if (!is_fp16) {
      const float *x = in.getData<float>() + b * feature_len + off;
      float *y = out.getData<float>() + b * feature_len + off;
      gelu_parallel(x, y, n);
    } else {
#ifdef ENABLE_FP16
      const _FP16 *x = in.getData<_FP16>() + b * feature_len + off;
      _FP16 *y = out.getData<_FP16>() + b * feature_len + off;
      gelu_parallel(x, y, n);
#else
      throw std::invalid_argument("[vjepa_gelu] enable-fp16 is not set!");
#endif
    }
  }
}

void VjepaGeluLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  throw std::runtime_error("[vjepa_gelu] Training is not supported yet.");
}

void VjepaGeluLayer::setProperty(const std::vector<std::string> &values) {
  NNTR_THROW_IF(!values.empty(), std::invalid_argument)
    << "[vjepa_gelu] does not take properties";
}

void VjepaGeluLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  context.updateInput(SINGLE_INOUT_IDX, input_dimensions[0]);
  context.updateOutput(SINGLE_INOUT_IDX, input_dimensions[0]);
}

#ifdef PLUGGABLE
nntrainer::Layer *create_vjepa_gelu_layer() { return new VjepaGeluLayer(); }
void destroy_vjepa_gelu_layer(nntrainer::Layer *layer) { delete layer; }
extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_vjepa_gelu_layer,
                                                   destroy_vjepa_gelu_layer};
}
#endif

} // namespace causallm
