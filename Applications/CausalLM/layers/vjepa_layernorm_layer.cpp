// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   vjepa_layernorm_layer.cpp
 * @date   22 May 2026
 * @brief  Per-token-parallel LayerNorm over the width axis.
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "vjepa_layernorm_layer.h"

#include <cmath>
#include <nntrainer_error.h>
#include <stdexcept>
#include <thread_manager.h>

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;
enum LNParams { GAMMA = 0, BETA = 1 };

void VjepaLayerNormLayer::finalize(nntrainer::InitLayerContext &context) {
  const auto &in_dim = context.getInputDimensions()[0];
  context.setOutputDimensions({in_dim});

  // gamma/beta over the width axis only: [1, 1, 1, W] (matches the core
  // layer_normalization axis=3 weight layout, so weight bins are compatible).
  // LayerNorm weights stay FP32 even in the Q4_0/FP16-activation model (norms
  // are unquantized and stored FP32 on disk), so request FP32 explicitly. Using
  // the activation dtype here would, under FP16 activation, bind FP32 weight
  // bytes into an FP16 tensor and corrupt gamma/beta.
  nntrainer::TensorDim norm_dim(context.getFormat(),
                                ml::train::TensorDim::DataType::FP32);
  norm_dim.width(in_dim.width());

  wt_idx[GAMMA] = context.requestWeight(norm_dim, nntrainer::Initializer::ONES,
                                        nntrainer::WeightRegularizer::NONE,
                                        1.0f, 0.0f, "gamma", true);
  wt_idx[BETA] = context.requestWeight(norm_dim, nntrainer::Initializer::ZEROS,
                                       nntrainer::WeightRegularizer::NONE, 1.0f,
                                       0.0f, "beta", true);
}

// LayerNorm of `num_rows` contiguous rows of width W, parallelized over the
// pool. y = (x-mean)/sqrt(var+eps)*gamma + beta, var biased (matches core).
// Reads/writes the activation in its native dtype T (FP32 or FP16) but does the
// reduction with FP32 partial accumulation: under FP16 activation the row is
// never converted into an FP32 buffer — each element is widened to float inline
// for the mean/variance sums and the normalized value is cast back to T on
// store. gamma/beta are always FP32 (norms are unquantized).
template <typename T>
static void layernorm_parallel(const T *X, T *Y, const float *gamma,
                               const float *beta, size_t num_rows,
                               unsigned int W, float eps) {
  if (num_rows == 0)
    return;
  auto &tm = nntrainer::ThreadManager::Global();
  const unsigned int nt = tm.getComputeThreadCount();
  const float invW = 1.0f / static_cast<float>(W);
  auto do_rows = [=](size_t rs, size_t re) {
    for (size_t r = rs; r < re; ++r) {
      const T *x = X + r * W;
      T *y = Y + r * W;
      float mean = 0.0f;
      for (unsigned int j = 0; j < W; ++j)
        mean += static_cast<float>(x[j]);
      mean *= invW;
      float var = 0.0f;
      for (unsigned int j = 0; j < W; ++j) {
        float dv = static_cast<float>(x[j]) - mean;
        var += dv * dv;
      }
      var *= invW;
      const float inv = 1.0f / std::sqrt(var + eps);
      for (unsigned int j = 0; j < W; ++j)
        y[j] = static_cast<T>((static_cast<float>(x[j]) - mean) * inv *
                                gamma[j] +
                              beta[j]);
    }
  };
  if (nt <= 1 || num_rows < 4) {
    do_rows(0, num_rows);
    return;
  }
  tm.parallel_for(0, static_cast<size_t>(nt), [=](size_t t) {
    do_rows((num_rows * t) / nt, (num_rows * (t + 1)) / nt);
  });
}

void VjepaLayerNormLayer::forwarding(nntrainer::RunLayerContext &context,
                                     bool training) {
  const float eps = std::get<props::VjepaLnEpsilon>(layernorm_props).get();
  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &gamma = context.getWeight(wt_idx[GAMMA]);
  nntrainer::Tensor &beta = context.getWeight(wt_idx[BETA]);

  const nntrainer::TensorDim dim = in.getDim();
  const unsigned int W = dim.width();
  const size_t rows = (size_t)dim.batch() * dim.channel() * dim.height();
  const float *g = gamma.getData<float>();
  const float *bt = beta.getData<float>();
  if (in.getDataType() == ml::train::TensorDim::DataType::FP32) {
    layernorm_parallel<float>(in.getData<float>(), out.getData<float>(), g, bt,
                              rows, W, eps);
  } else if (in.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    layernorm_parallel<_FP16>(in.getData<_FP16>(), out.getData<_FP16>(), g, bt,
                              rows, W, eps);
#else
    throw std::invalid_argument("[vjepa_layernorm] enable-fp16 is not set!");
#endif
  } else {
    throw std::invalid_argument("[vjepa_layernorm] unsupported data type");
  }
}

void VjepaLayerNormLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  const float eps = std::get<props::VjepaLnEpsilon>(layernorm_props).get();
  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &gamma = context.getWeight(wt_idx[GAMMA]);
  nntrainer::Tensor &beta = context.getWeight(wt_idx[BETA]);

  const nntrainer::TensorDim dim = in.getDim();
  const unsigned int W = dim.width();
  const size_t feature_len = dim.getFeatureLen();
  const size_t rows_per_bc = to - from;
  const float *g = gamma.getData<float>();
  const float *bt = beta.getData<float>();
  const bool is_fp16 = in.getDataType() == ml::train::TensorDim::DataType::FP16;
  if (!is_fp16 && in.getDataType() != ml::train::TensorDim::DataType::FP32)
    throw std::invalid_argument("[vjepa_layernorm] unsupported data type");
  for (unsigned int b = 0; b < dim.batch(); ++b) {
    for (unsigned int c = 0; c < dim.channel(); ++c) {
      const size_t off = (size_t)b * feature_len +
                         (size_t)c * dim.height() * W + (size_t)from * W;
      if (!is_fp16) {
        layernorm_parallel<float>(in.getData<float>() + off,
                                  out.getData<float>() + off, g, bt,
                                  rows_per_bc, W, eps);
      } else {
#ifdef ENABLE_FP16
        layernorm_parallel<_FP16>(in.getData<_FP16>() + off,
                                  out.getData<_FP16>() + off, g, bt,
                                  rows_per_bc, W, eps);
#else
        throw std::invalid_argument("[vjepa_layernorm] enable-fp16 is not set!");
#endif
      }
    }
  }
}

void VjepaLayerNormLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  throw std::runtime_error("[vjepa_layernorm] Training is not supported yet.");
}

void VjepaLayerNormLayer::setProperty(const std::vector<std::string> &values) {
  auto remain = loadProperties(values, layernorm_props);
  NNTR_THROW_IF(!remain.empty(), std::invalid_argument)
    << "[vjepa_layernorm] unknown properties";
}

void VjepaLayerNormLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  context.updateInput(SINGLE_INOUT_IDX, input_dimensions[0]);
  context.updateOutput(SINGLE_INOUT_IDX, input_dimensions[0]);
}

#ifdef PLUGGABLE
nntrainer::Layer *create_vjepa_layernorm_layer() {
  return new VjepaLayerNormLayer();
}
void destroy_vjepa_layernorm_layer(nntrainer::Layer *layer) { delete layer; }
extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{
  create_vjepa_layernorm_layer, destroy_vjepa_layernorm_layer};
}
#endif

} // namespace causallm
