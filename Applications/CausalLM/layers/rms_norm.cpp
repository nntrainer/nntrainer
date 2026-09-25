// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   rms_norm.cpp
 * @date   19 July 2023
 * @brief  Implementation of custom RMS normalization function
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @author Niket Agarwal <niket.a@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <cmath>
#include <vector>
#include <cpu_backend.h>
#include <iostream>

#include "rms_norm.h"

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void RMSNormLayer::finalize(nntrainer::InitLayerContext &context) {
  std::vector<nntrainer::TensorDim> dim = context.getInputDimensions();
  context.setOutputDimensions(dim);

  if (!std::get<nntrainer::props::SkipPrefill>(rms_props).empty())
    skip_prefill = std::get<nntrainer::props::SkipPrefill>(rms_props).get();

  // gamma is unquantized and stored as FP32 in the bin. Request it as FP32
  // regardless of the activation dtype; declaring it FP16 reinterprets the
  // on-disk FP32 bytes as FP16 and corrupts gamma (≈FP16-max garbage). The
  // FP16 forward path casts gamma down to FP16 at the multiply site.
  nntrainer::TensorDim gamma_dim(
    1, 1, 1, dim[0].width(),
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     nntrainer::TensorDim::DataType::FP32));
  wt_idx[RMSParams::gamma] = context.requestWeight(
    gamma_dim, nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "gamma", true);
}

void RMSNormLayer::forwarding(nntrainer::RunLayerContext &context,
                              bool training) {
  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  computeRMSNorm(context, 0, in.getDim().height());
}

void RMSNormLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                          unsigned int from, unsigned int to,
                                          bool training) {
  bool is_prefill = !from || (to - from) > 1;
  if (skip_prefill && is_prefill)
    return;

  computeRMSNorm(context, from, to);
}

void RMSNormLayer::computeRMSNorm(nntrainer::RunLayerContext &context,
                                  unsigned int from, unsigned int to) {
  auto &epsilon = std::get<nntrainer::props::Epsilon>(rms_props).get();

  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &gamma = context.getWeight(wt_idx[RMSParams::gamma]);

  ml::train::TensorDim in_dim = in.getDim();
  ml::train::TensorDim out_dim = out.getDim();

  ml::train::TensorDim in_step_dim = in_dim;
  ml::train::TensorDim out_step_dim = out_dim;

  in_step_dim.batch(1);
  in_step_dim.height(to - from);
  out_step_dim.batch(1);
  out_step_dim.height(to - from);

  unsigned int b_size = in_dim.batch();

  for (unsigned int b = 0; b < b_size; ++b) {
    nntrainer::Tensor in_step =
      in.getSharedDataTensor(in_step_dim, b * in_dim.getFeatureLen(), true);
    nntrainer::Tensor out_step =
      out.getSharedDataTensor(out_step_dim, b * out_dim.getFeatureLen(), true);

    if (in_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
      const auto &dim = in_step.getDim();
#ifdef ENABLE_FP16
      nntrainer::rms_norm_wrt_width_fp32_intrinsic(
        in_step.getData<float>(), out_step.getData<float>(), dim.height(),
        dim.width(), epsilon);

      // DO NOT USE rms_norm_wrt_width_fp16_intrinsic. It causes overflow!

      // nntrainer::rms_norm_wrt_width_fp16_intrinsic(
      //   in_step.getData<float>(), out_step.getData<float>(), dim.height(),
      //   dim.width(), epsilon);
#else

      nntrainer::rms_norm_wrt_width_fp32_intrinsic(
        in_step.getData<float>(), out_step.getData<float>(), dim.height(),
        dim.width(), epsilon);
#endif
#ifdef ENABLE_FP16
    } else if (in_step.getDataType() == ml::train::TensorDim::DataType::FP16) {
      const auto &dim = in_step.getDim();
      // FP16 activation: this kernel accumulates the sum-of-squares in FP32
      // (so a wide residual row cannot overflow FP16) and reads/writes FP16.
      nntrainer::rms_norm_wrt_width_fp16_intrinsic(
        in_step.getData<_FP16>(), out_step.getData<_FP16>(), dim.height(),
        dim.width(), epsilon);
#endif
    } else {
      throw std::invalid_argument(
        "Error: not yet implemented for this data type");
    }
    // gamma (unquantized) may be stored at a different dtype than the FP16
    // activation; cast it to match before the elementwise multiply.
    if (gamma.getDataType() != out_step.getDataType()) {
      nntrainer::Tensor gamma_cast = gamma.clone(out_step.getDataType());
      out_step.multiply_i(gamma_cast);
    } else {
      out_step.multiply_i(gamma);
    }
#ifdef DEBUG
    std::cout << context.getName() << " \n input:" << in_step
              << "output:" << out_step << "gamma:" << gamma << std::endl;
#endif
  }
}

void RMSNormLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  context.updateInput(SINGLE_INOUT_IDX, input_dimensions[0]);
  context.updateOutput(SINGLE_INOUT_IDX, input_dimensions[0]);
}

/**
 * @brief calcDerivative for RMSNorm.
 * @details y_i = x_i * inv_rms * gamma_i, with
 *          inv_rms = 1 / sqrt(mean(x^2) + eps).
 *          dL/dx_j = inv_rms * (gamma_j*dy_j)
 *                    - inv_rms^3 * x_j * mean_i(gamma_i*dy_i*x_i)
 *          dgamma is computed separately in calcGradient(), which the
 *          framework only calls when the layer is trainable (gamma stays
 *          frozen under LoRA-only training).
 */
void RMSNormLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  auto &epsilon = std::get<nntrainer::props::Epsilon>(rms_props).get();

  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &gamma = context.getWeight(wt_idx[RMSParams::gamma]);
  const nntrainer::Tensor &dy =
    context.getIncomingDerivative(SINGLE_INOUT_IDX);
  nntrainer::Tensor &dx = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  NNTR_THROW_IF(in.getDataType() != ml::train::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "[rms_norm] calcDerivative only supports FP32 for now";

  nntrainer::Tensor gamma_fp32 =
    (gamma.getDataType() == ml::train::TensorDim::DataType::FP32)
      ? gamma
      : gamma.clone(ml::train::TensorDim::DataType::FP32);

  const ml::train::TensorDim &in_dim = in.getDim();
  const unsigned int width = in_dim.width();
  const unsigned int rows_per_batch = in_dim.getFeatureLen() / width;
  const unsigned int batch = in_dim.batch();

  const float *x = in.getData<float>();
  const float *dy_ = dy.getData<float>();
  const float *g = gamma_fp32.getData<float>();
  float *dx_ = dx.getData<float>();

  for (unsigned int b = 0; b < batch; ++b) {
    const float *x_b = x + b * in_dim.getFeatureLen();
    const float *dy_b = dy_ + b * in_dim.getFeatureLen();
    float *dx_b = dx_ + b * in_dim.getFeatureLen();

    for (unsigned int r = 0; r < rows_per_batch; ++r) {
      const float *x_row = x_b + r * width;
      const float *dy_row = dy_b + r * width;
      float *dx_row = dx_b + r * width;

      float ms = 0.0f;
      for (unsigned int w = 0; w < width; ++w)
        ms += x_row[w] * x_row[w];
      ms /= width;
      float inv_rms = 1.0f / std::sqrt(ms + epsilon);
      float inv_rms3 = inv_rms * inv_rms * inv_rms;

      float sum_gdyx = 0.0f;
      for (unsigned int w = 0; w < width; ++w)
        sum_gdyx += g[w] * dy_row[w] * x_row[w];
      float mean_gdyx = sum_gdyx / width;

      for (unsigned int w = 0; w < width; ++w)
        dx_row[w] =
          inv_rms * (g[w] * dy_row[w]) - inv_rms3 * x_row[w] * mean_gdyx;
    }
  }
}

/**
 * @brief calcGradient for RMSNorm.
 * @details y[r][w] = x[r][w] * inv_rms[r] * gamma[w], so
 *          dL/dgamma[w] = sum over all rows of dy[r][w] * x[r][w] *
 *          inv_rms[r].
 *
 * @note Accumulated in double and written once per element, so a long
 *       sequence cannot lose precision to repeated float rounding.
 * @note Honours isGradientFirstAccess(): the gradient is overwritten on the
 *       first visit and accumulated afterwards, so a gamma shared between
 *       several layers sums their contributions instead of discarding all
 *       but the last.
 * @note Safe to read the input here: the framework runs calcGradient before
 *       calcDerivative, and it is calcDerivative that overwrites the input
 *       buffer with dx (see the aliasing note there).
 */
void RMSNormLayer::calcGradient(nntrainer::RunLayerContext &context) {
  auto &epsilon = std::get<nntrainer::props::Epsilon>(rms_props).get();

  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  const nntrainer::Tensor &dy =
    context.getIncomingDerivative(SINGLE_INOUT_IDX);
  nntrainer::Tensor &dgamma = context.getWeightGrad(wt_idx[RMSParams::gamma]);

  NNTR_THROW_IF(in.getDataType() != ml::train::TensorDim::DataType::FP32 ||
                  dgamma.getDataType() !=
                    ml::train::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "[rms_norm] calcGradient only supports FP32 for now";

  const ml::train::TensorDim &in_dim = in.getDim();
  const unsigned int width = in_dim.width();
  const unsigned int rows_per_batch = in_dim.getFeatureLen() / width;
  const unsigned int batch = in_dim.batch();

  const float *x = in.getData<float>();
  const float *dy_ = dy.getData<float>();

  std::vector<double> acc(width, 0.0);

  for (unsigned int b = 0; b < batch; ++b) {
    const float *x_b = x + b * in_dim.getFeatureLen();
    const float *dy_b = dy_ + b * in_dim.getFeatureLen();

    for (unsigned int r = 0; r < rows_per_batch; ++r) {
      const float *x_row = x_b + r * width;
      const float *dy_row = dy_b + r * width;

      float ms = 0.0f;
      for (unsigned int w = 0; w < width; ++w)
        ms += x_row[w] * x_row[w];
      ms /= width;
      const float inv_rms = 1.0f / std::sqrt(ms + epsilon);

      for (unsigned int w = 0; w < width; ++w)
        acc[w] += static_cast<double>(dy_row[w]) * x_row[w] * inv_rms;
    }
  }

  float *dg = dgamma.getData<float>();
  if (context.isGradientFirstAccess(wt_idx[RMSParams::gamma])) {
    for (unsigned int w = 0; w < width; ++w)
      dg[w] = static_cast<float>(acc[w]);
  } else {
    for (unsigned int w = 0; w < width; ++w)
      dg[w] += static_cast<float>(acc[w]);
  }
}

#ifdef PLUGGABLE

nntrainer::Layer *create_rms_norm_layer() {
  auto layer = new RMSNormLayer();
  return layer;
}

void destroy_rms_norm_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_rms_norm_layer,
                                                   destroy_rms_norm_layer};
}

#endif

} // namespace causallm
