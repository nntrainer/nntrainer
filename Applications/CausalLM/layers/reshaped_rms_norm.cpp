// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   reshaped_rms_norm.cpp
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
#include <reshaped_rms_norm.h>

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void ReshapedRMSNormLayer::finalize(nntrainer::InitLayerContext &context) {
  std::vector<nntrainer::TensorDim> dim = context.getInputDimensions();
  context.setOutputDimensions(dim);
  feature_size = std::get<props::FeatureSize>(rms_props);
  use_gamma = std::get<props::UseGamma>(rms_props).get();

  if (!std::get<nntrainer::props::SkipPrefill>(rms_props).empty())
    skip_prefill = std::get<nntrainer::props::SkipPrefill>(rms_props).get();

  if (!std::get<nntrainer::props::SkipPrefill>(rms_props).empty())
    skip_prefill = std::get<nntrainer::props::SkipPrefill>(rms_props).get();

  NNTR_THROW_IF(dim[0].width() % feature_size != 0, std::invalid_argument)
    << "feature size must be a divisor of width";

  if (use_gamma) {
    // gamma is unquantized FP32 on disk; request FP32 regardless of activation
    // dtype (FP16 would reinterpret the FP32 bytes and corrupt gamma). The FP16
    // path casts gamma down at the multiply site.
    nntrainer::TensorDim gamma_dim(
      1, 1, 1, feature_size,
      nntrainer::TensorDim::TensorType(context.getFormat(),
                                       nntrainer::TensorDim::DataType::FP32));
    wt_idx[RMSParams::gamma] = context.requestWeight(
      gamma_dim, nntrainer::props::InitializerInfo::Enum::NONE,
      nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "gamma", true);
  }
}

void ReshapedRMSNormLayer::forwarding(nntrainer::RunLayerContext &context,
                                      bool training) {
  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  computeRMSNorm(context, 0, in.getDim().height());
}

void ReshapedRMSNormLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  bool is_prefill = !from || (to - from) > 1;
  if (skip_prefill && is_prefill)
    return;

  computeRMSNorm(context, from, to);
}

void ReshapedRMSNormLayer::computeRMSNorm(nntrainer::RunLayerContext &context,
                                          unsigned int from,
                                          unsigned int to) {
  auto &epsilon = std::get<nntrainer::props::Epsilon>(rms_props).get();

  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);

  ml::train::TensorDim in_dim = in.getDim();
  ml::train::TensorDim out_dim = out.getDim();

  ml::train::TensorDim in_step_dim = in_dim;
  ml::train::TensorDim out_step_dim = out_dim;

  in_step_dim.batch(1);
  in_step_dim.height(to - from);
  out_step_dim.batch(1);
  out_step_dim.height(to - from);

  // set reshaped dim to (1, 1, -1, feature_size)
  ml::train::TensorDim step_reshaped_dim = in_step_dim;

  step_reshaped_dim.width(feature_size);
  step_reshaped_dim.height(in_step_dim.height() *
                           (in_dim.width() / feature_size));

  unsigned int b_size = in_dim.batch();

  for (unsigned int b = 0; b < b_size; ++b) {
    nntrainer::Tensor in_step =
      in.getSharedDataTensor(in_step_dim, b * in_dim.getFeatureLen(), true);
    nntrainer::Tensor out_step =
      out.getSharedDataTensor(out_step_dim, b * out_dim.getFeatureLen(), true);

    // reshape in_step
    // reshape out_step
    in_step.reshape(step_reshaped_dim);
    out_step.reshape(step_reshaped_dim);

    if (in_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
      ///@todo rms_norm_wrt_width_something() should be refactored to
      /// nntrainer::Tensor operation.
#ifdef ENABLE_FP16
      nntrainer::rms_norm_wrt_width_fp32_intrinsic(
        in_step.getData<float>(), out_step.getData<float>(),
        in_step.getDim().height(), in_step.getDim().width(), epsilon);

      // DO NOT USE rms_norm_wrt_width_fp16_intrinsic. It causes overflow!

      // nntrainer::rms_norm_wrt_width_fp16_intrinsic(
      //   in_step.getData<float>(), out_step.getData<float>(),
      //   in_step.getDim().height(), in_step.getDim().width(), epsilon);
#else
      nntrainer::rms_norm_wrt_width_fp32_intrinsic(
        in_step.getData<float>(), out_step.getData<float>(),
        in_step.getDim().height(), in_step.getDim().width(), epsilon);
#endif
#ifdef ENABLE_FP16
    } else if (in_step.getDataType() == ml::train::TensorDim::DataType::FP16) {
      // FP16 activation: kernel accumulates squares in FP32 (no overflow).
      nntrainer::rms_norm_wrt_width_fp16_intrinsic(
        in_step.getData<_FP16>(), out_step.getData<_FP16>(),
        in_step.getDim().height(), in_step.getDim().width(), epsilon);
#endif
    } else {
      throw std::invalid_argument(
        "Error: not yet implemented for this data type");
    }
    if (use_gamma) {
      nntrainer::Tensor &gamma = context.getWeight(wt_idx[RMSParams::gamma]);
      if (gamma.getDataType() != out_step.getDataType()) {
        nntrainer::Tensor gamma_cast = gamma.clone(out_step.getDataType());
        out_step.multiply_i(gamma_cast);
      } else {
        out_step.multiply_i(gamma);
      }
    }

    // reshape again out_step
    out_step.reshape(out_step_dim);

#ifdef DEBUG
    std::cout << context.getName() << " \n input:" << in_step
              << "output:" << out_step << std::endl;
#endif
  }
}

void ReshapedRMSNormLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  context.updateInput(SINGLE_INOUT_IDX, input_dimensions[0]);
  context.updateOutput(SINGLE_INOUT_IDX, input_dimensions[0]);
}

/**
 * @brief calcDerivative for ReshapedRMSNorm.
 * @details Identical math to RMSNormLayer::calcDerivative, but each row of
 *          width `feature_size` (rather than the full tensor width) is
 *          normalized independently, matching the reshape done in
 *          computeRMSNorm(). dgamma is computed in calcGradient(), which
 *          the framework only calls when the layer is trainable.
 */
void ReshapedRMSNormLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  auto &epsilon = std::get<nntrainer::props::Epsilon>(rms_props).get();

  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  const nntrainer::Tensor &dy =
    context.getIncomingDerivative(SINGLE_INOUT_IDX);
  nntrainer::Tensor &dx = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  NNTR_THROW_IF(in.getDataType() != ml::train::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "[reshaped_rms_norm] calcDerivative only supports FP32 for now";

  nntrainer::Tensor gamma_fp32;
  const float *g = nullptr;
  if (use_gamma) {
    nntrainer::Tensor &gamma = context.getWeight(wt_idx[RMSParams::gamma]);
    gamma_fp32 = (gamma.getDataType() == ml::train::TensorDim::DataType::FP32)
                   ? gamma
                   : gamma.clone(ml::train::TensorDim::DataType::FP32);
    g = gamma_fp32.getData<float>();
  }

  const ml::train::TensorDim &in_dim = in.getDim();
  const unsigned int chunks_per_batch = in_dim.getFeatureLen() / feature_size;
  const unsigned int batch = in_dim.batch();

  const float *x = in.getData<float>();
  const float *dy_ = dy.getData<float>();
  float *dx_ = dx.getData<float>();

  for (unsigned int b = 0; b < batch; ++b) {
    const float *x_b = x + b * in_dim.getFeatureLen();
    const float *dy_b = dy_ + b * in_dim.getFeatureLen();
    float *dx_b = dx_ + b * in_dim.getFeatureLen();

    for (unsigned int r = 0; r < chunks_per_batch; ++r) {
      const float *x_row = x_b + r * feature_size;
      const float *dy_row = dy_b + r * feature_size;
      float *dx_row = dx_b + r * feature_size;

      float ms = 0.0f;
      for (unsigned int w = 0; w < feature_size; ++w)
        ms += x_row[w] * x_row[w];
      ms /= feature_size;
      float inv_rms = 1.0f / std::sqrt(ms + epsilon);
      float inv_rms3 = inv_rms * inv_rms * inv_rms;

      float sum_gdyx = 0.0f;
      for (unsigned int w = 0; w < feature_size; ++w) {
        float gdy = use_gamma ? g[w] * dy_row[w] : dy_row[w];
        sum_gdyx += gdy * x_row[w];
      }
      float mean_gdyx = sum_gdyx / feature_size;

      for (unsigned int w = 0; w < feature_size; ++w) {
        float gdy = use_gamma ? g[w] * dy_row[w] : dy_row[w];
        dx_row[w] = inv_rms * gdy - inv_rms3 * x_row[w] * mean_gdyx;
      }
    }
  }
}

/**
 * @brief calcGradient for ReshapedRMSNorm.
 * @details Same accumulation as RMSNormLayer::calcGradient, but gamma spans
 *          one `feature_size`-wide chunk and every chunk in the tensor
 *          contributes to it:
 *          dL/dgamma[w] = sum over chunks of dy[c][w]*x[c][w]*inv_rms[c].
 *          No-op when use_gamma is false, since then there is no weight.
 */
void ReshapedRMSNormLayer::calcGradient(nntrainer::RunLayerContext &context) {
  if (!use_gamma)
    return;

  auto &epsilon = std::get<nntrainer::props::Epsilon>(rms_props).get();

  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  const nntrainer::Tensor &dy =
    context.getIncomingDerivative(SINGLE_INOUT_IDX);
  nntrainer::Tensor &dgamma = context.getWeightGrad(wt_idx[RMSParams::gamma]);

  NNTR_THROW_IF(in.getDataType() != ml::train::TensorDim::DataType::FP32 ||
                  dgamma.getDataType() !=
                    ml::train::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "[reshaped_rms_norm] calcGradient only supports FP32 for now";

  const ml::train::TensorDim &in_dim = in.getDim();
  const unsigned int chunks_per_batch = in_dim.getFeatureLen() / feature_size;
  const unsigned int batch = in_dim.batch();

  const float *x = in.getData<float>();
  const float *dy_ = dy.getData<float>();

  std::vector<double> acc(feature_size, 0.0);

  for (unsigned int b = 0; b < batch; ++b) {
    const float *x_b = x + b * in_dim.getFeatureLen();
    const float *dy_b = dy_ + b * in_dim.getFeatureLen();

    for (unsigned int r = 0; r < chunks_per_batch; ++r) {
      const float *x_row = x_b + r * feature_size;
      const float *dy_row = dy_b + r * feature_size;

      float ms = 0.0f;
      for (unsigned int w = 0; w < feature_size; ++w)
        ms += x_row[w] * x_row[w];
      ms /= feature_size;
      const float inv_rms = 1.0f / std::sqrt(ms + epsilon);

      for (unsigned int w = 0; w < feature_size; ++w)
        acc[w] += static_cast<double>(dy_row[w]) * x_row[w] * inv_rms;
    }
  }

  float *dg = dgamma.getData<float>();
  if (context.isGradientFirstAccess(wt_idx[RMSParams::gamma])) {
    for (unsigned int w = 0; w < feature_size; ++w)
      dg[w] = static_cast<float>(acc[w]);
  } else {
    for (unsigned int w = 0; w < feature_size; ++w)
      dg[w] += static_cast<float>(acc[w]);
  }
}

#ifdef PLUGGABLE

nntrainer::Layer *create_rms_norm_layer() {
  auto layer = new ReshapedRMSNormLayer();
  return layer;
}

void destroy_rms_norm_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_rms_norm_layer,
                                                   destroy_rms_norm_layer};
}

#endif

} // namespace causallm
