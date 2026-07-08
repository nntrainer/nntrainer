// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 hyeonseok Lee <hs89.lee@samsung.com>
 *
 * @file   layer_normalization_layer.cpp
 * @date   25 July 2022
 * @see    https://github.com/nntrainer/nntrainer
 *         https://arxiv.org/abs/1607.06450
 * @author hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Layer Normalization Layer Class for Neural Network
 *
 */

#include <algorithm>
#include <numeric>

#include <base_properties.h>
#include <cpu_backend.h>
#include <layer_context.h>
#include <layer_normalization_layer.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum LNParams {
  gamma,
  beta,
  deviation,
  variance,
  inv_std_dev,
  temp_origin_size,
  temp_normalized_size,
};

LayerNormalizationLayer::LayerNormalizationLayer() :
  Layer(),
  layer_normalization_props(std::vector<props::Axis>(), props::Epsilon(),
                            props::GammaInitializer(), props::BetaInitializer(),
                            props::WeightDecay(), props::BiasDecay()) {
  wt_idx.fill(std::numeric_limits<unsigned>::max());
}

void LayerNormalizationLayer::finalize(InitLayerContext &context) {
  if (context.getNumInputs() != 1) {
    throw std::invalid_argument(
      "Only one input is allowed for layer normalization layer");
  }

  auto gamma_initializer =
    std::get<props::GammaInitializer>(layer_normalization_props).get();
  auto beta_initializer =
    std::get<props::BetaInitializer>(layer_normalization_props).get();
  auto weight_decay = std::get<props::WeightDecay>(layer_normalization_props);
  auto bias_decay = std::get<props::BiasDecay>(layer_normalization_props);

  auto const &input_dim = context.getInputDimensions()[0];
  context.setOutputDimensions({input_dim});

  std::vector<props::Axis> axes_prop =
    std::get<std::vector<props::Axis>>(layer_normalization_props);

  NNTR_THROW_IF(axes_prop.empty(), std::invalid_argument)
    << "[Layer normalization]axis property is empty";

  normalize_axes.insert(normalize_axes.end(), axes_prop.begin(),
                        axes_prop.end());
  std::sort(normalize_axes.begin(), normalize_axes.end());
  normalize_axes.erase(
    std::unique(normalize_axes.begin(), normalize_axes.end()),
    normalize_axes.end());

  TensorDim normalize_dim(context.getFormat(), context.getWeightDataType());
  for (unsigned int axis : normalize_axes) {
    normalize_dim.setTensorDim(axis, input_dim.getTensorDim(axis));
  }

  wt_idx[LNParams::gamma] = context.requestWeight(
    normalize_dim, gamma_initializer, WeightRegularizer::NONE, 1.0f,
    weight_decay, "gamma", true);
  wt_idx[LNParams::beta] = context.requestWeight(
    normalize_dim, beta_initializer, WeightRegularizer::NONE, 1.0f, bias_decay,
    "beta", true);

  TensorDim remain_dim(context.getFormat(), context.getWeightDataType());
  std::vector<unsigned int> total_axes;
  total_axes.resize(ml::train::TensorDim::MAXDIM);
  std::iota(total_axes.begin(), total_axes.end(), 0u);
  std::set_difference(total_axes.begin(), total_axes.end(),
                      normalize_axes.begin(), normalize_axes.end(),
                      std::back_inserter(remain_axes));
  for (unsigned int axis : remain_axes) {
    remain_dim.setTensorDim(axis, input_dim.getTensorDim(axis));
  }

  /** caches the deviation -> input - avg(input) */
  wt_idx[LNParams::deviation] =
    context.requestTensor(input_dim, "deviation", Initializer::NONE, false,
                          TensorLifespan::ITERATION_LIFESPAN);
  /** caches variance + epsilon as well */
  wt_idx[LNParams::variance] =
    context.requestTensor(remain_dim, "variance", Initializer::NONE, false,
                          TensorLifespan::ITERATION_LIFESPAN);
  /** caches the inverse standard deviation */
  wt_idx[LNParams::inv_std_dev] =
    context.requestTensor(remain_dim, "inv_std_dev", Initializer::NONE, false,
                          TensorLifespan::ITERATION_LIFESPAN);

  /** temporary tensor (origin size) */
  wt_idx[LNParams::temp_origin_size] =
    context.requestTensor(input_dim, "temp_origin_size", Initializer::NONE,
                          false, TensorLifespan::CALC_DERIV_LIFESPAN);
  /** temporary tensor (normalized size) */
  wt_idx[LNParams::temp_normalized_size] =
    context.requestTensor(remain_dim, "temp_normalized_size", Initializer::NONE,
                          false, TensorLifespan::CALC_DERIV_LIFESPAN);
}

void LayerNormalizationLayer::setProperty(
  const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, layer_normalization_props);
  NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
    << "[Layer Normalization Layer] Unknown Layer Properties count " +
         std::to_string(values.size());
}

void LayerNormalizationLayer::forwarding(RunLayerContext &context,
                                         bool training) {
  const float epsilon =
    std::get<props::Epsilon>(layer_normalization_props).get();

  const Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  Tensor &output = context.getOutput(SINGLE_INOUT_IDX);

  Tensor &gamma = context.getWeight(wt_idx[LNParams::gamma]);
  Tensor &beta = context.getWeight(wt_idx[LNParams::beta]);

  Tensor &deviation = context.getTensor(wt_idx[LNParams::deviation]);
  Tensor &variance = context.getTensor(wt_idx[LNParams::variance]);
  Tensor &inv_std_dev = context.getTensor(wt_idx[LNParams::inv_std_dev]);

  Tensor &temp_full_size = output;
  Tensor &temp_norm_size = inv_std_dev;

  input.average(normalize_axes, temp_norm_size);
  input.subtract(temp_norm_size, deviation);

  deviation.pow(2.0, temp_full_size);
  temp_full_size.average(normalize_axes, variance);

  variance.add_i(epsilon);
  variance.pow(-0.5, inv_std_dev);

  deviation.multiply(inv_std_dev, output);
  output.multiply_i(gamma);
  output.add_i(beta);
}

void LayerNormalizationLayer::incremental_forwarding(RunLayerContext &context,
                                                     unsigned int from,
                                                     unsigned int to,
                                                     bool training) {
  const float epsilon =
    std::get<props::Epsilon>(layer_normalization_props).get();

  const Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  Tensor &output = context.getOutput(SINGLE_INOUT_IDX);

  Tensor &gamma = context.getWeight(wt_idx[LNParams::gamma]);
  Tensor &beta = context.getWeight(wt_idx[LNParams::beta]);

  Tensor &deviation = context.getTensor(wt_idx[LNParams::deviation]);
  Tensor &variance = context.getTensor(wt_idx[LNParams::variance]);
  Tensor &inv_std_dev = context.getTensor(wt_idx[LNParams::inv_std_dev]);

  // @todo: consider NHWC format
  bool is_height_normalize =
    std::find(normalize_axes.begin(), normalize_axes.end(), 1) !=
        normalize_axes.end()
      ? true
      : false;

  TensorDim input_dim = input.getDim();
  TensorDim output_dim = output.getDim();
  TensorDim normalize_dim = gamma.getDim();
  TensorDim remain_dim = variance.getDim();

  TensorDim input_step_dim = input_dim;
  TensorDim output_step_dim = output_dim;
  TensorDim normalize_step_dim = normalize_dim;
  TensorDim remain_step_dim = remain_dim;

  input_step_dim.height(to - from);
  output_step_dim.height(to - from);
  normalize_step_dim.height(is_height_normalize ? to - from : 1);
  remain_step_dim.height(is_height_normalize ? 1 : to - from);

  Tensor &temp_full_size = output;
  Tensor &temp_norm_size = inv_std_dev;

  input.average(normalize_axes, temp_norm_size);
  input.subtract(temp_norm_size, deviation);

#ifndef ENABLE_FP16
  deviation.pow(2.0f, temp_full_size);
  temp_full_size.average(normalize_axes, variance);

  variance.add_i(epsilon);
  variance.pow(-0.5f, inv_std_dev);
  deviation.multiply(inv_std_dev, output);
#else
  // Optimized path for FP32 tensors with width-axis-only normalization.
  // Uses NEON intrinsic (rms_norm_wrt_width_fp16_intrinsic) which:
  // - Accumulates in FP16 for better performance
  // - Computes: output = deviation / sqrt(mean(deviation^2) + epsilon)
  // - Supports incremental decoding with [from, to) row range
  // - Processes all batches and channels
  const bool width_axis_only =
    normalize_axes.size() == 1 &&
    normalize_axes[0] == ml::train::TensorDim::getNumDim() - 1;

  // The FP32 rms_norm_wrt_width_fp16_intrinsic overload is a deprecated stub on
  // every backend (ARM neon_impl_fp16.cpp and x86 fallback both throw
  // "deprecated due to overflow in fp16"). So it must never be taken for FP32
  // tensors; force the generic, numerically-correct path. This makes FP32 Layer
  // Normalization work on ENABLE_FP16 builds (e.g. Android arm64). The FP16
  // (non-FP32) tensor case still uses the generic fallback below as before.
  constexpr bool kFp32WidthNormIntrinsicAvailable = false;

  if (kFp32WidthNormIntrinsicAvailable &&
      deviation.getDataType() == ml::train::TensorDim::DataType::FP32 &&
      width_axis_only) {
    const unsigned int W = input_dim.width();
    const unsigned int row_dim_v = input_dim.height();
    const unsigned int row_begin = (to > from && to <= row_dim_v) ? from : 0u;
    const unsigned int row_end =
      (to > from && to <= row_dim_v) ? to : row_dim_v;
    const size_t row_count = row_end - row_begin;

    for (unsigned int b = 0; b < input_dim.batch(); ++b) {
      for (unsigned int c = 0; c < input_dim.channel(); ++c) {
        const float *src = deviation.getAddress<float>(b, c, row_begin, 0);
        float *dst = output.getAddress<float>(b, c, row_begin, 0);
        // NOTE: rms_norm_wrt_width_fp32_intrinsic is used here for Layer
        // Normalization. Since 'deviation' is already centered (X - mean),
        // computing:
        //   output = deviation / sqrt(mean(deviation^2) + epsilon)
        // is equivalent to Layer Norm's formula:
        //   output = (X - mean_X) / sqrt(variance + epsilon)
        // DO NOT USE rms_norm_wrt_width_fp16_intrinsic here — it accumulates
        // squared values in FP16, which overflows for typical LayerNorm inputs
        // (e.g. DeBERTa). Use the FP32-accumulator variant instead.
        nntrainer::rms_norm_wrt_width_fp32_intrinsic(src, dst, row_count, W,
                                                     epsilon);
      }
    }
  } else {
    // Fallback: generic tensor operations for all data types.
    // Used when: FP16 not enabled, non-FP32 tensor, or multi-axis
    // normalization.
    deviation.pow(2.0f, temp_full_size);
    temp_full_size.average(normalize_axes, variance);
    variance.add_i(epsilon);
    variance.pow(-0.5f, inv_std_dev);
    deviation.multiply(inv_std_dev, output);
  }
#endif
  output.multiply_i(gamma);
  output.add_i(beta);
}

void LayerNormalizationLayer::calcDerivative(RunLayerContext &context) {
  const bool trainable = context.getTrainable();

  TensorDim::TensorType weight_tensor_type =
    context.getWeight(wt_idx[LNParams::gamma]).getTensorType();

  Tensor empty =
    Tensor("empty", weight_tensor_type.format, weight_tensor_type.data_type);

  Tensor &outgoing_derivative = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  const Tensor &incoming_derivative =
    context.getIncomingDerivative(SINGLE_INOUT_IDX);

  const Tensor &gamma = context.getWeight(wt_idx[LNParams::gamma]);
  Tensor &d_gamma =
    trainable ? context.getWeightGrad(wt_idx[LNParams::gamma]) : empty;

  Tensor &deviation = context.getTensor(wt_idx[LNParams::deviation]);
  Tensor &variance = context.getTensor(wt_idx[LNParams::variance]);
  Tensor &inv_std_dev = context.getTensor(wt_idx[LNParams::inv_std_dev]);

  Tensor &temp_origin_size =
    context.getTensor(wt_idx[LNParams::temp_origin_size]);
  Tensor &temp_normalized_size =
    context.getTensor(wt_idx[LNParams::temp_normalized_size]);

  incoming_derivative.multiply(deviation, temp_origin_size);
  temp_origin_size.average(normalize_axes, temp_normalized_size);
  temp_normalized_size.divide_i(variance);
  deviation.multiply_i(temp_normalized_size);

  if (trainable) {
    /** calculate d_gamma */
    temp_origin_size.multiply_i(inv_std_dev);
    temp_origin_size.sum(remain_axes, d_gamma);
  }
  incoming_derivative.average(normalize_axes, temp_normalized_size);
  incoming_derivative.subtract(temp_normalized_size, outgoing_derivative);
  outgoing_derivative.subtract_i(deviation);

  inv_std_dev.multiply_i(gamma);
  outgoing_derivative.multiply_i(inv_std_dev);
}

void LayerNormalizationLayer::calcGradient(RunLayerContext &context) {
  /** d_gamma is calculated in calcDerivative. d_beta is calculated here */
  const Tensor &incoming_derivative =
    context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &d_beta = context.getWeightGrad(wt_idx[LNParams::beta]);

  incoming_derivative.sum(remain_axes, d_beta);
}

void LayerNormalizationLayer::exportTo(
  Exporter &exporter, const ml::train::ExportMethods &method) const {
  exporter.saveResult(layer_normalization_props, method, this);
}

void LayerNormalizationLayer::setBatch(RunLayerContext &context,
                                       unsigned int batch) {
  context.updateTensor(wt_idx[LNParams::deviation], batch);
  context.updateTensor(wt_idx[LNParams::variance], batch);
  context.updateTensor(wt_idx[LNParams::inv_std_dev], batch);
  context.updateTensor(wt_idx[LNParams::temp_origin_size], batch);
  context.updateTensor(wt_idx[LNParams::temp_normalized_size], batch);
}

} /* namespace nntrainer */
