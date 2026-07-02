// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Joonseok Oh <jrock.oh@samsung.com>
 *
 * @file   rms_reverse_norm.cpp
 * @date   27 March 2026
 * @brief  This is Reverse RMS Norm Layer Class
 * @see    https://github.com/nntrainer/nntrainer
 * @author Joonseok Oh <jrock.oh@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <cmath>
#include <iostream>

#include "rms_reverse_norm.h"

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum RMSReverseParams { weight, out_scale };

void RMSReverseNormLayer::finalize(nntrainer::InitLayerContext &context) {
  std::vector<nntrainer::TensorDim> dim = context.getInputDimensions();

  context.setOutputDimensions(dim);

  // Initialize weight and out_scale parameters
  auto weight_init = nntrainer::props::InitializerInfo::Enum::ONES;
  auto outscale_init = nntrainer::props::InitializerInfo::Enum::ONES;

  if (!std::get<props::RMS_REVERSE_NORM_WEIGHT_INIT>(rms_props).empty()) {
    weight_init =
      std::get<props::RMS_REVERSE_NORM_WEIGHT_INIT>(rms_props).get();
  }

  if (!std::get<props::RMS_REVERSE_NORM_OUTSCALE_INIT>(rms_props).empty()) {
    outscale_init =
      std::get<props::RMS_REVERSE_NORM_OUTSCALE_INIT>(rms_props).get();
  }

  if (!std::get<nntrainer::props::SkipPrefill>(rms_props).empty()) {
    skip_prefill = std::get<nntrainer::props::SkipPrefill>(rms_props).get();
  }

  // Request weight parameter (learnable multiplicative weight applied BEFORE
  // norm)
  nntrainer::TensorDim weight_dim(
    1, 1, 1, dim[0].width(),
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()));
  wt_idx[RMSReverseParams::weight] = context.requestWeight(
    weight_dim, weight_init, nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f,
    "weight", true);

  // Request out_scale parameter (learnable scale applied AFTER norm)
  nntrainer::TensorDim outscale_dim(
    1, 1, 1, 1,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()));
  wt_idx[RMSReverseParams::out_scale] = context.requestWeight(
    outscale_dim, outscale_init, nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f,
    "out_scale", true);
}

void RMSReverseNormLayer::forwarding(nntrainer::RunLayerContext &context,
                                     bool training) {}

void RMSReverseNormLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  auto &epsilon = std::get<nntrainer::props::Epsilon>(rms_props).get();

  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &weight =
    context.getWeight(wt_idx[RMSReverseParams::weight]);
  nntrainer::Tensor &out_scale =
    context.getWeight(wt_idx[RMSReverseParams::out_scale]);

  ml::train::TensorDim in_dim = in.getDim();
  ml::train::TensorDim out_dim = out.getDim();

  ml::train::TensorDim in_step_dim = in_dim;
  ml::train::TensorDim out_step_dim = out_dim;

  bool is_prefill = !from;
  if (from) {
    // Normalize to 0-based while preserving step size for multi-token prefill
    to = to - from;
    from = 0;
  } else if (skip_prefill && is_prefill)
    return;

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
      // ReverseRMSNorm order: x * weight → normalize → multiply by out_scale

      // Step 1: Multiply input by weight (BEFORE normalization)
      in_step.multiply_i(weight);

      // Step 2: Compute RMS normalization
      // rsqrt(average(x^2) + eps)
      auto t = in_step.multiply(in_step).average(3).add(epsilon);
      t.inv_sqrt_i();

      // Step 3: Apply normalization
      in_step.multiply(t, out_step);

      // Step 4: Apply output scale (AFTER normalization)
      out_step.multiply_i(out_scale);

      // TODO : Implement Fast Route for FP16
    } else if (in_step.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
      ml::train::TensorDim instep_dim = in_step_dim;
      ml::train::TensorDim outstep_dim = out_step_dim;

      instep_dim.setDataType(ml::train::TensorDim::DataType::FP32);
      outstep_dim.setDataType(ml::train::TensorDim::DataType::FP32);

      nntrainer::Tensor in_step32(instep_dim, true);
      nntrainer::Tensor out_step32(outstep_dim, true);

      in_step32.copyData(in_step);

      // ReverseRMSNorm order: x * weight → normalize → multiply by out_scale

      // Step 1: Multiply input by weight (BEFORE normalization)
      in_step32.multiply_i(weight);

      // Step 2: Compute RMS normalization
      auto t = in_step32.multiply(in_step32).average(3).add(epsilon);
      t.inv_sqrt_i();

      // Step 3: Apply normalization
      in_step32.multiply(t, out_step32);

      // Step 4: Apply output scale (AFTER normalization)
      out_step32.multiply_i(out_scale);

      out_step.copyData(out_step32);
#else
      throw std::invalid_argument("Error: enable-fp16 is not set");
#endif
    }

#ifdef DEBUG
    std::cout << context.getName() << " \n input:" << in_step
              << "output:" << out_step << "weight:" << weight
              << "out_scale:" << out_scale << std::endl;
#endif
  }
}

void RMSReverseNormLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  ml::train::TensorDim input_dim = context.getInput(SINGLE_INOUT_IDX).getDim();
  ml::train::TensorDim output_dim =
    context.getOutput(SINGLE_INOUT_IDX).getDim();

  input_dim.height(input_dimensions[0].height());
  output_dim.height(input_dimensions[0].height());

  context.updateInput(SINGLE_INOUT_IDX, input_dim);
  context.updateOutput(SINGLE_INOUT_IDX, output_dim);
}

void RMSReverseNormLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  // Training not implemented yet
  // std::throw_with_nested(std::runtime_error("Training is not supported
  // yet."));
}

#ifdef PLUGGABLE

nntrainer::Layer *create_rms_reverse_norm_layer() {
  auto layer = new RMSReverseNormLayer();
  return layer;
}

void destroy_rms_reverse_norm_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{
  create_rms_reverse_norm_layer, destroy_rms_reverse_norm_layer};
}

#endif

} // namespace causallm
