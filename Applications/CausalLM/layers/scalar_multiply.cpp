// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   scalar_multiply.cpp
 * @date   7 April 2026
 * @brief  Implementation of scalar multiplication layer
 * @see    https://github.com/nntrainer/nntrainer
 * @author Joonseok Oh <jrock.oh@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <cmath>
#include <iostream>

#include "scalar_multiply.h"

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void ScalarMultiplyLayer::finalize(nntrainer::InitLayerContext &context) {
  std::vector<nntrainer::TensorDim> dim = context.getInputDimensions();
  context.setOutputDimensions(dim);
  if (!std::get<nntrainer::props::SkipPrefill>(scalar_multiply_props).empty())
    skip_prefill =
      std::get<nntrainer::props::SkipPrefill>(scalar_multiply_props).get();

  bool use_weight = std::get<props::UseWeight>(scalar_multiply_props).get();

  if (use_weight) {
    // Request weight for scalar value (single element)
    nntrainer::TensorDim scalar_dim(
      1, 1, 1, 1,
      nntrainer::TensorDim::TensorType(context.getFormat(),
                                       context.getWeightDataType()));
    wt_idx[0] = context.requestWeight(
      scalar_dim, nntrainer::props::InitializerInfo::Enum::NONE,
      nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "scalar_multiplier",
      false);
  }
}

void ScalarMultiplyLayer::forwarding(nntrainer::RunLayerContext &context,
                                     bool training) {
  // Use incremental_forwarding for actual computation
  auto &in = context.getInput(SINGLE_INOUT_IDX);
  auto &out = context.getOutput(SINGLE_INOUT_IDX);

  bool use_weight = std::get<props::UseWeight>(scalar_multiply_props).get();

  float multiplier;
  if (use_weight) {
    nntrainer::Tensor &weight = context.getWeight(wt_idx[0]);
    multiplier = weight.getValue<float>(0, 0, 0, 0);
  } else {
    multiplier = std::get<props::ScalarMultiplier>(scalar_multiply_props).get();
  }

  in.multiply(multiplier, out);
}

void ScalarMultiplyLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  bool is_prefill = !from;
  if (skip_prefill && is_prefill)
    return;

  bool use_weight = std::get<props::UseWeight>(scalar_multiply_props).get();

  float multiplier;
  if (use_weight) {
    nntrainer::Tensor &weight = context.getWeight(wt_idx[0]);
    multiplier = weight.getValue<float>(0, 0, 0, 0);
  } else {
    multiplier = std::get<props::ScalarMultiplier>(scalar_multiply_props).get();
  }

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

  unsigned int b_size = in_dim.batch();

  for (unsigned int b = 0; b < b_size; ++b) {
    nntrainer::Tensor in_step =
      in.getSharedDataTensor(in_step_dim, b * in_dim.getFeatureLen(), true);
    nntrainer::Tensor out_step =
      out.getSharedDataTensor(out_step_dim, b * out_dim.getFeatureLen(), true);

    in_step.multiply(multiplier, out_step);

#ifdef DEBUG
    std::cout << context.getName() << " \n input:" << in_step
              << "output:" << out_step << "multiplier:" << multiplier
              << std::endl;
#endif
  }
}

void ScalarMultiplyLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  context.updateInput(SINGLE_INOUT_IDX, input_dimensions[0]);
  context.updateOutput(SINGLE_INOUT_IDX, input_dimensions[0]);
}

void ScalarMultiplyLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  std::throw_with_nested(std::runtime_error("Training is not supported yet."));
}

#ifdef PLUGGABLE

nntrainer::Layer *create_scalar_multiply_layer() {
  auto layer = new ScalarMultiplyLayer();
  return layer;
}

void destroy_scalar_multiply_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{
  create_scalar_multiply_layer, destroy_scalar_multiply_layer};
}

#endif

} // namespace causallm
