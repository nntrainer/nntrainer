// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   addition_layer.cpp
 * @date   30 July 2020
 * @see    https://github.com/nntrainer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Addition Layer Class for Neural Network
 *
 */

#include <addition_layer.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

#include <layer_context.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void AdditionLayer::finalize(InitLayerContext &context) {
  [[maybe_unused]] auto [output_dims, weight_dims, tensor_dims] =
    getLayerDimensions(context);
  context.setOutputDimensions(output_dims);

  if (!std::get<props::SkipPrefill>(add_props).empty())
    skip_prefill = std::get<props::SkipPrefill>(add_props).get();
}

std::vector<TensorDim>
AdditionLayer::updateTensorsByInputDimensions(InitLayerContext &init_context,
                                              RunLayerContext &run_context) {
  [[maybe_unused]] auto [output_dims, weight_dims, tensor_dims] =
    getLayerDimensions(init_context);

  for (size_t i = 0; i < run_context.getNumInputs(); ++i) {
    run_context.updateInput(i, init_context.getInputDimensions()[i]);
  }
  run_context.updateOutput(SINGLE_INOUT_IDX, output_dims[SINGLE_INOUT_IDX]);

  return output_dims;
}

void AdditionLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);

  /** @todo check possibility for in-place of addition layer */
  for (unsigned int idx = 0; idx < context.getNumInputs(); ++idx) {
    const Tensor &input_ = context.getInput(idx);
    if (!idx) {
      hidden_.copy(input_);
    } else {
      hidden_.add_i(input_);
    }
  }
}

void AdditionLayer::incremental_forwarding(RunLayerContext &context,
                                           unsigned int from, unsigned int to,
                                           bool training) {
  bool is_prefill = !from || (to - from) > 1;
  if (skip_prefill && is_prefill)
    return;

  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  TensorDim hidden_dim = hidden_.getDim();
  TensorDim hidden_step_dim = hidden_dim;

  hidden_step_dim.batch(1);
  hidden_step_dim.height(to - from);

  for (unsigned int b = 0; b < hidden_.batch(); ++b) {
    Tensor hidden_step = hidden_.getSharedDataTensor(
      hidden_step_dim, b * hidden_dim.getFeatureLen(), true);

    /** @todo check possibility for in-place of addition layer */
    for (unsigned int idx = 0; idx < context.getNumInputs(); ++idx) {
      const Tensor &input_ = context.getInput(idx);
      TensorDim input_dim = input_.getDim();

      TensorDim input_step_dim = input_dim;
      input_step_dim.batch(1);
      input_step_dim.height(to - from);

      Tensor input_step = input_.getSharedDataTensor(
        input_step_dim, b * input_dim.getFeatureLen(), true);
      if (!idx) {
        hidden_step.copy(input_step);
      } else {
        hidden_step.add_i(input_step);
      }
    }
  }
}

void AdditionLayer::calcDerivative(RunLayerContext &context) {

  for (unsigned int idx = 0; idx < context.getNumInputs(); ++idx) {
    /**
     * TODO: replace this with tensor assignment during optimization.
     * Tensor assignment needs to make sure that the previous connected layers
     * are not inplace
     */
    context.getOutgoingDerivative(idx).copy(
      context.getIncomingDerivative(SINGLE_INOUT_IDX));
  }
}

void AdditionLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, add_props);
  if (!remain_props.empty()) {
    std::string msg = "[AdditionLayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}

std::array<std::vector<TensorDim>, 3>
AdditionLayer::getLayerDimensions(InitLayerContext &context) {
  std::vector<TensorDim> output_dims(1);
  output_dims[SINGLE_INOUT_IDX] =
    context.getInputDimensions()[SINGLE_INOUT_IDX];

  NNTR_THROW_IF(std::any_of(context.getInputDimensions().cbegin(),
                            context.getInputDimensions().cend(),
                            [&output_dims](const TensorDim &dim) {
                              return output_dims[SINGLE_INOUT_IDX] != dim;
                            }),
                std::invalid_argument)
    << "All of input dimensions of addition layer SHOULD BE identical";

  return {output_dims, {}, {}};
}

} /* namespace nntrainer */
