// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   activation_layer.cpp
 * @date   17 June 2020
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Activation Layer Class for Neural Network
 *
 */

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <activation_layer.h>
#include <common_properties.h>
#include <cpu_backend.h>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <tensor.h>
#include <tensor_wrap_specs.h>
#include <util_func.h>

namespace nntrainer {
ActivationLayer::ActivationLayer() :
  Layer(),
  activation_props(new PropTypes(props::Activation(), props::SkipPrefill())) {
  acti_func.setActiFunc(ActivationType::ACT_NONE);
}

static constexpr size_t SINGLE_INOUT_IDX = 0;

void ActivationLayer::finalize(InitLayerContext &context) {
  auto &act = std::get<props::Activation>(*activation_props);
  if (!std::get<props::SkipPrefill>(*activation_props).empty())
    skip_prefill = std::get<props::SkipPrefill>(*activation_props).get();
  NNTR_THROW_IF(act.empty(), std::invalid_argument)
    << "activation has not been set!";
  if (context.getActivationDataType() == TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    acti_func.setActiFunc<_FP16>(act.get());
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  } else if (context.getActivationDataType() == TensorDim::DataType::FP32) {
    acti_func.setActiFunc<float>(act.get());
  }

  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "activation layer, " << context.getName()
    << "requires exactly one input, but given: " << context.getNumInputs()
    << ", check graph connection if it is correct";

  /// @todo for only certain types of activation needs lifespan of
  /// forward_derivative order
  std::vector<VarGradSpecV2> out_specs;
  out_specs.push_back(
    InitLayerContext::outSpec(context.getInputDimensions()[0], "out",
                              TensorLifespan::FORWARD_DERIV_LIFESPAN));
  context.requestOutputs(std::move(out_specs));
  acti_func.setInPlace(context.getInPlace());
}

void ActivationLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  acti_func.run_fn(input_, hidden_);
}

void ActivationLayer::incremental_forwarding(RunLayerContext &context,
                                             unsigned int from, unsigned int to,
                                             bool training) {
  (void)training;
  bool is_prefill = !from;
  if (skip_prefill && is_prefill)
    return;

  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  TensorDim input_dim = input_.getDim();
  TensorDim hidden_dim = hidden_.getDim();

  TensorDim input_step_dim = input_dim;
  TensorDim hidden_step_dim = hidden_dim;

  input_step_dim.batch(1);
  hidden_step_dim.batch(1);

  if (input_dim.height() > 1)
    input_step_dim.height(to - from);
  if (hidden_dim.height() > 1)
    hidden_step_dim.height(to - from);

  for (unsigned int b = 0; b < hidden_.batch(); ++b) {
    Tensor input_step = input_.getSharedDataTensor(
      input_step_dim, b * input_dim.getFeatureLen(), true);
    Tensor hidden_step = hidden_.getSharedDataTensor(
      hidden_step_dim, b * hidden_dim.getFeatureLen(), true);

    acti_func.run_fn(input_step, hidden_step);
  }
}

void ActivationLayer::calcDerivative(RunLayerContext &context) {
  const Tensor &deriv = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &ret = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  Tensor &out = context.getOutput(SINGLE_INOUT_IDX);

  acti_func.run_prime_fn(in, out, ret, deriv);
}

void ActivationLayer::exportTo(Exporter &exporter,
                               const ml::train::ExportMethods &method) const {
  exporter.saveResult(*activation_props, method, this);
}

void ActivationLayer::setProperty(const std::vector<std::string> &values) {
  auto left = loadProperties(values, *activation_props);
  NNTR_THROW_IF(!left.empty(), std::invalid_argument)
    << "Failed to set property";

  auto &act = std::get<props::Activation>(*activation_props);
  if (!act.empty())
    acti_func.setActiFunc(act.get());
}

}; // namespace nntrainer
