// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   gather_layer.cpp
 * @date   02 April 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    It's not implemented operation yet. Just a draft for compilation.
 * @brief  This is gather layer class (operation layer)
 */

#include "common_properties.h"
#include <gather_layer.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <stdexcept>
#include <util_func.h>

#include <layer_context.h>

namespace nntrainer {

void GatherLayer::finalize(InitLayerContext &context) {
  axis = std::get<props::Axis>(gather_props).get();
  TensorDim inputDim = context.getInputDimensions()[0];
  TensorDim indexDim = context.getInputDimensions()[1];

  if (axis < 1 || axis > 3) {
    throw std::invalid_argument(
      "The axis property of GatherLayer should be between 1 and 3.");
  }

  if (inputDim[0] != indexDim[0]) {
    throw std::invalid_argument(
      "The batch size of the input and index should be same.");
  }

  TensorDim outputDim = TensorDim(inputDim);
  outputDim.setTensorDim(axis, indexDim[axis]);
  context.setOutputDimensions({outputDim});
}

void GatherLayer::forwarding_operation(const Tensor &input, const Tensor &index,
                                       Tensor &output) {
  // TODO: implement forwarding operation
  throw std::runtime_error("forwarding operation is not implemented yet");
}

void GatherLayer::calcDerivative(RunLayerContext &context) {
  // TODO: implement derivative calculation
  throw std::runtime_error("derivative calculation is not implemented yet");
}

void GatherLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, gather_props);
  if (!remain_props.empty()) {
    std::string msg = "[GatherLayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}

} /* namespace nntrainer */
