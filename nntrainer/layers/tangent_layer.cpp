// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   tangent_layer.cpp
 * @date   19 March 2025
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is tangent layer class (operation layer)
 */

#include "common_properties.h"
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <tangent_layer.h>
#include <util_func.h>

#include <layer_context.h>

namespace nntrainer {

void TangentLayer::finalize(InitLayerContext &context) {
  context.setOutputDimensions({context.getInputDimensions()[0]});
}

void TangentLayer::forwarding_operation(const Tensor &input, Tensor &hidden) {
  input.tan(hidden);
}

void TangentLayer::calcDerivative(RunLayerContext &context) {
  auto &deriv = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  context.getInput(SINGLE_INOUT_IDX).cos(deriv);
  deriv.pow_i(-2.0f);
  deriv.multiply_i(context.getIncomingDerivative(SINGLE_INOUT_IDX));
}

void TangentLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, tangent_props);
  if (!remain_props.empty()) {
    std::string msg = "[TangentLayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}

} /* namespace nntrainer */
