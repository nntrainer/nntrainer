// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   cosine_layer.cpp
 * @date   19 March 2025
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is cosine layer class (operation layer)
 *
 */

#include "common_properties.h"
#include <cosine_layer.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

#include <layer_context.h>

namespace nntrainer {

void CosineLayer::finalize(InitLayerContext &context) {
  context.setOutputDimensions({context.getInputDimensions()[0]});
}

void CosineLayer::forwarding_operation(const Tensor &input, Tensor &hidden) {
  input.cos(hidden);
}

void CosineLayer::calcDerivative(RunLayerContext &context) {
  auto &deriv = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  context.getInput(SINGLE_INOUT_IDX).sin(deriv);
  deriv.multiply_i(context.getIncomingDerivative(SINGLE_INOUT_IDX));
  deriv.multiply_i(-1.0f);
}

void CosineLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, cosine_props);
  if (!remain_props.empty()) {
    std::string msg = "[CosineLayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}
} /* namespace nntrainer */
