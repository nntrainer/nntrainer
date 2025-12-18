// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Sumon Nath <sumon.nath@samsung.com>
 *
 * @file   negative_layer.cpp
 * @date   3 July 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sumon Nath <sumon.nath@samsung.com>
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is negative layer class (operation layer)
 */

#include "common_properties.h"
#include <negative_layer.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

#include <layer_context.h>

namespace nntrainer {

void NegativeLayer::finalize(InitLayerContext &context) {
  context.setOutputDimensions({context.getInputDimensions()[0]});
}

void NegativeLayer::forwarding_operation(const Tensor &input, Tensor &hidden) {
  input.multiply(-1, hidden);
}

void NegativeLayer::calcDerivative(RunLayerContext &context) {
  auto &deriv = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  deriv.copy(context.getIncomingDerivative(SINGLE_INOUT_IDX));
  deriv.multiply(-1.0f);
}

void NegativeLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, negative_props);
  if (!remain_props.empty()) {
    std::string msg = "[NegativeLayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}

} /* namespace nntrainer */
