// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   cast_layer.cpp
 * @date   04 April 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is cast layer class (operation layer)
 *
 */

#include "base_properties.h"
#include "common_properties.h"
#include <cast_layer.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

#include <layer_context.h>
namespace nntrainer {

void CastLayer::finalize(InitLayerContext &context) {
  props::TensorDataType dtype =
    std::get<props::TensorDataType>(cast_props).get();
  TensorDim out_dim = TensorDim(context.getInputDimensions()[0]);
  out_dim.setDataType(dtype);
  context.setOutputDimensions({out_dim});
}

void CastLayer::forwarding_operation(const Tensor &input, Tensor &output) {
  // Casting type is performed in copyData function
  output.copyData(input);
}

void CastLayer::calcDerivative(RunLayerContext &context) {
  context.getOutgoingDerivative(SINGLE_INOUT_IDX)
    .copyData(context.getIncomingDerivative(SINGLE_INOUT_IDX));
}

void CastLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, cast_props);
  if (!remain_props.empty()) {
    std::string msg = "[CastLayer] Unknown Layer Properties count " +
                      std::to_string(remain_props.size());
    throw exception::not_supported(msg);
  }
}

} /* namespace nntrainer */
