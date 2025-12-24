// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   gather_layer.cpp
 * @date   02 April 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
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
  for (unsigned int b = 0; b < index.getDim().batch(); ++b) {
    for (unsigned int i = 0; i < index.getDim().channel(); ++i) {
      for (unsigned int j = 0; j < index.getDim().height(); ++j) {
        for (unsigned int k = 0; k < index.getDim().width(); ++k) {
#ifdef ENABLE_FP16
          auto selected = (size_t)index.getValue<_FP16>(b, i, j, k);
          if (selected >= input.getDim()[axis]) {
            throw std::invalid_argument("The index value is out of range.");
          }
          switch (axis) {
          case 1:
            output.setValue(b, i, j, k,
                            input.getValue<_FP16>(b, selected, j, k));
            break;
          case 2:
            output.setValue(b, i, j, k,
                            input.getValue<_FP16>(b, i, selected, k));
            break;
          case 3:
            output.setValue(b, i, j, k,
                            input.getValue<_FP16>(b, i, j, selected));
          default:
            break;
          }
#else
          auto selected = (size_t)index.getValue(b, i, j, k);
          if (selected >= input.getDim()[axis]) {
            throw std::invalid_argument("The index value is out of range.");
          }
          switch (axis) {
          case 1:
            output.setValue(b, i, j, k, input.getValue(b, selected, j, k));
            break;
          case 2:
            output.setValue(b, i, j, k, input.getValue(b, i, selected, k));
            break;
          case 3:
            output.setValue(b, i, j, k, input.getValue(b, i, j, selected));
          default:
            break;
          }
#endif
        }
      }
    }
  }
}

void GatherLayer::calcDerivative(RunLayerContext &context) {
  const Tensor &index = context.getInput(1);
  const Tensor &inDeriv = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &outDeriv = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  for (unsigned int b = 0; b < index.getDim().batch(); ++b) {
    for (unsigned int i = 0; i < index.getDim().channel(); ++i) {
      for (unsigned int j = 0; j < index.getDim().height(); ++j) {
        for (unsigned int k = 0; k < index.getDim().width(); ++k) {
          auto inDerivValue = inDeriv.getValue(b, i, j, k);
          auto selected = (size_t)index.getValue(b, i, j, k);
          switch (axis) {
          case 0:
            outDeriv.addValue(b, selected, j, k, inDerivValue, 1);
            break;
          case 1:
            outDeriv.addValue(b, i, selected, k, inDerivValue, 1);
            break;
          case 2:
            outDeriv.addValue(b, i, j, selected, inDerivValue, 1);
          default:
            break;
          }
        }
      }
    }
  }
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
