// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   dropout.cpp
 * @date   16 June 2020
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug	   No known bugs except for NYI items
 * @brief  This is Dropout Layer Class for Neural Network
 *
 */

#include <dropout.h>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void DropOutLayer::finalize(InitLayerContext &context) {
  auto const &input_dims = context.getInputDimensions();
  context.setOutputDimensions(input_dims);

  mask_idx.reserve(input_dims.size());
  for (auto &t : input_dims) {
    mask_idx.push_back(context.requestTensor(
      t, "Mask", Initializer::NONE, false, TensorLifespan::ITERATION_LIFESPAN));
  }
}

void DropOutLayer::forwarding(RunLayerContext &context, bool training) {
  auto &rate_ = std::get<props::DropOutRate>(dropout_rate).get();

  // Dropout currently writes to the output tensor. During inference or when the
  // dropout rate is effectively zero, dropout is a no-op and the input is copied
  // to the output tensor.
  for (unsigned int i = 0; i < context.getNumInputs(); ++i) {
    Tensor &input_ = context.getInput(i);
    Tensor &output_ = context.getOutput(i);

    if (training && rate_ > epsilon) {
      Tensor &mask_ = context.getTensor(mask_idx[i]);
      if (!context.reStoreData()) {
        mask_.dropout_mask(rate_);
      }

      input_.multiply(mask_, output_);
    } else {
      output_.fill(input_);
    }
  }
}

void DropOutLayer::calcDerivative(RunLayerContext &context) {
  // Backpropagation also writes to the outgoing derivative tensor.
  auto &rate_ = std::get<props::DropOutRate>(dropout_rate).get();

  for (unsigned int i = 0; i < context.getNumInputs(); ++i) {
    const Tensor &derivative_ = context.getIncomingDerivative(i);
    Tensor &ret_ = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

    if (rate_ > epsilon) {
      Tensor &mask_ = context.getTensor(mask_idx[i]);
      derivative_.multiply(mask_, ret_);
    } else {
      ret_.fill(derivative_);
    }
  }
}

void DropOutLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, dropout_rate);
  if (!remain_props.empty()) {
    std::string msg = "[DropOutLayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}

} /* namespace nntrainer */
