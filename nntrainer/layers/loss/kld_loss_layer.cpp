// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   kld_loss_layer.cpp
 * @date   25 November 2021
 * @brief  KLD (Kullback-Leibler Divergence) loss implementation
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#include <kld_loss_layer.h>
#include <layer_context.h>
#include <string>
#include <vector>

namespace nntrainer {
static constexpr size_t SINGLE_INOUT_IDX = 0;

void KLDLossLayer::forwarding(RunLayerContext &context, bool training) {
  // Result = (P * (P / Q).log()).sum()
  // KL(P ∣∣ Q) whereP denotes the distribution of the observations in datasets
  // and Q denotes the model output.

  nntrainer::Tensor &predicted = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  if (context.isLabelAvailable(SINGLE_INOUT_IDX)) {
    nntrainer::Tensor &label = context.getLabel(SINGLE_INOUT_IDX);
    nntrainer::Tensor temp; // temp output
    /**
     * 1. Output = label / predicted
     * 2. Output = log(Output)
     * 3. Output = Output * label
     * 4. Output = sum(output)
     */
    label.divide(predicted, temp);
    temp.apply<float>(logf, temp);
    temp.multiply_i(label);
    output.fill(temp.sum({0, 1, 2, 3}));
  }
}

void KLDLossLayer::calcDerivative(RunLayerContext &context) {
  /**
   * d/dQ = -P/Q
   */
  nntrainer::Tensor &predicted = context.getInput(SINGLE_INOUT_IDX); // Q
  const nntrainer::Tensor &label = context.getLabel(SINGLE_INOUT_IDX); // P
  nntrainer::Tensor &deriv = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  label.multiply(-1.0f, deriv);
  deriv.divide_i(predicted);
}

} // namespace nntrainer
