// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   optimizer_context.h
 * @date   30 July 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is the layer context for each layer
 */

#include <optimizer_context.h>
#include <weight.h>

namespace nntrainer {

/**
 * @brief Get the Weight tensor object
 */
Tensor &RunOptimizerContext::getWeight() const {
  return weight->getVariableRef();
}

/**
 * @brief Get the Weight FP32 tensor object (master weight for mixed precision)
 */
Tensor &RunOptimizerContext::getWeightFP32() const {
  return weight->getVariableFP32Ref();
}

/**
 * @brief Get the Weight Gradient tensor object
 */
Tensor &RunOptimizerContext::getGradient() const {
  return weight->getGradientRef();
}

/**
 * @brief Get the optimizer variable associated to this weight
 */
Tensor &RunOptimizerContext::getOptimizerVariable(unsigned int idx) const {
  return weight->getOptimizerVariableRef(idx);
}

/**
 * @brief   Apply the gradient with the given learning rate
 */
void RunOptimizerContext::applyGradient(double lr) const {
  weight->applyGradient(lr);
}

/**
 * @brief   Apply the gradient with the given learning rate and gradient
 */
void RunOptimizerContext::applyGradient(double lr, Tensor &updated_grad) const {
  weight->applyGradient(lr, updated_grad);
}

/**
 * @brief   Apply loss scale to gradient (full precision)
 */
void RunOptimizerContext::applyLossScale(Tensor &fp32_grad) {
  if (!weight->isMixedPrecision())
    return;
  if (fp32_grad.getDataType() != ml::train::TensorDim::DataType::FP32)
    throw std::invalid_argument(
      "gradient should be fullprecsion to maintain accuracy");
  float loss_scale = weight->getLossScale();
  fp32_grad.divide_i(loss_scale);
}

/**
 * @brief Return if the underlying weight is mixed precision
 */
bool RunOptimizerContext::isMixedPrecision() const {
  return weight->isMixedPrecision();
}
} // namespace nntrainer
