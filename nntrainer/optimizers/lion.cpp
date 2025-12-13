// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jeonghun Park <top231902@naver.com>
 *
 * @file   lion.cpp
 * @date   1 December 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jeonghun Park <top231902@naver.com>
 * @author Minseo Kim <ms05251@naver.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is the Lion Optimizer.
 */

#include <cmath>
#include <fstream>

#include <lion.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

namespace nntrainer {

Lion::Lion() : lion_props(PropsB1(), PropsB2(), PropsWeightDecayLion()) {
  auto &[beta1, beta2, weight_decay] = lion_props;
  beta1.set(0.9f);
  beta2.set(0.99f);
  weight_decay.set(0.0f);
}

Lion::~Lion() {}

enum LionParams { m };

std::vector<TensorDim> Lion::getOptimizerVariableDim(const TensorDim &dim) {
  TensorDim m_dim(dim);
  m_dim.setDataType(ml::train::TensorDim::DataType::FP32);
  return {m_dim};
}

void Lion::exportTo(Exporter &exporter,
                    const ml::train::ExportMethods &method) const {
  exporter.saveResult(lion_props, method, this);
  Optimizer::exportTo(exporter, method);
}

void Lion::setProperty(const std::vector<std::string> &values) {
  auto left = loadProperties(values, lion_props);
  Optimizer::setProperty(left);
}

void Lion::applyGradient(RunOptimizerContext &context) {
  // 1. Get Tensors and Properties
  Tensor empty_tensor;
  Tensor &x_grad =
    context.getGradient().getDataType() == ml::train::TensorDim::DataType::FP32
      ? context.getGradient()
      : empty_tensor;

  if (x_grad.empty()) {
    x_grad = context.getGradient().clone(ml::train::TensorDim::DataType::FP32);
  }

  context.applyLossScale(x_grad);

  Tensor &m = context.getOptimizerVariable(LionParams::m);

  auto &beta1 = std::get<PropsB1>(lion_props).get();
  auto &beta2 = std::get<PropsB2>(lion_props).get();
  auto &weight_decay = std::get<PropsWeightDecayLion>(lion_props).get();
  float lr = context.getLearningRate();

  Tensor original_x_grad = x_grad.clone();

  // 2. Calculate interpolated momentum: c_t = beta1 * m_t + (1 - beta1) * g_t
  x_grad.multiply_i(1.0 - beta1);
  x_grad.add_i(m, beta1);

  // 3. Update momentum for next iteration: m_{t+1} = beta2 * m_t + (1 - beta2)
  // * g_t
  m.multiply_i(beta2);
  m.add_i(original_x_grad, 1.0 - beta2);

  // 4. Take the sign of the interpolated momentum
  std::function<float(float)> sign_func = [](float val) {
    if (val > 0.0f)
      return 1.0f;
    if (val < 0.0f)
      return -1.0f;
    return 0.0f;
  };
  x_grad.apply_i<float>(sign_func);

  // 5. Add decoupled weight decay term. w = w - lr * wd * w
  if (weight_decay > 0.0) {
    Tensor &w = context.isMixedPrecision() ? context.getWeightFP32()
                                           : context.getWeight();
    w.multiply_i(1.0f - (context.getLearningRate() * weight_decay));
  }

  // 6. Apply the final gradient update
  context.applyGradient(lr, x_grad);
}

} // namespace nntrainer
