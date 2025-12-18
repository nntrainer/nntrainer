// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Daniel Jang <minhyukjang@snu.ac.kr>
 *
 * @file   adamw.cpp
 * @date   3 November 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @author Daniel Jang <minhyukjang@snu.ac.kr>
 * @bug    No known bugs except for NYI items
 * @brief  This is the AdamW Optimizer.
 */

#include <cmath>
#include <fstream>

#include <adamw.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

namespace nntrainer {

AdamW::AdamW() :
  adam_props(PropsB1(), PropsB2(), PropsEpsilon(), TorchRef(),
             PropsWeightDecayW()) {
  /** default properties */
  auto &[b1, b2, eps, torch_ref, weight_decay] = adam_props;
  b1.set(0.9f);
  b2.set(0.999f);
  eps.set(1.0e-8f);
  torch_ref.set(false);
  weight_decay.set(0.0f);
}

AdamW::~AdamW() {}

enum AdamParams { wm, wv };

std::vector<TensorDim> AdamW::getOptimizerVariableDim(const TensorDim &dim) {
  /**
   * @note We assume the optimizer parameters should be full precision to
   * maintain the accuracy even in mixed precision training.
   */
  TensorDim wm_dim(dim);
  TensorDim wv_dim(dim);
  wm_dim.setDataType(ml::train::TensorDim::DataType::FP32);
  wv_dim.setDataType(ml::train::TensorDim::DataType::FP32);
  return {wm_dim, wv_dim};
}

void AdamW::exportTo(Exporter &exporter,
                     const ml::train::ExportMethods &method) const {
  exporter.saveResult(adam_props, method, this);
  Optimizer::exportTo(exporter, method);
}

void AdamW::setProperty(const std::vector<std::string> &values) {
  auto left = loadProperties(values, adam_props);
  Optimizer::setProperty(left);
}

double AdamW::getUpdatedLearningRate(unsigned int iteration, double lr) const {
  auto &beta1 = std::get<PropsB1>(adam_props).get();
  auto &beta2 = std::get<PropsB2>(adam_props).get();
  auto biasCorrection = [&](double f) {
    return 1.0 - (double)pow(f, iteration + 1);
  };
  lr *= sqrt(biasCorrection(beta2)) / biasCorrection(beta1);
  return lr;
}

void AdamW::applyGradient(RunOptimizerContext &context) {
  Tensor empty_tensor;

  Tensor &x_grad =
    context.getGradient().getDataType() == ml::train::TensorDim::DataType::FP32
      ? context.getGradient()
      : empty_tensor;

  if (x_grad.empty()) {
    x_grad = context.getGradient().clone(ml::train::TensorDim::DataType::FP32);
  }

  context.applyLossScale(x_grad);

  auto &beta1 = std::get<PropsB1>(adam_props).get();
  auto &beta2 = std::get<PropsB2>(adam_props).get();
  auto &epsilon = std::get<PropsEpsilon>(adam_props).get();
  auto &weight_decay = std::get<PropsWeightDecayW>(adam_props).get();

  Tensor &wm = context.getOptimizerVariable(AdamParams::wm);
  Tensor &wv = context.getOptimizerVariable(AdamParams::wv);

  wm.multiply_i(beta1);
  wm.add_i(x_grad, 1.0f - beta1);

  wv.multiply_i(beta2);
  wv.add_i(x_grad.multiply(x_grad), 1.0f - beta2);

  // Decoupled weight decay: w = w - lr * wd * w
  if (weight_decay > 0.0) {
    Tensor &w = context.isMixedPrecision() ? context.getWeightFP32()
                                           : context.getWeight();
    w.multiply_i(1.0f - (context.getLearningRate() * weight_decay));
  }

  // Adam update with bias-corrected lr
  double lr_t =
    getUpdatedLearningRate(context.getIteration(), context.getLearningRate());

  std::function<double(double)> sqrtEps = [epsilon](double f) {
    return 1.0 / (sqrtDouble(f) + epsilon);
  };
  x_grad = wv.apply<float>(sqrtEps, x_grad);
  x_grad.multiply_i(wm);
  context.applyGradient(lr_t, x_grad);
}

} // namespace nntrainer
