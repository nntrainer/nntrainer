// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   adam.cpp
 * @date   6 October 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is the Adam optimizer.
 */

#include <cmath>
#include <fstream>

#include <adam.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

namespace nntrainer {

Adam::Adam() : adam_props(PropsB1(), PropsB2(), PropsEpsilon(), TorchRef()) {
  /** default properties */
  auto &[b1, b2, eps, torch_ref] = adam_props;
  b1.set(0.9f);
  b2.set(0.999f);
  eps.set(1.0e-7f);
  torch_ref.set(false);
}

Adam::~Adam() {}

enum AdamParams { wm, wv };

std::vector<TensorDim> Adam::getOptimizerVariableDim(const TensorDim &dim) {
  /**
   * @note We assume the optimizer parameters should be full precsion to
   * maintain the accuracy even in mixed precision training.
   */
  TensorDim wm_dim(dim);
  TensorDim wv_dim(dim);
  wm_dim.setDataType(ml::train::TensorDim::DataType::FP32);
  wv_dim.setDataType(ml::train::TensorDim::DataType::FP32);
  return {wm_dim, wv_dim};
}

void Adam::exportTo(Exporter &exporter,
                    const ml::train::ExportMethods &method) const {
  exporter.saveResult(adam_props, method, this);
  Optimizer::exportTo(exporter, method);
}

void Adam::setProperty(const std::vector<std::string> &values) {
  auto left = loadProperties(values, adam_props);
  Optimizer::setProperty(left);
}

double Adam::getUpdatedLearningRate(unsigned int iteration, double lr) const {
  auto &beta1 = std::get<PropsB1>(adam_props).get();
  auto &beta2 = std::get<PropsB2>(adam_props).get();

  /// @note This change suppresses C4244 warnings due to precision loss.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244)
#endif
  std::function<float(double)> biasCorrection = [&](float f) {
    return 1.0f - pow(f, iteration + 1);
  };

  lr *= sqrt(biasCorrection(beta2)) / biasCorrection(beta1);
#ifdef _MSC_VER
#pragma warning(pop)
#endif

  return lr;
}

void Adam::applyGradient(RunOptimizerContext &context) {
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
  auto &torch_ref = std::get<TorchRef>(adam_props).get();

  // This is implementation of adam from original paper.
  // This is not deleted intentionally.
  unsigned int iteration = context.getIteration();
  float biasCorrection1 = 1.0 - pow((double)beta1, iteration + 1);
  float biasCorrection2 = 1.0 - pow((double)beta2, iteration + 1);
  Tensor &wm = context.getOptimizerVariable(AdamParams::wm);
  Tensor &wv = context.getOptimizerVariable(AdamParams::wv);

  wm.multiply_i(beta1);
  wm.add_i(x_grad, 1.0f - beta1);

  wv.multiply_i(beta2);
  wv.add_i(x_grad.multiply(x_grad), 1.0f - beta2);

  if (torch_ref) {
    Tensor denom = wv.apply<float>(sqrtFloat<float>);
    denom.divide_i(sqrtFloat(biasCorrection2));
    denom.add_i(epsilon);
    wm.divide(denom, x_grad);

    context.applyGradient(context.getLearningRate() / biasCorrection1, x_grad);

  } else {
    /// @note This change suppresses C4244 warnings due to precision loss.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244)
#endif
    std::function<double(double)> sqrtEps = [epsilon](double f) {
      return 1.0 / (sqrtDouble(f) + epsilon);
    };

    x_grad = wv.apply<float>(sqrtEps, x_grad);
#ifdef _MSC_VER
#pragma warning(pop)
#endif
    x_grad.multiply_i(wm);
    context.applyGradient(
      getUpdatedLearningRate(context.getIteration(), context.getLearningRate()),
      x_grad);
  }
}

} // namespace nntrainer
