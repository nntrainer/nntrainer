// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Minseo Kim <ms05251@naver.com>
 *
 * @file   sophia.cpp
 * @date   16 December 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Minseo Kim <ms05251@naver.com>
 * @author Jeonghun Park <top231902@naver.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is the Sophia Optimizer.
 */

#include <algorithm>
#include <cmath>
#include <node_exporter.h>
#include <sophia.h>

namespace nntrainer {

Sophia::Sophia() :
  sophia_props(PropsB1(), PropsB2(), PropsEpsilon(), PropsRho(),
               PropsWeightDecaySophia(), PropsK()) {
  auto &[beta1, beta2, eps, rho, wd, kprop] = sophia_props;
  beta1.set(0.9f);
  beta2.set(0.99f);
  eps.set(1.0e-12f);
  rho.set(0.03f);
  wd.set(0.0f);
  kprop.set(10u);
}

Sophia::~Sophia() = default;

enum SophiaParams { m, h };

std::vector<TensorDim> Sophia::getOptimizerVariableDim(const TensorDim &dim) {
  /**
   * m: first-moment (momentum), h: hessian diagonal estimate (EMA)
   * Keep in FP32 for stability, even under mixed precision.
   */
  TensorDim m_dim(dim);
  TensorDim h_dim(dim);
  m_dim.setDataType(ml::train::TensorDim::DataType::FP32);
  h_dim.setDataType(ml::train::TensorDim::DataType::FP32);
  return {m_dim, h_dim};
}

void Sophia::exportTo(Exporter &exporter,
                      const ml::train::ExportMethods &method) const {
  exporter.saveResult(sophia_props, method, this);
  Optimizer::exportTo(exporter, method);
}

void Sophia::setProperty(const std::vector<std::string> &values) {
  auto left = loadProperties(values, sophia_props);
  Optimizer::setProperty(left);
}

void Sophia::applyGradient(RunOptimizerContext &context) {
  // Prepare gradient in FP32 and apply loss scaling if any
  Tensor empty_tensor;
  Tensor &x_grad =
    context.getGradient().getDataType() == ml::train::TensorDim::DataType::FP32
      ? context.getGradient()
      : empty_tensor;

  if (x_grad.empty()) {
    x_grad = context.getGradient().clone(ml::train::TensorDim::DataType::FP32);
  }

  context.applyLossScale(x_grad);

  // State tensors
  Tensor &m_t = context.getOptimizerVariable(SophiaParams::m);
  Tensor &h_t = context.getOptimizerVariable(SophiaParams::h);

  // Hyper-parameters
  const float beta1 = std::get<PropsB1>(sophia_props).get();
  const float beta2 = std::get<PropsB2>(sophia_props).get();
  const float eps = std::get<PropsEpsilon>(sophia_props).get();
  const float rho = std::get<PropsRho>(sophia_props).get();
  const float wd = std::get<PropsWeightDecaySophia>(sophia_props).get();
  const unsigned int iter = context.getIteration();
  const unsigned int K = std::get<PropsK>(sophia_props).get();

  /**
   * 1) Hessian EMA update (Approximated GNB)
   * We use empirical Fisher (grad^2) to approximate the diagonal Hessian.
   * Consistent with Algorithm 2, we scale the gradient square by batch size B.
   * h_t = beta2 * h_{t-k} + (1 - beta2) * (B * g \odot g)
   */
  if (K > 0 && ((iter + 1) % K) == 0) {
    unsigned int B = context.getBatchSize();

    h_t.multiply_i(beta2);

    Tensor grad_sq = x_grad.multiply(x_grad);
    if (B > 1u) {
      grad_sq.multiply_i(static_cast<float>(B));
    }

    h_t.add_i(grad_sq, 1.0f - beta2);
  }

  /**
   * 2) First moment (momentum) update
   * m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
   */
  m_t.multiply_i(beta1);
  m_t.add_i(x_grad, 1.0f - beta1);

  /**
   * 3) Preconditioned Gradient Calculation
   * ratio = m_t / (h_t + eps)
   */
  Tensor denom = h_t.clone();
  denom.add_i(eps);

  Tensor ratio = m_t.divide(denom);

  /**
   * 4) Element-wise Clipping [-rho, rho]
   * Paper: theta = theta - lr * clip( m_t / (rho * h_t), 1)
   * Code : theta = theta - (lr / rho) * clip( m_t / h_t, rho)
   */
  std::function<float(float)> clamp_func = [rho](float val) {
    if (val > rho)
      return rho;
    if (val < -rho)
      return -rho;
    return val;
  };
  ratio.apply_i<float>(clamp_func);

  /**
   * 5) Decoupled Weight Decay
   * We multiply 'rho' here because the final update is scaled by (1/rho).
   * theta = theta - lr * weight_decay * theta
   */
  if (wd > 0.0f) {
    Tensor &decay_src = context.isMixedPrecision() ? context.getWeightFP32()
                                                   : context.getWeight();
    ratio.add_i(decay_src, wd * rho);
  }

  /**
   * 6) Final Update
   * Apply update with the scaled learning rate (lr / rho).
   */
  const float lr = context.getLearningRate();
  const float scaled_lr = lr / rho;
  context.applyGradient(scaled_lr, ratio);
}

} // namespace nntrainer
