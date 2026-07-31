// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_lora_backward_gradcheck.cpp
 * @date   31 July 2026
 * @brief  Finite-difference gradient checks for the calcDerivative() and
 *         calcGradient() implementations added to support LoRA training on
 *         the Qwen3 CausalLM layer stack.
 * @bug    No known bugs except for NYI items
 */

#include <layer_context.h>
#include <var_grad.h>
#include <weight.h>

#include <reshaped_rms_norm.h>
#include <rms_norm.h>

#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <vector>

namespace {

std::vector<nntrainer::Weight *>
makeWeightView(std::vector<nntrainer::Weight> &weights) {
  std::vector<nntrainer::Weight *> view;
  view.reserve(weights.size());
  for (auto &weight : weights)
    view.push_back(&weight);
  return view;
}

std::vector<nntrainer::Var_Grad *>
makeVarGradView(std::vector<nntrainer::Var_Grad> &vars) {
  std::vector<nntrainer::Var_Grad *> view;
  view.reserve(vars.size());
  for (auto &var : vars)
    view.push_back(&var);
  return view;
}

nntrainer::RunLayerContext
makeRunContext(const std::string &name, std::vector<nntrainer::Weight> &weights,
              std::vector<nntrainer::Var_Grad> &inputs,
              std::vector<nntrainer::Var_Grad> &outputs,
              std::vector<nntrainer::Var_Grad> &tensors) {
  return nntrainer::RunLayerContext(
    name, true, 0.0f, false, 1.0f, nullptr, false, makeWeightView(weights),
    makeVarGradView(inputs), makeVarGradView(outputs),
    makeVarGradView(tensors));
}

/**
 * @brief Central-difference numerical gradient check: perturbs each element
 *        of `x`, re-runs `forward`, and compares (dL/dx)_i against
 *        `analytic[i]`, where L(x) := dot(forward(x), dy) so that the exact
 *        gradient of L w.r.t. x is what a correct backward pass should
 *        produce for the fixed incoming derivative `dy`.
 */
void checkGradient(float *x, const float *analytic, const float *dy,
                   size_t n, size_t out_n,
                   const std::function<const float *()> &forward,
                   float eps = 1e-3f, float tol = 5e-2f) {
  auto loss = [&](const float *out) {
    double l = 0.0;
    for (size_t j = 0; j < out_n; ++j)
      l += static_cast<double>(out[j]) * static_cast<double>(dy[j]);
    return l;
  };

  for (size_t i = 0; i < n; ++i) {
    float orig = x[i];

    x[i] = orig + eps;
    double lp = loss(forward());

    x[i] = orig - eps;
    double lm = loss(forward());

    x[i] = orig;

    double numeric = (lp - lm) / (2.0 * eps);
    EXPECT_NEAR(analytic[i], numeric, tol)
      << "gradient mismatch at index " << i << " (analytic=" << analytic[i]
      << ", numeric=" << numeric << ")";
  }
}

} // namespace

TEST(CausalLmLoraBackward, RmsNormCalcDerivativeMatchesFiniteDifference) {
  const unsigned int height = 2, width = 4;

  causallm::RMSNormLayer layer;
  nntrainer::InitLayerContext init_context(
    {nntrainer::TensorDim({1, 1, height, width})}, {true}, false,
    "rms_norm_gradcheck", "", 0.0f, {"NCHW", "FP32", "FP32"});
  ASSERT_NO_THROW(layer.finalize(init_context));

  std::vector<nntrainer::Weight> weights;
  std::vector<nntrainer::Var_Grad> inputs, outputs, tensors;

  nntrainer::Tensor gamma_tensor(nntrainer::TensorDim({1, 1, 1, width}), true);
  float gamma_vals[width] = {1.0f, 2.0f, 0.5f, 1.5f};
  std::copy(gamma_vals, gamma_vals + width, gamma_tensor.getData<float>());
  weights.emplace_back(gamma_tensor, nntrainer::Tensor(), nntrainer::Tensor(),
                       "gamma");

  inputs.emplace_back(init_context.getInputDimensions()[0],
                      nntrainer::Initializer::NONE, true, true, "input");
  outputs.emplace_back(init_context.getOutSpecs()[0].variable_spec.dim,
                       nntrainer::Initializer::NONE, true, true, "output");

  float x[height * width] = {0.2f, -0.1f, 0.4f, 0.3f, -0.3f, 0.5f, -0.2f, 0.1f};
  std::copy(x, x + height * width, inputs[0].getVariableRef().getData<float>());

  auto run_context = makeRunContext("rms_norm_gradcheck", weights, inputs,
                                    outputs, tensors);

  float dy[height * width] = {0.5f, -0.3f, 0.2f, 0.1f, 0.4f, -0.2f, 0.3f, -0.1f};
  std::copy(dy, dy + height * width,
           run_context.getOutputGradUnsafe(0).getData<float>());

  layer.forwarding(run_context, true);
  layer.calcDerivative(run_context);

  std::vector<float> analytic(height * width);
  std::copy(run_context.getOutgoingDerivative(0).getData<float>(),
           run_context.getOutgoingDerivative(0).getData<float>() +
             height * width,
           analytic.begin());

  auto forward = [&]() -> const float * {
    layer.forwarding(run_context, true);
    return run_context.getOutput(0).getData<float>();
  };

  checkGradient(inputs[0].getVariableRef().getData<float>(), analytic.data(),
               dy, height * width, height * width, forward);
}

TEST(CausalLmLoraBackward,
    ReshapedRmsNormCalcDerivativeMatchesFiniteDifference) {
  const unsigned int height = 1, width = 8, feature_size = 4;

  causallm::ReshapedRMSNormLayer layer;
  ASSERT_NO_THROW(
    layer.setProperty({"feature_size=" + std::to_string(feature_size)}));
  nntrainer::InitLayerContext init_context(
    {nntrainer::TensorDim({1, 1, height, width})}, {true}, false,
    "reshaped_rms_norm_gradcheck", "", 0.0f, {"NCHW", "FP32", "FP32"});
  ASSERT_NO_THROW(layer.finalize(init_context));

  std::vector<nntrainer::Weight> weights;
  std::vector<nntrainer::Var_Grad> inputs, outputs, tensors;

  nntrainer::Tensor gamma_tensor(nntrainer::TensorDim({1, 1, 1, feature_size}),
                                true);
  float gamma_vals[feature_size] = {1.0f, 2.0f, 0.5f, 1.5f};
  std::copy(gamma_vals, gamma_vals + feature_size,
           gamma_tensor.getData<float>());
  weights.emplace_back(gamma_tensor, nntrainer::Tensor(), nntrainer::Tensor(),
                       "gamma");

  inputs.emplace_back(init_context.getInputDimensions()[0],
                      nntrainer::Initializer::NONE, true, true, "input");
  outputs.emplace_back(init_context.getOutSpecs()[0].variable_spec.dim,
                       nntrainer::Initializer::NONE, true, true, "output");

  float x[height * width] = {0.2f, -0.1f, 0.4f, 0.3f, -0.3f, 0.5f, -0.2f, 0.1f};
  std::copy(x, x + height * width, inputs[0].getVariableRef().getData<float>());

  auto run_context = makeRunContext("reshaped_rms_norm_gradcheck", weights,
                                    inputs, outputs, tensors);

  float dy[height * width] = {0.5f, -0.3f, 0.2f, 0.1f, 0.4f, -0.2f, 0.3f, -0.1f};
  std::copy(dy, dy + height * width,
           run_context.getOutputGradUnsafe(0).getData<float>());

  layer.forwarding(run_context, true);
  layer.calcDerivative(run_context);

  std::vector<float> analytic(height * width);
  std::copy(run_context.getOutgoingDerivative(0).getData<float>(),
           run_context.getOutgoingDerivative(0).getData<float>() +
             height * width,
           analytic.begin());

  auto forward = [&]() -> const float * {
    layer.forwarding(run_context, true);
    return run_context.getOutput(0).getData<float>();
  };

  checkGradient(inputs[0].getVariableRef().getData<float>(), analytic.data(),
               dy, height * width, height * width, forward);
}

TEST(CausalLmLoraBackward, RmsNormCalcGradientMatchesFiniteDifference) {
  const unsigned int height = 3, width = 4;

  causallm::RMSNormLayer layer;
  nntrainer::InitLayerContext init_context(
    {nntrainer::TensorDim({1, 1, height, width})}, {true}, false,
    "rms_norm_gradw", "", 0.0f, {"NCHW", "FP32", "FP32"});
  ASSERT_NO_THROW(layer.finalize(init_context));

  std::vector<nntrainer::Weight> weights;
  std::vector<nntrainer::Var_Grad> inputs, outputs, tensors;

  // Weight needs a real gradient buffer for calcGradient to write into.
  nntrainer::Tensor gamma_v(nntrainer::TensorDim({1, 1, 1, width}), true);
  nntrainer::Tensor gamma_g(nntrainer::TensorDim({1, 1, 1, width}), true);
  const float gamma_vals[width] = {1.0f, 2.0f, 0.5f, 1.5f};
  std::copy(gamma_vals, gamma_vals + width, gamma_v.getData<float>());
  gamma_g.setZero();
  weights.emplace_back(gamma_v, gamma_g, nntrainer::Tensor(), "gamma");

  inputs.emplace_back(init_context.getInputDimensions()[0],
                      nntrainer::Initializer::NONE, true, true, "input");
  outputs.emplace_back(init_context.getOutSpecs()[0].variable_spec.dim,
                       nntrainer::Initializer::NONE, true, true, "output");

  const unsigned int n = height * width;
  std::vector<float> x(n), dy(n);
  for (unsigned int i = 0; i < n; ++i) {
    x[i] = std::sin(0.6f * i + 0.4f) * 0.7f;
    dy[i] = std::cos(0.8f * i + 0.1f) * 0.5f;
  }
  std::copy(x.begin(), x.end(), inputs[0].getVariableRef().getData<float>());

  auto rc = makeRunContext("rms_norm_gradw", weights, inputs, outputs, tensors);
  std::copy(dy.begin(), dy.end(), rc.getOutputGradUnsafe(0).getData<float>());

  layer.forwarding(rc, true);
  layer.calcGradient(rc);

  std::vector<float> analytic(rc.getWeightGrad(0).getData<float>(),
                              rc.getWeightGrad(0).getData<float>() + width);

  auto forward = [&]() -> const float * {
    layer.forwarding(rc, true);
    return rc.getOutput(0).getData<float>();
  };
  checkGradient(rc.getWeight(0).getData<float>(), analytic.data(), dy.data(),
                width, n, forward);
}

/**
 * @brief Finite-difference check of dL/dgamma for ReshapedRMSNorm, where
 *        every feature_size-wide chunk contributes to the same gamma.
 */
TEST(CausalLmLoraBackward,
     ReshapedRmsNormCalcGradientMatchesFiniteDifference) {
  const unsigned int height = 2, width = 8, feature_size = 4;

  causallm::ReshapedRMSNormLayer layer;
  ASSERT_NO_THROW(
    layer.setProperty({"feature_size=" + std::to_string(feature_size)}));
  nntrainer::InitLayerContext init_context(
    {nntrainer::TensorDim({1, 1, height, width})}, {true}, false,
    "reshaped_gradw", "", 0.0f, {"NCHW", "FP32", "FP32"});
  ASSERT_NO_THROW(layer.finalize(init_context));

  std::vector<nntrainer::Weight> weights;
  std::vector<nntrainer::Var_Grad> inputs, outputs, tensors;

  nntrainer::Tensor gamma_v(nntrainer::TensorDim({1, 1, 1, feature_size}),
                            true);
  nntrainer::Tensor gamma_g(nntrainer::TensorDim({1, 1, 1, feature_size}),
                            true);
  const float gamma_vals[feature_size] = {1.0f, 2.0f, 0.5f, 1.5f};
  std::copy(gamma_vals, gamma_vals + feature_size, gamma_v.getData<float>());
  gamma_g.setZero();
  weights.emplace_back(gamma_v, gamma_g, nntrainer::Tensor(), "gamma");

  inputs.emplace_back(init_context.getInputDimensions()[0],
                      nntrainer::Initializer::NONE, true, true, "input");
  outputs.emplace_back(init_context.getOutSpecs()[0].variable_spec.dim,
                       nntrainer::Initializer::NONE, true, true, "output");

  const unsigned int n = height * width;
  std::vector<float> x(n), dy(n);
  for (unsigned int i = 0; i < n; ++i) {
    x[i] = std::sin(0.5f * i + 0.2f) * 0.6f;
    dy[i] = std::cos(0.7f * i + 0.3f) * 0.4f;
  }
  std::copy(x.begin(), x.end(), inputs[0].getVariableRef().getData<float>());

  auto rc = makeRunContext("reshaped_gradw", weights, inputs, outputs, tensors);
  std::copy(dy.begin(), dy.end(), rc.getOutputGradUnsafe(0).getData<float>());

  layer.forwarding(rc, true);
  layer.calcGradient(rc);

  std::vector<float> analytic(rc.getWeightGrad(0).getData<float>(),
                              rc.getWeightGrad(0).getData<float>() +
                                feature_size);

  auto forward = [&]() -> const float * {
    layer.forwarding(rc, true);
    return rc.getOutput(0).getData<float>();
  };
  checkGradient(rc.getWeight(0).getData<float>(), analytic.data(), dy.data(),
                feature_size, n, forward);
}
