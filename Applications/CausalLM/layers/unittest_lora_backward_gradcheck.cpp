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

#include <lm_head.h>
#include <mha_core.h>
#include <reshaped_rms_norm.h>
#include <rms_norm.h>
#include <swiglu.h>
#include <tie_word_embedding.h>

#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <limits>
#include <memory>
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

TEST(CausalLmLoraBackward, SwiGLUCalcDerivativeMatchesFiniteDifference) {
  const unsigned int height = 2, width = 3;

  causallm::SwiGLULayer layer;
  nntrainer::InitLayerContext init_context(
    {nntrainer::TensorDim({1, 1, height, width}),
     nntrainer::TensorDim({1, 1, height, width})},
    {true}, false, "swiglu_gradcheck", "", 0.0f, {"NCHW", "FP32", "FP32"});
  ASSERT_NO_THROW(layer.finalize(init_context));

  std::vector<nntrainer::Weight> weights;
  std::vector<nntrainer::Var_Grad> inputs, outputs, tensors;

  inputs.emplace_back(init_context.getInputDimensions()[0],
                      nntrainer::Initializer::NONE, true, true, "gate");
  inputs.emplace_back(init_context.getInputDimensions()[1],
                      nntrainer::Initializer::NONE, true, true, "up");
  outputs.emplace_back(init_context.getOutSpecs()[0].variable_spec.dim,
                       nntrainer::Initializer::NONE, true, true, "output");

  float gate[height * width] = {0.2f, -0.5f, 0.8f, -0.3f, 0.1f, 1.2f};
  float up[height * width] = {0.4f, 0.9f, -0.6f, 0.3f, -1.1f, 0.2f};
  std::copy(gate, gate + height * width,
           inputs[0].getVariableRef().getData<float>());
  std::copy(up, up + height * width,
           inputs[1].getVariableRef().getData<float>());

  auto run_context =
    makeRunContext("swiglu_gradcheck", weights, inputs, outputs, tensors);

  float dy[height * width] = {0.5f, -0.3f, 0.2f, 0.1f, 0.4f, -0.2f};
  std::copy(dy, dy + height * width,
           run_context.getOutputGradUnsafe(0).getData<float>());

  layer.forwarding(run_context, true);
  layer.calcDerivative(run_context);

  std::vector<float> analytic_gate(height * width), analytic_up(height * width);
  std::copy(run_context.getOutgoingDerivative(0).getData<float>(),
           run_context.getOutgoingDerivative(0).getData<float>() +
             height * width,
           analytic_gate.begin());
  std::copy(run_context.getOutgoingDerivative(1).getData<float>(),
           run_context.getOutgoingDerivative(1).getData<float>() +
             height * width,
           analytic_up.begin());

  auto forward = [&]() -> const float * {
    layer.forwarding(run_context, true);
    return run_context.getOutput(0).getData<float>();
  };

  checkGradient(inputs[0].getVariableRef().getData<float>(),
               analytic_gate.data(), dy, height * width, height * width,
               forward);
  checkGradient(inputs[1].getVariableRef().getData<float>(),
               analytic_up.data(), dy, height * width, height * width,
               forward);
}

TEST(CausalLmLoraBackward, LmHeadCalcDerivativeMatchesFiniteDifference) {
  const unsigned int height = 3, width = 4, unit = 5;
  causallm::g_lm_head_read_row = 1; // simulate a right-padded sample: the
                                    // last *real* token is row 1, not
                                    // row (height - 1) = 2.

  causallm::LmHeadLayer layer;
  ASSERT_NO_THROW(layer.setProperty({"unit=" + std::to_string(unit)}));
  nntrainer::InitLayerContext init_context(
    {nntrainer::TensorDim({1, 1, height, width})}, {true}, false,
    "lm_head_gradcheck", "", 0.0f, {"NCHW", "FP32", "FP32"});
  ASSERT_NO_THROW(layer.finalize(init_context));

  std::vector<nntrainer::Weight> weights;
  std::vector<nntrainer::Var_Grad> inputs, outputs, tensors;

  nntrainer::Tensor weight_tensor(nntrainer::TensorDim({1, 1, width, unit}),
                                 true);
  float weight_vals[width * unit] = {0.1f,  0.2f,  -0.3f, 0.4f,  0.5f,
                                     -0.2f, 0.3f,  0.1f,  -0.4f, 0.2f,
                                     0.3f,  -0.1f, 0.2f,  0.4f,  -0.3f,
                                     0.1f,  -0.2f, 0.3f,  0.4f,  -0.1f};
  std::copy(weight_vals, weight_vals + width * unit,
           weight_tensor.getData<float>());
  weights.emplace_back(weight_tensor, nntrainer::Tensor(), nntrainer::Tensor(),
                       "weight");

  nntrainer::Tensor bias_tensor(nntrainer::TensorDim({1, 1, 1, unit}), true);
  float bias_vals[unit] = {0.05f, -0.1f, 0.15f, -0.05f, 0.1f};
  std::copy(bias_vals, bias_vals + unit, bias_tensor.getData<float>());
  weights.emplace_back(bias_tensor, nntrainer::Tensor(), nntrainer::Tensor(),
                       "bias");

  inputs.emplace_back(init_context.getInputDimensions()[0],
                      nntrainer::Initializer::NONE, true, true, "input");
  outputs.emplace_back(init_context.getOutSpecs()[0].variable_spec.dim,
                       nntrainer::Initializer::NONE, true, true, "output");

  float x[height * width] = {0.2f,  -0.1f, 0.4f, 0.3f, -0.3f, 0.5f,
                             -0.2f, 0.1f,  0.6f, -0.4f, 0.2f, -0.5f};
  std::copy(x, x + height * width, inputs[0].getVariableRef().getData<float>());

  auto run_context =
    makeRunContext("lm_head_gradcheck", weights, inputs, outputs, tensors);

  float dy[unit] = {0.5f, -0.3f, 0.2f, 0.1f, -0.2f};
  std::copy(dy, dy + unit, run_context.getOutputGradUnsafe(0).getData<float>());

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
               dy, height * width, unit, forward);

  causallm::g_lm_head_read_row = std::numeric_limits<unsigned int>::max();
}

TEST(CausalLmLoraBackward,
    TieWordEmbeddingLmHeadCalcDerivativeMatchesFiniteDifference) {
  const unsigned int height = 3, width = 4, unit = 5;
  causallm::g_tie_embedding_lm_head_read_row =
    1; // simulate a right-padded sample.

  causallm::TieWordEmbedding layer;
  ASSERT_NO_THROW(layer.setProperty({"unit=" + std::to_string(unit)}));
  nntrainer::InitLayerContext init_context(
    {nntrainer::TensorDim({1, 1, height, width})}, {true}, false,
    "tie_word_embedding_lmhead_gradcheck", "", 0.0f, {"NCHW", "FP32", "FP32"});
  ASSERT_NO_THROW(layer.finalize(init_context));

  std::vector<nntrainer::Weight> weights;
  std::vector<nntrainer::Var_Grad> inputs, outputs, tensors;

  // tied weight layout is [vocab=unit, hidden=width] (shared with embedding).
  nntrainer::Tensor weight_tensor(nntrainer::TensorDim({1, 1, unit, width}),
                                 true);
  float weight_vals[unit * width] = {0.1f,  0.2f, -0.3f, 0.4f,  0.5f, -0.2f,
                                     0.3f,  0.1f, -0.4f, 0.2f,  0.3f, -0.1f,
                                     0.2f,  0.4f, -0.3f, 0.1f,  -0.2f, 0.3f,
                                     0.4f, -0.1f};
  std::copy(weight_vals, weight_vals + unit * width,
           weight_tensor.getData<float>());
  weights.emplace_back(weight_tensor, nntrainer::Tensor(), nntrainer::Tensor(),
                       "weight");

  nntrainer::Tensor bias_tensor(nntrainer::TensorDim({1, 1, 1, unit}), true);
  float bias_vals[unit] = {0.05f, -0.1f, 0.15f, -0.05f, 0.1f};
  std::copy(bias_vals, bias_vals + unit, bias_tensor.getData<float>());
  weights.emplace_back(bias_tensor, nntrainer::Tensor(), nntrainer::Tensor(),
                       "bias");

  inputs.emplace_back(init_context.getInputDimensions()[0],
                      nntrainer::Initializer::NONE, true, true, "input");
  outputs.emplace_back(init_context.getOutSpecs()[0].variable_spec.dim,
                       nntrainer::Initializer::NONE, true, true, "output");

  float x[height * width] = {0.2f,  -0.1f, 0.4f, 0.3f, -0.3f, 0.5f,
                             -0.2f, 0.1f,  0.6f, -0.4f, 0.2f, -0.5f};
  std::copy(x, x + height * width, inputs[0].getVariableRef().getData<float>());

  auto run_context = makeRunContext("tie_word_embedding_lmhead_gradcheck",
                                    weights, inputs, outputs, tensors);

  float dy[unit] = {0.5f, -0.3f, 0.2f, 0.1f, -0.2f};
  std::copy(dy, dy + unit, run_context.getOutputGradUnsafe(0).getData<float>());

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
               dy, height * width, unit, forward);

  causallm::g_tie_embedding_lm_head_read_row =
    std::numeric_limits<unsigned int>::max();
}

TEST(CausalLmLoraBackward, MhaCoreCalcDerivativeMatchesFiniteDifference) {
  // 4 Q heads, 2 KV heads (GQA group size 2), head_dim=4, seq_len=3.
  const unsigned int seq_len = 3, num_heads_q = 4, num_heads_kv = 2,
                     head_dim = 4;
  const unsigned int q_width = num_heads_q * head_dim;   // 16
  const unsigned int kv_width = num_heads_kv * head_dim; // 8

  causallm::MHACoreLayer layer;
  ASSERT_NO_THROW(layer.setProperty(
    {"num_heads=" + std::to_string(num_heads_q),
     "num_heads_KV=" + std::to_string(num_heads_kv), "max_timestep=8"}));

  nntrainer::InitLayerContext init_context(
    {nntrainer::TensorDim({1, 1, seq_len, q_width}),
     nntrainer::TensorDim({1, 1, seq_len, kv_width}),
     nntrainer::TensorDim({1, 1, seq_len, kv_width})},
    {true}, false, "mha_core_gradcheck", "", 0.0f, {"NCHW", "FP32", "FP32"});
  ASSERT_NO_THROW(layer.finalize(init_context));

  std::vector<nntrainer::Weight> weights; // no weights (use_sink=false)
  std::vector<nntrainer::Var_Grad> inputs, outputs, tensors;

  for (const auto &dim : init_context.getInputDimensions())
    inputs.emplace_back(dim, nntrainer::Initializer::NONE, true, true, "in");
  outputs.emplace_back(init_context.getOutSpecs()[0].variable_spec.dim,
                       nntrainer::Initializer::NONE, true, true, "output");
  // Build the 5 requested tensors (cache_key, cache_value, train_q_roped,
  // train_k_roped, train_attn_wt) directly from the specs finalize()
  // produced, so their exact dims/dtypes always match what the layer
  // actually requested.
  for (const auto &spec : init_context.getTensorsSpec())
    tensors.emplace_back(spec, true);

  auto run_context =
    makeRunContext("mha_core_gradcheck", weights, inputs, outputs, tensors);

  float *q = inputs[0].getVariableRef().getData<float>();
  float *k = inputs[1].getVariableRef().getData<float>();
  float *v = inputs[2].getVariableRef().getData<float>();
  for (unsigned int i = 0; i < seq_len * q_width; ++i)
    q[i] = std::sin(0.7f * i + 1.0f) * 0.5f;
  for (unsigned int i = 0; i < seq_len * kv_width; ++i)
    k[i] = std::cos(0.5f * i + 0.3f) * 0.5f;
  for (unsigned int i = 0; i < seq_len * kv_width; ++i)
    v[i] = std::sin(0.3f * i + 0.9f) * 0.4f;

  std::vector<float> dy(seq_len * q_width);
  for (unsigned int i = 0; i < dy.size(); ++i)
    dy[i] = std::cos(0.9f * i + 0.2f) * 0.3f;
  std::copy(dy.begin(), dy.end(),
           run_context.getOutputGradUnsafe(0).getData<float>());

  layer.forwarding(run_context, true);
  layer.calcDerivative(run_context);

  std::vector<float> analytic_q(seq_len * q_width),
    analytic_k(seq_len * kv_width), analytic_v(seq_len * kv_width);
  std::copy(run_context.getOutgoingDerivative(0).getData<float>(),
           run_context.getOutgoingDerivative(0).getData<float>() +
             seq_len * q_width,
           analytic_q.begin());
  std::copy(run_context.getOutgoingDerivative(1).getData<float>(),
           run_context.getOutgoingDerivative(1).getData<float>() +
             seq_len * kv_width,
           analytic_k.begin());
  std::copy(run_context.getOutgoingDerivative(2).getData<float>(),
           run_context.getOutgoingDerivative(2).getData<float>() +
             seq_len * kv_width,
           analytic_v.begin());

  auto forward = [&]() -> const float * {
    layer.forwarding(run_context, true);
    return run_context.getOutput(0).getData<float>();
  };

  checkGradient(q, analytic_q.data(), dy.data(), seq_len * q_width,
               seq_len * q_width, forward, 1e-3f, 5e-3f);
  checkGradient(k, analytic_k.data(), dy.data(), seq_len * kv_width,
               seq_len * q_width, forward, 1e-3f, 5e-3f);
  checkGradient(v, analytic_v.data(), dy.data(), seq_len * kv_width,
               seq_len * q_width, forward, 1e-3f, 5e-3f);
}

/**
 * @brief mha_core's training forward (full-sequence, internal-cache) must
 *        agree with the trusted inference prefill path
 *        (incremental_forwarding over [0, seq_len)) for identical Q/K/V,
 *        since both compute the same dense causal GQA attention.
 */
TEST(CausalLmLoraBackward, MhaCoreTrainForwardMatchesInferencePrefill) {
  const unsigned int seq_len = 5, num_heads_q = 4, num_heads_kv = 2,
                     head_dim = 4;
  const unsigned int q_width = num_heads_q * head_dim;
  const unsigned int kv_width = num_heads_kv * head_dim;

  std::vector<float> q(seq_len * q_width), k(seq_len * kv_width),
    v(seq_len * kv_width);
  for (unsigned int i = 0; i < q.size(); ++i)
    q[i] = std::sin(0.7f * i + 1.0f) * 0.5f;
  for (unsigned int i = 0; i < k.size(); ++i)
    k[i] = std::cos(0.5f * i + 0.3f) * 0.5f;
  for (unsigned int i = 0; i < v.size(); ++i)
    v[i] = std::sin(0.3f * i + 0.9f) * 0.4f;

  auto run = [&](ml::train::ExecutionMode mode) -> std::vector<float> {
    causallm::MHACoreLayer layer;
    layer.setProperty({"num_heads=" + std::to_string(num_heads_q),
                       "num_heads_KV=" + std::to_string(num_heads_kv),
                       "max_timestep=" + std::to_string(seq_len)});
    nntrainer::InitLayerContext init_context(
      {nntrainer::TensorDim({1, 1, seq_len, q_width}),
       nntrainer::TensorDim({1, 1, seq_len, kv_width}),
       nntrainer::TensorDim({1, 1, seq_len, kv_width})},
      {true}, false, "mha_cmp", "", 0.0f, {"NCHW", "FP32", "FP32"}, 1.0f, mode);
    layer.finalize(init_context);

    std::vector<nntrainer::Weight> weights;
    std::vector<nntrainer::Var_Grad> inputs, outputs, tensors;
    for (const auto &d : init_context.getInputDimensions())
      inputs.emplace_back(d, nntrainer::Initializer::NONE, true, true, "in");
    outputs.emplace_back(init_context.getOutSpecs()[0].variable_spec.dim,
                         nntrainer::Initializer::NONE, true, true, "out");
    for (const auto &spec : init_context.getTensorsSpec())
      tensors.emplace_back(spec, true);

    std::copy(q.begin(), q.end(), inputs[0].getVariableRef().getData<float>());
    std::copy(k.begin(), k.end(), inputs[1].getVariableRef().getData<float>());
    std::copy(v.begin(), v.end(), inputs[2].getVariableRef().getData<float>());

    auto rc = makeRunContext("mha_cmp", weights, inputs, outputs, tensors);
    rc.getOutput(0).setZero();
    if (mode == ml::train::ExecutionMode::TRAIN)
      layer.forwarding(rc, true);
    else
      layer.incremental_forwarding(rc, 0, seq_len, false);

    const float *o = rc.getOutput(0).getData<float>();
    return std::vector<float>(o, o + seq_len * q_width);
  };

  auto inference_out = run(ml::train::ExecutionMode::INFERENCE);
  auto training_out = run(ml::train::ExecutionMode::TRAIN);

  ASSERT_EQ(inference_out.size(), training_out.size());
  // Tolerance is set by the inference path storing its KV cache in 16-bit
  // (FP16/UINT16) while the training path keeps Q/K/V in FP32, so the two
  // agree only to FP16 precision.
  for (size_t i = 0; i < inference_out.size(); ++i)
    EXPECT_NEAR(inference_out[i], training_out[i], 2e-3)
      << "attention output mismatch at " << i << " (row " << i / q_width
      << ", head " << (i % q_width) / head_dim << ")";
}

/**
 * @brief Regression test: nntrainer aliases each outgoing derivative onto its
 *        input's own buffer, so a calcDerivative that writes one output
 *        before reading every input it needs will silently consume its own
 *        output. SwiGLU is the case that matters (d_up aliases up, which is
 *        still needed to form d_gate).
 *
 * @note The other gradient checks in this file build each Var_Grad with
 *       independent variable/gradient storage, which does NOT reproduce the
 *       aliasing the real graph uses — hence this dedicated test.
 */
TEST(CausalLmLoraBackward, SwiGLUCalcDerivativeIsSafeWhenGradAliasesInput) {
  const unsigned int height = 2, width = 3, n = height * width;

  const std::vector<float> gate0 = {0.2f, -0.5f, 0.8f, -0.3f, 0.1f, 1.2f};
  const std::vector<float> up0 = {0.4f, 0.9f, -0.6f, 0.3f, -1.1f, 0.2f};
  const std::vector<float> dy0 = {0.5f, -0.3f, 0.2f, 0.1f, 0.4f, -0.2f};

  auto build = [&](bool alias) {
    causallm::SwiGLULayer layer;
    nntrainer::InitLayerContext init_context(
      {nntrainer::TensorDim({1, 1, height, width}),
       nntrainer::TensorDim({1, 1, height, width})},
      {true}, false, "swiglu_alias", "", 0.0f, {"NCHW", "FP32", "FP32"});
    layer.finalize(init_context);

    // Storage kept alive by the caller via the returned vectors.
    auto gate_t = std::make_shared<nntrainer::Tensor>(
      init_context.getInputDimensions()[0], true);
    auto up_t = std::make_shared<nntrainer::Tensor>(
      init_context.getInputDimensions()[1], true);
    auto gate_g = alias ? gate_t
                        : std::make_shared<nntrainer::Tensor>(
                            init_context.getInputDimensions()[0], true);
    auto up_g = alias ? up_t
                      : std::make_shared<nntrainer::Tensor>(
                          init_context.getInputDimensions()[1], true);
    std::copy(gate0.begin(), gate0.end(), gate_t->getData<float>());
    std::copy(up0.begin(), up0.end(), up_t->getData<float>());

    std::vector<nntrainer::Weight> weights;
    std::vector<nntrainer::Var_Grad> inputs, outputs, tensors;
    inputs.emplace_back(gate_t.get(), gate_g.get(), false);
    inputs.emplace_back(up_t.get(), up_g.get(), false);
    outputs.emplace_back(init_context.getOutSpecs()[0].variable_spec.dim,
                         nntrainer::Initializer::NONE, true, true, "out");

    auto rc = makeRunContext("swiglu_alias", weights, inputs, outputs, tensors);
    std::copy(dy0.begin(), dy0.end(),
              rc.getOutputGradUnsafe(0).getData<float>());

    layer.forwarding(rc, true);
    layer.calcDerivative(rc);

    std::vector<float> dg(gate_g->getData<float>(),
                          gate_g->getData<float>() + n);
    std::vector<float> du(up_g->getData<float>(), up_g->getData<float>() + n);
    return std::make_pair(dg, du);
  };

  auto separate = build(false);
  auto aliased = build(true);

  for (unsigned int i = 0; i < n; ++i) {
    EXPECT_NEAR(separate.first[i], aliased.first[i], 1e-6)
      << "d_gate differs under aliasing at " << i;
    EXPECT_NEAR(separate.second[i], aliased.second[i], 1e-6)
      << "d_up differs under aliasing at " << i;
  }
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

TEST(CausalLmLoraBackward, LmHeadCalcGradientMatchesFiniteDifference) {
  const unsigned int height = 3, width = 4, unit = 5;
  causallm::g_lm_head_read_row = 1; // right-padded: last real token is row 1

  causallm::LmHeadLayer layer;
  ASSERT_NO_THROW(layer.setProperty(
    {"unit=" + std::to_string(unit), "disable_bias=true"}));
  nntrainer::InitLayerContext init_context(
    {nntrainer::TensorDim({1, 1, height, width})}, {true}, false,
    "lm_head_gradw", "", 0.0f, {"NCHW", "FP32", "FP32"});
  ASSERT_NO_THROW(layer.finalize(init_context));

  std::vector<nntrainer::Weight> weights;
  std::vector<nntrainer::Var_Grad> inputs, outputs, tensors;

  nntrainer::Tensor w_v(nntrainer::TensorDim({1, 1, width, unit}), true);
  nntrainer::Tensor w_g(nntrainer::TensorDim({1, 1, width, unit}), true);
  for (unsigned int i = 0; i < width * unit; ++i)
    w_v.getData<float>()[i] = std::sin(0.4f * i + 0.7f) * 0.3f;
  w_g.setZero();
  weights.emplace_back(w_v, w_g, nntrainer::Tensor(), "weight");

  inputs.emplace_back(init_context.getInputDimensions()[0],
                      nntrainer::Initializer::NONE, true, true, "input");
  outputs.emplace_back(init_context.getOutSpecs()[0].variable_spec.dim,
                       nntrainer::Initializer::NONE, true, true, "output");

  for (unsigned int i = 0; i < height * width; ++i)
    inputs[0].getVariableRef().getData<float>()[i] =
      std::sin(0.9f * i + 0.2f) * 0.5f;

  auto rc = makeRunContext("lm_head_gradw", weights, inputs, outputs, tensors);
  std::vector<float> dy(unit);
  for (unsigned int j = 0; j < unit; ++j)
    dy[j] = std::cos(1.1f * j + 0.5f) * 0.6f;
  std::copy(dy.begin(), dy.end(), rc.getOutputGradUnsafe(0).getData<float>());

  layer.forwarding(rc, true);
  layer.calcGradient(rc);

  std::vector<float> analytic(rc.getWeightGrad(0).getData<float>(),
                              rc.getWeightGrad(0).getData<float>() +
                                width * unit);

  auto forward = [&]() -> const float * {
    layer.forwarding(rc, true);
    return rc.getOutput(0).getData<float>();
  };
  checkGradient(rc.getWeight(0).getData<float>(), analytic.data(), dy.data(),
                width * unit, unit, forward);

  causallm::g_lm_head_read_row = std::numeric_limits<unsigned int>::max();
}

TEST(CausalLmLoraBackward,
     TieWordEmbeddingLmHeadCalcGradientMatchesFiniteDifference) {
  const unsigned int height = 3, hidden = 4, vocab = 5;
  causallm::g_tie_embedding_lm_head_read_row = 1;

  causallm::TieWordEmbedding layer;
  ASSERT_NO_THROW(layer.setProperty(
    {"unit=" + std::to_string(vocab), "disable_bias=true"}));
  nntrainer::InitLayerContext init_context(
    {nntrainer::TensorDim({1, 1, height, hidden})}, {true}, false,
    "twe_gradw", "", 0.0f, {"NCHW", "FP32", "FP32"});
  ASSERT_NO_THROW(layer.finalize(init_context));

  std::vector<nntrainer::Weight> weights;
  std::vector<nntrainer::Var_Grad> inputs, outputs, tensors;

  nntrainer::Tensor w_v(nntrainer::TensorDim({1, 1, vocab, hidden}), true);
  nntrainer::Tensor w_g(nntrainer::TensorDim({1, 1, vocab, hidden}), true);
  for (unsigned int i = 0; i < vocab * hidden; ++i)
    w_v.getData<float>()[i] = std::sin(0.35f * i + 0.9f) * 0.3f;
  w_g.setZero();
  weights.emplace_back(w_v, w_g, nntrainer::Tensor(), "weight");

  inputs.emplace_back(init_context.getInputDimensions()[0],
                      nntrainer::Initializer::NONE, true, true, "input");
  outputs.emplace_back(init_context.getOutSpecs()[0].variable_spec.dim,
                       nntrainer::Initializer::NONE, true, true, "output");

  for (unsigned int i = 0; i < height * hidden; ++i)
    inputs[0].getVariableRef().getData<float>()[i] =
      std::sin(0.75f * i + 0.15f) * 0.5f;

  auto rc = makeRunContext("twe_gradw", weights, inputs, outputs, tensors);
  std::vector<float> dy(vocab);
  for (unsigned int v = 0; v < vocab; ++v)
    dy[v] = std::cos(0.95f * v + 0.25f) * 0.6f;
  std::copy(dy.begin(), dy.end(), rc.getOutputGradUnsafe(0).getData<float>());

  layer.forwarding(rc, true);
  layer.calcGradient(rc);

  std::vector<float> analytic(rc.getWeightGrad(0).getData<float>(),
                              rc.getWeightGrad(0).getData<float>() +
                                vocab * hidden);

  auto forward = [&]() -> const float * {
    layer.forwarding(rc, true);
    return rc.getOutput(0).getData<float>();
  };
  checkGradient(rc.getWeight(0).getData<float>(), analytic.data(), dy.data(),
                vocab * hidden, vocab, forward);

  causallm::g_tie_embedding_lm_head_read_row =
    std::numeric_limits<unsigned int>::max();
}

TEST(CausalLmLoraBackward, TieWordEmbeddingEmbeddingModeCalcGradientScatters) {
  const unsigned int seq_len = 4, hidden = 3, vocab = 6;
  const float scale = 2.0f;

  causallm::TieWordEmbedding layer;
  ASSERT_NO_THROW(layer.setProperty({"in_dim=" + std::to_string(vocab),
                                     "out_dim=" + std::to_string(hidden),
                                     "scale=" + std::to_string(scale)}));
  nntrainer::InitLayerContext init_context(
    {nntrainer::TensorDim({1, 1, 1, seq_len})}, {true}, false, "twe_emb_gradw",
    "", 0.0f, {"NCHW", "FP32", "FP32"});
  ASSERT_NO_THROW(layer.finalize(init_context));

  std::vector<nntrainer::Weight> weights;
  std::vector<nntrainer::Var_Grad> inputs, outputs, tensors;

  nntrainer::Tensor w_v(nntrainer::TensorDim({1, 1, vocab, hidden}), true);
  nntrainer::Tensor w_g(nntrainer::TensorDim({1, 1, vocab, hidden}), true);
  w_v.setZero();
  w_g.setZero();
  weights.emplace_back(w_v, w_g, nntrainer::Tensor(), "Embedding");

  inputs.emplace_back(init_context.getInputDimensions()[0],
                      nntrainer::Initializer::NONE, true, true, "input");
  outputs.emplace_back(init_context.getOutSpecs()[0].variable_spec.dim,
                       nntrainer::Initializer::NONE, true, true, "output");

  // ids: 3, 1, 3, 0  -> row 3 receives positions 0 and 2 summed; row 0 gets
  // the trailing pad position; rows 2,4,5 stay zero.
  const float ids[seq_len] = {3.0f, 1.0f, 3.0f, 0.0f};
  std::copy(ids, ids + seq_len, inputs[0].getVariableRef().getData<float>());

  auto rc = makeRunContext("twe_emb_gradw", weights, inputs, outputs, tensors);

  std::vector<float> dy(seq_len * hidden);
  for (unsigned int i = 0; i < dy.size(); ++i)
    dy[i] = 0.1f * static_cast<float>(i + 1);
  std::copy(dy.begin(), dy.end(), rc.getOutputGradUnsafe(0).getData<float>());

  layer.calcGradient(rc);

  const float *dw = rc.getWeightGrad(0).getData<float>();
  for (unsigned int h = 0; h < hidden; ++h) {
    // row 3 <- positions 0 and 2
    EXPECT_NEAR(dw[3 * hidden + h],
                (dy[0 * hidden + h] + dy[2 * hidden + h]) * scale, 1e-6)
      << "row 3 (repeated id) must accumulate both positions, h=" << h;
    // row 1 <- position 1
    EXPECT_NEAR(dw[1 * hidden + h], dy[1 * hidden + h] * scale, 1e-6)
      << "row 1, h=" << h;
    // row 0 <- position 3 (the pad slot)
    EXPECT_NEAR(dw[0 * hidden + h], dy[3 * hidden + h] * scale, 1e-6)
      << "row 0, h=" << h;
    // untouched ids
    EXPECT_NEAR(dw[2 * hidden + h], 0.0f, 1e-6) << "row 2 must stay zero";
    EXPECT_NEAR(dw[4 * hidden + h], 0.0f, 1e-6) << "row 4 must stay zero";
    EXPECT_NEAR(dw[5 * hidden + h], 0.0f, 1e-6) << "row 5 must stay zero";
  }
}
