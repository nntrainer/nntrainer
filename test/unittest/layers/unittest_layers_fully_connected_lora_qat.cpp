// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_layers_fully_connected_lora_qat.cpp
 * @date   31 July 2026
 * @brief  Correctness checks for FullyConnectedLayer's per-block Q4_0 QAT
 *         fake-quantization (EMA-calibrated) and its straight-through
 *         estimator (STE) backward.
 * @bug    No known bugs except for NYI items
 */

#include <layer_context.h>
#include <var_grad.h>
#include <weight.h>

#include <fc_layer.h>

#include <gtest/gtest.h>

#include <cmath>
#include <utility>
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
 * @brief Reference recomputation of FullyConnectedLayer::fakeQuantizeQ4_0's
 *        forward math, given the per-block EMA scale already calibrated by
 *        a real forward pass (retrieved via getRegisteredBlockScales()).
 *        This is NOT a reimplementation of the EMA calibration itself --
 *        only of the quantize-with-a-known-scale step -- so a test built on
 *        it still exercises the real calibration code.
 */
float fakeQuantWithScale(float v, float d) {
  if (d < 1e-10f)
    return v;
  float q = std::round(v / d);
  q = std::max(-8.0f, std::min(7.0f, q));
  return q * d;
}

/** Blocks are indexed in N x K layout; x is stored K x N (nntrainer/GGML
 * convention: height=K, width=N), matching fakeQuantizeQ4_0 exactly. */
nntrainer::Tensor applyFakeQuantWithScales(const nntrainer::Tensor &x,
                                           const std::vector<float> &block_d) {
  const size_t K = x.getDim().height();
  const size_t N = x.getDim().width();
  nntrainer::Tensor out = x.clone();
  for (size_t k = 0; k < K; ++k) {
    for (size_t n = 0; n < N; ++n) {
      const size_t nk = n * K + k;
      const size_t b = nk / 32;
      out.getValue<float>(k * N + n) =
        fakeQuantWithScale(x.getValue<float>(k * N + n), block_d[b]);
    }
  }
  return out;
}

/**
 * @brief Common fixture: a FullyConnectedLayer with lora_qat enabled, a
 *        zeroed (and frozen) base weight and bias disabled -- so the
 *        layer's entire output is the LoRA path, letting these tests check
 *        the QAT/STE math in isolation from the base matmul.
 */
struct FCLoraQATFixture {
  static constexpr unsigned int height = 2, in_dim = 4, unit = 6,
                                lora_rank = 32;

  nntrainer::FullyConnectedLayer layer;
  nntrainer::InitLayerContext init_context;
  std::vector<nntrainer::Weight> weights;
  std::vector<nntrainer::Var_Grad> inputs, outputs, tensors;
  std::vector<float> loraA_init, loraB_init;
  // rc's initializer calls build(), which fills the members above -- keep
  // rc declared last so it's constructed last (member init order follows
  // declaration order, not the constructor's initializer-list order).
  nntrainer::RunLayerContext rc;

  FCLoraQATFixture(const std::string &name) :
    init_context({nntrainer::TensorDim({1, 1, height, in_dim})}, {true}, false,
                name, "", 0.0f, {"NCHW", "FP32", "FP32"}),
    rc(build(name)) {}

  nntrainer::RunLayerContext build(const std::string &name) {
    layer.setProperty({"unit=" + std::to_string(unit),
                       "lora_rank=" + std::to_string(lora_rank), "lora_qat=true",
                       "disable_bias=true", "weight_initializer=zeros"});
    layer.finalize(init_context);

    // weight (base, in_dim x unit): left at its ZEROS initializer value, so
    // the layer's output is exactly the LoRA contribution.
    nntrainer::Tensor weight_v(nntrainer::TensorDim({1, 1, in_dim, unit}),
                              true);
    weight_v.setZero();
    weights.emplace_back(weight_v, nntrainer::Tensor(), nntrainer::Tensor(),
                         "weight");

    // loraA (in_dim x lora_rank = 128 elements = 4 Q4_0 blocks)
    nntrainer::Tensor loraA_v(nntrainer::TensorDim({1, 1, in_dim, lora_rank}),
                             true);
    loraA_init.resize(in_dim * lora_rank);
    for (unsigned int i = 0; i < loraA_init.size(); ++i)
      loraA_init[i] = std::sin(0.37f * i + 0.11f) * 0.6f;
    std::copy(loraA_init.begin(), loraA_init.end(), loraA_v.getData<float>());
    nntrainer::Tensor loraA_g(nntrainer::TensorDim({1, 1, in_dim, lora_rank}),
                             true);
    loraA_g.setZero();
    weights.emplace_back(loraA_v, loraA_g, nntrainer::Tensor(), "loraA");

    // loraB (lora_rank x unit = 192 elements = 6 Q4_0 blocks)
    nntrainer::Tensor loraB_v(nntrainer::TensorDim({1, 1, lora_rank, unit}),
                             true);
    loraB_init.resize(lora_rank * unit);
    for (unsigned int i = 0; i < loraB_init.size(); ++i)
      loraB_init[i] = std::sin(0.53f * i + 0.29f) * 0.4f;
    std::copy(loraB_init.begin(), loraB_init.end(), loraB_v.getData<float>());
    nntrainer::Tensor loraB_g(nntrainer::TensorDim({1, 1, lora_rank, unit}),
                             true);
    loraB_g.setZero();
    weights.emplace_back(loraB_v, loraB_g, nntrainer::Tensor(), "loraB");

    // loraTmp/loraOut: scratch tensors finalize() also requests via
    // context.requestTensor(); forwarding()/calcGradient() index into the
    // RunLayerContext's tensors vector by the same relative position, so
    // these must be present even though the test never reads loraOut
    // directly.
    tensors.emplace_back(nntrainer::TensorDim({1, 1, height, lora_rank}),
                         nntrainer::Initializer::NONE, true, true,
                         "hidden_tmp_lora");
    tensors.emplace_back(nntrainer::TensorDim({1, 1, height, unit}),
                         nntrainer::Initializer::NONE, true, true,
                         "hidden_lora");

    inputs.emplace_back(init_context.getInputDimensions()[0],
                        nntrainer::Initializer::NONE, true, true, "input");
    for (unsigned int i = 0; i < height * in_dim; ++i)
      inputs[0].getVariableRef().getData<float>()[i] =
        std::sin(0.71f * i + 0.05f) * 0.5f;

    outputs.emplace_back(init_context.getOutSpecs()[0].variable_spec.dim,
                         nntrainer::Initializer::NONE, true, true, "output");

    return makeRunContext(name, weights, inputs, outputs, tensors);
  }
};

} // namespace

TEST(FullyConnectedLoraQAT, ForwardMatchesReferenceFakeQuantAndIsNotIdentity) {
  FCLoraQATFixture fx("fc_qat_forward");

  fx.layer.forwarding(fx.rc, true);

  auto [a_bd, b_bd] =
    nntrainer::FullyConnectedLayer::getRegisteredBlockScales("fc_qat_forward");
  ASSERT_EQ(a_bd.size(), (fx.in_dim * fx.lora_rank) / 32u);
  ASSERT_EQ(b_bd.size(), (fx.lora_rank * fx.unit) / 32u);

  nntrainer::Tensor loraA_ref(
    nntrainer::TensorDim({1, 1, fx.in_dim, fx.lora_rank}), true);
  std::copy(fx.loraA_init.begin(), fx.loraA_init.end(),
           loraA_ref.getData<float>());
  nntrainer::Tensor loraB_ref(
    nntrainer::TensorDim({1, 1, fx.lora_rank, fx.unit}), true);
  std::copy(fx.loraB_init.begin(), fx.loraB_init.end(),
           loraB_ref.getData<float>());

  nntrainer::Tensor a_fq_ref = applyFakeQuantWithScales(loraA_ref, a_bd);
  nntrainer::Tensor b_fq_ref = applyFakeQuantWithScales(loraB_ref, b_bd);

  // Sanity: quantization actually changed something. If this ever fails,
  // the test data no longer exercises rounding and the checks below would
  // pass vacuously even with a broken STE wiring.
  float max_abs_diff = 0.0f;
  for (unsigned int i = 0; i < fx.loraA_init.size(); ++i)
    max_abs_diff = std::max(
      max_abs_diff, std::abs(a_fq_ref.getData<float>()[i] - fx.loraA_init[i]));
  EXPECT_GT(max_abs_diff, 1e-6f)
    << "fake-quant left loraA unchanged; test data no longer discriminates";

  nntrainer::Tensor lora_tmp_ref(
    nntrainer::TensorDim({1, 1, fx.height, fx.lora_rank}), true);
  fx.inputs[0].getVariableRef().dot(a_fq_ref, lora_tmp_ref, false, false);
  nntrainer::Tensor expected_out(
    nntrainer::TensorDim({1, 1, fx.height, fx.unit}), true);
  lora_tmp_ref.dot(b_fq_ref, expected_out, false, false);

  const float *actual = fx.rc.getOutput(0).getData<float>();
  const float *expected = expected_out.getData<float>();
  for (unsigned int i = 0; i < fx.height * fx.unit; ++i)
    EXPECT_NEAR(actual[i], expected[i], 1e-5f)
      << "forward output mismatch at " << i;
}

TEST(FullyConnectedLoraQAT, CalcDerivativeAndCalcGradientUseFakeQuantizedWeightsSTE) {
  FCLoraQATFixture fx("fc_qat_backward");

  fx.layer.forwarding(fx.rc, true);

  auto [a_bd, b_bd] =
    nntrainer::FullyConnectedLayer::getRegisteredBlockScales("fc_qat_backward");

  nntrainer::Tensor loraA_ref(
    nntrainer::TensorDim({1, 1, fx.in_dim, fx.lora_rank}), true);
  std::copy(fx.loraA_init.begin(), fx.loraA_init.end(),
           loraA_ref.getData<float>());
  nntrainer::Tensor loraB_ref(
    nntrainer::TensorDim({1, 1, fx.lora_rank, fx.unit}), true);
  std::copy(fx.loraB_init.begin(), fx.loraB_init.end(),
           loraB_ref.getData<float>());

  nntrainer::Tensor a_fq_ref = applyFakeQuantWithScales(loraA_ref, a_bd);
  nntrainer::Tensor b_fq_ref = applyFakeQuantWithScales(loraB_ref, b_bd);

  std::vector<float> dy(fx.height * fx.unit);
  for (unsigned int i = 0; i < dy.size(); ++i)
    dy[i] = std::cos(0.61f * i + 0.17f) * 0.3f;
  std::copy(dy.begin(), dy.end(), fx.rc.getOutputGradUnsafe(0).getData<float>());

  fx.layer.calcDerivative(fx.rc);
  fx.layer.calcGradient(fx.rc);

  // --- calcDerivative: dL/dx must use a_fq . b_fq (STE), not raw loraA/loraB.
  nntrainer::Tensor lora_contrib_ref(
    nntrainer::TensorDim({1, 1, fx.in_dim, fx.unit}), true);
  a_fq_ref.dot(b_fq_ref, lora_contrib_ref, false, false);
  // base weight is zero, so the effective weight the real calcDerivative
  // adds against is exactly lora_contrib_ref (lora_scaling defaults to 1
  // since lora_alpha was not set).
  nntrainer::Tensor expected_deriv(
    nntrainer::TensorDim({1, 1, fx.height, fx.in_dim}), true);
  nntrainer::Tensor dy_tensor(nntrainer::TensorDim({1, 1, fx.height, fx.unit}),
                             true);
  std::copy(dy.begin(), dy.end(), dy_tensor.getData<float>());
  expected_deriv.dot_deriv_wrt_1(lora_contrib_ref, dy_tensor, false, false);

  const float *actual_deriv = fx.rc.getOutgoingDerivative(0).getData<float>();
  const float *exp_deriv = expected_deriv.getData<float>();
  for (unsigned int i = 0; i < fx.height * fx.in_dim; ++i)
    EXPECT_NEAR(actual_deriv[i], exp_deriv[i], 1e-5f)
      << "calcDerivative mismatch at " << i
      << " -- outgoing derivative does not match the fake-quantized-weight "
         "(STE) formula";

  // --- calcGradient: dL/dloraA and dL/dloraB must use the STE chain
  // through b_fq/a_fq, matching exactly what forwarding() computed.
  nntrainer::Tensor lora_tmp_ref(
    nntrainer::TensorDim({1, 1, fx.height, fx.lora_rank}), true);
  fx.inputs[0].getVariableRef().dot(a_fq_ref, lora_tmp_ref, false, false);

  nntrainer::Tensor expected_djdtmp(
    nntrainer::TensorDim({1, 1, fx.height, fx.lora_rank}), true);
  expected_djdtmp.dot_deriv_wrt_1(b_fq_ref, dy_tensor, false, false);

  nntrainer::Tensor expected_djdla(
    nntrainer::TensorDim({1, 1, fx.in_dim, fx.lora_rank}), true);
  fx.inputs[0].getVariableRef().dot_deriv_wrt_2(expected_djdla, expected_djdtmp,
                                                false, false);

  nntrainer::Tensor expected_djdlb(
    nntrainer::TensorDim({1, 1, fx.lora_rank, fx.unit}), true);
  lora_tmp_ref.dot_deriv_wrt_2(expected_djdlb, dy_tensor, false, false);

  const float *actual_djdla = fx.rc.getWeightGrad(1).getData<float>();
  const float *exp_djdla = expected_djdla.getData<float>();
  for (unsigned int i = 0; i < fx.in_dim * fx.lora_rank; ++i)
    EXPECT_NEAR(actual_djdla[i], exp_djdla[i], 1e-5f)
      << "calcGradient dL/dloraA mismatch at " << i;

  const float *actual_djdlb = fx.rc.getWeightGrad(2).getData<float>();
  const float *exp_djdlb = expected_djdlb.getData<float>();
  for (unsigned int i = 0; i < fx.lora_rank * fx.unit; ++i)
    EXPECT_NEAR(actual_djdlb[i], exp_djdlb[i], 1e-5f)
      << "calcGradient dL/dloraB mismatch at " << i;
}

TEST(FullyConnectedLoraQAT, EMABootstrapsThenBlendsAndFreezesDuringEval) {
  FCLoraQATFixture fx("fc_qat_ema");

  // First training forward: EMA has no history yet, so it must bootstrap
  // directly to the fresh per-block scale (not blend against zero).
  fx.layer.forwarding(fx.rc, true);
  auto [a_bd_1, b_bd_1] =
    nntrainer::FullyConnectedLayer::getRegisteredBlockScales("fc_qat_ema");
  ASSERT_FALSE(a_bd_1.empty());

  // A second training forward with different loraA values must blend the
  // EMA: block_d = (1 - momentum) * old + momentum * fresh, momentum = 0.1.
  for (unsigned int i = 0; i < fx.in_dim * fx.lora_rank; ++i)
    fx.weights[1].getVariableRef().getData<float>()[i] =
      std::sin(1.9f * i + 0.4f) * 1.2f;
  fx.layer.forwarding(fx.rc, true);
  auto [a_bd_2, b_bd_2] =
    nntrainer::FullyConnectedLayer::getRegisteredBlockScales("fc_qat_ema");

  // Recompute the expected fresh per-block scale for the second call and
  // confirm the registry reflects an EMA blend, not a hard overwrite.
  const nntrainer::Tensor &loraA_now = fx.weights[1].getVariableRef();
  const size_t K = fx.in_dim, N = fx.lora_rank;
  for (size_t b = 0; b < a_bd_2.size(); ++b) {
    float max_abs = 0.0f;
    for (size_t j = 0; j < 32; ++j) {
      const size_t nk = b * 32 + j;
      const size_t n = nk / K, k = nk % K;
      max_abs =
        std::max(max_abs, std::abs(loraA_now.getValue<float>(k * N + n)));
    }
    const float d_fresh = (max_abs > 1e-8f) ? max_abs / 8.0f : 0.0f;
    const float expected_blend = 0.9f * a_bd_1[b] + 0.1f * d_fresh;
    EXPECT_NEAR(a_bd_2[b], expected_blend, 1e-5f)
      << "EMA block " << b << " did not blend as (1-momentum)*old + "
                              "momentum*fresh";
  }

  // A validation-mode (training=false) forward must NOT update the EMA:
  // the registry should be unchanged, and calling it twice in a row must
  // be bit-identical (no residual state mutation on repeated eval calls).
  fx.layer.forwarding(fx.rc, false);
  auto [a_bd_3, b_bd_3] =
    nntrainer::FullyConnectedLayer::getRegisteredBlockScales("fc_qat_ema");
  ASSERT_EQ(a_bd_3.size(), a_bd_2.size());
  for (size_t b = 0; b < a_bd_3.size(); ++b)
    EXPECT_FLOAT_EQ(a_bd_3[b], a_bd_2[b])
      << "validation-mode forward updated EMA block " << b;

  fx.layer.forwarding(fx.rc, false);
  auto [a_bd_4, b_bd_4] =
    nntrainer::FullyConnectedLayer::getRegisteredBlockScales("fc_qat_ema");
  for (size_t b = 0; b < a_bd_4.size(); ++b)
    EXPECT_FLOAT_EQ(a_bd_4[b], a_bd_3[b])
      << "repeated validation-mode forward is not idempotent at block " << b;
}
