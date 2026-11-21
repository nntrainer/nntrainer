// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 hyeonseok Lee <hs89.lee@samsung.com>
 *
 * @file unittest_layers_loss.cpp
 * @date 29 July 2022
 * @brief Layer Normalization Layer Test
 * @see	https://github.com/nntrainer/nntrainer
 * @author hyeonseok Lee <hs89.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <cmath>
#include <random>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include <layer_context.h>
#include <layer_normalization_layer.h>
#include <layers_common_tests.h>
#include <var_grad.h>
#include <weight.h>

auto semantic_layer_normalization = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::LayerNormalizationLayer>,
  nntrainer::LayerNormalizationLayer::type, {"axis=1"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

auto ln_option = LayerGoldenTestParamOptions::SKIP_COSINE_SIMILARITY;

GTEST_PARAMETER_TEST(LayerNormalization, LayerSemantics,
                     ::testing::Values(semantic_layer_normalization));

auto ln_axis_1 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::LayerNormalizationLayer>, {"axis=1"},
  "2:4:2:3", "ln_axis_1.nnlayergolden", ln_option, "nchw", "fp32", "fp32");

auto ln_axis_2 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::LayerNormalizationLayer>, {"axis=2"},
  "2:4:2:3", "ln_axis_2.nnlayergolden", ln_option, "nchw", "fp32", "fp32");

auto ln_axis_3 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::LayerNormalizationLayer>, {"axis=3"},
  "2:4:2:3", "ln_axis_3.nnlayergolden", ln_option, "nchw", "fp32", "fp32");

auto ln_axis_1_2 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::LayerNormalizationLayer>, {"axis=1, 2"},
  "2:4:2:3", "ln_axis_1_2.nnlayergolden", ln_option, "nchw", "fp32", "fp32");

auto ln_axis_2_3 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::LayerNormalizationLayer>, {"axis=2, 3"},
  "2:4:2:3", "ln_axis_2_3.nnlayergolden", ln_option, "nchw", "fp32", "fp32");

auto ln_axis_1_3 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::LayerNormalizationLayer>, {"axis=1, 3"},
  "2:4:2:3", "ln_axis_1_3.nnlayergolden", ln_option, "nchw", "fp32", "fp32");

auto ln_axis_1_2_3 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::LayerNormalizationLayer>, {"axis=1, 2, 3"},
  "2:4:2:3", "ln_axis_1_2_3.nnlayergolden", ln_option, "nchw", "fp32", "fp32");

GTEST_PARAMETER_TEST(LayerNormalization, LayerGoldenTest,
                     ::testing::Values(ln_axis_1, ln_axis_2, ln_axis_3,
                                       ln_axis_1_2, ln_axis_2_3, ln_axis_1_3,
                                       ln_axis_1_2_3));

// Inference-mode coverage for the ComputeOps::layer_norm whole-op path.
// Every case above runs forwarding(rc, training=true) and therefore takes the
// composite host path; without these the new op path would have ZERO golden
// coverage. FORWARD_MODE_INFERENCE additionally selects incremental_forwarding
// (same enum bit as USE_INC_FORWARD), so one case covers both the inference
// branch and the (active_rows, row_offset) window arithmetic. Only axis=3 is
// listed: the op's contract is "normalize over width", and any other axis
// deliberately keeps the composite path.
auto ln_axis_3_inference = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::LayerNormalizationLayer>, {"axis=3"},
  "2:4:2:3", "ln_axis_3.nnlayergolden",
  ln_option | LayerGoldenTestParamOptions::FORWARD_MODE_INFERENCE |
    LayerGoldenTestParamOptions::SKIP_CALC_GRAD |
    LayerGoldenTestParamOptions::SKIP_CALC_DERIV,
  "nchw", "fp32", "fp32");

GTEST_PARAMETER_TEST(LayerNormalizationInference, LayerGoldenTest,
                     ::testing::Values(ln_axis_3_inference));

#ifdef ENABLE_FP16
auto ln_axis_1_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::LayerNormalizationLayer>, {"axis=1"},
  "2:4:2:3", "ln_axis_1_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

auto ln_axis_2_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::LayerNormalizationLayer>, {"axis=2"},
  "2:4:2:3", "ln_axis_2_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

auto ln_axis_3_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::LayerNormalizationLayer>, {"axis=3"},
  "2:4:2:3", "ln_axis_3_w16a16.nnlayergolden", ln_option, "nchw", "fp16",
  "fp16");

auto ln_axis_1_2_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::LayerNormalizationLayer>, {"axis=1, 2"},
  "2:4:2:3", "ln_axis_1_2_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

auto ln_axis_2_3_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::LayerNormalizationLayer>, {"axis=2, 3"},
  "2:4:2:3", "ln_axis_2_3_w16a16.nnlayergolden", ln_option, "nchw", "fp16",
  "fp16");

auto ln_axis_1_3_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::LayerNormalizationLayer>, {"axis=1, 3"},
  "2:4:2:3", "ln_axis_1_3_w16a16.nnlayergolden", ln_option, "nchw", "fp16",
  "fp16");

auto ln_axis_1_2_3_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::LayerNormalizationLayer>, {"axis=1, 2, 3"},
  "2:4:2:3", "ln_axis_1_2_3_w16a16.nnlayergolden", ln_option, "nchw", "fp16",
  "fp16");

GTEST_PARAMETER_TEST(LayerNormalization16, LayerGoldenTest,
                     ::testing::Values(ln_axis_1_w16a16, ln_axis_2_w16a16,
                                       ln_axis_3_w16a16, ln_axis_1_2_w16a16,
                                       ln_axis_2_3_w16a16,
                                       ln_axis_1_3_w16a16,
                                       ln_axis_1_2_3_w16a16));

// fp16 inference-mode coverage for the whole-op path (see the fp32 case).
auto ln_axis_3_w16a16_inference = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::LayerNormalizationLayer>, {"axis=3"},
  "2:4:2:3", "ln_axis_3_w16a16.nnlayergolden",
  ln_option | LayerGoldenTestParamOptions::FORWARD_MODE_INFERENCE |
    LayerGoldenTestParamOptions::SKIP_CALC_GRAD |
    LayerGoldenTestParamOptions::SKIP_CALC_DERIV,
  "nchw", "fp16", "fp16");

GTEST_PARAMETER_TEST(LayerNormalizationInference16, LayerGoldenTest,
                     ::testing::Values(ln_axis_3_w16a16_inference));
#endif

namespace {

/**
 * @brief Run one incremental step [from, to) of an axis=3 LayerNorm and check
 * that the first channel * (to - from) rows of each batch are normalized and
 * the other rows are left untouched, the window ActivationLayer uses.
 *
 * @param act activation and weight dtype ("FP32" or "FP16")
 * @param training run the training path; it applies gamma / beta to the
 * whole output, so only the live rows are checked
 */
void runIncrementalWindow(const std::string &act, unsigned int from,
                          unsigned int to, float tol, bool training = false) {
  using nntrainer::Tensor;
  using nntrainer::TensorDim;
  const bool fp16 = act == "FP16";
  const TensorDim::DataType dt =
    fp16 ? TensorDim::DataType::FP16 : TensorDim::DataType::FP32;
  const unsigned int B = 2, C = 2, H = 4, W = 24;
  TensorDim in_dim(B, C, H, W,
                   TensorDim::TensorType(nntrainer::Tformat::NCHW, dt));

  nntrainer::LayerNormalizationLayer layer;
  layer.setProperty({"axis=3", "epsilon=0.00001"});
  nntrainer::InitLayerContext init_ctx({in_dim}, {true}, false, "ln", "", 0.0,
                                       {"NCHW", act, act}, 1.0,
                                       ml::train::ExecutionMode::INFERENCE);
  layer.finalize(init_ctx);

  auto get = [&](const Tensor &t, size_t i) -> float {
#ifdef ENABLE_FP16
    if (fp16)
      return static_cast<float>(t.getData<_FP16>()[i]);
#endif
    return t.getData<float>()[i];
  };
  auto set = [&](Tensor &t, size_t i, float v) {
#ifdef ENABLE_FP16
    if (fp16) {
      t.getData<_FP16>()[i] = static_cast<_FP16>(v);
      return;
    }
#endif
    t.getData<float>()[i] = v;
  };

  std::mt19937 rng(20261117);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

  std::vector<nntrainer::Weight> weights;
  weights.reserve(2);
  for (unsigned int i = 0; i < 2; ++i) {
    const TensorDim w_dim = std::get<0>(init_ctx.getWeightsSpec()[i]);
    weights.emplace_back(
      w_dim, nntrainer::Initializer::ZEROS, nntrainer::WeightRegularizer::NONE,
      1.0f, 0.0f, 0.0f, false, true, std::get<8>(init_ctx.getWeightsSpec()[i]));
    for (unsigned int k = 0; k < W; ++k)
      set(weights[i].getVariableRef(), k, dist(rng));
  }
  const Tensor &gamma = weights[0].getVariableRef();
  const Tensor &beta = weights[1].getVariableRef();

  const float sentinel = 7.0f;
  nntrainer::Var_Grad input(in_dim, nntrainer::Initializer::NONE, false, true,
                            "ln:input");
  nntrainer::Var_Grad output(init_ctx.getOutSpecs()[0].variable_spec.dim,
                             nntrainer::Initializer::NONE, false, true,
                             "ln:output");
  Tensor &in = input.getVariableRef();
  Tensor &out = output.getVariableRef();
  for (size_t i = 0; i < in.size(); ++i) {
    set(in, i, dist(rng));
    set(out, i, sentinel);
  }

  std::vector<nntrainer::Var_Grad> tensors;
  tensors.reserve(init_ctx.getTensorsSpec().size());
  for (auto const &spec : init_ctx.getTensorsSpec())
    tensors.emplace_back(spec, true);
  std::vector<nntrainer::Var_Grad *> tensor_ptrs;
  for (auto &t : tensors)
    tensor_ptrs.push_back(&t);

  nntrainer::RunLayerContext run_ctx("ln", false, 0.0f, false, 1.0f, nullptr,
                                     false, {&weights[0], &weights[1]},
                                     {&input}, {&output}, tensor_ptrs);
  layer.incremental_forwarding(run_ctx, from, to, training);

  const unsigned int rows = C * (to - from);
  for (unsigned int bt = 0; bt < B; ++bt) {
    for (unsigned int h = 0; h < C * H; ++h) {
      const size_t base = ((size_t)bt * C * H + h) * W;
      if (h >= rows) {
        if (training)
          continue;
        for (unsigned int k = 0; k < W; ++k)
          EXPECT_EQ(get(out, base + k), sentinel)
            << "batch " << bt << " row " << h << " was written";
        continue;
      }
      double mean = 0.0, var = 0.0;
      for (unsigned int k = 0; k < W; ++k)
        mean += get(in, base + k);
      mean /= W;
      for (unsigned int k = 0; k < W; ++k)
        var += (get(in, base + k) - mean) * (get(in, base + k) - mean);
      const double inv = 1.0 / std::sqrt(var / W + 0.00001);
      for (unsigned int k = 0; k < W; ++k) {
        const double ref =
          (get(in, base + k) - mean) * inv * get(gamma, k) + get(beta, k);
        EXPECT_NEAR(get(out, base + k), ref, tol)
          << "batch " << bt << " row " << h << " col " << k;
      }
    }
  }
}

} // namespace

/**
 * @brief a decode step past the prompt reads and writes row 0, not row from
 */
TEST(LayerNormalizationIncremental, fp32_step_rows_from_zero_p) {
  runIncrementalWindow("FP32", 2, 3, 1e-5f);
}

/**
 * @brief a multi-row chunk past position 0 covers rows [0, to - from)
 */
TEST(LayerNormalizationIncremental, fp32_chunk_rows_from_zero_p) {
  runIncrementalWindow("FP32", 1, 3, 1e-5f);
}

/**
 * @brief the training path (FP32 rms fast path with ENABLE_FP16) uses the
 * same live rows
 */
TEST(LayerNormalizationIncremental, fp32_training_chunk_rows_from_zero_p) {
  runIncrementalWindow("FP32", 1, 3, 1e-5f, true);
}

#ifdef ENABLE_FP16
/**
 * @brief FP16 activation follows the same window as FP32
 */
TEST(LayerNormalizationIncremental, fp16_step_rows_from_zero_p) {
  runIncrementalWindow("FP16", 2, 3, 2e-2f);
}

/**
 * @brief FP16 multi-row chunk past position 0
 */
TEST(LayerNormalizationIncremental, fp16_chunk_rows_from_zero_p) {
  runIncrementalWindow("FP16", 1, 3, 2e-2f);
}
#endif
