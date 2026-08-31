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
#include <tuple>

#include <gtest/gtest.h>

#include <layer_normalization_layer.h>
#include <layers_common_tests.h>

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
