// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file unittest_layers_fully_connected.cpp
 * @date 15 June 2021
 * @brief Fully Connected Layer Test
 * @see	https://github.com/nntrainer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <fc_layer.h>
#include <layer_context.h>
#include <layers_common_tests.h>

auto semantic_fc = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::FullyConnectedLayer>,
  nntrainer::FullyConnectedLayer::type, {"unit=1"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

GTEST_PARAMETER_TEST(FullyConnected, LayerSemantics,
                     ::testing::Values(semantic_fc));

auto fc_basic_plain = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::FullyConnectedLayer>, {"unit=5"},
  "3:1:1:10", "fc_plain.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT,
  "nchw", "fp32", "fp32");
auto fc_basic_single_batch = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::FullyConnectedLayer>, {"unit=4"},
  "1:1:1:10", "fc_single_batch.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");
auto fc_basic_no_decay = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::FullyConnectedLayer>,
  {"unit=5", "weight_decay=0.0", "bias_decay=0.0"}, "3:1:1:10",
  "fc_plain.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT, "nchw",
  "fp32", "fp32");

auto fc_basic_plain_nhwc = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::FullyConnectedLayer>, {"unit=5"},
  "3:10:1:1", "fc_plain.nnlayergolden",
  LayerGoldenTestParamOptions::SKIP_CALC_DERIV |
    LayerGoldenTestParamOptions::SKIP_CALC_GRAD,
  "nhwc", "fp32", "fp32");

auto fc_basic_single_batch_nhwc = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::FullyConnectedLayer>, {"unit=4"},
  "1:10:1:1", "fc_single_batch.nnlayergolden",
  LayerGoldenTestParamOptions::SKIP_CALC_DERIV |
    LayerGoldenTestParamOptions::SKIP_CALC_GRAD,
  "nhwc", "fp32", "fp32");

auto fc_basic_no_decay_nhwc = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::FullyConnectedLayer>,
  {"unit=5", "weight_decay=0.0", "bias_decay=0.0"}, "3:10:1:1",
  "fc_plain.nnlayergolden",
  LayerGoldenTestParamOptions::SKIP_CALC_DERIV |
    LayerGoldenTestParamOptions::SKIP_CALC_GRAD,
  "nhwc", "fp32", "fp32");

GTEST_PARAMETER_TEST(FullyConnected, LayerGoldenTest,
                     ::testing::Values(fc_basic_plain, fc_basic_single_batch,
                                       fc_basic_no_decay, fc_basic_plain_nhwc,
                                       fc_basic_single_batch_nhwc,
                                       fc_basic_no_decay_nhwc));

#ifdef ENABLE_FP16
auto fc_basic_plain_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::FullyConnectedLayer>, {"unit=5"},
  "3:1:1:10", "fc_plain_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

auto fc_basic_single_batch_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::FullyConnectedLayer>, {"unit=4"},
  "1:1:1:10", "fc_single_batch_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

auto fc_basic_no_decay_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::FullyConnectedLayer>,
  {"unit=5", "weight_decay=0.0", "bias_decay=0.0"}, "3:1:1:10",
  "fc_plain_w16a16.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT,
  "nchw", "fp16", "fp16");

GTEST_PARAMETER_TEST(FullyConnected16, LayerGoldenTest,
                     ::testing::Values(fc_basic_plain_w16a16,
                                       fc_basic_single_batch_w16a16,
                                       fc_basic_no_decay_w16a16));
#endif

/**
 * @brief A fused activation epilogue has no backward, so finalizing a layer
 * that carries one for anything other than an inference graph must throw
 * rather than silently drop the activation derivative during training.
 */
TEST(FullyConnected, fusedActivationOnTrainingGraph_n) {
  auto layer = nntrainer::createLayer<nntrainer::FullyConnectedLayer>();
  EXPECT_NO_THROW(layer->setProperty({"unit=5", "fused_activation=relu"}));

  std::vector<ml::train::TensorDim> input_dims(
    1, ml::train::TensorDim({1, 1, 1, 4}));
  nntrainer::InitLayerContext train_context(
    input_dims, {true}, false, "fc", "", 0.0f, {"NCHW", "FP32", "FP32"}, 1.0f,
    ml::train::ExecutionMode::TRAIN);
  EXPECT_THROW(layer->finalize(train_context), std::invalid_argument);
}

/**
 * @brief The same property is accepted on an inference graph, which is the
 * only mode the FusionRealizer sets it in.
 */
TEST(FullyConnected, fusedActivationOnInferenceGraph_p) {
  auto layer = nntrainer::createLayer<nntrainer::FullyConnectedLayer>();
  EXPECT_NO_THROW(layer->setProperty({"unit=5", "fused_activation=relu"}));

  std::vector<ml::train::TensorDim> input_dims(
    1, ml::train::TensorDim({1, 1, 1, 4}));
  nntrainer::InitLayerContext infer_context(
    input_dims, {true}, false, "fc", "", 0.0f, {"NCHW", "FP32", "FP32"}, 1.0f,
    ml::train::ExecutionMode::INFERENCE);
  EXPECT_NO_THROW(layer->finalize(infer_context));
}

/**
 * @brief Finalize a fully connected layer over an eight-row input and report
 * the height the layer planned its output at.
 *
 * @param props properties to set on the layer before finalizing
 * @return unsigned int planned output height
 */
static unsigned int plannedOutputHeight(const std::vector<std::string> &props) {
  auto layer = nntrainer::createLayer<nntrainer::FullyConnectedLayer>();
  layer->setProperty(props);

  std::vector<ml::train::TensorDim> input_dims(
    1, ml::train::TensorDim({1, 1, 8, 4}));
  nntrainer::InitLayerContext context(input_dims, {true}, false, "fc", "", 0.0f,
                                      {"NCHW", "FP32", "FP32"}, 1.0f,
                                      ml::train::ExecutionMode::INFERENCE);
  layer->finalize(context);

  return context.getOutSpecs().at(0).variable_spec.dim.height();
}

/**
 * @brief Without the property the output keeps the graph-build height, which
 * is the behaviour of every layer that does not carry it.
 */
TEST(FullyConnected, planLastRowOnlyUnset_p) {
  EXPECT_EQ(plannedOutputHeight({"unit=5"}), 8u);
  EXPECT_EQ(plannedOutputHeight({"unit=5", "skip_prefill=true"}), 8u);
}

/**
 * @brief A layer the model has declared to produce only its last row plans a
 * single output row instead of a plane whose rows above the first are never
 * written.
 */
TEST(FullyConnected, planLastRowOnlySet_p) {
  EXPECT_EQ(plannedOutputHeight(
              {"unit=5", "skip_prefill=true", "plan_last_row_only=true"}),
            1u);
}

/**
 * @brief The property is read together with skip_prefill: on a layer that does
 * fill every row, shortening the plan would drop the rows it writes, so the
 * collapse is not applied.
 */
TEST(FullyConnected, planLastRowOnlyWithoutSkipPrefill_p) {
  EXPECT_EQ(plannedOutputHeight({"unit=5", "plan_last_row_only=true"}), 8u);
}

/**
 * @brief Setting the property false is the same as leaving it unset.
 */
TEST(FullyConnected, planLastRowOnlyFalse_p) {
  EXPECT_EQ(plannedOutputHeight(
              {"unit=5", "skip_prefill=true", "plan_last_row_only=false"}),
            8u);
}
