// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Donghak Park <donghak.park@samsung.com>
 *
 * @file unittest_layers_direct.cpp
 * @date 15 December 2025
 * @brief Unified Direct Unit Tests for Layer Types using Parameterized Tests
 * @see https://github.com/nnstreamer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <gtest/gtest.h>

#include <layers_common_tests.h>

// Layer headers
#include <gru.h>
#include <grucell.h>
#include <layer_normalization_layer.h>
#include <lstm.h>
#include <lstmcell.h>
#include <multi_head_attention_layer.h>
#include <permute_layer.h>
#include <reduce_mean_layer.h>
#include <reduce_sum_layer.h>
#include <rnn.h>
#include <rnncell.h>
#include <split_layer.h>
#include <upsample2d_layer.h>
#include <zoneout_lstmcell.h>

/**
 * @brief Layer Direct Test Configuration
 * Contains layer type info, expected type string, and whether it supports
 * backwarding
 */
struct LayerDirectTestConfig {
  LayerFactoryType factory;
  std::string expected_type;
  bool supports_backwarding;
  std::string invalid_property_for_type_test;
};

/**
 * @brief Test fixture for direct layer tests
 */
class LayerDirectTest : public ::testing::TestWithParam<LayerDirectTestConfig> {
protected:
  void SetUp() override {
    auto config = GetParam();
    layer = config.factory({});
  }

  std::unique_ptr<nntrainer::Layer> layer;
};

/**
 * @brief Test getType returns correct type
 */
TEST_P(LayerDirectTest, getType) {
  auto config = GetParam();
  EXPECT_EQ(layer->getType(), config.expected_type);
}

/**
 * @brief Test supportBackwarding returns expected value
 */
TEST_P(LayerDirectTest, supportBackwarding) {
  auto config = GetParam();
  EXPECT_EQ(layer->supportBackwarding(), config.supports_backwarding);
}

/**
 * @brief Test setProperty with invalid property name
 */
TEST_P(LayerDirectTest, setProperty_invalid_name) {
  EXPECT_THROW(layer->setProperty({"invalid_property=100"}), std::exception);
}

/**
 * @brief Test setProperty with invalid value type
 */
TEST_P(LayerDirectTest, setProperty_invalid_type) {
  auto config = GetParam();
  EXPECT_THROW(layer->setProperty({config.invalid_property_for_type_test}),
               std::exception);
}

// Define test configurations for each layer type
// clang-format off
static const LayerDirectTestConfig kLayerDirectTestConfigs[] = {
  {nntrainer::createLayer<nntrainer::LSTMLayer>, "lstm", true, "unit=not_a_number"},
  {nntrainer::createLayer<nntrainer::RNNLayer>, "rnn", true, "unit=not_a_number"},
  {nntrainer::createLayer<nntrainer::GRULayer>, "gru", true, "unit=not_a_number"},
  {nntrainer::createLayer<nntrainer::GRUCellLayer>, "grucell", true, "unit=not_a_number"},
  {nntrainer::createLayer<nntrainer::LSTMCellLayer>, "lstmcell", true, "unit=not_a_number"},
  {nntrainer::createLayer<nntrainer::RNNCellLayer>, "rnncell", true, "unit=not_a_number"},
  {nntrainer::createLayer<nntrainer::ZoneoutLSTMCellLayer>, "zoneout_lstmcell", true, "unit=not_a_number"},
  {nntrainer::createLayer<nntrainer::MultiHeadAttentionLayer>, "multi_head_attention", true, "num_heads=not_a_number"},
  {nntrainer::createLayer<nntrainer::ReduceMeanLayer>, "reduce_mean", true, "axis=not_a_number"},
  {nntrainer::createLayer<nntrainer::ReduceSumLayer>, "reduce_sum", true, "axis=not_a_number"},
  {nntrainer::createLayer<nntrainer::LayerNormalizationLayer>, "layer_normalization", true, "epsilon=not_a_number"},
  {nntrainer::createLayer<nntrainer::PermuteLayer>, "permute", true, "direction=not_valid"},
  {nntrainer::createLayer<nntrainer::SplitLayer>, "split", true, "split_dimension=not_a_number"},
  {nntrainer::createLayer<nntrainer::Upsample2dLayer>, "upsample2d", true, "upsample_mode=not_valid"},
};
// clang-format on

INSTANTIATE_TEST_SUITE_P(
  LayerDirect, LayerDirectTest, ::testing::ValuesIn(kLayerDirectTestConfigs),
  [](const ::testing::TestParamInfo<LayerDirectTestConfig> &info) {
    // Create test name from layer type (replace special chars)
    std::string name = info.param.expected_type;
    std::replace(name.begin(), name.end(), '_', 'X');
    return name;
  });
