// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Donghak Park <donghak.park@samsung.com>
 *
 * @file unittest_layers_layer_normalization_direct.cpp
 * @date 12 December 2025
 * @brief Layer Normalization Layer Direct Unit Tests
 * @see https://github.com/nnstreamer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layer_normalization_layer.h>
#include <layers_common_tests.h>

/**
 * @brief Direct test for LayerNormalizationLayer setProperty
 */
TEST(LayerNormalizationLayerDirect, setProperty_valid) {
  auto layer = nntrainer::createLayer<nntrainer::LayerNormalizationLayer>({});
  EXPECT_NO_THROW(layer->setProperty({"epsilon=1e-5"}));
}

/**
 * @brief Test getType returns correct type
 */
TEST(LayerNormalizationLayerDirect, getType) {
  auto layer = nntrainer::createLayer<nntrainer::LayerNormalizationLayer>({});
  EXPECT_EQ(layer->getType(), "layer_normalization");
}

/**
 * @brief Test supportBackwarding returns true
 */
TEST(LayerNormalizationLayerDirect, supportBackwarding) {
  auto layer = nntrainer::createLayer<nntrainer::LayerNormalizationLayer>({});
  EXPECT_TRUE(layer->supportBackwarding());
}

/**
 * @brief Test with different epsilon values
 */
TEST(LayerNormalizationLayerDirect, setProperty_various) {
  auto layer1 = nntrainer::createLayer<nntrainer::LayerNormalizationLayer>({});
  EXPECT_NO_THROW(layer1->setProperty({"epsilon=1e-6"}));

  auto layer2 = nntrainer::createLayer<nntrainer::LayerNormalizationLayer>({});
  EXPECT_NO_THROW(layer2->setProperty({"epsilon=1e-4"}));
}
