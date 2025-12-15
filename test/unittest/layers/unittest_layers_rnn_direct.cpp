// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Donghak Park <donghak.park@samsung.com>
 *
 * @file unittest_layers_rnn_direct.cpp
 * @date 12 December 2025
 * @brief RNN Layer Direct Unit Tests
 * @see https://github.com/nnstreamer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <rnn.h>

/**
 * @brief Direct test for RNNLayer setProperty
 */
TEST(RNNLayerDirect, setProperty_valid) {
  auto layer = nntrainer::createLayer<nntrainer::RNNLayer>({});
  EXPECT_NO_THROW(layer->setProperty({"unit=32"}));
}

/**
 * @brief Test getType returns correct type
 */
TEST(RNNLayerDirect, getType) {
  auto layer = nntrainer::createLayer<nntrainer::RNNLayer>({});
  EXPECT_EQ(layer->getType(), "rnn");
}

/**
 * @brief Test supportBackwarding returns true
 */
TEST(RNNLayerDirect, supportBackwarding) {
  auto layer = nntrainer::createLayer<nntrainer::RNNLayer>({});
  EXPECT_TRUE(layer->supportBackwarding());
}

/**
 * @brief Test with various unit sizes
 */
TEST(RNNLayerDirect, setProperty_various_units) {
  auto layer1 = nntrainer::createLayer<nntrainer::RNNLayer>({});
  EXPECT_NO_THROW(layer1->setProperty({"unit=64"}));

  auto layer2 = nntrainer::createLayer<nntrainer::RNNLayer>({});
  EXPECT_NO_THROW(layer2->setProperty({"unit=128", "return_sequences=true"}));

  auto layer3 = nntrainer::createLayer<nntrainer::RNNLayer>({});
  EXPECT_NO_THROW(
    layer3->setProperty({"unit=256", "hidden_state_activation=tanh"}));
}

/**
 * @brief Test setProperty with invalid property name
 */
TEST(RNNLayerDirect, setProperty_invalid_name) {
  auto layer = nntrainer::createLayer<nntrainer::RNNLayer>({});
  EXPECT_THROW(layer->setProperty({"invalid_property=100"}), std::exception);
}

/**
 * @brief Test setProperty with invalid value type
 */
TEST(RNNLayerDirect, setProperty_invalid_type) {
  auto layer = nntrainer::createLayer<nntrainer::RNNLayer>({});
  // unit expects a positive integer, passing a string
  EXPECT_THROW(layer->setProperty({"unit=not_a_number"}), std::exception);
}
