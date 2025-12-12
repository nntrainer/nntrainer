// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Donghak Park <donghak.park@samsung.com>
 *
 * @file unittest_layers_lstm_direct.cpp
 * @date 12 December 2025
 * @brief LSTM Layer Direct Unit Tests
 * @see https://github.com/nnstreamer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <lstm.h>
#include <layers_common_tests.h>

/**
 * @brief Direct test for LSTMLayer setProperty
 */
TEST(LSTMLayerDirect, setProperty_valid) {
  auto layer = nntrainer::createLayer<nntrainer::LSTMLayer>({});
  EXPECT_NO_THROW(layer->setProperty({"unit=32"}));
}

/**
 * @brief Test getType returns correct type
 */
TEST(LSTMLayerDirect, getType) {
  auto layer = nntrainer::createLayer<nntrainer::LSTMLayer>({});
  EXPECT_EQ(layer->getType(), "lstm");
}

/**
 * @brief Test supportBackwarding returns true
 */
TEST(LSTMLayerDirect, supportBackwarding) {
  auto layer = nntrainer::createLayer<nntrainer::LSTMLayer>({});
  EXPECT_TRUE(layer->supportBackwarding());
}

/**
 * @brief Test with various unit sizes and properties
 */
TEST(LSTMLayerDirect, setProperty_various) {
  auto layer1 = nntrainer::createLayer<nntrainer::LSTMLayer>({});
  EXPECT_NO_THROW(layer1->setProperty({"unit=64"}));

  auto layer2 = nntrainer::createLayer<nntrainer::LSTMLayer>({});
  EXPECT_NO_THROW(layer2->setProperty({"unit=128", "return_sequences=true"}));

  auto layer3 = nntrainer::createLayer<nntrainer::LSTMLayer>({});
  EXPECT_NO_THROW(layer3->setProperty({"unit=256", "hidden_state_activation=tanh"}));
}
