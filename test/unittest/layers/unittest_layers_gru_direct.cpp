// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Donghak Park <donghak.park@samsung.com>
 *
 * @file unittest_layers_gru_direct.cpp
 * @date 12 December 2025
 * @brief GRU Layer Direct Unit Tests
 * @see https://github.com/nnstreamer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <gru.h>
#include <layers_common_tests.h>

/**
 * @brief Direct test for GRULayer setProperty
 */
TEST(GRULayerDirect, setProperty_valid) {
  auto layer = nntrainer::createLayer<nntrainer::GRULayer>({});
  EXPECT_NO_THROW(layer->setProperty({"unit=32"}));
}

/**
 * @brief Test getType returns correct type
 */
TEST(GRULayerDirect, getType) {
  auto layer = nntrainer::createLayer<nntrainer::GRULayer>({});
  EXPECT_EQ(layer->getType(), "gru");
}

/**
 * @brief Test supportBackwarding returns true
 */
TEST(GRULayerDirect, supportBackwarding) {
  auto layer = nntrainer::createLayer<nntrainer::GRULayer>({});
  EXPECT_TRUE(layer->supportBackwarding());
}

/**
 * @brief Test with various unit sizes and properties
 */
TEST(GRULayerDirect, setProperty_various) {
  auto layer1 = nntrainer::createLayer<nntrainer::GRULayer>({});
  EXPECT_NO_THROW(layer1->setProperty({"unit=64"}));

  auto layer2 = nntrainer::createLayer<nntrainer::GRULayer>({});
  EXPECT_NO_THROW(layer2->setProperty({"unit=128", "return_sequences=true"}));

  auto layer3 = nntrainer::createLayer<nntrainer::GRULayer>({});
  EXPECT_NO_THROW(layer3->setProperty({"unit=256", "hidden_state_activation=tanh"}));
}
