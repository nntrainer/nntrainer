// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Donghak Park <donghak.park@samsung.com>
 *
 * @file unittest_layers_lstmcell_direct.cpp
 * @date 12 December 2025
 * @brief LSTMCell Layer Direct Unit Tests
 * @see https://github.com/nnstreamer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <lstmcell.h>

/**
 * @brief Direct test for LSTMCellLayer setProperty
 */
TEST(LSTMCellLayerDirect, setProperty_valid) {
  auto layer = nntrainer::createLayer<nntrainer::LSTMCellLayer>({});
  EXPECT_NO_THROW(layer->setProperty({"unit=32"}));
}

/**
 * @brief Test getType returns correct type
 */
TEST(LSTMCellLayerDirect, getType) {
  auto layer = nntrainer::createLayer<nntrainer::LSTMCellLayer>({});
  EXPECT_EQ(layer->getType(), "lstmcell");
}

/**
 * @brief Test supportBackwarding returns true
 */
TEST(LSTMCellLayerDirect, supportBackwarding) {
  auto layer = nntrainer::createLayer<nntrainer::LSTMCellLayer>({});
  EXPECT_TRUE(layer->supportBackwarding());
}

/**
 * @brief Test with various unit sizes
 */
TEST(LSTMCellLayerDirect, setProperty_various) {
  auto layer1 = nntrainer::createLayer<nntrainer::LSTMCellLayer>({});
  EXPECT_NO_THROW(layer1->setProperty({"unit=64"}));

  auto layer2 = nntrainer::createLayer<nntrainer::LSTMCellLayer>({});
  EXPECT_NO_THROW(layer2->setProperty({"unit=128"}));
}

/**
 * @brief Test setProperty with invalid property name
 */
TEST(LSTMCellLayerDirect, setProperty_invalid_name) {
  auto layer = nntrainer::createLayer<nntrainer::LSTMCellLayer>({});
  EXPECT_THROW(layer->setProperty({"invalid_property=100"}), std::exception);
}

/**
 * @brief Test setProperty with invalid value type
 */
TEST(LSTMCellLayerDirect, setProperty_invalid_type) {
  auto layer = nntrainer::createLayer<nntrainer::LSTMCellLayer>({});
  // unit expects a positive integer, passing a string
  EXPECT_THROW(layer->setProperty({"unit=not_a_number"}), std::exception);
}
