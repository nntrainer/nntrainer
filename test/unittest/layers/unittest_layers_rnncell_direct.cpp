// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Donghak Park <donghak.park@samsung.com>
 *
 * @file unittest_layers_rnncell_direct.cpp
 * @date 12 December 2025
 * @brief RNNCell Layer Direct Unit Tests
 * @see https://github.com/nnstreamer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <rnncell.h>
#include <layers_common_tests.h>

/**
 * @brief Direct test for RNNCellLayer setProperty
 */
TEST(RNNCellLayerDirect, setProperty_valid) {
  auto layer = nntrainer::createLayer<nntrainer::RNNCellLayer>({});
  EXPECT_NO_THROW(layer->setProperty({"unit=32"}));
}

/**
 * @brief Test getType returns correct type
 */
TEST(RNNCellLayerDirect, getType) {
  auto layer = nntrainer::createLayer<nntrainer::RNNCellLayer>({});
  EXPECT_EQ(layer->getType(), "rnncell");
}

/**
 * @brief Test supportBackwarding returns true
 */
TEST(RNNCellLayerDirect, supportBackwarding) {
  auto layer = nntrainer::createLayer<nntrainer::RNNCellLayer>({});
  EXPECT_TRUE(layer->supportBackwarding());
}

/**
 * @brief Test with various unit sizes
 */
TEST(RNNCellLayerDirect, setProperty_various) {
  auto layer1 = nntrainer::createLayer<nntrainer::RNNCellLayer>({});
  EXPECT_NO_THROW(layer1->setProperty({"unit=64"}));

  auto layer2 = nntrainer::createLayer<nntrainer::RNNCellLayer>({});
  EXPECT_NO_THROW(layer2->setProperty({"unit=128"}));
}
