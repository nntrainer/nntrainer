// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Donghak Park <donghak.park@samsung.com>
 *
 * @file unittest_layers_zoneout_lstmcell_direct.cpp
 * @date 12 December 2025
 * @brief ZoneoutLSTMCell Layer Direct Unit Tests
 * @see https://github.com/nnstreamer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <zoneout_lstmcell.h>
#include <layers_common_tests.h>

/**
 * @brief Direct test for ZoneoutLSTMCellLayer setProperty
 */
TEST(ZoneoutLSTMCellLayerDirect, setProperty_valid) {
  auto layer = nntrainer::createLayer<nntrainer::ZoneoutLSTMCellLayer>({});
  EXPECT_NO_THROW(layer->setProperty({"unit=32"}));
}

/**
 * @brief Test getType returns correct type
 */
TEST(ZoneoutLSTMCellLayerDirect, getType) {
  auto layer = nntrainer::createLayer<nntrainer::ZoneoutLSTMCellLayer>({});
  EXPECT_EQ(layer->getType(), "zoneout_lstmcell");
}

/**
 * @brief Test supportBackwarding returns true
 */
TEST(ZoneoutLSTMCellLayerDirect, supportBackwarding) {
  auto layer = nntrainer::createLayer<nntrainer::ZoneoutLSTMCellLayer>({});
  EXPECT_TRUE(layer->supportBackwarding());
}

/**
 * @brief Test with various properties
 */
TEST(ZoneoutLSTMCellLayerDirect, setProperty_various) {
  auto layer1 = nntrainer::createLayer<nntrainer::ZoneoutLSTMCellLayer>({});
  EXPECT_NO_THROW(layer1->setProperty({"unit=64"}));

  auto layer2 = nntrainer::createLayer<nntrainer::ZoneoutLSTMCellLayer>({});
  EXPECT_NO_THROW(layer2->setProperty({"unit=128", "hidden_state_zoneout_rate=0.1"}));
}
