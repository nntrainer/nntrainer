// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Donghak Park <donghak.park@samsung.com>
 *
 * @file unittest_layers_grucell_direct.cpp
 * @date 12 December 2025
 * @brief GRUCell Layer Direct Unit Tests
 * @see https://github.com/nnstreamer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <grucell.h>
#include <layers_common_tests.h>

/**
 * @brief Direct test for GRUCellLayer setProperty
 */
TEST(GRUCellLayerDirect, setProperty_valid) {
  auto layer = nntrainer::createLayer<nntrainer::GRUCellLayer>({});
  EXPECT_NO_THROW(layer->setProperty({"unit=32"}));
}

/**
 * @brief Test getType returns correct type
 */
TEST(GRUCellLayerDirect, getType) {
  auto layer = nntrainer::createLayer<nntrainer::GRUCellLayer>({});
  EXPECT_EQ(layer->getType(), "grucell");
}

/**
 * @brief Test supportBackwarding returns true
 */
TEST(GRUCellLayerDirect, supportBackwarding) {
  auto layer = nntrainer::createLayer<nntrainer::GRUCellLayer>({});
  EXPECT_TRUE(layer->supportBackwarding());
}

/**
 * @brief Test with various unit sizes
 */
TEST(GRUCellLayerDirect, setProperty_various) {
  auto layer1 = nntrainer::createLayer<nntrainer::GRUCellLayer>({});
  EXPECT_NO_THROW(layer1->setProperty({"unit=64"}));

  auto layer2 = nntrainer::createLayer<nntrainer::GRUCellLayer>({});
  EXPECT_NO_THROW(layer2->setProperty({"unit=128"}));
}
