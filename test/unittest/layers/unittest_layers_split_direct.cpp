// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Donghak Park <donghak.park@samsung.com>
 *
 * @file unittest_layers_split_direct.cpp
 * @date 12 December 2025
 * @brief Split Layer Direct Unit Tests
 * @see https://github.com/nnstreamer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <split_layer.h>
#include <layers_common_tests.h>

/**
 * @brief Direct test for SplitLayer setProperty
 */
TEST(SplitLayerDirect, setProperty_valid) {
  auto layer = nntrainer::createLayer<nntrainer::SplitLayer>({});
  EXPECT_NO_THROW(layer->setProperty({"split_dimension=1", "split_number=2"}));
}

/**
 * @brief Test getType returns correct type
 */
TEST(SplitLayerDirect, getType) {
  auto layer = nntrainer::createLayer<nntrainer::SplitLayer>({});
  EXPECT_EQ(layer->getType(), "split");
}

/**
 * @brief Test supportBackwarding returns true
 */
TEST(SplitLayerDirect, supportBackwarding) {
  auto layer = nntrainer::createLayer<nntrainer::SplitLayer>({});
  EXPECT_TRUE(layer->supportBackwarding());
}

/**
 * @brief Test with different dimensions
 */
TEST(SplitLayerDirect, setProperty_various) {
  auto layer1 = nntrainer::createLayer<nntrainer::SplitLayer>({});
  EXPECT_NO_THROW(layer1->setProperty({"split_dimension=2", "split_number=4"}));

  auto layer2 = nntrainer::createLayer<nntrainer::SplitLayer>({});
  EXPECT_NO_THROW(layer2->setProperty({"split_dimension=3", "split_number=2"}));
}
