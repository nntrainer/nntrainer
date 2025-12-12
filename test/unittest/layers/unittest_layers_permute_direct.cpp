// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Donghak Park <donghak.park@samsung.com>
 *
 * @file unittest_layers_permute_direct.cpp
 * @date 12 December 2025
 * @brief Permute Layer Direct Unit Tests
 * @see https://github.com/nnstreamer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <permute_layer.h>
#include <layers_common_tests.h>

/**
 * @brief Direct test for PermuteLayer setProperty
 */
TEST(PermuteLayerDirect, setProperty_valid) {
  auto layer = nntrainer::createLayer<nntrainer::PermuteLayer>({});
  EXPECT_NO_THROW(layer->setProperty({"direction=0:2:1:3"}));
}

/**
 * @brief Test getType returns correct type
 */
TEST(PermuteLayerDirect, getType) {
  auto layer = nntrainer::createLayer<nntrainer::PermuteLayer>({});
  EXPECT_EQ(layer->getType(), "permute");
}

/**
 * @brief Test supportBackwarding returns true
 */
TEST(PermuteLayerDirect, supportBackwarding) {
  auto layer = nntrainer::createLayer<nntrainer::PermuteLayer>({});
  EXPECT_TRUE(layer->supportBackwarding());
}

/**
 * @brief Test with different permutation directions
 */
TEST(PermuteLayerDirect, setProperty_various) {
  auto layer1 = nntrainer::createLayer<nntrainer::PermuteLayer>({});
  EXPECT_NO_THROW(layer1->setProperty({"direction=0:1:3:2"}));

  auto layer2 = nntrainer::createLayer<nntrainer::PermuteLayer>({});
  EXPECT_NO_THROW(layer2->setProperty({"direction=0:3:1:2"}));
}
