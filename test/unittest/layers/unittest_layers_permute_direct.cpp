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
#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <permute_layer.h>

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
 * @brief Test setProperty with invalid property name
 */
TEST(PermuteLayerDirect, setProperty_invalid_name) {
  auto layer = nntrainer::createLayer<nntrainer::PermuteLayer>({});
  EXPECT_THROW(layer->setProperty({"invalid_property=100"}), std::exception);
}

/**
 * @brief Test setProperty with invalid value type
 */
TEST(PermuteLayerDirect, setProperty_invalid_type) {
  auto layer = nntrainer::createLayer<nntrainer::PermuteLayer>({});
  // direction expects a valid permutation format, passing invalid
  EXPECT_THROW(layer->setProperty({"direction=not_valid"}), std::exception);
}
