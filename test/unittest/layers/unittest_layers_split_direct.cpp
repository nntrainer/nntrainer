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
#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <split_layer.h>

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
 * @brief Test setProperty with invalid property name
 */
TEST(SplitLayerDirect, setProperty_invalid_name) {
  auto layer = nntrainer::createLayer<nntrainer::SplitLayer>({});
  EXPECT_THROW(layer->setProperty({"invalid_property=100"}), std::exception);
}

/**
 * @brief Test setProperty with invalid value type
 */
TEST(SplitLayerDirect, setProperty_invalid_type) {
  auto layer = nntrainer::createLayer<nntrainer::SplitLayer>({});
  // split_dimension expects an integer, passing a string
  EXPECT_THROW(layer->setProperty({"split_dimension=not_a_number"}),
               std::exception);
}
