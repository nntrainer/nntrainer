// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Donghak Park <donghak.park@samsung.com>
 *
 * @file unittest_layers_upsample2d_direct.cpp
 * @date 12 December 2025
 * @brief Upsample2D Layer Direct Unit Tests
 * @see https://github.com/nnstreamer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <upsample2d_layer.h>

/**
 * @brief Test getType returns correct type
 */
TEST(Upsample2dLayerDirect, getType) {
  auto layer = nntrainer::createLayer<nntrainer::Upsample2dLayer>({});
  EXPECT_EQ(layer->getType(), "upsample2d");
}

/**
 * @brief Test supportBackwarding returns true
 */
TEST(Upsample2dLayerDirect, supportBackwarding) {
  auto layer = nntrainer::createLayer<nntrainer::Upsample2dLayer>({});
  EXPECT_TRUE(layer->supportBackwarding());
}

/**
 * @brief Test setProperty with invalid property name
 */
TEST(Upsample2dLayerDirect, setProperty_invalid_name) {
  auto layer = nntrainer::createLayer<nntrainer::Upsample2dLayer>({});
  EXPECT_THROW(layer->setProperty({"invalid_property=100"}), std::exception);
}

/**
 * @brief Test setProperty with invalid value type
 */
TEST(Upsample2dLayerDirect, setProperty_invalid_type) {
  auto layer = nntrainer::createLayer<nntrainer::Upsample2dLayer>({});
  // upsample expects a valid format, passing invalid
  EXPECT_THROW(layer->setProperty({"upsample_mode=not_valid"}), std::exception);
}
