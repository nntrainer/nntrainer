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
#include <tuple>

#include <gtest/gtest.h>

#include <upsample2d_layer.h>
#include <layers_common_tests.h>

/**
 * @brief Direct test for Upsample2dLayer setProperty
 */
TEST(Upsample2dLayerDirect, setProperty_valid) {
  auto layer = nntrainer::createLayer<nntrainer::Upsample2dLayer>({});
  EXPECT_NO_THROW(layer->setProperty({"upsample=2x2"}));
}

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
 * @brief Test with different upsample factors
 */
TEST(Upsample2dLayerDirect, setProperty_various) {
  auto layer1 = nntrainer::createLayer<nntrainer::Upsample2dLayer>({});
  EXPECT_NO_THROW(layer1->setProperty({"upsample=3x3"}));

  auto layer2 = nntrainer::createLayer<nntrainer::Upsample2dLayer>({});
  EXPECT_NO_THROW(layer2->setProperty({"upsample=4x4"}));
}
