// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Donghak Park <donghak.park@samsung.com>
 *
 * @file unittest_layers_reduce_mean_direct.cpp
 * @date 12 December 2025
 * @brief Reduce Mean Layer Direct Unit Tests
 * @see https://github.com/nnstreamer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <reduce_mean_layer.h>
#include <layers_common_tests.h>

/**
 * @brief Direct test for ReduceMeanLayer setProperty
 */
TEST(ReduceMeanLayerDirect, setProperty_valid) {
  auto layer = nntrainer::createLayer<nntrainer::ReduceMeanLayer>({});
  EXPECT_NO_THROW(layer->setProperty({"axis=1"}));
}

/**
 * @brief Test getType returns correct type
 */
TEST(ReduceMeanLayerDirect, getType) {
  auto layer = nntrainer::createLayer<nntrainer::ReduceMeanLayer>({});
  EXPECT_EQ(layer->getType(), "reduce_mean");
}

/**
 * @brief Test supportBackwarding returns true
 */
TEST(ReduceMeanLayerDirect, supportBackwarding) {
  auto layer = nntrainer::createLayer<nntrainer::ReduceMeanLayer>({});
  EXPECT_TRUE(layer->supportBackwarding());
}

/**
 * @brief Test with different axes
 */
TEST(ReduceMeanLayerDirect, setProperty_various) {
  auto layer1 = nntrainer::createLayer<nntrainer::ReduceMeanLayer>({});
  EXPECT_NO_THROW(layer1->setProperty({"axis=1"}));

  auto layer2 = nntrainer::createLayer<nntrainer::ReduceMeanLayer>({});
  EXPECT_NO_THROW(layer2->setProperty({"axis=2"}));

  auto layer3 = nntrainer::createLayer<nntrainer::ReduceMeanLayer>({});
  EXPECT_NO_THROW(layer3->setProperty({"axis=3"}));
}
