// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Donghak Park <donghak.park@samsung.com>
 *
 * @file unittest_layers_gather.cpp
 * @date 12 December 2025
 * @brief Gather Layer Test
 * @see https://github.com/nnstreamer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <gather_layer.h>
#include <layers_common_tests.h>

auto semantic_gather = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::GatherLayer>, nntrainer::GatherLayer::type,
  {"axis=1"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 2);

GTEST_PARAMETER_TEST(Gather, LayerSemantics,
                     ::testing::Values(semantic_gather));

/**
 * @brief Direct test for GatherLayer setProperty
 */
TEST(GatherLayerDirect, setProperty_valid) {
  auto layer = nntrainer::createLayer<nntrainer::GatherLayer>({});
  EXPECT_NO_THROW(layer->setProperty({"axis=2"}));
}

/**
 * @brief Test setProperty with invalid property
 */
TEST(GatherLayerDirect, setProperty_invalid) {
  auto layer = nntrainer::createLayer<nntrainer::GatherLayer>({});
  EXPECT_THROW(layer->setProperty({"invalid_prop=value"}),
               nntrainer::exception::not_supported);
}

/**
 * @brief Test getType returns correct type
 */
TEST(GatherLayerDirect, getType) {
  auto layer = nntrainer::createLayer<nntrainer::GatherLayer>({});
  EXPECT_EQ(layer->getType(), "gather");
}

/**
 * @brief Test supportBackwarding returns true
 */
TEST(GatherLayerDirect, supportBackwarding) {
  auto layer = nntrainer::createLayer<nntrainer::GatherLayer>({});
  EXPECT_TRUE(layer->supportBackwarding());
}
