// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Donghak Park <donghak.park@samsung.com>
 *
 * @file unittest_layers_slice.cpp
 * @date 12 December 2025
 * @brief Slice Layer Test
 * @see https://github.com/nnstreamer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <slice_layer.h>
#include <layers_common_tests.h>

auto semantic_slice = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::SliceLayer>, nntrainer::SliceLayer::type,
  {"axis=1", "start_index=1", "end_index=3"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

GTEST_PARAMETER_TEST(Slice, LayerSemantics, ::testing::Values(semantic_slice));

/**
 * @brief Direct test for SliceLayer setProperty
 */
TEST(SliceLayerDirect, setProperty_valid) {
  auto layer = nntrainer::createLayer<nntrainer::SliceLayer>({});
  EXPECT_NO_THROW(
    layer->setProperty({"axis=2", "start_index=1", "end_index=5"}));
}

/**
 * @brief Test setProperty with invalid property
 */
TEST(SliceLayerDirect, setProperty_invalid) {
  auto layer = nntrainer::createLayer<nntrainer::SliceLayer>({});
  EXPECT_THROW(layer->setProperty({"invalid_prop=value"}),
               nntrainer::exception::not_supported);
}

/**
 * @brief Test getType returns correct type
 */
TEST(SliceLayerDirect, getType) {
  auto layer = nntrainer::createLayer<nntrainer::SliceLayer>({});
  EXPECT_EQ(layer->getType(), "slice");
}

/**
 * @brief Test supportBackwarding returns true
 */
TEST(SliceLayerDirect, supportBackwarding) {
  auto layer = nntrainer::createLayer<nntrainer::SliceLayer>({});
  EXPECT_TRUE(layer->supportBackwarding());
}
