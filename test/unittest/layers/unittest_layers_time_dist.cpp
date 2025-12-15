// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Donghak Park <donghak.park@samsung.com>
 *
 * @file unittest_layers_time_dist.cpp
 * @date 12 December 2025
 * @brief Time Distributed Layer Test
 * @see https://github.com/nnstreamer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <time_dist.h>

// TimeDistLayer requires a distributed layer to be set, which makes
// LayerSemantics testing complex. We only test direct APIs here.

/**
 * @brief Test getType returns correct type
 */
TEST(TimeDistLayerDirect, getType) {
  auto layer = nntrainer::createLayer<nntrainer::TimeDistLayer>({});
  EXPECT_EQ(layer->getType(), "time_dist");
}
