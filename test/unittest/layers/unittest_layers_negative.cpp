// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Sumon Nath <sumon.nath@samsung.com>
 *
 * @file unittest_layers_neg.cpp
 * @date 5 July 2025
 * @brief Neg Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Sumon Nath <sumon.nath@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <negative_layer.h>

auto semantic_negative = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::NegativeLayer>,
  nntrainer::NegativeLayer::type, {},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

auto semantic_negative_multi = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::NegativeLayer>,
  nntrainer::NegativeLayer::type, {},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 2);

GTEST_PARAMETER_TEST(Negative, LayerSemantics,
                     ::testing::Values(semantic_negative,
                                       semantic_negative_multi));
