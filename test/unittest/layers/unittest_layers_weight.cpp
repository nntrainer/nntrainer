// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file unittest_layers_weight.cpp
 * @date 30 July 2024
 * @brief Weight Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <weight_layer.h>

auto semantic_weight = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::WeightLayer>, nntrainer::WeightLayer::type,
  {"dim=1:1:1", "tensor_dtype=FP32", "weight_name=unittest_weight1"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

auto semantic_weight_multi = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::WeightLayer>, nntrainer::WeightLayer::type,
  {"dim=1:1:1", "tensor_dtype=FP32", "weight_name=unittest_weight2"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 2);

GTEST_PARAMETER_TEST(Weight, LayerSemantics,
                     ::testing::Values(semantic_weight, semantic_weight_multi));
