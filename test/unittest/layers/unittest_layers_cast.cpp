// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file unittest_layers_cast.cpp
 * @date 7 April 2025
 * @brief Cast Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <cast_layer.h>
#include <layers_common_tests.h>

auto semantic_cast = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::CastLayer>, nntrainer::CastLayer::type,
  {"tensor_dtype=FP16"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 2);

auto semantic_cast_multi = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::CastLayer>, nntrainer::CastLayer::type,
  {"tensor_dtype=FP16"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 2);

GTEST_PARAMETER_TEST(Cast, LayerSemantics,
                     ::testing::Values(semantic_cast, semantic_cast_multi));
