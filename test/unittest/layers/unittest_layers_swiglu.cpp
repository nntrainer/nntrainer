// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Niket Agarwal <niket.a@samsung.com>
 *
 * @file unittest_layers_swiglu.cpp
 * @date 07 Apr 2026
 * @brief SwiGLU Layer Test
 * @see	https://github.com/nntrainer/nntrainer
 * @author Niket Agarwal <niket.a@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <gtest/gtest.h>
#include <layers_common_tests.h>
#include <swiglu.h>
#include <tuple>

auto swiglu_golden = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::SwiGLULayer>, {}, "2:3:3:3,2:3:3:3",
  "swiglu.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32",
  "fp32");

GTEST_PARAMETER_TEST(SwiGLU, LayerGoldenTest, ::testing::Values(swiglu_golden));
