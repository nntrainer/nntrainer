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

// Test for FP32 (forward and backward both)
auto swiglu_golden = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::SwiGLULayer>, {}, "1:3:3:3,1:3:3:3",
  "swiglu_batch1.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT, "nchw",
  "fp32", "fp32");

GTEST_PARAMETER_TEST(SwiGLU, LayerGoldenTest, ::testing::Values(swiglu_golden));

#ifdef ENABLE_FP16
// Test for FP16 forward (backward not implemented yet)
auto swiglufp16_golden =
  LayerGoldenTestParamType(nntrainer::createLayer<nntrainer::SwiGLULayer>, {},
                           "1:3:3:3,1:3:3:3", "swiglufp16_batch1.nnlayergolden",
                           LayerGoldenTestParamOptions::DEFAULT |
                             LayerGoldenTestParamOptions::SKIP_CALC_DERIV,
                           "nchw", "fp16", "fp16");

GTEST_PARAMETER_TEST(SwiGLU16, LayerGoldenTest,
                     ::testing::Values(swiglufp16_golden));
#endif
