// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Niket Agarwal <niket.a@samsung.com>
 *
 * @file unittest_layers_swiglu_cl.cpp
 * @date 6th June 2024
 * @brief Swiglu Layer Test
 * @see	https://github.com/nntrainer/nntrainer
 * @author Niket Agarwal <niket.a@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <swiglu_layer.h>

auto semantic_swiglu_gpu = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::SwiGLULayer>, nntrainer::SwiGLULayer::type,
  {}, LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

GTEST_PARAMETER_TEST(SwigluGPU, LayerSemanticsGpu,
                     ::testing::Values(semantic_swiglu_gpu));

auto swiglu_basic_plain =
  LayerGoldenTestParamType(nntrainer::createLayer<nntrainer::SwiGLULayer>, {},
                           "2:3:3:3,2:3:3:3", "swiglu.nnlayergolden",
                           LayerGoldenTestParamOptions::SKIP_CALC_DERIV |
                             LayerGoldenTestParamOptions::SKIP_CALC_GRAD |
                             LayerGoldenTestParamOptions::USE_INC_FORWARD,
                           "nchw", "fp32", "fp32");

GTEST_PARAMETER_TEST(SwigluGPU, LayerGoldenTest,
                     ::testing::Values(swiglu_basic_plain));

#ifdef ENABLE_FP16
auto swiglu_basic_plain_w16a16 =
  LayerGoldenTestParamType(nntrainer::createLayer<nntrainer::SwiGLULayer>, {},
                           "2:3:3:3,2:3:3:3", "swiglufp16.nnlayergolden",
                           LayerGoldenTestParamOptions::SKIP_CALC_DERIV |
                             LayerGoldenTestParamOptions::SKIP_CALC_GRAD |
                             LayerGoldenTestParamOptions::USE_INC_FORWARD,
                           "nchw", "fp16", "fp16");

GTEST_PARAMETER_TEST(SwigluGPU16, LayerGoldenTest,
                     ::testing::Values(swiglu_basic_plain_w16a16));
#endif
