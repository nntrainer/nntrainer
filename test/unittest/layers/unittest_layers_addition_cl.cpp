// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Yash Singh <yash.singh@samsung.com>
 *
 * @file unittest_layers_addition_cl.cpp
 * @date 17 May 2024
 * @brief Addition Layer Test
 * @see	https://github.com/nntrainer/nntrainer
 * @author Yash Singh <yash.singh@samsung.com>
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <addition_layer.h>
#include <layers_common_tests.h>

auto semantic_addition_gpu = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::AdditionLayer>,
  nntrainer::AdditionLayer::type, {},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

auto semantic_addition_multi_gpu = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::AdditionLayer>,
  nntrainer::AdditionLayer::type, {},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 2);

GTEST_PARAMETER_TEST(AdditionGPU, LayerSemanticsGpu,
                     ::testing::Values(semantic_addition_gpu,
                                       semantic_addition_multi_gpu));

auto addition_w32a32_gpu = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::AdditionLayer>, {}, "2:3:3:3,2:3:3:3",
  "added_w32a32.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT, "nchw",
  "fp32", "fp32");

auto addition_w32a32_2_gpu = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::AdditionLayer>, {}, "3:4:3:4,3:4:3:4",
  "added_w32a32_2.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT, "nchw",
  "fp32", "fp32");

GTEST_PARAMETER_TEST(AdditionGPU, LayerGoldenTest,
                     ::testing::Values(addition_w32a32_gpu,
                                       addition_w32a32_2_gpu));

#ifdef ENABLE_FP16
auto addition_w16a16_gpu = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::AdditionLayer>, {}, "2:3:3:3,2:3:3:3",
  "added_w16a16.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT, "nchw",
  "fp16", "fp16");

GTEST_PARAMETER_TEST(AdditionGPU16, LayerGoldenTest,
                     ::testing::Values(addition_w16a16_gpu));
#endif
