// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Eunju Yang <ej.yang@samsung.com>
 *
 * @file unittest_layers_causallm_swiglu.cpp
 * @date 01 Apr 2026
 * @brief CausalLM SwiGLU Layer Backward Test
 * @see	https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <swiglu.h>

auto causallm_swiglu_golden = LayerGoldenTestParamType(
  nntrainer::createLayer<causallm::SwiGLULayer>, {}, "2:1:1:10,2:1:1:10",
  "causallm_swiglu.nnlayergolden",
  LayerGoldenTestParamOptions::SKIP_CALC_GRAD, "nchw", "fp32", "fp32");

GTEST_PARAMETER_TEST(CausalLMSwiGLU, LayerGoldenTest,
                     ::testing::Values(causallm_swiglu_golden));
