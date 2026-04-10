// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Eunju Yang <ej.yang@samsung.com>
 *
 * @file unittest_layers_causallm_lm_head.cpp
 * @date 31 Mar 2026
 * @brief CausalLM LM Head Layer Backward Test
 * @see	https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <lm_head.h>

auto causallm_lmhead_golden = LayerGoldenTestParamType(
  nntrainer::createLayer<causallm::LmHeadLayer>,
  {"unit=5", "disable_bias=true"}, "2:1:1:10",
  "causallm_lmhead.nnlayergolden",
  LayerGoldenTestParamOptions::SKIP_CALC_GRAD, "nchw", "fp32", "fp32");

GTEST_PARAMETER_TEST(CausalLMLmHead, LayerGoldenTest,
                     ::testing::Values(causallm_lmhead_golden));
