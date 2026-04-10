// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Eunju Yang <ej.yang@samsung.com>
 *
 * @file unittest_layers_causallm_rmsnorm.cpp
 * @date 31 Mar 2026
 * @brief CausalLM RMS Norm Layer Backward Test
 * @see	https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <rms_norm.h>

auto causallm_rms_golden = LayerGoldenTestParamType(
  nntrainer::createLayer<causallm::RMSNormLayer>, {"epsilon=0.001"},
  "2:3:3:3", "causallm_rmsnorm.nnlayergolden",
  LayerGoldenTestParamOptions::SKIP_CALC_GRAD,
  "nchw", "fp32", "fp32");

GTEST_PARAMETER_TEST(CausalLMRMSNorm, LayerGoldenTest,
                     ::testing::Values(causallm_rms_golden));
