// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Niket Agarwal <niket.a@samsung.com>
 *
 * @file unittest_layers_reshaped_rms_norm.cpp
 * @date 10 Apr 2026
 * @brief Reshaped RMS Norm Layer Test
 * @see		https://github.com/nntrainer/nntrainer
 * @author Niket Agarwal <niket.a@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <gtest/gtest.h>
#include <layers_common_tests.h>
#include <reshaped_rms_norm.h>
#include <tuple>

auto reshaped_rms_norm_golden = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::ReshapedRMSNormLayer>,
  {"epsilon=0.001", "feature_size=3"}, "2:3:3:9",
  "reshaped_rms_norm_test.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT,
  "nchw", "fp32", "fp32");

GTEST_PARAMETER_TEST(ReshapedRMSNorm, LayerGoldenTest,
                     ::testing::Values(reshaped_rms_norm_golden));
