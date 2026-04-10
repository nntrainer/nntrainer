// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Niket Agarwal <niket.a@samsung.com>
 *
 * @file unittest_layers_rms_norm.cpp
 * @date 10 Apr 2026
 * @brief RMS Norm Layer Test
 * @see	https://github.com/nntrainer/nntrainer
 * @author Niket Agarwal <niket.a@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <gtest/gtest.h>
#include <layers_common_tests.h>
#include <rms_norm.h>
#include <tuple>

auto rms_norm_golden = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::RMSNormLayer>, {"epsilon=0.001"}, "2:3:3:3",
  "rms_normtest_gamma_trainable.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

GTEST_PARAMETER_TEST(RMSNorm, LayerGoldenTest,
                     ::testing::Values(rms_norm_golden));
