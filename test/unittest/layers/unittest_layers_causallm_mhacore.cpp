// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Eunju Yang <ej.yang@samsung.com>
 *
 * @file unittest_layers_causallm_mhacore.cpp
 * @date 01 Apr 2026
 * @brief CausalLM MHA Core Layer Forward/Backward Test
 * @see	https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <mha_core.h>

auto causallm_mhacore_golden = LayerGoldenTestParamType(
  nntrainer::createLayer<causallm::MHACoreLayer>,
  {"num_heads=4", "num_heads_kv=2", "max_timestep=16", "is_causal=true",
   "rope_theta=10000", "max_position_embeddings=128"},
  "2:1:4:32,2:1:4:16,2:1:4:16", "causallm_mhacore.nnlayergolden",
  LayerGoldenTestParamOptions::SKIP_CALC_GRAD |
    LayerGoldenTestParamOptions::SKIP_CALC_DERIV |
    LayerGoldenTestParamOptions::SKIP_COSINE_SIMILARITY,
  "nchw", "fp32", "fp32");

GTEST_PARAMETER_TEST(CausalLMMHACore, LayerGoldenTest,
                     ::testing::Values(causallm_mhacore_golden));
