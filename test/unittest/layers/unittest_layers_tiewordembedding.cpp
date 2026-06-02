// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Eunju Yang <ej.yang@samsung.com>
 *
 * @file unittest_layers_tiewordembedding.cpp
 * @date 01 Apr 2026
 * @brief CausalLM TieWordEmbedding Layer Backward Test (lm_head mode)
 * @see	https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <tuple>

#include <gtest/gtest.h>

#include "tie_word_embedding.h"
#include <layers_common_tests.h>

auto causallm_tiewordembedding_golden = LayerGoldenTestParamType(
  nntrainer::createLayer<causallm::TieWordEmbedding>,
  {"in_dim=100", "out_dim=10"}, "2:1:1:10", "tiewordembedding.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

GTEST_PARAMETER_TEST(CausalLMTieWordEmbedding, LayerGoldenTest,
                     ::testing::Values(causallm_tiewordembedding_golden));