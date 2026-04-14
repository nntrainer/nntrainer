// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Eunju Yang <ej.yang@samsung.com>
 *
 * @file unittest_layers_causallm_embedding.cpp
 * @date 05 Apr 2026
 * @brief CausalLM Embedding Layer Backward Test
 * @see	https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include "embedding_layer.h"

auto causallm_embedding_golden = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::EmbeddingLayer>, {"in_dim=100", "out_dim=10"},
  "2:1:1:10", "causallm_embedding_layer.nnlayergolden",
  LayerGoldenTestParamOptions::USE_INC_FORWARD,
  "nchw", "fp32", "fp32");

GTEST_PARAMETER_TEST(CausalLMEmbedding, LayerGoldenTest,
                     ::testing::Values(causallm_embedding_golden));