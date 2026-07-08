// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_causallm_gpt_oss_cached_slim.cpp
 * @date   15 June 2026
 * @brief  Tiny GptOssCachedSlim CausalLM model unit tests
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jungwon Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <causallm_test_utils.h>

#include <gtest/gtest.h>

#include <gptoss_cached_slim_causallm.h>
#include <layer.h>
#include <layer_context.h>

namespace {

using TinyGptOssCachedSlimCausalLM =
  causallm_test::CausalLMTestAdapter<causallm::GptOssCachedSlimCausalLM>;

/**
 * @brief Make the tiny GptOssCachedSlim model config
 */
causallm::json makeTinyGptOssCachedSlimConfig() {
  return {
    {"architectures", {"GptOssCachedSlimCausalLM"}},
    {"bos_token_id", 0},
    {"eos_token_id", {31}},
    {"head_dim", 8},
    {"hidden_size", 64},
    {"intermediate_size", 64},
    {"is_causal", true},
    {"layer_types", {"sliding_attention"}},
    {"max_position_embeddings", 8},
    {"moe_intermediate_size", 64},
    {"num_attention_heads", 8},
    {"num_hidden_layers", 1},
    {"num_key_value_heads", 4},
    {"num_local_experts", 4},
    {"num_experts_per_tok", 2},
    {"rms_norm_eps", 1e-5},
    {"rope_theta", 10000},
    {"rope_scaling", {{"factor", 1.0}, {"type", "yarn"}}},
    {"sliding_window", 4},
    {"tie_word_embeddings", true},
    {"vocab_size", 32},
  };
}

/**
 * @brief Verify that GptOssCachedSlimCausalLM can be instantiated and that
 *        greedy generation selects the argmax token from supplied logits.
 *
 * WeightRoundTrip and PromptProducesExpectedLogits are not included here
 * because GptOssMoELayerCached::forwarding() calls Tensor::activate() for
 * lazy mmap-based expert weight loading, which is incompatible with the
 * in-memory weight setup used by the shared test helpers.
 */
TEST(GptOssCachedSlimTinyModelTest, GreedyGenerationSelectsArgmaxLogit) {
  auto tokenizer_path =
    causallm_test::makeTinyCausalLMFiles("GptOssCachedSlimTinyModelTest",
                                         "GreedyGenerationSelectsArgmaxLogit",
                                         "GptOssCachedSlim_FP32")
      .tokenizer_path;

  auto model_cfg = makeTinyGptOssCachedSlimConfig();
  auto gen_cfg = causallm_test::makeTinyGenerationConfig();
  auto nntr_cfg = causallm_test::makeTinyNntrainerConfig(
    tokenizer_path, causallm_test::makeTinyFp32DataType());

  auto model = std::make_unique<TinyGptOssCachedSlimCausalLM>(
    model_cfg, gen_cfg, nntr_cfg);
  causallm_test::expectGreedyGenerationSelectsArgmax(*model);
}

#ifdef ENABLE_FP16
TEST(GptOssCachedSlimTinyModelTest,
     GreedyGenerationSelectsArgmaxLogit_Q40FP16) {
  auto tokenizer_path =
    causallm_test::makeTinyCausalLMFiles("GptOssCachedSlimTinyModelTest",
                                         "GreedyGenerationSelectsArgmaxLogit",
                                         "GptOssCachedSlim_Q40_FP16")
      .tokenizer_path;

  auto model_cfg = makeTinyGptOssCachedSlimConfig();
  auto gen_cfg = causallm_test::makeTinyGenerationConfig();
  auto nntr_cfg = causallm_test::makeTinyNntrainerConfig(
    tokenizer_path, causallm_test::makeTinyQ40Fp16DataType());

  auto model = std::make_unique<TinyGptOssCachedSlimCausalLM>(
    model_cfg, gen_cfg, nntr_cfg);
  causallm_test::expectGreedyGenerationSelectsArgmax(*model);
}
#endif

} // namespace
