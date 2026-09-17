// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_causallm_lfm2_cached_slim_moe.cpp
 * @date   06 July 2026
 * @brief  Tiny LFM2 Cached-Slim MoE CausalLM model unit tests
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jungwon Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <causallm_test_utils.h>

#include <gtest/gtest.h>

#if !defined(_WIN32)

#include <layer.h>
#include <layer_context.h>
#include <lfm2_cached_slim_moe_causallm.h>

#include <map>

namespace {

using TinyLfm2CachedSlimMoeCausalLM =
  causallm_test::CausalLMTestAdapter<causallm::Lfm2CachedSlimMoeCausalLM>;

/**
 * @brief Make the tiny LFM2 Cached-Slim MoE model config
 */
causallm::json makeTinyLfm2CachedSlimMoeConfig() {
  return {
    {"architectures", {"Lfm2CachedSlimMoeForCausalLM"}},
    {"bos_token_id", 0},
    {"conv_L_cache", 3},
    {"conv_bias", false},
    {"conv_dim", 64},
    {"conv_dim_out", 64},
    {"eos_token_id", {31}},
    {"head_dim", 8},
    {"hidden_size", 64},
    {"intermediate_size", 64},
    {"is_causal", true},
    {"layer_types", {"attention", "conv"}},
    {"max_position_embeddings", 8},
    {"num_attention_heads", 8},
    {"num_hidden_layers", 2},
    {"num_key_value_heads", 4},
    {"num_experts", 4},
    {"num_experts_per_tok", 2},
    {"moe_intermediate_size", 64},
    {"num_dense_layers", 0},
    {"norm_topk_prob", true},
    {"use_expert_bias", true},
    {"routed_scaling_factor", 1.0},
    {"rms_norm_eps", 1e-6},
    {"rope_theta", 10000},
    {"tie_word_embeddings", true},
    {"vocab_size", 32},
  };
}

/**
 * @brief Verify that Lfm2CachedSlimMoe can be instantiated and that greedy
 *        generation selects the argmax token from supplied logits.
 *
 * See unittest_causallm_lfm2_slim_moe.cpp for why WeightRoundTrip /
 * PromptProducesExpectedLogits are not included here.
 */
TEST(Lfm2CachedSlimMoeTinyModelTest, GreedyGenerationSelectsArgmaxLogit) {
  auto tokenizer_path =
    causallm_test::makeTinyCausalLMFiles("Lfm2CachedSlimMoeTinyModelTest",
                                         "GreedyGenerationSelectsArgmaxLogit",
                                         "Lfm2CachedSlimMoe_FP32")
      .tokenizer_path;

  auto model_cfg = makeTinyLfm2CachedSlimMoeConfig();
  auto gen_cfg = causallm_test::makeTinyGenerationConfig();
  auto nntr_cfg = causallm_test::makeTinyNntrainerConfig(
    tokenizer_path, causallm_test::makeTinyFp32DataType());

  auto model = std::make_unique<TinyLfm2CachedSlimMoeCausalLM>(model_cfg,
                                                               gen_cfg, nntr_cfg);
  causallm_test::expectGreedyGenerationSelectsArgmax(*model);
}

} // namespace

#endif // !_WIN32
