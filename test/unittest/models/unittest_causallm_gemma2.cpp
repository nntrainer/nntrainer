// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_causallm_gemma2.cpp
 * @date   28 Jul 2026
 * @brief  Tiny Gemma2 CausalLM config-plumbing unit tests
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <causallm_test_utils.h>

#include <gtest/gtest.h>

#include <gemma2_causallm.h>

#include <memory>

namespace {

constexpr int tiny_gemma2_num_layers = 2;

/**
 * @brief Make the tiny Gemma2 model config
 */
causallm::json makeTinyGemma2Config() {
  return {
    {"architectures", {"Gemma2ForCausalLM"}},
    {"bos_token_id", 0},
    {"eos_token_id", {31}},
    {"head_dim", 8},
    {"hidden_size", 64},
    {"intermediate_size", 64},
    {"is_causal", true},
    {"max_position_embeddings", 8},
    {"num_attention_heads", 8},
    {"num_hidden_layers", tiny_gemma2_num_layers},
    {"num_key_value_heads", 4},
    {"rms_norm_eps", 1e-6},
    {"rope_theta", 10000},
    {"sliding_window", 4},
    {"sliding_window_pattern", 2},
    {"tie_word_embeddings", true},
    {"vocab_size", 32},
  };
}

/**
 * @brief Tiny Gemma2 probe exposing config-derived attention parameters
 *
 * Construct-only: it never compiles a graph nor loads weights, so every
 * assertion observes exactly what the constructor chain parsed out of cfg.
 * Transformer is a virtual base of every CausalLM model, so this most-derived
 * class initializes it with Gemma2's sanitized configs (mirroring
 * TinyGemma3CausalLM in unittest_causallm_gemma3.cpp).
 */
class TinyGemma2ConfigProbe final : public causallm::Gemma2CausalLM {
public:
  /**
   * @brief Construct a tiny Gemma2 config probe
   */
  TinyGemma2ConfigProbe(causallm::json &cfg, causallm::json &generation_cfg,
                        causallm::json &nntr_cfg) :
    causallm::Transformer(sanitizeConfig(cfg),
                          sanitizeGenerationConfig(generation_cfg, cfg),
                          nntr_cfg, causallm::ModelType::CAUSALLM),
    causallm::Gemma2CausalLM(cfg, generation_cfg, nntr_cfg) {}

  /**
   * @brief Attention-logit soft-cap as parsed by the constructor chain
   */
  float attnLogitSoftcapping() const { return ATTN_LOGIT_SOFTCAPPING; }

  /**
   * @brief Per-layer sliding/full attention pattern derived from the config
   */
  const std::vector<std::string> &layerTypes() const { return layer_types; }
};

/**
 * @brief Construct a tiny Gemma2 probe from a model config
 */
std::unique_ptr<TinyGemma2ConfigProbe> makeGemma2Probe(causallm::json cfg) {
  auto generation_cfg = causallm_test::makeTinyGenerationConfig();
  auto nntr_cfg = causallm_test::makeTinyCtorOnlyNntrainerConfig();
  return std::make_unique<TinyGemma2ConfigProbe>(cfg, generation_cfg, nntr_cfg);
}

} // namespace

/**
 * @brief config.json "attn_logit_softcapping" must reach the model object
 *
 * Gemma2 uses attention-logit soft-capping and every real gemma2 config ships
 * "attn_logit_softcapping": 50.0. This asserts the CONTRACT -- a cfg key the
 * model declares support for actually lands in the member every consumer
 * reads -- not where the parse is written, so it stays valid if the parse
 * moves again.
 *
 * Regression guard: upstream this parse lived only in
 * Gemma2Transformer::setupParameters, which is unreachable. setupParameters is
 * dispatched from base-class CONSTRUCTOR bodies (Transformer:: and CausalLM::),
 * and a virtual call inside a base constructor resolves to the BASE override --
 * never a derived one. Gemma2's ctor bodies add no call of their own, so the
 * override never ran and gemma2 silently used the 0.0f default; every consumer
 * is gated `if (attn_logit_softcapping > 0.0f)`, so `c*tanh(s/c)` was skipped
 * on every layer and every backend.
 */
TEST(Gemma2ConfigPlumbingTest, AttnLogitSoftcappingReachesModel) {
  auto cfg = makeTinyGemma2Config();
  cfg["attn_logit_softcapping"] = 50.0;

  auto probe = makeGemma2Probe(cfg);

  EXPECT_FLOAT_EQ(probe->attnLogitSoftcapping(), 50.0f);
}

/**
 * @brief A config without the key keeps the no-soft-cap default
 */
TEST(Gemma2ConfigPlumbingTest, AbsentAttnLogitSoftcappingKeepsDefault) {
  auto probe = makeGemma2Probe(makeTinyGemma2Config());

  EXPECT_FLOAT_EQ(probe->attnLogitSoftcapping(), 0.0f);
}

/**
 * @brief An explicitly null key keeps the no-soft-cap default
 */
TEST(Gemma2ConfigPlumbingTest, NullAttnLogitSoftcappingKeepsDefault) {
  auto cfg = makeTinyGemma2Config();
  cfg["attn_logit_softcapping"] = nullptr;

  auto probe = makeGemma2Probe(cfg);

  EXPECT_FLOAT_EQ(probe->attnLogitSoftcapping(), 0.0f);
}

/**
 * @brief sanitizeConfig derives the alternating sliding/full attention pattern
 *
 * HF gemma2 configs carry neither "sliding_window_pattern" nor "layer_types",
 * so Gemma2Transformer::sanitizeConfig injects the canonical period-2 pattern
 * and expands it. createAttention reads layer_types to decide which layers get
 * a finite sliding_window, so an empty vector would silently make every layer
 * global.
 */
TEST(Gemma2ConfigPlumbingTest, SanitizeConfigDerivesAlternatingLayerTypes) {
  auto cfg = makeTinyGemma2Config();
  cfg.erase("sliding_window_pattern");

  auto probe = makeGemma2Probe(cfg);

  const auto &types = probe->layerTypes();
  ASSERT_EQ(types.size(), static_cast<size_t>(tiny_gemma2_num_layers));
  EXPECT_EQ(types[0], "sliding_attention");
  EXPECT_EQ(types[1], "full_attention");
}
