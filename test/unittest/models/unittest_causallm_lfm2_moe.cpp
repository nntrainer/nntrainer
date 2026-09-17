// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_causallm_lfm2_moe.cpp
 * @date   06 July 2026
 * @brief  Tiny LFM2-MoE CausalLM model unit tests
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jungwon Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <causallm_test_utils.h>

#include <gtest/gtest.h>

#include <layer.h>
#include <layer_context.h>
#include <lfm2_moe_causallm.h>

#include <map>

namespace {

constexpr int tiny_lfm2_moe_num_layers = 2;
constexpr int tiny_lfm2_moe_num_experts = 4;
constexpr int tiny_lfm2_moe_num_experts_per_tok = 2;

/**
 * @brief Tiny LFM2-MoE CausalLM adapter for common model tests
 */
using TinyLfm2MoeCausalLM =
  causallm_test::CausalLMTestAdapter<causallm::Lfm2MoeCausalLM>;

/**
 * @brief Populate deterministic tiny LFM2-MoE weights for golden token tests
 *
 * With num_dense_layers=0 every FFN is an MoE block. Zeroing all FP32 weights
 * inside the MoE layer (router gate, expert bias and every expert projection)
 * makes the MoE branch output zero, so the residual stream carries the hidden
 * state unchanged — the expected logits are identical to the dense LFM2 tiny
 * model. Attention/conv blocks are set up exactly as the dense LFM2 test.
 */
void setupLfm2MoeDeterministicWeights(TinyLfm2MoeCausalLM &model) {
  model.forEachLayer(
    [](ml::train::Layer &layer, nntrainer::RunLayerContext &context, void *) {
      if (layer.getName() == "output_of_causallm")
        return;

      if (layer.getType() == "lfm2_moe") {
        // Zero every FP32 weight (gate, expert_bias, expert projections) so the
        // MoE branch contributes zero to the residual stream.
        for (unsigned int i = 0; i < context.getNumWeights(); ++i) {
          auto &w = context.getWeight(i);
          if (w.getDataType() == ml::train::TensorDim::DataType::FP32)
            w.setValue(0.0f);
        }
        return;
      }

      for (unsigned int i = 0; i < context.getNumWeights(); ++i) {
        auto &weight = context.getWeight(i);
        if (weight.getDataType() != ml::train::TensorDim::DataType::FP32)
          continue;

        weight.setValue(0.0f);
        if (layer.getType() == "rms_norm" ||
            layer.getType() == "reshaped_rms_norm") {
          weight.setValue(1.0f);
        } else if (layer.getName() == "embedding0") {
          weight.setValue(0.0f);
          weight.setValue(0, 0, 1, 0, 1.0f);
          weight.setValue(0, 0, 4, 0, 2.0f);
        }
      }
    });
}

/**
 * @brief Make the tiny LFM2-MoE model config
 *
 * Uses layer_types=["attention","conv"] to exercise the MoE FFN on both the
 * attention and conv hybrid paths, with num_dense_layers=0 so both FFNs are
 * MoE blocks.
 */
causallm::json makeTinyLfm2MoeConfig() {
  return {
    {"architectures", {"Lfm2MoeForCausalLM"}},
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
    {"num_hidden_layers", tiny_lfm2_moe_num_layers},
    {"num_key_value_heads", 4},
    {"num_experts", tiny_lfm2_moe_num_experts},
    {"num_experts_per_tok", tiny_lfm2_moe_num_experts_per_tok},
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
 * @brief Make the expected tiny LFM2-MoE prefill logits
 *
 * Identical to the dense LFM2 tiny model: with all MoE branch weights zeroed
 * the residual stream is unchanged, so the final logits match the dense case.
 */
std::vector<float> makeExpectedLfm2MoeLogits() {
  std::vector<float> logits(32, 0.0f);
  logits[1] = 7.9999361f;
  logits[4] = 15.9998722f;
  return logits;
}

/**
 * @brief Make the tiny LFM2-MoE layer dtype map (FP32 only variant)
 */
std::map<std::string, ml::train::TensorDim::DataType>
makeLfm2MoeLayerDtypeMap(const causallm_test::TinyCausalLMDataType &data_type) {
  std::map<std::string, ml::train::TensorDim::DataType> dtype_map;

  if (data_type.embedding_dtype != "FP32")
    dtype_map["embedding0"] =
      causallm_test::toTensorDataType(data_type.embedding_dtype);

  if (data_type.lmhead_dtype != "FP32")
    dtype_map["output_of_causallm"] =
      causallm_test::toTensorDataType(data_type.lmhead_dtype);

  // MoE (layer{i}_ffn_down) and attention/conv FC layers stay FP32: the router
  // gate width (num_experts=4) is not divisible by 32 and cannot be Q4_0.
  return dtype_map;
}

/**
 * @brief Make a LFM2-MoE tiny CausalLM test case
 */
causallm_test::TinyCausalLMCase
makeLfm2MoeCase(const causallm_test::TinyCausalLMDataType &data_type) {
  return {
    "LFM2Moe_" + data_type.name,
    data_type,
    {"hello tok4", makeExpectedLfm2MoeLogits(),
     data_type.name == "FP32" ? 1e-4f : 1e-3f},
    makeTinyLfm2MoeConfig,
    makeLfm2MoeLayerDtypeMap,
    [](causallm::json &cfg, causallm::json &generation_cfg,
       causallm::json &nntr_cfg) {
      return std::make_unique<TinyLfm2MoeCausalLM>(cfg, generation_cfg,
                                                   nntr_cfg);
    },
    [](causallm_test::TinyCausalLMRunner &runner) {
      setupLfm2MoeDeterministicWeights(
        static_cast<TinyLfm2MoeCausalLM &>(runner));
    },
  };
}

/**
 * @brief Parameterized fixture for tiny LFM2-MoE model cases
 */
class Lfm2MoeTinyModelTest
  : public ::testing::TestWithParam<causallm_test::TinyCausalLMCase> {
protected:
  causallm_test::TinyCausalLMFiles makeFiles() const {
    const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string suite_name = "Lfm2MoeTinyModelTest";
    std::string test_name = "Unknown";

    if (info != nullptr) {
      suite_name = info->test_suite_name();
      test_name = info->name();
    }

    return causallm_test::makeTinyCausalLMFiles(suite_name, test_name,
                                                GetParam().name);
  }
};

TEST_P(Lfm2MoeTinyModelTest, GreedyGenerationSelectsArgmaxLogit) {
  const auto files = makeFiles();
  auto config =
    causallm_test::makeTinyCausalLMConfig(GetParam(), files.tokenizer_path);
  auto model =
    GetParam().create_model(config.model, config.generation, config.nntrainer);

  causallm_test::expectGreedyGenerationSelectsArgmax(*model);
}

TEST_P(Lfm2MoeTinyModelTest, WeightRoundTripProducesSameLogits) {
  const auto files = makeFiles();
  causallm_test::expectWeightRoundTripProducesSameLogits(GetParam(), files);
}

TEST_P(Lfm2MoeTinyModelTest, PromptProducesExpectedLogits) {
  const auto files = makeFiles();
  causallm_test::expectPromptProducesExpectedLogits(GetParam(), files);
}

// Q4_0 variant is intentionally omitted for LFM2-MoE (see qwen3_moe): the MoE
// router gate is FP32 with width num_experts (=4), not divisible by 32.
INSTANTIATE_TEST_SUITE_P(
  LFM2Moe, Lfm2MoeTinyModelTest,
  ::testing::Values(makeLfm2MoeCase(causallm_test::makeTinyFp32DataType())),
  [](const ::testing::TestParamInfo<causallm_test::TinyCausalLMCase> &info) {
    return info.param.name;
  });

} // namespace
