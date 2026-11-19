// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_causallm_paligemma.cpp
 * @date   12 Nov 2026
 * @brief  Tiny PaliGemma CausalLM model unit tests
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <causallm_test_utils.h>

#include <gtest/gtest.h>

#include <layer.h>
#include <layer_context.h>
#include <paligemma/paligemma_causallm.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <utility>

namespace {

constexpr int tiny_paligemma_num_layers = 2;

/**
 * @brief Tiny PaliGemma adapter for common model tests
 *
 * Thin subclass of the shared CausalLMTestAdapter: only the constructor
 * differs because PaliGemma sanitizes its configs before initializing the
 * (virtual) Transformer base. All inference methods come from the adapter
 * (token-id path: the standard prefill/decode calls; the model graph is
 * is_causal=false, so the adapter's single prefill is the bidirectional
 * prefix block and its 1-row decode steps are the causal suffix — the same
 * prefix-LM semantics run_with_embeddings() orchestrates).
 */
class TinyPaliGemmaCausalLM final
  : public causallm_test::CausalLMTestAdapter<causallm::PaliGemmaCausalLM> {
public:
  /**
   * @brief Construct a tiny PaliGemma test adapter
   */
  TinyPaliGemmaCausalLM(causallm::json &cfg, causallm::json &generation_cfg,
                        causallm::json &nntr_cfg) :
    causallm::Transformer(sanitizeConfig(cfg),
                          sanitizeGenerationConfig(generation_cfg, cfg),
                          nntr_cfg, causallm::ModelType::CAUSALLM),
    causallm_test::CausalLMTestAdapter<causallm::PaliGemmaCausalLM>(
      cfg, generation_cfg, nntr_cfg) {}
};

/**
 * @brief Populate deterministic tiny PaliGemma weights for golden token tests
 *
 * Zero all FC weights; set RMS norm scales to 1; set embedding[1][0]=1,
 * embedding[4][0]=2. Attention and MLP outputs vanish, so each residual
 * carries the (scaled) embeddings unchanged and the prefill logits are
 * analytically known.
 */
void setupPaliGemmaDeterministicWeights(TinyPaliGemmaCausalLM &model) {
  model.forEachLayer(
    [](ml::train::Layer &layer, nntrainer::RunLayerContext &context, void *) {
      if (layer.getName() == "output_of_causallm")
        return;

      for (unsigned int i = 0; i < context.getNumWeights(); ++i) {
        auto &weight = context.getWeight(i);
        if (weight.getDataType() != ml::train::TensorDim::DataType::FP32)
          continue;

        weight.setValue(0.0f);
        if (causallm_test::isRmsNormLayerType(layer.getType())) {
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
 * @brief Make the tiny PaliGemma model config
 *
 * The text half of google/paligemma-3b-pt-224 shrunk to test dimensions:
 * Gemma1 decoder (no q/k norms, no softcapping), MQA, tied embedding,
 * EMBEDDING_SCALE = sqrt(hidden). is_causal is omitted on purpose — the
 * model sanitizes it to false (prefix-LM).
 */
causallm::json makeTinyPaliGemmaConfig() {
  return {
    {"architectures", {"PaliGemmaForConditionalGeneration"}},
    {"bos_token_id", 0},
    {"eos_token_id", {31}},
    {"head_dim", 8},
    {"hidden_size", 64},
    {"image_token_index", 31},
    {"intermediate_size", 64},
    {"max_position_embeddings", 8},
    {"num_attention_heads", 8},
    {"num_hidden_layers", tiny_paligemma_num_layers},
    {"num_key_value_heads", 4},
    {"rms_norm_eps", 1e-6},
    {"rope_theta", 10000},
    {"tie_word_embeddings", true},
    {"vocab_size", 32},
  };
}

/**
 * @brief Make the expected tiny PaliGemma prefill logits
 *
 * Derivation (all FC weights = 0, all RMS norm scales = 1):
 *   EMBEDDING_SCALE = sqrt(64) = 8, so the residual of the last prompt row
 *   (tok4) is [2*8, 0..0]. output_norm: RMS = sqrt(256/64) = 2, normed =
 *   [8, 0..0]. Tied LM head (no scale): logit[j] = 8 * emb[j][0] ->
 *   logit[1] = 8, logit[4] = 16.
 */
std::vector<float> makeExpectedPaliGemmaLogits() {
  std::vector<float> logits(32, 0.0f);
  logits[1] = 8.0f;
  logits[4] = 16.0f;
  return logits;
}

/**
 * @brief Make the tiny PaliGemma layer dtype map
 */
std::map<std::string, ml::train::TensorDim::DataType>
makePaliGemmaLayerDtypeMap(
  const causallm_test::TinyCausalLMDataType &data_type) {
  std::map<std::string, ml::train::TensorDim::DataType> dtype_map;

  if (data_type.embedding_dtype != "FP32")
    dtype_map["embedding0"] =
      causallm_test::toTensorDataType(data_type.embedding_dtype);

  if (data_type.fc_layer_dtype != "FP32") {
    const auto dtype =
      causallm_test::toTensorDataType(data_type.fc_layer_dtype);
    for (int i = 0; i < tiny_paligemma_num_layers; ++i) {
      const std::string prefix = "layer" + std::to_string(i);
      dtype_map[prefix + "_wq"] = dtype;
      dtype_map[prefix + "_wk"] = dtype;
      dtype_map[prefix + "_wv"] = dtype;
      dtype_map[prefix + "_attention_out"] = dtype;
      dtype_map[prefix + "_ffn_gate"] = dtype;
      dtype_map[prefix + "_ffn_up"] = dtype;
      dtype_map[prefix + "_ffn_down"] = dtype;
    }
  }

  if (data_type.lmhead_dtype != "FP32")
    dtype_map["output_of_causallm"] =
      causallm_test::toTensorDataType(data_type.lmhead_dtype);

  return dtype_map;
}

/**
 * @brief Make a PaliGemma tiny CausalLM test case
 */
causallm_test::TinyCausalLMCase
makePaliGemmaCase(const causallm_test::TinyCausalLMDataType &data_type) {
  return {
    "PaliGemma_" + data_type.name,
    data_type,
    {"hello tok4", makeExpectedPaliGemmaLogits(),
     data_type.name == "FP32" ? 1e-4f : 1e-3f},
    makeTinyPaliGemmaConfig,
    makePaliGemmaLayerDtypeMap,
    [](causallm::json &cfg, causallm::json &generation_cfg,
       causallm::json &nntr_cfg) {
      return std::make_unique<TinyPaliGemmaCausalLM>(cfg, generation_cfg,
                                                     nntr_cfg);
    },
    [](causallm_test::TinyCausalLMRunner &runner) {
      setupPaliGemmaDeterministicWeights(
        static_cast<TinyPaliGemmaCausalLM &>(runner));
    },
  };
}

/**
 * @brief Parameterized fixture for tiny PaliGemma model cases
 */
class PaliGemmaTinyModelTest
  : public ::testing::TestWithParam<causallm_test::TinyCausalLMCase> {
protected:
  /**
   * @brief Make test files for the current parameterized case
   */
  causallm_test::TinyCausalLMFiles makeFiles() const {
    const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string suite_name = "PaliGemmaTinyModelTest";
    std::string test_name = "Unknown";

    if (info != nullptr) {
      suite_name = info->test_suite_name();
      test_name = info->name();
    }

    return causallm_test::makeTinyCausalLMFiles(suite_name, test_name,
                                                GetParam().name);
  }
};

/**
 * @brief Test that greedy generation chooses the argmax logit
 */
TEST_P(PaliGemmaTinyModelTest, GreedyGenerationSelectsArgmaxLogit) {
  const auto files = makeFiles();
  auto config =
    causallm_test::makeTinyCausalLMConfig(GetParam(), files.tokenizer_path);
  auto model =
    GetParam().create_model(config.model, config.generation, config.nntrainer);

  causallm_test::expectGreedyGenerationSelectsArgmax(*model);

  // A resumed embedding request must account for the occupied cache rows.
  // Reject it before forwarding if the prompt leaves no generation slot.
  config.nntrainer["use_embedding"] = true;
  auto resumed = std::make_unique<TinyPaliGemmaCausalLM>(
    config.model, config.generation, config.nntrainer);
  resumed->initialize();
  resumed->setSessionTokenPosition(
    config.nntrainer["max_seq_len"].get<unsigned int>() - 1);
  std::vector<float> plane(64, 0.0f);
  EXPECT_THROW(
    resumed->run_with_embeddings(plane.data(), 1, 1, {4}, false, false),
    std::invalid_argument);
}

/**
 * @brief Test that a save/load round-trip preserves logits
 */
TEST_P(PaliGemmaTinyModelTest, WeightRoundTripProducesSameLogits) {
  const auto files = makeFiles();
  causallm_test::expectWeightRoundTripProducesSameLogits(GetParam(), files);
}

/**
 * @brief Test that a prompt produces the expected golden logits
 */
TEST_P(PaliGemmaTinyModelTest, PromptProducesExpectedLogits) {
  const auto files = makeFiles();
  causallm_test::expectPromptProducesExpectedLogits(GetParam(), files);
}

INSTANTIATE_TEST_SUITE_P(
  PaliGemma, PaliGemmaTinyModelTest,
  ::testing::Values(
    makePaliGemmaCase(causallm_test::makeTinyFp32DataType()),
    makePaliGemmaCase(causallm_test::makeTinyQ40Fp32DataType())),
  [](const ::testing::TestParamInfo<causallm_test::TinyCausalLMCase> &info) {
    return info.param.name;
  });

} // namespace
