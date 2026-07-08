// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_causallm_gemma4.cpp
 * @date   15 June 2026
 * @brief  Tiny Gemma4 CausalLM model unit tests
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jungwon Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <causallm_test_utils.h>

#include <gtest/gtest.h>

#include <gemma4_causallm.h>
#include <layer.h>
#include <layer_context.h>

#include <map>

namespace {

constexpr int tiny_gemma4_num_layers = 2;

/**
 * @brief Tiny Gemma4 CausalLM adapter for common model tests
 *
 * Thin subclass of the shared CausalLMTestAdapter: only the constructor
 * differs because Gemma4 must sanitize its configs (flattening text_config)
 * before initializing the (virtual) Transformer base.
 */
class TinyGemma4CausalLM final
  : public causallm_test::CausalLMTestAdapter<causallm::Gemma4CausalLM> {
public:
  /**
   * @brief Construct a tiny Gemma4 CausalLM test adapter
   */
  TinyGemma4CausalLM(causallm::json &cfg, causallm::json &generation_cfg,
                     causallm::json &nntr_cfg) :
    causallm::Transformer(sanitizeConfig(cfg),
                          sanitizeGenerationConfig(generation_cfg, cfg),
                          nntr_cfg, causallm::ModelType::CAUSALLM),
    causallm_test::CausalLMTestAdapter<causallm::Gemma4CausalLM>(
      cfg, generation_cfg, nntr_cfg) {}
};

/**
 * @brief Populate deterministic tiny Gemma4 weights for golden token tests
 */
void setupGemma4DeterministicWeights(TinyGemma4CausalLM &model) {
  model.forEachLayer(
    [](ml::train::Layer &layer, nntrainer::RunLayerContext &context, void *) {
      if (layer.getName() == "output_of_causallm")
        return;

      for (unsigned int i = 0; i < context.getNumWeights(); ++i) {
        auto &weight = context.getWeight(i);
        if (weight.getDataType() != ml::train::TensorDim::DataType::FP32)
          continue;

        weight.setValue(0.0f);
        if (layer.getType() == "rms_norm" ||
            layer.getType() == "reshaped_rms_norm") {
          weight.setValue(1.0f);
        } else if (layer.getName() == "embedding0") {
          weight.setValue(0, 0, 1, 0, 1.0f);
          weight.setValue(0, 0, 4, 0, 2.0f);
        } else if (layer.getName().find("_layer_scalar") != std::string::npos) {
          // layer_scalar scales decoder_output (including residual) before the
          // next layer receives it.  A value of 0 zeros out the entire hidden
          // state; 1 preserves it so the residual path is exercised.
          weight.setValue(1.0f);
        }
      }
    });
}

/**
 * @brief Make the tiny Gemma4 model config
 *
 * Fields are wrapped in text_config as the real HF config would be.
 * sanitizeConfig() in TinyGemma4CausalLM flattens them before construction.
 */
causallm::json makeTinyGemma4Config() {
  return {
    {"architectures", {"Gemma4ForCausalLM"}},
    {"bos_token_id", 0},
    {"eos_token_id", {31}},
    {"text_config",
     {
       {"head_dim", 8},
       {"hidden_size", 64},
       {"hidden_size_per_layer_input", 32},
       {"intermediate_size", 64},
       {"layer_types", {"sliding_attention", "full_attention"}},
       {"max_position_embeddings", 8},
       {"num_attention_heads", 8},
       {"num_hidden_layers", tiny_gemma4_num_layers},
       {"num_key_value_heads", 4},
       {"rms_norm_eps", 1e-6},
       {"rope_theta", 1000000},
       {"sliding_window", 4},
       {"tie_word_embeddings", true},
       {"vocab_size", 32},
       {"vocab_size_per_layer_input", 32},
     }},
  };
}

/**
 * @brief Make the tiny Gemma4 layer dtype map
 */
std::map<std::string, ml::train::TensorDim::DataType>
makeGemma4LayerDtypeMap(const causallm_test::TinyCausalLMDataType &data_type) {
  std::map<std::string, ml::train::TensorDim::DataType> dtype_map;

  if (data_type.embedding_dtype != "FP32") {
    const auto emb_dtype =
      causallm_test::toTensorDataType(data_type.embedding_dtype);
    dtype_map["embedding0"] = emb_dtype;
    // per_layer_input_embedding: [vocab_per_layer, num_layers*hidden_per_layer]
    // with hidden_size_per_layer_input=32: width=64, divisible by 32
    dtype_map["per_layer_input_embedding"] = emb_dtype;
  }

  if (data_type.fc_layer_dtype != "FP32") {
    const auto dtype =
      causallm_test::toTensorDataType(data_type.fc_layer_dtype);
    for (int i = 0; i < tiny_gemma4_num_layers; ++i) {
      const std::string prefix = "layer" + std::to_string(i);
      dtype_map[prefix + "_wq"] = dtype;
      dtype_map[prefix + "_wk"] = dtype;
      dtype_map[prefix + "_wv"] = dtype;
      dtype_map[prefix + "_attention_out"] = dtype;
      dtype_map[prefix + "_ffn_gate"] = dtype;
      dtype_map[prefix + "_ffn_up"] = dtype;
      dtype_map[prefix + "_ffn_down"] = dtype;
      // Gemma4-specific per-layer FC weights
      // hidden_size_per_layer_input=32 ensures width is divisible by 32
      dtype_map[prefix + "_per_layer_input_gate"] = dtype;
      dtype_map[prefix + "_per_layer_input_proj"] = dtype;
    }
    dtype_map["per_layer_input_projection"] = dtype;
  }

  if (data_type.lmhead_dtype != "FP32")
    dtype_map["output_of_causallm"] =
      causallm_test::toTensorDataType(data_type.lmhead_dtype);

  return dtype_map;
}

/**
 * @brief Make the expected tiny Gemma4 prefill logits
 *
 * With deterministic weights (embedding[1,0]=1, embedding[4,0]=2, all FC=0,
 * all rms_norm=1, all scalar_multiply=0), the hidden state passes unchanged
 * through zero-output decoder layers.  The final rms_norm normalises the
 * embedding vector, and the tied word-embedding lm_head projects it back:
 *   logit[j] = hidden_norm[0] * embedding[j,0]
 * giving logit[1]=8, logit[4]=16, all others=0.
 */
std::vector<float> makeExpectedGemma4Logits() {
  std::vector<float> logits(32, 0.0f);
  logits[1] = 8.0f;
  logits[4] = 16.0f;
  return logits;
}

/**
 * @brief Make a Gemma4 tiny CausalLM test case
 */
causallm_test::TinyCausalLMCase
makeGemma4Case(const causallm_test::TinyCausalLMDataType &data_type) {
  return {
    "Gemma4_" + data_type.name,
    data_type,
    {"hello tok4", makeExpectedGemma4Logits(),
     data_type.name == "FP32"       ? 1e-4f
     : data_type.name == "Q40_FP16" ? 2e-2f
                                    : 1e-3f},
    makeTinyGemma4Config,
    makeGemma4LayerDtypeMap,
    [](causallm::json &cfg, causallm::json &generation_cfg,
       causallm::json &nntr_cfg) {
      return std::make_unique<TinyGemma4CausalLM>(cfg, generation_cfg,
                                                  nntr_cfg);
    },
    [](causallm_test::TinyCausalLMRunner &runner) {
      setupGemma4DeterministicWeights(
        static_cast<TinyGemma4CausalLM &>(runner));
    },
  };
}

/**
 * @brief Parameterized fixture for tiny Gemma4 model cases
 */
class Gemma4TinyModelTest
  : public ::testing::TestWithParam<causallm_test::TinyCausalLMCase> {
protected:
  /**
   * @brief Make test files for the current parameterized case
   */
  causallm_test::TinyCausalLMFiles makeFiles() const {
    const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string suite_name = "Gemma4TinyModelTest";
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
TEST_P(Gemma4TinyModelTest, GreedyGenerationSelectsArgmaxLogit) {
  const auto files = makeFiles();
  auto config =
    causallm_test::makeTinyCausalLMConfig(GetParam(), files.tokenizer_path);
  auto model =
    GetParam().create_model(config.model, config.generation, config.nntrainer);

  causallm_test::expectGreedyGenerationSelectsArgmax(*model);
}

/**
 * @brief Test that a save/load round-trip preserves logits
 */
TEST_P(Gemma4TinyModelTest, WeightRoundTripProducesSameLogits) {
  const auto files = makeFiles();
  causallm_test::expectWeightRoundTripProducesSameLogits(GetParam(), files);
}

/**
 * @brief Test that a prompt produces the expected golden logits
 */
TEST_P(Gemma4TinyModelTest, PromptProducesExpectedLogits) {
  const auto files = makeFiles();
  causallm_test::expectPromptProducesExpectedLogits(GetParam(), files);
}

INSTANTIATE_TEST_SUITE_P(
  Gemma4, Gemma4TinyModelTest,
  ::testing::Values(makeGemma4Case(causallm_test::makeTinyFp32DataType()),
                    makeGemma4Case(causallm_test::makeTinyQ40Fp32DataType())),
  [](const ::testing::TestParamInfo<causallm_test::TinyCausalLMCase> &info) {
    return info.param.name;
  });

#ifdef ENABLE_FP16
INSTANTIATE_TEST_SUITE_P(
  Gemma4Fp16, Gemma4TinyModelTest,
  ::testing::Values(makeGemma4Case(causallm_test::makeTinyQ40Fp16DataType())),
  [](const ::testing::TestParamInfo<causallm_test::TinyCausalLMCase> &info) {
    return info.param.name;
  });
#endif

} // namespace
