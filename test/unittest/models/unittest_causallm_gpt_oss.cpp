// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_causallm_gpt_oss.cpp
 * @date   15 June 2026
 * @brief  Tiny GptOss CausalLM model unit tests
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jungwon Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <causallm_test_utils.h>

#include <gtest/gtest.h>

#include <gptoss_causallm.h>
#include <layer.h>
#include <layer_context.h>

#include <map>

namespace {

constexpr int tiny_gpt_oss_num_layers = 1;
constexpr int tiny_gpt_oss_num_experts = 4;
constexpr int tiny_gpt_oss_num_experts_per_tok = 2;

using TinyGptOssCausalLM =
  causallm_test::CausalLMTestAdapter<causallm::GptOssForCausalLM>;

/**
 * @brief Populate deterministic tiny GptOss weights for golden token tests
 *
 * GptOss attention layers include bias (disable_bias=false) with
 * weight_initializer="ones".  All FP32 weights and biases are zeroed so that
 * the attention and MoE branches contribute zero to the residual stream and
 * the expected logits equal those of the base embedding-only models.
 *
 * MoE gate weights are set to non-uniform values for deterministic routing;
 * expert weights are zeroed so MoE output is zero.
 */
void setupGptOssDeterministicWeights(TinyGptOssCausalLM &model) {
  model.forEachLayer(
    [](ml::train::Layer &layer, nntrainer::RunLayerContext &context, void *) {
      if (layer.getName() == "output_of_causallm")
        return;

      if (layer.getType() == "gpt_oss_moe") {
        // Weight 0 is the router gate (always FP32).
        auto &gate = context.getWeight(0);
        const auto dim = gate.getDim();
        const unsigned hidden = dim.height();
        const unsigned num_exp = dim.width();
        for (unsigned h = 0; h < hidden; ++h)
          for (unsigned e = 0; e < num_exp; ++e)
            gate.setValue(0, 0, h, e, 1.0f / (e + 1));

        for (unsigned int i = 1; i < context.getNumWeights(); ++i) {
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
        if (layer.getType() == "rms_norm") {
          weight.setValue(1.0f);
        } else if (layer.getName() == "embedding0") {
          weight.setValue(0, 0, 1, 0, 1.0f);
          weight.setValue(0, 0, 4, 0, 2.0f);
        }
      }
    });
}

/**
 * @brief Make the tiny GptOss model config
 */
causallm::json makeTinyGptOssConfig() {
  return {
    {"architectures", {"GptOssForCausalLM"}},
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
    {"num_hidden_layers", tiny_gpt_oss_num_layers},
    {"num_key_value_heads", 4},
    {"num_local_experts", tiny_gpt_oss_num_experts},
    {"num_experts_per_tok", tiny_gpt_oss_num_experts_per_tok},
    {"rms_norm_eps", 1e-5},
    {"rope_theta", 10000},
    {"rope_scaling", {{"factor", 1.0}, {"type", "yarn"}}},
    {"sliding_window", 4},
    {"tie_word_embeddings", true},
    {"vocab_size", 32},
  };
}

/**
 * @brief Make the tiny GptOss layer dtype map
 *
 * GptOss attention layers include bias (two weight tensors per layer).
 * The MoE layer (layer{i}_ffn_down) is omitted for the same reason as
 * Qwen3MoE: the gate weight has width=num_experts which may not be
 * Q4_0-quantizable.
 */
std::map<std::string, ml::train::TensorDim::DataType>
makeGptOssLayerDtypeMap(const causallm_test::TinyCausalLMDataType &data_type) {
  std::map<std::string, ml::train::TensorDim::DataType> dtype_map;

  if (data_type.embedding_dtype != "FP32")
    dtype_map["embedding0"] =
      causallm_test::toTensorDataType(data_type.embedding_dtype);

  if (data_type.fc_layer_dtype != "FP32") {
    const auto dtype =
      causallm_test::toTensorDataType(data_type.fc_layer_dtype);
    for (int i = 0; i < tiny_gpt_oss_num_layers; ++i) {
      const std::string prefix = "layer" + std::to_string(i);
      dtype_map[prefix + "_wq"] = dtype;
      dtype_map[prefix + "_wk"] = dtype;
      dtype_map[prefix + "_wv"] = dtype;
      dtype_map[prefix + "_attention_out"] = dtype;
      // MoE layer omitted — see Qwen3MoE test for details.
    }
  }

  if (data_type.lmhead_dtype != "FP32")
    dtype_map["output_of_causallm"] =
      causallm_test::toTensorDataType(data_type.lmhead_dtype);

  return dtype_map;
}

/**
 * @brief Expected logits: attention bias is zeroed, MoE output is zero,
 *        residuals carry the embedding vector unchanged.
 */
std::vector<float> makeExpectedGptOssLogits() {
  std::vector<float> logits(32, 0.0f);
  logits[1] = 7.99936008f;
  logits[4] = 15.99872017f;
  return logits;
}

causallm_test::TinyCausalLMCase
makeGptOssCase(const causallm_test::TinyCausalLMDataType &data_type) {
  return {
    "GptOss_" + data_type.name,
    data_type,
    {"hello tok4", makeExpectedGptOssLogits(),
     data_type.name == "FP32"       ? 1e-4f
     : data_type.name == "Q40_FP16" ? 2e-3f
                                    : 1e-3f},
    makeTinyGptOssConfig,
    makeGptOssLayerDtypeMap,
    [](causallm::json &cfg, causallm::json &generation_cfg,
       causallm::json &nntr_cfg) {
      return std::make_unique<TinyGptOssCausalLM>(cfg, generation_cfg,
                                                  nntr_cfg);
    },
    [](causallm_test::TinyCausalLMRunner &runner) {
      setupGptOssDeterministicWeights(
        static_cast<TinyGptOssCausalLM &>(runner));
    },
  };
}

class GptOssTinyModelTest
  : public ::testing::TestWithParam<causallm_test::TinyCausalLMCase> {
protected:
  causallm_test::TinyCausalLMFiles makeFiles() const {
    const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string suite_name = "GptOssTinyModelTest";
    std::string test_name = "Unknown";

    if (info != nullptr) {
      suite_name = info->test_suite_name();
      test_name = info->name();
    }

    return causallm_test::makeTinyCausalLMFiles(suite_name, test_name,
                                                GetParam().name);
  }
};

TEST_P(GptOssTinyModelTest, GreedyGenerationSelectsArgmaxLogit) {
  const auto files = makeFiles();
  auto config =
    causallm_test::makeTinyCausalLMConfig(GetParam(), files.tokenizer_path);
  auto model =
    GetParam().create_model(config.model, config.generation, config.nntrainer);

  causallm_test::expectGreedyGenerationSelectsArgmax(*model);
}

TEST_P(GptOssTinyModelTest, WeightRoundTripProducesSameLogits) {
  const auto files = makeFiles();
  causallm_test::expectWeightRoundTripProducesSameLogits(GetParam(), files);
}

TEST_P(GptOssTinyModelTest, PromptProducesExpectedLogits) {
  const auto files = makeFiles();
  causallm_test::expectPromptProducesExpectedLogits(GetParam(), files);
}

// Q4_0 and Q4_0-FP16 variants are omitted — GptOssMoELayer uses
// ThreadManager::parallel_for which deadlocks or produces non-deterministic
// results when multiple model instances run inference sequentially in the same
// process.  See unittest_causallm_qwen3_moe.cpp for details.
INSTANTIATE_TEST_SUITE_P(
  GptOss, GptOssTinyModelTest,
  ::testing::Values(makeGptOssCase(causallm_test::makeTinyFp32DataType())),
  [](const ::testing::TestParamInfo<causallm_test::TinyCausalLMCase> &info) {
    return info.param.name;
  });

} // namespace
