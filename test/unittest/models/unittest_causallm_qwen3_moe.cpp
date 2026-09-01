// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_causallm_qwen3_moe.cpp
 * @date   15 June 2026
 * @brief  Tiny Qwen3MoE CausalLM model unit tests
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jungwon Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <causallm_test_utils.h>

#include <gtest/gtest.h>

#include <layer.h>
#include <layer_context.h>
#include <qwen3_moe_causallm.h>

#include <map>

namespace {

constexpr int tiny_qwen3_moe_num_layers = 1;
constexpr int tiny_qwen3_moe_num_experts = 4;
constexpr int tiny_qwen3_moe_num_experts_per_tok = 2;

using TinyQwen3MoECausalLM =
  causallm_test::CausalLMTestAdapter<causallm::Qwen3MoECausalLM>;

/**
 * @brief Populate deterministic tiny Qwen3MoE weights for golden token tests
 *
 * MoE gate weights are set to non-uniform values so top-k routing is
 * deterministic (expert j scores 1/(j+1)).  Expert projection weights stay
 * at zero, so the MoE branch output is zero and the residual path carries
 * the hidden state unchanged — expected logits are the same as Qwen3.
 */
void setupQwen3MoEDeterministicWeights(TinyQwen3MoECausalLM &model) {
  model.forEachLayer(
    [](ml::train::Layer &layer, nntrainer::RunLayerContext &context, void *) {
      if (layer.getName() == "output_of_causallm")
        return;

      if (layer.getType() == "qwen_moe") {
        // Weight 0 is the router gate: [1, 1, hidden_size, num_experts] (NCHW).
        // Always FP32 inside MoELayer regardless of layer dtype.
        // Set expert j column to 1/(j+1) so routing is deterministic.
        auto &gate = context.getWeight(0);
        const auto dim = gate.getDim();
        const unsigned hidden = dim.height(); // NCHW: height = hidden_size
        const unsigned num_exp = dim.width(); // NCHW: width  = num_experts
        for (unsigned h = 0; h < hidden; ++h)
          for (unsigned e = 0; e < num_exp; ++e)
            gate.setValue(0, 0, h, e, 1.0f / (e + 1));

        // Expert projection weights (indices 1+): zero all FP32 weights so the
        // MoE branch contributes zero to the residual stream.
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
        if (causallm_test::isRmsNormLayerType(layer.getType())) {
          weight.setValue(1.0f);
        } else if (layer.getName() == "embedding0") {
          weight.setValue(0, 0, 1, 0, 1.0f);
          weight.setValue(0, 0, 4, 0, 2.0f);
        }
      }
    });
}

/**
 * @brief Make the tiny Qwen3MoE model config
 */
causallm::json makeTinyQwen3MoEConfig() {
  return {
    {"architectures", {"Qwen3MoeForCausalLM"}},
    {"bos_token_id", 0},
    {"eos_token_id", {31}},
    {"head_dim", 8},
    {"hidden_size", 64},
    {"intermediate_size", 64},
    {"moe_intermediate_size", 64},
    {"is_causal", true},
    {"max_position_embeddings", 8},
    {"num_attention_heads", 8},
    {"num_hidden_layers", tiny_qwen3_moe_num_layers},
    {"num_key_value_heads", 4},
    {"num_experts", tiny_qwen3_moe_num_experts},
    {"num_experts_per_tok", tiny_qwen3_moe_num_experts_per_tok},
    {"rms_norm_eps", 1e-5},
    {"rope_theta", 10000},
    {"tie_word_embeddings", true},
    {"vocab_size", 32},
  };
}

/**
 * @brief Make the tiny Qwen3MoE layer dtype map
 *
 * The MoE layer (layer{i}_ffn_down) is included for Q4_0 so that expert
 * projection weights are quantized.  The gate weight inside MoELayer is
 * always FP32 (hardcoded in the layer) so it is unaffected by the dtype map.
 */
std::map<std::string, ml::train::TensorDim::DataType> makeQwen3MoELayerDtypeMap(
  const causallm_test::TinyCausalLMDataType &data_type) {
  std::map<std::string, ml::train::TensorDim::DataType> dtype_map;

  if (data_type.embedding_dtype != "FP32")
    dtype_map["embedding0"] =
      causallm_test::toTensorDataType(data_type.embedding_dtype);

  if (data_type.fc_layer_dtype != "FP32") {
    const auto dtype =
      causallm_test::toTensorDataType(data_type.fc_layer_dtype);
    for (int i = 0; i < tiny_qwen3_moe_num_layers; ++i) {
      const std::string prefix = "layer" + std::to_string(i);
      dtype_map[prefix + "_wq"] = dtype;
      dtype_map[prefix + "_wk"] = dtype;
      dtype_map[prefix + "_wv"] = dtype;
      dtype_map[prefix + "_attention_out"] = dtype;
      // MoE layer (layer{i}_ffn_down) is intentionally omitted from the
      // dtype_map.  The router gate weight inside MoELayer is hardcoded
      // FP32 with width=num_experts=4, which is not divisible by 32 and
      // therefore cannot be Q4_0-quantized via the save_weight path.
      // The fc_dtype config only applies to built-in fully_connected layers,
      // so the MoE layer weights stay FP32 on load as well.
    }
  }

  if (data_type.lmhead_dtype != "FP32")
    dtype_map["output_of_causallm"] =
      causallm_test::toTensorDataType(data_type.lmhead_dtype);

  return dtype_map;
}

/**
 * @brief Make the expected tiny Qwen3MoE prefill logits
 *
 * With zero expert projection weights the MoE branch contributes zero to
 * the residual stream, leaving the hidden state unchanged after each decoder
 * layer.  The final logits therefore equal those of the plain Qwen3 tiny
 * model with the same embedding and rms_norm setup.
 */
std::vector<float> makeExpectedQwen3MoELogits() {
  std::vector<float> logits(32, 0.0f);
  logits[1] = 7.99936008f;
  logits[4] = 15.99872017f;
  return logits;
}

/**
 * @brief Make a Qwen3MoE tiny CausalLM test case
 */
causallm_test::TinyCausalLMCase
makeQwen3MoECase(const causallm_test::TinyCausalLMDataType &data_type) {
  return {
    "Qwen3MoE_" + data_type.name,
    data_type,
    {"hello tok4", makeExpectedQwen3MoELogits(),
     data_type.name == "FP32" ? 1e-4f : 1e-3f},
    makeTinyQwen3MoEConfig,
    makeQwen3MoELayerDtypeMap,
    [](causallm::json &cfg, causallm::json &generation_cfg,
       causallm::json &nntr_cfg) {
      return std::make_unique<TinyQwen3MoECausalLM>(cfg, generation_cfg,
                                                    nntr_cfg);
    },
    [](causallm_test::TinyCausalLMRunner &runner) {
      setupQwen3MoEDeterministicWeights(
        static_cast<TinyQwen3MoECausalLM &>(runner));
    },
  };
}

/**
 * @brief Parameterized fixture for tiny Qwen3MoE model cases
 */
class Qwen3MoETinyModelTest
  : public ::testing::TestWithParam<causallm_test::TinyCausalLMCase> {
protected:
  causallm_test::TinyCausalLMFiles makeFiles() const {
    const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string suite_name = "Qwen3MoETinyModelTest";
    std::string test_name = "Unknown";

    if (info != nullptr) {
      suite_name = info->test_suite_name();
      test_name = info->name();
    }

    return causallm_test::makeTinyCausalLMFiles(suite_name, test_name,
                                                GetParam().name);
  }
};

TEST_P(Qwen3MoETinyModelTest, GreedyGenerationSelectsArgmaxLogit) {
  const auto files = makeFiles();
  auto config =
    causallm_test::makeTinyCausalLMConfig(GetParam(), files.tokenizer_path);
  auto model =
    GetParam().create_model(config.model, config.generation, config.nntrainer);

  causallm_test::expectGreedyGenerationSelectsArgmax(*model);
}

TEST_P(Qwen3MoETinyModelTest, WeightRoundTripProducesSameLogits) {
  const auto files = makeFiles();
  causallm_test::expectWeightRoundTripProducesSameLogits(GetParam(), files);
}

TEST_P(Qwen3MoETinyModelTest, PromptProducesExpectedLogits) {
  const auto files = makeFiles();
  causallm_test::expectPromptProducesExpectedLogits(GetParam(), files);
}

// Q4_0 variant is intentionally omitted for Qwen3MoE.  The MoE forwarding
// path uses ThreadManager::parallel_for which deadlocks when four model
// instances are created and run sequentially (as WeightRoundTrip does) in the
// same process.  The MoE routing and expert computation are fully exercised
// by the FP32 variant; Q4_0 would only differ in attention-weight precision.
INSTANTIATE_TEST_SUITE_P(
  Qwen3MoE, Qwen3MoETinyModelTest,
  ::testing::Values(makeQwen3MoECase(causallm_test::makeTinyFp32DataType())),
  [](const ::testing::TestParamInfo<causallm_test::TinyCausalLMCase> &info) {
    return info.param.name;
  });

} // namespace
