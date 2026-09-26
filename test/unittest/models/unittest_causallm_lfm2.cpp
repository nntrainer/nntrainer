// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_causallm_lfm2.cpp
 * @date   19 June 2026
 * @brief  Tiny LFM2 CausalLM model unit tests
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jungwon Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <causallm_test_utils.h>

#include <gtest/gtest.h>

#include <layer.h>
#include <layer_context.h>
#include <lfm2_causallm.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <utility>

namespace {

constexpr int tiny_lfm2_num_layers = 2;

/**
 * @brief Tiny LFM2 CausalLM adapter for common model tests
 */
using TinyLfm2CausalLM =
  causallm_test::CausalLMTestAdapter<causallm::Lfm2CausalLM>;

/**
 * @brief Populate deterministic tiny LFM2 weights for golden token tests
 *
 * Zero all FC weights; set RMS norm scales to 1; set embedding[1][0]=1,
 * embedding[4][0]=2. This produces analytically known prefill logits.
 */
void setupLfm2DeterministicWeights(TinyLfm2CausalLM &model) {
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
 * @brief Make the tiny LFM2 model config
 *
 * Uses layer_types=["attention","conv"] to exercise both the attention and
 * conv hybrid paths with a single tiny model.
 */
causallm::json makeTinyLfm2Config() {
  return {
    {"architectures", {"Lfm2ForCausalLM"}},
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
    {"num_hidden_layers", tiny_lfm2_num_layers},
    {"num_key_value_heads", 4},
    {"rms_norm_eps", 1e-6},
    {"rope_theta", 10000},
    {"tie_word_embeddings", true},
    {"vocab_size", 32},
  };
}

/**
 * @brief Make the expected tiny LFM2 prefill logits
 *
 * Derivation: all FC weights = 0, all RMS norm scales = 1.
 *   layer0 (attention): residual carries embedding[tok4]=[2,0..] unchanged.
 *   layer1 (conv):      explicit residual (input.add(proj_back)) does the same.
 *   output_norm on [2,0..0] (64-dim): RMS≈0.25, normed≈[8,0..0].
 *   tied LM head: logit[j] ≈ 8*emb[j][0].
 *   LFM2's Q/K reshaped_rms_norm reduces numerical error vs Qwen family:
 *   logit[1] ≈ 7.9999361, logit[4] ≈ 15.9998722 (empirically confirmed).
 */
std::vector<float> makeExpectedLfm2Logits() {
  std::vector<float> logits(32, 0.0f);
  logits[1] = 7.9999361f;
  logits[4] = 15.9998722f;
  return logits;
}

/**
 * @brief Make the tiny LFM2 layer dtype map
 */
std::map<std::string, ml::train::TensorDim::DataType>
makeLfm2LayerDtypeMap(const causallm_test::TinyCausalLMDataType &data_type) {
  std::map<std::string, ml::train::TensorDim::DataType> dtype_map;

  if (data_type.embedding_dtype != "FP32")
    dtype_map["embedding0"] =
      causallm_test::toTensorDataType(data_type.embedding_dtype);

  if (data_type.fc_layer_dtype != "FP32") {
    const auto dtype =
      causallm_test::toTensorDataType(data_type.fc_layer_dtype);
    // layer0: attention block FC layers
    dtype_map["layer0_wq"] = dtype;
    dtype_map["layer0_wk"] = dtype;
    dtype_map["layer0_wv"] = dtype;
    dtype_map["layer0_attention_out"] = dtype;
    dtype_map["layer0_ffn_up"] = dtype;
    dtype_map["layer0_ffn_gate"] = dtype;
    dtype_map["layer0_ffn_down"] = dtype;
    // layer1: conv block FC layers (causal_conv1d is always FP32 by design)
    dtype_map["layer1_conv_in_proj"] = dtype;
    dtype_map["layer1_conv_out_proj"] = dtype;
    dtype_map["layer1_ffn_up"] = dtype;
    dtype_map["layer1_ffn_gate"] = dtype;
    dtype_map["layer1_ffn_down"] = dtype;
  }

  if (data_type.lmhead_dtype != "FP32")
    dtype_map["output_of_causallm"] =
      causallm_test::toTensorDataType(data_type.lmhead_dtype);

  return dtype_map;
}

/**
 * @brief Make a LFM2 tiny CausalLM test case
 */
causallm_test::TinyCausalLMCase
makeLfm2Case(const causallm_test::TinyCausalLMDataType &data_type) {
  return {
    "LFM2_" + data_type.name,
    data_type,
    {"hello tok4", makeExpectedLfm2Logits(),
     data_type.name == "FP32" ? 1e-4f : 1e-3f},
    makeTinyLfm2Config,
    makeLfm2LayerDtypeMap,
    [](causallm::json &cfg, causallm::json &generation_cfg,
       causallm::json &nntr_cfg) {
      return std::make_unique<TinyLfm2CausalLM>(cfg, generation_cfg, nntr_cfg);
    },
    [](causallm_test::TinyCausalLMRunner &runner) {
      setupLfm2DeterministicWeights(static_cast<TinyLfm2CausalLM &>(runner));
    },
  };
}

/**
 * @brief Parameterized fixture for tiny LFM2 model cases
 */
class Lfm2TinyModelTest
  : public ::testing::TestWithParam<causallm_test::TinyCausalLMCase> {
protected:
  causallm_test::TinyCausalLMFiles makeFiles() const {
    const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string suite_name = "Lfm2TinyModelTest";
    std::string test_name = "Unknown";

    if (info != nullptr) {
      suite_name = info->test_suite_name();
      test_name = info->name();
    }

    return causallm_test::makeTinyCausalLMFiles(suite_name, test_name,
                                                GetParam().name);
  }
};

TEST_P(Lfm2TinyModelTest, GreedyGenerationSelectsArgmaxLogit) {
  const auto files = makeFiles();
  auto config =
    causallm_test::makeTinyCausalLMConfig(GetParam(), files.tokenizer_path);
  auto model =
    GetParam().create_model(config.model, config.generation, config.nntrainer);

  causallm_test::expectGreedyGenerationSelectsArgmax(*model);
}

TEST_P(Lfm2TinyModelTest, WeightRoundTripProducesSameLogits) {
  const auto files = makeFiles();
  causallm_test::expectWeightRoundTripProducesSameLogits(GetParam(), files);
}

TEST_P(Lfm2TinyModelTest, PromptProducesExpectedLogits) {
  const auto files = makeFiles();
  causallm_test::expectPromptProducesExpectedLogits(GetParam(), files);
}

INSTANTIATE_TEST_SUITE_P(
  LFM2, Lfm2TinyModelTest,
  ::testing::Values(makeLfm2Case(causallm_test::makeTinyFp32DataType()),
                    makeLfm2Case(causallm_test::makeTinyQ40Fp32DataType())),
  [](const ::testing::TestParamInfo<causallm_test::TinyCausalLMCase> &info) {
    return info.param.name;
  });

} // namespace
