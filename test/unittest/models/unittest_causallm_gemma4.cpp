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
#include <gemma4_moe_causallm.h>
#include <layer.h>
#include <layer_context.h>

#include <map>
#include <set>

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
 * @brief Tiny Gemma4 MoE CausalLM adapter for common model tests
 */
class TinyGemma4MoECausalLM final
  : public causallm_test::CausalLMTestAdapter<causallm::Gemma4MoECausalLM> {
public:
  TinyGemma4MoECausalLM(causallm::json &cfg, causallm::json &generation_cfg,
                        causallm::json &nntr_cfg) :
    causallm::Transformer(sanitizeConfig(cfg),
                          sanitizeGenerationConfig(generation_cfg, cfg),
                          nntr_cfg, causallm::ModelType::CAUSALLM),
    causallm_test::CausalLMTestAdapter<causallm::Gemma4MoECausalLM>(
      cfg, generation_cfg, nntr_cfg) {}
};

/**
 * @brief Populate deterministic tiny Gemma4 weights for golden token tests
 */
template <typename Gemma4Model>
void setupGemma4DeterministicWeights(Gemma4Model &model) {
  model.forEachLayer(
    [](ml::train::Layer &layer, nntrainer::RunLayerContext &context, void *) {
      if (layer.getName() == "output_of_causallm")
        return;

      if (layer.getType() == "gemma4_moe") {
        // Keep routing deterministic while expert projections remain zero.
        auto &router = context.getWeight(0);
        for (unsigned int hidden = 0; hidden < router.height(); ++hidden)
          for (unsigned int expert = 0; expert < router.width(); ++expert)
            router.setValue(0, 0, hidden, expert,
                            1.0f / static_cast<float>(expert + 1));
        context.getWeight(1).setValue(1.0f); // router.scale
        context.getWeight(2).setValue(1.0f); // router.per_expert_scale
        for (unsigned int i = 3; i < context.getNumWeights(); ++i) {
          auto &weight = context.getWeight(i);
          if (weight.getDataType() == ml::train::TensorDim::DataType::FP32)
            weight.setValue(0.0f);
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
 * @brief Make a tiny Gemma4 MoE config matching the official feature switches
 */
causallm::json makeTinyGemma4MoEConfig() {
  auto cfg = makeTinyGemma4Config();
  auto &text = cfg["text_config"];
  text["hidden_size_per_layer_input"] = 0;
  text["attention_k_eq_v"] = true;
  text["global_head_dim"] = 8;
  text["num_global_key_value_heads"] = 4;
  text["enable_moe_block"] = true;
  text["num_experts"] = 4;
  text["top_k_experts"] = 2;
  text["moe_intermediate_size"] = 32;
  return cfg;
}

causallm_test::TinyCausalLMCase
makeGemma4Case(const causallm_test::TinyCausalLMDataType &data_type);
causallm_test::TinyCausalLMCase
makeGemma4MoECase(const causallm_test::TinyCausalLMDataType &data_type);

/**
 * @brief Initialize a tiny Gemma4 graph and return its layer names
 */
std::set<std::string> getGemma4LayerNames(causallm::json model_cfg,
                                          const std::string &case_name) {
  const auto files = causallm_test::makeTinyCausalLMFiles("Gemma4GraphTest",
                                                          case_name, case_name);
  const auto data_type = causallm_test::makeTinyFp32DataType();
  auto test_case = makeGemma4Case(data_type);
  test_case.make_model_config = [model_cfg]() { return model_cfg; };
  auto config =
    causallm_test::makeTinyCausalLMConfig(test_case, files.tokenizer_path);

  TinyGemma4CausalLM model(config.model, config.generation, config.nntrainer);
  model.initializeModel();

  std::set<std::string> layer_names;
  model.forEachLayer(
    [&layer_names](ml::train::Layer &layer, nntrainer::RunLayerContext &,
                   void *) { layer_names.insert(layer.getName()); });
  return layer_names;
}

/**
 * @brief Initialize a tiny Gemma4 MoE graph and return its layer names
 */
std::set<std::string> getGemma4MoELayerNames(causallm::json model_cfg,
                                             const std::string &case_name) {
  const auto files = causallm_test::makeTinyCausalLMFiles("Gemma4GraphTest",
                                                          case_name, case_name);
  const auto data_type = causallm_test::makeTinyFp32DataType();
  auto test_case = makeGemma4MoECase(data_type);
  test_case.make_model_config = [model_cfg]() { return model_cfg; };
  auto config =
    causallm_test::makeTinyCausalLMConfig(test_case, files.tokenizer_path);

  TinyGemma4MoECausalLM model(config.model, config.generation,
                              config.nntrainer);
  model.initializeModel();

  std::set<std::string> layer_names;
  model.forEachLayer(
    [&layer_names](ml::train::Layer &layer, nntrainer::RunLayerContext &,
                   void *) { layer_names.insert(layer.getName()); });
  return layer_names;
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
      dtype_map[prefix + "_sparse_moe"] = dtype;
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
 * @brief Make a Gemma4 MoE tiny CausalLM test case
 */
causallm_test::TinyCausalLMCase
makeGemma4MoECase(const causallm_test::TinyCausalLMDataType &data_type) {
  return {
    "Gemma4MoE_" + data_type.name,
    data_type,
    {"hello tok4", makeExpectedGemma4Logits(),
     data_type.name == "FP32" ? 1e-4f : 1e-3f},
    makeTinyGemma4MoEConfig,
    makeGemma4LayerDtypeMap,
    [](causallm::json &cfg, causallm::json &generation_cfg,
       causallm::json &nntr_cfg) {
      return std::make_unique<TinyGemma4MoECausalLM>(cfg, generation_cfg,
                                                     nntr_cfg);
    },
    [](causallm_test::TinyCausalLMRunner &runner) {
      setupGemma4DeterministicWeights(
        static_cast<TinyGemma4MoECausalLM &>(runner));
    },
  };
}

/**
 * @brief PLE-disabled Gemma4 configurations omit every per-layer input layer
 */
TEST(Gemma4GraphTest, PerLayerEmbeddingCanBeDisabled) {
  auto cfg = makeTinyGemma4Config();
  cfg["text_config"]["hidden_size_per_layer_input"] = 0;

  const auto layer_names = getGemma4LayerNames(cfg, "PLEDisabled");
  EXPECT_EQ(layer_names.count("per_layer_input_embedding"), 0u);
  EXPECT_EQ(layer_names.count("per_layer_input_projection"), 0u);
  EXPECT_EQ(layer_names.count("layer0_per_layer_input_gate"), 0u);
  EXPECT_EQ(layer_names.count("layer1_per_layer_input_gate"), 0u);
}

/**
 * @brief Full attention reuses K projection for V when attention_k_eq_v is set
 */
TEST(Gemma4GraphTest, FullAttentionCanShareKeyAndValueProjection) {
  auto cfg = makeTinyGemma4Config();
  cfg["text_config"]["hidden_size_per_layer_input"] = 0;
  cfg["text_config"]["attention_k_eq_v"] = true;
  cfg["text_config"]["num_global_key_value_heads"] = 4;

  const auto layer_names = getGemma4LayerNames(cfg, "AttentionKEqV");
  EXPECT_EQ(layer_names.count("layer0_wv"), 1u);
  EXPECT_EQ(layer_names.count("layer1_wv"), 0u);
  EXPECT_EQ(layer_names.count("layer0_v_norm"), 1u);
  EXPECT_EQ(layer_names.count("layer1_v_norm"), 1u);
}

/**
 * @brief MoE configurations build dense and sparse feed-forward branches
 */
TEST(Gemma4GraphTest, MoEUsesDenseAndSparseBranches) {
  auto cfg = makeTinyGemma4Config();
  cfg["text_config"]["hidden_size_per_layer_input"] = 0;
  cfg["text_config"]["enable_moe_block"] = true;
  cfg["text_config"]["num_experts"] = 4;
  cfg["text_config"]["top_k_experts"] = 2;
  cfg["text_config"]["moe_intermediate_size"] = 32;

  const auto layer_names = getGemma4MoELayerNames(cfg, "MoEDenseSparse");
  for (int layer = 0; layer < tiny_gemma4_num_layers; ++layer) {
    const std::string prefix = "layer" + std::to_string(layer);
    EXPECT_EQ(layer_names.count(prefix + "_ffn_down"), 1u);
    EXPECT_EQ(layer_names.count(prefix + "_sparse_moe"), 1u);
    EXPECT_EQ(layer_names.count(prefix + "_post_ffn_norm_1"), 1u);
    EXPECT_EQ(layer_names.count(prefix + "_pre_ffn_norm_2"), 1u);
    EXPECT_EQ(layer_names.count(prefix + "_post_ffn_norm_2"), 1u);
    EXPECT_EQ(layer_names.count(prefix + "_combine_ffn"), 1u);
  }
}

/**
 * @brief Q4_0 expert projections are virtual and respect the mmap LRU bound
 */
#if !defined(_WIN32)
TEST(Gemma4VirtualExpertCacheTest, Q40UsesBoundedVirtualExpertWeights) {
  const auto files = causallm_test::makeTinyCausalLMFiles(
    "Gemma4VirtualExpertCacheTest", "Q40UsesBoundedVirtualExpertWeights",
    "Gemma4MoE_Q40_FP32");
  const auto fp32_type = causallm_test::makeTinyFp32DataType();
  const auto q40_type = causallm_test::makeTinyQ40Fp32DataType();

  auto source_case = makeGemma4MoECase(fp32_type);
  auto source_config =
    causallm_test::makeTinyCausalLMConfig(source_case, files.tokenizer_path);
  TinyGemma4MoECausalLM source(source_config.model, source_config.generation,
                               source_config.nntrainer);
  source.initializeModel();
  setupGemma4DeterministicWeights(source);
  source.saveWeightWithDtype(files.weight_path.string(),
                             makeGemma4LayerDtypeMap(q40_type));

  auto cached_case = makeGemma4MoECase(q40_type);
  auto cached_config =
    causallm_test::makeTinyCausalLMConfig(cached_case, files.tokenizer_path);
  cached_config.nntrainer["moe_cache_size"] = 1;
  TinyGemma4MoECausalLM cached(cached_config.model, cached_config.generation,
                               cached_config.nntrainer);
  cached.initializeModel();
  cached.loadWeight(files.weight_path.string());

  const auto logits = cached.prefillLogits("hello tok4");
  const auto expected = makeExpectedGemma4Logits();
  ASSERT_EQ(logits.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i)
    EXPECT_NEAR(logits[i], expected[i], 1.0e-3f);

  unsigned int sparse_layers = 0;
  cached.forEachLayer([&sparse_layers](ml::train::Layer &layer,
                                       nntrainer::RunLayerContext &context,
                                       void *) {
    if (layer.getType() != "gemma4_moe")
      return;

    ++sparse_layers;
    unsigned int allocated_expert_weights = 0;
    for (unsigned int weight = 0; weight < context.getNumWeights(); ++weight) {
      auto &tensor = context.getWeight(weight);
      if (weight < 3) {
        EXPECT_FALSE(tensor.isVirtual());
        continue;
      }
      EXPECT_TRUE(tensor.isVirtual());
      if (tensor.isAllocated())
        ++allocated_expert_weights;
    }
    EXPECT_EQ(allocated_expert_weights, 3u);
  });
  EXPECT_EQ(sparse_layers, tiny_gemma4_num_layers);
}
#endif

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
  ::testing::Values(
    makeGemma4Case(causallm_test::makeTinyFp32DataType()),
    makeGemma4Case(causallm_test::makeTinyQ40Fp32DataType()),
    makeGemma4MoECase(causallm_test::makeTinyFp32DataType()),
    makeGemma4MoECase(causallm_test::makeTinyQ40Fp32DataType())),
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
