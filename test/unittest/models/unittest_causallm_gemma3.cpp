// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_causallm_gemma3.cpp
 * @date   18 May 2026
 * @brief  Tiny Gemma3 CausalLM and EmbeddingGemma model unit tests
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <causallm_test_utils.h>

#include <gtest/gtest.h>

#include <embedding_gemma.h>
#include <gemma3_causallm.h>
#include <layer.h>
#include <layer_context.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <utility>

namespace {

constexpr int tiny_gemma3_num_layers = 2;

/**
 * @brief Tiny Gemma3 CausalLM adapter for common model tests
 *
 * Thin subclass of the shared CausalLMTestAdapter: only the constructor
 * differs because Gemma3 must sanitize its configs before initializing the
 * (virtual) Transformer base. All inference methods come from the adapter.
 */
class TinyGemma3CausalLM final
  : public causallm_test::CausalLMTestAdapter<causallm::Gemma3CausalLM> {
public:
  /**
   * @brief Construct a tiny Gemma3 CausalLM test adapter
   */
  TinyGemma3CausalLM(causallm::json &cfg, causallm::json &generation_cfg,
                     causallm::json &nntr_cfg) :
    causallm::Transformer(sanitizeConfig(cfg),
                          sanitizeGenerationConfig(generation_cfg, cfg),
                          nntr_cfg, causallm::ModelType::CAUSALLM),
    causallm_test::CausalLMTestAdapter<causallm::Gemma3CausalLM>(
      cfg, generation_cfg, nntr_cfg) {}
};

/**
 * @brief Populate deterministic tiny Gemma3 weights for golden token tests
 */
void setupGemma3DeterministicWeights(TinyGemma3CausalLM &model) {
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
        }
      }
    });
}

/**
 * @brief Files generated for one tiny EmbeddingGemma test invocation
 */
struct TinyEmbeddingGemmaFiles {
  std::filesystem::path dir;            /**< Temporary test directory */
  std::filesystem::path tokenizer_path; /**< Tiny tokenizer.json path */
  std::filesystem::path modules_path;   /**< Tiny modules.json path */
  std::filesystem::path weight_path;    /**< Tiny model weight path */
};

/**
 * @brief Tiny EmbeddingGemma adapter for model-level tests
 */
class TinyEmbeddingGemma final : public causallm::EmbeddingGemma {
public:
  /**
   * @brief Construct a tiny EmbeddingGemma test adapter
   */
  TinyEmbeddingGemma(causallm::json &cfg, causallm::json &generation_cfg,
                     causallm::json &nntr_cfg) :
    causallm::Transformer(sanitizeConfig(cfg),
                          sanitizeGenerationConfig(generation_cfg, cfg),
                          nntr_cfg, causallm::ModelType::EMBEDDING),
    causallm::EmbeddingGemma(cfg, generation_cfg, nntr_cfg) {}

  /**
   * @brief Initialize the tiny EmbeddingGemma model
   */
  void initializeModel() { initialize(); }

  /**
   * @brief Save tiny EmbeddingGemma weights with dtype conversion
   */
  void saveWeightWithDtype(
    const std::string &path,
    const std::map<std::string, ml::train::TensorDim::DataType>
      &layer_dtype_map) {
    save_weight(path, ml::train::TensorDim::DataType::NONE, layer_dtype_map);
  }

  /**
   * @brief Load tiny EmbeddingGemma model weights
   */
  void loadWeight(const std::string &path) { load_weight(path); }

  /**
   * @brief Set deterministic tiny EmbeddingGemma weights
   */
  void setDeterministicWeights() {
    auto set_weights = [](ml::train::Layer &layer,
                          nntrainer::RunLayerContext &context, void *) {
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
        } else if (layer.getName() == "2" || layer.getName() == "3") {
          weight.setValue(0.01f);
        }
      }
    };

    model->forEachLayer(set_weights, nullptr);
  }

  /**
   * @brief Encode one prompt and copy the embedding output
   */
  std::vector<float> encodePrompt(const std::string &prompt) {
    auto output = encode(prompt);
    std::vector<float> embedding(output[0], output[0] + BATCH_SIZE * DIM);
    for (auto &out : output)
      delete[] out;
    return embedding;
  }

  /**
   * @brief Return whether the tiny embedding model uses causal attention
   */
  bool isCausalForTest() const { return IS_CAUSAL; }
};

/**
 * @brief Write a string into a file
 */
void writeTextFile(const std::filesystem::path &path,
                   const std::string &content) {
  std::ofstream file(path, std::ios::binary);
  if (!file)
    throw std::runtime_error("failed to open " + path.string());

  file << content;
  if (!file.good())
    throw std::runtime_error("failed to write " + path.string());
}

/**
 * @brief Make tiny EmbeddingGemma SentenceTransformer module config files
 */
std::filesystem::path
writeTinyEmbeddingGemmaModules(const std::filesystem::path &dir) {
  auto modules_path = dir / "modules.json";
  auto pooling_dir = dir / "1_Pooling";
  auto first_dense_dir = dir / "2_Dense";
  auto second_dense_dir = dir / "3_Dense";
  std::filesystem::create_directories(pooling_dir);
  std::filesystem::create_directories(first_dense_dir);
  std::filesystem::create_directories(second_dense_dir);

  writeTextFile(modules_path, R"([
  {
    "idx": 0,
    "name": "0",
    "path": "",
    "type": "sentence_transformers.models.Transformer"
  },
  {
    "idx": 1,
    "name": "1",
    "path": "1_Pooling",
    "type": "sentence_transformers.models.Pooling"
  },
  {
    "idx": 2,
    "name": "2",
    "path": "2_Dense",
    "type": "sentence_transformers.models.Dense"
  },
  {
    "idx": 3,
    "name": "3",
    "path": "3_Dense",
    "type": "sentence_transformers.models.Dense"
  },
  {
    "idx": 4,
    "name": "4",
    "path": "",
    "type": "sentence_transformers.models.Normalize"
  }
])");

  writeTextFile(pooling_dir / "config.json", R"({
  "word_embedding_dimension": 64,
  "pooling_mode_cls_token": false,
  "pooling_mode_mean_tokens": true,
  "pooling_mode_max_tokens": false,
  "pooling_mode_mean_sqrt_len_tokens": false,
  "pooling_mode_weightedmean_tokens": false,
  "pooling_mode_lasttoken": false,
  "include_prompt": true
})");

  const char *dense_config = R"({
  "in_features": 64,
  "out_features": 64,
  "bias": false,
  "activation_function": "torch.nn.modules.linear.Identity"
})";
  writeTextFile(first_dense_dir / "config.json", dense_config);
  writeTextFile(second_dense_dir / "config.json", dense_config);

  return modules_path;
}

/**
 * @brief Make the tiny Gemma3 model config
 */
causallm::json makeTinyGemma3Config() {
  return {
    {"architectures", {"Gemma3ForCausalLM"}},
    {"bos_token_id", 0},
    {"eos_token_id", {31}},
    {"head_dim", 8},
    {"hidden_size", 64},
    {"intermediate_size", 64},
    {"is_causal", true},
    {"max_position_embeddings", 8},
    {"num_attention_heads", 8},
    {"num_hidden_layers", tiny_gemma3_num_layers},
    {"num_key_value_heads", 4},
    {"rms_norm_eps", 1e-6},
    {"rope_theta", 1000000},
    {"sliding_window", 4},
    {"sliding_window_pattern", 2},
    {"tie_word_embeddings", true},
    {"vocab_size", 32},
  };
}

/**
 * @brief Make the tiny EmbeddingGemma model config
 */
causallm::json makeTinyEmbeddingGemmaConfig() {
  auto cfg = makeTinyGemma3Config();
  cfg["architectures"] = {"Gemma3TextModel"};
  cfg["use_bidirectional_attention"] = true;
  cfg.erase("is_causal");
  return cfg;
}

/**
 * @brief Make tiny EmbeddingGemma test files
 */
TinyEmbeddingGemmaFiles makeEmbeddingGemmaFiles() {
  const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
  std::string suite_name = "EmbeddingGemmaTinyModelTest";
  std::string test_name = "Unknown";

  if (info != nullptr) {
    suite_name = info->test_suite_name();
    test_name = info->name();
  }

  auto files = causallm_test::makeTinyCausalLMFiles(suite_name, test_name,
                                                    "EmbeddingGemma_Q40_FP32");

  return {
    files.dir,
    files.tokenizer_path,
    writeTinyEmbeddingGemmaModules(files.dir),
    files.dir / "embedding_gemma_tiny.bin",
  };
}

/**
 * @brief Make the tiny EmbeddingGemma nntrainer config
 */
causallm::json makeTinyEmbeddingGemmaNntrainerConfig(
  const TinyEmbeddingGemmaFiles &files,
  const causallm_test::TinyCausalLMDataType &data_type) {
  auto cfg =
    causallm_test::makeTinyNntrainerConfig(files.tokenizer_path, data_type);
  cfg["model_type"] = "Embedding";
  cfg["module_config_path"] = files.modules_path.string();
  return cfg;
}

/**
 * @brief Make the tiny Gemma3 layer dtype map
 */
std::map<std::string, ml::train::TensorDim::DataType>
makeGemma3LayerDtypeMap(const causallm_test::TinyCausalLMDataType &data_type) {
  std::map<std::string, ml::train::TensorDim::DataType> dtype_map;

  if (data_type.embedding_dtype != "FP32")
    dtype_map["embedding0"] =
      causallm_test::toTensorDataType(data_type.embedding_dtype);

  if (data_type.fc_layer_dtype != "FP32") {
    const auto dtype =
      causallm_test::toTensorDataType(data_type.fc_layer_dtype);
    for (int i = 0; i < tiny_gemma3_num_layers; ++i) {
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
 * @brief Make the tiny EmbeddingGemma Q4_0 layer dtype map
 */
std::map<std::string, ml::train::TensorDim::DataType>
makeEmbeddingGemmaQ40LayerDtypeMap() {
  auto data_type = causallm_test::makeTinyQ40Fp32DataType();
  auto dtype_map = makeGemma3LayerDtypeMap(data_type);
  const auto dtype = ml::train::TensorDim::DataType::Q4_0;

  dtype_map["2"] = dtype;
  dtype_map["3"] = dtype;

  return dtype_map;
}

/**
 * @brief Create a loaded tiny EmbeddingGemma model
 */
std::unique_ptr<TinyEmbeddingGemma>
makeLoadedEmbeddingGemma(const TinyEmbeddingGemmaFiles &files) {
  const auto fp32_data_type = causallm_test::makeTinyFp32DataType();
  const auto q40_data_type = causallm_test::makeTinyQ40Fp32DataType();
  auto source_model_cfg = makeTinyEmbeddingGemmaConfig();
  auto source_generation_cfg = causallm_test::makeTinyGenerationConfig();
  auto source_nntr_cfg =
    makeTinyEmbeddingGemmaNntrainerConfig(files, fp32_data_type);

  TinyEmbeddingGemma source(source_model_cfg, source_generation_cfg,
                            source_nntr_cfg);
  source.initializeModel();
  source.setDeterministicWeights();
  source.saveWeightWithDtype(files.weight_path.string(),
                             makeEmbeddingGemmaQ40LayerDtypeMap());

  auto loaded_model_cfg = makeTinyEmbeddingGemmaConfig();
  auto loaded_generation_cfg = causallm_test::makeTinyGenerationConfig();
  auto loaded_nntr_cfg =
    makeTinyEmbeddingGemmaNntrainerConfig(files, q40_data_type);
  auto loaded = std::make_unique<TinyEmbeddingGemma>(
    loaded_model_cfg, loaded_generation_cfg, loaded_nntr_cfg);
  loaded->initializeModel();
  loaded->loadWeight(files.weight_path.string());

  return loaded;
}

/**
 * @brief Make the expected tiny Gemma3 prefill logits
 */
std::vector<float> makeExpectedGemma3Logits() {
  std::vector<float> logits(32, 0.0f);
  logits[1] = 8.0f;
  logits[4] = 16.0f;
  return logits;
}

/**
 * @brief Make a Gemma3 tiny CausalLM test case
 */
causallm_test::TinyCausalLMCase
makeGemma3Case(const causallm_test::TinyCausalLMDataType &data_type) {
  return {
    "Gemma3_" + data_type.name,
    data_type,
    {"hello tok4", makeExpectedGemma3Logits(),
     data_type.name == "FP32"       ? 1e-4f
     : data_type.name == "Q40_FP16" ? 2e-3f
                                    : 1e-3f},
    makeTinyGemma3Config,
    makeGemma3LayerDtypeMap,
    [](causallm::json &cfg, causallm::json &generation_cfg,
       causallm::json &nntr_cfg) {
      return std::make_unique<TinyGemma3CausalLM>(cfg, generation_cfg,
                                                  nntr_cfg);
    },
    [](causallm_test::TinyCausalLMRunner &runner) {
      setupGemma3DeterministicWeights(
        static_cast<TinyGemma3CausalLM &>(runner));
    },
  };
}

/**
 * @brief Parameterized fixture for tiny Gemma3 model cases
 */
class Gemma3TinyModelTest
  : public ::testing::TestWithParam<causallm_test::TinyCausalLMCase> {
protected:
  /**
   * @brief Make test files for the current parameterized case
   */
  causallm_test::TinyCausalLMFiles makeFiles() const {
    const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string suite_name = "Gemma3TinyModelTest";
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
TEST_P(Gemma3TinyModelTest, GreedyGenerationSelectsArgmaxLogit) {
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
TEST_P(Gemma3TinyModelTest, WeightRoundTripProducesSameLogits) {
  const auto files = makeFiles();
  causallm_test::expectWeightRoundTripProducesSameLogits(GetParam(), files);
}

/**
 * @brief Test that a prompt produces the expected golden logits
 */
TEST_P(Gemma3TinyModelTest, PromptProducesExpectedLogits) {
  const auto files = makeFiles();
  causallm_test::expectPromptProducesExpectedLogits(GetParam(), files);
}

INSTANTIATE_TEST_SUITE_P(
  Gemma3, Gemma3TinyModelTest,
  ::testing::Values(makeGemma3Case(causallm_test::makeTinyFp32DataType()),
                    makeGemma3Case(causallm_test::makeTinyQ40Fp32DataType())),
  [](const ::testing::TestParamInfo<causallm_test::TinyCausalLMCase> &info) {
    return info.param.name;
  });

#ifdef ENABLE_FP16
INSTANTIATE_TEST_SUITE_P(
  Gemma3Fp16, Gemma3TinyModelTest,
  ::testing::Values(makeGemma3Case(causallm_test::makeTinyQ40Fp16DataType())),
  [](const ::testing::TestParamInfo<causallm_test::TinyCausalLMCase> &info) {
    return info.param.name;
  });
#endif

/**
 * @brief Test that tiny EmbeddingGemma can save/load and encode
 */
TEST(EmbeddingGemmaTinyModelTest,
     WeightRoundTripEncodesPromptWithQ40BidirectionalModel) {
  const auto files = makeEmbeddingGemmaFiles();
  auto model = makeLoadedEmbeddingGemma(files);

  EXPECT_FALSE(model->isCausalForTest());

  std::vector<float> embedding;
  ASSERT_NO_THROW(embedding = model->encodePrompt("hello tok4"));
  ASSERT_EQ(embedding.size(), 64u);

  bool has_non_zero = false;
  for (float value : embedding) {
    ASSERT_TRUE(std::isfinite(value));
    has_non_zero = has_non_zero || std::abs(value) > 1e-5f;
  }
  EXPECT_TRUE(has_non_zero);
}

} // namespace
