// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_causallm_qwen2.cpp
 * @date   18 May 2026
 * @brief  Tiny Qwen2 CausalLM model unit tests
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <causallm_test_utils.h>

#include <gtest/gtest.h>

#include <layer.h>
#include <layer_context.h>
#include <qwen2_causallm.h>
#include <qwen2_embedding.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

/**
 * @brief Anonymous namespace for tiny Qwen2 test helpers
 */
namespace {

/**
 * @brief Tiny Qwen2 CausalLM adapter (inference shared by CausalLMTestAdapter)
 */
using TinyQwen2CausalLM =
  causallm_test::CausalLMTestAdapter<causallm::Qwen2CausalLM>;

/**
 * @brief Populate deterministic tiny Qwen2 weights for golden token tests
 */
void setupQwen2DeterministicWeights(TinyQwen2CausalLM &model) {
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
          weight.setValue(0.0f);
          weight.setValue(0, 0, 1, 0, 1.0f);
          weight.setValue(0, 0, 4, 0, 2.0f);
        }
      }
    });
}

/**
 * @brief Files generated for one tiny Qwen2.5 embedding test invocation
 */
struct TinyQwen25EmbeddingFiles {
  std::filesystem::path dir;            /**< Temporary test directory */
  std::filesystem::path tokenizer_path; /**< Tiny tokenizer.json path */
  std::filesystem::path modules_path;   /**< Tiny modules.json path */
  std::filesystem::path weight_path;    /**< Tiny model weight path */
};

/**
 * @brief Tiny Qwen2.5 Embedding adapter for model-level tests
 */
class TinyQwen25Embedding final : public causallm::Qwen2Embedding {
public:
  /**
   * @brief Construct a tiny Qwen2.5 Embedding test adapter
   */
  TinyQwen25Embedding(causallm::json &cfg, causallm::json &generation_cfg,
                      causallm::json &nntr_cfg) :
    causallm::Transformer(cfg, generation_cfg, nntr_cfg,
                          causallm::ModelType::EMBEDDING),
    causallm::Qwen2Embedding(cfg, generation_cfg, nntr_cfg) {}

  /**
   * @brief Initialize the tiny Qwen2.5 embedding model
   */
  void initializeModel() { initialize(); }

  /**
   * @brief Save tiny Qwen2.5 embedding weights with dtype conversion
   */
  void saveWeightWithDtype(
    const std::string &path,
    const std::map<std::string, ml::train::TensorDim::DataType>
      &layer_dtype_map) {
    save_weight(path, ml::train::TensorDim::DataType::NONE, layer_dtype_map);
  }

  /**
   * @brief Load tiny Qwen2.5 embedding model weights
   */
  void loadWeight(const std::string &path) { load_weight(path); }

  /**
   * @brief Set deterministic tiny Qwen2.5 embedding weights
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
 * @brief Make tiny Qwen2.5 SentenceTransformer module config files
 */
std::filesystem::path
writeTinyQwen25EmbeddingModules(const std::filesystem::path &dir) {
  auto modules_path = dir / "modules.json";
  auto pooling_dir = dir / "1_Pooling";
  std::filesystem::create_directories(pooling_dir);

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
  "include_prompt": false
})");

  return modules_path;
}

/**
 * @brief Make the tiny Qwen2 model config
 */
causallm::json makeTinyQwen2Config() {
  return {
    {"architectures", {"Qwen2ForCausalLM"}},
    {"bos_token_id", 0},
    {"eos_token_id", {31}},
    {"hidden_size", 64},
    {"intermediate_size", 64},
    {"is_causal", true},
    {"max_position_embeddings", 8},
    {"num_attention_heads", 8},
    {"num_hidden_layers", 1},
    {"num_key_value_heads", 4},
    {"rms_norm_eps", 1e-5},
    {"rope_theta", 10000},
    {"tie_word_embeddings", true},
    {"vocab_size", 32},
  };
}

/**
 * @brief Make the tiny Qwen2.5 embedding model config
 */
causallm::json makeTinyQwen25EmbeddingConfig() {
  auto cfg = makeTinyQwen2Config();
  cfg["architectures"] = {"Qwen2Model"};
  cfg.erase("is_causal");
  return cfg;
}

/**
 * @brief Make tiny Qwen2.5 embedding test files
 */
TinyQwen25EmbeddingFiles makeQwen25EmbeddingFiles() {
  const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
  std::string suite_name = "Qwen25EmbeddingTinyModelTest";
  std::string test_name = "Unknown";

  if (info != nullptr) {
    suite_name = info->test_suite_name();
    test_name = info->name();
  }

  auto files = causallm_test::makeTinyCausalLMFiles(suite_name, test_name,
                                                    "Qwen25Embedding_Q40_FP32");

  return {
    files.dir,
    files.tokenizer_path,
    writeTinyQwen25EmbeddingModules(files.dir),
    files.dir / "qwen25_embedding_tiny.bin",
  };
}

/**
 * @brief Make the tiny Qwen2.5 embedding nntrainer config
 */
causallm::json makeTinyQwen25EmbeddingNntrainerConfig(
  const TinyQwen25EmbeddingFiles &files,
  const causallm_test::TinyCausalLMDataType &data_type) {
  auto cfg =
    causallm_test::makeTinyNntrainerConfig(files.tokenizer_path, data_type);
  cfg["model_type"] = "Embedding";
  cfg["module_config_path"] = files.modules_path.string();
  return cfg;
}

/**
 * @brief Make the tiny Qwen2 layer dtype map
 */
std::map<std::string, ml::train::TensorDim::DataType>
makeQwen2LayerDtypeMap(const causallm_test::TinyCausalLMDataType &data_type) {
  std::map<std::string, ml::train::TensorDim::DataType> dtype_map;

  if (data_type.embedding_dtype != "FP32")
    dtype_map["embedding0"] =
      causallm_test::toTensorDataType(data_type.embedding_dtype);

  if (data_type.fc_layer_dtype != "FP32") {
    const auto dtype =
      causallm_test::toTensorDataType(data_type.fc_layer_dtype);
    dtype_map["layer0_wq"] = dtype;
    dtype_map["layer0_wk"] = dtype;
    dtype_map["layer0_wv"] = dtype;
    dtype_map["layer0_attention_out"] = dtype;
    dtype_map["layer0_ffn_gate"] = dtype;
    dtype_map["layer0_ffn_up"] = dtype;
    dtype_map["layer0_ffn_down"] = dtype;
  }

  if (data_type.lmhead_dtype != "FP32")
    dtype_map["output_of_causallm"] =
      causallm_test::toTensorDataType(data_type.lmhead_dtype);

  return dtype_map;
}

/**
 * @brief Make the tiny Qwen2.5 embedding Q4_0 layer dtype map
 */
std::map<std::string, ml::train::TensorDim::DataType>
makeQwen25EmbeddingQ40LayerDtypeMap() {
  std::map<std::string, ml::train::TensorDim::DataType> dtype_map;
  const auto dtype = ml::train::TensorDim::DataType::Q4_0;

  dtype_map["embedding0"] = dtype;
  dtype_map["layer0_wq"] = dtype;
  dtype_map["layer0_wk"] = dtype;
  dtype_map["layer0_wv"] = dtype;
  dtype_map["layer0_attention_out"] = dtype;
  dtype_map["layer0_ffn_gate"] = dtype;
  dtype_map["layer0_ffn_up"] = dtype;
  dtype_map["layer0_ffn_down"] = dtype;

  return dtype_map;
}

/**
 * @brief Create a loaded tiny Qwen2.5 embedding model
 */
std::unique_ptr<TinyQwen25Embedding>
makeLoadedQwen25Embedding(const TinyQwen25EmbeddingFiles &files) {
  const auto fp32_data_type = causallm_test::makeTinyFp32DataType();
  const auto q40_data_type = causallm_test::makeTinyQ40Fp32DataType();
  auto source_model_cfg = makeTinyQwen25EmbeddingConfig();
  auto source_generation_cfg = causallm_test::makeTinyGenerationConfig();
  auto source_nntr_cfg =
    makeTinyQwen25EmbeddingNntrainerConfig(files, fp32_data_type);

  TinyQwen25Embedding source(source_model_cfg, source_generation_cfg,
                             source_nntr_cfg);
  source.initializeModel();
  source.setDeterministicWeights();
  source.saveWeightWithDtype(files.weight_path.string(),
                             makeQwen25EmbeddingQ40LayerDtypeMap());

  auto loaded_model_cfg = makeTinyQwen25EmbeddingConfig();
  auto loaded_generation_cfg = causallm_test::makeTinyGenerationConfig();
  auto loaded_nntr_cfg =
    makeTinyQwen25EmbeddingNntrainerConfig(files, q40_data_type);
  auto loaded = std::make_unique<TinyQwen25Embedding>(
    loaded_model_cfg, loaded_generation_cfg, loaded_nntr_cfg);
  loaded->initializeModel();
  loaded->loadWeight(files.weight_path.string());

  return loaded;
}

/**
 * @brief Make the expected tiny Qwen2 prefill logits
 */
std::vector<float> makeExpectedQwen2Logits() {
  std::vector<float> logits(32, 0.0f);
  logits[1] = 7.99936008f;
  logits[4] = 15.99872017f;
  return logits;
}

/**
 * @brief Make a Qwen2 tiny CausalLM test case
 */
causallm_test::TinyCausalLMCase
makeQwen2Case(const causallm_test::TinyCausalLMDataType &data_type) {
  return {
    "Qwen2_" + data_type.name,
    data_type,
    {"hello tok4", makeExpectedQwen2Logits(),
     data_type.name == "FP32" ? 1e-4f : 1e-3f},
    makeTinyQwen2Config,
    makeQwen2LayerDtypeMap,
    [](causallm::json &cfg, causallm::json &generation_cfg,
       causallm::json &nntr_cfg) {
      return std::make_unique<TinyQwen2CausalLM>(cfg, generation_cfg, nntr_cfg);
    },
    [](causallm_test::TinyCausalLMRunner &runner) {
      setupQwen2DeterministicWeights(static_cast<TinyQwen2CausalLM &>(runner));
    },
  };
}

/**
 * @brief Parameterized fixture for tiny Qwen2 CausalLM model cases
 */
class Qwen2CausalLMTinyModelTest
  : public ::testing::TestWithParam<causallm_test::TinyCausalLMCase> {
protected:
  /**
   * @brief Make test files for the current parameterized case
   */
  causallm_test::TinyCausalLMFiles makeFiles() const {
    const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string suite_name = "Qwen2CausalLMTinyModelTest";
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
 * @brief Logits processor that forces one token and records callbacks
 */
class ForcingLogitsProcessor final : public causallm::LogitsProcessor {
public:
  explicit ForcingLogitsProcessor(unsigned int token) : token(token) {}

  /**
   * @brief Mask every logit except the forced token
   */
  void process(float *logits, unsigned int vocab_size,
               unsigned int batch_index) override {
    ++process_count;
    last_batch_index = batch_index;
    last_vocab_size = vocab_size;

    for (unsigned int i = 0; i < vocab_size; ++i)
      logits[i] = -std::numeric_limits<float>::infinity();
    logits[token] = 100.0f;
  }

  /**
   * @brief Record the accepted token
   */
  void acceptToken(unsigned int token_id, unsigned int batch_index) override {
    ++accept_count;
    accepted_token = token_id;
    accepted_batch_index = batch_index;
  }

  /**
   * @brief Record reset calls
   */
  void reset() override { ++reset_count; }

  unsigned int token;
  unsigned int process_count = 0;
  unsigned int accept_count = 0;
  unsigned int reset_count = 0;
  unsigned int last_batch_index = 99;
  unsigned int accepted_batch_index = 99;
  unsigned int last_vocab_size = 0;
  unsigned int accepted_token = 0;
};

/**
 * @brief Make a direct tiny Qwen2 model for logits processor hook tests
 */
std::unique_ptr<TinyQwen2CausalLM>
makeDirectTinyQwen2Model(
  const causallm_test::TinyCausalLMFiles &files,
  const causallm_test::TinyCausalLMCase &test_case,
  const std::vector<unsigned int> &bad_word_ids = {}) {
  auto config =
    causallm_test::makeTinyCausalLMConfig(test_case, files.tokenizer_path);
  config.nntrainer["bad_word_ids"] = bad_word_ids;
  return std::make_unique<TinyQwen2CausalLM>(config.model, config.generation,
                                             config.nntrainer);
}

/**
 * @brief Test that Transformer exposes the configured vocabulary size
 */
TEST_P(Qwen2CausalLMTinyModelTest, TransformerReturnsConfiguredVocabSize) {
  const auto files = makeFiles();
  auto model = makeDirectTinyQwen2Model(files, GetParam());

  EXPECT_EQ(model->getVocabSize(), 32u);
}

/**
 * @brief Test that Transformer exposes its owned tokenizer
 */
TEST_P(Qwen2CausalLMTinyModelTest, TransformerReturnsOwnedTokenizer) {
  const auto files = makeFiles();
  auto model = makeDirectTinyQwen2Model(files, GetParam());

  EXPECT_NE(model->getTokenizer(), nullptr);
}

/**
 * @brief Test that embedding_file_name reaches the embedding layer sidecar path
 */
TEST_P(Qwen2CausalLMTinyModelTest,
       EmbeddingFileNameIsPassedToEmbeddingLayerSidecarPath) {
  const auto files = makeFiles();
  auto config =
    causallm_test::makeTinyCausalLMConfig(GetParam(), files.tokenizer_path);
  config.model["tie_word_embeddings"] = false;
  config.nntrainer["embedding_file_name"] =
    (files.dir / "missing_sidecar_lut.bin").string();
  auto model = std::make_unique<TinyQwen2CausalLM>(
    config.model, config.generation, config.nntrainer);

  EXPECT_THROW(model->initializeModel(), std::runtime_error);
}

/**
 * @brief Test that a logits processor can force greedy generation
 */
TEST_P(Qwen2CausalLMTinyModelTest,
       LogitsProcessorForcesGreedyGenerationAndReceivesAcceptedToken) {
  const auto files = makeFiles();
  auto model = makeDirectTinyQwen2Model(files, GetParam(), {7});
  ForcingLogitsProcessor processor(7);
  std::vector<float> logits(32, -2.0f);
  logits[3] = 5.0f;
  unsigned int input_ids[4] = {1, 0, 0, 0};

  model->setLogitsProcessor(&processor);
  auto ids = model->generateFromLogits(logits.data(), false, 1.0f, input_ids,
                                       1);

  ASSERT_EQ(ids.size(), 1u);
  EXPECT_EQ(ids[0], 7u);
  EXPECT_EQ(processor.process_count, 1u);
  EXPECT_EQ(processor.accept_count, 1u);
  EXPECT_EQ(processor.last_vocab_size, 32u);
  EXPECT_EQ(processor.last_batch_index, 0u);
  EXPECT_EQ(processor.accepted_token, 7u);
  EXPECT_EQ(processor.accepted_batch_index, 0u);
}

/**
 * @brief Test that detaching a logits processor restores greedy argmax
 */
TEST_P(Qwen2CausalLMTinyModelTest,
       DetachingLogitsProcessorRestoresGreedyArgmax) {
  const auto files = makeFiles();
  auto model = makeDirectTinyQwen2Model(files, GetParam());
  ForcingLogitsProcessor processor(7);
  unsigned int input_ids[4] = {1, 0, 0, 0};

  model->setLogitsProcessor(&processor);
  model->setLogitsProcessor(nullptr);

  std::vector<float> logits(32, -2.0f);
  logits[3] = 5.0f;
  logits[7] = 4.0f;
  auto ids = model->generateFromLogits(logits.data(), false, 1.0f, input_ids,
                                       1);

  ASSERT_EQ(ids.size(), 1u);
  EXPECT_EQ(ids[0], 3u);
  EXPECT_EQ(processor.process_count, 0u);
  EXPECT_EQ(processor.accept_count, 0u);
}

/**
 * @brief Test that resetLogitsProcessor forwards to the attached processor
 */
TEST_P(Qwen2CausalLMTinyModelTest, ResetLogitsProcessorForwardsReset) {
  const auto files = makeFiles();
  auto model = makeDirectTinyQwen2Model(files, GetParam());
  ForcingLogitsProcessor processor(7);

  model->setLogitsProcessor(&processor);
  model->resetLogitsProcessor();

  EXPECT_EQ(processor.reset_count, 1u);
}

/**
 * @brief Test that greedy generation chooses the argmax logit
 */
TEST_P(Qwen2CausalLMTinyModelTest, GreedyGenerationSelectsArgmaxLogit) {
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
TEST_P(Qwen2CausalLMTinyModelTest, WeightRoundTripProducesSameLogits) {
  const auto files = makeFiles();
  causallm_test::expectWeightRoundTripProducesSameLogits(GetParam(), files);
}

/**
 * @brief Test that a prompt produces the expected golden logits
 */
TEST_P(Qwen2CausalLMTinyModelTest, PromptProducesExpectedLogits) {
  const auto files = makeFiles();
  causallm_test::expectPromptProducesExpectedLogits(GetParam(), files);
}

INSTANTIATE_TEST_SUITE_P(
  Qwen2, Qwen2CausalLMTinyModelTest,
  ::testing::Values(makeQwen2Case(causallm_test::makeTinyFp32DataType()),
                    makeQwen2Case(causallm_test::makeTinyQ40Fp32DataType())),
  [](const ::testing::TestParamInfo<causallm_test::TinyCausalLMCase> &info) {
    return info.param.name;
  });

/**
 * @brief Test that a tiny Qwen2.5 embedding model can save/load and encode
 */
TEST(Qwen25EmbeddingTinyModelTest,
     WeightRoundTripEncodesPromptWithQ40BidirectionalModel) {
  const auto files = makeQwen25EmbeddingFiles();
  auto model = makeLoadedQwen25Embedding(files);

  EXPECT_FALSE(model->isCausalForTest());
  EXPECT_EQ(model->getEmbeddingDim(), 64);

  std::vector<float> embedding;
  ASSERT_NO_THROW(embedding = model->encodePrompt("hello tok4"));
  ASSERT_EQ(embedding.size(), 64u);

  bool has_non_zero = false;
  for (float value : embedding) {
    ASSERT_TRUE(std::isfinite(value));
    has_non_zero = has_non_zero || value != 0.0f;
  }
  EXPECT_TRUE(has_non_zero);
}

} // namespace
