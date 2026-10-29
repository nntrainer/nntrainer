// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_causallm_qwen3.cpp
 * @date   15 May 2026
 * @brief  Tiny Qwen3 model unit tests
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <causallm_test_utils.h>

#include <gtest/gtest.h>

#include <layer.h>
#include <layer_context.h>
#include <qwen3_causallm.h>
#include <qwen3_embedding.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <utility>

/**
 * @brief Helpers for tiny Qwen3 model tests
 */
namespace {

/**
 * @brief Tiny Qwen3 CausalLM adapter (inference shared by CausalLMTestAdapter)
 */
using TinyQwen3CausalLM =
  causallm_test::CausalLMTestAdapter<causallm::Qwen3CausalLM>;

/**
 * @brief Populate deterministic tiny Qwen3 weights for golden token tests
 */
void setupQwen3DeterministicWeights(TinyQwen3CausalLM &model) {
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
 * @brief Files generated for one tiny Qwen3 embedding test invocation
 */
struct TinyQwen3EmbeddingFiles {
  std::filesystem::path dir;            /**< Temporary test directory */
  std::filesystem::path tokenizer_path; /**< Tiny tokenizer.json path */
  std::filesystem::path modules_path;   /**< Tiny modules.json path */
  std::filesystem::path weight_path;    /**< Tiny model weight path */
};

/**
 * @brief Tiny Qwen3 Embedding adapter for model-level tests
 */
class TinyQwen3Embedding final : public causallm::Qwen3Embedding {
public:
  /**
   * @brief Construct a tiny Qwen3 Embedding test adapter
   */
  TinyQwen3Embedding(causallm::json &cfg, causallm::json &generation_cfg,
                     causallm::json &nntr_cfg) :
    causallm::Transformer(cfg, generation_cfg, nntr_cfg,
                          causallm::ModelType::EMBEDDING),
    causallm::Qwen3Embedding(cfg, generation_cfg, nntr_cfg) {}

  /**
   * @brief Initialize the tiny Qwen3 embedding model
   */
  void initializeModel() { initialize(); }

  /**
   * @brief Save tiny Qwen3 embedding weights with dtype conversion
   */
  void saveWeightWithDtype(
    const std::string &path,
    const std::map<std::string, ml::train::TensorDim::DataType>
      &layer_dtype_map) {
    save_weight(path, ml::train::TensorDim::DataType::NONE, layer_dtype_map);
  }

  /**
   * @brief Load tiny Qwen3 embedding model weights
   */
  void loadWeight(const std::string &path) { load_weight(path); }

  /**
   * @brief Set deterministic tiny Qwen3 embedding weights
   */
  void setDeterministicWeights() {
    auto set_weights = [](ml::train::Layer &layer,
                          nntrainer::RunLayerContext &context, void *) {
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
 * @brief Make tiny SentenceTransformer module config files
 */
std::filesystem::path
writeTinyQwen3EmbeddingModules(const std::filesystem::path &dir) {
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
  }
])");

  writeTextFile(pooling_dir / "config.json", R"({
  "word_embedding_dimension": 64,
  "pooling_mode_cls_token": false,
  "pooling_mode_mean_tokens": false,
  "pooling_mode_max_tokens": false,
  "pooling_mode_mean_sqrt_len_tokens": false,
  "pooling_mode_weightedmean_tokens": false,
  "pooling_mode_lasttoken": true,
  "include_prompt": true
})");

  return modules_path;
}

/**
 * @brief Make the tiny Qwen3 model config
 */
causallm::json makeTinyQwen3Config() {
  return {
    {"architectures", {"Qwen3ForCausalLM"}},
    {"bos_token_id", 0},
    {"eos_token_id", {31}},
    {"head_dim", 8},
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
 * @brief Make the tiny Qwen3 embedding model config
 */
causallm::json makeTinyQwen3EmbeddingConfig() {
  auto cfg = makeTinyQwen3Config();
  cfg["vocab_size"] = 33;
  return cfg;
}

/**
 * @brief Make tiny Qwen3 embedding test files
 */
TinyQwen3EmbeddingFiles makeQwen3EmbeddingFiles() {
  const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
  std::string suite_name = "Qwen3EmbeddingTinyModelTest";
  std::string test_name = "Unknown";

  if (info != nullptr) {
    suite_name = info->test_suite_name();
    test_name = info->name();
  }

  auto files = causallm_test::makeTinyCausalLMFiles(suite_name, test_name,
                                                    "Qwen3Embedding_Q40_FP32");

  return {
    files.dir,
    files.tokenizer_path,
    writeTinyQwen3EmbeddingModules(files.dir),
    files.dir / "qwen3_embedding_tiny.bin",
  };
}

/**
 * @brief Make the tiny Qwen3 embedding nntrainer config
 */
causallm::json makeTinyQwen3EmbeddingNntrainerConfig(
  const TinyQwen3EmbeddingFiles &files,
  const causallm_test::TinyCausalLMDataType &data_type) {
  auto cfg =
    causallm_test::makeTinyNntrainerConfig(files.tokenizer_path, data_type);
  cfg["model_type"] = "Embedding";
  cfg["module_config_path"] = files.modules_path.string();
  return cfg;
}

/**
 * @brief Make the tiny Qwen3 layer dtype map
 */
std::map<std::string, ml::train::TensorDim::DataType>
makeQwen3LayerDtypeMap(const causallm_test::TinyCausalLMDataType &data_type) {
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
    dtype_map["layer0_ffn_up"] = dtype;
    dtype_map["layer0_ffn_gate"] = dtype;
    dtype_map["layer0_ffn_down"] = dtype;
  }

  if (data_type.lmhead_dtype != "FP32")
    dtype_map["output_of_causallm"] =
      causallm_test::toTensorDataType(data_type.lmhead_dtype);

  return dtype_map;
}

/**
 * @brief Make the tiny Qwen3 embedding Q4_0 layer dtype map
 */
std::map<std::string, ml::train::TensorDim::DataType>
makeQwen3EmbeddingQ40LayerDtypeMap() {
  std::map<std::string, ml::train::TensorDim::DataType> dtype_map;
  const auto dtype = ml::train::TensorDim::DataType::Q4_0;

  dtype_map["embedding0"] = dtype;
  dtype_map["layer0_wq"] = dtype;
  dtype_map["layer0_wk"] = dtype;
  dtype_map["layer0_wv"] = dtype;
  dtype_map["layer0_attention_out"] = dtype;
  dtype_map["layer0_ffn_up"] = dtype;
  dtype_map["layer0_ffn_gate"] = dtype;
  dtype_map["layer0_ffn_down"] = dtype;

  return dtype_map;
}

/**
 * @brief Create a loaded tiny Qwen3 embedding model
 */
std::unique_ptr<TinyQwen3Embedding>
makeLoadedQwen3Embedding(const TinyQwen3EmbeddingFiles &files) {
  const auto fp32_data_type = causallm_test::makeTinyFp32DataType();
  const auto q40_data_type = causallm_test::makeTinyQ40Fp32DataType();
  auto source_model_cfg = makeTinyQwen3EmbeddingConfig();
  auto source_generation_cfg = causallm_test::makeTinyGenerationConfig();
  auto source_nntr_cfg =
    makeTinyQwen3EmbeddingNntrainerConfig(files, fp32_data_type);

  TinyQwen3Embedding source(source_model_cfg, source_generation_cfg,
                            source_nntr_cfg);
  source.initializeModel();
  source.setDeterministicWeights();
  source.saveWeightWithDtype(files.weight_path.string(),
                             makeQwen3EmbeddingQ40LayerDtypeMap());

  auto loaded_model_cfg = makeTinyQwen3EmbeddingConfig();
  auto loaded_generation_cfg = causallm_test::makeTinyGenerationConfig();
  auto loaded_nntr_cfg =
    makeTinyQwen3EmbeddingNntrainerConfig(files, q40_data_type);
  auto loaded = std::make_unique<TinyQwen3Embedding>(
    loaded_model_cfg, loaded_generation_cfg, loaded_nntr_cfg);
  loaded->initializeModel();
  loaded->loadWeight(files.weight_path.string());

  return loaded;
}

/**
 * @brief Make the expected tiny Qwen3 prefill logits
 */
std::vector<float> makeExpectedQwen3Logits() {
  std::vector<float> logits(32, 0.0f);
  logits[1] = 7.99936008f;
  logits[4] = 15.99872017f;
  return logits;
}

/**
 * @brief Make a Qwen3 tiny CausalLM test case
 */
causallm_test::TinyCausalLMCase
makeQwen3Case(const causallm_test::TinyCausalLMDataType &data_type) {
  return {
    "Qwen3_" + data_type.name,
    data_type,
    {"hello tok4", makeExpectedQwen3Logits(),
     data_type.name == "FP32"       ? 1e-4f
     : data_type.name == "Q40_FP16" ? 2e-3f
                                    : 1e-3f},
    makeTinyQwen3Config,
    makeQwen3LayerDtypeMap,
    [](causallm::json &cfg, causallm::json &generation_cfg,
       causallm::json &nntr_cfg) {
      return std::make_unique<TinyQwen3CausalLM>(cfg, generation_cfg, nntr_cfg);
    },
    [](causallm_test::TinyCausalLMRunner &runner) {
      setupQwen3DeterministicWeights(static_cast<TinyQwen3CausalLM &>(runner));
    },
  };
}

/**
 * @brief Parameterized fixture for tiny CausalLM model cases
 */
class CausalLMTinyModelTest
  : public ::testing::TestWithParam<causallm_test::TinyCausalLMCase> {
protected:
  /**
   * @brief Make test files for the current parameterized case
   */
  causallm_test::TinyCausalLMFiles makeFiles() const {
    const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string suite_name = "CausalLMTinyModelTest";
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
TEST_P(CausalLMTinyModelTest, GreedyGenerationSelectsArgmaxLogit) {
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
TEST_P(CausalLMTinyModelTest, WeightRoundTripProducesSameLogits) {
  const auto files = makeFiles();
  causallm_test::expectWeightRoundTripProducesSameLogits(GetParam(), files);
}

/**
 * @brief Test that a prompt produces the expected golden logits
 */
TEST_P(CausalLMTinyModelTest, PromptProducesExpectedLogits) {
  const auto files = makeFiles();
  causallm_test::expectPromptProducesExpectedLogits(GetParam(), files);
}

INSTANTIATE_TEST_SUITE_P(
  Qwen3, CausalLMTinyModelTest,
  ::testing::Values(makeQwen3Case(causallm_test::makeTinyFp32DataType()),
                    makeQwen3Case(causallm_test::makeTinyQ40Fp32DataType())),
  [](const ::testing::TestParamInfo<causallm_test::TinyCausalLMCase> &info) {
    return info.param.name;
  });

#ifdef ENABLE_FP16
INSTANTIATE_TEST_SUITE_P(
  Qwen3Fp16, CausalLMTinyModelTest,
  ::testing::Values(makeQwen3Case(causallm_test::makeTinyQ40Fp16DataType())),
  [](const ::testing::TestParamInfo<causallm_test::TinyCausalLMCase> &info) {
    return info.param.name;
  });
#endif

/** @brief Real two-attention graph with a batch-1, eight-token CPU cache. */
class QwenCpuKVGraphTest : public ::testing::Test {
protected:
  void SetUp() override {
    const auto files = causallm_test::makeTinyCausalLMFiles(
      "QwenCpuKVGraphTest",
      ::testing::UnitTest::GetInstance()->current_test_info()->name(), "FP32");
    config_ = makeTinyQwen3Config();
    config_["num_hidden_layers"] = 2;
    generation_ = causallm_test::makeTinyGenerationConfig();
    nntrainer_ = causallm_test::makeTinyNntrainerConfig(
      files.tokenizer_path, causallm_test::makeTinyFp32DataType());
    nntrainer_["init_seq_len"] = 8;
    model_ =
      std::make_unique<TinyQwen3CausalLM>(config_, generation_, nntrainer_);
    ASSERT_NO_THROW(model_->initializeModel());
    setupQwen3DeterministicWeights(*model_);
  }

  void rejectForeignPlane(bool key) {
    model_->prepareKVCache();
    auto &cache = model_->cacheManager();
    ASSERT_TRUE(cache.usesCpuStorage());
    auto &plane = key ? cache.getKeyCache(0) : cache.getValueCache(0);
    const auto original = plane.getMemoryData();
    const auto offset = plane.getOffset();
    nntrainer::Tensor foreign(plane.getDim(), true);
    std::memset(foreign.getData<char>(), 0x5a, foreign.bytes());
    plane.setData(foreign.getMemoryData(), foreign.getOffset(), false);

    try {
      model_->stepLogitsFromIds({1, 4}, 0);
      FAIL() << "MHA accepted a cache plane outside its active lease";
    } catch (const std::logic_error &error) {
      EXPECT_STREQ(error.what(),
                   "CPU MHA cache does not belong to the active KV step");
    }
    EXPECT_EQ(cache.getPosition(), 0u);
    const char *bytes = foreign.getData<char>();
    EXPECT_TRUE(std::all_of(bytes, bytes + foreign.bytes(),
                            [](char value) { return value == 0x5a; }));

    plane.setData(original, offset, false);
    EXPECT_NO_THROW(model_->stepLogitsFromIds({1, 4}, 0));
    EXPECT_EQ(cache.getPosition(), 2u);
  }

  causallm_test::ScopedEnv engine_{"NNTR_ENGINE", "cpu"};
  causallm_test::ScopedEnv legacy_{"NNTR_CPU_KV_LEGACY", "0"};
  causallm::json config_, generation_, nntrainer_;
  std::unique_ptr<TinyQwen3CausalLM> model_;
};

TEST_F(QwenCpuKVGraphTest, CommitsAtCapacityAndResetsRepeatedly) {
  for (unsigned int repetition = 0; repetition < 3; ++repetition) {
    if (repetition)
      model_->cacheManager().reset();
    const auto logits = model_->stepLogitsFromIds({1, 4, 4, 4, 4, 4, 4}, 0);
    ASSERT_EQ(logits.size(), 32u);
    EXPECT_NEAR(logits[4], makeExpectedQwen3Logits()[4], 1e-4f);
    EXPECT_EQ(model_->getKvLen(), 7);
    EXPECT_NO_THROW(model_->stepLogitsFromIds({4}, 7));
    EXPECT_EQ(model_->getKvLen(), 8);
    EXPECT_THROW(model_->stepLogitsFromIds({4}, 8), std::out_of_range);
    EXPECT_EQ(model_->getKvLen(), 8);
  }
}

TEST_F(QwenCpuKVGraphTest, RejectsMissingProducerAfterBothAttentionLayersRun) {
  model_->addUnproducedCacheOwner();
  model_->prepareKVCache();
  auto &cache = model_->cacheManager();
  ASSERT_TRUE(cache.usesCpuStorage());
  ASSERT_EQ(cache.getNumLayers(), 3u);
  for (unsigned int layer = 0; layer < 3; ++layer) {
    auto &value = cache.getValueCache(layer);
    std::memset(value.getData<char>(), 0x5a, value.bytes());
  }
  try {
    model_->stepLogitsFromIds({1, 4}, 0);
    FAIL() << "A graph with an unmatched producer published its KV position";
  } catch (const std::logic_error &error) {
    EXPECT_STREQ(error.what(), "KV commit before all producers completed");
  }
  EXPECT_EQ(cache.getPosition(), 0u);
  // The real attention producers replaced their sentinels before commit
  // rejected the third owner. This is graph execution, not a mocked lease.
  EXPECT_NE(cache.getValueCache(0).getData<char>()[0], 0x5a);
  EXPECT_NE(cache.getValueCache(1).getData<char>()[0], 0x5a);
  EXPECT_EQ(cache.getValueCache(2).getData<char>()[0], 0x5a);
  EXPECT_NO_THROW(cache.reset());
}

TEST_F(QwenCpuKVGraphTest, RejectsForeignKeyBeforeCacheWrites) {
  rejectForeignPlane(true);
}

TEST_F(QwenCpuKVGraphTest, RejectsForeignValueBeforeCacheWrites) {
  rejectForeignPlane(false);
}

TEST_F(QwenCpuKVGraphTest, RejectsNonCommittedIntervalAndCanContinue) {
  ASSERT_NO_THROW(model_->stepLogitsFromIds({1, 4}, 0));
  ASSERT_EQ(model_->getKvLen(), 2);
  EXPECT_THROW(model_->stepLogitsFromIds({4}, 1), std::logic_error);
  EXPECT_EQ(model_->getKvLen(), 2);
  EXPECT_NO_THROW(model_->stepLogitsFromIds({4}, 2));
  EXPECT_EQ(model_->getKvLen(), 3);
}

/**
 * @brief Test that a tiny Qwen3 embedding model can save/load and encode
 */
TEST(Qwen3EmbeddingTinyModelTest,
     WeightRoundTripEncodesPromptWithQ40VocabRemainder) {
  const auto files = makeQwen3EmbeddingFiles();
  auto model = makeLoadedQwen3Embedding(files);

  std::vector<float> embedding;
  ASSERT_NO_THROW(embedding = model->encodePrompt("hello tok4"));
  ASSERT_EQ(embedding.size(), 64u);

  bool has_non_zero_value = false;
  for (float value : embedding) {
    EXPECT_TRUE(std::isfinite(value));
    has_non_zero_value = has_non_zero_value || std::abs(value) > 1e-5f;
  }
  EXPECT_TRUE(has_non_zero_value);
}

} // namespace
