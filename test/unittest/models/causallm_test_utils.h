// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   causallm_test_utils.h
 * @date   15 May 2026
 * @brief  Shared helpers for tiny CausalLM model unit tests
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __CAUSALLM_TEST_UTILS_H__
#define __CAUSALLM_TEST_UTILS_H__

#include <algorithm>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <causal_lm.h>
#include <layer.h>
#include <layer_context.h>
#include <transformer.h>

namespace causallm_test {

/**
 * @brief Files generated for one tiny CausalLM test invocation
 */
struct TinyCausalLMFiles {
  std::filesystem::path dir;            /**< Temporary test directory */
  std::filesystem::path tokenizer_path; /**< Tiny tokenizer.json path */
  std::filesystem::path weight_path;    /**< Tiny model weight path */
};

/**
 * @brief Minimal configs required to construct one CausalLM model
 */
struct TinyCausalLMConfig {
  causallm::json model;      /**< config.json equivalent */
  causallm::json generation; /**< generation_config.json equivalent */
  causallm::json nntrainer;  /**< nntrainer_config.json equivalent */
};

/**
 * @brief Data type variant used by one tiny CausalLM model case
 */
struct TinyCausalLMDataType {
  std::string name;              /**< Data type name used by gtest */
  std::string embedding_dtype;   /**< Embedding layer weight dtype */
  std::string fc_layer_dtype;    /**< Fully connected layer weight dtype */
  std::string lmhead_dtype;      /**< LM head weight dtype */
  std::string model_tensor_type; /**< Weight-activation tensor type */
};

/**
 * @brief Golden logits for one tiny CausalLM prompt
 */
struct TinyCausalLMExpectedLogits {
  std::string prompt;        /**< Prompt text */
  std::vector<float> logits; /**< Expected prefill logits */
  float logits_tolerance;    /**< Absolute logits tolerance */
};

/**
 * @brief Common runner interface exposed by model-specific tiny adapters
 */
class TinyCausalLMRunner {
public:
  /**
   * @brief Destroy the TinyCausalLMRunner object
   */
  virtual ~TinyCausalLMRunner() = default;

  /**
   * @brief Initialize the model graph and weights
   */
  virtual void initializeModel() = 0;

  /**
   * @brief Save model weights
   * @param path Target weight file path
   */
  virtual void saveWeight(const std::string &path) = 0;

  /**
   * @brief Save model weights with per-layer data type conversion
   * @param path Target weight file path
   * @param layer_dtype_map Per-layer target data types
   */
  virtual void saveWeightWithDtype(
    const std::string &path,
    const std::map<std::string, ml::train::TensorDim::DataType>
      &layer_dtype_map) = 0;

  /**
   * @brief Load model weights
   * @param path Source weight file path
   */
  virtual void loadWeight(const std::string &path) = 0;

  /**
   * @brief Run one prompt through the model
   * @param prompt Prompt text
   */
  virtual void runPrompt(const std::string &prompt) = 0;

  /**
   * @brief Run prefill and return logits before token sampling
   * @param prompt Prompt text
   * @return Prefill logits copied from the model output
   */
  virtual std::vector<float> prefillLogits(const std::string &prompt) = 0;

  /**
   * @brief Get generated output text
   * @param batch_idx Batch index
   * @return Generated output text
   */
  virtual std::string getOutputText(int batch_idx = 0) const = 0;

  /**
   * @brief Get whether the model has completed run()
   * @return true if run() completed
   */
  virtual bool hasRun() const = 0;

  /**
   * @brief Read one token from the model input/output history
   * @param idx Token history index
   * @return Token id
   */
  virtual unsigned int tokenAt(size_t idx) const = 0;

  /**
   * @brief Generate ids from logits through CausalLM decoding logic
   * @param logits Logit buffer
   * @param do_sample Whether sampling is enabled
   * @param repetition_penalty Repetition penalty
   * @param input_ids Input ids used by repetition penalty
   * @param num_input_ids Number of input ids
   * @return Generated token ids
   */
  virtual std::vector<unsigned int>
  generateFromLogits(float *logits, bool do_sample, float repetition_penalty,
                     unsigned int *input_ids, unsigned int num_input_ids) = 0;

  /**
   * @brief Run prefill with raw token IDs and return logits before sampling
   * @param ids Token id sequence
   * @return Prefill logits for the last token position
   */
  virtual std::vector<float>
  prefillLogitsFromIds(const std::vector<unsigned int> &ids) {
    throw std::logic_error(
      "prefillLogitsFromIds not implemented for this model");
  }

  /**
   * @brief Run prefill then n greedy decoding steps using raw token IDs
   * @param ids Prompt token id sequence
   * @param n Number of greedy tokens to generate
   * @return Generated token ids (length n)
   */
  virtual std::vector<unsigned int>
  greedyGenerateFromIds(const std::vector<unsigned int> &ids, size_t n) {
    throw std::logic_error(
      "greedyGenerateFromIds not implemented for this model");
  }

  /**
   * @brief Encode one prompt and return its output embedding vector
   *
   * Used by embedding/encoder models (SentenceTransformer / BERT / DeBERTa).
   * @param prompt Prompt text
   * @param out_len Number of floats to copy from the model output (the fixture
   *                reference embedding length; pooled models = hidden_size,
   *                raw encoders = seq_len * hidden_size)
   * @return Output embedding vector of length out_len
   */
  virtual std::vector<float> embedPrompt(const std::string &prompt,
                                         size_t out_len) {
    throw std::logic_error("embedPrompt not implemented for this model");
  }
};

/**
 * @brief Generic tiny CausalLM test adapter shared by all model families
 *
 * Wraps a concrete CausalLM model (ModelBase) and implements the
 * TinyCausalLMRunner inference interface once, so per-model test files only
 * differ in how their weights are populated (deterministic setup vs loading a
 * committed reference fixture).
 *
 * @tparam ModelBase Concrete CausalLM model (e.g. causallm::Qwen3CausalLM)
 */
template <typename ModelBase>
class CausalLMTestAdapter : public ModelBase, public TinyCausalLMRunner {
public:
  /**
   * @brief Construct the adapter from the three standard CausalLM configs
   *
   * Transformer is a virtual base of every CausalLM model, so it must be
   * initialized here by the most-derived adapter. Models that need to
   * pre-process configs (e.g. Gemma3's sanitizeConfig) should derive a thin
   * subclass that initializes Transformer with the processed configs; this
   * mem-initializer is then ignored for that subclass per virtual-base rules.
   */
  CausalLMTestAdapter(causallm::json &cfg, causallm::json &generation_cfg,
                      causallm::json &nntr_cfg) :
    causallm::Transformer(cfg, generation_cfg, nntr_cfg,
                          causallm::ModelType::CAUSALLM),
    ModelBase(cfg, generation_cfg, nntr_cfg) {}

  /**
   * @brief Initialize the model graph and weights
   */
  void initializeModel() override { this->initialize(); }

  /**
   * @brief Save model weights
   */
  void saveWeight(const std::string &path) override { this->save_weight(path); }

  /**
   * @brief Save model weights with per-layer data type conversion
   */
  void saveWeightWithDtype(
    const std::string &path,
    const std::map<std::string, ml::train::TensorDim::DataType>
      &layer_dtype_map) override {
    this->save_weight(path, ml::train::TensorDim::DataType::NONE,
                      layer_dtype_map);
  }

  /**
   * @brief Load model weights
   */
  void loadWeight(const std::string &path) override { this->load_weight(path); }

  /**
   * @brief Run one prompt through the model
   */
  void runPrompt(const std::string &prompt) override {
    this->run(prompt, false, "", "", false);
  }

  /**
   * @brief Run prefill and return logits before token sampling
   */
  std::vector<float> prefillLogits(const std::string &prompt) override {
    auto encoded = this->tokenizer->Encode(prompt);
    if (encoded.empty())
      throw std::invalid_argument("tiny CausalLM prompt encoded to no tokens");

    const unsigned int num_allow_str =
      this->MAX_SEQ_LEN - this->NUM_TO_GENERATE;
    const unsigned int init_len = static_cast<unsigned int>(
      std::min<size_t>(encoded.size(), num_allow_str));
    std::vector<unsigned int> ids(encoded.begin(), encoded.begin() + init_len);
    return prefillLogitsFromIds(ids);
  }

  /**
   * @brief Run prefill with raw token IDs and return last-token logits
   */
  std::vector<float>
  prefillLogitsFromIds(const std::vector<unsigned int> &ids) override {
    this->allocateAndBindKVCache();

    const unsigned int init_len = static_cast<unsigned int>(ids.size());
    std::vector<float> input_sample(
      static_cast<size_t>(this->BATCH_SIZE) * this->MAX_SEQ_LEN, 0.0f);

    for (unsigned int b = 0; b < this->BATCH_SIZE; ++b) {
      for (unsigned int i = 0; i < init_len; ++i) {
        input_sample[static_cast<size_t>(b) * this->MAX_SEQ_LEN + i] =
          static_cast<float>(ids[i]);
        this->ids_history[static_cast<size_t>(b) * this->MAX_SEQ_LEN + i] =
          ids[i];
      }
    }

    auto [input, cache_inputs] = buildCacheInput(input_sample);
    std::vector<float *> label;
    this->setKVCachePosition(0);
    auto output = this->model->incremental_inference(
      this->BATCH_SIZE, input, label, init_len, 0, init_len, false);
    std::vector<float> logits(output[0], output[0] + this->NUM_VOCAB);
    for (auto &out : output)
      delete[] out;

    return logits;
  }

  /**
   * @brief Run prefill then n greedy decoding steps with raw token IDs
   */
  std::vector<unsigned int>
  greedyGenerateFromIds(const std::vector<unsigned int> &ids,
                        size_t n) override {
    this->allocateAndBindKVCache();

    const unsigned int init_len = static_cast<unsigned int>(ids.size());
    std::vector<float> input_sample(
      static_cast<size_t>(this->BATCH_SIZE) * this->MAX_SEQ_LEN, 0.0f);

    for (unsigned int b = 0; b < this->BATCH_SIZE; ++b)
      for (unsigned int i = 0; i < init_len; ++i)
        input_sample[static_cast<size_t>(b) * this->MAX_SEQ_LEN + i] =
          static_cast<float>(ids[i]);

    auto [input, cache_inputs] = buildCacheInput(input_sample);
    std::vector<float *> label;

    this->setKVCachePosition(0);
    auto output = this->model->incremental_inference(
      this->BATCH_SIZE, input, label, init_len, 0, init_len, false);

    std::vector<unsigned int> generated;
    generated.reserve(n);

    for (size_t step = 0; step < n; ++step) {
      unsigned int next_tok = static_cast<unsigned int>(std::distance(
        output[0], std::max_element(output[0], output[0] + this->NUM_VOCAB)));
      generated.push_back(next_tok);
      for (auto &out : output)
        delete[] out;
      output.clear();

      if (step + 1 >= n)
        break;

      std::fill(input_sample.begin(), input_sample.end(), 0.0f);
      input_sample[0] = static_cast<float>(next_tok);

      unsigned int from = init_len + static_cast<unsigned int>(step);
      unsigned int to = from + 1;
      this->setKVCachePosition(from);
      output = this->model->incremental_inference(this->BATCH_SIZE, input,
                                                  label, 1, from, to, false);
    }

    return generated;
  }

  /**
   * @brief Get generated output text
   */
  std::string getOutputText(int batch_idx = 0) const override {
    return this->getOutput(batch_idx);
  }

  /**
   * @brief Get whether the model has completed run()
   */
  bool hasRun() const override { return causallm::CausalLM::hasRun(); }

  /**
   * @brief Read one token from the model input/output history
   */
  unsigned int tokenAt(size_t idx) const override {
    return this->ids_history[idx];
  }

  /**
   * @brief Generate ids from logits through CausalLM decoding logic
   */
  std::vector<unsigned int>
  generateFromLogits(float *logits, bool do_sample, float repetition_penalty,
                     unsigned int *input_ids,
                     unsigned int num_input_ids) override {
    return this->generate(logits, do_sample, repetition_penalty, input_ids,
                          num_input_ids);
  }

  /**
   * @brief Apply a callback to every layer (used to populate weights in tests)
   * @param fn Callback receiving (layer, run-context, user-data)
   */
  void forEachLayer(std::function<void(ml::train::Layer &,
                                       nntrainer::RunLayerContext &, void *)>
                      fn) {
    this->model->forEachLayer(fn, nullptr);
  }

private:
  /**
   * @brief Build the (input, cache_inputs) pair for incremental_inference
   *
   * The returned cache_inputs keeps the string keys alive; input holds raw
   * pointers into input_sample and cache buffers.
   */
  std::pair<std::vector<float *>, std::vector<std::pair<std::string, float *>>>
  buildCacheInput(std::vector<float> &input_sample) {
    std::vector<std::pair<std::string, float *>> cache_inputs;
    cache_inputs.reserve(static_cast<size_t>(this->NUM_LAYERS) * 2);
    for (int i = 0; i < this->NUM_LAYERS; ++i) {
      cache_inputs.emplace_back(
        "cache_k_l" + std::to_string(i),
        reinterpret_cast<float *>(this->kv_cache.getKeyCache(i).getData()));
      cache_inputs.emplace_back(
        "cache_v_l" + std::to_string(i),
        reinterpret_cast<float *>(this->kv_cache.getValueCache(i).getData()));
    }
    std::sort(cache_inputs.begin(), cache_inputs.end(),
              [](const auto &l, const auto &r) { return l.first < r.first; });

    std::vector<float *> input;
    input.reserve(1 + cache_inputs.size());
    input.push_back(input_sample.data());
    for (const auto &ci : cache_inputs)
      input.push_back(ci.second);

    return {std::move(input), std::move(cache_inputs)};
  }
};

/**
 * @brief Generic tiny embedding/encoder model test adapter
 *
 * Wraps a concrete embedding model (ModelBase) — any class exposing an
 * encode() that returns a std::vector<float *> (SentenceTransformer-derived
 * models, BertTransformer, DebertaV2). It initializes the virtual Transformer
 * base with ModelType::EMBEDDING and drives the model through its own encode(),
 * so it absorbs each family's input format (1-input, 3-input, KV-cache) for
 * free.
 *
 * Models that sanitize their configs in the constructor (EmbeddingGemma,
 * MultilingualTinyBert, DebertaV2) must derive a thin subclass that initializes
 * Transformer with the processed configs (see TinyGemma3CausalLM for the
 * CausalLM analogue); this mem-initializer is then ignored for that subclass
 * per virtual-base rules.
 *
 * @tparam ModelBase Concrete embedding model (e.g. causallm::Qwen3Embedding)
 */
template <typename ModelBase>
class EmbeddingTestAdapter : public ModelBase, public TinyCausalLMRunner {
public:
  /**
   * @brief Construct the adapter from the three standard configs
   */
  EmbeddingTestAdapter(causallm::json &cfg, causallm::json &generation_cfg,
                       causallm::json &nntr_cfg) :
    causallm::Transformer(cfg, generation_cfg, nntr_cfg,
                          causallm::ModelType::EMBEDDING),
    ModelBase(cfg, generation_cfg, nntr_cfg) {}

  /**
   * @brief Initialize the model graph and weights
   */
  void initializeModel() override { this->initialize(); }

  /**
   * @brief Save model weights
   */
  void saveWeight(const std::string &path) override { this->save_weight(path); }

  /**
   * @brief Save model weights with per-layer data type conversion
   */
  void saveWeightWithDtype(
    const std::string &path,
    const std::map<std::string, ml::train::TensorDim::DataType>
      &layer_dtype_map) override {
    this->save_weight(path, ml::train::TensorDim::DataType::NONE,
                      layer_dtype_map);
  }

  /**
   * @brief Load model weights
   */
  void loadWeight(const std::string &path) override { this->load_weight(path); }

  /**
   * @brief Encode one prompt and copy out_len floats from the output embedding
   */
  std::vector<float> embedPrompt(const std::string &prompt,
                                 size_t out_len) override {
    auto output = this->encode(prompt);
    if (output.empty() || output[0] == nullptr)
      throw std::runtime_error("embedding model produced no output");
    std::vector<float> embedding(output[0], output[0] + out_len);
    for (auto &out : output)
      delete[] out;
    return embedding;
  }

  // --- CausalLM-specific interface methods: not applicable to embeddings ---

  void runPrompt(const std::string &) override {
    throw std::logic_error("runPrompt not supported for embedding models");
  }
  std::vector<float> prefillLogits(const std::string &) override {
    throw std::logic_error("prefillLogits not supported for embedding models");
  }
  std::string getOutputText(int = 0) const override {
    throw std::logic_error("getOutputText not supported for embedding models");
  }
  bool hasRun() const override { return false; }
  unsigned int tokenAt(size_t) const override {
    throw std::logic_error("tokenAt not supported for embedding models");
  }
  std::vector<unsigned int> generateFromLogits(float *, bool, float,
                                               unsigned int *,
                                               unsigned int) override {
    throw std::logic_error(
      "generateFromLogits not supported for embedding models");
  }
};

/**
 * @brief One tiny CausalLM model case reusable by common tests
 */
struct TinyCausalLMCase {
  std::string name;                           /**< Case name used by gtest */
  TinyCausalLMDataType data_type;             /**< Data type variant */
  TinyCausalLMExpectedLogits expected_logits; /**< Expected prefill logits */
  std::function<causallm::json()> make_model_config;
  std::function<std::map<std::string, ml::train::TensorDim::DataType>(
    const TinyCausalLMDataType &)>
    make_layer_dtype_map;
  std::function<std::unique_ptr<TinyCausalLMRunner>(
    causallm::json &, causallm::json &, causallm::json &)>
    create_model;
  std::function<void(TinyCausalLMRunner &)>
    setup_weights; /**< Populate deterministic weights before saving */
  /**
   * @brief Optional nntrainer config factory override.
   *
   * When non-null, called instead of makeTinyNntrainerConfig() to build the
   * nntrainer config (e.g. flash-attention cases need init_seq_len=32).
   * When null, makeTinyNntrainerConfig() is used.
   */
  std::function<causallm::json(const std::filesystem::path &,
                               const TinyCausalLMDataType &)>
    make_nntrainer_config = nullptr;
};

/**
 * @brief Golden reference fixture loaded from a committed directory
 */
struct ReferenceFixture {
  std::filesystem::path dir;           /**< Fixture directory */
  std::vector<unsigned int> input_ids; /**< Fixed input token IDs */
  std::vector<float> reference_logits; /**< HF prefill last-token logits */
  std::vector<unsigned int> reference_tokens; /**< HF greedy token sequence */
  float logits_atol_fp32;  /**< FP32 logit absolute tolerance */
  float logits_atol_q40;   /**< Q4_0 logit absolute tolerance */
  size_t prefix_match_min; /**< Min greedy prefix match length */

  // Embedding/encoder model fields (unused by CausalLM fixtures)
  std::vector<float> reference_embedding; /**< HF output embedding vector */
  std::string prompt;                     /**< Prompt text fed to encode() */
  float embedding_atol;     /**< FP32 embedding absolute tolerance */
  float cosine_min;         /**< FP32 min cosine similarity */
  float embedding_atol_q40; /**< Q4_0 embedding absolute tolerance */
  float cosine_min_q40;     /**< Q4_0 min cosine similarity */
};

/**
 * @brief Locate the fixture directory for a given fixture name
 *
 * Checks NNTRAINER_CAUSALLM_FIXTURE_DIR env first, then uses
 * a path relative to this header's source location.
 *
 * @param fixture_name Sub-directory name (e.g. "qwen3_tiny")
 * @return Absolute path (may not exist if fixtures were not generated)
 */
std::filesystem::path findFixtureDir(const std::string &fixture_name);

/**
 * @brief Load a reference fixture from disk
 * @param dir Fixture directory
 * @return Loaded fixture; empty vectors if files are absent
 */
ReferenceFixture loadReferenceFixture(const std::filesystem::path &dir);

/**
 * @brief Assert that two logit vectors agree within an absolute tolerance
 * @param got Logits produced by nntrainer
 * @param ref Reference logits from HF
 * @param atol Absolute tolerance
 */
void expectLogitsNear(const std::vector<float> &got,
                      const std::vector<float> &ref, float atol);

/**
 * @brief Assert that two token sequences have at least min_match tokens
 *        in common at the start
 * @param got Tokens produced by nntrainer
 * @param ref Reference tokens from HF
 * @param min_match Minimum matching prefix length
 */
void expectTokenPrefixMatch(const std::vector<unsigned int> &got,
                            const std::vector<unsigned int> &ref,
                            size_t min_match);

/**
 * @brief Assert two embedding vectors agree element-wise and by cosine
 * @param got Embedding produced by nntrainer
 * @param ref Reference embedding from HF
 * @param atol Per-element absolute tolerance
 * @param cosine_min Minimum acceptable cosine similarity
 */
void expectEmbeddingNear(const std::vector<float> &got,
                         const std::vector<float> &ref, float atol,
                         float cosine_min);

/**
 * @brief One model entry for the generic differential reference tests
 */
struct DifferentialModel {
  std::string fixture_name; /**< Fixture sub-directory (e.g. "qwen3_tiny") */
  std::function<std::unique_ptr<TinyCausalLMRunner>(
    causallm::json &, causallm::json &, causallm::json &)>
    make_model; /**< Adapter factory from loaded fixture configs */
};

/**
 * @brief Run the FP32 differential checks for a model against its fixture
 *
 * Skips (via GTEST_SKIP) when the fixture or FP32 weights are absent.
 * Verifies prefill logits and greedy tokens match the HF reference.
 *
 * @param model Differential model descriptor
 */
void runFp32DifferentialChecks(const DifferentialModel &model);

/**
 * @brief Run the Q4_0 differential checks for a model against its fixture
 *
 * Skips when the fixture or the nntr_quantize binary (NNTR_QUANTIZE_BIN) are
 * absent. Quantizes the FP32 fixture to Q4_0 and verifies the resulting logits
 * stay within tolerance of both the HF reference and the nntrainer FP32 logits.
 *
 * @param model Differential model descriptor
 */
void runQ40DifferentialChecks(const DifferentialModel &model);

/**
 * @brief Run the FP32 differential checks for an embedding model
 *
 * Skips (via GTEST_SKIP) when the fixture or FP32 weights are absent.
 * Loads the model in embedding mode, encodes the fixture prompt, and verifies
 * the output embedding matches the HF reference (per-element atol + cosine).
 *
 * @param model Differential model descriptor (make_model returns an
 *              EmbeddingTestAdapter-derived runner)
 */
void runFp32EmbeddingDifferentialChecks(const DifferentialModel &model);

/**
 * @brief Run the Q4_0 differential checks for an embedding model
 *
 * Skips when the fixture, NNTR_QUANTIZE_BIN, or Q4_0 support for the
 * architecture are absent. Quantizes the FP32 fixture, encodes the prompt
 * with the Q4_0 model, and verifies the embedding stays within Q4_0 tolerance
 * of both the HF FP32 reference and the nntrainer FP32 embedding.
 *
 * @param model Differential model descriptor
 */
void runQ40EmbeddingDifferentialChecks(const DifferentialModel &model);

/**
 * @brief Make FP32 data type variant
 * @return Tiny FP32 data type descriptor
 */
TinyCausalLMDataType makeTinyFp32DataType();

/**
 * @brief Make Q4_0 weights with FP32 activations data type variant
 * @return Tiny Q4_0-FP32 data type descriptor
 */
TinyCausalLMDataType makeTinyQ40Fp32DataType();

/**
 * @brief Make Q4_0 weights with FP16 activations data type variant
 * @return Tiny Q4_0-FP16 data type descriptor
 */
TinyCausalLMDataType makeTinyQ40Fp16DataType();

/**
 * @brief Convert a test dtype string to an nntrainer tensor data type
 * @param dtype Test dtype string
 * @return nntrainer tensor data type
 */
ml::train::TensorDim::DataType toTensorDataType(const std::string &dtype);

/**
 * @brief Make files for one tiny CausalLM test invocation
 * @param suite_name GTest suite name
 * @param test_name GTest test name
 * @param case_name Tiny CausalLM model case name
 * @return Generated file paths
 */
TinyCausalLMFiles makeTinyCausalLMFiles(const std::string &suite_name,
                                        const std::string &test_name,
                                        const std::string &case_name);

/**
 * @brief Make minimal generation config shared by tiny CausalLM tests
 * @return generation_config.json equivalent
 */
causallm::json makeTinyGenerationConfig();

/**
 * @brief Make minimal nntrainer config shared by tiny CausalLM tests
 * @param tokenizer_path Tiny tokenizer path
 * @param data_type Tiny CausalLM data type variant
 * @return nntrainer_config.json equivalent
 */
causallm::json
makeTinyNntrainerConfig(const std::filesystem::path &tokenizer_path,
                        const TinyCausalLMDataType &data_type);

/**
 * @brief Make complete tiny configs for one model case
 * @param test_case Model case descriptor
 * @param tokenizer_path Tiny tokenizer path
 * @return Complete tiny CausalLM configs
 */
TinyCausalLMConfig
makeTinyCausalLMConfig(const TinyCausalLMCase &test_case,
                       const std::filesystem::path &tokenizer_path);

/**
 * @brief Verify greedy decoding chooses the maximum logit token
 * @param model Tiny model runner
 */
void expectGreedyGenerationSelectsArgmax(TinyCausalLMRunner &model);

/**
 * @brief Verify save/load round-trip preserves tiny model logits
 * @param test_case Model case descriptor
 * @param files Generated test file paths
 */
void expectWeightRoundTripProducesSameLogits(const TinyCausalLMCase &test_case,
                                             const TinyCausalLMFiles &files);

/**
 * @brief Verify a tiny model emits the expected logits for a prompt
 * @param test_case Model case descriptor
 * @param files Generated test file paths
 */
void expectPromptProducesExpectedLogits(const TinyCausalLMCase &test_case,
                                        const TinyCausalLMFiles &files);

} // namespace causallm_test

#endif // __CAUSALLM_TEST_UTILS_H__
