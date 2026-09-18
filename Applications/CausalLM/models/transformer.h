// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   transformer.h
 * @brief  Base Transformer class shared by CausalLM and encoder models.
 * @date   31 Dec 2025
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   This transformer.h constructs a class for Transformer model which can
 * be a parent of CausalLM and Encoder models with transformer structure.
 * @note   This transformer assumes the following structure :
 *
 *           [Input]
 *              |
 *         [Embedding]
 *              |
 *        [Decoder Block] (repeated N times)
 *              |
 *          [RMSNorm]
 *
 */
#ifndef __TRANSFORMER_H__
#define __TRANSFORMER_H__

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#define WSTR std::string
#define WCHAR_P std::string &
#else
#define WIN_EXPORT
#define WSTR std::string
#define WCHAR_P std::string &
#endif

#include <layer.h>
#include <map>
#include <model.h>
#include <random>
#include <stdexcept>
#include <tensor_api.h>
#include <utility>
#include <vector>

#include <limits.h>

#include "json.hpp"
#include "performance_metrics.h"
#include <fstream>
#include <tokenizers_c.h>
#include <tokenizers_cpp.h>

namespace causallm {

/*** ALIAS ****/
using LayerHandle = ml::train::LayerHandle;
using Tensor = ml::train::Tensor;
using ModelHandle = std::unique_ptr<ml::train::Model>;

using json = nlohmann::json;

/**
 * @brief Model Type Enum
 */
enum class ModelType { MODEL, CAUSALLM, EMBEDDING, UNKNOWN };

/**
 * @brief {data, size} pointer pair produced/consumed by multimodal vision
 *        models. The buffer is heap-allocated by the producer (run_image) and
 *        ownership transfers to the caller.
 */
using multimodal_pointer = std::pair<void *, size_t>;

/**
 * @brief Non-owning logits processor hook for token generation
 */
class LogitsProcessor {
public:
  /**
   * @brief Destroy the LogitsProcessor object
   */
  virtual ~LogitsProcessor() = default;

  /**
   * @brief Mutate one batch row of logits before token selection
   * @param logits FP32 logits for a single batch row
   * @param vocab_size Number of logits in the row
   * @param batch_index Batch row index
   */
  virtual void process(float *logits, unsigned int vocab_size,
                       unsigned int batch_index) = 0;

  /**
   * @brief Receive the selected token after token selection
   * @param token_id Selected token id
   * @param batch_index Batch row index
   */
  virtual void acceptToken(unsigned int token_id, unsigned int batch_index) = 0;

  /**
   * @brief Reset processor state when requested by the caller
   */
  virtual void reset() {}
};

/**
 * @brief Transformer Class
 */
WIN_EXPORT class Transformer {

public:
  /**
   * @brief Construct a new Transformer object
   * @param cfg Configuration for the model (config.json)
   * @param generation_cfg Configuration for the generation (generation.json)
   * @param nntr_cfg Configuration for nntrainer (nntrainer_config.json)
   * @param model_type Type of the model (default: ModelType::MODEL)
   */
  Transformer(json &cfg, json &generation_cfg, json &nntr_cfg,
              ModelType model_type = ModelType::MODEL);

  /**
   * @brief Empty constructor for Transformer.
   * @brief Child Class Needs to implement all features of the original
   * Transformer constructor
   */
  Transformer() {}

  /**
   * @brief Destroy the Transformer object
   */
  virtual ~Transformer() {}

  /**
   * @brief Initialize and Construct the Transformer model
   */
  virtual void initialize();

  /**
   * @brief Load the model weights from a file
   */
  virtual void load_weight(const std::string &weight_path);

  /**
   * @brief Repack all QS4CX weights after loading
   * @note Must be called after load_weight() for QS4CX quantized tensors
   * @note Prepares weights for efficient computation by eagerly packing them
   */
  virtual void repack_weight();

  /**
   * @brief Save the weight to a file
   */
  virtual void save_weight(const std::string &weight_path);
  /**
   * @brief Save the weight to a file with type conversion
   * @param weight_path Path to save the weight file
   * @param dtype Global target data type for all layers (NONE = keep original)
   * @param layer_dtype_map Per-layer data type overrides (layer_name -> dtype)
   * @param target_isa Target ISA for quantization (default: DEFAULT)
   */
  virtual void
  save_weight(const std::string &weight_path,
              ml::train::TensorDim::DataType dtype,
              const std::map<std::string, ml::train::TensorDim::DataType>
                &layer_dtype_map = {},
              ml::train::ISA target_isa = ml::train::ISA::DEFAULT);

  /**
   * @brief run the Transformer model
   */
  virtual void run(const WSTR prompt, bool do_sample = false,
                   const WSTR system_prompt = WSTR(),
                   const WSTR tail_prompt = WSTR(), bool log_output = true);

  // ── Multimodal composition interface (model-agnostic) ──────────────────
  // Lets a generic composer drive any [vision producer, LLM consumer] pair
  // through base pointers, without knowing the concrete model type.
  // Default implementations mean "this role is not supported by this model".

  /** Embedding-CONSUMER (LLM): bytes of one token embedding (0 ⇒ no table). */
  virtual size_t embeddingBytesPerToken() const { return 0; }

  /** Embedding-CONSUMER (LLM): embedding of @p token_id, or nullptr. */
  virtual const void *lookupEmbedding(int token_id) const {
    (void)token_id;
    return nullptr;
  }

  /** Embedding-CONSUMER (LLM): (scale, offset) of the embedding quant space. */
  virtual std::pair<float, int> get_embedding_info() { return {1.0f, 0}; }

  /** Embedding-CONSUMER (LLM): run generation from precomputed embeddings. */
  virtual void run_with_embeddings(const void *prefill_embeds, size_t n_tokens,
                                   std::vector<int> seed_tokens, bool do_sample,
                                   bool log_output) {
    (void)prefill_embeds;
    (void)n_tokens;
    (void)seed_tokens;
    (void)do_sample;
    (void)log_output;
    throw std::runtime_error("run_with_embeddings not supported by this model");
  }

  /** Embedding-PRODUCER (vision): set the (scale, offset) it should emit in. */
  virtual void set_quant_param(float scale, int offset) {
    (void)scale;
    (void)offset;
  }

  /** Embedding-PRODUCER (vision): encode an image into LLM-space embeddings.
   *  Returns a heap buffer (caller frees) of size {bytes}; the default
   *  {nullptr,0} means "this model is not a vision producer". */
  virtual multimodal_pointer
  run_image(const WSTR prompt, multimodal_pointer image, int image_height,
            int image_width, bool do_sample = false,
            const WSTR system_prompt = WSTR(), const WSTR tail_prompt = WSTR(),
            bool log_output = true) {
    (void)prompt;
    (void)image;
    (void)image_height;
    (void)image_width;
    (void)do_sample;
    (void)system_prompt;
    (void)tail_prompt;
    (void)log_output;
    return {nullptr, 0};
  }

  /** Current KV-cache length (0 if the model has no persistent KV cache). */
  virtual int getKvLen() const { return 0; }

  /**
   * @brief Get TransformerPerformanceMetrics
   */
  TransformerPerformanceMetrics getPerformanceMetrics() const {
    return performance_metrics;
  }

  /**
   * @brief get the status of run
   */
  bool hasRun() const { return has_run_; }

  /**
   * @brief Get configured vocabulary size
   * @return Vocabulary size
   */
  unsigned int getVocabSize() const { return NUM_VOCAB; }

  /**
   * @brief Get tokenizer owned by this model, or nullptr if no tokenizer exists
   */
  tokenizers::Tokenizer *getTokenizer() { return tokenizer.get(); }

  /**
   * @brief Attach a non-owning logits processor
   * @param processor Processor pointer, or nullptr to detach
   */
  virtual void setLogitsProcessor(LogitsProcessor *) {}

  /**
   * @brief Reset attached logits processor state
   */
  virtual void resetLogitsProcessor() {}

protected:
  /**
   * @brief Setup the parameters for the Transformer model
   */
  virtual void setupParameters(json &cfg, json &generation_cfg, json &nntr_cfg);

  /**
   * @brief Construct Model
   * @return {input_tensor, output_tensor} pair representing the symbolic
   *         tensor graph. Derived classes can extend by taking the output
   *         and feeding additional layers before returning.
   */
  virtual std::pair<Tensor, Tensor> constructModel();

  /**
   * @brief Build common CausalLM embedding layer properties
   * @param name Layer name
   * @param in_dim Vocabulary/input dimension
   * @param out_dim Embedding output dimension
   * @param weight_dtype Layer weight dtype
   * @param scale Embedding scale
   * @param quantized_lut_path Optional sidecar LUT path
   * @return Layer property strings
   */
  std::vector<std::string>
  buildEmbeddingLayerProperties(const std::string &name, unsigned int in_dim,
                                unsigned int out_dim,
                                const std::string &weight_dtype, float scale,
                                const std::string &quantized_lut_path) const;

  /**
   * @brief Create one Transformer decoder block (norm + attention + residual +
   *        norm + ffn + residual)
   * @param layer_id index of the decoder block
   * @param input    symbolic input tensor for this block
   * @return symbolic output tensor of the block
   */
  virtual Tensor createTransformerDecoderBlock(const int layer_id,
                                               Tensor input);

  /**
   * @brief Create the attention sub-graph (Q/K/V projections + mha_core +
   *        output projection)
   * @return symbolic output tensor of the attention sub-graph
   */
  virtual Tensor createAttention(const int layer_id, int seq_len, int n_heads,
                                 int head_dim, Tensor query, Tensor key,
                                 Tensor value);

  /**
   * @brief Create the feed-forward sub-graph
   * @return symbolic output tensor of the FFN sub-graph
   */
  virtual Tensor createMlp(const int layer_id, int dim, int hidden_dim,
                           Tensor input);

  /**
   * @brief Create the per-layer external KV-cache placeholder Tensors that
   *        feed mha_core's input slots 3 and 4. The actual storage is owned
   *        by the host (e.g. KVCacheManager) and is bound at runtime via
   *        Model::setExternalTensors using the names
   *          "cache_k_l<layer_id>" and "cache_v_l<layer_id>".
   * @param layer_id  attention layer index
   * @param n_heads   total query heads (used together with GQA_SIZE to derive
   *                  the KV head count)
   * @return {cache_k, cache_v} symbolic placeholder tensors
   */
  std::pair<Tensor, Tensor> createKVCachePlaceholders(const int layer_id,
                                                      int n_heads);

  /** @brief Host buffer offered for one named model input. */
  struct NamedModelInput {
    std::string name; /**< compiled input layer name */
    float *data;      /**< host buffer, reinterpreted to the slot dtype */
    size_t bytes;     /**< usable size of @a data in bytes */
  };

  /**
   * @brief Arrange named external inputs in compiled positional order.
   *
   * Every compiled input must be matched by name; unmatched entries are
   * collected in @a dropped (expected for hybrid architectures like LFM2
   * whose conv blocks have no cache inputs).
   */
  std::vector<float *> buildOrderedInferenceInputs(
    const std::vector<NamedModelInput> &named_inputs,
    std::vector<std::string> *dropped = nullptr) const;

  /** @brief Name of the primary (token/embedding) input layer. */
  static constexpr const char *PRIMARY_INPUT_NAME = "input0";

  /**
   * @brief register CustomLayers
   */
  virtual void registerCustomLayers();

  /**
   * @brief Get model format from weight file extension.
   * @param weight_path Path to the weight file.
   * @return Model format for the given file extension.
   */
  virtual ml::train::ModelFormat
  formatFromExtension(const std::string &weight_path);

  /**
   * @brief register Outputs
   */
  bool is_initialized = false; /**< Flag to check if the model is initialized */
  ModelHandle model;

  /** tokenizer */
  std::unique_ptr<tokenizers::Tokenizer> tokenizer;

  unsigned int NUM_VOCAB;
  int DIM;
  int HEAD_DIM;
  int INTERMEDIATE_SIZE;
  int NUM_LAYERS;
  bool USE_VOCAB_SELECTION;
  bool TIE_WORD_EMBEDDINGS;
  unsigned int MAX_SEQ_LEN;
  int NUM_HEADS;
  int NUM_KEY_VALUE_HEADS;
  int NUM_TO_GENERATE;
  std::string MODEL_TENSOR_TYPE;
  std::string EMBEDDING_DTYPE; /** embedding dtype */
  std::string FC_LAYER_DTYPE;  /** custom_fc_lora */
  std::string EMBEDDING_FILE_NAME;
  std::string PLE_FILE_NAME;

  unsigned int SLIDING_WINDOW = UINT_MAX;
  unsigned int SLIDING_WINDOW_PATTERN = 5;
  unsigned int ROPE_THETA = 10000; /**< RoPE theta value */
  float NORM_EPS = 1e-5;           /**< RMSNorm epsilon value */
  float EMBEDDING_SCALE = 1.0f;
  int GQA_SIZE;

  unsigned int BATCH_SIZE;              /**< Batch size for the model */
  unsigned int INIT_SEQ_LEN;            /**< Initial sequence length */
  unsigned int MAX_POSITION_EMBEDDINGS; /**< max_position embeddings */
  bool MEMORY_SWAP;                     /**< memory swap option */
  unsigned int FSU_LOOKAHEAD;
  float ATTN_LOGIT_SOFTCAPPING = 0.0f; /**< attention logit softcapping */
  bool IS_CAUSAL = true;

  // Performance metrics
  TransformerPerformanceMetrics performance_metrics;

  bool has_run_ = false;
};
/**
 * Loads JSON data from a file with detailed error handling
 * @param file_path Path to JSON file
 * @return JSON object
 * @throws std::runtime_error on file open or parse failure
 */
inline json LoadJsonFile(const std::string &file_path) {
  std::ifstream file(file_path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + file_path +
                             " | Reason: " + std::strerror(errno));
  }

  try {
    json data;
    file >> data;
    return data;
  } catch (const json::parse_error &e) {
    throw std::runtime_error("JSON parse error in " + file_path +
                             " | Details: " + e.what());
  }
}
} // namespace causallm

#endif
