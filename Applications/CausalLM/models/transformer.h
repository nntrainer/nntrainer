// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   transformer.h
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

// Forward declaration for BaseStreamer (used by setStreamer)
extern "C" {
struct BaseStreamer;
}

namespace causallm {

// Forward declaration for XGrammar (grammar-constrained generation)
class XGrammar;

/*** ALIAS ****/
using LayerHandle = ml::train::LayerHandle;
using Tensor = ml::train::Tensor;
using ModelHandle = std::unique_ptr<ml::train::Model>;

using json = nlohmann::json;

// Memory pointer and its size
typedef std::pair<void *, size_t> multimodal_pointer;

/**
 * @brief Model Type Enum
 */
enum class ModelType { MODEL, CAUSALLM, EMBEDDING, UNKNOWN };

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
   * @brief Initialize and Construct the Transformer model with native library
   * directory
   * @param native_lib_dir Native library directory path (from Android
   * ApplicationInfo.nativeLibraryDir)
   */
  virtual void initialize(const std::string &native_lib_dir);

  /**
   * @brief Load the model weights from a file
   */
  virtual void load_weight(const std::string &weight_path);

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

  tokenizers::Tokenizer *getTokenizer() { return tokenizer.get(); }

  /**
   * @brief Get vocabulary size
   */
  unsigned int getVocabSize() const { return NUM_VOCAB; }

  /**
   * @brief run the Transformer model
   */
  virtual void run(const WSTR prompt, bool do_sample = false,
                   const WSTR system_prompt = WSTR(),
                   const WSTR tail_prompt = WSTR(), bool log_output = true);

  /**
   * @brief run the Transformer model, but with multimodal input and arbitrary
   * output
   */
  virtual multimodal_pointer
  run_image(const WSTR prompt, multimodal_pointer image, int image_height,
            int image_width, bool do_sample = false,
            const WSTR system_prompt = "", const WSTR tail_prompt = "",
            bool log_output = true);

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

  /** Current KV-cache length (0 if the model has no persistent KV cache). */
  virtual int getKvLen() const { return 0; }

  /** Producer: FP32 element count run_image expects (0 ⇒ legacy 512² calc). */
  virtual size_t expectedPixelElems() const { return 0; }

  /** Consumer: token id used as the image placeholder (<0 ⇒ use "<|image|>"). */
  virtual int imagePlaceholderTokenId() const { return -1; }

  /**
   * @brief Run a full video→text pipeline from pre-loaded frames.
   *
   * Video-language models (e.g. Lfm2VLVJepa21BModel) own their vision encoder,
   * projector and LM internally and expose this single entry point. @p frames
   * holds @c num_frames tensors, each laid out [C, H, W]. Default
   * implementation means "this model is not a video-language model".
   */
  virtual void run_video(const std::vector<std::vector<float>> &frames,
                         const std::string &prompt, bool do_sample,
                         bool log_output) {
    (void)frames;
    (void)prompt;
    (void)do_sample;
    (void)log_output;
    throw std::runtime_error("run_video not supported by this model");
  }

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
   * @brief Attach (or detach) a BaseStreamer to intercept per-token
   *        output during the next call to run().
   *        Default implementation does nothing - subclasses can override.
   */
  virtual void setStreamer(::BaseStreamer *streamer) { (void)streamer; }

  /**
   * @brief Get the generated output text.
   *        Default implementation returns empty string - subclasses can
   * override.
   */
  virtual std::string getOutput(int batch_idx = 0) const {
    (void)batch_idx;
    return "";
  }

  /**
   * @brief Request cancellation of the current run().
   *        Thread-safe: can be called from any thread.
   *        Default implementation does nothing - subclasses can override.
   */
  virtual void requestStop() { /* no-op by default */
  }

  /**
   * @brief Attach an XGrammar instance for grammar-constrained generation.
   *        Default implementation does nothing - subclasses can override.
   */
  virtual void setXGrammar(XGrammar *grammar) { (void)grammar; }

  /**
   * @brief Reset the XGrammar matcher state after generation.
   *        Default implementation does nothing - subclasses can override.
   */
  virtual void resetXGrammar() { /* no-op by default */
  }

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
  bool USE_FLASH_ATTENTION = true; /**< Enable flash GEMM attention path for
                                        prefill (decode always uses the
                                        per-row dot path). Defaults to true;
                                        read from nntr_config.json key
                                        "use_flash_attention". */

  // Performance metrics
  TransformerPerformanceMetrics performance_metrics;

  bool has_run_ = false;

  /** Native library directory for loading shared libraries (e.g., QNN context)
   */
  std::string native_lib_dir_;
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
