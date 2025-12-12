// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 * Copyright (C) 2025 Seungback Hong <sb92.hong@samsung.com>
 * Copyright (C) 2025 Hyeonseok Lee <hs89.lee@samsung.com>
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   causal_lm.h
 * @date   10 July 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   This causal_lm.h constructs a class for Transformer-based Causal
 * Language Model (CausalLM). It aims to support AutoModelForCausalLM with
 * nntrainer. It supports the following models:
 *          - Qwen3
 *          - Qwen3-MoE
 * @note   This CausalLM assumes the Decoder-based model, which structure is
 *
 *           [Input]
 *              |
 *         [Embedding]
 *              |
 *        [Decoder Block] (repeated N times)
 *              |
 *          [RMSNorm]
 *              |
 *           [LMHead]
 */

#ifndef __CAUSAL_LM_H__
#define __CAUSAL_LM_H__

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#define WSTR std::wstring
#define WCHAR_P wchar_t *
#else
#define WIN_EXPORT
#define WSTR std::string
#define WCHAR_P std::string &
#endif

#include <layer.h>
#include <model.h>
#include <random>

#include <limits.h>

#include "json.hpp"
#include <fstream>
#include <tokenizers_c.h>
#include <tokenizers_cpp.h>

namespace causallm {

/*** ALIAS ****/
using LayerHandle = std::shared_ptr<ml::train::Layer>;
using ModelHandle = std::unique_ptr<ml::train::Model>;

using json = nlohmann::json;

/**
 * @brief CausalLM Class
 */
WIN_EXPORT class CausalLM {

public:
  /**
   * @brief Construct a new CausalLM object
   * @param cfg Configuration for the model (config.json)
   * @param generation_cfg Configuration for the generation
   * (generation_config.json)
   * @param nntr_cfg Configuration for nntrainer (nntrainer_config.json)
   */
  CausalLM(json &cfg, json &generation_cfg, json &nntr_cfg);

  /**
   * @brief Destroy the CausalLM object
   */
  virtual ~CausalLM() { free(ids_history); }

  /**
   * @brief Initialize and Construct the CausalLM model
   */
  void initialize();

  /**
   * @brief Load the model weights from a file
   */
  void load_weight(const std::string &weight_path);

  /**
   * @brief Save the weight to a file
   */
  void save_weight(const std::string &weight_path);

  /**
   * @brief run the CausalLM model
   */
  void run(const WSTR prompt, bool do_sample = false,
           const WSTR system_prompt = "", const WSTR tail_prompt = "");

protected:
  /**
   * @brief Setup the parameters for the CausalLM model
   */
  virtual void setupParameters(json &cfg, json &generation_cfg, json &nntr_cfg);

  /**
   * @brief Construct Model
   */
  virtual void constructModel();

  /**
   * @brief create Decoder Part
   */
  virtual std::vector<LayerHandle>
  createTransformerDecoderBlock(const int layer_id, std::string input_name);

  /**
   * @brief create Attention Layer
   */
  virtual std::vector<LayerHandle>
  createAttention(const int layer_id, int seq_len, int n_heads, int head_dim,
                  std::string query_name, std::string key_name,
                  std::string value_name);

  /**
   * @brief create Feed Forward Layer
   */
  virtual std::vector<LayerHandle> createMlp(const int layer_id, int dim,
                                             int hidden_dim,
                                             std::string input_name);

  /**
   * @brief register CustomLayers
   */
  virtual void registerCustomLayers();

  /**
   * @brief register Outputs
   */
  virtual void
  registerOutputs(std::unique_ptr<tokenizers::Tokenizer> &tokenizer,
                  std::vector<unsigned int> ids, unsigned int pos,
                  const std::vector<bool> &eos_list);

  /**
   * @brief save kv cache
   */
  WIN_EXPORT virtual void save_kvcache(std::string path, int to);

  /**
   * @brief load kv cache
   */
  WIN_EXPORT virtual void load_kvcache(std::string path, int to);

  /**
   * @brief generate
   */
  std::vector<unsigned int> generate(float *logits, bool do_sample,
                                     float repetition_penalty = 1,
                                     unsigned int *input_ids = nullptr,
                                     unsigned int NUM_INPUT_IDS = 0);

  bool is_initialized = false; /**< Flag to check if the model is initialized */
  ModelHandle model;

  /** internal buffer */
  std::vector<std::string>
    output_list;             /**< List of output names for the model */
  unsigned int *ids_history; /**< History of input IDs for the model */

  /** tokenizer */
  std::unique_ptr<tokenizers::Tokenizer> tokenizer;
  std::vector<int> pending_ids_;

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
  std::string LMHEAD_DTYPE;    /** embedding dtype */
  std::string FC_LAYER_DTYPE;  /** custom_fc_lora */
  std::vector<unsigned int> EOS_TOKEN_ID;
  unsigned int BOS_TOKEN_ID;
  float TEMPERATURE;
  unsigned int TOP_K;
  float TOP_P;
  unsigned int SLIDING_WINDOW = UINT_MAX;
  unsigned int SLIDING_WINDOW_PATTERN = 5;
  unsigned int ROPE_THETA = 10000; /**< RoPE theta value */
  float NORM_EPS = 1e-5;           /**< RMSNorm epsilon value */
  int GQA_SIZE;

  std::vector<unsigned int> BAD_WORD_IDS; /**< List of bad word IDs */
  unsigned int NUM_BADWORDS;              /**< Number of bad words */
  unsigned int BATCH_SIZE;                /**< Batch size for the model */
  unsigned int INIT_SEQ_LEN;              /**< Initial sequence length */
  unsigned int MAX_POSITION_EMBEDDINGS;   /**< max position embeddings */
  bool MEMORY_SWAP;                       /**< Memory swap option */
  unsigned int FSU_LOOKAHEAD;
  unsigned int SYS_PROMP_LEN;
  std::string PRE_COMPUTED_CACHE_PATH;
  std::string TAIL_PROMPT;
  bool SAVE_KVCACHE;
  bool USE_KVCACHE;
  unsigned int global_token_len;

  std::mt19937 rng; /**< Random Number Gen */
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
