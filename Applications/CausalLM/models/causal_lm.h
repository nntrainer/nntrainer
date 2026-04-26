// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 * Copyright (C) 2025 Seungback Hong <sb92.hong@samsung.com>
 * Copyright (C) 2025 Hyeonseok Lee <hs89.lee@samsung.com>
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   causal_lm.h
 * @date   10 July 2025
 * @see    https://github.com/nntrainer/nntrainer
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
 *        [Transformer]
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

#include <transformer.h>

namespace causallm {

/**
 * @brief CausalLM Class
 */
WIN_EXPORT class CausalLM : virtual public Transformer {

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
  virtual ~CausalLM() {
    if (ids_history)
      free(ids_history);
  }

  /**
   * @brief run the CausalLM model
   */
  void run(const WSTR prompt, bool do_sample = false,
           const WSTR system_prompt = "", const WSTR tail_prompt = "",
           bool log_output = true) override;

  /**
   * @brief Get the generated output text
   * @param batch_idx Index of the batch item
   * @return Generated text string
   */
  std::string getOutput(int batch_idx = 0) const;

  /**
   * @brief get the status of run
   */
  bool hasRun() const { return has_run_; }

protected:
  /**
   * @brief Setup the parameters for the CausalLM model
   */
  virtual void setupParameters(json &cfg, json &generation_cfg,
                               json &nntr_cfg) override;

  /**
   * @brief Construct Model — extends Transformer's symbolic graph with the
   *        LM-head and returns the final {input, logits} pair.
   */
  virtual std::pair<Tensor, Tensor> constructModel() override;

  /**
   * @brief register Outputs
   */
  virtual void
  registerOutputs(std::unique_ptr<tokenizers::Tokenizer> &tokenizer,
                  std::vector<unsigned int> ids, unsigned int pos,
                  const std::vector<bool> &eos_list, bool log_output = true);

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

  /**
   * @brief registerCutomLayers
   */
  void registerCustomLayers() override;

  /** internal buffer */
  std::vector<std::string>
    output_list;             /**< List of output names for the model */
  unsigned int *ids_history; /**< History of input IDs for the model */

  std::vector<int> pending_ids_;

  std::string LMHEAD_DTYPE; /** embedding dtype */
  std::vector<unsigned int> EOS_TOKEN_ID;
  unsigned int BOS_TOKEN_ID;
  float TEMPERATURE;
  unsigned int TOP_K;
  float TOP_P;

  std::vector<unsigned int> BAD_WORD_IDS; /**< List of bad word IDs */
  unsigned int NUM_BADWORDS;              /**< Number of bad words */

  unsigned int SYS_PROMP_LEN;
  std::string PRE_COMPUTED_CACHE_PATH;
  std::string TAIL_PROMPT;
  bool SAVE_KVCACHE;
  bool USE_KVCACHE;
  unsigned int global_token_len;

  bool has_run_ = false;

  std::mt19937 rng; /**< Random Number Gen */
};

} // namespace causallm

#endif
