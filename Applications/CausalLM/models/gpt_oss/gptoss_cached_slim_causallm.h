// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   gptoss_causallm.h
 * @date   26 Aug 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   Please refer to the following code :
 * https://github.com/huggingface/transformers/blob/e68146fbe7052a6dc8456f48edabe705dc1f7381/src/transformers/models/gpt_oss/modeling_gpt_oss.py
 */

#ifndef __GPTOSS_CACHED_SLIM_CAUSALLM_H__
#define __GPTOSS_CACHED_SLIM_CAUSALLM_H__

#include <causal_lm.h>

namespace causallm {

/**
 * @brief GptOssCachedSlimCausalLM
 */
class GptOssCachedSlimCausalLM : public CausalLM {
public:
  static constexpr const char *architectures = "GptOssCachedSlimCausalLM";

  GptOssCachedSlimCausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
    CausalLM(cfg, generation_cfg, nntr_cfg) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  virtual ~GptOssCachedSlimCausalLM() = default;

  /**
   * @brief createAttention
   * @note sink attention with sliding window
   */
  std::vector<LayerHandle> createAttention(const int layer_id, int seq_len,
                                           int n_heads, int head_dim,
                                           std::string query_name,
                                           std::string key_name,
                                           std::string value_name) override;

  /**
   * @brief MoE layer
   */
  std::vector<LayerHandle> createMlp(const int layer_id, int dim,
                                     int hidden_dim,
                                     std::string input_name) override;

  /**
   * @brief setupParameters
   */
  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  /**
   * @brief registerCutomLayers
   */
  void registerCustomLayers() override;

private:
  unsigned int NUM_EXPERTS;
  unsigned int NUM_EXPERTS_PER_TOK;
  std::vector<std::string> LAYER_TYPES;
  float ATTENTION_ROPE_SCALING_FACTOR;
};

} // namespace causallm

#endif /** __GPTOSS_CAUSALLM_H__ */
