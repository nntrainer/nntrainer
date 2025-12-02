// SPDX-License-Identifier: Apache-2.0
/**
 *
 * @file   ernie_causallm.h
 * @brief  ernie 4.5 causallm header
 * @date   02 December 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef NNTRAINER_ERNIE_CAUSALLM_H
#define NNTRAINER_ERNIE_CAUSALLM_H
#include <causal_lm.h>

namespace causallm {

/**
 * @class Ernie4_5_MoeForCausalLM
 * @brief Mixture of Expert Layer for ERNIE 4.5
 */
class Ernie4_5_MoeForCausalLM : public CausalLM {
public:
  static constexpr const char *architecture = "Ernie4_5_MoeForCausalLM";
  Ernie4_5_MoeForCausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
    CausalLM(cfg, generation_cfg, nntr_cfg) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  virtual ~Ernie4_5_MoeForCausalLM() = default;

  /**
   * @brief MoE layer
   */
  /**
   * @brief Create MLP layer
   * @param layer_id Layer ID
   * @param dim Dimension
   * @param hidden_dim Hidden dimension
   * @param input_name Input name
   * @return std::vector<LayerHandle> Vector of layer handles
   */
  std::vector<LayerHandle> createMlp(const int layer_id, int dim,
                                     int hidden_dim,
                                     std::string input_name) override;

  /**
   * @brief Create Attention layer
   * @param layer_id Layer ID
   * @param seq_len Sequence length
   * @param n_heads Number of heads
   * @param head_dim Head dimension
   * @param query_name Query name
   * @param key_name Key name
   * @param value_name Value name
   * @return std::vector<LayerHandle> Vector of layer handles
   */
  std::vector<LayerHandle> createAttention(int layer_id, int seq_len,
                                           int n_heads, int head_dim,
                                           std::string query_name,
                                           std::string key_name,
                                           std::string value_name) override;
  /**
   * @brief Setup parameters for the model
   * @param cfg Configuration json
   * @param generation_cfg Generation configuration json
   * @param nntr_cfg NNtrainer configuration json
   */
  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  /**
   * @brief Register custom layers
   */
  void registerCustomLayers() override;

private:
  unsigned int NUM_EXPERTS;           /**< Number of experts */
  unsigned int NUM_EXPERTS_PER_TOK;   /**< Number of experts per token */
  unsigned int NUM_SHARED_EXPERTS;    /**< Number of shared experts */
  unsigned int MOE_INTERMEDIATE_SIZE; /**< MoE intermediate size */
  float MOE_NORM_MIN;                 /**< MoE normalization minimum */

  std::vector<std::string> LAYER_TYPES; /**< Layer types */
  float ATTENTION_ROPE_SCALING_FACTOR;  /**< Attention RoPE scaling factor */
};

} // namespace causallm
#endif // NNTRAINER_ERNIE_CAUSALLM_H
