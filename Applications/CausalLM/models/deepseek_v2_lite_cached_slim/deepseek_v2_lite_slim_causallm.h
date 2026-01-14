//
// Created by donghak on 25. 12. 4..
//

#ifndef NNTRAINER_DEEPSEEK_V2_LITE_SLIM_CAUSALLM_H
#define NNTRAINER_DEEPSEEK_V2_LITE_SLIM_CAUSALLM_H
#include <causal_lm.h>

namespace causallm {

/**
 * @class DeepseekV2ForCausalLM
 * @brief Mixture of Expert Layer for DeepSeek_V2_Lite
 */
class DeepseekV2SlimCausalLM : public CausalLM {
public:
  static constexpr const char *architecture = "DeepseekV2SlimCausalLM";
  DeepseekV2SlimCausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg, ModelType::CAUSALLM),
    CausalLM(cfg, generation_cfg, nntr_cfg) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  virtual ~DeepseekV2SlimCausalLM() = default;

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
  unsigned int NUM_GROUP_EXPERTS;     /**< Number of group experts */
  bool NORM_TOPK_PROB;                /**< Normalize top-k probabilities */

  std::vector<std::string> LAYER_TYPES; /**< Layer types */
  float ATTENTION_ROPE_SCALING_FACTOR;  /**< Attention RoPE scaling factor */
  float ROPE_SCALING_BETA_FAST;         /**< RoPE scaling beta fast */
  float ROPE_SCALING_BETA_SLOW;         /**< RoPE scaling beta slow */
  float ROPE_SCALING_MSCALE;            /**< RoPE scaling mscale */
  float ROPE_SCALING_MSCALE_ALL_DIM;    /**< RoPE scaling mscale all dim */
  std::string ROPE_SCALING_TYPE;        /**< RoPE scaling type */
  unsigned int ROPE_SCALING_MAX_POSITION_EMBEDDINGS; /**< RoPE scaling max position embeddings */
  float ROUTED_SCALING_FACTOR;                       /**< Routed scaling factor */

  // MLA specific parameters
  unsigned int Q_LORA_RANK;      /**< Q LoRA rank */
  unsigned int KV_LORA_RANK;     /**< KV LoRA rank */
  unsigned int QK_NOPE_HEAD_DIM; /**< QK non-RoPE head dimension */
  unsigned int QK_ROPE_HEAD_DIM; /**< QK RoPE head dimension */
  unsigned int V_HEAD_DIM;       /**< Value head dimension */
};

} // namespace causallm
#endif // NNTRAINER_DEEPSEEK_V2_LITE_SLIM_CAUSALLM_H
