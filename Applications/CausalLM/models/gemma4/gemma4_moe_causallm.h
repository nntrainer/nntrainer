// SPDX-License-Identifier: Apache-2.0
/**
 * @file   gemma4_moe_causallm.h
 * @brief  Gemma4 MoE causal language model implementation.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug    No known bugs
 */

#ifndef __GEMMA4_MOE_CAUSAL_LM_H__
#define __GEMMA4_MOE_CAUSAL_LM_H__

#include <gemma4_causallm.h>

namespace causallm {

/**
 * @brief Gemma4 sparse MoE variant.
 */
class Gemma4MoECausalLM : public Gemma4CausalLM {
public:
  Gemma4MoECausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(sanitizeConfig(cfg),
                sanitizeGenerationConfig(generation_cfg, cfg), nntr_cfg,
                ModelType::CAUSALLM),
    Gemma4CausalLM(cfg, generation_cfg, nntr_cfg) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  ~Gemma4MoECausalLM() override = default;

  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;
  void registerCustomLayers() override;

protected:
  Tensor createFeedForwardBlock(const int layer_id, Tensor post_attention,
                                bool is_kv_shared_layer) override;
  bool fuseNormResidualAdd() const override { return fuse_norm_add; }

private:
  unsigned int num_experts = 0;
  unsigned int top_k_experts = 0;
  unsigned int moe_intermediate_size = 0;
  unsigned int moe_cache_size = 0;
  /**
   * @brief build the three residual adds of each decoder block (after
   * post_attention_norm, post_ffn_norm_2 and post_ffn_norm) as two-input
   * rms_norm layers that add in the norm kernel. Set for the gpu engine with
   * FP16 activations (the only arm with a fused kernel, bit-identical to the
   * separate norm and add); NNTR_G4MOE_NORM_ADD=0 keeps the separate layers.
   */
  bool fuse_norm_add = false;
  /**
   * @brief weight dtype of the expert projections (nntr_config
   * "moe_expert_dtype", defaults to fc_layer_dtype), e.g. Q4_0, QS4CX, TQ2CX
   */
  std::string moe_expert_dtype;
  /**
   * @brief the router RMSNorm has a learned scale (nntr_config
   * "moe_router_norm_weight", default false as in HF Gemma4)
   */
  bool moe_router_norm_weight = false;
};

} // namespace causallm

#endif // __GEMMA4_MOE_CAUSAL_LM_H__
