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

private:
  unsigned int num_experts = 0;
  unsigned int top_k_experts = 0;
  unsigned int moe_intermediate_size = 0;
  unsigned int moe_cache_size = 0;
};

} // namespace causallm

#endif // __GEMMA4_MOE_CAUSAL_LM_H__
