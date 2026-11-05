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
  /**
   * @brief The sliding-window KV ring (and the chunked prefill that comes with
   * it) is this model's default. Measured with NNTR_KV_WINDOW_RING=1 against
   * the linear cache: the generated text is identical on short prompts and on
   * a prompt that wraps the 1024-row window, the sliding layers store
   * kvRingCap() rows instead of max_seq_len (-200 MiB of GPU memory on
   * Adreno at the pack's 512-row plane), and a prompt longer than the plane
   * is fed in chunks instead of being cut. The ring is granted only where a
   * ring-aware attention arm resolves (causallm::kvRingEnabled), so the cpu
   * engine keeps the linear cache; NNTR_KV_WINDOW_RING=0 is the opt-out of
   * both the ring and the chunked feed.
   * The cuda engine is not measured for this model yet and keeps the ring an
   * explicit opt-in (NNTR_KV_WINDOW_RING=1).
   */
  bool kvRingByDefault() const override { return nntr_engine_env() != "cuda"; }

private:
  unsigned int num_experts = 0;
  unsigned int top_k_experts = 0;
  unsigned int moe_intermediate_size = 0;
  unsigned int moe_cache_size = 0;
  /** stream the experts from the file into device slots (gpu engine) */
  bool moe_fsu = false;
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
