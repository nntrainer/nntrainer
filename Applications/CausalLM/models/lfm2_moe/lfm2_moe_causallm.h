// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jungwon-Lee <jungone.lee@samsung.com>
 *
 * @file   lfm2_moe_causallm.h
 * @date   06 July 2026
 * @brief  This declares the LFM2-8B-A1B Mixture-of-Experts causal LM.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   Inherits the dense LFM2 model (Lfm2CausalLM) and replaces the dense
 *         SwiGLU FFN with MoE routing on layers >= num_dense_layers.
 */

#ifndef __LFM2_MOE_CAUSALLM_H__
#define __LFM2_MOE_CAUSALLM_H__

#include "lfm2_causallm.h"

namespace causallm {

/**
 * @brief Lfm2MoeCausalLM - LFM2-8B-A1B MoE (Base variant, "lfm2_moe" layer)
 */
class Lfm2MoeCausalLM : public Lfm2CausalLM {

public:
  static constexpr const char *architectures = "Lfm2MoeForCausalLM";

  Lfm2MoeCausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg, ModelType::CAUSALLM),
    Lfm2CausalLM(cfg, generation_cfg, nntr_cfg) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  virtual ~Lfm2MoeCausalLM() = default;

  Tensor createMlp(const int layer_id, int dim, int hidden_dim,
                   Tensor input) override;

  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  void registerCustomLayers() override;

protected:
  unsigned int NUM_EXPERTS = 0;
  unsigned int NUM_EXPERTS_PER_TOK = 0;
  unsigned int MOE_INTERMEDIATE_SIZE = 0;
  unsigned int NUM_DENSE_LAYERS = 0;

  /**
   * @brief Create the variant-specific MoE layer for a given layer id.
   * @note Overridden by the Slim / CachedSlim variants to emit their layer type.
   */
  virtual Tensor createMoeLayer(const int layer_id, Tensor input);
};

} // namespace causallm

#endif /* __LFM2_MOE_CAUSALLM_H__ */
