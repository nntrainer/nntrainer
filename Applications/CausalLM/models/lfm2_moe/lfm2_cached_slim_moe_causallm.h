// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jungwon-Lee <jungone.lee@samsung.com>
 *
 * @file   lfm2_cached_slim_moe_causallm.h
 * @date   06 July 2026
 * @brief  LFM2-8B-A1B Cached-Slim Mixture-of-Experts causal LM.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __LFM2_CACHED_SLIM_MOE_CAUSALLM_H__
#define __LFM2_CACHED_SLIM_MOE_CAUSALLM_H__

#include "lfm2_moe_causallm.h"

namespace causallm {

/**
 * @brief Lfm2CachedSlimMoeCausalLM - shares the base MoE weights but streams
 *        experts through a bounded LRU cache ("lfm2_moe_cached_slim").
 */
class Lfm2CachedSlimMoeCausalLM : public Lfm2MoeCausalLM {

public:
  static constexpr const char *architectures = "Lfm2CachedSlimMoeForCausalLM";

  Lfm2CachedSlimMoeCausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg, ModelType::CAUSALLM),
    Lfm2MoeCausalLM(cfg, generation_cfg, nntr_cfg) {}

  virtual ~Lfm2CachedSlimMoeCausalLM() = default;

  void registerCustomLayers() override;

protected:
  Tensor createMoeLayer(const int layer_id, Tensor input) override;
};

} // namespace causallm

#endif /* __LFM2_CACHED_SLIM_MOE_CAUSALLM_H__ */
