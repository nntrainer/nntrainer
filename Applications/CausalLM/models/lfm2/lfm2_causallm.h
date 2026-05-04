// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   lfm2_causallm.h
 * @date   4 May 2026
 * @brief  This defines an LFM2 causal language model.
 */

#ifndef __LFM2_CAUSAL_LM_H__
#define __LFM2_CAUSAL_LM_H__

#include <causal_lm.h>

namespace causallm {

/**
 * @brief LFM2 transformer class
 */
class Lfm2Transformer : virtual public Transformer {
public:
  static constexpr const char *architectures = "Lfm2Model";

  Lfm2Transformer(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(sanitizeConfig(cfg),
                sanitizeGenerationConfig(generation_cfg, cfg), nntr_cfg) {
    setupParameters(sanitizeConfig(cfg),
                    sanitizeGenerationConfig(generation_cfg, cfg), nntr_cfg);
  }

  virtual ~Lfm2Transformer() = default;

  std::vector<LayerHandle>
  createTransformerDecoderBlock(const int layer_id,
                                std::string input_name) override;

  std::vector<LayerHandle> createAttention(const int layer_id, int seq_len,
                                           int n_heads, int head_dim,
                                           std::string query_name,
                                           std::string key_name,
                                           std::string value_name) override;

  std::vector<LayerHandle> createMlp(const int layer_id, int dim,
                                     int hidden_dim,
                                     std::string input_name) override;

  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  void registerCustomLayers() override;

protected:
  static json &sanitizeConfig(json &cfg);
  static json &sanitizeGenerationConfig(json &gen_cfg, const json &cfg);

  std::vector<std::string> layer_types;
  unsigned int CONV_KERNEL_SIZE = 3;
};

/**
 * @brief LFM2 CausalLM class
 */
class Lfm2CausalLM : public CausalLM, public Lfm2Transformer {
public:
  static constexpr const char *architectures = "Lfm2ForCausalLM";

  Lfm2CausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(sanitizeConfig(cfg),
                sanitizeGenerationConfig(generation_cfg, cfg), nntr_cfg,
                ModelType::CAUSALLM),
    CausalLM(sanitizeConfig(cfg), sanitizeGenerationConfig(generation_cfg, cfg),
             nntr_cfg),
    Lfm2Transformer(sanitizeConfig(cfg),
                    sanitizeGenerationConfig(generation_cfg, cfg), nntr_cfg) {}

  virtual ~Lfm2CausalLM() = default;

  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override {
    CausalLM::setupParameters(cfg, generation_cfg, nntr_cfg);
    Lfm2Transformer::setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  void registerCustomLayers() override;
};

} // namespace causallm

#endif // __LFM2_CAUSAL_LM_H__
