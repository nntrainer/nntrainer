// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   gemma3_causallm.h
 * @date   24 Dec 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __GEMMA3_CAUSAL_LM_H__
#define __GEMMA3_CAUSAL_LM_H__

#include <causal_lm.h>

namespace causallm {

/**
 * @brief Gemma3Transformer class
 */
class Gemma3Transformer : virtual public Transformer {

public:
  static constexpr const char *architectures = "Gemma3Transformer";

  Gemma3Transformer(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(sanitizeConfig(cfg),
                sanitizeGenerationConfig(generation_cfg, cfg), nntr_cfg) {
    if (cfg.contains("layer_types")) {
      layer_types = cfg["layer_types"].get<std::vector<std::string>>();
    }
    EMBEDDING_SCALE = std::sqrt(static_cast<float>(cfg["hidden_size"]));
  }

  virtual ~Gemma3Transformer() = default;

protected:
  static json &sanitizeConfig(json &cfg);
  static json &sanitizeGenerationConfig(json &gen_cfg, const json &cfg);

  std::vector<std::string> layer_types;

public:
  Tensor createAttention(const int layer_id, int seq_len, int n_heads,
                         int head_dim, Tensor query, Tensor key,
                         Tensor value) override;

  Tensor createTransformerDecoderBlock(const int layer_id,
                                       Tensor input) override;

  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  Tensor createMlp(const int layer_id, int dim, int hidden_dim,
                   Tensor input) override;

  void registerCustomLayers() override;
};

/**
 * @brief Gemma3CausalLM class
 */
class Gemma3CausalLM : public CausalLM, public Gemma3Transformer {

public:
  static constexpr const char *architectures = "Gemma3ForCausalLM";

  Gemma3CausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(sanitizeConfig(cfg),
                sanitizeGenerationConfig(generation_cfg, cfg), nntr_cfg),
    CausalLM(sanitizeConfig(cfg), sanitizeGenerationConfig(generation_cfg, cfg),
             nntr_cfg),
    Gemma3Transformer(sanitizeConfig(cfg),
                      sanitizeGenerationConfig(generation_cfg, cfg), nntr_cfg) {
  }

  virtual ~Gemma3CausalLM() = default;

  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override {
    CausalLM::setupParameters(cfg, generation_cfg, nntr_cfg);
    Gemma3Transformer::setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  void registerCustomLayers() override;

private:
};
} // namespace causallm

#endif /* __GEMMA3_CAUSAL_LM_H__ */
