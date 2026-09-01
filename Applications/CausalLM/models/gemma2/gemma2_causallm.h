// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   gemma2_causallm.h
 * @date   08 Jun 2026
 * @brief  Gemma2 layer-graph causal language model
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   Gemma2 layer-graph model. Architecturally Gemma2 is Gemma3 minus the
 *         per-head q/k RMSNorm; it keeps the sandwich (pre/post) norms, GeGLU
 *         MLP, attention-logit softcapping, alternating sliding-window pattern,
 *         input embedding scale (sqrt(hidden)) and tied embeddings. The
 *         final_logit_softcapping (30) is a generation-time detail that the
 *         layer-graph LM-head does not apply yet (affects last-token selection
 *         only, not prefill throughput).
 */

#ifndef __GEMMA2_CAUSAL_LM_H__
#define __GEMMA2_CAUSAL_LM_H__

#include <cmath>

#include <causal_lm.h>

namespace causallm {

/**
 * @brief Gemma2Transformer class
 */
class Gemma2Transformer : virtual public Transformer {

public:
  static constexpr const char *architectures = "Gemma2Transformer";

  Gemma2Transformer(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(sanitizeConfig(cfg),
                sanitizeGenerationConfig(generation_cfg, cfg), nntr_cfg) {
    if (cfg.contains("layer_types")) {
      layer_types = cfg["layer_types"].get<std::vector<std::string>>();
    }
    EMBEDDING_SCALE = std::sqrt(static_cast<float>(cfg["hidden_size"]));
  }

  virtual ~Gemma2Transformer() = default;

protected:
  static json &sanitizeConfig(json &cfg);
  static json &sanitizeGenerationConfig(json &gen_cfg, const json &cfg);

  std::vector<std::string> layer_types;

  /** @copydoc Transformer::getLayerSlidingWindow(int) — the layer_types table
   *  names the sliding layers; with no table every layer slides. */
  unsigned int getLayerSlidingWindow(int layer_id) const override {
    if (!layer_types.empty()) {
      if (layer_id < static_cast<int>(layer_types.size()) &&
          layer_types[layer_id] == "sliding_attention")
        return SLIDING_WINDOW;
      return UINT_MAX;
    }
    return SLIDING_WINDOW;
  }

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

  /** @copydoc Transformer::getModelFeatures() — gemma2: no q/k-norm, GeGLU,
   *  sandwich norm, alternating sliding window, attn+final soft-cap, tied
   *  lm_head, decode-GPU on (attn; rope host). */
  nntrainer::ModelFeatures getModelFeatures() const override {
    nntrainer::ModelFeatures f;
    f.mlp_kind = nntrainer::MlpKind::GEGLU;
    f.norm_style = nntrainer::NormStyle::SANDWICH;
    f.sliding_window = true;
    f.attn_softcap = true;
    f.final_softcap = true;
    f.lmhead_kind = nntrainer::LmHeadKind::TIED;
    f.decode_gpu = true;
    f.decode_rope_gpu =
      false; // gemma2: GPU attn but HOST decode-RoPE (diverges)
    return f;
  }
};

/**
 * @brief Gemma2CausalLM class
 */
class Gemma2CausalLM : public CausalLM, public Gemma2Transformer {

public:
  static constexpr const char *architectures = "Gemma2ForCausalLM";

  Gemma2CausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(sanitizeConfig(cfg),
                sanitizeGenerationConfig(generation_cfg, cfg), nntr_cfg,
                ModelType::CAUSALLM),
    CausalLM(sanitizeConfig(cfg), sanitizeGenerationConfig(generation_cfg, cfg),
             nntr_cfg),
    Gemma2Transformer(sanitizeConfig(cfg),
                      sanitizeGenerationConfig(generation_cfg, cfg), nntr_cfg) {
  }

  virtual ~Gemma2CausalLM() = default;

  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override {
    CausalLM::setupParameters(cfg, generation_cfg, nntr_cfg);
    Gemma2Transformer::setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  void registerCustomLayers() override;

private:
};
} // namespace causallm

#endif /* __GEMMA2_CAUSAL_LM_H__ */
