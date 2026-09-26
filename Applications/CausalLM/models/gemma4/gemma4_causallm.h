// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   gemma4_causallm.h
 * @brief  Gemma4 causal language model implementation.
 * @date   07 Apr 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Joonseok Oh <jrock.oh@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __GEMMA4_CAUSAL_LM_H__
#define __GEMMA4_CAUSAL_LM_H__

#include <causal_lm.h>

namespace causallm {

/**
 * @brief Gemma4Transformer class
 */
class Gemma4Transformer : virtual public Transformer {

public:
  static constexpr const char *architectures = "Gemma4Transformer";

  Gemma4Transformer(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(sanitizeConfig(cfg),
                sanitizeGenerationConfig(generation_cfg, cfg), nntr_cfg) {
    if (cfg.contains("layer_types")) {
      layer_types = cfg["layer_types"].get<std::vector<std::string>>();
    }

    setupParameters(cfg, generation_cfg,
                    nntr_cfg); // call this after setting up)
  }

  virtual ~Gemma4Transformer() = default;

protected:
  static json &sanitizeConfig(json &cfg);
  static json &sanitizeGenerationConfig(json &gen_cfg, const json &cfg);

  std::vector<std::string> layer_types;

  unsigned int GLOBAL_HEAD_DIM = 0;
  unsigned int NUM_GLOBAL_KEY_VALUE_HEADS = 0;
  bool ATTENTION_K_EQ_V = false;

  /** Per-layer-type RoPE theta from Gemma4 rope_parameters */
  unsigned int FULL_ATTENTION_ROPE_THETA = 0;
  unsigned int SLIDING_ATTENTION_ROPE_THETA = 0;

  unsigned int HIDDEN_SIZE_PER_LAYER_INPUT = 0;
  unsigned int VOCAB_SIZE_PER_LAYER_INPUT = 0;
  int NUM_KV_SHARED_LAYERS = 0;
  bool USE_DOUBLE_WIDE_MLP = false;
  float EMBEDDING_PER_LAYER_SCALE = 1.0f;

  std::string FULL_ATTENTION_ROPE_TYPE = "default";
  std::string SLIDING_ATTENTION_ROPE_TYPE = "default";
  float FULL_ATTENTION_ROPE_PARTIAL_ROTARY_FACTOR = 1.0f;
  float SLIDING_ATTENTION_ROPE_PARTIAL_ROTARY_FACTOR = 1.0f;
  float FINAL_LOGIT_SOFTCAPPING = 0.0f;
  bool ENABLE_SKIP_PREFILL_OPT = false;

  /** nntr_config "folded_lut": the sidecar embedding LUTs are PRE-FOLDED.
   *  The token table already contains the sqrt(hidden) embedding scaling, and
   *  the PLE table already contains the whole per-layer-input combine
   *  (projection -> 1/sqrt(hidden) -> rms_norm -> +16*ple_row -> sqrt(0.5)),
   *  so both runtime embedding scales collapse to 1 and constructModel wires
   *  the LUT output straight into the per-layer slices. Requires both
   *  embedding_file_name and ple_file_name. */
  bool FOLDED_LUT = false;

  bool isKVSharedLayer(int layer_id) const;
  bool isSlidingAttentionLayer(int layer_id) const;

  /**
   * @brief [kv-share] The layer whose K/V a KV-shared layer reads, or -1 when
   *        this layer owns its K/V.
   *
   * @details `num_kv_shared_layers` makes the trailing layers reuse an earlier
   *          layer's KV cache; the source is the LAST layer of the same
   *          `layer_types` entry among the non-shared prefix. This is the ONE
   *          source of truth for that rule -- both the graph builder
   *          (createTransformerDecoderBlock -> createSharedAttention, which
   *          wires the source's k/v norms into mha_core) and the KV allocator
   *          (CausalLM::allocateAndBindKVCache, through the
   *          getKVSourceLayer() hook, which declares the aliases to
   *          KVCacheManager) call it, exactly as every consumer of the
   *          per-layer window calls getLayerSlidingWindow().
   *
   *          They MUST agree: if allocation and graph building resolved
   *          different sources, a layer would attend over the wrong layer's
   *          K/V -- fluent, wrong, and with no crash to notice it by.
   *
   * @param[in] layer_id decoder layer index
   * @return source layer id (< layer_id), or -1 when the layer owns its cache
   */
  int getSharedKVSourceLayer(int layer_id) const;

  unsigned int getAttentionHeadDim(int layer_id) const;
  unsigned int getKVHeadCount(int layer_id) const;
  unsigned int getKVCacheWidth(int layer_id) const override;
  /** @copydoc Transformer::getLayerSlidingWindow(int) — the layer_types table
   *  names the sliding layers; with no table every layer slides. */
  unsigned int getLayerSlidingWindow(int layer_id) const override {
    return isSlidingAttentionLayer(layer_id) ? SLIDING_WINDOW : UINT_MAX;
  }
  void appendSkipPrefillIfNeeded(std::vector<std::string> &props,
                                 bool enable_skip) const;
  std::pair<Tensor, Tensor>
  createGemma4KVCachePlaceholders(const int layer_id, unsigned int kv_width);
  /**
   * @brief Per-layer KV placeholder factory used by the base
   *        wireAttentionKVCache. The cache width is per-layer here
   *        (sliding vs global head_dim), so n_heads is unused.
   */
  std::pair<Tensor, Tensor> createKVCachePlaceholders(const int layer_id,
                                                      int n_heads) override;

public:
  Tensor createAttention(const int layer_id, int seq_len, int n_heads,
                         int head_dim, Tensor query, Tensor key,
                         Tensor value) override;
  Tensor createSharedAttention(const int layer_id, const int shared_kv_layer_id,
                               int seq_len, int n_heads, int head_dim,
                               Tensor query);

  Tensor createTransformerDecoderBlock(const int layer_id,
                                       Tensor input) override;

  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  std::pair<Tensor, Tensor> constructModel() override;

  Tensor createMlp(const int layer_id, int dim, int hidden_dim,
                   Tensor input) override;

  void registerCustomLayers() override;

  /** @copydoc Transformer::getModelFeatures() — gemma4: q/k/v-norm, GeGLU,
   *  sandwich norm, dual head_dim (sliding/global), PLE, KV-share+skip-prefill,
   *  attn+final soft-cap, untie-able QINT4 lm_head, decode-GPU ON. */
  nntrainer::ModelFeatures getModelFeatures() const override {
    nntrainer::ModelFeatures f;
    f.has_qk_norm = true;
    f.has_v_norm = true;
    f.mlp_kind = nntrainer::MlpKind::GEGLU;
    f.norm_style = nntrainer::NormStyle::SANDWICH;
    f.sliding_window = true;
    f.kv_share_skip_prefill = true;
    f.dual_head_dim = true;
    f.ple = true;
    f.attn_softcap = true;
    f.final_softcap = true;
    f.lmhead_kind = nntrainer::LmHeadKind::UNTIED_QINT4;
    f.decode_gpu = true;
    f.decode_rope_gpu = true; // gemma4: decode RoPE on GPU (token-identical)
    return f;
  }

protected:
  Tensor per_layer_input;
  std::vector<Tensor> layer_k_norms;
  std::vector<Tensor> layer_v_norms;
};

/**
 * @brief Gemma4CausalLM class
 */
class Gemma4CausalLM : public CausalLM, public Gemma4Transformer {

public:
  static constexpr const char *architectures = "Gemma4ForCausalLM";

  Gemma4CausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(sanitizeConfig(cfg),
                sanitizeGenerationConfig(generation_cfg, cfg), nntr_cfg,
                ModelType::CAUSALLM),
    CausalLM(sanitizeConfig(cfg), sanitizeGenerationConfig(generation_cfg, cfg),
             nntr_cfg),
    Gemma4Transformer(sanitizeConfig(cfg),
                      sanitizeGenerationConfig(generation_cfg, cfg), nntr_cfg) {
  }

  virtual ~Gemma4CausalLM() = default;

  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override {
    CausalLM::setupParameters(cfg, generation_cfg, nntr_cfg);
    Gemma4Transformer::setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  std::pair<Tensor, Tensor> constructModel() override;

  void registerCustomLayers() override;
};
} // namespace causallm

#endif /* __GEMMA4_CAUSAL_LM_H__ */
