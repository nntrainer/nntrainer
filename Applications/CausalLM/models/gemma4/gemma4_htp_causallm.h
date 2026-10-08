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
#include <set>

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

  /** MoE block beside the dense MLP (config enable_moe_block; doc 55). The
   *  expert layer is the LFM2 MoE layer with router_type softmax_scale, so
   *  the same nntr_config keys as LFM2 drive it: moe_layer_dtype,
   *  moe_engine, moe_htp_layers, and moe_cache_experts (the per-layer expert
   *  cache C; NNTR_MOE_CACHE_EXPERTS in the environment overrides it). */
  bool ENABLE_MOE_BLOCK = false;
  unsigned int NUM_EXPERTS = 0;
  unsigned int NUM_EXPERTS_PER_TOK = 0;
  unsigned int MOE_INTERMEDIATE_SIZE = 0;
  std::string MOE_LAYER_DTYPE;
  std::string MOE_ENGINE = "cpu";
  std::set<int> MOE_HTP_LAYERS;
  unsigned int MOE_CACHE_EXPERTS = 0;

  /** Engines of the attention projections (q, k, v, o) and of the dense
   *  MLP's three fully connected layers, under the nntr_config keys
   *  Lfm2CausalLM reads (doc 50): attn_proj_engine / attn_proj_htp_layers
   *  and dense_ffn_engine / dense_ffn_htp_layers. "htp" moves only their
   *  prefill matmul to the accelerator -- FloatTensor::dot declines M == 1,
   *  so decode keeps the CPU Q4_0 kernel on the same weight, which the Q4_0
   *  registration path leaves resident. An empty layer list means every
   *  layer. Default "cpu" leaves every existing run unchanged. */
  /** @brief The tied lm_head takes the final norm and the logit softcap
   *  (doc 57 section 9.7): constructModel then ends at the last block's
   *  output and Gemma4CausalLM's lm_head norms it. */
  bool FOLD_OUTPUT_NORM = false;
  /** @brief lmhead_engine: "htp" runs the tied lm_head (norm, head,
   *  softcap) as one NPU call; "cpu" (default) keeps it on the CPU. */
  std::string LMHEAD_ENGINE = "cpu";
  std::string ATTN_PROJ_ENGINE = "cpu";
  std::set<int> ATTN_PROJ_HTP_LAYERS;
  std::string FFN_ENGINE = "cpu";
  std::set<int> FFN_HTP_LAYERS;

  std::string FULL_ATTENTION_ROPE_TYPE = "default";
  std::string SLIDING_ATTENTION_ROPE_TYPE = "default";
  float FULL_ATTENTION_ROPE_PARTIAL_ROTARY_FACTOR = 1.0f;
  float SLIDING_ATTENTION_ROPE_PARTIAL_ROTARY_FACTOR = 1.0f;
  float FINAL_LOGIT_SOFTCAPPING = 0.0f;
  bool ENABLE_SKIP_PREFILL_OPT = false;

  bool isKVSharedLayer(int layer_id) const;
  bool isSlidingAttentionLayer(int layer_id) const;
  unsigned int getAttentionHeadDim(int layer_id) const;
  unsigned int getKVHeadCount(int layer_id) const;
  unsigned int getKVCacheWidth(int layer_id) const;
  void appendSkipPrefillIfNeeded(std::vector<std::string> &props,
                                 bool enable_skip) const;
  std::pair<Tensor, Tensor>
  createGemma4KVCachePlaceholders(const int layer_id, unsigned int kv_width);
  /** The per-layer input embedding, projection and norm (hidden_size_per_
   *  layer_input != 0); leaves the result in per_layer_input. */
  void constructPerLayerInput(Tensor x, Tensor h);

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
  /** The MoE half of a Gemma-4 FFN block: router on @a router_input, experts
   *  on @a input, both already normed by the caller. */
  Tensor createMoe(const int layer_id, Tensor input);

  void registerCustomLayers() override;

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

protected:
  void allocateAndBindKVCache() override;
};
} // namespace causallm

#endif /* __GEMMA4_CAUSAL_LM_H__ */
