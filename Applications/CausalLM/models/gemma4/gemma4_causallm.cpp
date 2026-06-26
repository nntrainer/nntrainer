// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   gemma4_causallm.cpp
 * @date   07 Apr 2026
 * @brief  This defines a Gemma4 causal language model.
 * @see    https://github.com/nnstreamer/
 * @author Joonseok Oh <jrock.oh@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <gemma4_causallm.h>

#include <algorithm>
#include <cmath>

#include <app_context.h>
#include <engine.h>
#include <llm_util.hpp>
#include <logit_softcapping.h>
#include <model.h>
#include <per_layer_slice.h>
#include <reshaped_rms_norm.h>
#include <scalar_multiply.h>

namespace causallm {

bool Gemma4Transformer::isKVSharedLayer(int layer_id) const {
  const int first_kv_shared_layer_idx = NUM_LAYERS - NUM_KV_SHARED_LAYERS;
  return layer_id >= first_kv_shared_layer_idx && first_kv_shared_layer_idx > 0;
}

bool Gemma4Transformer::isSlidingAttentionLayer(int layer_id) const {
  if (!layer_types.empty() && layer_id < static_cast<int>(layer_types.size())) {
    return layer_types[layer_id] == "sliding_attention";
  }

  return true;
}

unsigned int Gemma4Transformer::getAttentionHeadDim(int layer_id) const {
  return isSlidingAttentionLayer(layer_id) ? static_cast<unsigned int>(HEAD_DIM)
                                           : GLOBAL_HEAD_DIM;
}

unsigned int Gemma4Transformer::getKVHeadCount(int layer_id) const {
  const bool is_sliding = isSlidingAttentionLayer(layer_id);
  return (is_sliding || !ATTENTION_K_EQ_V) ? NUM_KEY_VALUE_HEADS
                                           : NUM_GLOBAL_KEY_VALUE_HEADS;
}

unsigned int Gemma4Transformer::getKVCacheWidth(int layer_id) const {
  return getAttentionHeadDim(layer_id) * getKVHeadCount(layer_id);
}

void Gemma4Transformer::appendSkipPrefillIfNeeded(
  std::vector<std::string> &props, bool enable_skip) const {
  if (enable_skip && ENABLE_SKIP_PREFILL_OPT) {
    props.emplace_back(withKey("skip_prefill", "true"));
  }
}

json &Gemma4Transformer::sanitizeConfig(json &cfg) {
  if (cfg.contains("text_config") && cfg["text_config"].is_object()) {
    const auto &text_cfg = cfg["text_config"];
    for (auto it = text_cfg.begin(); it != text_cfg.end(); ++it) {
      if (!cfg.contains(it.key())) {
        cfg[it.key()] = it.value();
      }
    }
  }

  if (!cfg.contains("tie_word_embeddings")) {
    cfg["tie_word_embeddings"] = true;
  }

  if (!cfg.contains("head_dim") && cfg.contains("hidden_size") &&
      cfg.contains("num_attention_heads")) {
    cfg["head_dim"] = cfg["hidden_size"].get<unsigned int>() /
                      cfg["num_attention_heads"].get<unsigned int>();
  }

  return cfg;
}

json &Gemma4Transformer::sanitizeGenerationConfig(json &gen_cfg,
                                                  const json &cfg) {
  if (!gen_cfg.contains("eos_token_id")) {
    if (cfg.contains("eos_token_id")) {
      auto eos = cfg["eos_token_id"];
      if (eos.is_number()) {
        gen_cfg["eos_token_id"] =
          std::vector<unsigned int>{eos.get<unsigned int>()};
      } else {
        gen_cfg["eos_token_id"] = eos;
      }
    }
  } else {
    auto eos = gen_cfg["eos_token_id"];
    if (eos.is_number()) {
      gen_cfg["eos_token_id"] =
        std::vector<unsigned int>{eos.get<unsigned int>()};
    }
  }

  return gen_cfg;
}

void Gemma4Transformer::setupParameters(json &cfg, json &generation_cfg,
                                        json &nntr_cfg) {
  Transformer::setupParameters(cfg, generation_cfg, nntr_cfg);

  if (cfg.contains("layer_types")) {
    layer_types = cfg["layer_types"].get<std::vector<std::string>>();
  }

  if (cfg.contains("attn_logit_softcapping") &&
      !cfg["attn_logit_softcapping"].is_null()) {
    ATTN_LOGIT_SOFTCAPPING = cfg["attn_logit_softcapping"].get<float>();
  }
  if (cfg.contains("final_logit_softcapping") &&
      !cfg["final_logit_softcapping"].is_null()) {
    FINAL_LOGIT_SOFTCAPPING = cfg["final_logit_softcapping"].get<float>();
  }

  GLOBAL_HEAD_DIM =
    cfg.contains("global_head_dim") && !cfg["global_head_dim"].is_null()
      ? cfg["global_head_dim"].get<unsigned int>()
      : HEAD_DIM;

  NUM_GLOBAL_KEY_VALUE_HEADS =
    cfg.contains("num_global_key_value_heads") &&
        !cfg["num_global_key_value_heads"].is_null()
      ? cfg["num_global_key_value_heads"].get<unsigned int>()
      : NUM_KEY_VALUE_HEADS;

  ATTENTION_K_EQ_V =
    cfg.contains("attention_k_eq_v") && cfg["attention_k_eq_v"].get<bool>();

  NNTR_THROW_IF(!cfg.contains("hidden_size_per_layer_input") ||
                  cfg["hidden_size_per_layer_input"].is_null() ||
                  cfg["hidden_size_per_layer_input"].get<unsigned int>() == 0,
                std::invalid_argument)
    << "[Gemma4] hidden_size_per_layer_input must be provided and > 0";
  NNTR_THROW_IF(!cfg.contains("vocab_size_per_layer_input") ||
                  cfg["vocab_size_per_layer_input"].is_null() ||
                  cfg["vocab_size_per_layer_input"].get<unsigned int>() == 0,
                std::invalid_argument)
    << "[Gemma4] vocab_size_per_layer_input must be provided and > 0";
  HIDDEN_SIZE_PER_LAYER_INPUT =
    cfg["hidden_size_per_layer_input"].get<unsigned int>();
  VOCAB_SIZE_PER_LAYER_INPUT =
    cfg["vocab_size_per_layer_input"].get<unsigned int>();

  FULL_ATTENTION_ROPE_THETA = ROPE_THETA;
  SLIDING_ATTENTION_ROPE_THETA = ROPE_THETA;
  FULL_ATTENTION_ROPE_TYPE = "default";
  SLIDING_ATTENTION_ROPE_TYPE = "default";
  FULL_ATTENTION_ROPE_PARTIAL_ROTARY_FACTOR = 1.0f;
  SLIDING_ATTENTION_ROPE_PARTIAL_ROTARY_FACTOR = 1.0f;

  NUM_KV_SHARED_LAYERS = cfg.contains("num_kv_shared_layers") &&
                             !cfg["num_kv_shared_layers"].is_null()
                           ? cfg["num_kv_shared_layers"].get<int>()
                           : 0;
  USE_DOUBLE_WIDE_MLP = cfg.contains("use_double_wide_mlp") &&
                        cfg["use_double_wide_mlp"].get<bool>();
  ENABLE_SKIP_PREFILL_OPT =
    nntr_cfg.contains("skip_prefill") && nntr_cfg["skip_prefill"].get<bool>();

  if (cfg.contains("rope_parameters") && cfg["rope_parameters"].is_object()) {
    const auto &rope_params = cfg["rope_parameters"];
    if (rope_params.contains("full_attention") &&
        rope_params["full_attention"].contains("rope_theta")) {
      FULL_ATTENTION_ROPE_THETA =
        rope_params["full_attention"]["rope_theta"].get<unsigned int>();
    }
    if (rope_params.contains("full_attention") &&
        rope_params["full_attention"].contains("rope_type") &&
        !rope_params["full_attention"]["rope_type"].is_null()) {
      FULL_ATTENTION_ROPE_TYPE =
        rope_params["full_attention"]["rope_type"].get<std::string>();
    }
    if (rope_params.contains("full_attention") &&
        rope_params["full_attention"].contains("partial_rotary_factor") &&
        !rope_params["full_attention"]["partial_rotary_factor"].is_null()) {
      FULL_ATTENTION_ROPE_PARTIAL_ROTARY_FACTOR =
        rope_params["full_attention"]["partial_rotary_factor"].get<float>();
    }

    if (rope_params.contains("sliding_attention") &&
        rope_params["sliding_attention"].contains("rope_theta")) {
      SLIDING_ATTENTION_ROPE_THETA =
        rope_params["sliding_attention"]["rope_theta"].get<unsigned int>();
    }

    if (rope_params.contains("sliding_attention") &&
        rope_params["sliding_attention"].contains("rope_type") &&
        !rope_params["sliding_attention"]["rope_type"].is_null()) {
      SLIDING_ATTENTION_ROPE_TYPE =
        rope_params["sliding_attention"]["rope_type"].get<std::string>();
    }
    if (rope_params.contains("sliding_attention") &&
        rope_params["sliding_attention"].contains("partial_rotary_factor") &&
        !rope_params["sliding_attention"]["partial_rotary_factor"].is_null()) {
      SLIDING_ATTENTION_ROPE_PARTIAL_ROTARY_FACTOR =
        rope_params["sliding_attention"]["partial_rotary_factor"].get<float>();
    }
  }

  EMBEDDING_SCALE = std::sqrt(static_cast<float>(DIM));
  EMBEDDING_PER_LAYER_SCALE =
    std::sqrt(static_cast<float>(HIDDEN_SIZE_PER_LAYER_INPUT));
}

std::pair<Tensor, Tensor>
Gemma4Transformer::createGemma4KVCachePlaceholders(const int layer_id,
                                                   unsigned int kv_width) {
  const unsigned int max_timestep = static_cast<unsigned int>(MAX_SEQ_LEN);
#ifdef ENABLE_FP16
  ml::train::TensorDim cache_dim(
    {BATCH_SIZE, 1, max_timestep, kv_width},
    {ml::train::TensorDim::Format::NCHW, ml::train::TensorDim::DataType::FP16});

  Tensor cache_k(cache_dim, "cache_k_l" + std::to_string(layer_id));
  Tensor cache_v(cache_dim, "cache_v_l" + std::to_string(layer_id));
  return {cache_k, cache_v};
#else
  const std::string cache_shape = std::to_string(BATCH_SIZE) +
                                  ":1:" + std::to_string(max_timestep) + ":" +
                                  std::to_string(kv_width);

  LayerHandle cache_k_input(createLayer(
    "input",
    {withKey("name", "cache_k_l" + std::to_string(layer_id)),
     withKey("input_shape", cache_shape), withKey("input_dtype", "UINT16")}));
  LayerHandle cache_v_input(createLayer(
    "input",
    {withKey("name", "cache_v_l" + std::to_string(layer_id)),
     withKey("input_shape", cache_shape), withKey("input_dtype", "UINT16")}));

  return {cache_k_input(Tensor()), cache_v_input(Tensor())};
#endif
}

std::pair<Tensor, Tensor> Gemma4Transformer::constructModel() {

  Tensor x =
    Tensor({1, 1, 1, static_cast<unsigned int>(INIT_SEQ_LEN)}, "input0");

  const std::string embedding_type =
    TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "embedding_layer";

  NNTR_THROW_IF(TIE_WORD_EMBEDDINGS && !EMBEDDING_FILE_NAME.empty(),
                std::invalid_argument)
    << "embedding_file_name requires untied embedding_layer";
  LayerHandle embedding(createLayer(
    embedding_type,
    buildEmbeddingLayerProperties("embedding0", NUM_VOCAB, DIM, EMBEDDING_DTYPE,
                                  EMBEDDING_SCALE, EMBEDDING_FILE_NAME)));
  Tensor h = embedding(x);

  const unsigned int per_layer_total_dim =
    NUM_LAYERS * HIDDEN_SIZE_PER_LAYER_INPUT;

  // try using same low bit precision as fc layers
  LayerHandle per_layer_embedding(createLayer(
    "embedding_layer",
    buildEmbeddingLayerProperties("per_layer_input_embedding",
                                  VOCAB_SIZE_PER_LAYER_INPUT,
                                  per_layer_total_dim, FC_LAYER_DTYPE,
                                  EMBEDDING_PER_LAYER_SCALE, PLE_FILE_NAME)));
  Tensor per_layer_embedding_out = per_layer_embedding(x);

  LayerHandle per_layer_projection(createLayer(
    "fully_connected",
    {withKey("name", "per_layer_input_projection"),
     withKey("unit", std::to_string(per_layer_total_dim)),
     withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));
  Tensor per_layer_projected = per_layer_projection(h);

  float ple_proj_scale = 1.0f / std::sqrt(static_cast<float>(DIM));
  LayerHandle model_proj_scale(createLayer(
    "scalar_multiply",
    {withKey("name", "per_layer_model_proj_scale"), withKey("packed", "false"),
     withKey("multiplier", std::to_string(ple_proj_scale))}));
  Tensor scaled_projection = model_proj_scale(per_layer_projected);

  LayerHandle projection_norm(createLayer(
    "reshaped_rms_norm",
    {
      withKey("name", "per_layer_projection_norm"),
      withKey("epsilon", std::to_string(NORM_EPS)),
      withKey("feature_size", std::to_string(HIDDEN_SIZE_PER_LAYER_INPUT)),
      withKey("packed", "false"),
    }));
  Tensor normalized_projection = projection_norm(scaled_projection);

  LayerHandle per_layer_sum(
    createLayer("addition", {withKey("name", "per_layer_input_sum")}));
  Tensor per_layer_sum_out =
    per_layer_sum({per_layer_embedding_out, normalized_projection});

  // TODO : change per_layer_input_scale to non hard-coded way

  float per_layer_input_scale = std::sqrt(0.5f);

  LayerHandle per_layer_input_scale_layer(
    createLayer("scalar_multiply",
                {
                  withKey("name", "per_layer_input_scale"),
                  withKey("packed", "false"),
                  withKey("multiplier", std::to_string(per_layer_input_scale)),
                }));
  per_layer_input = per_layer_input_scale_layer(per_layer_sum_out);

  layer_k_norms.assign(NUM_LAYERS, Tensor());
  layer_v_norms.assign(NUM_LAYERS, Tensor());
  for (int i = 0; i < NUM_LAYERS; ++i) {
    h = createTransformerDecoderBlock(i, h);
  }

  std::vector<std::string> output_norm_props = {
    withKey("name", "output_norm"),
    withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false")};
  appendSkipPrefillIfNeeded(output_norm_props, true);
  LayerHandle out_norm(createLayer("rms_norm", output_norm_props));
  h = out_norm(h);

  return {x, h};
}

Tensor Gemma4Transformer::createTransformerDecoderBlock(const int layer_id,
                                                        Tensor input) {

  // Gemma4TextRMSNorm scales by `weight` (initialized to ones), which matches
  // NNTrainer `rms_norm` behavior used here.
  const bool is_kv_shared_layer = isKVSharedLayer(layer_id);
  std::vector<std::string> attn_norm_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_attention_norm"),
    withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false")};
  appendSkipPrefillIfNeeded(attn_norm_props, is_kv_shared_layer);
  LayerHandle attn_norm(createLayer("rms_norm", attn_norm_props));
  Tensor normed = attn_norm(input);

  int shared_kv_layer_id = -1;

  const int first_kv_shared_layer_idx = NUM_LAYERS - NUM_KV_SHARED_LAYERS;

  if (is_kv_shared_layer && !layer_types.empty() &&
      first_kv_shared_layer_idx <= static_cast<int>(layer_types.size())) {
    const auto &curr_layer_type = layer_types[layer_id];
    const std::vector<std::string> prev_layers(
      layer_types.begin(), layer_types.begin() + first_kv_shared_layer_idx);
    auto rev_it =
      std::find(prev_layers.rbegin(), prev_layers.rend(), curr_layer_type);
    NNTR_THROW_IF(rev_it == prev_layers.rend(), std::invalid_argument)
      << "[Gemma4] Could not find shared KV source layer for layer " << layer_id
      << " with layer_type=" << curr_layer_type;
    shared_kv_layer_id =
      static_cast<int>(prev_layers.size()) - 1 -
      static_cast<int>(std::distance(prev_layers.rbegin(), rev_it));
  }

  Tensor att_out;
  if (shared_kv_layer_id >= 0) {
    att_out = createSharedAttention(layer_id, shared_kv_layer_id, INIT_SEQ_LEN,
                                    NUM_HEADS, HEAD_DIM, normed);
  } else {
    att_out = createAttention(layer_id, INIT_SEQ_LEN, NUM_HEADS, HEAD_DIM,
                              normed, normed, normed);
  }

  std::vector<std::string> post_attn_norm_props = {
    withKey("name",
            "layer" + std::to_string(layer_id) + "_post_attention_norm"),
    withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false")};
  appendSkipPrefillIfNeeded(post_attn_norm_props, is_kv_shared_layer);
  LayerHandle post_attn_norm(createLayer("rms_norm", post_attn_norm_props));
  Tensor post_normed = post_attn_norm(att_out);

  std::vector<std::string> post_attention_add_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_post_attention")};
  appendSkipPrefillIfNeeded(post_attention_add_props, is_kv_shared_layer);
  LayerHandle post_attention_add(
    createLayer("addition", post_attention_add_props));
  Tensor post_attention = post_attention_add({input, post_normed});

  std::vector<std::string> pre_ffn_norm_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_pre_ffn_norm"),
    withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false")};
  appendSkipPrefillIfNeeded(pre_ffn_norm_props, is_kv_shared_layer);
  LayerHandle pre_ffn_norm(createLayer("rms_norm", pre_ffn_norm_props));
  Tensor pre_ffn = pre_ffn_norm(post_attention);

  Tensor ffn_out = createMlp(layer_id, DIM, INTERMEDIATE_SIZE, pre_ffn);

  std::vector<std::string> post_ffn_norm_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_post_ffn_norm"),
    withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false")};
  appendSkipPrefillIfNeeded(post_ffn_norm_props, is_kv_shared_layer);
  LayerHandle post_ffn_norm(createLayer("rms_norm", post_ffn_norm_props));
  Tensor post_ffn = post_ffn_norm(ffn_out);

  std::vector<std::string> decoder_output_base_props = {withKey(
    "name", "layer" + std::to_string(layer_id) + "_decoder_output_base")};
  appendSkipPrefillIfNeeded(decoder_output_base_props, is_kv_shared_layer);
  LayerHandle decoder_output_base_layer(
    createLayer("addition", decoder_output_base_props));
  Tensor decoder_output_base =
    decoder_output_base_layer({post_attention, post_ffn});

  // Select [B, S, hidden_size_per_layer_input] from packed per-layer input
  // [B, S, num_layers*hidden_size_per_layer_input]
  std::vector<std::string> per_layer_slice_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_per_layer_input"),
    withKey("feature_size", std::to_string(HIDDEN_SIZE_PER_LAYER_INPUT)),
    withKey("layer_index", std::to_string(layer_id))};
  appendSkipPrefillIfNeeded(per_layer_slice_props, is_kv_shared_layer);
  LayerHandle per_layer_slice(
    createLayer("per_layer_slice", per_layer_slice_props));
  Tensor per_layer_input_slice = per_layer_slice(per_layer_input);

  std::vector<std::string> per_layer_input_gate_props = {
    withKey("name",
            "layer" + std::to_string(layer_id) + "_per_layer_input_gate"),
    withKey("unit", std::to_string(HIDDEN_SIZE_PER_LAYER_INPUT)),
    withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
    withKey("weight_dtype", FC_LAYER_DTYPE)};
  appendSkipPrefillIfNeeded(per_layer_input_gate_props, is_kv_shared_layer);
  LayerHandle per_layer_input_gate(
    createLayer("fully_connected", per_layer_input_gate_props));
  Tensor per_layer_input_gate_out = per_layer_input_gate(decoder_output_base);

  std::vector<std::string> per_layer_input_act_props = {
    withKey("name",
            "layer" + std::to_string(layer_id) + "_per_layer_input_act"),
    withKey("activation", "tanh_gelu")};
  appendSkipPrefillIfNeeded(per_layer_input_act_props, is_kv_shared_layer);
  LayerHandle per_layer_input_act(
    createLayer("activation", per_layer_input_act_props));
  Tensor per_layer_input_activated =
    per_layer_input_act(per_layer_input_gate_out);

  std::vector<std::string> per_layer_input_mul_props = {withKey(
    "name", "layer" + std::to_string(layer_id) + "_per_layer_input_mul")};
  appendSkipPrefillIfNeeded(per_layer_input_mul_props, is_kv_shared_layer);
  LayerHandle per_layer_input_mul(
    createLayer("multiply", per_layer_input_mul_props));
  Tensor per_layer_input_multiplied =
    per_layer_input_mul({per_layer_input_activated, per_layer_input_slice});

  std::vector<std::string> per_layer_input_proj_props = {
    withKey("name",
            "layer" + std::to_string(layer_id) + "_per_layer_input_proj"),
    withKey("unit", std::to_string(DIM)), withKey("disable_bias", "true"),
    withKey("weight_initializer", "ones"),
    withKey("weight_dtype", FC_LAYER_DTYPE)};
  appendSkipPrefillIfNeeded(per_layer_input_proj_props, is_kv_shared_layer);
  LayerHandle per_layer_input_proj(
    createLayer("fully_connected", per_layer_input_proj_props));
  Tensor per_layer_input_projected =
    per_layer_input_proj(per_layer_input_multiplied);

  std::vector<std::string> post_per_layer_input_norm_props = {
    withKey("name",
            "layer" + std::to_string(layer_id) + "_post_per_layer_input_norm"),
    withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false")};
  appendSkipPrefillIfNeeded(post_per_layer_input_norm_props,
                            is_kv_shared_layer);
  LayerHandle post_per_layer_input_norm(
    createLayer("rms_norm", post_per_layer_input_norm_props));
  Tensor per_layer_input_normed =
    post_per_layer_input_norm(per_layer_input_projected);

  std::vector<std::string> decoder_output_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_decoder_output")};
  appendSkipPrefillIfNeeded(decoder_output_props, is_kv_shared_layer);
  LayerHandle decoder_output_layer(
    createLayer("addition", decoder_output_props));
  Tensor decoder_output =
    decoder_output_layer({decoder_output_base, per_layer_input_normed});

  std::vector<std::string> layer_scalar_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_layer_scalar"),
    withKey("packed", "false"),
    withKey("use_weight", "true"),
  };
  appendSkipPrefillIfNeeded(layer_scalar_props, is_kv_shared_layer);
  LayerHandle layer_scalar(createLayer("scalar_multiply", layer_scalar_props));

  return layer_scalar(decoder_output);
}

Tensor Gemma4Transformer::createSharedAttention(const int layer_id,
                                                const int shared_kv_layer_id,
                                                int seq_len, int n_heads,
                                                int head_dim, Tensor query) {
  (void)seq_len;
  (void)head_dim;

  const std::string Q = "layer" + std::to_string(layer_id) + "_wq";
  const std::string Q_norm = "layer" + std::to_string(layer_id) + "_q_norm";
  const std::string A = "layer" + std::to_string(layer_id) + "_attention";
  const std::string O = "layer" + std::to_string(layer_id) + "_attention_out";
  const std::string Q_scaled = "layer" + std::to_string(layer_id) + "_q_scaled";

  const bool is_kv_shared_layer = isKVSharedLayer(layer_id);
  const bool is_sliding = isSlidingAttentionLayer(layer_id);

  int curr_head_dim = static_cast<int>(getAttentionHeadDim(layer_id));
  int curr_kv_heads = static_cast<int>(getKVHeadCount(layer_id));

  NNTR_THROW_IF(shared_kv_layer_id < 0 ||
                  shared_kv_layer_id >= static_cast<int>(layer_k_norms.size()),
                std::invalid_argument)
    << "[Gemma4] invalid shared KV source layer " << shared_kv_layer_id;

  // Q layer [B, S, H] -> [B, S, Nq*Dh]
  std::vector<std::string> q_params = {
    withKey("name", Q), withKey("unit", curr_head_dim * n_heads),
    withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
    withKey("weight_dtype", FC_LAYER_DTYPE)};
  appendSkipPrefillIfNeeded(q_params, is_kv_shared_layer);
  LayerHandle wq(createLayer("fully_connected", q_params));
  Tensor q = wq(query);

  // q_norm on per-head projection [B, S, Nq*Dh]
  std::vector<std::string> q_norm_params = {
    withKey("name", Q_norm), withKey("packed", "false"),
    withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("feature_size", std::to_string(curr_head_dim))};
  appendSkipPrefillIfNeeded(q_norm_params, is_kv_shared_layer);
  LayerHandle q_norm(createLayer("reshaped_rms_norm", q_norm_params));
  Tensor q_normed = q_norm(q);

  // Gemma4TextAttention uses scaling=1.0 after q_norm/k_norm.
  // mha_core backend applies 1/sqrt(head_dim) to QK, so pre-scale Q by
  // sqrt(head_dim) to preserve Gemma4 semantics.

  // TODO : fix AVX kernel to not make it divide by 1/sqrt(head_dim) on gemma4
  LayerHandle q_scale(createLayer(
    "scalar_multiply",
    {withKey("name", Q_scaled), withKey("packed", "false"),
     withKey("multiplier",
             std::to_string(std::sqrt(static_cast<float>(curr_head_dim))))}));
  Tensor q_scaled = q_scale(q_normed);

  unsigned int window_size = is_sliding ? SLIDING_WINDOW : UINT_MAX;
  unsigned int rope_theta =
    is_sliding ? SLIDING_ATTENTION_ROPE_THETA : FULL_ATTENTION_ROPE_THETA;

  const std::string &rope_type =
    is_sliding ? SLIDING_ATTENTION_ROPE_TYPE : FULL_ATTENTION_ROPE_TYPE;
  const float rope_partial_rotary_factor =
    is_sliding ? SLIDING_ATTENTION_ROPE_PARTIAL_ROTARY_FACTOR
               : FULL_ATTENTION_ROPE_PARTIAL_ROTARY_FACTOR;

  auto [cache_k, cache_v] =
    createGemma4KVCachePlaceholders(layer_id, getKVCacheWidth(layer_id));

  Tensor shared_k_norm = layer_k_norms[shared_kv_layer_id];
  Tensor shared_v_norm = layer_v_norms[shared_kv_layer_id];
  layer_k_norms[layer_id] = shared_k_norm;
  layer_v_norms[layer_id] = shared_v_norm;

  // Shared attention core receives [Q_norm, shared_K_norm, shared_V_norm].
  std::vector<std::string> a_params = {
    withKey("name", A),
    withKey("num_heads", n_heads),
    withKey("num_heads_kv", curr_kv_heads),
    withKey("max_timestep", std::to_string(MAX_SEQ_LEN)),
    withKey("max_position_embeddings", std::to_string(MAX_POSITION_EMBEDDINGS)),
    withKey("sliding_window", window_size),
    withKey("use_rope", "true"),
    withKey("rope_theta", std::to_string(rope_theta)),
    withKey("rope_scaling_type", rope_type),
    withKey("rope_partial_rotary_factor",
            std::to_string(rope_partial_rotary_factor)),
    withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
    withKey("attn_logit_softcapping", std::to_string(ATTN_LOGIT_SOFTCAPPING)),
    withKey("is_causal", IS_CAUSAL ? "true" : "false")};
  appendSkipPrefillIfNeeded(a_params, is_kv_shared_layer);
  LayerHandle mha(createLayer("mha_core", a_params));
  Tensor a = mha({q_scaled, shared_k_norm, shared_v_norm, cache_k, cache_v});

  // O layer [B, S, Nq*Dh] -> [B, S, H]
  std::vector<std::string> o_params = {withKey("name", O), withKey("unit", DIM),
                                       withKey("disable_bias", "true"),
                                       withKey("weight_initializer", "ones"),
                                       withKey("weight_dtype", FC_LAYER_DTYPE)};
  appendSkipPrefillIfNeeded(o_params, is_kv_shared_layer);
  LayerHandle wo(createLayer("fully_connected", o_params));

  return wo(a);
}

Tensor Gemma4Transformer::createAttention(const int layer_id, int seq_len,
                                          int n_heads, int head_dim,
                                          Tensor query, Tensor key,
                                          Tensor value) {
  (void)seq_len;
  (void)head_dim;

  const std::string Q = "layer" + std::to_string(layer_id) + "_wq";
  const std::string Q_norm = "layer" + std::to_string(layer_id) + "_q_norm";
  const std::string K = "layer" + std::to_string(layer_id) + "_wk";
  const std::string K_norm = "layer" + std::to_string(layer_id) + "_k_norm";
  const std::string V = "layer" + std::to_string(layer_id) + "_wv";
  const std::string V_norm = "layer" + std::to_string(layer_id) + "_v_norm";
  const std::string A = "layer" + std::to_string(layer_id) + "_attention";
  const std::string O = "layer" + std::to_string(layer_id) + "_attention_out";
  const std::string Q_scaled = "layer" + std::to_string(layer_id) + "_q_scaled";

  const bool is_sliding = isSlidingAttentionLayer(layer_id);
  const bool is_kv_shared_layer = isKVSharedLayer(layer_id);
  const int curr_head_dim = static_cast<int>(getAttentionHeadDim(layer_id));
  const int curr_kv_heads = static_cast<int>(getKVHeadCount(layer_id));

  // Q layer [B, S, H] -> [B, S, Nq*Dh]
  std::vector<std::string> q_params = {
    withKey("name", Q), withKey("unit", curr_head_dim * n_heads),
    withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
    withKey("weight_dtype", FC_LAYER_DTYPE)};
  appendSkipPrefillIfNeeded(q_params, is_kv_shared_layer);
  LayerHandle wq(createLayer("fully_connected", q_params));
  Tensor q = wq(query);

  // K layer [B, S, H] -> [B, S, Nk*Dh]
  std::vector<std::string> k_params = {
    withKey("name", K), withKey("unit", curr_head_dim * curr_kv_heads),
    withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
    withKey("weight_dtype", FC_LAYER_DTYPE)};
  appendSkipPrefillIfNeeded(k_params, is_kv_shared_layer);
  LayerHandle wk(createLayer("fully_connected", k_params));
  Tensor k = wk(key);

  // V layer [B, S, H] -> [B, S, Nk*Dh]
  std::vector<std::string> v_params = {
    withKey("name", V), withKey("unit", curr_head_dim * curr_kv_heads),
    withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
    withKey("weight_dtype", FC_LAYER_DTYPE)};
  appendSkipPrefillIfNeeded(v_params, is_kv_shared_layer);
  LayerHandle wv(createLayer("fully_connected", v_params));
  Tensor v = wv(value);

  // q_norm on per-head projection [B, S, Nq*Dh]
  std::vector<std::string> q_norm_params = {
    withKey("name", Q_norm), withKey("packed", "false"),
    withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("feature_size", std::to_string(curr_head_dim))};
  appendSkipPrefillIfNeeded(q_norm_params, is_kv_shared_layer);
  LayerHandle q_norm(createLayer("reshaped_rms_norm", q_norm_params));
  Tensor q_normed = q_norm(q);

  // Gemma4TextAttention uses scaling=1.0 after q_norm/k_norm.
  // mha_core backend applies 1/sqrt(head_dim) to QK, so pre-scale Q by
  // sqrt(head_dim) to preserve Gemma4 semantics.
  LayerHandle q_scale(createLayer(
    "scalar_multiply",
    {withKey("name", Q_scaled), withKey("packed", "false"),
     withKey("multiplier",
             std::to_string(std::sqrt(static_cast<float>(curr_head_dim))))}));
  Tensor q_scaled = q_scale(q_normed);

  // k_norm on per-head projection [B, S, Nk*Dh]
  std::vector<std::string> k_norm_params = {
    withKey("name", K_norm), withKey("packed", "false"),
    withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("feature_size", std::to_string(curr_head_dim))};
  appendSkipPrefillIfNeeded(k_norm_params, is_kv_shared_layer);
  LayerHandle k_norm(createLayer("reshaped_rms_norm", k_norm_params));
  Tensor k_normed = k_norm(k);

  // v_norm on per-head projection [B, S, Nk*Dh] (no learned scale)
  std::vector<std::string> v_norm_params = {
    withKey("name", V_norm), withKey("packed", "false"),
    withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("feature_size", std::to_string(curr_head_dim))};
  v_norm_params.push_back(withKey("use_gamma", "false"));
  appendSkipPrefillIfNeeded(v_norm_params, is_kv_shared_layer);
  LayerHandle v_norm(createLayer("reshaped_rms_norm", v_norm_params));
  Tensor v_normed = v_norm(v);

  if (layer_id >= static_cast<int>(layer_k_norms.size())) {
    layer_k_norms.resize(layer_id + 1);
    layer_v_norms.resize(layer_id + 1);
  }
  layer_k_norms[layer_id] = k_normed;
  layer_v_norms[layer_id] = v_normed;

  unsigned int window_size = is_sliding ? SLIDING_WINDOW : UINT_MAX;
  unsigned int rope_theta =
    is_sliding ? SLIDING_ATTENTION_ROPE_THETA : FULL_ATTENTION_ROPE_THETA;
  const std::string &rope_type =
    is_sliding ? SLIDING_ATTENTION_ROPE_TYPE : FULL_ATTENTION_ROPE_TYPE;
  const float rope_partial_rotary_factor =
    is_sliding ? SLIDING_ATTENTION_ROPE_PARTIAL_ROTARY_FACTOR
               : FULL_ATTENTION_ROPE_PARTIAL_ROTARY_FACTOR;

  auto [cache_k, cache_v] =
    createGemma4KVCachePlaceholders(layer_id, getKVCacheWidth(layer_id));

  // Attention core receives [Q_norm, K_norm, V_norm].
  std::vector<std::string> a_params = {
    withKey("name", A),
    withKey("num_heads", n_heads),
    withKey("num_heads_kv", curr_kv_heads),
    withKey("max_timestep", std::to_string(MAX_SEQ_LEN)),
    withKey("max_position_embeddings", std::to_string(MAX_POSITION_EMBEDDINGS)),
    withKey("sliding_window", window_size),
    withKey("use_rope", "true"),
    withKey("rope_theta", std::to_string(rope_theta)),
    withKey("rope_scaling_type", rope_type),
    withKey("rope_partial_rotary_factor",
            std::to_string(rope_partial_rotary_factor)),
    withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
    withKey("attn_logit_softcapping", std::to_string(ATTN_LOGIT_SOFTCAPPING)),
    withKey("is_causal", IS_CAUSAL ? "true" : "false")};
  appendSkipPrefillIfNeeded(a_params, is_kv_shared_layer);
  LayerHandle mha(createLayer("mha_core", a_params));
  Tensor a = mha({q_scaled, k_normed, v_normed, cache_k, cache_v});

  // O layer [B, S, Nq*Dh] -> [B, S, H]
  std::vector<std::string> o_params = {withKey("name", O), withKey("unit", DIM),
                                       withKey("disable_bias", "true"),
                                       withKey("weight_initializer", "ones"),
                                       withKey("weight_dtype", FC_LAYER_DTYPE)};
  appendSkipPrefillIfNeeded(o_params, is_kv_shared_layer);
  LayerHandle wo(createLayer("fully_connected", o_params));

  return wo(a);
}

Tensor Gemma4Transformer::createMlp(const int layer_id, int dim, int hidden_dim,
                                    Tensor input) {
  const bool is_kv_shared_layer = isKVSharedLayer(layer_id);
  const int curr_hidden_dim =
    hidden_dim * ((USE_DOUBLE_WIDE_MLP && is_kv_shared_layer) ? 2 : 1);

  std::vector<std::string> ffn_gate_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_ffn_gate"),
    withKey("unit", curr_hidden_dim), withKey("disable_bias", "true"),
    withKey("weight_initializer", "ones"),
    withKey("weight_dtype", FC_LAYER_DTYPE)};
  appendSkipPrefillIfNeeded(ffn_gate_props, is_kv_shared_layer);
  LayerHandle ffn_gate(createLayer("fully_connected", ffn_gate_props));
  Tensor gate = ffn_gate(input);

  std::vector<std::string> ffn_gate_gelu_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_ffn_gate_gelu"),
    withKey("activation", "tanh_gelu")};
  appendSkipPrefillIfNeeded(ffn_gate_gelu_props, is_kv_shared_layer);
  LayerHandle ffn_gate_gelu(createLayer("activation", ffn_gate_gelu_props));
  Tensor gate_gelu = ffn_gate_gelu(gate);

  std::vector<std::string> ffn_up_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_ffn_up"),
    withKey("unit", curr_hidden_dim), withKey("disable_bias", "true"),
    withKey("weight_initializer", "ones"),
    withKey("weight_dtype", FC_LAYER_DTYPE)};
  appendSkipPrefillIfNeeded(ffn_up_props, is_kv_shared_layer);
  LayerHandle ffn_up(createLayer("fully_connected", ffn_up_props));
  Tensor up = ffn_up(input);

  std::vector<std::string> ffn_geglu_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_ffn_geglu")};
  appendSkipPrefillIfNeeded(ffn_geglu_props, is_kv_shared_layer);
  LayerHandle ffn_geglu(createLayer("multiply", ffn_geglu_props));
  Tensor geglu = ffn_geglu({gate_gelu, up});

  std::vector<std::string> ffn_down_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
    withKey("unit", dim), withKey("disable_bias", "true"),
    withKey("weight_initializer", "ones"),
    withKey("weight_dtype", FC_LAYER_DTYPE)};
  appendSkipPrefillIfNeeded(ffn_down_props, is_kv_shared_layer);
  LayerHandle ffn_down(createLayer("fully_connected", ffn_down_props));

  return ffn_down(geglu);
}

void Gemma4Transformer::registerCustomLayers() {
  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

  auto tryRegister = [&](auto factory_fn) {
    try {
      app_context->registerFactory(factory_fn);
    } catch (std::invalid_argument &e) {
      std::cerr << "failed to register factory, reason: " << e.what()
                << std::endl;
    }
  };

  tryRegister(nntrainer::createLayer<causallm::ReshapedRMSNormLayer>);
  tryRegister(nntrainer::createLayer<causallm::PerLayerSliceLayer>);
  tryRegister(nntrainer::createLayer<causallm::ScalarMultiplyLayer>);
  tryRegister(nntrainer::createLayer<causallm::LogitSoftCappingLayer>);
}

void Gemma4CausalLM::registerCustomLayers() {
  CausalLM::registerCustomLayers();
  Gemma4Transformer::registerCustomLayers();
}

std::pair<Tensor, Tensor> Gemma4CausalLM::constructModel() {
  auto [x, h] = Gemma4Transformer::constructModel();

  // create lm_head layer (using fully_connected option)
  const std::string lmhead_type =
    TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "lm_head";

  // add lmhead
  std::vector<std::string> lmhead_prop = {
    withKey("name", "output_of_causallm"),
    withKey("unit", NUM_VOCAB),
    withKey("disable_bias", "true"),
    withKey("weight_dtype", LMHEAD_DTYPE),
  };
  appendSkipPrefillIfNeeded(lmhead_prop, true);

  if (TIE_WORD_EMBEDDINGS)
    lmhead_prop.emplace_back(withKey("shared_from", "embedding0"));

  LayerHandle lmhead(createLayer(lmhead_type, lmhead_prop));
  Tensor y = lmhead(h);

  if (FINAL_LOGIT_SOFTCAPPING > 0.0f) {
    std::vector<std::string> final_softcap_props = {
      withKey("name", "output_of_causallm_softcapped"),
      withKey("activation_type", "tanh"), withKey("apply_rows", "1"),
      withKey("softcap_value", std::to_string(FINAL_LOGIT_SOFTCAPPING))};
    appendSkipPrefillIfNeeded(final_softcap_props, true);
    LayerHandle final_softcap(
      createLayer("logit_softcapping", final_softcap_props));
    y = final_softcap(y);
  }

  return {x, y};
}

void Gemma4CausalLM::allocateAndBindKVCache() {
  if (!kv_cache.isAllocated()) {
#ifdef ENABLE_FP16
    const auto cache_dtype = ml::train::TensorDim::DataType::FP16;
#else
    const auto cache_dtype = ml::train::TensorDim::DataType::UINT16;
#endif
    std::vector<unsigned int> kv_widths;
    kv_widths.reserve(static_cast<size_t>(NUM_LAYERS));
    for (int i = 0; i < NUM_LAYERS; ++i) {
      kv_widths.push_back(getKVCacheWidth(i));
    }

    kv_cache.allocate(static_cast<unsigned int>(NUM_LAYERS), BATCH_SIZE,
                      static_cast<unsigned int>(MAX_SEQ_LEN), kv_widths,
                      cache_dtype);
    kv_cache_bound = false;
  }

  if (kv_cache_bound)
    return;

  for (int i = 0; i < NUM_LAYERS; ++i) {
    auto &kc = kv_cache.getKeyCache(i);
    auto &vc = kv_cache.getValueCache(i);

    auto find_cache_placeholder = [this](const std::string &base_name) {
      for (const auto &suffix : {":0", ":input0", ":out0", ""}) {
        auto *tensor = model->getTensor(base_name + suffix);
        if (tensor != nullptr)
          return tensor;
      }
      return static_cast<nntrainer::Tensor *>(nullptr);
    };

    auto *kp =
      model->getTensor("layer" + std::to_string(i) + "_attention:input3");
    auto *vp =
      model->getTensor("layer" + std::to_string(i) + "_attention:input4");
    if (kp == nullptr)
      kp = find_cache_placeholder("cache_k_l" + std::to_string(i));
    if (vp == nullptr)
      vp = find_cache_placeholder("cache_v_l" + std::to_string(i));
    NNTR_THROW_IF(kp == nullptr || vp == nullptr, std::runtime_error)
      << "Gemma4 allocateAndBindKVCache: cache_k_l" << i << " / cache_v_l" << i
      << " input placeholder not found in compiled graph";
    NNTR_THROW_IF(kp->getDataType() != kc.getDataType() ||
                    vp->getDataType() != vc.getDataType(),
                  std::runtime_error)
      << "Gemma4 allocateAndBindKVCache: cache placeholder dtype mismatch for "
         "layer "
      << i;
    NNTR_THROW_IF(kp->getDim() != kc.getDim() || vp->getDim() != vc.getDim(),
                  std::runtime_error)
      << "Gemma4 allocateAndBindKVCache: cache placeholder shape mismatch for "
         "layer "
      << i;

    kp->setData(kc.getMemoryData(), kc.getOffset(), false);
    vp->setData(vc.getMemoryData(), vc.getOffset(), false);
  }

  kv_cache_bound = true;
}

} // namespace causallm
