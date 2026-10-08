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

#include "gemma4_causallm.h"

#include <algorithm>
#include <cmath>

#include <app_context.h>
#include <dense_ffn_layer.h>
#include <engine.h>
#include <lfm2_causallm.h>
#include <lfm2_moe_layer.h>
#include <llm_util.hpp>
#include <logit_softcapping.h>
#include <model.h>
#include <per_layer_slice.h>
#include <qkv_layer.h>
#include <reshaped_rms_norm.h>
#include <residual_add.h>
#include <scalar_multiply.h>

namespace causallm {

namespace {
/** @brief @a engine for @a layer_id when the layer is in @a ids or @a ids is
 *  empty, "cpu" otherwise -- the moe_htp_layers rule, and Lfm2CausalLM's
 *  projEngine. */
std::string engineFor(const std::string &engine, const std::set<int> &ids,
                      int layer_id) {
  return (ids.empty() || ids.count(layer_id)) ? engine : "cpu";
}
} // namespace

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

  // 0 (the MoE checkpoints) means no per-layer input path at all.
  HIDDEN_SIZE_PER_LAYER_INPUT = cfg.value("hidden_size_per_layer_input", 0u);
  VOCAB_SIZE_PER_LAYER_INPUT = cfg.value("vocab_size_per_layer_input", 0u);
  NNTR_THROW_IF(HIDDEN_SIZE_PER_LAYER_INPUT != 0 &&
                  VOCAB_SIZE_PER_LAYER_INPUT == 0,
                std::invalid_argument)
    << "[Gemma4] vocab_size_per_layer_input must be > 0 when "
       "hidden_size_per_layer_input is";

  ENABLE_MOE_BLOCK = cfg.value("enable_moe_block", false);
  if (ENABLE_MOE_BLOCK) {
    NUM_EXPERTS = cfg.at("num_experts").get<unsigned int>();
    NUM_EXPERTS_PER_TOK = cfg.at("top_k_experts").get<unsigned int>();
    MOE_INTERMEDIATE_SIZE = cfg.at("moe_intermediate_size").get<unsigned int>();
    // Same keys and defaults as Lfm2MoeCausalLM::setupParameters.
    MOE_LAYER_DTYPE = nntr_cfg.value("moe_layer_dtype", FC_LAYER_DTYPE);
    MOE_ENGINE = nntr_cfg.value("moe_engine", std::string("cpu"));
    MOE_HTP_LAYERS =
      parseLayerIdList(nntr_cfg.value("moe_htp_layers", std::string("")));
    MOE_CACHE_EXPERTS = nntr_cfg.value("moe_cache_experts", 0u);
  }

  ATTN_PROJ_ENGINE = nntr_cfg.value("attn_proj_engine", std::string("cpu"));
  LMHEAD_ENGINE = nntr_cfg.value("lmhead_engine", std::string("cpu"));
  ATTN_PROJ_HTP_LAYERS =
    parseLayerIdList(nntr_cfg.value("attn_proj_htp_layers", std::string("")));
  FFN_ENGINE = nntr_cfg.value("dense_ffn_engine", std::string("cpu"));
  FFN_HTP_LAYERS =
    parseLayerIdList(nntr_cfg.value("dense_ffn_htp_layers", std::string("")));

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

  if (HIDDEN_SIZE_PER_LAYER_INPUT != 0)
    constructPerLayerInput(x, h);

  layer_k_norms.assign(NUM_LAYERS, Tensor());
  layer_v_norms.assign(NUM_LAYERS, Tensor());
  for (int i = 0; i < NUM_LAYERS; ++i) {
    h = createTransformerDecoderBlock(i, h);
  }

  if (FOLD_OUTPUT_NORM)
    return {x, h};

  std::vector<std::string> output_norm_props = {
    withKey("name", "output_norm"),
    withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false")};
  appendSkipPrefillIfNeeded(output_norm_props, true);
  LayerHandle out_norm(createLayer("rms_norm", output_norm_props));
  h = out_norm(h);

  return {x, h};
}

void Gemma4Transformer::constructPerLayerInput(Tensor x, Tensor h) {
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
}

Tensor Gemma4Transformer::createTransformerDecoderBlock(const int layer_id,
                                                        Tensor input) {

  // Gemma4TextRMSNorm scales by `weight` (initialized to ones), which matches
  // NNTrainer `rms_norm` behavior used here.
  const bool is_kv_shared_layer = isKVSharedLayer(layer_id);
  // The norms around the projections ride the fused layers (qkv_layer's
  // in_norm, dense_ffn's in_norm / out_norm, the MoE layer's three), where
  // one accelerator call covers norm and matmul (doc 57 section 5 step 4).
  // A KV-shared layer keeps the separate norm layers: its attention and
  // MLP are the older layers, which carry skip_prefill.
  Tensor normed = input;
  if (is_kv_shared_layer) {
    std::vector<std::string> attn_norm_props = {
      withKey("name", "layer" + std::to_string(layer_id) + "_attention_norm"),
      withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false")};
    appendSkipPrefillIfNeeded(attn_norm_props, is_kv_shared_layer);
    LayerHandle attn_norm(createLayer("rms_norm", attn_norm_props));
    normed = attn_norm(input);
  }

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

  // input + post_attention_norm(att_out): the norm's gamma lives on the
  // add, so the two are one accelerator call (doc 57 section 5 step 4).
  // The layer keeps the norm's name: the weight file's order is unchanged.
  std::vector<std::string> post_attn_props = {
    withKey("name",
            "layer" + std::to_string(layer_id) + "_post_attention_norm"),
    withKey("in_norm", "true"), withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("engine",
            engineFor(ATTN_PROJ_ENGINE, ATTN_PROJ_HTP_LAYERS, layer_id))};
  appendSkipPrefillIfNeeded(post_attn_props, is_kv_shared_layer);
  LayerHandle post_attention_add(createLayer("residual_add", post_attn_props));
  Tensor post_attention = post_attention_add({input, att_out});

  Tensor pre_ffn = post_attention;
  if (is_kv_shared_layer) {
    std::vector<std::string> pre_ffn_norm_props = {
      withKey("name", "layer" + std::to_string(layer_id) + "_pre_ffn_norm"),
      withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false")};
    appendSkipPrefillIfNeeded(pre_ffn_norm_props, is_kv_shared_layer);
    LayerHandle pre_ffn_norm(createLayer("rms_norm", pre_ffn_norm_props));
    pre_ffn = pre_ffn_norm(post_attention);
  }

  Tensor ffn_out = createMlp(layer_id, DIM, INTERMEDIATE_SIZE, pre_ffn);

  // post_attention + post_ffn_norm(ffn [+ moe]), then the layer scalar: one
  // residual_add, so the sum, the norm, the add and the scalar are one
  // accelerator call. With a per-layer input the scalar comes after that
  // path instead and stays its own layer.
  const bool fold_scalar = HIDDEN_SIZE_PER_LAYER_INPUT == 0;
  std::vector<Tensor> ffn_terms = {post_attention, ffn_out};
  if (ENABLE_MOE_BLOCK) {
    // Gemma4TextDecoderLayer: norm_1(mlp) + norm_2(experts(norm(residual)))
    // with the router reading its own norm of the residual (doc 55 §6.2;
    // its scale and hidden^-0.5 are folded into that norm's gamma by the
    // converter). norm_1 is the dense layer's out_norm; the MoE layer
    // holds the other three (in_norm, router_norm, out_norm).
    ffn_terms.push_back(createMoe(layer_id, post_attention));
  }
  std::vector<std::string> post_ffn_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_post_ffn_norm"),
    withKey("in_norm", "true"), withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("use_weight", fold_scalar ? "true" : "false"),
    withKey("engine", engineFor(FFN_ENGINE, FFN_HTP_LAYERS, layer_id))};
  appendSkipPrefillIfNeeded(post_ffn_props, is_kv_shared_layer);
  LayerHandle post_ffn_add(createLayer("residual_add", post_ffn_props));
  Tensor decoder_output_base = post_ffn_add(ffn_terms);

  if (fold_scalar)
    return decoder_output_base;

  std::vector<std::string> layer_scalar_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_layer_scalar"),
    withKey("packed", "false"),
    withKey("use_weight", "true"),
  };
  appendSkipPrefillIfNeeded(layer_scalar_props, is_kv_shared_layer);
  LayerHandle layer_scalar(createLayer("scalar_multiply", layer_scalar_props));

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

  return layer_scalar(decoder_output);
}

Tensor Gemma4Transformer::createMoe(const int layer_id, Tensor input) {
  const std::string engine =
    (MOE_HTP_LAYERS.empty() || MOE_HTP_LAYERS.count(layer_id)) ? MOE_ENGINE
                                                               : "cpu";
  std::vector<std::string> props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_sparse_moe"),
    withKey("unit", MOE_INTERMEDIATE_SIZE),
    withKey("num_experts", NUM_EXPERTS),
    withKey("num_experts_per_token", NUM_EXPERTS_PER_TOK),
    withKey("moe_activation", "tanh_gelu"),
    withKey("router_type", "softmax_scale"),
    withKey("in_norm", "true"),
    withKey("router_norm", "true"),
    withKey("out_norm", "true"),
    withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("weight_dtype", MOE_LAYER_DTYPE),
    withKey("engine", engine)};
  if (MOE_CACHE_EXPERTS != 0)
    props.push_back(withKey("cache_experts", MOE_CACHE_EXPERTS));
  appendSkipPrefillIfNeeded(props, isKVSharedLayer(layer_id));
  LayerHandle moe(createLayer("lfm2_moe", props));
  return moe(input);
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
    withKey("name", Q),
    withKey("unit", curr_head_dim * n_heads),
    withKey("disable_bias", "true"),
    withKey("weight_initializer", "ones"),
    withKey("weight_dtype", FC_LAYER_DTYPE),
    withKey("engine",
            engineFor(ATTN_PROJ_ENGINE, ATTN_PROJ_HTP_LAYERS, layer_id))};
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
  LayerHandle mha(createAttentionCore(a_params));
  Tensor a = mha({q_scaled, shared_k_norm, shared_v_norm, cache_k, cache_v});

  // O layer [B, S, Nq*Dh] -> [B, S, H]
  std::vector<std::string> o_params = {
    withKey("name", O),
    withKey("unit", DIM),
    withKey("disable_bias", "true"),
    withKey("weight_initializer", "ones"),
    withKey("weight_dtype", FC_LAYER_DTYPE),
    withKey("engine",
            engineFor(ATTN_PROJ_ENGINE, ATTN_PROJ_HTP_LAYERS, layer_id))};
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

  const std::string QKV = "layer" + std::to_string(layer_id) + "_qkv";
  const std::string A = "layer" + std::to_string(layer_id) + "_attention";
  const std::string O = "layer" + std::to_string(layer_id) + "_attention_out";

  const bool is_sliding = isSlidingAttentionLayer(layer_id);
  const bool is_kv_shared_layer = isKVSharedLayer(layer_id);
  const int curr_head_dim = static_cast<int>(getAttentionHeadDim(layer_id));
  const int curr_kv_heads = static_cast<int>(getKVHeadCount(layer_id));

  // q, k, v and their per-head norms as one layer (qkv_layer, doc 51
  // section 2.21): the three projections share one activation, so an
  // accelerator takes them as one call instead of three that each quantize
  // and ship the same rows; on the CPU it is the same GEMMs and the same
  // norm kernel. The weights stay in the file's order (q, q_norm, k,
  // k_norm, v), so a model file is read identically. Folded in with them:
  // the gamma-less v_norm, and the sqrt(head_dim) on q that cancels the
  // attention core's 1/sqrt(head_dim) (this attention's scaling is 1.0).
  // attention_k_eq_v: a full-attention layer has no v_proj and V is the
  // raw K projection (before k_norm and RoPE), v_norm'ed like any V.
  // (the one caller passes the same normed tensor as query, key and value)
  (void)key;
  (void)value;
  unsigned int rope_theta =
    is_sliding ? SLIDING_ATTENTION_ROPE_THETA : FULL_ATTENTION_ROPE_THETA;
  const std::string &rope_type =
    is_sliding ? SLIDING_ATTENTION_ROPE_TYPE : FULL_ATTENTION_ROPE_TYPE;
  const float rope_partial_rotary_factor =
    is_sliding ? SLIDING_ATTENTION_ROPE_PARTIAL_ROTARY_FACTOR
               : FULL_ATTENTION_ROPE_PARTIAL_ROTARY_FACTOR;
  // RoPE too (doc 57 section 5 step 4): q and k leave the projection call
  // rotated, so the attention core only converts k into its cache.
  std::vector<std::string> qkv_params = {
    withKey("name", QKV),
    withKey("q_unit", curr_head_dim * n_heads),
    withKey("k_unit", curr_head_dim * curr_kv_heads),
    withKey("v_unit", curr_head_dim * curr_kv_heads),
    withKey("feature_size", std::to_string(curr_head_dim)),
    withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("v_norm", "true"),
    withKey("q_scale",
            std::to_string(std::sqrt(static_cast<float>(curr_head_dim)))),
    withKey("in_norm", "true"),
    withKey("rope", "true"),
    withKey("rope_theta", std::to_string(rope_theta)),
    withKey("rope_scaling_type", rope_type),
    withKey("rope_partial_rotary_factor",
            std::to_string(rope_partial_rotary_factor)),
    withKey("max_timestep", std::to_string(MAX_SEQ_LEN)),
    withKey("weight_dtype", FC_LAYER_DTYPE),
    withKey("engine",
            engineFor(ATTN_PROJ_ENGINE, ATTN_PROJ_HTP_LAYERS, layer_id))};
  if (ATTENTION_K_EQ_V && !is_sliding)
    qkv_params.push_back(withKey("v_from_k", "true"));
  appendSkipPrefillIfNeeded(qkv_params, is_kv_shared_layer);
  LayerHandle qkv(createLayer("qkv_layer", qkv_params));
  Tensor qkv_out = qkv(query);
  Tensor q_scaled = qkv_out.output(0);
  Tensor k_normed = qkv_out.output(1);
  Tensor v_normed = qkv_out.output(2);

  if (layer_id >= static_cast<int>(layer_k_norms.size())) {
    layer_k_norms.resize(layer_id + 1);
    layer_v_norms.resize(layer_id + 1);
  }
  layer_k_norms[layer_id] = k_normed;
  layer_v_norms[layer_id] = v_normed;

  unsigned int window_size = is_sliding ? SLIDING_WINDOW : UINT_MAX;

  auto [cache_k, cache_v] =
    createGemma4KVCachePlaceholders(layer_id, getKVCacheWidth(layer_id));

  // Attention core receives [Q_norm, K_norm, V_norm], already rotated.
  std::vector<std::string> a_params = {
    withKey("name", A),
    withKey("num_heads", n_heads),
    withKey("num_heads_kv", curr_kv_heads),
    withKey("max_timestep", std::to_string(MAX_SEQ_LEN)),
    withKey("max_position_embeddings", std::to_string(MAX_POSITION_EMBEDDINGS)),
    withKey("sliding_window", window_size),
    withKey("use_rope", "false"),
    withKey("rope_theta", std::to_string(rope_theta)),
    withKey("rope_scaling_type", rope_type),
    withKey("rope_partial_rotary_factor",
            std::to_string(rope_partial_rotary_factor)),
    withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
    withKey("attn_logit_softcapping", std::to_string(ATTN_LOGIT_SOFTCAPPING)),
    withKey("is_causal", IS_CAUSAL ? "true" : "false")};
  appendSkipPrefillIfNeeded(a_params, is_kv_shared_layer);
  LayerHandle mha(createAttentionCore(a_params));
  Tensor a = mha({q_scaled, k_normed, v_normed, cache_k, cache_v});

  // O layer [B, S, Nq*Dh] -> [B, S, H]
  std::vector<std::string> o_params = {
    withKey("name", O),
    withKey("unit", DIM),
    withKey("disable_bias", "true"),
    withKey("weight_initializer", "ones"),
    withKey("weight_dtype", FC_LAYER_DTYPE),
    withKey("engine",
            engineFor(ATTN_PROJ_ENGINE, ATTN_PROJ_HTP_LAYERS, layer_id))};
  appendSkipPrefillIfNeeded(o_params, is_kv_shared_layer);
  LayerHandle wo(createLayer("fully_connected", o_params));

  return wo(a);
}

Tensor Gemma4Transformer::createMlp(const int layer_id, int dim, int hidden_dim,
                                    Tensor input) {
  const bool is_kv_shared_layer = isKVSharedLayer(layer_id);
  const int curr_hidden_dim =
    hidden_dim * ((USE_DOUBLE_WIDE_MLP && is_kv_shared_layer) ? 2 : 1);
  const std::string ffn_engine =
    engineFor(FFN_ENGINE, FFN_HTP_LAYERS, layer_id);

  // gate, up, GeGLU and down as one layer (dense_ffn, doc 51), so an
  // accelerator takes the block in one call with the GeGLU in its
  // epilogue; the same three weights in the file's order (gate, up, down),
  // so a model file is read identically, and on the CPU the same GEMMs.
  // ponytail: a KV-shared layer keeps the three layers below, which carry
  // skip_prefill; dense_ffn does not, and no model here has such a layer.
  if (!is_kv_shared_layer) {
    LayerHandle ffn(createLayer(
      "dense_ffn",
      {withKey("name", "layer" + std::to_string(layer_id) + "_ffn"),
       withKey("unit", curr_hidden_dim), withKey("gate_first", "true"),
       withKey("glu_activation", "tanh_gelu"), withKey("in_norm", "true"),
       withKey("out_norm", ENABLE_MOE_BLOCK ? "true" : "false"),
       withKey("epsilon", std::to_string(NORM_EPS)),
       withKey("weight_dtype", FC_LAYER_DTYPE), withKey("engine", ffn_engine)}));
    return ffn(input);
  }


  std::vector<std::string> ffn_gate_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_ffn_gate"),
    withKey("unit", curr_hidden_dim),
    withKey("disable_bias", "true"),
    withKey("weight_initializer", "ones"),
    withKey("weight_dtype", FC_LAYER_DTYPE),
    withKey("engine", ffn_engine)};
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
    withKey("unit", curr_hidden_dim),
    withKey("disable_bias", "true"),
    withKey("weight_initializer", "ones"),
    withKey("weight_dtype", FC_LAYER_DTYPE),
    withKey("engine", ffn_engine)};
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
    withKey("unit", dim),
    withKey("disable_bias", "true"),
    withKey("weight_initializer", "ones"),
    withKey("weight_dtype", FC_LAYER_DTYPE),
    withKey("engine", ffn_engine)};
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
  // the fused projections (Transformer::registerCustomLayers registers them
  // too; this model's factory may be the only one called)
  tryRegister(nntrainer::createLayer<causallm::QKVLayer>);
  tryRegister(nntrainer::createLayer<causallm::DenseFfnLayer>);
  tryRegister(nntrainer::createLayer<causallm::ResidualAddLayer>);
  if (ENABLE_MOE_BLOCK)
    tryRegister(nntrainer::createLayer<causallm::Lfm2MoELayer>);
}

void Gemma4CausalLM::registerCustomLayers() {
  CausalLM::registerCustomLayers();
  Gemma4Transformer::registerCustomLayers();
}

std::pair<Tensor, Tensor> Gemma4CausalLM::constructModel() {
  // A tied head takes the final norm and softcap (the gamma is read where
  // the output_norm layer's was, so a model file reads as before).
  FOLD_OUTPUT_NORM = TIE_WORD_EMBEDDINGS;
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

  if (TIE_WORD_EMBEDDINGS) {
    lmhead_prop.emplace_back(withKey("shared_from", "embedding0"));
    lmhead_prop.emplace_back(withKey("in_norm", "true"));
    lmhead_prop.emplace_back(withKey("epsilon", std::to_string(NORM_EPS)));
    lmhead_prop.emplace_back(withKey(
      "softcap", std::to_string(std::max(0.0f, FINAL_LOGIT_SOFTCAPPING))));
    lmhead_prop.emplace_back(withKey("engine", LMHEAD_ENGINE));
  }

  LayerHandle lmhead(createLayer(lmhead_type, lmhead_prop));
  Tensor y = lmhead(h);

  if (!TIE_WORD_EMBEDDINGS && FINAL_LOGIT_SOFTCAPPING > 0.0f) {
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

    installKVCacheSharedAllocator();
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
