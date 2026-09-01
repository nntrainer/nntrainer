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
#include <engine.h>
#include <llm_util.hpp>

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
#include <cuda_context.h>
#endif
#include <model.h>
#include <per_layer_slice.h>
#include <reshaped_rms_norm.h>

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

int Gemma4Transformer::getSharedKVSourceLayer(int layer_id) const {
  // The ONE place the "which layer's K/V does this layer read" rule lives.
  // See the header for why both the graph builder and the KV allocator must
  // resolve it here rather than each re-deriving it.
  if (!isKVSharedLayer(layer_id) || layer_types.empty())
    return -1;

  const int first_kv_shared_layer_idx = NUM_LAYERS - NUM_KV_SHARED_LAYERS;
  if (first_kv_shared_layer_idx > static_cast<int>(layer_types.size()) ||
      layer_id >= static_cast<int>(layer_types.size()))
    return -1;

  // Search the non-shared prefix [0, first_kv_shared_layer_idx) backwards for
  // the last layer of the same type. Reverse-iterating the prefix in place
  // (instead of copying it into a temporary vector, as this rule did while it
  // was inline) keeps the answer identical and the cost O(1) allocations.
  const auto &curr_layer_type = layer_types[layer_id];
  const std::vector<std::string>::const_reverse_iterator prefix_rbegin(
    layer_types.begin() + first_kv_shared_layer_idx);
  const auto prefix_rend = layer_types.rend();

  const auto rev_it = std::find(prefix_rbegin, prefix_rend, curr_layer_type);
  NNTR_THROW_IF(rev_it == prefix_rend, std::invalid_argument)
    << "[Gemma4] Could not find shared KV source layer for layer " << layer_id
    << " with layer_type=" << curr_layer_type;

  return first_kv_shared_layer_idx - 1 -
         static_cast<int>(std::distance(prefix_rbegin, rev_it));
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

  // attn_logit_softcapping is now parsed in Transformer::setupParameters
  // (base); final_logit_softcapping stays here (Gemma4-specific member).
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

  // NOTE(Adreno image attn): gemma4's dual head-dim geometry (sliding d=256,
  // global_head_dim=512) runs correctly on the OHWI image kernels — the
  // earlier "d>256 corrupts" observation was an artifact of mixing image and
  // flash layers per call (state desync), not a kernel tiling limit. With
  // the kernels' sliding-window mask and the uniform per-process image
  // state, no model-level vetting is needed here (device-validated:
  // uniform-image gemma4 @999 tok coherent, prefill 2323 vs flash 1060 TPS).

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
  // Folded-LUT packaging: the fold lives in the sidecar tables, so the flag
  // is meaningless without both of them.
  FOLDED_LUT =
    nntr_cfg.contains("folded_lut") && nntr_cfg["folded_lut"].get<bool>();
  NNTR_THROW_IF(FOLDED_LUT &&
                  (EMBEDDING_FILE_NAME.empty() || PLE_FILE_NAME.empty()),
                std::invalid_argument)
    << "[Gemma4] folded_lut requires both embedding_file_name and "
       "ple_file_name: the folded scales/projection live in those sidecar "
       "LUTs, not in the model bin";

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

  // Folded LUTs carry these scales inside the table content (the token rows
  // are sqrt(hidden)*embedding, the PLE rows the finished per-layer input),
  // so applying them again at lookup time would double-scale.
  EMBEDDING_SCALE = FOLDED_LUT ? 1.0f : std::sqrt(static_cast<float>(DIM));
  EMBEDDING_PER_LAYER_SCALE =
    FOLDED_LUT ? 1.0f
               : std::sqrt(static_cast<float>(HIDDEN_SIZE_PER_LAYER_INPUT));
}

std::pair<Tensor, Tensor>
Gemma4Transformer::createGemma4KVCachePlaceholders(const int layer_id,
                                                   unsigned int kv_width) {
  const unsigned int cache_rows = static_cast<unsigned int>(MAX_SEQ_LEN);
#ifdef ENABLE_FP16
  ml::train::TensorDim cache_dim(
    {BATCH_SIZE, 1, cache_rows, kv_width},
    {ml::train::TensorDim::Format::NCHW, ml::train::TensorDim::DataType::FP16});

  Tensor cache_k(cache_dim, "cache_k_l" + std::to_string(layer_id));
  Tensor cache_v(cache_dim, "cache_v_l" + std::to_string(layer_id));
  return {cache_k, cache_v};
#else
  const std::string cache_shape = std::to_string(BATCH_SIZE) +
                                  ":1:" + std::to_string(cache_rows) + ":" +
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

/**
 * @brief Placeholder factory hook for the base wireAttentionKVCache. Forwards
 * to the model's own factory with the per-layer cache width (sliding vs global
 * head_dim), so n_heads -- the uniform-geometry input the base formula uses --
 * is deliberately unused here.
 */
std::pair<Tensor, Tensor>
Gemma4Transformer::createKVCachePlaceholders(const int layer_id, int n_heads) {
  (void)n_heads;
  return createGemma4KVCachePlaceholders(layer_id, getKVCacheWidth(layer_id));
}

std::pair<Tensor, Tensor> Gemma4Transformer::constructModel() {

  Tensor x =
    Tensor({1, 1, 1, static_cast<unsigned int>(INIT_SEQ_LEN)}, "input0");

  // TieWordEmbedding exists to share the table with a tied lm_head. With
  // LMHEAD_UNTIE the head is a separate FC, so embedding0 is a plain
  // embedding_layer — lookup-identical (mirrored dequant/scale/handoff code,
  // same weight record) and it unlocks the mmap'd sidecar (embedding_file_name)
  // that TieWordEmbedding structurally lacks.
  const bool embedding_tied = TIE_WORD_EMBEDDINGS && !LMHEAD_UNTIE;
  const std::string embedding_type =
    embedding_tied ? "tie_word_embeddings" : "embedding_layer";

  NNTR_THROW_IF(embedding_tied && !EMBEDDING_FILE_NAME.empty(),
                std::invalid_argument)
    << "embedding_file_name requires an untied embedding_layer (tied lm_head "
       "scans every row per decode step, so a sidecar saves nothing)";
  std::vector<std::string> embedding0_props =
    buildEmbeddingLayerProperties("embedding0", NUM_VOCAB, DIM, EMBEDDING_DTYPE,
                                  EMBEDDING_SCALE, EMBEDDING_FILE_NAME);
  if (!embedding_tied && !EMBD_SIDECAR_EXPORT.empty())
    embedding0_props.emplace_back(
      withKey("sidecar_export_path", EMBD_SIDECAR_EXPORT));
  LayerHandle embedding(createLayer(embedding_type, embedding0_props));
  Tensor h = embedding(x);

  const unsigned int per_layer_total_dim =
    NUM_LAYERS * HIDDEN_SIZE_PER_LAYER_INPUT;

  // per-layer input embedding is a lookup table -> use EMBEDDING_DTYPE.
  // (EmbeddingLayer save/load supports FP32/Q4_0/Q6_K, not QINT4, so it must
  // not inherit FC_LAYER_DTYPE; quantize.cpp maps it to embd_dtype to match.)
  // ple_file_name (nntr_config.json) points at a GGML q4_0/q6_k sidecar
  // manifest: the table then lives OUTSIDE the model .bin, mmap'd and
  // dequantized per token on demand instead of held resident (~GB saved).
  // ple_sidecar_export is the matching extraction key (nntr_quantize
  // --ple_sidecar); mutually exclusive with ple_file_name.
  std::vector<std::string> ple_embedding_props = {
    withKey("name", "per_layer_input_embedding"),
    withKey("in_dim", std::to_string(VOCAB_SIZE_PER_LAYER_INPUT)),
    withKey("out_dim", std::to_string(per_layer_total_dim)),
    withKey("weight_dtype", EMBEDDING_DTYPE),
    withKey("scale", EMBEDDING_PER_LAYER_SCALE)};
  if (!PLE_FILE_NAME.empty())
    ple_embedding_props.emplace_back(
      withKey("quantized_lut_path", PLE_FILE_NAME));
  if (!PLE_SIDECAR_EXPORT.empty())
    ple_embedding_props.emplace_back(
      withKey("sidecar_export_path", PLE_SIDECAR_EXPORT));
  LayerHandle per_layer_embedding(
    createLayer("embedding_layer", ple_embedding_props));
  Tensor per_layer_embedding_out = per_layer_embedding(x);

  LayerHandle per_layer_projection(createLayer(
    "fully_connected",
    {withKey("name", "per_layer_input_projection"),
     withKey("unit", std::to_string(per_layer_total_dim)),
     withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE),
     withKey("engine", causallm_engine())}));
  Tensor per_layer_projected = per_layer_projection(h);

  float ple_proj_scale = 1.0f / std::sqrt(static_cast<float>(DIM));
  LayerHandle model_proj_scale(createLayer(
    "scalar_multiply",
    {withKey("name", "per_layer_model_proj_scale"), withKey("packed", "false"),
     withKey("multiplier", std::to_string(ple_proj_scale)),
     withKey("engine", causallm_engine())}));
  Tensor scaled_projection = model_proj_scale(per_layer_projected);

  LayerHandle projection_norm(createLayer(
    "reshaped_rms_norm",
    {
      withKey("name", "per_layer_projection_norm"),
      withKey("epsilon", std::to_string(NORM_EPS)),
      withKey("feature_size", std::to_string(HIDDEN_SIZE_PER_LAYER_INPUT)),
      withKey("packed", "false"),
      withKey("engine", causallm_engine()), // S1.1: GPU_CLMEM output, no map
    }));
  Tensor normalized_projection = projection_norm(scaled_projection);

  if (!FOLDED_LUT) {
    LayerHandle per_layer_sum(
      createLayer("addition", {withKey("name", "per_layer_input_sum"),
                               withKey("engine", causallm_engine())}));
    Tensor per_layer_sum_out =
      per_layer_sum({per_layer_embedding_out, normalized_projection});

    // TODO : change per_layer_input_scale to non hard-coded way

    float per_layer_input_scale = std::sqrt(0.5f);

    LayerHandle per_layer_input_scale_layer(createLayer(
      "scalar_multiply",
      {
        withKey("name", "per_layer_input_scale"),
        withKey("packed", "false"),
        withKey("multiplier", std::to_string(per_layer_input_scale)),
        withKey("engine", causallm_engine()),
      }));
    per_layer_input = per_layer_input_scale_layer(per_layer_sum_out);
  } else {
    // folded_lut: the PLE sidecar rows already ARE
    //   sqrt(0.5) * (sqrt(256)*ple_row + rms_norm(projection(h)/sqrt(hidden)))
    // so the combine above must not run again -- the LUT lookup output is the
    // finished per-layer input, and the sqrt(0.5) tail is folded away.
    //
    // The projection/norm layers above are still BUILT, and kept reachable
    // through a zero-multiply: compile() discovers layers by walking back
    // from the model output (tensor_api_graph.cpp DFS), so a dead-ended
    // chain would be pruned from the graph -- and with it the
    // per_layer_input_projection + per_layer_projection_norm weight records,
    // shifting every positional file offset the loader derives from the
    // sorted graph (neuralnet.cpp load) and misloading the EXISTING bin,
    // which still carries those weights. Adding exactly 0 * norm keeps the
    // result bit-equal to the LUT row while keeping the weights in place.
    // The dead FC still executes each step; accepted waste for v1 -- a
    // folded bin without these records could drop this anchor entirely.
    // Caveat: 0 * (+-Inf) = NaN, so an FP16 overflow in the dead chain
    // would poison per_layer_input even though its value is unused; the
    // unfolded path had the same exposure (it USED the value), so this is
    // no regression, but a v2 could zero via a select and be immune.
    LayerHandle folded_proj_zero(createLayer(
      "scalar_multiply", {
                           withKey("name", "per_layer_folded_proj_zero"),
                           withKey("packed", "false"),
                           withKey("multiplier", std::to_string(0.0f)),
                           withKey("engine", causallm_engine()),
                         }));
    Tensor zeroed_projection = folded_proj_zero(normalized_projection);

    LayerHandle per_layer_sum(
      createLayer("addition", {withKey("name", "per_layer_input_sum"),
                               withKey("engine", causallm_engine())}));
    per_layer_input =
      per_layer_sum({per_layer_embedding_out, zeroed_projection});
  }

  layer_k_norms.assign(NUM_LAYERS, Tensor());
  layer_v_norms.assign(NUM_LAYERS, Tensor());
  for (int i = 0; i < NUM_LAYERS; ++i) {
    h = createTransformerDecoderBlock(i, h);
  }

  std::vector<std::string> output_norm_props = {
    withKey("name", "output_norm"),
    withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false")};
  output_norm_props.push_back(withKey("engine", causallm_engine()));
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
  attn_norm_props.push_back(withKey("engine", causallm_engine()));
  appendSkipPrefillIfNeeded(attn_norm_props, is_kv_shared_layer);
  LayerHandle attn_norm(createLayer("rms_norm", attn_norm_props));
  Tensor normed = attn_norm(input);

  // One source of truth -- see getSharedKVSourceLayer(). The KV allocator
  // asks the same function (through Transformer::getKVSourceLayer()) which
  // layers get storage, so the plane a layer attends over and the plane it was
  // allocated cannot drift apart.
  const int shared_kv_layer_id = getSharedKVSourceLayer(layer_id);

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
  post_attn_norm_props.push_back(withKey("engine", causallm_engine()));
  appendSkipPrefillIfNeeded(post_attn_norm_props, is_kv_shared_layer);
  LayerHandle post_attn_norm(createLayer("rms_norm", post_attn_norm_props));
  Tensor post_normed = post_attn_norm(att_out);

  std::vector<std::string> post_attention_add_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_post_attention")};
  post_attention_add_props.push_back(withKey("engine", causallm_engine()));
  appendSkipPrefillIfNeeded(post_attention_add_props, is_kv_shared_layer);
  LayerHandle post_attention_add(
    createLayer("addition", post_attention_add_props));
  Tensor post_attention = post_attention_add({input, post_normed});

  std::vector<std::string> pre_ffn_norm_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_pre_ffn_norm"),
    withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false")};
  pre_ffn_norm_props.push_back(withKey("engine", causallm_engine()));
  appendSkipPrefillIfNeeded(pre_ffn_norm_props, is_kv_shared_layer);
  LayerHandle pre_ffn_norm(createLayer("rms_norm", pre_ffn_norm_props));
  Tensor pre_ffn = pre_ffn_norm(post_attention);

  Tensor ffn_out = createMlp(layer_id, DIM, INTERMEDIATE_SIZE, pre_ffn);

  std::vector<std::string> post_ffn_norm_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_post_ffn_norm"),
    withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false")};
  post_ffn_norm_props.push_back(withKey("engine", causallm_engine()));
  appendSkipPrefillIfNeeded(post_ffn_norm_props, is_kv_shared_layer);
  LayerHandle post_ffn_norm(createLayer("rms_norm", post_ffn_norm_props));
  Tensor post_ffn = post_ffn_norm(ffn_out);

  std::vector<std::string> decoder_output_base_props = {withKey(
    "name", "layer" + std::to_string(layer_id) + "_decoder_output_base")};
  decoder_output_base_props.push_back(withKey("engine", causallm_engine()));
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
  per_layer_slice_props.push_back(withKey("engine", causallm_engine()));
  appendSkipPrefillIfNeeded(per_layer_slice_props, is_kv_shared_layer);
  LayerHandle per_layer_slice(
    createLayer("per_layer_slice", per_layer_slice_props));
  Tensor per_layer_input_slice = per_layer_slice(per_layer_input);

  std::vector<std::string> per_layer_input_gate_props = {
    withKey("name",
            "layer" + std::to_string(layer_id) + "_per_layer_input_gate"),
    withKey("unit", std::to_string(HIDDEN_SIZE_PER_LAYER_INPUT)),
    withKey("disable_bias", "true"),
    withKey("weight_initializer", "ones"),
    withKey("weight_dtype", FC_LAYER_DTYPE),
    withKey("engine", causallm_engine())};
  appendSkipPrefillIfNeeded(per_layer_input_gate_props, is_kv_shared_layer);
  LayerHandle per_layer_input_gate(
    createLayer("fully_connected", per_layer_input_gate_props));
  Tensor per_layer_input_gate_out = per_layer_input_gate(decoder_output_base);

  // Fused GeGLU: gelu_tanh(gate) * per_layer_input_slice on GPU. Replaces the
  // separate tanh_gelu activation + element-wise multiply (same gelu(a)*b
  // pattern as the FFN GeGLU); no CL activation/multiply exists and those CPU
  // ops break SVM/cl_mem residency.
  std::vector<std::string> per_layer_input_mul_props = {
    withKey("name",
            "layer" + std::to_string(layer_id) + "_per_layer_input_mul"),
    withKey("engine", causallm_engine())};
  appendSkipPrefillIfNeeded(per_layer_input_mul_props, is_kv_shared_layer);
  LayerHandle per_layer_input_mul(
    createLayer("geglu", per_layer_input_mul_props));
  Tensor per_layer_input_multiplied =
    per_layer_input_mul({per_layer_input_gate_out, per_layer_input_slice});

  std::vector<std::string> per_layer_input_proj_props = {
    withKey("name",
            "layer" + std::to_string(layer_id) + "_per_layer_input_proj"),
    withKey("unit", std::to_string(DIM)),
    withKey("disable_bias", "true"),
    withKey("weight_initializer", "ones"),
    withKey("weight_dtype", FC_LAYER_DTYPE),
    withKey("engine", causallm_engine())};
  appendSkipPrefillIfNeeded(per_layer_input_proj_props, is_kv_shared_layer);
  LayerHandle per_layer_input_proj(
    createLayer("fully_connected", per_layer_input_proj_props));
  Tensor per_layer_input_projected =
    per_layer_input_proj(per_layer_input_multiplied);

  std::vector<std::string> post_per_layer_input_norm_props = {
    withKey("name",
            "layer" + std::to_string(layer_id) + "_post_per_layer_input_norm"),
    withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false")};
  post_per_layer_input_norm_props.push_back(
    withKey("engine", causallm_engine()));
  appendSkipPrefillIfNeeded(post_per_layer_input_norm_props,
                            is_kv_shared_layer);
  LayerHandle post_per_layer_input_norm(
    createLayer("rms_norm", post_per_layer_input_norm_props));
  Tensor per_layer_input_normed =
    post_per_layer_input_norm(per_layer_input_projected);

  std::vector<std::string> decoder_output_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_decoder_output")};
  decoder_output_props.push_back(withKey("engine", causallm_engine()));
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
  layer_scalar_props.push_back(withKey("engine", causallm_engine()));
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
  std::vector<std::string> q_params = {withKey("name", Q),
                                       withKey("unit", curr_head_dim * n_heads),
                                       withKey("disable_bias", "true"),
                                       withKey("weight_initializer", "ones"),
                                       withKey("weight_dtype", FC_LAYER_DTYPE),
                                       withKey("engine", causallm_engine())};
  appendSkipPrefillIfNeeded(q_params, is_kv_shared_layer);
  LayerHandle wq(createLayer("fully_connected", q_params));
  Tensor q = wq(query);

  // q_norm on per-head projection [B, S, Nq*Dh]
  std::vector<std::string> q_norm_params = {
    withKey("name", Q_norm), withKey("packed", "false"),
    withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("feature_size", std::to_string(curr_head_dim)),
    withKey("engine", causallm_engine())}; // S1.1: GPU_CLMEM output, no map
  appendSkipPrefillIfNeeded(q_norm_params, is_kv_shared_layer);
  LayerHandle q_norm(createLayer("reshaped_rms_norm", q_norm_params));
  Tensor q_normed = q_norm(q);

  // Gemma4TextAttention uses scaling=1.0 after q_norm/k_norm.
  // mha_core backend applies 1/sqrt(head_dim) to QK, so pre-scale Q by
  // sqrt(head_dim) to preserve Gemma4 semantics.

  // TODO : fix AVX kernel to not make it divide by 1/sqrt(head_dim) on gemma4
  std::vector<std::string> q_scale_params = {
    withKey("name", Q_scaled), withKey("packed", "false"),
    withKey("multiplier",
            std::to_string(std::sqrt(static_cast<float>(curr_head_dim)))),
    withKey("engine", causallm_engine())};
  // Same skip as q_norm right above: on a KV-shared layer the scaled query
  // only feeds the attention, which skips the prefill big-step.
  appendSkipPrefillIfNeeded(q_scale_params, is_kv_shared_layer);
  LayerHandle q_scale(createLayer("scalar_multiply", q_scale_params));
  Tensor q_scaled = q_scale(q_normed);

  const unsigned int window_size = getLayerSlidingWindow(layer_id);
  unsigned int rope_theta =
    is_sliding ? SLIDING_ATTENTION_ROPE_THETA : FULL_ATTENTION_ROPE_THETA;

  const std::string &rope_type =
    is_sliding ? SLIDING_ATTENTION_ROPE_TYPE : FULL_ATTENTION_ROPE_TYPE;
  const float rope_partial_rotary_factor =
    is_sliding ? SLIDING_ATTENTION_ROPE_PARTIAL_ROTARY_FACTOR
               : FULL_ATTENTION_ROPE_PARTIAL_ROTARY_FACTOR;

  Tensor shared_k_norm = layer_k_norms[shared_kv_layer_id];
  Tensor shared_v_norm = layer_v_norms[shared_kv_layer_id];
  layer_k_norms[layer_id] = shared_k_norm;
  layer_v_norms[layer_id] = shared_v_norm;

  // Shared attention core receives [Q_norm, shared_K_norm, shared_V_norm].
  // use_gemm_attention=true routes prefill onto the GPU flash path
  // (mha_core.cpp:1941). The flash kernel handles d=256 sliding (window mask)
  // + GQA; the d=512 full layers fail the VPL<=8 check and fall back to the
  // (x86-FP16-Q-safe) CPU gemm_attention. Without this flag prefill attention
  // runs entirely on the slow CPU non-gemm path (~222 TPS @ M=1024).
  std::vector<std::string> a_params = {
    withKey("name", A), withKey("num_heads", n_heads),
    withKey("num_heads_kv", curr_kv_heads),
    withKey("max_timestep", std::to_string(MAX_SEQ_LEN)),
    withKey("max_position_embeddings", std::to_string(MAX_POSITION_EMBEDDINGS)),
    withKey("sliding_window", window_size), withKey("use_rope", "true"),
    withKey("rope_theta", std::to_string(rope_theta)),
    withKey("rope_scaling_type", rope_type),
    withKey("rope_partial_rotary_factor",
            std::to_string(rope_partial_rotary_factor)),
    withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
    withKey("attn_logit_softcapping", std::to_string(ATTN_LOGIT_SOFTCAPPING)),
    withKey("use_gemm_attention", "true"),
    // Decode-GPU path is token-identical for gemma4: both the flash decode
    // attention and the GPU-RoPE-decode are validated, so enable both by
    // default (no NNTR_MHA_GPU_DECODE env needed). The env flag still forces
    // them on globally and NNTR_NO_GPU_ROPE still disables (A).
    // decode-GPU gates now derive from getModelFeatures() (the
    // single source) instead of per-call literals. Values unchanged (gemma4:
    // all GPU), so token-identical.
    withKey("gpu_decode_attn",
            getModelFeatures().decode_gpu ? "true" : "false"),
    withKey("gpu_decode_rope",
            getModelFeatures().decode_rope_gpu ? "true" : "false"),
    withKey("gpu_ohwi_rope", getModelFeatures().decode_gpu ? "true" : "false"),
    withKey("is_causal", IS_CAUSAL ? "true" : "false")};
  appendSkipPrefillIfNeeded(a_params, is_kv_shared_layer);
  LayerHandle mha(createLayer("mha_core", a_params));
  Tensor a = wireAttentionKVCache(layer_id, n_heads, mha, q_scaled,
                                  shared_k_norm, shared_v_norm,
                                  /*use_int8=*/false);

  // O layer [B, S, Nq*Dh] -> [B, S, H]
  std::vector<std::string> o_params = {withKey("name", O), withKey("unit", DIM),
                                       withKey("disable_bias", "true"),
                                       withKey("weight_initializer", "ones"),
                                       withKey("weight_dtype", FC_LAYER_DTYPE)};
  o_params.push_back(withKey("engine", causallm_engine()));
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
  std::vector<std::string> q_params = {withKey("name", Q),
                                       withKey("unit", curr_head_dim * n_heads),
                                       withKey("disable_bias", "true"),
                                       withKey("weight_initializer", "ones"),
                                       withKey("weight_dtype", FC_LAYER_DTYPE),
                                       withKey("engine", causallm_engine())};
  appendSkipPrefillIfNeeded(q_params, is_kv_shared_layer);
  LayerHandle wq(createLayer("fully_connected", q_params));
  Tensor q = wq(query);

  // K layer [B, S, H] -> [B, S, Nk*Dh]
  std::vector<std::string> k_params = {
    withKey("name", K),
    withKey("unit", curr_head_dim * curr_kv_heads),
    withKey("disable_bias", "true"),
    withKey("weight_initializer", "ones"),
    withKey("weight_dtype", FC_LAYER_DTYPE),
    withKey("engine", causallm_engine())};
  appendSkipPrefillIfNeeded(k_params, is_kv_shared_layer);
  LayerHandle wk(createLayer("fully_connected", k_params));
  Tensor k = wk(key);

  // V layer [B, S, H] -> [B, S, Nk*Dh]
  std::vector<std::string> v_params = {
    withKey("name", V),
    withKey("unit", curr_head_dim * curr_kv_heads),
    withKey("disable_bias", "true"),
    withKey("weight_initializer", "ones"),
    withKey("weight_dtype", FC_LAYER_DTYPE),
    withKey("engine", causallm_engine())};
  appendSkipPrefillIfNeeded(v_params, is_kv_shared_layer);
  LayerHandle wv(createLayer("fully_connected", v_params));
  Tensor v = wv(value);

  // q_norm on per-head projection [B, S, Nq*Dh]
  std::vector<std::string> q_norm_params = {
    withKey("name", Q_norm), withKey("packed", "false"),
    withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("feature_size", std::to_string(curr_head_dim)),
    withKey("engine", causallm_engine())}; // S1.1: GPU_CLMEM output, no map
  appendSkipPrefillIfNeeded(q_norm_params, is_kv_shared_layer);
  LayerHandle q_norm(createLayer("reshaped_rms_norm", q_norm_params));
  Tensor q_normed = q_norm(q);

  // Gemma4TextAttention uses scaling=1.0 after q_norm/k_norm.
  // mha_core backend applies 1/sqrt(head_dim) to QK, so pre-scale Q by
  // sqrt(head_dim) to preserve Gemma4 semantics.
  std::vector<std::string> q_scale_params = {
    withKey("name", Q_scaled), withKey("packed", "false"),
    withKey("multiplier",
            std::to_string(std::sqrt(static_cast<float>(curr_head_dim)))),
    withKey("engine", causallm_engine())};
  // Same skip as q_norm right above: on a KV-shared layer the scaled query
  // only feeds the attention, which skips the prefill big-step.
  appendSkipPrefillIfNeeded(q_scale_params, is_kv_shared_layer);
  LayerHandle q_scale(createLayer("scalar_multiply", q_scale_params));
  Tensor q_scaled = q_scale(q_normed);

  // k_norm on per-head projection [B, S, Nk*Dh]
  std::vector<std::string> k_norm_params = {
    withKey("name", K_norm), withKey("packed", "false"),
    withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("feature_size", std::to_string(curr_head_dim)),
    withKey("engine", causallm_engine())}; // S1.1: GPU_CLMEM output, no map
  appendSkipPrefillIfNeeded(k_norm_params, is_kv_shared_layer);
  LayerHandle k_norm(createLayer("reshaped_rms_norm", k_norm_params));
  Tensor k_normed = k_norm(k);

  // v_norm on per-head projection [B, S, Nk*Dh] (no learned scale)
  std::vector<std::string> v_norm_params = {
    withKey("name", V_norm), withKey("packed", "false"),
    withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("feature_size", std::to_string(curr_head_dim)),
    withKey("engine", causallm_engine())}; // S1.1: GPU_CLMEM output, no map
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

  const unsigned int window_size = getLayerSlidingWindow(layer_id);
  unsigned int rope_theta =
    is_sliding ? SLIDING_ATTENTION_ROPE_THETA : FULL_ATTENTION_ROPE_THETA;
  const std::string &rope_type =
    is_sliding ? SLIDING_ATTENTION_ROPE_TYPE : FULL_ATTENTION_ROPE_TYPE;
  const float rope_partial_rotary_factor =
    is_sliding ? SLIDING_ATTENTION_ROPE_PARTIAL_ROTARY_FACTOR
               : FULL_ATTENTION_ROPE_PARTIAL_ROTARY_FACTOR;

  // Attention core receives [Q_norm, K_norm, V_norm].
  // use_gemm_attention=true routes prefill onto the GPU flash path
  // (mha_core.cpp:1941). The flash kernel handles d=256 sliding (window mask)
  // + GQA; the d=512 full layers fail the VPL<=8 check and fall back to the
  // (x86-FP16-Q-safe) CPU gemm_attention. Without this flag prefill attention
  // runs entirely on the slow CPU non-gemm path (~222 TPS @ M=1024).
  std::vector<std::string> a_params = {
    withKey("name", A), withKey("num_heads", n_heads),
    withKey("num_heads_kv", curr_kv_heads),
    withKey("max_timestep", std::to_string(MAX_SEQ_LEN)),
    withKey("max_position_embeddings", std::to_string(MAX_POSITION_EMBEDDINGS)),
    withKey("sliding_window", window_size), withKey("use_rope", "true"),
    withKey("rope_theta", std::to_string(rope_theta)),
    withKey("rope_scaling_type", rope_type),
    withKey("rope_partial_rotary_factor",
            std::to_string(rope_partial_rotary_factor)),
    withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
    withKey("attn_logit_softcapping", std::to_string(ATTN_LOGIT_SOFTCAPPING)),
    withKey("use_gemm_attention", "true"),
    // Decode-GPU path is token-identical for gemma4: both the flash decode
    // attention and the GPU-RoPE-decode are validated, so enable both by
    // default (no NNTR_MHA_GPU_DECODE env needed). The env flag still forces
    // them on globally and NNTR_NO_GPU_ROPE still disables (A).
    // decode-GPU gates now derive from getModelFeatures() (the
    // single source) instead of per-call literals. Values unchanged (gemma4:
    // all GPU), so token-identical.
    withKey("gpu_decode_attn",
            getModelFeatures().decode_gpu ? "true" : "false"),
    withKey("gpu_decode_rope",
            getModelFeatures().decode_rope_gpu ? "true" : "false"),
    withKey("gpu_ohwi_rope", getModelFeatures().decode_gpu ? "true" : "false"),
    withKey("is_causal", IS_CAUSAL ? "true" : "false")};
  appendSkipPrefillIfNeeded(a_params, is_kv_shared_layer);
  LayerHandle mha(createLayer("mha_core", a_params));
  Tensor a = wireAttentionKVCache(layer_id, n_heads, mha, q_scaled, k_normed,
                                  v_normed, /*use_int8=*/false);

  // O layer [B, S, Nq*Dh] -> [B, S, H]
  std::vector<std::string> o_params = {withKey("name", O), withKey("unit", DIM),
                                       withKey("disable_bias", "true"),
                                       withKey("weight_initializer", "ones"),
                                       withKey("weight_dtype", FC_LAYER_DTYPE)};
  o_params.push_back(withKey("engine", causallm_engine()));
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
    withKey("unit", curr_hidden_dim),
    withKey("disable_bias", "true"),
    withKey("weight_initializer", "ones"),
    withKey("weight_dtype", FC_LAYER_DTYPE),
    withKey("engine", causallm_engine())};
  appendSkipPrefillIfNeeded(ffn_gate_props, is_kv_shared_layer);
  LayerHandle ffn_gate(createLayer("fully_connected", ffn_gate_props));
  Tensor gate = ffn_gate(input);

  std::vector<std::string> ffn_up_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_ffn_up"),
    withKey("unit", curr_hidden_dim),
    withKey("disable_bias", "true"),
    withKey("weight_initializer", "ones"),
    withKey("weight_dtype", FC_LAYER_DTYPE),
    withKey("engine", causallm_engine())};
  appendSkipPrefillIfNeeded(ffn_up_props, is_kv_shared_layer);
  LayerHandle ffn_up(createLayer("fully_connected", ffn_up_props));
  Tensor up = ffn_up(input);

  // Fused GeGLU: gelu_tanh(gate) * up on GPU (GeGLULayerCl). Replaces the
  // separate tanh_gelu activation + element-wise multiply -- there is no CL
  // activation/multiply, and those CPU ops break SVM/cl_mem residency.
  std::vector<std::string> ffn_geglu_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_ffn_geglu"),
    withKey("engine", causallm_engine())};
  appendSkipPrefillIfNeeded(ffn_geglu_props, is_kv_shared_layer);
  LayerHandle ffn_geglu(createLayer("geglu", ffn_geglu_props));
  Tensor geglu = ffn_geglu({gate, up});

  std::vector<std::string> ffn_down_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
    withKey("unit", dim),
    withKey("disable_bias", "true"),
    withKey("weight_initializer", "ones"),
    withKey("weight_dtype", FC_LAYER_DTYPE),
    withKey("engine", causallm_engine())};
  appendSkipPrefillIfNeeded(ffn_down_props, is_kv_shared_layer);
  LayerHandle ffn_down(createLayer("fully_connected", ffn_down_props));

  return ffn_down(geglu);
}

void Gemma4Transformer::registerCustomLayers() {
  auto &ct_engine = nntrainer::Engine::Global();

  // One facade lambda for any backend — no static_cast to a concrete Context.
  // Inert (caught) when the named context is absent (e.g. "cuda" on a non-CUDA
  // build, "gpu" on CPU-only).
  auto tryRegister = [&](const char *engine, auto factory_fn) {
    try {
      ct_engine.registerLayerFactory(engine, factory_fn);
    } catch (std::invalid_argument &e) {
      std::cerr << "failed to register factory on " << engine
                << " ctx, reason: " << e.what() << std::endl;
    }
  };

  tryRegister("cpu", nntrainer::createLayer<causallm::ReshapedRMSNormLayer>);
  tryRegister("cpu", nntrainer::createLayer<causallm::PerLayerSliceLayer>);
  // scalar_multiply + logit_softcapping promoted to core — self-registered
  // on cpu/gpu/cuda contexts (app_context.cpp / cl_context.cpp /
  // cuda_context.cpp).

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
  // Additive CUDA backend: register the gemma4-specific host layers on the cuda
  // context (host-on-UVM). reshaped_rms_norm is centralized in
  // CausalLM::registerCustomLayers (cuda), so it is not repeated here.
  tryRegister("cuda", nntrainer::createLayer<causallm::PerLayerSliceLayer>);
#endif
  // S1.1 GPU-context registration of ReshapedRMSNormLayer is now centralized in
  // CausalLM::registerCustomLayers (shared by all models). The q/k/v_norm +
  // per_layer_projection_norm here still build with engine=GPU; pairs with
  // NNTR_VNORM_GPU=1 for the gamma-free v_norm/PLE-norm GPU path.
}

void Gemma4CausalLM::registerCustomLayers() {
  CausalLM::registerCustomLayers();
  Gemma4Transformer::registerCustomLayers();
}

std::pair<Tensor, Tensor> Gemma4CausalLM::constructModel() {
  auto [x, h] = Gemma4Transformer::constructModel();

  // create lm_head layer (using fully_connected option)
  // QINT4 lm_head (S4): UNTIE from the Q6_K input embedding so the output
  // projection runs as a v8c QINT4 GPU GEMV (dotCl_v8c, fully_connected path,
  // ~3ms) instead of the ALU-bound gpu_native Q6_K GEMV (~17.5ms/token). The
  // input embedding must stay Q6_K (row-gather dequant) so the two cannot share
  // one weight; output_of_causallm carries a separate, transposed
  // [hidden,vocab] QINT4 copy. Used by BOTH nntr_quantize (constructs this
  // model to quantize output_of_causallm as a per-channel section-A FC weight)
  // and inference. Gated on LMHEAD_DTYPE == QINT4 so Q6_K/Q4_0 lmheads keep the
  // tied path. Untie is a config flag (LMHEAD_UNTIE), NOT derived from
  // LMHEAD_DTYPE: the quantizer builds this same untied graph with an FP32
  // source weight (weight_dtype follows the source dtype) and the dtype map
  // quantizes output_of_causallm to QINT4 on save; inference rebuilds it with
  // the QINT4 weight. Gating on the dtype would force a QINT4 tensor at
  // quantize time and fail to load the FP32 source.
  //
  // The layer build itself is the shared CausalLM::buildLmHeadOutput helper;
  // only the skip_prefill decision is model-local (appendSkipPrefillIfNeeded's
  // `true` argument reduces to ENABLE_SKIP_PREFILL_OPT), so it is passed in.
  Tensor y = buildLmHeadOutput(h, ENABLE_SKIP_PREFILL_OPT);

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

} // namespace causallm
