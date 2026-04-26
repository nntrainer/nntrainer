// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file	gemma3_causallm.cpp
 * @date	24 Dec 2025
 * @brief	This defines a gemma3 causal language model.
 * @see		https://github.com/nnstreamer/
 * @author	Seungbaek Hong <sb92.hong@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */
#include <gemma3_causallm.h>

#include <app_context.h>
#include <engine.h>
#include <llm_util.hpp>
#include <reshaped_rms_norm.h>

namespace causallm {

json &Gemma3Transformer::sanitizeConfig(json &cfg) {
  if (!cfg.contains("tie_word_embeddings")) {
    cfg["tie_word_embeddings"] = true;
  }
  return cfg;
}

json &Gemma3Transformer::sanitizeGenerationConfig(json &gen_cfg,
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

void Gemma3Transformer::setupParameters(json &cfg, json &generation_cfg,
                                        json &nntr_cfg) {
  Transformer::setupParameters(cfg, generation_cfg, nntr_cfg);
  if (cfg.contains("layer_types")) {
    layer_types = cfg["layer_types"].get<std::vector<std::string>>();
  }
  if (cfg.contains("attn_logit_softcapping") &&
      !cfg["attn_logit_softcapping"].is_null()) {
    ATTN_LOGIT_SOFTCAPPING = cfg["attn_logit_softcapping"].get<float>();
  }
}

Tensor Gemma3Transformer::createTransformerDecoderBlock(const int layer_id,
                                                        Tensor input) {

  LayerHandle attn_norm(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention_norm"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));
  Tensor normed = attn_norm(input);

  Tensor att_out = createAttention(layer_id, INIT_SEQ_LEN, NUM_HEADS, HEAD_DIM,
                                   normed, normed, normed);

  LayerHandle post_attn_norm(createLayer(
    "rms_norm", {withKey("name", "layer" + std::to_string(layer_id) +
                                   "_post_attention_norm"),
                 withKey("epsilon", std::to_string(NORM_EPS)),
                 withKey("packed", "false")}));
  Tensor post_normed = post_attn_norm(att_out);

  LayerHandle post_attn_add(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_post_attention")}));
  Tensor post_attn = post_attn_add({input, post_normed});

  LayerHandle pre_ffn_norm(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "pre_ffn_norm"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));
  Tensor pre_ffn = pre_ffn_norm(post_attn);

  Tensor ffn_out = createMlp(layer_id, DIM, INTERMEDIATE_SIZE, pre_ffn);

  LayerHandle post_ffn_norm(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "post_ffn_norm"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));
  Tensor post_ffn = post_ffn_norm(ffn_out);

  LayerHandle decoder_output(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_decoder_output")}));
  return decoder_output({post_attn, post_ffn});
}

Tensor Gemma3Transformer::createAttention(const int layer_id, int seq_len,
                                          int n_heads, int head_dim,
                                          Tensor query, Tensor key,
                                          Tensor value) {

  // Q layer
  LayerHandle wq(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_wq"),
     withKey("unit", head_dim * n_heads), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));
  Tensor q = wq(query);

  // K layer
  LayerHandle wk(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_wk"),
     withKey("unit", head_dim * n_heads / GQA_SIZE),
     withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));
  Tensor k = wk(key);

  // V layer
  LayerHandle wv(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_wv"),
     withKey("unit", head_dim * n_heads / GQA_SIZE),
     withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));
  Tensor v = wv(value);

  // q_norm
  LayerHandle q_norm(createLayer(
    "reshaped_rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_q_norm"),
     withKey("packed", "false"), withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("feature_size", std::to_string(head_dim))}));
  Tensor q_normed = q_norm(q);

  // k_norm
  LayerHandle k_norm(createLayer(
    "reshaped_rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_k_norm"),
     withKey("packed", "false"), withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("feature_size", std::to_string(head_dim))}));
  Tensor k_normed = k_norm(k);

  // Attention core layer
  unsigned int window_size = UINT_MAX;
  if (!layer_types.empty()) {
    if (layer_id < layer_types.size()) {
      if (layer_types[layer_id] == "sliding_attention") {
        window_size = SLIDING_WINDOW;
      }
    }
  } else {
    window_size = SLIDING_WINDOW;
  }

  float rope_theta = ROPE_THETA; // Default global
  if (!layer_types.empty() && layer_id < layer_types.size()) {
    if (layer_types[layer_id] == "sliding_attention") {
      rope_theta = 10000.0f;
    }
  }

  // External KV cache placeholders (per-layer). Storage is owned by the host
  // (KVCacheManager) and bound at runtime via setExternalTensors.
  auto [cache_k, cache_v] = createKVCachePlaceholders(layer_id, n_heads);

  LayerHandle mha(createLayer(
    "mha_core",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention"),
     withKey("num_heads", n_heads), withKey("num_heads_kv", n_heads / GQA_SIZE),
     withKey("max_timestep", std::to_string(INIT_SEQ_LEN + NUM_TO_GENERATE)),
     withKey("sliding_window", window_size),
     withKey("rope_theta", std::to_string(rope_theta)),
     withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
     withKey("attn_logit_softcapping", std::to_string(ATTN_LOGIT_SOFTCAPPING)),
     withKey("is_causal", IS_CAUSAL ? "true" : "false")}));
  Tensor a = mha({q_normed, k_normed, v, cache_k, cache_v});

  // O layer
  LayerHandle wo(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention_out"),
     withKey("unit", DIM), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));
  return wo(a);
}

Tensor Gemma3Transformer::createMlp(const int layer_id, int dim, int hidden_dim,
                                    Tensor input) {

  // Gate projection
  LayerHandle ffn_gate(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_gate"),
     withKey("unit", hidden_dim), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));
  Tensor gate = ffn_gate(input);

  // GeLU
  LayerHandle gelu(createLayer(
    "activation",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_gate_gelu"),
     withKey("activation", "tanh_gelu")}));
  Tensor gate_gelu = gelu(gate);

  // Up projection
  LayerHandle ffn_up(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_up"),
     withKey("unit", hidden_dim), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));
  Tensor up = ffn_up(input);

  // Multiply (GeGLU = gate_gelu * up)
  LayerHandle mul(createLayer(
    "multiply",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_geglu")}));
  Tensor geglu = mul({gate_gelu, up});

  // Down projection
  LayerHandle ffn_down(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
     withKey("unit", dim), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));
  return ffn_down(geglu);
}

void Gemma3Transformer::registerCustomLayers() {
  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

  try {
    app_context->registerFactory(
      nntrainer::createLayer<causallm::ReshapedRMSNormLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }
}

void Gemma3CausalLM::registerCustomLayers() {
  CausalLM::registerCustomLayers();
  Gemma3Transformer::registerCustomLayers();
}

} // namespace causallm
