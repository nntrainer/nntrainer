// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file	gemma2_causallm.cpp
 * @date	08 Jun 2026
 * @brief	This defines a gemma2 causal language model.
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */
#include <gemma2_causallm.h>

#include <app_context.h>
#include <engine.h>
#include <llm_util.hpp>
#include <reshaped_rms_norm.h>

namespace causallm {

json &Gemma2Transformer::sanitizeConfig(json &cfg) {
  if (!cfg.contains("tie_word_embeddings")) {
    cfg["tie_word_embeddings"] = true;
  }
  // Gemma2 alternates local (sliding) / global (full) attention every other
  // layer. HF gemma2 config does not carry an explicit sliding_window_pattern,
  // so inject the canonical period-2 pattern when absent.
  if (!cfg.contains("sliding_window_pattern")) {
    cfg["sliding_window_pattern"] = 2;
  }
  if (!cfg.contains("layer_types") && cfg.contains("sliding_window_pattern") &&
      cfg.contains("num_hidden_layers")) {
    const unsigned int num_layers = cfg["num_hidden_layers"];
    const unsigned int sliding_window_pattern = cfg["sliding_window_pattern"];
    std::vector<std::string> layer_types;
    layer_types.reserve(num_layers);
    for (unsigned int i = 0; i < num_layers; ++i) {
      layer_types.push_back(sliding_window_pattern != 0 &&
                                (i + 1) % sliding_window_pattern == 0
                              ? "full_attention"
                              : "sliding_attention");
    }
    cfg["layer_types"] = layer_types;
  }
  return cfg;
}

json &Gemma2Transformer::sanitizeGenerationConfig(json &gen_cfg,
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

void Gemma2Transformer::setupParameters(json &cfg, json &generation_cfg,
                                        json &nntr_cfg) {
  Transformer::setupParameters(cfg, generation_cfg, nntr_cfg);
  EMBEDDING_SCALE = std::sqrt(static_cast<float>(DIM));
  if (cfg.contains("layer_types")) {
    layer_types = cfg["layer_types"].get<std::vector<std::string>>();
  }
  // attn_logit_softcapping is now parsed in Transformer::setupParameters
  // (base).
}

Tensor Gemma2Transformer::createTransformerDecoderBlock(const int layer_id,
                                                        Tensor input) {

  LayerHandle attn_norm(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention_norm"),
     withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false"),
     withKey("engine", causallm_engine())}));
  Tensor normed = attn_norm(input);

  Tensor att_out = createAttention(layer_id, INIT_SEQ_LEN, NUM_HEADS, HEAD_DIM,
                                   normed, normed, normed);

  // NNTR_FUSE_ADDNORM: fuse the post-norm RMSNorm with its residual add into a
  // single 2-input "rms_norm" node (out = rmsnorm(sublayer)*gamma + residual),
  // removing the separate AdditionLayerCL (v8c_add_h2h GPU kernel + dispatch
  // idle). Default off -> the original separate norm + addition graph.
  static const bool fuse_addnorm = std::getenv("NNTR_FUSE_ADDNORM") != nullptr;

  LayerHandle post_attn_norm(createLayer(
    "rms_norm",
    {withKey("name",
             "layer" + std::to_string(layer_id) + "_post_attention_norm"),
     withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false"),
     withKey("engine", causallm_engine())}));
  Tensor post_attn;
  if (fuse_addnorm) {
    post_attn = post_attn_norm({att_out, input});
  } else {
    Tensor post_normed = post_attn_norm(att_out);
    LayerHandle post_attn_add(createLayer(
      "addition",
      {withKey("name", "layer" + std::to_string(layer_id) + "_post_attention"),
       withKey("engine", causallm_engine())}));
    post_attn = post_attn_add({input, post_normed});
  }

  LayerHandle pre_ffn_norm(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "pre_ffn_norm"),
     withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false"),
     withKey("engine", causallm_engine())}));
  Tensor pre_ffn = pre_ffn_norm(post_attn);

  Tensor ffn_out = createMlp(layer_id, DIM, INTERMEDIATE_SIZE, pre_ffn);

  LayerHandle post_ffn_norm(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "post_ffn_norm"),
     withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false"),
     withKey("engine", causallm_engine())}));
  if (fuse_addnorm)
    return post_ffn_norm({ffn_out, post_attn});

  Tensor post_ffn = post_ffn_norm(ffn_out);

  LayerHandle decoder_output(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_decoder_output"),
     withKey("engine", causallm_engine())}));
  return decoder_output({post_attn, post_ffn});
}

Tensor Gemma2Transformer::createAttention(const int layer_id, int seq_len,
                                          int n_heads, int head_dim,
                                          Tensor query, Tensor key,
                                          Tensor value) {

  // Q layer
  LayerHandle wq(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_wq"),
     withKey("unit", head_dim * n_heads), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE),
     withKey("engine", causallm_engine())}));
  Tensor q = wq(query);

  // K layer
  LayerHandle wk(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_wk"),
     withKey("unit", head_dim * n_heads / GQA_SIZE),
     withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE),
     withKey("engine", causallm_engine())}));
  Tensor k = wk(key);

  // V layer
  LayerHandle wv(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_wv"),
     withKey("unit", head_dim * n_heads / GQA_SIZE),
     withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE),
     withKey("engine", causallm_engine())}));
  Tensor v = wv(value);

  // NOTE: Gemma2 has NO per-head q/k RMSNorm (unlike Gemma3). Q and K feed the
  // attention core directly after projection.

  // Attention core layer
  const unsigned int window_size = getLayerSlidingWindow(layer_id);

  // Gemma2 uses a single global RoPE theta (rope_theta, default 1e4) for both
  // sliding and full attention layers.
  float rope_theta = ROPE_THETA;

  LayerHandle mha(createLayer(
    "mha_core",
    {
      withKey("name", "layer" + std::to_string(layer_id) + "_attention"),
      withKey("num_heads", n_heads),
      withKey("num_heads_kv", n_heads / GQA_SIZE),
      withKey("max_timestep", std::to_string(MAX_SEQ_LEN)),
      withKey("sliding_window", window_size),
      withKey("rope_theta", std::to_string(rope_theta)),
      withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
      withKey("attn_logit_softcapping", std::to_string(ATTN_LOGIT_SOFTCAPPING)),
      withKey("is_causal", IS_CAUSAL ? "true" : "false"),
      withKey("use_gemm_attention", "true"),
      // Decode-GPU: flash decode attention (B) is token-identical for gemma2,
      // so enable it by default. GPU-RoPE-decode (A) DIVERGES on gemma2, so
      // keep it OFF -> decode runs GPU flash attention + HOST RoPE (the fast,
      // token-identical combination). NNTR_MHA_GPU_DECODE env still forces
      // both. derive from getModelFeatures() (single source). Values
      // unchanged (gemma2: attn/ohwi GPU, rope HOST), so token-identical.
      withKey("gpu_decode_attn",
              getModelFeatures().decode_gpu ? "true" : "false"),
      withKey("gpu_decode_rope",
              getModelFeatures().decode_rope_gpu ? "true" : "false"),
      withKey("gpu_ohwi_rope",
              getModelFeatures().decode_gpu ? "true" : "false"),
    }));
  Tensor a = wireAttentionKVCache(layer_id, n_heads, mha, q, k, v,
                                  /*use_int8=*/false);

  // O layer
  LayerHandle wo(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention_out"),
     withKey("unit", DIM), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE),
     withKey("engine", causallm_engine())}));
  return wo(a);
}

Tensor Gemma2Transformer::createMlp(const int layer_id, int dim, int hidden_dim,
                                    Tensor input) {

  // Gate projection
  LayerHandle ffn_gate(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_gate"),
     withKey("unit", hidden_dim), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE),
     withKey("engine", causallm_engine())}));
  Tensor gate = ffn_gate(input);

  // Up projection
  LayerHandle ffn_up(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_up"),
     withKey("unit", hidden_dim), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE),
     withKey("engine", causallm_engine())}));
  Tensor up = ffn_up(input);

  // Fused GeGLU: gelu_tanh(gate) * up (GPU GeGLULayerCl). Replaces the separate
  // tanh_gelu activation + element-wise multiply (no CL multiply/activation
  // exists, and those CPU ops break on SVM-resident tensors).
  LayerHandle geglu(createLayer(
    "geglu",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_geglu"),
     withKey("engine", causallm_engine())}));
  Tensor act = geglu({gate, up});

  // Down projection
  LayerHandle ffn_down(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
     withKey("unit", dim), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE),
     withKey("engine", causallm_engine())}));
  return ffn_down(act);
}

void Gemma2Transformer::registerCustomLayers() {
  auto &ct_engine = nntrainer::Engine::Global();

  try {
    ct_engine.registerLayerFactory(
      "cpu", nntrainer::createLayer<causallm::ReshapedRMSNormLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }
}

void Gemma2CausalLM::registerCustomLayers() {
  CausalLM::registerCustomLayers();
  Gemma2Transformer::registerCustomLayers();
}

} // namespace causallm
