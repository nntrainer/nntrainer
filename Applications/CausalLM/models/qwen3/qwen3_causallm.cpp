/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * @file	qwen3_causallm.cpp
 * @date	23 July 2025
 * @brief	This defines a qwen3 causal language model.
 * @see		https://github.com/nnstreamer/
 * @author	Eunju Yang <ej.yang@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */
#include <llm_util.hpp>
#include <model.h>
#include <qwen3_causallm.h>

#include <app_context.h>
#include <engine.h>
#include <reshaped_rms_norm.h>

namespace causallm {

Tensor Qwen3Transformer::createAttention(const int layer_id, int seq_len,
                                         int n_heads, int head_dim,
                                         Tensor query, Tensor key,
                                         Tensor value) {

  // Q layer
  LayerHandle wq(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_wq"),
     withKey("unit", head_dim * n_heads), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones"),
     withKey("engine", causallm_engine())}));
  Tensor q = wq(query);

  // Q-reshaped-norm layer (q_norm(q_proj.view(hidden_shape))).
  // engine=GPU (mirror of Gemma4 S1.1): keeps q_norm output GPU_CLMEM-resident
  // (the layer is registered on the cl context in registerCustomLayers) instead
  // of draining q to the host for a CPU RMS norm every layer. Decode ~+20% on
  // Adreno, prefill neutral, token-identical. q/k norm carry gamma so they take
  // the GPU coop kernel directly (no gamma-free v_norm fallback concern).
  std::vector<std::string> q_norm_params = {
    withKey("name", "layer" + std::to_string(layer_id) + "_q_norm"),
    withKey("packed", "false"), withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("feature_size", std::to_string(head_dim)),
    withKey("engine", causallm_engine())};
  LayerHandle q_norm(createLayer("reshaped_rms_norm", q_norm_params));
  Tensor q_normed = q_norm(q);

  // K layer
  LayerHandle wk(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_wk"),
     withKey("unit", head_dim * n_heads / GQA_SIZE),
     withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
     withKey("engine", causallm_engine())}));
  Tensor k = wk(key);

  // K-reshaped-norm layer (k_norm(k_proj.view(hidden_shape))). engine=GPU as
  // with q_norm above (GPU_CLMEM-resident, no per-layer host drain).
  std::vector<std::string> k_norm_params = {
    withKey("name", "layer" + std::to_string(layer_id) + "_k_norm"),
    withKey("packed", "false"), withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("feature_size", std::to_string(head_dim)),
    withKey("engine", causallm_engine())};
  LayerHandle k_norm(createLayer("reshaped_rms_norm", k_norm_params));
  Tensor k_normed = k_norm(k);

  // V layer
  LayerHandle wv(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_wv"),
     withKey("unit", head_dim * n_heads / GQA_SIZE),
     withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
     withKey("engine", causallm_engine())}));
  Tensor v = wv(value);

  // KV cache wiring. Default is external (5-input mha) with FP16
  // placeholders owned by KVCacheManager. When NNTR_KV_INT8=1, switch to
  // 3-input mode so mha_core allocates an INT8 cache + FP16 scale
  // tensors internally - createKVCachePlaceholders only emits FP16
  // tensors so it can't host the int8 path.
  static const bool _kv_int8_setup = std::getenv("NNTR_KV_INT8") != nullptr;

  // Attention core layer
  LayerHandle mha(createLayer(
    "mha_core",
    {
      withKey("name", "layer" + std::to_string(layer_id) + "_attention"),
      withKey("num_heads", n_heads),
      withKey("num_heads_kv", n_heads / GQA_SIZE),
      withKey("max_timestep", std::to_string(MAX_SEQ_LEN)),
      withKey("sliding_window", getLayerSlidingWindow(layer_id)),
      withKey("rope_theta", ROPE_THETA),
      withKey("max_position_embeddings", MAX_POSITION_EMBEDDINGS),
      withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
      withKey("is_causal", IS_CAUSAL ? "true" : "false"),
      // Decode-GPU: qwen3 flash decode attention DIVERGES (a separate
      // head_dim=128 bug) even with host RoPE, so keep BOTH the decode flash
      // attention (B) and the GPU-RoPE-decode (A) OFF for now (explicit; both
      // default false anyway). NNTR_MHA_GPU_DECODE env still forces them on for
      // testing.
      // derive from getModelFeatures() (single source). Values
      // unchanged (qwen3: host decode, head_dim=128 diverges), so
      // token-identical.
      withKey("gpu_decode_attn",
              getModelFeatures().decode_gpu ? "true" : "false"),
      withKey("gpu_decode_rope",
              getModelFeatures().decode_rope_gpu ? "true" : "false"),
      withKey("gpu_ohwi_rope",
              getModelFeatures().decode_gpu ? "true" : "false"),
    }));
  Tensor a = wireAttentionKVCache(layer_id, n_heads, mha, q_normed, k_normed, v,
                                  _kv_int8_setup);

  // O layer
  LayerHandle wo(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention_out"),
     withKey("unit", DIM), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones"),
     withKey("engine", causallm_engine())}));
  return wo(a);
}

void Qwen3Transformer::registerCustomLayers() {
  ///
  auto &ct_engine = nntrainer::Engine::Global();

  try {
    ct_engine.registerLayerFactory(
      "cpu", nntrainer::createLayer<causallm::ReshapedRMSNormLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }
  // GPU-context registration of ReshapedRMSNormLayer is centralized in
  // CausalLM::registerCustomLayers (shared by all models); q/k norm above build
  // with engine=GPU and resolve there to stay GPU_CLMEM-resident.
  //
  // ...but that chokepoint is only reachable from the CausalLM hierarchy.
  // Qwen3Embedding is `SentenceTransformer + Qwen3Transformer` and never
  // inherits CausalLM, so under engine=gpu its q_norm/k_norm asked for a
  // "reshaped_rms_norm" that was registered on the cpu context only — and an
  // unregistered type on a live context THROWS ("Key is not found for the
  // object. Key: reshaped_rms_norm") instead of falling back to cpu.
  // Registering here as well covers every model that mixes in this class; the
  // duplicate registration from CausalLM's central call is caught below, so
  // whichever runs first wins and the other is a benign no-op.
  try {
    ct_engine.registerLayerFactory(
      "gpu", nntrainer::createLayer<causallm::ReshapedRMSNormLayer>);
  } catch (std::invalid_argument &e) {
    // no "gpu" context (CPU-only build) or already registered — both benign.
  }
}

void Qwen3CausalLM::registerCustomLayers() {
  CausalLM::registerCustomLayers();
  Qwen3Transformer::registerCustomLayers();
}

} // namespace causallm
