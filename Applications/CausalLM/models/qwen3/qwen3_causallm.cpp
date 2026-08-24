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
#include <hexagon_context.h>
#include <reshaped_rms_norm.h>

namespace causallm {

Tensor Qwen3Transformer::createAttention(const int layer_id, int seq_len,
                                         int n_heads, int head_dim,
                                         Tensor query, Tensor key,
                                         Tensor value) {

  // Q/K/V layers batched into one qkv_layer: three independent Q4_0
  // matmuls sharing the same activation (query == key == value going in -
  // see Transformer::createAttention's callers), dispatched together so a
  // Hexagon cDSP ComputeOps can collapse them into one FastRPC round trip
  // (gemm_q4_0_batch_fp32) instead of three. Weight request order inside
  // QKVLayer (Q, then K, then V) matches the three-separate-layer order this
  // replaces, so existing .bin files load unchanged.
  LayerHandle wqkv(createLayer(
    "qkv_layer",
    withHexagonEngine(
      {withKey("name", "layer" + std::to_string(layer_id) + "_wqkv"),
       withKey("q_unit", head_dim * n_heads),
       withKey("k_unit", head_dim * n_heads / GQA_SIZE),
       withKey("v_unit", head_dim * n_heads / GQA_SIZE),
       withKey("disable_bias", "true"),
       withKey("weight_initializer", "ones")})));
  Tensor qkv_out = wqkv(query);
  Tensor q = qkv_out.output(0);
  Tensor k = qkv_out.output(1);
  Tensor v = qkv_out.output(2);

  // Q-reshaped-norm layer (q_norm(q_proj.view(hidden_shape)))
  LayerHandle q_norm(createLayer(
    "reshaped_rms_norm",
    withHexagonEngine(
      {withKey("name", "layer" + std::to_string(layer_id) + "_q_norm"),
       withKey("packed", "false"), withKey("epsilon", std::to_string(NORM_EPS)),
       withKey("feature_size", std::to_string(head_dim))})));
  Tensor q_normed = q_norm(q);

  // K-reshaped-norm layer (k_norm(k_proj.view(hidden_shape)))
  LayerHandle k_norm(createLayer(
    "reshaped_rms_norm",
    withHexagonEngine(
      {withKey("name", "layer" + std::to_string(layer_id) + "_k_norm"),
       withKey("packed", "false"), withKey("epsilon", std::to_string(NORM_EPS)),
       withKey("feature_size", std::to_string(head_dim))})));
  Tensor k_normed = k_norm(k);

  // External KV cache placeholders (per-layer). Storage is owned by the host
  // (KVCacheManager) and bound at runtime via setExternalTensors.
  auto [cache_k, cache_v] = createKVCachePlaceholders(layer_id, n_heads);

  // Attention core layer
  LayerHandle mha(createLayer(
    "mha_core",
    withHexagonEngine(
      {withKey("name", "layer" + std::to_string(layer_id) + "_attention"),
       withKey("num_heads", n_heads),
       withKey("num_heads_kv", n_heads / GQA_SIZE),
       withKey("max_timestep", std::to_string(MAX_SEQ_LEN)),
       withKey("sliding_window", SLIDING_WINDOW),
       withKey("rope_theta", ROPE_THETA),
       withKey("max_position_embeddings", MAX_POSITION_EMBEDDINGS),
       withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
       withKey("is_causal", IS_CAUSAL ? "true" : "false")})));
  Tensor a = mha({q_normed, k_normed, v, cache_k, cache_v});

  // O layer
  LayerHandle wo(createLayer(
    "fully_connected",
    withHexagonEngine(
      {withKey("name", "layer" + std::to_string(layer_id) + "_attention_out"),
       withKey("unit", DIM), withKey("disable_bias", "true"),
       withKey("weight_initializer", "ones")})));
  return wo(a);
}

void Qwen3Transformer::registerCustomLayers() {
  ///
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

  // Also register under "cdsp" so q_norm/k_norm can be tagged engine=cdsp
  // (see withHexagonEngine() at their construction sites above) - same
  // reasoning as Transformer::registerCustomLayers() registering
  // RMSNormLayer/MHACoreLayer there. Probed at runtime since "cdsp" is only
  // registered in builds compiled with -Denable-hexagon-cdsp=true.
  try {
    auto hexagon_context = static_cast<nntrainer::HexagonContext *>(
      ct_engine.getRegisteredContext("cdsp"));
    hexagon_context->registerFactory(
      nntrainer::createLayer<causallm::ReshapedRMSNormLayer>);
  } catch (const std::invalid_argument &) {
    // "cdsp" context not registered in this build - nothing to do.
  }
}

void Qwen3CausalLM::registerCustomLayers() {
  CausalLM::registerCustomLayers();
  Qwen3Transformer::registerCustomLayers();
}

} // namespace causallm
