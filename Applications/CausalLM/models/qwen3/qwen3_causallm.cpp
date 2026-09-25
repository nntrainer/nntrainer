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
 * @author	Pranjal Thapliyal <p.thapliyal@samsung.com>
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
  std::vector<std::string> wq_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_wq"),
    withKey("unit", head_dim * n_heads), withKey("disable_bias", "true"),
    withKey("weight_initializer", "ones")};
  if (hasLoRA("wq"))
    appendLoRAProps(wq_props);
  else if (LORA_RANK > 0)
    wq_props.push_back(withKey("trainable", "false"));
  LayerHandle wq(createLayer("fully_connected", wq_props));
  Tensor q = wq(query);

  // Q-reshaped-norm layer (q_norm(q_proj.view(hidden_shape))). Never a LoRA
  // target; frozen when LoRA is active unless TRAIN_NORMS asks for the
  // "LoRA + norms" recipe.
  std::vector<std::string> q_norm_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_q_norm"),
    withKey("packed", "false"), withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("feature_size", std::to_string(head_dim))};
  if (LORA_RANK > 0 && !TRAIN_NORMS)
    q_norm_props.push_back(withKey("trainable", "false"));
  LayerHandle q_norm(createLayer("reshaped_rms_norm", q_norm_props));
  Tensor q_normed = q_norm(q);

  // K layer
  std::vector<std::string> wk_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_wk"),
    withKey("unit", head_dim * n_heads / GQA_SIZE),
    withKey("disable_bias", "true"), withKey("weight_initializer", "ones")};
  if (hasLoRA("wk"))
    appendLoRAProps(wk_props);
  else if (LORA_RANK > 0)
    wk_props.push_back(withKey("trainable", "false"));
  LayerHandle wk(createLayer("fully_connected", wk_props));
  Tensor k = wk(key);

  // K-reshaped-norm layer (k_norm(k_proj.view(hidden_shape)))
  std::vector<std::string> k_norm_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_k_norm"),
    withKey("packed", "false"), withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("feature_size", std::to_string(head_dim))};
  if (LORA_RANK > 0 && !TRAIN_NORMS)
    k_norm_props.push_back(withKey("trainable", "false"));
  LayerHandle k_norm(createLayer("reshaped_rms_norm", k_norm_props));
  Tensor k_normed = k_norm(k);

  // V layer
  std::vector<std::string> wv_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_wv"),
    withKey("unit", head_dim * n_heads / GQA_SIZE),
    withKey("disable_bias", "true"), withKey("weight_initializer", "ones")};
  if (hasLoRA("wv"))
    appendLoRAProps(wv_props);
  else if (LORA_RANK > 0)
    wv_props.push_back(withKey("trainable", "false"));
  LayerHandle wv(createLayer("fully_connected", wv_props));
  Tensor v = wv(value);

  // Attention core layer (frozen: GQA has no learnable weights beyond
  // Q/K/V/O, so there's nothing here for LoRA to target)
  LayerHandle mha(createLayer(
    "mha_core",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention"),
     withKey("num_heads", n_heads), withKey("num_heads_kv", n_heads / GQA_SIZE),
     withKey("max_timestep", std::to_string(MAX_SEQ_LEN)),
     withKey("sliding_window", SLIDING_WINDOW),
     withKey("rope_theta", ROPE_THETA),
     withKey("max_position_embeddings", MAX_POSITION_EMBEDDINGS),
     withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
     withKey("is_causal", IS_CAUSAL ? "true" : "false")}));

  Tensor a;
  if (FOR_TRAINING) {
    // No KV-cache placeholders for training: mha_core's internal-cache
    // (3-input) construction is what its training-mode forward/backward
    // (trainForwarding()/calcDerivative() in mha_core.cpp) requires.
    a = mha({q_normed, k_normed, v});
  } else {
    // External KV cache placeholders (per-layer). Storage is owned by the
    // host (KVCacheManager) and bound at runtime via setExternalTensors.
    auto [cache_k, cache_v] = createKVCachePlaceholders(layer_id, n_heads);
    a = mha({q_normed, k_normed, v, cache_k, cache_v});
  }

  // O layer
  std::vector<std::string> wo_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_attention_out"),
    withKey("unit", DIM), withKey("disable_bias", "true"),
    withKey("weight_initializer", "ones")};
  if (hasLoRA("wo"))
    appendLoRAProps(wo_props);
  else if (LORA_RANK > 0)
    wo_props.push_back(withKey("trainable", "false"));
  LayerHandle wo(createLayer("fully_connected", wo_props));
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
}

void Qwen3CausalLM::registerCustomLayers() {
  CausalLM::registerCustomLayers();
  Qwen3Transformer::registerCustomLayers();
}

} // namespace causallm
