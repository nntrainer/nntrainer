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
 * @file	gptoss_causallm.cpp
 * @brief	This defines a gpt_oss causal language model.
 * @date    26 Aug 2025
 * @see		https://github.com/nnstreamer/
 * @author	Eunju Yang <ej.yang@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <gptoss_cached_slim_causallm.h>
#include <llm_util.hpp>
#include <model.h>

#include <app_context.h>
#include <engine.h>
#include <gpt_oss_moe_layer_cached.h>

namespace causallm {

Tensor GptOssCachedSlimCausalLM::createAttention(const int layer_id,
                                                 int seq_len, int n_heads,
                                                 int head_dim, Tensor query,
                                                 Tensor key, Tensor value) {

  ///@note Q/K/V/O has bias!

  // V layer
  LayerHandle wv(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_wv"),
     withKey("unit", head_dim * n_heads / GQA_SIZE),
     withKey("disable_bias", "false"), withKey("weight_initializer", "ones")}));
  Tensor v = wv(value);

  // K layer
  LayerHandle wk(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_wk"),
     withKey("unit", head_dim * n_heads / GQA_SIZE),
     withKey("disable_bias", "false"), withKey("weight_initializer", "ones")}));
  Tensor k = wk(key);

  // Q layer
  LayerHandle wq(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_wq"),
     withKey("unit", head_dim * n_heads), withKey("disable_bias", "false"),
     withKey("weight_initializer", "ones")}));
  Tensor q = wq(query);

  // External KV cache placeholders (per-layer). Storage is owned by the host
  // (KVCacheManager) and bound at runtime via setExternalTensors.
  auto [cache_k, cache_v] = createKVCachePlaceholders(layer_id, n_heads);

  // Attention core layer
  unsigned sliding_window =
    (LAYER_TYPES[layer_id] == "sliding_attention") ? SLIDING_WINDOW : UINT_MAX;
  // this attention uses sink!
  LayerHandle mha(createLayer(
    "mha_core",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention"),
     withKey("num_heads", n_heads), withKey("num_heads_kv", n_heads / GQA_SIZE),
     withKey("max_timestep", std::to_string(INIT_SEQ_LEN + NUM_TO_GENERATE)),
     withKey("sliding_window", sliding_window),
     withKey("rope_theta", ROPE_THETA),
     withKey("max_position_embeddings", MAX_POSITION_EMBEDDINGS),
     withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
     withKey("use_sink", "true"),
     withKey("rope_scaling_factor", ATTENTION_ROPE_SCALING_FACTOR),
     withKey("rope_scaling_type", "yarn"),
     withKey("rope_scaling_max_position_embeddings", 4096)}));
  Tensor a = mha({q, k, v, cache_k, cache_v});

  // O layer
  LayerHandle wo(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention_out"),
     withKey("unit", DIM), withKey("disable_bias", "false"),
     withKey("weight_initializer", "ones")}));
  return wo(a);
}

Tensor GptOssCachedSlimCausalLM::createMlp(const int layer_id, int dim,
                                           int hidden_dim, Tensor input) {

  LayerHandle moe(createLayer(
    "gpt_oss_moe_slim_cached",
    {
      withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
      withKey("unit", hidden_dim),
      withKey("num_experts", NUM_EXPERTS),
      withKey("num_experts_per_token", NUM_EXPERTS_PER_TOK),
    }));
  return moe(input);
}

void GptOssCachedSlimCausalLM::setupParameters(json &cfg, json &generation_cfg,
                                               json &nntr_cfg) {
  CausalLM(cfg, generation_cfg, nntr_cfg);

  try {
    NUM_EXPERTS = cfg["num_local_experts"].get<unsigned int>();
    NUM_EXPERTS_PER_TOK = cfg["num_experts_per_tok"].get<unsigned int>();
    LAYER_TYPES = cfg["layer_types"].get<std::vector<std::string>>();
    ATTENTION_ROPE_SCALING_FACTOR = cfg["rope_scaling"]["factor"];
  } catch (const std::exception &e) {
    throw std::runtime_error("GptOssCachedSlimCausalLM: config parsing error");
  }
}

void GptOssCachedSlimCausalLM::registerCustomLayers() {
  CausalLM::registerCustomLayers();
  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

  try {
    app_context->registerFactory(
      nntrainer::createLayer<causallm::CachedSlimGptOssMoELayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }
}

} // namespace causallm
