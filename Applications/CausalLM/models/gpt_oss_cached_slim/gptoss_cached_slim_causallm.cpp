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

std::vector<LayerHandle> GptOssCachedSlimCausalLM::createAttention(
  const int layer_id, int seq_len, int n_heads, int head_dim,
  std::string query_name, std::string key_name, std::string value_name) {

  std::vector<LayerHandle> layers;

  ///@note Q/K/V/O has bias!
  auto Q = "layer" + std::to_string(layer_id) + "_wq";
  auto K = "layer" + std::to_string(layer_id) + "_wk";
  auto V = "layer" + std::to_string(layer_id) + "_wv";
  auto A = "layer" + std::to_string(layer_id) + "_attention";
  auto O = "layer" + std::to_string(layer_id) + "_attention_out";

  // V layer
  std::vector<std::string> v_params = {
    withKey("name", V), withKey("unit", head_dim * n_heads / GQA_SIZE),
    withKey("disable_bias", "false"), withKey("input_layers", value_name),
    withKey("weight_initializer", "ones")};
  layers.push_back(createLayer("fully_connected", v_params));

  // K layer
  std::vector<std::string> k_params = {
    withKey("name", K), withKey("unit", head_dim * n_heads / GQA_SIZE),
    withKey("disable_bias", "false"), withKey("input_layers", key_name),
    withKey("weight_initializer", "ones")};
  layers.push_back(createLayer("fully_connected", k_params));

  // Q layer
  std::vector<std::string> q_params = {
    withKey("name", Q), withKey("unit", head_dim * n_heads),
    withKey("disable_bias", "false"), withKey("input_layers", query_name),
    withKey("weight_initializer", "ones")};
  layers.push_back(createLayer("fully_connected", q_params));

  // Attention core layer
  // layer_types[layer_id] == "sliding_attention"
  // layer_types[layer_id] == "full_attention"
  unsigned sliding_window =
    (LAYER_TYPES[layer_id] == "sliding_attention") ? SLIDING_WINDOW : UINT_MAX;
  // this attention use sink!
  std::vector<std::string> a_params = {
    withKey("name", A),
    withKey("num_heads", n_heads),
    withKey("num_heads_kv", n_heads / GQA_SIZE),
    withKey("max_timestep", std::to_string(INIT_SEQ_LEN + NUM_TO_GENERATE)),
    withKey("sliding_window", sliding_window),
    withKey("rope_theta", ROPE_THETA),
    withKey("max_position_embeddings", MAX_POSITION_EMBEDDINGS),
    withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
    withKey("use_sink", "true"),
    withKey("rope_scaling_factor", ATTENTION_ROPE_SCALING_FACTOR),
    withKey("rope_scaling_type", "yarn"),
    withKey("rope_scaling_max_position_embeddings", 4096),
    withKey("input_layers", {Q, K, V})};
  layers.push_back(createLayer("mha_core", a_params));

  // O layer
  std::vector<std::string> o_params = {
    withKey("name", O), withKey("unit", DIM), withKey("disable_bias", "false"),
    withKey("input_layers", A), withKey("weight_initializer", "ones")};
  layers.push_back(createLayer("fully_connected", o_params));

  return layers;
}

std::vector<LayerHandle>
GptOssCachedSlimCausalLM::createMlp(const int layer_id, int dim, int hidden_dim,
                                    std::string input_name) {

  std::vector<LayerHandle> layers;
  layers.push_back(createLayer(
    "gpt_oss_moe_slim_cached",
    {
      withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
      withKey("input_layers", input_name),
      withKey("unit", hidden_dim),
      withKey("num_experts", NUM_EXPERTS),
      withKey("num_experts_per_token", NUM_EXPERTS_PER_TOK),
    }));

  return layers;
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
