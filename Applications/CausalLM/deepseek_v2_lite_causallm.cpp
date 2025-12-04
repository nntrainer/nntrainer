// SPDX-License-Identifier: Apache-2.0
/**
 *
 * @file   deepseek_v2_lite_causallm.cpp
 * @brief  deepseek_v2_lite causallm source file
 * @date   04 December 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <app_context.h>
#include <engine.h>

#include <llm_util.hpp>

#include <deepseek_v2_lite_causallm.h>
#include <ernie_moe_layer.h>


namespace causallm {

std::vector<LayerHandle>
DeepseekV2ForCausalLM::createMlp(const int layer_id, int dim, int hidden_dim,
                                   std::string input_name) {
  std::vector<LayerHandle> layers;
  if (layer_id == 0) {
    int ffn_hidden_dim = INTERMEDIATE_SIZE; // Ernie's first layer

    layers.push_back(createLayer(
      "fully_connected",
      {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_up"),
       withKey("unit", ffn_hidden_dim), withKey("disable_bias", "true"),
       withKey("input_layers", input_name),
       withKey("weight_initializer", "ones")}));

    layers.push_back(createLayer(
      "fully_connected",
      {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_gate"),
       withKey("unit", ffn_hidden_dim), withKey("disable_bias", "true"),
       withKey("input_layers", input_name),
       withKey("weight_initializer", "ones")}));

    layers.push_back(createLayer(
      "swiglu",
      {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_swiglu"),
       withKey("input_layers", "layer" + std::to_string(layer_id) + "_ffn_up," +
                                 "layer" + std::to_string(layer_id) +
                                 "_ffn_gate")}));

    layers.push_back(createLayer(
      "fully_connected",
      {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
       withKey("unit", dim), withKey("disable_bias", "true"),
       withKey("input_layers",
               "layer" + std::to_string(layer_id) + "_ffn_swiglu"),
       withKey("weight_initializer", "ones")}));

  } else {
    layers.push_back(createLayer(
      "ernie_moe",
      {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
       withKey("input_layers", input_name),
       withKey("unit", MOE_INTERMEDIATE_SIZE),
       withKey("num_experts", NUM_EXPERTS),
       withKey("num_shared_experts", NUM_SHARED_EXPERTS),
       withKey("num_experts_per_token", NUM_EXPERTS_PER_TOK),
       withKey("moe_norm_min", std::to_string(MOE_NORM_MIN)),
       withKey("moe_activation", "swish")}));
  }
  return layers;
}
std::vector<LayerHandle> DeepseekV2ForCausalLM::createAttention(
  const int layer_id, int seq_len, int n_heads, int head_dim,
  std::string query_name, std::string key_name, std::string value_name) {

  std::vector<LayerHandle> layers;
  auto Q = "layer" + std::to_string(layer_id) + "_wq";
  auto K = "layer" + std::to_string(layer_id) + "_wk";
  auto V = "layer" + std::to_string(layer_id) + "_wv";
  auto A = "layer" + std::to_string(layer_id) + "_attention";
  auto O = "layer" + std::to_string(layer_id) + "_attention_out";

  // V layer
  std::vector<std::string> v_params = {
    withKey("name", V), withKey("unit", head_dim * n_heads / GQA_SIZE),
    withKey("disable_bias", "true"), withKey("input_layers", value_name),
    withKey("weight_initializer", "ones")};
  layers.push_back(createLayer("fully_connected", v_params));

  // K layer
  std::vector<std::string> k_params = {
    withKey("name", K), withKey("unit", head_dim * n_heads / GQA_SIZE),
    withKey("disable_bias", "true"), withKey("input_layers", key_name),
    withKey("weight_initializer", "ones")};
  layers.push_back(createLayer("fully_connected", k_params));

  // Q layer
  std::vector<std::string> q_params = {
    withKey("name", Q), withKey("unit", head_dim * n_heads),
    withKey("disable_bias", "true"), withKey("input_layers", query_name),
    withKey("weight_initializer", "ones")};
  layers.push_back(createLayer("fully_connected", q_params));

  // Attention core layer
  std::vector<std::string> a_params = {
    withKey("name", A),
    withKey("num_heads", n_heads),
    withKey("num_heads_kv", n_heads / GQA_SIZE),
    withKey("max_timestep", std::to_string(INIT_SEQ_LEN + NUM_TO_GENERATE)),
    withKey("sliding_window", SLIDING_WINDOW),
    withKey("rope_theta", ROPE_THETA),
    withKey("max_position_embeddings", MAX_POSITION_EMBEDDINGS),
    withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
    withKey("input_layers", {Q, K, V})};
  layers.push_back(createLayer("mha_core", a_params));

  // O layer
  std::vector<std::string> o_params = {
    withKey("name", O), withKey("unit", DIM), withKey("disable_bias", "true"),
    withKey("input_layers", A), withKey("weight_initializer", "ones")};
  layers.push_back(createLayer("fully_connected", o_params));

  return layers;
}

void DeepseekV2ForCausalLM::setupParameters(json &cfg, json &generation_cfg,
                                              json &nntr_cfg) {

  try {
    NUM_EXPERTS = cfg["moe_num_experts"].get<unsigned int>();
    NUM_EXPERTS_PER_TOK = cfg["num_experts_per_tok"].get<unsigned int>();
    MOE_INTERMEDIATE_SIZE = cfg["moe_intermediate_size"].get<unsigned int>();
    INTERMEDIATE_SIZE = cfg["moe_intermediate_size"].get<unsigned int>();
    NUM_SHARED_EXPERTS = cfg["moe_num_shared_experts"].get<unsigned int>();
    MOE_NORM_MIN =
      cfg.contains("moe_norm_min") ? cfg["moe_norm_min"].get<float>() : 1e-12f;

  } catch (const std::exception &e) {
    throw std::runtime_error("Ernie Causallm: config parsing error");
  }
}

void DeepseekV2ForCausalLM::registerCustomLayers() {
  CausalLM::registerCustomLayers();
  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

  try {
    app_context->registerFactory(
      nntrainer::createLayer<causallm::DeepSeekMoELayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }
}

} // namespace causallm
