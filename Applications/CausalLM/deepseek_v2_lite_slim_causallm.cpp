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

#include <deepseek_v2_lite_slim_causallm.h>
#include <deepseek_moe_layer_slim.h>
#include <deepseek_moe_layer_slim.h>
#include <mla_core.h>
#include <causallm_split_layer.h>
#include <iostream>

namespace causallm {

std::vector<LayerHandle>
DeepseekV2SlimCausalLM::createMlp(const int layer_id, int dim, int hidden_dim,
                                   std::string input_name) {
  std::vector<LayerHandle> layers;
  if (layer_id == 0) { //First Layer is Dense Layer
    int ffn_hidden_dim = INTERMEDIATE_SIZE;

    layers.push_back(createLayer(
      "fully_connected",
      {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_gate"),
       withKey("unit", ffn_hidden_dim), withKey("disable_bias", "true"),
       withKey("input_layers", input_name),
       withKey("weight_initializer", "ones")}));

    layers.push_back(createLayer(
      "fully_connected",
      {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_up"),
       withKey("unit", ffn_hidden_dim), withKey("disable_bias", "true"),
       withKey("input_layers", input_name),
       withKey("weight_initializer", "ones")}));

    layers.push_back(createLayer(
      "swiglu",
      {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_swiglu"),
       withKey("input_layers", "layer" + std::to_string(layer_id) + "_ffn_gate," +
                                 "layer" + std::to_string(layer_id) +
                                 "_ffn_up")}));

    layers.push_back(createLayer(
      "fully_connected",
      {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
       withKey("unit", dim), withKey("disable_bias", "true"),
       withKey("input_layers",
               "layer" + std::to_string(layer_id) + "_ffn_swiglu"),
       withKey("weight_initializer", "ones")}));

  } else {
    layers.push_back(createLayer(
      "deepseek_moe_slim",
      {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
       withKey("input_layers", input_name),
       withKey("unit", MOE_INTERMEDIATE_SIZE),
       withKey("num_experts", NUM_EXPERTS),
       withKey("num_shared_experts", NUM_SHARED_EXPERTS),
       withKey("num_experts_per_token", NUM_EXPERTS_PER_TOK),
       withKey("moe_norm_min", std::to_string(MOE_NORM_MIN)),
       withKey("num_group_experts", NUM_GROUP_EXPERTS),
       withKey("norm_topk_prob", NORM_TOPK_PROB ? "true" : "false"),
       withKey("routed_scaling_factor", std::to_string(ROUTED_SCALING_FACTOR)),
       withKey("moe_activation", "swish")}));
  }
  return layers;
}
std::vector<LayerHandle> DeepseekV2SlimCausalLM::createAttention(
  const int layer_id, int seq_len, int n_heads, int head_dim,
  std::string query_name, std::string key_name, std::string value_name) {

  std::vector<LayerHandle> layers;
  // MLA Implementation
  // Order adjusted to match weight loading order: KV_A -> Norm -> KV_B -> Q -> O

  // 1. KV Compression (Latent KV)
  // Input: [Batch, Seq, Hidden]
  // Output: [Batch, Seq, KV_LORA_RANK + QK_ROPE_HEAD_DIM]

  // KV_A_PROJ_WITH_MQA
  auto kv_a_proj_name =
    "layer" + std::to_string(layer_id) + "_kv_a_proj_with_mqa";
  int kv_a_out_dim = KV_LORA_RANK + QK_ROPE_HEAD_DIM;
  // Note: key_name and value_name are usually the same input (hidden_states)
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", kv_a_proj_name), withKey("unit", kv_a_out_dim),
     withKey("disable_bias", "true"), withKey("input_layers", query_name),
     withKey("weight_initializer", "ones")}));

  // Split KV_A output into LatentKV and KeyRoPE
  // LatentKV: [Batch, Seq, KV_LORA_RANK]
  // KeyRoPE: [Batch, Seq, QK_ROPE_HEAD_DIM]
  auto latent_kv_name = "layer" + std::to_string(layer_id) + "_latent_kv";
  auto key_rope_name = "layer" + std::to_string(layer_id) + "_key_rope";

  layers.push_back(createLayer(
    "causallm_split",
    {withKey("name", latent_kv_name),
     withKey("input_layers", kv_a_proj_name), withKey("axis", "3"),
     withKey("start_index", "1"),
     withKey("end_index", std::to_string(KV_LORA_RANK + 1))}));

  layers.push_back(createLayer(
    "causallm_split",
    {withKey("name", key_rope_name),
     withKey("input_layers", kv_a_proj_name), withKey("axis", "3"),
     withKey("start_index", std::to_string(KV_LORA_RANK + 1)),
     withKey("end_index",
             std::to_string(KV_LORA_RANK + QK_ROPE_HEAD_DIM + 1))}));

  // 2. KV_A_Norm on LatentKV
  auto kv_a_norm_name = "layer" + std::to_string(layer_id) + "_kv_a_layernorm";

  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", kv_a_norm_name), withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("input_layers", latent_kv_name)}));

  // 3. KV_B_PROJ
  // Input: LatentKV (normalized)
  // Output: [Batch, Seq, NumHeads * (QK_NOPE_HEAD_DIM + V_HEAD_DIM)]
  auto kv_b_proj_name = "layer" + std::to_string(layer_id) + "_kv_b_proj";
  int kv_b_out_dim = n_heads * (QK_NOPE_HEAD_DIM + V_HEAD_DIM);
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", kv_b_proj_name), withKey("unit", kv_b_out_dim),
     withKey("disable_bias", "true"), // kv_b_proj has no bias
     withKey("input_layers", kv_a_norm_name),
     withKey("weight_initializer", "ones")}));

  // 4. Q Projection (Standard or LoRA)
  // Input: [Batch, Seq, Hidden]
  // Output: [Batch, Seq, NumHeads * (QK_NOPE + QK_ROPE)]
  std::string q_proj_name;
    // Standard Linear Projection for Q
    q_proj_name = "layer" + std::to_string(layer_id) + "_q_proj";
    int q_out_dim = n_heads * (QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM);
    layers.push_back(createLayer(
      "fully_connected",
      {withKey("name", q_proj_name), withKey("unit", q_out_dim),
       withKey("disable_bias", "true"), withKey("input_layers", query_name),
       withKey("weight_initializer", "ones")}));


  // 5. MLA Core Layer
  // Inputs: Query (q_proj_name), KV_B_Output (kv_b_proj_name), KeyRoPE (key_rope_name)
  auto mla_core_name = "layer" + std::to_string(layer_id) + "_mla_core";
  std::vector<std::string> mla_params = {
    withKey("name", mla_core_name),
    withKey("num_heads", n_heads),
    withKey("num_heads_KV", 1), // MQA style, effectively 1 KV head shared
    withKey("kv_lora_rank", KV_LORA_RANK),
    withKey("qk_rope_dim", QK_ROPE_HEAD_DIM),
    withKey("qk_nope_dim", QK_NOPE_HEAD_DIM),
    withKey("v_head_dim", V_HEAD_DIM), // Pass V_HEAD_DIM
    withKey("max_timestep", std::to_string(INIT_SEQ_LEN + NUM_TO_GENERATE)),
    withKey("sliding_window", SLIDING_WINDOW),
    withKey("rope_theta", ROPE_THETA),
    withKey("max_position_embeddings", MAX_POSITION_EMBEDDINGS),
    withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
    withKey("input_layers",
            {q_proj_name, kv_b_proj_name, key_rope_name}),
    withKey("rope_scaling_type", ROPE_SCALING_TYPE),
    withKey("rope_scaling_factor", std::to_string(ATTENTION_ROPE_SCALING_FACTOR)),
    withKey("rope_scaling_beta_fast", std::to_string(ROPE_SCALING_BETA_FAST)),
    withKey("rope_scaling_beta_slow", std::to_string(ROPE_SCALING_BETA_SLOW)),
    withKey("rope_scaling_mscale", std::to_string(ROPE_SCALING_MSCALE)),
    withKey("rope_scaling_mscale_all_dim", std::to_string(ROPE_SCALING_MSCALE_ALL_DIM)),
    withKey("rope_scaling_max_position_embeddings", std::to_string(ROPE_SCALING_MAX_POSITION_EMBEDDINGS))};

  layers.push_back(createLayer("mla_core", mla_params));

  // 6. Output Projection
  // Input: [Batch, Seq, NumHeads * V_HEAD_DIM]
  // Output: [Batch, Seq, Hidden]
  auto o_proj_name = "layer" + std::to_string(layer_id) + "_attention_out";
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", o_proj_name), withKey("unit", DIM),
     withKey("disable_bias", "true"), withKey("input_layers", mla_core_name),
     withKey("weight_initializer", "ones")}));

  return layers;
}

void DeepseekV2SlimCausalLM::setupParameters(json &cfg, json &generation_cfg,
                                              json &nntr_cfg) {

  try {
    NUM_EXPERTS = cfg["n_routed_experts"].get<unsigned int>();
    NUM_EXPERTS_PER_TOK = cfg["num_experts_per_tok"].get<unsigned int>();
    MOE_INTERMEDIATE_SIZE = cfg["moe_intermediate_size"].get<unsigned int>();
    INTERMEDIATE_SIZE = cfg["intermediate_size"].get<unsigned int>();
    NUM_SHARED_EXPERTS = cfg["n_shared_experts"].get<unsigned int>();
    if (DIM == 2048 && NUM_SHARED_EXPERTS == 2) {
      NUM_SHARED_EXPERTS = 1;
    }
    MOE_NORM_MIN =
      cfg.contains("moe_norm_min") ? cfg["moe_norm_min"].get<float>() : 1e-12f;
    NUM_GROUP_EXPERTS = cfg.contains("n_group") ? cfg["n_group"].get<unsigned int>() : 1;
    NORM_TOPK_PROB = cfg.contains("norm_topk_prob") ? cfg["norm_topk_prob"].get<bool>() : true;

    // MLA parameters
    if (cfg.contains("q_lora_rank") && !cfg["q_lora_rank"].is_null()) {
      Q_LORA_RANK = cfg["q_lora_rank"].get<unsigned int>();
    } else {
      Q_LORA_RANK = 0;
    }
    KV_LORA_RANK = cfg["kv_lora_rank"].get<unsigned int>();
    QK_NOPE_HEAD_DIM = cfg["qk_nope_head_dim"].get<unsigned int>();
    QK_ROPE_HEAD_DIM = cfg["qk_rope_head_dim"].get<unsigned int>();
    QK_ROPE_HEAD_DIM = cfg["qk_rope_head_dim"].get<unsigned int>();
    V_HEAD_DIM = cfg["v_head_dim"].get<unsigned int>();
    if (cfg.contains("attention_bias")) {
      ATTENTION_BIAS = cfg["attention_bias"].get<bool>();
    }
    if (cfg.contains("rms_norm_eps")) {
      NORM_EPS = cfg["rms_norm_eps"].get<float>();
    }

    // RoPE Scaling (Yarn)
    ROPE_SCALING_TYPE = "default";
    ATTENTION_ROPE_SCALING_FACTOR = 1.0f;
    ROPE_SCALING_BETA_FAST = 32.0f;
    ROPE_SCALING_BETA_SLOW = 1.0f;
    ROPE_SCALING_MSCALE = 1.0f;
    ROPE_SCALING_MSCALE_ALL_DIM = 1.0f;
    ROPE_SCALING_MAX_POSITION_EMBEDDINGS = MAX_POSITION_EMBEDDINGS;

    if (cfg.contains("rope_scaling") && !cfg["rope_scaling"].is_null()) {
      auto &rs = cfg["rope_scaling"];
      if (rs.contains("type")) ROPE_SCALING_TYPE = rs["type"].get<std::string>();
      if (rs.contains("factor")) ATTENTION_ROPE_SCALING_FACTOR = rs["factor"].get<float>();
      if (rs.contains("beta_fast")) ROPE_SCALING_BETA_FAST = rs["beta_fast"].get<float>();
      if (rs.contains("beta_slow")) ROPE_SCALING_BETA_SLOW = rs["beta_slow"].get<float>();
      if (rs.contains("mscale")) ROPE_SCALING_MSCALE = rs["mscale"].get<float>();
      if (rs.contains("mscale_all_dim")) ROPE_SCALING_MSCALE_ALL_DIM = rs["mscale_all_dim"].get<float>();
      if (rs.contains("original_max_position_embeddings")) 
        ROPE_SCALING_MAX_POSITION_EMBEDDINGS = rs["original_max_position_embeddings"].get<unsigned int>();
    }

    ROUTED_SCALING_FACTOR = 1.0f;
    if (cfg.contains("routed_scaling_factor")) {
      ROUTED_SCALING_FACTOR = cfg["routed_scaling_factor"].get<float>();
    }

  } catch (const std::exception &e) {
    throw std::runtime_error("DeepseekV2 Causallm: config parsing error: " +
                             std::string(e.what()));
  }
}

void DeepseekV2SlimCausalLM::registerCustomLayers() {
  CausalLM::registerCustomLayers();
  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

  try {
    app_context->registerFactory(
      nntrainer::createLayer<causallm::DeepseekSlimMoELayer>);
    app_context->registerFactory(nntrainer::createLayer<causallm::MLACoreLayer>);
    app_context->registerFactory(nntrainer::createLayer<causallm::SplitLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }
}

} // namespace causallm
