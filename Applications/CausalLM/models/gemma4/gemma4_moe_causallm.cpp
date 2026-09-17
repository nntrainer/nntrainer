// SPDX-License-Identifier: Apache-2.0
/**
 * @file   gemma4_moe_causallm.cpp
 * @brief  Gemma4 MoE causal language model implementation.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug    No known bugs
 */

#include <app_context.h>
#include <engine.h>
#include <gemma4_moe_causallm.h>
#include <gemma4_moe_layer.h>
#include <llm_util.hpp>
#include <model.h>

#include <iostream>

namespace causallm {

void Gemma4MoECausalLM::setupParameters(json &cfg, json &generation_cfg,
                                        json &nntr_cfg) {
  Gemma4CausalLM::setupParameters(cfg, generation_cfg, nntr_cfg);

  NNTR_THROW_IF(!cfg.contains("num_experts") || cfg["num_experts"].is_null() ||
                  !cfg.contains("top_k_experts") ||
                  cfg["top_k_experts"].is_null() ||
                  !cfg.contains("moe_intermediate_size") ||
                  cfg["moe_intermediate_size"].is_null(),
                std::invalid_argument)
    << "[Gemma4MoE] num_experts, top_k_experts, and moe_intermediate_size "
       "must be provided";

  num_experts = cfg["num_experts"].get<unsigned int>();
  top_k_experts = cfg["top_k_experts"].get<unsigned int>();
  moe_intermediate_size = cfg["moe_intermediate_size"].get<unsigned int>();
  moe_cache_size =
    nntr_cfg.contains("moe_cache_size") && !nntr_cfg["moe_cache_size"].is_null()
      ? nntr_cfg["moe_cache_size"].get<unsigned int>()
      : 0;

  NNTR_THROW_IF(num_experts == 0 || top_k_experts == 0 ||
                  top_k_experts > num_experts || moe_intermediate_size == 0,
                std::invalid_argument)
    << "[Gemma4MoE] invalid expert configuration";
  NNTR_THROW_IF(NUM_KV_SHARED_LAYERS > 0, std::invalid_argument)
    << "[Gemma4MoE] shared KV layers are not supported";
}

Tensor Gemma4MoECausalLM::createFeedForwardBlock(const int layer_id,
                                                 Tensor post_attention,
                                                 bool is_kv_shared_layer) {
  std::vector<std::string> pre_ffn_norm_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_pre_ffn_norm"),
    withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false")};
  appendSkipPrefillIfNeeded(pre_ffn_norm_props, is_kv_shared_layer);
  LayerHandle pre_ffn_norm(createLayer("rms_norm", pre_ffn_norm_props));
  Tensor dense_input = pre_ffn_norm(post_attention);
  Tensor dense_output =
    createMlp(layer_id, DIM, INTERMEDIATE_SIZE, dense_input);

  LayerHandle post_dense_norm(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_post_ffn_norm_1"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));
  Tensor post_dense = post_dense_norm(dense_output);

  LayerHandle pre_sparse_norm(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_pre_ffn_norm_2"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));
  Tensor sparse_input = pre_sparse_norm(post_attention);
  LayerHandle sparse_moe(createLayer(
    "gemma4_moe",
    {withKey("name", "layer" + std::to_string(layer_id) + "_sparse_moe"),
     withKey("unit", std::to_string(moe_intermediate_size)),
     withKey("num_experts", std::to_string(num_experts)),
     withKey("num_experts_per_token", std::to_string(top_k_experts)),
     withKey("moe_cache_size", std::to_string(moe_cache_size)),
     withKey("moe_activation", "tanh_gelu"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));
  Tensor sparse_output = sparse_moe({sparse_input, post_attention});

  LayerHandle post_sparse_norm(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_post_ffn_norm_2"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));
  Tensor post_sparse = post_sparse_norm(sparse_output);
  LayerHandle combine_ffn(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_combine_ffn")}));
  Tensor combined_ffn = combine_ffn({post_dense, post_sparse});

  LayerHandle post_combined_norm(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_post_ffn_norm"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));
  return post_combined_norm(combined_ffn);
}

void Gemma4MoECausalLM::registerCustomLayers() {
  Gemma4CausalLM::registerCustomLayers();
  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));
  try {
    app_context->registerFactory(nntrainer::createLayer<Gemma4MoELayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }
}

} // namespace causallm
