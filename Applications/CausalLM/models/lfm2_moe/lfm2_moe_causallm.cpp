// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jungwon-Lee <jungone.lee@samsung.com>
 *
 * @file   lfm2_moe_causallm.cpp
 * @date   06 July 2026
 * @brief  This defines the LFM2-8B-A1B Mixture-of-Experts causal LM.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <lfm2_moe_causallm.h>
#include <lfm2_moe_layer.h>

#include <app_context.h>
#include <engine.h>
#include <llm_util.hpp>
#include <model.h>

namespace causallm {

void Lfm2MoeCausalLM::setupParameters(json &cfg, json &generation_cfg,
                                      json &nntr_cfg) {
  // Parse the LFM2 backbone parameters (dims, layer_types, conv, ...).
  Lfm2CausalLM::setupParameters(cfg, generation_cfg, nntr_cfg);

  // MoE-specific parameters.
  try {
    NUM_EXPERTS = cfg["num_experts"];
    NUM_EXPERTS_PER_TOK = cfg["num_experts_per_tok"];
    MOE_INTERMEDIATE_SIZE = cfg["moe_intermediate_size"];
  } catch (const std::exception &e) {
    throw std::runtime_error(
      "Lfm2Moe: num_experts, num_experts_per_tok and moe_intermediate_size "
      "must be specified in the config file");
  }
  // Layers [0, num_dense_layers) keep the dense SwiGLU FFN. Optional (default 0).
  NUM_DENSE_LAYERS = cfg.value("num_dense_layers", 0);
}

Tensor Lfm2MoeCausalLM::createMoeLayer(const int layer_id, Tensor input) {
  LayerHandle moe(createLayer(
    "lfm2_moe",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
     withKey("unit", MOE_INTERMEDIATE_SIZE), withKey("num_experts", NUM_EXPERTS),
     withKey("num_experts_per_token", NUM_EXPERTS_PER_TOK),
     withKey("moe_activation", "swish")}));
  return moe(input);
}

Tensor Lfm2MoeCausalLM::createMlp(const int layer_id, int dim, int hidden_dim,
                                  Tensor input) {
  // Dense SwiGLU FFN for the first NUM_DENSE_LAYERS layers.
  if (layer_id < static_cast<int>(NUM_DENSE_LAYERS))
    return Transformer::createMlp(layer_id, dim, hidden_dim, input);

  // MoE FFN for the remaining layers.
  return createMoeLayer(layer_id, input);
}

void Lfm2MoeCausalLM::registerCustomLayers() {

  Lfm2CausalLM::registerCustomLayers();
  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

  try {
    app_context->registerFactory(nntrainer::createLayer<causallm::Lfm2MoELayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register Lfm2MoELayer factory, reason: " << e.what()
              << std::endl;
  }
}

} // namespace causallm
