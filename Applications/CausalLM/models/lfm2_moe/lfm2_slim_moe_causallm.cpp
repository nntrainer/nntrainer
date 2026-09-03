// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jungwon-Lee <jungone.lee@samsung.com>
 *
 * @file   lfm2_slim_moe_causallm.cpp
 * @date   06 July 2026
 * @brief  LFM2-8B-A1B Slim (FSU) Mixture-of-Experts causal LM.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <lfm2_moe_layer_fsu.h>
#include <lfm2_slim_moe_causallm.h>

#include <app_context.h>
#include <engine.h>
#include <llm_util.hpp>
#include <model.h>

namespace causallm {

Tensor Lfm2SlimMoeCausalLM::createMoeLayer(const int layer_id, Tensor input) {
  LayerHandle moe(createLayer(
    "lfm2_moe_slim",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
     withKey("unit", MOE_INTERMEDIATE_SIZE), withKey("num_experts", NUM_EXPERTS),
     withKey("num_experts_per_token", NUM_EXPERTS_PER_TOK),
     withKey("moe_activation", "swish")}));
  return moe(input);
}

void Lfm2SlimMoeCausalLM::registerCustomLayers() {

  // Register the LFM2 backbone custom layers (skip the base MoE layer factory).
  Lfm2CausalLM::registerCustomLayers();
  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

  try {
    app_context->registerFactory(
      nntrainer::createLayer<causallm::Lfm2SlimMoELayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register Lfm2SlimMoELayer factory, reason: "
              << e.what() << std::endl;
  }
}

} // namespace causallm
