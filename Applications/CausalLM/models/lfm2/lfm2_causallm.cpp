// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   lfm2_causallm.cpp
 * @date   4 May 2026
 * @brief  This defines an LFM2 causal language model.
 */

#include <cmath>
#include <iostream>
#include <limits.h>
#include <stdexcept>

#include <app_context.h>
#include <engine.h>
#include <lfm2_causallm.h>
#include <lfm2_short_conv.h>
#include <llm_util.hpp>
#include <model.h>
#include <reshaped_rms_norm.h>

namespace causallm {

json &Lfm2Transformer::sanitizeConfig(json &cfg) {
  if (cfg.value("_nntrainer_lfm2_sanitized", false)) {
    return cfg;
  }

  if (!cfg.contains("num_attention_heads") && cfg.contains("num_heads")) {
    cfg["num_attention_heads"] = cfg["num_heads"];
  }

  if (!cfg.contains("tie_word_embeddings")) {
    cfg["tie_word_embeddings"] = cfg.value("tie_embedding", true);
  }

  if (!cfg.contains("rms_norm_eps")) {
    cfg["rms_norm_eps"] =
      cfg.value("norm_eps", cfg.value("block_norm_eps", 1e-5));
  }

  if (!cfg.contains("rope_theta")) {
    double theta = 1000000.0;
    if (cfg.contains("rope_parameters") && cfg["rope_parameters"].is_object() &&
        cfg["rope_parameters"].contains("rope_theta")) {
      theta = cfg["rope_parameters"]["rope_theta"].get<double>();
    }
    cfg["rope_theta"] = static_cast<unsigned int>(theta);
  }

  if (cfg.value("block_auto_adjust_ff_dim", false)) {
    int intermediate_size = cfg["intermediate_size"].get<int>();
    intermediate_size = static_cast<int>(2 * intermediate_size / 3);
    const double multiplier = cfg.value("block_ffn_dim_multiplier", 1.0);
    intermediate_size = static_cast<int>(multiplier * intermediate_size);
    const int multiple_of = cfg.value("block_multiple_of", 256);
    intermediate_size =
      multiple_of * ((intermediate_size + multiple_of - 1) / multiple_of);
    cfg["intermediate_size"] = intermediate_size;
  }

  cfg["_nntrainer_lfm2_sanitized"] = true;
  return cfg;
}

json &Lfm2Transformer::sanitizeGenerationConfig(json &gen_cfg,
                                                const json &cfg) {
  if (!gen_cfg.contains("eos_token_id") && cfg.contains("eos_token_id")) {
    gen_cfg["eos_token_id"] = cfg["eos_token_id"];
  }
  if (!gen_cfg.contains("bos_token_id") && cfg.contains("bos_token_id")) {
    gen_cfg["bos_token_id"] = cfg["bos_token_id"];
  }
  if (!gen_cfg.contains("pad_token_id") && cfg.contains("pad_token_id")) {
    gen_cfg["pad_token_id"] = cfg["pad_token_id"];
  }
  return gen_cfg;
}

void Lfm2Transformer::setupParameters(json &cfg, json &generation_cfg,
                                      json &nntr_cfg) {
  Transformer::setupParameters(cfg, generation_cfg, nntr_cfg);

  layer_types = cfg["layer_types"].get<std::vector<std::string>>();
  CONV_KERNEL_SIZE = cfg.value("conv_L_cache", 3);

  if (cfg.value("conv_bias", false)) {
    throw std::runtime_error("LFM2 conv_bias=true is not supported yet");
  }
}

std::vector<LayerHandle>
Lfm2Transformer::createTransformerDecoderBlock(const int layer_id,
                                               std::string input_name) {
  std::vector<LayerHandle> layers;

  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_operator_norm"),
     withKey("input_layers", input_name),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));

  const std::string operator_input =
    "layer" + std::to_string(layer_id) + "_operator_norm";
  const bool is_attention_layer =
    layer_id < static_cast<int>(layer_types.size()) &&
    layer_types[layer_id] == "full_attention";

  if (is_attention_layer) {
    auto att_layer =
      createAttention(layer_id, INIT_SEQ_LEN, NUM_HEADS, HEAD_DIM,
                      operator_input, operator_input, operator_input);
    layers.insert(layers.end(), att_layer.begin(), att_layer.end());
  } else {
    layers.push_back(createLayer(
      "lfm2_short_conv",
      {withKey("name", "layer" + std::to_string(layer_id) + "_short_conv"),
       withKey("input_layers", operator_input), withKey("unit", DIM),
       withKey("kernel_size", CONV_KERNEL_SIZE),
       withKey("weight_initializer", "ones"),
       withKey("weight_dtype", FC_LAYER_DTYPE)}));
  }

  const std::string operator_out =
    is_attention_layer ? "layer" + std::to_string(layer_id) + "_attention_out"
                       : "layer" + std::to_string(layer_id) + "_short_conv";

  layers.push_back(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_operator_add"),
     withKey("input_layers", input_name + "," + operator_out)}));

  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_norm"),
     withKey("input_layers",
             "layer" + std::to_string(layer_id) + "_operator_add"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));

  auto ffn_layer = createMlp(layer_id, DIM, INTERMEDIATE_SIZE,
                             "layer" + std::to_string(layer_id) + "_ffn_norm");
  layers.insert(layers.end(), ffn_layer.begin(), ffn_layer.end());

  layers.push_back(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_decoder_output"),
     withKey("input_layers", "layer" + std::to_string(layer_id) +
                               "_operator_add,layer" +
                               std::to_string(layer_id) + "_ffn_down")}));

  return layers;
}

std::vector<LayerHandle> Lfm2Transformer::createAttention(
  const int layer_id, int seq_len, int n_heads, int head_dim,
  std::string query_name, std::string key_name, std::string value_name) {
  std::vector<LayerHandle> layers;

  auto Q = "layer" + std::to_string(layer_id) + "_wq";
  auto Q_norm = "layer" + std::to_string(layer_id) + "_q_norm";
  auto K = "layer" + std::to_string(layer_id) + "_wk";
  auto K_norm = "layer" + std::to_string(layer_id) + "_k_norm";
  auto V = "layer" + std::to_string(layer_id) + "_wv";
  auto A = "layer" + std::to_string(layer_id) + "_attention";
  auto O = "layer" + std::to_string(layer_id) + "_attention_out";

  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", Q), withKey("unit", head_dim * n_heads),
     withKey("disable_bias", "true"), withKey("input_layers", query_name),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));

  layers.push_back(createLayer(
    "reshaped_rms_norm",
    {withKey("name", Q_norm), withKey("input_layers", Q),
     withKey("packed", "false"), withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("feature_size", std::to_string(head_dim))}));

  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", K), withKey("unit", head_dim * n_heads / GQA_SIZE),
     withKey("disable_bias", "true"), withKey("input_layers", key_name),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));

  layers.push_back(createLayer(
    "reshaped_rms_norm",
    {withKey("name", K_norm), withKey("input_layers", K),
     withKey("packed", "false"), withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("feature_size", std::to_string(head_dim))}));

  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", V), withKey("unit", head_dim * n_heads / GQA_SIZE),
     withKey("disable_bias", "true"), withKey("input_layers", value_name),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));

  layers.push_back(createLayer(
    "mha_core",
    {withKey("name", A), withKey("num_heads", n_heads),
     withKey("num_heads_kv", n_heads / GQA_SIZE),
     withKey("max_timestep", std::to_string(INIT_SEQ_LEN + NUM_TO_GENERATE)),
     withKey("sliding_window", UINT_MAX), withKey("rope_theta", ROPE_THETA),
     withKey("max_position_embeddings", MAX_POSITION_EMBEDDINGS),
     withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
     withKey("is_causal", IS_CAUSAL ? "true" : "false"),
     withKey("input_layers", {Q_norm, K_norm, V})}));

  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", O), withKey("unit", DIM),
     withKey("disable_bias", "true"), withKey("input_layers", A),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));

  return layers;
}

std::vector<LayerHandle> Lfm2Transformer::createMlp(const int layer_id, int dim,
                                                    int hidden_dim,
                                                    std::string input_name) {
  std::vector<LayerHandle> layers;

  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_gate"),
     withKey("unit", hidden_dim), withKey("disable_bias", "true"),
     withKey("input_layers", input_name), withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));

  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_up"),
     withKey("unit", hidden_dim), withKey("disable_bias", "true"),
     withKey("input_layers", input_name), withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));

  layers.push_back(createLayer(
    "swiglu",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_swiglu"),
     withKey("input_layers", "layer" + std::to_string(layer_id) +
                               "_ffn_gate,layer" + std::to_string(layer_id) +
                               "_ffn_up")}));

  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
     withKey("unit", dim), withKey("disable_bias", "true"),
     withKey("input_layers",
             "layer" + std::to_string(layer_id) + "_ffn_swiglu"),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));

  return layers;
}

void Lfm2Transformer::registerCustomLayers() {
  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

  try {
    app_context->registerFactory(
      nntrainer::createLayer<causallm::ReshapedRMSNormLayer>);
    app_context->registerFactory(
      nntrainer::createLayer<causallm::Lfm2ShortConvLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }
}

void Lfm2CausalLM::registerCustomLayers() {
  CausalLM::registerCustomLayers();
  Lfm2Transformer::registerCustomLayers();
}

} // namespace causallm
