// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   qwen35_causallm.cpp
 * @brief  Qwen3.5 text-only causal language model.
 */

#include <qwen35_causallm.h>

#include <app_context.h>
#include <engine.h>
#include <llm_util.hpp>
#include <model.h>
#include <qwen35_layers.h>
#include <reshaped_rms_norm.h>

namespace causallm {

namespace {

json &text_config(json &cfg) {
  if (cfg.contains("text_config") && cfg["text_config"].is_object())
    return cfg["text_config"];
  return cfg;
}

} // namespace

Qwen35CausalLM::Qwen35CausalLM(json &cfg, json &generation_cfg,
                               json &nntr_cfg) :
  Transformer(cfg, generation_cfg, nntr_cfg, ModelType::CAUSALLM),
  CausalLM(cfg, generation_cfg, nntr_cfg) {
  setupQwen35Parameters(cfg);
}

void Qwen35CausalLM::setupQwen35Parameters(json &cfg) {
  json &tcfg = text_config(cfg);

  layer_types_ = tcfg["layer_types"].get<std::vector<std::string>>();
  linear_num_key_heads = tcfg["linear_num_key_heads"].get<unsigned int>();
  linear_num_value_heads = tcfg["linear_num_value_heads"].get<unsigned int>();
  linear_key_head_dim = tcfg["linear_key_head_dim"].get<unsigned int>();
  linear_value_head_dim = tcfg["linear_value_head_dim"].get<unsigned int>();
  linear_conv_kernel_dim = tcfg["linear_conv_kernel_dim"].get<unsigned int>();

  if (tcfg.contains("rope_parameters") &&
      tcfg["rope_parameters"].contains("partial_rotary_factor")) {
    partial_rotary_factor =
      tcfg["rope_parameters"]["partial_rotary_factor"].get<float>();
  }
}

std::vector<LayerHandle> Qwen35CausalLM::createQwen35FullAttention(
  const int layer_id, std::string input_name) {
  std::vector<LayerHandle> layers;

  const unsigned int q_width = NUM_HEADS * HEAD_DIM;
  const unsigned int kv_width = NUM_KEY_VALUE_HEADS * HEAD_DIM;

  const auto Q = "layer" + std::to_string(layer_id) + "_wq";
  const auto K = "layer" + std::to_string(layer_id) + "_wk";
  const auto V = "layer" + std::to_string(layer_id) + "_wv";
  const auto Q_query = "layer" + std::to_string(layer_id) + "_wq_query";
  const auto Q_gate = "layer" + std::to_string(layer_id) + "_wq_gate";
  const auto Q_gate_act =
    "layer" + std::to_string(layer_id) + "_wq_gate_sigmoid";
  const auto Q_norm = "layer" + std::to_string(layer_id) + "_q_norm";
  const auto K_norm = "layer" + std::to_string(layer_id) + "_k_norm";
  const auto A = "layer" + std::to_string(layer_id) + "_attention";
  const auto A_gated = "layer" + std::to_string(layer_id) + "_attention_gated";
  const auto O = "layer" + std::to_string(layer_id) + "_attention_out";

  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", Q), withKey("unit", q_width * 2),
     withKey("disable_bias", "true"), withKey("input_layers", input_name),
     withKey("weight_initializer", "ones")}));
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", K), withKey("unit", kv_width),
     withKey("disable_bias", "true"), withKey("input_layers", input_name),
     withKey("weight_initializer", "ones")}));
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", V), withKey("unit", kv_width),
     withKey("disable_bias", "true"), withKey("input_layers", input_name),
     withKey("weight_initializer", "ones")}));

  layers.push_back(createLayer(
    "qwen35_head_pair_split",
    {withKey("name", Q_query), withKey("input_layers", Q),
     withKey("feature_size", HEAD_DIM), withKey("select_index", "0")}));
  layers.push_back(createLayer(
    "qwen35_head_pair_split",
    {withKey("name", Q_gate), withKey("input_layers", Q),
     withKey("feature_size", HEAD_DIM), withKey("select_index", "1")}));

  layers.push_back(createLayer(
    "reshaped_rms_norm",
    {withKey("name", Q_norm), withKey("input_layers", Q_query),
     withKey("packed", "false"), withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("feature_size", std::to_string(HEAD_DIM))}));
  layers.push_back(createLayer(
    "reshaped_rms_norm",
    {withKey("name", K_norm), withKey("input_layers", K),
     withKey("packed", "false"), withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("feature_size", std::to_string(HEAD_DIM))}));

  layers.push_back(createLayer(
    "mha_core",
    {withKey("name", A), withKey("num_heads", NUM_HEADS),
     withKey("num_heads_kv", NUM_KEY_VALUE_HEADS),
     withKey("max_timestep", std::to_string(INIT_SEQ_LEN + NUM_TO_GENERATE)),
     withKey("sliding_window", SLIDING_WINDOW),
     withKey("rope_theta", ROPE_THETA),
     withKey("max_position_embeddings", MAX_POSITION_EMBEDDINGS),
     withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
     withKey("partial_rotary_factor", partial_rotary_factor),
     withKey("is_causal", IS_CAUSAL ? "true" : "false"),
     withKey("input_layers", {Q_norm, K_norm, V})}));

  layers.push_back(createLayer(
    "activation",
    {withKey("name", Q_gate_act), withKey("input_layers", Q_gate),
     withKey("activation", "sigmoid")}));
  layers.push_back(createLayer(
    "multiply",
    {withKey("name", A_gated), withKey("input_layers", {A, Q_gate_act})}));
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", O), withKey("unit", DIM),
     withKey("disable_bias", "true"), withKey("input_layers", A_gated),
     withKey("weight_initializer", "ones")}));

  return layers;
}

std::vector<LayerHandle> Qwen35CausalLM::createQwen35LinearAttention(
  const int layer_id, std::string input_name) {
  std::vector<LayerHandle> layers;

  const unsigned int key_dim = linear_num_key_heads * linear_key_head_dim;
  const unsigned int value_dim =
    linear_num_value_heads * linear_value_head_dim;
  const unsigned int conv_dim = key_dim * 2 + value_dim;

  const auto prefix = "layer" + std::to_string(layer_id) + "_linear_attn";
  const auto QKV = prefix + "_qkv";
  const auto Conv = prefix + "_conv";
  const auto ConvAct = prefix + "_conv_silu";
  const auto Q = prefix + "_q";
  const auto K = prefix + "_k";
  const auto V = prefix + "_v";
  const auto QNorm = prefix + "_q_l2_norm";
  const auto KNorm = prefix + "_k_l2_norm";
  const auto Z = prefix + "_z";
  const auto ZAct = prefix + "_z_silu";
  const auto Beta = prefix + "_beta";
  const auto BetaAct = prefix + "_beta_sigmoid";
  const auto A = prefix + "_decay";
  const auto ABias = prefix + "_decay_bias";
  const auto ASoftplus = prefix + "_decay_softplus";
  const auto AScale = prefix + "_decay_scale";
  const auto Core = prefix + "_core";
  const auto CoreNorm = prefix + "_norm";
  const auto Gated = prefix + "_gated";
  const auto O = "layer" + std::to_string(layer_id) + "_attention_out";

  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", QKV), withKey("unit", conv_dim),
     withKey("disable_bias", "true"), withKey("input_layers", input_name),
     withKey("weight_initializer", "ones")}));
  layers.push_back(createLayer(
    "qwen35_causal_depthwise_conv1d",
    {withKey("name", Conv), withKey("input_layers", QKV),
     withKey("linear_conv_kernel_dim", linear_conv_kernel_dim)}));
  layers.push_back(createLayer(
    "activation",
    {withKey("name", ConvAct), withKey("input_layers", Conv),
     withKey("activation", "swish")}));

  layers.push_back(createLayer(
    "slice", {withKey("name", Q), withKey("input_layers", ConvAct),
              withKey("axis", "3"), withKey("start_index", "1"),
              withKey("end_index", std::to_string(key_dim + 1))}));
  layers.push_back(createLayer(
    "slice", {withKey("name", K), withKey("input_layers", ConvAct),
              withKey("axis", "3"),
              withKey("start_index", std::to_string(key_dim + 1)),
              withKey("end_index", std::to_string(key_dim * 2 + 1))}));
  layers.push_back(createLayer(
    "slice", {withKey("name", V), withKey("input_layers", ConvAct),
              withKey("axis", "3"),
              withKey("start_index", std::to_string(key_dim * 2 + 1)),
              withKey("end_index", std::to_string(conv_dim + 1))}));

  layers.push_back(createLayer(
    "reshaped_l2_norm",
    {withKey("name", QNorm), withKey("input_layers", Q),
     withKey("feature_size", linear_key_head_dim),
     withKey("epsilon", "1e-6")}));
  layers.push_back(createLayer(
    "reshaped_l2_norm",
    {withKey("name", KNorm), withKey("input_layers", K),
     withKey("feature_size", linear_key_head_dim),
     withKey("epsilon", "1e-6")}));

  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", Z), withKey("unit", value_dim),
     withKey("disable_bias", "true"), withKey("input_layers", input_name),
     withKey("weight_initializer", "ones")}));
  layers.push_back(createLayer(
    "activation",
    {withKey("name", ZAct), withKey("input_layers", Z),
     withKey("activation", "swish")}));

  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", Beta), withKey("unit", linear_num_value_heads),
     withKey("disable_bias", "true"), withKey("input_layers", input_name),
     withKey("weight_initializer", "ones")}));
  layers.push_back(createLayer(
    "activation",
    {withKey("name", BetaAct), withKey("input_layers", Beta),
     withKey("activation", "sigmoid")}));

  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", A), withKey("unit", linear_num_value_heads),
     withKey("disable_bias", "true"), withKey("input_layers", input_name),
     withKey("weight_initializer", "ones")}));
  layers.push_back(createLayer(
    "feature_bias",
    {withKey("name", ABias), withKey("input_layers", A)}));
  layers.push_back(createLayer(
    "activation",
    {withKey("name", ASoftplus), withKey("input_layers", ABias),
     withKey("activation", "softplus")}));
  layers.push_back(createLayer(
    "feature_scale",
    {withKey("name", AScale), withKey("input_layers", ASoftplus)}));

  layers.push_back(createLayer(
    "qwen35_gated_delta_core",
    {withKey("name", Core),
     withKey("input_layers", {QNorm, KNorm, V, BetaAct, AScale}),
     withKey("linear_num_key_heads", linear_num_key_heads),
     withKey("linear_num_value_heads", linear_num_value_heads),
     withKey("linear_key_head_dim", linear_key_head_dim),
     withKey("linear_value_head_dim", linear_value_head_dim)}));
  layers.push_back(createLayer(
    "reshaped_rms_norm",
    {withKey("name", CoreNorm), withKey("input_layers", Core),
     withKey("packed", "false"), withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("feature_size", linear_value_head_dim)}));
  layers.push_back(createLayer(
    "multiply",
    {withKey("name", Gated), withKey("input_layers", {CoreNorm, ZAct})}));
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", O), withKey("unit", DIM),
     withKey("disable_bias", "true"), withKey("input_layers", Gated),
     withKey("weight_initializer", "ones")}));

  return layers;
}

std::vector<LayerHandle>
Qwen35CausalLM::createTransformerDecoderBlock(const int layer_id,
                                              std::string input_name) {
  std::vector<LayerHandle> layers;

  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention_norm"),
     withKey("input_layers", input_name),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));

  std::vector<LayerHandle> token_mixer;
  if (layer_types_.at(layer_id) == "linear_attention") {
    token_mixer = createQwen35LinearAttention(
      layer_id, "layer" + std::to_string(layer_id) + "_attention_norm");
  } else if (layer_types_.at(layer_id) == "full_attention") {
    token_mixer = createQwen35FullAttention(
      layer_id, "layer" + std::to_string(layer_id) + "_attention_norm");
  } else {
    throw std::invalid_argument("Unsupported Qwen3.5 layer type: " +
                                layer_types_.at(layer_id));
  }
  layers.insert(layers.end(), token_mixer.begin(), token_mixer.end());

  layers.push_back(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_decoder_add"),
     withKey("input_layers", input_name + ",layer" + std::to_string(layer_id) +
                               "_attention_out")}));

  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_norm"),
     withKey("input_layers",
             "layer" + std::to_string(layer_id) + "_decoder_add"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));

  auto ffn_layer = createMlp(layer_id, DIM, INTERMEDIATE_SIZE,
                             "layer" + std::to_string(layer_id) + "_ffn_norm");
  layers.insert(layers.end(), ffn_layer.begin(), ffn_layer.end());

  layers.push_back(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_decoder_output"),
     withKey("input_layers", "layer" + std::to_string(layer_id) +
                               "_decoder_add,layer" + std::to_string(layer_id) +
                               "_ffn_down")}));

  return layers;
}

void Qwen35CausalLM::registerCustomLayers() {
  CausalLM::registerCustomLayers();

  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

  try {
    app_context->registerFactory(
      nntrainer::createLayer<causallm::ReshapedRMSNormLayer>);
    app_context->registerFactory(
      nntrainer::createLayer<causallm::ReshapedL2NormLayer>);
    app_context->registerFactory(
      nntrainer::createLayer<causallm::FeatureBiasLayer>);
    app_context->registerFactory(
      nntrainer::createLayer<causallm::FeatureScaleLayer>);
    app_context->registerFactory(
      nntrainer::createLayer<causallm::HeadPairSplitLayer>);
    app_context->registerFactory(
      nntrainer::createLayer<causallm::Qwen35CausalDepthwiseConv1DLayer>);
    app_context->registerFactory(
      nntrainer::createLayer<causallm::Qwen35GatedDeltaCoreLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }
}

} // namespace causallm
