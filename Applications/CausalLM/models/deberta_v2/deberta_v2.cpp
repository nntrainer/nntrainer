// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   deberta_v2.cpp
 * @date   14 January 2026
 * @brief  DeBERTa V2 encoder model for SentenceTransformer embeddings
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   Please refer to the following code :
 * https://github.com/huggingface/transformers/blob/5c1c72b/src/transformers/models/deberta_v2/modeling_deberta_v2.py
 */

#include <deberta_attention_layer.h>
#include <deberta_v2.h>
#include <shared_fully_connected_layer.h>

#include <app_context.h>
#include <engine.h>
#include <llm_util.hpp>
#include <model.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace causallm {

namespace {

/**
 * @brief Convert a float to a string with enough precision to preserve eps.
 */
std::string toStringPrecise(float v) {
  std::ostringstream oss;
  oss << std::setprecision(20) << v;
  return oss.str();
}

/**
 * @brief Parse DeBERTa pos_att_type config field into a string vector.
 */
std::vector<std::string> parsePosAttType(const json &cfg) {
  std::vector<std::string> result;
  if (!cfg.contains("pos_att_type") || cfg["pos_att_type"].is_null())
    return result;

  if (cfg["pos_att_type"].is_array()) {
    result = cfg["pos_att_type"].get<std::vector<std::string>>();
  } else if (cfg["pos_att_type"].is_string()) {
    std::stringstream ss(cfg["pos_att_type"].get<std::string>());
    std::string token;
    while (std::getline(ss, token, '|')) {
      token.erase(0, token.find_first_not_of(" \t"));
      token.erase(token.find_last_not_of(" \t") + 1);
      if (!token.empty())
        result.push_back(token);
    }
  }

  return result;
}

} // namespace

json &DebertaV2::sanitizeConfig(json &cfg) {
  if (!cfg.contains("rope_theta")) {
    cfg["rope_theta"] = 0u;
  }

  if (!cfg.contains("rms_norm_eps")) {
    cfg["rms_norm_eps"] = cfg.value("layer_norm_eps", 1e-7f);
  }

  if (!cfg.contains("tie_word_embeddings")) {
    cfg["tie_word_embeddings"] = false;
  }

  if (!cfg.contains("num_key_value_heads")) {
    cfg["num_key_value_heads"] = cfg["num_attention_heads"];
  }

  if (!cfg.contains("use_bidirectional_attention") &&
      !cfg.contains("is_causal")) {
    cfg["is_causal"] = false;
  }

  return cfg;
}

void DebertaV2::setupParameters(json &cfg, json &generation_cfg,
                                json &nntr_cfg) {
  SentenceTransformer::setupParameters(cfg, generation_cfg, nntr_cfg);

  const json *encoder_cfg = &cfg;
  try {
    if (nntr_cfg.contains("tokenizer_file")) {
      std::filesystem::path tokenizer_path(
        nntr_cfg["tokenizer_file"].get<std::string>());
      std::filesystem::path model_path = tokenizer_path.parent_path();
      std::filesystem::path encoder_config_path =
        model_path / "encoder_config" / "config.json";

      if (std::filesystem::exists(encoder_config_path)) {
        static json loaded_encoder_cfg;
        loaded_encoder_cfg =
          causallm::LoadJsonFile(encoder_config_path.string());
        encoder_cfg = &loaded_encoder_cfg;
      }
    }
  } catch (const std::exception &e) {
    std::cerr << "Warning: failed to load DeBERTa encoder_config: " << e.what()
              << std::endl;
  }

  MAX_RELATIVE_POSITIONS = encoder_cfg->value(
    "max_relative_positions", static_cast<int>(MAX_POSITION_EMBEDDINGS));
  if (MAX_RELATIVE_POSITIONS == -1)
    MAX_RELATIVE_POSITIONS = static_cast<int>(MAX_POSITION_EMBEDDINGS);

  const auto pos_att_type = parsePosAttType(*encoder_cfg);
  C2P = std::find(pos_att_type.begin(), pos_att_type.end(), "c2p") !=
        pos_att_type.end();
  P2C = std::find(pos_att_type.begin(), pos_att_type.end(), "p2c") !=
        pos_att_type.end();

  SHARE_ATT_KEY = encoder_cfg->value("share_att_key", true);
  RELATIVE_ATTENTION = encoder_cfg->value("relative_attention", true);
  POSITION_BUCKETS = encoder_cfg->value("position_buckets", -1);

  std::string norm_rel_ebd =
    encoder_cfg->value("norm_rel_ebd", std::string("none"));
  std::transform(norm_rel_ebd.begin(), norm_rel_ebd.end(), norm_rel_ebd.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  NORM_REL_EBD = norm_rel_ebd.find("layer_norm") != std::string::npos;
}

std::pair<Tensor, Tensor> DebertaV2::constructTransformerModule() {
  Tensor x({1, 1, 1, static_cast<unsigned int>(INIT_SEQ_LEN)}, "input0");

  LayerHandle embedding(createLayer(
    "embedding_layer",
    {withKey("name", "embedding0"), withKey("in_dim", NUM_VOCAB),
     withKey("weight_dtype", EMBEDDING_DTYPE), withKey("out_dim", DIM),
     withKey("scale", std::to_string(EMBEDDING_SCALE))}));
  Tensor h = embedding(x);

  LayerHandle embedding_norm(createLayer(
    "layer_normalization", {withKey("name", "embeddings_norm"),
                            withKey("epsilon", toStringPrecise(NORM_EPS)),
                            withKey("axis", 3), withKey("packed", "false")}));
  h = embedding_norm(h);

  const int rel_embed_size =
    (POSITION_BUCKETS > 0) ? POSITION_BUCKETS * 2
                           : ((MAX_RELATIVE_POSITIONS < 1)
                                ? static_cast<int>(MAX_POSITION_EMBEDDINGS) * 2
                                : MAX_RELATIVE_POSITIONS * 2);

  const std::string rel_shape =
    "1:1:" + std::to_string(rel_embed_size) + ":" + std::to_string(DIM);
  const std::string rel_input_shape =
    "1:1:" + std::to_string(INIT_SEQ_LEN) + ":" + std::to_string(DIM);
  LayerHandle rel_embedding(createLayer(
    "weight",
    {withKey("name", "rel_embeddings"), withKey("weight_name", "weight"),
     withKey("dim", rel_shape), withKey("input_shape", rel_input_shape),
     withKey("weight_initializer", "none")}));
  Tensor rel = rel_embedding(h);

  if (NORM_REL_EBD) {
    LayerHandle rel_embedding_norm(createLayer(
      "layer_normalization", {withKey("name", "rel_embeddings_norm"),
                              withKey("epsilon", toStringPrecise(NORM_EPS)),
                              withKey("axis", 3), withKey("packed", "false")}));
    rel = rel_embedding_norm(rel);
  }

  for (int i = 0; i < NUM_LAYERS; ++i) {
    h = createDebertaLayer(i, h, rel);
  }

  return {x, h};
}

Tensor DebertaV2::createDebertaLayer(const int layer_id, Tensor input,
                                     Tensor rel_embeddings) {
  const std::string prefix = "layer" + std::to_string(layer_id);

  Tensor att_out = createDebertaV2Attention(layer_id, input, rel_embeddings);

  LayerHandle attention_res(
    createLayer("addition", {withKey("name", prefix + "_attention_add")}));
  Tensor attention_residual = attention_res({input, att_out});

  LayerHandle attention_norm(createLayer(
    "layer_normalization", {withKey("name", prefix + "_attention_norm"),
                            withKey("epsilon", toStringPrecise(NORM_EPS)),
                            withKey("axis", 3), withKey("packed", "false")}));
  Tensor attention_normed = attention_norm(attention_residual);

  LayerHandle intermediate(createLayer(
    "fully_connected",
    {withKey("name", prefix + "_intermediate"),
     withKey("unit", INTERMEDIATE_SIZE), withKey("disable_bias", "false"),
     withKey("activation", "gelu"), withKey("weight_initializer", "ones")}));
  Tensor intermediate_out = intermediate(attention_normed);

  LayerHandle output_dense(createLayer(
    "fully_connected",
    {withKey("name", prefix + "_output_dense"), withKey("unit", DIM),
     withKey("disable_bias", "false"), withKey("weight_initializer", "ones")}));
  Tensor output_dense_out = output_dense(intermediate_out);

  LayerHandle output_res(
    createLayer("addition", {withKey("name", prefix + "_output_add")}));
  Tensor output_residual = output_res({attention_normed, output_dense_out});

  LayerHandle output_norm(createLayer(
    "layer_normalization", {withKey("name", prefix + "_output"),
                            withKey("epsilon", toStringPrecise(NORM_EPS)),
                            withKey("axis", 3), withKey("packed", "false")}));

  return output_norm(output_residual);
}

Tensor DebertaV2::createDebertaV2Attention(const int layer_id, Tensor input,
                                           Tensor rel_embeddings) {
  const std::string prefix = "layer" + std::to_string(layer_id);
  const std::string Q = prefix + "_wq";
  const std::string K = prefix + "_wk";
  const std::string V = prefix + "_wv";
  const std::string A = prefix + "_attention";
  const std::string O = prefix + "_attention_out";

  LayerHandle query(createLayer("shared_fully_connected",
                                {withKey("name", Q), withKey("unit", DIM),
                                 withKey("disable_bias", "false"),
                                 withKey("weight_initializer", "none")}));
  Tensor q = query(input);

  LayerHandle key(createLayer("shared_fully_connected",
                              {withKey("name", K), withKey("unit", DIM),
                               withKey("disable_bias", "false"),
                               withKey("weight_initializer", "none")}));
  Tensor k = key(input);

  LayerHandle value(
    createLayer("fully_connected", {withKey("name", V), withKey("unit", DIM),
                                    withKey("disable_bias", "false"),
                                    withKey("weight_initializer", "ones")}));
  Tensor v = value(input);

  std::vector<Tensor> attention_inputs = {q, k, v};

  if (P2C) {
    LayerHandle rel_query(createLayer(
      "shared_fully_connected",
      {withKey("name", prefix + "_wq_rel"), withKey("unit", DIM),
       withKey("shared_from", Q), withKey("shared_mode", "true"),
       withKey("full_input_range", "true"), withKey("disable_bias", "false"),
       withKey("weight_initializer", "none")}));
    attention_inputs.push_back(rel_query({rel_embeddings, q}));
  }

  if (C2P) {
    LayerHandle rel_key(createLayer(
      "shared_fully_connected",
      {withKey("name", prefix + "_wk_rel"), withKey("unit", DIM),
       withKey("shared_from", K), withKey("shared_mode", "true"),
       withKey("full_input_range", "true"), withKey("disable_bias", "false"),
       withKey("weight_initializer", "none")}));
    attention_inputs.push_back(rel_key({rel_embeddings, k}));
  }

  LayerHandle attention(createLayer(
    "deberta_attention",
    {withKey("name", A), withKey("num_heads", NUM_HEADS),
     withKey("max_position_embeddings", MAX_POSITION_EMBEDDINGS),
     withKey("max_relative_positions", MAX_RELATIVE_POSITIONS),
     withKey("c2p", C2P ? "true" : "false"),
     withKey("p2c", P2C ? "true" : "false"),
     withKey("share_att_key", SHARE_ATT_KEY ? "true" : "false"),
     withKey("position_buckets", POSITION_BUCKETS),
     withKey("relative_attention", RELATIVE_ATTENTION ? "true" : "false"),
     withKey("disable_bias", "false")}));
  Tensor a = attention(attention_inputs);

  LayerHandle output(
    createLayer("fully_connected", {withKey("name", O), withKey("unit", DIM),
                                    withKey("disable_bias", "false"),
                                    withKey("weight_initializer", "ones")}));

  return output(a);
}

std::vector<float *> DebertaV2::encode(const WSTR prompt,
                                       const WSTR system_prompt,
                                       const WSTR tail_prompt) {
  if (!is_initialized) {
    throw std::runtime_error("DebertaV2 is not initialized. Please call "
                             "initialize() before encode().");
  }

  std::string prompt_ = system_prompt + prompt + tail_prompt;
  ensureTokenizer(); // join the async load
  auto tokenized = tokenizer->Encode(prompt_, true);

  unsigned int input_len =
    std::min(static_cast<unsigned int>(tokenized.size()), INIT_SEQ_LEN);

  std::vector<float> input_sample(static_cast<size_t>(BATCH_SIZE) *
                                  INIT_SEQ_LEN);
  std::fill(input_sample.begin(), input_sample.end(), 0.0f);

  for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
    for (unsigned int i = 0; i < input_len; ++i) {
      input_sample[static_cast<size_t>(b) * INIT_SEQ_LEN + i] =
        static_cast<float>(tokenized[i]);
    }
  }

  std::vector<float *> input = {input_sample.data()};
  std::vector<float *> label;

  auto start_prefill = std::chrono::high_resolution_clock::now();
  auto output = model->incremental_inference(BATCH_SIZE, input, label,
                                             input_len, 0, input_len, false);
  auto finish_prefill = std::chrono::high_resolution_clock::now();
  auto prefill_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
    finish_prefill - start_prefill);

  std::cout << "prefill: " << input_len << " tokens, "
            << prefill_duration.count() << " ms";
  if (prefill_duration.count() > 0) {
    std::cout << ", " << ((double)input_len / prefill_duration.count() * 1000)
              << " TPS";
  }
  std::cout << '\n';

  return output;
}

void DebertaV2::registerCustomLayers() {
  SentenceTransformer::registerCustomLayers();

  const auto &ct_engine = nntrainer::Engine::Global();
  // cpu-context registration goes through Engine::registerLayerFactory below
  // (no static_cast to AppContext).

  try {
    ct_engine.registerLayerFactory(
      "cpu", nntrainer::createLayer<causallm::DebertaAttentionLayer>);
    ct_engine.registerLayerFactory(
      "cpu", nntrainer::createLayer<causallm::SharedFullyConnectedLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }
}

} // namespace causallm
