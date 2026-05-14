// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   bert_transformer.h
 * @date   29 April 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   Please refer to the following code :
 *  https://github.com/huggingface/transformers/blob/v4.52.3/src/transformers/models/bert/modeling_bert.py
 */

#include <bert_transformer.h>
#include <llm_util.hpp>
#include <model.h>

#include <app_context.h>

namespace causallm {

namespace {
/**
 * @brief Convert a float to a string with enough precision to preserve
 * values as small as BERT's layer_norm_eps (1e-12).
 */
std::string toStringPrecise(float v) {
  std::ostringstream oss;
  oss << std::setprecision(20) << v;
  return oss.str();
}
} // namespace

json &BertTransformer::sanitizeConfig(json &cfg) {
  if (!cfg.contains("rope_theta")) {
    cfg["rope_theta"] = 0u;
  }

  if (!cfg.contains("rms_norm_eps")) {
    float layer_norm_eps = cfg.value("layer_norm_eps", 1e-12f);
    cfg["rms_norm_eps"] = layer_norm_eps;
  }

  if (!cfg.contains("tie_word_embeddings")) {
    cfg["tie_word_embeddings"] = false;
  }

  if (!cfg.contains("use_bidirectional_attention") &&
      !cfg.contains("is_causal")) {
    cfg["is_causal"] = false;
  }

  if (!cfg.contains("num_key_value_heads")) {
    cfg["num_key_value_heads"] = cfg["num_attention_heads"];
  }

  return cfg;
}

void BertTransformer::setupParameters(json &cfg, json &generation_cfg,
                                      json &nntr_cfg) {
  Transformer::setupParameters(cfg, generation_cfg, nntr_cfg);
}

void BertTransformer::initialize() {
  registerCustomLayers();

  model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
  model->setProperty({withKey("batch_size", BATCH_SIZE), withKey("epochs", "1"),
                      withKey("model_tensor_type", MODEL_TENSOR_TYPE)});

  auto [inputs, output] = constructBertGraph();
  std::vector<Tensor> outputs = {output};
  if (model->compile(inputs, outputs, ml::train::ExecutionMode::INFERENCE)) {
    throw std::invalid_argument("Model compilation failed.");
  }

  is_initialized = true;
}

std::pair<Tensor, Tensor> BertTransformer::constructModel() {
  auto [inputs, output] = constructBertGraph();
  return {inputs.front(), output};
}

std::pair<std::vector<Tensor>, Tensor> BertTransformer::constructBertGraph() {
  /** --------- Inputs --------- */
  Tensor input({1, 1, 1, static_cast<unsigned int>(INIT_SEQ_LEN)}, "input0");
  Tensor position_ids({1, 1, 1, static_cast<unsigned int>(INIT_SEQ_LEN)},
                      "position_ids");
  Tensor token_type_ids({1, 1, 1, static_cast<unsigned int>(INIT_SEQ_LEN)},
                        "token_type_ids");

  /** --------- Token / Position / TokenType Embeddings --------- */
  const std::string embedding_type =
    TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "embedding_layer";

  LayerHandle word_embedding(createLayer(
    embedding_type,
    {withKey("name", "embedding0"), withKey("in_dim", NUM_VOCAB),
     withKey("weight_dtype", EMBEDDING_DTYPE), withKey("out_dim", DIM)}));
  Tensor word = word_embedding(input);

  LayerHandle position_embedding(
    createLayer("embedding_layer", {withKey("name", "position_embedding"),
                                    withKey("in_dim", MAX_POSITION_EMBEDDINGS),
                                    withKey("weight_dtype", EMBEDDING_DTYPE),
                                    withKey("out_dim", DIM)}));
  Tensor position = position_embedding(position_ids);

  LayerHandle token_type_embedding(
    createLayer("embedding_layer", {withKey("name", "token_type_embedding"),
                                    withKey("in_dim", TYPE_VOCAB_SIZE),
                                    withKey("weight_dtype", EMBEDDING_DTYPE),
                                    withKey("out_dim", DIM)}));
  Tensor token_type = token_type_embedding(token_type_ids);

  LayerHandle embedding_sum(
    createLayer("addition", {withKey("name", "embedding_sum")}));
  Tensor h = embedding_sum({word, position, token_type});

  LayerHandle embedding_norm(createLayer(
    "layer_normalization", {withKey("name", "embedding_norm"),
                            withKey("epsilon", toStringPrecise(NORM_EPS)),
                            withKey("axis", 3), withKey("packed", "false")}));
  h = embedding_norm(h);

  /** --------- Encoder blocks --------- */
  for (int i = 0; i < NUM_LAYERS; ++i) {
    h = createTransformerDecoderBlock(i, h);
  }

  return {{input, position_ids, token_type_ids}, h};
}

Tensor BertTransformer::createTransformerDecoderBlock(const int layer_id,
                                                      Tensor input) {

  // Self-attention sub-block
  Tensor att_out = createAttention(layer_id, INIT_SEQ_LEN, NUM_HEADS, HEAD_DIM,
                                   input, input, input);

  // Residual (input + attention_out) + post LayerNorm
  LayerHandle attention_res(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention_res")}));
  Tensor attention_residual = attention_res({input, att_out});

  LayerHandle attention_norm(createLayer(
    "layer_normalization",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention_norm"),
     withKey("epsilon", toStringPrecise(NORM_EPS)), withKey("axis", 3),
     withKey("packed", "false")}));
  Tensor attention_normed = attention_norm(attention_residual);

  // Feed-forward sub-block
  auto ffn_layers =
    createMlp(layer_id, DIM, INTERMEDIATE_SIZE, attention_normed);

  // Residual (normed + ffn_down) + post LayerNorm
  LayerHandle ffn_res(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_res")}));
  Tensor ffn_residual = ffn_res({attention_normed, ffn_layers});

  LayerHandle ffn_norm(createLayer(
    "layer_normalization",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_norm"),
     withKey("epsilon", toStringPrecise(NORM_EPS)), withKey("axis", 3),
     withKey("packed", "false")}));

  return ffn_norm(ffn_residual);
}

Tensor BertTransformer::createAttention(const int layer_id, int seq_len,
                                        int n_heads, int head_dim, Tensor query,
                                        Tensor key, Tensor value) {

  auto Q = "layer" + std::to_string(layer_id) + "_wq";
  auto K = "layer" + std::to_string(layer_id) + "_wk";
  auto V = "layer" + std::to_string(layer_id) + "_wv";
  auto A = "layer" + std::to_string(layer_id) + "_attention";
  auto O = "layer" + std::to_string(layer_id) + "_attention_out";

  // Q layer (bias enabled for BERT)
  LayerHandle wq(createLayer(
    "fully_connected",
    {withKey("name", Q), withKey("unit", head_dim * n_heads),
     withKey("disable_bias", "false"), withKey("weight_initializer", "ones")}));
  Tensor q = wq(query);

  // K layer (bias enabled for BERT)
  LayerHandle wk(createLayer(
    "fully_connected",
    {withKey("name", K), withKey("unit", head_dim * n_heads / GQA_SIZE),
     withKey("disable_bias", "false"), withKey("weight_initializer", "ones")}));
  Tensor k = wk(key);

  // V layer (bias enabled for BERT)
  LayerHandle wv(createLayer(
    "fully_connected",
    {withKey("name", V), withKey("unit", head_dim * n_heads / GQA_SIZE),
     withKey("disable_bias", "false"), withKey("weight_initializer", "ones")}));
  Tensor v = wv(value);

  // Attention core layer (bidirectional, no RoPE)
  std::vector<std::string> a_params = {
    withKey("name", A),
    withKey("num_heads", n_heads),
    withKey("num_heads_kv", n_heads / GQA_SIZE),
    withKey("max_timestep", std::to_string(INIT_SEQ_LEN)),
    withKey("rope_theta", ROPE_THETA),
    withKey("is_causal", "false")};
  LayerHandle mha(createLayer("mha_core", a_params));
  Tensor a = mha({q, k, v});

  // O layer (bias enabled for BERT)
  LayerHandle wo(
    createLayer("fully_connected", {withKey("name", O), withKey("unit", DIM),
                                    withKey("disable_bias", "false"),
                                    withKey("weight_initializer", "ones")}));

  return wo(a);
}

Tensor BertTransformer::createMlp(const int layer_id, int dim, int hidden_dim,
                                  Tensor input) {
  LayerHandle fc1(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_fc1"),
     withKey("unit", hidden_dim), withKey("disable_bias", "false"),
     withKey("weight_initializer", "ones")}));
  Tensor fc1_out = fc1(input);

  LayerHandle act(createLayer(
    "activation",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_act"),
     withKey("activation", "gelu")}));
  Tensor activated = act(fc1_out);

  LayerHandle down(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
     withKey("unit", dim), withKey("disable_bias", "false"),
     withKey("weight_initializer", "ones")}));

  return down(activated);
}

void BertTransformer::run(const WSTR prompt, bool do_sample,
                          const WSTR system_prompt, const WSTR tail_prompt,
                          bool log_output) {

  try {
    std::vector<float *> results = encode(prompt, system_prompt, tail_prompt);

    if (log_output) {

      std::cout << "Embedding Result (" << BATCH_SIZE
                << " batch(es)):" << std::endl;
      for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
        std::cout << "Batch " << b << ": [";
        // Print first few elements as sample
        int print_dim = (DIM > 10) ? 10 : DIM;
        for (int i = 0; i < print_dim; ++i) {
          std::cout << results[0][b * DIM + i]
                    << (i == print_dim - 1 ? "" : ", ");
        }
        if (DIM > 10)
          std::cout << ", ...";
        std::cout << "] (Total DIM: " << DIM << ")" << std::endl;
      }
    }

    // output should be deallocated after use.
    for (auto out : results) {
      delete[] out;
    }
  } catch (const std::exception &e) {
    std::cerr << "Error during embedding run: " << e.what() << std::endl;
  }
}

void BertTransformer::registerCustomLayers() {
  Transformer::registerCustomLayers();
}

} // namespace causallm
