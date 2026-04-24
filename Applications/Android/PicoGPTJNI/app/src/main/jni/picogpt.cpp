// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 Jijoong Moon <jijong.moon@samsung.com>
 *
 * @file   picogpt.cpp
 * @date   20 March 2023
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Picogpt Application for android
 *
 */

#include "picogpt.h"
#include "encoder.hpp"
#include "tensor_dim.h"
#include <algorithm>
#include <ctime>
#include <sstream>
#include <tensor_api.h>

const unsigned int BATCH_SIZE = 1;
const unsigned int NUM_LAYERS = 12;
const unsigned int NUM_HEADS = 12;
const unsigned int MODEL_DIM = 768;
const unsigned int FC_UNIT = 3072; // 768*4

const unsigned int NUM_VOCAB = 50257;
const unsigned int NUM_CTX = 1024;
const unsigned int NUM_TOKENS_TO_GENERATE = 40;

unsigned int init_input_seq_len;
const unsigned int MAX_TOKEN_LEN = 10 + NUM_TOKENS_TO_GENERATE;

bool fsu = false;
bool optimize = false;
bool optimize_attention = false;

template <typename T>
T unwrap(std::optional<T> &&value, const std::string &error_msg) {
  if (value.has_value()) {
    return value.value();
  } else {
    throw std::runtime_error(error_msg);
  }
}

/** cache loss values post training for test */
float training_loss = 0.0;
float validation_loss = 0.0;

ml::train::RunStats training;
ml::train::RunStats validation;
ModelHandle model;
bool stop = false;
std::string test_result = "";
std::string infer_result = "";
bool model_destroyed = true;
bool last = false;

ml::train::Model *createPicogpt() {
  using namespace ml::train;

  model = createModel(ModelType::NEURAL_NET);
  model->setProperty({"batch_size=" + std::to_string(BATCH_SIZE),
                      "memory_optimization=false",
                      fsu ? "fsu=true" : "fsu=false"});

  // Symbolic input tensors
  Tensor wte_in({1, 1, 1, MAX_TOKEN_LEN}, "wte_input");
  Tensor wpe_in({1, 1, 1, MAX_TOKEN_LEN}, "wpe_input");

  // Embedding layers
  LayerHandle wte = createLayer("embedding",
    {"name=wte", "in_dim=" + std::to_string(NUM_VOCAB),
     "out_dim=" + std::to_string(MODEL_DIM)});
  LayerHandle wpe = createLayer("embedding",
    {"name=wpe", "in_dim=" + std::to_string(NUM_CTX),
     "out_dim=" + std::to_string(MODEL_DIM)});
  auto wte_out = wte(wte_in);
  auto wpe_out = wpe(wpe_in);

  // Add embeddings
  LayerHandle add_emb = createLayer("Addition", {"name=add"});
  auto prev = add_emb({wte_out, wpe_out});

  for (unsigned int i = 0; i < NUM_LAYERS; ++i) {
    std::string prefix = "layer" + std::to_string(i);

    // Layer Norm 1
    LayerHandle ln1 = createLayer("layer_normalization",
      {"name=" + prefix + "/ln1", "axis=3", "epsilon=1e-5"});
    auto ln1_out = ln1(prev);

    Tensor attn_out;
    if (optimize) {
      // Per-head attention with separate Q/K/V projections
      std::vector<Tensor> attn_heads;

      for (unsigned int j = 0; j < NUM_HEADS; ++j) {
        unsigned int idx = NUM_HEADS - 1 - j;
        std::string head_prefix =
          prefix + "/multi_head_attention";

        LayerHandle v_fc = createLayer("fully_connected",
          {"name=" + head_prefix + "/v_fc" + std::to_string(idx),
           "unit=" + std::to_string(MODEL_DIM / NUM_HEADS)});
        LayerHandle k_fc = createLayer("fully_connected",
          {"name=" + head_prefix + "/k_fc" + std::to_string(idx),
           "unit=" + std::to_string(MODEL_DIM / NUM_HEADS)});
        LayerHandle q_fc = createLayer("fully_connected",
          {"name=" + head_prefix + "/q_fc" + std::to_string(idx),
           "unit=" + std::to_string(MODEL_DIM / NUM_HEADS)});

        auto v = v_fc(ln1_out);
        auto k = k_fc(ln1_out);
        auto q = q_fc(ln1_out);

        LayerHandle attn = createLayer("attention",
          {"name=" + head_prefix + "/attention" + std::to_string(idx),
           "scaled_dot_product=true", "causal_mask=true"});
        attn_heads.push_back(attn({q, v, k}));
      }

      // Reverse so concat order is attention0, attention1, ..., attention11
      std::reverse(attn_heads.begin(), attn_heads.end());

      LayerHandle concat = createLayer("concat",
        {"name=" + prefix + "/multi_head_attention/concat", "axis=3"});
      auto concat_out = concat(attn_heads);

      LayerHandle attn_fc = createLayer("fully_connected",
        {"name=" + prefix + "/multi_head_attention/fc",
         "unit=" + std::to_string(MODEL_DIM)});
      auto fc_out = attn_fc(concat_out);

      LayerHandle identity = createLayer("identity",
        {"name=" + prefix + "/multi_head_attention"});
      attn_out = identity(fc_out);
    } else {
      LayerHandle mha = createLayer("multi_head_attention",
        {"name=" + prefix + "/multi_head_attention",
         "num_heads=" + std::to_string(NUM_HEADS)});
      attn_out = mha({ln1_out, ln1_out, ln1_out});
    }

    // Skip connection 1: prev + attention output
    LayerHandle add1 = createLayer("Addition",
      {"name=" + prefix + "/add1"});
    auto add1_out = add1({prev, attn_out});

    // Layer Norm 2
    LayerHandle ln2 = createLayer("layer_normalization",
      {"name=" + prefix + "/ln2", "axis=3", "epsilon=1e-5"});
    auto ln2_out = ln2(add1_out);

    // FFN
    LayerHandle fc1 = createLayer("fully_connected",
      {"name=" + prefix + "/fc1",
       "unit=" + std::to_string(FC_UNIT), "activation=gelu"});
    auto fc1_out = fc1(ln2_out);

    LayerHandle fc2 = createLayer("fully_connected",
      {"name=" + prefix + "/fc2",
       "unit=" + std::to_string(MODEL_DIM)});
    auto fc2_out = fc2(fc1_out);

    // Skip connection 2: add1_out + fc2_out
    LayerHandle add2 = createLayer("Addition",
      {"name=" + prefix + "/add2"});
    prev = add2({add1_out, fc2_out});
  }

  // Final Layer Norm
  LayerHandle final_ln = createLayer("layer_normalization",
    {"name=layer_normalization", "axis=3", "epsilon=1e-5"});
  auto output = final_ln(prev);

  model->setOptimizer(
    ml::train::createOptimizer("sgd", {"learning_rate = 0.1"}));

  // Graph-based compile (handles compile + initialize + allocate)
  model->compile(wte_in, output);

  return model.get();
}

std::string displayProgress(const int count, float loss, int batch_size) {
  int barWidth = 20;
  std::stringstream ssInt;
  ssInt << count * batch_size;
  std::string str = ssInt.str();
  int len = str.size();
  std::string ret;

  int pad_left = (barWidth - len) / 2;
  int pad_right = barWidth - pad_left - len;
  std::string out_str =
    std::string(pad_left, ' ') + str + std::string(pad_right, ' ');

  ret = " [ " + out_str + " ] " + " ( Training Loss: " + std::to_string(loss) +
        " ) ";

  return ret;
}

bool modelDestroyed() { return model_destroyed; }

std::string getInferResult() { return infer_result; }

std::string inferModel(std::string path, std::string sentence,
                       ml::train::Model *model_) {

  infer_result = "";
  std::string text = sentence;

  // Model is already compiled and initialized by createPicogpt()

  std::string weight_file_name =
    optimize ? path + "/pico_gpt.bin" : path + "/pico_gpt_124_mha.bin";

  model_->load(weight_file_name, ml::train::ModelFormat::MODEL_FORMAT_BIN);

  float *wte_input = new float[MAX_TOKEN_LEN];
  float *wpe_input = new float[MAX_TOKEN_LEN];

  std::string vocab_file_name = path + "/vocab.json";
  std::string merge_file_name = path + "/merges.txt";

  auto tokenizer = unwrap(GPT2Encoder::load(vocab_file_name, merge_file_name),
                          "Error initialising GPT2 tokenizer\n");

  auto init_input = tokenizer.encode(text);
  init_input_seq_len = init_input.size();

  for (unsigned int i = 0; i < init_input_seq_len; ++i) {
    ((unsigned int *)(wte_input))[i] = init_input[i];
  }

  for (unsigned int i = 0; i < init_input_seq_len; ++i) {
    ((unsigned int *)(wpe_input))[i] = i;
  }

  std::shared_ptr<ml::train::Layer> wte_embedding_layer;
  model_->getLayer("wte", &wte_embedding_layer);
  const std::vector<float *> wte_weights_buf =
    wte_embedding_layer->getWeights();
  auto wte_weight = ml::train::Tensor::fromData(
    {NUM_VOCAB, MODEL_DIM}, wte_weights_buf[0]);

  std::vector<float *> output_bufs;

  for (unsigned int i = init_input_seq_len;
       i < init_input_seq_len + NUM_TOKENS_TO_GENERATE; ++i) {
    output_bufs = model_->incremental_inference(
      BATCH_SIZE, {wte_input, wpe_input}, {}, init_input_seq_len, i - 1);

    auto output = ml::train::Tensor::fromData(
      {BATCH_SIZE, 1, i, MODEL_DIM}, output_bufs[0]);
    auto incremented_output = output.getSharedDataTensor(
      {BATCH_SIZE, 1, 1, MODEL_DIM}, BATCH_SIZE * (i - 1) * MODEL_DIM);
    auto next = incremented_output.dot(wte_weight, false, true);

    std::vector<unsigned int> ids = next.argmax();

    ((unsigned int *)(wte_input))[i] = ids[0];
    ((unsigned int *)(wpe_input))[i] = i;

    std::vector<int64_t> token_ids;
    for (auto element : ids) {
      token_ids.push_back(static_cast<int64_t>(element));
    }
    auto decoded_str = tokenizer.decode(token_ids);

    infer_result += decoded_str + " ";
    ANDROID_LOG_D("%s ", decoded_str.c_str());
  }

  infer_result += "\n";

  model_destroyed = true;

  return infer_result;
}
