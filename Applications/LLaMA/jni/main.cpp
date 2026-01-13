// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   main.cpp
 * @date   7 August 2023
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#include <string.h>

#include <algorithm>
#include <ctime>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <layer.h>
#include <model.h>
#include <optimizer.h>
#include <util_func.h>

#include <app_context.h>
#include <custom_multi_head_attention_layer.h>
#include <engine.h>
#include <optional>
#include <rms_norm.h>
#include <rotary_embedding.h>
#include <swiglu.h>
#include <transpose_layer.h>

#if defined(ENABLE_TRANSFORMER)
#include "json.hpp"
#include <codecvt>
#include <encoder.hpp>
#include <locale>
using json = nlohmann::json;
#endif

using LayerHandle = std::shared_ptr<ml::train::Layer>;
using ModelHandle = std::unique_ptr<ml::train::Model>;

ModelHandle g_model;

// Hyper params for LLaMA
int const DIM = 2304;
int const INTERMEDIATE_SIZE = 2048;
int const NUM_LAYERS = 28;
int const NUM_HEADS = 18;

float const NORM_EPS = 0.000001f;
float const EPS = 0.000001f;
int const NUM_VOCAB = 96000;
int MAX_SEQ_LEN = 1024;
int NUM_TO_GENERATE = 100;

constexpr unsigned int INIT_SEQ_LEN = 2;
unsigned int batch_size = 1;
unsigned int epoch = 1;

/** cache loss values post training for test */
float training_loss = 0.0;
float validation_loss = 0.0;
bool fsu = false;

bool optimize = false;

/**
 * @brief Apply temperature & top-k & top-p to logits
 * @return Max logit for softmax
 */
float applyTKP(float *logits, int len, float temperature, unsigned int top_k,
               float top_p) {
  // Apply temperature & Sort logits
  std::vector<std::pair<int, float>> top_indices_and_logits;
  for (int i = 0; i < len; ++i) {
    if (temperature > 1e-5)
      logits[i] = logits[i] / temperature;
    top_indices_and_logits.push_back({i, logits[i]});
  }
  std::partial_sort(top_indices_and_logits.begin(),
                    top_indices_and_logits.begin() + top_k,
                    top_indices_and_logits.end(),
                    [](auto &a, auto &b) { return a.second > b.second; });

  // Accumulate logits
  float cum_prob = 0;
  unsigned int top_index = 0;
  while (cum_prob <= top_p && top_index < top_k) {
    cum_prob += top_indices_and_logits[top_index].second;
    ++top_index;
  }

  // Apply Top-K and Top-P
  std::fill_n(logits, len, -INFINITY);
  for (unsigned int i = 0; i < top_index; ++i) {
    logits[top_indices_and_logits[i].first] = top_indices_and_logits[i].second;
  }

  return top_indices_and_logits[0].second;
}

/**
 * @brief Apply repetition penalty to logits
 */
void applyRepetitionPenalty(float *logits, unsigned int *input_ids,
                            unsigned int NUM_INPUT_IDS,
                            float repetition_penalty = 1) {
  for (unsigned int i = 0; i < NUM_INPUT_IDS; ++i) {
    if (logits[input_ids[i]] < 0) {
      logits[input_ids[i]] *= repetition_penalty;
    } else {
      logits[input_ids[i]] /= repetition_penalty;
    }
  }
}

/**
 * @brief Apply bad words penalty
 */
void applyBadWordsPenalty(float *logits, unsigned int *bad_words_ids,
                          unsigned int NUM_BAD_WORDS_IDS) {
  for (unsigned int i = 0; i < NUM_BAD_WORDS_IDS; ++i) {
    logits[bad_words_ids[i]] = -INFINITY;
  }
}

std::vector<int> generate(float *logits, unsigned int NUM_VOCAB = 0,
                          unsigned int NUM_BATCH = 1, bool do_sample = false,
                          float temperature = 1, unsigned int top_k = 1,
                          float top_p = 1, float repetition_penalty = 1,
                          unsigned int *input_ids = nullptr,
                          unsigned int NUM_INPUT_IDS = 0,
                          unsigned int *bad_words_ids = nullptr,
                          unsigned int NUM_BAD_WORDS_IDS = 0) {
  std::vector<int> outputs;
  for (unsigned int iteration = 0; iteration < NUM_BATCH; ++iteration) {

    // apply repetition penalty
    if (repetition_penalty != 1 && input_ids != nullptr && NUM_INPUT_IDS != 0) {
      applyRepetitionPenalty(logits, input_ids, NUM_INPUT_IDS,
                             repetition_penalty);
    }

    // apply bad words penalty
    if (bad_words_ids != nullptr && NUM_BAD_WORDS_IDS != 0) {
      applyBadWordsPenalty(logits, bad_words_ids, NUM_BAD_WORDS_IDS);
    }

    // return argmax if do_sample is false
    if (do_sample == false) {
      int argmax_idx =
        std::distance(logits, std::max_element(logits, logits + NUM_VOCAB));
      outputs.push_back(argmax_idx);
    } else {
      // apply temperature & top-k & top-p to logits
      float max_logits = applyTKP(logits, NUM_VOCAB, temperature, top_k, top_p);
      // transform logits to softmax
      float sum_exp_logits = 0;
      for (unsigned int i = 0; i < NUM_VOCAB; i++) {
        float exp_x = exp(logits[i] - max_logits);
        sum_exp_logits += exp_x;
        logits[i] = exp_x;
      }

      for (unsigned int i = 0; i < NUM_VOCAB; ++i) {
        logits[i] /= sum_exp_logits;
      }

      // sample from final logits
      std::random_device rd;
      std::mt19937 gen(rd());
      std::discrete_distribution<int> dist(logits, logits + NUM_VOCAB);
      int sampled_idx = dist(gen);

      // add sampled word
      outputs.push_back(sampled_idx);
    }

    // set batch offset
    logits = logits + NUM_VOCAB;
    input_ids = input_ids + batch_size;
  }

  return outputs;
}

std::vector<int> generate_multi_tokens(
  float *logits, unsigned int NUM_VOCAB = 0, unsigned int NUM_TARGET_TOKENS = 1,
  float repetition_penalty = 1, unsigned int *input_ids = nullptr,
  unsigned int NUM_INPUT_IDS = 0, unsigned int *bad_words_ids = nullptr,
  unsigned int NUM_BAD_WORDS_IDS = 0) {
  std::vector<int> outputs;
  // apply repetition penalty
  if (abs(repetition_penalty - 1) < EPS && input_ids != nullptr &&
      NUM_INPUT_IDS != 0) {
    applyRepetitionPenalty(logits, input_ids, NUM_INPUT_IDS,
                           repetition_penalty);
  }

  // apply bad words penalty
  if (bad_words_ids != nullptr && NUM_BAD_WORDS_IDS != 0) {
    applyBadWordsPenalty(logits, bad_words_ids, NUM_BAD_WORDS_IDS);
  }

  // Sort and generate multiple tokens
  std::vector<std::pair<int, float>> top_indices_and_logits;
  for (unsigned int i = 0; i < NUM_VOCAB; ++i) {
    top_indices_and_logits.push_back({i, logits[i]});
  }
  std::partial_sort(top_indices_and_logits.begin(),
                    top_indices_and_logits.begin() + NUM_TARGET_TOKENS,
                    top_indices_and_logits.end(),
                    [](auto &a, auto &b) { return a.second > b.second; });

  // add sampled words
  for (unsigned int i = 0; i < NUM_TARGET_TOKENS; ++i) {
    outputs.push_back(top_indices_and_logits[i].first);
  }

  return outputs;
}

template <typename T>
T unwrap(std::optional<T> &&value, const std::string &error_msg) {
  if (value.has_value()) {
    return value.value();
  } else {
    throw std::runtime_error(error_msg);
  }
}

/**
 * @brief Create Attention Layer for the seperate impelemntation
 */
std::vector<LayerHandle> createAttentionLayer(const int layer_id, int seq_len,
                                              int n_heads, int head_dim,
                                              std::string query_name,
                                              std::string key_name,
                                              std::string value_name) {
  using ml::train::createLayer;

  std::vector<LayerHandle> layers;

  if (optimize) {
    // linear transformation of q
    for (int i = 0; i < n_heads; i++) {
      layers.push_back(createLayer(
        "fully_connected",
        {nntrainer::withKey("name", "layer" + std::to_string(layer_id) +
                                      "_wq_" + std::to_string(i)),
         nntrainer::withKey("unit", head_dim),
         nntrainer::withKey("disable_bias", "true"),
         nntrainer::withKey("input_layers", query_name)}));
    }

    // linear transformation of k
    for (int i = 0; i < n_heads; i++) {
      layers.push_back(createLayer(
        "fully_connected",
        {nntrainer::withKey("name", "layer" + std::to_string(layer_id) +
                                      "_wk_" + std::to_string(i)),
         nntrainer::withKey("unit", head_dim),
         nntrainer::withKey("disable_bias", "true"),
         nntrainer::withKey("input_layers", key_name)}));
    }

    // linear transformation of v
    for (int i = 0; i < n_heads; i++) {
      layers.push_back(createLayer(
        "fully_connected",
        {nntrainer::withKey("name", "layer" + std::to_string(layer_id) +
                                      "_wv_" + std::to_string(i)),
         nntrainer::withKey("unit", head_dim),
         nntrainer::withKey("disable_bias", "true"),
         nntrainer::withKey("input_layers", value_name)}));
    }

    std::string concat_input = "";
    // apply rotary embedding and dot_product attention
    for (int i = 0; i < n_heads; i++) {
      // reshape q, k, v (apply num_heads)
      layers.push_back(createLayer(
        "reshape",
        {nntrainer::withKey("name", "layer" + std::to_string(layer_id) +
                                      "_q_reshape_" + std::to_string(i)),
         nntrainer::withKey("target_shape", "1:" + std::to_string(seq_len) +
                                              ":" + std::to_string(head_dim)),
         nntrainer::withKey("input_layers", "layer" + std::to_string(layer_id) +
                                              "_wq_" + std::to_string(i))}));

      layers.push_back(createLayer(
        "reshape",
        {nntrainer::withKey("name", "layer" + std::to_string(layer_id) +
                                      "_k_reshape_" + std::to_string(i)),
         nntrainer::withKey("target_shape", "1:" + std::to_string(seq_len) +
                                              ":" + std::to_string(head_dim)),
         nntrainer::withKey("input_layers", "layer" + std::to_string(layer_id) +
                                              "_wk_" + std::to_string(i))}));

      layers.push_back(createLayer(
        "reshape",
        {nntrainer::withKey("name", "layer" + std::to_string(layer_id) +
                                      "_v_reshape_" + std::to_string(i)),
         nntrainer::withKey("target_shape", "1:" + std::to_string(seq_len) +
                                              ":" + std::to_string(head_dim)),
         nntrainer::withKey("input_layers", "layer" + std::to_string(layer_id) +
                                              "_wv_" + std::to_string(i))}));

      // apply rotary embedding to q, k
      layers.push_back(createLayer(
        "rotary_embedding",
        {nntrainer::withKey("name", "layer" + std::to_string(layer_id) +
                                      "_q_rotary_" + std::to_string(i)),
         nntrainer::withKey("input_layers", "layer" + std::to_string(layer_id) +
                                              "_q_reshape_" +
                                              std::to_string(i))}));

      layers.push_back(createLayer(
        "rotary_embedding",
        {nntrainer::withKey("name", "layer" + std::to_string(layer_id) +
                                      "_k_rotary_" + std::to_string(i)),
         nntrainer::withKey("input_layers", "layer" + std::to_string(layer_id) +
                                              "_k_reshape_" +
                                              std::to_string(i))}));

      // apply scaled-dot product attention
      layers.push_back(ml::train::layer::Attention(
        {"name=layer" + std::to_string(layer_id) + "_attention_" +
           std::to_string(i),
         "input_layers=layer" + std::to_string(layer_id) + "_q_rotary_" +
           std::to_string(i) + ",layer" + std::to_string(layer_id) +
           "_v_reshape_" + std::to_string(i) + ",layer" +
           std::to_string(layer_id) + "_k_rotary_" + std::to_string(i),
         "scaled_dot_product=true", "causal_mask=true"}));

      layers.push_back(createLayer(
        "reshape",
        {nntrainer::withKey("name", "layer" + std::to_string(layer_id) +
                                      "_attention_output_" + std::to_string(i)),
         nntrainer::withKey("target_shape", "1:" + std::to_string(seq_len) +
                                              ":" + std::to_string(head_dim)),
         nntrainer::withKey("input_layers", "layer" + std::to_string(layer_id) +
                                              "_attention_" +
                                              std::to_string(i))}));

      concat_input += "layer" + std::to_string(layer_id) +
                      "_attention_output_" + std::to_string(i);

      if (i != n_heads - 1) {
        concat_input += ",";
      }
    }

    // concat attention output
    layers.push_back(createLayer(
      "concat", {nntrainer::withKey("name", "layer" + std::to_string(layer_id) +
                                              "_attention_concat"),
                 nntrainer::withKey("axis", 3),
                 nntrainer::withKey("input_layers", concat_input)}));

    // reshape for flatten
    layers.push_back(createLayer(
      "reshape",
      {nntrainer::withKey("name", "layer" + std::to_string(layer_id) +
                                    "_attention_flatten"),
       nntrainer::withKey("target_shape", "1:" + std::to_string(seq_len) + ":" +
                                            std::to_string(n_heads * head_dim)),
       nntrainer::withKey("input_layers", "layer" + std::to_string(layer_id) +
                                            "_attention_concat")}));

    // linear transformation of attention output
    layers.push_back(createLayer(
      "fully_connected",
      {nntrainer::withKey("name", "layer" + std::to_string(layer_id) +
                                    "_attention_out"),
       nntrainer::withKey("unit", head_dim * n_heads),
       nntrainer::withKey("disable_bias", "true"),
       nntrainer::withKey("input_layers", "layer" + std::to_string(layer_id) +
                                            "_attention_flatten")}));
  } else {
    layers.push_back(createLayer(
      "custom_multi_head_attention",
      {nntrainer::withKey("name", "layer" + std::to_string(layer_id) +
                                    "_attention_out"),
       nntrainer::withKey("num_heads", std::to_string(NUM_HEADS)),
       nntrainer::withKey("max_timestep", std::to_string(MAX_SEQ_LEN)),
       nntrainer::withKey("disable_bias", "true"),
       nntrainer::withKey("input_layers",
                          {query_name, key_name, value_name})}));
  }

  return layers;
}

/**
 * @brief Create FF Layers
 */
std::vector<LayerHandle> createFeedForwardLayer(const int layer_id, int dim,
                                                int intermediate_size,
                                                std::string input_name) {
  using ml::train::createLayer;
  std::vector<LayerHandle> layers;

  layers.push_back(createLayer(
    "fully_connected",
    {nntrainer::withKey("name", "layer" + std::to_string(layer_id) + "_ffn_1"),
     nntrainer::withKey("unit", intermediate_size),
     nntrainer::withKey("disable_bias", "true"),
     nntrainer::withKey("input_layers", input_name)}));
  layers.push_back(createLayer(
    "fully_connected",
    {nntrainer::withKey("name", "layer" + std::to_string(layer_id) + "_ffn_2"),
     nntrainer::withKey("unit", intermediate_size),
     nntrainer::withKey("disable_bias", "true"),
     nntrainer::withKey("input_layers", input_name)}));

  layers.push_back(createLayer(
    "swiglu",
    {nntrainer::withKey("name",
                        "layer" + std::to_string(layer_id) + "_ffn_swiglu"),
     nntrainer::withKey("input_layers",
                        "layer" + std::to_string(layer_id) + "_ffn_1," +
                          "layer" + std::to_string(layer_id) + "_ffn_2")}));

  layers.push_back(createLayer(
    "fully_connected",
    {nntrainer::withKey("name",
                        "layer" + std::to_string(layer_id) + "_ffn_output"),
     nntrainer::withKey("unit", dim),
     nntrainer::withKey("disable_bias", "true"),
     nntrainer::withKey("input_layers",
                        "layer" + std::to_string(layer_id) + "_ffn_swiglu")}));

  return layers;
}

/**
 * @brief Create Decoder
 */
std::vector<LayerHandle> createTransformerDecoder(const int layer_id,
                                                  std::string input_name) {
  using ml::train::createLayer;
  std::vector<LayerHandle> layers;

  layers.push_back(createLayer(
    "rms_norm", {nntrainer::withKey("name", "layer" + std::to_string(layer_id) +
                                              "_attention_norm"),
                 nntrainer::withKey("input_layers", input_name),
                 nntrainer::withKey("epsilon", std::to_string(NORM_EPS)),
                 nntrainer::withKey("packed", "false")}));

  auto att_layer = createAttentionLayer(
    layer_id, INIT_SEQ_LEN, NUM_HEADS, DIM / NUM_HEADS,
    "layer" + std::to_string(layer_id) + "_attention_norm",
    "layer" + std::to_string(layer_id) + "_attention_norm",
    "layer" + std::to_string(layer_id) + "_attention_norm");
  layers.insert(layers.end(), att_layer.begin(), att_layer.end());

  layers.push_back(createLayer(
    "addition", {nntrainer::withKey("name", "layer" + std::to_string(layer_id) +
                                              "_decoder_add"),
                 nntrainer::withKey("input_layers", input_name + ",layer" +
                                                      std::to_string(layer_id) +
                                                      "_attention_out")}));

  layers.push_back(createLayer(
    "rms_norm",
    {nntrainer::withKey("name",
                        "layer" + std::to_string(layer_id) + "_ffn_norm"),
     nntrainer::withKey("input_layers",
                        "layer" + std::to_string(layer_id) + "_decoder_add"),
     nntrainer::withKey("epsilon", std::to_string(NORM_EPS)),
     nntrainer::withKey("packed", "false")}));

  auto ffn_layer =
    createFeedForwardLayer(layer_id, DIM, INTERMEDIATE_SIZE,
                           "layer" + std::to_string(layer_id) + "_ffn_norm");
  layers.insert(layers.end(), ffn_layer.begin(), ffn_layer.end());

  layers.push_back(createLayer(
    "addition", {nntrainer::withKey("name", "layer" + std::to_string(layer_id) +
                                              "_decoder_output"),
                 nntrainer::withKey(
                   "input_layers",
                   "layer" + std::to_string(layer_id) + "_decoder_add,layer" +
                     std::to_string(layer_id) + "_ffn_output")}));

  return layers;
}

/**
 * @brief Create LLaMA2 Model
 */
ModelHandle createLLaMA() {
  using ml::train::createLayer;

  ModelHandle model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  std::vector<LayerHandle> layers;

  layers.push_back(createLayer(
    "input", {nntrainer::withKey("name", "input0"),
              nntrainer::withKey("input_shape",
                                 "1:1:" + std::to_string(INIT_SEQ_LEN))}));

  layers.push_back(ml::train::layer::Embedding(
    {"name=embedding0", "in_dim=" + std::to_string(NUM_VOCAB), "packed=false",
     "out_dim=" + std::to_string(DIM)}));

  for (int i = 0; i < NUM_LAYERS; i++) {
    std::vector<LayerHandle> transformer;
    if (i == 0)
      transformer = createTransformerDecoder(i, "embedding0");
    else
      transformer = createTransformerDecoder(
        i, "layer" + std::to_string(i - 1) + "_decoder_output");
    layers.insert(layers.end(), transformer.begin(), transformer.end());
  }

  int last_layer = NUM_LAYERS - 1;

  layers.push_back(createLayer(
    "rms_norm",
    {nntrainer::withKey("name", "output_norm"),
     nntrainer::withKey("epsilon", std::to_string(NORM_EPS)),
     nntrainer::withKey("input_layers", "layer" + std::to_string(last_layer) +
                                          "_decoder_output"),
     nntrainer::withKey("packed", "false")}));

  layers.push_back(createLayer(
    "fully_connected", {nntrainer::withKey("name", "output_of_llama"),
                        nntrainer::withKey("unit", NUM_VOCAB),
                        nntrainer::withKey("disable_bias", "true"),
                        nntrainer::withKey("input_layers", "output_norm"),
                        nntrainer::withKey("packed", "false")}));

  for (auto &layer : layers) {
    model->addLayer(layer);
  }

  return model;
}

/**
 * @brief to run for every text sequence
 */
void run(std::string text, std::string vocab_file_path,
         std::string merge_file_path, bool apply_temperature = false) {

  std::vector<float *> input;
  std::vector<float *> label;

  int data_size = batch_size * INIT_SEQ_LEN;

  float *input_sample = (float *)malloc(sizeof(float) * data_size);

  if (input_sample == nullptr) {
    std::cerr << "failed to malloc()" << std::endl;
    exit(EXIT_SUCCESS);
  }

  unsigned int input_len = INIT_SEQ_LEN;

  unsigned int init_len;

#if defined(ENABLE_TRANSFORMER)
  auto tokenizer = unwrap(GPT2Encoder::load(vocab_file_path, merge_file_path),
                          "Error initializing GPT2 tokenizer\n");

  std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;

  auto init_input = tokenizer.encode(converter.from_bytes(text));
  init_len = init_input.size();

  input_len = (init_len > INIT_SEQ_LEN) ? INIT_SEQ_LEN : init_len;

  for (unsigned int i = 0; i < input_len; ++i) {
    input_sample[i] = static_cast<float>(init_input[i]);
  }

  input.push_back(input_sample);

#else
  float init_input[INIT_SEQ_LEN] = {0, 21};
  memcpy(input_sample, init_input, sizeof(float) * INIT_SEQ_LEN);
  input.push_back(input_sample);
  init_len = 2;
#endif

  std::vector<int64_t> token_ids;

  auto output =
    g_model->incremental_inference(1, input, label, MAX_SEQ_LEN, 0, input_len);

  unsigned int ids = std::distance(
    output[0], std::max_element(output[0], output[0] + NUM_VOCAB));

  input_sample[0] = static_cast<float>(ids);

#ifdef ENABLE_FP16
  for (auto o : output) {
    delete[] o;
  }
#endif
  std::cout << " Progress Reading: 100 % " << std::endl;
  std::cout << std::endl << "### Output : " << std::endl;
  if (init_len < INIT_SEQ_LEN) {
#if defined(ENABLE_TRANSFORMER)
    auto decoded_str = tokenizer.decode({static_cast<int64_t>(ids)});
    std::cout << decoded_str << " ";
    std::cout.flush();
#endif
  }

  for (unsigned int i = input_len + 1; i < input_len + NUM_TO_GENERATE; ++i) {
    auto output_interval =
      g_model->incremental_inference(1, input, label, MAX_SEQ_LEN, i - 1, i);

    unsigned int ids = generate(output[0], NUM_VOCAB)[0];

    if (i < input_len) {
      input_sample[0] = static_cast<float>(init_input[i]);
    } else {
      input_sample[0] = static_cast<float>(ids);
#if defined(ENABLE_TRANSFORMER)
      auto decoded_str = tokenizer.decode({static_cast<int64_t>(ids)});
      std::cout << decoded_str << " ";
      std::cout.flush();
#endif
    }

#ifdef ENABLE_FP16
    for (auto o : output_interval) {
      delete[] o;
    }
#endif
  }

  std::cout << std::endl;
  free(input_sample);
}

/**
 * @brief to creaet model
 */
void createAndRun(unsigned int epochs, unsigned int batch_size,
                  std::string weight_path) {
  // setup model
  g_model = createLLaMA();
  g_model->setProperty({nntrainer::withKey("batch_size", batch_size),
                        nntrainer::withKey("epochs", epochs),
#ifdef ENABLE_FP16
                        nntrainer::withKey("model_tensor_type", "FP16-FP16"),
#endif
                        nntrainer::withKey("save_path", "test_model.bin")});

  auto optimizer = ml::train::createOptimizer("sgd", {"learning_rate=0.001"});
  int status = g_model->setOptimizer(std::move(optimizer));
  if (status) {
    throw std::invalid_argument("failed to set optimizer!");
  }

  status = g_model->compile();
  if (status) {
    throw std::invalid_argument("model compilation failed!");
  }

  status = g_model->initialize();
  if (status) {
    throw std::invalid_argument("model initialization failed!");
  }

  g_model->load(weight_path);
}

#if defined(ENABLE_TRANSFORMER)
std::wstring decodeUnicodeEscape(const std::wstring &input) {
  std::wstringstream result;

  for (size_t i = 0; i < input.length(); ++i) {
    if (i + 5 < input.length() && input[i] == L'\\' && input[i + 1] == L'u') {
      std::wstring unicodeSeq;
      for (int j = 0; j < 4; ++j)
        unicodeSeq += input[i + 2 + j];

      result << static_cast<wchar_t>(std::stoi(unicodeSeq, nullptr, 16));
      i += 5;
    } else if (input[i] == L'\\' && input[i + 1] == L'n') {
      result << static_cast<wchar_t>('\n');
      i++;
    } else if (input[i] == L' ')
      result << static_cast<wchar_t>(' ');
    else
      result << input[i];
  }

  return result.str();
}
#endif
int main(int argc, char *argv[]) {
  // Setting locale
  std::locale::global(std::locale("ko_KR.UTF-8"));

  std::string text = "";

#if defined(ENABLE_TRANSFORMER)
  // Getting arguments From terminal
  std::wstring input;
  std::getline(std::wcin, input);
  std::wstring test = decodeUnicodeEscape(input);
  std::wstring_convert<std::codecvt_utf16<wchar_t>> converter;
  text = converter.to_bytes(test);
#endif
  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

  try {
    app_context->registerFactory(
      nntrainer::createLayer<custom::MultiHeadAttentionLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
    return 1;
  }

  try {
    app_context->registerFactory(nntrainer::createLayer<custom::SwiGLULayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
    return 1;
  }

  try {
    app_context->registerFactory(nntrainer::createLayer<custom::RMSNormLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
    return 1;
  }

  try {
    const std::vector<std::string> args(argv + 1, argv + argc);
    if (args.size() != 4) {
      std::cerr << "Error: Invalid number of arguments.\n"

                   "Usage: program_name <weight.bin path> <vocab.json path> "
                   "<merges.txt path> <temperature (0 or 1)>"
                << std::endl;

      throw std::invalid_argument("Invalid number of arguments");
    }

    std::string weight_path = args[0].c_str();
    std::string vocab_file_name = args[1].c_str();
    std::string merge_file_name = args[2].c_str();
#ifdef _WIN32
    bool apply_temp = _stricmp("true", args[3].c_str()) == 0;
#else
    bool apply_temp = (strcasecmp("true", args[3].c_str()) == 0);
#endif

    createAndRun(epoch, batch_size, weight_path);

    run(text, vocab_file_name, merge_file_name, apply_temp);

  } catch (const std::exception &e) {
    std::cerr << "uncaught error while running! details: " << e.what()
              << std::endl;
    return EXIT_FAILURE;
  }

  int status = EXIT_SUCCESS;
  return status;
}
