// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Hyeonseok Lee <hs89.lee@samsung.com>
 *
 * @file   main.cpp
 * @date   19 May 2023
 * @brief  task runner for the pico gpt
 * @see    https://github.com/nntrainer/nntrainer
 *         https://github.com/jaymody/picoGPT
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <algorithm>
#include <app_context.h>
#include <fstream>
#include <model.h>
#include <string.h>
#include <tensor_api.h>

#if defined(ENABLE_TRANSFORMER)
#include "encoder.hpp"
#endif

#include <iostream>

const unsigned int BATCH_SIZE = 1;
const unsigned int NUM_LAYERS = 12;
const unsigned int NUM_HEADS = 12;
const unsigned int MODEL_DIM = 768;
/** @todo: Need to check **/
const unsigned int FC_UNIT = MODEL_DIM * 4;

const unsigned int NUM_VOCAB = 50257;
const unsigned int NUM_CTX = 1024;
const unsigned int NUM_TOKENS_TO_GENERATE = 40;

unsigned int init_input_seq_len;
// Todo: fix this
const unsigned int MAX_TOKEN_LEN = 10 + NUM_TOKENS_TO_GENERATE;

bool fsu = false;
bool optimize = false;
// bool optimize = true;
bool optimize_attention = false;

#if defined(ENABLE_TRANSFORMER)
template <typename T>
T unwrap(std::optional<T> &&value, const std::string &error_msg) {
  if (value.has_value()) {
    return value.value();
  } else {
    throw std::runtime_error(error_msg);
  }
}
#endif

std::shared_ptr<ml::train::Model> genModel() {
  using namespace ml::train;

  auto model = createModel(ModelType::NEURAL_NET);
  model->setProperty({"batch_size=" + std::to_string(BATCH_SIZE),
                      "model_tensor_type=FP16-FP16",
                      fsu ? "fsu=true" : "fsu=false"});

  // Symbolic input tensors
  Tensor wte_in({1, 1, 1, 1}, "wte_input");
  Tensor wpe_in({1, 1, 1, 1}, "wpe_input");

  // Embedding layers
  LayerHandle wte =
    createLayer("embedding", {"name=wte", "in_dim=" + std::to_string(NUM_VOCAB),
                              "out_dim=" + std::to_string(MODEL_DIM)});
  LayerHandle wpe =
    createLayer("embedding", {"name=wpe", "in_dim=" + std::to_string(NUM_CTX),
                              "out_dim=" + std::to_string(MODEL_DIM)});
  auto wte_out = wte(wte_in);
  auto wpe_out = wpe(wpe_in);

  // Add embeddings
  LayerHandle add_emb = createLayer("Addition", {"name=add"});
  auto prev = add_emb({wte_out, wpe_out});

  for (unsigned int i = 0; i < NUM_LAYERS; ++i) {
    std::string prefix = "layer" + std::to_string(i);

    // Layer Norm 1
    LayerHandle ln1 =
      createLayer("layer_normalization",
                  {"name=" + prefix + "/ln1", "axis=3", "epsilon=1e-5"});
    auto ln1_out = ln1(prev);

    Tensor attn_out;
    if (optimize) {
      // Per-head attention with separate Q/K/V projections
      std::vector<Tensor> attn_heads;

      for (unsigned int j = 0; j < NUM_HEADS; ++j) {
        unsigned int idx = NUM_HEADS - 1 - j;
        std::string head_prefix = prefix + "/multi_head_attention";

        LayerHandle v_fc =
          createLayer("fully_connected",
                      {"name=" + head_prefix + "/v_fc" + std::to_string(idx),
                       "unit=" + std::to_string(MODEL_DIM / NUM_HEADS)});
        LayerHandle k_fc =
          createLayer("fully_connected",
                      {"name=" + head_prefix + "/k_fc" + std::to_string(idx),
                       "unit=" + std::to_string(MODEL_DIM / NUM_HEADS)});
        LayerHandle q_fc =
          createLayer("fully_connected",
                      {"name=" + head_prefix + "/q_fc" + std::to_string(idx),
                       "unit=" + std::to_string(MODEL_DIM / NUM_HEADS)});

        auto v = v_fc(ln1_out);
        auto k = k_fc(ln1_out);
        auto q = q_fc(ln1_out);

        LayerHandle attn = createLayer(
          "attention",
          {"name=" + head_prefix + "/attention" + std::to_string(idx),
           "scaled_dot_product=true", "causal_mask=true"});
        attn_heads.push_back(attn({q, v, k}));
      }

      // Reverse so concat order is attention0, attention1, ..., attention11
      std::reverse(attn_heads.begin(), attn_heads.end());

      LayerHandle concat = createLayer(
        "concat",
        {"name=" + prefix + "/multi_head_attention/concat", "axis=3"});
      auto concat_out = concat(attn_heads);

      LayerHandle attn_fc = createLayer(
        "fully_connected", {"name=" + prefix + "/multi_head_attention/fc",
                            "unit=" + std::to_string(MODEL_DIM)});
      auto fc_out = attn_fc(concat_out);

      LayerHandle identity =
        createLayer("identity", {"name=" + prefix + "/multi_head_attention"});
      attn_out = identity(fc_out);
    } else {
      LayerHandle mha = createLayer("multi_head_attention",
                                    {"name=" + prefix + "/multi_head_attention",
                                     "num_heads=" + std::to_string(NUM_HEADS)});
      attn_out = mha({ln1_out, ln1_out, ln1_out});
    }

    // Skip connection 1: prev + attention output
    LayerHandle add1 = createLayer("Addition", {"name=" + prefix + "/add1"});
    auto add1_out = add1({prev, attn_out});

    // Layer Norm 2
    LayerHandle ln2 =
      createLayer("layer_normalization",
                  {"name=" + prefix + "/ln2", "axis=3", "epsilon=1e-5"});
    auto ln2_out = ln2(add1_out);

    // FFN
    LayerHandle fc1 =
      createLayer("fully_connected",
                  {"name=" + prefix + "/fc1", "unit=" + std::to_string(FC_UNIT),
                   "activation=gelu"});
    auto fc1_out = fc1(ln2_out);

    LayerHandle fc2 =
      createLayer("fully_connected", {"name=" + prefix + "/fc2",
                                      "unit=" + std::to_string(MODEL_DIM)});
    auto fc2_out = fc2(fc1_out);

    // Skip connection 2: add1_out + fc2_out
    LayerHandle add2 = createLayer("Addition", {"name=" + prefix + "/add2"});
    prev = add2({add1_out, fc2_out});
  }

  // Final Layer Norm
  LayerHandle final_ln =
    createLayer("layer_normalization",
                {"name=layer_normalization", "axis=3", "epsilon=1e-5"});
  auto output = final_ln(prev);

  int status = model->setOptimizer(
    ml::train::createOptimizer("sgd", {"learning_rate = 0.1"}));
  if (status) {
    throw std::invalid_argument("failed to set optimizer!");
  }

  // Graph-based compile (handles compile + initialize + allocate)
  status = model->compile(wte_in, output);
  if (status) {
    throw std::invalid_argument("failed to compile model from graph!");
  }

  return model;
}

int main(int argc, char *argv[]) {
  try {
    if (argc < 2) {
      std::cout << "Usage: " << argv[0] << " <text>\n";
      return 1;
    }

    const std::vector<std::string> args(argv + 1, argv + argc);
    std::string text = args[0];

    auto model = genModel();
    model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);

    std::string weight_file_name =
      optimize ? "./res/app/PicoGPT/pico_gpt_124.bin"
               : "./res/app/PicoGPT/pico_gpt_mha_fp16.bin";
    // : "./res/app/PicoGPT/pico_gpt_124_mha.bin";
    try {
      model->load(weight_file_name, ml::train::ModelFormat::MODEL_FORMAT_BIN);
    } catch (const std::exception &e) {
      std::cerr << "Error during load: " << e.what() << "\n";
      return 1;
    }

    // model->save("pico_gpt_fp16.bin");

    float *wte_input = new float[1];
    float *wpe_input = new float[1];

    std::vector<int64_t> init_input;

#if defined(ENABLE_TRANSFORMER)

    std::string vocab_file_name = "../Applications/PicoGPT/jni/vocab.json";
    std::string merge_file_name = "../Applications/PicoGPT/jni/merges.txt";

    auto tokenizer = unwrap(GPT2Encoder::load(vocab_file_name, merge_file_name),
                            "Error initialising GPT2 tokenizer\n");

    init_input = tokenizer.encode(text);
#else
    text = "Elan Turing is";
    init_input = {36235, 39141, 18765, 1143, 326, 9061, 561, 530, 1110, 1716};
#endif
    init_input_seq_len = init_input.size();

    ((unsigned int *)(wte_input))[0] = init_input[0];
    ((unsigned int *)(wpe_input))[0] = 0;

    std::vector<float *> output_bufs;
    std::shared_ptr<ml::train::Layer> wte_embedding_layer;

    model->getLayer("wte", &wte_embedding_layer);
    const std::vector<float *> wte_weights_buf =
      wte_embedding_layer->getWeights();
    auto wte_weight =
      ml::train::Tensor::fromData({NUM_VOCAB, MODEL_DIM}, wte_weights_buf[0]);

    for (unsigned int i = 1; i < init_input_seq_len + NUM_TOKENS_TO_GENERATE;
         ++i) {
      output_bufs = model->incremental_inference(
        BATCH_SIZE, {wte_input, wpe_input}, {}, init_input_seq_len, i - 1, i);

      auto output = ml::train::Tensor::fromData({BATCH_SIZE, 1, i, MODEL_DIM},
                                                output_bufs[0]);

      std::shared_ptr<ml::train::Layer> wte_embedding_layer;
      model->getLayer("wte", &wte_embedding_layer);
      const std::vector<float *> wte_weights_buf =
        wte_embedding_layer->getWeights();
      auto wte_weight =
        ml::train::Tensor::fromData({NUM_VOCAB, MODEL_DIM}, wte_weights_buf[0]);
      auto logits = output.dot(wte_weight, false, true);
      auto next = logits.getSharedDataTensor({1, NUM_VOCAB},
                                             BATCH_SIZE * (i - 1) * NUM_VOCAB);

      std::vector<unsigned int> ids = next.argmax();

      if (i < init_input_seq_len) {
        ((unsigned int *)(wte_input))[0] = init_input[i];
      } else {
        ((unsigned int *)(wte_input))[0] = ids[0];
      }

      ((unsigned int *)(wpe_input))[0] = i;

#if defined(ENABLE_TRANSFORMER)
      std::vector<int64_t> token_ids;
      for (auto element : ids) {
        token_ids.push_back(static_cast<int64_t>(element));
      }

      if (i >= init_input_seq_len) {
        auto decoded_str = tokenizer.decode(token_ids);
        std::cerr << decoded_str << " " << std::flush;
      }
#endif
    }

    for (auto v : wte_weights_buf) {
      delete v;
    }

    for (auto v : output_bufs) {
      delete v;
    }

    std::cout << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "uncaught error while running! details: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
