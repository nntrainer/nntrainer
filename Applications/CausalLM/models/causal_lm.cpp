// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 * Copyright (C) 2025 Seungback Hong <sb92.hong@samsung.com>
 * Copyright (C) 2025 Hyeonseok Lee <hs89.lee@samsung.com>
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   causal_lm.cpp
 * @date   10 July 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This file defines CausalLM's basic actions
 * @note   This causal_lm.h constructs a class for Transformer-based Causal
 * Language Model (CausalLM). It aims to support AutoModelForCausalLM with
 * nntrainer. It supports the following models:
 *          - Llama
 */

#include <fstream>

#include <app_context.h>
#include <engine.h>
#include <model.h>

#include <causal_lm.h>
#include <llm_util.hpp>
#include <tokenizers_cpp.h>

#include <embedding_layer.h>
#include <mha_core.h>
#include <rms_norm.h>
#include <swiglu.h>
#include <tie_word_embedding.h>

namespace causallm {

std::string LoadBytesFromFile(const std::string &path);

CausalLM::CausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) {

  // Initialize the model with the provided configurations
  // This is where you would set up the model layers, parameters, etc.
  setupParameters(cfg, generation_cfg, nntr_cfg);

  // Initialize output list
  for (unsigned int i = 0; i < BATCH_SIZE; ++i)
    output_list.push_back("");

  // allocate memory for the internal buffer
  ids_history = (unsigned int *)malloc(static_cast<size_t>(BATCH_SIZE) *
                                       MAX_SEQ_LEN * sizeof(unsigned int));

  // prep tokenizer
  tokenizer = tokenizers::Tokenizer::FromBlobJSON(
    LoadBytesFromFile(nntr_cfg["tokenizer_file"]));
};

void CausalLM::setupParameters(json &cfg, json &generation_cfg,
                               json &nntr_cfg) {

  /** Initialize nntr prameters */
  BATCH_SIZE = nntr_cfg["batch_size"].get<unsigned int>();
  MODEL_TENSOR_TYPE = nntr_cfg["model_tensor_type"].get<std::string>();
  INIT_SEQ_LEN = nntr_cfg["init_seq_len"];
  MAX_SEQ_LEN = nntr_cfg["max_seq_len"];
  NUM_TO_GENERATE = nntr_cfg["num_to_generate"];
  MODEL_TENSOR_TYPE = nntr_cfg["model_tensor_type"];
  BAD_WORD_IDS = nntr_cfg["bad_word_ids"].get<std::vector<unsigned int>>();
  NUM_BADWORDS = BAD_WORD_IDS.size();
  MEMORY_SWAP = nntr_cfg.contains("fsu") ? nntr_cfg["fsu"].get<bool>() : false;
  FSU_LOOKAHEAD = nntr_cfg.contains("fsu_lookahead")
                    ? nntr_cfg["fsu_lookahead"].get<unsigned int>()
                    : 1;
  EMBEDDING_DTYPE = nntr_cfg["embedding_dtype"];
  LMHEAD_DTYPE = nntr_cfg.contains("lmhead_dtype")
                   ? nntr_cfg["lmhead_dtype"]
                   : nntr_cfg["embedding_dtype"];
  FC_LAYER_DTYPE = nntr_cfg["fc_layer_dtype"];

  USE_KVCACHE = false;
  PRE_COMPUTED_CACHE_PATH = "";
  SYS_PROMP_LEN = 0;

  if (nntr_cfg.contains("system_prompt") &&
      nntr_cfg["system_prompt"].contains("kvcache")) {
    USE_KVCACHE = true;
    PRE_COMPUTED_CACHE_PATH =
      nntr_cfg["system_prompt"]["kvcache"]["pre_computed_cache_path"];
    if (nntr_cfg["system_prompt"]["kvcache"].contains("sys_prompt_token_size"))
      SYS_PROMP_LEN =
        nntr_cfg["system_prompt"]["kvcache"]["sys_prompt_token_size"]
          .get<unsigned int>();
  }

  /** Initialize model parameters */
  NUM_VOCAB = cfg["vocab_size"];
  DIM = cfg["hidden_size"];
  INTERMEDIATE_SIZE = cfg["intermediate_size"];
  NUM_LAYERS = cfg["num_hidden_layers"];
  NUM_HEADS = cfg["num_attention_heads"];
  HEAD_DIM = cfg.contains("head_dim")
               ? cfg["head_dim"].get<int>()
               : DIM / NUM_HEADS; // default value is hidden_size / num_heads
  NUM_KEY_VALUE_HEADS = cfg.contains("num_key_value_heads")
                          ? cfg["num_key_value_heads"].get<int>()
                          : NUM_HEADS;
  SLIDING_WINDOW =
    cfg.contains("sliding_window") && !cfg["sliding_window"].is_null()
      ? cfg["sliding_window"].get<unsigned int>()
      : UINT_MAX;
  SLIDING_WINDOW_PATTERN = cfg.contains("sliding_window_pattern")
                             ? cfg["sliding_window_pattern"].get<unsigned int>()
                             : 1;
  MAX_POSITION_EMBEDDINGS = cfg["max_position_embeddings"].get<unsigned int>();
  ROPE_THETA = cfg["rope_theta"].get<unsigned int>();
  TIE_WORD_EMBEDDINGS = cfg["tie_word_embeddings"].get<bool>();
  NORM_EPS = cfg["rms_norm_eps"];
  GQA_SIZE = NUM_HEADS / NUM_KEY_VALUE_HEADS;

  EOS_TOKEN_ID =
    generation_cfg["eos_token_id"].get<std::vector<unsigned int>>();
  BOS_TOKEN_ID = generation_cfg["bos_token_id"].get<unsigned int>();
  TOP_K = generation_cfg.contains("top_k")
            ? generation_cfg["top_k"].get<unsigned int>()
            : 20;
  TOP_P = generation_cfg.contains("top_p")
            ? generation_cfg["top_p"].get<float>()
            : 0.95;
  TEMPERATURE = generation_cfg.contains("temperature")
                  ? generation_cfg["temperature"].get<float>()
                  : 0.7;
  global_token_len = 0;
  return;
};

void CausalLM::initialize() {

  // RegisterCustomLayers
  registerCustomLayers();

  // construct causalLM model
  constructModel();

  // setup model property
  std::vector<std::string> model_props = {
    withKey("batch_size", BATCH_SIZE), withKey("epochs", "1"),
    withKey("model_tensor_type", MODEL_TENSOR_TYPE)};
  if (MEMORY_SWAP) {
    model_props.emplace_back(withKey("fsu", "true"));
    model_props.emplace_back(withKey("fsu_lookahead", FSU_LOOKAHEAD));
  }

  model->setProperty(model_props);

  if (model->compile(ml::train::ExecutionMode::INFERENCE)) {
    throw std::invalid_argument("Model compilation failed.");
  }

  if (model->initialize(ml::train::ExecutionMode::INFERENCE)) {
    throw std::invalid_argument("Model initialization failed.");
  }

  is_initialized = true;

#ifdef DEBUG
  model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);
#endif
}

void CausalLM::constructModel() {

  // layers used in the model
  std::vector<LayerHandle> layers;

  // create model
  model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  // create input layer
  layers.push_back(createLayer(
    "input", {withKey("name", "input0"),
              withKey("input_shape", "1:1:" + std::to_string(INIT_SEQ_LEN))}));

  // create embedding layer
  const std::string embedding_type =
    TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "embedding_layer";
  const std::string lmhead_type =
    TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "fully_connected";
  layers.push_back(createLayer(
    embedding_type,
    {"name=embedding0", "in_dim=" + std::to_string(NUM_VOCAB),
     "weight_dtype=" + EMBEDDING_DTYPE, "out_dim=" + std::to_string(DIM)}));

  // create transformer layers
  for (int i = 0; i < NUM_LAYERS; ++i) {
    std::vector<LayerHandle> transformer;
    if (i == 0)
      transformer = createTransformerDecoderBlock(0, "embedding0");
    else
      transformer = createTransformerDecoderBlock(
        i, "layer" + std::to_string(i - 1) + "_decoder_output");
    layers.insert(layers.end(), transformer.begin(), transformer.end());
  }

  // create rms_norm
  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", "output_norm"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("input_layers",
             "layer" + std::to_string(NUM_LAYERS - 1) + "_decoder_output"),
     withKey("packed", "false")}));

  // add lmhead
  std::vector<std::string> lmhead_prop = {
    withKey("name", "output_of_causallm"),
    withKey("unit", NUM_VOCAB),
    withKey("disable_bias", "true"),
    withKey("input_layers", "output_norm"),
    withKey("weight_dtype", LMHEAD_DTYPE),
  };
  if (TIE_WORD_EMBEDDINGS)
    lmhead_prop.emplace_back(withKey("shared_from", "embedding0"));
  layers.push_back(createLayer(lmhead_type, lmhead_prop));

  // add created layers into the model
  for (auto &layer : layers) {
    model->addLayer(layer);
  }
};

void CausalLM::load_weight(const std::string &weight_path) {

  if (!is_initialized) {
    throw std::runtime_error("CausalLM model is not initialized. Please call "
                             "initialize() before load_weight().");
  }

  try {
    model->load(weight_path, ml::train::ModelFormat::MODEL_FORMAT_BIN);
  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to load model weights: " +
                             std::string(e.what()));
  }
};

void CausalLM::save_weight(const std::string &weight_path) {

  if (!is_initialized) {
    throw std::runtime_error("CausalLM model is not initialized. Please call "
                             "initialize() before save_weight().");
  }

  try {
    model->save(weight_path, ml::train::ModelFormat::MODEL_FORMAT_BIN);
  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to save model weights: " +
                             std::string(e.what()));
  }
};

void CausalLM::run(const WSTR prompt, bool do_sample, const WSTR system_prompt,
                   const WSTR tail_prompt) {

  if (!is_initialized) {
    throw std::runtime_error("CausalLM model is not initialized. Please call "
                             "initialize() before run().");
  }

  output_list.clear();
  for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
    output_list.push_back("");
  }

  if (MAX_SEQ_LEN < INIT_SEQ_LEN) {
    throw std::invalid_argument(
      "MAX_SEQ_LEN must be greater than or equal to INIT_SEQ_LEN");
  }

  /**
   * Variables for Log
   */
  unsigned int generation_cnt = 0;
  int64_t total_generation_duration = 0;

  /**
   * INPUT PREPARATION
   */
  std::vector<float *> input;
  std::vector<float *> label;

  /**
   * SAVE_KVCACHE ?
   *  if USE_KVCACHE && system_prompt is given && but the
   * PRE_COMPUTED_CACHE_PATH does not exist
   */
  SAVE_KVCACHE = (USE_KVCACHE && system_prompt != "" &&
                  !std::filesystem::exists(PRE_COMPUTED_CACHE_PATH));

#if defined(_WIN32)
  std::wcout << L"" << system_prompt << L"" << text_ << std::endl;
  std::wstring prompt_ = prompt;
  if (!SAVE_KVCACHE)
    prompt_ += TAIL_PROMPT;
  std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
  auto _input = tokenizer->Encode(converter.to_bytes(prompt_));
#else
  // print input text
  std::cout << system_prompt << prompt << tail_prompt << std::endl;

  // actual prompt to be used in computation
  std::string prompt_;

  if (USE_KVCACHE) {
    prompt_ = SAVE_KVCACHE ? system_prompt : (prompt + tail_prompt);
  } else {
    prompt_ = system_prompt + prompt + tail_prompt;
  }

  if (USE_KVCACHE && !SAVE_KVCACHE && SYS_PROMP_LEN == 0)
    SYS_PROMP_LEN = tokenizer->Encode(system_prompt).size();

  auto _input = tokenizer->Encode(prompt_);
#endif

  // | <------------------- MAX_SEQ_LEN -------------------> |
  //                       ||             ||
  // |<-- System prompt -->||<-- input -->||<-- generate -->|

  std::vector<int64_t> init_input;
  unsigned int _len = _input.size();
  unsigned int num_allow_str = MAX_SEQ_LEN - NUM_TO_GENERATE;
  unsigned text_len = _len;

  if (_len > num_allow_str)
    text_len = num_allow_str;

  // feed only available length
  // if _input is allowed, it feeds all of the _input
  // otherwise, feeds only a part of _input
  for (unsigned int i = 0; i < text_len; ++i)
    init_input.push_back(_input[i]);

  ///@todo currently, the whole sequence may not be fed into the model
  /// This should be handled later.
  _input.clear();

  unsigned int init_len = init_input.size();
  float *input_sample =
    (float *)malloc(sizeof(float) * BATCH_SIZE * MAX_SEQ_LEN);
  std::vector<bool> eos_list(BATCH_SIZE, false);

  unsigned int input_len = init_len;
  unsigned int token_generation_idx = input_len + 1;

  for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
    for (unsigned int i = 0; i < input_len; ++i) {
      input_sample[static_cast<size_t>(b) * MAX_SEQ_LEN + i] =
        static_cast<float>(init_input[i]);
      ids_history[static_cast<size_t>(b) * MAX_SEQ_LEN + i] = init_input[i];
    }
  }

  /**
   * PREFILL
   */
  std::vector<int64_t> token_ids;
  input.push_back(input_sample);

  ///@note contains possible bug
  // std::vector<ml::train::TensorDim> input_dims;
  // ml::train::TensorDim input_dim(1, 1, input_len, DIM);
  // input_dims.push_back(input_dim);
  // model->resetInputDimension(input_dims);

  auto start_prefill = std::chrono::high_resolution_clock::now();

  std::vector<float *> output;

  if (SAVE_KVCACHE) {
    //@note This is for the save the kv cache. precomputed kv cache should be
    // always located at the begining of the prompt.
    // Therefore, it start from 0. and system prompt should be saved in the
    // init_input, so that we can compute system prompt size properly
    //
    // The structure of this precomputed K,V Cache is :
    //
    //  //<-- System Prompt -->/<-- Input Tokens -->/<-- Tail prompt --> //
    //  //< Precomputed cache >/<--given as input-->/<--- from json ---->//
    //

    std::cout << "\n==============[KV CACHE SAVE MODE]================\n";
    output = model->incremental_inference(BATCH_SIZE, input, label, input_len,
                                          0 + global_token_len,
                                          input_len + global_token_len, false);

    SYS_PROMP_LEN = input_len;
    save_kvcache(PRE_COMPUTED_CACHE_PATH, SYS_PROMP_LEN);

    std::cout
      << "kv caches are saved in " << PRE_COMPUTED_CACHE_PATH << std::endl
      << "and the size of prompt is " << SYS_PROMP_LEN << ".\n"
      << "You may need this prompt lenth to set the \"sys_prompt_token_size\""
      << "\n==================================================\n"
      << std::endl;
    return;
  }

  if (USE_KVCACHE) {
    load_kvcache(PRE_COMPUTED_CACHE_PATH, SYS_PROMP_LEN);
  } else {
    SYS_PROMP_LEN = 0;
  }
  output = model->incremental_inference(BATCH_SIZE, input, label, init_len,
                                        SYS_PROMP_LEN,
                                        SYS_PROMP_LEN + input_len, false);

  // post process of model output
  std::vector<unsigned int> id_list(generate_multi_tokens(
    output[0], NUM_VOCAB, BATCH_SIZE, 1, ids_history, _len));

  if (init_len < INIT_SEQ_LEN)
    registerOutputs(tokenizer, id_list, init_len, eos_list);

  auto finish_prefill = std::chrono::high_resolution_clock::now();
  auto prefill_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
    finish_prefill - start_prefill);

  /**
   * TOKEN GENERATION
   */

  input_len += SYS_PROMP_LEN;

  // Update generated token by prefill as an input
  for (unsigned int b = 0; b < BATCH_SIZE; ++b)
    input_sample[static_cast<size_t>(b) * MAX_SEQ_LEN] =
      static_cast<float>(id_list[b]);

  auto start_generation = std::chrono::high_resolution_clock::now();

  for (token_generation_idx = input_len + 1;
       token_generation_idx < input_len + 1 + NUM_TO_GENERATE;
       ++token_generation_idx) {

    auto output_interval =
      model->incremental_inference(BATCH_SIZE, input, label, input_len,
                                   token_generation_idx - 1 + global_token_len,
                                   token_generation_idx + global_token_len);
    std::vector<unsigned int> ids_list(generate(output_interval[0], do_sample));
    if (token_generation_idx < input_len) {
      for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
        input_sample[static_cast<size_t>(b) * MAX_SEQ_LEN] =
          static_cast<float>(init_input[token_generation_idx - SYS_PROMP_LEN]);
      }
      registerOutputs(tokenizer, ids_list, token_generation_idx, eos_list);
    } else {
      for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
        input_sample[static_cast<size_t>(b) * MAX_SEQ_LEN] =
          static_cast<float>(ids_list[b]);
      }
      registerOutputs(tokenizer, ids_list, token_generation_idx, eos_list);
    }
    ++generation_cnt;

    // check FINISH
    for (unsigned int j = 0; j < BATCH_SIZE; ++j) {
      if (!eos_list[j] && (std::find(EOS_TOKEN_ID.begin(), EOS_TOKEN_ID.end(),
                                     ids_list[j]) != EOS_TOKEN_ID.end())) {
        eos_list[j] = true;
      }
    }

    bool is_finish = true;
    for (unsigned int j = 0; j < BATCH_SIZE; ++j) {
      if (!eos_list[j]) {
        is_finish = false;
        break;
      }
    }

    if (is_finish) {
      free(input_sample);
      break;
    }
  }

  global_token_len += (generation_cnt + init_len);

  auto finish_generation = std::chrono::high_resolution_clock::now();
  auto generation_duration =
    std::chrono::duration_cast<std::chrono::milliseconds>(finish_generation -
                                                          start_generation);

  std::cout << "\n\n";
  std::cout << "=================[ LLM with NNTrainer ]===================\n";
  std::cout << "prefill: " << init_len << " tokens, "
            << prefill_duration.count() << " ms, "
            << ((double)init_len / prefill_duration.count() * 1000) << " TPS\n";
  std::cout << "generation: " << generation_cnt << " tokens, "
            << generation_duration.count() << " ms, "
            << ((double)generation_cnt / generation_duration.count() * 1000)
            << " TPS\n";
  std::cout << "==========================================================\n";
};

std::vector<unsigned int> CausalLM::generate(float *logits, bool do_sample,
                                             float repetition_penalty,
                                             unsigned int *input_ids,
                                             unsigned int NUM_INPUT_IDS) {

  std::vector<unsigned int> outputs;
  for (unsigned int iteration = 0; iteration < BATCH_SIZE; ++iteration) {

    // apply repetition penalty
    if (repetition_penalty != 1 && input_ids != nullptr && NUM_INPUT_IDS != 0) {
      applyRepetitionPenalty(logits, input_ids, NUM_INPUT_IDS,
                             repetition_penalty);
    }

    // apply bad words penalty
    if (BAD_WORD_IDS.size() != 0 && NUM_BADWORDS != 0) {
      applyBadWordsPenalty(logits, BAD_WORD_IDS.data(), NUM_BADWORDS);
    }

    // return argmax if do_sample is false
    if (do_sample == false) {
      unsigned int argmax_idx =
        std::distance(logits, std::max_element(logits, logits + NUM_VOCAB));
      outputs.push_back(argmax_idx);
    } else {
      // apply temperature & top-k & top-p to logits
      float max_logits = applyTKP(logits, NUM_VOCAB, TEMPERATURE, TOP_K, TOP_P);
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
      std::discrete_distribution<int> dist(logits, logits + NUM_VOCAB);
      unsigned int sampled_idx = dist(rng);

      // add sampled word
      outputs.push_back(sampled_idx);
    }

    // set batch offset
    logits = logits + NUM_VOCAB;
    input_ids = input_ids + MAX_SEQ_LEN;
  }

  return outputs;
};

std::vector<LayerHandle>
CausalLM::createTransformerDecoderBlock(const int layer_id,
                                        std::string input_name) {

  std::vector<LayerHandle> layers;

  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention_norm"),
     withKey("input_layers", input_name),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));

  auto att_layer =
    createAttention(layer_id, INIT_SEQ_LEN, NUM_HEADS, HEAD_DIM,
                    "layer" + std::to_string(layer_id) + "_attention_norm",
                    "layer" + std::to_string(layer_id) + "_attention_norm",
                    "layer" + std::to_string(layer_id) + "_attention_norm");

  layers.insert(layers.end(), att_layer.begin(), att_layer.end());

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

std::vector<LayerHandle>
CausalLM::createAttention(const int layer_id, int seq_len, int n_heads,
                          int head_dim, std::string query_name,
                          std::string key_name, std::string value_name) {

  std::vector<LayerHandle> layers;

  auto Q = "layer" + std::to_string(layer_id) + "_wq";
  auto K = "layer" + std::to_string(layer_id) + "_wk";
  auto V = "layer" + std::to_string(layer_id) + "_wv";
  auto A = "layer" + std::to_string(layer_id) + "_attention";
  auto O = "layer" + std::to_string(layer_id) + "_attention_out";

  // V layer
  std::vector<std::string> v_params = {
    withKey("name", V), withKey("unit", head_dim * n_heads / GQA_SIZE),
    withKey("disable_bias", "true"), withKey("input_layers", value_name),
    withKey("weight_initializer", "ones")};
  layers.push_back(createLayer("fully_connected", v_params));

  // K layer
  std::vector<std::string> k_params = {
    withKey("name", K), withKey("unit", head_dim * n_heads / GQA_SIZE),
    withKey("disable_bias", "true"), withKey("input_layers", key_name),
    withKey("weight_initializer", "ones")};
  layers.push_back(createLayer("fully_connected", k_params));

  // Q layer
  std::vector<std::string> q_params = {
    withKey("name", Q), withKey("unit", head_dim * n_heads),
    withKey("disable_bias", "true"), withKey("input_layers", query_name),
    withKey("weight_initializer", "ones")};
  layers.push_back(createLayer("fully_connected", q_params));

  // Attention core layer
  std::vector<std::string> a_params = {
    withKey("name", A),
    withKey("num_heads", n_heads),
    withKey("num_heads_kv", n_heads / GQA_SIZE),
    withKey("max_timestep", std::to_string(INIT_SEQ_LEN + NUM_TO_GENERATE)),
    withKey("sliding_window", (layer_id + 1) % SLIDING_WINDOW_PATTERN
                                ? SLIDING_WINDOW
                                : UINT_MAX),
    withKey("rope_theta", ROPE_THETA),
    withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
    withKey("input_layers", {Q, K, V})};
  layers.push_back(createLayer("mha_core", a_params));

  // O layer
  std::vector<std::string> o_params = {
    withKey("name", O), withKey("unit", DIM), withKey("disable_bias", "true"),
    withKey("input_layers", A), withKey("weight_initializer", "ones")};
  layers.push_back(createLayer("fully_connected", o_params));

  return layers;
}

std::vector<LayerHandle> CausalLM::createMlp(const int layer_id, int dim,
                                             int hidden_dim,
                                             std::string input_name) {

  std::vector<LayerHandle> layers;

  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_up"),
     withKey("unit", hidden_dim), withKey("disable_bias", "true"),
     withKey("input_layers", input_name),
     withKey("weight_initializer", "ones")}));
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_gate"),
     withKey("unit", hidden_dim), withKey("disable_bias", "true"),
     withKey("input_layers", input_name),
     withKey("weight_initializer", "ones")}));

  layers.push_back(createLayer(
    "swiglu",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_swiglu"),
     withKey("input_layers", "layer" + std::to_string(layer_id) + "_ffn_up," +
                               "layer" + std::to_string(layer_id) +
                               "_ffn_gate")}));

  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
     withKey("unit", dim), withKey("disable_bias", "true"),
     withKey("input_layers",
             "layer" + std::to_string(layer_id) + "_ffn_swiglu"),
     withKey("weight_initializer", "ones")}));

  return layers;
}

void CausalLM::registerCustomLayers() {
  ///
  const auto &ct_engine = nntrainer::Engine::Global();
  const auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

  try {
    app_context->registerFactory(nntrainer::createLayer<causallm::SwiGLULayer>);
    app_context->registerFactory(
      nntrainer::createLayer<causallm::RMSNormLayer>);
    app_context->registerFactory(
      nntrainer::createLayer<causallm::MHACoreLayer>);
    app_context->registerFactory(
      nntrainer::createLayer<causallm::TieWordEmbedding>);
    app_context->registerFactory(
      nntrainer::createLayer<causallm::EmbeddingLayer>);

  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }
}

void CausalLM::registerOutputs(
  std::unique_ptr<tokenizers::Tokenizer> &tokenizer,
  std::vector<unsigned int> ids, unsigned int pos,
  const std::vector<bool> &eos_list) {

  static const std::vector<char> puncts{',', '!', ':', ';', '?'};
  for (size_t b = 0; b < ids.size(); ++b) {
    if (!eos_list[b]) {
      pending_ids_.push_back(static_cast<int>(ids[b]));
      ids_history[b * MAX_SEQ_LEN + pos] = ids[b];
      std::string decoded_str = tokenizer->Decode(pending_ids_);

      if (std::find(puncts.begin(), puncts.end(), decoded_str.back()) !=
          puncts.end()) {
        // last symbol is a punctuation, hold on
      } else if (decoded_str.size() >= 3 &&
                 decoded_str.compare(decoded_str.size() - 3, 3, "�") == 0) {
        // ends with an incomplete token, hold on
      } else {
#if defined(_WIN32)
        std::wcout << L"" << utf8_to_wstring(decoded_str);
        std::wcout.flush();
#else
        std::cout << decoded_str;
        std::cout.flush();
#endif
        output_list[b].append(decoded_str);
        pending_ids_.clear();
      }
    }
  }
}

void CausalLM::save_kvcache(std::string path, int to_) {
  auto f = nntrainer::checkedOpenStream<std::ofstream>(
    path, std::ios::out | std::ios::binary | std::ios::trunc);

  std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &, void *)>
    fn = [&f](ml::train::Layer &l, nntrainer::RunLayerContext &context,
              void *idx) {
      if (l.getType() == causallm::MHACoreLayer::type) {
        int to = static_cast<int>(reinterpret_cast<intptr_t>(idx));
        auto k_cache = context.getTensor(0);
        auto v_cache = context.getTensor(1);
        ml::train::TensorDim k_dim = k_cache.getDim();
        ml::train::TensorDim v_dim = v_cache.getDim();
        k_dim.height(to);
        v_dim.height(to);
        nntrainer::Tensor k_cache_prompt =
          k_cache.getSharedDataTensor(k_dim, 0, true);
        nntrainer::Tensor v_cache_prompt =
          v_cache.getSharedDataTensor(v_dim, 0, true);
        k_cache_prompt.save(f);
        v_cache_prompt.save(f);
      }
    };
  void *arg = reinterpret_cast<void *>(static_cast<intptr_t>(to_));
  model->forEachLayer(fn, arg);
  f.close();
}

void CausalLM::load_kvcache(std::string path, int to_) {
  auto f = nntrainer::checkedOpenStream<std::ifstream>(
    path, std::ios::in | std::ios::binary);

  model->allocate(ml::train::ExecutionMode::INFERENCE);

  std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &, void *)>
    fn = [&f](ml::train::Layer &l, nntrainer::RunLayerContext &context,
              void *idx) {
      if (l.getType() == causallm::MHACoreLayer::type) {
        int to = static_cast<int>(reinterpret_cast<intptr_t>(idx));
        auto k_cache = context.getTensor(0);
        auto v_cache = context.getTensor(1);
        ml::train::TensorDim k_dim = k_cache.getDim();
        ml::train::TensorDim v_dim = v_cache.getDim();
        k_dim.height(to);
        v_dim.height(to);
        nntrainer::Tensor k_cache_prompt =
          k_cache.getSharedDataTensor(k_dim, 0, true);
        nntrainer::Tensor v_cache_prompt =
          v_cache.getSharedDataTensor(v_dim, 0, true);
        k_cache_prompt.read(f);
        v_cache_prompt.read(f);
      }
    };
  void *arg = reinterpret_cast<void *>(static_cast<intptr_t>(to_));

  model->forEachLayer(fn, arg);
  f.close();
}

std::string LoadBytesFromFile(const std::string &path) {
  std::ifstream fs(path, std::ios::in | std::ios::binary);
  if (fs.fail()) {
    std::cerr << "Cannot open " << path << std::endl;
    exit(1);
  }
  std::string data;
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  data.resize(size);
  fs.read(data.data(), size);
  return data;
};

} // namespace causallm
