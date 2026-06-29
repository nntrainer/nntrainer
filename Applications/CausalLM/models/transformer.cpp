// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   transformer.cpp
 * @date   10 July 2025
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @author Sumon Nath <sumon.nath@samsung.com>
 * @author Pranjal Thapliyal <p.thapliyal@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This file defines Transformer's basic actions
 */

#include <cmath>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <unordered_set>

#include <app_context.h>
#include <engine.h>
#include <model.h>

#include <llm_util.hpp>
#include <tokenizers_cpp.h>
#include <transformer.h>

#include <embedding_layer.h>
#include <mha_core.h>
#include <neuralnet.h>
#include <qs4cx_tensor.h>
#include <rms_norm.h>
#include <swiglu.h>
#include <tie_word_embedding.h>

namespace causallm {

/**
 * @brief Load a file as a binary string.
 */
ml::train::ModelFormat
Transformer::formatFromExtension(const std::string &weight_path) {
  const auto dot = weight_path.find_last_of('.');
  if (dot != std::string::npos) {
    const std::string ext = weight_path.substr(dot + 1);
    if (ext == "safetensors")
      return ml::train::ModelFormat::MODEL_FORMAT_SAFETENSORS;
  }
  return ml::train::ModelFormat::MODEL_FORMAT_BIN;
}

std::string LoadBytesFromFile(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + path);
  }
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::string buffer(size, ' ');
  if (!file.read(&buffer[0], size)) {
    throw std::runtime_error("Failed to read file: " + path);
  }
  return buffer;
}

/**
 * @brief Convert model_type text from config to ModelType.
 */
ModelType strToModelType(std::string model_type) {

  std::string model_type_lower = model_type;
  std::transform(model_type_lower.begin(), model_type_lower.end(),
                 model_type_lower.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  static const std::unordered_map<std::string, ModelType> model_type_map = {
    {"model", ModelType::MODEL},
    {"causallm", ModelType::CAUSALLM},
    {"embedding", ModelType::EMBEDDING}};

  if (model_type_map.find(model_type_lower) == model_type_map.end()) {
    return ModelType::UNKNOWN;
  }

  return model_type_map.at(model_type_lower);
}

/**
 * @brief Construct a Transformer and initialize shared config state.
 */
Transformer::Transformer(json &cfg, json &generation_cfg, json &nntr_cfg,
                         ModelType model_type) {

  std::string config_model_type_str = "Model";
  if (nntr_cfg.contains("model_type")) {
    config_model_type_str = nntr_cfg["model_type"].get<std::string>();
  }

  ModelType config_model_type = strToModelType(config_model_type_str);

  if (model_type != config_model_type) {
    throw std::runtime_error("model_type mismatch. Class Type: " +
                             std::to_string(static_cast<int>(model_type)) +
                             ", Config Type: " + config_model_type_str);
  }

  const bool skip_tokenizer = nntr_cfg.contains("skip_tokenizer") &&
                              nntr_cfg["skip_tokenizer"].get<bool>();

  // Initialize the model with the provided configurations. Vision models such
  // as TimmViT defer this to their derived constructor because the base
  // Transformer setup expects text-model fields.
  if (!(skip_tokenizer && model_type == ModelType::MODEL)) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  // Skip tokenizer if specified, or when no tokenizer_file is configured
  // (e.g. vision-encoder sub-models composed into a multimodal handle, whose
  // config carries no tokenizer). Avoids a json type_error on a null path.
  if (skip_tokenizer || !nntr_cfg.contains("tokenizer_file") ||
      nntr_cfg["tokenizer_file"].is_null()) {
    tokenizer = nullptr; // No tokenizer for this model
  } else {
    tokenizer = tokenizers::Tokenizer::FromBlobJSON(
      LoadBytesFromFile(nntr_cfg["tokenizer_file"]));
  }
};

/**
 * @brief Set common transformer parameters from model configs.
 */
void Transformer::setupParameters(json &cfg, json &generation_cfg,
                                  json &nntr_cfg) {

  /** Initialize nntr prameters */
  BATCH_SIZE = nntr_cfg["batch_size"].get<unsigned int>();
  MODEL_TENSOR_TYPE = nntr_cfg["model_tensor_type"].get<std::string>();
  INIT_SEQ_LEN = nntr_cfg["init_seq_len"];
  MAX_SEQ_LEN = nntr_cfg["max_seq_len"];
  NUM_TO_GENERATE = nntr_cfg["num_to_generate"];
  MODEL_TENSOR_TYPE = nntr_cfg["model_tensor_type"];
  MEMORY_SWAP = nntr_cfg.contains("fsu") ? nntr_cfg["fsu"].get<bool>() : false;
  FSU_LOOKAHEAD = nntr_cfg.contains("fsu_lookahead")
                    ? nntr_cfg["fsu_lookahead"].get<unsigned int>()
                    : 1;
  EMBEDDING_DTYPE = nntr_cfg["embedding_dtype"];
  FC_LAYER_DTYPE = nntr_cfg["fc_layer_dtype"];
  EMBEDDING_FILE_NAME = nntr_cfg.value("embedding_file_name", std::string());
  PLE_FILE_NAME = nntr_cfg.value("ple_file_name", std::string());

  if (cfg.contains("is_causal")) {
    IS_CAUSAL = cfg["is_causal"].get<bool>();
  } else if (cfg.contains("use_bidirectional_attention") &&
             !cfg["use_bidirectional_attention"].is_null()) {
    IS_CAUSAL = !cfg["use_bidirectional_attention"].get<bool>();
  } else if (nntr_cfg.contains("model_type") &&
             strToModelType(nntr_cfg["model_type"].get<std::string>()) ==
               ModelType::EMBEDDING &&
             cfg.contains("architectures") && cfg["architectures"].is_array() &&
             !cfg["architectures"].empty() &&
             cfg["architectures"][0].get<std::string>() == "Qwen2Model") {
    IS_CAUSAL = false;
  }

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
  if (cfg.contains("rope_theta")) {
    ROPE_THETA = cfg["rope_theta"].get<unsigned int>();
  } else if (cfg.contains("rope_parameters") &&
             cfg["rope_parameters"].contains("rope_theta")) {
    ROPE_THETA = cfg["rope_parameters"]["rope_theta"].get<unsigned int>();
  } else if (cfg.contains("rope_parameters") &&
             cfg["rope_parameters"].contains("sliding_attention")) {
    json &rope_cfg = cfg["rope_parameters"]["sliding_attention"];
    ROPE_THETA = rope_cfg.value("rope_theta", 10000);
  } else {
    ROPE_THETA = cfg.value("rope_theta", 10000);
  }
  TIE_WORD_EMBEDDINGS = cfg["tie_word_embeddings"].get<bool>();
  NORM_EPS = cfg["rms_norm_eps"];
  GQA_SIZE = NUM_HEADS / NUM_KEY_VALUE_HEADS;

  LORA_RANK = nntr_cfg.contains("lora_rank")
                ? nntr_cfg["lora_rank"].get<unsigned int>()
                : 0;
  LORA_ALPHA = nntr_cfg.contains("lora_alpha")
                 ? nntr_cfg["lora_alpha"].get<unsigned int>()
                 : 0;
  LORA_TARGET = nntr_cfg.contains("lora_target")
                  ? nntr_cfg["lora_target"].get<std::vector<std::string>>()
                  : std::vector<std::string>{};

  return;
};

bool Transformer::hasLoRA(const std::string &module_type) const {
  if (LORA_RANK == 0 || LORA_TARGET.empty())
    return false;
  return std::find(LORA_TARGET.begin(), LORA_TARGET.end(), module_type) !=
         LORA_TARGET.end();
}

void Transformer::appendLoRAProps(std::vector<std::string> &props) const {
  props.push_back(withKey("lora_rank", LORA_RANK));
  if (LORA_ALPHA > 0)
    props.push_back(withKey("lora_alpha", LORA_ALPHA));
}

/**
 * @brief Build and compile the symbolic transformer graph.
 */
void Transformer::initialize() {

  // RegisterCustomLayers
  registerCustomLayers();

  // create model and apply properties before compile()
  model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  std::vector<std::string> model_props = {
    withKey("batch_size", BATCH_SIZE), withKey("epochs", "1"),
    withKey("model_tensor_type", MODEL_TENSOR_TYPE)};
  if (MEMORY_SWAP) {
    model_props.emplace_back(withKey("fsu", "true"));
    model_props.emplace_back(withKey("fsu_lookahead", FSU_LOOKAHEAD));
  }
  model->setProperty(model_props);

  // build symbolic tensor graph and compile from (input, output)
  auto [x, y] = constructModel();

  if (model->compile(x, y, ml::train::ExecutionMode::INFERENCE)) {
    throw std::invalid_argument("Model compilation failed.");
  }

  is_initialized = true;
#ifdef DEBUG
  model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);
#endif
}

/**
 * @brief Construct the default decoder-only transformer graph.
 */
std::pair<Tensor, Tensor> Transformer::constructModel() {

  // input
  Tensor x =
    Tensor({1, 1, 1, static_cast<unsigned int>(INIT_SEQ_LEN)}, "input0");

  // embedding
  const std::string embedding_type =
    TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "embedding_layer";

  NNTR_THROW_IF(TIE_WORD_EMBEDDINGS && !EMBEDDING_FILE_NAME.empty(),
                std::invalid_argument)
    << "embedding_file_name requires untied embedding_layer";
  auto emb_props =
    buildEmbeddingLayerProperties("embedding0", NUM_VOCAB, DIM, EMBEDDING_DTYPE,
                                  EMBEDDING_SCALE, EMBEDDING_FILE_NAME);
  if (LORA_RANK > 0)
    emb_props.push_back(withKey("trainable", "false"));
  LayerHandle embedding(createLayer(embedding_type, emb_props));
  Tensor h = embedding(x);

  // transformer decoder blocks
  for (int i = 0; i < NUM_LAYERS; ++i) {
    h = createTransformerDecoderBlock(i, h);
  }

  // final rms_norm (frozen in LoRA mode)
  {
    std::vector<std::string> norm_params = {
      withKey("name", "output_norm"),
      withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false")};
    if (LORA_RANK > 0)
      norm_params.push_back(withKey("trainable", "false"));
    LayerHandle out_norm(createLayer("rms_norm", norm_params));
    h = out_norm(h);
  }

  return {x, h};
};

std::vector<std::string> Transformer::buildEmbeddingLayerProperties(
  const std::string &name, unsigned int in_dim, unsigned int out_dim,
  const std::string &weight_dtype, float scale,
  const std::string &quantized_lut_path) const {
  std::vector<std::string> props = {
    withKey("name", name),
    withKey("in_dim", std::to_string(in_dim)),
    withKey("weight_dtype", weight_dtype),
    withKey("out_dim", std::to_string(out_dim)),
    withKey("scale", std::to_string(scale)),
  };

  if (!quantized_lut_path.empty())
    props.emplace_back(withKey("quantized_lut_path", quantized_lut_path));

  return props;
}

void Transformer::initializeForTraining(float lr, unsigned int epochs) {
  registerCustomLayers();
  constructModel();

  try {
    model->addLayer(ml::train::createLayer("cross_softmax", {"name=loss"}));
  } catch (const std::exception &e) {
    std::cerr << "[initializeForTraining] loss layer: " << e.what()
              << std::endl;
  }

  std::vector<std::string> model_props = {
    withKey("batch_size", BATCH_SIZE), withKey("epochs", epochs),
    withKey("model_tensor_type", MODEL_TENSOR_TYPE)};
  model->setProperty(model_props);

  auto optimizer =
    ml::train::createOptimizer("adam", {"learning_rate=" + std::to_string(lr)});
  if (model->setOptimizer(std::move(optimizer)))
    throw std::invalid_argument("Failed to set optimizer.");

  int compile_ret = model->compile(ml::train::ExecutionMode::TRAIN);
  if (compile_ret) {
    std::cerr << "[initializeForTraining] compile() returned " << compile_ret
              << std::endl;
    throw std::invalid_argument("Model compilation for training failed.");
  }

  int init_ret = model->initialize(ml::train::ExecutionMode::TRAIN);
  if (init_ret) {
    std::cerr << "[initializeForTraining] initialize() returned " << init_ret
              << std::endl;
    throw std::invalid_argument("Model initialization for training failed.");
  }

  is_initialized = true;
}

// Returns the ordered layer names matching NeuralNetwork::save() graph
// traversal. Used by load_weight, save_weight_lora, and load_weight_lora to
// iterate weights in a consistent, deterministic order.
static std::vector<std::string> buildOrderedLayerNames(int num_layers) {
  std::vector<std::string> names;
  names.push_back("embedding0");
  for (int i = 0; i < num_layers; ++i) {
    std::string p = "layer" + std::to_string(i);
    names.push_back(p + "_attention_norm");
    names.push_back(p + "_wq");
    names.push_back(p + "_q_norm");
    names.push_back(p + "_wk");
    names.push_back(p + "_k_norm");
    names.push_back(p + "_wv");
    names.push_back(p + "_mha_core" + std::to_string(i));
    names.push_back(p + "_attention_out");
    names.push_back(p + "_ffn_norm");
    names.push_back(p + "_ffn_up");
    names.push_back(p + "_ffn_gate");
    names.push_back(p + "_ffn_down");
    names.push_back(p + "_swiglu");
    names.push_back(p + "_attention_add");
    names.push_back(p + "_ffn_add");
  }
  names.push_back("output_norm");
  names.push_back("output_of_causallm");
  return names;
}

// Returns the byte count that the nntrainer .bin serialiser writes for a
// weight tensor.  Block-quantised formats have sub-byte per-element cost, so
// getDataLen() * getDataTypeSize() is wrong for Q4_0, Q4_K, Q6_K.
static size_t weight_bytes(const ml::train::TensorDim &dim) {
  using DT = ml::train::TensorDim::DataType;
  size_t n = static_cast<size_t>(dim.getDataLen());
  switch (dim.getDataType()) {
  case DT::Q4_0:
    return (n / 32) * 18;
  case DT::Q4_K:
    return (n / 256) * 144;
  case DT::Q6_K:
    return (n / 256) * 210;
  default:
    return n * dim.getDataTypeSize();
  }
}

/**
 * @brief Load model weights from a binary nntrainer model file.
 */
void Transformer::load_weight(const std::string &weight_path) {
  if (!is_initialized) {
    throw std::runtime_error(
      "Transformer model is not initialized. Please call "
      "initialize() before load_weight().");
  }

  // The pretrained BIN file was saved WITHOUT LoRA adapter slots.
  // model->load() assigns offsets positionally, so every loraA/loraB weight
  // in the LoRA model shifts subsequent base weights by ~32-64 KB, completely
  // scrambling the loaded pretrained weights.
  //
  // Fix: read the file manually, advancing the file pointer only for base
  // weights (not loraA/loraB), so each base weight reads from its correct
  // position in the pretrained file.
  std::ifstream f(weight_path, std::ios::binary);
  if (!f.is_open())
    throw std::runtime_error("Failed to open model weights: " + weight_path);

  auto layer_names = buildOrderedLayerNames(NUM_LAYERS);

  std::unordered_set<float *> visited;

  for (const auto &lname : layer_names) {
    std::shared_ptr<ml::train::Layer> layer;
    try {
      if (model->getLayer(lname.c_str(), &layer) != 0)
        continue;
    } catch (...) {
      continue;
    }

    std::vector<float *> wdata;
    std::vector<ml::train::TensorDim> wdims;
    try {
      layer->getWeights(wdata, wdims);
    } catch (...) {
      continue;
    }

    for (unsigned int wi = 0; wi < wdata.size(); ++wi) {
      if (!wdata[wi])
        continue;
      // Deduplicate shared tensors (e.g. TieWordEmbedding "Embedding").
      if (!visited.insert(wdata[wi]).second)
        continue;

      const std::string &wname = layer->getWeightName(wi);
      bool is_lora = (wname.find(":loraA") != std::string::npos ||
                      wname.find(":loraB") != std::string::npos);
      if (is_lora)
        continue; // Skip: not in pretrained file. Keeps initialized value.

      size_t bytes = weight_bytes(wdims[wi]);
      f.read(reinterpret_cast<char *>(wdata[wi]), bytes);
      if (!f)
        throw std::runtime_error("load_weight: read failed at weight '" +
                                 wname + "' (offset " +
                                 std::to_string(f.tellg()) + ")");
    }
  }
  std::cout << "[load_weight] Loaded base weights from " << weight_path
            << " (LoRA adapters kept at initialized values)\n";
};

/**
 * @brief Save model weights to a binary nntrainer model file.
 */
void Transformer::save_weight(const std::string &weight_path) {

  if (!is_initialized) {
    throw std::runtime_error(
      "Transformer model is not initialized. Please call "
      "initialize() before save_weight().");
  }

  try {
    model->save(weight_path, formatFromExtension(weight_path));
  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to save model weights: " +
                             std::string(e.what()));
  }
};

/**
 * @brief Save model weights with optional dtype conversion.
 */
void Transformer::save_weight(
  const std::string &weight_path, ml::train::TensorDim::DataType dtype,
  const std::map<std::string, ml::train::TensorDim::DataType> &layer_dtype_map,
  ml::train::ISA target_isa) {

  if (!is_initialized) {
    throw std::runtime_error(
      "Transformer model is not initialized. Please call "
      "initialize() before save_weight().");
  }

  try {
    model->save(weight_path, formatFromExtension(weight_path), dtype,
                layer_dtype_map, target_isa);

  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to save model weights with dtype: " +
                             std::string(e.what()));
  }
};

/**
 * @brief Repack all QS4CX weights after loading.
 */
void Transformer::repack_weight() {
  if (!is_initialized) {
    throw std::runtime_error(
      "Transformer model is not initialized. Please call "
      "initialize() before repack_weight().");
  }
  std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &, void *)>
    fn = [](ml::train::Layer &l, nntrainer::RunLayerContext &context, void *) {
      auto weights = context.getWeights();
      for (auto &w : weights) {
        if (w->getVariableRef().getDataType() ==
            ml::train::TensorDim::DataType::QS4CX) {
          w->getVariableRef().pack();
        }
      }
    };
  try {
    model->forEachLayer(fn, nullptr);
    ml_logd("QS4CX weights repacked successfully");
  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to repack weights: " +
                             std::string(e.what()));
  }
};

void Transformer::save_weight_lora(const std::string &weight_path) {
  if (!is_initialized)
    throw std::runtime_error(
      "Model not initialized before save_weight_lora().");

  std::ofstream f(weight_path, std::ios::binary);
  if (!f.is_open())
    throw std::runtime_error("Failed to open " + weight_path + " for writing.");

  auto layer_names = buildOrderedLayerNames(NUM_LAYERS);
  std::unordered_set<float *> visited;
  size_t total_bytes = 0;

  for (const auto &lname : layer_names) {
    std::shared_ptr<ml::train::Layer> layer;
    try {
      if (model->getLayer(lname.c_str(), &layer) != 0)
        continue;
    } catch (...) {
      continue;
    }

    std::vector<float *> wdata;
    std::vector<ml::train::TensorDim> wdims;
    try {
      layer->getWeights(wdata, wdims);
    } catch (...) {
      continue;
    }

    for (unsigned int wi = 0; wi < wdata.size(); ++wi) {
      if (!wdata[wi])
        continue;
      if (!visited.insert(wdata[wi]).second)
        continue;

      const std::string &wname = layer->getWeightName(wi);
      if (wname.find(":loraA") == std::string::npos &&
          wname.find(":loraB") == std::string::npos)
        continue;

      size_t bytes =
        static_cast<size_t>(wdims[wi].getDataLen()) * sizeof(float);
      f.write(reinterpret_cast<const char *>(wdata[wi]), bytes);
      total_bytes += bytes;
    }
  }

  std::cout << "[save_weight_lora] Saved LoRA adapters to " << weight_path
            << " (" << (total_bytes / 1024 / 1024) << " MB)\n";
}

void Transformer::load_weight_lora(const std::string &base_path,
                                   const std::string &lora_path) {
  load_weight(base_path);

  std::ifstream f(lora_path, std::ios::binary);
  if (!f.is_open())
    throw std::runtime_error("Failed to open LoRA adapters: " + lora_path);

  auto layer_names = buildOrderedLayerNames(NUM_LAYERS);
  std::unordered_set<float *> visited;

  for (const auto &lname : layer_names) {
    std::shared_ptr<ml::train::Layer> layer;
    try {
      if (model->getLayer(lname.c_str(), &layer) != 0)
        continue;
    } catch (...) {
      continue;
    }

    std::vector<float *> wdata;
    std::vector<ml::train::TensorDim> wdims;
    try {
      layer->getWeights(wdata, wdims);
    } catch (...) {
      continue;
    }

    for (unsigned int wi = 0; wi < wdata.size(); ++wi) {
      if (!wdata[wi])
        continue;
      if (!visited.insert(wdata[wi]).second)
        continue;

      const std::string &wname = layer->getWeightName(wi);
      if (wname.find(":loraA") == std::string::npos &&
          wname.find(":loraB") == std::string::npos)
        continue;

      size_t bytes =
        static_cast<size_t>(wdims[wi].getDataLen()) * sizeof(float);
      f.read(reinterpret_cast<char *>(wdata[wi]), bytes);
      if (!f)
        throw std::runtime_error("load_weight_lora: read failed at '" + wname +
                                 "'");
    }
  }

  std::cout << "[load_weight_lora] Loaded LoRA adapters from " << lora_path
            << "\n";
}

void Transformer::setDataset(const ml::train::DatasetModeType &mode,
                             std::shared_ptr<ml::train::Dataset> dataset) {
  if (!is_initialized)
    throw std::runtime_error("Model not initialized before setDataset().");
  if (model->setDataset(mode, dataset))
    throw std::runtime_error("Failed to set dataset on model.");
}

void Transformer::train() {
  if (!is_initialized)
    throw std::runtime_error("Model not initialized before train().");
  if (model->train())
    throw std::runtime_error("model->train() returned error.");
}

void Transformer::train(std::function<void(void *)> epoch_cb, void *epoch_data,
                        std::function<bool(void *)> stop_cb, void *stop_data) {
  if (!is_initialized)
    throw std::runtime_error("Model not initialized before train().");
  auto actual_stop = stop_cb ? stop_cb : [](void *) -> bool { return false; };
  if (model->train({}, actual_stop, stop_data, epoch_cb, epoch_data))
    throw std::runtime_error("model->train() returned error.");
}

ml::train::RunStats Transformer::getTrainingStats() {
  return model->getTrainingStats();
}

ml::train::RunStats Transformer::getValidStats() {
  return model->getValidStats();
}

void Transformer::summarize(std::ostream &out, unsigned int type) {
  if (!is_initialized)
    throw std::runtime_error("Model not initialized before summarize().");
  model->summarize(out, static_cast<ml_train_summary_type_e>(type));
}

void Transformer::exportWeightsToFile(const std::string &path) {
  if (!is_initialized)
    throw std::runtime_error(
      "Model not initialized before exportWeightsToFile().");
  std::ofstream f(path);
  if (!f.is_open())
    throw std::runtime_error("Cannot open " + path + " for weight export.");

  std::vector<std::string> layer_names;
  layer_names.push_back("embedding0");
  for (int i = 0; i < NUM_LAYERS; ++i) {
    std::string p = "layer" + std::to_string(i);
    layer_names.push_back(p + "_attention_norm");
    layer_names.push_back(p + "_wq");
    layer_names.push_back(p + "_q_norm"); // Qwen3 QK norm
    layer_names.push_back(p + "_wk");
    layer_names.push_back(p + "_k_norm"); // Qwen3 QK norm
    layer_names.push_back(p + "_wv");
    layer_names.push_back(p + "_attention_out");
    layer_names.push_back(p + "_ffn_norm");
    layer_names.push_back(p + "_ffn_up");
    layer_names.push_back(p + "_ffn_gate");
    layer_names.push_back(p + "_ffn_down");
  }
  layer_names.push_back("output_norm");
  layer_names.push_back("output_of_causallm");

  for (const auto &lname : layer_names) {
    std::shared_ptr<ml::train::Layer> layer;
    try {
      if (model->getLayer(lname.c_str(), &layer) != 0)
        continue;
    } catch (...) {
      continue;
    }

    std::vector<float *> wdata;
    std::vector<ml::train::TensorDim> wdims;
    try {
      layer->getWeights(wdata, wdims);
    } catch (...) {
      continue;
    }

    for (unsigned int wi = 0; wi < wdata.size(); ++wi) {
      try {
        const std::string &wname = layer->getWeightName(wi);
        unsigned int n = wdims[wi].getDataLen();
        double norm = 0.0;
        if (wdata[wi]) {
          for (unsigned int k = 0; k < n; ++k)
            norm += static_cast<double>(wdata[wi][k]) * wdata[wi][k];
          norm = std::sqrt(norm);
        }
        f << lname << "/" << wname << ": " << std::fixed << std::setprecision(6)
          << norm << "\n";
      } catch (...) {
        continue;
      }
    }
  }
}

/**
 * @brief Run a transformer model for a prompt.
 */
void Transformer::run(const WSTR prompt, bool do_sample,
                      const WSTR system_prompt, const WSTR tail_prompt,
                      bool log_output) {
  if (!is_initialized) {
    throw std::runtime_error(
      "Transformer model is not initialized. Please call "
      "initialize() before run().");
  }
  ///@note This part should be filled in.
  /// The run action can be defined by the precedent classes.
}

/**
 * @brief Create one decoder block with attention and feed-forward layers.
 */
Tensor Transformer::createTransformerDecoderBlock(const int layer_id,
                                                  Tensor input) {

  std::vector<std::string> attn_norm_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_attention_norm"),
    withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false")};
  if (LORA_RANK > 0)
    attn_norm_props.push_back(withKey("trainable", "false"));
  LayerHandle attn_norm(createLayer("rms_norm", attn_norm_props));
  Tensor normed = attn_norm(input);

  Tensor att_out = createAttention(layer_id, INIT_SEQ_LEN, NUM_HEADS, HEAD_DIM,
                                   normed, normed, normed);

  LayerHandle decoder_add(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_decoder_add")}));
  Tensor residual = decoder_add({input, att_out});

  std::vector<std::string> ffn_norm_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_ffn_norm"),
    withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false")};
  if (LORA_RANK > 0)
    ffn_norm_props.push_back(withKey("trainable", "false"));
  LayerHandle ffn_norm(createLayer("rms_norm", ffn_norm_props));
  Tensor ffn_normed = ffn_norm(residual);

  Tensor ffn_out = createMlp(layer_id, DIM, INTERMEDIATE_SIZE, ffn_normed);

  LayerHandle decoder_output(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_decoder_output")}));
  return decoder_output({residual, ffn_out});
}

/**
 * @brief Create external KV-cache placeholder tensors for one layer.
 */
std::pair<Tensor, Tensor>
Transformer::createKVCachePlaceholders(const int layer_id, int n_heads) {
  const unsigned int max_timestep = static_cast<unsigned int>(MAX_SEQ_LEN);
  const unsigned int kv_width =
    static_cast<unsigned int>(HEAD_DIM * n_heads / GQA_SIZE);
#ifdef ENABLE_FP16
  ml::train::TensorDim cache_dim(
    {BATCH_SIZE, 1, max_timestep, kv_width},
    {ml::train::TensorDim::Format::NCHW, ml::train::TensorDim::DataType::FP16});

  Tensor cache_k(cache_dim, "cache_k_l" + std::to_string(layer_id));
  Tensor cache_v(cache_dim, "cache_v_l" + std::to_string(layer_id));
  return {cache_k, cache_v};
#else
  const std::string cache_shape = std::to_string(BATCH_SIZE) +
                                  ":1:" + std::to_string(max_timestep) + ":" +
                                  std::to_string(kv_width);

  LayerHandle cache_k_input(createLayer(
    "input",
    {withKey("name", "cache_k_l" + std::to_string(layer_id)),
     withKey("input_shape", cache_shape), withKey("input_dtype", "UINT16")}));
  LayerHandle cache_v_input(createLayer(
    "input",
    {withKey("name", "cache_v_l" + std::to_string(layer_id)),
     withKey("input_shape", cache_shape), withKey("input_dtype", "UINT16")}));

  return {cache_k_input(Tensor()), cache_v_input(Tensor())};
#endif
}

/**
 * @brief Create the default attention subgraph.
 */
Tensor Transformer::createAttention(const int layer_id, int seq_len,
                                    int n_heads, int head_dim, Tensor query,
                                    Tensor key, Tensor value) {

  // Q layer
  std::vector<std::string> wq_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_wq"),
    withKey("unit", head_dim * n_heads), withKey("disable_bias", "true"),
    withKey("weight_initializer", "ones")};
  if (hasLoRA("wq"))
    appendLoRAProps(wq_props);
  else if (LORA_RANK > 0)
    wq_props.push_back(withKey("trainable", "false"));
  LayerHandle wq(createLayer("fully_connected", wq_props));
  Tensor q = wq(query);

  // K layer
  std::vector<std::string> wk_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_wk"),
    withKey("unit", head_dim * n_heads / GQA_SIZE),
    withKey("disable_bias", "true"), withKey("weight_initializer", "ones")};
  if (hasLoRA("wk"))
    appendLoRAProps(wk_props);
  else if (LORA_RANK > 0)
    wk_props.push_back(withKey("trainable", "false"));
  LayerHandle wk(createLayer("fully_connected", wk_props));
  Tensor k = wk(key);

  // V layer
  std::vector<std::string> wv_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_wv"),
    withKey("unit", head_dim * n_heads / GQA_SIZE),
    withKey("disable_bias", "true"), withKey("weight_initializer", "ones")};
  if (hasLoRA("wv"))
    appendLoRAProps(wv_props);
  else if (LORA_RANK > 0)
    wv_props.push_back(withKey("trainable", "false"));
  LayerHandle wv(createLayer("fully_connected", wv_props));
  Tensor v = wv(value);

  // External KV cache placeholders (per-layer). Their actual storage is owned
  // by the host (KVCacheManager) and bound at runtime via setExternalTensors.
  auto [cache_k, cache_v] = createKVCachePlaceholders(layer_id, n_heads);

  // Attention core layer
  LayerHandle mha(createLayer(
    "mha_core",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention"),
     withKey("num_heads", n_heads), withKey("num_heads_kv", n_heads / GQA_SIZE),
     withKey("max_timestep", std::to_string(MAX_SEQ_LEN)),
     withKey("sliding_window", (layer_id + 1) % SLIDING_WINDOW_PATTERN
                                 ? SLIDING_WINDOW
                                 : UINT_MAX),
     withKey("rope_theta", ROPE_THETA),
     withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
     withKey("is_causal", IS_CAUSAL ? "true" : "false")}));
  Tensor a = mha({q, k, v, cache_k, cache_v});

  // O layer
  std::vector<std::string> wo_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_attention_out"),
    withKey("unit", DIM), withKey("disable_bias", "true"),
    withKey("weight_initializer", "ones")};
  if (hasLoRA("wo"))
    appendLoRAProps(wo_props);
  else if (LORA_RANK > 0)
    wo_props.push_back(withKey("trainable", "false"));
  LayerHandle wo(createLayer("fully_connected", wo_props));
  return wo(a);
}

/**
 * @brief Create the default feed-forward subgraph.
 */
Tensor Transformer::createMlp(const int layer_id, int dim, int hidden_dim,
                              Tensor input) {

  std::vector<std::string> ffn_up_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_ffn_up"),
    withKey("unit", hidden_dim), withKey("disable_bias", "true"),
    withKey("weight_initializer", "ones")};
  if (hasLoRA("ffn_up"))
    appendLoRAProps(ffn_up_props);
  else if (LORA_RANK > 0)
    ffn_up_props.push_back(withKey("trainable", "false"));
  LayerHandle ffn_up(createLayer("fully_connected", ffn_up_props));
  Tensor up = ffn_up(input);

  std::vector<std::string> ffn_gate_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_ffn_gate"),
    withKey("unit", hidden_dim), withKey("disable_bias", "true"),
    withKey("weight_initializer", "ones")};
  if (hasLoRA("ffn_gate"))
    appendLoRAProps(ffn_gate_props);
  else if (LORA_RANK > 0)
    ffn_gate_props.push_back(withKey("trainable", "false"));
  LayerHandle ffn_gate(createLayer("fully_connected", ffn_gate_props));
  Tensor gate = ffn_gate(input);

  /// @note nntrainer binary stores mlp weights in up, gate order.
  /// For backward compatibility,
  /// * layers are in up, gate order
  /// * swiglu input[0] = gate
  /// * swiglu input[1] = up
  LayerHandle swiglu(createLayer(
    "swiglu",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_swiglu")}));
  Tensor act = swiglu({up, gate}, {1, 0});

  std::vector<std::string> ffn_down_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
    withKey("unit", dim), withKey("disable_bias", "true"),
    withKey("weight_initializer", "ones")};
  if (hasLoRA("ffn_down"))
    appendLoRAProps(ffn_down_props);
  else if (LORA_RANK > 0)
    ffn_down_props.push_back(withKey("trainable", "false"));
  LayerHandle ffn_down(createLayer("fully_connected", ffn_down_props));
  return ffn_down(act);
}

/**
 * @brief Register custom CausalLM layers in the nntrainer app context.
 */
void Transformer::registerCustomLayers() {
  static std::once_flag registered;
  std::call_once(registered, []() {
    const auto &ct_engine = nntrainer::Engine::Global();
    const auto app_context = static_cast<nntrainer::AppContext *>(
      ct_engine.getRegisteredContext("cpu"));

    app_context->registerFactory(nntrainer::createLayer<causallm::SwiGLULayer>);
    app_context->registerFactory(
      nntrainer::createLayer<causallm::RMSNormLayer>);
    app_context->registerFactory(
      nntrainer::createLayer<causallm::MHACoreLayer>);
    app_context->registerFactory(
      nntrainer::createLayer<causallm::TieWordEmbedding>);
    app_context->registerFactory(
      nntrainer::createLayer<causallm::EmbeddingLayer>);
  });
}

} // namespace causallm
