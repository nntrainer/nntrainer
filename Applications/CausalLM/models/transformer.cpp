// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   transformer.cpp
 * @date   10 July 2025
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @author Pranjal Thapliyal <p.thapliyal@samsung.com>
 * @author Sumon Nath <sumon.nath@samsung.com>
 * @author Niket Agarwal <niket.a@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This file defines Transformer's basic actions
 */

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <mutex>
#include <unordered_map>

#include <app_context.h>
#include <engine.h>
#include <model.h>

#include <llm_util.hpp>
#include <tokenizers_cpp.h>
#include <transformer.h>

#include <cpu_backend.h>
#include <embedding_layer.h>
#include <fc_layer.h>
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
  INTERMEDIATE_SIZE =
    cfg.contains("intermediate_size") ? cfg["intermediate_size"].get<int>() : 0;
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
  TIE_WORD_EMBEDDINGS = cfg.contains("tie_word_embeddings")
                          ? cfg["tie_word_embeddings"].get<bool>()
                          : false;
  NORM_EPS =
    cfg.contains("rms_norm_eps") ? cfg["rms_norm_eps"].get<float>() : 1e-5;
  GQA_SIZE = NUM_HEADS / NUM_KEY_VALUE_HEADS;

  LORA_RANK = nntr_cfg.contains("lora_rank")
                ? nntr_cfg["lora_rank"].get<unsigned int>()
                : 0;
  LORA_ALPHA = nntr_cfg.contains("lora_alpha")
                 ? nntr_cfg["lora_alpha"].get<unsigned int>()
                 : 0;
  LORA_TARGET = nntr_cfg.contains("lora_target")
                  ? nntr_cfg["lora_target"].get<std::vector<std::string>>()
                  : std::vector<std::string>();
  LORA_CLIP_GRAD = nntr_cfg.contains("lora_clip_grad_by_norm")
                     ? nntr_cfg["lora_clip_grad_by_norm"].get<float>()
                     : 0.0f;
  TRAIN_NORMS = nntr_cfg.contains("lora_train_norms") &&
                nntr_cfg["lora_train_norms"].get<bool>();
  LORA_QAT =
    nntr_cfg.contains("lora_qat") && nntr_cfg["lora_qat"].get<bool>();
  LORA_WEIGHT_Q4 = nntr_cfg.contains("lora_weight_q4") &&
                  nntr_cfg["lora_weight_q4"].get<bool>();

  return;
};

bool Transformer::hasLoRA(const std::string &module_type) const {
  return LORA_RANK > 0 &&
        std::find(LORA_TARGET.begin(), LORA_TARGET.end(), module_type) !=
          LORA_TARGET.end();
}

void Transformer::appendLoRAProps(std::vector<std::string> &props) const {
  props.emplace_back(withKey("lora_rank", std::to_string(LORA_RANK)));
  if (LORA_ALPHA > 0)
    props.emplace_back(withKey("lora_alpha", std::to_string(LORA_ALPHA)));
  if (LORA_CLIP_GRAD > 0.0f)
    props.emplace_back(
      withKey("clip_grad_by_norm", std::to_string(LORA_CLIP_GRAD)));
  if (LORA_QAT)
    props.emplace_back(withKey("lora_qat", "true"));
  if (LORA_WEIGHT_Q4)
    props.emplace_back(withKey("lora_weight_q4", "true"));
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
 * @brief Build and compile the symbolic transformer graph for LoRA training.
 */
void Transformer::initializeForTraining(float lr, unsigned int epochs) {
  FOR_TRAINING = true;

  registerCustomLayers();

  model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  std::vector<std::string> model_props = {
    withKey("batch_size", BATCH_SIZE), withKey("epochs", std::to_string(epochs)),
    withKey("model_tensor_type", MODEL_TENSOR_TYPE)};
  model->setProperty(model_props);

  auto optimizer =
    ml::train::createOptimizer("adam", {withKey("learning_rate", lr)});
  if (model->setOptimizer(std::move(optimizer))) {
    throw std::invalid_argument("Failed to set optimizer.");
  }

  auto [x, h] = constructModel();

  LayerHandle loss(
    createLayer("cross_softmax", {withKey("name", "loss")}));
  Tensor y = loss(h);

  // compile(Tensor, Tensor, mode) internally compiles + initializes.
  if (model->compile(x, y, ml::train::ExecutionMode::TRAIN)) {
    throw std::invalid_argument("Training model compilation failed.");
  }

  is_initialized = true;
}

/**
 * @brief Set the dataset used for training/validation.
 */
void Transformer::setDataset(const ml::train::DatasetModeType &mode,
                             std::shared_ptr<ml::train::Dataset> dataset) {
  if (model->setDataset(mode, dataset)) {
    throw std::invalid_argument("Failed to set dataset.");
  }
}

/**
 * @brief Run training for the epochs configured in initializeForTraining().
 */
void Transformer::train(std::function<void(void *)> epoch_complete_cb,
                        void *epoch_data, std::function<bool(void *)> stop_cb,
                        void *stop_data) {
  if (!stop_cb)
    stop_cb = [](void *) { return false; };
  if (!epoch_complete_cb)
    epoch_complete_cb = [](void *) {};

  if (model->train({}, stop_cb, stop_data, epoch_complete_cb, epoch_data)) {
    throw std::runtime_error("Training failed.");
  }
}

ml::train::RunStats Transformer::getTrainingStats() {
  return model->getTrainingStats();
}

ml::train::RunStats Transformer::getValidStats() {
  return model->getValidStats();
}

void Transformer::forEachLayer(
  std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &, void *)>
    fn,
  void *user_data) {
  model->forEachLayer(fn, user_data);
}

/**
 * @brief Save only the loraA/loraB weights (raw FP32) to a file.
 */
void Transformer::save_weight_lora(const std::string &lora_path) {
  std::ofstream file(lora_path, std::ios::binary);
  NNTR_THROW_IF(!file.is_open(), std::runtime_error)
    << "Failed to open lora output file: " << lora_path;

  std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &, void *)>
    fn = [&file](ml::train::Layer &l, nntrainer::RunLayerContext &context,
                 void *) {
      if (l.getType() != "fully_connected")
        return;
      for (auto &w : context.getWeights()) {
        const std::string &name = w->getName();
        if (name.find(":loraA") != std::string::npos ||
            name.find(":loraB") != std::string::npos)
          w->getVariableRef().save(file);
      }
    };
  model->forEachLayer(fn, nullptr);
}

/**
 * @brief Load a pretrained (non-LoRA) base checkpoint into a graph that has
 *        LoRA weights, then optionally overlay a saved LoRA adapter.
 */
void Transformer::load_weight_lora(const std::string &base_path,
                                   const std::string &lora_path) {
  NNTR_THROW_IF(!is_initialized, std::runtime_error)
    << "Transformer model is not initialized. Please call "
       "initializeForTraining() before load_weight_lora().";

  // Step 1: build a throwaway copy of the graph that is byte-for-byte the
  // graph the checkpoint was written from, and load the base checkpoint into
  // it with the ordinary (already-correct) loader.
  const unsigned int saved_lora_rank = LORA_RANK;
  const bool saved_for_training = FOR_TRAINING;
  LORA_RANK = 0;
  FOR_TRAINING = false;

  auto restore_graph_flags = [&]() {
    LORA_RANK = saved_lora_rank;
    FOR_TRAINING = saved_for_training;
  };

  ModelHandle base_model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
  base_model->setProperty(
    {withKey("batch_size", BATCH_SIZE), withKey("epochs", "1"),
     withKey("model_tensor_type", MODEL_TENSOR_TYPE)});
  auto [bx, by] = constructModel();
  if (base_model->compile(bx, by, ml::train::ExecutionMode::INFERENCE)) {
    restore_graph_flags();
    throw std::invalid_argument(
      "Base (no-LoRA) model compilation failed during load_weight_lora.");
  }
  base_model->load(base_path, formatFromExtension(base_path));

  restore_graph_flags();

  // Step 2: index the base model's weights by their (layer-prefixed) name.
  std::unordered_map<std::string, const nntrainer::Tensor *> base_weights;
  std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &, void *)>
    collect = [&base_weights](ml::train::Layer &, nntrainer::RunLayerContext &context,
                              void *) {
      for (auto &w : context.getWeights())
        base_weights.emplace(w->getName(), &w->getVariableRef());
    };
  base_model->forEachLayer(collect, nullptr);

  // Step 3: copy every matching weight (by name) into this model. Only
  // loraA/loraB are expected to go unmatched (they do not exist in a
  // pretrained checkpoint); anything else unmatched means the two graphs
  // disagree and the model would silently train from random weights, so
  // fail loudly instead.
  //
  // For Q4_0/Q6_K tensors, copyData() is not supported, so we use a raw
  // memcpy of the backing buffer instead. Both tensors come from the same
  // layer in the same graph topology, so they have the same dtype and size.
  std::vector<std::string> unmatched;
  unsigned int matched = 0;
  std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &, void *)>
    apply = [&](ml::train::Layer &, nntrainer::RunLayerContext &context,
                void *) {
      for (auto &w : context.getWeights()) {
        const std::string &name = w->getName();
        auto it = base_weights.find(name);
        if (it != base_weights.end()) {
          nntrainer::Tensor &dst = w->getVariableRef();
          const nntrainer::Tensor &src = *it->second;
          const size_t bytes = src.getMemoryBytes();
          std::memcpy(dst.getData<char>(), src.getData<char>(), bytes);
          ++matched;
        } else if (name.find(":loraA") == std::string::npos &&
                   name.find(":loraB") == std::string::npos) {
          unmatched.push_back(name);
        }
      }
    };
  model->forEachLayer(apply, nullptr);

  NNTR_THROW_IF(!unmatched.empty(), std::runtime_error)
    << "load_weight_lora: " << unmatched.size()
    << " non-LoRA weight(s) had no counterpart in the base checkpoint graph "
       "(first: "
    << unmatched.front()
    << "). The base graph must match the graph the checkpoint was saved "
       "from.";
  NNTR_THROW_IF(matched == 0, std::runtime_error)
    << "load_weight_lora: no weights were loaded from " << base_path;

  // Step 4: overlay a previously saved LoRA adapter, if given.
  if (!lora_path.empty()) {
    std::ifstream lora_file(lora_path, std::ios::binary);
    NNTR_THROW_IF(!lora_file.is_open(), std::runtime_error)
      << "Failed to open lora file: " << lora_path;

    std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &, void *)>
      load_lora = [&lora_file](ml::train::Layer &l,
                               nntrainer::RunLayerContext &context, void *) {
        if (l.getType() != "fully_connected")
          return;
        for (auto &w : context.getWeights()) {
          const std::string &name = w->getName();
          if (name.find(":loraA") != std::string::npos ||
              name.find(":loraB") != std::string::npos)
            w->getVariableRef().read(lora_file);
        }
      };
    model->forEachLayer(load_lora, nullptr);
  }
}

namespace {

// ---------------------------------------------------------------------------
// Q4_0 helpers (block = 32 elements, 18 bytes: 2B FP16 scale + 16B nibbles)
// ---------------------------------------------------------------------------

/**
 * @brief Build repacked Q4_0 bytes from an FP32 weight stored in nntrainer's
 *        (K, N) layout. Mirrors GgmlQuantizer's own pipeline: transpose
 *        K x N -> N x K, then quantize_q4_0, then repack_q4_0. The repacked
 *        format is required by the W4A8 GEMM kernel.
 */
std::vector<uint8_t> build_q4_0_natural(const float *data_KN, size_t K,
                                        size_t N) {
  std::vector<float> transposed(N * K);
  for (size_t n = 0; n < N; ++n)
    for (size_t k = 0; k < K; ++k)
      transposed[n * K + k] = data_KN[k * N + n];

  size_t out_size = (N * K / 32) * 18;
  std::vector<uint8_t> tmp(out_size);
  nntrainer::quantize_q4_0(transposed.data(), tmp.data(), (int64_t)N,
                          (int64_t)K, nullptr);

  std::vector<uint8_t> out(out_size);
  nntrainer::repack_q4_0(out.data(), tmp.data(), out_size, (unsigned int)N,
                        (unsigned int)K);
  return out;
}

/** Minimal FP32 -> FP16 conversion for writing Q4_0 block scale fields. */
uint16_t q40_fp32_to_fp16(float v) {
  union {
    float f;
    uint32_t u;
  } x{v};
  const uint32_t s = (x.u >> 16) & 0x8000u;
  const int e = ((x.u >> 23) & 0xFFu) - 127 + 15;
  const uint32_t m = x.u & 0x7FFFFFu;
  if (e <= 0)
    return (uint16_t)s;
  if (e >= 31)
    return (uint16_t)(s | 0x7C00u);
  return (uint16_t)(s | ((uint32_t)e << 10) | (m >> 13));
}

/**
 * @brief Build repacked Q4_0 bytes using pre-specified per-block EMA scales
 *        (force-feed), instead of recomputing a natural per-block scale
 *        from the data. block_d_NK must be indexed in N x K layout -- the
 *        same layout FullyConnectedLayer::fakeQuantizeQ4_0 tracks -- so the
 *        saved adapter matches exactly what training quantized against.
 *        Output format matches build_q4_0_natural exactly.
 */
std::vector<uint8_t> build_q4_0_forced_blocks(const float *data_KN, size_t K,
                                              size_t N,
                                              const std::vector<float> &block_d_NK) {
  std::vector<float> transposed(N * K);
  for (size_t n = 0; n < N; ++n)
    for (size_t k = 0; k < K; ++k)
      transposed[n * K + k] = data_KN[k * N + n];

  const size_t num_blocks = N * K / 32;
  const size_t out_size = num_blocks * 18;
  std::vector<uint8_t> tmp(out_size, 0);

  for (size_t b = 0; b < num_blocks; ++b) {
    const float *blk_data = transposed.data() + b * 32;
    float d = (b < block_d_NK.size() && block_d_NK[b] > 1e-10f)
                ? block_d_NK[b]
                : 1.0f;

    uint16_t d_fp16 = q40_fp32_to_fp16(d);
    uint8_t *blk = tmp.data() + b * 18;
    std::memcpy(blk, &d_fp16, 2);

    // Q4_0: quant stored as q+8 in [0,15]; lower nibble = elem[j], upper =
    // elem[j+16].
    for (int j = 0; j < 16; ++j) {
      int q0 = (int)std::round(blk_data[j] / d) + 8;
      int q1 = (int)std::round(blk_data[j + 16] / d) + 8;
      q0 = std::max(0, std::min(15, q0));
      q1 = std::max(0, std::min(15, q1));
      blk[2 + j] = (uint8_t)((q0 & 0x0F) | ((q1 & 0x0F) << 4));
    }
  }

  std::vector<uint8_t> out(out_size);
  nntrainer::repack_q4_0(out.data(), tmp.data(), out_size, (unsigned)N,
                        (unsigned)K);
  return out;
}

} // namespace

/**
 * @brief Save LoRA adapters as Q4_0. When LORA_QAT calibrated per-block EMA
 *        scales during training, force-feed them; otherwise fall back to a
 *        natural (recomputed) per-block scale, i.e. plain PTQ.
 */
void Transformer::save_weight_lora_q4(const std::string &path) {
  NNTR_THROW_IF(!is_initialized, std::runtime_error)
    << "Model not initialized before save_weight_lora_q4().";

  std::ofstream f(path, std::ios::binary);
  NNTR_THROW_IF(!f.is_open(), std::runtime_error)
    << "Failed to open " << path << " for writing.";

  size_t total_blocks = 0;
  std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &, void *)>
    fn = [&](ml::train::Layer &l, nntrainer::RunLayerContext &context, void *) {
      if (l.getType() != "fully_connected")
        return;
      const std::string lname = l.getName();

      for (auto &w : context.getWeights()) {
        const std::string &wname = w->getName();
        const bool is_loraA = wname.find(":loraA") != std::string::npos;
        const bool is_loraB = wname.find(":loraB") != std::string::npos;
        if (!is_loraA && !is_loraB)
          continue;

        const nntrainer::Tensor &t = w->getVariableRef();
        const size_t K = t.getDim().height();
        const size_t N = t.getDim().width();
        const uint32_t total_elems = static_cast<uint32_t>(K * N);

        // QAT: force-feed the EMA block scales (N x K layout) calibrated
        // during training. Non-QAT: fall back to natural (PTQ) scales.
        std::vector<uint8_t> q4_bytes;
        if (LORA_QAT) {
          auto [a_bd, b_bd] =
            nntrainer::FullyConnectedLayer::getRegisteredBlockScales(lname);
          const std::vector<float> &block_d = is_loraA ? a_bd : b_bd;
          q4_bytes = block_d.empty()
                       ? build_q4_0_natural(t.getData<float>(), K, N)
                       : build_q4_0_forced_blocks(t.getData<float>(), K, N,
                                                  block_d);
        } else {
          q4_bytes = build_q4_0_natural(t.getData<float>(), K, N);
        }

        f.write(reinterpret_cast<const char *>(&total_elems),
               sizeof(total_elems));
        f.write(reinterpret_cast<const char *>(q4_bytes.data()),
               q4_bytes.size());
        total_blocks += q4_bytes.size() / 18;
      }
    };
  model->forEachLayer(fn, nullptr);

  std::cout << "[save_weight_lora_q4] Saved Q4_0 LoRA adapters to " << path
            << " (" << total_blocks << " blocks, " << (total_blocks * 18 / 1024)
            << " KB)" << std::endl;
}

/**
 * @brief Load a pretrained (non-LoRA) base checkpoint, then load Q4_0 LoRA
 *        adapters directly into Q4_0-dtype loraA/loraB tensors.
 */
void Transformer::load_weight_lora_q4(const std::string &base_path,
                                      const std::string &lora_q4_path) {
  NNTR_THROW_IF(!is_initialized, std::runtime_error)
    << "Transformer model is not initialized. Please call "
       "initialize() before load_weight_lora_q4().";

  // The current graph has LoRA weights (loraA/loraB) spliced into every
  // targeted FC layer, but the checkpoint at base_path has no such slots.
  // model->load() assigns file offsets by walking the graph positionally, so
  // loading the checkpoint directly into this model (as a naive load_weight()
  // call would) desyncs every weight after the first LoRA slot it hits --
  // this crashed with a heap-corrupting read inside NeuralNetwork::load().
  // Fix: build a throwaway no-LoRA graph matching the checkpoint's own
  // topology, load into that, then copy every matching weight over by name
  // (mirroring load_weight_lora()'s FP32 path).
  const unsigned int saved_lora_rank = LORA_RANK;
  LORA_RANK = 0;
  auto restore_lora_rank = [&]() { LORA_RANK = saved_lora_rank; };

  ModelHandle base_model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
  base_model->setProperty(
    {withKey("batch_size", BATCH_SIZE), withKey("epochs", "1"),
     withKey("model_tensor_type", MODEL_TENSOR_TYPE)});
  auto [bx, by] = constructModel();
  if (base_model->compile(bx, by, ml::train::ExecutionMode::INFERENCE)) {
    restore_lora_rank();
    throw std::invalid_argument(
      "Base (no-LoRA) model compilation failed during load_weight_lora_q4.");
  }
  base_model->load(base_path, formatFromExtension(base_path));

  restore_lora_rank();

  std::unordered_map<std::string, const nntrainer::Tensor *> base_weights;
  std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &, void *)>
    collect = [&base_weights](ml::train::Layer &, nntrainer::RunLayerContext &context,
                              void *) {
      for (auto &w : context.getWeights())
        base_weights.emplace(w->getName(), &w->getVariableRef());
    };
  base_model->forEachLayer(collect, nullptr);

  std::vector<std::string> unmatched;
  unsigned int matched = 0;
  std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &, void *)>
    apply = [&](ml::train::Layer &, nntrainer::RunLayerContext &context, void *) {
      for (auto &w : context.getWeights()) {
        const std::string &name = w->getName();
        auto it = base_weights.find(name);
        if (it != base_weights.end()) {
          nntrainer::Tensor &dst = w->getVariableRef();
          const nntrainer::Tensor &src = *it->second;
          const size_t bytes = src.getMemoryBytes();
          std::memcpy(dst.getData<char>(), src.getData<char>(), bytes);
          ++matched;
        } else if (name.find(":loraA") == std::string::npos &&
                   name.find(":loraB") == std::string::npos) {
          unmatched.push_back(name);
        }
      }
    };
  model->forEachLayer(apply, nullptr);

  NNTR_THROW_IF(!unmatched.empty(), std::runtime_error)
    << "load_weight_lora_q4: " << unmatched.size()
    << " non-LoRA weight(s) had no counterpart in the base checkpoint graph "
       "(first: "
    << unmatched.front()
    << "). The base graph must match the graph the checkpoint was saved from.";
  NNTR_THROW_IF(matched == 0, std::runtime_error)
    << "load_weight_lora_q4: no weights were loaded from " << base_path;

  std::ifstream f(lora_q4_path, std::ios::binary);
  NNTR_THROW_IF(!f.is_open(), std::runtime_error)
    << "Failed to open Q4_0 LoRA file: " << lora_q4_path;

  std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &, void *)>
    fn = [&](ml::train::Layer &l, nntrainer::RunLayerContext &context, void *) {
      if (l.getType() != "fully_connected")
        return;

      for (auto &w : context.getWeights()) {
        const std::string &wname = w->getName();
        if (wname.find(":loraA") == std::string::npos &&
            wname.find(":loraB") == std::string::npos)
          continue;

        uint32_t n = 0;
        f.read(reinterpret_cast<char *>(&n), sizeof(n));
        NNTR_THROW_IF(!f, std::runtime_error)
          << "load_weight_lora_q4: failed reading element count at '"
          << wname << "'";

        nntrainer::Tensor &dst = w->getVariableRef();
        const uint32_t expected =
          static_cast<uint32_t>(dst.getDim().getDataLen());
        NNTR_THROW_IF(n != expected, std::runtime_error)
          << "load_weight_lora_q4: element count mismatch for '" << wname
          << "': file=" << n << " model=" << expected;

        // For Q4_0 tensors, read directly into the tensor buffer using memcpy
        const size_t bytes = dst.getMemoryBytes();
        void *buf = dst.getData<char>();
        NNTR_THROW_IF(!buf, std::runtime_error)
          << "load_weight_lora_q4: tensor buffer not allocated for '" << wname << "'";
        f.read(reinterpret_cast<char *>(buf), bytes);
        NNTR_THROW_IF(!f, std::runtime_error)
          << "load_weight_lora_q4: failed reading Q4_0 data for '" << wname << "'";
      }
    };
  model->forEachLayer(fn, nullptr);

  std::cout << "[load_weight_lora_q4] Loaded Q4_0 LoRA adapters from "
            << lora_q4_path << std::endl;
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
  auto embedding_props = buildEmbeddingLayerProperties(
    "embedding0", NUM_VOCAB, DIM, EMBEDDING_DTYPE, EMBEDDING_SCALE,
    EMBEDDING_FILE_NAME);
  // embedding0 is never a LoRA target: freeze it whenever LoRA is active.
  if (LORA_RANK > 0)
    embedding_props.push_back(withKey("trainable", "false"));
  LayerHandle embedding(createLayer(embedding_type, embedding_props));
  Tensor h = embedding(x);

  // transformer decoder blocks
  for (int i = 0; i < NUM_LAYERS; ++i) {
    h = createTransformerDecoderBlock(i, h);
  }

  // final rms_norm (never a LoRA target: freeze it whenever LoRA is active)
  std::vector<std::string> out_norm_props = {
    withKey("name", "output_norm"), withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("packed", "false")};
  if (LORA_RANK > 0 && !TRAIN_NORMS)
    out_norm_props.push_back(withKey("trainable", "false"));
  LayerHandle out_norm(createLayer("rms_norm", out_norm_props));
  h = out_norm(h);

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

/**
 * @brief Load model weights from a binary nntrainer model file.
 */
void Transformer::load_weight(const std::string &weight_path) {
  if (!is_initialized) {
    throw std::runtime_error(
      "Transformer model is not initialized. Please call "
      "initialize() before load_weight().");
  }

  try {
    model->load(weight_path, formatFromExtension(weight_path));
  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to load model weights: " +
                             std::string(e.what()));
  }
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
      // Each layer determines whether it repacks its own weights or not
      static_cast<nntrainer::LayerNode &>(l).pack(context);
    };
  try {
    model->forEachLayer(fn, nullptr);
    ml_logd("weights repacked successfully");
  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to repack weights: " +
                             std::string(e.what()));
  }
};

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

  // norms are never a LoRA target: freeze them whenever LoRA is active.
  std::vector<std::string> attn_norm_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_attention_norm"),
    withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false")};
  if (LORA_RANK > 0 && !TRAIN_NORMS)
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
  if (LORA_RANK > 0 && !TRAIN_NORMS)
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
  const std::string cache_shape = std::to_string(BATCH_SIZE) +
                                  ":1:" + std::to_string(max_timestep) + ":" +
                                  std::to_string(kv_width);

  // KV caches MUST be created as "input" layers (not plain Tensors). Plain
  // Tensors shrink the graph's input-layer set, which changes the tensor-pool
  // in-place/flatten behavior so that the first transformer layer's input is no
  // longer a synced dependent of the model input placeholder. On ARM that broke
  // USE_EMBEDDING prefill: the embedding reached input0's output but never
  // layer0_conv_norm (all-zero activations → <pad>). The x86 (#else) path
  // always used input layers and worked; this keeps both paths symmetric,
  // differing only in the external dtype (FP16 on ARM, UINT16 elsewhere).
#ifdef ENABLE_FP16
  const char *cache_dtype = "FP16";
#else
  const char *cache_dtype = "UINT16";
#endif

  LayerHandle cache_k_input(createLayer(
    "input", {withKey("name", "cache_k_l" + std::to_string(layer_id)),
              withKey("input_shape", cache_shape),
              withKey("input_dtype", cache_dtype)}));
  LayerHandle cache_v_input(createLayer(
    "input", {withKey("name", "cache_v_l" + std::to_string(layer_id)),
              withKey("input_shape", cache_shape),
              withKey("input_dtype", cache_dtype)}));

  return {cache_k_input(Tensor()), cache_v_input(Tensor())};
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

  // Attention core layer (frozen: GQA has no learnable weights beyond
  // Q/K/V/O, so there's nothing here for LoRA to target)
  LayerHandle mha(createLayer(
    "mha_core",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention"),
     withKey("num_heads", n_heads), withKey("num_heads_kv", n_heads / GQA_SIZE),
     withKey("max_timestep", std::to_string(MAX_SEQ_LEN)),
     withKey("sliding_window", (layer_id + 1) % SLIDING_WINDOW_PATTERN
                                 ? SLIDING_WINDOW
                                 : UINT_MAX),
     withKey("rope_theta", ROPE_THETA),
     withKey("max_position_embeddings", MAX_POSITION_EMBEDDINGS),
     withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
     withKey("is_causal", IS_CAUSAL ? "true" : "false")}));

  Tensor a;
  if (FOR_TRAINING) {
    // No KV-cache placeholders for training: mha_core's internal-cache
    // (3-input) construction is what its training-mode forward/backward
    // (trainForwarding()/calcDerivative() in mha_core.cpp) requires.
    a = mha({q, k, v});
  } else {
    // External KV cache placeholders (per-layer). Their actual storage is
    // owned by the host (KVCacheManager) and bound at runtime via
    // setExternalTensors.
    auto [cache_k, cache_v] = createKVCachePlaceholders(layer_id, n_heads);
    a = mha({q, k, v, cache_k, cache_v});
  }

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
