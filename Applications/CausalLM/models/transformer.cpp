// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   transformer.cpp
 * @date   10 July 2025
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This file defines Transformer's basic actions
 */

#include <array>
#include <filesystem>
#include <fstream>
#include <mutex>

#include <app_context.h>
#include <engine.h>
#include <model.h>

#include <llm_util.hpp>
#include <tokenizers_cpp.h>
#include <transformer.h>

#include <dense_ffn_layer.h>
#include <embedding_layer.h>
#include <htp_wh_layout.h>
#include <lfm2_moe_layer.h>
#include <lfm2_moe_pool_layer.h>
#include <mha_core.h>
#include <neuralnet.h>
#include <qs4cx_tensor.h>
#include <rms_norm.h>
#include <swiglu.h>
#include <tie_word_embedding.h>

namespace causallm {

namespace {
/** @brief [#260] The two MoE layer classes the load path registers alike:
 *  #4415's lfm2_moe (Gemma) and the one-PD pool's lfm2_moe_pool (LFM2). */
bool isMoeLayer(const ml::train::Layer &l) {
  return l.getType() == Lfm2MoELayer::type ||
         l.getType() == Lfm2MoePoolLayer::type;
}
} // namespace

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
  ATTENTION_ENGINE = nntr_cfg.value("attention_engine", std::string());
  ATTENTION_KV_DTYPE = nntr_cfg.value("attention_kv_dtype", std::string());
  EMBEDDING_FILE_NAME = nntr_cfg.value("embedding_file_name", std::string());
  PLE_FILE_NAME = nntr_cfg.value("ple_file_name", std::string());
  FC_WH_FILE_NAME = nntr_cfg.value("fc_wh_file_name", std::string());
  FC_WH_FORMAT = nntr_cfg.value("fc_wh_format", std::string());
  ACT16_ENC_FILE = nntr_cfg.value("act16_enc_file", std::string());

  if (cfg.contains("is_causal")) {
    IS_CAUSAL = cfg["is_causal"].get<bool>();
  } else if (cfg.contains("use_bidirectional_attention") &&
             !cfg["use_bidirectional_attention"].is_null()) {
    // Gemma-4 writes a string here: "all" is bidirectional, "vision" means
    // only the vision tokens are (Gemma4TextAttention: is_causal = value !=
    // "all"), so text stays causal. Older configs write a bool.
    const auto &bidir = cfg["use_bidirectional_attention"];
    IS_CAUSAL = bidir.is_string() ? bidir.get<std::string>() != "all"
                                  : !bidir.get<bool>();
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

  return;
};

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
  LayerHandle embedding(createLayer(
    embedding_type,
    buildEmbeddingLayerProperties("embedding0", NUM_VOCAB, DIM, EMBEDDING_DTYPE,
                                  EMBEDDING_SCALE, EMBEDDING_FILE_NAME)));
  Tensor h = embedding(x);

  // transformer decoder blocks
  for (int i = 0; i < NUM_LAYERS; ++i) {
    h = createTransformerDecoderBlock(i, h);
  }

  // final rms_norm
  LayerHandle out_norm(
    createLayer("rms_norm", {withKey("name", "output_norm"),
                             withKey("epsilon", std::to_string(NORM_EPS)),
                             withKey("packed", "false")}));
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

  WEIGHT_DIR = std::filesystem::path(weight_path).parent_path().string();
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
  // [doc 50 section 3.2] FC weights an accelerator will hold are collected
  // here and registered AFTER the walk, once every MoE arena chunk is
  // mapped. The DSP's 32-bit address space maps each 256 MiB chunk at a
  // 256 MiB boundary (doc 46 section 41); a heap allocation between two
  // chunk mappings -- which is where a registered FC weight lives --
  // strands the space from the heap's end to the next boundary. Measured:
  // registered in graph order (conv in_proj between the MoE layers) the
  // arena stopped at 3584 MiB, 14 chunks, against the 3840 it reaches on
  // its own.
  struct PendingFc {
    nntrainer::ComputeOps *ops;
    void *data;
    const float *scale; ///< QS4CX per-channel scales; null for Q4_0
    unsigned int K, N;
  };
  std::vector<PendingFc> fc_pending;
  struct PendingDense {
    nntrainer::ComputeOps *ops;
    void *up, *gate, *down;
    const float *up_s, *gate_s, *down_s; /**< QS4CX scales; null for Q4_0 */
    unsigned int K, I, N;
  };
  std::vector<PendingDense> dense_pending;
  struct PendingConv {
    nntrainer::ComputeOps *ops;
    void *in_proj, *out_proj;
    const float *conv_w;
    unsigned int K, C, N;
  };
  std::vector<PendingConv> conv_pending;
  // [#132 Part B E5g] the MoE warm-up call, kept until every MoE weight is
  // mapped: it grows the session's DSP heap by the M=512 scratch, and a heap
  // growth between two arena chunk mappings strands the space up to the next
  // chunk boundary (the note above). In the walk it landed between the first
  // MoE layer's chunks and the rest, and the two-session sittings saw S1's
  // last chunk refused (AEE_EMMAP) on some boots.
  struct PendingMoeWarm {
    nntrainer::ComputeOps *ops = nullptr;
    std::vector<void *> gu_data, dn_data;
    std::vector<float *> gu_scale, dn_scale;
    unsigned int K = 0, inter = 0, N_out = 0;
    unsigned int w_bits = 0u; /**< 0 QS4CX, 4 QS4CX_WH, 2 QS2CX_WH */
  } moe_warm;

  // The tied lm_head, prepared after the walk: on its accelerator it is
  // placed in the arena, which must come after every other weight (the
  // chunk-boundary note above), else its blocked CPU twin is built. Both
  // at load, with every weight loaded, rather than on the first lm_head
  // call inside the first prefill. The context is the node's own and
  // outlives the walk.
  struct PendingTie {
    TieWordEmbedding *layer = nullptr;
    nntrainer::RunLayerContext *context = nullptr;
  } tie_pending;
  // The attention layers' quantized KV caches, registered after the
  // lm_head for the same reason it waits: they are DSP heap.
  std::vector<std::pair<MHACoreLayer *, nntrainer::RunLayerContext *>>
    mha_pending;
  // a16: each QS4CX weight's input encoding, by weight name, handed to its
  // layer's ops in the walk -- before any of the registrations below
  std::map<std::string, std::array<float, 2>> act16;
  if (!ACT16_ENC_FILE.empty()) {
    const std::filesystem::path p(ACT16_ENC_FILE);
    std::ifstream f(p.is_absolute() ? p
                                    : std::filesystem::path(WEIGHT_DIR) / p);
    NNTR_THROW_IF(!f, std::runtime_error)
      << "act16_enc_file " << ACT16_ENC_FILE << " cannot be read";
    // named: items() iterates the json it is called on
    const nlohmann::json enc = nlohmann::json::parse(f);
    for (const auto &[name, e] : enc.items())
      act16[name] = {e.at(0).get<float>(), e.at(1).get<float>()};
  }
  unsigned int act16_bound = 0;

  std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &, void *)>
    fn = [&fc_pending, &dense_pending, &conv_pending, &moe_warm, &tie_pending,
          &mha_pending, &act16, &act16_bound](
           ml::train::Layer &l, nntrainer::RunLayerContext &context, void *) {
      if (!act16.empty() && context.getComputeOps()) {
        for (auto *w : context.getWeights()) {
          auto &t = w->getVariableRef();
          auto e = act16.find(w->getName());
          if (e == act16.end() || t.isVirtual() ||
              t.getDataType() != ml::train::TensorDim::DataType::QS4CX)
            continue;
          context.getComputeOps()->set_act16_enc(t.getData<char>(),
                                                 e->second[0], e->second[1]);
          ++act16_bound;
        }
      }
      // forEachLayer hands out LayerNodes.
      if (l.getType() == MHACoreLayer::type) {
        if (auto *m = dynamic_cast<MHACoreLayer *>(
              static_cast<nntrainer::LayerNode &>(l).getLayer()))
          mha_pending.push_back({m, &context});
        return;
      }
      if (l.getType() == TieWordEmbedding::type) {
        auto *tw = dynamic_cast<TieWordEmbedding *>(
          static_cast<nntrainer::LayerNode &>(l).getLayer());
        if (tw) {
          tie_pending.layer = tw;
          tie_pending.context = &context;
        }
        return;
      }
      // repack FC and MoE FFN layers -- both can hold QS4CX weights.
      // FloatTensor::dotQs4cx only needs a weight packed when it falls
      // through to the CPU KleidiAI path (getPackedData()); the HTP path
      // (gemm_qs4cx_accel_fp32) reads getData()/getScale() directly and
      // never touches packed_data, so packing an HTP-dispatched expert's
      // weight here is a harmless one-time no-op cost, not a correctness
      // requirement -- but a CPU-dispatched MoE layer (any layer_id not in
      // moe_htp_layers) throws "pack before run model" on its first token
      // without this, because lfm2_moe was missing from this filter.
      // A dense_ffn layer's three Q4_0 weights are registered as one
      // fused set (doc 51), after the walk like the FCs below -- not as
      // three FC weights, which the loop below would otherwise do.
      if (l.getType() == "dense_ffn") {
        // By name: with its norms folded in (in_norm / out_norm) the layer
        // also holds two FP32 gammas, and gate_first swaps the first two.
        // Matched on the weight order of a size()==3 test, the folded
        // layer skipped this and registered inside the first prefill: a
        // whole scalar Q4_0 -> QS4CX conversion per layer under the
        // prefill timer (doc 57 section 9.10).
        nntrainer::Tensor *up_p = nullptr, *gate_p = nullptr, *down_p = nullptr;
        for (auto *w : context.getWeights()) {
          const std::string &name = w->getName();
          auto ends_with = [&name](const char *tail) {
            const size_t n = std::strlen(tail);
            return name.size() >= n &&
                   name.compare(name.size() - n, n, tail) == 0;
          };
          if (ends_with(":up"))
            up_p = &w->getVariableRef();
          else if (ends_with(":gate"))
            gate_p = &w->getVariableRef();
          else if (ends_with(":down"))
            down_p = &w->getVariableRef();
        }
        // A CPU-engine dense_ffn reads QS4CX through the KleidiAI pack, so
        // pack as the FC branch below does (a CPU layer has ComputeOps too,
        // so the engine cannot be told from them here).
        for (auto *p : {up_p, gate_p, down_p})
          if (p && p->getDataType() == ml::train::TensorDim::DataType::QS4CX &&
              !p->isVirtual())
            p->pack();
        if (up_p && gate_p && down_p && context.getComputeOps()) {
          auto &up = *up_p;
          auto &gate = *gate_p;
          auto &down = *down_p;
          const auto wtype = up.getDataType();
          const bool qs4cx = wtype == ml::train::TensorDim::DataType::QS4CX;
          if ((wtype == ml::train::TensorDim::DataType::Q4_0 || qs4cx) &&
              gate.getDataType() == wtype && down.getDataType() == wtype) {
            dense_pending.push_back({context.getComputeOps(),
                                     up.getData<char>(), gate.getData<char>(),
                                     down.getData<char>(),
                                     qs4cx ? up.getScale<float>() : nullptr,
                                     qs4cx ? gate.getScale<float>() : nullptr,
                                     qs4cx ? down.getScale<float>() : nullptr,
                                     static_cast<unsigned int>(up.height()),
                                     static_cast<unsigned int>(up.width()),
                                     static_cast<unsigned int>(down.width())});
          }
        }
        return;
      }
      // A conv_block layer's in_proj and out_proj, likewise (doc 51
      // section 2): the conv weight is FP32 and rides with each call.
      if (l.getType() == "conv_block") {
        auto weights = context.getWeights();
        if (weights.size() == 3 && context.getComputeOps()) {
          auto &in_proj = weights[0]->getVariableRef();
          auto &conv = weights[1]->getVariableRef();
          auto &out_proj = weights[2]->getVariableRef();
          if (in_proj.getDataType() == ml::train::TensorDim::DataType::Q4_0 &&
              out_proj.getDataType() == ml::train::TensorDim::DataType::Q4_0) {
            conv_pending.push_back(
              {context.getComputeOps(), in_proj.getData<char>(),
               out_proj.getData<char>(), conv.getData<float>(),
               static_cast<unsigned int>(in_proj.height()),
               static_cast<unsigned int>(conv.width()),
               static_cast<unsigned int>(out_proj.width())});
          }
        }
        return;
      }
      // residual_add: its proj weight, when it holds the o-proj (doc 57
      // section 9.32); its FP32 gamma / scalar are skipped below.
      if (l.getType() != "fully_connected" &&
          l.getType() != "shared_fully_connected" &&
          l.getType() != "qkv_layer" && !isMoeLayer(l) &&
          l.getType() != "residual_add")
        return;

      // An accelerator-dispatched MoE layer registers its expert weights
      // here, at load, instead of on the first token: the layer's ComputeOps
      // is only non-null when nntr_config.json routed it to an engine, and
      // a backend that has nothing to register says so and is skipped. The
      // expert weight tensors are [K, N] as the kernel sees them; the router
      // gate and expert bias are FP32 and never reach this branch.
      auto *ops = isMoeLayer(l) ? context.getComputeOps() : nullptr;
      // [doc 50] A fully_connected layer under engine=htp registers its
      // Q4_0 weight here for the same reason: gemm_q4_0_accel_fp32's first
      // call converts and registers it, and that first call is in the
      // first prefill. A CPU-engine layer's ops answer false; nothing else
      // changes for it. The weight is [K, N] as dot() reads it.
      auto *fc_ops = isMoeLayer(l) ? nullptr : context.getComputeOps();

      auto weights = context.getWeights();
      std::vector<void *> gu_data, dn_data;
      std::vector<float *> gu_scale, dn_scale;
      unsigned int gu_h = 0, gu_w = 0, dn_h = 0, dn_w = 0;
      unsigned int w_bits = 0u;
      for (auto &w : weights) {
        auto &t = w->getVariableRef();
        const auto dtype = t.getDataType();
        if (dtype == ml::train::TensorDim::DataType::QS4CX && !t.isVirtual()) {
          t.pack();
        }
        // A QS4CX FC weight (quantized offline, no Q4_0 detour) is
        // registered here too, through the key gemm_qs4cx_accel_fp32 looks
        // up: the weight's own data pointer and scales.
        if (fc_ops && (dtype == ml::train::TensorDim::DataType::Q4_0 ||
                       (dtype == ml::train::TensorDim::DataType::QS4CX &&
                        !t.isVirtual()))) {
          const bool qs4cx = dtype == ml::train::TensorDim::DataType::QS4CX;
          fc_pending.push_back({fc_ops, t.getData<char>(),
                                qs4cx ? t.getScale<float>() : nullptr,
                                static_cast<unsigned int>(t.height()),
                                static_cast<unsigned int>(t.width())});
        }
        if (ops && (dtype == ml::train::TensorDim::DataType::QS4CX ||
                    dtype == ml::train::TensorDim::DataType::QS4CX_WH ||
                    dtype == ml::train::TensorDim::DataType::QS2CX_WH)) {
          const auto h = static_cast<unsigned int>(t.height());
          const auto wd = static_cast<unsigned int>(t.width());
          w_bits = dtype == ml::train::TensorDim::DataType::QS4CX_WH   ? 4u
                   : dtype == ml::train::TensorDim::DataType::QS2CX_WH ? 2u
                                                                       : 0u;
          // [doc 52] A virtual expert has no bytes here: preloadExperts
          // below reads it from the model file into the accelerator's slot
          // pool, keyed by the tensor itself, and the warm-up passes that
          // key with a null scale the way the layer's forward does.
          void *key =
            t.isVirtual() ? static_cast<void *>(&t) : t.getData<char>();
          float *scale = t.isVirtual() ? nullptr : t.getScale<float>();
          if (!t.isVirtual())
            ops->register_qs4cx_weight(key, scale, h, wd, w_bits);
          // The expert weights come in two shapes: gate_up is [K, 2*inter]
          // and down [inter, N_out]. Sorted here for the warm-up below by
          // the identity that tells them apart, gate_up.width == 2 *
          // down.height, checked once both are seen.
          if (gu_h == 0 || (h == gu_h && wd == gu_w)) {
            if (gu_h == 0) {
              gu_h = h;
              gu_w = wd;
            }
            gu_data.push_back(key);
            gu_scale.push_back(scale);
          } else {
            dn_h = h;
            dn_w = wd;
            dn_data.push_back(key);
            dn_scale.push_back(scale);
          }
        }
      }
      // [doc 52] Virtual experts into the shared pool now, in layer order
      // until it is full, so the first prefill maps no arena chunk. A
      // layer whose experts did not all fit cannot host the warm-up call.
      bool moe_all_resident = true;
      if (ops && isMoeLayer(l)) {
        auto *layer = static_cast<nntrainer::LayerNode &>(l).getLayer();
        if (auto *moe = dynamic_cast<Lfm2MoELayer *>(layer))
          moe_all_resident = moe->preloadExperts(context);
        else if (auto *pool = dynamic_cast<Lfm2MoePoolLayer *>(layer))
          moe_all_resident = pool->preloadExperts(context);
      }
      // [doc 47 section 20.1, lever 6] One warm-up call through the layer
      // kernel at load, so the first prefill does not pay the first call's
      // costs -- the session scratch's growth to the prefill size, the
      // first touch of every page of it and of the FastRPC buffers, the
      // DSP's own first-call setup: read as +10 ms on the first MoE layer
      // of the first prefill (doc 47 section 15). One layer's call is
      // enough, since all of that is per session, not per layer. Zero
      // input, output discarded. Skipped, harmlessly, when the shapes do
      // not read as an expert pair.
      if (ops && moe_warm.ops == nullptr && moe_all_resident && isMoeLayer(l) &&
          ops->supports_gemm_qs4cx_moe_layer_fp32() && !gu_data.empty() &&
          gu_data.size() == dn_data.size() && gu_w == 2 * dn_h &&
          gu_h == dn_w) {
        moe_warm.ops = ops;
        moe_warm.gu_data = gu_data;
        moe_warm.gu_scale = gu_scale;
        moe_warm.dn_data = dn_data;
        moe_warm.dn_scale = dn_scale;
        moe_warm.K = gu_h;
        moe_warm.inter = dn_h;
        moe_warm.N_out = dn_w;
        moe_warm.w_bits = w_bits;
      }
    };
  try {
    model->forEachLayer(fn, nullptr);
    if (!act16.empty())
      std::cout << "act16: " << act16_bound << " QS4CX weights at a static "
                << "u16 input encoding (" << ACT16_ENC_FILE << ")" << std::endl;
    // [#225] The FC WH sidecar, handed to every backend that registers a
    // pending FC before it registers any. A CPU-engine layer's ops are
    // pending too and answer false, so a model whose keys route no FC to
    // an accelerator (the CPU-only and no-keys configs) never opens it.
    if (!FC_WH_FILE_NAME.empty()) {
      NNTR_THROW_IF(FC_WH_FORMAT != nntrainer::FCWH_FORMAT, std::runtime_error)
        << "fc_wh_format \"" << FC_WH_FORMAT << "\" in nntr_config.json, this "
        << "build reads \"" << nntrainer::FCWH_FORMAT
        << "\": re-run nntr_quantize_stream --fc_wh_sidecar";
      const std::filesystem::path p(FC_WH_FILE_NAME);
      const std::string path =
        p.is_absolute() ? p.string()
                        : (std::filesystem::path(WEIGHT_DIR) / p).string();
      std::set<nntrainer::ComputeOps *> backends;
      for (const auto &f : fc_pending)
        backends.insert(f.ops);
      for (const auto &d : dense_pending)
        backends.insert(d.ops);
      for (const auto &c : conv_pending)
        backends.insert(c.ops);
#ifdef ENABLE_HEXKL
      // [#234 P4] the one-PD decode list binds its FC / DENSE_FFN ops to
      // the sidecar's images too, so a model that keys no FC (Gemma 4)
      // opens it for that bind
      if (nntrainer::get_htp_ops()->has_decode_graph_q4_0())
        backends.insert(nntrainer::get_htp_ops());
#endif
      for (auto *b : backends)
        b->set_fc_wh_file(path.c_str());
    }
    // The MoE warm-up (doc 47 section 20.1, lever 6): one call through the
    // layer kernel at load, now after the walk (see PendingMoeWarm).
    if (moe_warm.ops != nullptr) {
      // M as wide as a prefill can be: the kernel's scratch keeps its act
      // and out copies (M x K, M x N_out f32) and the staging pools their
      // size classes, so a 512-row warm-up left the first 1024-row prefill
      // call to grow and touch them, 2.1 ms a call on average over 31
      // (doc 57 section 9.26). The expert slots' bound is the same either way.
      const unsigned int M = std::max(512u, INIT_SEQ_LEN), K = moe_warm.K,
                         N_out = moe_warm.N_out;
      const unsigned int E = static_cast<unsigned int>(moe_warm.gu_data.size());
      const unsigned int per = 64; // one 64-row block per expert
      std::vector<unsigned int> row_index, row_count(E, per);
      std::vector<float> row_weight;
      for (unsigned int e = 0; e < E; ++e) {
        for (unsigned int r = 0; r < per; ++r) {
          row_index.push_back((e * per + r) % M);
          row_weight.push_back(0.0f);
        }
      }
      std::vector<float> act(static_cast<size_t>(M) * K, 0.0f);
      std::vector<float> out(static_cast<size_t>(M) * N_out, 0.0f);
      moe_warm.ops->gemm_qs4cx_moe_layer_fp32(
        moe_warm.gu_data, moe_warm.gu_scale, moe_warm.dn_data,
        moe_warm.dn_scale, row_index, row_count, row_weight, act.data(),
        out.data(), M, K, moe_warm.inter, N_out, moe_warm.w_bits);
      ml_logd("MoE HTP kernel warmed up at load (M=%u, %u experts)", M, E);
    }
    // The deferred FC registrations (see fc_pending above). A CPU-engine
    // layer's ops answer false and cost nothing. One warm-up call per
    // session after the first accelerated one, like the MoE one: the first
    // FC call otherwise grows the staging buffers to the prefill size and
    // touches every page of them inside the first prefill. In graph order
    // that first weight is a conv in_proj (the widest N) whenever those are
    // routed, so one call at M=512 covers every later shape's buffers.
    // A QS4CX weight has no register-only entry that slices it the way
    // the FC call does (register_qs4cx_weight is the one-handle expert
    // form), so its call registers it: 64 rows, one row block, for every
    // weight but the warm-up one.
    bool fc_warmed = false;
    for (const auto &p : fc_pending) {
      const bool qs4cx = p.scale != nullptr;
      if (qs4cx ? !p.ops->supports_gemm_qs4cx_accel_fp32()
                : !p.ops->register_q4_0_weight(p.data, p.K, p.N))
        continue;
      if (fc_warmed && !qs4cx)
        continue;
      if (!qs4cx && !p.ops->supports_gemm_q4_0_accel_fp32())
        continue;
      const unsigned int M = fc_warmed ? 64 : 512;
      fc_warmed = true;
      std::vector<float> act(static_cast<size_t>(M) * p.K, 0.0f);
      std::vector<float> out(static_cast<size_t>(M) * p.N, 0.0f);
      if (qs4cx)
        p.ops->gemm_qs4cx_accel_fp32(p.data, const_cast<float *>(p.scale),
                                     act.data(), out.data(), M, p.N, p.K);
      else
        p.ops->gemm_q4_0_accel_fp32(p.data, act.data(), out.data(), M, p.N,
                                    p.K);
      ml_logd("FC HTP kernel warmed up at load (M=%u, K=%u, N=%u)", M, p.K,
              p.N);
    }
    // Same for the dense FFNs: the registration converts and packs three
    // weights into I / w expert pairs, and the one warm-up call grows the
    // MoE layer kernel's scratch to this shape's row count.
    bool dense_warmed = false;
    for (const auto &p : dense_pending) {
      if (!p.ops->register_q4_0_dense_ffn(p.up, p.gate, p.down, p.K, p.I, p.N,
                                          p.up_s, p.gate_s, p.down_s))
        continue;
      if (dense_warmed || !p.ops->supports_gemm_q4_0_dense_ffn_fp32())
        continue;
      dense_warmed = true;
      const unsigned int M = 512;
      std::vector<float> act(static_cast<size_t>(M) * p.K, 0.0f);
      std::vector<float> out(static_cast<size_t>(M) * p.N, 0.0f);
      p.ops->gemm_q4_0_dense_ffn_fp32(
        p.up, p.gate, p.down, act.data(), out.data(), M, p.K, p.I, p.N, false,
        nullptr, nullptr, 0.0f, p.up_s, p.gate_s, p.down_s);
      ml_logd("dense FFN HTP kernel warmed up at load (M=%u, K=%u, I=%u, N=%u)",
              M, p.K, p.I, p.N);
    }
    // And the conv blocks: in_proj's three slices and out_proj registered,
    // one warm-up call so the first prefill finds the kernel's scratch and
    // the staging buffers already grown to this shape.
    bool conv_warmed = false;
    for (const auto &p : conv_pending) {
      if (!p.ops->register_q4_0_conv_block(p.in_proj, p.out_proj, p.K, p.C,
                                           p.N))
        continue;
      if (conv_warmed || !p.ops->supports_gemm_q4_0_conv_block_fp32())
        continue;
      conv_warmed = true;
      const unsigned int M = 512;
      std::vector<float> act(static_cast<size_t>(M) * p.K, 0.0f);
      std::vector<float> out(static_cast<size_t>(M) * p.N, 0.0f);
      std::vector<float> state(static_cast<size_t>(2) * p.C, 0.0f);
      p.ops->gemm_q4_0_conv_block_fp32(p.in_proj, p.conv_w, p.out_proj,
                                       act.data(), out.data(), state.data(), M,
                                       p.K, p.C, p.N);
      ml_logd(
        "conv block HTP kernel warmed up at load (M=%u, K=%u, C=%u, N=%u)", M,
        p.K, p.C, p.N);
    }
    if (tie_pending.layer != nullptr &&
        !tie_pending.layer->placeLmheadOnAccelerator(*tie_pending.context))
      tie_pending.layer->prepareLmhead(*tie_pending.context);
    for (auto &p : mha_pending)
      p.first->registerQuantizedCache(*p.second);
    ml_logd("QS4CX weights repacked successfully");
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

  LayerHandle attn_norm(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention_norm"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));
  Tensor normed = attn_norm(input);

  Tensor att_out = createAttention(layer_id, INIT_SEQ_LEN, NUM_HEADS, HEAD_DIM,
                                   normed, normed, normed);

  LayerHandle decoder_add(createLayer(
    RESIDUAL_ADD_TYPE,
    {withKey("name", "layer" + std::to_string(layer_id) + "_decoder_add")}));
  Tensor residual = decoder_add({input, att_out});

  LayerHandle ffn_norm(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_norm"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));
  Tensor ffn_normed = ffn_norm(residual);

  Tensor ffn_out = createMlp(layer_id, DIM, INTERMEDIATE_SIZE, ffn_normed);

  LayerHandle decoder_output(createLayer(
    RESIDUAL_ADD_TYPE,
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

LayerHandle Transformer::createAttentionCore(std::vector<std::string> props) {
  // "attention_engine" puts the layer on that engine's ComputeOps (e.g.
  // "htp": attention on the DSP; the engine falls back to the CPU path per
  // call when it cannot take a shape); "attention_kv_dtype" adds the
  // quantized cache mirror on top of it.
  if (!ATTENTION_ENGINE.empty()) {
    props.emplace_back(withKey("engine", ATTENTION_ENGINE));
  }
  if (!ATTENTION_KV_DTYPE.empty()) {
    props.emplace_back(withKey("kv_cache_quant", ATTENTION_KV_DTYPE));
  }
  return createLayer("mha_core", props);
}

/**
 * @brief Create the default attention subgraph.
 */
Tensor Transformer::createAttention(const int layer_id, int seq_len,
                                    int n_heads, int head_dim, Tensor query,
                                    Tensor key, Tensor value) {

  // Q layer
  LayerHandle wq(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_wq"),
     withKey("unit", head_dim * n_heads), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones")}));
  Tensor q = wq(query);

  // K layer
  LayerHandle wk(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_wk"),
     withKey("unit", head_dim * n_heads / GQA_SIZE),
     withKey("disable_bias", "true"), withKey("weight_initializer", "ones")}));
  Tensor k = wk(key);

  // V layer
  LayerHandle wv(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_wv"),
     withKey("unit", head_dim * n_heads / GQA_SIZE),
     withKey("disable_bias", "true"), withKey("weight_initializer", "ones")}));
  Tensor v = wv(value);

  // External KV cache placeholders (per-layer). Their actual storage is owned
  // by the host (KVCacheManager) and bound at runtime via setExternalTensors.
  auto [cache_k, cache_v] = createKVCachePlaceholders(layer_id, n_heads);

  // Attention core layer
  LayerHandle mha(createAttentionCore(
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
  Tensor a = mha({q, k, v, cache_k, cache_v});

  // O layer
  LayerHandle wo(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention_out"),
     withKey("unit", DIM), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones")}));
  return wo(a);
}

/**
 * @brief Create the default feed-forward subgraph.
 */
Tensor Transformer::createMlp(const int layer_id, int dim, int hidden_dim,
                              Tensor input) {
  // FFN_ENGINE (doc 50): "htp" moves only the three FCs' prefill matmuls;
  // decode (M == 1) stays on the CPU kernel, as does the SwiGLU between.
  const std::string eng =
    (FFN_HTP_LAYERS.empty() || FFN_HTP_LAYERS.count(layer_id)) ? FFN_ENGINE
                                                               : "cpu";
  if (eng != "cpu") {
    // One layer for the block, so the accelerator takes up, gate, SwiGLU
    // and down in one call (doc 51). Same three weights in the file's
    // order, so the model file loads unchanged.
    LayerHandle ffn(
      createLayer("dense_ffn",
                  {withKey("name", "layer" + std::to_string(layer_id) + "_ffn"),
                   withKey("unit", hidden_dim), withKey("engine", eng)}));
    return ffn(input);
  }

  LayerHandle ffn_up(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_up"),
     withKey("unit", hidden_dim), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones"), withKey("engine", eng)}));
  Tensor up = ffn_up(input);

  LayerHandle ffn_gate(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_gate"),
     withKey("unit", hidden_dim), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones"), withKey("engine", eng)}));
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

  LayerHandle ffn_down(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
     withKey("unit", dim), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones"), withKey("engine", eng)}));
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
      nntrainer::createLayer<causallm::DenseFfnLayer>);
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
