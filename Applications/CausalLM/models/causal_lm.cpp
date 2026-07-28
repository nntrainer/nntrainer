// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 * Copyright (C) 2025 Seungback Hong <sb92.hong@samsung.com>
 * Copyright (C) 2025 Hyeonseok Lee <hs89.lee@samsung.com>
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   causal_lm.cpp
 * @date   10 July 2025
 * @see    https://github.com/nntrainer/nntrainer
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

#include <algorithm>
#include <app_context.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <engine.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <common.h>
#include <compute_ops.h>
#include <layer_context.h>
#include <lm_head.h>
#include <mha_core.h>
#include <neuralnet.h>
#include <nntrainer_error.h>
#include <residency_policy.h>
#include <tensor.h>

#include <causal_lm.h>
#include <llm_util.hpp>
#include <utf8_stream_util.h>

#include "api/streamer.h"

namespace causallm {

namespace {

/**
 * @brief Wrap an external host buffer as a Tensor of @p dim.
 *
 * Byte-for-byte the same dtype dispatch as neuralnet.cpp's file-local
 * mapExternalTensor() (which is in an anonymous namespace and therefore not
 * reachable from here). Kept in sync deliberately: incrementalInference()
 * below must behave identically to the base float* overload for every input
 * that is NOT a KV-cache buffer.
 */
nntrainer::Tensor mapExternalInput(float *buf,
                                   const nntrainer::TensorDim &dim) {
  const unsigned int bytes = static_cast<unsigned int>(
    static_cast<size_t>(dim.getDataLen()) * dim.getDataTypeSize());

  switch (dim.getDataType()) {
  case nntrainer::TensorDim::DataType::FP16:
  case nntrainer::TensorDim::DataType::UINT16:
  case nntrainer::TensorDim::DataType::QINT16:
    return nntrainer::Tensor::Map<uint16_t>(reinterpret_cast<uint16_t *>(buf),
                                            bytes, dim, 0);
  case nntrainer::TensorDim::DataType::UINT8:
  case nntrainer::TensorDim::DataType::UINT4:
  case nntrainer::TensorDim::DataType::QINT8:
  case nntrainer::TensorDim::DataType::QINT4:
  case nntrainer::TensorDim::DataType::Q4_K:
  case nntrainer::TensorDim::DataType::Q6_K:
  case nntrainer::TensorDim::DataType::Q4_0:
    return nntrainer::Tensor::Map<uint8_t>(reinterpret_cast<uint8_t *>(buf),
                                           bytes, dim, 0);
  case nntrainer::TensorDim::DataType::UINT32:
  case nntrainer::TensorDim::DataType::BCQ:
    return nntrainer::Tensor::Map<uint32_t>(reinterpret_cast<uint32_t *>(buf),
                                            bytes, dim, 0);
  case nntrainer::TensorDim::DataType::FP32:
  case nntrainer::TensorDim::DataType::NONE:
  default:
    return nntrainer::Tensor::Map<float>(buf, bytes, dim, 0);
  }
}

} // namespace

CausalLM::CausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
  Transformer(cfg, generation_cfg, nntr_cfg, ModelType::CAUSALLM) {
  // Declare CausalLM's static-residency boundaries. Core ships the MECHANISM
  // (ResidencyPolicy::global(), read by manager.cpp's engine_neutral test and
  // tensor_pool.cpp's planner build) but deliberately carries no app-specific
  // layer names; the POLICY is the application's to declare. Nothing populated
  // it here, so `isEngineNeutral()` answered false for every type and the
  // mechanism was dead code.
  //
  // `mha_core` is CPU-registered but binds and consumes Q/K/V on the GPU plane
  // (it takes the cl_mem handles directly and bridges its host stages through
  // clmem_lower_cl / clmem_raise_cl). Undeclared, it counts as a CPU consumer,
  // so `all_consumers_gpu` is false for every wq/wk/wv output and the planner
  // downgrades the whole attention neighbourhood GPU_CLMEM -> SVM. Observable
  // proof the declaration is what arms the path: without it NNTR_CLMEM_MHA_OFF
  // (which nulls exactly those handles) cannot change the output at all,
  // because they are already null.
  //
  // NOT declared here, deliberately: the input-boundary RAISE
  // ("embedding0:out0") and output-boundary LOWER ("output_norm:out0") that the
  // reference tree also sets. Both are a REGRESSION on this tree -- measured
  // 2026-07-28, gemma4 goes from the golden "**Seoul**" to <pad> spam -- so the
  // raise/lower implementations they feed are not fully on the ladder yet. They
  // are still reachable for A/B via NNTR_CLMEM_RAISE / NNTR_CLMEM_LOWER.
  {
    auto &rp = nntrainer::ResidencyPolicy::global();
    if (rp.engine_neutral_types.empty())
      rp.engine_neutral_types = {"mha_core"};
  }
  setupParameters(cfg, generation_cfg, nntr_cfg);
}

void CausalLM::prepareForRun() {
  stop_requested_.store(false, std::memory_order_release);
  stop_prepared_for_run_.store(true, std::memory_order_release);
}

void CausalLM::prepareStopRequestForRun() {
  if (!stop_prepared_for_run_.exchange(false, std::memory_order_acq_rel)) {
    stop_requested_.store(false, std::memory_order_release);
  }
}

void CausalLM::setLogitsProcessor(LogitsProcessor *processor) {
  logits_processor = processor;
}

void CausalLM::resetLogitsProcessor() {
  if (logits_processor != nullptr)
    logits_processor->reset();
}

void CausalLM::setupParameters(json &cfg, json &generation_cfg,
                               json &nntr_cfg) {
  // Initialize output list
  for (unsigned int i = 0; i < BATCH_SIZE; ++i)
    output_list.push_back("");

  // allocate memory for the internal buffer
  ids_history = (unsigned int *)malloc(static_cast<size_t>(BATCH_SIZE) *
                                       MAX_SEQ_LEN * sizeof(unsigned int));

  BAD_WORD_IDS = nntr_cfg["bad_word_ids"].get<std::vector<unsigned int>>();
  NUM_BADWORDS = BAD_WORD_IDS.size();

  LMHEAD_DTYPE = nntr_cfg.contains("lmhead_dtype")
                   ? nntr_cfg["lmhead_dtype"]
                   : nntr_cfg["embedding_dtype"];

  SKIP_PREFILL = nntr_cfg.contains("skip_prefill")
                   ? nntr_cfg["skip_prefill"].get<bool>()
                   : false;

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

  if (generation_cfg["eos_token_id"].is_array()) {
    EOS_TOKEN_ID =
      generation_cfg["eos_token_id"].empty()
        ? cfg["eos_token_id"].get<std::vector<unsigned int>>()
        : generation_cfg["eos_token_id"].get<std::vector<unsigned int>>();
  } else {
    EOS_TOKEN_ID.clear();
    EOS_TOKEN_ID.push_back(generation_cfg["eos_token_id"].get<unsigned int>());
  }
  BOS_TOKEN_ID = generation_cfg["bos_token_id"].empty()
                   ? cfg["bos_token_id"].get<unsigned int>()
                   : generation_cfg["bos_token_id"].get<unsigned int>();
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
}

void CausalLM::allocateAndBindKVCache() {
  if (!kv_cache.isAllocated()) {
    // dtype matches mha_core's cache placeholders so external cache storage
    // is interpreted consistently across platforms.
#ifdef ENABLE_FP16
    const auto cache_dtype = ml::train::TensorDim::DataType::FP16;
#else
    const auto cache_dtype = ml::train::TensorDim::DataType::UINT16;
#endif

    const unsigned int max_timestep = static_cast<unsigned int>(MAX_SEQ_LEN);

    kv_cache.allocate(static_cast<unsigned int>(NUM_LAYERS), BATCH_SIZE,
                      max_timestep,
                      static_cast<unsigned int>(NUM_KEY_VALUE_HEADS),
                      static_cast<unsigned int>(HEAD_DIM), cache_dtype);
    kv_cache_bound = false;
  }

  if (kv_cache_bound)
    return;

  // Bind each (layer, K|V) buffer into the corresponding input layer
  // declared by Transformer::createKVCachePlaceholders(). The names here
  // must match what createKVCachePlaceholders() registers with the model.
  // We look up each placeholder by name and point it at our cache slab;
  // this is the same wiring Model::setExternalTensors used to do, just
  // without going through that API.
  for (int i = 0; i < NUM_LAYERS; ++i) {
    auto &kc = kv_cache.getKeyCache(i);
    auto &vc = kv_cache.getValueCache(i);

    auto find_cache_placeholder = [this](const std::string &base_name) {
      for (const auto &suffix : {":0", ":input0", ":out0", ""}) {
        auto *tensor = model->getTensor(base_name + suffix);
        if (tensor != nullptr)
          return tensor;
      }
      return static_cast<nntrainer::Tensor *>(nullptr);
    };

    auto *kp =
      model->getTensor("layer" + std::to_string(i) + "_attention:input3");
    auto *vp =
      model->getTensor("layer" + std::to_string(i) + "_attention:input4");
    if (kp == nullptr)
      kp = find_cache_placeholder("cache_k_l" + std::to_string(i));
    if (vp == nullptr)
      vp = find_cache_placeholder("cache_v_l" + std::to_string(i));
    if (kp == nullptr && vp == nullptr) {
      /// This layer has no attention sub-graph (e.g., a conv-only block in a
      /// hybrid architecture like LFM2). Skip KV-cache binding for it.
      continue;
    }
    NNTR_THROW_IF(kp == nullptr || vp == nullptr, std::runtime_error)
      << "allocateAndBindKVCache: cache_k_l" << i << " / cache_v_l" << i
      << " partially found in compiled graph (one placeholder exists but "
         "the other does not)";
    NNTR_THROW_IF(kp->getDataType() != kc.getDataType() ||
                    vp->getDataType() != vc.getDataType(),
                  std::runtime_error)
      << "allocateAndBindKVCache: cache placeholder dtype mismatch for layer "
      << i;

    kp->setData(kc.getMemoryData(), kc.getOffset(), false);
    vp->setData(vc.getMemoryData(), vc.getOffset(), false);
  }

  kv_cache_bound = true;
}

std::vector<float *> CausalLM::incrementalInference(
  unsigned int batch_size, const std::vector<float *> &input,
  unsigned int init_seq_len, unsigned int from, unsigned int to) {
  // Same contract as NeuralNetwork::incremental_inference(float* ...), except
  // inputs whose raw pointer belongs to a KVCacheManager cache tensor are fed
  // as the REAL tensor (sharing its MemoryData, so isSVM() set by the SVM
  // MemoryPool survives the per-call fillPlaceholder/syncDependents). With
  // in-place input layers the mha_core cache views are dependents of the
  // input placeholder; a fresh Tensor::Map MemoryData of the same pointer
  // would clobber the SVM flag and drop attention to the host path.
  auto *nn = static_cast<nntrainer::NeuralNetwork *>(model.get());

  std::unordered_map<const void *, nntrainer::Tensor *> cache_by_ptr;
  if (kv_cache.isAllocated()) {
    for (unsigned int i = 0; i < kv_cache.getNumLayers(); ++i) {
      auto &kc = kv_cache.getKeyCache(i);
      auto &vc = kv_cache.getValueCache(i);
      if (kc.empty() || vc.empty())
        continue; // mixed KV mode: internal-int8 layer, no external tensor
      cache_by_ptr.emplace(reinterpret_cast<const void *>(kc.getData()), &kc);
      cache_by_ptr.emplace(reinterpret_cast<const void *>(vc.getData()), &vc);
    }
  }

  auto in_dim = nn->getInputDimension();
  NNTR_THROW_IF(input.size() < in_dim.size(), std::invalid_argument)
    << "incrementalInference: model expects " << in_dim.size()
    << " inputs, got " << input.size();

  nntrainer::sharedConstTensors input_tensors;
  input_tensors.reserve(in_dim.size());
  for (unsigned int idx = 0; idx < in_dim.size(); idx++) {
    auto it = cache_by_ptr.find(reinterpret_cast<const void *>(input[idx]));
    if (it != cache_by_ptr.end()) {
      // shallow copy: shares the cache's MemoryData (isSVM intact)
      input_tensors.emplace_back(MAKE_SHARED_TENSOR(*it->second));
    } else {
      in_dim[idx].batch(batch_size);
      input_tensors.emplace_back(
        MAKE_SHARED_TENSOR(mapExternalInput(input[idx], in_dim[idx])));
    }
  }

  nntrainer::sharedConstTensors output_tensors =
    nn->incremental_inference(input_tensors, init_seq_len, from, to);

  // Output conversion identical to the float* overload in neuralnet.cpp.
  std::vector<float *> output;
  output.reserve(output_tensors.size());
  for (auto &out : output_tensors) {
    auto out_t = *out.get();
    const size_t buf_size =
      static_cast<size_t>(batch_size) * out_t.getDim().getFeatureLen();
    float *last_out_buf_data = new float[buf_size];

    if (out->getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
      nntrainer::getComputeOps()->scopy_fp16_to_fp32(
        buf_size, out_t.getData<_FP16>(), 1, last_out_buf_data, 1);
#else
      delete[] last_out_buf_data;
      throw std::invalid_argument("Error: enable-fp16 is not set");
#endif
    } else if (out->getDataType() == ml::train::TensorDim::DataType::FP32) {
      std::memcpy(last_out_buf_data, out_t.getData(), sizeof(float) * buf_size);
    }

    output.push_back(last_out_buf_data);
  }

  return output;
}

void CausalLM::setKVCachePosition(unsigned int pos) {
  kv_cache.setPosition(pos);
  std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &, void *)>
    fn = [pos](ml::train::Layer &l, nntrainer::RunLayerContext &, void *) {
      if (l.getType() == causallm::MHACoreLayer::type)
        l.setProperty({"cache_index=" + std::to_string(pos)});
    };
  model->forEachLayer(fn, nullptr);
}

void CausalLM::advanceKVCachePosition(unsigned int step_size) {
  // mha_core advances its own cache_index inside forwarding(), so the host
  // only has to keep KVCacheManager's tracked position in sync.
  kv_cache.advance(step_size);
}

/**
 * [lmhead-untie] When nntr_config.json sets lmhead_untie, build
 * output_of_causallm as an independent fully_connected layer with its own
 * weight even for a tied-embedding model, so the lm_head can carry a
 * different dtype than the embedding (untied-serialized packages such as
 * gemma4_qs4cx_fp16 ship a separate transposed [hidden, vocab] head record
 * that a tied graph cannot load). Untie is the config flag, NOT derived from
 * LMHEAD_DTYPE: a quantizer constructs this same untied graph from the FP32
 * source and quantizes output_of_causallm via the dtype map on save.
 * skip_prefill keeps the FC lm_head decode-only, the same contract the tied
 * lm_head types implement internally. Flag off = byte-identical graph.
 */
Tensor CausalLM::buildLmHeadOutput(Tensor h, bool add_skip_prefill) {
  const bool lmhead_untied = LMHEAD_UNTIE;
  const std::string lmhead_type =
    lmhead_untied ? "fully_connected"
                  : (TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "lm_head");
  std::vector<std::string> lmhead_prop = {
    withKey("name", "output_of_causallm"),
    withKey("unit", NUM_VOCAB),
    withKey("disable_bias", "true"),
    withKey("weight_dtype", LMHEAD_DTYPE),
  };
  // The head must carry the graph's engine. It is the LAST node, so it reads
  // output_norm's activation -- which, once the rest of the graph is
  // engine-stamped (38de03c46 / 2c2b0d96e), lives on the gpu context's
  // cl_mem/SVM plane. A host head reads the stale host shadow of that plane, so
  // the logits are garbage and every model degenerates to one repeated token.
  // Measured on this tree: unstamped gemma4 answered "<pad>"-class garbage at
  // 0.23 TPS decode (a 262144-row QS4CX head on the host); stamped it answers
  // "The capital of South Korea is **Seoul**." at 20.6 TPS.
  // Both reachable types have a real gpu-context factory, so neither throws
  // exception::not_supported from createLayer:
  //   fully_connected     -> FullyConnectedLayerCl (cl_context.cpp
  //                          add_default_object)
  //   tie_word_embeddings -> TieWordEmbedding      (cl_context.cpp, gated on
  //                          registerGeGLUClKernels; same class on
  //                          cpu/gpu/cuda, it selects its Q6_K/Q4_0 GPU GEMV
  //                          internally)
  // "lm_head" (untied via config.json tie_word_embeddings=false, i.e. NOT
  // LMHEAD_UNTIE) has NO gpu registration and stays unstamped -- no in-tree
  // package reaches it, and stamping it would throw.
  if (lmhead_type != "lm_head")
    lmhead_prop.emplace_back(withKey("engine", causallm_engine()));
  if (add_skip_prefill)
    lmhead_prop.emplace_back(withKey("skip_prefill", "true"));
  if (TIE_WORD_EMBEDDINGS && !lmhead_untied)
    lmhead_prop.emplace_back(withKey("shared_from", "embedding0"));
  LayerHandle lmhead(createLayer(lmhead_type, lmhead_prop));
  return lmhead(h);
}

std::pair<Tensor, Tensor> CausalLM::constructModel() {

  // base transformer (input, output_norm)
  auto [x, h] = Transformer::constructModel();

  Tensor y = buildLmHeadOutput(h, LMHEAD_UNTIE && SKIP_PREFILL);

  return {x, y};
}

void CausalLM::registerOutputs(
  std::unique_ptr<tokenizers::Tokenizer> &tokenizer,
  std::vector<unsigned int> ids, unsigned int pos,
  const std::vector<bool> &eos_list, bool log_output) {

  static const std::vector<char> puncts{',', '!', ':', ';', '?'};
  for (size_t b = 0; b < ids.size(); ++b) {
    if (!eos_list[b]) {
      pending_ids_.push_back(static_cast<int>(ids[b]));
      ids_history[b * MAX_SEQ_LEN + pos] = ids[b];
      std::string decoded_str = tokenizer->Decode(pending_ids_);

      if (decoded_str.empty()) {
        continue;
      }

      if (std::find(puncts.begin(), puncts.end(), decoded_str.back()) !=
          puncts.end()) {
        // last symbol is a punctuation, hold on
      } else if (utf8stream::shouldHold(decoded_str, pending_ids_.size())) {
      } else {
        if (log_output && streamer_ == nullptr) {
          std::cout << decoded_str;
          std::cout.flush();
        }
        output_list[b].append(decoded_str);
        if (streamer_ != nullptr &&
            streamer_put(streamer_, decoded_str.c_str()) != 0) {
          requestStop();
        }
        pending_ids_.clear();
      }
    }
  }
}

void CausalLM::save_kvcache(std::string path, int to_) {
  if (!kv_cache.isAllocated()) {
    throw std::runtime_error(
      "save_kvcache called before allocateAndBindKVCache()");
  }
  kv_cache.save(path, static_cast<unsigned int>(to_));
}

void CausalLM::load_kvcache(std::string path, int to_) {
  if (!kv_cache.isAllocated()) {
    allocateAndBindKVCache();
  }
  kv_cache.load(path, static_cast<unsigned int>(to_));
  // mha_core layers each track their own cache_index; sync them all to the
  // newly-loaded position so the next forwarding() writes at the right slot.
  setKVCachePosition(static_cast<unsigned int>(to_));
}

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

    if (logits_processor != nullptr)
      logits_processor->process(logits, NUM_VOCAB, iteration);

    unsigned int output_id;

    // return argmax if do_sample is false
    if (do_sample == false) {
      output_id =
        std::distance(logits, std::max_element(logits, logits + NUM_VOCAB));
    } else {
      // apply temperature & top-k & top-p and sample with original logits
      // unchanged
      output_id = applyTKP(logits, NUM_VOCAB, TEMPERATURE, TOP_K, TOP_P, rng);
    }

    outputs.push_back(output_id);

    if (logits_processor != nullptr)
      logits_processor->acceptToken(output_id, iteration);

    // set batch offset
    logits = logits + NUM_VOCAB;
    if (input_ids != nullptr)
      input_ids = input_ids + MAX_SEQ_LEN;
  }

  return outputs;
};

void CausalLM::registerCustomLayers() {
  Transformer::registerCustomLayers();
  const auto &ct_engine = nntrainer::Engine::Global();
  const auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));
  // lm_head is a core layer now (nntrainer/layers/llm), registered by
  // AppContext itself.
  (void)app_context;
}

void CausalLM::run(const WSTR prompt, bool do_sample, const WSTR system_prompt,
                   const WSTR tail_prompt, bool log_output) {

  auto start_total = std::chrono::high_resolution_clock::now();
  if (!is_initialized) {
    throw std::runtime_error("CausalLM model is not initialized. Please call "
                             "initialize() before run().");
  }

  struct StreamerEndGuard {
    BaseStreamer *streamer;
    ~StreamerEndGuard() { streamer_end(streamer); }
  } streamer_end_guard{streamer_};

  // Allocate the host-owned KV cache and bind it to mha_core's external cache
  // input slots. Idempotent: only the first call does work; subsequent runs
  // reuse the same buffers and continue from the computed absolute token
  // position below.
  allocateAndBindKVCache();

  has_run_ = false;
  prepareStopRequestForRun();

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

  /**
   * SAVE_KVCACHE ?
   *  if USE_KVCACHE && system_prompt is given && but the
   * PRE_COMPUTED_CACHE_PATH does not exist
   */
  SAVE_KVCACHE = (USE_KVCACHE && system_prompt != "" &&
                  !std::filesystem::exists(PRE_COMPUTED_CACHE_PATH));

  // print input text
  if (log_output)
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

  ///@note add_special_tokens=true lets each model's OWN tokenizer decide
  /// whether
  /// to prepend a BOS, rather than hard-coding it. The 1-arg Encode skips
  /// special tokens, so the leading BOS that Gemma2 (TemplateProcessing,
  /// add_bos_token= true) needs was dropped -> short prompts degenerated into
  /// pure repetition
  /// ("The capital of France is" -> "is is is..."); long prompts masked it.
  /// Verified to match HF add_special_tokens=True per model: Gemma2 gains its
  /// BOS(2); models whose tokenizer adds no BOS (e.g. Qwen3 — ByteLevel post-
  /// processor, add_bos_token=false) are byte-identical to the old behavior, so
  /// they are unaffected. (sentence_transformer.cpp already encodes this way.)
  auto _input = tokenizer->Encode(prompt_, /*add_special_tokens=*/true);

  // | <------------------- MAX_SEQ_LEN -------------------> |
  //                       ||             ||
  // |<-- System prompt -->||<-- input -->||<-- generate -->|

  std::vector<int64_t> init_input;
  unsigned int _len = _input.size();
  unsigned int num_allow_str = MAX_SEQ_LEN - NUM_TO_GENERATE;
  unsigned int text_len = _len;

  if (_len > num_allow_str) {
    text_len = num_allow_str;
    // Truncation drops tokens from the tail of the prompt, which is where
    // instructions in "summarize this document"-style prompts live: a
    // silently truncated prompt can make the model continue the body
    // instead of following a dropped trailing instruction. Always warn
    // with the exact counts.
    std::cerr << "[CausalLM] WARNING: prompt (" << _len
              << " tokens) exceeds the max allowed prefill length ("
              << num_allow_str
              << " = max_seq_len - num_to_generate); "
                 "truncating "
              << (_len - num_allow_str) << " tail tokens." << std::endl;
  }

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
    (float *)calloc(BATCH_SIZE * MAX_SEQ_LEN, sizeof(float));
  std::vector<bool> eos_list(BATCH_SIZE, false);

  unsigned int input_len = init_len;

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
  auto build_inference_inputs = [&]() {
    std::vector<std::pair<std::string, float *>> cache_inputs;
    cache_inputs.reserve(static_cast<size_t>(NUM_LAYERS) * 2);
    for (int i = 0; i < NUM_LAYERS; ++i) {
      cache_inputs.emplace_back(
        "cache_k_l" + std::to_string(i),
        reinterpret_cast<float *>(kv_cache.getKeyCache(i).getData()));
      cache_inputs.emplace_back(
        "cache_v_l" + std::to_string(i),
        reinterpret_cast<float *>(kv_cache.getValueCache(i).getData()));
    }

    std::sort(
      cache_inputs.begin(), cache_inputs.end(),
      [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });

    std::vector<float *> inference_inputs;
    inference_inputs.reserve(1 + cache_inputs.size());
    inference_inputs.push_back(input_sample);
    for (const auto &cache_input : cache_inputs)
      inference_inputs.push_back(cache_input.second);
    return inference_inputs;
  };
  input = build_inference_inputs();

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

    if (log_output)
      std::cout << "\n==============[KV CACHE SAVE MODE]================\n";
    allocateAndBindKVCache();
    setKVCachePosition(0);
    output = incrementalInference(BATCH_SIZE, input, input_len, 0, input_len);

    SYS_PROMP_LEN = input_len;
    save_kvcache(PRE_COMPUTED_CACHE_PATH, SYS_PROMP_LEN);

    if (log_output) {

      std::cout << "kv caches are saved in " << PRE_COMPUTED_CACHE_PATH
                << std::endl
                << "and the size of prompt is " << SYS_PROMP_LEN << ".\n"
                << "You may need this prompt length to set the "
                   "\"sys_prompt_token_size\""
                << "\n==================================================\n"
                << std::endl;
    }
    return;
  }

  if (USE_KVCACHE) {
    load_kvcache(PRE_COMPUTED_CACHE_PATH, SYS_PROMP_LEN);
  } else {
    SYS_PROMP_LEN = 0;
  }
  allocateAndBindKVCache();
  const unsigned int prefill_from = SYS_PROMP_LEN + global_token_len;
  std::vector<unsigned int> id_list;

  if (SKIP_PREFILL && init_len > 1) {
    // Prefill only N-1 tokens; the last input token will be used as the first
    // token in the generation phase (assigned directly, not sampled).
    unsigned int skipped_token =
      static_cast<unsigned int>(init_input[init_len - 1]);

    const unsigned int prefill_to = prefill_from + input_len - 1;
    setKVCachePosition(prefill_from);
    output = incrementalInference(BATCH_SIZE, input, init_len - 1, prefill_from,
                                  prefill_to);

    for (unsigned int b = 0; b < BATCH_SIZE; ++b)
      id_list.push_back(skipped_token);

    // Adjust lengths so the generation loop processes the skipped token
    // at the correct KV cache position.
    input_len -= 1;
    init_len -= 1;
  } else {
    const unsigned int prefill_to = prefill_from + input_len;
    setKVCachePosition(prefill_from);
    output = incrementalInference(BATCH_SIZE, input, init_len, prefill_from,
                                  prefill_to);

    // post process of model output
    id_list = generate(output[0], do_sample, 1, ids_history, init_len);

    if (init_len < INIT_SEQ_LEN)
      registerOutputs(tokenizer, id_list, init_len, eos_list, log_output);
  }
  // output should be deallocated after use
  for (auto &out : output) {
    delete[] out;
  }

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

  for (unsigned int token_generation_idx = input_len + 1;
       token_generation_idx < input_len + 1 + NUM_TO_GENERATE &&
       !stop_requested_.load(std::memory_order_acquire);
       ++token_generation_idx) {

    allocateAndBindKVCache();
    auto output_interval = incrementalInference(
      BATCH_SIZE, input, input_len, token_generation_idx - 1 + global_token_len,
      token_generation_idx + global_token_len);
    std::vector<unsigned int> ids_list(generate(output_interval[0], do_sample));

    // Feed the newly generated token back as the next input token.
    // token_generation_idx always starts at input_len + 1, so we are
    // always in the auto-regressive generation phase here.
    for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
      input_sample[static_cast<size_t>(b) * MAX_SEQ_LEN] =
        static_cast<float>(ids_list[b]);
    }
    registerOutputs(tokenizer, ids_list, token_generation_idx, eos_list,
                    log_output);
    ++generation_cnt;

    // output should be deallocated after use
    for (auto out : output_interval) {
      delete[] out;
    }

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
      break;
    }

    if (stop_requested_.load(std::memory_order_acquire)) {
      break;
    }
  }

  // Always release the input buffer after the generation loop, whether
  // the loop exited early (EOS found) or ran to the maximum token limit.
  free(input_sample);

  global_token_len += (generation_cnt + init_len);

  auto finish_generation = std::chrono::high_resolution_clock::now();
  auto generation_duration =
    std::chrono::duration_cast<std::chrono::milliseconds>(finish_generation -
                                                          start_generation);

  auto finish_total = std::chrono::high_resolution_clock::now();
  auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
    finish_total - start_total);
  size_t peak_memory = getPeakMemoryKb();

  if (log_output) {

    std::cout << "\n\n";
    std::cout << "=================[ LLM with NNTrainer ]===================\n";
    std::cout << "prefill: " << init_len << " tokens, "
              << prefill_duration.count() << " ms, "
              << ((double)init_len / prefill_duration.count() * 1000)
              << " TPS\n";
    std::cout << "generation: " << generation_cnt << " tokens, "
              << generation_duration.count() << " ms, "
              << ((double)generation_cnt / generation_duration.count() * 1000)
              << " TPS\n";
    std::cout << "total: " << total_duration.count() << " ms\n";
    std::cout << "peak memory: " << peak_memory << " KB\n";
    std::cout << "==========================================================\n";
  }

  performance_metrics.prefill_tokens = init_len;
  performance_metrics.prefill_duration_ms = prefill_duration.count();
  performance_metrics.generation_tokens = generation_cnt;
  performance_metrics.generation_duration_ms = generation_duration.count();
  performance_metrics.total_duration_ms = total_duration.count();
  performance_metrics.peak_memory_kb = peak_memory;

  has_run_ = true;
}

std::string CausalLM::getOutput(int batch_idx) const {
  if (batch_idx < 0 || batch_idx >= static_cast<int>(output_list.size())) {
    return "";
  }
  return output_list[batch_idx];
}

} // namespace causallm
