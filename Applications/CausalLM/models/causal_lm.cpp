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

#include <compute_ops.h>
#include <env_compat.h> // nntr_env_on: value-checked read of auto-injected flags
#include <neuralnet.h>

#if defined(ENABLE_OPENCL)
#include <cl_context.h> // OpenCL-only; registration uses the Engine facade.
#endif
#include <common.h>

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
#include <cuda_context.h>
#endif
#include <layer_context.h>
#include <mha_core.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <reshaped_rms_norm.h>
#include <residency_policy.h>
#include <rms_reverse_norm.h>
#include <tensor.h>

#include <causal_lm.h>
#include <llm_util.hpp>

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
#include <cuda_context_manager.h>
#include <cuda_elementwise.h>
#include <cuda_runtime.h>
#include <cuda_stream_manager.h>
#endif

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
namespace nntrainer::cuda {
// Frees all cuBLAS int8 caches at the prefill -> decode boundary.
void cuda_fc_qs4cx_free_i8_caches();
} // namespace nntrainer::cuda

namespace {
// On-GPU greedy argmax. incrementalInference()
// stashes the DEVICE-resident lm_head logits pointer + dtype here (the tensor
// data, before the host copy), so generate() can reduce it to the 4-byte token
// id on the GPU instead of running host std::max_element over the full-vocab
// D->H copy. One batch row (BATCH_SIZE==1 only, like the CL argmax gating).
// Reset every call; valid only when the FP32/FP16 output was confirmed
// device-accessible.
const void *g_cuda_logits_dev = nullptr;
bool g_cuda_logits_fp16 = false;
bool cuda_argmax_enabled() {
  // NNTR_CUDA_DEV_ARGMAX, default ON with =0 opting out. The reduction is
  // numerically identical to the host one it replaces (same fp32 values, ties
  // to the lowest index), so there is nothing to trade off per lane; it just
  // has to be on to be worth anything.
  static const bool on = []() {
    const char *e = std::getenv("NNTR_CUDA_DEV_ARGMAX");
    return e == nullptr || e[0] != '0';
  }();
  return on;
}

// Deferred host logits.
//
// Greedy decoding reads exactly ONE number out of the lm_head output, but the
// output conversion below materializes the whole [vocab] row on the host every
// token: a 1MB allocation, a 524KB D2H, and an fp16->fp32 pass over 262144
// elements -- all of it discarded after a max_element. When the previous
// generate() was greedy, incrementalInference skips that work and records here
// what it skipped. generate() then either takes the on-GPU token (the normal
// case) or, if anything about this call disqualifies the device path,
// materializes the row itself from the stashed pointer. The greedy flag is a
// HINT, never a correctness input: a wrong guess costs a deferred conversion,
// it cannot produce a wrong token or an unfilled read.
struct PendingLogits {
  const void *dev = nullptr; /**< device/managed logits row */
  float *host = nullptr;     /**< caller-owned destination, not yet filled */
  size_t count = 0;
  bool fp16 = false;
};
PendingLogits g_pending_logits;
bool g_greedy_hint = false;

/** @brief Fill the deferred host row (no-op unless one is outstanding). */
void materialize_pending_logits() {
  PendingLogits p = g_pending_logits;
  g_pending_logits = PendingLogits{};
  if (p.dev == nullptr || p.host == nullptr)
    return;
  nntrainer::cuda::StreamManager::Global().finish();
  if (p.fp16) {
#ifdef ENABLE_FP16
    std::vector<_FP16> stage(p.count);
    cudaMemcpy(stage.data(), p.dev, p.count * sizeof(_FP16),
               cudaMemcpyDeviceToHost);
    nntrainer::getComputeOps()->scopy_fp16_to_fp32(p.count, stage.data(), 1,
                                                   p.host, 1);
#endif
  } else {
    cudaMemcpy(p.host, p.dev, p.count * sizeof(float), cudaMemcpyDeviceToHost);
  }
  cudaGetLastError();
}
} // namespace
#endif

#include <utf8_stream_util.h>

#include "api/streamer.h"

namespace causallm {

CausalLM::CausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
  Transformer(cfg, generation_cfg, nntr_cfg, ModelType::CAUSALLM) {
  // Declare CausalLM's static-residency boundaries so the core planner
  // (tensor_pool / manager) needs no app-specific tensor/layer names. These are
  // byte-identical to the former hardcoded core defaults: the embedding output
  // is host-produced but uploaded to cl_mem (RAISE), the final norm output is
  // GPU-produced but read back once by the host lm_head (LOWER), and mha_core
  // is a CPU-registered layer that binds/consumes Q/K/V on the GPU plane
  // (engine- neutral). NNTR_CLMEM_RAISE/LOWER still override the raise/lower
  // patterns.
  auto &rp = nntrainer::ResidencyPolicy::global();
#ifndef _WIN32
  if (rp.raise_patterns.empty())
    rp.raise_patterns = "embedding0:out0";
#else
  // The embedding-output RAISE (host-written per token, then
  // uploaded to the cl_mem plane) is the single tensor whose cl_mem binding
  // makes Windows/Intel runs nondeterministic: keep-one bisection over every
  // GPU_CLMEM class reproduced run-to-run divergence ONLY here, and excluding
  // just this tensor is bit-reproducible 6/6 with NO drain at baseline cost
  // (2079/18.5 vs base 2075/18.2 — zero). The raise's own rationale (layer0
  // coarse-SVM ingress) did not reproduce in any of those runs. Keep the
  // embedding on SVM on Windows; NNTR_CLMEM_RAISE overrides for A/B.
#endif
  if (rp.lower_patterns.empty())
    rp.lower_patterns = "output_norm:out0";
  if (rp.engine_neutral_types.empty())
    rp.engine_neutral_types = {"mha_core"};
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
  // a legacy int4 lm_head loads as the canonical QS4CX class
  // (on-disk stays legacy QINT4, transcoded at read time). See transformer.cpp.
  if (LMHEAD_DTYPE == "QINT4")
    LMHEAD_DTYPE = "QS4CX";

  // LMHEAD_UNTIE is parsed in Transformer::setupParameters (member moved
  // there so <model>Transformer::constructModel can gate embedding0's type).

  SKIP_PREFILL = nntr_cfg.contains("skip_prefill")
                   ? nntr_cfg["skip_prefill"].get<bool>()
                   : false;

  // Repetition penalty. applyRepetitionPenalty() has always been here but
  // nothing plumbed a value into it, so generate() only ever saw the
  // default 1 and the penalty was unreachable. Both keys are optional; the
  // defaults below are exactly the arguments the old call sites passed, so a
  // config without them decodes byte-identically (penalty == 1 in particular
  // is what keeps generate()'s on-GPU argmax engaged).
  REPETITION_PENALTY = nntr_cfg.contains("repetition_penalty")
                         ? nntr_cfg["repetition_penalty"].get<float>()
                         : 1.0f;
  REPETITION_WINDOW = nntr_cfg.contains("repetition_window")
                        ? nntr_cfg["repetition_window"].get<unsigned int>()
                        : 128;

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
  if (EOS_TOKEN_ID.empty()) {
    // Without an EOS id nothing can break the decode loop early, so a config
    // with no explicit num_to_generate will always run to the context window.
    std::cerr << "[CausalLM] WARNING: no eos_token_id in the config; "
                 "generation can only stop at num_to_generate or at the end of "
                 "the context window."
              << std::endl;
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

    // Per-layer width via the getKVCacheWidth() hook: uniform
    // (NUM_KEY_VALUE_HEADS * HEAD_DIM) by default, overridden by
    // variable-geometry models. One vector path now serves every model, so the
    // per-model allocateAndBindKVCache overrides are gone. (The scalar
    // allocate this used before only additionally set the KVCacheManager
    // num_heads_kv_/head_dim_ members, which have no reader -- getKVWidth() is
    // uncalled -- so this is behavior-identical.)
    std::vector<unsigned int> kv_widths(static_cast<size_t>(NUM_LAYERS), 0u);
    for (int i = 0; i < NUM_LAYERS; ++i)
      kv_widths[static_cast<size_t>(i)] = getKVCacheWidth(i);

    kv_cache.allocate(static_cast<unsigned int>(NUM_LAYERS), BATCH_SIZE,
                      max_timestep, kv_widths, cache_dtype);
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
    NNTR_THROW_IF(kp->getDim() != kc.getDim() || vp->getDim() != vc.getDim(),
                  std::runtime_error)
      << "allocateAndBindKVCache: cache placeholder shape mismatch for layer "
      << i
      << " (the placeholder shape from createKVCachePlaceholders and the "
         "KVCacheManager allocation must agree)";

    kp->setData(kc.getMemoryData(), kc.getOffset(), false);
    vp->setData(vc.getMemoryData(), vc.getOffset(), false);
  }

  kv_cache_bound = true;
}

float *CausalLM::acquireLogitsBuf(size_t count) {
  for (auto it = logits_pool_free_.begin(); it != logits_pool_free_.end();
       ++it) {
    if (it->first == count) {
      float *buf = it->second;
      logits_pool_free_.erase(it);
      return buf;
    }
  }
  float *buf = new float[count];
  logits_pool_sizes_.emplace(buf, count);
  return buf;
}

void CausalLM::releaseLogitsBuf(float *buf) {
  if (buf == nullptr)
    return;
  auto it = logits_pool_sizes_.find(buf);
  if (it == logits_pool_sizes_.end()) {
    delete[] buf; // not pool-owned (defensive; all in-tree callers are)
    return;
  }
  logits_pool_free_.emplace_back(it->second, buf);
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
      input_tensors.emplace_back(MAKE_SHARED_TENSOR(nntrainer::Tensor::Map(
        input[idx], in_dim[idx].getDataLen() * sizeof(float), in_dim[idx], 0)));
    }
  }

  nntrainer::sharedConstTensors output_tensors =
    nn->incremental_inference(input_tensors, init_seq_len, from, to);

  // Output conversion identical to the float* overload in neuralnet.cpp.
  std::vector<float *> output;
  output.reserve(output_tensors.size());
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
  // Invalidate any stale device-logits stash; re-armed below
  // only when this call's first output is device-accessible (UVM / managed /
  // device) so generate() can run the on-GPU argmax instead of host
  // max_element.
  g_cuda_logits_dev = nullptr;
  // A row deferred by the previous call and never claimed belongs to a buffer
  // the caller has since freed; drop it rather than write through it later.
  g_pending_logits = PendingLogits{};
  // first_output is read only by the device-logits stash checks below
  // (both the fp16 and fp32 branches); without CUDA there is no reader.
  bool first_output = true;
#endif
  for (auto &out : output_tensors) {
    auto out_t = *out.get();
    const size_t buf_size = (size_t)batch_size * out_t.getDim().getFeatureLen();
    // Pooled host staging: decode paid a fresh 1MB new[]/delete[] round trip
    // per token here (mostly never even written under the deferred-logits
    // path); the pool recycles the row instead. Contents are as uninitialized
    // as a fresh new[] -- everything below either fills it or defers it.
    float *last_out_buf_data = acquireLogitsBuf(buf_size);

    if (out->getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
      const _FP16 *out_src = out_t.getData<_FP16>();
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
      // Per-token cudart touches (pointer probes + stream drains) are cuda-run
      // only: on a non-cuda run of the unified binary the first cudart call
      // boots the statically-linked runtime inside this (timed) path.
      std::vector<_FP16> out_host;
      bool defer_host_logits = false;
      if (causallm_engine() == "cuda") {
        // Stash the device logits pointer (before the D2H copy) when
        // device-accessible, for generate()'s on-GPU argmax. batch_size==1
        // only (the argmax reduces a single [vocab] row).
        if (cuda_argmax_enabled() && first_output && batch_size == 1) {
          cudaPointerAttributes pa0{};
          if (cudaPointerGetAttributes(&pa0, out_src) == cudaSuccess &&
              (pa0.type == cudaMemoryTypeDevice ||
               pa0.type == cudaMemoryTypeManaged)) {
            g_cuda_logits_dev = out_src;
            g_cuda_logits_fp16 = true;
            // Greedy last time -> generate() will read one index off the GPU
            // and never touch the host row. Hand it the ingredients instead of
            // the row (see PendingLogits).
            if (g_greedy_hint) {
              g_pending_logits = {out_src, last_out_buf_data, buf_size, true};
              defer_host_logits = true;
            }
          }
          cudaGetLastError();
        }
        // Device-only activation pool (NNTR_CUDA_DEV_ACT): the model output is
        // real device memory, not host-addressable. Drain the backend stream
        // and copy it D2H into a host buffer before the host fp16->fp32
        // convert (=the one sync-per-token boundary). For UVM the pointer is
        // host-coherent so this is skipped.
        cudaPointerAttributes pa{};
        if (defer_host_logits) {
          // nothing to stage: nobody is going to read the host row
        } else if (cudaPointerGetAttributes(&pa, out_src) == cudaSuccess &&
                   pa.type == cudaMemoryTypeDevice) {
          nntrainer::cuda::StreamManager::Global().finish();
          out_host.resize(buf_size);
          cudaMemcpy(out_host.data(), out_src, buf_size * sizeof(_FP16),
                     cudaMemcpyDeviceToHost);
          out_src = out_host.data();
        } else {
          // UVM/managed pointer: host-coherent for ADDRESSING, but the
          // producing work may still be in flight -- under NNTR_CUDA_ASYNC
          // (eager, no per-op drain) and, since the sync consolidation in
          // CudaContext::runDecode, after an M2B graph replay in sync mode
          // too (the replay no longer stream-syncs; each host consumer
          // drains once itself). Full finish(), not finishIfAsync().
          nntrainer::cuda::StreamManager::Global().finish();
        }
        cudaGetLastError();
      }
#endif
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
      if (!defer_host_logits)
#endif
        nntrainer::getComputeOps()->scopy_fp16_to_fp32(buf_size, out_src, 1,
                                                       last_out_buf_data, 1);
#else
      throw std::invalid_argument("Error: enable-fp16 is not set");
#endif
    } else if (out->getDataType() == ml::train::TensorDim::DataType::FP32) {
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
      bool defer_host_logits = false;
      // Per-token cudart touches are cuda-run only (see the fp16 branch note).
      if (causallm_engine() == "cuda") {
        // Stash the device logits pointer (the tensor data,
        // before the host memcpy below) when device-accessible. UVM/managed
        // pointers are host-coherent, so this same pointer feeds both the
        // on-GPU argmax kernel and -- as the fallback -- the host memcpy.
        if (cuda_argmax_enabled() && first_output && batch_size == 1) {
          const float *out_src = out_t.getData();
          cudaPointerAttributes pa0{};
          if (cudaPointerGetAttributes(&pa0, out_src) == cudaSuccess &&
              (pa0.type == cudaMemoryTypeDevice ||
               pa0.type == cudaMemoryTypeManaged)) {
            g_cuda_logits_dev = out_src;
            g_cuda_logits_fp16 = false;
            if (g_greedy_hint) {
              g_pending_logits = {out_src, last_out_buf_data, buf_size, false};
              defer_host_logits = true;
            }
          }
          cudaGetLastError();
        }
        // Host read of the GPU-produced logits: sync first so the read is
        // coherent -- under NNTR_CUDA_ASYNC and, since the runDecode sync
        // consolidation, after a sync-mode M2B replay as well (the replay no
        // longer drains; each host consumer waits once itself). Skipped when
        // the row is deferred: nobody reads it, and the on-GPU argmax path
        // does its own single finish() before the 4-byte D2H.
        if (!defer_host_logits)
          nntrainer::cuda::StreamManager::Global().finish();
      }
      // Nothing to materialize when the row is deferred: no reader for it.
      if (!defer_host_logits)
        std::memcpy(last_out_buf_data, out_t.getData(),
                    sizeof(float) * buf_size);
#else
      std::memcpy(last_out_buf_data, out_t.getData(), sizeof(float) * buf_size);
#endif
    }

    output.push_back(last_out_buf_data);
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
    first_output = false;
#endif
  }

  // Keep the host-side KV tracker at the absolute position just written:
  // mha_core advances only its own internal cache_index during forwarding,
  // so without this getKvLen()/save_kvcache() would report the stale
  // prefill-start position (advanceKVCachePosition had no caller).
  if (kv_cache.isAllocated() && to <= kv_cache.getMaxSeqLen())
    kv_cache.setPosition(to);

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
 * @brief Build the output_of_causallm lm_head from the transformer hidden
 * state h. Shared by the generic and the diamond-inheritance model
 * constructModel paths, which previously copy-pasted this block: pick
 * fully_connected / tie_word_embeddings / lm_head, assemble the
 * name/unit/disable_bias/weight_dtype/engine props, conditionally append
 * skip_prefill and shared_from, create the layer and apply it. The
 * skip_prefill decision differs per path, so it is passed in rather than
 * recomputed here.
 *
 * When the config unties the head (lmhead_untie) the layer is an independent
 * fully_connected with its own weight even for a tied-embedding model, so the
 * head can carry a different dtype than the input embedding (e.g. an int4 GEMV
 * head over a Q6_K gather embedding -- the tied full-vocab GEMV is the single
 * largest decode kernel on every backend). Untie is the config flag, NOT
 * derived from the head dtype: the quantizer constructs this same untied graph
 * from the FP32 source and quantizes output_of_causallm via the dtype map on
 * save.
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
    withKey("engine", causallm_engine()),
  };
  // skip_prefill must agree with the caller's runtime skip-prefill flag: the
  // layer property alone makes fc_layer early-return during prefill, but run()
  // only routes around that when the model-level flag is set too -- tagging the
  // layer without the flag silently yields garbage prefill logits.
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

  // ModelFeatures x DeviceCaps matcher SHADOW: the model declares its
  // features, the resolver combines them with the chosen backend's device caps
  // into an ExecPlan. Log-only (no decision site reads the plan yet) — the
  // model graph is unchanged, so this is byte-identical. docs §10 T11.
  if (const auto *ct =
        nntrainer::Engine::Global().getRegisteredContext(causallm_engine())) {
    const auto feats = getModelFeatures();
    const auto plan = nntrainer::resolveExecPlan(ct->caps(), feats);
    ml_logi("[CausalLM] %s | %s (shadow)", feats.toString().c_str(),
            plan.toString().c_str());
  }

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

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
  // Terminal drain for the selective-sync (NNTR_CUDA_ASYNC) decode path: ensure
  // the GPU has finished producing the logits before the host reads them here.
  // No-op in default mode (every GPU op already drained per-op). cuda engine
  // ONLY: in a dual-enabled (CUDA+OpenCL) binary this ran on OpenCL runs too,
  // and StreamManager::Global() lazily CREATES the CUDA context -- the first
  // stray CUDA touch on a non-cuda run (NVIDIA VRAM burned on an Intel run).
  if (causallm_engine() == "cuda")
    nntrainer::cuda::StreamManager::Global().finish();
#endif

  std::vector<unsigned int> outputs;
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
  // Repetition penalty / bad-words activity, with EXACTLY the host-path
  // predicates below (generate() applies each only under its own triple
  // check, so the device route must mirror both or it would penalize where
  // the host would not).
  const bool rp_active =
    repetition_penalty != 1 && input_ids != nullptr && NUM_INPUT_IDS != 0;
  const bool bw_active = BAD_WORD_IDS.size() != 0 && NUM_BADWORDS != 0;
  // Tell the NEXT incrementalInference whether the full-vocab host row is
  // going to be read at all. Recomputed every call, so a run that switches
  // sampling on pays one deferred conversion and then stops deferring.
  g_greedy_hint = cuda_argmax_enabled() && do_sample == false &&
                  logits_processor == nullptr && BATCH_SIZE == 1 &&
                  !rp_active && !bw_active;
#endif
  for (unsigned int iteration = 0; iteration < BATCH_SIZE; ++iteration) {
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
    // CUDA on-GPU greedy argmax: reduce the device-resident
    // lm_head logits to the token id on the GPU and read back only 4 bytes,
    // skipping the host std::max_element over the full-vocab buffer. Gated to
    // greedy (no sampling, no logits processor -- those consume logits on the
    // host) and only when incrementalInference stashed a device-accessible
    // logits pointer for this (single, BATCH_SIZE==1) row. A repetition or
    // bad-words penalty mutates the logits on the host, so those rows keep the
    // host path.
    if (cuda_argmax_enabled() && g_cuda_logits_dev != nullptr &&
        do_sample == false && logits_processor == nullptr && !rp_active &&
        !bw_active) {
      unsigned int tok = 0;
      const bool ok =
        g_cuda_logits_fp16
          ? nntrainer::cuda::cuda_argmax_fp16(
              reinterpret_cast<const unsigned short *>(g_cuda_logits_dev),
              NUM_VOCAB, &tok)
          : nntrainer::cuda::cuda_argmax_fp32(
              reinterpret_cast<const float *>(g_cuda_logits_dev), NUM_VOCAB,
              &tok);
      // Consume the stash regardless (it belongs to this call's logits row).
      g_cuda_logits_dev = nullptr;
      if (ok) {
        // The deferred host row was never needed -- drop it unfilled.
        g_pending_logits = PendingLogits{};
        outputs.push_back(tok);
        logits = logits + NUM_VOCAB;
        input_ids = input_ids + MAX_SEQ_LEN;
        continue;
      }
      // else: fall through to the host path below.
    }
    // Reached only when the device path did not take this row: either the gate
    // above refused it or the reduction failed. Either way the host row may
    // still be outstanding, so fill it before anyone reads `logits`.
    materialize_pending_logits();
#endif

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
  // lm_head promoted to core app_context.cpp (cpu).

  // Register ReshapedRMSNormLayer on the GPU (cl) context once, centrally, so
  // ANY model can build its per-head q/k/v norms with engine=GPU and keep them
  // GPU_CLMEM-resident (Gemma4 S1.1 / Qwen3 q/k norm) instead of draining to
  // the host every layer. Future q/k-norm models get GPU residency for free:
  // they only need their existing cpu-context registration plus
  // engine=causallm_engine() on the reshaped_rms_norm layer. Inert (skipped)
  // when there is no GPU context (CPU-only / NNTR_ENGINE=cpu builds). The
  // per-model registerCustomLayers still registers it on the cpu context.
  // Goes through Engine's registration facade — no static_cast to ClContext.
  try {
    ct_engine.registerLayerFactory(
      "gpu", nntrainer::createLayer<causallm::ReshapedRMSNormLayer>);
  } catch (std::invalid_argument &e) {
    // no "gpu" context (CPU-only build) or already registered — both benign.
  }

  // PLE post_norm (RMSReverseNormLayer) on the GPU context: its
  // incremental_forwarding runs the reverse-norm as a GPU op (no host op inside
  // the async GPU graph). Same central-registration pattern as the reshaped
  // norm above; inert on CPU-only builds.
  try {
    ct_engine.registerLayerFactory(
      "gpu", nntrainer::createLayer<causallm::RMSReverseNormLayer>);
  } catch (std::invalid_argument &e) {
    // no "gpu" context or already registered — both benign.
  }

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
  // Centralized cuda-context registration of ReshapedRMSNormLayer (mirror of
  // the cl block above). engine=cuda tensors are UVM (host-coherent) and not
  // isSVM()-flagged, so its forwarding takes the correct host path.
  try {
    ct_engine.registerLayerFactory(
      "cuda", nntrainer::createLayer<causallm::ReshapedRMSNormLayer>);
  } catch (std::invalid_argument &e) {
    // no "cuda" context or already registered — both benign.
  }
  // PLE post_norm (RMSReverseNormLayer) on the cuda context: UVM
  // tensors are host-coherent and not isSVM()-flagged, so its forwarding
  // takes the correct host FP32-temp path (mirror of the "gpu" block above).
  try {
    ct_engine.registerLayerFactory(
      "cuda", nntrainer::createLayer<causallm::RMSReverseNormLayer>);
  } catch (std::invalid_argument &e) {
    // no "cuda" context or already registered — both benign.
  }
#endif
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
  std::vector<float *> label;

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

  // join the async tokenizer load before first use
  // (covers both Encode calls below and every later Decode this run).
  ensureTokenizer();

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
  // The prefill activation buffers are built at INIT_SEQ_LEN (transformer.cpp
  // constructModel, {1,1,1,INIT_SEQ_LEN}); resetInputDimension is disabled, so
  // ONE forward pass cannot process more than INIT_SEQ_LEN query rows without
  // overflowing the shared activation tensor (getSharedDataTensor bounds-check
  // throw), so the prompt is capped by INIT_SEQ_LEN as well as by the KV
  // budget below -- whichever is smaller.
  //
  // NUM_TO_GENERATE <= 0 means "no explicit cap": generation runs until EOS or
  // until the window is full, so nothing has to be held back for it beyond the
  // single position the decode loop starts at (input_len + 1). Reserving 1
  // keeps the prompt budget at MAX_SEQ_LEN - 1 and that loop in bounds.
  const bool unlimited_generation = (NUM_TO_GENERATE <= 0);
  const unsigned int reserved_for_generation =
    unlimited_generation ? 1u : static_cast<unsigned int>(NUM_TO_GENERATE);
  const unsigned int kv_budget = MAX_SEQ_LEN > reserved_for_generation
                                   ? MAX_SEQ_LEN - reserved_for_generation
                                   : 0u;
  unsigned int num_allow_str = std::min<unsigned int>(INIT_SEQ_LEN, kv_budget);
  unsigned int text_len = _len;

  if (_len > num_allow_str) {
    text_len = num_allow_str;
    // Silent tail truncation loses whatever the prompt ENDS with (round-13
    // field case: a summarization instruction at the tail was dropped and
    // the model continued the body instead). Unexpected state -> always warn.
    // The named limit follows the budget actually applied: without an explicit
    // cap it is max_seq_len - 1, not max_seq_len - num_to_generate.
    std::fprintf(
      stderr,
      "[causallm] WARNING: prompt (%u tokens) exceeds the prefill window "
      "(init_seq_len=%u, %s=%u); truncating %u "
      "tail tokens. Raise init_seq_len to fit the prompt.\n",
      _len, static_cast<unsigned int>(INIT_SEQ_LEN),
      unlimited_generation ? "max_seq_len-1 (generating until EOS)"
                           : "max_seq_len-num_to_generate",
      kv_budget, _len - num_allow_str);
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
    (float *)malloc(sizeof(float) * BATCH_SIZE * MAX_SEQ_LEN);
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

  // [resume-block] A resumed prefill (multi-turn continuation or a restored
  // KV cache, prefill_from > 0) runs as ONE block call — same shape as the
  // first prefill but with `from` at the absolute position — now that the
  // shared llm activation layers accept from>0 with to-from>1 (their row
  // math is base+live-window on every backend; attention/mha_core always
  // supported it). NNTR_RESUME_BLOCK=0 restores the legacy token-by-token
  // decode-shape feed (prefill at decode TPS) as an escape hatch.
  static const char *_rb_env = std::getenv("NNTR_RESUME_BLOCK");
  static const bool resume_block_on = !(_rb_env && _rb_env[0] == '0');

  auto do_prefill = [&](unsigned int n_tok,
                        unsigned int from_pos) -> std::vector<float *> {
    // Legacy token-by-token resumed prefill (debug escape hatch) -- unchanged.
    if (!resume_block_on && from_pos > 0 && n_tok > 1) {
      std::vector<float *> out;
      for (unsigned int i = 0; i < n_tok; ++i) {
        for (unsigned int b = 0; b < BATCH_SIZE; ++b)
          input_sample[static_cast<size_t>(b) * MAX_SEQ_LEN] =
            static_cast<float>(init_input[i]);
        const unsigned int p = from_pos + i;
        auto so = incrementalInference(BATCH_SIZE, input, p, p, p + 1);
        if (i + 1 < n_tok)
          for (auto &o : so)
            releaseLogitsBuf(o);
        else
          out = std::move(so);
      }
      return out;
    }
    // Single block: one forward over the whole prompt window.
    return incrementalInference(BATCH_SIZE, input, n_tok, from_pos,
                                from_pos + n_tok);
  };

  if (SKIP_PREFILL && init_len > 1) {
    // Prefill only N-1 tokens; the last input token will be used as the first
    // token in the generation phase (assigned directly, not sampled).
    unsigned int skipped_token =
      static_cast<unsigned int>(init_input[init_len - 1]);

    setKVCachePosition(prefill_from);
    output = do_prefill(init_len - 1, prefill_from);

    for (unsigned int b = 0; b < BATCH_SIZE; ++b)
      id_list.push_back(skipped_token);

    // Adjust lengths so the generation loop processes the skipped token
    // at the correct KV cache position.
    input_len -= 1;
    init_len -= 1;
  } else {
    setKVCachePosition(prefill_from);
    output = do_prefill(init_len, prefill_from);

    // post process of model output
    id_list = generate(output[0], do_sample, 1, ids_history, init_len);

    if (init_len < INIT_SEQ_LEN)
      registerOutputs(tokenizer, id_list, init_len, eos_list, log_output);
  }
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
  // NNTR_CUDA_I8_EPHEMERAL=1: the prefill just finished and
  // decode (M=1, dp4a) never reads the cuBLAS-i8 caches -- free them here so
  // the decode phase runs without their VRAM residency (~1.2GB measured).
  // Multi-turn: the next prefill lazily rebuilds (slower TTFT on that turn).
  {
    static const bool i8_ephemeral = []() {
      const char *e = std::getenv("NNTR_CUDA_I8_EPHEMERAL");
      return e != nullptr && e[0] == '1';
    }();
    if (i8_ephemeral && causallm_engine() == "cuda")
      nntrainer::cuda::cuda_fc_qs4cx_free_i8_caches();
  }
#endif
  // output should be released after use (returns the row to the pool)
  for (auto &out : output) {
    releaseLogitsBuf(out);
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

  // registerOutputs() writes ids_history[b * MAX_SEQ_LEN + idx] with no bounds
  // check, so the loop index has to stay inside the row stride the buffer was
  // allocated with. A budget that fits the window is not enough on its own: the
  // loop starts one past input_len, and input_len carries SYS_PROMP_LEN (added
  // just above) on top of the already-truncated prompt, so
  // input_len + NUM_TO_GENERATE can still reach MAX_SEQ_LEN. Derive the end
  // from the window too and stop at whichever comes first.
  //
  // Without an explicit cap (NUM_TO_GENERATE <= 0) the budget is the whole
  // window; the std::min below is then what actually bounds the loop, so
  // generation runs until EOS breaks out of it or the window is exhausted.
  const unsigned int generation_budget =
    NUM_TO_GENERATE > 0 ? static_cast<unsigned int>(NUM_TO_GENERATE)
                        : MAX_SEQ_LEN;
  const unsigned int generation_begin = input_len + 1;
  const unsigned int generation_end =
    generation_begin < MAX_SEQ_LEN
      ? generation_begin +
          std::min(MAX_SEQ_LEN - generation_begin, generation_budget)
      : generation_begin;

  for (unsigned int token_generation_idx = generation_begin;
       token_generation_idx < generation_end &&
       !stop_requested_.load(std::memory_order_acquire);
       ++token_generation_idx) {

    allocateAndBindKVCache();
    auto output_interval = incrementalInference(
      BATCH_SIZE, input, input_len, token_generation_idx - 1 + global_token_len,
      token_generation_idx + global_token_len);
    // Repetition-penalty window. ids_history is one MAX_SEQ_LEN-strided row of
    // token ids per batch entry -- the same layout generate() walks with
    // `input_ids += MAX_SEQ_LEN` -- and registerOutputs() below has already
    // filled [generation_begin, token_generation_idx) with the tokens THIS
    // loop produced, so the window is just a suffix of that span and the
    // pointer can be offset directly. Two deliberate boundaries:
    //   - the prompt is excluded. Penalizing it would suppress exactly the
    //     vocabulary a summarization task has to reuse, and with a 29K prompt
    //     it would also swamp the generated tail it is meant to police.
    //   - the span is empty on the first decode step (nothing generated yet to
    //     repeat), which is also the one step that keeps the on-GPU argmax.
    // With the penalty off both arguments collapse to generate()'s defaults,
    // so the greedy device path is bit-for-bit what it was.
    const unsigned int rep_window_len =
      REPETITION_PENALTY != 1.0f
        ? std::min(token_generation_idx - generation_begin, REPETITION_WINDOW)
        : 0u;
    std::vector<unsigned int> ids_list(
      generate(output_interval[0], do_sample,
               rep_window_len != 0 ? REPETITION_PENALTY : 1.0f,
               rep_window_len != 0
                 ? ids_history + (token_generation_idx - rep_window_len)
                 : nullptr,
               rep_window_len));

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

    // output should be released after use (returns the row to the pool; the
    // deferred-logits stash referencing it was already consumed by generate())
    for (auto out : output_interval) {
      releaseLogitsBuf(out);
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
