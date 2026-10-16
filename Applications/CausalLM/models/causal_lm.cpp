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
 * @note   This causal_lm.cpp implements a class for Transformer-based Causal
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
#include <future>
#include <htp_fixed_opts.h>
#include <iostream>
#include <iterator>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <compute_ops.h>
#include <env_compat.h> // nntr_env_on: value-checked read of auto-injected flags
#include <neuralnet.h>

#if defined(ENABLE_OPENCL)
#include <attention_kernels.h>     // [reload] cl_attention_release_caches
#include <blas_kernel_interface.h> // [reload] cl_fc_release_caches
#include <blas_kernels.h> // the lm_head GEMV's deferred-logits / argmax hooks
#include <cl_context.h>   // OpenCL-only; registration uses the Engine facade.

namespace nntrainer::opencl {
/**
 * @brief GPU memory ledger phase boundary.
 *
 * Declared rather than included: the ledger lives behind the OpenCL backend's
 * own headers and the app wants exactly this one call, so that the phases a
 * footprint question turns on (load / prefill / decode) are named by the only
 * code that knows where they are. Inert unless NNTR_GPU_MEM_ACCT is set.
 */
void clMemAcctDump(const char *phase);
/** Launch ledger (NNTR_CL_LAUNCH_STAT): the decode loop names its token
 *  boundaries so the per-token launch table is over steady tokens only. */
void clLaunchStatBegin(const char *phase);
void clLaunchStatTick();
void clLaunchStatDump(const char *phase);
} // namespace nntrainer::opencl
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
#include <embedding_layer.h> // [lmhead-tie-lut] QuantLut / get_or_load_quant_lut
#include <footprint_sampler.h> // per-run honest peak (RssAnon + accelerator)
#include <htp_decode_hook.h>
#include <htp_mha_core.h>
#include <llm_util.hpp>
#if !defined(_WIN32)
#include <sys/mman.h> // [lmhead-tie-lut] madvise the uploaded LUT mmap away
#endif

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
#include <cuda_attention.h>
#include <cuda_context_manager.h>
#include <cuda_elementwise.h>
#include <cuda_emb_gather.h> // [reload] gather LUT teardown
#include <cuda_fc_qs4cx.h>   // [lmhead-tie-lut] tied-head LUT registration
#include <cuda_runtime.h>
#include <cuda_stream_manager.h>
#endif

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
// cuda_fc_qs4cx_free_i8_caches() -- frees all cuBLAS int8 caches at the
// prefill -> decode boundary -- now comes from <cuda_fc_qs4cx.h> above; a local
// re-declaration here would trip -Werror=redundant-decls.

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

#if defined(ENABLE_OPENCL)
namespace {
// Deferred host logits, OpenCL lane -- the counterpart of the CUDA
// PendingLogits above, one level further out.
//
// On this backend the logits reach the host in stages: the lm_head GEMV reads
// the device row into the FC output tensor, a monotone post-op (gemma4's final
// logit softcapping) rewrites it into the model's output tensor, and this
// function widens THAT into the fp32 buffer generate() scans. Under the greedy
// hint the first two stand down (the deferral in blas_kernels, which the
// reduction absorbs), so the widening has nothing to widen: record what it
// skipped instead. generate() then either takes the on-GPU token -- and the
// row is never materialized at all -- or fills it here, replaying exactly the
// stages that were skipped and nothing else.
struct ClPendingLogits {
  const void *src_fp16 = nullptr; /**< the model's fp16 row, left unwritten */
  float *host = nullptr;          /**< caller-owned fp32 row, not yet filled */
  size_t count = 0;
};
ClPendingLogits g_cl_pending_logits;

/** @brief Fill the deferred host row (no-op unless one is outstanding). */
void materialize_cl_pending_logits() {
  const ClPendingLogits p = g_cl_pending_logits;
  g_cl_pending_logits = ClPendingLogits{};
  if (p.src_fp16 == nullptr || p.host == nullptr)
    return;
#ifdef ENABLE_FP16
  // Device -> the FC row, replay the post-op into the model's row, then widen.
  nntrainer::cl_lmhead_materialize_logits();
  nntrainer::getComputeOps()->scopy_fp16_to_fp32(
    p.count, static_cast<const _FP16 *>(p.src_fp16), 1, p.host, 1);
#endif
}
} // namespace
#endif

namespace {
/**
 * @brief NNTR_SAMPLE_DEVICE_TOPK, default ON with "=0" opting out: sampled
 *        decoding on the OpenCL / CUDA lanes takes its top-k candidates from
 *        the GPU instead of reading back and scanning the whole logits row.
 * @note The candidates give the same token as the full row (the sampler
 *       declines and the row is read whenever they cannot prove it), so the
 *       env is a bisection lever, not a policy knob.
 */
bool sample_device_topk_enabled() {
  static const bool on = []() {
    const char *e = std::getenv("NNTR_SAMPLE_DEVICE_TOPK");
    return e == nullptr || e[0] != '0';
  }();
  return on;
}
} // namespace

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
  // (engine- neutral). Patterns already set on the ResidencyPolicy before this
  // constructor runs are kept.
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
  // embedding on SVM on Windows; a raise pattern set on the ResidencyPolicy
  // beforehand overrides for A/B.
#endif
  if (rp.lower_patterns.empty())
    rp.lower_patterns = "output_norm:out0";
  // The KV cache is the one activation the application itself binds and reads
  // through two different paths -- the KVCacheManager's own allocator plus the
  // attention layer's host path -- so it must stay on the shared plane. Core
  // used to hard-code this exclusion in the planner; declaring it here is the
  // same rule, moved to the side that owns the tensor, and without it the
  // cache is placed on the device plane while a host reader still holds the
  // shared pointer.
  if (rp.exclude_patterns.empty())
    rp.exclude_patterns = "cache_";
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
  // A pack declares whether the prefill K rotation has to drain before
  // k_scatter_ohwi_cl reads the SVM cache slice it wrote. Default false --
  // the per-write no-drain -- and true only where the missing drain has been
  // observed to decide a token (mha_core.h, setPrefillKvDrain). Not inferred
  // from any other property and never from a model name; see the comment at
  // the call site for why the earlier skip_prefill proxy was wrong.
  setPrefillKvDrain(nntr_cfg.contains("prefill_kv_drain")
                      ? nntr_cfg["prefill_kv_drain"].get<bool>()
                      : false);

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
  // Decoding follows generation_config.json. A missing key takes the Hugging
  // Face default: do_sample false (greedy), and -- consulted only when
  // do_sample is true -- temperature 1.0, top_k 50, top_p 1.0.
  DO_SAMPLE = false;
  if (generation_cfg.contains("do_sample") &&
      generation_cfg["do_sample"].is_boolean())
    DO_SAMPLE = generation_cfg["do_sample"].get<bool>();
  TOP_K = generation_cfg.contains("top_k")
            ? generation_cfg["top_k"].get<unsigned int>()
            : 50;
  TOP_P = generation_cfg.contains("top_p")
            ? generation_cfg["top_p"].get<float>()
            : 1.0f;
  TEMPERATURE = generation_cfg.contains("temperature")
                  ? generation_cfg["temperature"].get<float>()
                  : 1.0f;
  // One fixed seed everywhere unless the pack names one; every run() re-seeds
  // with it (beginSamplingRun), so sampled output is a function of the prompt.
  if (generation_cfg.contains("seed") &&
      generation_cfg["seed"].is_number_unsigned())
    SAMPLING_SEED = generation_cfg["seed"].get<uint64_t>();
  else if (nntr_cfg.contains("seed") && nntr_cfg["seed"].is_number_unsigned())
    SAMPLING_SEED = nntr_cfg["seed"].get<uint64_t>();
  else
    SAMPLING_SEED = causallm::kDefaultSamplingSeed;
  rng.seed(SAMPLING_SEED);
  global_token_len = 0;
}

CausalLM::KVCacheLayout CausalLM::planKVCache() {
  KVCacheLayout layout;
  layout.max_timestep = static_cast<unsigned int>(MAX_SEQ_LEN);

  // [kv-window-ring] Per-layer physical row capacity: a sliding-window layer
  // stores a Wcap-row ring, every other layer keeps max_seq. Same
  // getKVCacheRows() hook the placeholder factories use, so the two shapes
  // line up at bind time (the bind asserts they do).
  std::vector<unsigned int> layer_caps(static_cast<size_t>(NUM_LAYERS), 0u);
  for (int i = 0; i < NUM_LAYERS; ++i)
    layer_caps[static_cast<size_t>(i)] = getKVCacheRows(i);
  kv_cache.setLayerCaps(layer_caps);

  // Per-layer width via the getKVCacheWidth() hook: uniform
  // (NUM_KEY_VALUE_HEADS * HEAD_DIM) by default, overridden by
  // variable-geometry models. One vector path now serves every model, so the
  // per-model allocateAndBindKVCache overrides are gone. (The scalar
  // allocate this used before only additionally set the KVCacheManager
  // num_heads_kv_/head_dim_ members, which have no reader -- getKVWidth() is
  // uncalled -- so this is behavior-identical.)
  layout.widths.assign(static_cast<size_t>(NUM_LAYERS), 0u);
  for (int i = 0; i < NUM_LAYERS; ++i)
    layout.widths[static_cast<size_t>(i)] = getKVCacheWidth(i);

  // Second hook, same shape as the one above: which layers need storage at
  // all. A KV-shared layer (gemma4 `num_kv_shared_layers`) has no wk/wv and
  // no k_norm/v_norm of its own -- it attends over an earlier layer's K/V
  // plane -- so the private slab it gets otherwise is filled with
  // byte-identical values every step. Declaring the source makes those
  // layers aliases of it (same MemoryData, same offset) instead of copies.
  //
  // The map comes from getKVSourceLayer(), which resolves through the SAME
  // function the model's graph builder used to wire the shared attention, so
  // the plane a layer reads is by construction the plane it was given.
  //
  // Declared only when some layer actually shares: a model that inherits the
  // -1 default for every layer never calls setLayerKVSources at all, so it
  // keeps the literal pre-aliasing path through KVCacheManager (empty map =>
  // validate early-returns, both allocation loops are untouched, no witness
  // line) rather than an all-(-1) map that merely behaves the same.
  std::vector<int> kv_sources(static_cast<size_t>(NUM_LAYERS), -1);
  bool any_kv_shared = false;
  for (int i = 0; i < NUM_LAYERS; ++i) {
    kv_sources[static_cast<size_t>(i)] = getKVSourceLayer(i);
    any_kv_shared |= (kv_sources[static_cast<size_t>(i)] >= 0);
  }
  if (any_kv_shared)
    kv_cache.setLayerKVSources(std::move(kv_sources));
  return layout;
}

void CausalLM::allocateKVCacheStorage(const KVCacheLayout &layout) {
  // dtype matches mha_core's cache placeholders so external cache storage
  // is interpreted consistently across platforms.
#ifdef ENABLE_FP16
  const auto cache_dtype = ml::train::TensorDim::DataType::FP16;
#else
  const auto cache_dtype = ml::train::TensorDim::DataType::UINT16;
#endif
  kv_cache.allocate(static_cast<unsigned int>(NUM_LAYERS), BATCH_SIZE,
                    layout.max_timestep, layout.widths, cache_dtype);
}

void CausalLM::load_weight(const std::string &weight_path) {
  // NNTR_KV_EARLY: unset/anything else = allocate on a side thread during the
  // load (default on the OpenCL gpu engine), "sync" = after the load on this
  // thread, "0" = lazily in the first run() as before. Other engines keep the
  // lazy path unless NNTR_KV_EARLY is set: the early build was measured on
  // the gpu engine only. A model whose plane can still grow sizes its
  // KV from the first prompt, so it keeps the lazy path whatever the env says.
  // The CUDA engine always takes "sync": its KV zero fill is a kernel on the
  // stream the weight upload is using, and its context sets environment
  // variables lazily, which a getenv on another thread must not race.
  enum class Mode { lazy, sync, overlap };
  static const Mode mode = []() {
    const char *e = std::getenv("NNTR_KV_EARLY");
    if (e == nullptr && causallm_engine() != "gpu")
      return Mode::lazy;
    if (e != nullptr && e[0] == '0')
      return Mode::lazy;
    if (e != nullptr && std::strcmp(e, "sync") == 0)
      return Mode::sync;
    return Mode::overlap;
  }();
  const bool early = prepare_run_at_load_ && mode != Mode::lazy &&
                     !kv_cache.isAllocated() && !prefillPlaneCanGrow();

  const bool overlap = mode == Mode::overlap && causallm_engine() != "cuda";
  std::future<void> kv_alloc;
  if (early && overlap) {
    // The hooks run here, on this thread, before the load starts; the side
    // thread only allocates and zeroes kv_cache, which nothing in the weight
    // load reads or writes. The future is joined below, or by its destructor
    // if the load throws, so the thread never outlives this call.
    KVCacheLayout layout = planKVCache();
    kv_alloc = std::async(std::launch::async,
                          [this, layout]() { allocateKVCacheStorage(layout); });
  }

  Transformer::load_weight(weight_path);

  if (kv_alloc.valid())
    kv_alloc.get(); // rethrows an allocation failure here, not in run()
  if (early) {
    // Binds the placeholders of the graph that was just loaded (and, in sync
    // mode, allocates first). run() finds it done and skips both.
    allocateAndBindKVCache();
  }
}

void CausalLM::allocateAndBindKVCache() {
  if (!kv_cache.isAllocated()) {
    allocateKVCacheStorage(planKVCache());
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
#ifdef ENABLE_HEXKL
  // Shared CPU layers may also appear in an HTP model. Only this model's
  // inference may enter its resident DSP graph, even with other handles live.
  nntrainer::HtpDecodeScope htp_scope(htp_execution_profile_);
#endif
  // Same contract as NeuralNetwork::incremental_inference(float* ...), except
  // inputs whose raw pointer belongs to a KVCacheManager cache tensor are fed
  // as the REAL tensor (sharing its MemoryData, so isSVM() set by the SVM
  // MemoryPool survives the per-call fillPlaceholder/syncDependents). With
  // in-place input layers the mha_core cache views are dependents of the
  // input placeholder; a fresh Tensor::Map MemoryData of the same pointer
  // would clobber the SVM flag and drop attention to the host path.
  auto *nn = static_cast<nntrainer::NeuralNetwork *>(model.get());

  // Named KV inputs below normally resolve without a pointer lookup. A
  // linear fallback avoids rebuilding a hash table for every decode token;
  // key then value in layer order preserves the old map's first-match rule
  // when two cache views share the same address.
  auto cache_by_ptr = [&](const void *ptr) -> nntrainer::Tensor * {
    if (!kv_cache.isAllocated())
      return nullptr;
    for (unsigned int i = 0; i < kv_cache.getNumLayers(); ++i) {
      auto &kc = kv_cache.getKeyCache(i);
      if (ptr == reinterpret_cast<const void *>(kc.getData()))
        return &kc;
      auto &vc = kv_cache.getValueCache(i);
      if (ptr == reinterpret_cast<const void *>(vc.getData()))
        return &vc;
    }
    return nullptr;
  };

  auto in_dim = nn->getInputDimension();
  NNTR_THROW_IF(input.size() < in_dim.size(), std::invalid_argument)
    << "incrementalInference: model expects " << in_dim.size()
    << " inputs, got " << input.size();

  // The caller hands the cache pointers in NAME order (cache_k_l0, cache_k_l1,
  // cache_k_l10, ...), but the graph orders its inputs by node order. While
  // every cache plane has the same shape the two orders are interchangeable (a
  // consistent permutation of equally-sized slabs); as soon as planes differ --
  // a sliding-window layer that keeps fewer rows than a full-attention one --
  // a positional mismatch is a shape error. Resolve each cache input by the
  // NAME the graph gives that position ("cache_k_l<i>..." /
  // "cache_v_l<i>..."); the pointer match below remains for inputs the graph
  // does not name that way.
  const std::vector<std::string> &in_names = nn->getInputNames();
  auto cache_by_name = [&](const std::string &name) -> nntrainer::Tensor * {
    if (!kv_cache.isAllocated())
      return nullptr;
    const bool is_k = name.rfind("cache_k_l", 0) == 0;
    const bool is_v = name.rfind("cache_v_l", 0) == 0;
    if (!is_k && !is_v)
      return nullptr;
    size_t p = 9, e = p;
    while (e < name.size() && name[e] >= '0' && name[e] <= '9')
      ++e;
    if (e == p || (e < name.size() && name[e] != ':'))
      return nullptr;
    const unsigned long layer = std::stoul(name.substr(p, e - p));
    if (layer >= kv_cache.getNumLayers())
      return nullptr;
    return is_k ? &kv_cache.getKeyCache(static_cast<unsigned int>(layer))
                : &kv_cache.getValueCache(static_cast<unsigned int>(layer));
  };

  nntrainer::sharedConstTensors input_tensors;
  input_tensors.reserve(in_dim.size());
  for (unsigned int idx = 0; idx < in_dim.size(); idx++) {
    if (idx < in_names.size()) {
      if (auto *named = cache_by_name(in_names[idx])) {
        input_tensors.emplace_back(MAKE_SHARED_TENSOR(*named));
        continue;
      }
    }
    if (auto *cached = cache_by_ptr(input[idx])) {
      // shallow copy: shares the cache's MemoryData (isSVM intact)
      input_tensors.emplace_back(MAKE_SHARED_TENSOR(*cached));
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
#if defined(ENABLE_OPENCL)
  // Same rule as the CUDA stash below: a row deferred by the previous call and
  // never claimed names a buffer the caller has since returned to the pool.
  g_cl_pending_logits = ClPendingLogits{};
#endif
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

    // The completed HTP greedy row returned its token ID, so its CPU logits
    // are stale and generate() consumes only the ID. Preserve the pooled
    // buffer's ownership without copying a vocabulary-sized unused row.
    // Sampling, penalties and logit consumers request logits from the DSP;
    // the non-consuming completed-row check is false for all of those paths.
    if (htp_execution_profile_ && batch_size == 1u && to - from == 1u &&
        output_tensors.size() == 1u && buf_size == NUM_VOCAB &&
        out->getDataType() == ml::train::TensorDim::DataType::FP32 &&
        htpDecodeHasTokenId(from)) {
      output.push_back(last_out_buf_data);
      continue;
    }

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
#if defined(ENABLE_OPENCL)
      // The logits chain stood down for this row under the greedy hint, so
      // there is nothing here to widen yet. Record the ingredients instead --
      // generate() either takes the on-GPU token and drops the row, or fills it
      // through materialize_cl_pending_logits(). Keyed on the row address, so
      // it can only ever match the buffer that chain declined to fill.
      bool defer_cl_logits = false;
      if (batch_size == 1 && nntrainer::cl_lmhead_logits_deferred(out_src)) {
        g_cl_pending_logits = {out_src, last_out_buf_data, buf_size};
        defer_cl_logits = true;
      } else {
        // Safety net, and the reason a deferral can never corrupt a token: if
        // the GEMV left a row on the device that this output is NOT, the
        // widening below would read the previous step's numbers. Pay the
        // readback now instead. Costs exactly what the old code always paid.
        nntrainer::cl_lmhead_materialize_logits();
      }
#endif
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
      if (!defer_host_logits)
#endif
#if defined(ENABLE_OPENCL)
        if (!defer_cl_logits)
#endif
          nntrainer::getComputeOps()->scopy_fp16_to_fp32(buf_size, out_src, 1,
                                                         last_out_buf_data, 1);
#else
      throw std::invalid_argument("Error: enable-fp16 is not set");
#endif
    } else if (out->getDataType() == ml::train::TensorDim::DataType::FP32) {
#if defined(ENABLE_OPENCL)
      // Same safety net as the fp16 branch: this path copies the row wholesale,
      // so anything the GEMV deferred has to land first.
      nntrainer::cl_lmhead_materialize_logits();
#endif
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
  if (add_skip_prefill) {
    lmhead_prop.emplace_back(withKey("skip_prefill", "true"));
    // A skip-prefill head computes one position per step, so every row of its
    // output above the first is allocated and never written. At a
    // vocabulary-sized unit that dead plane is hundreds of megabytes
    // (gemma4: 1024 x 262144 fp16 = 512 MB), and it is not merely wasted
    // space: the caller reads the whole output through getDim(), so the
    // per-step fp16->fp32 logits copy in NeuralNetwork::incremental_inference
    // walks 268M elements instead of 262144 -- measured 53 ms per decoded
    // token and 355 ms of prefill on the Intel Xe3 gemma4 1K cell. The FC
    // layers honour this by planning the output at height 1
    // (props::PlanLastRowOnly, fc_layer.cpp / fc_layer_cl.cpp); without the
    // declaration the property is inert and the plane comes back.
    lmhead_prop.emplace_back(withKey("plan_last_row_only", "true"));
  }
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

std::string CausalLM::kvCacheModelTag() const {
  char buf[64];
  // Short and fixed-width on purpose: it has to fit the file header's tag
  // field, and it has to name everything a KV plane's interpretation depends
  // on.
  std::snprintf(buf, sizeof(buf), "L%dq%dh%dd%dv%u%s%s", NUM_LAYERS,
                NUM_KEY_VALUE_HEADS, HEAD_DIM, DIM, NUM_VOCAB,
                MODEL_TENSOR_TYPE.empty() ? "" : "-",
                MODEL_TENSOR_TYPE.c_str());
  return std::string(buf);
}

void CausalLM::compactKVCache(unsigned int pos, bool force) {
  if (!kv_cache.compact(pos) && !force)
    return;
  // ponytail: the n-th HTP attention layer in graph order is attention
  // layer n (the fused model builds them in that order); a model that shares
  // or reorders them needs the layer's own index instead.
  unsigned int n = 0;
  std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &, void *)>
    fn = [&](ml::train::Layer &l, nntrainer::RunLayerContext &, void *) {
      if (l.getType() == causallm::HtpMHACoreLayer::type &&
          n < kv_cache.getNumLayers())
        l.setProperty({"cache_base=" + std::to_string(kv_cache.getBase(n++))});
    };
  model->forEachLayer(fn, nullptr);
}

void CausalLM::save_kvcache(std::string path, int to_) {
  if (!kv_cache.isAllocated()) {
    throw std::runtime_error(
      "save_kvcache called before allocateAndBindKVCache()");
  }
  kv_cache.save(path, static_cast<unsigned int>(to_), kvCacheModelTag());
  // the int8 attention's scales of this prefill, for the run that loads it
  std::ofstream q2(path + ".q2", std::ios::binary);
  causallm::HtpMHACoreLayer::saveQ2Calibration(q2);
  if (!q2)
    throw std::runtime_error("save_kvcache: cannot write " + path + ".q2");
}

/** Retire a captured decode graph. A capture describes the host decisions taken
 *  at the key count it recorded at; putting a different sequence under it -- a
 *  restored cache, or a session moved to another absolute position -- leaves it
 *  describing a prefix that is no longer there, and the recorded key-count
 * deltas will happily certify it because every question in that table is of the
 * form "is the count still on this side of that bound", which stays true.
 *
 *  MEASURED, not defensive. Gemma-4 on a discrete part, 1K prompt split at 400
 *  tokens, 64 tokens generated: continuing in memory from position 400 produced
 *  28 tokens and an early EOS where the uninterrupted run and the
 *  same resume through a FILE both produced the same 64 tokens. The file path
 * differed only in calling this; with NNTR_CUDA_DECODE_GRAPH=0 both agreed. So
 * a reposition has to retire the capture as much as a reload does. */
static inline void retire_decode_capture() {
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
  nntrainer::cuda::kv_regime_invalidate();
#endif
}

void CausalLM::setSessionTokenPosition(unsigned int pos) {
  USE_KVCACHE = false;
  PRE_COMPUTED_CACHE_PATH.clear();
  SYS_PROMP_LEN = 0;
  global_token_len = pos;
  retire_decode_capture();
}

void CausalLM::setPrecomputedKVCache(const std::string &path,
                                     unsigned int sys_prompt_token_len) {
  USE_KVCACHE = !path.empty();
  PRE_COMPUTED_CACHE_PATH = path;
  SYS_PROMP_LEN = USE_KVCACHE ? sys_prompt_token_len : 0;
  global_token_len = 0;
  retire_decode_capture();
}

void CausalLM::load_kvcache(std::string path, int to_) {
  if (!kv_cache.isAllocated()) {
    allocateAndBindKVCache();
  }
  const unsigned int loaded =
    kv_cache.load(path, static_cast<unsigned int>(to_), kvCacheModelTag());
  // absent (a dump of an older build, a CPU run): the layers calibrate on
  // their first step as before
  if (std::ifstream q2(path + ".q2", std::ios::binary); q2)
    causallm::HtpMHACoreLayer::loadQ2Calibration(q2);
  // mha_core layers each track their own cache_index; sync them all to the
  // newly-loaded position so the next forwarding() writes at the right slot.
  setKVCachePosition(loaded);
  // a windowed layer's rows start at its base now
  compactKVCache(loaded, true);
  retire_decode_capture();
}

unsigned int CausalLM::resumeFromKVCacheFile(const std::string &path,
                                             unsigned int token_len) {
  if (!kv_cache.isAllocated()) {
    allocateAndBindKVCache();
  }
  const unsigned int loaded = kv_cache.load(path, token_len, kvCacheModelTag());
  setKVCachePosition(loaded);
  retire_decode_capture();
  setSessionTokenPosition(loaded);
  return loaded;
}

std::vector<unsigned int> CausalLM::generate(float *logits, bool do_sample,
                                             float repetition_penalty,
                                             unsigned int *input_ids,
                                             unsigned int NUM_INPUT_IDS) {
  if (htp_execution_profile_ && !do_sample && BATCH_SIZE == 1 &&
      repetition_penalty == 1 && logits_processor == nullptr) {
    unsigned int id = 0;
    if (htpDecodeTokenId(&id))
      return {id};
  }

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
  // Terminal drain for the selective-sync (NNTR_CUDA_ASYNC) decode path: ensure
  // the GPU has finished producing the logits before the host reads them here.
  // Called unconditionally; under drained (non-async) submission every GPU op
  // already drained per-op, so it has nothing left to wait for. cuda engine
  // ONLY: in a dual-enabled (CUDA+OpenCL) binary this ran on OpenCL runs too,
  // and StreamManager::Global() lazily CREATES the CUDA context -- the first
  // stray CUDA touch on a non-cuda run (NVIDIA VRAM burned on an Intel run).
  if (causallm_engine() == "cuda")
    nntrainer::cuda::StreamManager::Global().finish();
#endif

  std::vector<unsigned int> outputs;
  // Sampled decoding whose top-k fits the device selection: the token depends
  // only on the TOP_K best logits, so the GPU lanes select candidates on the
  // device and the host samples from those. More candidates than TOP_K are
  // asked for: fp16 logits tie often, and the sampler can only settle the cut
  // when the candidates reach past the tie at rank TOP_K (see
  // sampleTokenFromTopCandidates). Same exclusions as the greedy device path:
  // anything that rewrites the row on the host keeps the row.
  constexpr unsigned int kDeviceTopKMax = 64;
  [[maybe_unused]] const bool topk_shape =
    do_sample && logits_processor == nullptr && BATCH_SIZE == 1 && TOP_K > 0 &&
    TOP_K <= kDeviceTopKMax && TOP_K < NUM_VOCAB &&
    sample_device_topk_enabled();
  [[maybe_unused]] const unsigned int topk_want = 2 * TOP_K + 32;
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
  // Repetition penalty / bad-words activity, with EXACTLY the host-path
  // predicates below (generate() applies each only under its own triple
  // check, so the device route must mirror both or it would penalize where
  // the host would not).
  const bool rp_active =
    repetition_penalty != 1 && input_ids != nullptr && NUM_INPUT_IDS != 0;
  const bool bw_active = BAD_WORD_IDS.size() != 0 && NUM_BADWORDS != 0;
  // The penalized device argmax replicates the host math bit-for-bit
  // (cuda_argmax_penalized_fp16) but its id scratch is capped; oversize
  // windows keep the host path. fp16-ness of the logits is only known from
  // the stash inside the loop, so it is checked there, not here (a wrong
  // guess by this hint costs one deferred conversion, never a wrong token).
  const bool dev_penalty_fits =
    (!rp_active || NUM_INPUT_IDS <= 512) && (!bw_active || NUM_BADWORDS <= 64);
  // Tell the NEXT incrementalInference whether the full-vocab host row is
  // going to be read at all. Recomputed every call, so a run that switches
  // sampling on pays one deferred conversion and then stops deferring.
  const bool cuda_topk = topk_shape &&
                         topk_want <= nntrainer::cuda::CUDA_TOPK_MAX &&
                         !rp_active && !bw_active;
  g_greedy_hint = cuda_argmax_enabled() && logits_processor == nullptr &&
                  BATCH_SIZE == 1 &&
                  ((do_sample == false && dev_penalty_fits) || cuda_topk);
#endif
#if defined(ENABLE_OPENCL)
  // Same predicates the host path below applies, so the device route can never
  // skip a penalty the host would have applied.
  const bool cl_rp_active =
    repetition_penalty != 1 && input_ids != nullptr && NUM_INPUT_IDS != 0;
  const bool cl_bw_active = BAD_WORD_IDS.size() != 0 && NUM_BADWORDS != 0;
  // Tell the NEXT lm_head GEMV whether the full-vocabulary host row is going to
  // be read at all. Recomputed every call, so a run that switches sampling on
  // pays one deferred readback and then stops deferring. Unlike the CUDA lane
  // there is no on-device penalty kernel here, so any penalty keeps the host
  // row. Guarded on the engine: on a cuda or cpu run this arms nothing.
  const bool cl_topk = topk_shape &&
                       topk_want <= nntrainer::CL_LMHEAD_TOPK_MAX &&
                       !cl_rp_active && !cl_bw_active;
  const bool cl_greedy = causallm_engine() == "gpu" && do_sample == false &&
                         logits_processor == nullptr && BATCH_SIZE == 1 &&
                         !cl_rp_active && !cl_bw_active;
  nntrainer::cl_lmhead_set_greedy_hint(cl_greedy ||
                                       (causallm_engine() == "gpu" && cl_topk));
#endif
  for (unsigned int iteration = 0; iteration < BATCH_SIZE; ++iteration) {
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
    // CUDA on-GPU top-k for sampled decoding: select the best candidates on
    // the GPU (after the decode graph replay, like the argmax below -- the
    // selection is not captured), read back their (id, logit) pairs and draw
    // from them on the host. The logits here are the device row the host path
    // would widen (softcap included, it runs on the device on this lane), so
    // the draw is the full-row draw; when the candidates cannot prove it the
    // row is read as before and nothing of the RNG has been used.
    if (cuda_topk && cuda_argmax_enabled() && g_cuda_logits_dev != nullptr) {
      unsigned int cand_ids[nntrainer::cuda::CUDA_TOPK_MAX];
      float cand_logits[nntrainer::cuda::CUDA_TOPK_MAX];
      unsigned int n = 0;
      const bool ok =
        g_cuda_logits_fp16
          ? nntrainer::cuda::cuda_topk_fp16(
              reinterpret_cast<const unsigned short *>(g_cuda_logits_dev),
              NUM_VOCAB, topk_want, cand_ids, cand_logits, &n)
          : nntrainer::cuda::cuda_topk_fp32(
              reinterpret_cast<const float *>(g_cuda_logits_dev), NUM_VOCAB,
              topk_want, cand_ids, cand_logits, &n);
      unsigned int tok = 0;
      if (ok &&
          sampleTokenFromTopCandidates(cand_ids, cand_logits, n, TEMPERATURE,
                                       TOP_K, TOP_P, rng, &tok)) {
        g_cuda_logits_dev = nullptr;
        // The deferred host row was never needed -- drop it unfilled.
        g_pending_logits = PendingLogits{};
        // Same early page-warm of the next token's rows as the argmax path.
        nntrainer::cuda::emb_gather_notify_token(tok);
        outputs.push_back(tok);
        logits = logits + NUM_VOCAB;
        if (input_ids != nullptr)
          input_ids = input_ids + MAX_SEQ_LEN;
        continue;
      }
      // else: fall through; the host path below materializes the row.
    }
    // CUDA on-GPU greedy argmax: reduce the device-resident
    // lm_head logits to the token id on the GPU and read back only 4 bytes,
    // skipping the host std::max_element over the full-vocab buffer. Gated to
    // greedy (no sampling, no logits processor -- those consume logits on the
    // host) and only when incrementalInference stashed a device-accessible
    // logits pointer for this (single, BATCH_SIZE==1) row. Repetition penalty
    // and bad words no longer force the host path: for fp16 logits they run
    // on-device through cuda_argmax_penalized_fp16, which replays the host
    // penalty loops bit-for-bit in fp32 (see its contract) -- only fp32
    // logits or over-cap id counts still fall back.
    if (cuda_argmax_enabled() && g_cuda_logits_dev != nullptr &&
        do_sample == false && logits_processor == nullptr &&
        ((!rp_active && !bw_active) ||
         (g_cuda_logits_fp16 && dev_penalty_fits))) {
      unsigned int tok = 0;
      bool ok;
      if (rp_active || bw_active) {
        // Window ids walk batch rows like the host path (input_ids is
        // advanced by MAX_SEQ_LEN per row below); BATCH_SIZE==1 here.
        ok = nntrainer::cuda::cuda_argmax_penalized_fp16(
          reinterpret_cast<const unsigned short *>(g_cuda_logits_dev),
          NUM_VOCAB, rp_active ? input_ids : nullptr,
          rp_active ? NUM_INPUT_IDS : 0,
          bw_active ? BAD_WORD_IDS.data() : nullptr,
          bw_active ? NUM_BADWORDS : 0, repetition_penalty, &tok);
      } else {
        ok = g_cuda_logits_fp16
               ? nntrainer::cuda::cuda_argmax_fp16(
                   reinterpret_cast<const unsigned short *>(g_cuda_logits_dev),
                   NUM_VOCAB, &tok)
               : nntrainer::cuda::cuda_argmax_fp32(
                   reinterpret_cast<const float *>(g_cuda_logits_dev),
                   NUM_VOCAB, &tok);
      }
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

#if defined(ENABLE_OPENCL)
    // OpenCL on-GPU top-k for sampled decoding: the same idea as the greedy
    // reduction below, for the best few dozen logits instead of one. A
    // deferred softcap is applied to just those values on the host by the
    // layer's own code, so the draw is the full-row draw; when the candidates
    // cannot prove it the row is materialized below and the RNG is untouched.
    if (cl_topk && g_cl_pending_logits.host != nullptr &&
        g_cl_pending_logits.host == logits) {
      unsigned int cand_ids[nntrainer::CL_LMHEAD_TOPK_MAX];
      float cand_logits[nntrainer::CL_LMHEAD_TOPK_MAX];
      unsigned int n = 0;
      unsigned int tok = 0;
      if (nntrainer::cl_lmhead_dev_topk(NUM_VOCAB, topk_want, cand_ids,
                                        cand_logits, &n) &&
          sampleTokenFromTopCandidates(cand_ids, cand_logits, n, TEMPERATURE,
                                       TOP_K, TOP_P, rng, &tok)) {
        // The deferred row was never needed -- drop it unfilled.
        nntrainer::cl_lmhead_release_logits();
        g_cl_pending_logits = ClPendingLogits{};
        outputs.push_back(tok);
        logits = logits + NUM_VOCAB;
        if (input_ids != nullptr)
          input_ids = input_ids + MAX_SEQ_LEN;
        continue;
      }
    }
    // OpenCL on-GPU greedy argmax: reduce the device-resident lm_head row to a
    // token id on the GPU and read back 4 bytes, skipping the softcapping pass
    // over the vocabulary, the fp16 -> fp32 widening and the max_element. The
    // gate is the deferral itself -- the GEMV only defers under the greedy
    // hint -- plus this call's own predicates, re-checked because the hint was
    // armed one step earlier. Ties resolve to the lowest index exactly as
    // std::max_element does, so the token is bit-identical to the host path.
    if (g_cl_pending_logits.host != nullptr &&
        g_cl_pending_logits.host == logits && do_sample == false &&
        logits_processor == nullptr && !cl_rp_active && !cl_bw_active) {
      unsigned int tok = 0;
      if (nntrainer::cl_lmhead_dev_argmax(NUM_VOCAB, &tok)) {
        // The deferred host row was never needed -- drop it unfilled.
        g_cl_pending_logits = ClPendingLogits{};
        outputs.push_back(tok);
        logits = logits + NUM_VOCAB;
        if (input_ids != nullptr)
          input_ids = input_ids + MAX_SEQ_LEN;
        continue;
      }
      // else: the reduction refused this row; fall through to the host path,
      // which materializes it just below.
    }
    // Reached only when the device path did not take this row. Either way the
    // host row may still be outstanding, so fill it before anyone reads
    // `logits`.
    materialize_cl_pending_logits();
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
  /** The peak belongs to THIS request, so the second message does not inherit
   *  the first one's high-water. arm() rather than a reset because a caller
   *  that lazily loaded the model just before this call has already started
   *  measuring, and that load is part of the same request. */
  FootprintSampler::get().arm();
#if defined(ENABLE_OPENCL)
  /** Load is finished here and no forward has run: everything in the ledger at
   *  this point is weights, planes and mirrors, i.e. the part of the footprint
   *  that a prompt cannot change. */
  nntrainer::opencl::clMemAcctDump("after-load");
#endif
  if (!is_initialized) {
    throw std::runtime_error("CausalLM model is not initialized. Please call "
                             "initialize() before run().");
  }

  struct StreamerEndGuard {
    BaseStreamer *streamer;
    ~StreamerEndGuard() { streamer_end(streamer); }
  } streamer_end_guard{streamer_};

  // [prefill-plane] The KV cache allocation below reads the plane (the ring
  // capacity is a multiple of the prefill chunk, which is clamped to the
  // plane), so the plane has to be settled BEFORE it -- and the plane follows
  // this request's prompt, which is only known once the prompt is tokenized.
  // Both therefore moved past the Encode() below; nothing between here and
  // there touches the cache. The call stays idempotent, so a second turn
  // reuses the buffers exactly as before.
  has_run_ = false;
  prepareStopRequestForRun();
  // Same seed at the top of every run, so run N samples exactly what run 1
  // (or a fresh process) would. A caller seed from setSamplingSeed() wins.
  beginSamplingRun();

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

  // Record generated IDs without requesting logits or forcing a continuation.
  // This lets performance runs verify the normal DSP token-id path as well.
  const char *generated_ids_path = std::getenv("NNTR_GENERATED_IDS");
  std::vector<unsigned int> generated_ids;

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

  // the length the SAVE run prefilled: its Encode added the special tokens
  if (USE_KVCACHE && !SAVE_KVCACHE && SYS_PROMP_LEN == 0)
    SYS_PROMP_LEN = tokenizer->Encode(system_prompt, true).size();

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
  /// A prompt continued from a loaded cache is the middle of a sequence:
  /// a tokenizer that prepends a BOS (LFM2's <|startoftext|>) must not put
  /// one there. Byte-identical for a tokenizer that adds none (Gemma4).
  auto _input = tokenizer->Encode(
    prompt_, /*add_special_tokens=*/!(USE_KVCACHE && !SAVE_KVCACHE));

  // [prefill-plane] Now that the prompt's real token count is known, give this
  // request the activation plane (== prefill block) it can use, and no more: a
  // short prompt keeps the pack's plane and the pack's footprint, a long one
  // gets the validated 4096-row block and its prefill throughput. A no-op on
  // every later turn (the plane only ever grows, and it grows before the cache
  // exists) and on every model that does not chunk by design. See
  // Transformer::growPrefillPlane().
  if (!kv_cache.isAllocated())
    growPrefillPlane(static_cast<unsigned int>(_input.size()));

  // Allocate the host-owned KV cache and bind it to mha_core's external cache
  // input slots. Idempotent: only the first call does work; subsequent runs
  // reuse the same buffers and continue from the computed absolute token
  // position below. Sized from the plane settled just above.
  allocateAndBindKVCache();

  // | <------------------- MAX_SEQ_LEN -------------------> |
  //                       ||             ||
  // |<-- System prompt -->||<-- input -->||<-- generate -->|

  std::vector<int64_t> init_input;
  unsigned int _len = _input.size();
  // The prefill activation buffers are built at INIT_SEQ_LEN (transformer.cpp
  // constructModel, {1,1,1,INIT_SEQ_LEN}); resetInputDimension is disabled, so
  // ONE forward pass cannot process more than INIT_SEQ_LEN query rows without
  // overflowing the shared activation tensor (getSharedDataTensor bounds-check
  // throw). Normally that caps the prompt to INIT_SEQ_LEN. With chunking
  // (prefillChunk() > 0) the prefill is fed FORWARD in chunks of <=
  // INIT_SEQ_LEN rows -- each chunk fits the buffer -- so the prompt is bounded
  // by the KV budget alone and the activation plane stays INIT_SEQ_LEN-sized
  // however long the prompt is. That decoupling is the point of chunking:
  // INIT_SEQ_LEN is the activation/chunk height (small), MAX_SEQ_LEN the KV
  // capacity (large).
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
  // Whether the prefill is chunked is a property of the model, not of the
  // environment: the KV ring requests chunking on its own (prefillChunk() > 0)
  // even when NNTR_PREFILL_CHUNK is unset. Keying this off
  // getenv("NNTR_PREFILL_CHUNK") meant that chunking never opened the budget,
  // so a 29K-token prompt was cut to INIT_SEQ_LEN even though the prefill
  // could have fed it in chunks.
  const bool _prefill_chunking = prefillDriveChunk() > 0;
  unsigned int num_allow_str =
    _prefill_chunking ? kv_budget
                      : std::min<unsigned int>(INIT_SEQ_LEN, kv_budget);
  unsigned int text_len = _len;

  performance_metrics.prompt_truncated_tokens = 0;
  if (_len > num_allow_str) {
    text_len = num_allow_str;
    performance_metrics.prompt_truncated_tokens = _len - num_allow_str;
    // Silent tail truncation loses whatever the prompt ENDS with (round-13
    // field case: a summarization instruction at the tail was dropped and
    // the model continued the body instead). Unexpected state -> always warn.
    // The named limit follows the budget actually applied: without an explicit
    // cap it is max_seq_len - 1, not max_seq_len - num_to_generate.
    std::fprintf(
      stderr,
      "[causallm] WARNING: prompt (%u tokens) exceeds the prefill window "
      "(init_seq_len=%u%s, %s=%u); truncating %u "
      "tail tokens. Raise %s to fit the prompt.\n",
      _len, static_cast<unsigned int>(INIT_SEQ_LEN),
      _prefill_chunking ? " (chunked: not the prompt limit)" : "",
      unlimited_generation ? "max_seq_len-1 (generating until EOS)"
                           : "max_seq_len-num_to_generate",
      kv_budget, _len - num_allow_str,
      _prefill_chunking ? "max_seq_len" : "init_seq_len");
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
    // a prompt past the activation plane goes in plane-height chunks, a
    // windowed layer's window moved to the front before each
    const unsigned int save_chunk = prefillDriveChunk();
    for (unsigned int o = 0, step = 0; o < input_len; o += step) {
      step = save_chunk ? std::min(save_chunk, input_len - o) : input_len - o;
      for (unsigned int b = 0; b < BATCH_SIZE; ++b)
        for (unsigned int j = 0; j < step; ++j)
          input_sample[static_cast<size_t>(b) * MAX_SEQ_LEN + j] =
            static_cast<float>(init_input[o + j]);
      compactKVCache(o);
      for (auto &so : output)
        releaseLogitsBuf(so);
      output = incrementalInference(BATCH_SIZE, input, step, o, o + step);
    }

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
    const auto t0 = std::chrono::steady_clock::now();
    load_kvcache(PRE_COMPUTED_CACHE_PATH, SYS_PROMP_LEN);
    std::cerr << "[TIMING] kv_load_ms="
              << std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - t0)
                   .count()
              << " rows=" << SYS_PROMP_LEN << std::endl;
  } else {
    SYS_PROMP_LEN = 0;
  }
  allocateAndBindKVCache();
  const unsigned int prefill_from = SYS_PROMP_LEN + global_token_len;
  std::vector<unsigned int> id_list;

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
  // [lmhead-tie-lut] Re-tie the untied lm_head onto the embedding sidecar LUT.
  // The LUT is fetched through the same path-keyed cache the embedding layer
  // used at finalize (a pure cache hit -- same QuantLut, same mmap) and is
  // validated against the contract the tied GEMV assumes: sfixed4 rows, ONE
  // fp32 scale per row, [vocab x hidden] geometry, and a 1.0 runtime embedding
  // scale (the sqrt(hidden) fold lives in the LUT content, which is exactly
  // what the kernel's 1/sqrt(K) compensates). Anything off leaves the head on
  // the fp-activation QS4CX route, unchanged.
  if (LMHEAD_TIE_LUT && causallm_engine() == "cuda" &&
      !EMBEDDING_FILE_NAME.empty()) {
    std::shared_ptr<QuantLut> tie_lut;
    try {
      tie_lut = get_or_load_quant_lut(EMBEDDING_FILE_NAME, NUM_VOCAB, DIM);
    } catch (const std::exception &e) {
      ml_logw("[lmhead-tie-lut] embedding LUT load failed (%s)", e.what());
    }
    if (tie_lut &&
        !(tie_lut->is_signed4 && !tie_lut->is_raw_u16 &&
          tie_lut->ggml_dtype == nntrainer::TensorDim::DataType::NONE &&
          tie_lut->sfixed4_blocks == 1 &&
          tie_lut->in_dim == (size_t)NUM_VOCAB &&
          tie_lut->out_dim == (size_t)DIM && EMBEDDING_SCALE == 1.0f)) {
      ml_logw("[lmhead-tie-lut] embedding LUT does not satisfy the tied-head "
              "contract (sfixed4 per-row-symmetric [vocab x hidden], "
              "embedding scale 1.0)");
      tie_lut.reset();
    }
    if (tie_lut) {
      std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &,
                         void *)>
        fn = [&tie_lut](ml::train::Layer &l, nntrainer::RunLayerContext &ctx,
                        void *) {
          if (l.getType() != "fully_connected" ||
              l.getName() != "output_of_causallm")
            return;
          for (unsigned int w = 0; w < ctx.getNumWeights(); ++w) {
            auto &wt = ctx.getWeight(w);
            // QINT4 never appears here: layer_context coerces it to QS4CX.
            if (wt.getDataType() != ml::train::TensorDim::DataType::QS4CX)
              continue;
            if (wt.width() != (unsigned)tie_lut->in_dim ||
                wt.height() != (unsigned)tie_lut->out_dim)
              continue;
            // Registers the LUT as the head's weight and builds nothing else
            // for it: the upload inside register() replaces the dp4a cache the
            // fp-act route would have built.
            if (nntrainer::cuda::cuda_fc_lmhead_tie_lut_register(
                  wt.getData<uint8_t>(), tie_lut->data(),
                  tie_lut->payload_size(), tie_lut->row_scales.data(),
                  tie_lut->row_scales.size(), wt.width(), wt.height())) {
#if !defined(_WIN32)
              // The upload just read the ENTIRE payload mmap, faulting all of
              // its file pages resident, where demand paging kept only the
              // touched rows. Drop the PTEs now that the device holds a copy:
              // file-backed MADV_DONTNEED is non-destructive (pages re-fault
              // from the page cache), so the host prefill dequant that still
              // reads this mmap keeps working and re-faults only its working
              // set.
              if (tie_lut->mmap_ptr != nullptr)
                ::madvise(tie_lut->mmap_ptr, tie_lut->mmap_len, MADV_DONTNEED);
#endif
            }
          }
        };
      model->forEachLayer(fn, nullptr);
    }
    // The flag promised a tied head but nothing registered (no untied FC head
    // reached the walk, geometry mismatch, or the device upload failed): say so
    // once -- a decode that silently keeps the fp-act route is otherwise
    // indistinguishable from the tie working.
    if (!nntrainer::cuda::cuda_fc_lmhead_tie_lut_ready())
      ml_logw("[lmhead-tie-lut] requested but not activated -- lm_head keeps "
              "the fp-activation QS4CX route");
  }

  // Per-weight CUDA prewarm walk, gated by NNTR_CUDA_PREWARM (on unless it
  // starts with '0'; unset means on). Two things depend on it:
  //   - correctness: without it every QS4CX weight's derived device caches (the
  //     dp4a int4 repack, the fp16/fp32 per-channel scale copies) are built
  //     LAZILY inside the first forward that uses the weight. Those builds
  //     cudaMalloc and free while the stream is running under async submission,
  //     so the first big-K FC of the prefill can consume a buffer another
  //     allocation has already moved. Building them all here, before any
  //     forward, removes the mid-stream allocation entirely.
  //   - cost: the one-time plain -> packed int4 repack is a large share of the
  //     cold-run GPU time; doing it once here keeps it off the first prefill.
  // Idempotent (each cache is keyed by the weight pointer), so a second call
  // and the lazy in-path build are both no-ops afterwards.
  {
    // The guard is a MEMBER (cuda_prewarmed_), one per model object -- not a
    // function-local static. A static was process-lifetime: the second load of
    // a load/destroy loop skipped the prewarm entirely, and under a device-only
    // activation pool the lazy in-path rebuild that was left to do the work is
    // exactly the mid-stream cudaMalloc this walk exists to remove.
    static const char *_pw = std::getenv("NNTR_CUDA_PREWARM");
    static const bool cuda_prewarm_on = !(_pw && _pw[0] == '0');
    // cuda engine ONLY: a dual-enabled (CUDA + OpenCL) binary would otherwise
    // run the whole prewarm on OpenCL runs too, allocating every FC's derived
    // cache on the NVIDIA device.
    if (!cuda_prewarmed_ && cuda_prewarm_on && causallm_engine() == "cuda") {
      cuda_prewarmed_ = true;
      std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &,
                         void *)>
        fn = [](ml::train::Layer &l, nntrainer::RunLayerContext &ctx, void *) {
          if (l.getType() != "fully_connected")
            return;
          for (unsigned int w = 0; w < ctx.getNumWeights(); ++w) {
            nntrainer::Tensor &wt = ctx.getWeight(w);
            // QINT4 never appears here: layer_context coerces it to QS4CX.
            if (wt.getDataType() != ml::train::TensorDim::DataType::QS4CX)
              continue;
            const unsigned short *uS = nullptr;
            nntrainer::cuda::cuda_fc_qs4cx_scales_to_uvm_fp16(
              wt.getScale<float>(), wt.width(), &uS);
            // Same reason, fp32 copy, for the one weight whose decode takes the
            // fp-activation lm_head GEMV: that route keeps the scale in full
            // precision.
            if (wt.width() >= nntrainer::cuda::CUDA_FC_FPACT_MIN_N) {
              const float *uS32 = nullptr;
              nntrainer::cuda::cuda_fc_qs4cx_scales_to_uvm_fp32(
                wt.getScale<float>(), wt.width(), &uS32);
            }
            // A skip_prefill FC never sees prefill M>1 and the untied lm_head
            // decodes at M=1, so neither can reach the M>=32 cuBLAS-i8 gate:
            // their [K,N] int8 cache is dead VRAM. Exempt them from the EAGER
            // build; the lazy runtime build remains as the self-healing
            // fallback.
            bool i8_dead = l.getName() == "output_of_causallm";
            if (!i8_dead) {
              try {
                i8_dead = l.getProperty("skip_prefill") == "true";
              } catch (...) {
              }
            }
            if (i8_dead)
              nntrainer::cuda::cuda_fc_qs4cx_prewarm_exempt_i8(
                wt.getData<uint8_t>());
            nntrainer::cuda::cuda_fc_qs4cx_prewarm(wt.getData<uint8_t>(),
                                                   wt.width(), wt.height());
          }
        };
      model->forEachLayer(fn, nullptr);
      // The walk is load-time work, not prefill work: restart the prefill clock
      // so it is not billed to the first forward.
      start_prefill = std::chrono::high_resolution_clock::now();
    }
  }

  // Pre-grow the decode scratch buffers, once, before any timed phase. Both
  // grow-on-demand allocators FREE the previous buffer and cudaMalloc a larger
  // one; under async submission (NNTR_CUDA_ASYNC, the default here) kernels
  // launched earlier on the stream still reference the OLD pointer, so the
  // first decode step that outgrows a prefill-sized scratch silently reads
  // freed memory. Sizing them here for the largest shape the model can ask for
  // removes the regrow entirely -- and it is also what keeps the M=1 path from
  // cudaMalloc'ing inside a CUDA-graph capture.
  if (causallm_engine() == "cuda") {
    // 2*HEAD_DIM covers a model whose global-attention head_dim is wider than
    // its sliding one; the over-allocation is a few hundred KB.
    nntrainer::cuda::cuda_attention_splitkv_prewarm(
      static_cast<int>(MAX_SEQ_LEN), NUM_HEADS, 2 * HEAD_DIM);
    // Decode is M=1. K (the FC contraction dim) is bounded by
    // max(hidden, FFN intermediate) -- the down projection reads the FFN
    // intermediate activation, so DIM alone under-sizes the activation-quant
    // staging. N is bounded by the largest decode FC output = max(vocab, FFN
    // intermediate).
    nntrainer::cuda::cuda_fc_qs4cx_dp4a_prewarm(
      1u,
      std::max(static_cast<unsigned int>(DIM),
               static_cast<unsigned int>(INTERMEDIATE_SIZE)),
      std::max(NUM_VOCAB, static_cast<unsigned int>(INTERMEDIATE_SIZE)));
  }
#endif

  // [resume-block] A resumed prefill (multi-turn continuation or a restored
  // KV cache, prefill_from > 0) runs as ONE block call — same shape as the
  // first prefill but with `from` at the absolute position — now that the
  // shared llm activation layers accept from>0 with to-from>1 (their row
  // math is base+live-window on every backend; attention/mha_core always
  // supported it). NNTR_RESUME_BLOCK=0 restores the legacy token-by-token
  // decode-shape feed (prefill at decode TPS) as an escape hatch.
  static const char *_rb_env = std::getenv("NNTR_RESUME_BLOCK");
  static const bool resume_block_on = !(_rb_env && _rb_env[0] == '0');

  // [prefill-chunk] A chunk C (>0) drives the prefill FORWARD in
  // C-token chunks instead of one M-token block. Each chunk feeds its tokens at
  // input row 0 and writes its KV at the absolute range [from, from+clen); the
  // next chunk attends over the accumulated cache. For causal (and
  // sliding-window) attention every query sees the identical causal prefix
  // either way, so the result is bit-identical to the single block. This is the
  // exact token-feed pattern the legacy resumed-prefill loop below already uses
  // (tokens at input row 0, absolute KV position via `from`), generalized from
  // one token per call to C. C=0 (no ring, no override) is the single-block
  // path verbatim.
  // prefillChunk() is the one clamped answer -- the same number the prompt
  // budget above and the KV ring capacity are computed from, so a chunk always
  // fits the INIT_SEQ_LEN-height activation buffer and never straddles the
  // ring's wrap seam.
  // prefillDriveChunk() == prefillChunk() whenever the ring picked a chunk; it
  // only adds plane-height chunking for a chunk-by-design model whose ring was
  // refused (see transformer.h).
  const unsigned int prefill_chunk = prefillDriveChunk();
  // [kv-window-ring] A ringed layer stores absolute position p at row
  // (p % Wcap) and takes ONE contiguous slice per call, so every prefill call
  // [p, p+L) must satisfy (p % Wcap) + L <= Wcap. Wcap is a multiple of the
  // chunk C, so it is enough that no call crosses an ABSOLUTE multiple of C.
  // A first turn starts at 0 and the plain C-tiling already has that property;
  // a resumed session (a second prompt, a loaded KV file, a precomputed system
  // prompt) starts at an arbitrary residue, and tiling from there put a slice
  // across the seam (mha_core's guard threw: 8.5K prompt resumed at 3269 on a
  // 2048-row ring). Only the producer's phase can uphold the invariant -- no
  // finite Wcap does -- so with a ringed layer the first chunk is shortened to
  // the next absolute multiple of C and every later chunk starts C-aligned.
  // Identical to the plain tiling whenever from_pos % C == 0. Gated on the
  // model actually having a ringed layer: a linear cache has no seam, and
  // re-tiling it would only move fp16 rounding for no reason.
  const unsigned int ring_grid = [&]() -> unsigned int {
    if (prefill_chunk == 0)
      return 0u;
    for (int i = 0; i < NUM_LAYERS; ++i)
      if (getKVCacheRows(i) < static_cast<unsigned int>(MAX_SEQ_LEN))
        return prefill_chunk;
    return 0u;
  }();
  auto do_prefill = [&](unsigned int n_tok,
                        unsigned int from_pos) -> std::vector<float *> {
    // [prefill-long] Tell the attention layers the whole key span this prefill
    // will reach, so the tight-stride V image is laid out once instead of once
    // per chunk (each re-layout re-scatters every V row already written). The
    // guard clears it on every exit, so decode sees 0 = unknown.
    struct SpanHint {
      SpanHint(unsigned int s) { setPrefillSpanHint(s); }
      ~SpanHint() { setPrefillSpanHint(0); }
    } _span_hint(from_pos + n_tok);
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
    // Single block (default) when chunking is off or the prompt fits one chunk.
    const bool one_block_on_grid =
      ring_grid == 0 || (from_pos % ring_grid) + n_tok <= ring_grid;
    if (prefill_chunk == 0 || (n_tok <= prefill_chunk && one_block_on_grid))
      return incrementalInference(BATCH_SIZE, input, n_tok, from_pos,
                                  from_pos + n_tok);
    // Chunked forward prefill.
    std::vector<float *> out;
    unsigned int clen = 0;
    for (unsigned int o = 0; o < n_tok; o += clen) {
      // ring_grid: stop at the next absolute multiple of C (see above).
      const unsigned int room =
        ring_grid ? ring_grid - ((from_pos + o) % ring_grid) : prefill_chunk;
      clen = std::min(std::min(prefill_chunk, room), n_tok - o);
      for (unsigned int b = 0; b < BATCH_SIZE; ++b)
        for (unsigned int j = 0; j < clen; ++j)
          input_sample[static_cast<size_t>(b) * MAX_SEQ_LEN + j] =
            static_cast<float>(init_input[o + j]);
      const unsigned int cf = from_pos + o;
      // a windowed (sliding) layer keeps its window in front of the chunk
      compactKVCache(cf);
      auto so = incrementalInference(BATCH_SIZE, input, clen, cf, cf + clen);
      if (o + clen < n_tok)
        for (auto &oo : so)
          releaseLogitsBuf(oo);
      else
        out = std::move(so);
    }
    return out;
  };

  /** [perf-split] Stamped when a token first reaches the caller; default-
   *  constructed (epoch) means "not yet", i.e. the first token comes out of the
   *  decode loop. See ttft_ms in performance_metrics.h. */
  std::chrono::high_resolution_clock::time_point perf_split_first_out{};

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

    // The bound is ids_history's row stride (MAX_SEQ_LEN), not the prefill
    // block height: a chunked prefill runs prompts longer than INIT_SEQ_LEN.
    if (init_len < MAX_SEQ_LEN) {
      registerOutputs(tokenizer, id_list, init_len, eos_list, log_output);
      /** [perf-split] Without skip_prefill the FIRST token the caller sees
       *  is the one prefill's own logits produced, before the boundary below
       *  -- so TTFT is stamped here, not at the first decode step. (The
       *  measured model packs set skip_prefill, so this arm serves the
       *  others.) */
      perf_split_first_out = std::chrono::high_resolution_clock::now();
    }
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
  if (generated_ids_path)
    generated_ids.insert(generated_ids.end(), id_list.begin(), id_list.end());

#ifdef ENABLE_HEXKL
  // The resident decode graph only needs a row of the CPU prefill scratch.
  // Outputs have been consumed above; KV placeholders and persistent tensors
  // retain their pages. Keep this opt-in while measuring the memory profile.
  // Built in for the HTP profile (htp_fixed::RELEASE_PREFILL_SCRATCH);
  // NNTR_HTP_RELEASE_PREFILL_SCRATCH=0 keeps the scratch.
  const char *release = std::getenv("NNTR_HTP_RELEASE_PREFILL_SCRATCH");
  if (release != nullptr || (htp_execution_profile_ &&
                             nntrainer::htp_fixed::RELEASE_PREFILL_SCRATCH)) {
    if (release == nullptr || std::strcmp(release, "1") == 0) {
      if (auto *network =
            dynamic_cast<nntrainer::NeuralNetwork *>(model.get())) {
        const size_t bytes = network->releaseInferenceScratch();
        std::cerr << "[HTP] CPU prefill scratch discarded: " << (bytes >> 20)
                  << " MiB of virtual pages" << std::endl;
      }
    }
  }
#endif

  /** [perf-split] MEASUREMENT PATH ONLY. The host is here as soon as the last
   *  prefill chunk has been SUBMITTED; on both async backends the device is
   *  still working. Stopping the prefill clock now and starting the decode
   *  clock a few lines down charges that in-flight work to the first decode
   *  token, so prefill reads fast and decode (averaged over N tokens) reads
   *  slow by the same amount -- and both move with the prefill block size,
   *  which is how a block-size sweep ends up reading its own instrument.
   *
   *  Draining here makes prefill_duration_ms mean "the prompt is through the
   *  device" and leaves the first decode step measuring only the first decode
   *  step. It adds no GPU work and no net wall time (the same queue would have
   *  been drained by the first token's host read), but it DOES serialize the
   *  host against the device at a point where production currently overlaps
   *  the decode-graph setup with the prefill tail, so it stays behind
   *  NNTR_PERF_SPLIT_DRAIN=1, which only a measurement run sets; nothing on
   *  the app/production path sets it, and with it unset this block compiles to
   *  one already-cached bool test. It cannot change results: both calls are
   *  pure barriers.
   */
  double prefill_drain_ms = 0.0;
  {
    static const bool split_drain = []() {
      const char *e = std::getenv("NNTR_PERF_SPLIT_DRAIN");
      return e != nullptr && e[0] == '1';
    }();
    if (split_drain) {
      const auto drain_begin = std::chrono::high_resolution_clock::now();
      const std::string eng = causallm_engine();
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
      if (eng == "cuda")
        nntrainer::cuda::StreamManager::Global().finish();
#endif
#if defined(ENABLE_OPENCL)
      if (eng != "cuda" && eng != "cpu")
        nntrainer::cl_queue_finish();
#endif
      prefill_drain_ms =
        std::chrono::duration<double, std::milli>(
          std::chrono::high_resolution_clock::now() - drain_begin)
          .count();
    }
  }

  // HTP inference has completed; release prompt-sized staging before the
  // first decode step allocates its graph and expert cache.
  std::unordered_set<nntrainer::ComputeOps *> prefill_backends;
  model->forEachLayer(
    [&prefill_backends](ml::train::Layer &, nntrainer::RunLayerContext &context,
                        void *) {
      if (auto *ops = context.getComputeOps())
        prefill_backends.insert(ops);
    },
    nullptr);
  for (auto *ops : prefill_backends)
    ops->release_prefill_scratch();
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

#if defined(ENABLE_OPENCL)
  /** The GPU footprint peaks somewhere between "the graph is built" and "the
   *  first token is out", and which side of the prefill it peaks on decides
   *  which lever is worth building. Name the boundary. */
  nntrainer::opencl::clMemAcctDump("after-prefill");
  nntrainer::opencl::clLaunchStatBegin("decode");
#endif
  auto start_generation = std::chrono::high_resolution_clock::now();

  /** [perf-split] One host timestamp per decode step, taken after the token has
   *  been streamed to the caller. Free (a clock read per token), always on.
   *  The first step is kept apart from the rest: it is the one that pays for
   *  whatever prefill left in flight (and, on CUDA, for the first graph
   *  capture), so averaging it into "decode TPS" is what made decode TPS a
   *  function of the prefill block size. Token granularity, not delta
   *  granularity: registerOutputs() holds a partial UTF-8 sequence, so a
   *  delta can lag its token by a step. */
  std::vector<double> decode_step_ms;
  decode_step_ms.reserve(64);
  auto decode_step_mark = start_generation;

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

  // The windowed layers' caches from here on: the window, then one row a
  // generated token. The accelerator's decode caches of those layers are
  // sized and based the same (before its first decode row).
  // ponytail: one (rows, base) for all windowed layers -- they share one
  // window and one DSP cache; per-layer values if a model ever mixes them.
  compactKVCache(input_len + global_token_len);
  for (unsigned int l = 0; l < kv_cache.getNumLayers(); ++l)
    if (kv_cache.getRows(l) < kv_cache.getMaxSeqLen()) {
      const unsigned int base = kv_cache.getBase(l);
      htpDecodeKvWindow(input_len + global_token_len - base + NUM_TO_GENERATE,
                        base);
      break;
    }

  struct HtpDecodeReset {
    bool active;
    ~HtpDecodeReset() {
      if (active) {
        htpDecodeWantLogits(true);
        htpDecodeBan(nullptr, 0);
      }
    }
  } htp_reset{htp_execution_profile_};
  if (htp_execution_profile_) {
    htpDecodeWantLogits(do_sample || BATCH_SIZE != 1 ||
                        logits_processor != nullptr ||
                        REPETITION_PENALTY != 1.0f);
    htpDecodeBan(BAD_WORD_IDS.data(), NUM_BADWORDS);
  }
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
    if (generated_ids_path)
      generated_ids.insert(generated_ids.end(), ids_list.begin(),
                           ids_list.end());
    registerOutputs(tokenizer, ids_list, token_generation_idx, eos_list,
                    log_output);
    ++generation_cnt;
    {
      const auto step_end = std::chrono::high_resolution_clock::now();
      decode_step_ms.push_back(
        std::chrono::duration<double, std::milli>(step_end - decode_step_mark)
          .count());
      decode_step_mark = step_end;
      if (perf_split_first_out.time_since_epoch().count() == 0)
        perf_split_first_out = step_end;
    }
#if defined(ENABLE_OPENCL)
    nntrainer::opencl::clLaunchStatTick();
#endif

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

  if (generated_ids_path) {
    std::ofstream ids_file(generated_ids_path);
    for (unsigned int id : generated_ids)
      ids_file << id << '\n';
    if (!ids_file)
      throw std::runtime_error(std::string("Cannot write generated IDs: ") +
                               generated_ids_path);
  }
  auto finish_generation = std::chrono::high_resolution_clock::now();
  auto generation_duration =
    std::chrono::duration_cast<std::chrono::milliseconds>(finish_generation -
                                                          start_generation);

  auto finish_total = std::chrono::high_resolution_clock::now();
  auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
    finish_total - start_total);
#if defined(ENABLE_OPENCL)
  nntrainer::opencl::clMemAcctDump("after-decode");
  nntrainer::opencl::clLaunchStatDump("after-decode");
#endif
  const size_t maxrss_kb = getPeakMemoryKb();
  const size_t honest_kb = FootprintSampler::get().finish();
  const size_t peak_memory = resolvePeakMemoryKb(honest_kb, maxrss_kb);

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
    std::cout << "  (honest RssAnon+accel: " << honest_kb
              << " KB, ru_maxrss: " << maxrss_kb << " KB)\n";
    std::cout << "==========================================================\n";
  }

  performance_metrics.prefill_tokens = init_len;
  performance_metrics.prefill_duration_ms = prefill_duration.count();
  performance_metrics.generation_tokens = generation_cnt;
  performance_metrics.generation_duration_ms = generation_duration.count();
  performance_metrics.total_duration_ms = total_duration.count();
  performance_metrics.peak_memory_kb = peak_memory;
  performance_metrics.peak_rss_kb = maxrss_kb;

  /** [perf-split] The honest rows. See performance_metrics.h for why the two
   *  fields above cannot be trusted on their own. */
  performance_metrics.prefill_drain_ms = prefill_drain_ms;
  performance_metrics.ttft_ms =
    perf_split_first_out.time_since_epoch().count() != 0
      ? std::chrono::duration<double, std::milli>(perf_split_first_out -
                                                  start_total)
          .count()
      : 0.0;
  performance_metrics.first_token_ms =
    decode_step_ms.empty() ? 0.0 : decode_step_ms.front();
  // Median, not mean: one throttle notch or one scheduler hiccup in 127 steps
  // moves a mean and does not move a median, and the question these sweeps ask
  // is what a step costs, not what the worst step cost.
  if (decode_step_ms.size() >= 2) {
    std::vector<double> tail(decode_step_ms.begin() + 1, decode_step_ms.end());
    std::sort(tail.begin(), tail.end());
    const size_t n = tail.size();
    performance_metrics.decode_steady_ms =
      (n % 2 != 0) ? tail[n / 2] : 0.5 * (tail[n / 2 - 1] + tail[n / 2]);
    performance_metrics.decode_steady_tokens = static_cast<unsigned int>(n);
  } else {
    performance_metrics.decode_steady_ms = 0.0;
    performance_metrics.decode_steady_tokens = 0;
  }

  if (log_output) {
    // Its own row, per the measurement rule: TTFT is not prefill + a constant,
    // and prefill_ms is only queue-drained when the drain was armed.
    std::cout << "[PERF-SPLIT] prefill_tok=" << init_len
              << " prefill_ms=" << prefill_duration.count()
              << " drain_ms=" << performance_metrics.prefill_drain_ms
              << " ttft_ms=" << performance_metrics.ttft_ms
              << " first_token_ms=" << performance_metrics.first_token_ms
              << " steady_ms=" << performance_metrics.decode_steady_ms
              << " steady_tok=" << performance_metrics.decode_steady_tokens
              << " gen_tok=" << generation_cnt
              << " e2e_ms=" << total_duration.count() << "\n";
    std::cout.flush();
  }

  if (htp_execution_profile_) {
    if (const char *path = std::getenv("NNTR_HTP_TOKEN_TRACE")) {
      json trace;
      trace["input_ids"] = init_input;
      const bool prefill_emitted = !SKIP_PREFILL && init_len < MAX_SEQ_LEN;
      const size_t begin = input_len + (prefill_emitted ? 0u : 1u);
      const size_t count = generation_cnt + (prefill_emitted ? 1u : 0u);
      trace["output_ids"] = std::vector<unsigned int>(
        ids_history + begin, ids_history + begin + count);
      std::ofstream file(path);
      if (!file)
        throw std::runtime_error("Cannot open NNTR_HTP_TOKEN_TRACE");
      file << trace.dump(2) << '\n';
    }
  }

  has_run_ = true;
}

std::string CausalLM::getOutput(int batch_idx) const {
  if (batch_idx < 0 || batch_idx >= static_cast<int>(output_list.size())) {
    return "";
  }
  return output_list[batch_idx];
}

void CausalLM::releaseDeviceCaches() {
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
  // [reload] The model objects are gone (weights freed). Drop the
  // process-global device caches their weights seeded -- all keyed by HOST
  // pointers the next load's allocator recycles -- so that load neither leaks a
  // model's worth of device packs per cycle nor takes a stale hit and computes
  // with the previous model's weights. Each hook is a pure reset that makes no
  // driver call when its lane was never used, so a CPU/OpenCL-only process
  // never pokes cudart. The OpenCL caches follow below.
  nntrainer::cuda::cuda_fc_qs4cx_release_weight_caches();
  // The gather LUTs first: the embedding LUT they read may be the tied head's
  // VRAM copy, which the next call frees.
  nntrainer::cuda::emb_gather_release_luts();
  nntrainer::cuda::cuda_fc_lmhead_tie_lut_release();
  nntrainer::cuda::cuda_attention_release_caches();
  nntrainer::cuda_reset_decode_graph_cache();
#endif
#if defined(ENABLE_OPENCL)
  // [reload] The OpenCL half of the same class, and the same contract: a pure
  // reset per translation unit, no driver call when the lane was never used.
  // ORDER MATTERS. The FC hook drops the v8c weight packs, whose per-channel
  // scale and row-sum buffers are SUB-BUFFERS of the aux arena's chunks, so the
  // arena is released second -- freeing a parent while its sub-buffers are live
  // would be the mirror of the defect this fixes. Field signature of NOT doing
  // this on Intel: the third load in one process answers the empty string
  // (d41d8cd9), because the arena's partially-used tail chunk is re-entered by
  // the next load with a zero-filled staging image and the flush rewrites the
  // previous load's still-referenced scales as zeros.
  nntrainer::cl_fc_release_caches();
  nntrainer::v8c_release_aux_arena();
  nntrainer::v8c_release_lmhead_caches();
  nntrainer::cl_attention_release_caches();
#endif
}

} // namespace causallm
