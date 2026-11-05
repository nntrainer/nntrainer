// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   gemma4_moe_layer_gpu.cpp
 * @date   02 October 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Gemma4 sparse MoE layer for the OpenCL (gpu) engine.
 */

#include <gemma4_moe_layer_gpu.h>

// Built only with the OpenCL backend; a build list that names this file
// unconditionally (the Android app library) gets an empty unit otherwise.
#if defined(ENABLE_OPENCL)

#include <blas_kernel_interface.h> // clmem_lower_cl / clmem_raise_cl
#include <blas_kernels.h>
#include <compute_ops.h>
#include <cpu_backend.h>
#include <fp16.h>
#include <layer_prof.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <thread_manager.h>

#include <qs4cx_tensor.h>   // the released-payload registry
#include <ternary_tensor.h> // tq2cxSingleCopyOn

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <mutex>
#include <unordered_set>
#if !defined(_WIN32)
#include <sys/mman.h>
#endif
#include <stdexcept>
#include <vector>

namespace causallm {

namespace {

/**
 * @brief Holds a shared-plane tensor for a host read.
 * @details A kernel still in the queue may write the region (its producer):
 * map it, which blocks until the queue has run up to it, and give it back to
 * the device afterwards. Nothing to do when the queue last drained (every
 * kernel output is then complete and mapped back for the host) or for a
 * tensor that is not on the shared plane.
 */
class HostRead {
public:
  explicit HostRead(const nntrainer::Tensor &t) {
    const auto md = t.getMemoryData();
    if (!md || !md->isSVM() || t.isClMem())
      return;
    if (!nntrainer::cl_queue_may_be_busy())
      return;
    ptr_ = const_cast<void *>(static_cast<const void *>(t.getData<uint8_t>()));
    nntrainer::cl_svm_map_force(ptr_, t.bytes(), /*read_only=*/true);
  }
  HostRead(const HostRead &) = delete;
  HostRead &operator=(const HostRead &) = delete;
  ~HostRead() {
    if (ptr_ != nullptr)
      nntrainer::cl_svm_unmap_force(ptr_);
  }

private:
  void *ptr_ = nullptr;
};

bool isHalf(const nntrainer::Tensor &t) {
  return t.getDataType() == nntrainer::TensorDim::DataType::FP16;
}

/**
 * @brief Host fallback of moe_gather_rows, for an input on host memory.
 */
void hostGather(const nntrainer::Tensor &in, nntrainer::Tensor &out,
                const std::vector<unsigned int> &rows, unsigned int width) {
  HostRead hold(in);
  const size_t elem = isHalf(in) ? sizeof(uint16_t) : sizeof(float);
  const size_t bytes = (size_t)rows.size() * width * elem;
  uint8_t *dst = out.getData<uint8_t>();
  const bool svm = out.getMemoryData() && out.getMemoryData()->isSVM();
  if (svm)
    nntrainer::cl_svm_host_write_begin(dst, bytes);
  const uint8_t *src = in.getData<uint8_t>();
  for (size_t r = 0; r < rows.size(); ++r)
    std::memcpy(dst + r * width * elem, src + (size_t)rows[r] * width * elem,
                width * elem);
  if (svm)
    nntrainer::cl_svm_host_write_end(dst, bytes);
}

/**
 * @brief Host fallback of moe_scatter_add_rows, for an output on host memory.
 */
void hostScatterAdd(const nntrainer::Tensor &expert_out, nntrainer::Tensor &out,
                    const std::vector<unsigned int> &rows,
                    const std::vector<float> &weights, unsigned int width) {
  HostRead hold(expert_out);
  if (isHalf(expert_out)) {
    const uint16_t *src = expert_out.getData<uint16_t>();
    uint16_t *dst = out.getData<uint16_t>();
    for (size_t r = 0; r < rows.size(); ++r)
      for (unsigned int j = 0; j < width; ++j) {
        uint16_t &d = dst[(size_t)rows[r] * width + j];
        const float v =
          nntrainer::compute_fp16_to_fp32(d) +
          weights[r] * nntrainer::compute_fp16_to_fp32(src[r * width + j]);
        d = nntrainer::compute_fp32_to_fp16(
          std::min(65504.0f, std::max(-65504.0f, v)));
      }
    return;
  }
  const float *src = expert_out.getData<float>();
  float *dst = out.getData<float>();
  for (size_t r = 0; r < rows.size(); ++r)
    nntrainer::saxpy(width, weights[r], src + r * width, 1,
                     dst + (size_t)rows[r] * width, 1);
}

/**
 * @brief Leave the layer output with the device when the expert half ends.
 * @details The row ops map their output back for the host after each launch
 * (the shared-memory protocol of this backend). The MoE output's consumer
 * is the next device op (the post-expert norm); measured on Intel Xe3, that
 * norm read stale rows unless the last map-back was undone or made blocking,
 * so the output is handed back to the device here -- one enqueue, no drain.
 * A host reader maps it itself (it does whenever kernels may be in flight).
 */
void handOutputToDevice(const nntrainer::Tensor &output) {
  const auto md = output.getMemoryData();
  if (md && md->isSVM() && !output.isClMem())
    nntrainer::cl_svm_unmap_force(
      const_cast<uint8_t *>(output.getData<uint8_t>()));
}

/**
 * @brief Route a decode row on the device (NNTR_G4MOE_DEVICE_ROUTER=0 routes
 *        every step on the host).
 * @details Default on: the device kernels and the host router are the same
 * deterministic routine (moe_route_det.h), so they select the same experts
 * with the same weights, and a decode step on the device never waits for
 * the host. The host router serves prefill and any step the backend declines.
 */
bool deviceRouterOn() {
  static const bool on = []() {
    const char *e = std::getenv("NNTR_G4MOE_DEVICE_ROUTER");
    return e == nullptr || e[0] != '0';
  }();
  return on;
}

/**
 * @brief Route prefill rows on the device too (NNTR_G4MOE_DEVICE_ROUTER_PREFILL
 *        =0, or NNTR_G4MOE_DEVICE_ROUTER=0, routes them on the host).
 * @details Default on: the same deterministic router kernels as decode, over
 * every row (ComputeOps::moe_route), so the routing is the host router's bit
 * for bit. The router rows stay on the device; the host reads back only the
 * selected ids and weights (tokens x topk) instead of reading every router
 * row, widening it and routing it on the CPU while the GPU waits.
 */
bool devicePrefillRouterOn() {
  static const bool on = []() {
    const char *e = std::getenv("NNTR_G4MOE_DEVICE_ROUTER_PREFILL");
    return e == nullptr || e[0] != '0';
  }();
  return on;
}

/**
 * @brief NNTR_G4MOE_DEVICE_ROUTER_MIN_ROWS=n: prefill calls of fewer than n
 *        rows route on the host (default 0: every prefill call on the
 *        device). Read once.
 * @details A switch for the per-device A/B of short prompts; both routers
 *          select the same experts with the same weights, so it changes no
 *          output.
 */
unsigned int deviceRouterMinRows() {
  static const unsigned int v = []() {
    const char *e = std::getenv("NNTR_G4MOE_DEVICE_ROUTER_MIN_ROWS");
    if (e == nullptr || e[0] == '\0')
      return 0u;
    const long n = std::strtol(e, nullptr, 10);
    return n > 0 ? static_cast<unsigned int>(n) : 0u;
  }();
  return v;
}

/**
 * @brief The moe_fsu setting after NNTR_G4MOE_FSU (=1 on, =0 off, unset keeps
 *        @a configured).
 */
bool fsuEnvOverride(bool configured) {
  const char *e = std::getenv("NNTR_G4MOE_FSU");
  if (e == nullptr || e[0] == '\0')
    return configured;
  return e[0] == '1';
}

/**
 * @brief NNTR_G4MOE_FSU_STATS=1: decode statistics at exit (misses per
 *        token, MiB read per token, the token clock of the first MoE layer:
 *        first-100 and median ms per token). Read once.
 */
bool fsuStatsOn() {
  static const bool on = []() {
    const char *e = std::getenv("NNTR_G4MOE_FSU_STATS");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}

/** streamed-expert totals of the process, printed at exit when used */
struct FsuStats {
  std::mutex mu;
  unsigned int layers = 0, slots = 0;
  const char *policy = "lru";
  double lambda = 0.0, prime = 0.0;
  unsigned long long calls = 0, hits = 0, misses = 0;
  unsigned long long dec_calls = 0, dec_hits = 0, dec_misses = 0;
  /** decode misses placed in a slot reserved by a read ahead, and the
   *  residents the reservations evicted */
  unsigned long long dec_reserved = 0, reserve_evictions = 0;
  /** decode steps the device placed (NNTR_G4MOE_FSU_FLAGS), replayed at the
   *  layer's next prepare: the last token's steps never are, and the
   *  backend counts those in at exit (tally, before this line prints) */
  unsigned long long dec_placed = 0, dec_tallied = 0;
  size_t record_bytes = 0;
  ~FsuStats() {
    if (layers == 0)
      return;
    // with the flags protocol (dec_placed != 0) the device's steps are
    // counted at their replay plus, at exit, the last token's that never
    // replayed (tally): every decode step, as [TQ2CX-FSU-FLAGS] counts them
    // (its misses = expert_reads - prefill reads); reads ahead are not
    // counted here (they are in [TQ2CX-FSU-FLAGS])
    char flags_part[96] = "";
    if (dec_placed != 0)
      std::snprintf(flags_part, sizeof(flags_part),
                    " decode_steps=%llu of_them_tallied_at_exit=%llu",
                    dec_calls, dec_tallied);
    std::fprintf(stderr,
                 "[G4MOE-FSU] layers=%u slots_per_layer=%u calls=%llu "
                 "expert_hits=%llu expert_reads=%llu read_mib=%.1f "
                 "policy=%s lambda=%.3f prime=%.1f%s\n",
                 layers, slots, calls, hits, misses,
                 (double)misses * (double)record_bytes / 1048576.0, policy,
                 lambda, prime, flags_part);
    // with the flags protocol the backend's [TQ2CX-FSU-FLAGS] line has the
    // per-token figures over every decode token (hit_rate, misses,
    // miss_mib_per_token), so this one is not printed then
    if (!fsuStatsOn() || dec_calls == 0 || dec_placed != 0)
      return;
    const double tok = (double)dec_calls / (double)layers;
    std::fprintf(stderr,
                 "[G4MOE-FSU-STATS] decode_tokens=%.0f decode_hits=%llu "
                 "decode_misses=%llu hit_rate=%.4f misses_per_token=%.2f "
                 "miss_mib_per_token=%.2f prefill_reads=%llu "
                 "reserved_misses_per_token=%.2f "
                 "reserve_evictions_per_token=%.2f\n",
                 tok, dec_hits, dec_misses,
                 (double)dec_hits / (double)(dec_hits + dec_misses),
                 (double)dec_misses / tok,
                 (double)dec_misses * (double)record_bytes / 1048576.0 / tok,
                 misses - dec_misses, (double)dec_reserved / tok,
                 (double)reserve_evictions / tok);
  }
};

/** the decode token clock (NNTR_G4MOE_FSU_STATS): one stamp per decode call
 *  of the first gpu MoE layer, printed at exit */
struct DecodeClock {
  std::mutex mu;
  const void *layer = nullptr;
  std::vector<std::chrono::steady_clock::time_point> t;
  ~DecodeClock() {
    if (t.size() < 3)
      return;
    std::vector<double> d;
    for (size_t i = 1; i < t.size(); ++i)
      d.push_back(
        std::chrono::duration<double, std::milli>(t[i] - t[i - 1]).count());
    const size_t n100 = std::min<size_t>(100, d.size());
    double s100 = 0.0;
    for (size_t i = 0; i < n100; ++i)
      s100 += d[i];
    std::vector<double> m = d;
    std::sort(m.begin(), m.end());
    const double med = m[m.size() / 2];
    std::fprintf(stderr,
                 "[G4MOE-DECODE-CLOCK] intervals=%zu first100_ms=%.2f "
                 "first100_tps=%.2f median_ms=%.2f median_tps=%.2f\n",
                 d.size(), s100 / n100, 1000.0 * n100 / s100, med,
                 1000.0 / med);
  }
};

DecodeClock &decodeClock() {
  static DecodeClock c;
  return c;
}

FsuStats &fsuStats() {
  static FsuStats s;
  return s;
}

/**
 * @brief The streamed (moe_fsu) MoE layers of the process in graph order:
 *        finalize appends a layer, so layer i + 1 is the one that runs after
 *        layer i (a prefill reads its experts ahead).
 */
struct FsuChain {
  std::mutex mu;
  std::vector<Gemma4MoELayerGpu *> layers;
};

FsuChain &fsuChain() {
  static FsuChain c;
  return c;
}

/**
 * @brief NNTR_G4MOE_FSU_SPEC_MIN_TOKENS: rows from which a prefill call
 *        streams whole layers (default 64). Read once.
 */
unsigned int fsuSpecMinTokens() {
  static const unsigned int v = []() {
    const char *e = std::getenv("NNTR_G4MOE_FSU_SPEC_MIN_TOKENS");
    if (e == nullptr || e[0] == '\0')
      return 64u;
    const long n = std::strtol(e, nullptr, 10);
    return n > 0 ? static_cast<unsigned int>(n) : 64u;
  }();
  return v;
}

/**
 * @brief NNTR_G4MOE_FSU_PREFILL_LOOKAHEAD=0|1 (default 1): a prefill call of
 *        fewer rows than NNTR_G4MOE_FSU_SPEC_MIN_TOKENS (its layers are not
 *        streamed: each reads its routed misses) predicts the next streamed
 *        layer's experts -- that layer's router on this layer's router rows,
 *        every expert in some row's top-k (as the decode pre-gating does) --
 *        and hands them to the backend (MoeExpertFsu::next_ahead), which
 *        reads them into the next layer's slots while this layer computes.
 *        Changes no routing and no value. Read once.
 */
bool prefillLookaheadOn() {
  static const bool on = []() {
    const char *e = std::getenv("NNTR_G4MOE_FSU_PREFILL_LOOKAHEAD");
    return e == nullptr || e[0] != '0';
  }();
  return on;
}

/**
 * @brief The streamed experts' victim policy (ExpertResidency::setPolicy):
 *        NNTR_G4MOE_FSU_POLICY=lru|lfu|lrfu and NNTR_G4MOE_FSU_LRFU_LAMBDA.
 *        Default lrfu (every platform). LRFU's lambda: the env value, else
 *        0.1 below 32 slots per layer and 0.05 from 32 (a cache policy
 *        study over routing traces, best on both of them). Decode misses
 *        per token on Xe3, same tokens, equal to the simulation of the same
 *        routing with the model's semantics: P4 1K C=16 LRU 63.61 -> LRFU
 *        58.42, C=32 32.17 -> 30.23; P2-256 C=16 91.01 -> 85.04, C=32
 *        50.52 -> 45.85. Read once; @a slots = the layer's slots.
 */
std::pair<causallm::ExpertResidency::Policy, double>
fsuPolicy(unsigned int slots) {
  using P = causallm::ExpertResidency::Policy;
  static const int policy = []() {
    constexpr int kDefault = 2; // lrfu (measured: same tokens, fewer reads)
    const char *e = std::getenv("NNTR_G4MOE_FSU_POLICY");
    if (e == nullptr || e[0] == '\0')
      return kDefault;
    if (std::strcmp(e, "lru") == 0)
      return 0;
    if (std::strcmp(e, "lfu") == 0)
      return 1;
    if (std::strcmp(e, "lrfu") == 0)
      return 2;
    ml_logw("NNTR_G4MOE_FSU_POLICY=%s: not lru|lfu|lrfu, the default kept", e);
    return kDefault;
  }();
  static const double lambda = []() {
    const char *e = std::getenv("NNTR_G4MOE_FSU_LRFU_LAMBDA");
    const double v = e != nullptr && e[0] != '\0' ? std::strtod(e, nullptr) : 0;
    return v > 0.0 ? v : 0.0;
  }();
  if (policy == 0)
    return {P::lru, 0.0};
  if (policy == 1)
    return {P::lfu, 0.0};
  return {P::lrfu, lambda > 0.0 ? lambda : (slots >= 32 ? 0.05 : 0.1)};
}

/**
 * @brief NNTR_G4MOE_FSU_PRIME=S (default 8; 0 = off): after a prefill that
 *        streamed a layer (its slots start empty), seed the layer's LFU /
 *        LRFU history with S x (the prefill rows that chose an expert / the
 *        rows): the decode's first evictions then keep the experts the
 *        prompt used most. No read, no slot change; nothing for LRU. The
 *        routing traces, model semantics: 0 to -0.3 misses per token (LRFU,
 *        C=16 / 32; S=8 never worse there). Read once.
 */
double fsuPrimeScale() {
  static const double v = []() {
    const char *e = std::getenv("NNTR_G4MOE_FSU_PRIME");
    if (e == nullptr || e[0] == '\0')
      return 8.0;
    const double x = std::strtod(e, nullptr);
    return x > 0.0 ? x : 0.0;
  }();
  return v;
}

/**
 * @brief NNTR_G4MOE_FSU_FADV=dontneed (default) | keep: drop a streamed
 *        block's file range from the page cache once it is consumed.
 */
bool fsuFadviseDontneed() {
  static const bool v = []() {
    const char *e = std::getenv("NNTR_G4MOE_FSU_FADV");
    return e == nullptr || std::strcmp(e, "keep") != 0;
  }();
  return v;
}

/**
 * @brief NNTR_TQ2CX_ARENA=1: build each layer's resident expert copy in the
 *        backend's expert slot arena, every record read from the weight file
 *        through the backend's reader pool and I/O queue (the fill path of
 *        streamed experts, with every expert resident) instead of uploaded
 *        from the loader's file mapping. A test lane of that path. Read once.
 */
bool arenaPrebuildOn() {
  static const bool on = []() {
    const char *e = std::getenv("NNTR_TQ2CX_ARENA");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}

} // namespace

Gemma4MoELayerGpu::Gemma4MoELayerGpu() :
  Gemma4MoELayer(),
  gathered_idx(0),
  gate_out_idx(0),
  up_out_idx(0),
  act_out_idx(0),
  down_out_idx(0),
  prebuilt(false),
  exact_gelu(false),
  moe_fsu(false),
  expert_fd(-1),
  expert_record_bytes(0),
  fsu_index(std::numeric_limits<unsigned int>::max()) {}

Gemma4MoELayerGpu::~Gemma4MoELayerGpu() {
  if (fsu_index == std::numeric_limits<unsigned int>::max())
    return;
  auto &chain = fsuChain();
  std::lock_guard<std::mutex> lk(chain.mu);
  if (fsu_index < chain.layers.size() && chain.layers[fsu_index] == this)
    chain.layers[fsu_index] = nullptr;
}

void Gemma4MoELayerGpu::finalize(nntrainer::InitLayerContext &context) {
  // Streamed experts (moe_fsu / NNTR_G4MOE_FSU): the slots per layer are
  // moe_cache_size, or NNTR_MOE_CACHE_EXPERTS, or every expert. Set before the
  // base requests the expert weights: a non-zero cache makes them virtual.
  const bool fsu_configured = std::get<props::Gemma4MoEFsu>(moe_props).get();
  moe_fsu = fsuEnvOverride(fsu_configured);
  // NNTR_G4MOE_FSU=0 on a pack (or a default) that streams: every expert
  // resident, so the slot count goes with it (the guard below refuses a
  // cache without moe_fsu).
  if (fsu_configured && !moe_fsu)
    std::get<props::Gemma4MoECacheSize>(moe_props).set(0);
  if (moe_fsu) {
    auto &cache = std::get<props::Gemma4MoECacheSize>(moe_props);
    unsigned int c = cache.get();
    if (const char *e = std::getenv("NNTR_MOE_CACHE_EXPERTS")) {
      const long v = std::strtol(e, nullptr, 10);
      if (v > 0)
        c = static_cast<unsigned int>(v);
    }
    const unsigned int experts = std::get<props::NumExperts>(moe_props).get();
    if (c == 0 || c > experts)
      c = experts;
    cache.set(c);
  }

  Gemma4MoELayer::finalize(context);

  NNTR_THROW_IF(cache_size > 0 && !moe_fsu, std::invalid_argument)
    << "Gemma4 MoE on the gpu engine keeps every expert resident unless "
       "moe_fsu is set; moe_cache_size must be 0 without it (the mapped "
       "expert cache runs on the cpu engine only)";
  if (moe_fsu) {
    NNTR_THROW_IF(context.getWeightDataType() !=
                    nntrainer::TensorDim::DataType::TQ2CX,
                  std::invalid_argument)
      << "Gemma4 MoE moe_fsu streams TQ2CX experts only";
    NNTR_THROW_IF(cache_size < topk, std::invalid_argument)
      << "Gemma4 MoE moe_fsu needs at least " << topk
      << " expert slots per layer (moe_cache_size " << cache_size << ")";
    residency.addLayer(0, cache_size, num_experts);
    const auto pol = fsuPolicy(cache_size);
    residency.setPolicy(
      pol.first,
      pol.first == causallm::ExpertResidency::Policy::lrfu ? pol.second : 0.1);
    slot_map.assign(num_experts, causallm::ExpertResidency::kNone);
    if (fsu_index == std::numeric_limits<unsigned int>::max()) {
      auto &chain = fsuChain();
      std::lock_guard<std::mutex> lk(chain.mu);
      fsu_index = static_cast<unsigned int>(chain.layers.size());
      chain.layers.push_back(this);
    }
  }
  const auto act = std::get<props::MoEActivation>(moe_props).get();
  NNTR_THROW_IF(act != nntrainer::ActivationType::ACT_TANH_GELU &&
                  act != nntrainer::ActivationType::ACT_GELU,
                std::invalid_argument)
    << "Gemma4 MoE on the gpu engine supports the tanh_gelu and gelu expert "
       "gates only";
  exact_gelu = act == nntrainer::ActivationType::ACT_GELU;

  const auto &expert_dim = context.getInputDimensions()[0];
  const unsigned int rows = expert_dim.batch() * expert_dim.height();
  const unsigned int hidden_size = expert_dim.width();
  const unsigned int intermediate_size =
    std::get<nntrainer::props::Unit>(moe_props).get();

  // Workspace of the expert loop. Requested from the context, not built on
  // the heap: a context tensor of a gpu-engine layer is on the shared plane
  // and resolves the gpu op table, which the gather / FC / scatter kernels
  // need; a heap tensor would silently dispatch to the host.
  // in the data type of the rows it moves (the activation type)
  const nntrainer::TensorDim::TensorType act_type(context.getFormat(),
                                                  expert_dim.getDataType());
  auto request = [&](unsigned int width, const char *name) {
    return context.requestTensor(
      nntrainer::TensorDim({1, 1, rows, width}, act_type), name,
      nntrainer::Initializer::NONE, false,
      nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
  };
  gathered_idx = request(hidden_size, "expert_gathered");
  gate_out_idx = request(intermediate_size, "expert_gate_out");
  up_out_idx = request(intermediate_size, "expert_up_out");
  act_out_idx = request(intermediate_size, "expert_act_out");
  down_out_idx = request(hidden_size, "expert_down_out");
}

void Gemma4MoELayerGpu::prebuildExpertWeights(
  nntrainer::RunLayerContext &context) {
  if (prebuilt)
    return;
  auto *ops = context.getWeight(expert_gate_proj_indices[0]).getOps();
  for (unsigned int e = 0; e < num_experts; ++e) {
    ops->fc_prebuild_weight(context.getWeight(expert_gate_proj_indices[e]));
    ops->fc_prebuild_weight(context.getWeight(expert_up_proj_indices[e]));
    ops->fc_prebuild_weight(context.getWeight(expert_down_proj_indices[e]));
  }
  prebuilt = true;
}

nntrainer::MoeExpertGroup
Gemma4MoELayerGpu::expertWeightGroup(nntrainer::RunLayerContext &context) {
  nntrainer::MoeExpertGroup group;
  group.layer_key = this;
  group.gate.resize(num_experts);
  group.up.resize(num_experts);
  group.down.resize(num_experts);
  for (unsigned int e = 0; e < num_experts; ++e) {
    group.gate[e] = &context.getWeight(expert_gate_proj_indices[e]);
    group.up[e] = &context.getWeight(expert_up_proj_indices[e]);
    group.down[e] = &context.getWeight(expert_down_proj_indices[e]);
  }
  group.exact_gelu = exact_gelu;
  return group;
}

bool Gemma4MoELayerGpu::recordExpertFile(nntrainer::RunLayerContext &context,
                                         int fd) {
  if (fd < 0 || num_experts == 0)
    return false;
  std::vector<uint64_t> off(num_experts);
  size_t rec = 0;
  for (unsigned int e = 0; e < num_experts; ++e) {
    const nntrainer::Tensor &g = context.getWeight(expert_gate_proj_indices[e]);
    const nntrainer::Tensor &u = context.getWeight(expert_up_proj_indices[e]);
    const nntrainer::Tensor &d = context.getWeight(expert_down_proj_indices[e]);
    const size_t gb = g.getMemoryBytes(), ub = u.getMemoryBytes(),
                 db = d.getMemoryBytes();
    const size_t r = gb + ub + db;
    if (u.getFileOffset() != g.getFileOffset() + gb ||
        d.getFileOffset() != u.getFileOffset() + ub || (e != 0 && r != rec)) {
      ml_logw("Gemma4 MoE %s: expert %u's gate, up and down are not one "
              "record in the weight file; its experts cannot be read from "
              "the file",
              context.getName().c_str(), e);
      return false;
    }
    rec = r;
    off[e] = g.getFileOffset();
  }
  expert_fd = fd;
  expert_offset = std::move(off);
  expert_record_bytes = rec;
  return true;
}

void Gemma4MoELayerGpu::setRouterDesc(nntrainer::RunLayerContext &context) {
  router_desc.weight = &context.getWeight(router_idx);
  router_desc.scale = &context.getWeight(router_scale_idx);
  router_desc.norm = router_norm_idx != std::numeric_limits<unsigned int>::max()
                       ? &context.getWeight(router_norm_idx)
                       : nullptr;
  router_desc.per_expert_scale = &context.getWeight(per_expert_scale_idx);
  router_desc.epsilon = epsilon;
  router_desc.topk = topk;
}

void Gemma4MoELayerGpu::createExpertSlots(nntrainer::RunLayerContext &context) {
  // the router now: the layer before this one predicts with it from its
  // first call (a first prefill runs that layer before this one)
  setRouterDesc(context);
  file_desc = nntrainer::MoeExpertFsu();
  file_desc.fd = expert_fd;
  file_desc.offset = &expert_offset;
  file_desc.record_bytes = expert_record_bytes;
  file_desc.slots = cache_size;
  file_desc.layer_index = fsu_index;
  file_desc.slot_of = &slot_map;
  // a short prefill's read ahead reaches this layer through this
  // descriptor (the previous layer's next)
  file_desc.reserve = [this](unsigned int e,
                             const std::vector<unsigned int> &avoid) {
    const auto before = residency.reserveStats().evictions;
    const unsigned int s = residency.reserve(0, e, avoid, topk);
    if (residency.reserveStats().evictions != before) {
      auto &st = fsuStats();
      std::lock_guard<std::mutex> lk(st.mu);
      ++st.reserve_evictions;
    }
    return s;
  };
  file_desc.release = [this](unsigned int e, bool admit) {
    residency.release(0, e, admit);
  };
  file_desc.resident = [this](unsigned int e) {
    return residency.slotOf(0, e) != causallm::ExpertResidency::kNone;
  };
  setFlagsHooks(file_desc);
  // the layer's partition of the slot arena, now: every partition exists
  // before the first prefill, which lends some of them to its stream
  nntrainer::MoeExpertGroup group = expertWeightGroup(context);
  group.fsu = &file_desc;
  nntrainer::ComputeOps *ops = context.getComputeOps();
  if (ops == nullptr || !ops->moe_experts_prebuild(group))
    ml_logw("Gemma4 MoE %s: no expert slots at load (%s); they are created "
            "at the first call",
            context.getName().c_str(),
            ops == nullptr ? "no op table" : "declined");
}

void Gemma4MoELayerGpu::setFlagsHooks(nntrainer::MoeExpertFsu &d) {
  d.victim_order = [this](const std::vector<unsigned int> &exclude) {
    return residency.victimOrder(0, exclude);
  };
  // a decode step the device placed (replayed off its path): counted as a
  // decode call like acquire()'s
  d.place =
    [this](const std::vector<unsigned int> &need,
           const std::vector<std::pair<unsigned int, unsigned int>> &placed) {
      const auto a = residency.place(0, need, placed);
      slot_map = residency.slotMap(0);
      auto &st = fsuStats();
      std::lock_guard<std::mutex> lk(st.mu);
      ++st.calls;
      st.hits += a.hits;
      st.misses += a.misses.size();
      ++st.dec_calls;
      st.dec_hits += a.hits;
      st.dec_misses += a.misses.size();
      ++st.dec_placed;
    };
  d.admit = [this](unsigned int e, unsigned int s) {
    residency.admit(0, e, s);
    slot_map = residency.slotMap(0);
  };
  d.forget = [this](unsigned int e) {
    residency.forget(0, e);
    slot_map = residency.slotMap(0);
  };
  d.refresh = [this](const std::vector<unsigned int> &recency) {
    residency.refresh(0, recency);
  };
  // the steps never replayed, counted at exit: the totals only (the layer
  // may be gone by then); the totals object exists from here on, so it
  // outlives the backend's exit line that calls this
  (void)fsuStats();
  d.tally = [](unsigned int hits, unsigned int misses) {
    auto &st = fsuStats();
    std::lock_guard<std::mutex> lk(st.mu);
    ++st.calls;
    st.hits += hits;
    st.misses += misses;
    ++st.dec_calls;
    st.dec_hits += hits;
    st.dec_misses += misses;
    ++st.dec_placed;
    ++st.dec_tallied;
  };
}

nntrainer::MoeExpertFsu Gemma4MoELayerGpu::fsuDescriptor() {
  nntrainer::MoeExpertFsu d;
  d.fd = expert_fd;
  d.offset = &expert_offset;
  d.record_bytes = expert_record_bytes;
  d.slots = cache_size;
  d.slot_of = &slot_map;
  d.complete = residency.complete(0);
  d.layer_index = fsu_index;
  d.spec_min_tokens = fsuSpecMinTokens();
  d.fadvise_dontneed = fsuFadviseDontneed();
  {
    auto &chain = fsuChain();
    std::lock_guard<std::mutex> lk(chain.mu);
    const size_t nx = (size_t)fsu_index + 1;
    if (fsu_index != std::numeric_limits<unsigned int>::max() &&
        nx < chain.layers.size() && chain.layers[nx] != nullptr &&
        chain.layers[nx]->file_desc.fd >= 0) {
      d.next = &chain.layers[nx]->file_desc;
      if (chain.layers[nx]->router_desc.weight != nullptr)
        d.next_router = &chain.layers[nx]->router_desc;
    }
  }
  d.acquire = [this](const std::vector<unsigned int> &experts,
                     std::vector<std::pair<unsigned int, unsigned int>> &fill) {
    const auto a = residency.acquire(0, experts);
    // a slot being read into is not a victim for anyone until it is filled
    for (const auto &m : a.misses) {
      residency.hold(0, m.slot);
      fill.emplace_back(m.expert, m.slot);
    }
    slot_map = residency.slotMap(0);
    auto &st = fsuStats();
    std::lock_guard<std::mutex> lk(st.mu);
    ++st.calls;
    st.hits += a.hits;
    st.misses += a.misses.size();
    if (fsu_decode_call) {
      ++st.dec_calls;
      st.dec_hits += a.hits;
      st.dec_misses += a.misses.size();
      for (const auto &m : a.misses)
        st.dec_reserved += m.reserved ? 1 : 0;
    }
  };
  d.refresh = [this](const std::vector<unsigned int> &recency) {
    residency.refresh(0, recency);
  };
  // a short prefill orders its chunks resident experts first
  d.resident = [this](unsigned int e) {
    return residency.slotOf(0, e) != causallm::ExpertResidency::kNone;
  };
  setFlagsHooks(d);
  d.filled =
    [this](const std::vector<std::pair<unsigned int, unsigned int>> &fill,
           bool ok) {
      for (const auto &f : fill) {
        // a failed read leaves the slot without the expert's bytes
        if (!ok)
          residency.forget(0, f.first);
        residency.unhold(0, f.second);
      }
      if (!ok)
        slot_map = residency.slotMap(0);
    };
  return d;
}

bool Gemma4MoELayerGpu::buildDeviceExperts(
  nntrainer::RunLayerContext &context, const std::vector<const void *> *records,
  const nntrainer::MoeExpertFsu *fsu) {
  if (!nntrainer::tq2cxSingleCopyOn() || num_experts == 0)
    return false;
  nntrainer::MoeExpertGroup group = expertWeightGroup(context);
  if (group.gate[0]->getDataType() != nntrainer::TensorDim::DataType::TQ2CX)
    return false;
  group.records = records;
  group.fsu = fsu;
  // The layer's op table: a self-owned (heap) weight does not name one.
  nntrainer::ComputeOps *ops = context.getComputeOps();
  if (ops == nullptr || !ops->moe_experts_prebuild(group)) {
    ml_logw("Gemma4 MoE %s: the backend keeps no device copy of the experts "
            "(%s); they stay in host memory",
            context.getName().c_str(),
            ops == nullptr ? "no op table" : "declined");
    return false;
  }

  // The device stack owns the experts now. A filled payload is a heap block
  // of its own (the manager's single-copy arm): give its pages back, INWARD
  // page aligned. Either way record the range, so a host reader refuses it.
  auto release = [&](nntrainer::Tensor *w) {
    uint8_t *p = w->getData<uint8_t>();
    const size_t bytes = w->getMemoryBytes();
    if (records == nullptr && fsu == nullptr) {
#if !defined(_WIN32)
      const uintptr_t page = 4096;
      const uintptr_t lo = ((uintptr_t)p + page - 1) & ~(page - 1);
      const uintptr_t hi = ((uintptr_t)p + bytes) & ~(page - 1);
      if (hi > lo)
        (void)::madvise((void *)lo, (size_t)(hi - lo), MADV_DONTNEED);
#endif
    }
    nntrainer::markQs4cxPayloadDropped(p, bytes);
  };
  for (unsigned int e = 0; e < num_experts; ++e) {
    release(group.gate[e]);
    release(group.up[e]);
    release(group.down[e]);
  }
  return true;
}

void Gemma4MoELayerGpu::read(
  nntrainer::ReadSource src, nntrainer::RunLayerContext &run_context,
  bool opt_var, ml::train::ExecutionMode mode, bool trainable,
  nntrainer::TensorDim::DataType defineWeightDataType, bool fsu,
  size_t start_offset, bool read_from_offset, int file_fd) {
  const char *const *mapping = std::get_if<const char *>(&src);
  if (moe_fsu && !opt_var) {
    // Virtual experts: the base read fills the other weights and only notes
    // the file of the experts; remember where each expert's record is.
    Gemma4MoELayer::read(src, run_context, opt_var, mode, trainable,
                         defineWeightDataType, fsu, start_offset,
                         read_from_offset, file_fd);
    NNTR_THROW_IF(!read_from_offset || !recordExpertFile(run_context, file_fd),
                  std::runtime_error)
      << "Gemma4 MoE " << run_context.getName()
      << ": moe_fsu needs the expert records' file offsets (a safetensors "
         "or offset-read load with the weight file open)";
    createExpertSlots(run_context);
    auto &st = fsuStats();
    std::lock_guard<std::mutex> lk(st.mu);
    ++st.layers;
    st.slots = cache_size;
    st.record_bytes = expert_record_bytes;
    st.policy =
      residency.policy() == causallm::ExpertResidency::Policy::lrfu  ? "lrfu"
      : residency.policy() == causallm::ExpertResidency::Policy::lfu ? "lfu"
                                                                     : "lru";
    st.lambda = residency.lambda();
    st.prime = residency.policy() == causallm::ExpertResidency::Policy::lru
                 ? 0.0
                 : fsuPrimeScale();
    return;
  }
  const bool eligible = !opt_var && !fsu && nntrainer::tq2cxSingleCopyOn();
  auto readOthers = [&]() {
    std::unordered_set<unsigned int> experts;
    for (unsigned int e = 0; e < num_experts; ++e) {
      experts.insert(expert_gate_proj_indices[e]);
      experts.insert(expert_up_proj_indices[e]);
      experts.insert(expert_down_proj_indices[e]);
    }
    for (unsigned int i = 0; i < run_context.getNumWeights(); ++i)
      if (!experts.count(i) && run_context.isGradientFirstAccess(i))
        run_context.getWeight(i).read(src, start_offset, read_from_offset,
                                      file_fd);
  };
  if (eligible && read_from_offset && arenaPrebuildOn() &&
      recordExpertFile(run_context, file_fd)) {
    // Every expert into its slot of the backend's slot arena, read from the
    // file through the backend's reader pool (the streaming fill path with
    // every expert resident): the host payloads are never filled.
    std::vector<unsigned int> slot_of(num_experts);
    std::vector<std::pair<unsigned int, unsigned int>> fills(num_experts);
    for (unsigned int e = 0; e < num_experts; ++e) {
      slot_of[e] = e;
      fills[e] = {e, e};
    }
    nntrainer::MoeExpertFsu desc;
    desc.fd = expert_fd;
    desc.offset = &expert_offset;
    desc.record_bytes = expert_record_bytes;
    desc.slots = num_experts;
    desc.slot_of = &slot_of;
    desc.fill = &fills;
    desc.complete = true;
    if (buildDeviceExperts(run_context, nullptr, &desc)) {
      readOthers();
      return;
    }
  }
  if (eligible && read_from_offset && mapping != nullptr &&
      *mapping != nullptr) {
    // Upload the expert records straight from the mapping, before anything
    // is copied: the host payloads are then never filled.
    std::vector<const void *> records;
    records.reserve(3 * num_experts);
    for (unsigned int e = 0; e < num_experts; ++e)
      for (unsigned int idx :
           {expert_gate_proj_indices[e], expert_up_proj_indices[e],
            expert_down_proj_indices[e]})
        records.push_back(*mapping +
                          run_context.getWeight(idx).getFileOffset());
    if (buildDeviceExperts(run_context, &records)) {
      readOthers();
      return;
    }
  }
  Gemma4MoELayer::read(src, run_context, opt_var, mode, trainable,
                       defineWeightDataType, fsu, start_offset,
                       read_from_offset, file_fd);
  if (eligible)
    buildDeviceExperts(run_context, nullptr);
}

void Gemma4MoELayerGpu::read(
  std::ifstream &file, nntrainer::RunLayerContext &run_context, bool opt_var,
  ml::train::ExecutionMode mode, bool trainable,
  nntrainer::TensorDim::DataType defineWeightDataType, bool fsu,
  size_t start_offset, bool read_from_offset, int file_fd) {
  Gemma4MoELayer::read(file, run_context, opt_var, mode, trainable,
                       defineWeightDataType, fsu, start_offset,
                       read_from_offset, file_fd);
  if (moe_fsu && !opt_var) {
    NNTR_THROW_IF(!read_from_offset || !recordExpertFile(run_context, file_fd),
                  std::runtime_error)
      << "Gemma4 MoE " << run_context.getName()
      << ": moe_fsu needs the expert records' file offsets (a safetensors "
         "or offset-read load with the weight file open)";
    createExpertSlots(run_context);
    auto &st = fsuStats();
    std::lock_guard<std::mutex> lk(st.mu);
    ++st.layers;
    st.slots = cache_size;
    st.record_bytes = expert_record_bytes;
    st.policy =
      residency.policy() == causallm::ExpertResidency::Policy::lrfu  ? "lrfu"
      : residency.policy() == causallm::ExpertResidency::Policy::lfu ? "lfu"
                                                                     : "lru";
    st.lambda = residency.lambda();
    st.prime = residency.policy() == causallm::ExpertResidency::Policy::lru
                 ? 0.0
                 : fsuPrimeScale();
    return;
  }
  if (!opt_var && !fsu)
    buildDeviceExperts(run_context, nullptr);
}

void Gemma4MoELayerGpu::forwardTensors(nntrainer::RunLayerContext &context,
                                       nntrainer::Tensor &expert_input,
                                       nntrainer::Tensor &router_input,
                                       nntrainer::Tensor &output,
                                       nntrainer::Tensor &router_input_scaled,
                                       nntrainer::Tensor &router_logits) {
  // The router runs on host scratch (below), not on these: they are gpu
  // context tensors whose op table has no host GEMM.
  (void)router_input_scaled;
  (void)router_logits;

  const unsigned int total_tokens =
    expert_input.batch() * expert_input.height();
  const unsigned int hidden_size = expert_input.width();
  expert_input.reshape({total_tokens, 1, 1, hidden_size});
  router_input.reshape({total_tokens, 1, 1, hidden_size});
  output.reshape({total_tokens, 1, 1, hidden_size});

  // NNTR_LAYER_PROFILE scopes: the host read of the router rows waits for
  // the queue to reach them; the routing itself is host compute.
  const bool decode_step = total_tokens == 1;
  nntrainer::ComputeOps *ops = expert_input.getOps();
  if (moe_fsu && ops != nullptr) {
    // The previous step of this layer may still run its acquire / filled /
    // refresh hooks on a reader thread (NNTR_G4MOE_FSU_ASYNC): touch the
    // residency policy only once it is done
    nntrainer::MoeExpertGroup join;
    join.layer_key = this;
    join.tokens = total_tokens;
    ops->moe_experts_join(join);
  }
  fsu_decode_call = decode_step;
  if (decode_step && fsuStatsOn()) {
    auto &clk = decodeClock();
    std::lock_guard<std::mutex> lk(clk.mu);
    if (clk.layer == nullptr)
      clk.layer = this;
    if (clk.layer == this)
      clk.t.push_back(std::chrono::steady_clock::now());
  }

  // ---- decode: route on the device -------------------------------------
  // The whole step stays on the queue: no host read of the router row, no
  // drain. A backend that cannot route on the device declines having
  // enqueued nothing, and the step takes the host router below.
  // Streamed experts: once every expert of the layer is resident the device
  // router looks them up on the device; until then the backend waits for
  // the device's selection, reads the misses into their slots and writes
  // the slots' offsets before the expert GEMVs.
  nntrainer::MoeExpertFsu fsu_desc;
  if (moe_fsu) {
    // A long prefill streams whole layers through slots the backend lends
    // from every layer's partition: this layer's resident experts are gone
    // after it, so forget them now (decode starts from empty slots). Start
    // the reads before the router, so they overlap it.
    if (total_tokens >= fsuSpecMinTokens() && cache_size < num_experts) {
      residency.clear(0);
      slot_map.assign(num_experts, causallm::ExpertResidency::kNone);
    }
    setRouterDesc(context);
    fsu_desc = fsuDescriptor();
    if (total_tokens >= fsuSpecMinTokens() && cache_size < num_experts) {
      nntrainer::MoeExpertGroup group = expertWeightGroup(context);
      group.tokens = total_tokens;
      group.fsu = &fsu_desc;
      expert_input.getOps()->moe_experts_prefetch(group);
    }
  }
  if (decode_step && deviceRouterOn()) {
    nntrainer::MoeRouter rt;
    rt.input = &router_input;
    rt.weight = &context.getWeight(router_idx);
    rt.scale = &context.getWeight(router_scale_idx);
    rt.norm = router_norm_idx != std::numeric_limits<unsigned int>::max()
                ? &context.getWeight(router_norm_idx)
                : nullptr;
    rt.per_expert_scale = &context.getWeight(per_expert_scale_idx);
    rt.epsilon = epsilon;
    rt.topk = topk;
    nntrainer::MoeExpertGroup group = expertWeightGroup(context);
    group.input = &expert_input;
    group.output = &output;
    group.tokens = 1;
    group.router = &rt;
    bool deferred = false;
    if (moe_fsu) {
      group.fsu = &fsu_desc;
      // the expert half may be left for the output's reader (the next
      // norm): the dense MLP enqueued in between covers the expert reads
      group.output_deferred = &deferred;
    }
    bool ran = false;
    {
      nntrainer::LayerProfScope prof_exp("g4moe_experts_enqueue", true);
      ran = ops->moe_experts(group);
    }
    if (ran) {
      // deferred: the output is already the device's, and its kernels are
      // not enqueued yet
      if (!deferred)
        handOutputToDevice(output);
      return;
    }
  }

  // ---- routing, on the host --------------------------------------------
  // The same code as the cpu layer on host tensors, so the selected experts,
  // their order and their weights are identical to the cpu engine's.
  Routing routing;
  const bool device_routed =
    !decode_step && deviceRouterOn() && devicePrefillRouterOn() &&
    total_tokens >= deviceRouterMinRows() &&
    deviceRoute(context, ops, router_input, total_tokens, routing);
  if (!device_routed)
    hostRoute(context, router_input, total_tokens, hidden_size, routing);
  // a prefill that streams this layer left its slots empty: seed the
  // victim policy's history with the prompt's expert histogram
  if (moe_fsu && !decode_step && total_tokens >= fsuSpecMinTokens() &&
      cache_size < num_experts && fsuPrimeScale() > 0.0 &&
      routing.assignments.size() == num_experts) {
    std::vector<double> w(num_experts, 0.0);
    for (unsigned int e = 0; e < num_experts; ++e)
      w[e] = fsuPrimeScale() * (double)routing.assignments[e].size() /
             (double)total_tokens;
    residency.prime(0, w);
  }
  // a short prefill (not streamed): the next layer's experts, predicted
  std::vector<unsigned int> next_ahead;
  if (moe_fsu && !decode_step && total_tokens < fsuSpecMinTokens() &&
      cache_size < num_experts && prefillLookaheadOn() &&
      fsu_desc.next != nullptr && fsu_desc.next_router != nullptr &&
      fsu_desc.next->offset != nullptr &&
      fsu_desc.next->slots < fsu_desc.next->offset->size()) {
    predictNext(ops, router_input, total_tokens, *fsu_desc.next_router,
                next_ahead);
    if (!next_ahead.empty())
      fsu_desc.next_ahead = &next_ahead;
  }

  // ---- experts, on the device ------------------------------------------
  // An expert format whose kernels batch several experts per launch runs the
  // whole expert half in one call (see ComputeOps::moe_experts); the others
  // take the per-expert loop below.
  {
    nntrainer::MoeExpertGroup group = expertWeightGroup(context);
    group.input = &expert_input;
    group.output = &output;
    group.tokens = total_tokens;
    group.active = routing.active;
    group.assignments = &routing.assignments;
    if (moe_fsu)
      group.fsu = &fsu_desc;
    nntrainer::LayerProfScope prof_exp("g4moe_experts_enqueue", decode_step);
    if (ops->moe_experts(group)) {
      handOutputToDevice(output);
      return;
    }
    // virtual experts have no host payload for the per-expert loop
    NNTR_THROW_IF(moe_fsu, std::runtime_error)
      << "Gemma4 MoE " << context.getName()
      << ": the backend declined the streamed experts (moe_fsu)";
  }

  // The per-expert loop works on the shared plane: bring a device-only
  // input back to it, and the output to the device at the end.
  nntrainer::clmem_lower_cl(expert_input, 0);
  forwardExpertLoop(context, expert_input, output, routing, total_tokens,
                    hidden_size);
  nntrainer::clmem_raise_cl(output, 0);
}

void Gemma4MoELayerGpu::predictNext(nntrainer::ComputeOps *ops,
                                    nntrainer::Tensor &router_input,
                                    unsigned int total_tokens,
                                    const nntrainer::MoeRouter &next,
                                    std::vector<unsigned int> &ranked) {
  ranked.clear();
  if (ops == nullptr || next.weight == nullptr || next.topk == 0)
    return;
  nntrainer::MoeRouter rt = next;
  rt.input = &router_input;
  const unsigned int K = rt.topk;
  std::vector<uint32_t> ids((size_t)total_tokens * K);
  std::vector<float> w((size_t)total_tokens * K);
  {
    nntrainer::LayerProfScope prof("g4moe_ahead_route", false);
    if (!ops->moe_route(rt, total_tokens, ids.data(), w.data()))
      return;
  }
  // most predicted rows first, then the lower id
  std::vector<unsigned int> rows(num_experts, 0);
  for (uint32_t e : ids)
    if (e < num_experts)
      rows[e]++;
  for (unsigned int e = 0; e < num_experts; ++e)
    if (rows[e] != 0)
      ranked.push_back(e);
  std::stable_sort(
    ranked.begin(), ranked.end(),
    [&rows](unsigned int x, unsigned int y) { return rows[x] > rows[y]; });
}

bool Gemma4MoELayerGpu::deviceRoute(nntrainer::RunLayerContext &context,
                                    nntrainer::ComputeOps *ops,
                                    nntrainer::Tensor &router_input,
                                    unsigned int total_tokens,
                                    Routing &routing) {
  nntrainer::MoeRouter rt;
  rt.input = &router_input;
  rt.weight = &context.getWeight(router_idx);
  rt.scale = &context.getWeight(router_scale_idx);
  rt.norm = router_norm_idx != std::numeric_limits<unsigned int>::max()
              ? &context.getWeight(router_norm_idx)
              : nullptr;
  rt.per_expert_scale = &context.getWeight(per_expert_scale_idx);
  rt.epsilon = epsilon;
  rt.topk = topk;
  std::vector<uint32_t> ids((size_t)total_tokens * topk);
  std::vector<float> w((size_t)total_tokens * topk);
  {
    // enqueue + the one blocking read of the ids and weights
    nntrainer::LayerProfScope prof_wait("g4moe_router_wait", false);
    if (ops == nullptr ||
        !ops->moe_route(rt, total_tokens, ids.data(), w.data()))
      return false;
  }
  {
    // the bucketing of routeTokens: rows in token order per expert
    nntrainer::LayerProfScope prof_route("g4moe_router_host", false);
    routing.assignments.assign(num_experts, {});
    for (unsigned int t = 0; t < total_tokens; ++t)
      for (unsigned int k = 0; k < topk; ++k) {
        const uint32_t e = ids[(size_t)t * topk + k];
        NNTR_THROW_IF(e >= num_experts, std::runtime_error)
          << "Gemma4 MoE " << context.getName()
          << ": device router selected expert " << e;
        routing.assignments[e].emplace_back(t, w[(size_t)t * topk + k]);
      }
    routing.active.clear();
    for (unsigned int e = 0; e < num_experts; ++e)
      if (!routing.assignments[e].empty())
        routing.active.push_back(e);
  }
  return true;
}

void Gemma4MoELayerGpu::hostRoute(nntrainer::RunLayerContext &context,
                                  nntrainer::Tensor &router_input,
                                  unsigned int total_tokens,
                                  unsigned int hidden_size, Routing &routing) {
  const bool decode_step = total_tokens == 1;
  if (host_router_scaled.empty() ||
      host_router_scaled.batch() != total_tokens) {
    host_router_scaled = nntrainer::Tensor(total_tokens, 1, 1, hidden_size);
    host_router_logits = nntrainer::Tensor(total_tokens, 1, 1, num_experts);
  }
  // a device-only router row is read back first (a blocking read)
  nntrainer::clmem_lower_cl(router_input, 0);
  if (isHalf(router_input)) {
    // FP16 rows: widen them into host FP32 scratch, the router's input type
    if (host_router_input.empty() || host_router_input.batch() != total_tokens)
      host_router_input = nntrainer::Tensor(total_tokens, 1, 1, hidden_size);
    {
      nntrainer::LayerProfScope prof_wait("g4moe_router_wait", decode_step);
      HostRead hold(router_input);
      const uint16_t *src = router_input.getData<uint16_t>();
      float *dst = host_router_input.getData<float>();
      // the same exact conversion per element, rows spread over the threads
      auto widen_row = [&](size_t row) {
        const size_t o = row * hidden_size;
        for (size_t i = 0; i < hidden_size; ++i)
          dst[o + i] = nntrainer::compute_fp16_to_fp32(src[o + i]);
      };
      if (total_tokens > 1)
        nntrainer::ThreadManager::Global().parallel_for(0, total_tokens,
                                                        widen_row);
      else
        widen_row(0);
    }
    nntrainer::LayerProfScope prof_route("g4moe_router_host", decode_step);
    routeTokens(context, host_router_input, host_router_scaled,
                host_router_logits, total_tokens, hidden_size, routing);
  } else {
    HostRead hold(router_input);
    routeTokens(context, router_input, host_router_scaled, host_router_logits,
                total_tokens, hidden_size, routing);
  }
}

void Gemma4MoELayerGpu::forwardExpertLoop(nntrainer::RunLayerContext &context,
                                          nntrainer::Tensor &expert_input,
                                          nntrainer::Tensor &output,
                                          const Routing &routing,
                                          unsigned int total_tokens,
                                          unsigned int hidden_size) {
  nntrainer::ComputeOps *ops = expert_input.getOps();
  prebuildExpertWeights(context);

  nntrainer::Tensor &gathered_all = context.getTensor(gathered_idx);
  nntrainer::Tensor &gate_all = context.getTensor(gate_out_idx);
  nntrainer::Tensor &up_all = context.getTensor(up_out_idx);
  nntrainer::Tensor &act_all = context.getTensor(act_out_idx);
  nntrainer::Tensor &down_all = context.getTensor(down_out_idx);
  const auto tensor_type = gathered_all.getTensorType();
  const unsigned int intermediate_size = gate_all.width();

  // Device row movement when the planes allow it. An operand on host memory
  // (a host producer upstream, or a host consumer downstream) keeps the host
  // loop for that one step; the order of the additions is the same either way.
  const bool device_output = ops->moe_zero_rows(output, total_tokens);
  if (!device_output)
    output.setZero();

  std::vector<unsigned int> rows;
  std::vector<float> weights;
  for (const unsigned int expert : routing.active) {
    const auto &assigned = routing.assignments[expert];
    const unsigned int n = static_cast<unsigned int>(assigned.size());
    rows.resize(n);
    weights.resize(n);
    for (unsigned int i = 0; i < n; ++i) {
      rows[i] = assigned[i].first;
      weights[i] = assigned[i].second;
    }

    // Views over the workspace base: the FC reads M from the view height,
    // and a device plane can only be bound from its base.
    nntrainer::TensorDim hidden_dim({1, 1, n, hidden_size}, tensor_type);
    nntrainer::TensorDim inter_dim({1, 1, n, intermediate_size}, tensor_type);
    nntrainer::Tensor gathered =
      gathered_all.getSharedDataTensor(hidden_dim, 0, true);
    nntrainer::Tensor gate_out =
      gate_all.getSharedDataTensor(inter_dim, 0, true);
    nntrainer::Tensor up_out = up_all.getSharedDataTensor(inter_dim, 0, true);
    nntrainer::Tensor act_out = act_all.getSharedDataTensor(inter_dim, 0, true);
    nntrainer::Tensor down_out =
      down_all.getSharedDataTensor(hidden_dim, 0, true);

    if (!ops->moe_gather_rows(expert_input, gathered, rows.data(), n))
      hostGather(expert_input, gathered, rows, hidden_size);

    ops->fc_expert(
      gathered, context.getWeight(expert_gate_proj_indices[expert]), gate_out);
    ops->fc_expert(gathered, context.getWeight(expert_up_proj_indices[expert]),
                   up_out);
    if (exact_gelu)
      ops->geglu_erf(gate_out, up_out, act_out, n, 0);
    else
      ops->geglu(gate_out, up_out, act_out, n, 0);
    ops->fc_expert(act_out, context.getWeight(expert_down_proj_indices[expert]),
                   down_out);

    // Ascending expert id, one call per expert on the in-order queue: every
    // output element sums its contributions in the cpu layer's order.
    if (!device_output || !ops->moe_scatter_add_rows(
                            down_out, output, rows.data(), weights.data(), n))
      hostScatterAdd(down_out, output, rows, weights, hidden_size);
  }
  if (device_output)
    handOutputToDevice(output);
}

} // namespace causallm

#endif // ENABLE_OPENCL
