// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jungwon-Lee <jungone.lee@samsung.com>
 *
 * @file   lfm2_moe_layer.cpp
 * @date   06 July 2026
 * @brief  Mixture-of-Experts layer for the LFM2-8B-A1B (lfm2_moe) model.
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#if __has_include(<htp_runtime_config.h>)
#include <htp_runtime_config.h>
#else
#include <htp_backend/htp_runtime_config.h>
#endif
#include <acti_func.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <compute_ops.h>
#include <cpu_backend.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <htp_backend/htp_tl2_opts.h>
#include <htp_decode_hook.h>
#include <htp_wh_layout.h>
#include <htp_wh_palette.h>
#include <map>
#include <set>
#ifndef _WIN32
#include <unistd.h>
#endif
#include <htp_backend/htp_expert_pool.h>
#include <iostream>
#include <lfm2_moe_layer.h>
#include <lfm2_router_arm.h>
#include <node_exporter.h>
#include <rms_norm_rows.h>
#include <stdexcept>
#include <thread_manager.h>
#include <unordered_map>
#include <unordered_set>

/** Match the DSP's SwiGLU bit for bit so fusion preserves uint8
 *  quantization levels. */
#include <swiglu_det.h>

namespace causallm {

namespace {

/**
 * @brief ARM-side MoE stage timings, shared by the CPU and HTP paths.
 *
 * Matching stage boundaries separate shared routing/gather/scatter work
 * from the FFN. `other` reports wall time outside the named stages.
 *
 * The unlocked accumulator requires serial layer execution. Timers run
 * on the caller thread, including the gather's parallel_for interval.
 */
struct M0Stages {
  uint64_t setup = 0;   /**< reshape + output.setZero() */
  uint64_t router = 0;  /**< input.dot(gate_weights) */
  uint64_t topk = 0;    /**< buildExpertAssignments + max scan */
  uint64_t wksp = 0;    /**< per-layer workspace Tensor construction */
  uint64_t gather = 0;  /**< tokens -> contiguous per-expert input */
  uint64_t ffn = 0;     /**< the expert FFN itself: HTP dispatch or two dots */
  uint64_t route = 0;   /**< multiply by the routing weight */
  uint64_t scatter = 0; /**< add_i back into the layer output */
  uint64_t misses = 0;  /**< count of experts loaded from the model file */

  void reset() { *this = M0Stages{}; }
  uint64_t sum() const {
    return setup + router + topk + wksp + gather + ffn + route + scatter;
  }
};

M0Stages g_m0;
bool g_m0_on = false;

/** @brief NNTR_OP_TIME=1: the decode MoE layer's M0 slots summed
 *         over the run, split into the ARM part (setup + router + top-k)
 *         and the call (`ffn`: staging, queue write, ARM spin, response),
 *         one line at exit. */
struct OpTimeMoe {
  uint64_t calls = 0, cpu_us = 0, call_us = 0;

  static bool on() {
    static const bool v = [] {
      const char *e = std::getenv("NNTR_OP_TIME");
      return e != nullptr && std::strcmp(e, "1") == 0;
    }();
    return v;
  }

  ~OpTimeMoe() {
    if (calls != 0)
      std::fprintf(stderr,
                   "[OP-TIME] moe calls=%llu cpu_us=%llu call_us=%llu\n",
                   static_cast<unsigned long long>(calls),
                   static_cast<unsigned long long>(cpu_us),
                   static_cast<unsigned long long>(call_us));
  }
};

OpTimeMoe g_op_time_moe;

/** @brief Adds its lifetime to one M0Stages slot, and costs two predicted
 *         branches when profiling is off. */
class M0Timer {
public:
  explicit M0Timer(uint64_t *slot) : slot_(g_m0_on ? slot : nullptr) {
    if (slot_) {
      t0_ = std::chrono::steady_clock::now();
    }
  }
  ~M0Timer() { stop(); }

  /** @brief Ends the interval early and idempotently. */
  void stop() {
    if (!slot_) {
      return;
    }
    *slot_ += static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - t0_)
        .count());
    slot_ = nullptr;
  }

  M0Timer(const M0Timer &) = delete;
  M0Timer &operator=(const M0Timer &) = delete;

private:
  uint64_t *slot_;
  std::chrono::steady_clock::time_point t0_;
};

} // namespace

static constexpr size_t SINGLE_INOUT_IDX = 0;
/** Optional second input: what the router reads instead of the experts'
 *  input (props::RouterType). */
static constexpr size_t ROUTER_IN_IDX = 1;

/** LFM2-MoE router hyper-parameters (fixed for LFM2-8B-A1B). */
static constexpr bool NORM_TOPK_PROB = true;
static constexpr float ROUTED_SCALING_FACTOR = 1.0f;

/** Experts beyond the routed top-k whose recency the LRU refreshes. */
static constexpr unsigned EXTRA_TOPK = 5;

using ExpertFileDesc = nntrainer::ComputeOps::ExpertFileDesc;
using nntrainer::requireHtpExpertPool;

/** Read ahead by this many layers during prefill, bounded by free slots. */
static int expertPrefetchDepth() {
  return static_cast<int>(nntrainer::htpRuntimeConfig().moe.prefill_lookahead);
}

static size_t beginHtpExpertPrefill(nntrainer::ComputeOps *ops,
                                    unsigned int layers,
                                    unsigned int experts_per_layer) {
  auto *pool = ops ? ops->htpExpertPool() : nullptr;
  if (!pool)
    return 0;
  // Only fresh C16 native TL2 pools may start before the first MoE call.
  // A later pass keeps the existing layer-zero lifecycle unchanged.
  if (!ops || ops != pool->owner || pool->early_owner ||
      pool->prefetch_next != 0 || !pool->prefetch_batches.empty() ||
      !pool->prefetch_inflight.empty() || !pool->handed_layers.empty() ||
      pool->layers.size() != layers || pool->lru.capacity() != layers * 16u ||
      !experts_per_layer || expertPrefetchDepth() == 0)
    return 0;

  // Preload is a graph-order prefix and can end partway through a layer.
  // Validate the actual resident keys before using it as the read frontier.
  const size_t resident_count = pool->lru.size();
  const size_t resident_layers = resident_count / experts_per_layer;
  const size_t resident_tail = resident_count % experts_per_layer;
  if (resident_layers >= layers)
    return 0;

  std::vector<nntrainer::ExpertLru::Key> pinned;
  size_t ordinal = 0;
  for (size_t l = 0; l < pool->layers.size(); ++l) {
    if (pool->layers[l].size() != experts_per_layer)
      return 0;
    for (const ExpertFileDesc &d : pool->layers[l]) {
      const bool expected_resident = ordinal++ < resident_count;
      if (d.w_bits != HTP_TL2_WEIGHT_BITS ||
          pool->lru.resident(d.key_gu) != expected_resident)
        return 0;
      if (expected_resident)
        pinned.push_back(d.key_gu);
    }
  }

  size_t queued = 0;
  const int last =
    std::min(expertPrefetchDepth(), static_cast<int>(pool->layers.size()) - 1);
  for (pool->prefetch_next = static_cast<int>(resident_layers);
       pool->prefetch_next <= last; ++pool->prefetch_next) {
    const auto &layer = pool->layers[pool->prefetch_next];
    std::vector<ExpertFileDesc> suffix;
    if (pool->prefetch_next == static_cast<int>(resident_layers) &&
        resident_tail)
      suffix.assign(layer.begin() + resident_tail, layer.end());
    const auto &want = suffix.empty() ? layer : suffix;
    // Only the first incomplete layer needs a temporary descriptor list.
    // Full batches preserve the ordinary first layer's registration groups
    // and LRFU acquire order, retaining the preloaded prefix and leaving
    // the later partial batch to the normal layer-time refill.
    if (want.size() > pool->lru.room(pinned) ||
        !pool->lru.makeRoom(want.size(), pinned, [](const void *) {
          throw std::logic_error("early prefill evicted a preloaded expert");
        }))
      break;
    pool->lru.hold(want.size());
    bool started = false;
    try {
      started = ops->prefetch_qs4cx_wh_experts_begin(want);
    } catch (...) {
      pool->lru.unhold(want.size());
      throw;
    }
    if (!started) {
      pool->lru.unhold(want.size());
      break;
    }
    pool->prefetch_batches.emplace_back(pool->prefetch_next, want.size());
    for (const ExpertFileDesc &d : want)
      pool->prefetch_inflight.insert(d.key_gu);
    queued += want.size();
    pool->early_owner = ops;
  }
  if (!queued)
    pool->prefetch_next = 0;
  return queued;
}

size_t beginLfm2HtpExpertPrefill(nntrainer::ComputeOps *ops) {
  return beginHtpExpertPrefill(ops, 38, 64);
}

void retireHtpExpertPrefillKickoff(nntrainer::ComputeOps *ops) {
  auto *pool = ops ? ops->htpExpertPool() : nullptr;
  if (pool && pool->early_owner == ops)
    pool->early_owner = nullptr;
}

/** Routing trace: layer, token count, routed experts and extended top-k. */
static std::FILE *moeTrace(nntrainer::ComputeOps *ops) {
  const auto &path = nntrainer::htpRuntimeConfig().moe.trace;
  auto *pool = ops ? ops->htpExpertPool() : nullptr;
  if (path.empty() || !pool)
    return nullptr;
  if (!pool->trace_opened) {
    pool->trace_opened = true;
    pool->trace_file.reset(std::fopen(path.c_str(), "w"));
  }
  return pool->trace_file.get();
}

/** Per-layer cache share, falling back to the runtime policy and capped by
 * the layer's expert count. */
static unsigned int expertCacheCapacity(unsigned int num_experts,
                                        unsigned int from_props) {
  return std::min(num_experts,
                  from_props != 0u
                    ? from_props
                    : nntrainer::htpRuntimeConfig().moe.cache_experts);
}

/** M0 ordinal of prefill forwarding() calls, so each MoE layer
 * prints under a stable index in layer order. Decode never touches this --
 * it runs incremental_forwarding. */
static std::atomic<unsigned> g_m0_call_index{0};

Lfm2MoELayer::Lfm2MoELayer() :
  LayerImpl(),
  num_experts(0),
  topk(0),
  moe_props(props::NumExperts(), props::NumExpertsPerToken(),
            nntrainer::props::Unit(), props::MoEActivation(),
            props::RouterType(), props::CacheExperts(), props::InNorm(),
            props::RouterNorm(), props::OutNorm(), props::OutAdd(),
            nntrainer::props::Epsilon(), props::OutToHost()),
  softmax_router(false),
  gelu_act(false),
  expert_gate_up_proj_indices({}),
  expert_down_proj_indices({}),
  gate_idx(std::numeric_limits<unsigned>::max()),
  expert_bias_idx(std::numeric_limits<unsigned>::max()),
  experts_virtual(false),
  cache_per_layer(0),
  expert_layer_slot(-1),
  trace_layer(0),
  router_logits_idx(std::numeric_limits<unsigned>::max()),
  decode_expert_output_idx(std::numeric_limits<unsigned>::max()),
  decode_gate_up_output_idx(std::numeric_limits<unsigned>::max()),
  decode_activation_output_idx(std::numeric_limits<unsigned>::max()) {}

void Lfm2MoELayer::finalize(nntrainer::InitLayerContext &context) {

  // 1. Validate input/output dimensions
  NNTR_THROW_IF(context.getNumInputs() < 1 || context.getNumInputs() > 2,
                std::invalid_argument)
    << "MoE layer takes one input, or two when the router reads its own";
  const std::string &router_type = std::get<props::RouterType>(moe_props);
  NNTR_THROW_IF(router_type != "sigmoid_bias" && router_type != "softmax_scale",
                std::invalid_argument)
    << "MoE router_type must be sigmoid_bias or softmax_scale, got "
    << router_type;
  softmax_router = router_type == "softmax_scale";
  in_norm = std::get<props::InNorm>(moe_props).get();
  router_norm = std::get<props::RouterNorm>(moe_props).get();
  out_norm = std::get<props::OutNorm>(moe_props).get();
  out_add = std::get<props::OutAdd>(moe_props).get();
  out_to_host = std::get<props::OutToHost>(moe_props).get();
  NNTR_THROW_IF(out_add && context.getNumInputs() != 2, std::invalid_argument)
    << "MoE layer: out_add takes the dense branch's output as input 1";
  NNTR_THROW_IF(router_norm && !out_add && context.getNumInputs() == 2,
                std::invalid_argument)
    << "MoE layer: router_norm norms the input for the router itself; a "
       "second input is the other way to give the router its own rows";

  auto &weight_regularizer =
    std::get<nntrainer::props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<nntrainer::props::WeightRegularizerConstant>(*layer_impl_props);
  auto &weight_initializer =
    std::get<nntrainer::props::WeightInitializer>(*layer_impl_props);
  auto &weight_decay =
    std::get<nntrainer::props::WeightDecay>(*layer_impl_props);

  // 2. Set output dimensions (same as input)
  const auto &in_dim = context.getInputDimensions()[SINGLE_INOUT_IDX];
  NNTR_THROW_IF(context.getNumInputs() == 2 &&
                  context.getInputDimensions()[ROUTER_IN_IDX] != in_dim,
                std::invalid_argument)
    << "MoE router input must have the experts' input shape";
  const bool is_nchw = context.getFormat() == nntrainer::Tformat::NCHW;
  std::vector<nntrainer::TensorDim> output_dims(1);
  output_dims[SINGLE_INOUT_IDX] = in_dim;
  context.setOutputDimensions(output_dims);

  // 3. Get MoE properties
  num_experts = std::get<props::NumExperts>(moe_props).get();
  topk = std::get<props::NumExpertsPerToken>(moe_props).get();

  // Accelerator layers with a nonzero cache share request virtual weights:
  // the loader records file offsets, and the shared LRU loads experts into
  // accelerator slots on demand. CPU layers keep resident weights.
  trace_layer = nntrainer::HtpExpertPool::nextLayerOrdinal();
  cache_per_layer = expertCacheCapacity(
    num_experts, std::get<props::CacheExperts>(moe_props).get());
  experts_virtual =
    cache_per_layer != 0 &&
    context.getComputeEngineType() != ml::train::LayerComputeEngine::CPU;
  nntrainer::HtpExpertPool *pool = nullptr;
  if (experts_virtual) {
#ifdef ENABLE_HEXKL
    pool = &requireHtpExpertPool(nntrainer::get_htp_ops());
#else
    throw std::runtime_error("virtual MoE experts require an HTP build");
#endif
    pool->lru.addLayer(this, cache_per_layer);
  }
  const unsigned int intermediate_size =
    std::get<nntrainer::props::Unit>(moe_props).get();
  const unsigned int hidden_size = in_dim.width(); // Feature dimension
  if (pool) {
    const auto wdt = context.getWeightDataType();
    ExpertFileDesc shape{};
    shape.K = shape.N_out = hidden_size;
    shape.inter = intermediate_size;
    shape.w_bits = wdt == nntrainer::Tdatatype::TL2CX_WH   ? HTP_TL2_WEIGHT_BITS
                   : wdt == nntrainer::Tdatatype::QS2CX_WH ? 2u
                                                           : 4u;
    shape.affine = wdt == nntrainer::Tdatatype::QS4CX_WH_AFFINE;
    pool->class_capacity[nntrainer::HtpExpertPool::slotClass(shape)] +=
      cache_per_layer;
  }

  // activation function
  if (std::get<props::MoEActivation>(moe_props).empty()) {
    throw std::runtime_error("Activation type is not set for LFM2 MoE layer");
  }
  switch (context.getActivationDataType()) {
  case ml::train::TensorDim::DataType::FP32:
    acti_func.setActiFunc<float>(
      std::get<props::MoEActivation>(moe_props).get());
    break;
  default:
    throw std::runtime_error(
      "Unsupported activation data type for LFM2 MoE layer");
  }
  // The expert FFN is gate_up -> act(gate)*up -> down with the deterministic
  // kernels (swiglu_det.h here, hvx_swiglu_det.h on the DSP), not acti_func:
  // silu for LFM2, gelu_tanh for Gemma-4 (doc 55), nothing else.
  switch (std::get<props::MoEActivation>(moe_props).get()) {
  case nntrainer::ActivationType::ACT_SWISH:
    gelu_act = false;
    break;
  case nntrainer::ActivationType::ACT_TANH_GELU:
    gelu_act = true;
    break;
  default:
    throw std::runtime_error(
      "MoE layer moe_activation must be swish or tanh_gelu");
  }

  // 3b. The block's norms' gammas, where the file has them: the experts'
  // input norm and the router's before the router weight, the output's
  // after the experts (below).
  const nntrainer::TensorDim gamma_dim(
    1, 1, 1, hidden_size,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     nntrainer::TensorDim::DataType::FP32));
  auto request_gamma = [&](const char *name) {
    return context.requestWeight(
      gamma_dim, nntrainer::props::InitializerInfo::Enum::NONE,
      nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, name, true);
  };
  if (in_norm)
    in_gamma_idx = request_gamma("in_norm_gamma");
  if (router_norm)
    router_gamma_idx = request_gamma("router_norm_gamma");

  // 4. Initialize gate layer; SR15 keeps its affine INT4 grid packed.
  nntrainer::TensorDim gate_dim(
    1, is_nchw ? 1 : num_experts, is_nchw ? hidden_size : 1,
    is_nchw ? num_experts : hidden_size,
    nntrainer::TensorDim::TensorType(
      context.getFormat(), router_q4_
                             ? nntrainer::TensorDim::DataType::QS4CX_WH_AFFINE
                             : nntrainer::TensorDim::DataType::FP32),
    is_nchw ? 0b0011 : 0b0101);

  gate_idx = context.requestWeight(
    gate_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "gate", true);

  // 4b. Expert bias used only for top-k selection. Shape [1,1,1,E], FP32.
  nntrainer::TensorDim expert_bias_dim(
    1, 1, 1, num_experts,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     nntrainer::TensorDim::DataType::FP32),
    0b0001);

  expert_bias_idx = context.requestWeight(
    expert_bias_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "expert_bias", false);

  // 5. Initialize expert weights
  expert_gate_up_proj_indices.reserve(num_experts);
  expert_down_proj_indices.reserve(num_experts);

  nntrainer::TensorDim expert_gate_up_dim(
    1, is_nchw ? 1 : 2 * intermediate_size, is_nchw ? hidden_size : 1,
    is_nchw ? 2 * intermediate_size : hidden_size,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);

  nntrainer::TensorDim expert_down_dim(
    1, is_nchw ? 1 : hidden_size, is_nchw ? intermediate_size : 1,
    is_nchw ? hidden_size : intermediate_size,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);

  for (unsigned int i = 0; i < num_experts; ++i) {
    // Fused gate and up projection. Each output row is [gate | up].
    expert_gate_up_proj_indices.push_back(context.requestWeight(
      expert_gate_up_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay,
      "expert_gate_up_" + std::to_string(i), false, experts_virtual));

    // Down projection
    expert_down_proj_indices.push_back(context.requestWeight(
      expert_down_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay,
      "expert_down_" + std::to_string(i), false, experts_virtual));
  }

  if (out_norm)
    out_gamma_idx = request_gamma("out_norm_gamma");
  // the residual_add this replaces: its gamma, then the block scalar
  if (out_add) {
    add_gamma_idx = request_gamma("add_gamma");
    add_scale_idx = context.requestWeight(
      nntrainer::TensorDim(
        1, 1, 1, 1,
        nntrainer::TensorDim::TensorType(context.getFormat(),
                                         nntrainer::TensorDim::DataType::FP32)),
      nntrainer::props::InitializerInfo::Enum::NONE,
      nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "add_scalar", true);
  }
  // The normed rows for the experts and the router (the norm kernel is not
  // in place); the experts' scratch doubles as the output norm's and the
  // epilogue's.
  if (in_norm || out_norm || out_add) {
    experts_in_idx = context.requestTensor(
      in_dim, "experts_in", nntrainer::Initializer::NONE, false,
      nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
  }
  if (router_norm) {
    router_in_idx = context.requestTensor(
      in_dim, "router_in", nntrainer::Initializer::NONE, false,
      nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
  }

  // 6. Request intermediate tensor for router logits [batch*seq, 1, 1, E]
  const unsigned batch_size = in_dim.batch();
  const unsigned seq_len = in_dim.height();
  const unsigned total_tokens = batch_size * seq_len;

  router_logits_idx =
    context.requestTensor({total_tokens, 1, 1, num_experts}, "router_logits",
                          nntrainer::Initializer::NONE, false,
                          nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
  decode_expert_output_idx =
    context.requestTensor({1, 1, 1, hidden_size}, "decode_expert_output",
                          nntrainer::Initializer::NONE, false,
                          nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
  decode_gate_up_output_idx = context.requestTensor(
    {1, 1, 1, 2 * intermediate_size}, "decode_gate_up_output",
    nntrainer::Initializer::NONE, false,
    nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
  decode_activation_output_idx = context.requestTensor(
    {1, 1, 1, intermediate_size}, "decode_activation_output",
    nntrainer::Initializer::NONE, false,
    nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
}

void Lfm2MoELayer::glu(unsigned int n, float *dst, const float *gate,
                       const float *up) const {
  if (!gelu_act) {
    swiglu_det(n, dst, gate, up);
    return;
  }
  for (unsigned int j = 0; j < n; ++j)
    dst[j] = geglu_det_one(gate[j], up[j]);
}

void Lfm2MoELayer::buildExpertAssignments(
  const nntrainer::Tensor &router_logits, const nntrainer::Tensor &expert_bias,
  unsigned int total_tokens,
  std::vector<std::vector<std::pair<unsigned, float>>> &expert_assignments,
  std::vector<int> *extra_top_k) {

  const float *logits = router_logits.getData<float>();
  const float *bias = expert_bias.getData<float>();
  // Sorting the top-(k + EXTRA_TOPK) prefix instead of the top-k one leaves
  // the top-k set and its order what they were; only the LRU hint reads
  // the rest.
  const unsigned extended =
    extra_top_k ? std::min<unsigned>(topk + EXTRA_TOPK, num_experts) : topk;

  // Compute each token independently, then append in token order so every
  // expert's rows and scatter accumulation order remain unchanged.
  const bool parallel = total_tokens >= 128;
  std::vector<int> selected(parallel ? size_t(total_tokens) * extended : 0);
  std::vector<float> weights(parallel ? size_t(total_tokens) * topk : 0);
  const auto route = [&](unsigned begin, unsigned end) {
    std::vector<float> sig(num_experts);
    std::vector<std::pair<float, int>> scored(num_experts);
    for (unsigned int i = begin; i < end; ++i) {
      const float *lrow = logits + static_cast<size_t>(i) * num_experts;

      if (softmax_router) {
        // softmax_scale (Gemma-4): softmax in f32, selection and weights from
        // the same scores; the [E] weight scales the top-k weights below.
        const float mx = *std::max_element(lrow, lrow + num_experts);
        float sum = 0.0f;
        for (unsigned int e = 0; e < num_experts; ++e)
          sum += sig[e] = std::exp(lrow[e] - mx);
        for (unsigned int e = 0; e < num_experts; ++e)
          scored[e] = {sig[e] /= sum, static_cast<int>(e)};
      } else {
        // sigmoid scores (bias-free) and biased scores used only for selection
        for (unsigned int e = 0; e < num_experts; ++e) {
          const float s = 1.0f / (1.0f + std::exp(-lrow[e]));
          sig[e] = s;
          scored[e] = {s + bias[e], static_cast<int>(e)};
        }
      }

      // top-k experts by (sigmoid + bias); an exact tie goes to the lower
      // index -- a total order, so the selection does not depend on the
      // library's partial_sort, and the HTP router's rule (m1_ops_det.h)
      std::partial_sort(
        scored.begin(), scored.begin() + extended, scored.end(),
        [](const std::pair<float, int> &a, const std::pair<float, int> &b) {
          return a.first > b.first ||
                 (a.first == b.first && a.second < b.second);
        });

      // routing weights come from the bias-free sigmoid scores
      float wsum = 0.0f;
      for (unsigned int k = 0; k < topk; ++k)
        wsum += sig[scored[k].second];

      const float inv = softmax_router   ? 1.0f / wsum
                        : NORM_TOPK_PROB ? (1.0f / (wsum + 1e-6f))
                                         : 1.0f;
      for (unsigned int k = 0; k < topk; ++k) {
        const int expert_idx = scored[k].second;
        const float weight =
          sig[expert_idx] * inv *
          (softmax_router ? bias[expert_idx] : ROUTED_SCALING_FACTOR);
        if (parallel)
          weights[size_t(i) * topk + k] = weight;
        else
          expert_assignments[expert_idx].emplace_back(i, weight);
      }
      if (parallel) {
        for (unsigned int k = 0; k < extended; ++k)
          selected[size_t(i) * extended + k] = scored[k].second;
      } else if (extra_top_k) {
        for (unsigned int k = 0; k < extended; ++k)
          extra_top_k->push_back(scored[k].second);
      }
    }
  };
  if (parallel) {
    auto &tm = nntrainer::ThreadManager::Global();
    const unsigned jobs = std::min(total_tokens, tm.getComputeThreadCount());
    tm.parallel_for(0, jobs, [&](size_t job) {
      route(total_tokens * job / jobs, total_tokens * (job + 1) / jobs);
    });
    for (unsigned i = 0; i < total_tokens; ++i)
      for (unsigned k = 0; k < topk; ++k)
        expert_assignments[selected[size_t(i) * extended + k]].emplace_back(
          i, weights[size_t(i) * topk + k]);
    if (extra_top_k)
      extra_top_k->insert(extra_top_k->end(), selected.begin(), selected.end());
  } else {
    route(0, total_tokens);
  }
}

/** @brief One virtual expert from the model file into the
 *  accelerator's slot pool, keyed by its two weight tensors. The dims are
 *  the kernel's: gate_up is [K, 2 * inter], down [inter, N_out]. */
static ExpertFileDesc expertDesc(nntrainer::Tensor &gu, nntrainer::Tensor &dn) {
  const auto dt = gu.getDataType();
  if ((dt != nntrainer::Tdatatype::QS4CX_WH &&
       dt != nntrainer::Tdatatype::QS2CX_WH &&
       dt != nntrainer::Tdatatype::TL2CX_WH &&
       dt != nntrainer::Tdatatype::QS4CX_WH_AFFINE) ||
      dn.getDataType() != dt) {
    throw std::runtime_error(
      "Expert caching on an accelerator engine needs QS4CX_WH, "
      "QS2CX_WH, TL2CX_WH or QS4CX_WH_AFFINE expert weights (the file's "
      "bytes go to the DSP as they are); this model's are not");
  }
  return ExpertFileDesc{&gu,
                        &dn,
                        gu.getFd(),
                        gu.getFileOffset(),
                        dn.getFileOffset(),
                        static_cast<unsigned int>(gu.height()),
                        static_cast<unsigned int>(dn.height()),
                        static_cast<unsigned int>(dn.width()),
                        dt == nntrainer::Tdatatype::TL2CX_WH
                          ? HTP_TL2_WEIGHT_BITS
                        : dt == nntrainer::Tdatatype::QS2CX_WH ? 2u
                                                               : 4u,
                        dt == nntrainer::Tdatatype::QS4CX_WH_AFFINE};
}

static void loadVirtualExpert(nntrainer::ComputeOps *ops,
                              const ExpertFileDesc &d, bool at_load) {
  if (!ops->register_qs4cx_wh_expert_file(d, at_load)) {
    throw std::runtime_error(
      "this engine cannot load a virtual expert from the model file "
      "(register_qs4cx_wh_expert_file); check the model expert weight format");
  }
}

/** Publish startup residency only after every batch member is registered.
 * A failed read may leave a backend subset registered; retry is idempotent and
 * the LRU never advertises the unregistered remainder as cache hits.
 */
static void preloadNativeExpertBatch(nntrainer::ComputeOps *ops,
                                     const std::vector<ExpertFileDesc> &descs) {
  auto *pool = &requireHtpExpertPool(ops);
  std::vector<ExpertFileDesc> missing;
  std::vector<nntrainer::ExpertLru::Key> need;
  for (const auto &d : descs) {
    need.push_back(d.key_gu);
    if (!pool->lru.resident(d.key_gu))
      missing.push_back(d);
  }
  if (!missing.empty() &&
      !ops->register_qs4cx_wh_expert_files(missing, /*at_load=*/true))
    throw std::runtime_error("accelerator cannot preload native TL2 experts");
  if (!need.empty())
    pool->lru.acquire(
      need, [](nntrainer::ExpertLru::Key) {},
      [](nntrainer::ExpertLru::Key) {
        throw std::logic_error("native preload exceeded expert capacity");
      });
}

bool Lfm2MoELayer::preloadExperts(nntrainer::RunLayerContext &context) {
  if (!experts_virtual)
    return true;
  auto *ops = context.getComputeOps();
  if (ops == nullptr) {
    throw std::runtime_error("virtual MoE experts need the layer's engine to "
                             "provide ComputeOps; none is registered");
  }
  auto *pool = &requireHtpExpertPool(ops);
  std::vector<ExpertFileDesc> descs;
  descs.reserve(num_experts);
  for (unsigned int e = 0; e < num_experts; ++e)
    descs.push_back(
      expertDesc(context.getWeight(expert_gate_up_proj_indices[e]),
                 context.getWeight(expert_down_proj_indices[e])));
  const size_t limit = std::min(
    pool->lru.capacity(),
    static_cast<size_t>(nntrainer::htpRuntimeConfig().moe.preload_experts));
  if (nntrainer::htpRuntimeConfig().moe.descriptor_trace && !descs.empty())
    for (unsigned int e = 0; e < std::min<unsigned int>(2u, num_experts); ++e)
      std::fprintf(stderr,
                   "[MoE-DESC] layer=%u expert=%u K=%u inter=%u N_out=%u "
                   "w_bits=%u affine=%d off_gu=%llu off_dn=%llu\n",
                   trace_layer, e, descs[e].K, descs[e].inter, descs[e].N_out,
                   descs[e].w_bits, descs[e].affine ? 1 : 0,
                   (unsigned long long)descs[e].off_gu,
                   (unsigned long long)descs[e].off_dn);
  // Map all slots before FC registration, independently of weight reads:
  // the first layer of each slot size reserves that size's share.
  if (!descs.empty() &&
      pool->class_reserved.insert(nntrainer::HtpExpertPool::slotClass(descs[0]))
        .second) {
    pool->slots_reserved +=
      pool->class_capacity[nntrainer::HtpExpertPool::slotClass(descs[0])];
    ops->reserve_qs4cx_wh_expert_slots(pool->slots_reserved, &descs[0]);
  }
  if (pool->layers.empty()) {
    pool->owner = ops;
    if (descs.empty())
      ops->reserve_qs4cx_wh_expert_slots(pool->lru.capacity(), nullptr);
    std::fprintf(stderr, "[MoE] preload_experts=%zu cache_slots=%zu\n", limit,
                 pool->lru.capacity());
  } else if (pool->owner != ops) {
    pool->owner = nullptr;
  }
  if (expert_layer_slot < 0) {
    expert_layer_slot = static_cast<int>(pool->layers.size());
    pool->layers.push_back(descs);
  }

  std::vector<nntrainer::ExpertLru::Key> need;
  std::unordered_map<nntrainer::ExpertLru::Key, unsigned int> expert_of;
  const size_t room = limit > pool->lru.size() ? limit - pool->lru.size() : 0;
  for (unsigned int e = 0; e < num_experts && need.size() < room; ++e) {
    need.push_back(descs[e].key_gu);
    expert_of[descs[e].key_gu] = e;
  }
  if (!need.empty()) {
    const bool batch_native = descs.front().K == 2048u &&
                              descs.front().inter == 1536u &&
                              descs.front().N_out == 2048u &&
                              descs.front().w_bits == HTP_TL2_WEIGHT_BITS;
    if (batch_native) {
      std::vector<ExpertFileDesc> preload_batch;
      for (auto key : need)
        preload_batch.push_back(descs[expert_of[key]]);
      preloadNativeExpertBatch(ops, preload_batch);
    } else {
      pool->lru.acquire(
        need,
        [&](nntrainer::ExpertLru::Key k) {
          loadVirtualExpert(ops, descs[expert_of[k]], /*at_load=*/true);
        },
        [](nntrainer::ExpertLru::Key) {
          throw std::logic_error("preloadExperts evicted an expert: the pool "
                                 "filled past its capacity");
        });
    }
  }
  return need.size() == num_experts;
}

/** @brief dst = rms_norm(src) * gamma over every row of src (its width),
 *  the rms_norm layer's own kernel; src and dst must not alias. */
/** @brief The softmax router's selection as the accelerator returns it
 *  (ComputeOps::router_logits_fp32 with top_k): what to ask for, and the
 *  answer. */
struct AcceleratedRoute {
  unsigned int top_k = 0;
  unsigned int n_sel = 0;
  const float *scale = nullptr; /**< the [E] per-expert scale */
  std::vector<unsigned int> sel;
  std::vector<float> weight;
};

static void normRows(const nntrainer::Tensor &src, nntrainer::Tensor &dst,
                     const nntrainer::Tensor &gamma, float eps) {
  NNTR_THROW_IF(src.getDataType() != ml::train::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "MoE layer: in_norm / router_norm / out_norm are FP32 only";
  const unsigned int width = src.width();
  if (parallelRmsNormRows(src, dst, gamma, eps))
    return;
  nntrainer::rms_norm_wrt_width_fp32_intrinsic(
    src.getData<float>(), dst.getData<float>(), src.size() / width, width, eps);
  dst.multiply_i(gamma);
}

#if defined(__aarch64__) && defined(ENABLE_HEXKL)
/** Use the existing CPU-normalized LFM rows and CPU-resident FP32 router
 * weights. No backend ownership is published: the following plain MoE call
 * stages its input normally and consumes no new router handoff ticket. */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
static bool
lfmRouterOnArm(const nntrainer::Tensor &rows, const nntrainer::Tensor &weights,
               nntrainer::Tensor &logits, nntrainer::RunLayerContext &context,
               const std::vector<unsigned int> &gate_up_indices,
               const std::vector<unsigned int> &down_indices) {
  const auto fp32 = nntrainer::Tdatatype::FP32;
  const unsigned int count = rows.height();
  if ((count != 1024 && count != 2048) || rows.batch() != 1 ||
      rows.channel() != 1 || rows.width() != 2048 ||
      rows.getFormat() != nntrainer::Tformat::NCHW ||
      rows.getDataType() != fp32 || weights.batch() != 1 ||
      weights.channel() != 1 || weights.height() != 2048 ||
      weights.width() != 64 || weights.getDataType() != fp32 ||
      weights.getFormat() != rows.getFormat() || logits.batch() != count ||
      logits.channel() != 1 || logits.height() != 1 || logits.width() != 64 ||
      logits.getDataType() != fp32 || logits.getFormat() != rows.getFormat() ||
      !rows.getContiguous() || !weights.getContiguous() ||
      !logits.getContiguous() || gate_up_indices.size() != 64 ||
      down_indices.size() != 64)
    return false;
  auto *ops = rows.getOps();
  if (ops == nullptr || logits.getOps() != ops ||
      !ops->supports_router_logits_fp32() ||
      !ops->supports_gemm_qs4cx_moe_layer_fp32() ||
      ops != nntrainer::get_htp_ops())
    return false;
  const auto &row_context = rows.getContextData();
  const auto &weight_context = weights.getContextData();
  const auto &logit_context = logits.getContextData();
  if ((row_context && weight_context && row_context != weight_context) ||
      (row_context && logit_context && row_context != logit_context))
    return false;
  for (size_t e = 0; e < 64; ++e) {
    if (context.getWeight(gate_up_indices[e]).getDataType() !=
          nntrainer::Tdatatype::TL2CX_WH ||
        context.getWeight(down_indices[e]).getDataType() !=
          nntrainer::Tdatatype::TL2CX_WH)
      return false;
  }
  const auto &moe_config = nntrainer::htpRuntimeConfig().moe;
  const bool enabled = moe_config.lfm2_router_arm && !moe_config.router_u8i8;
  if (!enabled)
    return false;
  const float *input = rows.getData<float>();
  const float *weight = weights.getData<float>();
  float *output = logits.getData<float>();
  const size_t out_bytes = static_cast<size_t>(count) * 64 * sizeof(float);
  const uintptr_t out_address = reinterpret_cast<uintptr_t>(output);
  const auto disjoint = [out_address, out_bytes](const float *p, size_t size) {
    const uintptr_t address = reinterpret_cast<uintptr_t>(p);
    return out_address >= address ? out_address - address >= size
                                  : address - out_address >= out_bytes;
  };
  if (!input || !weight || !output ||
      !disjoint(input, static_cast<size_t>(count) * 2048 * sizeof(float)) ||
      !disjoint(weight, 2048 * 64 * sizeof(float)))
    return false;
  if (!lfm2RouterArmWeightSupported(weight))
    return false;
  std::atomic<bool> finite{true};
  nntrainer::ThreadManager::Global().parallel_for(
    0, count / 4, [input, weight, output, &finite](size_t task) {
      if (!lfm2RouterArmBlock(input + task * 4 * 2048, weight,
                              output + task * 4 * 64))
        finite.store(false, std::memory_order_relaxed);
    });
  // Non-finite outputs can encode different NaN payloads on ARM and HVX.
  // All jobs have joined; false lets the existing router overwrite the entire
  // output before selection. No handoff stamp has been created by this path.
  return finite.load(std::memory_order_relaxed);
}
#endif

/** @brief The router's logits on the accelerator at prefill: its own norm
 *  (gamma, or nullptr) folded into the call over the raw rows, so the CPU
 *  neither norms the router's copy nor runs the M x K x E dot (doc 57
 *  section 5 step 5). False when the backend has no such call, at a decode
 *  row (one FastRPC call costs more than the row's dot), or at a width the
 *  kernel does not take; the caller then norms and dots as before. */
static bool routerLogitsOnAccelerator(
  const nntrainer::Tensor &rows, const nntrainer::Tensor &gate_weights,
  const float *gamma, float eps, unsigned int total_tokens,
  unsigned int hidden_size, unsigned int num_experts, nntrainer::Tensor &logits,
  AcceleratedRoute *route) {
  auto *ops = rows.getOps();
  const auto f32 = ml::train::TensorDim::DataType::FP32;
  if (total_tokens <= 1 || ops == nullptr ||
      !ops->supports_router_logits_fp32() || num_experts % 32 != 0 ||
      num_experts > 128 || rows.getDataType() != f32 ||
      gate_weights.getDataType() != f32)
    return false;
  if (route) {
    route->sel.resize(static_cast<size_t>(total_tokens) * route->n_sel);
    route->weight.resize(static_cast<size_t>(total_tokens) * route->top_k);
  }
  ops->router_logits_fp32(
    total_tokens, hidden_size, num_experts, rows.getData<float>(), gamma, eps,
    gate_weights.getData<float>(), logits.getData<float>(),
    route ? route->top_k : 0u, route ? route->n_sel : 0u,
    route ? route->scale : nullptr, route ? route->sel.data() : nullptr,
    route ? route->weight.data() : nullptr);
  return true;
}

/** @brief Route packed router weights through the layer's accelerator.
 * Router scratch tensors can have CPU context even in an HTP MoE layer. */
static void routerDot(nntrainer::RunLayerContext &context,
                      nntrainer::Tensor &rows, nntrainer::Tensor &weight,
                      nntrainer::Tensor &logits, unsigned tokens,
                      unsigned hidden, unsigned experts) {
  if (weight.getDataType() != nntrainer::Tdatatype::QS4CX_WH_AFFINE) {
    rows.dot(weight, logits);
    return;
  }
  auto *ops = context.getComputeOps();
  if (!ops || !ops->supports_gemm_qs4cx_accel_fp32() ||
      !ops->set_fc_wh_source(weight.getData(), hidden, experts, true))
    throw std::runtime_error("Packed LFM2 router requires HTP FC support");
  ops->gemm_qs4cx_accel_fp32(weight.getData<char>(), weight.getScale<float>(),
                             rows.getData<float>(), logits.getData<float>(),
                             tokens, experts, hidden);
  ops->sync_to_host(logits.getData(), tokens * experts * sizeof(float));
}

/** @brief buildExpertAssignments' output from the accelerator's selection:
 *  token by token, its top_k experts in selection order with their
 *  weights, and the n_sel-long ranking as the prefetch hint. */
static void assignmentsFromRoute(
  const AcceleratedRoute &route, unsigned int total_tokens,
  std::vector<std::vector<std::pair<unsigned, float>>> &expert_assignments,
  std::vector<int> *extra_top_k) {
  for (unsigned int i = 0; i < total_tokens; ++i) {
    const unsigned int *sel =
      route.sel.data() + static_cast<size_t>(i) * route.n_sel;
    const float *w = route.weight.data() + static_cast<size_t>(i) * route.top_k;
    for (unsigned int k = 0; k < route.top_k; ++k)
      expert_assignments[sel[k]].emplace_back(i, w[k]);
    if (extra_top_k)
      for (unsigned int k = 0; k < route.n_sel; ++k)
        extra_top_k->push_back(static_cast<int>(sel[k]));
  }
}

void Lfm2MoELayer::addEpilogue(nntrainer::RunLayerContext &context,
                               const nntrainer::Tensor &raw,
                               const nntrainer::Tensor &x2,
                               nntrainer::Tensor &out,
                               nntrainer::Tensor &scratch, float eps) {
  // residual_add's own CPU order: the sum, the norm into out, the gamma,
  // the residual, the scalar
  scratch.copyData(out);
  scratch.add_i(x2);
  normRows(scratch, out, context.getWeight(add_gamma_idx), eps);
  out.add_i(raw);
  out.multiply_i(context.getWeight(add_scale_idx).getValue<float>(0, 0, 0, 0));
}

void Lfm2MoELayer::forwarding(nntrainer::RunLayerContext &context,
                              bool training) {
  /** M0 (doc 43 §5): ARM-side wall timer around the whole layer -- this
   * profile is invisible to the HTP profile's host column by design, and
   * in the htp run the one dispatched layer's figure includes its
   * registration (once, first forward) on top of dispatch. */
  const bool m0_profile = nntrainer::htpRuntimeConfig().moe.profile;
  const auto m0_t0 = std::chrono::steady_clock::now();
  g_m0_on = m0_profile;
  g_m0.reset();

  const float norm_eps = std::get<nntrainer::props::Epsilon>(moe_props).get();
  nntrainer::Tensor &raw_input = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  // The folded norms: the experts and the router read the layer's own
  // normed copies of the input.
  nntrainer::Tensor &input =
    in_norm ? context.getTensor(experts_in_idx) : raw_input;
  nntrainer::Tensor &router_in = router_norm ? context.getTensor(router_in_idx)
                                 : (context.getNumInputs() == 2 && !out_add)
                                   ? context.getInput(ROUTER_IN_IDX)
                                   : input;
  if (in_norm)
    normRows(raw_input, input, context.getWeight(in_gamma_idx), norm_eps);

  nntrainer::Tensor &router_logits = context.getTensor(router_logits_idx);

  const unsigned batch_size = input.batch();
  const unsigned seq_len = input.height();
  const unsigned hidden_size = input.width();
  const unsigned total_tokens = batch_size * seq_len;

  // routing, part one: the logits on the accelerator (its norm folded in,
  // and for the softmax router its selection too) or the CPU's norm of
  // the router's copy for the dot below
  AcceleratedRoute route;
  route.top_k = topk;
  route.n_sel = topk;
  route.scale = context.getWeight(expert_bias_idx).getData<float>();
  const bool router_on_accel = routerLogitsOnAccelerator(
    router_norm ? raw_input : router_in, context.getWeight(gate_idx),
    router_norm ? context.getWeight(router_gamma_idx).getData<float>()
                : nullptr,
    norm_eps, total_tokens, hidden_size, num_experts, router_logits,
    softmax_router ? &route : nullptr);
  if (router_norm && !router_on_accel)
    normRows(raw_input, router_in, context.getWeight(router_gamma_idx),
             norm_eps);

  {
    M0Timer t(&g_m0.setup);
    // reshape input: [B,1,S,H] -> [B*S,1,1,H]
    input.reshape({total_tokens, 1, 1, hidden_size});
    router_in.reshape({total_tokens, 1, 1, hidden_size});

    // reshape output: [B,1,S,H] -> [B*S,1,1,H]
    output.reshape({total_tokens, 1, 1, hidden_size});
    output.setZero();
  }

  // routing: raw logits -> top-k selection (props::RouterType)
  nntrainer::Tensor &gate_weights = context.getWeight(gate_idx);
  nntrainer::Tensor &expert_bias = context.getWeight(expert_bias_idx);
  if (!router_on_accel) {
    M0Timer t(&g_m0.router);
    routerDot(context, router_in, gate_weights, router_logits, total_tokens,
              hidden_size, num_experts);
  }

  std::vector<std::vector<std::pair<unsigned, float>>> expert_assignments(
    num_experts);
  size_t max_assigned_tokens = 0;
  {
    M0Timer t(&g_m0.topk);
    if (router_on_accel && softmax_router)
      assignmentsFromRoute(route, total_tokens, expert_assignments, nullptr);
    else
      buildExpertAssignments(router_logits, expert_bias, total_tokens,
                             expert_assignments);

    for (const auto &assignments : expert_assignments)
      max_assigned_tokens = std::max(max_assigned_tokens, assignments.size());
  }

  nntrainer::Tensor prefill_token_input;
  nntrainer::Tensor prefill_expert_output;
  nntrainer::Tensor prefill_gate_up_output;
  nntrainer::Tensor prefill_activation_output;
  ExpertWorkspace workspace{
    nullptr,
    &context.getTensor(decode_expert_output_idx),
    &context.getTensor(decode_gate_up_output_idx),
    &context.getTensor(decode_activation_output_idx),
  };
  if (max_assigned_tokens > 1) {
    M0Timer t(&g_m0.wksp);
    const unsigned int workspace_tokens =
      static_cast<unsigned int>(max_assigned_tokens);
    const unsigned int intermediate_size =
      std::get<nntrainer::props::Unit>(moe_props).get();
    prefill_token_input = nntrainer::Tensor(1, 1, workspace_tokens, hidden_size,
                                            input.getTensorType());
    prefill_expert_output = nntrainer::Tensor(
      workspace_tokens, 1, 1, hidden_size, output.getTensorType());
    prefill_gate_up_output = nntrainer::Tensor(
      1, 1, workspace_tokens, 2 * intermediate_size, input.getTensorType());
    prefill_activation_output = nntrainer::Tensor(
      1, 1, workspace_tokens, intermediate_size, input.getTensorType());
    // These are locally-constructed Tensors, not context-requested ones, so
    // they carry no ContextData of their own and dispatch would otherwise
    // silently fall back to the CPU table (checkContextCompatibility is
    // permissive when either side lacks ContextData) regardless of the
    // weights' engine. Inherit input's now, before any dot() call below --
    // unlike the kernel-output usage inheritContextTo's docstring warns
    // about, these tensors are already fully shaped, so there is no
    // CREATE_IF_EMPTY_DIMS reallocation to race.
    input.inheritContextTo(prefill_token_input);
    input.inheritContextTo(prefill_expert_output);
    input.inheritContextTo(prefill_gate_up_output);
    input.inheritContextTo(prefill_activation_output);
    workspace = {&prefill_token_input, &prefill_expert_output,
                 &prefill_gate_up_output, &prefill_activation_output};
  }

  // Serial outer loop: dot() parallelizes internally through ThreadManager.
  // Nesting another parallel_for here can deadlock on its non-recursive
  // execution mutex, regardless of the expert weight dtype.
  for (unsigned int expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
    const auto &assignments = expert_assignments[expert_idx];
    if (assignments.empty())
      continue;

    compute_expert_forward(
      input, output, assignments,
      context.getWeight(expert_gate_up_proj_indices[expert_idx]),
      context.getWeight(expert_down_proj_indices[expert_idx]), hidden_size,
      workspace);
  }

  // reshape output: [B*S,1,1,H] -> [B,1,S,H]
  output.reshape({batch_size, 1, seq_len, hidden_size});
  if (out_norm) {
    nntrainer::Tensor &scratch = context.getTensor(experts_in_idx);
    normRows(output, scratch, context.getWeight(out_gamma_idx), norm_eps);
    output.copyData(scratch);
  }
  if (out_add)
    addEpilogue(context, raw_input, context.getInput(1), output,
                context.getTensor(experts_in_idx), norm_eps);

  if (m0_profile && total_tokens > 1) {
    const auto m0_us = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - m0_t0)
        .count());
    // [P3] `other` is deliberately not folded into a neighbour: it is the
    // part no named stage claims, and the point of this instrument is that
    // such a part is visible rather than inferred.
    const uint64_t named = g_m0.sum();
    std::cout << "[M0-PROF] moe_layer[" << g_m0_call_index.fetch_add(1)
              << "] tokens=" << total_tokens << " us=" << m0_us
              << "  setup=" << g_m0.setup << " router=" << g_m0.router
              << " topk=" << g_m0.topk << " wksp=" << g_m0.wksp
              << " gather=" << g_m0.gather << " ffn=" << g_m0.ffn
              << " route=" << g_m0.route << " scatter=" << g_m0.scatter
              << " other=" << (m0_us > named ? m0_us - named : 0) << std::endl;
  }
  g_m0_on = false;
}

/**
 * @brief The whole MoE FFN layer in one accelerator call.
 *
 * Flattens the per-expert assignment lists into the three arrays the call
 * wants and hands the routing over, so the token gather, the routing
 * multiply and the scatter-add happen on the accelerator instead of here.
 *
 * @return true when the layer was computed; false when this build or these
 *         weights cannot take the path, and the caller's per-expert loop
 *         still has to run.
 */
static bool tryMoeLayerOnAccelerator(
  const nntrainer::Tensor &input, nntrainer::Tensor &output,
  const std::vector<std::vector<std::pair<unsigned, float>>>
    &expert_assignments,
  nntrainer::RunLayerContext &context,
  const std::vector<unsigned int> &gate_up_indices,
  const std::vector<unsigned int> &down_indices, unsigned int total_tokens,
  unsigned int hidden_size, unsigned int intermediate_size,
  bool experts_virtual, const std::vector<int> *extra_top_k,
  int expert_layer_slot, bool gelu, unsigned int trace_layer,
  const float *pre_gamma, const float *post_gamma, float eps,
  const float *add_x2 = nullptr, const float *add_gamma = nullptr,
  float add_scale = 1.0f, bool *add_done = nullptr, bool out_to_host = true) {

  auto *ops = context.getComputeOps();
  if (ops == nullptr || !ops->supports_gemm_qs4cx_moe_layer_fp32()) {
    if (experts_virtual) {
      throw std::runtime_error("virtual MoE experts "
                               "have no CPU path: the layer's engine offers "
                               "no MoE layer call");
    }
    return false;
  }

  auto *pool = experts_virtual ? &requireHtpExpertPool(ops) : nullptr;
  const size_t n_experts = expert_assignments.size();
  if (n_experts == 0 || gate_up_indices.size() < n_experts ||
      down_indices.size() < n_experts) {
    return false;
  }
  std::vector<void *> gu_data(n_experts), dn_data(n_experts);
  std::vector<float *> gu_scale(n_experts), dn_scale(n_experts);
  // QS4CX_WH is the same weight with its nibbles already in HMX tiles and a
  // column sum after the scales, built by the offline quantizer so the DSP
  // does not have to convert and bake at load. Both
  // experts' halves have to agree, and every expert with every other: the
  // call takes one flag for the layer, and a model that mixed the two would
  // otherwise read half its weights with the wrong layout.
  const auto wh = nntrainer::Tdatatype::QS4CX_WH;
  const auto wh2 = nntrainer::Tdatatype::QS2CX_WH;
  const auto tl2 = nntrainer::Tdatatype::TL2CX_WH;
  const auto whaff = nntrainer::Tdatatype::QS4CX_WH_AFFINE;
  const auto plain = nntrainer::Tdatatype::QS4CX;
  const auto dt0 = context.getWeight(gate_up_indices[0]).getDataType();
  const unsigned int w_bits = dt0 == tl2 ? HTP_TL2_WEIGHT_BITS
                              : (dt0 == wh || dt0 == whaff) ? 4u
                              : (dt0 == wh2)                ? 2u
                                                            : 0u;

  // Plain QS4CX decode stays on ARM; accelerator layouts require HTP.
  if (total_tokens <= 1 && w_bits == 0u) {
    return false;
  }

  for (size_t e = 0; e < n_experts; ++e) {
    nntrainer::Tensor &gu = context.getWeight(gate_up_indices[e]);
    nntrainer::Tensor &dn = context.getWeight(down_indices[e]);
    const auto want = w_bits == 0u ? plain : dt0;
    if (gu.getDataType() != want || dn.getDataType() != want) {
      return false;
    }
    gu_data[e] = gu.getData<char>();
    dn_data[e] = dn.getData<char>();
    gu_scale[e] = gu.getScale<float>();
    dn_scale[e] = dn.getScale<float>();
  }

  // A virtual layer hands all its experts and the
  // LRU's policy to the backend at its first call (the first prefill runs
  // the layers in order), for the one-PD token's miss path: the pool's
  // slots are the same slots tracked by this LRU.
  if (experts_virtual) {
    if (pool->handed_layers.insert(&context.getWeight(gate_up_indices[0]))
          .second) {
      std::vector<ExpertFileDesc> all;
      for (size_t e = 0; e < n_experts; ++e)
        all.push_back(expertDesc(context.getWeight(gate_up_indices[e]),
                                 context.getWeight(down_indices[e])));
      ops->set_decode_moe_experts(
        all, [pool](const std::vector<const void *> &need,
                    const std::function<void(const void *)> &load,
                    const std::function<void(const void *)> &evict) {
          pool->lru.acquire(need, load, evict);
        });
      ops->set_decode_pool_grow([pool](size_t n) { pool->lru.addCapacity(n); });
    }
  }

  // Virtual experts: the ones this call routes to are made
  // resident through the shared LRU (a miss releases the least recently
  // used expert's slot and reads this one from the file into it), the
  // arrays are compacted to them, and the weight tensors' addresses stand
  // in for the data pointers the accelerator keys its handles on, with a
  // null scale to say so. Compacting changes nothing numerically: an
  // expert with no rows adds nothing, and the kernel's scatter-add order
  // over the remaining experts is what the full array gave.
  std::vector<size_t> active;
  active.reserve(n_experts);
  for (size_t e = 0; e < n_experts; ++e) {
    if (!experts_virtual || !expert_assignments[e].empty())
      active.push_back(e);
  }
  auto release = [&](nntrainer::ExpertLru::Key k) {
    if (!ops->release_qs4cx_wh_expert(k)) {
      throw std::logic_error("expert LRU evicted an expert the "
                             "accelerator does not hold");
    }
  };

  // One call's worth: make @a group resident (virtual experts), and build
  // the arrays the kernel takes for it. Grouped by expert in expert order
  // -- the layout the kernel slices by running offsets, so the order here
  // is part of the contract, not a convenience.
  std::vector<nntrainer::ExpertLru::Key> need;
  std::vector<void *> call_gu, call_dn;
  std::vector<float *> call_gus, call_dns;
  std::vector<unsigned int> row_index, row_count;
  std::vector<float> row_weight;
  auto stage = [&](const std::vector<size_t> &group) {
    need.clear();
    call_gu.clear();
    call_dn.clear();
    call_gus.clear();
    call_dns.clear();
    row_index.clear();
    row_count.clear();
    row_weight.clear();
    if (experts_virtual) {
      std::unordered_map<nntrainer::ExpertLru::Key, size_t> expert_of;
      for (size_t e : group) {
        auto *key = &context.getWeight(gate_up_indices[e]);
        need.push_back(key);
        expert_of[key] = e;
      }
      // acquire() frees every slot the misses need
      // before it names the first miss, so the misses are gathered and
      // loaded together after it: one registration round trip, not one
      // per expert. Nothing reads them before the call below.
      std::vector<ExpertFileDesc> missed;
      g_m0.misses += pool->lru.acquire(
        need,
        [&](nntrainer::ExpertLru::Key k) {
          const size_t e = expert_of[k];
          missed.push_back(expertDesc(context.getWeight(gate_up_indices[e]),
                                      context.getWeight(down_indices[e])));
        },
        release);
      if (!missed.empty() && !ops->register_qs4cx_wh_expert_files(missed)) {
        throw std::runtime_error(
          "this engine cannot load a virtual expert from the model file "
          "(register_qs4cx_wh_expert_files); check the model expert weight "
          "format");
      }
    }
    for (size_t e : group) {
      if (experts_virtual) { // the tensor's address is the key, no scale
        call_gu.push_back(&context.getWeight(gate_up_indices[e]));
        call_dn.push_back(&context.getWeight(down_indices[e]));
        call_gus.push_back(nullptr);
        call_dns.push_back(nullptr);
      } else {
        call_gu.push_back(gu_data[e]);
        call_dn.push_back(dn_data[e]);
        call_gus.push_back(gu_scale[e]);
        call_dns.push_back(dn_scale[e]);
      }
      row_count.push_back(
        static_cast<unsigned int>(expert_assignments[e].size()));
      for (const auto &pair : expert_assignments[e]) {
        row_index.push_back(pair.first);
        row_weight.push_back(pair.second);
      }
    }
  };
  if (add_done)
    *add_done = false;
  // The layer's whole output (the epilogue in the call) may stay on the
  // accelerator when the builder says so; a partial one is read here.
  auto call = [&](float *dst, bool with_add, bool retain_plain = false) {
    ops->gemm_qs4cx_moe_layer_fp32(
      call_gu, call_gus, call_dn, call_dns, row_index, row_count, row_weight,
      input.getData<float>(), dst, total_tokens, hidden_size, intermediate_size,
      hidden_size, w_bits, gelu, pre_gamma, post_gamma, eps,
      with_add ? add_x2 : nullptr, with_add ? add_gamma : nullptr, add_scale,
      out_to_host || (!with_add && !retain_plain) || total_tokens == 1);
  };

  // If all routed experts cannot fit, split them into pool-sized groups
  // and sum their zero-initialized outputs on the host. This changes fp32
  // addition order across groups; numerical equivalence is not guaranteed.
  const size_t cap = pool ? pool->lru.capacity() : 0;
  if (experts_virtual && active.size() > cap) {
    // the host adds the groups and runs the epilogue: it reads the input
    // and the dense branch, which their calls may have kept (out_to_host)
    const size_t in_bytes =
      static_cast<size_t>(total_tokens) * hidden_size * sizeof(float);
    ops->sync_to_host(const_cast<float *>(input.getData<float>()), in_bytes);
    if (add_x2)
      ops->sync_to_host(const_cast<float *>(add_x2), in_bytes);
    float *out = output.getData<float>();
    const size_t n_out = static_cast<size_t>(total_tokens) * hidden_size;
    std::vector<float> part(n_out);
    for (size_t g0 = 0; g0 < active.size(); g0 += cap) {
      const size_t g1 = std::min(active.size(), g0 + cap);
      stage(std::vector<size_t>(active.begin() + g0, active.begin() + g1));
      if (g0 == 0) {
        call(out, false);
      } else {
        call(part.data(), false);
        for (size_t i = 0; i < n_out; ++i)
          out[i] += part[i];
      }
    }
  } else {
    // Prefetch missing experts up to the configured lookahead while
    // this call runs. Each layer registers its batch before staging.
    // Protect current and later layers' resident experts, reserve slots
    // for in-flight reads, and pause queuing when the pool is full.
    const int depth =
      experts_virtual && total_tokens > 1 && expert_layer_slot >= 0
        ? expertPrefetchDepth()
        : 0;
    const bool prune_prefill =
      nntrainer::htpRuntimeConfig().moe.lfm2_prefill_prune;
    std::vector<const void *> routed_keys;
    if (prune_prefill)
      for (size_t e : active)
        routed_keys.push_back(&context.getWeight(gate_up_indices[e]));
    auto take_batch = [&] {
      const bool current =
        pool->prefetch_batches.front().first == expert_layer_slot;
      const size_t held = pool->prefetch_batches.front().second;
      pool->prefetch_batches.pop_front();
      pool->lru.unhold(held);
      std::vector<const void *> cancelled;
      const std::vector<nntrainer::ExpertLru::Key> got =
        ops->prefetch_qs4cx_wh_experts_end(
          prune_prefill && current ? &routed_keys : nullptr, &cancelled);
      for (auto k : cancelled)
        pool->prefetch_inflight.erase(k);
      for (nntrainer::ExpertLru::Key k : got)
        pool->prefetch_inflight.erase(k);
      pool->lru.acquire(
        got, [](nntrainer::ExpertLru::Key) {},
        [](nntrainer::ExpertLru::Key) {
          throw std::logic_error("expert prefetch overfilled the LRU");
        });
    };
    if (depth > 0) {
      // Layer 0 normally retires an earlier pass. An explicit timed early
      // kickoff belongs to THIS pass, so do not join its future batches.
      if (expert_layer_slot == 0 && pool->early_owner &&
          pool->early_owner != ops)
        throw std::logic_error("early prefill backend changed");
      const bool early = expert_layer_slot == 0 && pool->early_owner;
      while (!pool->prefetch_batches.empty() &&
             ((expert_layer_slot == 0 && !early) ||
              pool->prefetch_batches.front().first <= expert_layer_slot))
        take_batch();
      if (expert_layer_slot == 0 && !early) {
        pool->prefetch_next = 1;
        pool->prefetch_inflight.clear(); // every batch was taken just above
      }
      if (expert_layer_slot == 0)
        pool->early_owner = nullptr;
    }

    stage(active);

    if (depth > 0) {
      pool->prefetch_next =
        std::max(pool->prefetch_next, expert_layer_slot + 1);
      const int n_layers = static_cast<int>(pool->layers.size());
      const int last = std::min(expert_layer_slot + depth, n_layers - 1);
      for (; pool->prefetch_next <= last; ++pool->prefetch_next) {
        std::vector<nntrainer::ExpertLru::Key> pinned(need);
        for (int l = expert_layer_slot + 1; l < n_layers; ++l)
          for (const ExpertFileDesc &d : pool->layers[l])
            if (pool->lru.resident(d.key_gu))
              pinned.push_back(d.key_gu);
        std::vector<ExpertFileDesc> want;
        for (const ExpertFileDesc &d : pool->layers[pool->prefetch_next])
          if (!pool->lru.resident(d.key_gu) &&
              pool->prefetch_inflight.count(d.key_gu) == 0)
            want.push_back(d);
        if (want.empty())
          continue;
        // As many of the layer as the pool has room for now, the rest when
        // a finished layer's slots free up: at 128 experts a layer and
        // C = 12 the pool holds this layer, the next and 104 more, and a
        // whole-layer batch left those 104 slots idle, so every layer's
        // experts had one layer's time to be read in (exposed wait 5-141 ms
        // a 1024-token prefill).
        const size_t room = pool->lru.room(pinned);
        const bool part = want.size() > room;
        if (part)
          want.resize(room);
        if (want.empty() || !pool->lru.makeRoom(want.size(), pinned, release))
          break;
        pool->lru.hold(want.size());
        bool queued = false;
        try {
          queued = ops->prefetch_qs4cx_wh_experts_begin(want);
        } catch (...) {
          pool->lru.unhold(want.size());
          throw;
        }
        if (queued) {
          pool->prefetch_batches.emplace_back(pool->prefetch_next, want.size());
          for (const ExpertFileDesc &d : want)
            pool->prefetch_inflight.insert(d.key_gu);
        } else {
          pool->lru.unhold(want.size());
        }
        if (part)
          break; // the rest of this layer first, a layer later
      }
    }

    // Only a complete plain result can be consumed through a mapped view.
    // Split expert groups above are always materialized for CPU accumulation.
    const bool prefill_views =
      nntrainer::htpRuntimeConfig().htp.lfm2_prefill_views;
    const bool host_diagnostics =
      !nntrainer::htpRuntimeConfig().htp.dump_dir.empty() ||
      std::getenv("NNTR_ACT_STATS") || std::getenv("NNTR_ACT_DUMP") ||
      std::getenv("NNTR_DUMP_ROWS_DIR") || std::getenv("NNTR_DUMP_ROWS_HASH");
    bool retain_plain = prefill_views && !out_to_host &&
                        post_gamma == nullptr && add_x2 == nullptr &&
                        !host_diagnostics;
    call(output.getData<float>(), add_x2 != nullptr, retain_plain);
    if (add_done)
      *add_done = add_x2 != nullptr;
  }

  // Recency from the routing's extended top-k, token by token, so the
  // last token's likely-next experts end up most recent -- the rule
  // Lfm2CachedSlimMoELayer applies after its own pass.
  if (experts_virtual && extra_top_k != nullptr && !extra_top_k->empty()) {
    // Walked back to front, as Lfm2CachedSlimMoELayer walks it: the list
    // is rank order within each token, and refresh() makes its LAST entry
    // the most recent, so reversing puts a token's top-ranked expert --
    // the likeliest to be routed to again -- at the back.
    std::vector<nntrainer::ExpertLru::Key> recency;
    const bool compact_recency =
      nntrainer::htpRuntimeConfig().moe.lfm2_compact_recency;
    if (compact_recency && total_tokens > 1) {
      // Only the last touch of each key affects the final LRU order. The
      // old loop traverses hints backwards, so these are the first forward
      // occurrences, reversed. Keep absent keys too: refresh skips them.
      std::vector<bool> seen(n_experts, false);
      recency.reserve(n_experts);
      for (int expert : *extra_top_k) {
        if (!seen[expert]) {
          seen[expert] = true;
          recency.push_back(&context.getWeight(gate_up_indices[expert]));
          if (recency.size() == n_experts)
            break;
        }
      }
      std::reverse(recency.begin(), recency.end());
    } else {
      recency.reserve(extra_top_k->size());
      for (auto it = extra_top_k->rbegin(); it != extra_top_k->rend(); ++it)
        recency.push_back(&context.getWeight(gate_up_indices[*it]));
    }
    pool->lru.refresh(recency);
  }
  return true;
}

inline void Lfm2MoELayer::compute_expert_forward(
  const nntrainer::Tensor &input, nntrainer::Tensor &output,
  const std::vector<std::pair<unsigned, float>> &token_assignments,
  const nntrainer::Tensor &gate_up_proj, const nntrainer::Tensor &down_proj,
  unsigned int hidden_size, ExpertWorkspace &workspace) {

  if (token_assignments.empty())
    return;

  nntrainer::Tensor expert_output =
    workspace.expert_output->getSharedDataTensor(
      {static_cast<unsigned int>(token_assignments.size()), 1, 1, hidden_size},
      0, true);
  compute_expert_forward_no_critical(input, expert_output, token_assignments,
                                     gate_up_proj, down_proj, hidden_size,
                                     workspace);

  M0Timer t(&g_m0.scatter);
  nntrainer::TensorDim token_step_dim({1, 1, 1, hidden_size},
                                      output.getTensorType());
  for (size_t i = 0; i < token_assignments.size(); ++i) {
    nntrainer::Tensor token_output = output.getSharedDataTensor(
      token_step_dim, token_assignments[i].first * hidden_size, true);
    nntrainer::Tensor expert_token_output =
      expert_output.getSharedDataTensor(token_step_dim, i * hidden_size, true);
    token_output.add_i(expert_token_output);
  }
}

inline void Lfm2MoELayer::compute_expert_forward_no_critical(
  const nntrainer::Tensor &input, nntrainer::Tensor &expert_output,
  const std::vector<std::pair<unsigned, float>> &token_assignments,
  const nntrainer::Tensor &gate_up_proj, const nntrainer::Tensor &down_proj,
  unsigned int hidden_size, ExpertWorkspace &workspace) {

  const unsigned intermediate_size = gate_up_proj.width() / 2;
  const unsigned num_tokens = token_assignments.size();

  if (num_tokens == 0)
    return;

  nntrainer::TensorDim token_input_dim({1, 1, num_tokens, hidden_size},
                                       input.getTensorType());
  nntrainer::TensorDim intermediate_dim({1, 1, num_tokens, intermediate_size},
                                        input.getTensorType());
  nntrainer::TensorDim gate_up_dim({1, 1, num_tokens, 2 * intermediate_size},
                                   input.getTensorType());
  nntrainer::TensorDim token_step_dim({1, 1, 1, hidden_size},
                                      input.getTensorType());

  nntrainer::Tensor token_input;
  {
    // Timed from the serial caller, so this is the gather's wall clock
    // including the fork/join, not the sum of the workers' time.
    M0Timer t(&g_m0.gather);
    if (num_tokens == 1) {
      token_input = input.getSharedDataTensor(
        token_input_dim, token_assignments[0].first * hidden_size, true);
    } else {
      token_input =
        workspace.token_input->getSharedDataTensor(token_input_dim, 0, true);
      auto &tm = nntrainer::ThreadManager::Global();
      tm.parallel_for(0, static_cast<size_t>(num_tokens), [&](size_t i) {
        nntrainer::Tensor source = input.getSharedDataTensor(
          token_step_dim, token_assignments[i].first * hidden_size, true);
        nntrainer::Tensor target = token_input.getSharedDataTensor(
          token_step_dim, i * hidden_size, true);
        target.copyData(source);
      });
    }
  }

  nntrainer::Tensor gate_up_out =
    workspace.gate_up_output->getSharedDataTensor(gate_up_dim, 0, true);
  nntrainer::Tensor acti_out =
    workspace.activation_output->getSharedDataTensor(intermediate_dim, 0, true);

  // Keep gate_up -> SwiGLU -> down on the DSP for prefill. Decode uses
  // the two-dot path because one token cannot amortize 64-row padding.
  constexpr bool kFusedSwigluEnabled = true;
  bool expert_ffn_done = false;
  // Use the same FFN timing boundary for fused and two-dot paths.
  M0Timer m0_ffn(&g_m0.ffn);
  if (kFusedSwigluEnabled && num_tokens > 1 && !gelu_act &&
      gate_up_proj.getDataType() == nntrainer::Tdatatype::QS4CX &&
      down_proj.getDataType() == nntrainer::Tdatatype::QS4CX) {
    auto *ops = token_input.getOps();
    if (ops->supports_gemm_qs4cx_fused_swiglu_fp32()) {
      std::vector<void *> wdata = {gate_up_proj.getData<char>(),
                                   down_proj.getData<char>()};
      std::vector<float *> wscale = {gate_up_proj.getScale<float>(),
                                     down_proj.getScale<float>()};
      std::vector<unsigned int> widths = {
        static_cast<unsigned int>(gate_up_proj.width()),
        static_cast<unsigned int>(down_proj.width())};
      ops->gemm_qs4cx_fused_swiglu_fp32(
        wdata, wscale, token_input.getData<float>(),
        expert_output.getData<float>(), num_tokens, widths, hidden_size);
      expert_ffn_done = true;
    }
  }

  if (!expert_ffn_done) {
    token_input.dot(gate_up_proj, gate_up_out);

    if (num_tokens == 1) {
      glu(acti_out.width(), acti_out.getData<float>(),
          gate_up_out.getData<float>(),
          gate_up_out.getData<float>() + intermediate_size);
    } else {
      auto &tm = nntrainer::ThreadManager::Global();
      tm.parallel_for(0, static_cast<size_t>(num_tokens), [&](size_t i) {
        const unsigned int offset = acti_out.getIndex(0, 0, i, 0);
        const unsigned int gate_up_offset = gate_up_out.getIndex(0, 0, i, 0);
        glu(acti_out.width(), acti_out.getData<float>() + offset,
            gate_up_out.getData<float>() + gate_up_offset,
            gate_up_out.getData<float>() + gate_up_offset + intermediate_size);
      });
    }

    acti_out.dot(down_proj, expert_output);
  }
  m0_ffn.stop();

  {
    M0Timer t(&g_m0.route);
    for (size_t i = 0; i < num_tokens; ++i) {
      nntrainer::Tensor expert_token_output = expert_output.getSharedDataTensor(
        token_step_dim, i * hidden_size, true);
      expert_token_output.multiply_i(token_assignments[i].second);
    }
  }
}

void Lfm2MoELayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                          unsigned int from, unsigned int to,
                                          bool training) {

  /** M0: same timer as forwarding() -- the runner drives prefill through
   * this path too (from=0,to=prompt_len), decode is the total_tokens==1
   * case and stays silent under the same guard. */
  const bool m0_profile = nntrainer::htpRuntimeConfig().moe.profile;
  const bool op_time = to - from == 1 && OpTimeMoe::on();
  const auto m0_t0 = std::chrono::steady_clock::now();
  g_m0_on = m0_profile || op_time;
  g_m0.reset();

  const float norm_eps = std::get<nntrainer::props::Epsilon>(moe_props).get();
  nntrainer::Tensor &raw_input_ = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output_ = context.getOutput(SINGLE_INOUT_IDX);
  // The folded norms: the experts and the router read the layer's own
  // normed copies of the input (per batch, below).
  nntrainer::Tensor &input_ =
    in_norm ? context.getTensor(experts_in_idx) : raw_input_;
  nntrainer::Tensor &router_in_ = router_norm ? context.getTensor(router_in_idx)
                                  : (context.getNumInputs() == 2 && !out_add)
                                    ? context.getInput(ROUTER_IN_IDX)
                                    : input_;

  nntrainer::Tensor &router_logits_ = context.getTensor(router_logits_idx);
  nntrainer::Tensor &gate_weights = context.getWeight(gate_idx);
  nntrainer::Tensor &expert_bias = context.getWeight(expert_bias_idx);

  nntrainer::TensorDim input_step_dim = input_.getDim();
  nntrainer::TensorDim output_step_dim = output_.getDim();
  nntrainer::TensorDim router_logits_step_dim = router_logits_.getDim();

  input_step_dim.batch(1);
  output_step_dim.batch(1);
  router_logits_step_dim.batch(to - from);

  input_step_dim.height(to - from);
  output_step_dim.height(to - from);

  for (unsigned int b = 0; b < input_.batch(); ++b) {
    // A resident decode row: the DSP
    // runs the block's norms, router and experts in the list, so nothing
    // here is read -- checked ahead of the folded norms below
    if (to - from == 1 && input_.batch() == 1 && htpDecodeRowResident(from))
      continue;

    auto input = input_.getSharedDataTensor(
      input_step_dim, b * input_step_dim.getFeatureLen(), true);
    auto router_in = router_in_.getSharedDataTensor(
      input_step_dim, b * input_step_dim.getFeatureLen(), true);
    auto raw = raw_input_.getSharedDataTensor(
      input_step_dim, b * input_step_dim.getFeatureLen(), true);
    // With out_add the experts' input norm rides the prefill MoE call (its
    // pre_gamma), which then has the raw rows on the DSP for the
    // epilogue's residual. Decode keeps the CPU norm.
    auto *moe_ops = context.getComputeOps();
    const bool norm_on_accel = out_add && in_norm && to - from > 1 &&
                               moe_ops != nullptr &&
                               moe_ops->supports_gemm_qs4cx_moe_layer_fp32();
    if (in_norm && !norm_on_accel)
      normRows(raw, input, context.getWeight(in_gamma_idx), norm_eps);
    nntrainer::Tensor x2;
    if (out_add)
      x2 = context.getInput(1).getSharedDataTensor(
        input_step_dim, b * input_step_dim.getFeatureLen(), true);
    auto output = output_.getSharedDataTensor(
      output_step_dim, b * output_step_dim.getFeatureLen(), true);
    auto router_logits =
      router_logits_.getSharedDataTensor(router_logits_step_dim, 0, true);

    const unsigned batch_size = input.batch();
    const unsigned seq_len = input.height();
    const unsigned hidden_size = input.width();
    const unsigned total_tokens = batch_size * seq_len;

    // routing, part one: the logits on the accelerator (its norm folded
    // in, and for the softmax router its selection too) or the CPU's norm
    // of the router's copy for the dot below
    const bool want_extra =
      experts_virtual || moeTrace(context.getComputeOps()) != nullptr;
    AcceleratedRoute route;
    route.top_k = topk;
    route.n_sel =
      want_extra ? std::min<unsigned>(topk + EXTRA_TOPK, num_experts) : topk;
    route.scale = expert_bias.getData<float>();
    bool router_on_arm = false;
#if defined(__aarch64__) && defined(ENABLE_HEXKL)
    if (in_norm && !norm_on_accel && !out_add && !router_norm && !gelu_act &&
        !softmax_router && num_experts == 64 &&
        std::get<nntrainer::props::Unit>(moe_props).get() == 1536 &&
        context.getNumInputs() == 1)
      router_on_arm =
        lfmRouterOnArm(router_in, gate_weights, router_logits, context,
                       expert_gate_up_proj_indices, expert_down_proj_indices);
#endif
    const bool router_ready =
      router_on_arm ||
      routerLogitsOnAccelerator(
        router_norm ? raw : router_in, gate_weights,
        router_norm ? context.getWeight(router_gamma_idx).getData<float>()
                    : nullptr,
        norm_eps, total_tokens, hidden_size, num_experts, router_logits,
        softmax_router ? &route : nullptr);
    if (router_norm && !router_ready)
      normRows(raw, router_in, context.getWeight(router_gamma_idx), norm_eps);

    {
      M0Timer t(&g_m0.setup);
      // reshape input: [B,1,S,H] -> [B*S,1,1,H]
      input.reshape({total_tokens, 1, 1, hidden_size});
      router_in.reshape({total_tokens, 1, 1, hidden_size});

      // reshape output: [B,1,S,H] -> [B*S,1,1,H]
      output.reshape({total_tokens, 1, 1, hidden_size});
    }

    // One decode row: the HTP routes, runs the experts and adds
    // the result into its resident residual in the same stretch, so the
    // layer writes nothing (the next norm hook returns the stream). The
    // fused call is the MoE wait, so it counts as ffn; a continue
    // stops the timer through its destructor.
    M0Timer fused(&g_m0.ffn);
    if (total_tokens == 1 && !out_add &&
        gate_weights.getDataType() == ml::train::TensorDim::DataType::FP32 &&
        htpDecodeRouter(from, input.getData<float>(), hidden_size,
                        gate_weights.getData<float>(), num_experts,
                        expert_bias.getData<float>())) {
      output.reshape({batch_size, 1, seq_len, hidden_size});
      continue;
    }
    fused.stop();

    // routing, part two: the dot the accelerator did not do
    if (!router_ready) {
      M0Timer t(&g_m0.router);
      routerDot(context, router_in, gate_weights, router_logits, total_tokens,
                hidden_size, num_experts);
    }

    std::vector<std::vector<std::pair<unsigned, float>>> expert_assignments(
      num_experts);
    std::vector<int> extra_top_k;
    size_t max_assigned_tokens = 0;
    {
      M0Timer t(&g_m0.topk);
      if (router_ready && softmax_router)
        assignmentsFromRoute(route, total_tokens, expert_assignments,
                             want_extra ? &extra_top_k : nullptr);
      else
        buildExpertAssignments(router_logits, expert_bias, total_tokens,
                               expert_assignments,
                               want_extra ? &extra_top_k : nullptr);
      if (std::FILE *tf = moeTrace(context.getComputeOps())) {
        std::fprintf(tf, "%u %u |", trace_layer, total_tokens);
        for (unsigned int e = 0; e < num_experts; ++e)
          if (!expert_assignments[e].empty())
            std::fprintf(tf, " %u", e);
        std::fputs(" |", tf);
        for (int e : extra_top_k)
          std::fprintf(tf, " %d", e);
        std::fputc('\n', tf);
      }

      for (const auto &assignments : expert_assignments)
        max_assigned_tokens = std::max(max_assigned_tokens, assignments.size());
    }

    // Decode's single token routes to num_experts_per_tok distinct experts,
    // all sharing that one token as gate_up's activation -- group them (see
    // computeGroupedDecodeExperts). Prefill (total_tokens != 1) and the
    // degenerate 0/1-expert case are unaffected: same loop as before.
    std::vector<unsigned int> single_token_experts;
    if (total_tokens == 1) {
      for (unsigned int expert_idx = 0; expert_idx < num_experts;
           ++expert_idx) {
        if (!expert_assignments[expert_idx].empty())
          single_token_experts.push_back(expert_idx);
      }
    }

    // One call for every expert, when the accelerator offers it.
    // The M > 1 gate that used to sit here moved inside: it does not apply to
    // QS4CX_WH weights, which have no ARM path to fall back to at M == 1.
    // Falls through to the per-expert loop below otherwise.
    /** M0: the one stage that had no timer. With the whole-layer call the
       ARM-side gather, route and scatter all read ~0 because the DSP does
       them, and this read 0 as well -- so nothing accounted for the layer's
       wall clock, and doc 46 section 18.1 had to quote the ARM remainder as
       a 2.1-5.0 ms range rather than a number. Scoped, not folded into the
       && above: a temporary M0Timer in an expression is destroyed before
       the call it was meant to wrap. */
    bool moe_layer_done = false;
    bool out_normed_on_accel = false;
    bool add_done = false; // the epilogue rode the call
    {
      M0Timer t(&g_m0.ffn);
      // The output norm rides the prefill call; a decode row keeps its
      // dspqueue transport and norms on the CPU below.
      moe_layer_done = tryMoeLayerOnAccelerator(
        norm_on_accel ? raw : input, output, expert_assignments, context,
        expert_gate_up_proj_indices, expert_down_proj_indices, total_tokens,
        hidden_size, std::get<nntrainer::props::Unit>(moe_props).get(),
        experts_virtual, &extra_top_k, expert_layer_slot, gelu_act, trace_layer,
        norm_on_accel ? context.getWeight(in_gamma_idx).getData<float>()
                      : nullptr,
        out_norm && total_tokens > 1
          ? context.getWeight(out_gamma_idx).getData<float>()
          : nullptr,
        norm_eps, norm_on_accel ? x2.getData<float>() : nullptr,
        norm_on_accel ? context.getWeight(add_gamma_idx).getData<float>()
                      : nullptr,
        norm_on_accel
          ? context.getWeight(add_scale_idx).getValue<float>(0, 0, 0, 0)
          : 1.0f,
        &add_done, out_to_host || (out_add && !norm_on_accel));
      out_normed_on_accel = moe_layer_done && out_norm && total_tokens > 1;
    }

    // The ARM path's own preparation, after the accelerator has had its
    // turn: it zero-fills the output itself (hexkl_mm_u8i4_moe.c, memset of
    // out_c) and never touches the workspace, while FloatTensor's allocation
    // is a zero-fill too (float_tensor.cpp, `new float[n]{}`), so both used
    // to cost every accelerated prefill layer a memset and a page-fault
    // storm for buffers it then freed unused (doc 47 section 16, E1).
    nntrainer::Tensor prefill_token_input;
    nntrainer::Tensor prefill_expert_output;
    nntrainer::Tensor prefill_gate_up_output;
    nntrainer::Tensor prefill_activation_output;
    ExpertWorkspace workspace{
      nullptr,
      &context.getTensor(decode_expert_output_idx),
      &context.getTensor(decode_gate_up_output_idx),
      &context.getTensor(decode_activation_output_idx),
    };
    if (!moe_layer_done) {
      if (norm_on_accel) { // the CPU loops below read the normed copy
        // ... of rows (and a dense branch) their calls may have kept
        if (auto *o = raw.getOps()) {
          o->sync_to_host(raw.getData<float>(), raw.size() * sizeof(float));
          o->sync_to_host(x2.getData<float>(), x2.size() * sizeof(float));
        }
        normRows(raw, input, context.getWeight(in_gamma_idx), norm_eps);
      }
      {
        M0Timer t(&g_m0.setup);
        output.setZero();
      }
      if (max_assigned_tokens > 1) {
        M0Timer t(&g_m0.wksp);
        const unsigned int workspace_tokens =
          static_cast<unsigned int>(max_assigned_tokens);
        const unsigned int intermediate_size =
          std::get<nntrainer::props::Unit>(moe_props).get();
        prefill_token_input = nntrainer::Tensor(
          1, 1, workspace_tokens, hidden_size, input.getTensorType());
        prefill_expert_output = nntrainer::Tensor(
          workspace_tokens, 1, 1, hidden_size, output.getTensorType());
        prefill_gate_up_output = nntrainer::Tensor(
          1, 1, workspace_tokens, 2 * intermediate_size, input.getTensorType());
        prefill_activation_output = nntrainer::Tensor(
          1, 1, workspace_tokens, intermediate_size, input.getTensorType());
        // See the identical comment in forwarding(): these locally-constructed
        // Tensors carry no ContextData of their own, so dispatch would
        // otherwise silently fall back to CPU regardless of the weights'
        // engine.
        input.inheritContextTo(prefill_token_input);
        input.inheritContextTo(prefill_expert_output);
        input.inheritContextTo(prefill_gate_up_output);
        input.inheritContextTo(prefill_activation_output);
        workspace = {&prefill_token_input, &prefill_expert_output,
                     &prefill_gate_up_output, &prefill_activation_output};
      }
    }

    if (moe_layer_done) {
      // nothing further: the accelerator zeroed the output, ran every
      // expert, applied the routing weights and scattered the results.
    } else if (single_token_experts.size() > 1) {
      computeGroupedDecodeExperts(context, input, output, single_token_experts,
                                  expert_assignments, hidden_size, workspace);
    } else {
      for (unsigned int expert_idx = 0; expert_idx < num_experts;
           ++expert_idx) {
        const auto &assignments = expert_assignments[expert_idx];
        if (assignments.empty())
          continue;

        compute_expert_forward(
          input, output, assignments,
          context.getWeight(expert_gate_up_proj_indices[expert_idx]),
          context.getWeight(expert_down_proj_indices[expert_idx]), hidden_size,
          workspace);
      }
    }

    // reshape output: [B*S,1,1,H] -> [B,1,S,H]
    output.reshape({batch_size, 1, seq_len, hidden_size});
    if (out_norm && !out_normed_on_accel) {
      auto scratch =
        context.getTensor(experts_in_idx)
          .getSharedDataTensor(output_step_dim,
                               b * output_step_dim.getFeatureLen(), true);
      normRows(output, scratch, context.getWeight(out_gamma_idx), norm_eps);
      output.copyData(scratch);
    }
    if (out_add && !add_done) {
      auto scratch =
        context.getTensor(experts_in_idx)
          .getSharedDataTensor(output_step_dim,
                               b * output_step_dim.getFeatureLen(), true);
      addEpilogue(context, raw, x2, output, scratch, norm_eps);
    }

    if (m0_profile && total_tokens > 1) {
      const auto m0_us = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - m0_t0)
          .count());
      const uint64_t named = g_m0.sum();
      std::cout << "[M0-PROF] moe_layer[" << g_m0_call_index.fetch_add(1)
                << "] tokens=" << total_tokens << " us=" << m0_us
                << "  setup=" << g_m0.setup << " router=" << g_m0.router
                << " topk=" << g_m0.topk << " wksp=" << g_m0.wksp
                << " gather=" << g_m0.gather << " ffn=" << g_m0.ffn
                << " route=" << g_m0.route << " scatter=" << g_m0.scatter
                << " other=" << (m0_us > named ? m0_us - named : 0)
                << " miss=" << g_m0.misses << std::endl;
    }
  }
  if (op_time) {
    ++g_op_time_moe.calls;
    g_op_time_moe.cpu_us += g_m0.setup + g_m0.router + g_m0.topk;
    g_op_time_moe.call_us += g_m0.ffn;
  }
  g_m0_on = false;
}

void Lfm2MoELayer::computeGroupedDecodeExperts(
  nntrainer::RunLayerContext &context, const nntrainer::Tensor &input,
  nntrainer::Tensor &output, const std::vector<unsigned int> &selected_experts,
  const std::vector<std::vector<std::pair<unsigned, float>>>
    &expert_assignments,
  unsigned int hidden_size, ExpertWorkspace &workspace) {

  const unsigned int n = static_cast<unsigned int>(selected_experts.size());
  const unsigned int intermediate_size =
    std::get<nntrainer::props::Unit>(moe_props).get();

  // Decode's one token (token index 0 in this reshaped call), shared by
  // every selected expert's gate_up projection.
  nntrainer::Tensor token_input =
    input.getSharedDataTensor({1, 1, 1, hidden_size}, 0, true);

  // One buffer holding every selected expert's [gate|up] side by side, so
  // the grouped dot() below can write each expert's slice through its own
  // view -- same shape decode_gate_up_output holds for a single expert,
  // just concatenated across the n experts in this call.
  nntrainer::Tensor batched_gate_up(1, 1, n, 2 * intermediate_size,
                                    input.getTensorType());
  input.inheritContextTo(batched_gate_up);

  std::vector<nntrainer::Tensor *> gate_up_weights(n);
  std::vector<nntrainer::Tensor> gate_up_views;
  gate_up_views.reserve(n);
  for (unsigned int i = 0; i < n; ++i) {
    gate_up_weights[i] =
      &context.getWeight(expert_gate_up_proj_indices[selected_experts[i]]);
    gate_up_views.push_back(batched_gate_up.getSharedDataTensor(
      {1, 1, 1, 2 * intermediate_size}, i * 2 * intermediate_size, true));
  }
  std::vector<nntrainer::Tensor *> gate_up_outs(n);
  for (unsigned int i = 0; i < n; ++i)
    gate_up_outs[i] = &gate_up_views[i];

  token_input.dot(gate_up_weights, gate_up_outs);

  nntrainer::TensorDim intermediate_dim({1, 1, 1, intermediate_size},
                                        input.getTensorType());
  nntrainer::TensorDim token_step_dim({1, 1, 1, hidden_size},
                                      input.getTensorType());

  // swiglu and down stay per-expert and sequential -- each expert's
  // activation differs after swiglu, so they cannot share a down call the
  // way gate_up shared its activation. Reuses the same single-expert
  // workspace slots compute_expert_forward_no_critical uses, one expert at
  // a time, exactly as the un-grouped loop already did.
  for (unsigned int i = 0; i < n; ++i) {
    const unsigned int expert_idx = selected_experts[i];
    nntrainer::Tensor acti_out =
      workspace.activation_output->getSharedDataTensor(intermediate_dim, 0,
                                                       true);
    glu(acti_out.width(), acti_out.getData<float>(),
        gate_up_views[i].getData<float>(),
        gate_up_views[i].getData<float>() + intermediate_size);

    nntrainer::Tensor expert_output =
      workspace.expert_output->getSharedDataTensor(token_step_dim, 0, true);
    acti_out.dot(context.getWeight(expert_down_proj_indices[expert_idx]),
                 expert_output);
    expert_output.multiply_i(expert_assignments[expert_idx][0].second);

    nntrainer::Tensor token_output =
      output.getSharedDataTensor(token_step_dim, 0, true);
    token_output.add_i(expert_output);
  }
}

void Lfm2MoELayer::save(std::ofstream &file,
                        nntrainer::RunLayerContext &run_context, bool opt_var,
                        ml::train::ExecutionMode mode, bool trainable,
                        ml::train::TensorDim::DataType dtype,
                        ml::train::ISA target_isa) const {
  if (opt_var) {
    for (unsigned int i = 0; i < run_context.getNumWeights(); ++i) {
      if (run_context.isGradientFirstAccess(i) && trainable) {
        if (run_context.weightHasGradient(i)) {
          for (unsigned int j = 0; j < run_context.getNumWeightOptVar(i); ++j)
            run_context.getWeightOptVar(i, j).save(file);
        }
      }
    }
    return;
  }

  for (unsigned int i = 0; i < run_context.getNumWeights(); ++i) {
    if (!run_context.isGradientFirstAccess(i))
      continue;

    auto &weight = run_context.getWeight(i);

    // Router gate and expert bias must never be quantized: the generic
    // save-with-quantization path only special-cases height==1 (bias-like)
    // tensors, but the gate is [hidden, num_experts] with num_experts
    // possibly divisible by 32 (e.g. 32 here), which would otherwise be
    // silently Q4_0-quantized and corrupt every tensor written after it.
    const ml::train::TensorDim::DataType effective_dtype =
      (i == gate_idx || i == expert_bias_idx)
        ? ml::train::TensorDim::DataType::NONE
        : dtype;

    if (effective_dtype == ml::train::TensorDim::DataType::NONE ||
        weight.getDataType() == effective_dtype) {
      weight.save(file);
      continue;
    }

    if (effective_dtype == ml::train::TensorDim::DataType::Q4_0) {
      NNTR_THROW_IF(weight.getDataType() !=
                      ml::train::TensorDim::DataType::FP32,
                    std::runtime_error)
        << "Save with quantization only supports for FP32 weight.";
      nntrainer::TensorDim dim = weight.getDim();
      unsigned int K = dim.height();
      unsigned int N = dim.width();

      if (K == 1) {
        weight.save(file);
        continue;
      }

      NNTR_THROW_IF(N % 32 != 0 || K % 32 != 0, std::invalid_argument)
        << "Q4_0 quantization requires both width and height to be "
           "divisible by 32, but got height="
        << K << ", width=" << N;

      nntrainer::Tensor weight_t = weight.transpose("0:2:1");
      nntrainer::Tensor quant_weight(
        dim.batch(), dim.channel(), K, N,
        {nntrainer::Tformat::NCHW, effective_dtype});
      std::vector<char> tmp(quant_weight.size());

      nntrainer::quantize_q4_0(weight_t.getData<float>(), tmp.data(), N, K,
                               nullptr);
      nntrainer::repack_q4_0(quant_weight.getData<uint8_t>(), tmp.data(),
                             quant_weight.size(), N, K, target_isa);
      quant_weight.save(file);
    } else {
      NNTR_THROW_IF(true, std::runtime_error)
        << "This dtype is not supported in save with quantization for "
           "Lfm2MoELayer";
    }
  }
}

void Lfm2MoELayer::setProperty(const std::vector<std::string> &values) {
  std::vector<std::string> filtered;
  for (const auto &v : values) {
    if (v == "router_dtype=QS4CX_WH_AFFINE")
      router_q4_ = true;
    else
      filtered.push_back(v);
  }
  auto remain_props = loadProperties(filtered, moe_props);
  nntrainer::LayerImpl::setProperty(remain_props);
}

void Lfm2MoELayer::calcDerivative(nntrainer::RunLayerContext &context) {
  throw std::runtime_error(
    "LFM2 MoE layer does not support derivative calculation");
}

void Lfm2MoELayer::calcGradient(nntrainer::RunLayerContext &context) {
  throw std::runtime_error(
    "LFM2 MoE layer does not support gradient calculation");
}

void Lfm2MoELayer::exportTo(nntrainer::Exporter &exporter,
                            const ml::train::ExportMethods &method) const {
  nntrainer::LayerImpl::exportTo(exporter, method);
  exporter.saveResult(moe_props, method, this);
}

} // namespace causallm
