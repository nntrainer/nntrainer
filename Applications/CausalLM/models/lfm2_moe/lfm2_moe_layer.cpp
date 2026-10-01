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

#include <acti_func.h>
#include <algorithm>
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
#include <htp_wh_layout.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#include <expert_lru.h>
#include <htp_decode_hook.h>
#include <iostream>
#include <lfm2_moe_layer.h>
#include <node_exporter.h>
#include <stdexcept>
#include <thread_manager.h>
#include <unordered_map>
#include <unordered_set>

/** [A1] The deterministic SwiGLU. nntrainer::swiglu is left alone on
    purpose -- every model on ARM uses it and nothing is wrong with it.
    What this path needs is not a better SwiGLU but the SAME one the DSP
    runs, down to the bit, so that enabling the fused kernel cannot move a
    single uint8 quantization level (doc 44 section 3.3). */
#include <swiglu_det.h>

namespace causallm {

namespace {

/**
 * @brief [P3] Where the MoE layer's ARM-side wall clock actually goes.
 *
 * NNTR_HTP_PROFILE accounts for everything inside a FastRPC call. M0
 * measures the layer's wall clock. Doc 43 section 4.3 subtracted one from
 * the other and got ~21 ms/layer that neither instrument sees, and the
 * whole HTP-vs-CPU verdict turns on it (doc 44 section 5): if that time is
 * HTP-specific, P1's batching removes it and HTP wins; if it is work the
 * CPU path does too, P1 changes nothing and the MoE FFN alone cannot beat
 * CPU.
 *
 * A residual cannot answer that, because most of what happens here --
 * routing, top-k, the token gather, the scatter-add -- runs on BOTH paths.
 * So this times the stages by name and prints them, and the same build
 * prints the same stages under moe_engine=cpu. Subtracting the two runs
 * cancels the shared work and leaves ffn_cpu against ffn_htp, which is the
 * comparison that decides P1.
 *
 * `other` is the layer wall clock minus every named stage. It is printed
 * rather than distributed, so a stage nobody thought to time shows up as a
 * number instead of hiding inside one that was.
 *
 * ponytail: one file-scope accumulator, no locking. Layers run one at a
 * time within a forward and each of these stages is entered from the
 * serial outer thread (the parallel_for inside the gather is timed from
 * its caller, not from the workers), so there is no race today. If layers
 * are ever run concurrently this needs to become per-call state threaded
 * through the two compute_expert_forward helpers -- which is why the
 * timers below take a pointer to their slot rather than an index.
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
  uint64_t misses = 0;  /**< a count, not a time: experts loaded from the
                             model file for this call (doc 52) */

  void reset() { *this = M0Stages{}; }
  uint64_t sum() const {
    return setup + router + topk + wksp + gather + ffn + route + scatter;
  }
};

M0Stages g_m0;
bool g_m0_on = false;

/** @brief [#150] NNTR_OP_TIME=1: the decode MoE layer's M0 slots summed
 *         over the run, split into the ARM part (setup + router + top-k)
 *         and the call (`ffn`: staging, queue write, ARM spin, response),
 *         one line at exit. Per call would be 22 lines per token on the
 *         timed run's stderr. */
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

  /** @brief Ends the interval early and idempotently, for a stage that
   *         finishes before its enclosing scope does. Without it the FFN
   *         timer below would have to be given its own block, and the
   *         re-indentation would bury a one-line instrument in a diff that
   *         looks like a rewrite of the dispatch. */
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

/** LFM2-MoE router hyper-parameters (fixed for LFM2-8B-A1B). */
static constexpr bool NORM_TOPK_PROB = true;
static constexpr float ROUTED_SCALING_FACTOR = 1.0f;

/** [doc 52] Experts beyond the routed top-k whose recency the LRU refreshes,
 *  Lfm2CachedSlimMoELayer's value. */
static constexpr unsigned EXTRA_TOPK = 5;

/** [doc 52] The one pool of resident experts every MoE layer draws on. Its
 *  capacity is NNTR_MOE_CACHE_EXPERTS per layer, summed over the layers
 *  that finalize with virtual experts; the accelerator holds one slot per
 *  resident expert (register_qs4cx_wh_expert_file). Shared rather than per
 *  layer because the layer call needs a whole layer's active experts
 *  resident at once -- 32 at prefill -- which 22 layers x 2 can lend one
 *  layer and a per-layer bound of 2 cannot. */
static causallm::ExpertLru g_expert_lru;

using ExpertFileDesc = nntrainer::ComputeOps::ExpertFileDesc;

/** [doc 52 section 10.10] Every virtual MoE layer's experts, as the backend
 *  needs them to read one from the file, in layer order (preloadExperts is
 *  called in graph order). Row i + 1 is what layer i reads ahead. */
static std::vector<std::vector<ExpertFileDesc>> g_expert_layers;

/** Finalize-order ordinal of MoE layers, for NNTR_MOE_TRACE. */
static std::atomic<unsigned> g_moe_layer_count{0};

/** @brief NNTR_MOE_PREFETCH=<k>: prefill reads the experts of the layers up
 *  to k ahead under the current layers' calls (doc 52 sections 10.10,
 *  10.20); 0 turns it off. Unset is every remaining layer (section 10.33):
 *  the pool's room is the real limit -- C=8 fits this layer's 32 and four
 *  more layers' in its 176 slots, C <= 3 not even one -- and where there is
 *  none, queuing just stops. A cap of 4 (section 10.26) was the same at
 *  C=8 but cost C=16 130 ms cold: its preload covers layers 0-10, so the
 *  first layer with anything to read is 11, and a 4-layer horizon did not
 *  start reading it until layer 7. */
static int expertPrefetchDepth() {
  static const int k = [] {
    const char *v = std::getenv("NNTR_MOE_PREFETCH");
    return v != nullptr ? std::max(0, std::atoi(v)) : 1 << 20;
  }();
  return k;
}

/** [doc 52 section 10.20] Read-ahead batches in flight, oldest first: the
 *  layer slot each was queued for and the LRU slots it holds. */
static std::deque<std::pair<int, size_t>> g_prefetch_batches;
/** The next layer slot to queue in the current prefill. */
static int g_prefetch_next = 0;

/**
 * @brief NNTR_MOE_TRACE=<path>: one line per MoE layer call,
 *        "<layer> <tokens> | <routed experts> | <top-(k+5) per token>",
 *        for tools/moe_expert_cache_sim.py to replay against cache
 *        policies offline (doc 52 section 10.10). The routing is the same
 *        whatever the cache does, so one resident run gives every C.
 */
static std::FILE *moeTrace() {
  static std::FILE *f = [] {
    const char *path = std::getenv("NNTR_MOE_TRACE");
    return (path != nullptr && *path != '\0') ? std::fopen(path, "w") : nullptr;
  }();
  return f;
}

/** @brief NNTR_MOE_CACHE_EXPERTS as a count, 0 when unset or unparsable
 *  (the resident path, unchanged). The same variable and the same parsing
 *  as Lfm2CachedSlimMoELayer, so one knob drives the CPU and HTP caches. */
static unsigned int expertCacheFromEnv(unsigned int num_experts) {
  const char *value = std::getenv("NNTR_MOE_CACHE_EXPERTS");
  if (value == nullptr || *value == '\0' || *value == '-')
    return 0;
  errno = 0;
  char *end = nullptr;
  const unsigned long parsed = std::strtoul(value, &end, 10);
  if (errno == ERANGE || end == value || *end != '\0')
    return 0;
  return static_cast<unsigned int>(
    std::min<unsigned long>(num_experts, parsed));
}

/** M0 (doc 43 §5): ordinal of prefill forwarding() calls, so each MoE layer
 * prints under a stable index in layer order. Decode never touches this --
 * it runs incremental_forwarding. */
static std::atomic<unsigned> g_m0_call_index{0};

Lfm2MoELayer::Lfm2MoELayer() :
  LayerImpl(),
  num_experts(0),
  topk(0),
  moe_props(props::NumExperts(), props::NumExpertsPerToken(),
            nntrainer::props::Unit(), props::MoEActivation(),
            props::MoERouter(), nntrainer::props::Epsilon()),
  expert_gate_up_proj_indices({}),
  expert_down_proj_indices({}),
  gate_idx(std::numeric_limits<unsigned>::max()),
  expert_bias_idx(std::numeric_limits<unsigned>::max()),
  softmax_router(false),
  router_scale_idx(std::numeric_limits<unsigned>::max()),
  experts_virtual(false),
  cache_per_layer(0),
  expert_layer_slot(-1),
  trace_layer(0),
  router_logits_idx(std::numeric_limits<unsigned>::max()),
  decode_expert_output_idx(std::numeric_limits<unsigned>::max()),
  decode_gate_up_output_idx(std::numeric_limits<unsigned>::max()),
  decode_activation_output_idx(std::numeric_limits<unsigned>::max()) {}

void Lfm2MoELayer::finalize(nntrainer::InitLayerContext &context) {

  // 1. Validate input/output dimensions. [plan 201 S4] The softmax router
  // (Gemma 4) reads a second input, the un-normed row, and runs its
  // experts on the accelerator only: the CPU expert path here is SwiGLU,
  // Gemma's CPU MoE is gemma4_moe.
  const std::string router = std::get<props::MoERouter>(moe_props).get();
  NNTR_THROW_IF(router != "sigmoid" && router != "softmax",
                std::invalid_argument)
    << "LFM2 MoE layer: moe_router=" << router << " (sigmoid or softmax)";
  softmax_router = router == "softmax";
  NNTR_THROW_IF(context.getNumInputs() != (softmax_router ? 2u : 1u) ||
                  (softmax_router && context.getInputDimensions()[1] !=
                                       context.getInputDimensions()[0]),
                std::invalid_argument)
    << "LFM2 MoE layer takes " << (softmax_router ? 2 : 1)
    << " input(s) of one shape with moe_router=" << router;
  NNTR_THROW_IF(softmax_router && context.getComputeEngineType() ==
                                    ml::train::LayerComputeEngine::CPU,
                std::invalid_argument)
    << "LFM2 MoE layer: moe_router=softmax needs an accelerator engine";

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
  const bool is_nchw = context.getFormat() == nntrainer::Tformat::NCHW;
  std::vector<nntrainer::TensorDim> output_dims(1);
  output_dims[SINGLE_INOUT_IDX] = in_dim;
  context.setOutputDimensions(output_dims);

  // 3. Get MoE properties
  num_experts = std::get<props::NumExperts>(moe_props).get();
  topk = std::get<props::NumExpertsPerToken>(moe_props).get();

  // [doc 52] With NNTR_MOE_CACHE_EXPERTS set and an accelerator engine,
  // the expert weights are requested virtual: the loader records their
  // file offsets and reads nothing, and the forward brings each expert
  // from the file into the accelerator's slot pool on demand, under one
  // LRU shared by every MoE layer (g_expert_lru). A CPU-engine layer keeps
  // its resident weights whatever the variable says -- the CPU cache is
  // Lfm2CachedSlimMoELayer's job.
  trace_layer = g_moe_layer_count.fetch_add(1);
  cache_per_layer = expertCacheFromEnv(num_experts);
  experts_virtual =
    cache_per_layer != 0 &&
    context.getComputeEngineType() != ml::train::LayerComputeEngine::CPU;
  if (experts_virtual)
    g_expert_lru.addLayer(this, cache_per_layer);
  const unsigned int intermediate_size =
    std::get<nntrainer::props::Unit>(moe_props).get();
  const unsigned int hidden_size = in_dim.width(); // Feature dimension

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

  // 4. Initialize gate layer (router). Always kept FP32.
  nntrainer::TensorDim gate_dim(
    1, is_nchw ? 1 : num_experts, is_nchw ? hidden_size : 1,
    is_nchw ? num_experts : hidden_size,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     nntrainer::TensorDim::DataType::FP32),
    is_nchw ? 0b0011 : 0b0101);

  gate_idx =
    context.requestWeight(gate_dim, weight_initializer, weight_regularizer,
                          weight_regularizer_constant, weight_decay,
                          softmax_router ? "router" : "gate", true);

  // 4b. Expert bias used only for top-k selection. Shape [1,1,1,E], FP32.
  // [plan 201 S4] The softmax router's router_scale [K] and per-expert
  // scale [E] instead, in gemma4_moe's (and its file's) order.
  const auto fp32_type = nntrainer::TensorDim::TensorType(
    context.getFormat(), nntrainer::TensorDim::DataType::FP32);
  if (softmax_router)
    router_scale_idx = context.requestWeight(
      nntrainer::TensorDim(1, 1, 1, hidden_size, fp32_type),
      nntrainer::Initializer::ONES, nntrainer::WeightRegularizer::NONE, 1.0f,
      0.0f, "router_scale", true);
  nntrainer::TensorDim expert_bias_dim(1, 1, 1, num_experts, fp32_type, 0b0001);

  expert_bias_idx = context.requestWeight(
    expert_bias_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay,
    softmax_router ? "router_per_expert_scale" : "expert_bias", false);

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

void Lfm2MoELayer::routeSoftmax(
  nntrainer::RunLayerContext &context, nntrainer::Tensor &router_in,
  nntrainer::Tensor &router_logits, unsigned int total_tokens,
  std::vector<std::vector<std::pair<unsigned, float>>> &expert_assignments,
  std::vector<int> *extra_top_k) {
  // gemma4_moe_layer.cpp's forwardTensors, the CPU reference, with the same
  // calls in the same order: rms_norm without gamma, times router_scale /
  // sqrt(hidden), the dot, softmax, topK, the renormalisation in topK's
  // order times the per-expert scale.
  nntrainer::Tensor &gate_weights = context.getWeight(gate_idx);
  nntrainer::Tensor &expert_bias = context.getWeight(expert_bias_idx);
  const unsigned int hidden_size = router_in.width();
  const float eps = std::get<nntrainer::props::Epsilon>(moe_props).get();
  nntrainer::Tensor scaled(total_tokens, 1, 1, hidden_size,
                           router_in.getTensorType());
  nntrainer::rms_norm_wrt_width_fp32_intrinsic(router_in.getData<float>(),
                                               scaled.getData<float>(),
                                               total_tokens, hidden_size, eps);
  const float *router_scale =
    context.getWeight(router_scale_idx).getData<float>();
  const float hidden_scale = 1.0f / std::sqrt(static_cast<float>(hidden_size));
  for (unsigned int t = 0; t < total_tokens; ++t) {
    float *row = scaled.getData<float>() + static_cast<size_t>(t) * hidden_size;
    for (unsigned int f = 0; f < hidden_size; ++f)
      row[f] *= router_scale[f] * hidden_scale;
  }
  scaled.dot(gate_weights, router_logits);
  router_logits.apply(nntrainer::ActiFunc::softmax<float>, router_logits);
  auto top = router_logits.topK(topk);
  const uint32_t *ids = std::get<1>(top).getData<uint32_t>();
  const float *p = std::get<0>(top).getData<float>();
  const float *scale = expert_bias.getData<float>();
  for (unsigned int t = 0; t < total_tokens; ++t) {
    float sum = 0.0f;
    for (unsigned int k = 0; k < topk; ++k)
      sum += p[t * topk + k];
    for (unsigned int k = 0; k < topk; ++k) {
      const unsigned int o = t * topk + k;
      expert_assignments[ids[o]].emplace_back(t, p[o] / sum * scale[ids[o]]);
    }
    if (extra_top_k) {
      // the LRU hint: the top (k + EXTRA_TOPK) by probability, ties to the
      // lower id (the chosen k are topK's)
      const float *row =
        router_logits.getData<float>() + static_cast<size_t>(t) * num_experts;
      std::vector<int> order(num_experts);
      for (unsigned int e = 0; e < num_experts; ++e)
        order[e] = static_cast<int>(e);
      const unsigned int ext = std::min(topk + EXTRA_TOPK, num_experts);
      std::partial_sort(order.begin(), order.begin() + ext, order.end(),
                        [row](int a, int b) {
                          return row[a] > row[b] || (row[a] == row[b] && a < b);
                        });
      extra_top_k->insert(extra_top_k->end(), order.begin(),
                          order.begin() + ext);
    }
  }
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

  // Reusable scratch buffers (per token).
  std::vector<float> sig(num_experts);
  std::vector<std::pair<float, int>> scored(num_experts);

  for (unsigned int i = 0; i < total_tokens; ++i) {
    const float *lrow = logits + static_cast<size_t>(i) * num_experts;

    // sigmoid scores (bias-free) and biased scores used only for selection
    for (unsigned int e = 0; e < num_experts; ++e) {
      const float s = 1.0f / (1.0f + std::exp(-lrow[e]));
      sig[e] = s;
      scored[e] = {s + bias[e], static_cast<int>(e)};
    }

    // top-k experts by (sigmoid + bias); an exact tie goes to the lower
    // index -- a total order, so the selection does not depend on the
    // library's partial_sort, and the HTP router's rule (m1_ops_det.h)
    std::partial_sort(
      scored.begin(), scored.begin() + extended, scored.end(),
      [](const std::pair<float, int> &a, const std::pair<float, int> &b) {
        return a.first > b.first || (a.first == b.first && a.second < b.second);
      });

    // routing weights come from the bias-free sigmoid scores
    float wsum = 0.0f;
    for (unsigned int k = 0; k < topk; ++k)
      wsum += sig[scored[k].second];

    const float inv = NORM_TOPK_PROB ? (1.0f / (wsum + 1e-6f)) : 1.0f;
    for (unsigned int k = 0; k < topk; ++k) {
      const int expert_idx = scored[k].second;
      const float weight = sig[expert_idx] * inv * ROUTED_SCALING_FACTOR;
      expert_assignments[expert_idx].emplace_back(i, weight);
    }
    if (extra_top_k) {
      for (unsigned int k = 0; k < extended; ++k)
        extra_top_k->push_back(scored[k].second);
    }
  }
}

/** @brief [doc 52] One virtual expert from the model file into the
 *  accelerator's slot pool, keyed by its two weight tensors. The dims are
 *  the kernel's: gate_up is [K, 2 * inter], down [inter, N_out]. */
static ExpertFileDesc expertDesc(nntrainer::Tensor &gu, nntrainer::Tensor &dn) {
  const auto dt = gu.getDataType();
  if ((dt != nntrainer::Tdatatype::QS4CX_WH &&
       dt != nntrainer::Tdatatype::QS2CX_WH) ||
      dn.getDataType() != dt) {
    throw std::runtime_error(
      "NNTR_MOE_CACHE_EXPERTS on an accelerator engine needs QS4CX_WH or "
      "QS2CX_WH expert weights (the file's bytes go to the DSP as they are); "
      "this model's are not");
  }
  return ExpertFileDesc{&gu,
                        &dn,
                        gu.getFd(),
                        gu.getFileOffset(),
                        dn.getFileOffset(),
                        static_cast<unsigned int>(gu.height()),
                        static_cast<unsigned int>(dn.height()),
                        static_cast<unsigned int>(dn.width()),
                        dt == nntrainer::Tdatatype::QS2CX_WH ? 2u : 4u};
}

static void loadVirtualExpert(nntrainer::ComputeOps *ops,
                              const ExpertFileDesc &d, bool at_load) {
  if (!ops->register_qs4cx_wh_expert_file(d, at_load)) {
    throw std::runtime_error(
      "this engine cannot load a virtual expert from the model file "
      "(register_qs4cx_wh_expert_file); unset NNTR_MOE_CACHE_EXPERTS");
  }
}

bool Lfm2MoELayer::preloadExperts(nntrainer::RunLayerContext &context) {
  if (!experts_virtual)
    return true;
  auto *ops = context.getComputeOps();
  if (ops == nullptr) {
    throw std::runtime_error("virtual MoE experts need the layer's engine to "
                             "provide ComputeOps; none is registered");
  }
  // Every virtual layer has finalized by now, so the pool's capacity is
  // final: size the backend's slot memory to it, once.
  static bool reserved = false;
  if (!reserved) {
    ops->reserve_qs4cx_wh_expert_slots(g_expert_lru.capacity());
    reserved = true;
  }

  std::vector<ExpertFileDesc> descs;
  descs.reserve(num_experts);
  for (unsigned int e = 0; e < num_experts; ++e)
    descs.push_back(
      expertDesc(context.getWeight(expert_gate_up_proj_indices[e]),
                 context.getWeight(expert_down_proj_indices[e])));
  if (expert_layer_slot < 0) {
    expert_layer_slot = static_cast<int>(g_expert_layers.size());
    g_expert_layers.push_back(descs);
  }

  std::vector<causallm::ExpertLru::Key> need;
  std::unordered_map<causallm::ExpertLru::Key, unsigned int> expert_of;
  const size_t room = g_expert_lru.capacity() - g_expert_lru.size();
  for (unsigned int e = 0; e < num_experts && need.size() < room; ++e) {
    need.push_back(descs[e].key_gu);
    expert_of[descs[e].key_gu] = e;
  }
  if (!need.empty()) {
    g_expert_lru.acquire(
      need,
      [&](causallm::ExpertLru::Key k) {
        loadVirtualExpert(ops, descs[expert_of[k]], /*at_load=*/true);
      },
      [](causallm::ExpertLru::Key) {
        throw std::logic_error("preloadExperts evicted an expert: the pool "
                               "filled past its capacity");
      });
  }
  return need.size() == num_experts;
}

void Lfm2MoELayer::forwarding(nntrainer::RunLayerContext &context,
                              bool training) {
  // [plan 201 S4] the CPU expert loop below is SwiGLU and the router the
  // sigmoid one: the softmax (Gemma 4) layer runs incremental_forwarding
  NNTR_THROW_IF(softmax_router, std::runtime_error)
    << "LFM2 MoE layer: moe_router=softmax has no forwarding() path";
  /** M0 (doc 43 §5): ARM-side wall timer around the whole layer -- this
   * profile is invisible to NNTR_HTP_PROFILE's host column by design, and
   * in the htp run the one dispatched layer's figure includes its
   * registration (once, first forward) on top of dispatch. */
  const bool m0_profile = std::getenv("NNTR_M0_PROFILE") != nullptr;
  const auto m0_t0 = std::chrono::steady_clock::now();
  g_m0_on = m0_profile;
  g_m0.reset();

  nntrainer::Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);

  nntrainer::Tensor &router_logits = context.getTensor(router_logits_idx);

  const unsigned batch_size = input.batch();
  const unsigned seq_len = input.height();
  const unsigned hidden_size = input.width();
  const unsigned total_tokens = batch_size * seq_len;

  {
    M0Timer t(&g_m0.setup);
    // reshape input: [B,1,S,H] -> [B*S,1,1,H]
    input.reshape({total_tokens, 1, 1, hidden_size});

    // reshape output: [B,1,S,H] -> [B*S,1,1,H]
    output.reshape({total_tokens, 1, 1, hidden_size});
    output.setZero();
  }

  // routing: raw logits -> sigmoid + expert-bias top-k selection
  nntrainer::Tensor &gate_weights = context.getWeight(gate_idx);
  nntrainer::Tensor &expert_bias = context.getWeight(expert_bias_idx);
  {
    M0Timer t(&g_m0.router);
    input.dot(gate_weights, router_logits);
  }

  std::vector<std::vector<std::pair<unsigned, float>>> expert_assignments(
    num_experts);
  size_t max_assigned_tokens = 0;
  {
    M0Timer t(&g_m0.topk);
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
 * @brief NNTR_MOE_DIFF=<rows>: for the first @a rows tokens of every MoE
 *        layer call, recompute the layer on the CPU from the model file's
 *        own bytes and report the SNR of the accelerator's output against
 *        it, with the layer's INPUT statistics beside it.
 *
 * The two numbers answer different halves of one question, which is why
 * they are printed together and why this is one instrument rather than two
 * runs (doc 56 section 6.1):
 *
 *   input mean/std/max already wild -> whatever is wrong is UPSTREAM of
 *                                      this layer (attention, embedding,
 *                                      the norms), and the MoE SNR is
 *                                      measuring agreement on garbage;
 *   input sane, SNR low             -> the MoE call itself, on these
 *                                      weights and this activation.
 *
 * Expect 30-45 dB when the kernel is right: the reference is pure f32 and
 * the kernel requantizes its activations to u8, so they do not agree
 * further than that. Near 0 dB or negative is a different computation, not
 * a quantization gap.
 *
 * 0 means every row. The cost is a full f32 MoE layer on the application
 * processor -- hidden * 2 * inter + inter * hidden MACs per routed token --
 * so the default of 8 rows keeps a prefill layer at seconds; every row is
 * minutes and is what NNTR_MOE_SHADOW needs.
 *
 * ponytail: virtual experts only (the layer's weights live in the file, not
 * in memory, which is what this reads). A resident-expert model has
 * NNTR_L2_DIFF on the same question one level down. The upgrade path, if a
 * resident model ever needs this one, is to read the tensors instead of the
 * file when getFd() is -1.
 *
 * ponytail: the layer call only. It sees the rows that come through
 * tryMoeLayerOnAccelerator with the expert pool on -- every prefill, and
 * hybrid decode rows no decode hook took. The one-PD token
 * (NNTR_HTP_E2E=1) never enters this layer, so it can be neither diffed
 * nor shadowed here; its instrument is the E2E --dump read by
 * htp_dump_eval.py. Hooking the token's MoE op is the upgrade path.
 */
static int moeDiffRows() {
  static const int rows = [] {
    const char *v = std::getenv("NNTR_MOE_DIFF");
    return v != nullptr ? std::max(0, std::atoi(v)) : -1;
  }();
  return rows;
}

/** @brief NNTR_MOE_SHADOW=1: compute the same reference for EVERY row and
 *  hand the model those values instead of the accelerator's. Text that
 *  comes back says the MoE values are the fault; text that stays broken
 *  says they are not, which no SNR number can decide. Implies the diff. */
static bool moeShadowEnabled() {
  static const bool on = std::getenv("NNTR_MOE_SHADOW") != nullptr;
  return on;
}

#ifndef _WIN32
/** @brief pread until @a len bytes are in, or false. */
static bool moeReadAt(int fd, void *dst, size_t len, uint64_t off) {
  auto *p = static_cast<uint8_t *>(dst);
  while (len != 0) {
    const ssize_t got = ::pread(fd, p, len, static_cast<off_t>(off));
    if (got <= 0)
      return false;
    p += got;
    off += static_cast<uint64_t>(got);
    len -= static_cast<size_t>(got);
  }
  return true;
}

/**
 * @brief One QS4CX_WH weight from the model file, dequantized to f32
 *        row-major [K][N].
 *
 * The file holds whBytes(K, N) nibbles, then N f32 scales, then N f32
 * column sums (the kernel's zero-point term, not needed here). A nibble is
 * the value plus 8, unsigned, where whSlot puts it.
 */
static bool moeReadExpertWeight(int fd, uint64_t off, uint32_t K, uint32_t N,
                                std::vector<float> &out) {
  const size_t nibbles = nntrainer::whBytes(K, N);
  std::vector<uint8_t> packed(nibbles);
  std::vector<float> scale(N);
  if (!moeReadAt(fd, packed.data(), nibbles, off) ||
      !moeReadAt(fd, scale.data(), N * sizeof(float), off + nibbles)) {
    return false;
  }
  std::vector<int8_t> q(static_cast<size_t>(K) * N);
  nntrainer::whUnpack(packed.data(), K, N, q.data());
  out.resize(q.size());
  auto &tm = nntrainer::ThreadManager::Global();
  tm.parallel_for(0, K, [&](size_t k) {
    const size_t row = k * N;
    for (uint32_t n = 0; n < N; ++n)
      out[row + n] = static_cast<float>(q[row + n]) * scale[n];
  });
  return true;
}

/** @brief The layer, in f32, from the file. See moeDiffRows(). */
static void moeDiff(const nntrainer::Tensor &input, nntrainer::Tensor &output,
                    const std::vector<std::vector<std::pair<unsigned, float>>>
                      &expert_assignments,
                    nntrainer::RunLayerContext &context,
                    const std::vector<unsigned int> &gate_up_indices,
                    const std::vector<unsigned int> &down_indices,
                    unsigned int total_tokens, unsigned int hidden_size,
                    unsigned int intermediate_size, bool gelu,
                    unsigned int trace_layer) {
  const bool shadow = moeShadowEnabled();
  const int want = moeDiffRows();
  const unsigned int rows =
    (shadow || want == 0)
      ? total_tokens
      : std::min<unsigned int>(total_tokens, want < 0 ? 8u : (unsigned)want);

  const float *in = input.getData<float>();
  const size_t in_n = static_cast<size_t>(total_tokens) * hidden_size;
  double sum = 0.0, sum2 = 0.0;
  float in_max = 0.0f;
  for (size_t i = 0; i < in_n; ++i) {
    sum += in[i];
    sum2 += static_cast<double>(in[i]) * in[i];
    in_max = std::max(in_max, std::fabs(in[i]));
  }
  const double in_mean = sum / in_n;
  const double in_std =
    std::sqrt(std::max(0.0, sum2 / in_n - in_mean * in_mean));

  std::vector<float> ref(in_n, 0.0f);
  std::vector<float> w_gu, w_dn, mid(intermediate_size);
  auto &tm = nntrainer::ThreadManager::Global();
  unsigned int experts_seen = 0;
  for (size_t e = 0; e < expert_assignments.size(); ++e) {
    const auto &rows_of_e = expert_assignments[e];
    if (std::none_of(rows_of_e.begin(), rows_of_e.end(),
                     [&](const std::pair<unsigned, float> &p) {
                       return p.first < rows;
                     })) {
      continue;
    }
    const ExpertFileDesc d = expertDesc(context.getWeight(gate_up_indices[e]),
                                        context.getWeight(down_indices[e]));
    if (d.fd < 0) {
      std::fprintf(stderr,
                   "[MOE-DIFF] layer=%u expert=%zu has no file fd; "
                   "NNTR_MOE_DIFF needs virtual experts\n",
                   trace_layer, e);
      return;
    }
    if (!moeReadExpertWeight(d.fd, d.off_gu, d.K, 2u * intermediate_size,
                             w_gu) ||
        !moeReadExpertWeight(d.fd, d.off_dn, intermediate_size, d.N_out,
                             w_dn)) {
      std::fprintf(stderr, "[MOE-DIFF] layer=%u expert=%zu: read failed\n",
                   trace_layer, e);
      return;
    }
    ++experts_seen;
    for (const auto &pair : rows_of_e) {
      if (pair.first >= rows)
        continue;
      const float *x = in + static_cast<size_t>(pair.first) * hidden_size;
      tm.parallel_for(0, intermediate_size, [&](size_t j) {
        float g = 0.0f, u = 0.0f;
        for (unsigned int k = 0; k < hidden_size; ++k) {
          const size_t row = static_cast<size_t>(k) * 2u * intermediate_size;
          g += x[k] * w_gu[row + j];
          u += x[k] * w_gu[row + intermediate_size + j];
        }
        mid[j] = gelu ? geglu_det_one(g, u) : swiglu_det_one(g, u);
      });
      float *dst = ref.data() + static_cast<size_t>(pair.first) * hidden_size;
      tm.parallel_for(0, hidden_size, [&](size_t n) {
        float acc = 0.0f;
        for (unsigned int j = 0; j < intermediate_size; ++j)
          acc += mid[j] * w_dn[static_cast<size_t>(j) * hidden_size + n];
        dst[n] += pair.second * acc;
      });
    }
  }

  float *got = output.getData<float>();
  double signal = 0.0, noise = 0.0, max_ref = 0.0, max_got = 0.0;
  for (size_t i = 0; i < static_cast<size_t>(rows) * hidden_size; ++i) {
    const double r = ref[i], diff = static_cast<double>(got[i]) - r;
    signal += r * r;
    noise += diff * diff;
    max_ref = std::max(max_ref, std::fabs(r));
    max_got = std::max(max_got, std::fabs(static_cast<double>(got[i])));
  }
  std::fprintf(stderr,
               "[MOE-DIFF] layer=%u M=%u rows=%u experts=%u snr=%.1fdB "
               "max_abs_ref=%g max_abs_got=%g in_mean=%g in_std=%g in_max=%g\n",
               trace_layer, total_tokens, rows, experts_seen,
               noise == 0.0 ? 999.0 : 10.0 * std::log10(signal / noise),
               max_ref, max_got, in_mean, in_std, static_cast<double>(in_max));
  if (shadow)
    std::memcpy(got, ref.data(), ref.size() * sizeof(float));
}
#endif

/**
 * @brief [doc 46] The whole MoE FFN layer in one accelerator call.
 *
 * Flattens the per-expert assignment lists into the three arrays the call
 * wants and hands the routing over, so the token gather, the routing
 * multiply and the scatter-add happen on the accelerator instead of here.
 * Doc 44 section 15.1 measured those at 6.3 ms/layer on the ARM side.
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
  int expert_layer_slot, bool gelu, unsigned int trace_layer) {

  auto *ops = input.getOps();
  if (ops == nullptr || !ops->supports_gemm_qs4cx_moe_layer_fp32()) {
    if (experts_virtual) {
      throw std::runtime_error("virtual MoE experts (NNTR_MOE_CACHE_EXPERTS) "
                               "have no CPU path: the layer's engine offers "
                               "no MoE layer call");
    }
    return false;
  }

  const size_t n_experts = expert_assignments.size();
  if (n_experts == 0 || gate_up_indices.size() < n_experts ||
      down_indices.size() < n_experts) {
    return false;
  }
  std::vector<void *> gu_data(n_experts), dn_data(n_experts);
  std::vector<float *> gu_scale(n_experts), dn_scale(n_experts);
  // QS4CX_WH is the same weight with its nibbles already in HMX tiles and a
  // column sum after the scales, built by the offline quantizer so the DSP
  // does not have to convert and bake at load (doc 46 section 35). Both
  // experts' halves have to agree, and every expert with every other: the
  // call takes one flag for the layer, and a model that mixed the two would
  // otherwise read half its weights with the wrong layout.
  const auto wh = nntrainer::Tdatatype::QS4CX_WH;
  const auto wh2 = nntrainer::Tdatatype::QS2CX_WH;
  const auto plain = nntrainer::Tdatatype::QS4CX;
  const auto dt0 = context.getWeight(gate_up_indices[0]).getDataType();
  const unsigned int w_bits = (dt0 == wh) ? 4u : (dt0 == wh2) ? 2u : 0u;

  // Decode's single token normally stays on the ARM side: it cannot amortize
  // the kernel's 64-row pad, which is why the fused path has the same gate.
  // QS4CX_WH has no ARM side to stay on -- FloatTensor implements no dot()
  // for it, deliberately, because those bytes are in HMX tile order and only
  // the DSP can read them -- so there the gate would pick a throw over a
  // slow-but-correct call.
  // ponytail: decode through the layer kernel pads M = 1 out to 64 rows for
  // each of the num_experts_per_tok active experts. It is correct and it is
  // the only option for these weights; a decode-shaped kernel, or moving the
  // grouped-decode gather onto the DSP, is the upgrade if decode TPS asks
  // for it.
  // NNTR_MOE_HTP_DECODE=1 lifts that gate for plain QS4CX: the layers in
  // moe_htp_layers take the kernel at M == 1 while the rest keep the ARM
  // path, which is the only way to put decode's CPU and HTP FFN side by
  // side in one run (doc 46 section 49). Measurement switch, not a default.
  static const bool htp_decode_forced =
    std::getenv("NNTR_MOE_HTP_DECODE") != nullptr;
  if (total_tokens <= 1 && w_bits == 0u && !htp_decode_forced) {
    return false;
  }

  for (size_t e = 0; e < n_experts; ++e) {
    nntrainer::Tensor &gu = context.getWeight(gate_up_indices[e]);
    nntrainer::Tensor &dn = context.getWeight(down_indices[e]);
    const auto want = (w_bits == 4u) ? wh : (w_bits == 2u) ? wh2 : plain;
    if (gu.getDataType() != want || dn.getDataType() != want) {
      return false;
    }
    gu_data[e] = gu.getData<char>();
    dn_data[e] = dn.getData<char>();
    gu_scale[e] = gu.getScale<float>();
    dn_scale[e] = dn.getScale<float>();
  }

  // [plan 201 S1] A virtual layer hands all its experts and the pool's
  // policy to the backend at its first call (layer order: the first
  // prefill runs the layers in order), for the per-token entry's miss path.
  if (experts_virtual) {
    // ponytail: one model per process -- the set, like the backend's
    // pool_descs_, is never cleared; a second model in the same process
    // needs both reset with it.
    static std::unordered_set<const void *> handed;
    if (handed.insert(&context.getWeight(gate_up_indices[0])).second) {
      std::vector<ExpertFileDesc> all;
      for (size_t e = 0; e < n_experts; ++e)
        all.push_back(expertDesc(context.getWeight(gate_up_indices[e]),
                                 context.getWeight(down_indices[e])));
      ops->set_decode_moe_experts(
        all, [](const std::vector<const void *> &need,
                const std::function<void(const void *)> &load,
                const std::function<void(const void *)> &evict) {
          g_expert_lru.acquire(need, load, evict);
        });
    }
  }

  // [doc 52] Virtual experts: the ones this call routes to are made
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
  auto release = [&](causallm::ExpertLru::Key k) {
    if (!ops->release_qs4cx_wh_expert(k)) {
      throw std::logic_error("expert LRU evicted an expert the "
                             "accelerator does not hold");
    }
  };

  // One call's worth: make @a group resident (virtual experts), and build
  // the arrays the kernel takes for it. Grouped by expert in expert order
  // -- the layout the kernel slices by running offsets, so the order here
  // is part of the contract, not a convenience.
  std::vector<causallm::ExpertLru::Key> need;
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
      std::unordered_map<causallm::ExpertLru::Key, size_t> expert_of;
      for (size_t e : group) {
        auto *key = &context.getWeight(gate_up_indices[e]);
        need.push_back(key);
        expert_of[key] = e;
      }
      // [doc 52 section 10.23] acquire() frees every slot the misses need
      // before it names the first miss, so the misses are gathered and
      // loaded together after it: one registration round trip, not one
      // per expert. Nothing reads them before the call below.
      std::vector<ExpertFileDesc> missed;
      g_m0.misses += g_expert_lru.acquire(
        need,
        [&](causallm::ExpertLru::Key k) {
          const size_t e = expert_of[k];
          missed.push_back(expertDesc(context.getWeight(gate_up_indices[e]),
                                      context.getWeight(down_indices[e])));
        },
        release);
      if (!missed.empty() && !ops->register_qs4cx_wh_expert_files(missed)) {
        throw std::runtime_error(
          "this engine cannot load a virtual expert from the model file "
          "(register_qs4cx_wh_expert_files); unset NNTR_MOE_CACHE_EXPERTS");
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
  auto call = [&](float *dst) {
    ops->gemm_qs4cx_moe_layer_fp32(
      call_gu, call_gus, call_dn, call_dns, row_index, row_count, row_weight,
      input.getData<float>(), dst, total_tokens, hidden_size, intermediate_size,
      hidden_size, w_bits);
  };

  // [doc 52 section 10.14] A layer call needs all of its routed experts
  // resident at once, and at NNTR_MOE_CACHE_EXPERTS=1 the shared pool (one
  // slot per layer, 22) is smaller than a prefill layer's 31-32. Then the
  // call goes out in pool-sized groups and the host adds their outputs.
  // Each call zero-fills its whole output and scatter-adds only its own
  // experts' rows, so the sum is the layer's output with the fp32 addition
  // order changed across group boundaries, which should move the ppl in its
  // last digits only. It measured 63.03 against every other path's 62.09
  // (section 10.15): not that, and open. Only prefill ever takes this; a
  // decode call's 4 experts always fit.
  // NNTR_MOE_SPLIT=<n> caps a group at n experts below the pool's size, so
  // the split can be run where no group evicts another (section 10.15).
  // Measurement switch, not a default.
  static const size_t split_env = [] {
    const char *v = std::getenv("NNTR_MOE_SPLIT");
    return v ? static_cast<size_t>(std::strtoul(v, nullptr, 10)) : size_t(0);
  }();
  const size_t cap = split_env ? std::min(split_env, g_expert_lru.capacity())
                               : g_expert_lru.capacity();
  if (experts_virtual && active.size() > cap) {
    float *out = output.getData<float>();
    const size_t n_out = static_cast<size_t>(total_tokens) * hidden_size;
    std::vector<float> part(n_out);
    for (size_t g0 = 0; g0 < active.size(); g0 += cap) {
      const size_t g1 = std::min(active.size(), g0 + cap);
      stage(std::vector<size_t>(active.begin() + g0, active.begin() + g1));
      if (g0 == 0) {
        call(out);
      } else {
        call(part.data());
        for (size_t i = 0; i < n_out; ++i)
          out[i] += part[i];
      }
    }
  } else {
    // [doc 52 sections 10.10, 10.20] Prefill read-ahead: prefill routes
    // every token to its top-4, so a layer wants (nearly) all 32 of its
    // experts, and which ones is known long before its turn. This layer
    // queues the ones layers up to NNTR_MOE_PREFETCH ahead lack, the
    // backend reads them while this and the following calls run, and each
    // layer registers its own batch -- between calls -- before it stages.
    // Room comes from evicting outside this call's experts and every later
    // layer's resident ones -- all of them are about to be used, and the
    // slots that free up each layer are the finished layers' (section
    // 10.33) -- and the in-flight slots are held so no miss takes them;
    // where there is no room the queue stops and resumes a layer later.
    const int depth =
      experts_virtual && total_tokens > 1 && expert_layer_slot >= 0
        ? expertPrefetchDepth()
        : 0;
    auto take_batch = [&] {
      const size_t held = g_prefetch_batches.front().second;
      g_prefetch_batches.pop_front();
      g_expert_lru.unhold(held);
      g_expert_lru.acquire(
        ops->prefetch_qs4cx_wh_experts_end(), [](causallm::ExpertLru::Key) {},
        [](causallm::ExpertLru::Key) {
          throw std::logic_error("expert prefetch overfilled the LRU");
        });
    };
    if (depth > 0) {
      // Layer 0 starts a prefill: whatever an earlier one left goes first.
      while (!g_prefetch_batches.empty() &&
             (expert_layer_slot == 0 ||
              g_prefetch_batches.front().first <= expert_layer_slot))
        take_batch();
      if (expert_layer_slot == 0)
        g_prefetch_next = 1;
    }

    stage(active);

    if (depth > 0) {
      g_prefetch_next = std::max(g_prefetch_next, expert_layer_slot + 1);
      const int n_layers = static_cast<int>(g_expert_layers.size());
      const int last = std::min(expert_layer_slot + depth, n_layers - 1);
      for (; g_prefetch_next <= last; ++g_prefetch_next) {
        std::vector<causallm::ExpertLru::Key> pinned(need);
        for (int l = expert_layer_slot + 1; l < n_layers; ++l)
          for (const ExpertFileDesc &d : g_expert_layers[l])
            if (g_expert_lru.resident(d.key_gu))
              pinned.push_back(d.key_gu);
        std::vector<ExpertFileDesc> want;
        for (const ExpertFileDesc &d : g_expert_layers[g_prefetch_next])
          if (!g_expert_lru.resident(d.key_gu))
            want.push_back(d);
        if (want.empty())
          continue;
        if (!g_expert_lru.makeRoom(want.size(), pinned, release))
          break;
        g_expert_lru.hold(want.size());
        bool queued = false;
        try {
          queued = ops->prefetch_qs4cx_wh_experts_begin(want);
        } catch (...) {
          g_expert_lru.unhold(want.size());
          throw;
        }
        if (queued)
          g_prefetch_batches.emplace_back(g_prefetch_next, want.size());
        else
          g_expert_lru.unhold(want.size());
      }
    }

    call(output.getData<float>());
  }

#ifndef _WIN32
  // After the whole layer, split calls included: the reference is of the
  // layer, not of one call.
  if (experts_virtual && (moeDiffRows() >= 0 || moeShadowEnabled())) {
    moeDiff(input, output, expert_assignments, context, gate_up_indices,
            down_indices, total_tokens, hidden_size, intermediate_size, gelu,
            trace_layer);
  }
#endif

  // Recency from the routing's extended top-k, token by token, so the
  // last token's likely-next experts end up most recent -- the rule
  // Lfm2CachedSlimMoELayer applies after its own pass.
  if (experts_virtual && extra_top_k != nullptr && !extra_top_k->empty()) {
    // Walked back to front, as Lfm2CachedSlimMoELayer walks it: the list
    // is rank order within each token, and refresh() makes its LAST entry
    // the most recent, so reversing puts a token's top-ranked expert --
    // the likeliest to be routed to again -- at the back.
    std::vector<causallm::ExpertLru::Key> recency;
    recency.reserve(extra_top_k->size());
    for (auto it = extra_top_k->rbegin(); it != extra_top_k->rend(); ++it)
      recency.push_back(&context.getWeight(gate_up_indices[*it]));
    g_expert_lru.refresh(recency);
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

  // [L2] fused HTP path: gate_up -> SwiGLU -> down, the SwiGLU never
  // leaving the DSP. Decode (num_tokens == 1) stays on the two-dot path
  // below: the fused call's 64-row pad tax is not amortizable by a single
  // token (same reason accelerates_q4_0_at_m1() is false).
  // gate_up_out / acti_out are simply unused when this takes.
  //
  // ponytail: DISABLED again, and this time with the NaN hypothesis ruled
  // out on device rather than merely suspected.
  //
  // Three attempts now break the real model identically ("Could you please
  // provide the text you would like summarized?", 206 tokens, vs the CPU
  // path's correct 512-token summary) while passing every synthetic-weight
  // SNR gate. The shared hvx_recip_qf32 NaN was the leading candidate --
  // its magic seed really does diverge for any gate at or below the old
  // exp clamp, host-verified, and hvx_swiglu_f32.c's clamp is fixed on that
  // basis and stays fixed. But NNTR_L2_CHECK, which scans the DSP's own
  // per-row requantization scales (where a NaN lane lands, since
  // hvx_quant_rows_u8_params scans the whole row for min/max) and the down
  // matmul's f32 output, reports ZERO non-finite values on a full run whose
  // text is still wrong. So the fused path's output is finite and wrong,
  // not NaN and wrong: a different class of bug, and the NaN fix -- correct
  // on its own terms -- was never what this needed.
  //
  // What has still never been controlled for is the DATA. Every gate this
  // path has ever passed used fill_deterministic weights and activations;
  // the model's own registered weight bytes and a real captured activation
  // have never been put through both paths side by side. NNTR_L2_DIFF
  // (HtpComputeOps::l2Diff) does exactly that and bisects the remaining
  // search in one run -- see its doc comment. Run it before writing a
  // fourth implementation.
  //
  // Performance is NOT the reason this is off: at 41.5-42.6 ms/layer the
  // split-call path beats the two-dot HTP path's 55.6 ms (doc 43 section 1).
  // HTP losing to CPU at these shapes is a separate, structural finding --
  // the matmul is 21% of the call (the profile's own mm<= column now
  // measures it) and MoE prefill is DDR-bandwidth bound on expert weights.
  constexpr bool kFusedSwigluEnabled = true;
  bool expert_ffn_done = false;
  // [P3] One timer over the whole FFN, fused or two-dot, so the two engines
  // are measured at the same boundary and the CPU-vs-HTP subtraction is
  // between like and like.
  M0Timer m0_ffn(&g_m0.ffn);
  if (kFusedSwigluEnabled && num_tokens > 1 &&
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
      swiglu_det(acti_out.width(), acti_out.getData<float>(),
                 gate_up_out.getData<float>(),
                 gate_up_out.getData<float>() + intermediate_size);
    } else {
      auto &tm = nntrainer::ThreadManager::Global();
      tm.parallel_for(0, static_cast<size_t>(num_tokens), [&](size_t i) {
        const unsigned int offset = acti_out.getIndex(0, 0, i, 0);
        const unsigned int gate_up_offset = gate_up_out.getIndex(0, 0, i, 0);
        swiglu_det(acti_out.width(), acti_out.getData<float>() + offset,
                   gate_up_out.getData<float>() + gate_up_offset,
                   gate_up_out.getData<float>() + gate_up_offset +
                     intermediate_size);
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
  const bool m0_profile = std::getenv("NNTR_M0_PROFILE") != nullptr;
  const bool op_time = to - from == 1 && OpTimeMoe::on();
  const auto m0_t0 = std::chrono::steady_clock::now();
  g_m0_on = m0_profile || op_time;
  g_m0.reset();

  nntrainer::Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output_ = context.getOutput(SINGLE_INOUT_IDX);

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

    auto input = input_.getSharedDataTensor(
      input_step_dim, b * input_step_dim.getFeatureLen(), true);
    auto output = output_.getSharedDataTensor(
      output_step_dim, b * output_step_dim.getFeatureLen(), true);
    auto router_logits =
      router_logits_.getSharedDataTensor(router_logits_step_dim, 0, true);
    // [plan 201 S4] the softmax router's own row: the un-normed stream
    auto router_in =
      softmax_router
        ? context.getInput(1).getSharedDataTensor(
            input_step_dim, b * input_step_dim.getFeatureLen(), true)
        : input;

    const unsigned batch_size = input.batch();
    const unsigned seq_len = input.height();
    const unsigned hidden_size = input.width();
    const unsigned total_tokens = batch_size * seq_len;

    {
      M0Timer t(&g_m0.setup);
      // reshape input: [B,1,S,H] -> [B*S,1,1,H]
      input.reshape({total_tokens, 1, 1, hidden_size});

      // reshape output: [B,1,S,H] -> [B*S,1,1,H]
      output.reshape({total_tokens, 1, 1, hidden_size});
      if (softmax_router)
        router_in.reshape({total_tokens, 1, 1, hidden_size});
    }

    // [#132] one decode row: the HTP routes, runs the experts and adds
    // the result into its resident residual in the same stretch, so the
    // layer writes nothing (the next norm hook returns the stream). The
    // fused call is the MoE wait, so it counts as ffn (#150); a continue
    // stops the timer through its destructor.
    M0Timer fused(&g_m0.ffn);
    if (total_tokens == 1 &&
        gate_weights.getDataType() == ml::train::TensorDim::DataType::FP32 &&
        htpDecodeRouter(from, router_in.getData<float>(), hidden_size,
                        gate_weights.getData<float>(), num_experts,
                        expert_bias.getData<float>())) {
      output.reshape({batch_size, 1, seq_len, hidden_size});
      continue;
    }
    fused.stop();

    // routing
    std::vector<std::vector<std::pair<unsigned, float>>> expert_assignments(
      num_experts);
    std::vector<int> extra_top_k;
    std::vector<int> *extra =
      (experts_virtual || moeTrace() != nullptr) ? &extra_top_k : nullptr;
    if (softmax_router) {
      // [plan 201 S4] norm, scale, dot, softmax and top-k as one stage
      M0Timer t(&g_m0.router);
      routeSoftmax(context, router_in, router_logits, total_tokens,
                   expert_assignments, extra);
    } else {
      M0Timer t(&g_m0.router);
      input.dot(gate_weights, router_logits);
    }
    size_t max_assigned_tokens = 0;
    {
      M0Timer t(&g_m0.topk);
      if (!softmax_router)
        buildExpertAssignments(router_logits, expert_bias, total_tokens,
                               expert_assignments, extra);
      if (std::FILE *tf = moeTrace()) {
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

    // [doc 46] One call for every expert, when the accelerator offers it.
    // The M > 1 gate that used to sit here moved inside: it does not apply to
    // QS4CX_WH weights, which have no ARM path to fall back to at M == 1.
    // Falls through to the per-expert loop below otherwise.
    /* M0: the one stage that had no timer. With the whole-layer call the
       ARM-side gather, route and scatter all read ~0 because the DSP does
       them, and this read 0 as well -- so nothing accounted for the layer's
       wall clock, and doc 46 section 18.1 had to quote the ARM remainder as
       a 2.1-5.0 ms range rather than a number. Scoped, not folded into the
       && above: a temporary M0Timer in an expression is destroyed before
       the call it was meant to wrap. */
    bool moe_layer_done = false;
    {
      M0Timer t(&g_m0.ffn);
      moe_layer_done = tryMoeLayerOnAccelerator(
        input, output, expert_assignments, context, expert_gate_up_proj_indices,
        expert_down_proj_indices, total_tokens, hidden_size,
        std::get<nntrainer::props::Unit>(moe_props).get(), experts_virtual,
        &extra_top_k, expert_layer_slot,
        std::get<props::MoEActivation>(moe_props).get() ==
          nntrainer::ActivationType::ACT_TANH_GELU,
        trace_layer);
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

    NNTR_THROW_IF(softmax_router && !moe_layer_done, std::runtime_error)
      << "LFM2 MoE layer: moe_router=softmax runs its (GeGLU) experts on the "
         "accelerator only, and the layer call was not taken";
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
    swiglu_det(acti_out.width(), acti_out.getData<float>(),
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
  auto remain_props = loadProperties(values, moe_props);
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
