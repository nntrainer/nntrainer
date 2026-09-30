// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   gemma4_moe_layer.cpp
 * @brief  Gemma4 sparse Mixture-of-Experts inference layer.
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug    No known bugs
 */

#include <gemma4_moe_layer.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cpu_backend.h>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <limits>
#include <list>
#include <mutex>
#include <node_exporter.h>
#include <stdexcept>
#include <string>
#include <thread_manager.h>
#include <unordered_map>

namespace causallm {

namespace {

struct Gemma4ExpertWeights {
  nntrainer::Tensor *gate;
  nntrainer::Tensor *up;
  nntrainer::Tensor *down;
};

/**
 * @brief Per-phase timing for the expert loop, opt-in via NNTR_MOE_PROFILE.
 *
 * The expert loop is serial, so plain statics are safe here. Accumulated
 * across one forwardTensors() call and printed as a single line, letting us
 * separate the threaded GEMMs from the single-threaded gather/activation/
 * scatter glue.
 */
struct Gemma4MoEPhaseStats {
  uint64_t gather_ns = 0;
  uint64_t gemm_ns = 0;
  uint64_t acti_ns = 0;
  uint64_t scatter_ns = 0;
  // pre-loop phases
  uint64_t setzero_ns = 0;
  uint64_t rmsnorm_ns = 0;
  uint64_t rscale_ns = 0;
  uint64_t rdot_ns = 0;
  uint64_t softmax_ns = 0;
  uint64_t topk_ns = 0;
  uint64_t renorm_ns = 0;
  uint64_t bucket_ns = 0;

  void reset() {
    gather_ns = gemm_ns = acti_ns = scatter_ns = 0;
    setzero_ns = rmsnorm_ns = rscale_ns = rdot_ns = 0;
    softmax_ns = topk_ns = renorm_ns = bucket_ns = 0;
  }
};

bool moeProfileEnabled() {
  static const bool enabled = std::getenv("NNTR_MOE_PROFILE") != nullptr;
  return enabled;
}

Gemma4MoEPhaseStats &moePhaseStats() {
  static Gemma4MoEPhaseStats stats;
  return stats;
}

inline uint64_t nowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
           std::chrono::high_resolution_clock::now().time_since_epoch())
    .count();
}

} // namespace

/**
 * @brief Bounded LRU for virtual Gemma4 expert weights.
 *
 * At most one asynchronous activation is issued ahead of the currently
 * executing expert. Capacity includes both resident and loading experts, so
 * the configured mmap bound is not temporarily exceeded during prefetch.
 */
class Gemma4ExpertCache {
public:
  /// @brief hit / miss / eviction tallies, reported under NNTR_MOE_PROFILE
  static std::atomic<uint64_t> &hits() {
    static std::atomic<uint64_t> v{0};
    return v;
  }
  static std::atomic<uint64_t> &misses() {
    static std::atomic<uint64_t> v{0};
    return v;
  }
  static std::atomic<uint64_t> &evictions() {
    static std::atomic<uint64_t> v{0};
    return v;
  }

  Gemma4ExpertCache() = default;

  ~Gemma4ExpertCache() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      shutting_down = true;
    }
    condition.notify_all();
    std::lock_guard<std::mutex> task_lock(task_mutex);
    if (prefetch_task.valid())
      prefetch_task.wait();
  }

  Gemma4ExpertCache(const Gemma4ExpertCache &) = delete;
  Gemma4ExpertCache &operator=(const Gemma4ExpertCache &) = delete;

  void registerWeights(std::vector<Gemma4ExpertWeights> expert_weights,
                       unsigned int cache_capacity) {
    std::lock_guard<std::mutex> lock(mutex);
    if (!weights.empty())
      return;
    NNTR_THROW_IF(cache_capacity == 0, std::invalid_argument)
      << "Gemma4 virtual expert cache requires non-zero capacity";

    weights = std::move(expert_weights);
    capacity = std::min<unsigned int>(cache_capacity, weights.size());
    status.assign(weights.size(), Status::UNLOADED);
    pin_count.assign(weights.size(), 0);
    errors.resize(weights.size());
  }

  bool registered() const {
    std::lock_guard<std::mutex> lock(mutex);
    return !weights.empty();
  }

  void prefetch(unsigned int expert) {
    std::lock_guard<std::mutex> task_lock(task_mutex);
    if (prefetch_task.valid())
      prefetch_task.get();

    {
      std::lock_guard<std::mutex> lock(mutex);
      if (shutting_down || expert >= status.size() ||
          status[expert] != Status::UNLOADED)
        return;
    }

    prefetch_task = std::async(std::launch::async,
                               [this, expert]() { ensureResident(expert); });
  }

  void acquire(unsigned int expert) {
    ensureResident(expert);

    std::lock_guard<std::mutex> lock(mutex);
    NNTR_THROW_IF(shutting_down, std::runtime_error)
      << "Gemma4 expert cache is shutting down";
    NNTR_THROW_IF(expert >= status.size(), std::out_of_range)
      << "Gemma4 expert index is out of range";
    if (status[expert] == Status::FAILED)
      std::rethrow_exception(errors[expert]);
    NNTR_THROW_IF(status[expert] != Status::RESIDENT, std::logic_error)
      << "Gemma4 expert did not become resident";
    ++pin_count[expert];
    touch(expert);
  }

  void release(unsigned int expert) {
    std::lock_guard<std::mutex> lock(mutex);
    NNTR_THROW_IF(expert >= pin_count.size() || pin_count[expert] == 0,
                  std::logic_error)
      << "Gemma4 expert cache pin count underflow";
    --pin_count[expert];
    condition.notify_all();
  }

private:
  enum class Status { UNLOADED, LOADING, RESIDENT, EVICTING, FAILED };

  void touch(unsigned int expert) {
    auto found = lru_position.find(expert);
    if (found != lru_position.end())
      lru.erase(found->second);
    lru.push_back(expert);
    lru_position[expert] = --lru.end();
  }

  size_t occupiedCount() const {
    return std::count_if(status.begin(), status.end(), [](Status current) {
      return current == Status::LOADING || current == Status::RESIDENT ||
             current == Status::EVICTING;
    });
  }

  void deactivate(const Gemma4ExpertWeights &target) {
    std::exception_ptr error;
    try {
      target.gate->deactivate();
    } catch (...) {
      error = std::current_exception();
    }
    try {
      target.up->deactivate();
    } catch (...) {
      if (error == nullptr)
        error = std::current_exception();
    }
    try {
      target.down->deactivate();
    } catch (...) {
      if (error == nullptr)
        error = std::current_exception();
    }
    if (error != nullptr)
      std::rethrow_exception(error);
  }

  void activate(unsigned int expert) {
    bool gate_attempted = false;
    bool up_attempted = false;
    bool down_attempted = false;
    try {
      gate_attempted = true;
      weights[expert].gate->activate();
      up_attempted = true;
      weights[expert].up->activate();
      down_attempted = true;
      weights[expert].down->activate();

      {
        std::lock_guard<std::mutex> lock(mutex);
        status[expert] = Status::RESIDENT;
        touch(expert);
      }
    } catch (...) {
      const auto activation_error = std::current_exception();
      try {
        if (down_attempted)
          weights[expert].down->deactivate();
      } catch (...) {
      }
      try {
        if (up_attempted)
          weights[expert].up->deactivate();
      } catch (...) {
      }
      try {
        if (gate_attempted)
          weights[expert].gate->deactivate();
      } catch (...) {
      }

      std::lock_guard<std::mutex> lock(mutex);
      errors[expert] = activation_error;
      status[expert] = Status::FAILED;
    }
    condition.notify_all();
  }

  void ensureResident(unsigned int expert) {
    bool counted = false;
    while (true) {
      Gemma4ExpertWeights eviction_target{};
      unsigned int eviction_index = 0;
      bool should_evict = false;

      {
        std::unique_lock<std::mutex> lock(mutex);
        NNTR_THROW_IF(expert >= status.size(), std::out_of_range)
          << "Gemma4 expert index is out of range";
        NNTR_THROW_IF(shutting_down, std::runtime_error)
          << "Gemma4 expert cache is shutting down";

        condition.wait(lock, [&]() {
          return (status[expert] != Status::LOADING &&
                  status[expert] != Status::EVICTING) ||
                 shutting_down;
        });
        NNTR_THROW_IF(shutting_down, std::runtime_error)
          << "Gemma4 expert cache is shutting down";

        if (status[expert] == Status::RESIDENT) {
          if (!counted)
            hits().fetch_add(1, std::memory_order_relaxed);
          touch(expert);
          return;
        }
        if (!counted) {
          misses().fetch_add(1, std::memory_order_relaxed);
          counted = true;
        }
        if (status[expert] == Status::FAILED)
          std::rethrow_exception(errors[expert]);

        if (occupiedCount() < capacity) {
          status[expert] = Status::LOADING;
          lock.unlock();
          activate(expert);
          lock.lock();
          if (status[expert] == Status::FAILED)
            std::rethrow_exception(errors[expert]);
          return;
        }

        evictions().fetch_add(1, std::memory_order_relaxed);
        auto candidate =
          std::find_if(lru.begin(), lru.end(), [&](unsigned int current) {
            return current != expert && status[current] == Status::RESIDENT &&
                   pin_count[current] == 0;
          });
        if (candidate == lru.end()) {
          condition.wait(lock);
          continue;
        }

        eviction_index = *candidate;
        eviction_target = weights[eviction_index];
        lru_position.erase(eviction_index);
        lru.erase(candidate);
        status[eviction_index] = Status::EVICTING;
        should_evict = true;
      }

      if (should_evict) {
        std::exception_ptr eviction_error;
        try {
          deactivate(eviction_target);
        } catch (...) {
          eviction_error = std::current_exception();
        }

        {
          std::lock_guard<std::mutex> lock(mutex);
          if (eviction_error == nullptr) {
            status[eviction_index] = Status::UNLOADED;
          } else {
            errors[eviction_index] = eviction_error;
            status[eviction_index] = Status::FAILED;
          }
        }
        condition.notify_all();
        if (eviction_error != nullptr)
          std::rethrow_exception(eviction_error);
      }
    }
  }

  mutable std::mutex mutex;
  std::mutex task_mutex;
  std::condition_variable condition;
  std::vector<Gemma4ExpertWeights> weights;
  std::vector<Status> status;
  std::vector<unsigned int> pin_count;
  std::vector<std::exception_ptr> errors;
  std::list<unsigned int> lru;
  std::unordered_map<unsigned int, std::list<unsigned int>::iterator>
    lru_position;
  unsigned int capacity = 0;
  bool shutting_down = false;
  std::future<void> prefetch_task;
};

namespace {

constexpr size_t EXPERT_INPUT_IDX = 0;
constexpr size_t ROUTER_INPUT_IDX = 1;

} // namespace

Gemma4MoELayer::Gemma4MoELayer() :
  LayerImpl(),
  num_experts(0),
  topk(0),
  cache_size(0),
  epsilon(1.0e-6f),
  moe_props(props::NumExperts(), props::NumExpertsPerToken(),
            nntrainer::props::Unit(), props::MoEActivation(),
            nntrainer::props::Epsilon(1.0e-6f), props::Gemma4MoECacheSize()),
  expert_cache(std::make_unique<Gemma4ExpertCache>()),
  expert_gate_proj_indices(),
  expert_up_proj_indices(),
  expert_down_proj_indices(),
  router_idx(std::numeric_limits<unsigned int>::max()),
  router_scale_idx(std::numeric_limits<unsigned int>::max()),
  per_expert_scale_idx(std::numeric_limits<unsigned int>::max()),
  router_input_scaled_idx(std::numeric_limits<unsigned int>::max()),
  router_logits_idx(std::numeric_limits<unsigned int>::max()),
  scratch_capacity(0) {}

Gemma4MoELayer::~Gemma4MoELayer() = default;

void Gemma4MoELayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 2, std::invalid_argument)
    << "Gemma4 MoE requires expert input and router input";

  const auto &expert_dim = context.getInputDimensions()[EXPERT_INPUT_IDX];
  const auto &router_dim = context.getInputDimensions()[ROUTER_INPUT_IDX];
  NNTR_THROW_IF(expert_dim != router_dim, std::invalid_argument)
    << "Gemma4 MoE expert and router input dimensions must match";
  NNTR_THROW_IF(context.getActivationDataType() !=
                  ml::train::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "Gemma4 MoE currently requires FP32 activations";

  context.setOutputDimensions({expert_dim});

  num_experts = std::get<props::NumExperts>(moe_props).get();
  topk = std::get<props::NumExpertsPerToken>(moe_props).get();
  cache_size = std::get<props::Gemma4MoECacheSize>(moe_props).get();
  epsilon = std::get<nntrainer::props::Epsilon>(moe_props).get();
  const unsigned int intermediate_size =
    std::get<nntrainer::props::Unit>(moe_props).get();
  const unsigned int hidden_size = expert_dim.width();
  NNTR_THROW_IF(num_experts == 0 || topk == 0 || topk > num_experts ||
                  intermediate_size == 0,
                std::invalid_argument)
    << "Gemma4 MoE requires non-zero experts, top-k, and intermediate size; "
       "top-k cannot exceed the number of experts";
  NNTR_THROW_IF(std::get<props::MoEActivation>(moe_props).empty(),
                std::invalid_argument)
    << "Gemma4 MoE activation must be specified";
#if defined(_WIN32)
  NNTR_THROW_IF(cache_size > 0, std::invalid_argument)
    << "Gemma4 virtual expert weights are not supported on Windows";
#endif
  acti_func.setActiFunc<float>(std::get<props::MoEActivation>(moe_props).get());

  auto &weight_regularizer =
    std::get<nntrainer::props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<nntrainer::props::WeightRegularizerConstant>(*layer_impl_props);
  auto &weight_initializer =
    std::get<nntrainer::props::WeightInitializer>(*layer_impl_props);
  auto &weight_decay =
    std::get<nntrainer::props::WeightDecay>(*layer_impl_props);

  const bool is_nchw = context.getFormat() == nntrainer::Tformat::NCHW;
  const auto fp32_type = nntrainer::TensorDim::TensorType(
    context.getFormat(), nntrainer::TensorDim::DataType::FP32);

  nntrainer::TensorDim router_dim_weight(
    1, is_nchw ? 1 : num_experts, is_nchw ? hidden_size : 1,
    is_nchw ? num_experts : hidden_size, fp32_type, is_nchw ? 0b0011 : 0b0101);
  router_idx = context.requestWeight(
    router_dim_weight, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "router", true);

  nntrainer::TensorDim router_scale_dim(1, 1, 1, hidden_size, fp32_type);
  router_scale_idx = context.requestWeight(
    router_scale_dim, nntrainer::Initializer::ONES,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "router_scale", true);

  nntrainer::TensorDim per_expert_scale_dim(1, 1, 1, num_experts, fp32_type);
  per_expert_scale_idx =
    context.requestWeight(per_expert_scale_dim, nntrainer::Initializer::ONES,
                          nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f,
                          "router_per_expert_scale", true);

  expert_gate_proj_indices.reserve(num_experts);
  expert_up_proj_indices.reserve(num_experts);
  expert_down_proj_indices.reserve(num_experts);

  nntrainer::TensorDim expert_gate_dim(
    1, is_nchw ? 1 : intermediate_size, is_nchw ? hidden_size : 1,
    is_nchw ? intermediate_size : hidden_size,
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
    expert_gate_proj_indices.push_back(context.requestWeight(
      expert_gate_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay,
      "expert_gate_" + std::to_string(i), false, cache_size > 0));
    expert_up_proj_indices.push_back(context.requestWeight(
      expert_gate_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay,
      "expert_up_" + std::to_string(i), false, cache_size > 0));
    expert_down_proj_indices.push_back(context.requestWeight(
      expert_down_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay,
      "expert_down_" + std::to_string(i), false, cache_size > 0));
  }

  const unsigned int total_tokens = expert_dim.batch() * expert_dim.height();
  router_input_scaled_idx = context.requestTensor(
    {total_tokens, 1, 1, hidden_size}, "router_input_scaled",
    nntrainer::Initializer::NONE, false,
    nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
  router_logits_idx =
    context.requestTensor({total_tokens, 1, 1, num_experts}, "router_logits",
                          nntrainer::Initializer::NONE, false,
                          nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
}

void Gemma4MoELayer::forwardTensors(nntrainer::RunLayerContext &context,
                                    nntrainer::Tensor &expert_input,
                                    nntrainer::Tensor &router_input,
                                    nntrainer::Tensor &output,
                                    nntrainer::Tensor &router_input_scaled,
                                    nntrainer::Tensor &router_logits) {
  const unsigned int total_tokens =
    expert_input.batch() * expert_input.height();
  const unsigned int hidden_size = expert_input.width();

  expert_input.reshape({total_tokens, 1, 1, hidden_size});
  router_input.reshape({total_tokens, 1, 1, hidden_size});
  output.reshape({total_tokens, 1, 1, hidden_size});
  router_input_scaled.reshape({total_tokens, 1, 1, hidden_size});
  router_logits.reshape({total_tokens, 1, 1, num_experts});
  const bool pre_profile = moeProfileEnabled();
  auto &pre_stats = moePhaseStats();
  if (pre_profile)
    pre_stats.reset();
  uint64_t tp = pre_profile ? nowNs() : 0;

  output.setZero();
  if (pre_profile) {
    uint64_t t = nowNs();
    pre_stats.setzero_ns += t - tp;
    tp = t;
  }

  nntrainer::rms_norm_wrt_width_fp32_intrinsic(
    router_input.getData<float>(), router_input_scaled.getData<float>(),
    total_tokens, hidden_size, epsilon);
  if (pre_profile) {
    uint64_t t = nowNs();
    pre_stats.rmsnorm_ns += t - tp;
    tp = t;
  }

  const float *router_scale =
    context.getWeight(router_scale_idx).getData<float>();
  float *scaled_data = router_input_scaled.getData<float>();
  const float hidden_scale = 1.0f / std::sqrt(static_cast<float>(hidden_size));
  for (unsigned int token = 0; token < total_tokens; ++token) {
    float *row = scaled_data + token * hidden_size;
    for (unsigned int feature = 0; feature < hidden_size; ++feature)
      row[feature] *= router_scale[feature] * hidden_scale;
  }
  if (pre_profile) {
    uint64_t t = nowNs();
    pre_stats.rscale_ns += t - tp;
    tp = t;
  }

  router_input_scaled.dot(context.getWeight(router_idx), router_logits);
  if (pre_profile) {
    uint64_t t = nowNs();
    pre_stats.rdot_ns += t - tp;
    tp = t;
  }

  router_logits.apply(nntrainer::ActiFunc::softmax<float>, router_logits);
  if (pre_profile) {
    uint64_t t = nowNs();
    pre_stats.softmax_ns += t - tp;
    tp = t;
  }

  auto topk_result = router_logits.topK(topk);
  if (pre_profile) {
    uint64_t t = nowNs();
    pre_stats.topk_ns += t - tp;
    tp = t;
  }
  auto topk_values = std::get<0>(topk_result);
  auto topk_indices = std::get<1>(topk_result);

  const uint32_t *indices_data = topk_indices.getData<uint32_t>();
  float *values_data = topk_values.getData<float>();
  const float *per_expert_scale =
    context.getWeight(per_expert_scale_idx).getData<float>();
  for (unsigned int token = 0; token < total_tokens; ++token) {
    float sum = 0.0f;
    for (unsigned int k = 0; k < topk; ++k)
      sum += values_data[token * topk + k];
    for (unsigned int k = 0; k < topk; ++k) {
      const unsigned int offset = token * topk + k;
      values_data[offset] =
        values_data[offset] / sum * per_expert_scale[indices_data[offset]];
    }
  }
  if (pre_profile) {
    uint64_t t = nowNs();
    pre_stats.renorm_ns += t - tp;
    tp = t;
  }

  std::vector<std::vector<std::pair<unsigned int, float>>> expert_assignments(
    num_experts);
  for (unsigned int token = 0; token < total_tokens; ++token) {
    for (unsigned int k = 0; k < topk; ++k) {
      const unsigned int offset = token * topk + k;
      expert_assignments[indices_data[offset]].emplace_back(
        token, values_data[offset]);
    }
  }

  std::vector<unsigned int> active_experts;
  active_experts.reserve(num_experts);
  for (unsigned int expert = 0; expert < num_experts; ++expert) {
    if (!expert_assignments[expert].empty())
      active_experts.push_back(expert);
  }

  if (pre_profile)
    pre_stats.bucket_ns += nowNs() - tp;

  if (cache_size > 0 && !active_experts.empty())
    expert_cache->prefetch(active_experts.front());

  const bool moe_profile = moeProfileEnabled();

  uint64_t loop_start = 0;
  if (moe_profile)
    loop_start = nowNs();

  for (size_t active = 0; active < active_experts.size(); ++active) {
    const unsigned int expert = active_experts[active];
    if (cache_size > 0)
      expert_cache->acquire(expert);

    try {
      if (cache_size > 0 && active + 1 < active_experts.size())
        expert_cache->prefetch(active_experts[active + 1]);
      computeExpertForward(expert_input, output, expert_assignments[expert],
                           context.getWeight(expert_gate_proj_indices[expert]),
                           context.getWeight(expert_up_proj_indices[expert]),
                           context.getWeight(expert_down_proj_indices[expert]),
                           hidden_size);
    } catch (...) {
      if (cache_size > 0)
        expert_cache->release(expert);
      throw;
    }

    if (cache_size > 0)
      expert_cache->release(expert);
  }

  if (moe_profile) {
    const auto &s = moePhaseStats();
    uint64_t loop_ns = nowNs() - loop_start;
    std::cout << "MOEPROF\ttokens=" << total_tokens
              << "\tactive_experts=" << active_experts.size()
              << "\tloop_ms=" << loop_ns / 1e6
              << "\tgather_ms=" << s.gather_ns / 1e6
              << "\tgemm_ms=" << s.gemm_ns / 1e6
              << "\tacti_ms=" << s.acti_ns / 1e6
              << "\tscatter_ms=" << s.scatter_ns / 1e6
              << "\tsetzero_ms=" << s.setzero_ns / 1e6
              << "\trmsnorm_ms=" << s.rmsnorm_ns / 1e6
              << "\trscale_ms=" << s.rscale_ns / 1e6
              << "\trdot_ms=" << s.rdot_ns / 1e6
              << "\tsoftmax_ms=" << s.softmax_ns / 1e6
              << "\ttopk_ms=" << s.topk_ns / 1e6
              << "\trenorm_ms=" << s.renorm_ns / 1e6
              << "\tbucket_ms=" << s.bucket_ns / 1e6
              << "\tcache_hit=" << Gemma4ExpertCache::hits().load()
              << "\tcache_miss=" << Gemma4ExpertCache::misses().load()
              << "\tcache_evict=" << Gemma4ExpertCache::evictions().load()
              << "\n";
  }
}

void Gemma4MoELayer::registerExpertCache(nntrainer::RunLayerContext &context) {
  if (cache_size == 0 || expert_cache->registered())
    return;

  std::vector<Gemma4ExpertWeights> weights;
  weights.reserve(num_experts);
  for (unsigned int expert = 0; expert < num_experts; ++expert) {
    weights.push_back({&context.getWeight(expert_gate_proj_indices[expert]),
                       &context.getWeight(expert_up_proj_indices[expert]),
                       &context.getWeight(expert_down_proj_indices[expert])});
  }
  expert_cache->registerWeights(std::move(weights), cache_size);
}

void Gemma4MoELayer::ensureExpertScratch(
  unsigned int tokens, unsigned int hidden_size, unsigned int intermediate_size,
  nntrainer::TensorDim::TensorType tensor_type) {
  if (tokens <= scratch_capacity)
    return;

  scratch_gathered = nntrainer::Tensor(1, 1, tokens, hidden_size, tensor_type);
  scratch_gate_out =
    nntrainer::Tensor(1, 1, tokens, intermediate_size, tensor_type);
  scratch_up_out =
    nntrainer::Tensor(1, 1, tokens, intermediate_size, tensor_type);
  scratch_activated =
    nntrainer::Tensor(1, 1, tokens, intermediate_size, tensor_type);
  scratch_down_out = nntrainer::Tensor(1, 1, tokens, hidden_size, tensor_type);
  scratch_capacity = tokens;
}

void Gemma4MoELayer::computeExpertForward(
  const nntrainer::Tensor &input, nntrainer::Tensor &output,
  const std::vector<std::pair<unsigned int, float>> &token_assignments,
  const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
  const nntrainer::Tensor &down_proj, unsigned int hidden_size) {
  const unsigned int num_tokens = token_assignments.size();
  const unsigned int intermediate_size = gate_proj.width();
  if (num_tokens == 0)
    return;

  const bool profile = moeProfileEnabled();
  auto &stats = moePhaseStats();
  uint64_t t0 = profile ? nowNs() : 0;

  const auto tensor_type = input.getTensorType();
  ensureExpertScratch(num_tokens, hidden_size, intermediate_size, tensor_type);

  // Slice the pre-sized scratch down to this expert's token count. The GEMM
  // reads M from the tensor height, so the view has to match num_tokens
  // exactly rather than the capacity.
  nntrainer::TensorDim hidden_dim({1, 1, num_tokens, hidden_size}, tensor_type);
  nntrainer::TensorDim intermediate_dim({1, 1, num_tokens, intermediate_size},
                                        tensor_type);
  nntrainer::Tensor gathered =
    scratch_gathered.getSharedDataTensor(hidden_dim, 0, true);
  nntrainer::Tensor gate_out =
    scratch_gate_out.getSharedDataTensor(intermediate_dim, 0, true);
  nntrainer::Tensor up_out =
    scratch_up_out.getSharedDataTensor(intermediate_dim, 0, true);
  nntrainer::Tensor activated =
    scratch_activated.getSharedDataTensor(intermediate_dim, 0, true);
  nntrainer::Tensor down_out =
    scratch_down_out.getSharedDataTensor(hidden_dim, 0, true);

  const float *input_data = input.getData<float>();
  float *gathered_data = gathered.getData<float>();
  // Writing into a small buffer that is reused for every expert keeps the
  // rows in L2 for the GEMM that immediately follows. Gathering all experts
  // up front into one large buffer instead measured ~23% slower for exactly
  // that reason, even though it cut the gather itself by 40%.
  // Safe to parallelize: the enclosing expert loop is serial, and the inner
  // GEMMs (which do call parallel_for) are not active here.
  if (num_tokens > 1) {
    nntrainer::ThreadManager::Global().parallel_for(
      0, static_cast<size_t>(num_tokens), [&](size_t i) {
        std::memcpy(gathered_data + i * hidden_size,
                    input_data + token_assignments[i].first * hidden_size,
                    hidden_size * sizeof(float));
      });
  } else {
    std::memcpy(gathered_data,
                input_data + token_assignments[0].first * hidden_size,
                hidden_size * sizeof(float));
  }
  if (profile) {
    uint64_t t = nowNs();
    stats.gather_ns += t - t0;
    t0 = t;
  }

  // gate and up share an activation, so handing both to the batched GEMM
  // (Tensor::dot(vector)) would quantize it to q8_0 once instead of twice.
  // Measured on SD 8 Gen 3 that is a ~21% prefill LOSS: with M ~= 54 the
  // activation is tiny next to the 3.3 MB of expert weights, so sharing its
  // quantization saves almost nothing while running both weights
  // concurrently doubles the resident weight working set. Keep them separate.
  gathered.dot(gate_proj, gate_out);
  gathered.dot(up_proj, up_out);
  if (profile) {
    uint64_t t = nowNs();
    stats.gemm_ns += t - t0;
    t0 = t;
  }

  acti_func.run_fn(gate_out, activated);
  activated.multiply_i(up_out);
  if (profile) {
    uint64_t t = nowNs();
    stats.acti_ns += t - t0;
    t0 = t;
  }

  activated.dot(down_proj, down_out);
  if (profile) {
    uint64_t t = nowNs();
    stats.gemm_ns += t - t0;
    t0 = t;
  }

  const float *down_data = down_out.getData<float>();
  float *output_data = output.getData<float>();
  // Each token appears at most once in a given expert's assignment list, so
  // distinct i never alias the same destination row - no race.
  auto scatter_row = [&](size_t i) {
    const unsigned int token = token_assignments[i].first;
    const float routing_weight = token_assignments[i].second;
    const float *source = down_data + i * hidden_size;
    float *destination = output_data + token * hidden_size;
    nntrainer::saxpy(hidden_size, routing_weight, source, 1, destination, 1);
  };
  if (num_tokens > 1) {
    nntrainer::ThreadManager::Global().parallel_for(
      0, static_cast<size_t>(num_tokens), scatter_row);
  } else {
    scatter_row(0);
  }
  if (profile)
    stats.scatter_ns += nowNs() - t0;
}

void Gemma4MoELayer::forwarding(nntrainer::RunLayerContext &context,
                                bool training) {
  (void)training;
  registerExpertCache(context);
  auto &expert_input = context.getInput(EXPERT_INPUT_IDX);
  auto &router_input = context.getInput(ROUTER_INPUT_IDX);
  auto &output = context.getOutput(0);
  auto &router_input_scaled = context.getTensor(router_input_scaled_idx);
  auto &router_logits = context.getTensor(router_logits_idx);

  const auto expert_dim = expert_input.getDim();
  const auto router_dim = router_input.getDim();
  const auto output_dim = output.getDim();
  forwardTensors(context, expert_input, router_input, output,
                 router_input_scaled, router_logits);
  expert_input.reshape(expert_dim);
  router_input.reshape(router_dim);
  output.reshape(output_dim);
}

void Gemma4MoELayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                            unsigned int from, unsigned int to,
                                            bool training) {
  (void)training;
  registerExpertCache(context);
  auto &expert_input_all = context.getInput(EXPERT_INPUT_IDX);
  auto &router_input_all = context.getInput(ROUTER_INPUT_IDX);
  auto &output_all = context.getOutput(0);
  auto &router_input_scaled_all = context.getTensor(router_input_scaled_idx);
  auto &router_logits_all = context.getTensor(router_logits_idx);

  auto expert_step_dim = expert_input_all.getDim();
  auto output_step_dim = output_all.getDim();
  auto scaled_step_dim = router_input_scaled_all.getDim();
  auto logits_step_dim = router_logits_all.getDim();
  expert_step_dim.batch(1);
  expert_step_dim.height(to - from);
  output_step_dim.batch(1);
  output_step_dim.height(to - from);
  scaled_step_dim.batch(to - from);
  logits_step_dim.batch(to - from);

  for (unsigned int batch = 0; batch < expert_input_all.batch(); ++batch) {
    auto expert_input = expert_input_all.getSharedDataTensor(
      expert_step_dim, batch * expert_step_dim.getFeatureLen(), true);
    auto router_input = router_input_all.getSharedDataTensor(
      expert_step_dim, batch * expert_step_dim.getFeatureLen(), true);
    auto output = output_all.getSharedDataTensor(
      output_step_dim, batch * output_step_dim.getFeatureLen(), true);
    auto router_input_scaled =
      router_input_scaled_all.getSharedDataTensor(scaled_step_dim, 0, true);
    auto router_logits =
      router_logits_all.getSharedDataTensor(logits_step_dim, 0, true);
    forwardTensors(context, expert_input, router_input, output,
                   router_input_scaled, router_logits);
    output.reshape(output_step_dim);
  }
}

void Gemma4MoELayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, moe_props);
  nntrainer::LayerImpl::setProperty(remain_props);
}

void Gemma4MoELayer::calcDerivative(nntrainer::RunLayerContext &context) {
  throw std::runtime_error(
    "Gemma4 MoE does not support derivative calculation");
}

void Gemma4MoELayer::calcGradient(nntrainer::RunLayerContext &context) {
  throw std::runtime_error("Gemma4 MoE does not support gradient calculation");
}

void Gemma4MoELayer::exportTo(nntrainer::Exporter &exporter,
                              const ml::train::ExportMethods &method) const {
  nntrainer::LayerImpl::exportTo(exporter, method);
  exporter.saveResult(moe_props, method, this);
}

void Gemma4MoELayer::save(std::ofstream &file,
                          nntrainer::RunLayerContext &run_context, bool opt_var,
                          ml::train::ExecutionMode mode, bool trainable,
                          nntrainer::TensorDim::DataType dtype,
                          ml::train::ISA target_isa) const {
  if (opt_var) {
    nntrainer::LayerImpl::save(file, run_context, opt_var, mode, trainable,
                               dtype, target_isa);
    return;
  }

  for (unsigned int i = 0; i < run_context.getNumWeights(); ++i) {
    if (!run_context.isGradientFirstAccess(i))
      continue;

    auto &weight = run_context.getWeight(i);
    const bool is_router_weight =
      i == router_idx || i == router_scale_idx || i == per_expert_scale_idx;
    const bool activate_for_save = weight.isVirtual() && !weight.isAllocated();
    if (activate_for_save)
      weight.activate();

    try {
      if (is_router_weight || dtype == nntrainer::TensorDim::DataType::NONE ||
          weight.getDataType() == dtype) {
        weight.save(file);
      } else {
        NNTR_THROW_IF(dtype != nntrainer::TensorDim::DataType::Q4_0,
                      std::runtime_error)
          << "Gemma4 MoE save supports only Q4_0 quantization";
        NNTR_THROW_IF(weight.getDataType() !=
                        nntrainer::TensorDim::DataType::FP32,
                      std::runtime_error)
          << "Gemma4 MoE quantization requires FP32 source weights";

        const nntrainer::TensorDim dim = weight.getDim();
        const unsigned int height = dim.height();
        const unsigned int width = dim.width();
        NNTR_THROW_IF(height % 32 != 0 || width % 32 != 0,
                      std::invalid_argument)
          << "Q4_0 requires height and width divisible by 32, got height="
          << height << ", width=" << width;

        nntrainer::Tensor transposed = weight.transpose("0:2:1");
        nntrainer::Tensor quantized(dim.batch(), dim.channel(), height, width,
                                    {nntrainer::Tformat::NCHW, dtype});
        std::vector<char> temporary(quantized.size());
        nntrainer::quantize_q4_0(transposed.getData<float>(), temporary.data(),
                                 width, height, nullptr);
        nntrainer::repack_q4_0(quantized.getData<uint8_t>(), temporary.data(),
                               quantized.size(), width, height, target_isa);
        quantized.save(file);
      }
    } catch (...) {
      if (activate_for_save)
        weight.deactivate();
      throw;
    }

    if (activate_for_save)
      weight.deactivate();
  }
}

} // namespace causallm
