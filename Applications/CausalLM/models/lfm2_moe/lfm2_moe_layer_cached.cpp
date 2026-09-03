// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jungwon-Lee <jungone.lee@samsung.com>
 *
 * @file   lfm2_moe_layer_cached.cpp
 * @date   06 July 2026
 * @brief  Cached-Slim MoE layer for LFM2-MoE (FSU experts + LRU expert cache).
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <acti_func.h>
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cpu_backend.h>
#include <cstdlib>
#include <lfm2_moe_layer_cached.h>
#include <node_exporter.h>
#include <stdexcept>
#include <thread_manager.h>

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

/** LFM2-MoE router hyper-parameters (fixed for LFM2-8B-A1B). */
static constexpr bool NORM_TOPK_PROB = true;
static constexpr float ROUTED_SCALING_FACTOR = 1.0f;
/** Extra experts (beyond top-k) tracked per token for LRU prefetch ordering. */
static constexpr unsigned EXTRA_TOPK = 5;
/** Default number of experts kept mmap-resident per layer. */
static constexpr unsigned int DEFAULT_CACHE_CAPACITY = 32;

static unsigned int getExpertCacheCapacity(unsigned int num_experts) {
  const char *value = std::getenv("NNTR_MOE_CACHE_EXPERTS");
  if (value == nullptr || *value == '\0' || *value == '-')
    return std::min(num_experts, DEFAULT_CACHE_CAPACITY);

  errno = 0;
  char *end = nullptr;
  const unsigned long parsed = std::strtoul(value, &end, 10);
  if (errno == ERANGE || end == value || *end != '\0')
    return std::min(num_experts, DEFAULT_CACHE_CAPACITY);

  return static_cast<unsigned int>(
    std::min<unsigned long>(num_experts, parsed));
}

Lfm2CachedSlimMoELayer::Lfm2CachedSlimMoELayer() :
  LayerImpl(),
  num_experts(0),
  topk(0),
  moe_props(props::NumExperts(), props::NumExpertsPerToken(),
            nntrainer::props::Unit(), props::MoEActivation()),
  expert_gate_up_proj_indices({}),
  expert_down_proj_indices({}),
  gate_idx(std::numeric_limits<unsigned>::max()),
  expert_bias_idx(std::numeric_limits<unsigned>::max()),
  loaded_expert_deque({}),
  need_load({}),
  cache_capacity(0),
  router_logits_idx(std::numeric_limits<unsigned>::max()),
  decode_expert_output_idx(std::numeric_limits<unsigned>::max()),
  decode_gate_up_output_idx(std::numeric_limits<unsigned>::max()),
  decode_activation_output_idx(std::numeric_limits<unsigned>::max()) {}

void Lfm2CachedSlimMoELayer::finalize(nntrainer::InitLayerContext &context) {

  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "LFM2 Cached-Slim MoE layer only supports single input";

  auto &weight_regularizer =
    std::get<nntrainer::props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<nntrainer::props::WeightRegularizerConstant>(*layer_impl_props);
  auto &weight_initializer =
    std::get<nntrainer::props::WeightInitializer>(*layer_impl_props);
  auto &weight_decay =
    std::get<nntrainer::props::WeightDecay>(*layer_impl_props);

  const auto &in_dim = context.getInputDimensions()[SINGLE_INOUT_IDX];
  const bool is_nchw = context.getFormat() == nntrainer::Tformat::NCHW;
  std::vector<nntrainer::TensorDim> output_dims(1);
  output_dims[SINGLE_INOUT_IDX] = in_dim;
  context.setOutputDimensions(output_dims);

  num_experts = std::get<props::NumExperts>(moe_props).get();
  topk = std::get<props::NumExpertsPerToken>(moe_props).get();
  const unsigned int intermediate_size =
    std::get<nntrainer::props::Unit>(moe_props).get();
  const unsigned int hidden_size = in_dim.width();

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

  nntrainer::TensorDim gate_dim(
    1, is_nchw ? 1 : num_experts, is_nchw ? hidden_size : 1,
    is_nchw ? num_experts : hidden_size,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     nntrainer::TensorDim::DataType::FP32),
    is_nchw ? 0b0011 : 0b0101);

  gate_idx = context.requestWeight(
    gate_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "gate", true);

  nntrainer::TensorDim expert_bias_dim(
    1, 1, 1, num_experts,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     nntrainer::TensorDim::DataType::FP32),
    0b0001);

  expert_bias_idx = context.requestWeight(
    expert_bias_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "expert_bias", false);

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
    expert_gate_up_proj_indices.push_back(context.requestWeight(
      expert_gate_up_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay,
      "expert_gate_up_" + std::to_string(i), false, true));

    expert_down_proj_indices.push_back(context.requestWeight(
      expert_down_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay,
      "expert_down_" + std::to_string(i), false, true));
    need_load.push_back(true);
  }
  cache_capacity = getExpertCacheCapacity(num_experts);

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

void Lfm2CachedSlimMoELayer::buildExpertAssignments(
  const nntrainer::Tensor &router_logits, const nntrainer::Tensor &expert_bias,
  unsigned int total_tokens,
  std::vector<std::vector<std::pair<unsigned, float>>> &expert_assignments,
  std::vector<int> &extra_top_k) {

  const float *logits = router_logits.getData<float>();
  const float *bias = expert_bias.getData<float>();

  const unsigned extended = std::min<unsigned>(topk + EXTRA_TOPK, num_experts);

  std::vector<float> sig(num_experts);
  std::vector<std::pair<float, int>> scored(num_experts);

  for (unsigned int i = 0; i < total_tokens; ++i) {
    const float *lrow = logits + static_cast<size_t>(i) * num_experts;

    for (unsigned int e = 0; e < num_experts; ++e) {
      const float s = 1.0f / (1.0f + std::exp(-lrow[e]));
      sig[e] = s;
      scored[e] = {s + bias[e], static_cast<int>(e)};
    }

    std::partial_sort(
      scored.begin(), scored.begin() + extended, scored.end(),
      [](const std::pair<float, int> &a, const std::pair<float, int> &b) {
        return a.first > b.first;
      });

    // Real top-k drives the output; weights come from bias-free sigmoid scores.
    float wsum = 0.0f;
    for (unsigned int k = 0; k < topk; ++k)
      wsum += sig[scored[k].second];

    const float inv = NORM_TOPK_PROB ? (1.0f / (wsum + 1e-6f)) : 1.0f;
    for (unsigned int k = 0; k < topk; ++k) {
      const int expert_idx = scored[k].second;
      const float weight = sig[expert_idx] * inv * ROUTED_SCALING_FACTOR;
      expert_assignments[expert_idx].emplace_back(i, weight);
    }

    // Extended set (top-k + extra) recorded for LRU prefetch ordering only.
    for (unsigned int k = 0; k < extended; ++k)
      extra_top_k.push_back(scored[k].second);
  }
}

void Lfm2CachedSlimMoELayer::forwarding(nntrainer::RunLayerContext &context,
                                        bool training) {
  // Cached-Slim variant only implements incremental_forwarding (used for both
  // prefill and decode by CausalLM::incremental_inference).
}

inline void Lfm2CachedSlimMoELayer::compute_expert_forward(
  const nntrainer::Tensor &input, nntrainer::Tensor &output,
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
                                      output.getTensorType());

  nntrainer::Tensor gate_up_out =
    workspace.gate_up_output->getSharedDataTensor(gate_up_dim, 0, true);
  nntrainer::Tensor acti_out =
    workspace.activation_output->getSharedDataTensor(intermediate_dim, 0, true);
  nntrainer::Tensor token_input;

  if (num_tokens == 1) {
    const size_t token_offset = token_assignments[0].first * hidden_size;
    token_input =
      input.getSharedDataTensor(token_input_dim, token_offset, true);
  } else {
    token_input =
      workspace.token_input->getSharedDataTensor(token_input_dim, 0, true);
    auto &tm = nntrainer::ThreadManager::Global();
    tm.parallel_for(0, static_cast<size_t>(num_tokens), [&](size_t i) {
      const size_t token_offset = token_assignments[i].first * hidden_size;
      nntrainer::Tensor src =
        input.getSharedDataTensor({1, 1, 1, hidden_size}, token_offset, true);
      nntrainer::Tensor dst = token_input.getSharedDataTensor(
        {1, 1, 1, hidden_size}, i * hidden_size, true);
      dst.copyData(src);
    });
  }

  token_input.dot(gate_up_proj, gate_up_out);

  if (num_tokens == 1) {
    nntrainer::swiglu(acti_out.width(), acti_out.getData<float>(),
                      gate_up_out.getData<float>(),
                      gate_up_out.getData<float>() + intermediate_size);
  } else {
    auto &tm = nntrainer::ThreadManager::Global();
    tm.parallel_for(0, static_cast<size_t>(num_tokens), [&](size_t i) {
      const unsigned int offset = acti_out.getIndex(0, 0, i, 0);
      const unsigned int gate_up_offset = gate_up_out.getIndex(0, 0, i, 0);
      nntrainer::swiglu(acti_out.width(), acti_out.getData<float>() + offset,
                        gate_up_out.getData<float>() + gate_up_offset,
                        gate_up_out.getData<float>() + gate_up_offset +
                          intermediate_size);
    });
  }

  acti_out.dot(down_proj, output);

  for (size_t i = 0; i < num_tokens; ++i) {
    nntrainer::Tensor expert_token_output =
      output.getSharedDataTensor(token_step_dim, i * hidden_size, true);
    expert_token_output.multiply_i(token_assignments[i].second);
  }
}

void Lfm2CachedSlimMoELayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {

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

    const unsigned batch_size = input.batch();
    const unsigned seq_len = input.height();
    const unsigned hidden_size = input.width();
    const unsigned total_tokens = batch_size * seq_len;

    input.reshape({total_tokens, 1, 1, hidden_size});
    output.reshape({total_tokens, 1, 1, hidden_size});
    output.setZero();

    input.dot(gate_weights, router_logits);

    std::vector<std::vector<std::pair<unsigned, float>>> expert_assignments(
      num_experts);
    std::vector<int> extra_top_k;
    buildExpertAssignments(router_logits, expert_bias, total_tokens,
                           expert_assignments, extra_top_k);

    std::vector<int> target_idx_vector;
    target_idx_vector.reserve(num_experts);
    for (int expert_idx = 0; expert_idx < static_cast<int>(num_experts);
         ++expert_idx) {
      if (!expert_assignments[expert_idx].empty())
        target_idx_vector.push_back(expert_idx);
    }

    size_t max_assigned_tokens = 0;
    for (int expert_idx : target_idx_vector)
      max_assigned_tokens =
        std::max(max_assigned_tokens, expert_assignments[expert_idx].size());

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
      workspace = {&prefill_token_input, &prefill_expert_output,
                   &prefill_gate_up_output, &prefill_activation_output};
    }

    auto deactivate_expert = [&](int expert_idx) {
      context.getWeight(expert_gate_up_proj_indices[expert_idx]).deactivate();
      context.getWeight(expert_down_proj_indices[expert_idx]).deactivate();
    };

    auto evict_lru_expert = [&]() {
      const int expert_idx = loaded_expert_deque.front();
      loaded_expert_deque.pop_front();
      iteration_map.erase(expert_idx);
      need_load[expert_idx] = true;
      deactivate_expert(expert_idx);
    };

    // Serial outer loop: expert dot() operations parallelize internally.
    // Enforce the cache limit before mapping a miss so resident mappings never
    // exceed NNTR_MOE_CACHE_EXPERTS.
    nntrainer::TensorDim token_step_dim({1, 1, 1, hidden_size},
                                        output.getTensorType());
    for (int expert_idx : target_idx_vector) {
      const auto &assignments = expert_assignments[expert_idx];
      nntrainer::Tensor expert_output =
        workspace.expert_output->getSharedDataTensor(
          {static_cast<unsigned int>(assignments.size()), 1, 1, hidden_size}, 0,
          true);
      bool temporary_mapping = false;

      if (need_load[expert_idx]) {
        while (cache_capacity > 0 &&
               loaded_expert_deque.size() >= cache_capacity)
          evict_lru_expert();

        context.getWeight(expert_gate_up_proj_indices[expert_idx]).activate();
        context.getWeight(expert_down_proj_indices[expert_idx]).activate();

        if (cache_capacity == 0) {
          temporary_mapping = true;
        } else {
          loaded_expert_deque.push_back(expert_idx);
          iteration_map[expert_idx] = --loaded_expert_deque.end();
          need_load[expert_idx] = false;
        }
      }

      try {
        compute_expert_forward(
          input, expert_output, assignments,
          context.getWeight(expert_gate_up_proj_indices[expert_idx]),
          context.getWeight(expert_down_proj_indices[expert_idx]), hidden_size,
          workspace);

        // Stream this expert's compact output into the final output before
        // computing the next expert, preserving ascending expert order.
        for (size_t i = 0; i < assignments.size(); ++i) {
          nntrainer::Tensor token_output = output.getSharedDataTensor(
            token_step_dim, assignments[i].first * hidden_size, true);
          nntrainer::Tensor expert_token_output =
            expert_output.getSharedDataTensor(token_step_dim, i * hidden_size,
                                              true);
          token_output.add_i(expert_token_output);
        }
      } catch (...) {
        if (temporary_mapping)
          deactivate_expert(expert_idx);
        throw;
      }

      if (temporary_mapping)
        deactivate_expert(expert_idx);
    }

    // Refresh LRU recency using the extended top-k set (most recent last).
    for (int i = static_cast<int>(extra_top_k.size()) - 1; i >= 0; --i) {
      auto it = iteration_map.find(extra_top_k[i]);
      if (it != iteration_map.end()) {
        loaded_expert_deque.erase(it->second);
        loaded_expert_deque.push_back(extra_top_k[i]);
        iteration_map[extra_top_k[i]] = --loaded_expert_deque.end();
      }
    }

    output.reshape({batch_size, 1, seq_len, hidden_size});
  }
}

void Lfm2CachedSlimMoELayer::save(std::ofstream &file,
                                  nntrainer::RunLayerContext &run_context,
                                  bool opt_var, ml::train::ExecutionMode mode,
                                  bool trainable,
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
           "Lfm2CachedSlimMoELayer";
    }
  }
}

void Lfm2CachedSlimMoELayer::setProperty(
  const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, moe_props);
  nntrainer::LayerImpl::setProperty(remain_props);
}

void Lfm2CachedSlimMoELayer::calcDerivative(
  nntrainer::RunLayerContext &context) {
  throw std::runtime_error(
    "LFM2 Cached-Slim MoE layer does not support derivative calculation");
}

void Lfm2CachedSlimMoELayer::calcGradient(nntrainer::RunLayerContext &context) {
  throw std::runtime_error(
    "LFM2 Cached-Slim MoE layer does not support gradient calculation");
}

void Lfm2CachedSlimMoELayer::exportTo(
  nntrainer::Exporter &exporter, const ml::train::ExportMethods &method) const {
  nntrainer::LayerImpl::exportTo(exporter, method);
  exporter.saveResult(moe_props, method, this);
}

void Lfm2CachedSlimMoELayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  ml::train::TensorDim input_dim = context.getInput(SINGLE_INOUT_IDX).getDim();
  ml::train::TensorDim output_dim =
    context.getOutput(SINGLE_INOUT_IDX).getDim();

  input_dim.height(input_dimensions[0].height());
  output_dim.height(input_dimensions[0].height());

  context.updateInput(SINGLE_INOUT_IDX, input_dim);
  context.updateOutput(SINGLE_INOUT_IDX, output_dim);
}

} // namespace causallm
