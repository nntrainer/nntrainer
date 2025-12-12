// SPDX-License-Identifier: Apache-2.0
/**
 *
 * @file   ernie_moe_layer.cpp
 * @brief  ernie 4.5 causallm header
 * @date   02 December 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <acti_func.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <ernie_moe_layer.h>
#include <node_exporter.h>
#include <omp.h>
#include <stdexcept>

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

ErnieMoELayer::ErnieMoELayer() :
  LayerImpl(),
  num_experts(0),
  num_shared_experts(0),
  topk(0),
  moe_props(props::NumExperts(), props::NumExpertsPerToken(),
            nntrainer::props::Unit(), props::MoEActivation(),
            props::NumSharedExperts(), props::MoENormMin()),
  expert_gate_proj_indices({}),
  expert_up_proj_indices({}),
  expert_down_proj_indices({}),
  loaded_expert_deque({}),
  need_load({}),
  gate_idx(std::numeric_limits<unsigned>::max()),
  e_score_correction_bias_idx(std::numeric_limits<unsigned>::max()),
  router_logits_idx(std::numeric_limits<unsigned>::max()),
  expert_mask_idx(std::numeric_limits<unsigned>::max()) {}

void ErnieMoELayer::finalize(nntrainer::InitLayerContext &context) {
  // 1. Validate input/output dimensions
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "MoE layer only supports single input";

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
  num_shared_experts = std::get<props::NumSharedExperts>(moe_props).get();
  topk = std::get<props::NumExpertsPerToken>(moe_props).get();
  float moe_norm_min = std::get<props::MoENormMin>(moe_props).get();
  if (moe_norm_min == 0.0f) {
    moe_norm_min = 1e-12f; // Default value if not set
    std::get<props::MoENormMin>(moe_props).set(moe_norm_min);
  }
  const unsigned int intermediate_size =
    std::get<nntrainer::props::Unit>(moe_props).get();
  const unsigned int hidden_size = in_dim.width(); // Feature dimension

  // activation function
  if (std::get<props::MoEActivation>(moe_props).empty()) {
    throw std::runtime_error("Activation type is not set for MoE layer");
  }
  switch (context.getActivationDataType()) {
  case ml::train::TensorDim::DataType::FP32:
    acti_func.setActiFunc<float>(
      std::get<props::MoEActivation>(moe_props).get());
    break;
  default:
    throw std::runtime_error("Unsupported activation data type for MoE layer");
  }

  // 4. Initialie gate layer (router)
  nntrainer::TensorDim gate_dim(
    1, is_nchw ? 1 : num_experts, is_nchw ? hidden_size : 1,
    is_nchw ? num_experts : hidden_size,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     nntrainer::TensorDim::DataType::FP32),
    is_nchw ? 0b0011 : 0b0101);

  gate_idx = context.requestWeight(
    gate_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "gate", true);

  // 4-1. Initialize e_score_correction_bias
  nntrainer::TensorDim e_score_correction_bias_dim(
    1, 1, 1, num_experts,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     nntrainer::TensorDim::DataType::FP32),
    is_nchw ? 0b0011 : 0b0101);

  e_score_correction_bias_idx = context.requestWeight(
    e_score_correction_bias_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "moe_statics", false);

  // 5. Initializer expert weights
  expert_gate_proj_indices.reserve(num_experts);
  expert_up_proj_indices.reserve(num_experts);
  expert_down_proj_indices.reserve(num_experts);

  if (num_shared_experts > 0) {
    nntrainer::TensorDim shared_expert_gate_dim(
      1, is_nchw ? 1 : num_shared_experts * intermediate_size,
      is_nchw ? hidden_size : 1,
      is_nchw ? num_shared_experts * intermediate_size : hidden_size,
      nntrainer::TensorDim::TensorType(context.getFormat(),
                                       context.getWeightDataType()),
      is_nchw ? 0b0011 : 0b0101);

    nntrainer::TensorDim shared_expert_down_dim(
      1, is_nchw ? 1 : hidden_size,
      is_nchw ? num_shared_experts * intermediate_size : 1,
      is_nchw ? hidden_size : num_shared_experts * intermediate_size,
      nntrainer::TensorDim::TensorType(context.getFormat(),
                                       context.getWeightDataType()),
      is_nchw ? 0b0011 : 0b0101);

    shared_up_proj_idx = context.requestWeight(
      shared_expert_gate_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay, "shared_experts_up", false);

    shared_gate_proj_idx = context.requestWeight(
      shared_expert_gate_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay, "shared_experts_gate", false);

    shared_down_proj_idx = context.requestWeight(
      shared_expert_down_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay, "shared_experts_down", false);
  }

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
    expert_up_proj_indices.push_back(context.requestWeight(
      expert_gate_dim, // Same dimensions as gate projection
      weight_initializer, weight_regularizer, weight_regularizer_constant,
      weight_decay, "expert_up_" + std::to_string(i), false, true));

    expert_gate_proj_indices.push_back(context.requestWeight(
      expert_gate_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay,
      "expert_gate_" + std::to_string(i), false, true));

    expert_down_proj_indices.push_back(context.requestWeight(
      expert_down_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay,
      "expert_down_" + std::to_string(i), false, true));

    need_load.push_back(true);
  }

  // 5-1. Initialize shared expert weights

  // 6. Request intermediate tensors
  const unsigned batch_size = in_dim.batch();
  const unsigned seq_len = in_dim.height();
  const unsigned total_tokens = batch_size * seq_len;

  // Router logits :  [batch * seq, num_experts]
  router_logits_idx =
    context.requestTensor({total_tokens, 1, 1, num_experts}, "router_logits",
                          nntrainer::Initializer::NONE, false,
                          nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);

  // Expert mask: [num_experts, batch*seq]
  expert_mask_idx =
    context.requestTensor({num_experts, 1, topk, total_tokens}, "expert_mask",
                          nntrainer::Initializer::ZEROS, false,
                          nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
}

void ErnieMoELayer::forwarding(nntrainer::RunLayerContext &context,
                               bool training) {}

inline void ErnieMoELayer::compute_expert_forward(
  const nntrainer::Tensor &input, nntrainer::Tensor &output,
  const std::vector<std::pair<unsigned, float>> &token_assignments,
  const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
  const nntrainer::Tensor &down_proj, unsigned int hidden_size) {

  const unsigned intermediate_size = gate_proj.width();
  const unsigned num_tokens = token_assignments.size();

  if (num_tokens == 0)
    return;

  // Create tensor dimensions for single token processing
  nntrainer::TensorDim token_input_dim({1, 1, num_tokens, hidden_size},
                                       input.getTensorType());
  nntrainer::TensorDim intermediate_dim({1, 1, num_tokens, intermediate_size},
                                        input.getTensorType());
  nntrainer::TensorDim token_output_dim({1, 1, num_tokens, hidden_size},
                                        input.getTensorType());
  nntrainer::TensorDim out_step_dim({1, 1, 1, hidden_size},
                                    input.getTensorType());
  nntrainer::TensorDim step_dim({1, 1, 1, intermediate_size},
                                input.getTensorType());
  // Create intermediate tensors for this token
  nntrainer::Tensor gate_out(intermediate_dim);
  nntrainer::Tensor acti_out(intermediate_dim);
  nntrainer::Tensor up_out(intermediate_dim);
  nntrainer::Tensor token_input(token_input_dim);
  // Down projection using optimized dot operation
  nntrainer::Tensor token_expert_output(token_output_dim);

  unsigned token_idx = token_assignments[0].first;
  float weight = token_assignments[0].second;

  if (num_tokens > 1) {
    /** if prefill, copy data to make a batch */
#pragma omp parallel for schedule(static) if (num_tokens > 4)
    for (size_t i = 0; i < num_tokens; ++i) {
      const unsigned token_idx = token_assignments[i].first;
      // Use tensor's optimized copy operation
      nntrainer::Tensor src_view = input.getSharedDataTensor(
        {1, 1, 1, hidden_size}, token_idx * hidden_size, true);
      nntrainer::Tensor dst_view = token_input.getSharedDataTensor(
        {1, 1, 1, hidden_size}, i * hidden_size, true);
      dst_view.copyData(src_view);
    }
  } else {
    /** if token generation, do not copy but get the shared tensor */
    // Create shared tensor for input token (no memory copy)
    size_t token_offset = token_idx * hidden_size;
    token_input =
      input.getSharedDataTensor(token_input_dim, token_offset, true);
  }

  // Gate projection using optimized dot operation
  token_input.dot(gate_proj, gate_out);

  // Up projection using optimized dot operation
  token_input.dot(up_proj, up_out);

  if (num_tokens == 1) {
    // Apply activation (silu)
    acti_func.run_fn(gate_out, acti_out);
    // Element-wise multiply: silu(gate_out) * up_out
    acti_out.multiply_i(up_out);
  } else {
#pragma omp parallel for schedule(static) if (num_tokens > 4)
    for (size_t i = 0; i < num_tokens; ++i) {
      const unsigned offset = acti_out.getIndex(0, 0, i, 0);
      nntrainer::swiglu(acti_out.width(), acti_out.getData<float>() + offset,
                        gate_out.getData<float>() + offset,
                        up_out.getData<float>() + offset);
    }
  }

  acti_out.dot(down_proj, token_expert_output);

  // accumulate to output
  for (size_t i = 0; i < num_tokens; ++i) {
    token_idx = token_assignments[i].first;
    weight = token_assignments[i].second;
    size_t output_offset = token_idx * hidden_size;
    nntrainer::Tensor token_output =
      output.getSharedDataTensor(out_step_dim, output_offset, true);
    nntrainer::Tensor target = token_expert_output.getSharedDataTensor(
      out_step_dim, i * hidden_size, true);
    target.multiply_i(weight);
    token_output.add(target, token_output);
  }
}

void ErnieMoELayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                           unsigned int from, unsigned int to,
                                           bool training) {

  nntrainer::Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output_ = context.getOutput(SINGLE_INOUT_IDX);

  nntrainer::Tensor &router_logits_ = context.getTensor(router_logits_idx);

  nntrainer::TensorDim input_step_dim = input_.getDim();
  nntrainer::TensorDim output_step_dim = output_.getDim();
  nntrainer::TensorDim router_logits_step_dim = router_logits_.getDim();
  nntrainer::Tensor &shared_gate_proj = context.getWeight(shared_gate_proj_idx);
  nntrainer::Tensor &shared_up_proj = context.getWeight(shared_up_proj_idx);
  nntrainer::Tensor &shared_down_proj = context.getWeight(shared_down_proj_idx);

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

    // reshape input: [B,1,S,H] -> [B*S,1,1,H]
    input.reshape({total_tokens, 1, 1, hidden_size});

    // reshape output: [B,1,S,H] -> [B*S,1,1,H]
    output.reshape({total_tokens, 1, 1, hidden_size});
    output.setZero();

    // Compute shared experts
    if (num_shared_experts > 0) {

      nntrainer::Tensor shared_output(total_tokens, 1, 1, hidden_size,
                                      output.getTensorType());
      const unsigned int intermediate_size =
        std::get<nntrainer::props::Unit>(moe_props).get();

      nntrainer::TensorDim intermediate_dim(
        {total_tokens, 1, 1, num_shared_experts * intermediate_size},
        input.getTensorType());

      nntrainer::Tensor gate_out(intermediate_dim);
      nntrainer::Tensor acti_out(intermediate_dim);
      nntrainer::Tensor up_out(intermediate_dim);

      input.dot(shared_gate_proj, gate_out);
      input.dot(shared_up_proj, up_out);

      acti_func.run_fn(gate_out, acti_out);
      acti_out.multiply_i(up_out);

      acti_out.dot(shared_down_proj, shared_output);

      // Add shared expert output to final output
      output.add_i(shared_output);
    }

    // routing
    nntrainer::Tensor &gate_weights = context.getWeight(gate_idx);
    input.dot(gate_weights, router_logits);

    router_logits.apply(nntrainer::ActiFunc::softmax<float>, router_logits);

    // Add e_score_correction_bias
    nntrainer::Tensor &e_score_correction_bias =
      context.getWeight(e_score_correction_bias_idx);

    nntrainer::Tensor biased_router_logits(router_logits.getDim());
    biased_router_logits.copyData(router_logits);
    biased_router_logits.add_i(e_score_correction_bias);

    auto topk_result = biased_router_logits.topK(topk);
    auto topk_values = std::get<0>(topk_result);
    auto topk_indices = std::get<1>(topk_result);

    const uint32_t *indices_data = topk_indices.getData<uint32_t>();
    std::vector<std::vector<std::pair<unsigned, float>>> expert_assignments(
      num_experts);
    // Set expert mask
    for (int i = 0; i < static_cast<int>(total_tokens); ++i) {
      float sum_prob = 0.0f;
      for (int k = 0; k < static_cast<int>(topk); ++k) {
        unsigned expert_idx = indices_data[i * topk + k];
        float weight = router_logits.getValue<float>(i, 0, 0, expert_idx);
        sum_prob += weight;
      }

      for (int k = 0; k < static_cast<int>(topk); ++k) {
        unsigned expert_idx = indices_data[i * topk + k];
        float weight = router_logits.getValue<float>(i, 0, 0, expert_idx);
        weight /=
          std::max(sum_prob, std::get<props::MoENormMin>(moe_props).get());
        expert_assignments[expert_idx].emplace_back(i, weight);
      }
    }

    // Parallel processing for multiple tokens with many active experts
    std::vector<nntrainer::Tensor> expert_outputs(num_experts);
#pragma omp parallel for schedule(static)
    for (int expert_idx = 0; expert_idx < static_cast<int>(num_experts);
         ++expert_idx) {
      if (!expert_assignments[expert_idx].empty()) {
        expert_outputs[expert_idx] = nntrainer::Tensor(
          total_tokens, 1, 1, hidden_size, output.getTensorType());
        expert_outputs[expert_idx].setZero();
      }
    }
    std::vector<int> target_idx_vector;

    for (int expert_idx = 0; expert_idx < static_cast<int>(num_experts);
         ++expert_idx) {
      const auto &assignments = expert_assignments[expert_idx];
      if (assignments.empty())
        continue;

      target_idx_vector.push_back(expert_idx);
    }

#pragma omp parallel for schedule(dynamic)
    for (int expert_idx : target_idx_vector) {
      const auto &assignments = expert_assignments[expert_idx];
      if (need_load[expert_idx]) {

        context.getWeight(expert_gate_proj_indices[expert_idx]).activate();
        context.getWeight(expert_up_proj_indices[expert_idx]).activate();
        context.getWeight(expert_down_proj_indices[expert_idx]).activate();

        {
          std::lock_guard<std::mutex> lock(cache_mutex);
          loaded_expert_deque.push_back(expert_idx);
          iteration_map[expert_idx] = --loaded_expert_deque.end();
          need_load[expert_idx] = false;
        }

        compute_expert_forward(
          input, expert_outputs[expert_idx], assignments,
          context.getWeight(expert_gate_proj_indices[expert_idx]),
          context.getWeight(expert_up_proj_indices[expert_idx]),
          context.getWeight(expert_down_proj_indices[expert_idx]), hidden_size);

      } else {

        compute_expert_forward(
          input, expert_outputs[expert_idx], assignments,
          context.getWeight(expert_gate_proj_indices[expert_idx]),
          context.getWeight(expert_up_proj_indices[expert_idx]),
          context.getWeight(expert_down_proj_indices[expert_idx]), hidden_size);
      }
    }

// Evict experts
#pragma omp parallel
    while (true) {
      int target_idx;
      {
        std::lock_guard<std::mutex> lock(cache_mutex);

        if (loaded_expert_deque.size() > 16)
          break;

        target_idx = loaded_expert_deque.front();
        loaded_expert_deque.pop_front();
        iteration_map.erase(target_idx);
        need_load[target_idx] = true;
      }
      context.getWeight(expert_gate_proj_indices[target_idx]).deactivate();
      context.getWeight(expert_up_proj_indices[target_idx]).deactivate();
      context.getWeight(expert_down_proj_indices[target_idx]).deactivate();
    }

    // Combine expert outputs
    for (int expert_idx : target_idx_vector) {
      output.add_i(expert_outputs[expert_idx]);
    }

    // reshape output: [B*S,1,1,H] -> [B,1,S,H]
    output.reshape({batch_size, 1, seq_len, hidden_size});
  }
}

void ErnieMoELayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, moe_props);
  nntrainer::LayerImpl::setProperty(remain_props);
}

void ErnieMoELayer::calcDerivative(nntrainer::RunLayerContext &context) {
  // MoE layer does not support derivative calculation
  throw std::runtime_error("MoE layer does not support derivative calculation");
}

void ErnieMoELayer::calcGradient(nntrainer::RunLayerContext &context) {
  // MoE layer does not support gradient calculation
  throw std::runtime_error("MoE layer does not support gradient calculation");
}

void ErnieMoELayer::exportTo(nntrainer::Exporter &exporter,
                             const ml::train::ExportMethods &method) const {
  nntrainer::LayerImpl::exportTo(exporter, method);
  exporter.saveResult(moe_props, method, this); // Save MoE specific properties
}

void ErnieMoELayer::updateTensorsByInputDimensions(
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
