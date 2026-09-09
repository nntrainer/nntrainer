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
#include <chrono>
#include <cmath>
#include <compute_ops.h>
#include <cpu_backend.h>
#include <cstdlib>
#include <iostream>
#include <lfm2_moe_layer.h>
#include <node_exporter.h>
#include <stdexcept>
#include <thread_manager.h>

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

/** LFM2-MoE router hyper-parameters (fixed for LFM2-8B-A1B). */
static constexpr bool NORM_TOPK_PROB = true;
static constexpr float ROUTED_SCALING_FACTOR = 1.0f;

/** M0 (doc 43 §5): ordinal of prefill forwarding() calls, so each MoE layer
 * prints under a stable index in layer order. Decode never touches this --
 * it runs incremental_forwarding. */
static std::atomic<unsigned> g_m0_call_index{0};

Lfm2MoELayer::Lfm2MoELayer() :
  LayerImpl(),
  num_experts(0),
  topk(0),
  moe_props(props::NumExperts(), props::NumExpertsPerToken(),
            nntrainer::props::Unit(), props::MoEActivation()),
  expert_gate_up_proj_indices({}),
  expert_down_proj_indices({}),
  gate_idx(std::numeric_limits<unsigned>::max()),
  expert_bias_idx(std::numeric_limits<unsigned>::max()),
  router_logits_idx(std::numeric_limits<unsigned>::max()),
  decode_expert_output_idx(std::numeric_limits<unsigned>::max()),
  decode_gate_up_output_idx(std::numeric_limits<unsigned>::max()),
  decode_activation_output_idx(std::numeric_limits<unsigned>::max()) {}

void Lfm2MoELayer::finalize(nntrainer::InitLayerContext &context) {

  // 1. Validate input/output dimensions
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "LFM2 MoE layer only supports single input";

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
      "expert_gate_up_" + std::to_string(i), false));

    // Down projection
    expert_down_proj_indices.push_back(context.requestWeight(
      expert_down_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay,
      "expert_down_" + std::to_string(i), false));
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

void Lfm2MoELayer::buildExpertAssignments(
  const nntrainer::Tensor &router_logits, const nntrainer::Tensor &expert_bias,
  unsigned int total_tokens,
  std::vector<std::vector<std::pair<unsigned, float>>> &expert_assignments) {

  const float *logits = router_logits.getData<float>();
  const float *bias = expert_bias.getData<float>();

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

    // top-k experts by (sigmoid + bias)
    std::partial_sort(
      scored.begin(), scored.begin() + topk, scored.end(),
      [](const std::pair<float, int> &a, const std::pair<float, int> &b) {
        return a.first > b.first;
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
  }
}

void Lfm2MoELayer::forwarding(nntrainer::RunLayerContext &context,
                              bool training) {
  /** M0 (doc 43 §5): ARM-side wall timer around the whole layer -- this
   * profile is invisible to NNTR_HTP_PROFILE's host column by design, and
   * in the htp run the one dispatched layer's figure includes its
   * registration (once, first forward) on top of dispatch. */
  const bool m0_profile = std::getenv("NNTR_M0_PROFILE") != nullptr;
  const auto m0_t0 = std::chrono::steady_clock::now();

  nntrainer::Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);

  nntrainer::Tensor &router_logits = context.getTensor(router_logits_idx);

  const unsigned batch_size = input.batch();
  const unsigned seq_len = input.height();
  const unsigned hidden_size = input.width();
  const unsigned total_tokens = batch_size * seq_len;

  // reshape input: [B,1,S,H] -> [B*S,1,1,H]
  input.reshape({total_tokens, 1, 1, hidden_size});

  // reshape output: [B,1,S,H] -> [B*S,1,1,H]
  output.reshape({total_tokens, 1, 1, hidden_size});
  output.setZero();

  // routing: raw logits -> sigmoid + expert-bias top-k selection
  nntrainer::Tensor &gate_weights = context.getWeight(gate_idx);
  nntrainer::Tensor &expert_bias = context.getWeight(expert_bias_idx);
  input.dot(gate_weights, router_logits);

  std::vector<std::vector<std::pair<unsigned, float>>> expert_assignments(
    num_experts);
  buildExpertAssignments(router_logits, expert_bias, total_tokens,
                         expert_assignments);

  size_t max_assigned_tokens = 0;
  for (const auto &assignments : expert_assignments)
    max_assigned_tokens = std::max(max_assigned_tokens, assignments.size());

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
    const auto m0_us = std::chrono::duration_cast<std::chrono::microseconds>(
                         std::chrono::steady_clock::now() - m0_t0)
                         .count();
    std::cout << "[M0-PROF] moe_layer[" << g_m0_call_index.fetch_add(1)
              << "] tokens=" << total_tokens << " us=" << m0_us << std::endl;
  }
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
      nntrainer::Tensor target =
        token_input.getSharedDataTensor(token_step_dim, i * hidden_size, true);
      target.copyData(source);
    });
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
  constexpr bool kFusedSwigluEnabled = false;
  bool expert_ffn_done = false;
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

    acti_out.dot(down_proj, expert_output);
  }

  for (size_t i = 0; i < num_tokens; ++i) {
    nntrainer::Tensor expert_token_output =
      expert_output.getSharedDataTensor(token_step_dim, i * hidden_size, true);
    expert_token_output.multiply_i(token_assignments[i].second);
  }
}

void Lfm2MoELayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                          unsigned int from, unsigned int to,
                                          bool training) {

  /** M0: same timer as forwarding() -- the runner drives prefill through
   * this path too (from=0,to=prompt_len), decode is the total_tokens==1
   * case and stays silent under the same guard. */
  const bool m0_profile = std::getenv("NNTR_M0_PROFILE") != nullptr;
  const auto m0_t0 = std::chrono::steady_clock::now();

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

    // reshape input: [B,1,S,H] -> [B*S,1,1,H]
    input.reshape({total_tokens, 1, 1, hidden_size});

    // reshape output: [B,1,S,H] -> [B*S,1,1,H]
    output.reshape({total_tokens, 1, 1, hidden_size});
    output.setZero();

    // routing
    input.dot(gate_weights, router_logits);

    std::vector<std::vector<std::pair<unsigned, float>>> expert_assignments(
      num_experts);
    buildExpertAssignments(router_logits, expert_bias, total_tokens,
                           expert_assignments);

    size_t max_assigned_tokens = 0;
    for (const auto &assignments : expert_assignments)
      max_assigned_tokens = std::max(max_assigned_tokens, assignments.size());

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

    if (single_token_experts.size() > 1) {
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
      const auto m0_us = std::chrono::duration_cast<std::chrono::microseconds>(
                           std::chrono::steady_clock::now() - m0_t0)
                           .count();
      std::cout << "[M0-PROF] moe_layer[" << g_m0_call_index.fetch_add(1)
                << "] tokens=" << total_tokens << " us=" << m0_us << std::endl;
    }
  }
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
    nntrainer::swiglu(acti_out.width(), acti_out.getData<float>(),
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
