// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   conv_block_layer.cpp
 * @date   22 September 2026
 * @brief  The LFM2 conv block (in_proj, gate, causal conv1d, gate, out_proj)
 *         as one layer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <conv_block_layer.h>

#include <compute_ops.h>
#include <cpu_backend.h>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <node_exporter.h>
#include <thread_manager.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;
enum ConvBlockParams { IN_PROJ, CONV_W, OUT_PROJ };
enum ConvBlockTensors { PROJ, GATED, CONV_OUT, STATE };

/** NNTR_CONV_BLOCK_DIFF: run both paths at prefill and print the SNR of
 *  the accelerator's output and conv state against the CPU's, per layer.
 *  NNTR_CONV_BLOCK_SHADOW: same, and hand the model the CPU's values. The
 *  same two discriminators the MoE path has (NNTR_L2_DIFF / NNTR_L2_SHADOW).
 *  Read the number as "two int4 grids disagree" (doc 51 section 2.14):
 *  the CPU path is itself Q4_0, so any other int4 recipe reads 10-20 dB
 *  against it without being worse. Quality is judged by NNTR_PPL. */
static bool convBlockDiffEnabled() {
  static const bool on = std::getenv("NNTR_CONV_BLOCK_DIFF") != nullptr;
  return on;
}
static bool convBlockShadowEnabled() {
  static const bool on = std::getenv("NNTR_CONV_BLOCK_SHADOW") != nullptr;
  return on;
}
static double snrDb(const float *ref, const float *got, size_t n) {
  double sig = 0.0, err = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double d = (double)ref[i] - (double)got[i];
    sig += (double)ref[i] * (double)ref[i];
    err += d * d;
  }
  return err == 0.0 ? 999.0 : 10.0 * std::log10(sig / err);
}

ConvBlockLayer::ConvBlockLayer() :
  LayerImpl(), conv_props(nntrainer::props::Unit()) {
  weight_idx.fill(std::numeric_limits<unsigned>::max());
  tensor_idx.fill(std::numeric_limits<unsigned>::max());
}

void ConvBlockLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "conv_block takes exactly one input";
  NNTR_THROW_IF(context.getFormat() != nntrainer::Tformat::NCHW,
                std::invalid_argument)
    << "conv_block supports NCHW only";

  auto &weight_regularizer =
    std::get<nntrainer::props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<nntrainer::props::WeightRegularizerConstant>(*layer_impl_props);
  auto weight_initializer = nntrainer::props::InitializerInfo::Enum::NONE;
  auto &weight_decay =
    std::get<nntrainer::props::WeightDecay>(*layer_impl_props);
  const unsigned int C = std::get<nntrainer::props::Unit>(conv_props).get();

  context.setEffDimFlagInputDimension(0, 0b1001);
  context.setDynDimFlagInputDimension(0, 0b1000);

  const auto &in_dim = context.getInputDimensions()[0];
  NNTR_THROW_IF(in_dim.channel() != 1, std::invalid_argument)
    << "conv_block input channel must be 1 (B x 1 x T x W)";
  const unsigned int K = in_dim.width();
  const nntrainer::TensorDim::TensorType act_type(
    context.getFormat(), context.getActivationDataType());
  const nntrainer::TensorDim::TensorType w_type(context.getFormat(),
                                                context.getWeightDataType());
  const nntrainer::TensorDim::TensorType f32_type(
    context.getFormat(), ml::train::TensorDim::DataType::FP32);

  // out_proj's width is the residual width: the block adds into it.
  nntrainer::TensorDim out_dim = in_dim;
  out_dim.setTensorType(act_type);
  context.setOutputDimensions({out_dim});

  // The file's order: in_proj, conv, out_proj. Shapes as the three layers
  // this replaces request them ([in, unit] for the FCs, [1, 1, 3, C] FP32
  // for the conv), so the same bytes load either way.
  nntrainer::TensorDim w_in(1, 1, K, 3 * C, w_type, 0b0011);
  weight_idx[IN_PROJ] = context.requestWeight(
    w_in, weight_initializer, weight_regularizer, weight_regularizer_constant,
    weight_decay, "in_proj", true);
  nntrainer::TensorDim w_conv({1, 1, KERNEL_SIZE, C}, f32_type);
  weight_idx[CONV_W] = context.requestWeight(
    w_conv, nntrainer::Initializer::NONE, nntrainer::WeightRegularizer::NONE,
    0.0f, 0.0f, "conv", false);
  nntrainer::TensorDim w_out(1, 1, C, K, w_type, 0b0011);
  weight_idx[OUT_PROJ] = context.requestWeight(
    w_out, weight_initializer, weight_regularizer, weight_regularizer_constant,
    weight_decay, "out_proj", true);

  // The CPU path's intermediates, sized for the graph's input length and
  // sliced per step below; the fused call never touches them.
  nntrainer::TensorDim proj(in_dim.batch(), 1, in_dim.height(), 3 * C,
                            act_type);
  tensor_idx[PROJ] =
    context.requestTensor(proj, "proj", nntrainer::Initializer::NONE, false,
                          nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
  nntrainer::TensorDim mid(in_dim.batch(), 1, in_dim.height(), C, act_type);
  tensor_idx[GATED] =
    context.requestTensor(mid, "gated", nntrainer::Initializer::NONE, false,
                          nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
  tensor_idx[CONV_OUT] =
    context.requestTensor(mid, "conv_out", nntrainer::Initializer::NONE, false,
                          nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
  // The conv state, as causal_conv1d_layer keeps it: [x_{t-2}, x_{t-1}].
  nntrainer::TensorDim state({in_dim.batch(), 1, KERNEL_SIZE - 1, C}, f32_type);
  tensor_idx[STATE] =
    context.requestTensor(state, "conv_state", nntrainer::Initializer::ZEROS,
                          false, nntrainer::TensorLifespan::MAX_LIFESPAN);
}

void ConvBlockLayer::exportTo(nntrainer::Exporter &exporter,
                              const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(conv_props, method, this);
}

void ConvBlockLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, conv_props);
  LayerImpl::setProperty(remain_props);
}

void ConvBlockLayer::forwarding(nntrainer::RunLayerContext &context,
                                bool training) {
  incremental_forwarding(context, 0,
                         context.getInput(SINGLE_INOUT_IDX).height(), training);
}

void ConvBlockLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                            unsigned int from, unsigned int to,
                                            bool training) {
  NNTR_THROW_IF(to <= from, std::invalid_argument)
    << "conv_block: invalid range from=" << from << " to=" << to;
  nntrainer::Tensor &in_w = context.getWeight(weight_idx[IN_PROJ]);
  nntrainer::Tensor &conv_w = context.getWeight(weight_idx[CONV_W]);
  nntrainer::Tensor &out_w = context.getWeight(weight_idx[OUT_PROJ]);
  nntrainer::Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output_ = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &state_t = context.getTensor(tensor_idx[STATE]);

  // Batch 1, rows at offset 0 of the step -- where the model places the
  // current token(s), as causal_conv1d_layer and dense_ffn_layer read them.
  const unsigned int rows = to - from;
  nntrainer::TensorDim in_step_dim = input_.getDim();
  in_step_dim.batch(1);
  in_step_dim.height(rows);
  nntrainer::Tensor in_step = input_.getSharedDataTensor(in_step_dim, 0, true);
  nntrainer::TensorDim out_step_dim = output_.getDim();
  out_step_dim.batch(1);
  out_step_dim.height(rows);
  nntrainer::Tensor out_step =
    output_.getSharedDataTensor(out_step_dim, 0, true);

  const unsigned int K = in_step_dim.width();
  const unsigned int C = conv_w.width();
  const unsigned int N = out_w.width();
  const float *w_ptr = conv_w.getData<float>();
  float *state = state_t.getData<float>();

  // The whole block as one accelerator call at prefill; it hands back the
  // conv state decode continues from. Decode's single row stays on the
  // CPU for the reason every FC gate has (accelerates_q4_0_at_m1): one
  // row cannot amortize the call.
  const auto q4 = ml::train::TensorDim::DataType::Q4_0;
  auto *ops = in_step.getOps();
  const bool use_htp = rows > 1 && ops != nullptr &&
                       ops->supports_gemm_q4_0_conv_block_fp32() &&
                       in_w.getDataType() == q4 && out_w.getDataType() == q4;
  const bool compare =
    use_htp && (convBlockDiffEnabled() || convBlockShadowEnabled());
  if (use_htp && !compare) {
    ops->gemm_q4_0_conv_block_fp32(
      in_w.getData<char>(), w_ptr, out_w.getData<char>(),
      in_step.getData<float>(), out_step.getData<float>(), state, rows, K, C,
      N);
    return;
  }

  // What createConvBlock's layers computed, with their kernels: in_proj
  // dot, a * c, the causal conv (decode's kernel keeps the state, the
  // prefill one is followed by the same state save), b * conv, out_proj
  // dot. Each multiply is one f32 operation, so the split and the two
  // custom_multiply layers this folds are byte for byte.
  nntrainer::TensorDim proj_dim = context.getTensor(tensor_idx[PROJ]).getDim();
  proj_dim.batch(1);
  proj_dim.height(rows);
  nntrainer::Tensor proj =
    context.getTensor(tensor_idx[PROJ]).getSharedDataTensor(proj_dim, 0, true);
  nntrainer::TensorDim mid_dim = context.getTensor(tensor_idx[GATED]).getDim();
  mid_dim.batch(1);
  mid_dim.height(rows);
  nntrainer::Tensor gated =
    context.getTensor(tensor_idx[GATED]).getSharedDataTensor(mid_dim, 0, true);
  nntrainer::Tensor conv_out = context.getTensor(tensor_idx[CONV_OUT])
                                 .getSharedDataTensor(mid_dim, 0, true);

  // Under compare the projections go through the CPU Q4_0 GEMM directly:
  // with engine=htp on this layer, dot() would route them to the HTP FC
  // path and the "reference" would carry the same quantization points as
  // the fused call (the first DIFF run measured that as 150 dB and proved
  // the kernel, not the numerics -- doc 51 section 2.9).
  if (compare) {
    nntrainer::gemm_q4_0<float>(rows, 3 * C, K, in_step.getData<float>(), K,
                                in_w.getData<char>(), 3 * C,
                                proj.getData<float>(), 3 * C);
  } else {
    in_step.dot(in_w, proj, false, false);
  }

  const float *p = proj.getData<float>();
  float *g = gated.getData<float>();
  float *y = conv_out.getData<float>();
  auto gate_pre = [&](size_t r) {
    const float *a = p + r * 3 * C;
    const float *c = a + 2 * C;
    float *gr = g + r * C;
    for (unsigned int j = 0; j < C; ++j)
      gr[j] = a[j] * c[j];
  };
  auto gate_post = [&](size_t r) {
    const float *b = p + r * 3 * C + C;
    float *yr = y + r * C;
    for (unsigned int j = 0; j < C; ++j)
      yr[j] = b[j] * yr[j];
  };
  if (rows == 1) {
    gate_pre(0);
    nntrainer::causal_depthwise_conv1d_k3_decode(g, w_ptr, state, y, C);
    gate_post(0);
  } else {
    nntrainer::ThreadManager::Global().parallel_for(
      0, static_cast<size_t>(rows), gate_pre);
    nntrainer::causal_depthwise_conv1d_k3(g, w_ptr, nullptr, y, 1, rows, C);
    std::memcpy(state, g + static_cast<size_t>(rows - 2) * C,
                C * sizeof(float));
    std::memcpy(state + C, g + static_cast<size_t>(rows - 1) * C,
                C * sizeof(float));
    nntrainer::ThreadManager::Global().parallel_for(
      0, static_cast<size_t>(rows), gate_post);
  }

  if (compare) {
    nntrainer::gemm_q4_0<float>(rows, N, C, y, C, out_w.getData<char>(), N,
                                out_step.getData<float>(), N);
  } else {
    conv_out.dot(out_w, out_step, false, false);
  }

  if (compare) {
    // The CPU path above is the reference; the accelerator recomputes
    // the same step from the same input into its own buffers.
    std::vector<float> h_out(static_cast<size_t>(rows) * N);
    std::vector<float> h_state(static_cast<size_t>(2) * C);
    ops->gemm_q4_0_conv_block_fp32(
      in_w.getData<char>(), w_ptr, out_w.getData<char>(),
      in_step.getData<float>(), h_out.data(), h_state.data(), rows, K, C, N);
    std::fprintf(stderr,
                 "[conv_block] %s rows=%u  SNR out %.1f dB  state %.1f dB%s\n",
                 context.getName().c_str(), rows,
                 snrDb(out_step.getData<float>(), h_out.data(), h_out.size()),
                 snrDb(state, h_state.data(), h_state.size()),
                 convBlockShadowEnabled() ? "  (shadow: CPU values used)" : "");
    if (!convBlockShadowEnabled()) {
      std::memcpy(out_step.getData<float>(), h_out.data(),
                  h_out.size() * sizeof(float));
      std::memcpy(state, h_state.data(), h_state.size() * sizeof(float));
    }
  }
}

void ConvBlockLayer::updateTensorsByInputDimensions(
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
