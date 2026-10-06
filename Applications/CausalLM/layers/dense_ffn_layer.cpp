// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   dense_ffn_layer.cpp
 * @date   21 September 2026
 * @brief  The dense SwiGLU FFN (up, gate, SwiGLU, down) as one layer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <dense_ffn_layer.h>

#include <compute_ops.h>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <node_exporter.h>
#include <swiglu_det.h>
#include <thread_manager.h>

#include <limits>

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;
enum DenseFfnParams { UP, GATE, DOWN };
enum DenseFfnTensors { UP_OUT, GATE_OUT, ACT };

DenseFfnLayer::DenseFfnLayer() :
  LayerImpl(), dense_props(nntrainer::props::Unit()) {
  weight_idx.fill(std::numeric_limits<unsigned>::max());
  tensor_idx.fill(std::numeric_limits<unsigned>::max());
}

void DenseFfnLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "dense_ffn takes exactly one input";
  NNTR_THROW_IF(context.getFormat() != nntrainer::Tformat::NCHW,
                std::invalid_argument)
    << "dense_ffn supports NCHW only";

  auto &weight_regularizer =
    std::get<nntrainer::props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<nntrainer::props::WeightRegularizerConstant>(*layer_impl_props);
  auto weight_initializer = nntrainer::props::InitializerInfo::Enum::NONE;
  auto &weight_decay =
    std::get<nntrainer::props::WeightDecay>(*layer_impl_props);
  const unsigned int inter =
    std::get<nntrainer::props::Unit>(dense_props).get();

  context.setEffDimFlagInputDimension(0, 0b1001);
  context.setDynDimFlagInputDimension(0, 0b1000);

  const auto &in_dim = context.getInputDimensions()[0];
  const unsigned int hidden = in_dim.width();
  const nntrainer::TensorDim::TensorType act_type(
    context.getFormat(), context.getActivationDataType());

  nntrainer::TensorDim out_dim = in_dim;
  out_dim.setTensorType(act_type);
  context.setOutputDimensions({out_dim});

  // The file's order: up, gate, down. Shapes as the three fully_connected
  // layers request them ([in, unit]), so the same bytes load either way.
  nntrainer::TensorDim w_up(1, 1, hidden, inter,
                            nntrainer::TensorDim::TensorType(
                              context.getFormat(), context.getWeightDataType()),
                            0b0011);
  weight_idx[UP] = context.requestWeight(
    w_up, weight_initializer, weight_regularizer, weight_regularizer_constant,
    weight_decay, "up", true);
  weight_idx[GATE] = context.requestWeight(
    w_up, weight_initializer, weight_regularizer, weight_regularizer_constant,
    weight_decay, "gate", true);
  nntrainer::TensorDim w_down(
    1, 1, inter, hidden,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()),
    0b0011);
  weight_idx[DOWN] = context.requestWeight(
    w_down, weight_initializer, weight_regularizer, weight_regularizer_constant,
    weight_decay, "down", true);

  // The CPU path's intermediates, sized for the graph's input length and
  // sliced per step below; the fused call never touches them.
  nntrainer::TensorDim mid(in_dim.batch(), 1, in_dim.height(), inter, act_type);
  tensor_idx[UP_OUT] =
    context.requestTensor(mid, "up_out", nntrainer::Initializer::NONE, false,
                          nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
  tensor_idx[GATE_OUT] =
    context.requestTensor(mid, "gate_out", nntrainer::Initializer::NONE, false,
                          nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
  tensor_idx[ACT] =
    context.requestTensor(mid, "act", nntrainer::Initializer::NONE, false,
                          nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
}

void DenseFfnLayer::exportTo(nntrainer::Exporter &exporter,
                             const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(dense_props, method, this);
}

void DenseFfnLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, dense_props);
  LayerImpl::setProperty(remain_props);
}

void DenseFfnLayer::forwarding(nntrainer::RunLayerContext &context,
                               bool training) {
  incremental_forwarding(context, 0,
                         context.getInput(SINGLE_INOUT_IDX).height(), training);
}

void DenseFfnLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                           unsigned int from, unsigned int to,
                                           bool training) {
  nntrainer::Tensor &up_w = context.getWeight(weight_idx[UP]);
  nntrainer::Tensor &gate_w = context.getWeight(weight_idx[GATE]);
  nntrainer::Tensor &down_w = context.getWeight(weight_idx[DOWN]);
  nntrainer::Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output_ = context.getOutput(SINGLE_INOUT_IDX);

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
  const unsigned int inter = up_w.width();
  const unsigned int N = down_w.width();

  // The whole block as one accelerator call at prefill. Decode's single
  // row stays on the CPU below for the reason every FC gate has
  // (accelerates_q4_0_at_m1): one row cannot amortize the call.
  const auto q4 = ml::train::TensorDim::DataType::Q4_0;
  auto *ops = in_step.getOps();
  if (rows > 1 && ops != nullptr && ops->supports_gemm_q4_0_dense_ffn_fp32() &&
      up_w.getDataType() == q4 && gate_w.getDataType() == q4 &&
      down_w.getDataType() == q4) {
    ops->gemm_q4_0_dense_ffn_fp32(
      up_w.getData<char>(), gate_w.getData<char>(), down_w.getData<char>(),
      in_step.getData<float>(), out_step.getData<float>(), rows, K, inter, N);
    return;
  }

  // What Transformer::createMlp's three layers computed: up and gate dots,
  // silu(gate) * up (swiglu_det: the bit-identical NEON/scalar form the
  // MoE layer's CPU path uses), then down.
  nntrainer::TensorDim mid_step_dim =
    context.getTensor(tensor_idx[UP_OUT]).getDim();
  mid_step_dim.batch(1);
  mid_step_dim.height(rows);
  nntrainer::Tensor up_out = context.getTensor(tensor_idx[UP_OUT])
                               .getSharedDataTensor(mid_step_dim, 0, true);
  nntrainer::Tensor gate_out = context.getTensor(tensor_idx[GATE_OUT])
                                 .getSharedDataTensor(mid_step_dim, 0, true);
  nntrainer::Tensor act = context.getTensor(tensor_idx[ACT])
                            .getSharedDataTensor(mid_step_dim, 0, true);

  in_step.dot(up_w, up_out, false, false);
  in_step.dot(gate_w, gate_out, false, false);

  float *act_p = act.getData<float>();
  const float *gate_p = gate_out.getData<float>();
  const float *up_p = up_out.getData<float>();
  auto one_row = [&](size_t r) {
    const size_t off = r * inter;
    swiglu_det(inter, act_p + off, gate_p + off, up_p + off);
  };
  if (rows == 1) {
    one_row(0);
  } else {
    nntrainer::ThreadManager::Global().parallel_for(
      0, static_cast<size_t>(rows), one_row);
  }

  act.dot(down_w, out_step, false, false);
}

void DenseFfnLayer::updateTensorsByInputDimensions(
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
