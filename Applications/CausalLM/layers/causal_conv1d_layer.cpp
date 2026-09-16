// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Hyeong-Gwon Hong
 *
 * @file   causal_conv1d_layer.cpp
 * @date   01 April 2026
 * @brief  Causal depthwise Conv1D layer with conv-state cache for CausalLM
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeong-Gwon Hong
 * @bug    No known bugs except for NYI items
 */

#include "causal_conv1d_layer.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <cpu_backend.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>

namespace causallm {

// ===========================================================================

CausalConv1DLayer::CausalConv1DLayer() : LayerImpl() {
  weight_idx.fill(std::numeric_limits<unsigned int>::max());
  tensor_idx.fill(std::numeric_limits<unsigned int>::max());
}

void CausalConv1DLayer::validateInputShape(
  const nntrainer::TensorDim &input_dim) const {
  NNTR_THROW_IF(input_dim.channel() != 1, std::invalid_argument)
    << "[CausalConv1DLayer] input channel must be 1 (B×1×T×W layout), got "
    << input_dim.channel();
  NNTR_THROW_IF(input_dim.height() < 1 || input_dim.width() < 1,
                std::invalid_argument)
    << "[CausalConv1DLayer] invalid input shape: H and W must be positive.";
}

void CausalConv1DLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "[CausalConv1DLayer] requires exactly 1 input, got "
    << context.getNumInputs();

  [[maybe_unused]] auto [output_dims, weight_dims, tensor_dims] =
    getLayerDimensions(context);

  weight_idx[weight] =
    context.requestWeight(weight_dims[weight], nntrainer::Initializer::NONE,
                          nntrainer::WeightRegularizer::NONE, 0.0f, 0.0f,
                          "causal_conv1d_weight", false);

  tensor_idx[conv_state] = context.requestTensor(
    tensor_dims[0], "conv_state", nntrainer::Initializer::ZEROS, false,
    nntrainer::TensorLifespan::MAX_LIFESPAN);

  context.setOutputDimensions(output_dims);
}

void CausalConv1DLayer::forwarding(nntrainer::RunLayerContext &context,
                                   bool training) {
  throw std::runtime_error(
    "[CausalConv1DLayer] forwarding() is not used – call "
    "incremental_forwarding() instead.");
}

void CausalConv1DLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {

  NNTR_THROW_IF(training, std::invalid_argument)
    << "[CausalConv1DLayer] training/backward is not supported.";
  NNTR_THROW_IF(to == 0 || to <= from, std::invalid_argument)
    << "[CausalConv1DLayer] invalid range: from=" << from << ", to=" << to;

  nntrainer::Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &w_tensor = context.getWeight(weight_idx[weight]);
  nntrainer::Tensor &state = context.getTensor(tensor_idx[conv_state]);

  const unsigned int B = input.batch();
  const unsigned int H = input.height(); // full sequence length (INIT_SEQ_LEN)
  const unsigned int W = input.width();  // feature dimension

  const float *w_ptr = w_tensor.getData<float>();
  float *state_data = state.getData<float>(); // [B, 1, KERNEL_SIZE-1, W]

  if (to - from == 1) {
    // ----------------------------------------------------------------
    // Decode path (hot): single-token inference.
    // NNTrainer places the current token at offset 0 within each batch
    // slice (not at offset `from`).
    // ----------------------------------------------------------------
    for (unsigned int b = 0; b < B; ++b) {
      const float *x_cur = input.getData<float>() + b * H * W;
      float *y_cur = output.getData<float>() + b * H * W;
      float *s = state_data + b * (KERNEL_SIZE - 1) * W;

      // y = w0*x + w1*s1 + w2*s0; state updated in-place by kernel
      nntrainer::causal_depthwise_conv1d_k3_decode(x_cur, w_ptr, s, y_cur, W);
    }

  } else {
    // ----------------------------------------------------------------
    // Prefill path: process all positions [0, to).
    // ----------------------------------------------------------------
    for (unsigned int b = 0; b < B; ++b) {
      const float *x = input.getData<float>() + b * H * W;
      float *y = output.getData<float>() + b * H * W;

      nntrainer::causal_depthwise_conv1d_k3(x, w_ptr, nullptr, y, 1, to, W);

      // Save last KERNEL_SIZE-1 tokens to state for decode steps
      float *s = state_data + b * (KERNEL_SIZE - 1) * W;

      if (to >= 2)
        std::memcpy(s, x + (to - 2) * W, W * sizeof(float)); // x_{to-2}
      else
        std::memset(s, 0, W * sizeof(float));

      std::memcpy(s + W, x + (to - 1) * W, W * sizeof(float)); // x_{to-1}
    }
  }
}

void CausalConv1DLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  throw std::runtime_error(
    "[CausalConv1DLayer] calcDerivative() not implemented (inference only).");
}

void CausalConv1DLayer::calcGradient(nntrainer::RunLayerContext &context) {
  throw std::runtime_error(
    "[CausalConv1DLayer] calcGradient() not implemented (inference only).");
}

std::array<std::vector<nntrainer::TensorDim>, 3>
CausalConv1DLayer::getLayerDimensions(nntrainer::InitLayerContext &context) {
  validateInputShape(context.getInputDimensions()[0]);

  const nntrainer::TensorDim &in_dim = context.getInputDimensions()[0];
  const unsigned int B = in_dim.batch();
  const unsigned int W = in_dim.width();

  // Weight: [1, 1, KERNEL_SIZE, W] FP32
  //   Row k  (offset k*W) = kernel weights for position k:
  //     k=0 → w0: applied to current token x_t
  //     k=1 → w1: applied to x_{t-1}
  //     k=2 → w2: applied to x_{t-2}
  nntrainer::TensorDim weight_dim(
    {1, 1, KERNEL_SIZE, W},
    {context.getFormat(), ml::train::TensorDim::DataType::FP32});

  // Conv-state cache: [B, 1, KERNEL_SIZE-1, W] FP32
  //   state[b, 0, 0, f] = x_{t-2}
  //   state[b, 0, 1, f] = x_{t-1}
  nntrainer::TensorDim state_dim(
    {B, 1, KERNEL_SIZE - 1, W},
    {context.getFormat(), ml::train::TensorDim::DataType::FP32});

  std::vector<nntrainer::TensorDim> output_dims = {in_dim};
  std::vector<nntrainer::TensorDim> weight_dims = {weight_dim};
  std::vector<nntrainer::TensorDim> tensor_dims = {state_dim};

  return std::array<std::vector<nntrainer::TensorDim>, 3>{
    output_dims, weight_dims, tensor_dims};
}

std::vector<nntrainer::TensorDim>
CausalConv1DLayer::updateTensorsByInputDimensions(
  nntrainer::InitLayerContext &init_context,
  nntrainer::RunLayerContext &run_context) {
  [[maybe_unused]] auto [output_dims, weight_dims, tensor_dims] =
    getLayerDimensions(init_context);

  run_context.updateInput(SINGLE_INOUT_IDX,
                          init_context.getInputDimensions()[SINGLE_INOUT_IDX]);
  run_context.updateOutput(SINGLE_INOUT_IDX, output_dims[SINGLE_INOUT_IDX]);

  run_context.updateTensor(tensor_idx[conv_state], tensor_dims[0]);

  return output_dims;
}

void CausalConv1DLayer::exportTo(nntrainer::Exporter &exporter,
                                 const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
}

void CausalConv1DLayer::setProperty(const std::vector<std::string> &values) {
  LayerImpl::setProperty(values);
}

} // namespace causallm

#ifdef PLUGGABLE
extern "C" {

nntrainer::Layer *create_causal_conv1d_layer() {
  return new causallm::CausalConv1DLayer();
}

void destroy_causal_conv1d_layer(nntrainer::Layer *layer) { delete layer; }

nntrainer::LayerPluggable ml_train_layer_pluggable{
  create_causal_conv1d_layer, destroy_causal_conv1d_layer,
  causallm::CausalConv1DLayer::type};
}
#endif
