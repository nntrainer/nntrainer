// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   swiglu.cpp
 * @date   14 July 2023
 * @brief  Implementation of SwiGLU activation function
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <util_simd.h>

#include "swiglu.h"

namespace causallm {

static constexpr size_t OUT_IDX = 0;
static constexpr size_t INPUT_IDX_1 = 0;
static constexpr size_t INPUT_IDX_2 = 1;

namespace ActivationOp {
/**
 * @brief activation function swiglu
 * @param x input
 * @retval swiglu(x)
 */
float swiglu(float x) { return x / (1 + nntrainer::exp_util(-x)); }
} // namespace ActivationOp

void SwiGLULayer::finalize(nntrainer::InitLayerContext &context) {
  context.setOutputDimensions({context.getInputDimensions()[0]});
  tensor_idx[SwiGLUParams::sigmoid_gate] =
    context.requestTensor(context.getInputDimensions()[0], "sigmoid_gate",
                          nntrainer::Initializer::NONE, false,
                          nntrainer::TensorLifespan::ITERATION_LIFESPAN);
}

void SwiGLULayer::forwarding(nntrainer::RunLayerContext &context,
                             bool training) {

  nntrainer::Tensor &in1 = context.getInput(INPUT_IDX_1);
  unsigned int height = in1.getDim().height();

  // Process the full sequence by calling incremental_forwarding once with
  // from=0 and to=height. This unifies the implementation and ensures
  // training/inference consistency.
  incremental_forwarding(context, 0, height, training);
}

void SwiGLULayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) {

  nntrainer::Tensor &in1 = context.getInput(INPUT_IDX_1);
  nntrainer::Tensor &in2 = context.getInput(INPUT_IDX_2);
  nntrainer::Tensor &out = context.getOutput(OUT_IDX);

  if (in1.getDataType() == ml::train::TensorDim::DataType::FP32) {
    for (unsigned int b = 0; b < in1.batch(); b++) {
      for (unsigned int c = 0; c < in1.channel(); c++) {
        for (unsigned int h = from; h < to; h++) {
          nntrainer::swiglu(in1.width(),
                            out.getData<float>() + out.getIndex(b, c, h, 0),
                            in1.getData<float>() + in1.getIndex(b, c, h, 0),
                            in2.getData<float>() + in2.getIndex(b, c, h, 0));
        }
      }
    }
  } else if (in1.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    for (unsigned int b = 0; b < in1.batch(); b++) {
      for (unsigned int c = 0; c < in1.channel(); c++) {
        for (unsigned int h = from; h < to; h++) {
          nntrainer::swiglu(in1.width(),
                            out.getData<_FP16>() + out.getIndex(b, c, h, 0),
                            in1.getData<_FP16>() + in1.getIndex(b, c, h, 0),
                            in2.getData<_FP16>() + in2.getIndex(b, c, h, 0));
        }
      }
    }
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }
}

void SwiGLULayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  ml::train::TensorDim input_dim1 = context.getInput(INPUT_IDX_1).getDim();
  ml::train::TensorDim input_dim2 = context.getInput(INPUT_IDX_2).getDim();
  ml::train::TensorDim output_dim = context.getOutput(OUT_IDX).getDim();

  input_dim1.height(input_dimensions[0].height());
  input_dim2.height(input_dimensions[0].height());
  output_dim.height(input_dimensions[0].height());

  context.updateInput(INPUT_IDX_1, input_dim1);
  context.updateInput(INPUT_IDX_2, input_dim2);
  context.updateOutput(OUT_IDX, output_dim);
}

void SwiGLULayer::calcDerivative(nntrainer::RunLayerContext &context) {

  const nntrainer::Tensor &incoming_deriv =
    context.getIncomingDerivative(OUT_IDX);
  nntrainer::Tensor &d_gate = context.getOutgoingDerivative(INPUT_IDX_1);
  nntrainer::Tensor &d_up = context.getOutgoingDerivative(INPUT_IDX_2);
  nntrainer::Tensor &gate = context.getInput(INPUT_IDX_1);
  nntrainer::Tensor &up = context.getInput(INPUT_IDX_2);
  nntrainer::Tensor &sig_gate =
    context.getTensor(tensor_idx[SwiGLUParams::sigmoid_gate]);

  if (gate.getDataType() == ml::train::TensorDim::DataType::FP32) {
    // Generate sig_gate on-the-fly, as forwarding now bypasses it for SIMD
    // speed
    gate.apply<float>(
      [](float x) { return 1.0f / (1.0f + nntrainer::exp_util(-x)); },
      sig_gate);
    // d_up = Swish(gate) * dy
    gate.multiply(sig_gate, d_up);
    d_up.multiply_i(incoming_deriv);

    // Swish'(gate) = sig_gate + gate * sig_gate * (1 - sig_gate)
    // We use d_gate as a memory-safe buffer to calculate (1 - sig_gate) to
    // avoid allocations
    d_gate.copyData(sig_gate);
    d_gate.multiply_i(-1.0f);
    d_gate.add_i(1.0f);

    // d_gate is now (1 - sig_gate)
    d_gate.multiply_i(gate);
    d_gate.multiply_i(sig_gate); // gate * sig_gate * (1 - sig_gate)
    d_gate.add_i(sig_gate);      // Swish'(gate)

    d_gate.multiply_i(up);
    d_gate.multiply_i(incoming_deriv);
  } else if (gate.getDataType() == ml::train::TensorDim::DataType::FP16) {
    throw std::invalid_argument(
      "SwiGLULayer calcDerivative for FP16 is not yet implemented");
  }
}

#ifdef PLUGGABLE

nntrainer::Layer *create_swiglu_layer() {
  auto layer = new SwiGLULayer();
  return layer;
}

void destroy_swiglu_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_swiglu_layer,
                                                   destroy_swiglu_layer};
}

#endif

} // namespace causallm