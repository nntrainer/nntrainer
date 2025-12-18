// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   swiglu.cpp
 * @date   14 July 2023
 * @brief  Implementation of SwiGLU activation function
 * @see    https://github.com/nnstreamer/nntrainer
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
}

void SwiGLULayer::forwarding(nntrainer::RunLayerContext &context,
                             bool training) {}

void SwiGLULayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) {
  nntrainer::Tensor &in1 = context.getInput(INPUT_IDX_1);
  nntrainer::Tensor &in2 = context.getInput(INPUT_IDX_2);
  nntrainer::Tensor &out = context.getOutput(OUT_IDX);

  unsigned int _from = from;

  int iter = to - from;

  if (in1.getDataType() == ml::train::TensorDim::DataType::FP32) {
    for (unsigned int b = 0; b < in1.batch(); b++) {
      for (unsigned int c = 0; c < in1.channel(); c++) {
        for (unsigned int h = 0; h < iter; h++) {
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
        for (unsigned int h = 0; h < iter; h++) {
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
  // std::throw_with_nested(std::runtime_error("Training is not supported
  // yet."));
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
