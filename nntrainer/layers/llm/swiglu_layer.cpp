// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Niket Agarwal <niket.a@samsung.com>
 *
 * @file   swiglu_layer.cpp
 * @date   6 June 2024
 * @see    https://github.com/nntrainer/nntrainer
 * @author Niket Agarwal <niket.a@samsung.com>
 * @author Eunju Yang <ej.yang@samsung.com>
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Backend-neutral SwiGLU activation: out = silu(gate) * up.
 */

#include "swiglu_layer.h"

#include <nntrainer_error.h>
#include <node_exporter.h>
#include <tensor.h>

namespace nntrainer {

static constexpr size_t OUT_IDX = 0;
static constexpr size_t INPUT_IDX_1 = 0; /**< gate */
static constexpr size_t INPUT_IDX_2 = 1; /**< up */

void SwiGLULayer::finalize(InitLayerContext &context) {
  context.setOutputDimensions({context.getInputDimensions()[0]});

  if (!std::get<props::SkipPrefill>(swiglu_props).empty())
    skip_prefill = std::get<props::SkipPrefill>(swiglu_props).get();
}

void SwiGLULayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, swiglu_props);
  if (!remain_props.empty()) {
    std::string msg = "[SwiGLULayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}

void SwiGLULayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &in1 = context.getInput(INPUT_IDX_1);
  Tensor &in2 = context.getInput(INPUT_IDX_2);
  Tensor &out = context.getOutput(OUT_IDX);

  // Rows are the (batch, channel, height) axes flattened; width() is the
  // per-row element count.
  in1.getOps()->swiglu(in1, in2, out,
                       in1.batch() * in1.channel() * in1.height(),
                       /*row_offset=*/0);
}

void SwiGLULayer::incremental_forwarding(RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) {
  // A multi-token step is a prefill step, so a KV-shared layer skips it here
  // exactly as it skips the from == 0 one: the branch produces nothing any
  // later layer reads.
  if (skip_prefill && (from == 0 || (to - from) > 1))
    return;

  Tensor &in1 = context.getInput(INPUT_IDX_1);
  Tensor &in2 = context.getInput(INPUT_IDX_2);
  Tensor &out = context.getOutput(OUT_IDX);

  // Every producer writes the live rows starting at the buffer base on every
  // backend, so the offset is always 0 and the count is the live-row window.
  // Rows are (batch, channel, height) flattened, so the window has to be
  // scaled by batch*channel or a batch>1 shape only computes its first
  // (to - from) rows.
  const unsigned int bc = in1.batch() * in1.channel();
  in1.getOps()->swiglu(in1, in2, out, (to - from) * bc, /*row_offset=*/0);
}

void SwiGLULayer::updateTensorsByInputDimensions(
  RunLayerContext &context, std::vector<TensorDim> input_dimensions) {
  TensorDim input_dim1 = context.getInput(INPUT_IDX_1).getDim();
  TensorDim input_dim2 = context.getInput(INPUT_IDX_2).getDim();
  TensorDim output_dim = context.getOutput(OUT_IDX).getDim();

  input_dim1.height(input_dimensions[0].height());
  input_dim2.height(input_dimensions[0].height());
  output_dim.height(input_dimensions[0].height());

  context.updateInput(INPUT_IDX_1, input_dim1);
  context.updateInput(INPUT_IDX_2, input_dim2);
  context.updateOutput(OUT_IDX, output_dim);
}

void SwiGLULayer::calcDerivative(RunLayerContext &context) {
  std::throw_with_nested(std::runtime_error("Training is not supported yet."));
}

} // namespace nntrainer
