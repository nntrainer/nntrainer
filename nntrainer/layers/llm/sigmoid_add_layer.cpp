// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   sigmoid_add_layer.cpp
 * @date   12 September 2026
 * @brief  Backend-neutral sigmoid-gated add: out = sigmoid(gate) + in2.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "sigmoid_add_layer.h"

#include <nntrainer_error.h>
#include <node_exporter.h>
#include <tensor.h>

namespace nntrainer {

static constexpr size_t OUT_IDX = 0;
static constexpr size_t INPUT_IDX_1 = 0; // gate (the sigmoid argument)
static constexpr size_t INPUT_IDX_2 = 1; // the operand added to the gate

void SigmoidAddLayer::finalize(InitLayerContext &context) {
  if (!std::get<props::SkipPrefill>(sigmoid_add_props).empty())
    skip_prefill = std::get<props::SkipPrefill>(sigmoid_add_props).get();
  context.setOutputDimensions({context.getInputDimensions()[0]});
}

void SigmoidAddLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, sigmoid_add_props);
  if (!remain_props.empty()) {
    std::string msg = "[SigmoidAddLayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}

void SigmoidAddLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &in1 = context.getInput(INPUT_IDX_1);
  Tensor &in2 = context.getInput(INPUT_IDX_2);
  Tensor &out = context.getOutput(OUT_IDX);
  in1.getOps()->sigmoid_add(in1, in2, out,
                            in1.batch() * in1.channel() * in1.height(),
                            /*row_offset=*/0);
}

void SigmoidAddLayer::incremental_forwarding(RunLayerContext &context,
                                             unsigned int from, unsigned int to,
                                             bool training) {
  // skip-prefill gate, identical to GeGLULayer's: multi-token steps count as
  // prefill (a resumed multi-turn / KV-restored prefill arrives as one from>0
  // block call), so they are skipped exactly like the from==0 first prefill.
  if (skip_prefill && (from == 0 || (to - from) > 1))
    return;
  Tensor &in1 = context.getInput(INPUT_IDX_1);
  Tensor &in2 = context.getInput(INPUT_IDX_2);
  Tensor &out = context.getOutput(OUT_IDX);

  // active-row decision, mirroring GeGLULayer exactly: the producers write the
  // live rows starting at the buffer base on every backend, so row_offset
  // stays 0 on every path (see sigmoid_add_cl_op for why) and the count is the
  // live-row window. Rows are (batch*channel*height) flattened, so the count
  // is scaled the way forwarding() scales it; a partial window (to<height)
  // with batch*channel>1 is inexpressible as one contiguous row span and stays
  // unsupported.
  const unsigned int bc = in1.batch() * in1.channel();
  const unsigned int active_rows = (to - from) * bc;

  in1.getOps()->sigmoid_add(in1, in2, out, active_rows, /*row_offset=*/0);
}

void SigmoidAddLayer::calcDerivative(RunLayerContext &context) {
  std::throw_with_nested(std::runtime_error("Training is not supported yet."));
}

} // namespace nntrainer
