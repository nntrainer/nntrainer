// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   geglu_layer.cpp
 * @date   29 June 2026
 * @brief  Backend-neutral GeGLU activation: out = gelu_tanh(gate) * up.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "geglu_layer.h"

#include <nntrainer_error.h>
#include <node_exporter.h>
#include <tensor.h>

namespace nntrainer {

static constexpr size_t OUT_IDX = 0;
static constexpr size_t INPUT_IDX_1 = 0; // gate
static constexpr size_t INPUT_IDX_2 = 1; // up

void GeGLULayer::finalize(InitLayerContext &context) {
  if (!std::get<props::SkipPrefill>(geglu_props).empty())
    skip_prefill = std::get<props::SkipPrefill>(geglu_props).get();
  context.setOutputDimensions({context.getInputDimensions()[0]});
}

void GeGLULayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, geglu_props);
  if (!remain_props.empty()) {
    std::string msg = "[GeGLULayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}

void GeGLULayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &in1 = context.getInput(INPUT_IDX_1);
  Tensor &in2 = context.getInput(INPUT_IDX_2);
  Tensor &out = context.getOutput(OUT_IDX);
  in1.getOps()->geglu(in1, in2, out, in1.batch() * in1.channel() * in1.height(),
                      /*row_offset=*/0);
}

void GeGLULayer::incremental_forwarding(RunLayerContext &context,
                                        unsigned int from, unsigned int to,
                                        bool training) {
  // skip-prefill gate: multi-token steps count as prefill (a resumed
  // multi-turn / KV-restored prefill arrives as one from>0 block call), so
  // they are skipped exactly like the from==0 first prefill.
  if (skip_prefill && (from == 0 || (to - from) > 1))
    return;
  Tensor &in1 = context.getInput(INPUT_IDX_1);
  Tensor &in2 = context.getInput(INPUT_IDX_2);
  Tensor &out = context.getOutput(OUT_IDX);

  // active-row decision (unifies the former GeGLULayerCl / CudaGeGLULayer
  // forks): the producers (v8c FC etc.) write the live rows starting at the
  // buffer base on every backend, so row_offset stays 0 on every path (see
  // geglu_cl_op for why) and the count is the live-row window.
  //
  // Rows are (batch*channel*height) flattened — scale the count like
  // forwarding() does, or batch/channel>1 shapes only process the first
  // (to-from) of batch*channel*height rows (stale output beyond them).
  // Production decode has batch=channel=1, where this is a no-op. A partial
  // window (to<height) with batch*channel>1 is inexpressible as one contiguous
  // row span and stays unsupported.
  //
  // (to-from)*bc unifies the former three-way branch: for
  // from==0 prefill to == to-from (the old `to * bc` cl_mem window), for
  // decode to-from == 1 (the old all-cl_mem-fp16 `bc` fast path), and a
  // resumed multi-token prefill (from>0, to-from>1) processes its to-from
  // rows at the base.
  const unsigned int bc = in1.batch() * in1.channel();
  const unsigned int active_rows = (to - from) * bc;

  in1.getOps()->geglu(in1, in2, out, active_rows, /*row_offset=*/0);
}

void GeGLULayer::calcDerivative(RunLayerContext &context) {
  std::throw_with_nested(std::runtime_error("Training is not supported yet."));
}

} // namespace nntrainer
