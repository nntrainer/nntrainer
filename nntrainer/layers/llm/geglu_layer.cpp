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
                      /** row_offset */ 0);
}

void GeGLULayer::incremental_forwarding(RunLayerContext &context,
                                        unsigned int from, unsigned int to,
                                        bool training) {
  if (skip_prefill && from == 0)
    return;
  Tensor &in1 = context.getInput(INPUT_IDX_1);
  Tensor &in2 = context.getInput(INPUT_IDX_2);
  Tensor &out = context.getOutput(OUT_IDX);

  if (from) {
    NNTR_THROW_IF(to - from != 1, std::invalid_argument)
      << "incremental step size is not 1";
  }

  // active-row decision (unifies the former GeGLULayerCl / CudaGeGLULayer
  // forks; CUDA/CPU tensors are never cl_mem so they fall to the `else`):
  //  - all-cl_mem fp16 decode: the producers (v8c FC) write the live token to
  //    row 0, so process exactly 1 row at the buffer base (O(1) at decode).
  //  - any other cl_mem (mixed / fp32): process the whole [0,to) window.
  //  - SVM/host/UVM: process the live rows [from, to) from the base (=1 at
  //    decode; the live token is at row 0 of the rebased activation buffers).
  // row_offset stays 0 on every path (see geglu_cl_op for why).
  const bool any_clmem = in1.isClMem() || in2.isClMem() || out.isClMem();
  const bool all_clmem = in1.isClMem() && in2.isClMem() && out.isClMem();
  const bool is_fp16 =
    in1.getDataType() == ml::train::TensorDim::DataType::FP16;

  // Rows are (batch*channel*height) flattened — scale the count like
  // forwarding() does, or batch/channel>1 shapes only process the first
  // (to-from) of batch*channel*height rows (stale output beyond them).
  // Production decode has batch=channel=1, where this is a no-op. A partial
  // window (to<height) with batch*channel>1 is inexpressible as one contiguous
  // row span and stays unsupported.
  const unsigned int bc = in1.batch() * in1.channel();

  unsigned int active_rows;
  if (from && all_clmem && is_fp16)
    active_rows = bc;
  else if (any_clmem)
    active_rows = to * bc;
  else
    active_rows = (to - from) * bc;

  in1.getOps()->geglu(in1, in2, out, active_rows, /** row_offset */ 0);
}

void GeGLULayer::calcDerivative(RunLayerContext &context) {
  std::throw_with_nested(std::runtime_error("Training is not supported yet."));
}

} // namespace nntrainer
