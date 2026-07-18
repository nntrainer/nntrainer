// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   swiglu_layer.cpp
 * @date   29 June 2026
 * @brief  Backend-neutral SwiGLU activation: out = silu(gate) * up.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "swiglu_layer.h"

#include <nntrainer_error.h>
#include <node_exporter.h>
#include <tensor.h>

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
#include <cuda_context_manager.h>
#include <cuda_elementwise.h>
#include <cuda_runtime.h>
#endif

namespace nntrainer {

static constexpr size_t OUT_IDX = 0;
static constexpr size_t INPUT_IDX_1 = 0; // gate
static constexpr size_t INPUT_IDX_2 = 1; // up

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
  in1.getOps()->swiglu(in1, in2, out,
                       in1.batch() * in1.channel() * in1.height(),
                       0); /* row_offset */
}

void SwiGLULayer::incremental_forwarding(RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) {
  Tensor &in1 = context.getInput(INPUT_IDX_1);
  Tensor &in2 = context.getInput(INPUT_IDX_2);
  Tensor &out = context.getOutput(OUT_IDX);

  // skip-prefill gate (merged from the former app fork): KV-shared layers
  // skip the prefill activation. Inert unless a model sets skip_prefill on
  // swiglu. Multi-token steps count as prefill (upstream 1a80cfa35), and the
  // gate must run BEFORE the step-size assert so a resumed multi-token prefill
  // is skipped instead of throwing.
  if (skip_prefill && (from == 0 || (to - from) > 1))
    return;

  if (from) {
    NNTR_THROW_IF(to - from != 1, std::invalid_argument)
      << "incremental step size is not 1";
  }

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1 && defined(ENABLE_FP16)
  // engine=cuda device-resident fp16: one kernel instead of the host loop (the
  // getOps host path below would fault on the device-only activation pool under
  // NNTR_CUDA_DEV_ACT). Gated on FP16 + batch/channel==1; falls through for
  // OpenCL/CPU (cl_mem / host) and non-device tensors.
  if (in1.getDataType() == ml::train::TensorDim::DataType::FP16 &&
      in1.batch() == 1 && in1.channel() == 1) {
    const size_t n = (size_t)(to - from) * in1.width();
    auto *a = reinterpret_cast<const unsigned short *>(in1.getData<_FP16>());
    auto *b = reinterpret_cast<const unsigned short *>(in2.getData<_FP16>());
    auto *o = reinterpret_cast<unsigned short *>(out.getData<_FP16>());
    const bool dev = a && nntrainer::cuda::dev_accessible(a);
    if (dev && n > 0 &&
        nntrainer::cuda::cuda_swiglu_fp16(a, b, o, (unsigned int)n))
      return;
  }
#endif

  // active-row decision -- mirror GeGLULayer EXACTLY: the producers write the
  // live decode token to row 0 on every backend (cl_mem, SVM, and host), so the
  // row offset is always 0; only the row COUNT differs. (The earlier
  // `row_offset = from` on the host branch was wrong -- it read the wrong row
  // for engine=cpu decode, diverging the FP32 reference; the GPU branches and
  // the GeGLU host branch all use offset 0, and engine=gpu was token-identical
  // only because it never took this host branch.)
  //  - all-cl_mem fp16 decode: exactly 1 row at the buffer base (also avoids
  //  the
  //    one-row-out-of-bounds cl_mem write the old [0,to) branch could trigger).
  //  - any other cl_mem (mixed / fp32): the whole [0,to) window.
  //  - SVM/host: [0, to-from) (== row 0 for decode, the live token's slot).
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

  in1.getOps()->swiglu(in1, in2, out, active_rows, 0); /* row_offset */
}

void SwiGLULayer::calcDerivative(RunLayerContext &context) {
  std::throw_with_nested(std::runtime_error("Training is not supported yet."));
}

} // namespace nntrainer
