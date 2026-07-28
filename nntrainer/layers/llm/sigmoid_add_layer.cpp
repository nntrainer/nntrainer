// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   sigmoid_add_layer.cpp
 * @date   06 July 2026
 * @brief  Backend-neutral Sigmoid-add: out = sigmoid(gate) + emb.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "sigmoid_add_layer.h"

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
static constexpr size_t INPUT_IDX_1 = 0; // gate (sigmoid arg)
static constexpr size_t INPUT_IDX_2 = 1; // emb

void SigmoidAddLayer::finalize(InitLayerContext &context) {
  context.setOutputDimensions({context.getInputDimensions()[0]});
  if (!std::get<props::SkipPrefill>(sigmoid_add_props).empty())
    skip_prefill = std::get<props::SkipPrefill>(sigmoid_add_props).get();
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
  Tensor &in1 = context.getInput(INPUT_IDX_1);
  Tensor &in2 = context.getInput(INPUT_IDX_2);
  Tensor &out = context.getOutput(OUT_IDX);

  // skip-prefill gate: KV-shared PLE layers skip the prefill mix.
  // Inert unless a model sets skip_prefill on sigmoid_add. Multi-token steps
  // count as prefill (upstream 1a80cfa35), and the gate must run BEFORE the
  // step-size assert so a multi-token prefill is skipped, not thrown on.
  if (skip_prefill && (from == 0 || (to - from) > 1))
    return;

    // [prefill-chunk] from>0 no longer implies a single-token step: a chunked
    // prefill (NNTR_PREFILL_CHUNK) arrives as a block call with
    // from == the absolute chunk start and to-from == the chunk length. The
    // producers write the live rows at the buffer BASE on every backend
    // regardless of `from`, so the row math below is step-count-agnostic --
    // which is why the old `to - from != 1` assert could simply go.

// TODO(#14): CUDA fp16 fast-path — enable once the cuda_sigmoid_add_fp16
// wrapper lands in cuda_elementwise. Until then a CUDA build must not reference
// the undefined symbol, so this block is compiled out; engine=cuda falls
// through to getOps()->sigmoid_add (CudaComputeOps).
#if 0 && defined(ENABLE_CUDA) && ENABLE_CUDA == 1 && defined(ENABLE_FP16)
  // engine=cuda device-resident fp16: one kernel instead of the host loop.
  // Gated on FP16 + batch/channel==1; falls through for OpenCL/CPU and
  // non-device tensors.
  if (in1.getDataType() == ml::train::TensorDim::DataType::FP16 &&
      in1.batch() == 1 && in1.channel() == 1) {
    const size_t n = (size_t)(to - from) * in1.width();
    auto *a = reinterpret_cast<const unsigned short *>(in1.getData<_FP16>());
    auto *b = reinterpret_cast<const unsigned short *>(in2.getData<_FP16>());
    auto *o = reinterpret_cast<unsigned short *>(out.getData<_FP16>());
    const bool dev = a && nntrainer::cuda::dev_accessible(a);
    if (dev && n > 0 &&
        nntrainer::cuda::cuda_sigmoid_add_fp16(a, b, o, (unsigned int)n))
      return;
  }
#endif

  // active-row decision -- mirror SwiGLULayer EXACTLY (swiglu_layer.cpp): the
  // producers write the live decode token to row 0 on every backend (cl_mem,
  // SVM, and host), so the row offset is always 0; only the row COUNT differs.
  //  - all-cl_mem fp16 decode: exactly 1 row at the buffer base (also avoids
  //    the one-row-out-of-bounds cl_mem write a [0,to) window could trigger).
  //  - any other cl_mem (mixed / fp32): the whole [0,to) window.
  //  - SVM/host: [0, to-from) (== row 0 for decode, the live token's slot).

  // Rows are (batch*channel*height) flattened -- scale the count like
  // forwarding() does, or batch/channel>1 shapes only process the first
  // (to-from) rows. Production decode has batch=channel=1 (a no-op).
  const unsigned int bc = in1.batch() * in1.channel();

  // [prefill-chunk] (to-from)*bc replaces the former three-way branch: for
  // from==0 prefill `to == to-from` (identical to the old `to * bc` cl_mem
  // window), for decode `to-from == 1` (identical to the old all-cl_mem-fp16
  // `bc` fast path, which also avoided the one-row-out-of-bounds cl_mem write
  // a [0,to) window could trigger), and a chunked prefill (from>0, to-from>1)
  // processes exactly its to-from rows at the base.
  const unsigned int active_rows = (to - from) * bc;

  in1.getOps()->sigmoid_add(in1, in2, out, active_rows, /*row_offset=*/0);
}

void SigmoidAddLayer::calcDerivative(RunLayerContext &context) {
  std::throw_with_nested(std::runtime_error("Training is not supported yet."));
}

} // namespace nntrainer
