// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   rms_norm_layer.cpp
 * @date   29 Jul 2026
 * @brief  Backend-neutral RMS normalization:
 *         out = in * rsqrt(mean(in^2) + eps) * gamma.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "rms_norm_layer.h"

#include <nntrainer_error.h>
#include <tensor.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;
enum RMSParams { gamma };

void RMSNormLayer::finalize(InitLayerContext &context) {
  std::vector<TensorDim> dim = context.getInputDimensions();
  context.setOutputDimensions(dim);

  if (!std::get<props::SkipPrefill>(rms_props).empty())
    skip_prefill = std::get<props::SkipPrefill>(rms_props).get();

  // gamma follows the model weight dtype: quantized bins with FP16 norm
  // weights store gamma as FP16, and an FP32 hardcode here would positionally
  // misread the packed weight file from this tensor onward. FP32-weight bins
  // request FP32 gamma as before; a dtype mismatch against the activation is
  // resolved inside the whole-op (gamma is cloned to the activation dtype at
  // the multiply).
  TensorDim gamma_dim(
    1, 1, 1, dim[0].width(),
    TensorDim::TensorType(context.getFormat(), context.getWeightDataType()));
  wt_idx[RMSParams::gamma] =
    context.requestWeight(gamma_dim, props::InitializerInfo::Enum::NONE,
                          WeightRegularizer::NONE, 1.0f, 0.0f, "gamma", true);
}

void RMSNormLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  incremental_forwarding(context, 0, in.getDim().height(), training);
}

void RMSNormLayer::incremental_forwarding(RunLayerContext &context,
                                          unsigned int from, unsigned int to,
                                          bool training) {
  // skip-prefill gate: KV-shared layers skip the prefill norm. Multi-token
  // steps count as prefill (a resumed prefill re-enters with from > 0 and
  // to - from > 1), matching the merged SwiGLU/GeGLU gate semantics.
  if (skip_prefill && (from == 0 || (to - from) > 1))
    return;

  auto &epsilon = std::get<props::Epsilon>(rms_props).get();
  Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &gamma = context.getWeight(wt_idx[RMSParams::gamma]);

  TensorDim in_dim = in.getDim();
  TensorDim out_dim = out.getDim();
  TensorDim in_step_dim = in_dim;
  TensorDim out_step_dim = out_dim;
  in_step_dim.batch(1);
  in_step_dim.height(to - from);
  out_step_dim.batch(1);
  out_step_dim.height(to - from);

  // Rows are (channel*height) flattened per batch — every in-tree consumer
  // has channel == 1, where this equals the step height.
  const unsigned int rows_per_b = in_step_dim.channel() * (to - from);

  for (unsigned int b = 0; b < in_dim.batch(); ++b) {
    Tensor in_step =
      in.getSharedDataTensor(in_step_dim, b * in_dim.getFeatureLen(), true);
    Tensor out_step =
      out.getSharedDataTensor(out_step_dim, b * out_dim.getFeatureLen(), true);
    in_step.getOps()->rms_norm(in_step, out_step, gamma, epsilon, rows_per_b,
                               0); /* row_offset */
  }
}

void RMSNormLayer::updateTensorsByInputDimensions(
  RunLayerContext &context, std::vector<TensorDim> input_dimensions) {
  context.updateInput(SINGLE_INOUT_IDX, input_dimensions[0]);
  context.updateOutput(SINGLE_INOUT_IDX, input_dimensions[0]);
}

void RMSNormLayer::calcDerivative(RunLayerContext &context) {
  std::throw_with_nested(std::runtime_error("Training is not supported yet."));
}

} // namespace nntrainer
