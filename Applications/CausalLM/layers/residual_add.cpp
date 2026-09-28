// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   residual_add.cpp
 * @date   28 Sep 2026
 * @brief  The residual add of a decoder block, with the HTP per-token
 *         decode hook (#132)
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 */

#include "residual_add.h"

#include <nntrainer_error.h>

#include "htp_decode_hook.h"

namespace causallm {

static constexpr size_t OUT_IDX = 0;

void ResidualAddLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 2, std::invalid_argument)
    << "ResidualAddLayer requires exactly 2 inputs (residual, addend)";
  context.setOutputDimensions({context.getInputDimensions()[0]});
}

void ResidualAddLayer::forwarding(nntrainer::RunLayerContext &context,
                                  bool training) {
  nntrainer::Tensor &out = context.getOutput(OUT_IDX);
  out.copy(context.getInput(0));
  out.add_i(context.getInput(1));
}

void ResidualAddLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  nntrainer::Tensor &out = context.getOutput(OUT_IDX);
  const nntrainer::Tensor &in0 = context.getInput(0);
  const nntrainer::Tensor &in1 = context.getInput(1);
  nntrainer::TensorDim out_step_dim = out.getDim();
  nntrainer::TensorDim in0_step_dim = in0.getDim();
  nntrainer::TensorDim in1_step_dim = in1.getDim();
  out_step_dim.batch(1);
  out_step_dim.height(to - from);
  in0_step_dim.batch(1);
  in0_step_dim.height(to - from);
  in1_step_dim.batch(1);
  in1_step_dim.height(to - from);

  for (unsigned int b = 0; b < out.batch(); ++b) {
    nntrainer::Tensor out_step = out.getSharedDataTensor(
      out_step_dim, b * out.getDim().getFeatureLen(), true);
    nntrainer::Tensor in0_step = in0.getSharedDataTensor(
      in0_step_dim, b * in0.getDim().getFeatureLen(), true);
    nntrainer::Tensor in1_step = in1.getSharedDataTensor(
      in1_step_dim, b * in1.getDim().getFeatureLen(), true);
    // [#132] one decode row: the HTP adds into its resident residual
    if (to - from == 1 && out.batch() == 1 &&
        out.getDataType() == ml::train::TensorDim::DataType::FP32 &&
        htpDecodeAdd(from, in1_step.getData<float>(), out_step.getData<float>(),
                     static_cast<unsigned>(out_step_dim.width())))
      continue;
    out_step.copy(in0_step);
    out_step.add_i(in1_step);
  }
}

void ResidualAddLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  for (unsigned int idx = 0; idx < context.getNumInputs(); ++idx)
    context.getOutgoingDerivative(idx).copy(
      context.getIncomingDerivative(OUT_IDX));
}

void ResidualAddLayer::setProperty(const std::vector<std::string> &values) {
  NNTR_THROW_IF(!values.empty(), std::invalid_argument)
    << "[ResidualAddLayer] Unknown Layer Properties count "
    << std::to_string(values.size());
}

} // namespace causallm
