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

#include <cpu_backend.h>
#include <nntrainer_error.h>

#include "htp_decode_hook.h"

namespace causallm {

static constexpr size_t OUT_IDX = 0;

void ResidualAddLayer::finalize(nntrainer::InitLayerContext &context) {
  const unsigned int n_in = context.getNumInputs();
  NNTR_THROW_IF(n_in != 2 && n_in != 3, std::invalid_argument)
    << "ResidualAddLayer takes 2 or 3 inputs (residual, addend[, addend2])";
  const std::vector<nntrainer::TensorDim> dim = context.getInputDimensions();
  context.setOutputDimensions({dim[0]});

  in_norm = std::get<props::InNorm>(props_).get();
  use_scale = std::get<props::UseWeight>(props_).get();
  proj = std::get<props::Proj>(props_).get();
  out_to_host = std::get<props::OutToHost>(props_).get();
  NNTR_THROW_IF(proj && n_in != 2, std::invalid_argument)
    << "ResidualAddLayer: proj takes (residual, the projection's input)";
  if (!std::get<nntrainer::props::SkipPrefill>(props_).empty())
    skip_prefill = std::get<nntrainer::props::SkipPrefill>(props_).get();
  const bool fused = in_norm || use_scale || n_in == 3 || proj;
  // ponytail: the folded forms are FP32 only (the one activation type the
  // model that uses them runs); the plain two-input add keeps every type.
  NNTR_THROW_IF(fused &&
                  dim[0].getDataType() != ml::train::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "ResidualAddLayer: in_norm / use_weight / a third input need FP32";

  const nntrainer::TensorDim::TensorType f32(
    context.getFormat(), ml::train::TensorDim::DataType::FP32);
  // the file's order: the projection (FullyConnectedLayer's weight shape),
  // the norm's gamma, then the block's scalar
  if (proj) {
    const bool is_nchw = context.getFormat() == nntrainer::Tformat::NCHW;
    const unsigned int unit = dim[0].width();
    proj_idx = context.requestWeight(
      nntrainer::TensorDim(1, is_nchw ? 1 : unit, is_nchw ? dim[1].width() : 1,
                           is_nchw ? unit : dim[1].channel(),
                           nntrainer::TensorDim::TensorType(
                             context.getFormat(), context.getWeightDataType()),
                           is_nchw ? 0b0011 : 0b0101),
      // ones, as the o-proj layer it replaces was built (a model file
      // overwrites it; the tiny test models run on it)
      nntrainer::props::InitializerInfo::Enum::ONES,
      nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "weight", true);
  }
  if (in_norm)
    gamma_idx = context.requestWeight(
      nntrainer::TensorDim(1, 1, 1, dim[0].width(), f32),
      nntrainer::props::InitializerInfo::Enum::NONE,
      nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "gamma", true);
  if (use_scale)
    scale_idx =
      context.requestWeight(nntrainer::TensorDim(1, 1, 1, 1, f32),
                            nntrainer::props::InitializerInfo::Enum::NONE,
                            nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f,
                            "scalar_multiplier", true);
  if (fused)
    sum_idx =
      context.requestTensor(dim[0], "sum", nntrainer::Initializer::NONE, false,
                            nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
}

void ResidualAddLayer::forwarding(nntrainer::RunLayerContext &context,
                                  bool training) {
  run(context, 0, context.getOutput(OUT_IDX).height());
}

void ResidualAddLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  const bool is_prefill = !from || (to - from) > 1;
  if (skip_prefill && is_prefill)
    return;
  run(context, from, to);
}

void ResidualAddLayer::run(nntrainer::RunLayerContext &context,
                           unsigned int from, unsigned int to) {
  nntrainer::Tensor &out = context.getOutput(OUT_IDX);
  const nntrainer::Tensor &in0 = context.getInput(0);
  const nntrainer::Tensor &in1 = context.getInput(1);
  const bool two = context.getNumInputs() == 3;
  const bool fused = in_norm || use_scale || two || proj;
  const float eps = std::get<nntrainer::props::Epsilon>(props_).get();
  const float *gamma =
    in_norm ? context.getWeight(gamma_idx).getData<float>() : nullptr;
  const float scale =
    use_scale ? context.getWeight(scale_idx).getValue<float>(0, 0, 0, 0) : 1.0f;
  const unsigned int rows = to - from;
  const unsigned int W = out.width();

  // the tensors are sized to the step (a decode row is row 0), as every
  // layer's incremental_forwarding takes them
  auto step = [&](const nntrainer::Tensor &t, unsigned int b) {
    nntrainer::TensorDim d = t.getDim();
    d.batch(1);
    d.height(rows);
    return t.getSharedDataTensor(d, b * t.getDim().getFeatureLen(), true);
  };

  for (unsigned int b = 0; b < out.batch(); ++b) {
    nntrainer::Tensor out_step = step(out, b);
    nntrainer::Tensor in0_step = step(in0, b);
    nntrainer::Tensor in1_step = step(in1, b);
    // [#132] one plain decode row: the HTP adds into its resident residual
    if (!fused && rows == 1 && out.batch() == 1 &&
        out.getDataType() == ml::train::TensorDim::DataType::FP32 &&
        htpDecodeAdd(from, in1_step.getData<float>(), out_step.getData<float>(),
                     W))
      continue;
    // [#260] a fused epilogue's row under the one-PD token: its norm, add
    // and scalar are the list's RMSNORM / ADD ops, run by the DSP
    if (fused && rows == 1 && out.batch() == 1 && htpDecodeRowResident(from))
      continue;
    if (!fused) {
      out_step.copy(in0_step);
      out_step.add_i(in1_step);
      continue;
    }
    const float *x2 =
      two ? step(context.getInput(2), b).getData<float>() : nullptr;
    nntrainer::ComputeOps *ops = out_step.getOps();
    if (proj) {
      // The projection and the epilogue as one accelerator call at
      // prefill; otherwise the projection here (an engine's FC through
      // dot(), decode's row included) and the epilogue on its output.
      nntrainer::Tensor &w = context.getWeight(proj_idx);
      if (rows > 1 && ops != nullptr && W % 32 == 0 &&
          w.getDataType() == ml::train::TensorDim::DataType::QS4CX &&
          ops->gemm_qs4cx_res_add_fp32(
            w.getData<char>(), w.getScale<float>(), in1_step.getData<float>(),
            rows, in1_step.width(), W, in0_step.getData<float>(), gamma, eps,
            scale, out_step.getData<float>(), out_to_host))
        continue;
      in1_step = step(context.getTensor(sum_idx), b);
      step(in1, b).dot(w, in1_step, false, false);
    }
    // The whole epilogue as one accelerator call at prefill; decode's one
    // row cannot amortize the call and stays here.
    if (rows > 1 && ops != nullptr && ops->supports_rmsnorm_add_fp32() &&
        W % 32 == 0) {
      ops->rmsnorm_add_fp32(rows, W, in0_step.getData<float>(),
                            in1_step.getData<float>(), x2, gamma, eps, scale,
                            out_step.getData<float>());
      continue;
    }
    nntrainer::Tensor sum = step(context.getTensor(sum_idx), b);
    sum.copy(in1_step);
    if (two)
      sum.add_i(step(context.getInput(2), b));
    if (in_norm) {
      // the norm kernel is not in place: sum -> out, gamma, then the add
      nntrainer::rms_norm_wrt_width_fp32_intrinsic(
        sum.getData<float>(), out_step.getData<float>(), rows, W, eps);
      out_step.multiply_i(context.getWeight(gamma_idx));
      out_step.add_i(in0_step);
    } else {
      out_step.copy(in0_step);
      out_step.add_i(sum);
    }
    if (use_scale)
      out_step.multiply_i(scale);
  }
}

void ResidualAddLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  for (unsigned int idx = 0; idx < context.getNumInputs(); ++idx)
    context.getOutgoingDerivative(idx).copy(
      context.getIncomingDerivative(OUT_IDX));
}

void ResidualAddLayer::setProperty(const std::vector<std::string> &values) {
  auto remain = loadProperties(values, props_);
  NNTR_THROW_IF(!remain.empty(), std::invalid_argument)
    << "[ResidualAddLayer] Unknown Layer Properties count "
    << std::to_string(remain.size());
}

} // namespace causallm
