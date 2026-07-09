// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   scalar_multiply.cpp
 * @date   7 April 2026
 * @brief  Implementation of scalar multiplication layer
 * @see    https://github.com/nntrainer/nntrainer
 * @author Joonseok Oh <jrock.oh@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <env_compat.h>
#include <iostream>
#include <string>

#include "scalar_multiply.h"

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

// NNTR_DUMP_LAYERS=1 -> per-call stats of the scalar_multiply input/output and
// the multiplier. Every Gemma4 decoder block ends with a layer_scalar
// scalar_multiply, so this traces the per-layer hidden state (and verifies the
// learned layer_scalar value, e.g. layer0 ~= 0.0178) for HF-parity debugging.
static std::string nntr_tensor_stats(nntrainer::Tensor &t) {
  const float *d = t.getData<float>();
  size_t n = t.size();
  double s = 0, s2 = 0, amax = 0;
  for (size_t i = 0; i < n; ++i) {
    double v = d[i];
    s += v;
    s2 += v * v;
    if (std::fabs(v) > amax)
      amax = std::fabs(v);
  }
  double mean = n ? s / n : 0;
  double var = n ? s2 / n - mean * mean : 0;
  char buf[160];
  std::snprintf(buf, sizeof buf, "mean=%.5g std=%.5g absmax=%.5g", mean,
                std::sqrt(var < 0 ? 0 : var), amax);
  return std::string(buf);
}

void ScalarMultiplyLayer::finalize(nntrainer::InitLayerContext &context) {
  std::vector<nntrainer::TensorDim> dim = context.getInputDimensions();
  context.setOutputDimensions(dim);
  if (!std::get<nntrainer::props::SkipPrefill>(scalar_multiply_props).empty())
    skip_prefill =
      std::get<nntrainer::props::SkipPrefill>(scalar_multiply_props).get();

  bool use_weight = std::get<props::UseWeight>(scalar_multiply_props).get();

  if (use_weight) {
    // Request weight for scalar value (single element)
    nntrainer::TensorDim scalar_dim(
      1, 1, 1, 1,
      nntrainer::TensorDim::TensorType(context.getFormat(),
                                       context.getWeightDataType()));
    wt_idx[0] = context.requestWeight(
      scalar_dim, nntrainer::props::InitializerInfo::Enum::NONE,
      nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "scalar_multiplier",
      false);
  }
}

void ScalarMultiplyLayer::forwarding(nntrainer::RunLayerContext &context,
                                     bool training) {
  // Use incremental_forwarding for actual computation
  auto &in = context.getInput(SINGLE_INOUT_IDX);
  auto &out = context.getOutput(SINGLE_INOUT_IDX);

  bool use_weight = std::get<props::UseWeight>(scalar_multiply_props).get();

  float multiplier;
  if (use_weight) {
    nntrainer::Tensor &weight = context.getWeight(wt_idx[0]);
    // dtype-aware scalar read: getValue<float> on an FP16 weight reads 4 bytes
    // from the 2-byte cell (garbage). gemma4 layer_scalar / q_scale /
    // model_proj_scale weights are FP16 in QINT4-FP16 models -> reading them as
    // float gave a huge multiplier and overflowed the residual to +Inf.
#ifdef ENABLE_FP16
    multiplier = (weight.getDataType() == ml::train::TensorDim::DataType::FP16)
                   ? static_cast<float>(weight.getValue<_FP16>(0, 0, 0, 0))
                   : weight.getValue<float>(0, 0, 0, 0);
#else
    multiplier = weight.getValue<float>(0, 0, 0, 0);
#endif
  } else {
    multiplier = std::get<props::ScalarMultiplier>(scalar_multiply_props).get();
  }

  in.multiply(multiplier, out);
}

void ScalarMultiplyLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  bool is_prefill = !from;
  if (skip_prefill && is_prefill)
    return;

  bool use_weight = std::get<props::UseWeight>(scalar_multiply_props).get();

  float multiplier;
  if (use_weight) {
    nntrainer::Tensor &weight = context.getWeight(wt_idx[0]);
    // dtype-aware scalar read: getValue<float> on an FP16 weight reads 4 bytes
    // from the 2-byte cell (garbage). gemma4 layer_scalar / q_scale /
    // model_proj_scale weights are FP16 in QINT4-FP16 models -> reading them as
    // float gave a huge multiplier and overflowed the residual to +Inf.
#ifdef ENABLE_FP16
    multiplier = (weight.getDataType() == ml::train::TensorDim::DataType::FP16)
                   ? static_cast<float>(weight.getValue<_FP16>(0, 0, 0, 0))
                   : weight.getValue<float>(0, 0, 0, 0);
#else
    multiplier = weight.getValue<float>(0, 0, 0, 0);
#endif
  } else {
    multiplier = std::get<props::ScalarMultiplier>(scalar_multiply_props).get();
  }

  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);

  ml::train::TensorDim in_dim = in.getDim();
  ml::train::TensorDim out_dim = out.getDim();

  ml::train::TensorDim in_step_dim = in_dim;
  ml::train::TensorDim out_step_dim = out_dim;

  in_step_dim.batch(1);
  in_step_dim.height(to - from);
  out_step_dim.batch(1);
  out_step_dim.height(to - from);

  unsigned int b_size = in_dim.batch();

  for (unsigned int b = 0; b < b_size; ++b) {
    nntrainer::Tensor in_step =
      in.getSharedDataTensor(in_step_dim, b * in_dim.getFeatureLen(), true);
    nntrainer::Tensor out_step =
      out.getSharedDataTensor(out_step_dim, b * out_dim.getFeatureLen(), true);

    // Whole-op dispatch: the layer keeps the chunk/step bookkeeping; the
    // attached ComputeOps table runs one chunk (an accelerator table may take
    // a device kernel under its own gates, the host table runs the plain
    // multiply).
    in_step.getOps()->scalar_mul(in_step, out_step, multiplier);

    static const bool dump_layers = std::getenv("NNTR_DUMP_LAYERS") != nullptr;
    if (dump_layers) {
      std::fprintf(stderr,
                   "[DUMP] %-34s from=%u to=%u mult=%.6g | in %s | out %s\n",
                   context.getName().c_str(), from, to, multiplier,
                   nntr_tensor_stats(in_step).c_str(),
                   nntr_tensor_stats(out_step).c_str());
    }

#ifdef DEBUG
    std::cout << context.getName() << " \n input:" << in_step
              << "output:" << out_step << "multiplier:" << multiplier
              << std::endl;
#endif
  }
}

void ScalarMultiplyLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  context.updateInput(SINGLE_INOUT_IDX, input_dimensions[0]);
  context.updateOutput(SINGLE_INOUT_IDX, input_dimensions[0]);
}

void ScalarMultiplyLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  std::throw_with_nested(std::runtime_error("Training is not supported yet."));
}

} // namespace nntrainer
