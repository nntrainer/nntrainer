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
#include <iostream>

#include "scalar_multiply.h"

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void ScalarMultiplyLayer::finalize(nntrainer::InitLayerContext &context) {
  std::vector<nntrainer::TensorDim> dim = context.getInputDimensions();
  context.setOutputDimensions(dim);
  if (!std::get<nntrainer::props::SkipPrefill>(scalar_multiply_props).empty())
    skip_prefill =
      std::get<nntrainer::props::SkipPrefill>(scalar_multiply_props).get();

  bool use_weight = std::get<props::UseWeight>(scalar_multiply_props).get();

  if (use_weight) {
    // Request weight for scalar value (single element).
    // @note The multiplier is a single un-quantized scalar coefficient and is
    // read back as float at the multiply site. Store it FP32 regardless of the
    // activation dtype: an FP16 scalar weight is both lossy and unreliable to
    // read here (getValue<float> on the FP16 slot returns garbage for some
    // layers), which silently zeroed the hidden state under FP16 activation.
    nntrainer::TensorDim scalar_dim(
      1, 1, 1, 1,
      nntrainer::TensorDim::TensorType(context.getFormat(),
                                       nntrainer::TensorDim::DataType::FP32));
    wt_idx[0] = context.requestWeight(
      scalar_dim, nntrainer::props::InitializerInfo::Enum::NONE,
      nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "scalar_multiplier",
      false);
  }
}

/**
 * @brief Resolve the multiplier for one step, memoizing a weight-borne one.
 *
 * `use_weight` puts the multiplier in a one-element static weight, and reading
 * it is a HOST read. On the GPU lanes the weight pool is unified memory
 * (cudaMallocManaged / SVM), so that 4-byte host read migrates the whole page
 * out of device memory -- and the neighbouring norm weights packed into the
 * same page are faulted straight back device-side by the next block's kernel.
 * That is one device->host->device page round trip per scalar_multiply per
 * step, for a value that never changes: measured at 55 4 KiB migrations and
 * about 0.13 ms per token on a 35-block model.
 *
 * So read it once per weight BUFFER and keep it. The key is the weight's data
 * address rather than a done flag, because deallocateTensors() +
 * allocateTensors() hands the layer a different buffer and the address
 * comparison re-arms the read there instead of serving a stale scalar. Taking
 * the address does not touch the page.
 *
 * @param context run context holding the (optional) scalar weight
 * @return the scalar to multiply the input by
 */
float ScalarMultiplyLayer::readMultiplier(nntrainer::RunLayerContext &context) {
  if (!std::get<props::UseWeight>(scalar_multiply_props).get())
    return std::get<props::ScalarMultiplier>(scalar_multiply_props).get();

  nntrainer::Tensor &weight = context.getWeight(wt_idx[0]);
  const void *addr = weight.getData<char>();
  if (addr != nullptr && addr == memo_weight_addr)
    return memo_multiplier;

  const float m = weight.getValue<float>(0, 0, 0, 0);
  memo_weight_addr = addr;
  memo_multiplier = m;
  return m;
}

void ScalarMultiplyLayer::forwarding(nntrainer::RunLayerContext &context,
                                     bool training) {
  // Use incremental_forwarding for actual computation
  auto &in = context.getInput(SINGLE_INOUT_IDX);
  auto &out = context.getOutput(SINGLE_INOUT_IDX);

  const float multiplier = readMultiplier(context);

  in.multiply(multiplier, out);
}

void ScalarMultiplyLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  bool is_prefill = !from || (to - from) > 1;
  if (skip_prefill && is_prefill)
    return;

  const float multiplier = readMultiplier(context);

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

    in_step.multiply(multiplier, out_step);

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
