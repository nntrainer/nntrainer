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
#include <cstdlib>
#include <env_compat.h>
#include <iostream>

#include "scalar_multiply.h"

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
#include <cuda_context_manager.h>
#include <cuda_elementwise.h>
#include <cuda_runtime.h>
#include <cuda_stream_manager.h>
#endif

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
    // @note The scalar is an unquantized weight read straight out of the model
    // .bin, whose layout has no per-tensor dtype: NeuralNetwork::load() derives
    // every weight's file offset by accumulating getMemoryBytes() over the
    // graph. The request must therefore reproduce the dtype the exporting graph
    // used -- getWeightDataType() -- or this scalar is misread *and* every
    // weight after it is read at the wrong offset. Upstream pinned this to FP32
    // because its multiply site did an unconditional getValue<float> on the
    // 2-byte cell; incremental_forwarding() below instead reads the scalar
    // dtype-aware (FP16 or FP32), which fixes that hazard without changing the
    // on-disk contract.
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

/**
 * @brief Resolve the multiplier for one step, memoizing a weight-borne one.
 *
 * `use_weight` puts the multiplier in a one-element STATIC weight, and reading
 * it is a HOST read. On the GPU lanes the weight pool is Unified Memory
 * (cudaMallocManaged / SVM), so that 2-byte host read migrates the whole 4 KiB
 * page out of device memory -- and the neighbouring RMSNorm gammas packed into
 * the same page are then faulted straight back device-side by the next block's
 * norm kernel. One device->host->device page round trip per scalar_multiply per
 * step: measured at 55 4 KiB UVM migrations and about 0.13 ms per token on a
 * 35-block model, plus the fault stalls hidden behind them, for a value that
 * never changes.
 *
 * So read it once per weight BUFFER and keep it. The key is the weight's data
 * address rather than a done-flag: deallocateTensors()+allocateTensors() (the
 * reset / KV-resume path) hands the layer a different buffer, and comparing
 * addresses re-arms the read there instead of serving a stale scalar. Taking
 * the address does not touch the page.
 *
 * NNTR_CUDA_UVM_PIN=0 restores the per-step read (A/B, or a caller that
 * rewrites the weight in place between steps).
 *
 * @param context run context holding the (optional) scalar weight
 * @return the scalar to multiply the input by
 */
float ScalarMultiplyLayer::readMultiplier(nntrainer::RunLayerContext &context) {
  if (!std::get<props::UseWeight>(scalar_multiply_props).get())
    return std::get<props::ScalarMultiplier>(scalar_multiply_props).get();

  nntrainer::Tensor &weight = context.getWeight(wt_idx[0]);
  // Value-checked, not presence-checked: the memoisation is the default and
  // only an explicit 0 turns it off.
  static const bool memo_on = []() {
    const char *e = std::getenv("NNTR_CUDA_UVM_PIN");
    return !(e != nullptr && e[0] == '0');
  }();
  const void *addr = weight.getData<char>();
  if (memo_on && addr != nullptr && addr == memo_weight_addr)
    return memo_multiplier;

#ifdef ENABLE_FP16
  // dtype-aware scalar read: getValue<float> on an FP16 weight reads 4 bytes
  // from the 2-byte cell (garbage). The per-layer scalar / q_scale /
  // model_proj_scale weights are FP16 in QINT4-FP16 models -> reading them as
  // float gave a huge multiplier and overflowed the residual to +Inf.
  const float m = (weight.getDataType() == ml::train::TensorDim::DataType::FP16)
                    ? static_cast<float>(weight.getValue<_FP16>(0, 0, 0, 0))
                    : weight.getValue<float>(0, 0, 0, 0);
#else
  const float m = weight.getValue<float>(0, 0, 0, 0);
#endif
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

    bool done = false;
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1 && defined(ENABLE_FP16)
    if (in_step.getDataType() == ml::train::TensorDim::DataType::FP16) {
      static const bool gpu = nntr_env_on("NNTR_CUDA_ELTWISE");
      if (gpu) {
        auto *ip =
          reinterpret_cast<const unsigned short *>(in_step.getData<_FP16>());
        auto *op =
          reinterpret_cast<unsigned short *>(out_step.getData<_FP16>());
        const bool dev = nntrainer::cuda::dev_accessible(ip);
        if (dev && nntrainer::cuda::cuda_scalar_mul_fp16(
                     ip, op, (unsigned int)in_step.size(), multiplier))
          done = true;
      }
    }
#endif
    if (!done) {
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
      // Host multiply() reads the GPU-produced UVM input on the CPU; sync first
      // in async mode (no-op in default sync mode).
      nntrainer::cuda::drain_if_async();
#endif
      in_step.multiply(multiplier, out_step);
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
