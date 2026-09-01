// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   logit_softcapping.cpp
 * @date   8 April 2026
 * @brief  Implementation of final logit softcapping layer
 * @see    https://github.com/nntrainer/nntrainer
 * @author Joonseok Oh <jrock.oh@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "logit_softcapping.h"

#include <algorithm>
#include <stdexcept>

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
#include <cuda_context_manager.h>
#include <cuda_elementwise.h>
#include <cuda_runtime.h>
#include <cuda_stream_manager.h>
#endif

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void LogitSoftCappingLayer::finalize(nntrainer::InitLayerContext &context) {
  std::vector<nntrainer::TensorDim> dim = context.getInputDimensions();
  context.setOutputDimensions(dim);
  if (!std::get<nntrainer::props::SkipPrefill>(logit_softcap_props).empty())
    skip_prefill =
      std::get<nntrainer::props::SkipPrefill>(logit_softcap_props).get();

  auto activation =
    std::get<props::LogitSoftcapActivation>(logit_softcap_props).get();
  auto softcap = std::get<props::SoftcapValue>(logit_softcap_props).get();

  NNTR_THROW_IF(softcap <= 0.0f, std::invalid_argument)
    << "[logit_softcapping] softcap_value must be > 0";
  NNTR_THROW_IF(activation == nntrainer::ActivationType::ACT_NONE,
                std::invalid_argument)
    << "[logit_softcapping] activation_type must be set";

  if (context.getActivationDataType() == nntrainer::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    acti_func.setActiFunc<_FP16>(activation);
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  } else if (context.getActivationDataType() ==
             nntrainer::TensorDim::DataType::FP32) {
    acti_func.setActiFunc<float>(activation);
  }
}

void LogitSoftCappingLayer::forwarding(nntrainer::RunLayerContext &context,
                                       bool training) {
  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  out.copyData(in);

  applyOnRange(context, 0, in.height());
}

void LogitSoftCappingLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  bool is_prefill = !from || (to - from) > 1;
  if (skip_prefill && is_prefill)
    return;

  applyOnRange(context, from, to);
}

void LogitSoftCappingLayer::applyOnRange(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to) {
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
  // Terminal drain for the selective-sync (NNTR_CUDA_ASYNC) path: this is the
  // first host read of the lm_head logits, so the one-per-token GPU pipeline
  // drains here. A no-op in default mode (every GPU op already drained).
  // cuda runs only: StreamManager::Global() would CREATE the CUDA context.
  if (nntrainer::cuda::engine_selected())
    nntrainer::cuda::StreamManager::Global().finish();
#endif
  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);

  const unsigned int apply_rows =
    std::get<props::ApplyRows>(logit_softcap_props).get();
  const float softcap =
    std::get<props::SoftcapValue>(logit_softcap_props).get();

  if (apply_rows > (to - from)) {
    throw std::invalid_argument(
      "[logit_softcapping] apply_rows cannot exceed " +
      std::to_string(to - from));
  }

  const auto input_dim = in.getDim();

  ml::train::TensorDim in_chunk_dim = input_dim;
  ml::train::TensorDim out_chunk_dim = input_dim;
  in_chunk_dim.batch(1);
  out_chunk_dim.batch(1);
  in_chunk_dim.height(apply_rows);
  out_chunk_dim.height(apply_rows);

  const unsigned int num_channels = input_dim.channel();
  const unsigned int batch_size = input_dim.batch();

  for (unsigned int b = 0; b < batch_size; ++b) {
    for (unsigned int c = 0; c < num_channels; ++c) {
      nntrainer::Tensor in_chunk =
        in.getSharedDataTensor(in_chunk_dim, 0, true);
      nntrainer::Tensor out_chunk =
        out.getSharedDataTensor(out_chunk_dim, 0, true);
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1 && defined(ENABLE_FP16)
      // Device-only activation pool: the logits are real device memory; the
      // host Tensor ops below would fault. softcap = cap*tanh(x/cap) in one GPU
      // kernel (mirrors the OpenCL GPU path).
      if (in_chunk.getDataType() == nntrainer::TensorDim::DataType::FP16) {
        unsigned short *ip =
          reinterpret_cast<unsigned short *>(in_chunk.getData<_FP16>());
        unsigned short *op =
          reinterpret_cast<unsigned short *>(out_chunk.getData<_FP16>());
        cudaPointerAttributes pa{};
        // Accept Managed (UVM) too, not just Device: on an integrated GPU the
        // activation pool is cudaMallocManaged, so a Device-only gate sends the
        // softcap to the host loop below -- which, inside a CUDA-graph capture,
        // reads the not-yet-run lm_head logits (stale) and is itself not
        // captured. Managed pointers run the GPU kernel fine.
        if (nntrainer::cuda::engine_selected() &&
            cudaPointerGetAttributes(&pa, ip) == cudaSuccess &&
            (pa.type == cudaMemoryTypeDevice ||
             pa.type == cudaMemoryTypeManaged) &&
            nntrainer::cuda::cuda_softcap_fp16(
              ip, op, (unsigned int)in_chunk.size(), softcap)) {
          cudaGetLastError();
          continue;
        }
        cudaGetLastError();
      }
#endif
      out_chunk.copyData(in_chunk);

      in_chunk.multiply(1.0f / softcap, out_chunk);
      acti_func.run_fn(out_chunk, out_chunk);
      out_chunk.multiply(softcap, out_chunk);
    }
  }
}

void LogitSoftCappingLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  context.updateInput(SINGLE_INOUT_IDX, input_dimensions[0]);
  context.updateOutput(SINGLE_INOUT_IDX, input_dimensions[0]);
}

void LogitSoftCappingLayer::calcDerivative(
  nntrainer::RunLayerContext &context) {
  std::throw_with_nested(std::runtime_error("Training is not supported yet."));
}

} // namespace nntrainer
