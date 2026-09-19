// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   addition_layer.cpp
 * @date   30 July 2020
 * @see    https://github.com/nntrainer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Addition Layer Class for Neural Network
 *
 */

#include <addition_layer.h>
#include <env_compat.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <tensor.h>
#include <util_func.h>

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
#include <cuda_context_manager.h>
#include <cuda_elementwise.h>
#include <cuda_runtime.h>
#include <cuda_stream_manager.h>
#endif
#include <layer_context.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void AdditionLayer::finalize(InitLayerContext &context) {
  if (!std::get<props::SkipPrefill>(add_props).empty())
    skip_prefill = std::get<props::SkipPrefill>(add_props).get();
  context.setOutputDimensions({context.getInputDimensions()[0]});
}

void AdditionLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);

  /** @todo check possibility for in-place of addition layer */
  for (unsigned int idx = 0; idx < context.getNumInputs(); ++idx) {
    const Tensor &input_ = context.getInput(idx);
    // The first operand copies into hidden, the rest accumulate. The active
    // backend's op table picks the residency path: the CPU table runs the same
    // host copy/add this used to call inline, an accelerator can keep the
    // residual stream in device memory.
    hidden_.getOps()->residual_op(hidden_, input_, /*accumulate=*/idx != 0);
  }
}

void AdditionLayer::incremental_forwarding(RunLayerContext &context,
                                           unsigned int from, unsigned int to,
                                           bool training) {
  bool is_prefill = !from || (to - from) > 1;
  if (skip_prefill && is_prefill)
    return;

  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  TensorDim hidden_dim = hidden_.getDim();
  TensorDim hidden_step_dim = hidden_dim;

  hidden_step_dim.batch(1);
  hidden_step_dim.height(to - from);

  for (unsigned int b = 0; b < hidden_.batch(); ++b) {
    Tensor hidden_step = hidden_.getSharedDataTensor(
      hidden_step_dim, b * hidden_dim.getFeatureLen(), true);

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1 && defined(ENABLE_FP16)
    // GPU residual add (the common 2-input fp16 device case): out = in0 + in1
    // in one kernel, keeping the residual on-device. Opt-in
    // (NNTR_CUDA_ELTWISE).
    if (context.getNumInputs() == 2 &&
        hidden_.getDataType() == ml::train::TensorDim::DataType::FP16) {
      static const bool gpu = nntr_env_on("NNTR_CUDA_ELTWISE");
      if (gpu) {
        const Tensor &i0 = context.getInput(0);
        const Tensor &i1 = context.getInput(1);
        TensorDim sd0 = i0.getDim(), sd1 = i1.getDim();
        sd0.batch(1);
        sd0.height(to - from);
        sd1.batch(1);
        sd1.height(to - from);
        Tensor s0 =
          i0.getSharedDataTensor(sd0, b * i0.getDim().getFeatureLen(), true);
        Tensor s1 =
          i1.getSharedDataTensor(sd1, b * i1.getDim().getFeatureLen(), true);
        auto *a = reinterpret_cast<const unsigned short *>(s0.getData<_FP16>());
        auto *bb =
          reinterpret_cast<const unsigned short *>(s1.getData<_FP16>());
        auto *o =
          reinterpret_cast<unsigned short *>(hidden_step.getData<_FP16>());
        if (nntrainer::cuda::dev_accessible(a) &&
            cuda::cuda_add_fp16(a, bb, o, (unsigned int)hidden_step.size()))
          continue;
      }
    }
#endif

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
    // The host residual path below reads GPU-produced UVM inputs on the CPU
    // (copy()/add_i()). In async mode the producing kernels have not drained,
    // so sync first; no-op in sync mode. Without this the residual stream is
    // read while the producing kernel is still in flight -- fluent, wrong
    // output that a debug sync makes disappear.
    nntrainer::cuda::drain_if_async();
#endif

    /** @todo check possibility for in-place of addition layer */
    for (unsigned int idx = 0; idx < context.getNumInputs(); ++idx) {
      const Tensor &input_ = context.getInput(idx);
      TensorDim input_dim = input_.getDim();

      TensorDim input_step_dim = input_dim;
      input_step_dim.batch(1);
      input_step_dim.height(to - from);

      Tensor input_step = input_.getSharedDataTensor(
        input_step_dim, b * input_dim.getFeatureLen(), true);
      hidden_step.getOps()->residual_op(hidden_step, input_step,
                                        /*accumulate=*/idx != 0);
    }
  }
}

void AdditionLayer::calcDerivative(RunLayerContext &context) {

  for (unsigned int idx = 0; idx < context.getNumInputs(); ++idx) {
    /**
     * TODO: replace this with tensor assignment during optimization.
     * Tensor assignment needs to make sure that the previous connected layers
     * are not inplace
     */
    context.getOutgoingDerivative(idx).copy(
      context.getIncomingDerivative(SINGLE_INOUT_IDX));
  }
}

void AdditionLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, add_props);
  if (!remain_props.empty()) {
    std::string msg = "[AdditionLayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}

void AdditionLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  for (size_t i = 0; i < context.getNumInputs(); ++i) {
    context.updateInput(i, input_dimensions[0]);
  }
  context.updateOutput(SINGLE_INOUT_IDX, input_dimensions[0]);
}

} /* namespace nntrainer */
