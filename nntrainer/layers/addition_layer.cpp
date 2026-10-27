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

#include <cstring>
#include <fstream>

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
  TensorDim out_dim = context.getInputDimensions()[0];
  // W8A8/PERCH: produce FP32 here instead of requantizing back to QINT8.
  // Root cause (confirmed via on-device bisection 2026-08-11): a QINT8
  // tensor produced by AdditionLayer does not correctly bind to its
  // downstream consumer(s) through nntrainer's graph/tensor-pool -- a
  // conv or concat reading this layer's output sees stale/never-written
  // memory, even though THIS layer's own post-write view of hidden_ is
  // correct (verified byte-exact against golden). Conv2d's per-channel
  // GEMM path and concat's FP32-input branch both already auto-quantize a
  // genuine FP32 input on the fly (the same mechanism the network stem
  // uses), so emitting FP32 here sidesteps the bug entirely.
  if (out_dim.getDataType() == nntrainer::Tdatatype::QINT8)
    out_dim.setDataType(nntrainer::Tdatatype::FP32);
  context.setOutputDimensions({out_dim});
}

void AdditionLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);

  bool any_qint8_input = false;
  for (unsigned int idx = 0; idx < context.getNumInputs(); ++idx) {
    if (context.getInput(idx).getDataType() == nntrainer::Tdatatype::QINT8) {
      any_qint8_input = true;
      break;
    }
  }

  if (any_qint8_input && hidden_.getDataType() == nntrainer::Tdatatype::FP32) {
    // See finalize(): dequantize QINT8 inputs and sum in FP32 directly,
    // skipping the QINT8 output requantization that doesn't bind correctly.
    static const bool perch_asym_env =
      std::getenv("NNTR_W8A8_PERCH") != nullptr &&
      std::getenv("NNTR_W8A8_SYM") == nullptr;
    constexpr float kActOff = 0.27846455f;
    float *out = hidden_.getData<float>();
    const size_t n = hidden_.size();
    for (unsigned int idx = 0; idx < context.getNumInputs(); ++idx) {
      const Tensor &input_ = context.getInput(idx);
      if (input_.getDataType() == nntrainer::Tdatatype::QINT8) {
        const int8_t *q = input_.getData<int8_t>();
        const float sc = input_.getScale<float>()[0];
        if (!idx) {
          if (perch_asym_env)
            for (size_t i = 0; i < n; ++i)
              out[i] = ((float)q[i] + 128.f) * sc - kActOff;
          else
            for (size_t i = 0; i < n; ++i)
              out[i] = sc * (float)q[i];
        } else {
          if (perch_asym_env)
            for (size_t i = 0; i < n; ++i)
              out[i] += ((float)q[i] + 128.f) * sc - kActOff;
          else
            for (size_t i = 0; i < n; ++i)
              out[i] += sc * (float)q[i];
        }
      } else {
        const float *d = input_.getData<float>();
        if (!idx)
          std::copy(d, d + n, out);
        else
          for (size_t i = 0; i < n; ++i)
            out[i] += d[i];
      }
    }
  } else {
    for (unsigned int idx = 0; idx < context.getNumInputs(); ++idx) {
      const Tensor &input_ = context.getInput(idx);
      // The first operand copies into hidden, the rest accumulate. The active
      // backend's op table picks the residency path: the CPU table runs the
      // same host copy/add this used to call inline, an accelerator can keep
      // the residual stream in device memory.
      hidden_.getOps()->residual_op(hidden_, input_, /*accumulate=*/idx != 0);
    }
  }

  if (const char *dl = std::getenv("NNTR_CONCAT_DEBUG");
      dl && std::strstr(dl, context.getName().c_str()) &&
      hidden_.getDataType() == nntrainer::Tdatatype::QINT8) {
    const int8_t *q = hidden_.getData<int8_t>();
    const float sc = hidden_.getScale<float>()[0];
    int8_t qmin = 127, qmax = -128;
    for (size_t i = 0; i < hidden_.size(); ++i) {
      if (q[i] < qmin)
        qmin = q[i];
      if (q[i] > qmax)
        qmax = q[i];
    }
    std::cerr << "[CONCATDBG] " << context.getName()
              << " (producer) scale=" << sc << " qmin=" << (int)qmin
              << " qmax=" << (int)qmax << " n=" << hidden_.size()
              << " ptr=" << (const void *)q << "\n"
              << std::flush;
  }

  // Dump one named layer's real output (from inside the untruncated
  // production forward pass), for external bisection.
  if (const char *dl = std::getenv("NNTR_DUMP_LAYER");
      dl && context.getName() == dl) {
    const char *dp = std::getenv("NNTR_DUMP_PATH");
    if (dp) {
      std::vector<float> buf(hidden_.size());
      if (hidden_.getDataType() == nntrainer::Tdatatype::FP16) {
#ifdef ENABLE_FP16
        const _FP16 *d = hidden_.getData<_FP16>();
        for (unsigned int i = 0; i < hidden_.size(); ++i)
          buf[i] = (float)d[i];
#endif
      } else if (hidden_.getDataType() == nntrainer::Tdatatype::QINT8) {
        const int8_t *q = hidden_.getData<int8_t>();
        const float sc = hidden_.getScale<float>()[0];
        static const bool perch_asym_env =
          std::getenv("NNTR_W8A8_PERCH") != nullptr &&
          std::getenv("NNTR_W8A8_SYM") == nullptr;
        constexpr float kActOff = 0.27846455f;
        for (unsigned int i = 0; i < hidden_.size(); ++i)
          buf[i] = perch_asym_env ? ((float)q[i] + 128.f) * sc - kActOff
                                  : sc * (float)q[i];
      } else {
        const float *d = hidden_.getData<float>();
        std::copy(d, d + hidden_.size(), buf.begin());
      }
      std::ofstream of(dp, std::ios::binary);
      of.write(reinterpret_cast<const char *>(buf.data()),
               buf.size() * sizeof(float));
    }
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

    // The two-input join -- the residual stream's shape in every decoder --
    // can be one dispatch instead of a copy and an accumulate. Ask the op
    // table; a backend that cannot do it declines and the per-operand loop
    // below runs unchanged.
    if (context.getNumInputs() == 2) {
      const Tensor &i0 = context.getInput(0);
      const Tensor &i1 = context.getInput(1);
      TensorDim s0d = i0.getDim(), s1d = i1.getDim();
      const unsigned int f0 = s0d.getFeatureLen(), f1 = s1d.getFeatureLen();
      s0d.batch(1);
      s0d.height(to - from);
      s1d.batch(1);
      s1d.height(to - from);
      Tensor s0 =
        i0.getSharedDataTensor(s0d, static_cast<size_t>(b) * f0, true);
      Tensor s1 =
        i1.getSharedDataTensor(s1d, static_cast<size_t>(b) * f1, true);
      if (hidden_step.getOps()->residual_op2(hidden_step, s0, s1))
        continue;
    }

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
