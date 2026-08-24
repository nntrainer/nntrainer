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
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

#include <dlfcn.h>
#include <layer_context.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

// DSP ADD bridge: lazily dlopen libggml-hexagon.so and dlsym
// nntr_htp_bridge_add. Returns nullptr if unavailable.
using nntr_htp_bridge_add_fn =
  int (*)(const float * a, const float * b, float * out,
          unsigned int M, unsigned int N);

static nntr_htp_bridge_add_fn get_add_bridge() {
  static nntr_htp_bridge_add_fn fn = nullptr;
  static bool tried = false;
  if (!tried) {
    tried = true;
    void * lib = dlopen("libggml-hexagon.so", RTLD_NOW | RTLD_NOLOAD);
    if (!lib)
      lib = dlopen("libggml-hexagon.so", RTLD_LAZY);
    if (lib)
      fn = (nntr_htp_bridge_add_fn)dlsym(lib, "nntr_htp_bridge_add");
  }
  return fn;
}


void AdditionLayer::finalize(InitLayerContext &context) {
  if (!std::get<props::SkipPrefill>(add_props).empty())
    skip_prefill = std::get<props::SkipPrefill>(add_props).get();
  context.setOutputDimensions({context.getInputDimensions()[0]});
}

void AdditionLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);

  // For 2-input residual add, try DSP bridge (HTP_OP_ADD) first.
  // Falls back to CPU if the bridge is unavailable or fails.
  if (context.getNumInputs() == 2) {
    const Tensor &in0 = context.getInput(0);
    const Tensor &in1 = context.getInput(1);
    auto * bridge = get_add_bridge();
    if (bridge &&
        context.getComputeEngineType() == ml::train::LayerComputeEngine::CDSP &&
        !getenv("NNTR_HEXAGON_NO_ELEM_OPS")) {

      const float * a = in0.getData();
      const float * b = in1.getData();
      float * out = hidden_.getData();
      unsigned int M = in0.batch() * in0.height();
      unsigned int N = in0.width() * in0.channel();
      if (bridge(a, b, out, M, N) == 0) {
        return;
      }
    }
    // CPU fallback
  }

  /** @todo check possibility for in-place of addition layer */
  for (unsigned int idx = 0; idx < context.getNumInputs(); ++idx) {
    const Tensor &input_ = context.getInput(idx);
    if (!idx) {
      hidden_.copy(input_);
    } else {
      hidden_.add_i(input_);
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

  // Try the DSP ADD bridge (HTP_OP_ADD) for the common 2-input residual-add
  // case. This is the function nntrainer's step-based prefill/decode loop
  // actually calls - forwarding() above (which already had this dispatch)
  // is never reached from that path, so the bridge call has to live here too,
  // not just there. Prefill-only, matching every other NPU dispatch gate in
  // this codebase: decode is a single row, all round-trip cost, no compute
  // to amortize it against. Primary gate is compute_engine (see rms_norm.cpp
  // for the same reasoning); NNTR_HEXAGON_NO_ELEM_OPS remains a manual
  // kill-switch on top.
  bool try_dsp = is_prefill && context.getNumInputs() == 2 &&
                context.getComputeEngineType() ==
                  ml::train::LayerComputeEngine::CDSP &&
                hidden_.getDataType() == ml::train::TensorDim::DataType::FP32 &&
                !getenv("NNTR_HEXAGON_NO_ELEM_OPS");
  auto *bridge = try_dsp ? get_add_bridge() : nullptr;

  for (unsigned int b = 0; b < hidden_.batch(); ++b) {
    Tensor hidden_step = hidden_.getSharedDataTensor(
      hidden_step_dim, b * hidden_dim.getFeatureLen(), true);

    if (bridge) {
      const Tensor &in0 = context.getInput(0);
      const Tensor &in1 = context.getInput(1);
      Tensor in0_step = in0.getSharedDataTensor(
        hidden_step_dim, b * hidden_dim.getFeatureLen(), true);
      Tensor in1_step = in1.getSharedDataTensor(
        hidden_step_dim, b * hidden_dim.getFeatureLen(), true);

      unsigned int M = to - from;
      unsigned int N = hidden_step_dim.width() * hidden_step_dim.channel();
      if (bridge(in0_step.getData<float>(), in1_step.getData<float>(),
                hidden_step.getData<float>(), M, N) == 0) {
        continue;
      }
      // DSP call failed for this batch slice - fall through to CPU below.
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
      if (!idx) {
        hidden_step.copy(input_step);
      } else {
        hidden_step.add_i(input_step);
      }
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
