// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   activation_layer.cpp
 * @date   17 June 2020
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Activation Layer Class for Neural Network
 *
 */

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <activation_layer.h>
#include <common_properties.h>
#include <cpu_backend.h>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <tensor.h>
#include <tensor_wrap_specs.h>
#include <util_func.h>

namespace nntrainer {
ActivationLayer::ActivationLayer() :
  Layer(),
  activation_props(new PropTypes(props::Activation(), props::SkipPrefill())) {
  acti_func.setActiFunc(ActivationType::ACT_NONE);
}

static constexpr size_t SINGLE_INOUT_IDX = 0;

void ActivationLayer::finalize(InitLayerContext &context) {
  auto &act = std::get<props::Activation>(*activation_props);
  if (!std::get<props::SkipPrefill>(*activation_props).empty())
    skip_prefill = std::get<props::SkipPrefill>(*activation_props).get();
  NNTR_THROW_IF(act.empty(), std::invalid_argument)
    << "activation has not been set!";
  act_type_int = (int)act.get();
  if (context.getActivationDataType() == TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    acti_func.setActiFunc<_FP16>(act.get());
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  } else if (context.getActivationDataType() == TensorDim::DataType::FP32) {
    acti_func.setActiFunc<float>(act.get());
  }

  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "activation layer, " << context.getName()
    << "requires exactly one input, but given: " << context.getNumInputs()
    << ", check graph connection if it is correct";

  /// @todo for only certain types of activation needs lifespan of
  /// forward_derivative order
  std::vector<VarGradSpecV2> out_specs;
  out_specs.push_back(
    InitLayerContext::outSpec(context.getInputDimensions()[0], "out",
                              TensorLifespan::FORWARD_DERIV_LIFESPAN));
  context.requestOutputs(std::move(out_specs));
  acti_func.setInPlace(context.getInPlace());
}

void ActivationLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  // W8A8 int8-resident GELU pass-through (NNTR_W8A8): see geluQint8.
  if (std::getenv("NNTR_W8A8") != nullptr && geluQint8(input_, hidden_))
    return;

  // Backend-neutral whole-op dispatch: one call, no preprocessor branch, no
  // backend name. The CPU table delegates every non-gelu mode back to this very
  // same ActiFunc, so each existing mode stays value-identical; acti_func is
  // kept for the backward path, which has no whole-op yet.
  const TensorDim in_dim = input_.getDim();
  input_.getOps()->activation(
    input_, hidden_, act_type_int,
    in_dim.batch() * in_dim.channel() * in_dim.height(), /*row_offset=*/0);
}

bool ActivationLayer::geluQint8(Tensor const &input_, Tensor &hidden_) {
  if (input_.getDataType() != Tdatatype::QINT8)
    return false;
  // Only GELU is reached on an int8 edge today (CED's ffn_gelu). If another
  // activation appears here, the caller falls back to acti_func.
  auto &act = std::get<props::Activation>(*activation_props);
  if (act.empty() || act.get() != ActivationType::ACT_GELU)
    return false;

  const unsigned int n = input_.size();
  const int8_t *qi = input_.getData<int8_t>();
  const float in_scale = input_.getScale<float>()[0];
  static thread_local std::vector<float> fbuf, fbuf_out;
  fbuf.resize(n);
  fbuf_out.resize(n);
  float *fp = fbuf.data();
  for (unsigned int i = 0; i < n; ++i)
    fp[i] = (float)qi[i] * in_scale;
  // Dequantize via the inline per-tensor scale -> FP32, apply exact-erf GELU
  // (matching ACT_GELU's float path), then requantize symmetric (amax/127 --
  // GELU output spans both signs, so the SiLU-specific affine offset does not
  // apply) and write the new scale. The output dtype inherits QINT8.
  gelu_v2(n, fp, fbuf_out.data());
  float *fo = fbuf_out.data();
  float amax = 0.f;
  for (unsigned int i = 0; i < n; ++i)
    amax = std::max(amax, std::fabs(fo[i]));
  const float sc = amax > 0.f ? amax / 127.f : 1.f;
  const float inv = amax > 0.f ? 127.f / amax : 0.f;
  int8_t *qo = hidden_.getData<int8_t>();
  for (unsigned int i = 0; i < n; ++i)
    qo[i] = (int8_t)std::max(-128.f, std::min(127.f, std::round(fo[i] * inv)));
  hidden_.getScale<float>()[0] = sc;
  return true;
}

void ActivationLayer::incremental_forwarding(RunLayerContext &context,
                                             unsigned int from, unsigned int to,
                                             bool training) {
  (void)training;
  bool is_prefill = !from || (to - from) > 1;
  if (skip_prefill && is_prefill)
    return;

  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  TensorDim input_dim = input_.getDim();

  // The former per-batch getSharedDataTensor(step_dim, b * featureLen) loop,
  // expressed as (active_rows, row_offset) numbers instead of views, because a
  // view does not carry the residency state a backend op reads. Rows are
  // (channel, height) flattened within a batch, the window starts at the batch
  // base, and a height == 1 broadcast shape keeps its single row -- exactly
  // what the step-dim guard did. in/out share shape: finalize() sets the output
  // dims to the input dims.
  const unsigned int rows_per_batch =
    input_dim.channel() * (input_dim.height() > 1 ? (to - from) : 1u);
  const unsigned int rows_in_batch = input_dim.channel() * input_dim.height();

  for (unsigned int b = 0; b < hidden_.batch(); ++b) {
    // W8A8 int8-resident GELU pass-through (NNTR_W8A8): see geluQint8. The
    // step views exist only for this check; the generic path below works on
    // (rows, offset) numbers so a backend op can read residency state.
    if (std::getenv("NNTR_W8A8") != nullptr) {
      TensorDim in_step_dim = input_dim;
      in_step_dim.batch(1);
      in_step_dim.height(input_dim.height() > 1 ? (to - from) : 1u);
      TensorDim out_step_dim = hidden_.getDim();
      out_step_dim.batch(1);
      out_step_dim.height(in_step_dim.height());
      Tensor input_step = input_.getSharedDataTensor(
        in_step_dim, b * input_dim.getFeatureLen(), true);
      Tensor hidden_step = hidden_.getSharedDataTensor(
        out_step_dim, b * hidden_.getDim().getFeatureLen(), true);
      if (geluQint8(input_step, hidden_step))
        continue;
    }
    input_.getOps()->activation(input_, hidden_, act_type_int, rows_per_batch,
                                b * rows_in_batch);
  }
}

void ActivationLayer::calcDerivative(RunLayerContext &context) {
  const Tensor &deriv = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &ret = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  Tensor &out = context.getOutput(SINGLE_INOUT_IDX);

  acti_func.run_prime_fn(in, out, ret, deriv);
}

void ActivationLayer::exportTo(Exporter &exporter,
                               const ml::train::ExportMethods &method) const {
  exporter.saveResult(*activation_props, method, this);
}

void ActivationLayer::setProperty(const std::vector<std::string> &values) {
  auto left = loadProperties(values, *activation_props);
  NNTR_THROW_IF(!left.empty(), std::invalid_argument)
    << "Failed to set property";

  auto &act = std::get<props::Activation>(*activation_props);
  if (!act.empty()) {
    acti_func.setActiFunc(act.get());
    act_type_int = (int)act.get();
  }
}

}; // namespace nntrainer
