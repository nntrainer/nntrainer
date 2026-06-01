// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   lm_head.cpp
 * @date   16 Jan 2026
 * @brief  This is lmhead layer
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include "lm_head.h"
#include <cpu_backend.h>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <tensor.h>
#include <tensor_dim.h>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum LmHeadParams {
  weight,
  bias,
};

LmHeadLayer::LmHeadLayer() : LayerImpl(), lmhead_props(props::Unit()) {
  weight_idx.fill(std::numeric_limits<unsigned>::max());
}

void LmHeadLayer::finalize(InitLayerContext &context) {
  auto &weight_regularizer =
    std::get<props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<props::WeightRegularizerConstant>(*layer_impl_props);
  auto weight_initializer = Initializer::ONES;
  auto &weight_decay = std::get<props::WeightDecay>(*layer_impl_props);
  auto &bias_decay = std::get<props::BiasDecay>(*layer_impl_props);
  auto &bias_initializer = std::get<props::BiasInitializer>(*layer_impl_props);
  auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);

  auto unit = std::get<props::Unit>(lmhead_props).get();

  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "lm head layer takes only one input";

  std::vector<ml::train::TensorDim> output_dims(1);

  /// @todo fc actaully supports multidimensions.
  /// EffDimFlag shouldn't be fixed like this.
  context.setEffDimFlagInputDimension(0, 0b1001);
  context.setDynDimFlagInputDimension(0, 0b1000);
  bool is_nchw = (context.getFormat() == Tformat::NCHW);

  /** set output dimensions */
  ///@note lm_head's output dimension: width (or channel for NHWC) is vocab size; height follows input height
  auto const &in_dim = context.getInputDimensions()[0];
  output_dims[0] = in_dim;
  if (is_nchw)
    output_dims[0].width(unit);
  else
    output_dims[0].channel(unit);
  output_dims[0].height(in_dim.height());

  output_dims[0].setTensorType(
    {context.getFormat(), context.getActivationDataType()});

  context.setOutputDimensions(output_dims);

  /** set weight specifications */
  ml::train::TensorDim bias_dim(
    1, is_nchw ? 1 : unit, 1, is_nchw ? unit : 1,
    ml::train::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()),
    is_nchw ? 0b0001 : 0b0100);

  ///@note LMHead layer's tensor dim is transposed dim of user-defined
  /// dim
  /// so it can reuse embedding layer.
  ml::train::TensorDim weight_dim(
    1, is_nchw ? 1 : unit, is_nchw ? in_dim.width() : 1,
    is_nchw ? unit : in_dim.channel(),
    ml::train::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);

  weight_idx[LmHeadParams::weight] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "weight", true);

  if (disable_bias.empty() || disable_bias.get() == false) {
    weight_idx[LmHeadParams::bias] =
      context.requestWeight(bias_dim, bias_initializer, WeightRegularizer::NONE,
                            1.0f, bias_decay, "bias", true);
  }
}

void LmHeadLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, lmhead_props);
  LayerImpl::setProperty(remain_props);
}

void LmHeadLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &weight = context.getWeight(weight_idx[LmHeadParams::weight]);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);

  input_.dot(weight, hidden_, false, false);

  if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    Tensor &bias = context.getWeight(weight_idx[LmHeadParams::bias]);
    hidden_.add_i(bias);
  }
}

void LmHeadLayer::incremental_forwarding(RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) {

  Tensor weight = context.getWeight(weight_idx[LmHeadParams::weight]);

  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);

  ml::train::TensorDim input_dim = input_.getDim();
  ml::train::TensorDim hidden_dim = hidden_.getDim();

  ml::train::TensorDim input_step_dim = input_dim;
  ml::train::TensorDim hidden_step_dim = hidden_dim;

  input_step_dim.batch(1);
  input_step_dim.height(1);
  hidden_step_dim.batch(1);

  unsigned int b_size = input_dim.batch();

  for (unsigned int b = 0; b < b_size; ++b) {
    // Use local step index (to - from - 1) for incremental decoding
    // In incremental mode, input/hidden contain only the current step window
    Tensor input_step = input_.getSharedDataTensor(
      input_step_dim,
      b * input_dim.getFeatureLen() + (to - from - 1) * input_.width(), true);
    Tensor hidden_step = hidden_.getSharedDataTensor(
      hidden_step_dim, b * hidden_dim.getFeatureLen(), true);

    input_step.dot(weight, hidden_step, false, false);

    if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
        disable_bias.empty() || disable_bias.get() == false) {
      Tensor &bias = context.getWeight(weight_idx[LmHeadParams::bias]);
      hidden_step.add_i(bias);
    }
  }
}

void LmHeadLayer::calcDerivative(RunLayerContext &context) {
  Tensor weight = context.getWeight(weight_idx[LmHeadParams::weight]);
  Tensor &dx = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  const Tensor &dy = context.getIncomingDerivative(SINGLE_INOUT_IDX);

  // dx = dy . weight^T
  dy.dot(weight, dx, false, true);
}

void LmHeadLayer::calcGradient(RunLayerContext &context) {
  Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  const Tensor &dy = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &dweight = context.getWeightGrad(weight_idx[LmHeadParams::weight]);

  // dweight = in^T . dy  (accumulate correctly across multiple backward passes)
  in.dot_deriv_wrt_2(dweight, dy, false, false,
                     !context.isGradientFirstAccess(weight_idx[LmHeadParams::weight]));

  if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    Tensor &dbias = context.getWeightGrad(weight_idx[LmHeadParams::bias]);
    if (context.isGradientFirstAccess(weight_idx[LmHeadParams::bias])) {
      dy.sum({0, 1, 2}, dbias);
    } else {
      Tensor t = dy.sum({0, 1, 2});
      dbias.add_i(t);
    }
  }
}

void LmHeadLayer::exportTo(Exporter &exporter,
                           const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(lmhead_props, method, this);
}

void LmHeadLayer::updateTensorsByInputDimensions(
  RunLayerContext &context, std::vector<TensorDim> input_dimensions) {
  TensorDim in_dim = context.getInput(SINGLE_INOUT_IDX).getDim();
  unsigned int height = input_dimensions[0].height();

  in_dim.height(height);
  context.updateInput(SINGLE_INOUT_IDX, in_dim);

  TensorDim out_dim = context.getOutput(SINGLE_INOUT_IDX).getDim();
  out_dim.height(height);
  context.updateOutput(SINGLE_INOUT_IDX, out_dim);
}

} // namespace nntrainer
