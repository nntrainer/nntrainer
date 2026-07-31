// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   gate_up_layer.cpp
 * @date   29 July 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @brief  See gate_up_layer.h.
 *
 * Weight request order is UP then GATE, matching
 * Applications/CausalLM/models/transformer.cpp's createMlp comment:
 * "nntrainer binary stores mlp weights in up, gate order" - this layer must
 * preserve that exact byte order to stay compatible with existing .bin
 * files. Weight loading (Layer::read(), layer_devel.h) reads
 * context.getWeight(i) for i in [0, getNumWeights()) strictly in request
 * order, with no names involved, so request order here must match what two
 * separate FC layers (ffn_up created before ffn_gate) would have produced.
 */

#include <gate_up_layer.h>

#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum GateUpParams { Up, Gate };

GateUpLayer::GateUpLayer() :
  LayerImpl(), gate_up_props(props::UpUnit(), props::GateUnit()) {
  weight_idx.fill(std::numeric_limits<unsigned>::max());
}

void GateUpLayer::finalize(InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "GateUpLayer takes only one input";

  auto &weight_regularizer =
    std::get<props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<props::WeightRegularizerConstant>(*layer_impl_props);
  auto weight_initializer = props::InitializerInfo::Enum::NONE;
  auto &weight_decay = std::get<props::WeightDecay>(*layer_impl_props);

  const auto &up_unit = std::get<props::UpUnit>(gate_up_props).get();
  const auto &gate_unit = std::get<props::GateUnit>(gate_up_props).get();

  std::vector<TensorDim> output_dims(2);

  context.setEffDimFlagInputDimension(0, 0b1001);
  context.setDynDimFlagInputDimension(0, 0b1000);

  bool is_nchw = (context.getFormat() == Tformat::NCHW);
  auto const &in_dim = context.getInputDimensions()[0];

  /** Up out */
  output_dims[GateUpParams::Up] = in_dim;
  is_nchw ? output_dims[GateUpParams::Up].width(up_unit)
          : output_dims[GateUpParams::Up].channel(up_unit);
  output_dims[GateUpParams::Up].setTensorType(
    {context.getFormat(), context.getActivationDataType()});

  /** Gate out */
  output_dims[GateUpParams::Gate] = in_dim;
  is_nchw ? output_dims[GateUpParams::Gate].width(gate_unit)
          : output_dims[GateUpParams::Gate].channel(gate_unit);
  output_dims[GateUpParams::Gate].setTensorType(
    {context.getFormat(), context.getActivationDataType()});

  context.setOutputDimensions(output_dims);

  /** Up weight - requested first: matches ffn_up being created before
   * ffn_gate in the two-separate-layer version this replaces. */
  TensorDim weight_dim(
    1, is_nchw ? 1 : up_unit, is_nchw ? in_dim.width() : 1,
    is_nchw ? up_unit : in_dim.channel(),
    TensorDim::TensorType(context.getFormat(), context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);
  weight_idx[GateUpParams::Up] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "upweight", true);

  /** Gate weight - requested second. */
  weight_dim.width(gate_unit);
  weight_idx[GateUpParams::Gate] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "gateweight", true);
}

void GateUpLayer::exportTo(Exporter &exporter,
                          const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(gate_up_props, method, this);
}

void GateUpLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, gate_up_props);
  LayerImpl::setProperty(remain_props);
}

void GateUpLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &Upweight = context.getWeight(weight_idx[GateUpParams::Up]);
  Tensor &Gateweight = context.getWeight(weight_idx[GateUpParams::Gate]);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  Tensor &Uphidden_ = context.getOutput(GateUpParams::Up);
  Tensor &Gatehidden_ = context.getOutput(GateUpParams::Gate);

  std::vector<Tensor *> Weights({&Upweight, &Gateweight});
  std::vector<Tensor *> Outputs({&Uphidden_, &Gatehidden_});

  input_.dot(Weights, Outputs);
}

void GateUpLayer::incremental_forwarding(RunLayerContext &context,
                                        unsigned int from, unsigned int to,
                                        bool training) {
  Tensor &Upweight = context.getWeight(weight_idx[GateUpParams::Up]);
  Tensor &Gateweight = context.getWeight(weight_idx[GateUpParams::Gate]);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  Tensor &Uphidden_ = context.getOutput(GateUpParams::Up);
  Tensor &Gatehidden_ = context.getOutput(GateUpParams::Gate);

  TensorDim input_dim = input_.getDim();
  TensorDim input_step_dim = input_dim;
  input_step_dim.batch(1);
  input_step_dim.height(to - from);

  Tensor input_step = input_.getSharedDataTensor(input_step_dim, 0, true);

  TensorDim Uphidden_step_dim = Uphidden_.getDim();
  Uphidden_step_dim.batch(1);
  Uphidden_step_dim.height(to - from);
  Tensor Uphidden_step = Uphidden_.getSharedDataTensor(Uphidden_step_dim, 0, true);

  TensorDim Gatehidden_step_dim = Gatehidden_.getDim();
  Gatehidden_step_dim.batch(1);
  Gatehidden_step_dim.height(to - from);
  Tensor Gatehidden_step =
    Gatehidden_.getSharedDataTensor(Gatehidden_step_dim, 0, true);

  std::vector<Tensor *> Weights({&Upweight, &Gateweight});
  std::vector<Tensor *> Outputs({&Uphidden_step, &Gatehidden_step});

  input_step.dot(Weights, Outputs);
}

void GateUpLayer::calcDerivative(RunLayerContext &context) {
  throw std::runtime_error(
    "GateUpLayer::calcDerivative not supported (inference-only layer)");
}

void GateUpLayer::calcGradient(RunLayerContext &context) {
  throw std::runtime_error(
    "GateUpLayer::calcGradient not supported (inference-only layer)");
}

void GateUpLayer::updateTensorsByInputDimensions(
  RunLayerContext &context, std::vector<TensorDim> input_dimensions) {
  TensorDim input_dim = context.getInput(SINGLE_INOUT_IDX).getDim();
  TensorDim Updim = context.getOutput(GateUpParams::Up).getDim();
  TensorDim Gatedim = context.getOutput(GateUpParams::Gate).getDim();

  input_dim.height(input_dimensions[0].height());
  Updim.height(input_dimensions[0].height());
  Gatedim.height(input_dimensions[0].height());

  context.updateInput(SINGLE_INOUT_IDX, input_dim);
  context.updateOutput(GateUpParams::Up, Updim);
  context.updateOutput(GateUpParams::Gate, Gatedim);
}

} // namespace nntrainer
