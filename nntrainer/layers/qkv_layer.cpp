/**
 * Copyright (C) 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * @file	qkv_layer.cpp
 * @date	14 May 2020
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Eunju Yang <ej.yang@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <qkv_layer.h>

#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum QKVParams { Q, K, V };

QKVLayer::QKVLayer() :
  LayerImpl(), qkv_props(props::QUnit(), props::KUnit(), props::VUnit()) {
  weight_idx.fill(std::numeric_limits<unsigned>::max());
}

void QKVLayer::finalize(InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "QKVLayer takes only one input";

  auto &weight_regularizer =
    std::get<props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<props::WeightRegularizerConstant>(*layer_impl_props);
  auto weight_initializer = props::InitializerInfo::Enum::NONE;
  auto &weight_decay = std::get<props::WeightDecay>(*layer_impl_props);

  const auto &q_unit = std::get<props::QUnit>(qkv_props).get();
  const auto &k_unit = std::get<props::KUnit>(qkv_props).get();
  const auto &v_unit = std::get<props::VUnit>(qkv_props).get();

  std::vector<TensorDim> output_dims(3);

  /// @todo fc actaully supports multidimensions. EffDimFlag shouldn't be fixed
  /// like this.
  context.setEffDimFlagInputDimension(0, 0b1001);
  context.setDynDimFlagInputDimension(0, 0b1000);

  bool is_nchw = (context.getFormat() == Tformat::NCHW);
  /** set output dimensions */
  auto const &in_dim = context.getInputDimensions()[0];

  /** Q out */
  output_dims[QKVParams::Q] = in_dim;
  is_nchw ? output_dims[QKVParams::Q].width(q_unit)
          : output_dims[QKVParams::Q].channel(q_unit);
  output_dims[QKVParams::Q].setTensorType(
    {context.getFormat(), context.getActivationDataType()});

  /** K out */
  output_dims[QKVParams::K] = in_dim;
  is_nchw ? output_dims[QKVParams::K].width(k_unit)
          : output_dims[QKVParams::K].channel(k_unit);
  output_dims[QKVParams::K].setTensorType(
    {context.getFormat(), context.getActivationDataType()});

  /** V out */
  output_dims[QKVParams::V] = in_dim;
  is_nchw ? output_dims[QKVParams::V].width(v_unit)
          : output_dims[QKVParams::V].channel(v_unit);
  output_dims[QKVParams::V].setTensorType(
    {context.getFormat(), context.getActivationDataType()});

  context.setOutputDimensions(output_dims);

  /** Q - requested first: matches wq being created before wk/wv in the
   * three-separate-layer version this replaces (weight loading reads
   * context.getWeight(i) strictly in request order, see Layer::read in
   * layer_devel.h - request order here must be preserved for .bin
   * compatibility). */
  TensorDim weight_dim(
    1, is_nchw ? 1 : q_unit, is_nchw ? in_dim.width() : 1,
    is_nchw ? q_unit : in_dim.channel(),
    TensorDim::TensorType(context.getFormat(), context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);
  weight_idx[QKVParams::Q] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "qweight", true);

  /** K - requested second. */
  weight_dim.width(k_unit);
  weight_idx[QKVParams::K] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "kweight", true);

  /** V - requested third. */
  weight_dim.width(v_unit);
  weight_idx[QKVParams::V] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "vweight", true);
}

void QKVLayer::exportTo(Exporter &exporter,
                        const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(qkv_props, method, this);
}

void QKVLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, qkv_props);
  LayerImpl::setProperty(remain_props);
}

void QKVLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &Qweight = context.getWeight(weight_idx[QKVParams::Q]);
  Tensor &Kweight = context.getWeight(weight_idx[QKVParams::K]);
  Tensor &Vweight = context.getWeight(weight_idx[QKVParams::V]);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  Tensor &Qhidden_ = context.getOutput(QKVParams::Q);
  Tensor &Khidden_ = context.getOutput(QKVParams::K);
  Tensor &Vhidden_ = context.getOutput(QKVParams::V);

  std::vector<Tensor *> Weights({&Qweight, &Kweight, &Vweight});
  std::vector<Tensor *> Outputs({&Qhidden_, &Khidden_, &Vhidden_});

  input_.dot(Weights, Outputs);
}

void QKVLayer::incremental_forwarding(RunLayerContext &context,
                                     unsigned int from, unsigned int to,
                                     bool training) {
  Tensor &Qweight = context.getWeight(weight_idx[QKVParams::Q]);
  Tensor &Kweight = context.getWeight(weight_idx[QKVParams::K]);
  Tensor &Vweight = context.getWeight(weight_idx[QKVParams::V]);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  Tensor &Qhidden_ = context.getOutput(QKVParams::Q);
  Tensor &Khidden_ = context.getOutput(QKVParams::K);
  Tensor &Vhidden_ = context.getOutput(QKVParams::V);

  TensorDim input_dim = input_.getDim();
  TensorDim input_step_dim = input_dim;
  input_step_dim.batch(1);
  input_step_dim.height(to - from);

  Tensor input_step = input_.getSharedDataTensor(input_step_dim, 0, true);

  TensorDim Qhidden_step_dim = Qhidden_.getDim();
  Qhidden_step_dim.batch(1);
  Qhidden_step_dim.height(to - from);
  Tensor Qhidden_step = Qhidden_.getSharedDataTensor(Qhidden_step_dim, 0, true);

  TensorDim Khidden_step_dim = Khidden_.getDim();
  Khidden_step_dim.batch(1);
  Khidden_step_dim.height(to - from);
  Tensor Khidden_step = Khidden_.getSharedDataTensor(Khidden_step_dim, 0, true);

  TensorDim Vhidden_step_dim = Vhidden_.getDim();
  Vhidden_step_dim.batch(1);
  Vhidden_step_dim.height(to - from);
  Tensor Vhidden_step = Vhidden_.getSharedDataTensor(Vhidden_step_dim, 0, true);

  std::vector<Tensor *> Weights({&Qweight, &Kweight, &Vweight});
  std::vector<Tensor *> Outputs(
    {&Qhidden_step, &Khidden_step, &Vhidden_step});

  input_step.dot(Weights, Outputs);
}

void QKVLayer::calcDerivative(RunLayerContext &context) {
  throw std::runtime_error(
    "QKVLayer::calcDerivative not supported (inference-only layer)");
}

void QKVLayer::calcGradient(RunLayerContext &context) {
  throw std::runtime_error(
    "QKVLayer::calcGradient not supported (inference-only layer)");
}

void QKVLayer::updateTensorsByInputDimensions(
  RunLayerContext &context, std::vector<TensorDim> input_dimensions) {
  TensorDim input_dim = context.getInput(SINGLE_INOUT_IDX).getDim();
  TensorDim Qoutput_dim = context.getOutput(QKVParams::Q).getDim();
  TensorDim Koutput_dim = context.getOutput(QKVParams::K).getDim();
  TensorDim Voutput_dim = context.getOutput(QKVParams::V).getDim();

  input_dim.height(input_dimensions[0].height());
  Qoutput_dim.height(input_dimensions[0].height());
  Koutput_dim.height(input_dimensions[0].height());
  Voutput_dim.height(input_dimensions[0].height());

  context.updateInput(SINGLE_INOUT_IDX, input_dim);
  context.updateOutput(QKVParams::Q, Qoutput_dim);
  context.updateOutput(QKVParams::K, Koutput_dim);
  context.updateOutput(QKVParams::V, Voutput_dim);
}

} // namespace nntrainer
