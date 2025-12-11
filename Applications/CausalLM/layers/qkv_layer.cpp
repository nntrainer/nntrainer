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
 * @brief	This is Fully Connected Layer Class for Neural Network
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Eunju Yang <ej.yang@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <qkv_layer.h>

#include <bs_thread_pool_manager.hpp>
#include <engine.h>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum QKVParams { Q, K, V };

QKVLayer::QKVLayer() :
  LayerImpl(), qkv_props(props::QUnit(), props::KUnit(), props::VUnit()) {
  weight_idx.fill(std::numeric_limits<unsigned>::max());
}

void QKVLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "Fully connected layer takes only one input";

  auto &weight_regularizer =
    std::get<nntrainer::props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<nntrainer::props::WeightRegularizerConstant>(*layer_impl_props);
  auto weight_initializer = nntrainer::props::InitializerInfo::Enum::NONE;
  auto &weight_decay =
    std::get<nntrainer::props::WeightDecay>(*layer_impl_props);

  const auto &q_unit = std::get<props::QUnit>(qkv_props).get();
  const auto &k_unit = std::get<props::KUnit>(qkv_props).get();
  const auto &v_unit = std::get<props::VUnit>(qkv_props).get();

  std::vector<nntrainer::TensorDim> output_dims(3);

  /// @todo fc actaully supports multidimensions. EffDimFlag shouldn't be fixed
  /// like this.
  context.setEffDimFlagInputDimension(0, 0b1001);
  context.setDynDimFlagInputDimension(0, 0b1000);

  bool is_nchw = (context.getFormat() == nntrainer::Tformat::NCHW);
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

  /** Q */
  nntrainer::TensorDim weight_dim(
    1, is_nchw ? 1 : q_unit, is_nchw ? in_dim.width() : 1,
    is_nchw ? q_unit : in_dim.channel(),
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);
  weight_idx[QKVParams::Q] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "qweight", true);

  /** K */
  weight_dim.width(k_unit);
  weight_idx[QKVParams::K] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "kweight", true);

  /** V */
  weight_dim.width(v_unit);
  weight_idx[QKVParams::V] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "vweight", true);
}

void QKVLayer::exportTo(nntrainer::Exporter &exporter,
                        const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(qkv_props, method, this);
}

void QKVLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, qkv_props);
  LayerImpl::setProperty(remain_props);
}

void QKVLayer::forwarding(nntrainer::RunLayerContext &context, bool training) {
  return;
}

void QKVLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                      unsigned int from, unsigned int to,
                                      bool training) {
  nntrainer::Tensor &Qweight = context.getWeight(weight_idx[QKVParams::Q]);
  nntrainer::Tensor &Kweight = context.getWeight(weight_idx[QKVParams::K]);
  nntrainer::Tensor &Vweight = context.getWeight(weight_idx[QKVParams::V]);
  nntrainer::Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &Qhidden_ = context.getOutput(QKVParams::Q);
  nntrainer::Tensor &Khidden_ = context.getOutput(QKVParams::K);
  nntrainer::Tensor &Vhidden_ = context.getOutput(QKVParams::V);

  nntrainer::TensorDim input_dim = input_.getDim();
  nntrainer::TensorDim input_step_dim = input_dim;
  input_step_dim.batch(1);
  input_step_dim.height(to - from);

  auto &pool =
    nntrainer::Engine::Global().getThreadPoolManager()->getThreadPool();

  nntrainer::Tensor input_step =
    input_.getSharedDataTensor(input_step_dim, 0, true);

  nntrainer::TensorDim Qhidden_dim = Qhidden_.getDim();
  nntrainer::TensorDim Qhidden_step_dim = Qhidden_.getDim();
  Qhidden_step_dim.batch(1);
  Qhidden_step_dim.height(to - from);
  nntrainer::Tensor Qhidden_step =
    Qhidden_.getSharedDataTensor(Qhidden_step_dim, 0, true);

  nntrainer::TensorDim Khidden_dim = Khidden_.getDim();
  nntrainer::TensorDim Khidden_step_dim = Khidden_.getDim();
  Khidden_step_dim.batch(1);
  Khidden_step_dim.height(to - from);
  nntrainer::Tensor Khidden_step =
    Khidden_.getSharedDataTensor(Khidden_step_dim, 0, true);

  nntrainer::TensorDim Vhidden_dim = Vhidden_.getDim();
  nntrainer::TensorDim Vhidden_step_dim = Vhidden_.getDim();
  Vhidden_step_dim.batch(1);
  Vhidden_step_dim.height(to - from);
  nntrainer::Tensor Vhidden_step =
    Vhidden_.getSharedDataTensor(Vhidden_step_dim, 0, true);

  std::vector<nntrainer::Tensor *> Weights({&Qweight, &Kweight, &Vweight});
  std::vector<nntrainer::Tensor *> Outputs(
    {&Qhidden_step, &Khidden_step, &Vhidden_step});

  input_step.dot(Weights, Outputs);
}

void QKVLayer::calcDerivative(nntrainer::RunLayerContext &context) { return; }

void QKVLayer::calcGradient(nntrainer::RunLayerContext &context) { return; }

void QKVLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  ml::train::TensorDim input_dim = context.getInput(SINGLE_INOUT_IDX).getDim();
  ml::train::TensorDim Qoutput_dim = context.getOutput(QKVParams::Q).getDim();
  ml::train::TensorDim Koutput_dim = context.getOutput(QKVParams::K).getDim();
  ml::train::TensorDim Voutput_dim = context.getOutput(QKVParams::V).getDim();

  input_dim.height(input_dimensions[0].height());
  Qoutput_dim.height(input_dimensions[0].height());
  Koutput_dim.height(input_dimensions[0].height());
  Voutput_dim.height(input_dimensions[0].height());

  context.updateInput(SINGLE_INOUT_IDX, input_dim);
  context.updateOutput(QKVParams::Q, Qoutput_dim);
  context.updateOutput(QKVParams::K, Koutput_dim);
  context.updateOutput(QKVParams::V, Voutput_dim);
}
} // namespace causallm
