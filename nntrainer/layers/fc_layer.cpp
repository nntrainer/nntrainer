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
 * @file	fc_layer.cpp
 * @date	14 May 2020
 * @brief	This is Fully Connected Layer Class for Neural Network
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <common_properties.h>
#include <fc_layer.h>
#include <layer_context.h>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

#include <iostream>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum FCParams { weight, bias };
enum LORAParams { loraA, loraB, loraTmp, loraOut };

FullyConnectedLayer::FullyConnectedLayer() :
  LayerImpl(),
  lora_scaling(1.0f),
  fc_props(props::Unit(), props::LoraRank(), props::LoraAlpha()),
  quantizer(nullptr) {
  weight_idx.fill(std::numeric_limits<unsigned>::max());
  lora_idx.fill(std::numeric_limits<unsigned>::max());
}

void FullyConnectedLayer::finalize(InitLayerContext &context) {
  auto &weight_regularizer =
    std::get<props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<props::WeightRegularizerConstant>(*layer_impl_props);
  auto &weight_initializer =
    std::get<props::WeightInitializer>(*layer_impl_props);
  auto &weight_decay = std::get<props::WeightDecay>(*layer_impl_props);
  auto &bias_decay = std::get<props::BiasDecay>(*layer_impl_props);
  auto &bias_initializer = std::get<props::BiasInitializer>(*layer_impl_props);
  auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);

  const auto &unit = std::get<props::Unit>(fc_props).get();
  const auto &lora_rank = (std::get<props::LoraRank>(fc_props).empty())
                            ? 0
                            : std::get<props::LoraRank>(fc_props).get();
  lora_scaling = (lora_rank && !std::get<props::LoraAlpha>(fc_props).empty())
                   ? (float)std::get<props::LoraAlpha>(fc_props) / lora_rank
                   : 1;

  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "Fully connected layer takes only one input";

  std::vector<TensorDim> output_dims(1);

  /// @todo fc actaully supports multidimensions. EffDimFlag shouldn't be fixed
  /// like this.
  context.setEffDimFlagInputDimension(0, 0b1001);
  context.setDynDimFlagInputDimension(0, 0b1000);

  bool is_nchw = (context.getFormat() == Tformat::NCHW);
  /** set output dimensions */
  auto const &in_dim = context.getInputDimensions()[0];
  output_dims[0] = in_dim;
  is_nchw ? output_dims[0].width(unit) : output_dims[0].channel(unit);

  output_dims[0].setTensorType(
    {context.getFormat(), context.getActivationDataType()});

  context.setOutputDimensions(output_dims);

  /** set weight specifications */
  // @todo : This NCHW format setting is just temporal, it needs to be set by
  // global configuration

  /** Bias Dimension : (1, 1, 1, unit) */
  /// @note bias is directly added to activation
  /// since we have no dequantizer for add operation,
  /// we have to set its data type as same as activation.
  /// This should be updated when the dequantizer is supported.
  TensorDim bias_dim(
    1, is_nchw ? 1 : unit, 1, is_nchw ? unit : 1,
    TensorDim::TensorType(context.getFormat(), context.getActivationDataType()),
    is_nchw ? 0b0001 : 0b0100);

  /** Weight Dimension : (1, 1, in_dim.width(), unit)*/
  TensorDim weight_dim(
    1, is_nchw ? 1 : unit, is_nchw ? in_dim.width() : 1,
    is_nchw ? unit : in_dim.channel(),
    TensorDim::TensorType(context.getFormat(), context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);

  weight_idx[FCParams::weight] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "weight", true);

  if (disable_bias.empty() || disable_bias.get() == false) {
    weight_idx[FCParams::bias] =
      context.requestWeight(bias_dim, bias_initializer, WeightRegularizer::NONE,
                            1.0f, bias_decay, "bias", true);
  }

  /** create weights for LoRA */
  if (lora_rank) {

    /** loraA Dimension : (1, 1, in_dim.width, lora_rank) */
    TensorDim loraA_dim(
      1, is_nchw ? 1 : lora_rank, is_nchw ? in_dim.width() : 1,
      is_nchw ? lora_rank : in_dim.channel(),
      TensorDim::TensorType(context.getFormat(), context.getWeightDataType()),
      is_nchw ? 0b0011 : 0b0101);

    /** loraB Dimension : (1, 1, lora_rank, unit) */
    TensorDim loraB_dim(
      1, is_nchw ? 1 : unit, is_nchw ? lora_rank : 1,
      is_nchw ? unit : lora_rank,
      TensorDim::TensorType(context.getFormat(), context.getWeightDataType()),
      is_nchw ? 0b0011 : 0b0101);

    /** loraTmp Dimension : (B, 1, in_dim.height(), lora_rank) */
    TensorDim loraTmp_dim(
      in_dim.batch(), is_nchw ? 1 : lora_rank, is_nchw ? in_dim.height() : 1,
      is_nchw ? lora_rank : in_dim.width(),
      TensorDim::TensorType(context.getFormat(),
                            context.getActivationDataType()),
      is_nchw ? 0b1011 : 0b1101);

    /** loraTmp Dimension : (B, 1, in_dim.height(), unit) */
    TensorDim loraOut_dim(
      in_dim.batch(), is_nchw ? 1 : unit, is_nchw ? in_dim.height() : 1,
      is_nchw ? unit : in_dim.width(),
      TensorDim::TensorType(context.getFormat(),
                            context.getActivationDataType()),
      is_nchw ? 0b1011 : 0b1101);

    lora_idx[LORAParams::loraA] = context.requestWeight(
      loraA_dim, Initializer::ZEROS, weight_regularizer,
      weight_regularizer_constant, weight_decay, "loraA", true);

    lora_idx[LORAParams::loraB] = context.requestWeight(
      loraB_dim, Initializer::LECUN_NORMAL, weight_regularizer,
      weight_regularizer_constant, weight_decay, "loraB", true);

    lora_idx[LORAParams::loraTmp] =
      context.requestTensor(loraTmp_dim, "hidden_tmp_lora", Initializer::NONE,
                            true, TensorLifespan::FORWARD_GRAD_LIFESPAN);

    lora_idx[LORAParams::loraOut] =
      context.requestTensor(loraOut_dim, "hidden_lora", Initializer::NONE, true,
                            TensorLifespan::FORWARD_FUNC_LIFESPAN);
  }

  ///@todo this quantizaer should be moved to tensor, not layer!
  switch (context.getWeightDataType()) {
  case ml::train::TensorDim::DataType::QINT4:
  case ml::train::TensorDim::DataType::QINT8:
  case ml::train::TensorDim::DataType::QINT16:
    quantizer =
      Quantization::createQuantizer(nntrainer::QScheme::PER_TENSOR_AFFINE);
    break;
  default:
    quantizer = nullptr;
    break;
  }
}

void FullyConnectedLayer::exportTo(
  Exporter &exporter, const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(fc_props, method, this);
}

void FullyConnectedLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, fc_props);
  LayerImpl::setProperty(remain_props);
}

void FullyConnectedLayer::setBatch(nntrainer::RunLayerContext &context,
                                   unsigned int batch) {
  if (!std::get<props::LoraRank>(fc_props).empty()) {
    // update Lora Tensor's batch info.
    context.updateTensor(lora_idx[LORAParams::loraTmp], batch);
    context.updateTensor(lora_idx[LORAParams::loraOut], batch);
  }
}

void FullyConnectedLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &weight = context.getWeight(weight_idx[FCParams::weight]);
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  ///@todo This dequantization action should be moved to tensor.dot()
  if (quantizer != nullptr) {
    Tensor weight_ = quantizer->dequantize(weight, input_.getDataType());
    input_.dot(weight_, hidden_, false, false);
  } else {
    input_.dot(weight, hidden_, false, false);
  }

  if (!std::get<props::LoraRank>(fc_props).empty()) {
    Tensor &loraA = context.getWeight(lora_idx[LORAParams::loraA]);
    Tensor &loraB = context.getWeight(lora_idx[LORAParams::loraB]);
    Tensor &hidden_tmp_lora = context.getTensor(lora_idx[LORAParams::loraTmp]);
    Tensor &hidden_out_lora = context.getTensor(lora_idx[LORAParams::loraOut]);

    input_.dot(loraA, hidden_tmp_lora, false, false);
    hidden_tmp_lora.dot(loraB, hidden_out_lora, false, false);
    hidden_out_lora.multiply_i(lora_scaling);
    hidden_.add_i(hidden_out_lora);
  }

  if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    Tensor &bias = context.getWeight(weight_idx[FCParams::bias]);
    hidden_.add_i(bias);
  }
}

void FullyConnectedLayer::incremental_forwarding(RunLayerContext &context,
                                                 unsigned int from,
                                                 unsigned int to,
                                                 bool training) {
  Tensor &weight = context.getWeight(weight_idx[FCParams::weight]);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  Tensor loraA, loraB, hidden_tmp_lora, hidden_out_lora;

  if (!std::get<props::LoraRank>(fc_props).empty()) {
    loraA = context.getWeight(lora_idx[LORAParams::loraA]);
    loraB = context.getWeight(lora_idx[LORAParams::loraB]);
    hidden_tmp_lora = context.getTensor(lora_idx[LORAParams::loraTmp]);
    hidden_out_lora = context.getTensor(lora_idx[LORAParams::loraOut]);
  }

  TensorDim input_dim = input_.getDim();
  TensorDim hidden_dim = hidden_.getDim();

  TensorDim input_step_dim = input_dim;
  TensorDim hidden_step_dim = hidden_dim;

  input_step_dim.batch(1);
  input_step_dim.height(to - from);
  hidden_step_dim.batch(1);
  hidden_step_dim.height(to - from);

  // @todo make it parallelized with batch axis
  for (unsigned int b = 0; b < hidden_.batch(); ++b) {
    Tensor input_step = input_.getSharedDataTensor(
      input_step_dim, b * hidden_dim.getFeatureLen(), true);
    Tensor hidden_step = hidden_.getSharedDataTensor(
      hidden_step_dim, b * hidden_dim.getFeatureLen(), true);

    input_step.dot(weight, hidden_step, false, false);

    if (!std::get<props::LoraRank>(fc_props).empty()) {
      nntrainer::TensorDim hidden_tmp_lora_step_dim = hidden_tmp_lora.getDim();
      hidden_tmp_lora_step_dim.batch(1);
      hidden_tmp_lora_step_dim.height(to - from);
      nntrainer::TensorDim hidden_out_lora_step_dim = hidden_out_lora.getDim();
      hidden_out_lora_step_dim.batch(1);
      hidden_out_lora_step_dim.height(to - from);

      nntrainer::Tensor hidden_tmp_lora_step =
        hidden_tmp_lora.getSharedDataTensor(
          hidden_tmp_lora_step_dim,
          b * hidden_tmp_lora.height() * hidden_tmp_lora.width(), true);
      nntrainer::Tensor hidden_out_lora_step =
        hidden_out_lora.getSharedDataTensor(
          hidden_out_lora_step_dim,
          b * hidden_out_lora.height() * hidden_out_lora.width(), true);

      input_step.dot(loraA, hidden_tmp_lora_step, false, false);
      hidden_tmp_lora_step.dot(loraB, hidden_out_lora_step, false, false);
      hidden_out_lora_step.multiply_i(lora_scaling);
      hidden_step.add_i(hidden_out_lora_step);
    }

    if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
        disable_bias.empty() || disable_bias.get() == false) {
      Tensor &bias = context.getWeight(weight_idx[FCParams::bias]);
      hidden_step.add_i(bias);
    }
  }
}

void FullyConnectedLayer::calcDerivative(RunLayerContext &context) {
  Tensor &weight = context.getWeight(weight_idx[FCParams::weight]);

  const Tensor &derivative_ = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &ret_ = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  if (!std::get<props::LoraRank>(fc_props).empty()) {
    Tensor &lora_A = context.getWeight(lora_idx[LORAParams::loraA]);
    Tensor &lora_B = context.getWeight(lora_idx[LORAParams::loraB]);
    ret_.dot_deriv_wrt_1(weight.add(lora_A.dot(lora_B).multiply(lora_scaling)),
                         derivative_, false, false);
  } else {
    ret_.dot_deriv_wrt_1(weight, derivative_, false, false);
  }
}

void FullyConnectedLayer::calcGradient(RunLayerContext &context) {

  /** (default) calcGradient - compute gradient of weight and bias */
  if (std::get<props::LoraRank>(fc_props).empty()) {
    Tensor &djdw = context.getWeightGrad(weight_idx[FCParams::weight]);
    djdw.setZero();

    const Tensor &derivative_ = context.getIncomingDerivative(SINGLE_INOUT_IDX);
    Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

    if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
        disable_bias.empty() || disable_bias.get() == false) {
      Tensor &djdb = context.getWeightGrad(weight_idx[FCParams::bias]);
      djdb.setZero();

      if (context.isGradientFirstAccess(weight_idx[FCParams::bias])) {
        derivative_.sum({0, 1, 2}, djdb);
      } else {
        /// @todo optimize below by adding beta to Tensor::sum
        Tensor t = derivative_.sum({0, 1, 2});
        djdb.add_i(t);
      }
    }

    input_.dot_deriv_wrt_2(
      djdw, derivative_, false, false,
      !context.isGradientFirstAccess(weight_idx[FCParams::weight]));
  } else {
    /** (lora) calcGradient - compute gradients of LoRA params only */
    Tensor &djdla = context.getWeightGrad(lora_idx[LORAParams::loraA]);
    Tensor &djdlb = context.getWeightGrad(lora_idx[LORAParams::loraB]);
    Tensor &djdtmp = context.getTensorGrad(lora_idx[LORAParams::loraTmp]);

    const Tensor &derivative_ = context.getIncomingDerivative(SINGLE_INOUT_IDX);
    Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
    Tensor &loraA = context.getWeight(lora_idx[LORAParams::loraA]);
    Tensor &loraB = context.getWeight(lora_idx[LORAParams::loraB]);
    Tensor &loraTmp = context.getTensor(lora_idx[LORAParams::loraTmp]);
    const auto &lora_derivative_ = derivative_.multiply(lora_scaling);

    loraTmp.dot_deriv_wrt_2(
      djdlb, lora_derivative_, false, false,
      !context.isGradientFirstAccess(lora_idx[LORAParams::loraB]));
    djdtmp.dot_deriv_wrt_1(
      loraB, lora_derivative_, false, false,
      !context.isGradientFirstAccess(lora_idx[LORAParams::loraTmp]));
    input_.dot_deriv_wrt_2(
      djdla, djdtmp, false, false,
      !context.isGradientFirstAccess(lora_idx[LORAParams::loraA]));
  }
}

} /* namespace nntrainer */
