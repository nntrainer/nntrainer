// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   shared_fully_connected_layer.cpp
 * @date   20 January 2026
 * @brief  Shared Fully Connected Layer Class implementation
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <fstream>
#include <iostream>
#include <layer_context.h>
#include <limits>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <shared_fully_connected_layer.h>
#include <tensor.h>
#include <tensor_dim.h>
#include <util_func.h>

/**
 * @brief Namespace for CausalLM application components
 */
namespace causallm {
using namespace nntrainer;

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum FCParams { weight, bias };

SharedFullyConnectedLayer::SharedFullyConnectedLayer() :
  nntrainer::LayerImpl() {
  weight_idx.fill(std::numeric_limits<unsigned>::max());
  shared_mode_ = std::get<props::SharedMode>(fc_props).get();
}

void SharedFullyConnectedLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() < 1, std::invalid_argument)
    << "SharedFullyConnectedLayer requires at least one input";

  auto &disable_bias = std::get<nntrainer::props::DisableBias>(fc_props);
  const auto &unit = std::get<nntrainer::props::Unit>(fc_props).get();

  auto weight_initializer = nntrainer::props::InitializerInfo::Enum::NONE;
  auto bias_initializer = nntrainer::props::InitializerInfo::Enum::NONE;
  std::vector<nntrainer::TensorDim> output_dims(1);

  /// @todo fc actaully supports multidimensions. EffDimFlag shouldn't be fixed
  /// like this.
  context.setEffDimFlagInputDimension(0, 0b1001);
  context.setDynDimFlagInputDimension(0, 0b1000);

  bool is_nchw = (context.getFormat() == ml::train::TensorDim::Format::NCHW);
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

  /** Weight Dimension : (1, 1, in_dim.width(), unit)*/
  nntrainer::TensorDim weight_dim(
    1, is_nchw ? 1 : unit, is_nchw ? in_dim.width() : 1,
    is_nchw ? unit : in_dim.channel(),
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);

  /** Bias Dimension : (1, 1, 1, unit) */
  nntrainer::TensorDim bias_dim(
    1, is_nchw ? 1 : unit, 1, is_nchw ? unit : 1,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getActivationDataType()),
    is_nchw ? 0b0001 : 0b0100);

  weight_idx[FCParams::weight] =
    context.requestWeight(weight_dim, weight_initializer,
                          WeightRegularizer::NONE, 1.0f, 0.0f, "weight", true);

  if (disable_bias.empty() || disable_bias.get() == false) {
    weight_idx[FCParams::bias] =
      context.requestWeight(bias_dim, bias_initializer, WeightRegularizer::NONE,
                            1.0f, 0.0f, "bias", true);
  }
}

void SharedFullyConnectedLayer::setProperty(
  const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, fc_props);
  LayerImpl::setProperty(remain_props);
  shared_mode_ = std::get<props::SharedMode>(fc_props).get();
}

void SharedFullyConnectedLayer::forwarding(nntrainer::RunLayerContext &context,
                                           bool training) {
  nntrainer::Tensor &weight = context.getWeight(weight_idx[FCParams::weight]);

  nntrainer::Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  input_.dot(weight, hidden_, false, false);

  if (auto &disable_bias = std::get<nntrainer::props::DisableBias>(fc_props);
      disable_bias.empty() || disable_bias.get() == false) {
    nntrainer::Tensor &bias = context.getWeight(weight_idx[FCParams::bias]);

    hidden_.add_i(bias);
  }
}

void SharedFullyConnectedLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  nntrainer::Tensor &weight = context.getWeight(weight_idx[FCParams::weight]);

  nntrainer::Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);

  bool FullInputRange = std::get<props::FullInputRange>(fc_props).get();

  if (FullInputRange) {
    input_.dot(weight, hidden_, false, false);
    if (auto &disable_bias = std::get<nntrainer::props::DisableBias>(fc_props);
        disable_bias.empty() || disable_bias.get() == false) {
      nntrainer::Tensor &bias = context.getWeight(weight_idx[FCParams::bias]);
      hidden_.add_i(bias);
    }
    return;
  }

  nntrainer::TensorDim input_dim = input_.getDim();
  nntrainer::TensorDim hidden_dim = hidden_.getDim();

  nntrainer::TensorDim input_step_dim = input_dim;
  nntrainer::TensorDim hidden_step_dim = hidden_dim;

  input_step_dim.batch(1);
  input_step_dim.height(to - from);
  hidden_step_dim.batch(1);
  hidden_step_dim.height(to - from);

  for (unsigned int b = 0; b < hidden_.batch(); ++b) {
    nntrainer::Tensor input_step = input_.getSharedDataTensor(
      input_step_dim, b * hidden_dim.getFeatureLen(), true);
    nntrainer::Tensor hidden_step = hidden_.getSharedDataTensor(
      hidden_step_dim, b * hidden_dim.getFeatureLen(), true);

    input_step.dot(weight, hidden_step, false, false);

    if (auto &disable_bias = std::get<nntrainer::props::DisableBias>(fc_props);
        disable_bias.empty() || disable_bias.get() == false) {
      nntrainer::Tensor &bias = context.getWeight(weight_idx[FCParams::bias]);
      hidden_step.add_i(bias);
    }
  }
}

void SharedFullyConnectedLayer::calcDerivative(
  nntrainer::RunLayerContext &context) {
  throw nntrainer::exception::not_supported(
    "calcDerivative for SharedFullyConnectedLayer is not supported");
}

void SharedFullyConnectedLayer::calcGradient(
  nntrainer::RunLayerContext &context) {
  throw nntrainer::exception::not_supported(
    "calcGradient for SharedFullyConnectedLayer is not supported");
}

void SharedFullyConnectedLayer::exportTo(
  nntrainer::Exporter &exporter, const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(fc_props, method, this);
}

void SharedFullyConnectedLayer::read(
  std::ifstream &file, nntrainer::RunLayerContext &context, bool opt_var,
  ml::train::ExecutionMode mode, bool trainable,
  nntrainer::TensorDim::DataType definedWeightDataType, bool fsu,
  size_t start_offset, bool read_from_offset, int file_fd) {
  if (!shared_mode_) {
    size_t current_offset = start_offset;
    for (unsigned int i = 0; i < context.getNumWeights(); ++i) {
      auto &w = context.getWeight(i);
      w.read(file, current_offset, read_from_offset, file_fd);

      if (read_from_offset &&
          current_offset != std::numeric_limits<size_t>::max()) {
        current_offset += w.bytes();
      }

      if (context.isMixedPrecision(i) && !context.getWeightFP32(i).empty()) {
        context.getWeightFP32(i).copyData(context.getWeight(i));
      }
    }
  }
}

void SharedFullyConnectedLayer::save(
  std::ofstream &file, nntrainer::RunLayerContext &run_context, bool opt_var,
  ml::train::ExecutionMode mode, bool trainable,
  nntrainer::TensorDim::DataType definedWeightDataType,
  ml::train::ISA target_isa) const {
  if (!shared_mode_) {
    nntrainer::Layer::save(file, run_context, opt_var, mode, trainable,
                           definedWeightDataType, target_isa);
  }
}

} // namespace causallm
