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

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum LmHeadParams {
  weight,
  bias,
};

LmHeadLayer::LmHeadLayer() :
  LayerImpl(), lmhead_props(nntrainer::props::Unit()) {
  weight_idx.fill(std::numeric_limits<unsigned>::max());
}

void LmHeadLayer::finalize(nntrainer::InitLayerContext &context) {
  auto &weight_regularizer =
    std::get<nntrainer::props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<nntrainer::props::WeightRegularizerConstant>(*layer_impl_props);
  auto weight_initializer = nntrainer::Initializer::ONES;
  auto &weight_decay =
    std::get<nntrainer::props::WeightDecay>(*layer_impl_props);
  auto &bias_decay = std::get<nntrainer::props::BiasDecay>(*layer_impl_props);
  auto &bias_initializer =
    std::get<nntrainer::props::BiasInitializer>(*layer_impl_props);
  auto &disable_bias =
    std::get<nntrainer::props::DisableBias>(*layer_impl_props);

  auto unit = std::get<nntrainer::props::Unit>(lmhead_props).get();

  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "lm head layer takes only one input";

  std::vector<ml::train::TensorDim> output_dims(1);

  /// @todo fc actaully supports multidimensions.
  /// EffDimFlag shouldn't be fixed like this.
  context.setEffDimFlagInputDimension(0, 0b1001);
  context.setDynDimFlagInputDimension(0, 0b1000);
  bool is_nchw = (context.getFormat() == nntrainer::Tformat::NCHW);

  /** set output dimensions */
  ///@note lm_head's output dimension (height is always 1 !)
  auto const &in_dim = context.getInputDimensions()[0];
  output_dims[0] = in_dim;
  if (is_nchw)
    output_dims[0].width(unit);
  else
    output_dims[0].channel(unit);
  output_dims[0].height(1);

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
    weight_idx[LmHeadParams::bias] = context.requestWeight(
      bias_dim, bias_initializer, nntrainer::WeightRegularizer::NONE, 1.0f,
      bias_decay, "bias", true);
  }
}

void LmHeadLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, lmhead_props);
  LayerImpl::setProperty(remain_props);
}

void LmHeadLayer::forwarding(nntrainer::RunLayerContext &context,
                             bool training) {

  nntrainer::Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  unsigned int height = input_.getDim().height();

  // Process the full sequence by calling incremental_forwarding once with
  // from=0 and to=height. This unifies the implementation and ensures
  // training/inference consistency.
  incremental_forwarding(context, 0, height, training);
}

void LmHeadLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) {

  nntrainer::Tensor weight =
    context.getWeight(weight_idx[LmHeadParams::weight]);

  nntrainer::Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);

  ml::train::TensorDim input_dim = input_.getDim();
  ml::train::TensorDim hidden_dim = hidden_.getDim();

  ml::train::TensorDim input_step_dim = input_dim;
  ml::train::TensorDim hidden_step_dim = hidden_dim;

  input_step_dim.batch(1);
  input_step_dim.height(1);
  hidden_step_dim.batch(1);

  unsigned int b_size = input_dim.batch();
  unsigned int step_height = to - from;

  for (unsigned int b = 0; b < b_size; ++b) {
    for (unsigned int h = 0; h < step_height; ++h) {
      // Process each token in the range [from, to)
      // For inference (single token), step_height=1, so h=0 and we process
      // token 'from' For training/test (full sequence), we process all tokens
      nntrainer::Tensor input_step = input_.getSharedDataTensor(
        input_step_dim,
        b * input_dim.getFeatureLen() + (from + h) * input_.width(), true);
      nntrainer::Tensor hidden_step = hidden_.getSharedDataTensor(
        hidden_step_dim,
        b * hidden_dim.getFeatureLen() + (from + h) * hidden_dim.width(), true);

      input_step.dot(weight, hidden_step, false, false);

      if (auto &disable_bias =
            std::get<nntrainer::props::DisableBias>(*layer_impl_props);
          disable_bias.empty() || disable_bias.get() == false) {
        nntrainer::Tensor &bias =
          context.getWeight(weight_idx[LmHeadParams::bias]);
        hidden_step.add_i(bias);
      }
    }
  }
}

void LmHeadLayer::calcDerivative(nntrainer::RunLayerContext &context) {

  nntrainer::Tensor weight =
    context.getWeight(weight_idx[LmHeadParams::weight]);
  nntrainer::Tensor &dx = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  const nntrainer::Tensor &dy = context.getIncomingDerivative(SINGLE_INOUT_IDX);

  // dx = dy . weight^T
  dy.dot(weight, dx, false, true);
}

void LmHeadLayer::calcGradient(nntrainer::RunLayerContext &context) {

  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  const nntrainer::Tensor &dy = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  nntrainer::Tensor &dweight =
    context.getWeightGrad(weight_idx[LmHeadParams::weight]);

  // dweight = in^T . dy
  in.dot(dy, dweight, true, false);

  if (auto &disable_bias =
        std::get<nntrainer::props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    nntrainer::Tensor &dbias =
      context.getWeightGrad(weight_idx[LmHeadParams::bias]);
    dbias.setZero();
    float *db_data = dbias.getData<float>();
    const float *dy_data = dy.getData<float>();

    size_t batch = dy.batch();
    size_t channel = dy.channel();
    size_t height = dy.height();
    size_t width = dy.width();

    for (size_t b = 0; b < batch; ++b) {
      for (size_t c = 0; c < channel; ++c) {
        for (size_t h = 0; h < height; ++h) {
          size_t offset =
            b * channel * height * width + c * height * width + h * width;
          for (size_t w = 0; w < width; ++w) {
            db_data[w] += dy_data[offset + w];
          }
        }
      }
    }
  }
}

void LmHeadLayer::exportTo(nntrainer::Exporter &exporter,
                           const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(lmhead_props, method, this);
}

void LmHeadLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  nntrainer::TensorDim in_dim = context.getInput(SINGLE_INOUT_IDX).getDim();

  unsigned int height = input_dimensions[0].height();

  // output dim's height is always 1 !
  in_dim.height(height);
  context.updateInput(SINGLE_INOUT_IDX, in_dim);
}

#ifdef PLUGGABLE

nntrainer::Layer *create_tie_word_embedding() {
  auto layer = new LmHeadLayer();
  std::cout << "embedding layer created\n";
  return layer;
}

void destroy_tie_word_embedding(nntrainer::Layer *layer) {
  std::cout << "embeddinglayer is deleted\n";
  delete layer;
}

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_tie_word_embedding,
                                                   destroy_tie_word_embedding};
}

#endif

} // namespace causallm