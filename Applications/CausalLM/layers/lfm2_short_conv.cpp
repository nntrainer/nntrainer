// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   lfm2_short_conv.cpp
 * @date   4 May 2026
 * @brief  LFM2 short convolution layer for CausalLM inference
 */

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

#include <layer_context.h>
#include <lfm2_short_conv.h>
#include <nntrainer_error.h>
#include <node_exporter.h>
#include <util_func.h>

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum Lfm2ShortConvParams { in_proj_weight, conv_weight, out_proj_weight };
enum Lfm2ShortConvTensors { conv_state };

Lfm2ShortConvLayer::Lfm2ShortConvLayer() :
  LayerImpl(),
  conv_props(nntrainer::props::Unit(), nntrainer::props::KernelSize()),
  hidden_size(0),
  kernel_size(0) {
  weight_idx.fill(std::numeric_limits<unsigned int>::max());
  tensor_idx.fill(std::numeric_limits<unsigned int>::max());
}

void Lfm2ShortConvLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "LFM2 short convolution layer takes only one input";

  auto &weight_regularizer =
    std::get<nntrainer::props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<nntrainer::props::WeightRegularizerConstant>(*layer_impl_props);
  auto weight_initializer = nntrainer::props::InitializerInfo::Enum::NONE;
  auto &weight_decay =
    std::get<nntrainer::props::WeightDecay>(*layer_impl_props);

  const auto &in_dim = context.getInputDimensions()[SINGLE_INOUT_IDX];
  hidden_size = std::get<nntrainer::props::Unit>(conv_props).empty()
                  ? in_dim.width()
                  : std::get<nntrainer::props::Unit>(conv_props).get();
  kernel_size = std::get<nntrainer::props::KernelSize>(conv_props).empty()
                  ? 3
                  : std::get<nntrainer::props::KernelSize>(conv_props).get();

  NNTR_THROW_IF(in_dim.width() != hidden_size, std::invalid_argument)
    << "LFM2 short convolution input width must match unit";

  context.setEffDimFlagInputDimension(0, 0b1001);
  context.setDynDimFlagInputDimension(0, 0b1000);

  std::vector<nntrainer::TensorDim> output_dims(1, in_dim);
  output_dims[0].setTensorType(
    {context.getFormat(), context.getActivationDataType()});
  context.setOutputDimensions(output_dims);

  nntrainer::TensorDim in_proj_dim(
    1, 1, hidden_size, hidden_size * 3,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()),
    0b0011);
  weight_idx[Lfm2ShortConvParams::in_proj_weight] = context.requestWeight(
    in_proj_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "in_proj_weight", true);

  nntrainer::TensorDim conv_weight_dim(
    1, 1, hidden_size, kernel_size,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()),
    0b0011);
  weight_idx[Lfm2ShortConvParams::conv_weight] = context.requestWeight(
    conv_weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "conv_weight", true);

  nntrainer::TensorDim out_proj_dim(
    1, 1, hidden_size, hidden_size,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()),
    0b0011);
  weight_idx[Lfm2ShortConvParams::out_proj_weight] = context.requestWeight(
    out_proj_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "out_proj_weight", true);

  nntrainer::TensorDim conv_state_dim(
    in_dim.batch(), 1, kernel_size, hidden_size,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getActivationDataType()),
    0b1011);
  tensor_idx[Lfm2ShortConvTensors::conv_state] = context.requestTensor(
    conv_state_dim, "conv_state", nntrainer::Initializer::ZEROS, false,
    nntrainer::TensorLifespan::MAX_LIFESPAN);
}

void Lfm2ShortConvLayer::forwarding(nntrainer::RunLayerContext &context,
                                    bool training) {
  incremental_forwarding(context, 0, context.getInput(SINGLE_INOUT_IDX).height(),
                         training);
}

void Lfm2ShortConvLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  nntrainer::Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &in_proj =
    context.getWeight(weight_idx[Lfm2ShortConvParams::in_proj_weight]);
  nntrainer::Tensor &conv =
    context.getWeight(weight_idx[Lfm2ShortConvParams::conv_weight]);
  nntrainer::Tensor &out_proj =
    context.getWeight(weight_idx[Lfm2ShortConvParams::out_proj_weight]);
  nntrainer::Tensor &state =
    context.getTensor(tensor_idx[Lfm2ShortConvTensors::conv_state]);

  NNTR_THROW_IF(input.getDataType() != ml::train::TensorDim::DataType::FP32 ||
                  output.getDataType() != ml::train::TensorDim::DataType::FP32 ||
                  in_proj.getDataType() !=
                    ml::train::TensorDim::DataType::FP32 ||
                  conv.getDataType() != ml::train::TensorDim::DataType::FP32 ||
                  out_proj.getDataType() !=
                    ml::train::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "LFM2 short convolution currently supports FP32 activations and "
       "weights only";

  nntrainer::TensorDim input_dim = input.getDim();
  nntrainer::TensorDim output_dim = output.getDim();

  unsigned int step_size = input_dim.height() > 1 ? to - from : 1;

  nntrainer::TensorDim input_step_dim = input_dim;
  input_step_dim.batch(1);
  if (input_dim.height() > 1)
    input_step_dim.height(step_size);

  nntrainer::TensorDim output_step_dim = output_dim;
  output_step_dim.batch(1);
  if (output_dim.height() > 1)
    output_step_dim.height(step_size);

  nntrainer::TensorDim projected_dim(
    1, 1, step_size, hidden_size * 3, input_dim.getTensorType(), 0b1011);
  nntrainer::TensorDim gated_dim(1, 1, step_size, hidden_size,
                                 input_dim.getTensorType(), 0b1011);
  nntrainer::TensorDim state_step_dim(1, 1, kernel_size, hidden_size,
                                      input_dim.getTensorType(), 0b1011);

  for (unsigned int b = 0; b < input_dim.batch(); ++b) {
    nntrainer::Tensor input_step = input.getSharedDataTensor(
      input_step_dim, b * input_dim.getFeatureLen(), true);
    nntrainer::Tensor output_step = output.getSharedDataTensor(
      output_step_dim, b * output_dim.getFeatureLen(), true);
    nntrainer::Tensor state_step = state.getSharedDataTensor(
      state_step_dim, b * kernel_size * hidden_size, true);

    nntrainer::Tensor projected(projected_dim, true);
    input_step.dot(in_proj, projected, false, false);

    nntrainer::Tensor gated(gated_dim, true);
    const float *projected_data = projected.getData<float>();
    const float *conv_weight = conv.getData<float>();
    float *gated_data = gated.getData<float>();
    float *state_data = state_step.getData<float>();

    std::vector<float> bx(static_cast<size_t>(step_size) * hidden_size);
    std::vector<float> c_gate(static_cast<size_t>(step_size) * hidden_size);

    for (unsigned int t = 0; t < step_size; ++t) {
      const size_t projected_base = static_cast<size_t>(t) * hidden_size * 3;
      const size_t token_base = static_cast<size_t>(t) * hidden_size;
      for (unsigned int d = 0; d < hidden_size; ++d) {
        const float b_gate = projected_data[projected_base + d];
        c_gate[token_base + d] = projected_data[projected_base + hidden_size + d];
        const float x_gate =
          projected_data[projected_base + hidden_size * 2 + d];
        bx[token_base + d] = b_gate * x_gate;
      }
    }

    if (from == 0) {
      std::fill(state_data, state_data + kernel_size * hidden_size, 0.0f);
      for (unsigned int t = 0; t < step_size; ++t) {
        const size_t token_base = static_cast<size_t>(t) * hidden_size;
        for (unsigned int d = 0; d < hidden_size; ++d) {
          float conv_out = 0.0f;
          for (unsigned int k = 0; k < kernel_size; ++k) {
            const int src_t =
              static_cast<int>(t) + static_cast<int>(k) -
              static_cast<int>(kernel_size) + 1;
            if (src_t >= 0) {
              conv_out +=
                bx[static_cast<size_t>(src_t) * hidden_size + d] *
                conv_weight[static_cast<size_t>(d) * kernel_size + k];
            }
          }
          gated_data[token_base + d] = c_gate[token_base + d] * conv_out;
        }
      }

      const unsigned int copy_tokens = std::min(step_size, kernel_size);
      const unsigned int state_offset = kernel_size - copy_tokens;
      for (unsigned int t = 0; t < copy_tokens; ++t) {
        const unsigned int src_t = step_size - copy_tokens + t;
        std::copy_n(bx.data() + static_cast<size_t>(src_t) * hidden_size,
                    hidden_size,
                    state_data + static_cast<size_t>(state_offset + t) *
                                   hidden_size);
      }
    } else {
      for (unsigned int t = 0; t < step_size; ++t) {
        std::move(state_data + hidden_size,
                  state_data + static_cast<size_t>(kernel_size) * hidden_size,
                  state_data);
        std::copy_n(bx.data() + static_cast<size_t>(t) * hidden_size,
                    hidden_size,
                    state_data +
                      static_cast<size_t>(kernel_size - 1) * hidden_size);

        const size_t token_base = static_cast<size_t>(t) * hidden_size;
        for (unsigned int d = 0; d < hidden_size; ++d) {
          float conv_out = 0.0f;
          for (unsigned int k = 0; k < kernel_size; ++k) {
            conv_out +=
              state_data[static_cast<size_t>(k) * hidden_size + d] *
              conv_weight[static_cast<size_t>(d) * kernel_size + k];
          }
          gated_data[token_base + d] = c_gate[token_base + d] * conv_out;
        }
      }
    }

    gated.dot(out_proj, output_step, false, false);
  }
}

void Lfm2ShortConvLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  std::throw_with_nested(std::runtime_error("Training is not supported yet."));
}

void Lfm2ShortConvLayer::calcGradient(nntrainer::RunLayerContext &context) {
  std::throw_with_nested(std::runtime_error("Training is not supported yet."));
}

void Lfm2ShortConvLayer::exportTo(
  nntrainer::Exporter &exporter,
  const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(conv_props, method, this);
}

void Lfm2ShortConvLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, conv_props);
  LayerImpl::setProperty(remain_props);
}

void Lfm2ShortConvLayer::setBatch(nntrainer::RunLayerContext &context,
                                  unsigned int batch) {
  context.updateTensor(tensor_idx[Lfm2ShortConvTensors::conv_state], batch);
}

void Lfm2ShortConvLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  nntrainer::TensorDim input_dim = context.getInput(SINGLE_INOUT_IDX).getDim();
  nntrainer::TensorDim output_dim =
    context.getOutput(SINGLE_INOUT_IDX).getDim();

  input_dim.height(input_dimensions[0].height());
  output_dim.height(input_dimensions[0].height());

  context.updateInput(SINGLE_INOUT_IDX, input_dim);
  context.updateOutput(SINGLE_INOUT_IDX, output_dim);
}

#ifdef PLUGGABLE

nntrainer::Layer *create_lfm2_short_conv_layer() {
  auto layer = new Lfm2ShortConvLayer();
  return layer;
}

void destroy_lfm2_short_conv_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable
  ml_train_layer_pluggable{create_lfm2_short_conv_layer,
                           destroy_lfm2_short_conv_layer};
}

#endif

} // namespace causallm
