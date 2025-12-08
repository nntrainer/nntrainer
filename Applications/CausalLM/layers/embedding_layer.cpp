// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   embedding.cpp
 * @date   04 March 2021
 * @brief  This is Embedding Layer Class of Neural Network
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   This embedding layer supports FP32/FP16/Q6_K data type only.
 */

#include <embedding_layer.h>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum EmbeddingParams { weight };

EmbeddingLayer::EmbeddingLayer() :
  LayerImpl(),
  embedding_props(nntrainer::props::InDim(), nntrainer::props::OutDim(),
                  nntrainer::props::Scale()),
  weight_idx(std::numeric_limits<unsigned>::max()) {}

void EmbeddingLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "Embedding layer takes only one input";

  const nntrainer::TensorDim &input_dim =
    context.getInputDimensions()[SINGLE_INOUT_IDX];
  NNTR_THROW_IF(input_dim.channel() != 1, std::invalid_argument)
    << "Embedding layer takes only one for channel size";

  NNTR_THROW_IF(input_dim.getDataType() != nntrainer::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "Embedding layer takes only FP32 input data";

  auto &weight_regularizer =
    std::get<nntrainer::props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<nntrainer::props::WeightRegularizerConstant>(*layer_impl_props);
  auto weight_initializer = nntrainer::props::InitializerInfo::Enum::NONE;
  auto &weight_decay =
    std::get<nntrainer::props::WeightDecay>(*layer_impl_props);

  size_t in_dim =
    static_cast<size_t>(std::get<nntrainer::props::InDim>(embedding_props));
  size_t out_dim =
    static_cast<size_t>(std::get<nntrainer::props::OutDim>(embedding_props));

  nntrainer::TensorDim output_dim = input_dim;

  // output_dim expected as hidden x num input (batch size)
  output_dim.height(input_dim.width());
  output_dim.width(out_dim);
  output_dim.setTensorType(
    {context.getFormat(), context.getActivationDataType()});
  context.setOutputDimensions({output_dim});

  nntrainer::TensorDim dim = output_dim;

  dim.setTensorType({context.getFormat(), context.getWeightDataType()});

  dim.height(in_dim);
  dim.width(out_dim);
  dim.batch(1);

  weight_idx = context.requestWeight(
    dim, weight_initializer, weight_regularizer, weight_regularizer_constant,
    weight_decay, "Embedding", true);
}

void EmbeddingLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, embedding_props);
  LayerImpl::setProperty(remain_props);
}

void EmbeddingLayer::forwarding(nntrainer::RunLayerContext &context,
                                bool training) {}

void EmbeddingLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                            unsigned int from, unsigned int to,
                                            bool training) {

  /// @todo get input and output dimension from input_ and hidden itself
  unsigned int in_dim = std::get<nntrainer::props::InDim>(embedding_props);
  unsigned int out_dim = std::get<nntrainer::props::OutDim>(embedding_props);
  float scale = std::get<nntrainer::props::Scale>(embedding_props).empty()
                  ? 1.0f
                  : std::get<nntrainer::props::Scale>(embedding_props).get();
  unsigned int _from = from;

  nntrainer::Tensor &weight = context.getWeight(weight_idx);
  nntrainer::Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  nntrainer::TensorDim out_tensor_dim =
    nntrainer::TensorDim({1, 1, 1, out_dim}, hidden_.getTensorType());

  unsigned int b_size = input_.batch();

  for (unsigned int b = 0; b < b_size; ++b) {
    float *in_data =
      input_.getAddress<float>(b * input_.getDim().getFeatureLen());
    nntrainer::Tensor batchsliced_hidden = hidden_.getBatchSlice(b, 1);

    int iter = to - from;

#pragma omp parallel for
    for (int i = 0; i < iter; ++i) {
      size_t embed_idx = static_cast<size_t>(in_data[i]);
      if (embed_idx >= in_dim) {
        throw std::invalid_argument("input word index is greater than in_dim");
      }

      nntrainer::Tensor cur_weight =
        weight.getSharedDataTensor(out_tensor_dim, out_dim * embed_idx);
      nntrainer::Tensor out_tensor =
        batchsliced_hidden.getSharedDataTensor(out_tensor_dim, out_dim * (i));

      if (weight.getDataType() == nntrainer::TensorDim::DataType::Q6_K) {
        ///@note this should be replaced with quantizer operation
        int num_blocks_per_row = (weight.width() + 256 - 1) / 256;
        nntrainer::dequantize_row_q6_K(
          (void *)((char *)weight.getData<uint8_t>() +
                   (210 * num_blocks_per_row) * embed_idx),
          out_tensor.getData(), out_dim);
      } else if (weight.getDataType() == nntrainer::TensorDim::DataType::Q4_0) {
        ///@note this should be replaced with quantizer operation
        int num_blocks_per_row = (weight.width() + 32 - 1) / 32;
        nntrainer::dequantize_row_q4_0(
          (void *)((char *)weight.getData<uint8_t>() +
                   (18 * num_blocks_per_row) * embed_idx),
          out_tensor.getData(), out_dim);
      } else {
        out_tensor.copyData(cur_weight);
      }

      if (scale != 1.0f) {
        out_tensor.multiply_i(scale);
      }
    }

#ifdef DEBUG
    std::cout << context.getName() << " : "
              << "\n input:" << input_ << "\n weight: " << weight
              << "\n hidden: " << hidden_ << std::endl;
#endif
  }
}

void EmbeddingLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  throw nntrainer::exception::not_supported(
    "calcDerivative for Embedding layer is not supported");
}

void EmbeddingLayer::calcGradient(nntrainer::RunLayerContext &context) {}

void EmbeddingLayer::exportTo(nntrainer::Exporter &exporter,
                              const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(embedding_props, method, this);
}

#ifdef PLUGGABLE

nntrainer::Layer *create_embedding_layer() {
  auto layer = new EmbeddingLayer();
  std::cout << "embedding layer created\n";
  return layer;
}

void destroy_embedding_layer(nntrainer::Layer *layer) {
  std::cout << "embeddinglayer is deleted\n";
  delete layer;
}

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_embedding_layer,
                                                   destroy_embedding_layer};
}

#endif

} // namespace causallm
