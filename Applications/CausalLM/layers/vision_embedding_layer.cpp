// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Hyeonseok Lee <hs89.lee@samsung.com>
 *
 * @file   vision_embedding_layer.cpp
 * @date   17 April 2026
 * @brief  This is Embedding Layer Class of Neural Network
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   This embedding layer supports FP32/FP16/Q6_K data type only.
 */

#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>
#include <vision_embedding_layer.h>

namespace causallm {

static constexpr size_t OUT_IDX = 0;
static constexpr size_t TOKEN_IDX = 0;
static constexpr size_t IMAGE_IDX = 1;

enum EmbeddingParams { weight };

VisionEmbeddingLayer::VisionEmbeddingLayer() :
  LayerImpl(),
  vision_embedding_props(nntrainer::props::InDim(), nntrainer::props::OutDim(),
                         props::ImageStartToken(), nntrainer::props::Scale()),
  weight_idx(std::numeric_limits<unsigned>::max()) {}

void VisionEmbeddingLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 2, std::invalid_argument)
    << "VisionEmbedding layer takes 2 input";

  const nntrainer::TensorDim &token_dim =
    context.getInputDimensions()[TOKEN_IDX];
  NNTR_THROW_IF(token_dim.channel() != 1, std::invalid_argument)
    << "VisionEmbedding layer takes only one for channel size for token input";

  const nntrainer::TensorDim &image_dim =
    context.getInputDimensions()[IMAGE_IDX];
  NNTR_THROW_IF(image_dim.channel() != 1, std::invalid_argument)
    << "VisionEmbedding layer takes only one for channel size for image input";

  auto &weight_regularizer =
    std::get<nntrainer::props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<nntrainer::props::WeightRegularizerConstant>(*layer_impl_props);
  auto weight_initializer = nntrainer::props::InitializerInfo::Enum::NONE;
  auto &weight_decay =
    std::get<nntrainer::props::WeightDecay>(*layer_impl_props);

  size_t in_dim = static_cast<size_t>(
    std::get<nntrainer::props::InDim>(vision_embedding_props));
  size_t out_dim = static_cast<size_t>(
    std::get<nntrainer::props::OutDim>(vision_embedding_props));

  nntrainer::TensorDim output_dim = token_dim;

  output_dim.height(token_dim.width() + image_dim.height());

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
    weight_decay, "VisionEmbedding", true);
}

void VisionEmbeddingLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, vision_embedding_props);
  LayerImpl::setProperty(remain_props);
}

void VisionEmbeddingLayer::forwarding(nntrainer::RunLayerContext &context,
                                      bool training) {
  unsigned int in_dim =
    std::get<nntrainer::props::InDim>(vision_embedding_props);
  unsigned int out_dim =
    std::get<nntrainer::props::OutDim>(vision_embedding_props);
  unsigned int image_start_token =
    std::get<props::ImageStartToken>(vision_embedding_props);
  float scale =
    std::get<nntrainer::props::Scale>(vision_embedding_props).empty()
      ? 1.0f
      : std::get<nntrainer::props::Scale>(vision_embedding_props).get();

  nntrainer::Tensor &weight = context.getWeight(weight_idx);
  nntrainer::Tensor &hidden_ = context.getOutput(OUT_IDX);
  nntrainer::Tensor &tokens = context.getInput(TOKEN_IDX);
  nntrainer::Tensor &image = context.getInput(IMAGE_IDX);

  nntrainer::TensorDim out_token_dim =
    nntrainer::TensorDim({1, 1, 1, out_dim}, hidden_.getTensorType());
  nntrainer::TensorDim out_image_dim = nntrainer::TensorDim(
    {1, 1, image.height(), out_dim}, hidden_.getTensorType());

  unsigned int b_size = tokens.batch();

  for (unsigned int b = 0; b < b_size; ++b) {
    float *tokens_data =
      tokens.getAddress<float>(b * tokens.getDim().getFeatureLen());
    nntrainer::Tensor batchsliced_hidden = hidden_.getBatchSlice(b, 1);
    nntrainer::Tensor batchsliced_image = image.getBatchSlice(b, 1);

    size_t image_start_token_idx = 0;

    for (int i = 0; i < (int)tokens.width(); ++i) {
      size_t embed_idx = static_cast<size_t>(tokens_data[i]);
      if (embed_idx == image_start_token) {
        image_start_token_idx = i;
      }
    }

    for (int i = 0; i < (int)tokens.width(); ++i) {
      size_t embed_idx = static_cast<size_t>(tokens_data[i]);
      if (embed_idx >= in_dim) {
        throw std::invalid_argument("input word index is greater than in_dim");
      }

      nntrainer::Tensor out_tensor = batchsliced_hidden.getSharedDataTensor(
        out_token_dim, i <= image_start_token_idx
                         ? i * out_dim
                         : (i + image.height()) * out_dim);
      nntrainer::Tensor cur_weight =
        weight.getSharedDataTensor(out_token_dim, out_dim * embed_idx);

      if (weight.getDataType() == nntrainer::TensorDim::DataType::Q6_K) {
        ///@note this should be replaced with quantizer operation
        int num_blocks_per_row = (weight.width() + 256 - 1) / 256;
        const void *src = (void *)((char *)weight.getData<uint8_t>() +
                                   (210 * num_blocks_per_row) * embed_idx);
        if (out_tensor.getDataType() == nntrainer::TensorDim::DataType::FP32) {
          nntrainer::dequantize_row_q6_K(src, out_tensor.getData(), out_dim);
        } else {
          nntrainer::TensorDim fp32_dim(
            {1, 1, 1, out_dim},
            nntrainer::TensorDim::TensorType(
              out_token_dim.getFormat(), nntrainer::TensorDim::DataType::FP32));
          nntrainer::Tensor tmp(fp32_dim, true);
          nntrainer::dequantize_row_q6_K(src, tmp.getData(), out_dim);
          out_tensor.copyData(tmp);
        }
      } else if (weight.getDataType() == nntrainer::TensorDim::DataType::Q4_0) {
        ///@note this should be replaced with quantizer operation
        int num_blocks_per_row = (weight.width() + 32 - 1) / 32;
        const void *src = (void *)((char *)weight.getData<uint8_t>() +
                                   (18 * num_blocks_per_row) * embed_idx);
        if (out_tensor.getDataType() == nntrainer::TensorDim::DataType::FP32) {
          nntrainer::dequantize_row_q4_0(src, out_tensor.getData(), out_dim);
        } else {
          nntrainer::TensorDim fp32_dim(
            {1, 1, 1, out_dim},
            nntrainer::TensorDim::TensorType(
              out_token_dim.getFormat(), nntrainer::TensorDim::DataType::FP32));
          nntrainer::Tensor tmp(fp32_dim, true);
          nntrainer::dequantize_row_q4_0(src, tmp.getData(), out_dim);
          out_tensor.copyData(tmp);
        }
      } else {
        out_tensor.copyData(cur_weight);
      }

      if (scale != 1.0f) {
        out_tensor.multiply_i(scale);
      }

      if (embed_idx == image_start_token) {
        nntrainer::Tensor out_image_tensor =
          batchsliced_hidden.getSharedDataTensor(out_image_dim,
                                                 (i + 1) * out_dim);
        out_image_tensor.copyData(batchsliced_image);

        if (scale != 1.0f) {
          out_image_tensor.multiply_i(scale);
        }
      }
    }
  }
}

void VisionEmbeddingLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {

  unsigned int in_dim =
    std::get<nntrainer::props::InDim>(vision_embedding_props);
  unsigned int out_dim =
    std::get<nntrainer::props::OutDim>(vision_embedding_props);
  unsigned int image_start_token =
    std::get<props::ImageStartToken>(vision_embedding_props);
  float scale =
    std::get<nntrainer::props::Scale>(vision_embedding_props).empty()
      ? 1.0f
      : std::get<nntrainer::props::Scale>(vision_embedding_props).get();
  nntrainer::Tensor &weight = context.getWeight(weight_idx);
  nntrainer::Tensor &hidden_ = context.getOutput(OUT_IDX);
  nntrainer::Tensor &tokens = context.getInput(TOKEN_IDX);
  nntrainer::Tensor &image = context.getInput(IMAGE_IDX);

  nntrainer::TensorDim out_token_dim =
    nntrainer::TensorDim({1, 1, 1, out_dim}, hidden_.getTensorType());
  nntrainer::TensorDim out_image_dim = nntrainer::TensorDim(
    {1, 1, image.height(), out_dim}, hidden_.getTensorType());

  unsigned int b_size = tokens.batch();

  for (unsigned int b = 0; b < b_size; ++b) {
    float *tokens_data =
      tokens.getAddress<float>(b * tokens.getDim().getFeatureLen());
    nntrainer::Tensor batchsliced_hidden = hidden_.getBatchSlice(b, 1);
    nntrainer::Tensor batchsliced_image = image.getBatchSlice(b, 1);

    int iter = to - from;

    size_t image_start_token_idx = 0;

    for (int i = 0; i < (int)tokens.width(); ++i) {
      size_t embed_idx = static_cast<size_t>(tokens_data[i]);
      if (embed_idx == image_start_token) {
        image_start_token_idx = i;
      }
    }

    for (int i = 0; i < iter; ++i) {
      size_t embed_idx = static_cast<size_t>(tokens_data[from + i]);
      if (embed_idx >= in_dim) {
        throw std::invalid_argument("input word index is greater than in_dim");
      }

      unsigned int out_slot = from + i;
      nntrainer::Tensor out_tensor = batchsliced_hidden.getSharedDataTensor(
        out_token_dim, out_slot <= image_start_token_idx
                         ? out_slot * out_dim
                         : (out_slot + image.height()) * out_dim);
      nntrainer::Tensor cur_weight =
        weight.getSharedDataTensor(out_token_dim, out_dim * embed_idx);

      if (weight.getDataType() == nntrainer::TensorDim::DataType::Q6_K) {
        ///@note this should be replaced with quantizer operation
        int num_blocks_per_row = (weight.width() + 256 - 1) / 256;
        const void *src = (void *)((char *)weight.getData<uint8_t>() +
                                   (210 * num_blocks_per_row) * embed_idx);
        if (out_tensor.getDataType() == nntrainer::TensorDim::DataType::FP32) {
          nntrainer::dequantize_row_q6_K(src, out_tensor.getData(), out_dim);
        } else {
          nntrainer::TensorDim fp32_dim(
            {1, 1, 1, out_dim},
            nntrainer::TensorDim::TensorType(
              out_token_dim.getFormat(), nntrainer::TensorDim::DataType::FP32));
          nntrainer::Tensor tmp(fp32_dim, true);
          nntrainer::dequantize_row_q6_K(src, tmp.getData(), out_dim);
          out_tensor.copyData(tmp);
        }
      } else if (weight.getDataType() == nntrainer::TensorDim::DataType::Q4_0) {
        ///@note this should be replaced with quantizer operation
        int num_blocks_per_row = (weight.width() + 32 - 1) / 32;
        const void *src = (void *)((char *)weight.getData<uint8_t>() +
                                   (18 * num_blocks_per_row) * embed_idx);
        if (out_tensor.getDataType() == nntrainer::TensorDim::DataType::FP32) {
          nntrainer::dequantize_row_q4_0(src, out_tensor.getData(), out_dim);
        } else {
          nntrainer::TensorDim fp32_dim(
            {1, 1, 1, out_dim},
            nntrainer::TensorDim::TensorType(
              out_token_dim.getFormat(), nntrainer::TensorDim::DataType::FP32));
          nntrainer::Tensor tmp(fp32_dim, true);
          nntrainer::dequantize_row_q4_0(src, tmp.getData(), out_dim);
          out_tensor.copyData(tmp);
        }
      } else {
        out_tensor.copyData(cur_weight);
      }

      if (scale != 1.0f) {
        out_tensor.multiply_i(scale);
      }

      if (embed_idx == image_start_token) {
        nntrainer::Tensor out_image_tensor =
          batchsliced_hidden.getSharedDataTensor(out_image_dim,
                                                 (out_slot + 1) * out_dim);
        out_image_tensor.copyData(batchsliced_image);

        if (scale != 1.0f) {
          out_image_tensor.multiply_i(scale);
        }
      }
    }
  }
}

void VisionEmbeddingLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  throw nntrainer::exception::not_supported(
    "calcDerivative for VisionEmbedding layer is not supported");
}

void VisionEmbeddingLayer::calcGradient(nntrainer::RunLayerContext &context) {}

void VisionEmbeddingLayer::exportTo(
  nntrainer::Exporter &exporter, const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(vision_embedding_props, method, this);
}

void VisionEmbeddingLayer::save(std::ofstream &file,
                                nntrainer::RunLayerContext &run_context,
                                bool opt_var, ml::train::ExecutionMode mode,
                                bool trainable,
                                nntrainer::TensorDim::DataType dtype,
                                ml::train::ISA target_isa) const {
  // @note shared weights are only be saved at the first access
  for (unsigned int i = 0; i < run_context.getNumWeights(); ++i) {
    if (run_context.isGradientFirstAccess(i)) {
      auto &weight = run_context.getWeight(i);
      if (dtype == nntrainer::TensorDim::DataType::NONE ||
          weight.getDataType() == dtype)
        weight.save(file);
      else {
        NNTR_THROW_IF(weight.getDataType() !=
                        nntrainer::TensorDim::DataType::FP32,
                      std::runtime_error)
          << "Save with quantization only supports for FP32 weight.";
        ///@note The codelines below can be replaced with quantizer's
        /// quantize()
        nntrainer::TensorDim dim = weight.getDim();
        unsigned int K = dim.height();
        unsigned int N = dim.width();

        if (dtype == nntrainer::TensorDim::DataType::Q4_0) {

          // Skip quantization for bias-like tensors (1D with height == 1)
          // as they are not suitable for Q4_0 block quantization
          if (K == 1) {
            weight.save(file);
          } else {
            NNTR_THROW_IF(N % 32 != 0 || K % 32 != 0, std::invalid_argument)
              << "Q4_0 quantization requires both width and height to be "
                 "divisible by 32, but got height="
              << K << ", width=" << N;
            //////////////////////////////////////////////////////////////////
            ///@note Please note that VisionEmbedding layer doesn't need to be
            /// transposed!
            //////////////////////////////////////////////////////////////////
            nntrainer::Tensor quant_weight(dim.batch(), dim.channel(), K, N,
                                           {nntrainer::Tformat::NCHW, dtype});
            nntrainer::quantize_q4_0(weight.getData<float>(),
                                     quant_weight.getData<uint8_t>(), K, N,
                                     nullptr);
            quant_weight.save(file);
          }
        } else if (dtype == nntrainer::TensorDim::DataType::Q6_K) {
          //////////////////////////////////////////////////////////////////
          ///@note Please note that VisionEmbedding layer doesn't need to be
          /// transposed!
          //////////////////////////////////////////////////////////////////
          nntrainer::Tensor quant_weight(dim.batch(), dim.channel(), K, N,
                                         {nntrainer::Tformat::NCHW, dtype});
          nntrainer::quantize_q6_K(weight.getData<float>(),
                                   quant_weight.getData<uint8_t>(), K, N,
                                   nullptr);
          quant_weight.save(file);
        } else {
          NNTR_THROW_IF(true, std::runtime_error)
            << "This dtype is not supported in save with quantization";
        }
      }
    }
  }
}

void VisionEmbeddingLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  nntrainer::TensorDim token_dim = context.getInput(TOKEN_IDX).getDim();
  nntrainer::TensorDim image_dim = context.getInput(IMAGE_IDX).getDim();
  nntrainer::TensorDim out_dim = context.getOutput(OUT_IDX).getDim();

  unsigned int token_len = input_dimensions[TOKEN_IDX].width();
  unsigned int image_len = input_dimensions[IMAGE_IDX].height();

  token_dim.width(token_len);
  image_dim.height(image_len);
  out_dim.height(token_len + image_len);

  context.updateInput(TOKEN_IDX, token_dim);
  context.updateInput(IMAGE_IDX, image_dim);
  context.updateOutput(OUT_IDX, out_dim);
}

#ifdef PLUGGABLE

nntrainer::Layer *create_vision_embedding_layer() {
  auto layer = new VisionEmbeddingLayer();
  std::cout << "vision embedding layer created\n";
  return layer;
}

void destroy_vision_embedding_layer(nntrainer::Layer *layer) {
  std::cout << "vision embeddinglayer is deleted\n";
  delete layer;
}

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{
  create_vision_embedding_layer, destroy_vision_embedding_layer};
}

#endif

} // namespace causallm
