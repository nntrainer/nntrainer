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
 *
 */

#include <embedding.h>
#include <layer_context.h>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <thread_manager.h>
#include <util_func.h>

#include <cstring>
#include <iostream>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum EmbeddingParams { weight };

EmbeddingLayer::EmbeddingLayer() :
  LayerImpl(),
  embedding_props(props::InDim(), props::OutDim()),
  weight_idx(std::numeric_limits<unsigned>::max()) {}

void EmbeddingLayer::finalize(InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "Embedding layer takes only one input";

  const TensorDim &input_dim = context.getInputDimensions()[SINGLE_INOUT_IDX];
  NNTR_THROW_IF(input_dim.channel() != 1, std::invalid_argument)
    << "Embedding layer takes only one for channel size";

  // Token-ID input expected (caller responsibility). Input dtype check
  // removed so the layer can sit between an FP32 input layer and FP16
  // activation downstream.

  auto &weight_regularizer =
    std::get<props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<props::WeightRegularizerConstant>(*layer_impl_props);
  auto &weight_initializer =
    std::get<props::WeightInitializer>(*layer_impl_props);
  auto &weight_decay = std::get<props::WeightDecay>(*layer_impl_props);

  unsigned int in_dim = std::get<props::InDim>(embedding_props);
  unsigned int out_dim = std::get<props::OutDim>(embedding_props);

  TensorDim output_dim = input_dim;

  output_dim.height(input_dim.width());
  output_dim.width(out_dim);
  output_dim.setTensorType(
    {context.getFormat(), context.getActivationDataType()});
  context.setOutputDimensions({output_dim});

  TensorDim dim = output_dim;

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

void EmbeddingLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  forwardRows(context, input_.width());
}

void EmbeddingLayer::incremental_forwarding(RunLayerContext &context,
                                            unsigned int from, unsigned int to,
                                            bool training) {

  if (from) {
    // Normalize to 0-based while preserving step size for multi-token prefill
    to = to - from;
    from = 0;
  }

  forwardRows(context, to - from);
}

void EmbeddingLayer::forwardRows(RunLayerContext &context,
                                 unsigned int token_count) {
  const unsigned int in_dim = std::get<props::InDim>(embedding_props);
  const unsigned int out_dim = std::get<props::OutDim>(embedding_props);

  Tensor &weight = context.getWeight(weight_idx);
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  const Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  const TensorDim &input_dim = input_.getDim();
  const TensorDim &hidden_dim = hidden_.getDim();

  NNTR_THROW_IF(token_count > input_.width(), std::invalid_argument)
    << "embedding token count exceeds input width";

  const size_t task_count = static_cast<size_t>(input_.batch()) * token_count;
  auto get_index = [&](size_t task) {
    const size_t batch = task / token_count;
    const size_t token = task % token_count;
    const float value =
      input_.getValue<float>(batch * input_dim.getFeatureLen() + token);
    const unsigned int embed_idx = static_cast<unsigned int>(value);
    NNTR_THROW_IF(embed_idx >= in_dim, std::invalid_argument)
      << "input word index is greater than in_dim";
    return embed_idx;
  };

  const auto data_type = weight.getDataType();
  const bool direct_copy = weight.getContiguous() && hidden_.getContiguous() &&
                           data_type == hidden_.getDataType() &&
                           (data_type == TensorDim::DataType::FP32 ||
                            data_type == TensorDim::DataType::FP16);

  if (direct_copy) {
    const size_t row_bytes =
      static_cast<size_t>(out_dim) * weight.getDim().getDataTypeSize();
    const auto *weight_data =
      reinterpret_cast<const unsigned char *>(weight.getData());
    auto *output_data = reinterpret_cast<unsigned char *>(hidden_.getData());
    auto copy_row = [&](size_t task, unsigned int embed_idx) {
      const size_t batch = task / token_count;
      const size_t token = task % token_count;
      const size_t output_offset =
        (batch * hidden_dim.getFeatureLen() + token * out_dim) *
        hidden_dim.getDataTypeSize();
      std::memcpy(output_data + output_offset,
                  weight_data + static_cast<size_t>(embed_idx) * row_bytes,
                  row_bytes);
    };

    constexpr size_t PARALLEL_COPY_THRESHOLD_BYTES = 64 * 1024;
    if (task_count * row_bytes >= PARALLEL_COPY_THRESHOLD_BYTES) {
      std::vector<unsigned int> indices(task_count);
      for (size_t task = 0; task < task_count; ++task) {
        indices[task] = get_index(task);
      }
      ThreadManager::Global().parallel_for(
        0, task_count, [&](size_t task) { copy_row(task, indices[task]); });
    } else {
      for (size_t task = 0; task < task_count; ++task) {
        copy_row(task, get_index(task));
      }
    }
    return;
  }

  TensorDim out_tensor_dim =
    TensorDim({1, 1, 1, out_dim}, hidden_.getTensorType());

  for (unsigned int b = 0; b < input_.batch(); ++b) {
    Tensor batchsliced_hidden = hidden_.getBatchSlice(b, 1);
    for (unsigned int i = 0; i < token_count; ++i) {
      const unsigned int embed_idx = get_index(b * token_count + i);
      Tensor cur_weight =
        weight.getSharedDataTensor(out_tensor_dim, out_dim * embed_idx);
      Tensor out_tensor =
        batchsliced_hidden.getSharedDataTensor(out_tensor_dim, out_dim * i);
      out_tensor.copyData(cur_weight);
    }
  }
}

void EmbeddingLayer::calcDerivative(RunLayerContext &context) {
  throw exception::not_supported(
    "calcDerivative for Embedding layer is not supported");
}

void EmbeddingLayer::calcGradient(RunLayerContext &context) {
  unsigned int out_dim = std::get<props::OutDim>(embedding_props);

  Tensor &djdw = context.getWeightGrad(weight_idx);
  const Tensor &derivative_ = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  djdw.setZero();

  // TODO:
  // This is to calculate gradient with current implementation of optimizer.
  // In order to accelerate, we need to better way like using index to weight.

  /// @todo
  // Current nntrainer gradient Tensor shape is identical to its
  // weight shape. However, this creates a sparse Tensor since we are only using
  // certain indices of the Tensor that we are interested in. Since we have such
  // indices before accessing to the Tensor, we can optimize it by deleting the
  // sparse-value indices. Also left as an Issue as well.

  for (unsigned int b = 0; b < input_.batch(); ++b) {
    float *in_data =
      input_.getAddress<float>(b * input_.getDim().getFeatureLen());

    if (djdw.getDataType() == TensorDim::DataType::FP32) {
      for (unsigned int i = 0; i < input_.width(); ++i) {
        unsigned int embed_idx = (unsigned int)((float *)(in_data))[i];
        // Assume padding is 0 and index always start from 1.
        // If in_data[i] - 1 < 0, then it skips.
        // if (embed_idx == 0)
        //   continue;

        float *djdw_data = djdw.getAddress<float>(embed_idx * out_dim);
        const float *grad_data = derivative_.getAddress<float>(
          b * derivative_.getDim().getFeatureLen() + i * out_dim);

        std::transform(djdw_data, djdw_data + out_dim, grad_data, djdw_data,
                       std::plus<float>());
      }
    } else if (djdw.getDataType() == TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
      for (unsigned int i = 0; i < input_.width(); ++i) {
        unsigned int embed_idx = (unsigned int)((float *)(in_data))[i];
        // Assume padding is 0 and index always start from 1.
        // If in_data[i] - 1 < 0, then it skips.
        // if (embed_idx == 0)
        //   continue;

        _FP16 *djdw_data = djdw.getAddress<_FP16>(embed_idx * out_dim);
        const _FP16 *grad_data = derivative_.getAddress<_FP16>(
          b * derivative_.getDim().getFeatureLen() + i * out_dim);

        std::transform(djdw_data, djdw_data + out_dim, grad_data, djdw_data,
                       std::plus<_FP16>());
      }
#else
      throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
    }
  }
}

void EmbeddingLayer::exportTo(Exporter &exporter,
                              const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(embedding_props, method, this);
}

} // namespace nntrainer
