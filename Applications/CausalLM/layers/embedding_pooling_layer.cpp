// SPDX-License-Identifier: Apatche-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file    embedding_pooling_layer.cpp
 * @date    02 Jan 2026
 * @brief   This is Embedding Pooling Layer Class
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Eunju Yang <ej.yang@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#include <embedding_pooling_layer.h>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

EmbeddingPoolingLayer::EmbeddingPoolingLayer() :
  LayerImpl(),
  pooling_props(
    props::WordEmbeddingDimension(), props::PoolingModeClsToken(false),
    props::PoolingModeMeanTokens(false), props::PoolingModeMaxTokens(false),
    props::PoolingModeMeanSqrtLenTokens(false),
    props::PoolingModeWeightedMeanTokens(false),
    props::PoolingModeLastToken(false), props::IncludePrompt(true)) {}

void EmbeddingPoolingLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "EmbeddingPooling layer takes only one input";

  const nntrainer::TensorDim &input_dim =
    context.getInputDimensions()[SINGLE_INOUT_IDX];

  unsigned int word_embed_dim =
    std::get<props::WordEmbeddingDimension>(pooling_props);

  if (input_dim.width() != word_embed_dim) {
    ml_logw("Input dimension width (%zu) does not match "
            "word_embedding_dimension (%d)",
            input_dim.width(), word_embed_dim);
  }

  // Output dimension for Pooling is [batch, 1, 1, word_embed_dim]
  nntrainer::TensorDim output_dim = input_dim;
  output_dim.height(1);

  context.setOutputDimensions({output_dim});

  bool mode_cls = std::get<props::PoolingModeClsToken>(pooling_props);
  bool mode_mean = std::get<props::PoolingModeMeanTokens>(pooling_props);
  bool mode_max = std::get<props::PoolingModeMaxTokens>(pooling_props);
  bool mode_mean_sqrt =
    std::get<props::PoolingModeMeanSqrtLenTokens>(pooling_props);
  bool mode_weighted_mean =
    std::get<props::PoolingModeWeightedMeanTokens>(pooling_props);

  if (mode_cls || mode_max || mode_mean_sqrt || mode_weighted_mean) {
    throw nntrainer::exception::not_supported(
      "Only pooling_mode_lasttoken and pooling_mode_mean_tokens are currently "
      "supported in EmbeddingPoolingLayer");
  }
}

void EmbeddingPoolingLayer::setProperty(
  const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, pooling_props);
  LayerImpl::setProperty(remain_props);
}

void EmbeddingPoolingLayer::forwarding(nntrainer::RunLayerContext &context,
                                       bool training) {
  nntrainer::Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);

  unsigned int batch = input.batch();
  unsigned int seq_len = input.height();
  unsigned int dim = input.width();

  bool mode_lasttoken = std::get<props::PoolingModeLastToken>(pooling_props);
  bool mode_mean = std::get<props::PoolingModeMeanTokens>(pooling_props);

  if (mode_lasttoken) {
    for (unsigned int b = 0; b < batch; ++b) {
      // Last token index = seq_len - 1
      nntrainer::Tensor source = input.getSharedDataTensor(
        {1, 1, 1, dim}, b * seq_len * dim + (seq_len - 1) * dim);

      nntrainer::Tensor dest =
        output.getSharedDataTensor({1, 1, 1, dim}, b * dim);
      dest.copyData(source);
    }
  } else if (mode_mean) {
    for (unsigned int b = 0; b < batch; ++b) {
      nntrainer::Tensor source =
        input.getSharedDataTensor({1, 1, seq_len, dim}, b * seq_len * dim);
      nntrainer::Tensor dest =
        output.getSharedDataTensor({1, 1, 1, dim}, b * dim);

      // Calculate mean along average dim (height/seq_len)
      dest.copyData(source.average(2));
    }
  } else {
    output.setZero();
  }
}

void EmbeddingPoolingLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  nntrainer::Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);

  unsigned int batch = input.batch();
  unsigned int dim = input.width();
  size_t feature_len = input.getDim().getFeatureLen(); // height * width

  bool mode_lasttoken = std::get<props::PoolingModeLastToken>(pooling_props);

  if (mode_lasttoken) {
    for (unsigned int b = 0; b < batch; ++b) {
      // Use feature_len for batch stride
      // The last token processed is at index `to-1` in the absolute sequence.
      size_t offset = static_cast<size_t>(b) * feature_len + (to - 1) * dim;

      nntrainer::Tensor source =
        input.getSharedDataTensor({1, 1, 1, dim}, offset);
      nntrainer::Tensor dest =
        output.getSharedDataTensor({1, 1, 1, dim}, b * dim);

      dest.copyData(source);
    }
  } else if (std::get<props::PoolingModeMeanTokens>(pooling_props)) {
    for (unsigned int b = 0; b < batch; ++b) {
      unsigned int len = to - from;
      size_t offset = static_cast<size_t>(b) * feature_len + from * dim;

      nntrainer::Tensor source =
        input.getSharedDataTensor({1, 1, len, dim}, offset);
      nntrainer::Tensor dest =
        output.getSharedDataTensor({1, 1, 1, dim}, b * dim);

      dest.copyData(source.average(2));
    }
  } else {
    output.setZero();
  }
}

void EmbeddingPoolingLayer::calcDerivative(
  nntrainer::RunLayerContext &context) {
  throw nntrainer::exception::not_supported(
    "calcDerivative for EmbeddingPooling layer is not supported");
}

void EmbeddingPoolingLayer::calcGradient(nntrainer::RunLayerContext &context) {
  throw nntrainer::exception::not_supported(
    "calcGradient for EmbeddingPooling layer is not supported");
}

void EmbeddingPoolingLayer::exportTo(
  nntrainer::Exporter &exporter, const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(pooling_props, method, this);
}
} // namespace causallm
