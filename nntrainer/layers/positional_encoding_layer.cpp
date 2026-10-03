// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 Hyeonseok Lee <hs89.lee@samsung.com>
 *
 * @file   positional_encoding_layer.cpp
 * @date   16 August 2022
 * @brief  This file contains the positional encoding layer in transformer
 * @see    https://github.com/nntrainer/nntrainer
 *         https://arxiv.org/abs/1607.06450
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <math.h>
#include <regex>

#include <positional_encoding_layer.h>
#include <tensor_dim.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum PositionalEncodingParams {
  positional_encoding,
  denominator,
};

/**
 * @brief positional encoding of position @a i at model dimension @a j
 */
static inline float positionalEncodingValue(unsigned int i, unsigned int j,
                                            const float *denom) {
  float value = i / denom[j];
  return (j & 1) ? cosf(value) : sinf(value);
}

/**
 * @brief fill a contiguous (num_tokens x model_dim) buffer
 */
template <typename T>
static void fillPositionalEncoding(T *pe, unsigned int num_tokens,
                                   unsigned int model_dim, const float *denom) {
  for (unsigned int i = 0; i < num_tokens; ++i)
    for (unsigned int j = 0; j < model_dim; ++j)
      pe[static_cast<size_t>(i) * model_dim + j] =
        static_cast<T>(positionalEncodingValue(i, j, denom));
}

/**
 * @brief fill pe with the positional encoding of its first pe.height()
 * positions
 * @param pe contiguous tensor to fill, laid out as (positions, model dim)
 * @param denom FP32 scratch tensor holding at least model dim elements
 */
static void calculatePositionalEncoding(Tensor &pe, Tensor &denom) {
  unsigned int num_tokens = pe.height();
  unsigned int model_dim = pe.width();

  float *denom_data = denom.getData<float>();
  for (unsigned int j = 0; j < model_dim; ++j) {
    unsigned int jj = (j >> 1) << 1;
    denom_data[j] = powf(10000.0f, jj / (float)model_dim);
  }

  switch (pe.getDataType()) {
  case TensorDim::DataType::FP32:
    fillPositionalEncoding(pe.getData<float>(), num_tokens, model_dim,
                           denom_data);
    break;
#ifdef ENABLE_FP16
  case TensorDim::DataType::FP16:
    fillPositionalEncoding(pe.getData<_FP16>(), num_tokens, model_dim,
                           denom_data);
    break;
#endif
  default:
    for (unsigned int i = 0; i < num_tokens; ++i)
      for (unsigned int j = 0; j < model_dim; ++j)
        pe.setValue(0, 0, i, j, positionalEncodingValue(i, j, denom_data));
    break;
  }
}

PositionalEncodingLayer::PositionalEncodingLayer() :
  positional_encoding_props(props::MaxTimestep()) {
  tensor_idx.fill(std::numeric_limits<unsigned>::max());
}

PositionalEncodingLayer::~PositionalEncodingLayer() {}

void PositionalEncodingLayer::finalize(InitLayerContext &context) {
  unsigned int max_token_size =
    std::get<props::MaxTimestep>(positional_encoding_props);

  std::vector<ml::train::TensorDim> input_dims = context.getInputDimensions();
  NNTR_THROW_IF(input_dims[SINGLE_INOUT_IDX].height() > max_token_size,
                std::invalid_argument)
    << "[positional encoding layer] " << context.getName() << ": input length "
    << input_dims[SINGLE_INOUT_IDX].height() << " exceeds max_timestep "
    << max_token_size;
  context.setOutputDimensions(input_dims);

  unsigned int model_dim = input_dims[SINGLE_INOUT_IDX].width();

  ml::train::TensorDim pe_dim(
    {max_token_size, model_dim},
    {context.getFormat(), context.getWeightDataType()});
  tensor_idx[PositionalEncodingParams::positional_encoding] =
    context.requestTensor(pe_dim, "positional_encoding",
                          nntrainer::Initializer::NONE, false,
                          nntrainer::TensorLifespan::MAX_LIFESPAN);

  ml::train::TensorDim denom_dim(
    {1, model_dim},
    {context.getFormat(), ml::train::TensorDim::DataType::FP32});
  tensor_idx[PositionalEncodingParams::denominator] = context.requestTensor(
    denom_dim, "positional_encoding_denominator", nntrainer::Initializer::NONE,
    false, nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
}

void PositionalEncodingLayer::forwarding(RunLayerContext &context,
                                         bool training) {
  const nntrainer::Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);

  nntrainer::Tensor &pe = context.getTensor(
    tensor_idx[PositionalEncodingParams::positional_encoding]);

  TensorDim input_dim = input.getDim();
  TensorDim pe_partial_dim({input_dim.height(), input_dim.width()},
                           pe.getTensorType());
  nntrainer::Tensor pe_partial = pe.getSharedDataTensor(pe_partial_dim, 0);

  calculatePositionalEncoding(
    pe_partial,
    context.getTensor(tensor_idx[PositionalEncodingParams::denominator]));

  input.add(pe_partial, output);
}

void PositionalEncodingLayer::calcDerivative(RunLayerContext &context) {
  const nntrainer::Tensor &incoming_derivative =
    context.getIncomingDerivative(SINGLE_INOUT_IDX);
  nntrainer::Tensor &outgoing_derivative =
    context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  outgoing_derivative.copyData(incoming_derivative);
}

void PositionalEncodingLayer::setProperty(
  const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, positional_encoding_props);
  NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
    << "[positional encoding layer] Unknown Layer Properties count " +
         std::to_string(values.size());
}

void PositionalEncodingLayer::exportTo(
  Exporter &exporter, const ml::train::ExportMethods &method) const {
  exporter.saveResult(positional_encoding_props, method, this);
}

} /* namespace nntrainer */
