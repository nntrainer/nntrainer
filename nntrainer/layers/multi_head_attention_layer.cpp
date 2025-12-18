// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 hyeonseok Lee <hs89.lee@samsung.com>
 *
 * @file   multi_head_attention_layer.cpp
 * @date   08 July 2022
 * @see    https://github.com/nnstreamer/nntrainer
 *         https://arxiv.org/abs/1706.03762
 * @author hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is MultiHeadAttention Layer Class for Neural Network
 *
 */

#include <cmath>

#include <layer_context.h>
#include <multi_head_attention_layer.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>

namespace nntrainer {

MultiHeadAttentionLayer::MultiHeadAttentionLayer() :
  multi_head_attention_props(
    props::NumHeads(), props::ProjectedKeyDim(), props::ProjectedValueDim(),
    props::OutputShape(), props::DropOutRate(), props::ReturnAttentionWeight(),
    props::AverageAttentionWeight()),
  sm(ActivationType::ACT_SOFTMAX),
  epsilon(1e-3f) {
  weight_idx.fill(std::numeric_limits<unsigned>::max());
}

MultiHeadAttentionLayer::~MultiHeadAttentionLayer() {}

enum INOUT_INDEX {
  /** input index */
  QUERY = 0,
  KEY = 1,
  VALUE = 2,
  MASK = 3,
  /** output index */
  OUTPUT = 0,
  RETURN_ATTENTION_WEIGHT = 1,
};

enum AttentionParams {
  query_fc_weight,
  query_fc_bias,
  key_fc_weight,
  key_fc_bias,
  value_fc_weight,
  value_fc_bias,
  fc_weight,
  fc_bias,
  projected_query,
  projected_key,
  projected_value,
  cache_key,
  cache_value,
  /** intended comment for later use of attention_mask */
  // attention_mask,
  attention_weight,
  dropout_mask,
  attention_output,
};

void MultiHeadAttentionLayer::finalize(InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() < 3 || context.getNumInputs() > 4,
                std::invalid_argument)
    << "Multi head Attention layer needs 3 or 4 inputs. (query, key, value and "
       "mask is optional";
  const bool provide_attention_mask = context.getNumInputs() == 4;

  TensorDim::TensorType weight_type = {context.getFormat(),
                                       context.getWeightDataType()};

  TensorDim::TensorType activation_type = {context.getFormat(),
                                           context.getActivationDataType()};

  TensorDim empty_dim(activation_type);

  const std::vector<TensorDim> &input_dims = context.getInputDimensions();
  const TensorDim &query_dim = input_dims[INOUT_INDEX::QUERY];
  const TensorDim &key_dim = input_dims[INOUT_INDEX::KEY];
  const TensorDim &value_dim = input_dims[INOUT_INDEX::VALUE];
  const TensorDim &mask_dim =
    provide_attention_mask ? input_dims[INOUT_INDEX::MASK] : empty_dim;

  const unsigned int batch_size = query_dim.batch();
  const unsigned int query_height = query_dim.height();
  const unsigned int query_width = query_dim.width();
  const unsigned int key_height = key_dim.height();
  const unsigned int key_width = key_dim.width();
  const unsigned int value_height = value_dim.height();
  const unsigned int value_width = value_dim.width();

  const bool disable_bias =
    std::get<props::DisableBias>(*layer_impl_props).get();
  auto &weight_initializer =
    std::get<props::WeightInitializer>(*layer_impl_props).get();
  auto &weight_regularizer =
    std::get<props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<props::WeightRegularizerConstant>(*layer_impl_props);
  const float &weight_decay =
    std::get<props::WeightDecay>(*layer_impl_props).get();

  NNTR_THROW_IF(std::get<props::NumHeads>(multi_head_attention_props).empty(),
                std::invalid_argument)
    << "num_heads property is not provided for layer " << context.getName();
  const unsigned int num_heads =
    std::get<props::NumHeads>(multi_head_attention_props).get();

  if (std::get<props::ProjectedKeyDim>(multi_head_attention_props).empty()) {
    NNTR_THROW_IF(query_width % num_heads, std::invalid_argument)
      << "query_width: " << query_width
      << " is not divisible by num_heads: " << num_heads << " for layer "
      << context.getName();

    ml_logw("[multi head attention] ProjectedKeyDim property is not given. "
            "Default value(query_width / num_heads) is set");

    std::get<props::ProjectedKeyDim>(multi_head_attention_props)
      .set(query_width / num_heads);
  }
  const unsigned int projected_key_dim_prop =
    std::get<props::ProjectedKeyDim>(multi_head_attention_props).get();

  if (std::get<props::ProjectedValueDim>(multi_head_attention_props).empty()) {
    std::get<props::ProjectedValueDim>(multi_head_attention_props)
      .set(projected_key_dim_prop);
  }
  const unsigned int projected_value_dim_prop =
    std::get<props::ProjectedValueDim>(multi_head_attention_props).get();

  if (std::get<props::OutputShape>(multi_head_attention_props).empty()) {
    std::get<props::OutputShape>(multi_head_attention_props).set(query_width);
  }
  const unsigned int output_shape =
    std::get<props::OutputShape>(multi_head_attention_props).get();

  const float dropout_rate =
    std::get<props::DropOutRate>(multi_head_attention_props).get();

  if (std::get<props::AverageAttentionWeight>(multi_head_attention_props)
        .empty()) {
    std::get<props::AverageAttentionWeight>(multi_head_attention_props)
      .set(true);
  }
  const bool average_attention_weight =
    std::get<props::AverageAttentionWeight>(multi_head_attention_props).get();

  const props::ReturnAttentionWeightInfo::Enum return_attention_weight =
    std::get<props::ReturnAttentionWeight>(multi_head_attention_props).get();

  const unsigned int projected_query_dim_prop = projected_key_dim_prop;

  if (activation_type.data_type == TensorDim::DataType::FP32) {
    sm.setActiFunc(ActivationType::ACT_SOFTMAX);
  } else if (activation_type.data_type == TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    sm.setActiFunc<_FP16>(ActivationType::ACT_SOFTMAX);
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  }

  // sm.setActiFunc(ActivationType::ACT_SOFTMAX);

  NNTR_THROW_IF(query_dim.channel() != 1, std::invalid_argument)
    << "Dimension of input query channel: " << query_dim.channel()
    << " is not 1 for layer " << context.getName();
  NNTR_THROW_IF(key_dim.channel() != 1, std::invalid_argument)
    << "Dimension of input key channel: " << key_dim.channel()
    << " is not 1 for layer " << context.getName();
  NNTR_THROW_IF(value_dim.channel() != 1, std::invalid_argument)
    << "Dimension of input value channel: " << value_dim.channel()
    << " is not 1 for layer " << context.getName();
  NNTR_THROW_IF(provide_attention_mask && mask_dim.channel() != num_heads,
                std::invalid_argument)
    << "Dimension of input mask channel: " << mask_dim.channel()
    << " is not matched with num_heads property: " << num_heads << " for layer "
    << context.getName();

  NNTR_THROW_IF(key_height != value_height, std::invalid_argument)
    << "Dimension of input key height: " << key_height
    << " is not matched with Dimension of input value height: " << value_height
    << " for layer " << context.getName();
  NNTR_THROW_IF(provide_attention_mask && mask_dim.height() != query_height,
                std::invalid_argument)
    << "Dimension of input mask height: " << mask_dim.height()
    << " is not matched with Dimension of input query height: " << query_height
    << " for layer " << context.getName();

  NNTR_THROW_IF(provide_attention_mask && mask_dim.width() != key_height,
                std::invalid_argument)
    << "Dimension of input mask width: " << mask_dim.width()
    << " is not matched with Dimension of input key height: " << key_height
    << " for layer " << context.getName();

  /** weight/bias for query fc */
  TensorDim query_fc_weight_dim(
    {1, 1, query_width, num_heads * projected_query_dim_prop}, weight_type);

  weight_idx[AttentionParams::query_fc_weight] = context.requestWeight(
    query_fc_weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "query_fc_weight", true);
  if (!disable_bias) {
    TensorDim query_fc_bias_dim({1, 1, 1, num_heads * projected_query_dim_prop},
                                weight_type);
    weight_idx[AttentionParams::query_fc_bias] = context.requestWeight(
      query_fc_bias_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay, "query_fc_bias", true);
  }

  /** weight/bias for key fc */
  TensorDim key_fc_weight_dim(
    {1, 1, key_width, num_heads * projected_key_dim_prop}, weight_type);
  weight_idx[AttentionParams::key_fc_weight] = context.requestWeight(
    key_fc_weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "key_fc_weight", true);
  if (!disable_bias) {
    TensorDim key_fc_bias_dim({1, 1, 1, num_heads * projected_key_dim_prop},
                              weight_type);
    weight_idx[AttentionParams::key_fc_bias] = context.requestWeight(
      key_fc_bias_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay, "key_fc_bias", true);
  }

  /** weight/bias for value fc */
  TensorDim value_fc_weight_dim(
    {1, 1, value_width, num_heads * projected_value_dim_prop}, weight_type);
  weight_idx[AttentionParams::value_fc_weight] = context.requestWeight(
    value_fc_weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "value_fc_weight", true);
  if (!disable_bias) {
    TensorDim value_fc_bias_dim({1, 1, 1, num_heads * projected_value_dim_prop},
                                weight_type);
    weight_idx[AttentionParams::value_fc_bias] = context.requestWeight(
      value_fc_bias_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay, "value_fc_bias", true);
  }

  /** weight/bias for out fc */
  TensorDim fc_weight_dim(
    {1, 1, num_heads * projected_value_dim_prop, output_shape}, weight_type);
  weight_idx[AttentionParams::fc_weight] = context.requestWeight(
    fc_weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "fc_weight", true);
  if (!disable_bias) {
    TensorDim fc_bias_dim({1, 1, 1, output_shape}, weight_type);
    weight_idx[AttentionParams::fc_bias] = context.requestWeight(
      fc_bias_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay, "fc_bias", true);
  }

  /** tensor for output of query fc */
  TensorDim projected_query_dim(
    {batch_size, 1, query_height, num_heads * projected_query_dim_prop},
    activation_type);
  weight_idx[AttentionParams::projected_query] = context.requestTensor(
    projected_query_dim, "projected_query", Initializer::NONE, true,
    TensorLifespan::ITERATION_LIFESPAN);
  /** tensor for output of key fc */
  TensorDim projected_key_dim(
    {batch_size, 1, key_height, num_heads * projected_key_dim_prop},
    activation_type);
  weight_idx[AttentionParams::projected_key] =
    context.requestTensor(projected_key_dim, "projected_key", Initializer::NONE,
                          true, TensorLifespan::ITERATION_LIFESPAN);
  /** tensor for output of value fc */
  TensorDim projected_value_dim(
    {batch_size, 1, value_height, num_heads * projected_value_dim_prop},
    activation_type);
  weight_idx[AttentionParams::projected_value] = context.requestTensor(
    projected_value_dim, "projected_value", Initializer::NONE, true,
    TensorLifespan::ITERATION_LIFESPAN);

  weight_idx[AttentionParams::cache_key] =
    context.requestTensor(projected_key_dim, "cache_key", Initializer::NONE,
                          true, TensorLifespan::MAX_LIFESPAN);

  weight_idx[AttentionParams::cache_value] =
    context.requestTensor(projected_value_dim, "cache_value", Initializer::NONE,
                          true, TensorLifespan::MAX_LIFESPAN);

  if (provide_attention_mask) {
    /** Intended comment for bool type mask */
    // TensorDim attention_mask_dim(
    //   {batch_size, num_heads, query_height, key_height});
    // weight_idx[AttentionParams::attention_mask] = context.requestTensor(
    //   attention_mask_dim, "attention_mask", Initializer::NONE, false,
    //   TensorLifespan::FORWARD_FUNC_LIFESPAN);
  }
  /** tensor for attention weight */
  TensorDim attention_weight_dim(
    {batch_size, num_heads, query_height, key_height}, activation_type);
  weight_idx[AttentionParams::attention_weight] = context.requestTensor(
    attention_weight_dim, "attention_weight", Initializer::NONE, true,
    TensorLifespan::ITERATION_LIFESPAN);
  if (dropout_rate > epsilon) {
    /** tensor for dropout mask */
    TensorDim dropout_mask_dim(
      {batch_size, num_heads, query_height, key_height}, activation_type);
    weight_idx[AttentionParams::dropout_mask] =
      context.requestTensor(dropout_mask_dim, "dropout_mask", Initializer::NONE,
                            false, TensorLifespan::ITERATION_LIFESPAN);
  }

  /** tensor for attention output */
  TensorDim attention_output_dim(
    {batch_size, 1, query_height, num_heads * projected_value_dim_prop},
    activation_type);
  weight_idx[AttentionParams::attention_output] = context.requestTensor(
    attention_output_dim, "attention_output", Initializer::NONE, true,
    TensorLifespan::ITERATION_LIFESPAN);

  TensorDim output_dim({batch_size, 1, query_height, output_shape},
                       activation_type);
  if (return_attention_weight != props::ReturnAttentionWeightInfo::Enum::none) {
    TensorDim return_attention_weight_dim(
      {batch_size, average_attention_weight ? 1 : num_heads, query_height,
       key_height},
      activation_type);
    context.setOutputDimensions({output_dim, return_attention_weight_dim});
  } else {
    context.setOutputDimensions({output_dim});
  }
}

void MultiHeadAttentionLayer::forwarding(RunLayerContext &context,
                                         bool training) {
  const bool disable_bias =
    std::get<props::DisableBias>(*layer_impl_props).get();

  const unsigned int num_heads =
    std::get<props::NumHeads>(multi_head_attention_props).get();
  const unsigned int projected_key_dim_prop =
    std::get<props::ProjectedKeyDim>(multi_head_attention_props).get();
  const unsigned int projected_value_dim_prop =
    std::get<props::ProjectedValueDim>(multi_head_attention_props).get();
  const float dropout_rate =
    std::get<props::DropOutRate>(multi_head_attention_props).get();
  const props::ReturnAttentionWeightInfo::Enum return_attention_weight =
    std::get<props::ReturnAttentionWeight>(multi_head_attention_props).get();
  const bool average_attention_weight =
    std::get<props::AverageAttentionWeight>(multi_head_attention_props).get();

  const bool provide_attention_mask = context.getNumInputs() == 4;
  const unsigned int projected_query_dim_prop = projected_key_dim_prop;
  const bool enable_dropout = dropout_rate > epsilon;

  Tensor empty_tensor;

  /** get inputs/outputs */
  Tensor &query = context.getInput(INOUT_INDEX::QUERY);
  Tensor &key = context.getInput(INOUT_INDEX::KEY);
  Tensor &value = context.getInput(INOUT_INDEX::VALUE);
  Tensor &mask =
    provide_attention_mask ? context.getInput(INOUT_INDEX::MASK) : empty_tensor;

  Tensor &output = context.getOutput(INOUT_INDEX::OUTPUT);
  Tensor &ret_attention_weight =
    return_attention_weight != props::ReturnAttentionWeightInfo::Enum::none
      ? context.getOutput(INOUT_INDEX::RETURN_ATTENTION_WEIGHT)
      : empty_tensor;

  /** get weights */
  Tensor &query_fc_weight =
    context.getWeight(weight_idx[AttentionParams::query_fc_weight]);
  Tensor &query_fc_bias =
    disable_bias
      ? empty_tensor
      : context.getWeight(weight_idx[AttentionParams::query_fc_bias]);
  Tensor &key_fc_weight =
    context.getWeight(weight_idx[AttentionParams::key_fc_weight]);
  Tensor &key_fc_bias =
    disable_bias ? empty_tensor
                 : context.getWeight(weight_idx[AttentionParams::key_fc_bias]);
  Tensor &value_fc_weight =
    context.getWeight(weight_idx[AttentionParams::value_fc_weight]);
  Tensor &value_fc_bias =
    disable_bias
      ? empty_tensor
      : context.getWeight(weight_idx[AttentionParams::value_fc_bias]);
  Tensor &fc_weight = context.getWeight(weight_idx[AttentionParams::fc_weight]);
  Tensor &fc_bias = disable_bias
                      ? empty_tensor
                      : context.getWeight(weight_idx[AttentionParams::fc_bias]);

  /** get tensors */
  Tensor &projected_query =
    context.getTensor(weight_idx[AttentionParams::projected_query]);
  Tensor &projected_key =
    context.getTensor(weight_idx[AttentionParams::projected_key]);
  Tensor &projected_value =
    context.getTensor(weight_idx[AttentionParams::projected_value]);

  Tensor &attention_weight =
    context.getTensor(weight_idx[AttentionParams::attention_weight]);
  Tensor &attention_output =
    context.getTensor(weight_idx[AttentionParams::attention_output]);

  const TensorDim query_dim = query.getDim();
  const unsigned int batch_size = query_dim.batch();
  const unsigned int query_height = query_dim.height();
  const TensorDim key_dim = key.getDim();
  const unsigned int key_height = key_dim.height();
  const TensorDim value_dim = value.getDim();
  const unsigned int value_height = value_dim.height();

  query.dot(query_fc_weight, projected_query);
  if (!disable_bias) {
    projected_query.add_i(query_fc_bias);
  }
  key.dot(key_fc_weight, projected_key);
  if (!disable_bias) {
    projected_key.add_i(key_fc_bias);
  }
  value.dot(value_fc_weight, projected_value);
  if (!disable_bias) {
    projected_value.add_i(value_fc_bias);
  }

  projected_query.reshape(
    TensorDim({batch_size, query_height, num_heads, projected_query_dim_prop}));
  projected_key.reshape(
    TensorDim({batch_size, key_height, num_heads, projected_key_dim_prop}));
  projected_value.reshape(
    TensorDim({batch_size, value_height, num_heads, projected_value_dim_prop}));

  projected_query = projected_query.transpose("1:0:2");
  projected_key = projected_key.transpose("1:0:2");
  projected_value = projected_value.transpose("1:0:2");

  /** set tensor name to restore origin name cause origin name was remove during
   * transpose */
  projected_query.setName("multi_head_attention:projected_query");
  projected_key.setName("multi_head_attention:projected_key");
  projected_value.setName("multi_head_attention:projected_value");

  projected_query.reshape(TensorDim(
    {batch_size * num_heads, 1, query_height, projected_query_dim_prop}));
  projected_key.reshape(
    TensorDim({batch_size * num_heads, 1, key_height, projected_key_dim_prop}));
  projected_value.reshape(TensorDim(
    {batch_size * num_heads, 1, value_height, projected_value_dim_prop}));

  attention_weight.reshape(
    TensorDim({batch_size * num_heads, 1, query_height, key_height}));
  attention_output.reshape(TensorDim(
    {batch_size * num_heads, 1, query_height, projected_value_dim_prop}));

  /** scaled dot product attention */
  projected_query.dotBatched(projected_key, attention_weight, false, true);
  attention_weight.multiply_i(1.0f /
                              std::sqrt((float)projected_query_dim_prop));

  if (provide_attention_mask) {
    // Tensor &attention_mask =
    //   context.getTensor(weight_idx[AttentionParams::attention_mask]);
    /** @todo: enable bool type tensor */
    // if (torch_ref) {
    //   attention_mask.setValue(-1e9);
    // } else {
    //   // flip
    //   attention_mask.setValue(1);
    //   attention_mask.subtract_i(mask);

    //   attention_mask.multiply_i(-1e9);
    // }
    // attention_mask.multiply_i(mask);
    // attention_weight.add_i(attention_mask);

    attention_weight.reshape(
      TensorDim({batch_size, num_heads, query_height, key_height}));
    attention_weight.add_i(mask);
    attention_weight.reshape(
      TensorDim({batch_size * num_heads, 1, query_height, key_height}));
  }

  sm.run_fn(attention_weight, attention_weight);

  if (return_attention_weight ==
      props::ReturnAttentionWeightInfo::Enum::before) {
    ret_attention_weight.copyData(attention_weight);
  }

  if (enable_dropout) {
    Tensor &dropout_mask =
      context.getTensor(weight_idx[AttentionParams::dropout_mask]);
    dropout_mask.dropout_mask(dropout_rate);
    attention_weight.multiply_i(dropout_mask);
  }

  if (return_attention_weight ==
      props::ReturnAttentionWeightInfo::Enum::after) {
    if (average_attention_weight) {
      attention_weight.reshape(
        TensorDim({batch_size, num_heads, query_height, key_height}));
      attention_weight.sum(1, ret_attention_weight, 1, 0);
      ret_attention_weight.divide_i(static_cast<float>(num_heads));
      attention_weight.reshape(
        TensorDim({batch_size * num_heads, 1, query_height, key_height}));
    } else {
      ret_attention_weight.copyData(attention_weight);
    }
  }

  attention_weight.dotBatched(projected_value, attention_output);

  attention_output.reshape(
    TensorDim({batch_size, num_heads, query_height, projected_value_dim_prop}));

  attention_output = attention_output.transpose("1:0:2");

  /** set tensor name to restore origin name cause origin name was remove during
   * transpose */
  attention_output.setName("multi_head_attention:attention_output");

  attention_output.reshape(TensorDim(
    {batch_size * query_height, 1, 1, num_heads * projected_value_dim_prop}));

  attention_output.dot(fc_weight, output);
  if (!disable_bias) {
    output.add_i(fc_bias);
  }

  /** restore shape */
  projected_query.reshape(TensorDim(
    {batch_size, 1, query_height, num_heads * projected_query_dim_prop}));
  projected_key.reshape(
    TensorDim({batch_size, 1, key_height, num_heads * projected_key_dim_prop}));
  projected_value.reshape(TensorDim(
    {batch_size, 1, value_height, num_heads * projected_value_dim_prop}));

  attention_weight.reshape(
    TensorDim({batch_size, num_heads, query_height, key_height}));
  attention_output.reshape(TensorDim(
    {batch_size, 1, query_height, num_heads * projected_value_dim_prop}));
}

void MultiHeadAttentionLayer::incremental_forwarding(RunLayerContext &context,
                                                     unsigned int from,
                                                     unsigned int to,
                                                     bool training) {
  const bool disable_bias =
    std::get<props::DisableBias>(*layer_impl_props).get();

  const unsigned int num_heads =
    std::get<props::NumHeads>(multi_head_attention_props).get();
  const unsigned int projected_key_dim_prop =
    std::get<props::ProjectedKeyDim>(multi_head_attention_props).get();
  const unsigned int projected_value_dim_prop =
    std::get<props::ProjectedValueDim>(multi_head_attention_props).get();
  const float dropout_rate =
    std::get<props::DropOutRate>(multi_head_attention_props).get();
  const props::ReturnAttentionWeightInfo::Enum return_attention_weight =
    std::get<props::ReturnAttentionWeight>(multi_head_attention_props).get();
  const bool average_attention_weight =
    std::get<props::AverageAttentionWeight>(multi_head_attention_props).get();

  const bool provide_attention_mask = context.getNumInputs() == 4;
  const unsigned int projected_query_dim_prop = projected_key_dim_prop;
  const bool enable_dropout = dropout_rate > epsilon;

  /** get inputs/outputs */
  Tensor &query = context.getInput(INOUT_INDEX::QUERY);
  Tensor &key = context.getInput(INOUT_INDEX::KEY);
  Tensor &value = context.getInput(INOUT_INDEX::VALUE);

  Tensor empty_tensor("empty", value.getFormat(), value.getDataType());

  Tensor &mask =
    provide_attention_mask ? context.getInput(INOUT_INDEX::MASK) : empty_tensor;

  TensorDim query_dim = query.getDim();
  TensorDim key_dim = key.getDim();
  TensorDim value_dim = value.getDim();

  Tensor &output = context.getOutput(INOUT_INDEX::OUTPUT);

  TensorDim output_dim = output.getDim();
  Tensor &ret_attention_weight =
    return_attention_weight != props::ReturnAttentionWeightInfo::Enum::none
      ? context.getOutput(INOUT_INDEX::RETURN_ATTENTION_WEIGHT)
      : empty_tensor;

  /** get weights */
  Tensor &query_fc_weight =
    context.getWeight(weight_idx[AttentionParams::query_fc_weight]);
  Tensor &query_fc_bias =
    disable_bias
      ? empty_tensor
      : context.getWeight(weight_idx[AttentionParams::query_fc_bias]);
  Tensor &key_fc_weight =
    context.getWeight(weight_idx[AttentionParams::key_fc_weight]);
  Tensor &key_fc_bias =
    disable_bias ? empty_tensor
                 : context.getWeight(weight_idx[AttentionParams::key_fc_bias]);
  Tensor &value_fc_weight =
    context.getWeight(weight_idx[AttentionParams::value_fc_weight]);
  Tensor &value_fc_bias =
    disable_bias
      ? empty_tensor
      : context.getWeight(weight_idx[AttentionParams::value_fc_bias]);
  Tensor &fc_weight = context.getWeight(weight_idx[AttentionParams::fc_weight]);
  Tensor &fc_bias = disable_bias
                      ? empty_tensor
                      : context.getWeight(weight_idx[AttentionParams::fc_bias]);

  /** get tensors */
  Tensor &projected_query =
    context.getTensor(weight_idx[AttentionParams::projected_query]);
  Tensor &projected_key =
    context.getTensor(weight_idx[AttentionParams::projected_key]);
  Tensor &projected_value =
    context.getTensor(weight_idx[AttentionParams::projected_value]);
  Tensor &cache_key = context.getTensor(weight_idx[AttentionParams::cache_key]);
  Tensor &cache_value =
    context.getTensor(weight_idx[AttentionParams::cache_value]);

  TensorDim projected_query_dim = projected_query.getDim();
  TensorDim projected_key_dim = projected_key.getDim();
  TensorDim projected_value_dim = projected_value.getDim();
  TensorDim cache_key_dim = cache_key.getDim();
  TensorDim cache_value_dim = cache_value.getDim();

  TensorDim projected_query_step_dim = projected_query_dim;

  TensorDim projected_key_step_dim = projected_key_dim;
  TensorDim projected_value_step_dim = projected_value_dim;
  TensorDim cache_key_step_dim = cache_key_dim;
  TensorDim cache_value_step_dim = cache_value_dim;
  projected_query_step_dim.height(to - from);

  projected_key_step_dim.height(to);
  projected_value_step_dim.height(to);
  cache_key_step_dim.height(to - from);
  cache_value_step_dim.height(to - from);

  Tensor projected_query_step =
    projected_query.getSharedDataTensor(projected_query_step_dim, 0, true);
  Tensor projected_key_step =
    projected_key.getSharedDataTensor(projected_key_step_dim, 0, true);
  Tensor projected_value_step =
    projected_value.getSharedDataTensor(projected_value_step_dim, 0, true);

  Tensor cache_key_step = cache_key.getSharedDataTensor(
    cache_key_step_dim, from * cache_key_dim.width(), true);
  Tensor cache_value_step = cache_value.getSharedDataTensor(
    cache_value_step_dim, from * cache_value_dim.width(), true);

  TensorDim cached_key_dim = {cache_key_dim.batch(), cache_key_dim.channel(),
                              to, cache_key_dim.width(),
                              cache_key.getTensorType()};
  TensorDim cached_value_dim = {
    cache_value_dim.batch(), cache_value_dim.channel(), to,
    cache_value_dim.width(), cache_value.getTensorType()};
  Tensor cached_key = cache_key.getSharedDataTensor(cached_key_dim, 0, true);
  Tensor cached_value =
    cache_value.getSharedDataTensor(cached_value_dim, 0, true);

  Tensor &attention_weight =
    context.getTensor(weight_idx[AttentionParams::attention_weight]);
  Tensor &attention_output =
    context.getTensor(weight_idx[AttentionParams::attention_output]);
  TensorDim attention_weight_dim = attention_weight.getDim();

  TensorDim attention_weight_step_dim = attention_weight_dim;
  attention_weight_step_dim.height(to - from);
  attention_weight_step_dim.width(to);

  Tensor attention_weight_step =
    attention_weight.getSharedDataTensor(attention_weight_step_dim, 0, true);

  TensorDim attention_output_dim = attention_output.getDim();
  TensorDim attention_output_step_dim = attention_output_dim;
  attention_output_step_dim.height(to - from);

  Tensor attention_output_step =
    attention_output.getSharedDataTensor(attention_output_step_dim, 0, true);

  const unsigned int batch_size = query_dim.batch();
  const unsigned int query_height = query_dim.height();
  const unsigned int key_height = key_dim.height();
  const unsigned int value_height = value_dim.height();

  query.dot(query_fc_weight, projected_query_step);
  if (!disable_bias) {
    projected_query_step.add_i(query_fc_bias);
  }
  key.dot(key_fc_weight, cache_key_step);
  if (!disable_bias) {
    cache_key_step.add_i(key_fc_bias);
  }
  value.dot(value_fc_weight, cache_value_step);
  if (!disable_bias) {
    cache_value_step.add_i(value_fc_bias);
  }

  projected_query_step.reshape(
    TensorDim({batch_size, 1, num_heads, projected_query_dim_prop}));
  cached_key.reshape(
    TensorDim({batch_size, to, num_heads, projected_key_dim_prop}));
  cached_value.reshape(
    TensorDim({batch_size, to, num_heads, projected_value_dim_prop}));

  projected_query_step.transpose("1:0:2", projected_query_step);
  cached_key.transpose("1:0:2", projected_key_step);
  cached_value.transpose("1:0:2", projected_value_step);

  projected_query_step.reshape(
    TensorDim({batch_size * num_heads, 1, 1, projected_query_dim_prop}));
  projected_key_step.reshape(
    TensorDim({batch_size * num_heads, 1, to, projected_key_dim_prop}));
  projected_value_step.reshape(
    TensorDim({batch_size * num_heads, 1, to, projected_value_dim_prop}));

  attention_weight_step.reshape(TensorDim({batch_size * num_heads, 1, 1, to}));
  attention_output_step.reshape(
    TensorDim({batch_size * num_heads, 1, 1, projected_value_dim_prop}));

  /** scaled dot product attention */
  projected_query_step.dotBatched(projected_key_step, attention_weight_step,
                                  false, true);
  attention_weight_step.multiply_i(1 / sqrt((float)projected_query_dim_prop));

  if (!from) {
    unsigned int mask_size = attention_weight_step.getDim().width();
    unsigned int mask_dim_height = mask_size;
    unsigned int mask_dim_width = mask_size;

    Tensor causal_mask(TensorDim{1, 1, mask_size, mask_size,
                                 attention_weight_step.getTensorType()});

    causal_mask.setZero();

#ifdef ENABLE_FP16
#define _MASK_NUM -1e4
#else
#define _MASK_NUM -1e10
#endif

    for (unsigned int i = 0; i < mask_dim_height; ++i) {
      for (unsigned int j = i + 1; j < mask_dim_width; ++j) {
        causal_mask.setValue(0, 0, i, j, _MASK_NUM);
      }
    }

    attention_weight_step.add_i(causal_mask);
  }

  sm.run_fn(attention_weight_step, attention_weight_step);

  attention_weight_step.dotBatched(projected_value_step, attention_output_step);

  attention_output_step.reshape(
    TensorDim({batch_size, num_heads, to - from, projected_value_dim_prop}));

  attention_output_step = attention_output_step.transpose("1:0:2");

  attention_output_step.reshape(TensorDim(
    {batch_size * (to - from), 1, 1, num_heads * projected_value_dim_prop}));

  attention_output_step.dot(fc_weight, output);
  if (!disable_bias) {
    output.add_i(fc_bias);
  }
}

void MultiHeadAttentionLayer::calcCommonDerivative(RunLayerContext &context) {
  const unsigned int num_heads =
    std::get<props::NumHeads>(multi_head_attention_props).get();
  const unsigned int projected_key_dim_prop =
    std::get<props::ProjectedKeyDim>(multi_head_attention_props).get();
  const unsigned int projected_value_dim_prop =
    std::get<props::ProjectedValueDim>(multi_head_attention_props).get();
  const float dropout_rate =
    std::get<props::DropOutRate>(multi_head_attention_props).get();
  const props::ReturnAttentionWeightInfo::Enum return_attention_weight =
    std::get<props::ReturnAttentionWeight>(multi_head_attention_props).get();
  const bool average_attention_weight =
    std::get<props::AverageAttentionWeight>(multi_head_attention_props).get();

  const bool provide_attention_mask = context.getNumInputs() == 4;
  const unsigned int projected_query_dim_prop = projected_key_dim_prop;

  Tensor empty_tensor;

  Tensor &query = context.getInput(INOUT_INDEX::QUERY);
  Tensor &key = context.getInput(INOUT_INDEX::KEY);
  Tensor &value = context.getInput(INOUT_INDEX::VALUE);
  const Tensor &incoming_derivative =
    context.getIncomingDerivative(INOUT_INDEX::OUTPUT);
  const Tensor &d_ret_attention_weight =
    return_attention_weight != props::ReturnAttentionWeightInfo::Enum::none
      ? context.getIncomingDerivative(INOUT_INDEX::RETURN_ATTENTION_WEIGHT)
      : empty_tensor;

  Tensor &fc_weight = context.getWeight(weight_idx[AttentionParams::fc_weight]);

  Tensor &projected_query =
    context.getTensor(weight_idx[AttentionParams::projected_query]);
  Tensor &d_projected_query =
    context.getTensorGrad(weight_idx[AttentionParams::projected_query]);
  Tensor &projected_key =
    context.getTensor(weight_idx[AttentionParams::projected_key]);
  Tensor &d_projected_key =
    context.getTensorGrad(weight_idx[AttentionParams::projected_key]);
  Tensor &projected_value =
    context.getTensor(weight_idx[AttentionParams::projected_value]);
  Tensor &d_projected_value =
    context.getTensorGrad(weight_idx[AttentionParams::projected_value]);

  Tensor &attention_weight =
    context.getTensor(weight_idx[AttentionParams::attention_weight]);
  Tensor &d_attention_weight =
    context.getTensorGrad(weight_idx[AttentionParams::attention_weight]);
  Tensor &d_attention_output =
    context.getTensorGrad(weight_idx[AttentionParams::attention_output]);

  const TensorDim query_dim = query.getDim();
  const unsigned int batch_size = query_dim.batch();
  const unsigned int query_height = query_dim.height();
  const TensorDim key_dim = key.getDim();
  const unsigned int key_height = key_dim.height();
  const TensorDim value_dim = value.getDim();
  const unsigned int value_height = value_dim.height();

  d_attention_output.dot_deriv_wrt_1(fc_weight, incoming_derivative);

  d_attention_output.reshape(
    TensorDim({batch_size, query_height, num_heads, projected_value_dim_prop}));

  d_attention_output = d_attention_output.transpose("1:0:2");

  /** set tensor name to restore origin name cause origin name was remove
   * during transpose */
  d_attention_output.setName("multi_head_attention:attention_output:grad");

  projected_query.reshape(TensorDim(
    {batch_size * num_heads, 1, query_height, projected_query_dim_prop}));
  d_projected_query.reshape(TensorDim(
    {batch_size * num_heads, 1, query_height, projected_query_dim_prop}));
  projected_key.reshape(
    TensorDim({batch_size * num_heads, 1, key_height, projected_key_dim_prop}));
  d_projected_key.reshape(
    TensorDim({batch_size * num_heads, 1, key_height, projected_key_dim_prop}));
  projected_value.reshape(TensorDim(
    {batch_size * num_heads, 1, value_height, projected_value_dim_prop}));
  d_projected_value.reshape(TensorDim(
    {batch_size * num_heads, 1, value_height, projected_value_dim_prop}));

  attention_weight.reshape(
    TensorDim({batch_size * num_heads, 1, query_height, key_height}));
  d_attention_weight.reshape(
    TensorDim({batch_size * num_heads, 1, query_height, key_height}));
  d_attention_output.reshape(TensorDim(
    {batch_size * num_heads, 1, query_height, projected_value_dim_prop}));

  d_attention_weight.dot_batched_deriv_wrt_1(projected_value,
                                             d_attention_output);
  attention_weight.dot_batched_deriv_wrt_2(d_projected_value,
                                           d_attention_output);

  if (return_attention_weight ==
      props::ReturnAttentionWeightInfo::Enum::after) {
    const float scale = average_attention_weight ? 1 / (float)num_heads : 1;
    d_attention_weight.add_i(d_ret_attention_weight, scale);
  }

  if (dropout_rate > epsilon) {
    Tensor &dropout_mask =
      context.getTensor(weight_idx[AttentionParams::dropout_mask]);
    d_attention_weight.multiply_i(dropout_mask);
  }

  if (return_attention_weight ==
      props::ReturnAttentionWeightInfo::Enum::before) {
    d_attention_weight.add_i(d_ret_attention_weight);
  }

  sm.run_prime_fn(attention_weight, d_attention_weight, d_attention_weight);
  if (provide_attention_mask) {
    Tensor &d_mask = context.getOutgoingDerivative(INOUT_INDEX::MASK);
    d_mask.copyData(d_attention_weight);
  }
  d_attention_weight.multiply_i(
    1 / sqrt((float)projected_query_dim_prop)); /** scale */

  d_projected_query.dot_batched_deriv_wrt_1(projected_key, d_attention_weight,
                                            false, true);
  projected_query.dot_batched_deriv_wrt_2(d_projected_key, d_attention_weight,
                                          false, true);

  d_projected_query.reshape(
    TensorDim({batch_size, num_heads, query_height, projected_query_dim_prop}));
  d_projected_key.reshape(
    TensorDim({batch_size, num_heads, key_height, projected_key_dim_prop}));
  d_projected_value.reshape(
    TensorDim({batch_size, num_heads, value_height, projected_value_dim_prop}));

  d_projected_query = d_projected_query.transpose("1:0:2");
  d_projected_key = d_projected_key.transpose("1:0:2");
  d_projected_value = d_projected_value.transpose("1:0:2");

  /** set tensor name to restore origin name cause origin name was remove
   * during transpose */
  d_projected_query.setName("multi_head_attention:projected_query:grad");
  d_projected_key.setName("multi_head_attention:projected_key:grad");
  d_projected_value.setName("multi_head_attention:projected_value:grad");

  /** restore shape */
  projected_query.reshape(TensorDim(
    {batch_size, 1, query_height, num_heads * projected_query_dim_prop}));
  d_projected_query.reshape(TensorDim(
    {batch_size * query_height, 1, 1, num_heads * projected_query_dim_prop}));
  projected_key.reshape(
    TensorDim({batch_size, 1, key_height, num_heads * projected_key_dim_prop}));
  d_projected_key.reshape(TensorDim(
    {batch_size * key_height, 1, 1, num_heads * projected_key_dim_prop}));
  projected_value.reshape(TensorDim(
    {batch_size, 1, value_height, num_heads * projected_value_dim_prop}));
  d_projected_value.reshape(TensorDim(
    {batch_size * value_height, 1, 1, num_heads * projected_value_dim_prop}));

  attention_weight.reshape(
    TensorDim({batch_size, num_heads, query_height, key_height}));
  d_attention_weight.reshape(
    TensorDim({batch_size, num_heads, query_height, key_height}));
  d_attention_output.reshape(TensorDim(
    {batch_size, 1, query_height, num_heads * projected_value_dim_prop}));
}

void MultiHeadAttentionLayer::calcDerivative(RunLayerContext &context) {
  if (!context.getTrainable()) {
    calcCommonDerivative(context);
  }

  Tensor &query = context.getInput(INOUT_INDEX::QUERY);
  Tensor &d_query = context.getOutgoingDerivative(INOUT_INDEX::QUERY);
  Tensor &key = context.getInput(INOUT_INDEX::KEY);
  Tensor &d_key = context.getOutgoingDerivative(INOUT_INDEX::KEY);
  Tensor &value = context.getInput(INOUT_INDEX::VALUE);
  Tensor &d_value = context.getOutgoingDerivative(INOUT_INDEX::VALUE);
  /** d_mask will be calculated in calcCommonDerivative */

  Tensor &query_fc_weight =
    context.getWeight(weight_idx[AttentionParams::query_fc_weight]);
  Tensor &key_fc_weight =
    context.getWeight(weight_idx[AttentionParams::key_fc_weight]);
  Tensor &value_fc_weight =
    context.getWeight(weight_idx[AttentionParams::value_fc_weight]);

  Tensor &d_projected_query =
    context.getTensorGrad(weight_idx[AttentionParams::projected_query]);
  Tensor &d_projected_key =
    context.getTensorGrad(weight_idx[AttentionParams::projected_key]);
  Tensor &d_projected_value =
    context.getTensorGrad(weight_idx[AttentionParams::projected_value]);

  const TensorDim query_dim = query.getDim();
  const TensorDim key_dim = key.getDim();
  const TensorDim value_dim = value.getDim();

  d_query.dot_deriv_wrt_1(query_fc_weight, d_projected_query);
  d_key.dot_deriv_wrt_1(key_fc_weight, d_projected_key);
  d_value.dot_deriv_wrt_1(value_fc_weight, d_projected_value, false, false);
}

void MultiHeadAttentionLayer::calcGradient(RunLayerContext &context) {
  calcCommonDerivative(context);

  const bool disable_bias =
    std::get<props::DisableBias>(*layer_impl_props).get();

  const unsigned int num_heads =
    std::get<props::NumHeads>(multi_head_attention_props).get();
  const unsigned int projected_key_dim_prop =
    std::get<props::ProjectedKeyDim>(multi_head_attention_props).get();
  const unsigned int projected_value_dim_prop =
    std::get<props::ProjectedValueDim>(multi_head_attention_props).get();
  const unsigned int output_shape =
    std::get<props::OutputShape>(multi_head_attention_props).get();

  const unsigned int projected_query_dim_prop = projected_key_dim_prop;

  Tensor &query = context.getInput(INOUT_INDEX::QUERY);
  Tensor &key = context.getInput(INOUT_INDEX::KEY);
  Tensor &value = context.getInput(INOUT_INDEX::VALUE);
  const Tensor &incoming_derivative =
    context.getIncomingDerivative(INOUT_INDEX::OUTPUT);

  Tensor &d_query_fc_weight =
    context.getWeightGrad(weight_idx[AttentionParams::query_fc_weight]);
  Tensor &d_key_fc_weight =
    context.getWeightGrad(weight_idx[AttentionParams::key_fc_weight]);
  Tensor &d_value_fc_weight =
    context.getWeightGrad(weight_idx[AttentionParams::value_fc_weight]);
  Tensor &d_fc_weight =
    context.getWeightGrad(weight_idx[AttentionParams::fc_weight]);

  Tensor empty_tensor;
  Tensor &d_query_fc_bias =
    disable_bias
      ? empty_tensor
      : context.getWeightGrad(weight_idx[AttentionParams::query_fc_bias]);
  Tensor &d_key_fc_bias =
    disable_bias
      ? empty_tensor
      : context.getWeightGrad(weight_idx[AttentionParams::key_fc_bias]);
  Tensor &d_value_fc_bias =
    disable_bias
      ? empty_tensor
      : context.getWeightGrad(weight_idx[AttentionParams::value_fc_bias]);
  Tensor &d_fc_bias =
    disable_bias ? empty_tensor
                 : context.getWeightGrad(weight_idx[AttentionParams::fc_bias]);

  Tensor &d_projected_query =
    context.getTensorGrad(weight_idx[AttentionParams::projected_query]);
  Tensor &d_projected_key =
    context.getTensorGrad(weight_idx[AttentionParams::projected_key]);
  Tensor &d_projected_value =
    context.getTensorGrad(weight_idx[AttentionParams::projected_value]);

  Tensor &attention_output =
    context.getTensor(weight_idx[AttentionParams::attention_output]);

  const TensorDim query_dim = query.getDim();
  const unsigned int batch_size = query_dim.batch();
  const unsigned int query_height = query_dim.height();
  const TensorDim key_dim = key.getDim();
  const unsigned int key_height = key_dim.height();
  const TensorDim value_dim = value.getDim();
  const unsigned int value_height = value_dim.height();

  attention_output.dot_deriv_wrt_2(
    d_fc_weight, incoming_derivative, false, false,
    !context.isGradientFirstAccess(weight_idx[AttentionParams::fc_weight]));

  if (!disable_bias) {
    Tensor incoming_derivative_ = incoming_derivative;
    incoming_derivative_.reshape(
      TensorDim({batch_size * query_height, 1, 1, output_shape}));
    incoming_derivative_.sum(
      0, d_fc_bias, 1,
      !context.isGradientFirstAccess(weight_idx[AttentionParams::fc_bias]));
  }

  query.dot_deriv_wrt_2(d_query_fc_weight, d_projected_query, false, false,
                        !context.isGradientFirstAccess(
                          weight_idx[AttentionParams::query_fc_weight]));
  if (!disable_bias) {
    d_projected_query.reshape(TensorDim(
      {batch_size * query_height, 1, 1, num_heads * projected_query_dim_prop}));
    d_projected_query.sum(0, d_query_fc_bias, 1,
                          !context.isGradientFirstAccess(
                            weight_idx[AttentionParams::query_fc_bias]));
    d_projected_query.reshape(TensorDim(
      {batch_size, 1, query_height, num_heads * projected_query_dim_prop}));
  }

  key.dot_deriv_wrt_2(
    d_key_fc_weight, d_projected_key, false, false,
    !context.isGradientFirstAccess(weight_idx[AttentionParams::key_fc_weight]));
  if (!disable_bias) {
    d_projected_key.reshape(TensorDim(
      {batch_size * key_height, 1, 1, num_heads * projected_key_dim_prop}));
    d_projected_key.sum(
      0, d_key_fc_bias, 1,
      !context.isGradientFirstAccess(weight_idx[AttentionParams::key_fc_bias]));
    d_projected_key.reshape(TensorDim(
      {batch_size, 1, key_height, num_heads * projected_key_dim_prop}));
  }

  value.dot_deriv_wrt_2(d_value_fc_weight, d_projected_value, false, false,
                        !context.isGradientFirstAccess(
                          weight_idx[AttentionParams::value_fc_weight]));
  if (!disable_bias) {
    d_projected_value.reshape(TensorDim(
      {batch_size * value_height, 1, 1, num_heads * projected_value_dim_prop}));
    d_projected_value.sum(0, d_value_fc_bias, 1,
                          !context.isGradientFirstAccess(
                            weight_idx[AttentionParams::value_fc_bias]));
    d_projected_value.reshape(TensorDim(
      {batch_size, 1, value_height, num_heads * projected_value_dim_prop}));
  }
}

void MultiHeadAttentionLayer::setProperty(
  const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, multi_head_attention_props);
  LayerImpl::setProperty(remain_props);
}

void MultiHeadAttentionLayer::setBatch(RunLayerContext &context,
                                       unsigned int batch) {
  const float dropout_rate =
    std::get<props::DropOutRate>(multi_head_attention_props).get();

  context.updateTensor(weight_idx[AttentionParams::projected_query], batch);
  context.updateTensor(weight_idx[AttentionParams::projected_key], batch);
  context.updateTensor(weight_idx[AttentionParams::projected_value], batch);
  context.updateTensor(weight_idx[AttentionParams::cache_key], batch);
  context.updateTensor(weight_idx[AttentionParams::cache_value], batch);
  // context.updateTensor(weight_idx[AttentionParams::cache_value], batch);
  context.updateTensor(weight_idx[AttentionParams::attention_weight], batch);
  if (dropout_rate > epsilon) {
    context.updateTensor(weight_idx[AttentionParams::dropout_mask], batch);
  }
  context.updateTensor(weight_idx[AttentionParams::attention_output], batch);
}

void MultiHeadAttentionLayer::exportTo(
  Exporter &exporter, const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(multi_head_attention_props, method, this);
}

} /* namespace nntrainer */
