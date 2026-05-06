// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   qwen35_layers.cpp
 * @brief  Missing Qwen3.5 token mixer helper layers.
 */

#include <qwen35_layers.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <nntrainer_error.h>

namespace causallm {

namespace {

static constexpr size_t SINGLE_INOUT_IDX = 0;
static constexpr float DEFAULT_L2_EPSILON = 1.0e-6f;

void require_fp32(const nntrainer::Tensor &tensor, const std::string &name) {
  NNTR_THROW_IF(tensor.getDataType() != ml::train::TensorDim::DataType::FP32,
                std::invalid_argument)
    << name << " supports FP32 tensors only";
}

void normalize_sequence_range(unsigned int &from, unsigned int &to,
                              const nntrainer::Tensor &input) {
  const unsigned int step_size = to - from;
  NNTR_THROW_IF(step_size > input.height(), std::invalid_argument)
    << "incremental step size is larger than local input height";
  from = 0;
  to = step_size;
}

} // namespace

ReshapedL2NormLayer::ReshapedL2NormLayer() :
  props_(props::FeatureSize(), nntrainer::props::Epsilon()),
  feature_size(0) {}

void ReshapedL2NormLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "reshaped_l2_norm takes one input";

  const auto &input_dim = context.getInputDimensions()[SINGLE_INOUT_IDX];
  feature_size = std::get<props::FeatureSize>(props_).get();
  NNTR_THROW_IF(input_dim.width() % feature_size != 0, std::invalid_argument)
    << "feature_size should divide input width";

  context.setOutputDimensions(context.getInputDimensions());
}

void ReshapedL2NormLayer::forwarding(nntrainer::RunLayerContext &context,
                                     bool training) {
  incremental_forwarding(context, 0, context.getInput(SINGLE_INOUT_IDX).height(),
                         training);
}

void ReshapedL2NormLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  nntrainer::Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  require_fp32(input, getType());
  require_fp32(output, getType());
  normalize_sequence_range(from, to, input);

  const float eps = std::get<nntrainer::props::Epsilon>(props_).empty()
                      ? DEFAULT_L2_EPSILON
                      : std::get<nntrainer::props::Epsilon>(props_).get();
  const unsigned int width = input.width();

  for (unsigned int b = 0; b < input.batch(); ++b) {
    for (unsigned int c = 0; c < input.channel(); ++c) {
      for (unsigned int h = from; h < to; ++h) {
        const float *src = input.getData<float>() + input.getIndex(b, c, h, 0);
        float *dst = output.getData<float>() + output.getIndex(b, c, h, 0);
        for (unsigned int offset = 0; offset < width; offset += feature_size) {
          float sum = 0.0f;
          for (unsigned int i = 0; i < feature_size; ++i) {
            const float v = src[offset + i];
            sum += v * v;
          }
          const float scale = 1.0f / std::sqrt(sum + eps);
          for (unsigned int i = 0; i < feature_size; ++i) {
            dst[offset + i] = src[offset + i] * scale;
          }
        }
      }
    }
  }
}

void ReshapedL2NormLayer::calcDerivative(
  nntrainer::RunLayerContext &context) {
  throw nntrainer::exception::not_supported(
    "calcDerivative for reshaped_l2_norm is not supported");
}

void ReshapedL2NormLayer::setProperty(
  const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, props_);
  NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
    << "[reshaped_l2_norm] Unknown layer properties";
}

void ReshapedL2NormLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  context.updateInput(SINGLE_INOUT_IDX, input_dimensions[0]);
  context.updateOutput(SINGLE_INOUT_IDX, input_dimensions[0]);
}

FeatureBiasLayer::FeatureBiasLayer() :
  bias_idx(std::numeric_limits<unsigned int>::max()) {}

void FeatureBiasLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "feature_bias takes one input";

  const auto &input_dim = context.getInputDimensions()[SINGLE_INOUT_IDX];
  context.setOutputDimensions(context.getInputDimensions());

  auto weight_type = nntrainer::TensorDim::TensorType(
    context.getFormat(), context.getWeightDataType());
  bias_idx = context.requestWeight(
    nntrainer::TensorDim(1, 1, 1, input_dim.width(), weight_type),
    nntrainer::Initializer::NONE, nntrainer::WeightRegularizer::NONE, 1.0f,
    0.0f, "bias", false);
}

void FeatureBiasLayer::forwarding(nntrainer::RunLayerContext &context,
                                  bool training) {
  incremental_forwarding(context, 0, context.getInput(SINGLE_INOUT_IDX).height(),
                         training);
}

void FeatureBiasLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  nntrainer::Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &bias = context.getWeight(bias_idx);
  require_fp32(input, getType());
  require_fp32(output, getType());
  require_fp32(bias, getType());
  normalize_sequence_range(from, to, input);

  const float *b_data = bias.getData<float>();
  for (unsigned int b = 0; b < input.batch(); ++b) {
    for (unsigned int c = 0; c < input.channel(); ++c) {
      for (unsigned int h = from; h < to; ++h) {
        const float *src = input.getData<float>() + input.getIndex(b, c, h, 0);
        float *dst = output.getData<float>() + output.getIndex(b, c, h, 0);
        for (unsigned int w = 0; w < input.width(); ++w)
          dst[w] = src[w] + b_data[w];
      }
    }
  }
}

void FeatureBiasLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  throw nntrainer::exception::not_supported(
    "calcDerivative for feature_bias is not supported");
}

void FeatureBiasLayer::setProperty(const std::vector<std::string> &values) {
  NNTR_THROW_IF(!values.empty(), std::invalid_argument)
    << "[feature_bias] Unknown layer properties";
}

void FeatureBiasLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  context.updateInput(SINGLE_INOUT_IDX, input_dimensions[0]);
  context.updateOutput(SINGLE_INOUT_IDX, input_dimensions[0]);
}

FeatureScaleLayer::FeatureScaleLayer() :
  scale_idx(std::numeric_limits<unsigned int>::max()) {}

void FeatureScaleLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "feature_scale takes one input";

  const auto &input_dim = context.getInputDimensions()[SINGLE_INOUT_IDX];
  context.setOutputDimensions(context.getInputDimensions());

  auto weight_type = nntrainer::TensorDim::TensorType(
    context.getFormat(), context.getWeightDataType());
  scale_idx = context.requestWeight(
    nntrainer::TensorDim(1, 1, 1, input_dim.width(), weight_type),
    nntrainer::Initializer::NONE, nntrainer::WeightRegularizer::NONE, 1.0f,
    0.0f, "scale", false);
}

void FeatureScaleLayer::forwarding(nntrainer::RunLayerContext &context,
                                   bool training) {
  incremental_forwarding(context, 0, context.getInput(SINGLE_INOUT_IDX).height(),
                         training);
}

void FeatureScaleLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  nntrainer::Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &scale = context.getWeight(scale_idx);
  require_fp32(input, getType());
  require_fp32(output, getType());
  require_fp32(scale, getType());
  normalize_sequence_range(from, to, input);

  const float *s_data = scale.getData<float>();
  for (unsigned int b = 0; b < input.batch(); ++b) {
    for (unsigned int c = 0; c < input.channel(); ++c) {
      for (unsigned int h = from; h < to; ++h) {
        const float *src = input.getData<float>() + input.getIndex(b, c, h, 0);
        float *dst = output.getData<float>() + output.getIndex(b, c, h, 0);
        for (unsigned int w = 0; w < input.width(); ++w)
          dst[w] = src[w] * s_data[w];
      }
    }
  }
}

void FeatureScaleLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  throw nntrainer::exception::not_supported(
    "calcDerivative for feature_scale is not supported");
}

void FeatureScaleLayer::setProperty(const std::vector<std::string> &values) {
  NNTR_THROW_IF(!values.empty(), std::invalid_argument)
    << "[feature_scale] Unknown layer properties";
}

void FeatureScaleLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  context.updateInput(SINGLE_INOUT_IDX, input_dimensions[0]);
  context.updateOutput(SINGLE_INOUT_IDX, input_dimensions[0]);
}

HeadPairSplitLayer::HeadPairSplitLayer() :
  props_(props::FeatureSize(), props::Qwen35PairSelectIndex()),
  feature_size(0),
  select_index(0) {}

void HeadPairSplitLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "qwen35_head_pair_split takes one input";

  const auto &input_dim = context.getInputDimensions()[SINGLE_INOUT_IDX];
  feature_size = std::get<props::FeatureSize>(props_).get();
  select_index = std::get<props::Qwen35PairSelectIndex>(props_).get();
  NNTR_THROW_IF(select_index > 1, std::invalid_argument)
    << "select_index should be 0 or 1";
  NNTR_THROW_IF(input_dim.width() % (feature_size * 2) != 0,
                std::invalid_argument)
    << "input width should be divisible by feature_size * 2";

  auto output_dim = input_dim;
  output_dim.width(input_dim.width() / 2);
  context.setOutputDimensions({output_dim});
}

void HeadPairSplitLayer::forwarding(nntrainer::RunLayerContext &context,
                                    bool training) {
  incremental_forwarding(context, 0, context.getInput(SINGLE_INOUT_IDX).height(),
                         training);
}

void HeadPairSplitLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  nntrainer::Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  require_fp32(input, getType());
  require_fp32(output, getType());
  normalize_sequence_range(from, to, input);

  const unsigned int output_width = output.width();
  for (unsigned int b = 0; b < input.batch(); ++b) {
    for (unsigned int c = 0; c < input.channel(); ++c) {
      for (unsigned int h = from; h < to; ++h) {
        const float *src = input.getData<float>() + input.getIndex(b, c, h, 0);
        float *dst = output.getData<float>() + output.getIndex(b, c, h, 0);
        for (unsigned int out_offset = 0; out_offset < output_width;
             out_offset += feature_size) {
          const unsigned int pair_offset = out_offset * 2;
          const unsigned int src_offset =
            pair_offset + select_index * feature_size;
          std::copy(src + src_offset, src + src_offset + feature_size,
                    dst + out_offset);
        }
      }
    }
  }
}

void HeadPairSplitLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  throw nntrainer::exception::not_supported(
    "calcDerivative for qwen35_head_pair_split is not supported");
}

void HeadPairSplitLayer::setProperty(
  const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, props_);
  NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
    << "[qwen35_head_pair_split] Unknown layer properties";
}

void HeadPairSplitLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  context.updateInput(SINGLE_INOUT_IDX, input_dimensions[0]);
  auto output_dim = input_dimensions[0];
  output_dim.width(input_dimensions[0].width() / 2);
  context.updateOutput(SINGLE_INOUT_IDX, output_dim);
}

Qwen35CausalDepthwiseConv1DLayer::Qwen35CausalDepthwiseConv1DLayer() :
  wt_idx(), tensor_idx(), props_(props::Qwen35LinearConvKernelDim()) {
  wt_idx.fill(std::numeric_limits<unsigned int>::max());
  tensor_idx.fill(std::numeric_limits<unsigned int>::max());
}

void Qwen35CausalDepthwiseConv1DLayer::finalize(
  nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "qwen35_causal_depthwise_conv1d takes one input";

  const auto &input_dim = context.getInputDimensions()[SINGLE_INOUT_IDX];
  const unsigned int kernel =
    std::get<props::Qwen35LinearConvKernelDim>(props_).get();

  context.setOutputDimensions(context.getInputDimensions());

  auto tensor_type = nntrainer::TensorDim::TensorType(
    context.getFormat(), context.getActivationDataType());
  auto weight_type = nntrainer::TensorDim::TensorType(
    context.getFormat(), context.getWeightDataType());

  wt_idx[ConvWeight] = context.requestWeight(
    nntrainer::TensorDim(1, 1, input_dim.width(), kernel, weight_type),
    nntrainer::Initializer::NONE, nntrainer::WeightRegularizer::NONE, 1.0f,
    0.0f, "weight", true);
  tensor_idx[ConvState] = context.requestTensor(
    nntrainer::TensorDim(input_dim.batch(), 1, input_dim.width(), kernel,
                         tensor_type),
    "conv_state", nntrainer::Initializer::ZEROS, false,
    nntrainer::TensorLifespan::MAX_LIFESPAN);
}

void Qwen35CausalDepthwiseConv1DLayer::forwarding(
  nntrainer::RunLayerContext &context, bool training) {
  incremental_forwarding(context, 0, context.getInput(SINGLE_INOUT_IDX).height(),
                         training);
}

void Qwen35CausalDepthwiseConv1DLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  nntrainer::Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &weight = context.getWeight(wt_idx[ConvWeight]);
  nntrainer::Tensor &state = context.getTensor(tensor_idx[ConvState]);
  require_fp32(input, getType());
  require_fp32(output, getType());
  require_fp32(weight, getType());
  require_fp32(state, getType());
  normalize_sequence_range(from, to, input);

  const unsigned int width = input.width();
  const unsigned int kernel =
    std::get<props::Qwen35LinearConvKernelDim>(props_).get();
  const float *w_data = weight.getData<float>();

  for (unsigned int b = 0; b < input.batch(); ++b) {
    float *state_b =
      state.getData<float>() + static_cast<size_t>(b) * state.getDim().getFeatureLen();
    for (unsigned int h = from; h < to; ++h) {
      const float *src = input.getData<float>() + input.getIndex(b, 0, h, 0);
      float *dst = output.getData<float>() + output.getIndex(b, 0, h, 0);
      for (unsigned int c = 0; c < width; ++c) {
        float *channel_state = state_b + static_cast<size_t>(c) * kernel;
        for (unsigned int k = 0; k + 1 < kernel; ++k)
          channel_state[k] = channel_state[k + 1];
        channel_state[kernel - 1] = src[c];

        const float *kernel_w = w_data + static_cast<size_t>(c) * kernel;
        float sum = 0.0f;
        for (unsigned int k = 0; k < kernel; ++k)
          sum += channel_state[k] * kernel_w[k];
        dst[c] = sum;
      }
    }
  }
}

void Qwen35CausalDepthwiseConv1DLayer::calcDerivative(
  nntrainer::RunLayerContext &context) {
  throw nntrainer::exception::not_supported(
    "calcDerivative for qwen35_causal_depthwise_conv1d is not supported");
}

void Qwen35CausalDepthwiseConv1DLayer::setProperty(
  const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, props_);
  NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
    << "[qwen35_causal_depthwise_conv1d] Unknown layer properties";
}

void Qwen35CausalDepthwiseConv1DLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  context.updateInput(SINGLE_INOUT_IDX, input_dimensions[0]);
  context.updateOutput(SINGLE_INOUT_IDX, input_dimensions[0]);
}

Qwen35GatedDeltaCoreLayer::Qwen35GatedDeltaCoreLayer() :
  tensor_idx(),
  props_(props::Qwen35LinearNumKeyHeads(),
         props::Qwen35LinearNumValueHeads(),
         props::Qwen35LinearKeyHeadDim(),
         props::Qwen35LinearValueHeadDim()) {
  tensor_idx.fill(std::numeric_limits<unsigned int>::max());
}

void Qwen35GatedDeltaCoreLayer::finalize(
  nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != InputCount, std::invalid_argument)
    << "qwen35_gated_delta_core takes query, key, value, beta, and decay";

  const unsigned int num_k_heads =
    std::get<props::Qwen35LinearNumKeyHeads>(props_).get();
  const unsigned int num_v_heads =
    std::get<props::Qwen35LinearNumValueHeads>(props_).get();
  const unsigned int key_head_dim =
    std::get<props::Qwen35LinearKeyHeadDim>(props_).get();
  const unsigned int value_head_dim =
    std::get<props::Qwen35LinearValueHeadDim>(props_).get();

  const auto &query_dim = context.getInputDimensions()[Query];
  const auto &key_dim = context.getInputDimensions()[Key];
  const auto &value_dim = context.getInputDimensions()[Value];
  const auto &beta_dim = context.getInputDimensions()[Beta];
  const auto &decay_dim = context.getInputDimensions()[Decay];

  NNTR_THROW_IF(query_dim.width() != num_k_heads * key_head_dim ||
                  key_dim.width() != num_k_heads * key_head_dim,
                std::invalid_argument)
    << "query/key width does not match linear key head configuration";
  NNTR_THROW_IF(value_dim.width() != num_v_heads * value_head_dim,
                std::invalid_argument)
    << "value width does not match linear value head configuration";
  NNTR_THROW_IF(beta_dim.width() != num_v_heads ||
                  decay_dim.width() != num_v_heads,
                std::invalid_argument)
    << "beta/decay width should match number of value heads";

  context.setOutputDimensions({value_dim});

  auto tensor_type = nntrainer::TensorDim::TensorType(
    context.getFormat(), context.getActivationDataType());
  tensor_idx[RecurrentState] = context.requestTensor(
    nntrainer::TensorDim(query_dim.batch(), num_v_heads, key_head_dim,
                         value_head_dim, tensor_type),
    "recurrent_state", nntrainer::Initializer::ZEROS, false,
    nntrainer::TensorLifespan::MAX_LIFESPAN);
}

void Qwen35GatedDeltaCoreLayer::forwarding(
  nntrainer::RunLayerContext &context, bool training) {
  incremental_forwarding(context, 0, context.getInput(Query).height(),
                         training);
}

void Qwen35GatedDeltaCoreLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  nntrainer::Tensor &query = context.getInput(Query);
  nntrainer::Tensor &key = context.getInput(Key);
  nntrainer::Tensor &value = context.getInput(Value);
  nntrainer::Tensor &beta = context.getInput(Beta);
  nntrainer::Tensor &decay = context.getInput(Decay);
  nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &state = context.getTensor(tensor_idx[RecurrentState]);
  require_fp32(query, getType());
  require_fp32(key, getType());
  require_fp32(value, getType());
  require_fp32(beta, getType());
  require_fp32(decay, getType());
  require_fp32(output, getType());
  require_fp32(state, getType());
  normalize_sequence_range(from, to, query);

  const unsigned int num_k_heads =
    std::get<props::Qwen35LinearNumKeyHeads>(props_).get();
  const unsigned int num_v_heads =
    std::get<props::Qwen35LinearNumValueHeads>(props_).get();
  const unsigned int key_head_dim =
    std::get<props::Qwen35LinearKeyHeadDim>(props_).get();
  const unsigned int value_head_dim =
    std::get<props::Qwen35LinearValueHeadDim>(props_).get();
  const unsigned int head_repeat = num_v_heads / num_k_heads;
  const float query_scale = 1.0f / std::sqrt(static_cast<float>(key_head_dim));

  for (unsigned int b = 0; b < query.batch(); ++b) {
    float *state_b =
      state.getData<float>() + static_cast<size_t>(b) * state.getDim().getFeatureLen();
    for (unsigned int h = from; h < to; ++h) {
      const float *q_base = query.getData<float>() + query.getIndex(b, 0, h, 0);
      const float *k_base = key.getData<float>() + key.getIndex(b, 0, h, 0);
      const float *v_base = value.getData<float>() + value.getIndex(b, 0, h, 0);
      const float *beta_base =
        beta.getData<float>() + beta.getIndex(b, 0, h, 0);
      const float *decay_base =
        decay.getData<float>() + decay.getIndex(b, 0, h, 0);
      float *out_base = output.getData<float>() + output.getIndex(b, 0, h, 0);

      for (unsigned int vh = 0; vh < num_v_heads; ++vh) {
        const unsigned int kh = vh / head_repeat;
        const float *q = q_base + kh * key_head_dim;
        const float *k = k_base + kh * key_head_dim;
        const float *v = v_base + vh * value_head_dim;
        float *out = out_base + vh * value_head_dim;
        float *head_state =
          state_b + static_cast<size_t>(vh) * key_head_dim * value_head_dim;
        const float g = std::exp(decay_base[vh]);

        for (unsigned int kd = 0; kd < key_head_dim; ++kd) {
          float *row = head_state + static_cast<size_t>(kd) * value_head_dim;
          for (unsigned int vd = 0; vd < value_head_dim; ++vd)
            row[vd] *= g;
        }

        for (unsigned int vd = 0; vd < value_head_dim; ++vd) {
          float kv_mem = 0.0f;
          for (unsigned int kd = 0; kd < key_head_dim; ++kd)
            kv_mem += head_state[static_cast<size_t>(kd) * value_head_dim +
                                 vd] *
                      k[kd];
          const float delta = (v[vd] - kv_mem) * beta_base[vh];
          for (unsigned int kd = 0; kd < key_head_dim; ++kd) {
            head_state[static_cast<size_t>(kd) * value_head_dim + vd] +=
              k[kd] * delta;
          }
        }

        for (unsigned int vd = 0; vd < value_head_dim; ++vd) {
          float sum = 0.0f;
          for (unsigned int kd = 0; kd < key_head_dim; ++kd)
            sum += head_state[static_cast<size_t>(kd) * value_head_dim + vd] *
                   q[kd] * query_scale;
          out[vd] = sum;
        }
      }
    }
  }
}

void Qwen35GatedDeltaCoreLayer::calcDerivative(
  nntrainer::RunLayerContext &context) {
  throw nntrainer::exception::not_supported(
    "calcDerivative for qwen35_gated_delta_core is not supported");
}

void Qwen35GatedDeltaCoreLayer::setProperty(
  const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, props_);
  NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
    << "[qwen35_gated_delta_core] Unknown layer properties";
}

void Qwen35GatedDeltaCoreLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  for (unsigned int i = 0; i < InputCount; ++i)
    context.updateInput(i, input_dimensions[i]);
  context.updateOutput(SINGLE_INOUT_IDX, input_dimensions[Value]);
}

} // namespace causallm
