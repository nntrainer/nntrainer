// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file node_exporter.cpp
 * @date 09 April 2021
 * @brief NNTrainer Node exporter
 * @see	https://github.com/nntrainer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <node_exporter.h>

#ifdef ENABLE_TFLITE_INTERPRETER
#include <activation_layer.h>
#include <bitset>
#include <common_properties.h>
#include <fc_layer.h>
#include <map>
#include <node_exporter.h>
#include <tf_schema_generated.h>
#include <tflite_opnode.h>
#endif

namespace {

#ifdef ENABLE_TFLITE_INTERPRETER
tflite::Padding tflite_padding(const std::string &padding) {
  std::map<std::string, tflite::Padding> m = {{"same", tflite::Padding_SAME},
                                              {"valid", tflite::Padding_VALID}};
  return m[padding];
}
#endif

} // namespace

namespace nntrainer {

constexpr const unsigned int CONV2D_DIM = 2;
constexpr const unsigned int POOLING2D_DIM = 2;

/**
 * @brief Construct a new Exporter object
 *
 */
Exporter::Exporter() : stored_result(nullptr), is_exported(false) {
#ifdef ENABLE_TFLITE_INTERPRETER
  tf_node = nullptr;
  fbb = nullptr;
#endif
}

#ifdef ENABLE_TFLITE_INTERPRETER
/**
 * @brief Construct a new Exporter object with flatbuffer builder
 *
 */
Exporter::Exporter(flatbuffers::FlatBufferBuilder *fbb) :
  fbb(fbb), stored_result(nullptr), is_exported(false) {}
#endif

/**
 * @brief Destroy the Exporter object
 *
 */
Exporter::~Exporter() = default;

template <>
std::unique_ptr<std::vector<std::pair<std::string, std::string>>>
Exporter::getResult<ml::train::ExportMethods::METHOD_STRINGVECTOR>() {
  return std::move(stored_result);
}

#ifdef ENABLE_TFLITE_INTERPRETER
template <>
std::unique_ptr<TfOpNode>
Exporter::getResult<ml::train::ExportMethods::METHOD_TFLITE>() {
  tf_node->finalize();
  return std::move(tf_node);
}

template <>
void Exporter::saveTflResult(const std::tuple<> &props,
                             const nntrainer::Layer *self) {
  createIfNull(tf_node);
}

template <>
void Exporter::saveTflResult(
  const std::tuple<props::Name, props::Distribute, props::Trainable,
                   std::vector<props::InputConnection>,
                   std::vector<props::InputShape>, props::WeightDimension,
                   props::SharedFrom, props::ClipGradByGlobalNorm,
                   props::Packed, props::WeightDtype, props::LossScaleForMixed,
                   props::ComputeEngine> &props,
  const LayerNode *self) {
  createIfNull(tf_node);
  tf_node->setLayerNode(*self);
}

template <>
void Exporter::saveTflResult(
  const std::tuple<props::WeightRegularizer, props::WeightRegularizerConstant,
                   props::WeightInitializer, props::WeightDecay,
                   props::BiasDecay, props::BiasInitializer, props::DisableBias,
                   props::Print> &props,
  const LayerImpl *self) { /// layer impl has nothing to serialize so do nothing
}

template <>
void Exporter::saveTflResult(
  const std::tuple<props::Unit, props::LoraRank, props::LoraAlpha> &props,
  const FullyConnectedLayer *self) {
  createIfNull(tf_node);
  tf_node->setOpType(tflite::BuiltinOperator_FULLY_CONNECTED);
  auto options = tflite::CreateFullyConnectedOptions(*fbb).Union();
  tf_node->setBuiltinOptions(tflite::BuiltinOptions_FullyConnectedOptions,
                             options);
}

template <>
void Exporter::saveTflResult(const std::tuple<props::Activation> &props,
                             const ActivationLayer *self) {
  createIfNull(tf_node);

  auto activation = std::get<props::Activation>(props);
  switch (activation.get()) {
  case ActivationType::ACT_RELU: {
    tf_node->setOpType(tflite::BuiltinOperator_RELU);
    tf_node->setBuiltinOptions(tflite::BuiltinOptions_NONE,
                               flatbuffers::Offset<void>() /** no options **/);
    break;
  }
  case ActivationType::ACT_SOFTMAX: {
    tf_node->setOpType(tflite::BuiltinOperator_SOFTMAX);
    auto options = tflite::CreateSoftmaxOptions(*fbb, 1.0).Union();
    tf_node->setBuiltinOptions(tflite::BuiltinOptions_SoftmaxOptions, options);
    break;
  }
  default:
    throw std::runtime_error{"Unsupported activation type"};
  }
}

template <>
void Exporter::saveTflResult(
  const std::tuple<props::Epsilon, props::MuInitializer, props::VarInitializer,
                   props::BetaInitializer, props::GammaInitializer,
                   props::Momentum, props::Axis, props::WeightDecay,
                   props::BiasDecay> &props,
  const BatchNormalizationLayer *self) {
  createIfNull(tf_node);

  auto epsilon = std::get<props::Epsilon>(props).get();
  tf_node->AppendAdditionalProps(epsilon);

  tf_node->setOpType(tflite::BuiltinOperator_MUL);
  auto options =
    tflite::CreateMulOptions(*fbb, tflite::ActivationFunctionType_NONE).Union();
  tf_node->setBuiltinOptions(tflite::BuiltinOptions_MulOptions, options);
}

template <>
void Exporter::saveTflResult(
  const std::tuple<props::FilterSize, std::array<props::KernelSize, CONV2D_DIM>,
                   std::array<props::Stride, CONV2D_DIM>, props::Padding2D,
                   std::array<props::Dilation, CONV2D_DIM>> &props,
  const Conv2DLayer *self) {
  createIfNull(tf_node);

  auto weight_transform = [](std::vector<const Tensor *> &old_weights) {
    std::vector<Tensor> new_weights;

    auto &filter_weight = *old_weights[0];
    // tflite filter has shape format {channel_out, height, width, channel_in}
    Tensor filter(filter_weight.transpose("1:2:0"));
    new_weights.push_back(filter);

    auto &bias_weight = *old_weights[1];
    TensorDim bias_dim{bias_weight.getTensorType(), std::bitset<4>(0b0001)};
    bias_dim.setTensorDim(
      3 /** index **/,
      bias_weight
        .channel() /** value **/); // effective dimension = {bias->channel()}
    Tensor bias(bias_dim);
    bias.copyData(bias_weight.transpose("1:2:0"));
    bias.setName(bias_weight.getName());

    new_weights.push_back(bias);

    return new_weights;
  };
  tf_node->setWeightTransformFn(weight_transform);

  tf_node->setOpType(tflite::BuiltinOperator_CONV_2D);

  auto &strides = std::get<std::array<props::Stride, CONV2D_DIM>>(props);
  assert(strides.size() == CONV2D_DIM);
  const auto &padding = std::get<props::Padding2D>(props).get();
  if (padding != "same" && padding != "valid") {
    std::ostringstream ss;
    ss << "Unsupported padding type; \"" << padding
       << "\" is not supported. Use \"same\" or \"valid\".";
    throw std::runtime_error(ss.str());
  }
  auto options = tflite::CreateConv2DOptions(*fbb, tflite_padding(padding),
                                             strides.at(0), strides.at(1))
                   .Union();

  tf_node->AppendProps(tflite_padding(padding));
  tf_node->AppendProps(strides.at(0));
  tf_node->AppendProps(strides.at(1));

  tf_node->setBuiltinOptions(tflite::BuiltinOptions_Conv2DOptions, options);
}

template <>
void Exporter::saveTflResult(
  const std::tuple<props::Normalization, props::Standardization> &props,
  const InputLayer *self) {
  createIfNull(tf_node);
  // input layer exports to Transpose operator (NCHW -> NHWC)
  tf_node->setOpType(tflite::BuiltinOperator_TRANSPOSE);
  tf_node->setBuiltinOptions(tflite::BuiltinOptions_TransposeOptions,
                             flatbuffers::Offset<void>());

  auto input_transform = [](std::vector<const Tensor *> &inputs) {
    std::vector<Tensor> new_inputs;
    assert(inputs.size() == 1);
    new_inputs.reserve(inputs.size() + 1 /** perm **/);
    new_inputs.push_back(*inputs[0]);
    // create "perm" tensor for Transpose operator
    // @todo : This NCHW format setting is just temporal, it needs to be set by
    //  global configuration
    TensorDim perm_dim{inputs[0]->getTensorType(), std::bitset<4>(0b0001)};
    perm_dim.setTensorDim(3 /** index **/,
                          4 /** value **/); // effective dimension = {4}
    new_inputs.emplace_back(perm_dim);
    auto &perm = new_inputs.back();
    perm.setName("nntrainer_internal_perm");
    perm.setValueInt(0, 0 /** N **/);
    perm.setValueInt(1, 2 /** H **/);
    perm.setValueInt(2, 3 /** W **/);
    perm.setValueInt(3, 1 /** C **/);
    return new_inputs;
  };
  tf_node->setInputTransformFn(input_transform);

  assert(tf_node->getOutputs().size() == 1);
  auto output_tensor = const_cast<Tensor *>(tf_node->getOutputs()[0]);
  // Transpose op needs buffer
  output_tensor->allocate();
}

template <>
void Exporter::saveTflResult(
  const std::tuple<props::PoolingType, std::vector<props::PoolSize>,
                   std::array<props::Stride, POOLING2D_DIM>, props::Padding2D>
    &props,
  const Pooling2DLayer *self) {
  createIfNull(tf_node);

  auto poolingType = std::get<props::PoolingType>(props);
  auto strides = std::get<std::array<props::Stride, POOLING2D_DIM>>(props);
  assert(strides.size() == POOLING2D_DIM);
  auto poolSize = std::get<std::vector<props::PoolSize>>(props);
  assert(poolSize.size() == POOLING2D_DIM);
  const auto &padding = std::get<props::Padding2D>(props).get();
  assert(padding == "same" || padding == "valid");

  switch (poolingType.get()) {
  case props::PoolingTypeInfo::Enum::average: {
    tf_node->setOpType(tflite::BuiltinOperator_AVERAGE_POOL_2D);
    auto options =
      tflite::CreatePool2DOptions(*fbb, tflite_padding(padding), strides.at(0),
                                  strides.at(1), poolSize.at(0), poolSize.at(1))
        .Union();
    tf_node->setBuiltinOptions(tflite::BuiltinOptions_Pool2DOptions, options);
    break;
  }
  default:
    throw std::runtime_error{"Unsupported pooling type"};
  }
}

template <>
void Exporter::saveTflResult(const std::tuple<props::TargetShape> &props,
                             const ReshapeLayer *self) {
  createIfNull(tf_node);

  tf_node->setOpType(tflite::BuiltinOperator_RESHAPE);
  const auto &targetShape = std::get<props::TargetShape>(props).get();
  std::vector<int32_t> new_shape_vec = {
    static_cast<int32_t>(targetShape.batch()),
    static_cast<int32_t>(targetShape.height()),
    static_cast<int32_t>(targetShape.width()),
    static_cast<int32_t>(targetShape.channel())};
  auto new_shape = fbb->CreateVector(new_shape_vec);
  auto options = tflite::CreateReshapeOptions(*fbb, new_shape).Union();
  tf_node->setBuiltinOptions(tflite::BuiltinOptions_ReshapeOptions, options);
}

template <>
void Exporter::saveTflResult(const std::tuple<props::TargetShape> &props,
                             const FlattenLayer *self) {
  createIfNull(tf_node);

  tf_node->setOpType(tflite::BuiltinOperator_RESHAPE);
  auto &targetShape = std::get<props::TargetShape>(props).get();

  /// @todo new shape should be 2 rank {batch, channel * height * width}
  std::vector<int32_t> new_shape_vec = {
    static_cast<int32_t>(targetShape.batch()),
    static_cast<int32_t>(targetShape.height()),
    static_cast<int32_t>(targetShape.width()),
    static_cast<int32_t>(targetShape.channel())};
  auto new_shape = fbb->CreateVector(new_shape_vec);
  auto options = tflite::CreateReshapeOptions(*fbb, new_shape).Union();
  tf_node->setBuiltinOptions(tflite::BuiltinOptions_ReshapeOptions, options);
}

template <>
void Exporter::saveTflResult(const std::tuple<> &props,
                             const AdditionLayer *self) {
  createIfNull(tf_node);

  tf_node->setOpType(tflite::BuiltinOperator_ADD);
  auto options = tflite::CreateAddOptions(*fbb).Union();
  tf_node->setBuiltinOptions(tflite::BuiltinOptions_AddOptions, options);
}
#endif

} // namespace nntrainer
