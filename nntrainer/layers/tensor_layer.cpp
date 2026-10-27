// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   tensor_layer.cpp
 * @date   17 Jan 2025
 * @brief  This is QNN Tensor Layer Class of Neural Network
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <iostream>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <string>
#include <tensor_layer.h>
#include <tensor_wrap_specs.h>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

TensorLayer::TensorLayer() : Layer(), tensor_props({}, {}, {}, {}) {}

void TensorLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, tensor_props);
  NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
    << "[TensorLayer] Unknown Layer Properties count " +
         std::to_string(values.size());
}

void TensorLayer::finalize(InitLayerContext &context) {
  auto &dims = std::get<std::vector<props::TensorDimension>>(tensor_props);
  std::vector<TensorDim> t_dims(dims.begin(), dims.end());

  auto &t_dtype = std::get<std::vector<props::TensorDataType>>(tensor_props);
  auto &t_name = std::get<std::vector<props::TensorName>>(tensor_props);
  auto &t_life = std::get<std::vector<props::TensorLife>>(tensor_props);

  NNTR_THROW_IF(t_dims.empty(), std::invalid_argument)
    << "Tensor dimension is not provided";
  n_tensor = t_dims.size();

  if (t_dtype.empty()) {
    ml_logi("Set Activation Tensor DataType");
    t_dtype.reserve(t_dims.size());
    for (auto t : t_dims)
      t_dtype.push_back(context.getActivationDataType());
  }

  if (t_life.empty()) {
    ml_logi("Set max Tensor LifeSpan");
    t_life.reserve(t_dims.size());
    for (auto t : t_dims)
      t_life.push_back(nntrainer::TensorLifespan::MAX_LIFESPAN);
  }

  auto engine = context.getComputeEngineType();

  NNTR_THROW_IF((t_dims.size() != t_dtype.size() ||
                 t_dims.size() != t_life.size() ||
                 t_dims.size() != t_name.size()),
                std::invalid_argument)
    << "Size of Dimensions, Types, Formats should be matched!";

  tensor_idx.reserve(t_dims.size());

  for (unsigned int i = 0; i < t_dims.size(); ++i) {
    t_dims[i].setFormat(context.getFormat());
    t_dims[i].setDataType(t_dtype[i]);
    std::string name = context.getName() + "_t" + std::to_string(i);
    if (!t_name.empty())
      name = t_name[i];

    tensor_idx.push_back(context.requestTensor(
      t_dims[i], name, Initializer::NONE, true, t_life[i], true, engine));
  }

  context.setOutputDimensions(t_dims);
}

void TensorLayer::forwarding(RunLayerContext &context, bool training) {
  if (!context.getInPlace()) {
    for (unsigned int i = 0; i < n_tensor; ++i) {
      Tensor &input_ = context.getInput(i);
      Tensor &hidden_ = context.getOutput(i);
      hidden_.copyData(input_);
    }
  }
}

void TensorLayer::calcDerivative(RunLayerContext &context) {
  throw exception::not_supported(
    "calcDerivative for input layer is not supported");
}

void TensorLayer::exportTo(Exporter &exporter,
                           const ml::train::ExportMethods &method) const {
  exporter.saveResult(tensor_props, method, this);
}

} /* namespace nntrainer */
