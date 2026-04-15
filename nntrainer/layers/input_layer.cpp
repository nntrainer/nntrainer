/**
 * Copyright (C) 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * @file	input_layer.cpp
 * @date	14 May 2020
 * @brief	This is Input Layer Class for Neural Network
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <input_layer.h>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

InputLayer::InputLayer() :
  Layer(), input_props(props::Normalization(), props::Standardization()) {}

void InputLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, input_props);
  NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
    << "[InputLayer] Unknown Layer Properties count " +
         std::to_string(values.size());
}

void InputLayer::forwarding(RunLayerContext &context, bool training) {

  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  std::unique_ptr<nntrainer::Quantizer> quantizer;
  if (!context.getInPlace()) {
    Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
    ///@note: The following code simply copies incoming values (fp32) to the
    // input tensor. Supported types include QINT4, QINT8, UINT8, UINT16,
    // UINT32, FP16, and FP32.
    hidden_.copyData(input_);
  }

  if (std::get<props::Normalization>(input_props))
    hidden_.normalization_i();
  if (std::get<props::Standardization>(input_props))
    hidden_.standardization_i();
}

void InputLayer::calcDerivative(RunLayerContext &context) {
  throw exception::not_supported(
    "calcDerivative for input layer is not supported");
}

void InputLayer::exportTo(Exporter &exporter,
                          const ml::train::ExportMethods &method) const {
  exporter.saveResult(input_props, method, this);
}

void InputLayer::finalize(InitLayerContext &context) {
  TensorDim::DataType input_dtype =
    context.getInputDimensions()[SINGLE_INOUT_IDX].getDataType();
  TensorDim::DataType output_dtype = context.getActivationDataType();

  [[maybe_unused]] auto [output_dims, weight_dims, tensor_dims] =
    getLayerDimensions(context);
  context.setOutputDimensions(output_dims);

  is_inplace = true;
  if (context.getActivationDataType() != ml::train::TensorDim::DataType::FP32)
    is_inplace = false;
}

std::vector<TensorDim>
InputLayer::updateTensorsByInputDimensions(InitLayerContext &init_context,
                                           RunLayerContext &run_context) {
  [[maybe_unused]] auto [output_dims, weight_dims, tensor_dims] =
    getLayerDimensions(init_context);

  run_context.updateInput(SINGLE_INOUT_IDX,
                          init_context.getInputDimensions()[SINGLE_INOUT_IDX]);
  run_context.updateOutput(SINGLE_INOUT_IDX, output_dims[SINGLE_INOUT_IDX]);

  return output_dims;
}

std::array<std::vector<TensorDim>, 3>
InputLayer::getLayerDimensions(InitLayerContext &context) {
  TensorDim::DataType output_dtype = context.getActivationDataType();

  std::vector<TensorDim> output_dims = context.getInputDimensions();
  for (auto &d : output_dims) {
    d.setDataType(output_dtype);
  }

  return {output_dims, {}, {}};
}

} /* namespace nntrainer */
