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
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  ///@note: The following code simply copies incoming values to the output
  // tensor. Supported types include QINT4, QINT8, UINT8, UINT16, UINT32,
  // FP16, and FP32.
  //
  // The copy is skipped only when input and output genuinely share the same
  // backing memory. is_inplace is decided in finalize() purely from matching
  // dims, which assumes the manager aliased input and output. That assumption
  // breaks when the input is an external placeholder (Model::inference /
  // setExternalTensors, UNMANAGED) and the output is a managed pool tensor —
  // e.g. an rpcmem buffer that a downstream QNN-graph layer reads: the manager
  // cannot alias an external buffer onto a managed one, so it allocates a
  // separate output yet still reports in-place, and the (skipped) copy leaves
  // the output all-zero. Guard on the real data pointer so genuine in-place
  // input layers still skip the redundant self-copy, but externally-fed inputs
  // get copied into the output the consumer actually reads.
  if (input_.getData<char>() != nullptr &&
      hidden_.getData<char>() != input_.getData<char>()) {
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

  std::vector<TensorDim> output_dims = context.getInputDimensions();
  for (auto &d : output_dims) {
    if (d.getDataType() == ml::train::TensorDim::DataType::FP32)
      d.setDataType(context.getActivationDataType());
  }

  context.setOutputDimensions(output_dims);
  is_inplace = output_dims == context.getInputDimensions();
}

void InputLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  context.updateInput(SINGLE_INOUT_IDX, input_dimensions[0]);
  context.updateOutput(SINGLE_INOUT_IDX, input_dimensions[0]);
}

} /* namespace nntrainer */
