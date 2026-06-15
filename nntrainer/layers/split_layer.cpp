// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   split_layer.cpp
 * @date   21 May 2021
 * @see    https://github.com/nntrainer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Split Layer Class for Neural Network
 *
 */

#include <cstring>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <split_layer.h>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

SplitLayer::SplitLayer() :
  Layer(),
  leading_helper_dim(1),
  split_props(props::SplitDimension(), props::SplitNumber()) {}

void SplitLayer::finalize(InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "Error: only a single input is supported with split layer";

  unsigned int split_dimension = std::get<props::SplitDimension>(split_props);

  const TensorDim &in_dim = context.getInputDimensions()[0];

  if (std::get<props::SplitNumber>(split_props).empty()) {
    std::get<props::SplitNumber>(split_props)
      .set(in_dim.getTensorDim(split_dimension));
  }
  unsigned int split_number = std::get<props::SplitNumber>(split_props);

  /**
   * The split is only done along the split_dimension dimension.
   * (Assumes input data is continuous)
   * For example, consider input dimension [b,c,h,w], split_number = n
   * 1. axis = 1, output_dim = [b,c//n,h,w], num_outputs = n
   * 2. axis = 2, output_dim = [b,c,h//n,w], num_outputs = n
   * 3. axis = 3, output_dim = [b,c,h,w//n], num_outputs = n
   */
  NNTR_THROW_IF(split_number != context.getNumRequestedOutputs(),
                std::invalid_argument)
    << "Given split number does not match with number of outputs";

  NNTR_THROW_IF(in_dim.getTensorDim(split_dimension) % split_number,
                std::invalid_argument)
    << "Split dimension cannot be split into given number of split_number";

  const unsigned int split_size =
    in_dim.getTensorDim(split_dimension) / split_number;

  TensorDim d = in_dim;
  d.setTensorDim(split_dimension, split_size);

  std::vector<TensorDim> output_dim(context.getNumRequestedOutputs());
  for (auto &out_dim : output_dim) {
    out_dim = d;
  }
  context.setOutputDimensions(output_dim);

  /**
   * Setup input_reshape_helper to which input will be reshaped in forwarding
   * to facilitate easier processing.
   *
   * The helper shape consolidates all the dimensions before the split_dimension
   * together and all the dimensions after the split_dimension to facilitate
   * easier splitting of the data.
   */
  leading_helper_dim = 1;
  input_reshape_helper.channel(1);
  input_reshape_helper.height(1);
  input_reshape_helper.width(1);
  for (unsigned int idx = 1; idx < split_dimension; ++idx) {
    leading_helper_dim *= in_dim.getTensorDim(idx);
  }

  input_reshape_helper.height(in_dim.getTensorDim(split_dimension));

  for (unsigned int idx = split_dimension + 1;
       idx < ml::train::TensorDim::MAXDIM; ++idx) {
    input_reshape_helper.width(input_reshape_helper.width() *
                               in_dim.getTensorDim(idx));
  }

  /**
   * Setup output_reshape_helper to which input will be reshaped in forwarding
   * to facilitate easier processing.
   */
  output_reshape_helper = input_reshape_helper;
  output_reshape_helper.height(split_size);

  setBatch(in_dim.batch());
}

void SplitLayer::forwarding(RunLayerContext &context, bool training) {
  unsigned int split_number = std::get<props::SplitNumber>(split_props);

  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  const TensorDim in_dim = input_.getDim();
  input_.reshape(input_reshape_helper);

  for (unsigned int idx = 0; idx < split_number; idx++) {
    Tensor &output_ = context.getOutput(idx);
    const TensorDim out_dim = output_.getDim();
    output_.reshape(output_reshape_helper);

    for (unsigned int batch = 0; batch < input_.batch(); batch++) {
      const Tensor source_tensor = Tensor::Map(
        input_.getAddress(batch, 0, idx * output_reshape_helper.height(), 0),
        output_reshape_helper.height() * input_reshape_helper.width() *
          sizeof(float),
        {1, 1, output_reshape_helper.height(), input_reshape_helper.width()});
      Tensor dest_tensor = Tensor::Map(
        output_.getAddress(batch, 0, 0, 0),
        output_reshape_helper.height() * output_reshape_helper.width() *
          sizeof(float),
        {1, 1, output_reshape_helper.height(), output_reshape_helper.width()});
      dest_tensor.copy(source_tensor);
    }

    output_.reshape(out_dim);
  }

  input_.reshape(in_dim);
}

void SplitLayer::incremental_forwarding(RunLayerContext &context,
                                        unsigned int from, unsigned int to,
                                        bool training) {
  unsigned int split_number = std::get<props::SplitNumber>(split_props);
  unsigned int split_dimension =
    std::get<props::SplitDimension>(split_props).get();

  // Only optimize the width-split (axis=3) case with a direct memcpy.
  // For other axes (height, channel) or for prefill, fall back to forwarding().
  if (split_dimension != 3) {
    forwarding(context, training);
    return;
  }

  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  const unsigned int B = input_.batch();
  const unsigned int num_steps = to - from; // 1 for decode
  const unsigned int split_w = input_.width() / split_number;

  // For each actual batch and each valid time step, copy only split_w floats
  // from the chunk at [b, 0, s, idx*split_w] into output[b, 0, s, 0].
  // This is O(B * num_steps * split_number * split_w) vs the full
  // O(B * INIT_SEQ_LEN * split_number * split_w) of forwarding().
  for (unsigned int b = 0; b < B; ++b) {
    for (unsigned int s = 0; s < num_steps; ++s) {
      for (unsigned int idx = 0; idx < split_number; ++idx) {
        Tensor &output_ = context.getOutput(idx);
        const float *src = input_.getAddress(b, 0, s, idx * split_w);
        float *dst = output_.getAddress(b, 0, s, 0);
        std::memcpy(dst, src, split_w * sizeof(float));
      }
    }
  }
}

void SplitLayer::calcDerivative(RunLayerContext &context) {
  unsigned int split_number = std::get<props::SplitNumber>(split_props);

  Tensor &input_ = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  const TensorDim in_dim = input_.getDim();
  input_.reshape(input_reshape_helper);

  for (unsigned int idx = 0; idx < split_number; idx++) {
    Tensor output_ = context.getIncomingDerivative(idx);
    const TensorDim out_dim = output_.getDim();
    output_.reshape(output_reshape_helper);

    for (unsigned int batch = 0; batch < input_.batch(); batch++) {
      Tensor dest_tensor = Tensor::Map(
        input_.getAddress(batch, 0, idx * output_reshape_helper.height(), 0),
        output_reshape_helper.height() * input_reshape_helper.width() *
          sizeof(float),
        {1, 1, output_reshape_helper.height(), input_reshape_helper.width()});
      const Tensor source_tensor = Tensor::Map(
        output_.getAddress(batch, 0, 0, 0),
        output_reshape_helper.height() * output_reshape_helper.width() *
          sizeof(float),
        {1, 1, output_reshape_helper.height(), output_reshape_helper.width()});
      dest_tensor.copy(source_tensor);
    }

    output_.reshape(out_dim);
  }

  input_.reshape(in_dim);
}

void SplitLayer::exportTo(Exporter &exporter,
                          const ml::train::ExportMethods &method) const {
  exporter.saveResult(split_props, method, this);
}

void SplitLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, split_props);
  NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
    << "[SplitLayer] Unknown Layer Properties count " +
         std::to_string(values.size());
}

} /* namespace nntrainer */
