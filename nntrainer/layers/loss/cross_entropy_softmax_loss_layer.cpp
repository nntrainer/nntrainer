// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   cross_entropy_Softmax_loss_layer.cpp
 * @date   24 June 2021
 * @brief  This is MSE Loss Layer Class of Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <cmath>

#include <cross_entropy_softmax_loss_layer.h>

#include <acti_func.h>
#include <layer_context.h>
#include <lazy_tensor.h>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void CrossEntropySoftmaxLossLayer::forwarding(RunLayerContext &context,
                                              bool training) {
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &y = context.getInput(SINGLE_INOUT_IDX);

  // fill the output
  auto dataType = y.getDataType();
  if (dataType == ml::train::TensorDim::DataType::FP32) {
    hidden_ = y.apply(ActiFunc::softmax<float>, hidden_);

    if (context.isLabelAvailable(SINGLE_INOUT_IDX)) {
      Tensor &y2 = context.getLabel(SINGLE_INOUT_IDX);
      l = y2.multiply(hidden_.apply<float>(logFloat<float>))
            .sum_by_batch()
            .multiply(-1.0f);

      // update the loss value
      LossLayer::updateLoss(context, l);
    }
  } else if (dataType == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    hidden_ = y.apply(ActiFunc::softmax<_FP16>, hidden_);

    if (context.isLabelAvailable(SINGLE_INOUT_IDX)) {
      Tensor &y2 = context.getLabel(SINGLE_INOUT_IDX);
      l = y2.multiply(hidden_.apply<_FP16>(logFloat<_FP16>))
            .sum_by_batch()
            .multiply(-1);

      // update the loss value
      LossLayer::updateLoss(context, l);
    }
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  }
}

void CrossEntropySoftmaxLossLayer::calcDerivative(RunLayerContext &context) {
  Tensor &ret_derivative = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  const Tensor &y2 = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &y = context.getInput(SINGLE_INOUT_IDX);

  auto dataType = y.getDataType();
  Tensor ret = Tensor("ret", y.getFormat(), y.getDataType());
  if (dataType == ml::train::TensorDim::DataType::FP32) {
    y.apply(ActiFunc::softmax<float>, ret);
  } else if (dataType == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    y.apply(ActiFunc::softmax<_FP16>, ret);
#else
    throw std::runtime_error("enable-fp16 is not enabled");
#endif
  }

  /// @note y and ret_derivative can be same here, so this has to be out-place
  /// operation
  // TODO: verify y and ret_derivative must not be same as loss layer is not
  // working in-place
  ret.subtract(y2, ret_derivative);
  if (ret_derivative.divide_i(ret.batch()) != ML_ERROR_NONE) {
    throw std::runtime_error("[CrossEntropySoftmaxLossLayer::calcDerivative] "
                             "Error when calculating loss");
  }
}

} // namespace nntrainer
