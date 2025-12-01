// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   weight.cpp
 * @date   22 September 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Weight Class for Neural Network
 *
 */

#include <util_func.h>
#include <weight.h>

#include <nntrainer_error.h>

namespace nntrainer {

Weight::Weight(const TensorDim &dim, const Initializer init,
               const WeightRegularizer reg, const float reg_const,
               const float decay_const, const float max_norm, bool train,
               bool alloc_now_, std::string name, unsigned int axis,
               float loss_scale_, bool is_mixed_) :
  Var_Grad(dim, init, train, alloc_now_, name),
  regularizer(reg),
  regularizer_constant(reg_const),
  decay(decay_const),
  clip_by_global_norm(max_norm),
  output_axis(axis),
  loss_scale(loss_scale_),
  is_mixed(is_mixed_) {
  if (init == Initializer::NONE)
    throw std::invalid_argument("Weight initializer cannot be none");
  if (regularizer == WeightRegularizer::UNKNOWN)
    throw std::invalid_argument("Weight regularizer unknown");

  std::string var32_suffix = ":fp32";
  std::string var32_name = name + var32_suffix;

  /**
   * @note We assume if the Weight Data Type is not FP32, then FP32 Weight is
   * necessary to maintain the accuracy.
   * We could think it can be other data type and if there is the case to
   * support other data type, then the code below needs to be udpated.
   *
   * Also, the loss_scale is not used in Weight but leave as it is for later
   * usage.
   */

  if (train && dim.getDataType() != ml::train::TensorDim::DataType::FP32) {
    TensorDim var32_dim(dim);
    var32_dim.setDataType(ml::train::TensorDim::DataType::FP32);

    var32 = std::make_shared<Tensor>(var32_dim, alloc_now_, init, var32_name);
  } else {
    var32 = std::make_shared<Tensor>(var32_name);
  }
}

Weight::Weight(const TensorDim &dim_v, const TensorDim &dim_g,
               const Initializer init, const WeightRegularizer reg,
               const float reg_const, const float decay_const,
               const float max_norm, bool train, bool alloc_now_,
               std::string name, unsigned int axis, float loss_scale_,
               bool is_mixed_) :
  Var_Grad(dim_v, dim_g, init, train, alloc_now_, name),
  regularizer(reg),
  regularizer_constant(reg_const),
  decay(decay_const),
  clip_by_global_norm(max_norm),
  output_axis(axis),
  loss_scale(loss_scale_),
  is_mixed(is_mixed_) {
  if (init == Initializer::NONE)
    throw std::invalid_argument("Weight initializer cannot be none");
  if (regularizer == WeightRegularizer::UNKNOWN)
    throw std::invalid_argument("Weight regularizer unknown");

  std::string var32_suffix = ":fp32";
  std::string var32_name = name + var32_suffix;

  if (train && dim_v.getDataType() != ml::train::TensorDim::DataType::FP32) {
    TensorDim var32_dim(dim_v);
    var32_dim.setDataType(ml::train::TensorDim::DataType::FP32);
    var32 = std::make_shared<Tensor>(var32_dim, alloc_now_, init, var32_name);
  } else {
    var32 = std::make_shared<Tensor>(var32_name);
  }
}

Weight::Weight(const Tensor &v, const Tensor &g, const Tensor &v32,
               const std::string &n, bool is_dependent,
               unsigned int output_axis_) :
  Var_Grad(v, g, n, is_dependent),
  regularizer(WeightRegularizer::NONE),
  regularizer_constant(1.0f),
  decay(0.0f),
  clip_by_global_norm(0.0f),
  output_axis(output_axis_),
  loss_scale(1.0),
  is_mixed(false),
  var32(std::make_shared<Tensor>(n + ":fp32")) {

  if (!g.empty() && isMixedPrecision()) {
    TensorDim var32_dim(v.getDim());
    var32_dim.setDataType(ml::train::TensorDim::DataType::FP32);
    if (!v32.empty())
      var32 = std::make_shared<Tensor>(
        v32.getSharedDataTensor(var32_dim, 0, false, n + ":fp32"));
  }
}

Weight::Weight(Tensor *v, Tensor *g, Tensor *v32, const WeightRegularizer reg,
               const float reg_const, const float decay, bool is_dependent,
               const float max_norm, unsigned int output_axis_,
               float loss_scale_, bool is_mixed_) :
  Var_Grad(v, g, is_dependent),
  regularizer(reg),
  regularizer_constant(reg_const),
  decay(decay),
  clip_by_global_norm(max_norm),
  output_axis(output_axis_),
  loss_scale(loss_scale_),
  is_mixed(is_mixed_),
  var32(std::shared_ptr<Tensor>(v32, [](void *) {})) {
  if (!v32)
    var32 = std::make_shared<Tensor>();
}

void Weight::applyGradient(double lr, Tensor &updated_grad) {
  if (isMixedPrecision() &&
      updated_grad.getDataType() == ml::train::TensorDim::DataType::FP32 &&
      var->getDataType() != ml::train::TensorDim::DataType::FP32) {
    var32->add_i(updated_grad, -lr);
    quantizeWeight();
    return;
  } else {
    /** FP32 (or matching dtype) path: apply the provided updated_grad directly */
    var->add_i(updated_grad, -lr);
  }
}

void Weight::quantizeWeight() {
  if (!isMixedPrecision())
    return;

  Tensor &var = getVariableRef();
  ml::train::TensorDim::DataType type = var.getDataType();
  switch (type) {
  case ml::train::TensorDim::DataType::QINT4:
    // NYI
    break;
  case ml::train::TensorDim::DataType::QINT8:
    // NYI
    break;
  case ml::train::TensorDim::DataType::FP16:
    getVariableRef().copyData(getVariableFP32Ref());
    break;
  case ml::train::TensorDim::DataType::FP32:
    break;
  default:
    break;
  }

  return;
}

} // namespace nntrainer
