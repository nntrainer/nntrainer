// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Jijoong Moon <jijoong.moon@@samsung.com>
 *
 * @file   tensor_api_ops.cpp
 * @date   11 December 2023
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Eager numeric operations for ml::train::Tensor
 *
 * @note This is experimental API and not stable.
 */

#include "tensor_api_impl.h"

#include <tensor.h>

#include <cstring>
#include <functional>
#include <stdexcept>

namespace ml {
namespace train {

Tensor Tensor::add(float value) const {
  auto r = asInternal(getInternalPtr())->add(value);
  return wrapResult(&r);
}

Tensor Tensor::subtract(float value) const {
  auto r = asInternal(getInternalPtr())->subtract(value);
  return wrapResult(&r);
}

Tensor Tensor::subtract(const Tensor &other) const {
  auto r =
    asInternal(getInternalPtr())->subtract(*asInternal(other.getInternalPtr()));
  return wrapResult(&r);
}

Tensor Tensor::multiply(float value) const {
  auto r = asInternal(getInternalPtr())->multiply(value);
  return wrapResult(&r);
}

Tensor Tensor::divide(float value) const {
  auto r = asInternal(getInternalPtr())->divide(value);
  return wrapResult(&r);
}

Tensor Tensor::divide(const Tensor &other) const {
  auto r =
    asInternal(getInternalPtr())->divide(*asInternal(other.getInternalPtr()));
  return wrapResult(&r);
}

Tensor Tensor::dot(const Tensor &other, bool trans, bool trans_in) const {
  auto r = asInternal(getInternalPtr())
             ->dot(*asInternal(other.getInternalPtr()), trans, trans_in);
  return wrapResult(&r);
}

Tensor Tensor::transpose(const std::string &direction) const {
  auto r = asInternal(getInternalPtr())->transpose(direction);
  return wrapResult(&r);
}

Tensor Tensor::pow(float exponent) const {
  auto r = asInternal(getInternalPtr())->pow(exponent);
  return wrapResult(&r);
}

Tensor Tensor::sum(unsigned int axis, float alpha) const {
  auto r = asInternal(getInternalPtr())->sum(axis, alpha);
  return wrapResult(&r);
}

Tensor Tensor::average(unsigned int axis) const {
  auto r = asInternal(getInternalPtr())->average(axis);
  return wrapResult(&r);
}

Tensor Tensor::average() const {
  auto r = asInternal(getInternalPtr())->average();
  return wrapResult(&r);
}

float Tensor::l2norm() const { return asInternal(getInternalPtr())->l2norm(); }

std::vector<unsigned int> Tensor::argmax() const {
  return asInternal(getInternalPtr())->argmax();
}

// --- Tensor manipulation ---

Tensor Tensor::getBatchSlice(unsigned int offset, unsigned int size) const {
  auto r = asInternal(getInternalPtr())->getBatchSlice(offset, size);
  return wrapResult(&r);
}

Tensor Tensor::getSharedDataTensor(const TensorDim &dim, size_t offset) const {
  auto r =
    asInternal(getInternalPtr())->getSharedDataTensor(dim, offset, false);
  return wrapResult(&r);
}

Tensor Tensor::apply(std::function<float(float)> f) const {
  auto *internal = asInternal(getInternalPtr());
  nntrainer::Tensor r = internal->clone();
  float *d = r.getData<float>();
  size_t len = r.size();
  for (size_t i = 0; i < len; ++i) {
    d[i] = f(d[i]);
  }
  return wrapResult(&r);
}

void Tensor::apply_i(std::function<float(float)> f) {
  auto *internal = asInternal(getInternalPtr());
  float *d = internal->getData<float>();
  size_t len = internal->size();
  for (size_t i = 0; i < len; ++i) {
    d[i] = f(d[i]);
  }
}

Tensor Tensor::cat(const std::vector<Tensor> &tensors, int axis) {
  if (tensors.empty()) {
    throw std::invalid_argument("cat: tensors must not be empty");
  }

  std::vector<nntrainer::Tensor> internals;
  internals.reserve(tensors.size());
  for (auto &t : tensors) {
    internals.push_back(*asInternal(t.getInternalPtr()));
  }

  nntrainer::Tensor output;
  internals[0].concat(
    std::vector<nntrainer::Tensor>(internals.begin() + 1, internals.end()),
    axis, output);
  return wrapResult(&output);
}

} // namespace train
} // namespace ml
