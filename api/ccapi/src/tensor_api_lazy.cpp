// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Jijoong Moon <jijoong.moon@@samsung.com>
 *
 * @file   tensor_api_lazy.cpp
 * @date   11 December 2023
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Lazy operation chaining for ml::train::Tensor
 *
 * @note This is experimental API and not stable.
 */

#include "tensor_api_impl.h"

#include <tensor.h>

#include <stdexcept>

namespace ml {
namespace train {

Tensor &Tensor::chain() {
  if (!impl_) {
    throw std::runtime_error("Cannot chain on invalid tensor");
  }
  impl_->call_chain.clear();
  return *this;
}

Tensor &Tensor::add_i(float value) {
  if (!impl_) {
    throw std::runtime_error("Cannot add_i on invalid tensor");
  }
  impl_->call_chain.push_back(
    [value](nntrainer::Tensor &t) { t.add_i(value); });
  return *this;
}

Tensor &Tensor::add_i(const Tensor &other, float alpha) {
  if (!impl_) {
    throw std::runtime_error("Cannot add_i on invalid tensor");
  }
  auto other_impl = other.impl_.get();
  impl_->call_chain.push_back([other_impl, alpha](nntrainer::Tensor &t) {
    nntrainer::Tensor *src = other_impl->bound_tensor
                               ? other_impl->bound_tensor
                               : other_impl->eager_data.get();
    if (!src)
      throw std::runtime_error("add_i: other tensor not materialized");
    t.add_i(*src, alpha);
  });
  return *this;
}

Tensor &Tensor::subtract_i(float value) {
  if (!impl_) {
    throw std::runtime_error("Cannot subtract_i on invalid tensor");
  }
  impl_->call_chain.push_back(
    [value](nntrainer::Tensor &t) { t.subtract_i(value); });
  return *this;
}

Tensor &Tensor::subtract_i(const Tensor &other) {
  if (!impl_) {
    throw std::runtime_error("Cannot subtract_i on invalid tensor");
  }
  auto other_impl = other.impl_.get();
  impl_->call_chain.push_back([other_impl](nntrainer::Tensor &t) {
    nntrainer::Tensor *src = other_impl->bound_tensor
                               ? other_impl->bound_tensor
                               : other_impl->eager_data.get();
    if (!src)
      throw std::runtime_error("subtract_i: other tensor not materialized");
    t.subtract_i(*src);
  });
  return *this;
}

Tensor &Tensor::multiply_i(float value) {
  if (!impl_) {
    throw std::runtime_error("Cannot multiply_i on invalid tensor");
  }
  impl_->call_chain.push_back(
    [value](nntrainer::Tensor &t) { t.multiply_i(value); });
  return *this;
}

Tensor &Tensor::multiply_i(const Tensor &other) {
  if (!impl_) {
    throw std::runtime_error("Cannot multiply_i on invalid tensor");
  }
  auto other_impl = other.impl_.get();
  impl_->call_chain.push_back([other_impl](nntrainer::Tensor &t) {
    nntrainer::Tensor *src = other_impl->bound_tensor
                               ? other_impl->bound_tensor
                               : other_impl->eager_data.get();
    if (!src)
      throw std::runtime_error("multiply_i: other tensor not materialized");
    t.multiply_i(*src);
  });
  return *this;
}

Tensor &Tensor::divide_i(float value) {
  if (!impl_) {
    throw std::runtime_error("Cannot divide_i on invalid tensor");
  }
  impl_->call_chain.push_back(
    [value](nntrainer::Tensor &t) { t.divide_i(value); });
  return *this;
}

Tensor &Tensor::divide_i(const Tensor &other) {
  if (!impl_) {
    throw std::runtime_error("Cannot divide_i on invalid tensor");
  }
  auto other_impl = other.impl_.get();
  impl_->call_chain.push_back([other_impl](nntrainer::Tensor &t) {
    nntrainer::Tensor *src = other_impl->bound_tensor
                               ? other_impl->bound_tensor
                               : other_impl->eager_data.get();
    if (!src)
      throw std::runtime_error("divide_i: other tensor not materialized");
    t.divide_i(*src);
  });
  return *this;
}

Tensor &Tensor::pow_i(float exponent) {
  if (!impl_) {
    throw std::runtime_error("Cannot pow_i on invalid tensor");
  }
  impl_->call_chain.push_back(
    [exponent](nntrainer::Tensor &t) { t.pow_i(exponent); });
  return *this;
}

Tensor &Tensor::inv_sqrt_i() {
  if (!impl_) {
    throw std::runtime_error("Cannot inv_sqrt_i on invalid tensor");
  }
  impl_->call_chain.push_back([](nntrainer::Tensor &t) { t.inv_sqrt_i(); });
  return *this;
}

Tensor &Tensor::eval() {
  if (!impl_ || !isMaterialized()) {
    throw std::runtime_error("Cannot eval: tensor is not materialized");
  }
  nntrainer::Tensor *target =
    impl_->bound_tensor ? impl_->bound_tensor : impl_->eager_data.get();
  for (auto &op : impl_->call_chain) {
    op(*target);
  }
  impl_->call_chain.clear();
  return *this;
}

} // namespace train
} // namespace ml
