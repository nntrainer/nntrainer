// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Jijoong Moon <jijoong.moon@@samsung.com>
 *
 * @file   tensor_api.cpp
 * @date   11 December 2023
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug	   No known bugs except for NYI items
 * @brief  This is Tensor interface for c++ API
 *
 * @note This is experimental API and not stable.
 */

#include "tensor_api_impl.h"

#include <memory_data.h>
#include <tensor.h>

#include <cstring>
#include <stdexcept>

namespace ml {
namespace train {

// --- Constructors / Destructor ---

Tensor::Tensor() : impl_(std::make_unique<Impl>()) {}

Tensor::Tensor(const TensorDim &dim, const std::string &name) :
  impl_(std::make_unique<Impl>(dim, name)) {}

Tensor::~Tensor() = default;

Tensor::Tensor(Tensor &&rhs) noexcept = default;

Tensor &Tensor::operator=(Tensor &&rhs) noexcept = default;

Tensor::Tensor(const Tensor &rhs) :
  impl_(rhs.impl_ ? std::make_unique<Impl>(*rhs.impl_)
                  : std::make_unique<Impl>()) {}

Tensor &Tensor::operator=(const Tensor &rhs) {
  if (this != &rhs) {
    impl_ =
      rhs.impl_ ? std::make_unique<Impl>(*rhs.impl_) : std::make_unique<Impl>();
  }
  return *this;
}

Tensor Tensor::clone() const {
  Tensor t;
  if (impl_) {
    t.impl_ = std::make_unique<Impl>(*impl_);
    if (impl_->eager_data && !impl_->external) {
      t.impl_->eager_data =
        std::make_shared<nntrainer::Tensor>(impl_->eager_data->clone());
    }
  }
  return t;
}

// --- Accessors ---

bool Tensor::isValid() const { return impl_ && impl_->valid; }

const TensorDim &Tensor::shape() const {
  if (!impl_ || !impl_->valid) {
    throw std::runtime_error("Cannot get shape of invalid tensor");
  }
  return impl_->dim;
}

const std::string &Tensor::name() const {
  if (!impl_ || !impl_->valid) {
    throw std::runtime_error("Cannot get name of invalid tensor");
  }
  return impl_->name;
}

TensorDim::DataType Tensor::dtype() const {
  if (!impl_ || !impl_->valid) {
    throw std::runtime_error("Cannot get dtype of invalid tensor");
  }
  return impl_->dim.getDataType();
}

// --- State queries ---

bool Tensor::isExternal() const { return impl_ && impl_->external; }

bool Tensor::isMaterialized() const {
  return impl_ &&
         (impl_->eager_data != nullptr || impl_->bound_tensor != nullptr);
}

// --- Data access ---

template <typename T> const T *Tensor::data() const {
  if (!impl_) {
    throw std::runtime_error("Tensor is not materialized");
  }
  if (impl_->bound_tensor) {
    return impl_->bound_tensor->getData<T>();
  }
  if (!impl_->eager_data) {
    throw std::runtime_error("Tensor is not materialized");
  }
  return impl_->eager_data->getData<T>();
}

template <typename T> T *Tensor::mutable_data() {
  if (!impl_) {
    throw std::runtime_error("Tensor is not materialized");
  }
  if (impl_->bound_tensor) {
    return impl_->bound_tensor->getData<T>();
  }
  if (!impl_->eager_data) {
    throw std::runtime_error("Tensor is not materialized");
  }
  return impl_->eager_data->getData<T>();
}

// Explicit instantiations
template const float *Tensor::data<float>() const;
template float *Tensor::mutable_data<float>();

float Tensor::getValue(unsigned int b, unsigned int c, unsigned int h,
                       unsigned int w) const {
  if (!impl_) {
    throw std::runtime_error("Tensor is not materialized");
  }
  if (impl_->bound_tensor) {
    return impl_->bound_tensor->getValue<float>(b, c, h, w);
  }
  if (!impl_->eager_data) {
    throw std::runtime_error("Tensor is not materialized");
  }
  return impl_->eager_data->getValue<float>(b, c, h, w);
}

void Tensor::setValue(unsigned int b, unsigned int c, unsigned int h,
                      unsigned int w, float value) {
  if (!impl_) {
    throw std::runtime_error("Tensor is not materialized");
  }
  if (impl_->bound_tensor) {
    impl_->bound_tensor->setValue(b, c, h, w, value);
    return;
  }
  if (!impl_->eager_data) {
    throw std::runtime_error("Tensor is not materialized");
  }
  impl_->eager_data->setValue(b, c, h, w, value);
}

void Tensor::copyFrom(const void *src) {
  if (!src) {
    throw std::invalid_argument("copyFrom: source pointer must not be null");
  }
  if (!impl_) {
    throw std::runtime_error("Tensor is not materialized");
  }
  if (impl_->bound_tensor) {
    std::memcpy(impl_->bound_tensor->getData(), src,
                impl_->bound_tensor->bytes());
    return;
  }
  if (!impl_->eager_data) {
    throw std::runtime_error("Tensor is not materialized");
  }
  std::memcpy(impl_->eager_data->getData(), src, impl_->eager_data->bytes());
}

void Tensor::setData(void *new_ptr) {
  if (!impl_) {
    throw std::runtime_error("setData: tensor is not materialized");
  }
  if (!new_ptr) {
    throw std::invalid_argument("setData: pointer must not be null");
  }

  // Bound case: this Tensor is wired to a graph-owned placeholder by
  // Model::compile(inputs, outputs, mode). Pointing the placeholder's
  // MemoryData at the host buffer turns subsequent forward()/inference
  // calls into a zero-copy read of the caller's memory — no separate
  // setExternalTensors() round trip needed.
  if (impl_->bound_tensor) {
    impl_->external_ptr = new_ptr;
    impl_->bound_tensor->setData(
      std::make_shared<nntrainer::MemoryData>(new_ptr), 0, false);
    return;
  }

  // fromData (eager external) case: same idea but on the user-side
  // eager_data tensor that was created at fromData() time.
  if (impl_->external && impl_->eager_data) {
    impl_->external_ptr = new_ptr;
    impl_->eager_data->setData(std::make_shared<nntrainer::MemoryData>(new_ptr),
                               0, false);
    return;
  }

  throw std::runtime_error(
    "setData: tensor is neither bound to a graph placeholder nor a fromData "
    "external tensor");
}

// --- Private helpers ---

void *Tensor::getInternalPtr() const {
  if (!impl_)
    throw std::runtime_error("Tensor is not materialized");
  if (impl_->bound_tensor)
    return impl_->bound_tensor;
  if (impl_->eager_data)
    return impl_->eager_data.get();
  throw std::runtime_error("Tensor is not materialized");
}

Tensor Tensor::wrapResult(const void *internal_tensor) {
  const auto &internal =
    *static_cast<const nntrainer::Tensor *>(internal_tensor);
  Tensor result;
  result.impl_->dim = internal.getDim();
  result.impl_->valid = true;
  result.impl_->external = false;
  result.impl_->eager_data = std::make_shared<nntrainer::Tensor>(internal);
  return result;
}

// --- Immediate in-place operations ---

void Tensor::setZero() { asInternal(getInternalPtr())->setZero(); }

void Tensor::fill(const Tensor &from) {
  asInternal(getInternalPtr())->fill(*asInternal(from.getInternalPtr()));
}

void Tensor::copyData(const Tensor &from) {
  asInternal(getInternalPtr())->copyData(*asInternal(from.getInternalPtr()));
}

// --- Convenience dimension accessors ---

size_t Tensor::size() const {
  if (!impl_ || !impl_->valid) {
    throw std::runtime_error("Cannot get size of invalid tensor");
  }
  return impl_->dim.getDataLen();
}

bool Tensor::empty() const {
  return !impl_ || !impl_->valid || impl_->dim.getDataLen() == 0;
}

size_t Tensor::batch() const { return shape().batch(); }

size_t Tensor::channel() const { return shape().channel(); }

size_t Tensor::height() const { return shape().height(); }

size_t Tensor::width() const { return shape().width(); }

// --- Factory methods ---

Tensor Tensor::fromData(const TensorDim &dim, void *data,
                        const std::string &name) {
  if (!data) {
    throw std::invalid_argument("fromData: data pointer must not be null");
  }
  Tensor t;
  t.impl_->dim = dim;
  t.impl_->name = name;
  t.impl_->valid = true;
  t.impl_->external = true;
  t.impl_->external_ptr = data;
  // Create internal tensor structure, then point to external memory (zero-copy)
  t.impl_->eager_data = std::make_shared<nntrainer::Tensor>(dim, true);
  t.impl_->eager_data->setData(std::make_shared<nntrainer::MemoryData>(data), 0,
                               false);
  return t;
}

Tensor Tensor::zeros(const TensorDim &dim, const std::string &name) {
  Tensor t;
  t.impl_->dim = dim;
  t.impl_->name = name;
  t.impl_->valid = true;
  t.impl_->external = false;
  t.impl_->eager_data = std::make_shared<nntrainer::Tensor>(
    dim, true, nntrainer::Initializer::ZEROS, name);
  t.impl_->eager_data->initialize();
  return t;
}

Tensor Tensor::ones(const TensorDim &dim, const std::string &name) {
  Tensor t;
  t.impl_->dim = dim;
  t.impl_->name = name;
  t.impl_->valid = true;
  t.impl_->external = false;
  t.impl_->eager_data = std::make_shared<nntrainer::Tensor>(
    dim, true, nntrainer::Initializer::ONES, name);
  t.impl_->eager_data->initialize();
  return t;
}

// --- Source layer (backward compatible) ---

void Tensor::setSrcLayer(std::shared_ptr<Layer> l) {
  if (impl_) {
    impl_->src_layer = l;
  }
}

std::shared_ptr<Layer> Tensor::getSrcLayer() const {
  return impl_ ? impl_->src_layer : nullptr;
}

} // namespace train
} // namespace ml
