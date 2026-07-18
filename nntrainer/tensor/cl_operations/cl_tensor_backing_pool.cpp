// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cl_tensor_backing_pool.cpp
 * @date    27 May 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Implementation of TensorBackingPool — string → backing map
 *          shared across GPU layers within one process.
 */
#include "cl_tensor_backing_pool.h"

namespace nntrainer::tv {

TensorBackingPool &TensorBackingPool::Global() {
  static TensorBackingPool g;
  return g;
}

std::shared_ptr<TensorBacking>
TensorBackingPool::set(const std::string &name,
                       std::shared_ptr<TensorBacking> backing) {
  std::lock_guard<std::mutex> lock(mtx_);
  map_[name] = backing;
  return backing;
}

std::shared_ptr<TensorBacking>
TensorBackingPool::get(const std::string &name) const {
  std::lock_guard<std::mutex> lock(mtx_);
  auto it = map_.find(name);
  return it == map_.end() ? nullptr : it->second;
}

void TensorBackingPool::erase(const std::string &name) {
  std::lock_guard<std::mutex> lock(mtx_);
  map_.erase(name);
}

void TensorBackingPool::clear() {
  std::lock_guard<std::mutex> lock(mtx_);
  map_.clear();
}

size_t TensorBackingPool::size() const {
  std::lock_guard<std::mutex> lock(mtx_);
  return map_.size();
}

} // namespace nntrainer::tv
