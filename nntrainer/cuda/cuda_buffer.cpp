// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_buffer.cpp
 * @date    22 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   CUDA device buffer implementation.
 */

#include "cuda_buffer.h"
#include "cuda_common.h"
#include "cuda_context_manager.h"

namespace nntrainer::cuda {

Buffer::Buffer(size_t size_in_bytes, const void *data) : size_(size_in_bytes) {
  ContextManager::Global().EnsureCurrent();
  if (size_ == 0)
    return;
  if (!cudaCheck(cudaMalloc(&dev_ptr_, size_), "cudaMalloc")) {
    dev_ptr_ = nullptr;
    size_ = 0;
    return;
  }
  if (data)
    WriteData(data);
}

void Buffer::Release() {
  if (dev_ptr_) {
    cudaFree(dev_ptr_);
    dev_ptr_ = nullptr;
    size_ = 0;
  }
}

Buffer::~Buffer() { Release(); }

Buffer::Buffer(Buffer &&o) : dev_ptr_(o.dev_ptr_), size_(o.size_) {
  o.dev_ptr_ = nullptr;
  o.size_ = 0;
}

Buffer &Buffer::operator=(Buffer &&o) {
  if (this != &o) {
    Release();
    dev_ptr_ = o.dev_ptr_;
    size_ = o.size_;
    o.dev_ptr_ = nullptr;
    o.size_ = 0;
  }
  return *this;
}

bool Buffer::WriteData(const void *data) {
  return dev_ptr_ != nullptr &&
         cudaCheck(cudaMemcpy(dev_ptr_, data, size_, cudaMemcpyHostToDevice),
                   "cudaMemcpy H2D");
}

bool Buffer::ReadData(void *data) {
  return dev_ptr_ != nullptr &&
         cudaCheck(cudaMemcpy(data, dev_ptr_, size_, cudaMemcpyDeviceToHost),
                   "cudaMemcpy D2H");
}

bool Buffer::WriteDataRegion(size_t n, const void *data, size_t boff,
                             size_t hoff) {
  return dev_ptr_ != nullptr &&
         cudaCheck(cudaMemcpy(static_cast<char *>(dev_ptr_) + boff,
                              static_cast<const char *>(data) + hoff, n,
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy H2D region");
}

bool Buffer::ReadDataRegion(size_t n, void *data, size_t boff, size_t hoff) {
  return dev_ptr_ != nullptr &&
         cudaCheck(cudaMemcpy(static_cast<char *>(data) + hoff,
                              static_cast<const char *>(dev_ptr_) + boff, n,
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy D2H region");
}

} // namespace nntrainer::cuda
