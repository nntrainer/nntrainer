// SPDX-License-Identifier: Apache-2.0
/**
 * @file	rpcmem_allocator.cpp
 * @date	15 August 2026
 * @brief	RAII wrapper for an rpcmem (dma-buf) allocation.
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include "rpcmem_allocator.h"

#include <climits>

#include <rpcmem.h>

namespace nntrainer::hexagon {

RpcmemBuffer::RpcmemBuffer(size_t size) : size_(size) {
  if (size > (size_t)INT_MAX)
    return; /* stays invalid; rpcmem sizes are int */
  data_ = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, (int)size);
  if (data_ != nullptr) {
    fd_ = rpcmem_to_fd(data_);
    if (fd_ < 0) {
      rpcmem_free(data_);
      data_ = nullptr;
    }
  }
}

RpcmemBuffer::~RpcmemBuffer() {
  if (data_ != nullptr)
    rpcmem_free(data_);
}

RpcmemBuffer::RpcmemBuffer(RpcmemBuffer &&other) noexcept :
  data_(other.data_), fd_(other.fd_), size_(other.size_) {
  other.data_ = nullptr;
  other.fd_ = -1;
  other.size_ = 0;
}

} // namespace nntrainer::hexagon
