// SPDX-License-Identifier: Apache-2.0
/**
 * @file	rpcmem_allocator.h
 * @date	15 August 2026
 * @brief	RAII wrapper for an rpcmem (dma-buf) allocation shareable
 *		zero-copy between CPU and cDSP.
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#ifndef __HEXAGON_RPCMEM_ALLOCATOR_H__
#define __HEXAGON_RPCMEM_ALLOCATOR_H__

#include <cstddef>
#include <cstdint>

namespace nntrainer::hexagon {

/**
 * @class RpcmemBuffer
 * @brief One rpcmem allocation. Check valid() after construction.
 */
class RpcmemBuffer {
public:
  explicit RpcmemBuffer(size_t size);
  ~RpcmemBuffer();
  RpcmemBuffer(RpcmemBuffer &&other) noexcept;
  RpcmemBuffer(const RpcmemBuffer &) = delete;
  RpcmemBuffer &operator=(const RpcmemBuffer &) = delete;
  RpcmemBuffer &operator=(RpcmemBuffer &&) = delete;

  bool valid() const { return data_ != nullptr; }
  void *data() const { return data_; }
  int fd() const { return fd_; }
  size_t size() const { return size_; }

private:
  void *data_ = nullptr;
  int fd_ = -1;
  size_t size_ = 0;
};

} // namespace nntrainer::hexagon
#endif // __HEXAGON_RPCMEM_ALLOCATOR_H__
