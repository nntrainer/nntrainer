// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_buffer.h
 * @date    22 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   CUDA wrapper for device memory. Peer of nntrainer::opencl::Buffer:
 *          RAII over cudaMalloc/cudaFree with synchronous H2D/D2H copies.
 */

#ifndef __CUDA_BUFFER_H__
#define __CUDA_BUFFER_H__

#include <cstddef>

#include "noncopyable.h"

namespace nntrainer::cuda {

/**
 * @class Buffer
 * @brief RAII owner of a cudaMalloc'd device allocation.
 */
class Buffer : public Noncopyable {
public:
  /**
   * @brief Default (empty) buffer
   */
  Buffer() {}

  /**
   * @brief Allocate @p size_in_bytes of device memory, optionally seeding it
   *        with @p data (host pointer, copied H2D).
   */
  Buffer(size_t size_in_bytes, const void *data = nullptr);

  /**
   * @brief Move constructor (transfers ownership)
   */
  Buffer(Buffer &&buffer);

  /**
   * @brief Move assignment (transfers ownership)
   */
  Buffer &operator=(Buffer &&buffer);

  /**
   * @brief Free the device allocation
   */
  ~Buffer();

  /**
   * @brief Get the raw device pointer (cast to CUdeviceptr / T* as needed)
   */
  void *GetBuffer() const { return dev_ptr_; }

  /**
   * @brief Allocation size in bytes
   */
  size_t size() const { return size_; }

  /**
   * @brief Synchronous full-buffer host->device copy
   */
  bool WriteData(const void *data);

  /**
   * @brief Synchronous full-buffer device->host copy
   */
  bool ReadData(void *data);

  /**
   * @brief Synchronous host->device copy of a sub-region
   */
  bool WriteDataRegion(size_t size_in_bytes, const void *data,
                       size_t buffer_offset = 0, size_t host_offset = 0);

  /**
   * @brief Synchronous device->host copy of a sub-region
   */
  bool ReadDataRegion(size_t size_in_bytes, void *data,
                      size_t buffer_offset = 0, size_t host_offset = 0);

private:
  void Release();

  void *dev_ptr_{nullptr};
  size_t size_{0};
};

} // namespace nntrainer::cuda

#endif // __CUDA_BUFFER_H__
