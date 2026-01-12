// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd.
 *
 * @file    cuda_buffer_manager.h
 * @date    11 Dec 2025
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Donghak Jung <dk11.jung@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   This file contains global Buffer objects and manages them for CUDA
 */

#ifndef __CUDA_BUFFER_MANAGER_H__
#define __CUDA_BUFFER_MANAGER_H__

#include <string>
#include <vector>

#include "cuda/cuda_context_manager.h"
#include <nntrainer_log.h>

#include "singleton.h"

namespace nntrainer {

/**
 * @class CudaBufferManager contains Buffer object management
 * @brief Support for Buffer management on CUDA
 */

class CudaBufferManager : public Singleton<CudaBufferManager> {

private:
  /**
   * @brief CUDA context global instance
   *
   */
  cuda::ContextManager &context_inst_ = cuda::ContextManager::Global();

  /**
   * @brief Buffer size in bytes preset
   */
  const size_t buffer_size_bytes = 1024 * 8192 * sizeof(float);
  const size_t unused_buffer_bytes = sizeof(float);
  const unsigned int max_qs = 3;

  /// @note this size might be changed
  const size_t scale_q4_0_size =
    3072 * (8192 / 32) * 2;                   /** buffer size of quants */
  const size_t quant_q4_0_size = 3072 * 8192; /** buffer size of scales */

  void *inBufferA = nullptr;
  void *inBufferB = nullptr;
  void *inBufferC = nullptr;
  void *outBufferA = nullptr;
  void *outBufferB = nullptr;

  void *data_input = nullptr;
  std::vector<void *> scale_vec;
  std::vector<void *> quant_vec;
  std::vector<void *> output_vec;

public:
  /**
   * @brief Initialize Buffer objects.
   */
  void initBuffers();

  /**
   * @brief Get read only inBufferA.
   * @return void* or nullptr if initBuffers() is not called
   */
  void *getInBufferA() { return inBufferA; }

  /**
   * @brief Get read only inBufferB.
   * @return void* or nullptr if initBuffers() is not called
   */
  void *getInBufferB() { return inBufferB; }

  /**
   * @brief Get read only inBufferC.
   * @return void* or nullptr if initBuffers() is not called
   */
  void *getInBufferC() { return inBufferC; }

  /**
   * @brief Get read-write outBufferA.
   * @return void* or nullptr if initBuffers() is not called
   */
  void *getOutBufferA() { return outBufferA; }

  /**
   * @brief Get read-write outBufferB.
   * @return void* or nullptr if initBuffers() is not called
   */
  void *getOutBufferB() { return outBufferB; }

  /**
   * @brief Get the device pointer to data_input
   */
  void *getDeviceInput() { return data_input; }

  /**
   * @brief Get the device pointer to output
   *
   */
  void *getDeviceOutput(unsigned int idx = 0) {
    if (idx >= output_vec.size())
      return nullptr;

    return output_vec[idx];
  }

  /**
   * @brief Get the device pointer to scale
   */
  void *getDeviceScale(unsigned int idx = 0) {
    if (idx >= scale_vec.size())
      return nullptr;

    return scale_vec[idx];
  }

  /**
   * @brief Get the device pointer to quant
   */
  void *getDeviceQuant(unsigned int idx = 0) {
    if (idx >= quant_vec.size())
      return nullptr;

    return quant_vec[idx];
  }

  /**
   * @brief Destroy Buffer pointers.
   *
   */
  ~CudaBufferManager();
};
} // namespace nntrainer

#endif /* __CUDA_BUFFER_MANAGER_H__ */
