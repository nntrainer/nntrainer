// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd.
 *
 * @file    cuda_context_manager.h
 * @date    11 Dec 2025
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Donghak Jung <dk11.jung@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   CUDA wrapper for context management
 *
 */

#ifndef __CUDA_CONTEXT_MANAGER_H__
#define __CUDA_CONTEXT_MANAGER_H__

#include <cuda_runtime.h>
#include <mutex>

#include "singleton.h"

namespace nntrainer::cuda {

/**
 * @class ContextManager contains wrappers for managing CUDA context
 * @brief CUDA context wrapper
 *
 */
class ContextManager : public Singleton<ContextManager> {

  /**
   * @brief Create a Default GPU Device object
   *
   * @return true if successful or false otherwise
   */
  bool CreateDefaultGPUDevice();

  /**
   * @brief Create CUDA context
   *
   * @return true if successful or false otherwise
   */
  bool CreateContext();

public:
  /**
   * @brief Release CUDA context
   *
   */
  void ReleaseContext();

  /**
   * @brief Get the Device Id
   *
   * @return int device id
   */
  int GetDeviceId();

  /**
   * @brief allocate Device memory
   *
   * @param size size of the memory to be allocated
   * @return void* pointer to the allocated memory
   */
  void *allocateDeviceMemory(size_t size);

  /**
   * @brief deallocate Device memory
   *
   * @param ptr pointer to the memory to be deallocated
   */
  void releaseDeviceMemory(void *ptr);

  /**
   * @brief Destroy the Context Manager object
   *
   */
  ~ContextManager();

private:
  int device_id_ = 0;
  bool initialized_ = false;
};
} // namespace nntrainer::cuda
#endif // __CUDA_CONTEXT_MANAGER_H__
