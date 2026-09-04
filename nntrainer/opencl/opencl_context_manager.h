// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file    opencl_context_manager.h
 * @date    06 Feb 2024
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Debadri Samaddar <s.debadri@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   OpenCL wrapper for context management
 *
 */

#ifndef __OPENCL_CONTEXT_MANAGER_H__
#define __OPENCL_CONTEXT_MANAGER_H__

#include <mutex>

#include "CL/cl.h"
#include "opencl_device_info.h"

#include "singleton.h"

namespace nntrainer::opencl {

/**
 * @class ContextManager contains wrappers for managing OpenCL context
 * @brief OpenCL context wrapper
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
   * @brief Create OpenCL context
   *
   * @return true if successful or false otherwise
   */
  bool CreateCLContext();

public:
  /**
   * @brief Get the OpenCL context object
   *
   * @return const cl_context
   */
  const cl_context &GetContext();

  /**
   * @brief Release OpenCL context
   *
   */
  void ReleaseContext();

  /**
   * @brief Get the Device Id object
   *
   * @return const cl_device_id
   */
  const cl_device_id GetDeviceId();

  /**
   * @brief Get a stable signature of the active device (name + driver
   *        version). Used to key the on-disk kernel binary cache so a binary
   *        built for a different GPU, or before a driver update, is never
   *        loaded.
   *
   * @return std::string "<device-name>|<driver-version>"
   */
  std::string GetDeviceSignature();

  /**
   * @brief Get the Device Info
   *
   * @return const DeviceInfo *
   */
  const DeviceInfo *getDeviceInfo() const { return device_info_.get(); }

  /**
   * @brief allocate SVM memory
   *
   * @param size size of the memory to be allocated
   * @return void* pointer to the allocated SVM memory
   */
  void *createSVMRegion(size_t size);

  /**
   * @brief deallocate SVM memory
   *
   * @param svm_ptr pointer to the SVM memory to be deallocated
   */
  void releaseSVMRegion(void *svm_ptr);

  /**
   * @brief Destroy the Context Manager object
   *
   */
  ~ContextManager();

private:
  cl_platform_id platform_id_ = nullptr;
  cl_device_id device_id_ = nullptr;
  cl_context context_ = nullptr;
  std::unique_ptr<DeviceInfo> device_info_;
};
} // namespace nntrainer::opencl
#endif // __OPENCL_CONTEXT_MANAGER_H__
