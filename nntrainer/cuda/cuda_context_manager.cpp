// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd.
 *
 * @file    cuda_context_manager.cpp
 * @date    11 Dec 2025
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Donghak Jung <dk11.jung@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   CUDA wrapper for context management
 *
 */

#include "cuda_context_manager.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include "cuda_util.h"

namespace nntrainer::cuda {

bool ContextManager::CreateDefaultGPUDevice() {
  if (initialized_)
    return true;

  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));

  if (device_count == 0) {
    ml_loge("No CUDA compatible GPU found based on cudaGetDeviceCount");
    return false;
  }

  // Use the first device by default
  device_id_ = 0;
  CUDA_CHECK(cudaSetDevice(device_id_));

  return true;
}

bool ContextManager::CreateContext() {
  if (!CreateDefaultGPUDevice()) {
    return false;
  }

  // Check if primary context is active using Driver API
  // Note: We need to include <cuda.h> and link against cuda.lib (Driver API)
  // CUDA Runtime API initializes context lazily, but here we check status explicitly.
  
  CUdevice device;
  CU_CHECK(cuDeviceGet(&device, device_id_));

  unsigned int flags = 0;
  int active = 0;
  CU_CHECK(cuDevicePrimaryCtxGetState(device, &flags, &active));

  if (active) {
      initialized_ = true;
      return true;
  }
  
  // Try to force initialization with a dummy sync or similar if needed,
  // but if it's inactive and we just setDevice, typical runtime calls will activate it.
  // Let's force a sync to verify activation if not already.
  CUDA_CHECK(cudaDeviceSynchronize());
  initialized_ = true;
  return true;
}

void ContextManager::ReleaseContext() {
  if (initialized_) {
      CUDA_CHECK(cudaDeviceReset());
      initialized_ = false;
  }
}

int ContextManager::GetDeviceId() { return device_id_; }

#include <iostream>

void *ContextManager::allocateDeviceMemory(size_t size) {
#ifdef DEBUG
  std::cout << "[DEBUG] allocDeviceMemory called with size: " << size << std::endl;
#endif
  if (!initialized_) {
#ifdef DEBUG
      std::cout << "[DEBUG] Context not initialized, preventing creation" << std::endl;
#endif
      if (!CreateContext()) {
          throw std::runtime_error("Failed to initialize CUDA context");
      }
#ifdef DEBUG
      std::cout << "[DEBUG] Context initialized successfully" << std::endl;
#endif
  }

  void *ptr = nullptr;
  CUDA_CHECK(cudaMalloc(&ptr, size));
#ifdef DEBUG
  std::cout << "[DEBUG] cudaMalloc success, ptr: " << ptr << std::endl;
#endif
  return ptr;
}

void ContextManager::releaseDeviceMemory(void *ptr) {
  if (ptr) {
    CUDA_CHECK(cudaFree(ptr));
  }
}

ContextManager::~ContextManager() { 
    try {
        ReleaseContext(); 
    } catch (...) {
        ml_loge("Failed to release CUDA context in destructor");
    }
}

} // namespace nntrainer::cuda
