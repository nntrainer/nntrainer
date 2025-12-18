// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file    opencl_context_manager.cpp
 * @date    06 Feb 2024
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Debadri Samaddar <s.debadri@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   OpenCL wrapper for context management
 *
 */

#include "opencl_context_manager.h"

#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "CL/cl.h"
#include "opencl_loader.h"

#include <nntrainer_log.h>

namespace nntrainer::opencl {

/**
 * @brief Get the OpenCL context object
 *
 * @return const cl_context
 */
const cl_context &ContextManager::GetContext() {
  // loading the OpenCL library and required functions
  bool result = LoadOpenCL();

  if (!result) {
    context_ = nullptr;
    return context_;
  }

  if (context_) {
    // increments the context reference count
    auto error_code = clRetainContext(context_);
    if (error_code != CL_SUCCESS) {
      ml_loge("Failed to specify the OpenCL context to retain. OpenCL error "
              "code: %d : %s",
              error_code, OpenCLErrorCodeToString(error_code));
    }

    return context_;
  }

  do {
    result = CreateDefaultGPUDevice();
    if (!result) {
      break;
    }

    result = CreateCLContext();
    if (!result) {
      break;
    }

    // increments the context reference count
    clRetainContext(context_);

  } while (false);

  if (!result) {
    ml_loge("Failed to create OpenCL Context");
    context_ = nullptr;
  }

  return context_;
}

void ContextManager::ReleaseContext() {
  if (context_) {
    // decrements the context reference count
    clReleaseContext(context_);
  }
}

/**
 * @brief Get the Device Id object
 *
 * @return const cl_device_id
 */
const cl_device_id ContextManager::GetDeviceId() { return device_id_; }

void *ContextManager::createSVMRegion(size_t size) {
  if (context_)
    return clSVMAlloc(context_, CL_MEM_READ_WRITE, size, 0);
  else
    return nullptr;
}

void ContextManager::releaseSVMRegion(void *svm_ptr) {
  if (svm_ptr) {
    // deallocates the SVM memory
    clSVMFree(context_, svm_ptr);
  } else {
    ml_logw("Attempted to deallocate a null pointer");
  }
  svm_ptr = nullptr;
}

/**
 * @brief Destroy the Context Manager object
 *
 */
ContextManager::~ContextManager() {
  if (context_) {
    // decrements the context reference count
    clReleaseContext(context_);
    context_ = nullptr;
  }
}

/**
 * @brief Create a Default GPU Device object
 *
 * @return true if successful or false otherwise
 */
bool ContextManager::CreateDefaultGPUDevice() {
  std::vector<std::pair<cl_platform_id, cl_device_id>> platform_device_pairs;

  platform_id_ = nullptr;
  device_id_ = nullptr;

  static constexpr cl_device_type kDefaultQueryDeviceType = CL_DEVICE_TYPE_GPU;

  ml_logi("Collecting OpenCL platforms ...");

  cl_uint num_platforms = 0;
  cl_int status = clGetPlatformIDs(0, nullptr, &num_platforms);
  if (status != CL_SUCCESS) {
    ml_loge("clGetPlatformIDs returned %d : %s", status,
            OpenCLErrorCodeToString(status));
    return false;
  }
  if (num_platforms == 0) {
    ml_loge("No supported OpenCL platform.");
    return false;
  }

  std::vector<cl_platform_id> platforms(num_platforms);
  status = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
  if (status != CL_SUCCESS) {
    ml_loge("clGetPlatformIDs returned %d : %s", status,
            OpenCLErrorCodeToString(status));
    return false;
  }

  for (size_t i = 0; i < static_cast<size_t>(num_platforms); i++) {
    ml_logi("Collecting OpenCL devices for platform %d / %d ...", (int32_t)i,
            (int32_t)num_platforms);

    cl_uint num_devices = 0;
    status = clGetDeviceIDs(platforms[i], kDefaultQueryDeviceType, 0, nullptr,
                            &num_devices);
    if (status != CL_SUCCESS) {
      ml_loge("clGetDeviceIDs returned %d : %s", status,
              OpenCLErrorCodeToString(status));
      continue;
    }
    if (num_devices == 0) {
      ml_loge("No GPU on current platform.");
      continue;
    }

    std::vector<cl_device_id> devices(num_devices);
    status = clGetDeviceIDs(platforms[i], kDefaultQueryDeviceType, num_devices,
                            devices.data(), nullptr);
    if (status != CL_SUCCESS) {
      ml_loge("clGetDeviceIDs returned %d : %s", status,
              OpenCLErrorCodeToString(status));
      continue;
    }

    for (size_t j = 0; j < static_cast<size_t>(num_devices); j++) {
      platform_device_pairs.push_back(
        std::make_pair<>(platforms[i], devices[j]));
    }
  }

  ml_logi("Looking for suitable OpenCL platform and device ...");

  // Vendor ID of Intel : 0x8086
  // Vendor ID of NVidia : 0x10DE / 0x13B5
  constexpr static cl_uint intel_igpu_vendor_id = 0x8086;
  constexpr static cl_device_type intel_igpu_device_type = CL_DEVICE_TYPE_GPU;
  constexpr static const char *const intel_igpu_device_name_pfx = "Intel";

#define SEARCH_BY_NAME 1

  for (const std::pair<cl_platform_id, cl_device_id> &platform_device :
       platform_device_pairs) {
    cl_platform_id platform = platform_device.first;
    cl_device_id device = platform_device.second;

    auto device_info = std::make_unique<DeviceInfo>();
    if (!device_info->read(device)) {
      ml_loge("Failed to read device info");
      return false;
    }

    const bool type_check =
      (device_info->getDeviceType() == intel_igpu_device_type);

#if SEARCH_BY_NAME
    std::string device_name_query(intel_igpu_device_name_pfx);

    const bool vendor_check = (device_info->getDeviceName().find(
                                 device_name_query) != std::string::npos);
#else
    const bool vendor_check =
      (device_info->getDeviceVendorID() == intel_igpu_vendor_id);
#endif

#undef SEARCH_BY_NAME

    if (vendor_check && type_check) {
      platform_id_ = platform;
      device_id_ = device;
      device_info_ = std::move(device_info);
      break;
    }
  }

  if ((nullptr == platform_id_) || (nullptr == device_id_)) {
    if (platform_device_pairs.empty()) {
      ml_loge("No suitable platforms / device found - aborting OpenCL context "
              "creation.");
      return false;
    }
    ml_loge("No suitable platform / device found - using default (first)");
    platform_id_ = platform_device_pairs[0].first;
    device_id_ = platform_device_pairs[0].second;
    device_info_ = std::make_unique<DeviceInfo>();
    if (!device_info_->read(device_id_)) {
      ml_loge("Failed to read device info");
      return false;
    }
  }

  // Raport device name
  ml_logi("Using device %s", device_info_->getDeviceName().data());
  device_info_->print();

  return true;
}

/**
 * @brief Create OpenCL context
 *
 * @return true if successful or false otherwise
 */
bool ContextManager::CreateCLContext() {
  int error_code;
  cl_context_properties properties[] = {CL_CONTEXT_PLATFORM,
                                        (cl_context_properties)platform_id_, 0};

  // creating valid ARM GPU OpenCL context, will return NULL with error code if
  // fails
  context_ =
    clCreateContext(properties, 1, &device_id_, nullptr, nullptr, &error_code);
  if (!context_) {
    ml_loge("Failed to create a compute context. OpenCL error code: %d : %s",
            error_code, OpenCLErrorCodeToString(error_code));
    return false;
  }

  return true;
}
} // namespace nntrainer::opencl
