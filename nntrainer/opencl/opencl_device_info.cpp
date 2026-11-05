// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   opencl_device_info.cpp
 * @date   18 August 2025
 * @brief  Read and store OpenCL Device Info
 * @see    https://github.com/nntrainer/nntrainer
 * @author Grzegorz Kisala <g.kisala@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#include "opencl_device_info.h"

#include <iostream>
#include <sstream>

#include "nntrainer_log.h"
#include "opencl_loader.h"

namespace nntrainer::opencl {

bool DeviceInfo::read(const cl_device_id device_id) {

  // CL_DEVICE_TYPE
  if (CL_SUCCESS != clGetDeviceInfo(device_id, CL_DEVICE_TYPE,
                                    sizeof(cl_device_type), &device_type_,
                                    nullptr)) {
    ml_loge("Failed to query for CL_DEVICE_TYPE");
    return false;
  }

  // CL_DEVICE_VENDOR_ID
  if (CL_SUCCESS != clGetDeviceInfo(device_id, CL_DEVICE_VENDOR_ID,
                                    sizeof(cl_uint), &device_vendor_id_,
                                    nullptr)) {
    ml_loge("Failed to query for CL_DEVICE_VENDOR_ID");
    return false;
  }

  // CL_DEVICE_MAX_COMPUTE_UNITS
  if (CL_SUCCESS != clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS,
                                    sizeof(cl_uint), &device_max_compute_units_,
                                    nullptr)) {
    ml_loge("Failed to query for CL_DEVICE_MAX_COMPUTE_UNITS");
    return false;
  }

  // CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS
  if (CL_SUCCESS !=
      clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                      sizeof(cl_uint), &device_max_work_item_dimensions_,
                      nullptr)) {
    ml_loge("Failed to query for CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS");
    return false;
  }

  // CL_DEVICE_MAX_WORK_ITEM_SIZES
  device_max_work_item_sizes_.resize(device_max_work_item_dimensions_);
  if (CL_SUCCESS !=
      clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                      sizeof(size_t) * device_max_work_item_sizes_.size(),
                      device_max_work_item_sizes_.data(), nullptr)) {
    ml_loge("Failed to query for CL_DEVICE_MAX_WORK_ITEM_SIZES");
    return false;
  }

  // CL_DEVICE_MAX_WORK_GROUP_SIZE
  if (CL_SUCCESS != clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                                    sizeof(size_t),
                                    &device_max_work_group_size_, nullptr)) {
    ml_loge("Failed to query for CL_DEVICE_MAX_WORK_GROUP_SIZE");
    return false;
  }

  // CL_DEVICE_MAX_MEM_ALLOC_SIZE
  if (CL_SUCCESS != clGetDeviceInfo(device_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                                    sizeof(cl_ulong),
                                    &device_max_mem_alloc_size_, nullptr)) {
    ml_loge("Failed to query for CL_DEVICE_MAX_MEM_ALLOC_SIZE");
    return false;
  }

  // CL_DEVICE_NAME
  size_t device_name_size = 0;

  if (CL_SUCCESS != clGetDeviceInfo(device_id, CL_DEVICE_NAME, 0, nullptr,
                                    &device_name_size)) {
    ml_loge("Failed to query for CL_DEVICE_NAME");
    return false;
  }

  device_name_.resize(device_name_size);
  if (CL_SUCCESS != clGetDeviceInfo(device_id, CL_DEVICE_NAME, device_name_size,
                                    device_name_.data(), nullptr)) {
    ml_loge("Failed to query for CL_DEVICE_NAME");
    return false;
  }

  // CL_DEVICE_VENDOR
  size_t device_vendor_size = 0;

  if (CL_SUCCESS != clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, 0, nullptr,
                                    &device_vendor_size)) {
    ml_loge("Failed to query for CL_DEVICE_VENDOR");
    return false;
  }

  device_vendor_.resize(device_vendor_size);
  if (CL_SUCCESS != clGetDeviceInfo(device_id, CL_DEVICE_VENDOR,
                                    device_vendor_size, device_vendor_.data(),
                                    nullptr)) {
    ml_loge("Failed to query for CL_DEVICE_VENDOR");
    return false;
  }

  // CL_DRIVER_VERSION
  size_t driver_version_size = 0;

  if (CL_SUCCESS != clGetDeviceInfo(device_id, CL_DRIVER_VERSION, 0, nullptr,
                                    &driver_version_size)) {
    ml_loge("Failed to query for CL_DRIVER_VERSION");
    return false;
  }

  driver_version_.resize(driver_version_size);
  if (CL_SUCCESS != clGetDeviceInfo(device_id, CL_DRIVER_VERSION,
                                    driver_version_size, driver_version_.data(),
                                    nullptr)) {
    ml_loge("Failed to query for CL_DRIVER_VERSION");
    return false;
  }

  // CL_DEVICE_EXTENSIONS
  size_t device_extensions_size = 0;

  if (CL_SUCCESS != clGetDeviceInfo(device_id, CL_DEVICE_EXTENSIONS, 0, nullptr,
                                    &device_extensions_size)) {
    ml_loge("Failed to query for CL_DEVICE_EXTENSIONS");
    return false;
  }

  device_extensions_.resize(device_extensions_size);
  if (CL_SUCCESS != clGetDeviceInfo(device_id, CL_DEVICE_EXTENSIONS,
                                    device_extensions_size,
                                    device_extensions_.data(), nullptr)) {
    ml_loge("Failed to query for CL_DEVICE_EXTENSIONS");
    return false;
  }

  // CL_DEVICE_SVM_CAPABILITIES
  if (CL_SUCCESS != clGetDeviceInfo(device_id, CL_DEVICE_SVM_CAPABILITIES,
                                    sizeof(cl_device_svm_capabilities),
                                    &device_svm_capabilities_, nullptr)) {
    ml_loge("Failed to query for CL_DEVICE_SVM_CAPABILITIES");
    return false;
  }

  // CL_DEVICE_SINGLE_FP_CONFIG (optional: a failed query leaves it 0)
  if (CL_SUCCESS != clGetDeviceInfo(device_id, CL_DEVICE_SINGLE_FP_CONFIG,
                                    sizeof(cl_device_fp_config),
                                    &device_single_fp_config_, nullptr))
    device_single_fp_config_ = 0;

  return true;
}

void DeviceInfo::print() const {
  ml_logi("Device info:");

  {
    std::ostringstream oss;
    oss << "    CL_DEVICE_TYPE: ";
    switch (getDeviceType()) {
    case CL_DEVICE_TYPE_DEFAULT:
      oss << "CL_DEVICE_TYPE_DEFAULT";
      break;
    case CL_DEVICE_TYPE_CPU:
      oss << "CL_DEVICE_TYPE_CPU";
      break;
    case CL_DEVICE_TYPE_GPU:
      oss << "CL_DEVICE_TYPE_GPU";
      break;
    case CL_DEVICE_TYPE_ACCELERATOR:
      oss << "CL_DEVICE_TYPE_ACCELERATOR";
      break;
    case CL_DEVICE_TYPE_CUSTOM:
      oss << "CL_DEVICE_TYPE_CUSTOM";
      break;
    default:
      oss << "UNKNOWN";
      break;
    }
    ml_logi("%s", oss.str().data());
  }

  {
    std::ostringstream oss;
    oss << "    CL_DEVICE_VENDOR_ID: " << std::hex << "0x"
        << getDeviceVendorId();
    ml_logi("%s", oss.str().data());
  }

  {
    std::ostringstream oss;
    oss << "    CL_DEVICE_MAX_COMPUTE_UNITS: " << getDeviceMaxComputeUnits();
    ml_logi("%s", oss.str().data());
  }

  {
    std::ostringstream oss;
    oss << "    CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: "
        << getDeviceMaxWorkItemDimensions();
    ml_logi("%s", oss.str().data());
  }

  {
    std::ostringstream oss;
    oss << "    CL_DEVICE_MAX_WORK_ITEM_SIZES:";
    for (auto size : getDeviceMaxWorkItemSizes()) {
      oss << " " << size;
    }
    ml_logi("%s", oss.str().data());
  }

  {
    std::ostringstream oss;
    oss << "    CL_DEVICE_MAX_WORK_GROUP_SIZE: " << getDeviceMaxWorkGroupSize();
    ml_logi("%s", oss.str().data());
  }

  {
    std::ostringstream oss;
    oss << "    CL_DEVICE_MAX_MEM_ALLOC_SIZE: " << getDeviceMaxMemAllocSize();
    ml_logi("%s", oss.str().data());
  }

  {
    std::ostringstream oss;
    oss << "    CL_DEVICE_NAME: " << getDeviceName();
    ml_logi("%s", oss.str().data());
  }

  {
    std::ostringstream oss;
    oss << "    CL_DEVICE_VENDOR: " << getDeviceVendor();
    ml_logi("%s", oss.str().data());
  }

  {
    std::ostringstream oss;
    oss << "    CL_DRIVER_VERSION: " << getDriverVersion();
    ml_logi("%s", oss.str().data());
  }

  {
    std::ostringstream oss;
    oss << "    CL_DEVICE_EXTENSIONS: " << getDeviceExtensions();
    ml_logi("%s", oss.str().data());
  }

  {
    std::ostringstream oss;
    oss << "    CL_DEVICE_SVM_CAPABILITIES: ";
    if (getDeviceSVMCapabilities() & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER) {
      oss << "CL_DEVICE_SVM_COARSE_GRAIN_BUFFER ";
    }
    if (getDeviceSVMCapabilities() & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) {
      oss << "CL_DEVICE_SVM_FINE_GRAIN_BUFFER ";
    }
    if (getDeviceSVMCapabilities() & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) {
      oss << "CL_DEVICE_SVM_FINE_GRAIN_SYSTEM ";
    }
    if (getDeviceSVMCapabilities() & CL_DEVICE_SVM_ATOMICS) {
      oss << "CL_DEVICE_SVM_ATOMICS ";
    }
    ml_logi("%s", oss.str().data());
  }
}

} // namespace nntrainer::opencl
