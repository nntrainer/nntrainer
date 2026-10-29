// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   opencl_device_info.h
 * @date   18 August 2025
 * @brief  Read and store OpenCL Device Info
 * @see    https://github.com/nntrainer/nntrainer
 * @author Grzegorz Kisala <g.kisala@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#ifndef __NNTRAINER_OPENCL_DEVICE_INFO__
#define __NNTRAINER_OPENCL_DEVICE_INFO__

#include <memory>
#include <string>
#include <vector>

#include "CL/cl.h"

namespace nntrainer::opencl {

/**
 * @class DeviceInfo contains OpenCL Device Info
 * @brief Load OpenCL Device Info
 *
 */
class DeviceInfo {
public:
  bool read(const cl_device_id device_id);

  void print() const;

  cl_device_type getDeviceType() const { return device_type_; }

  cl_uint getDeviceVendorId() const { return device_vendor_id_; }

  cl_uint getDeviceMaxComputeUnits() const { return device_max_compute_units_; }

  cl_uint getDeviceMaxWorkItemDimensions() const {
    return device_max_work_item_dimensions_;
  }

  const std::vector<size_t> &getDeviceMaxWorkItemSizes() const {
    return device_max_work_item_sizes_;
  }

  size_t getDeviceMaxWorkGroupSize() const {
    return device_max_work_group_size_;
  }

  cl_ulong getDeviceMaxMemAllocSize() const {
    return device_max_mem_alloc_size_;
  }

  const std::string &getDeviceName() const { return device_name_; }

  const std::string &getDeviceVendor() const { return device_vendor_; }

  const std::string &getDriverVersion() const { return driver_version_; }

  const std::string &getDeviceExtensions() const { return device_extensions_; }

  cl_device_svm_capabilities getDeviceSVMCapabilities() const {
    return device_svm_capabilities_;
  }

  /** CL_DEVICE_SINGLE_FP_CONFIG; 0 when the device did not answer */
  cl_device_fp_config getDeviceSingleFpConfig() const {
    return device_single_fp_config_;
  }

private:
  // Order of parameters based on
  // https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html#device-queries-table
  cl_device_type device_type_ = {};
  cl_uint device_vendor_id_ = {};
  cl_uint device_max_compute_units_ = {};
  cl_uint device_max_work_item_dimensions_ = {};
  std::vector<size_t> device_max_work_item_sizes_ = {};
  size_t device_max_work_group_size_ = {};
  cl_ulong device_max_mem_alloc_size_ = {};
  std::string device_name_ = {};
  std::string device_vendor_ = {};
  std::string driver_version_ = {};
  std::string device_extensions_ = {};
  cl_device_svm_capabilities device_svm_capabilities_ = {};
  cl_device_fp_config device_single_fp_config_ = {};
};

} // namespace nntrainer::opencl

#endif // __NNTRAINER_OPENCL_DEVICE_INFO__
