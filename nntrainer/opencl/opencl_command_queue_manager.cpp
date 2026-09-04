// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file    opencl_command_queue_manager.cpp
 * @date    06 Feb 2024
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Debadri Samaddar <s.debadri@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   OpenCL wrapper for command queue management
 *
 */

#include "opencl_command_queue_manager.h"

#include "opencl_context_manager.h"
#include "opencl_loader.h"

#include <nntrainer_error.h>
#include <nntrainer_log.h>

namespace nntrainer::opencl {

namespace {

/**
 * @brief Whether this device needs an explicit flush between a kernel that
 *        writes shared virtual memory and the kernel that reads it.
 *
 * An in-order queue does not by itself make a coarse-grain shared-memory
 * handoff visible to the next kernel; fine-grain buffer memory is coherent by
 * definition and needs nothing. CL_DEVICE_SVM_FINE_GRAIN_BUFFER is the
 * queryable difference, so the decision is derived rather than configured.
 * It is scoped to Intel, the vendor whose coarse-grain handoff is observably
 * not ordered by the queue alone; elsewhere the flush would only cost
 * throughput. Resolved once per process.
 */
bool needsCoarseSVMDrain() {
  static const bool drain = []() {
    const auto *device_info = ContextManager::Global().getDeviceInfo();
    if (!device_info)
      return false;
    constexpr cl_uint INTEL_VENDOR_ID = 0x8086;
    const cl_device_svm_capabilities svm =
      device_info->getDeviceSVMCapabilities();
    const bool fine_grain = (svm & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) != 0;
    return device_info->getDeviceVendorId() == INTEL_VENDOR_ID && !fine_grain;
  }();
  return drain;
}

} // namespace

/**
 * @brief Create a Command Queue object
 *
 * @return true if creation is successful or false otherwise
 */
bool CommandQueueManager::CreateCommandQueue() {
  if (command_queue_) {
    ml_logi("opencl_command_queue_manager: Retained command queue");
    // increments the command_queue reference count
    clRetainCommandQueue(command_queue_);
    return true;
  }

  int error_code;
  ContextManager &context_instance = ContextManager::Global();

  // OpenCL context is created
  cl_context context = context_instance.GetContext();

  // If context is invalid, return false
  if (context == nullptr) {
    return false;
  }

  // getting GPU device ID
  cl_device_id device_id = context_instance.GetDeviceId();

  // In-order queue. Nothing in this tree enqueues a wait list or waits on an
  // event, so out-of-order execution bought no overlap; what it did do is let
  // the device reorder two kernels that hand off through a shared buffer,
  // which is exactly how consecutive layers on the OpenCL memory allocator
  // communicate. Submission order is the only ordering guarantee those
  // handoffs have.
  command_queue_ = clCreateCommandQueue(context, device_id, 0, &error_code);
  if (!command_queue_) {
    ml_loge("Failed to create a command queue. OpenCL error code: %d : ",
            error_code, OpenCLErrorCodeToString(error_code));
    return false;
  }
  ml_logi("opencl_command_queue_manager: Created command queue");
  // increments the command_queue reference count
  clRetainCommandQueue(command_queue_);
  ml_logi("opencl_command_queue_manager: Retained command queue");
  return true;
}

/**
 * @brief Release th OpenCL command queue instance
 *
 */
void CommandQueueManager::ReleaseCommandQueue() {
  if (command_queue_) {
    ml_logi("opencl_command_queue_manager: Released command queue");
    clReleaseCommandQueue(command_queue_);
  }
}

/**
 * @brief Destroy the Command Queue Manager object
 *
 */
CommandQueueManager::~CommandQueueManager() {
  if (command_queue_) {
    ml_logi("opencl_command_queue_manager: Destroyed command queue");
    // decrements the command_queue reference count
    clReleaseCommandQueue(command_queue_);
    command_queue_ = nullptr;

    // releasing OpenCL context since it has been created by
    // CommandQueueManager::CreateCommandQueue
    ContextManager::Global().ReleaseContext();
  }
}

/**
 * @brief Get the OpenCL Command Queue object
 *
 * @return const cl_command_queue
 */
const cl_command_queue CommandQueueManager::GetCommandQueue() {
  return command_queue_;
}

/**
 * @brief Reading buffer object. Used from Buffer class
 *
 * @param buffer cl_mem buffer object
 * @param size_in_bytes size of data
 * @param data getting the data stored in buffer
 * @param async flag for asynchronous operation
 * @return true if reading is successful or false otherwise
 */
bool CommandQueueManager::EnqueueReadBuffer(cl_mem buffer, size_t size_in_bytes,
                                            void *data, bool async) {

  // managing synchronization
  const cl_bool blocking = async ? CL_FALSE : CL_TRUE;
  // returns NULL with error code if fails
  auto error_code =
    clEnqueueReadBuffer(command_queue_, buffer, blocking, 0, size_in_bytes,
                        data, 0, nullptr, nullptr);
  if (error_code != CL_SUCCESS) {
    ml_loge("Failed to read data from GPU (clEnqueueReadBuffer). OpenCL error "
            "code: %d : %s",
            error_code, OpenCLErrorCodeToString(error_code));
    return false;
  }

  return true;
}

bool CommandQueueManager::EnqueueReadBufferRegion(
  cl_mem buffer, size_t size_in_bytes, void *data, size_t host_origin_offset,
  size_t buffer_origin_offset, bool async) {

  // managing synchronization
  const cl_bool blocking = async ? CL_FALSE : CL_TRUE;

  // (x, y, z) offset in the memory region associated with buffer
  const size_t buffer_origin[] = {buffer_origin_offset, 0, 0};
  // (x, y, z) offset in the memory region associated with host
  const size_t host_origin[] = {host_origin_offset, 0, 0};
  // region defines the (width in bytes, height in rows, depth in slices)
  const size_t region[] = {size_in_bytes, 1, 1};
  // length of each row in bytes
  size_t row_pitch = region[0];
  // length of each 2D slice in bytes
  size_t slice_pitch = region[0] * region[1];

  // Buffer and host data are interpreted as 1D in this case
  // hence row and slice pitch are same for both
  cl_int error_code = clEnqueueReadBufferRect(
    command_queue_, buffer, blocking, buffer_origin, host_origin, region,
    row_pitch, slice_pitch, row_pitch, slice_pitch, data, 0, nullptr, nullptr);

  if (error_code != CL_SUCCESS) {
    ml_loge("Failed to write data region to GPU (clEnqueueReadBufferRect). "
            "OpenCL error "
            "code: %d : %s",
            error_code, OpenCLErrorCodeToString(error_code));
    return false;
  }

  return true;
}

/**
 * @brief Writing buffer object. Used from Buffer class
 *
 * @param buffer cl_mem buffer object
 * @param size_in_bytes size of data
 * @param data to be enqueued into the buffer
 * @param async flag for asynchronous operation
 * @return true if writing is successful or false otherwise
 */
bool CommandQueueManager::EnqueueWriteBuffer(cl_mem buffer,
                                             size_t size_in_bytes,
                                             const void *data, bool async) {

  // managing synchronization
  const cl_bool blocking = async ? CL_FALSE : CL_TRUE;
  // returns NULL with error code if fails
  auto error_code =
    clEnqueueWriteBuffer(command_queue_, buffer, blocking, 0, size_in_bytes,
                         data, 0, nullptr, nullptr);

  if (error_code != CL_SUCCESS) {
    ml_loge("Failed to upload data to GPU (clEnqueueWriteBuffer). OpenCL error "
            "code: %d : %s",
            error_code, OpenCLErrorCodeToString(error_code));
    return false;
  }

  return true;
}

bool CommandQueueManager::EnqueueWriteBufferRegion(
  cl_mem buffer, size_t size_in_bytes, const void *data,
  size_t host_origin_offset, size_t buffer_origin_offset, bool async) {

  // managing synchronization
  const cl_bool blocking = async ? CL_FALSE : CL_TRUE;

  // (x, y, z) offset in the memory region associated with buffer
  const size_t buffer_origin[] = {buffer_origin_offset, 0, 0};
  // (x, y, z) offset in the memory region associated with host
  const size_t host_origin[] = {host_origin_offset, 0, 0};
  // region defines the (width in bytes, height in rows, depth in slices)
  const size_t region[] = {size_in_bytes, 1, 1};
  // length of each row in bytes
  size_t row_pitch = region[0];
  // length of each 2D slice in bytes
  size_t slice_pitch = region[0] * region[1];

  // Buffer and host data are interpreted as 1D in this case
  // hence row and slice pitch are same for both
  cl_int error_code = clEnqueueWriteBufferRect(
    command_queue_, buffer, blocking, buffer_origin, host_origin, region,
    row_pitch, slice_pitch, row_pitch, slice_pitch, data, 0, nullptr, nullptr);

  if (error_code != CL_SUCCESS) {
    ml_loge("Failed to write data region to GPU (clEnqueueWriteBufferRect). "
            "OpenCL error "
            "code: %d : %s",
            error_code, OpenCLErrorCodeToString(error_code));
    return false;
  }

  return true;
}

/**
 * @brief Mapping a region of a buffer object into the host address space
 *
 * @param buffer cl_mem buffer object
 * @param offset_in_bytes offset of the region in the buffer object that is
 * being mapped
 * @param size_in_bytes size of the buffer object that is being mapped
 * @param read_only flag for read only mapping
 * @param async flag for asynchronous operation
 * @param event Object that identifies this command and can be used to query
 * or wait for this command to complete
 * @return void* pointer to the mapped region
 */
void *CommandQueueManager::EnqueueMapBuffer(cl_mem buffer,
                                            size_t offset_in_bytes,
                                            size_t size_in_bytes,
                                            bool read_only, bool async,
                                            cl_event *event) {
  // managing synchronization
  const cl_bool blocking = async ? CL_FALSE : CL_TRUE;
  // managing read/write flags
  const cl_map_flags map_flag = read_only ? CL_MAP_READ : CL_MAP_WRITE;

  cl_int error_code;

  void *host_mem_buf = clEnqueueMapBuffer(
    command_queue_, buffer, blocking, map_flag, offset_in_bytes, size_in_bytes,
    0, nullptr, event, &error_code);

  if (error_code != CL_SUCCESS) {
    ml_loge(
      "Failed to map buffer to host memory(clEnqueueMapBuffer). OpenCL error "
      "code: %d : %s",
      error_code, OpenCLErrorCodeToString(error_code));
    return nullptr;
  }
  return host_mem_buf;
}

/**
 * @brief Mapping a region of a buffer object into the host address space
 *
 * @param buffer cl_mem buffer object
 * @param mapped_ptr pointer to the mapped region
 * @param event Object that identifies this command and can be used to query
 * or wait for this command to complete
 * @return true if unmap is successful
 */
bool CommandQueueManager::EnqueueUnmapMemObject(cl_mem buffer, void *mapped_ptr,
                                                cl_event *event) {
  cl_int error_code = clEnqueueUnmapMemObject(command_queue_, buffer,
                                              mapped_ptr, 0, nullptr, event);
  if (error_code != CL_SUCCESS) {
    ml_loge("Failed to unmap buffer from host memory(clEnqueueUnmapMemObject). "
            "OpenCL error "
            "code: %d : %s",
            error_code, OpenCLErrorCodeToString(error_code));
    return false;
  }
  return true;
}

bool CommandQueueManager::enqueueSVMMap(void *svm_ptr, size_t size,
                                        bool read_only, cl_event *event,
                                        bool async) {
  // managing read/write flags
  const cl_map_flags map_flag = read_only ? CL_MAP_READ : CL_MAP_WRITE;

  // async = true makes the map non-blocking. That is sound only on an in-order
  // queue, where the map is ordered ahead of the next operation's unmap, and
  // only when nothing on the host touches the region before that next GPU op
  // runs -- which is exactly the GPU-to-GPU handoff the quantized GEMM below
  // performs. It removes a host stall that otherwise drains the queue to idle
  // between two device operations. The default keeps the blocking behaviour
  // every existing caller has.
  const cl_bool blocking = async ? CL_FALSE : CL_TRUE;

  // The event out-parameter was accepted and then dropped on the floor here,
  // so a caller could never wait on the map it just enqueued. Pass it through.
  cl_int error_code = clEnqueueSVMMap(command_queue_, blocking, map_flag,
                                      svm_ptr, size, 0, nullptr, event);

  if (error_code != CL_SUCCESS) {
    ml_loge(
      "Failed to map SVM memory (clEnqueueSVMMap). OpenCL error code: %d : %s",
      error_code, OpenCLErrorCodeToString(error_code));
    return false;
  }
  return true;
}

bool CommandQueueManager::enqueueSVMUnmap(void *svm_ptr, cl_event *event) {
  cl_int error_code =
    clEnqueueSVMUnmap(command_queue_, svm_ptr, 0, nullptr, event);

  if (error_code != CL_SUCCESS) {
    ml_loge(
      "Failed to unmap SVM memory (clEnqueueSVMUnmap). OpenCL error code: "
      "%d : %s",
      error_code, OpenCLErrorCodeToString(error_code));
    return false;
  }
  return true;
}

/**
 * @brief Function to initiate execution of the command queue.
 *
 * @param kernel OpenCL kernel
 * @param work_groups_count Total number of work items that will execute the
 * kernel function
 * @param work_group_size Number of work items that make up a work group
 * @param event Object that identifies this command and can be used to query
 * or wait for this command to complete
 * @return true if command queue execution is successful or false otherwise
 */
bool CommandQueueManager::DispatchCommand(
  Kernel kernel, const int (&work_groups_count)[3],
  const int (&work_group_size)[3], cl_event *event,
  std::vector<cl_event> events_to_wait) {

  // work_dim of 2 has been hardcoded, might be modified later based on
  // requirements

  // setting the local_work_size referred to as the size of the
  // work-group
  const size_t local[3] = {static_cast<size_t>(work_group_size[0]),
                           static_cast<size_t>(work_group_size[1]),
                           static_cast<size_t>(work_group_size[2])};

  // setting the global_work_size that describe the number of global work-items
  const size_t global[3] = {static_cast<size_t>(work_groups_count[0]),
                            static_cast<size_t>(work_groups_count[1]),
                            static_cast<size_t>(work_groups_count[2])};

  cl_kernel kernel_ = kernel.GetKernel();

  // returns NULL with error code if fails
  const int error_code =
    clEnqueueNDRangeKernel(command_queue_, kernel_, 3, nullptr, global, local,
                           events_to_wait.size(), events_to_wait.data(), event);
  // Always consume the flag, so it never leaks onto the next dispatch.
  const bool touched_svm = Kernel::takeDispatchTouchedSVM();
  if (error_code != CL_SUCCESS) {
    ml_loge("Failed to clEnqueueNDRangeKernel. OpenCL error code: %d : %s",
            error_code, OpenCLErrorCodeToString(error_code));
    return false;
  }

  if (touched_svm && needsCoarseSVMDrain())
    clFinish(command_queue_);

  return true;
}

bool CommandQueueManager::DispatchCommand(
  const std::shared_ptr<Kernel> &kernel_ptr, const int (&work_groups_count)[3],
  const int (&work_group_size)[3], cl_event *event,
  std::vector<cl_event> events_to_wait) {

  // work_dim of 2 has been hardcoded, might be modified later based on
  // requirements

  // setting the local_work_size referred to as the size of the
  // work-group
  const size_t local[3] = {static_cast<size_t>(work_group_size[0]),
                           static_cast<size_t>(work_group_size[1]),
                           static_cast<size_t>(work_group_size[2])};

  // setting the global_work_size that describe the number of global work-items
  const size_t global[3] = {static_cast<size_t>(work_groups_count[0]),
                            static_cast<size_t>(work_groups_count[1]),
                            static_cast<size_t>(work_groups_count[2])};

  cl_kernel kernel_ = kernel_ptr->GetKernel();

  // returns NULL with error code if fails
  const int error_code =
    clEnqueueNDRangeKernel(command_queue_, kernel_, 3, nullptr, global, local,
                           events_to_wait.size(), events_to_wait.data(), event);
  // Always consume the flag, so it never leaks onto the next dispatch.
  const bool touched_svm = Kernel::takeDispatchTouchedSVM();
  if (error_code != CL_SUCCESS) {
    ml_loge("Failed to clEnqueueNDRangeKernel. OpenCL error code: %d : %s",
            error_code, OpenCLErrorCodeToString(error_code));
    return false;
  }

  if (touched_svm && needsCoarseSVMDrain())
    clFinish(command_queue_);

  return true;
}

void CommandQueueManager::enqueueKernel(const cl_kernel kernel,
                                        const cl_uint work_dim,
                                        const size_t *global_work_size,
                                        const size_t *local_work_size,
                                        cl_uint num_events_in_wait_list,
                                        const cl_event *event_wait_list,
                                        cl_event *event) {

  const auto error_code = clEnqueueNDRangeKernel(
    command_queue_, kernel, work_dim, nullptr, global_work_size,
    local_work_size, num_events_in_wait_list, event_wait_list, event);

  // Always consume the flag, so it never leaks onto the next dispatch.
  const bool touched_svm = Kernel::takeDispatchTouchedSVM();

  NNTR_THROW_IF(error_code != CL_SUCCESS, std::runtime_error)
    << "clEnqueueNDRangeKernel failed. OpenCL error code: " << error_code
    << ", error: " << OpenCLErrorCodeToString(error_code);

  // The attention and rotary-embedding kernels come through here rather than
  // DispatchCommand, and they are the shared-memory producers and consumers,
  // so the flush that keeps their handoff coherent has to live here too.
  if (touched_svm && needsCoarseSVMDrain())
    clFinish(command_queue_);
}

} // namespace nntrainer::opencl
