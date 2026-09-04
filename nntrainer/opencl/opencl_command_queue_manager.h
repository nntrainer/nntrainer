// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file    opencl_command_queue_manager.h
 * @date    06 Feb 2024
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Debadri Samaddar <s.debadri@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   OpenCL wrapper for command queue management
 *
 */

#ifndef __OPENCL_COMMAND_QUEUE_MANAGER_H__
#define __OPENCL_COMMAND_QUEUE_MANAGER_H__

#include "CL/cl.h"
#include "opencl_kernel.h"
#include "singleton.h"
#include <memory>

namespace nntrainer::opencl {

/**
 * @class CommandQueueManager contains wrappers for managing OpenCL command
 * queue
 * @brief OpenCL command queue wrapper
 *
 */
class CommandQueueManager : public Singleton<CommandQueueManager> {

  /**
   * @brief cl_command_queue instance
   *
   */
  cl_command_queue command_queue_{nullptr};

public:
  /**
   * @brief Create a Command Queue object
   *
   * @return true if creation is successful or false otherwise
   */
  bool CreateCommandQueue();

  /**
   * @brief Release th OpenCL command queue instance
   *
   */
  void ReleaseCommandQueue();

  /**
   * @brief Reading buffer object. Used from Buffer class
   *
   * @param buffer cl_mem buffer object
   * @param size_in_bytes size of data
   * @param data getting the data stored in buffer
   * @param async flag for asynchronous operation
   * @return true if reading is successful or false otherwise
   */
  bool EnqueueReadBuffer(cl_mem buffer, size_t size_in_bytes, void *data,
                         bool async = false);

  /**
   * @brief Reading 1D region from a buffer object. Used from Buffer class
   *
   * @param buffer cl_mem buffer object
   * @param size_in_bytes size of data region
   * @param data pointer for the region
   * @param host_origin_offset offset in the host memory region
   * @param buffer_origin_offset offset in the buffer memory region
   * @param async flag for asynchronous operation
   * @return true if reading is successful or false otherwise
   */
  bool EnqueueReadBufferRegion(cl_mem buffer, size_t size_in_bytes, void *data,
                               size_t host_origin_offset = 0,
                               size_t buffer_origin_offset = 0,
                               bool async = false);

  /**
   * @brief Writing buffer object. Used from Buffer class
   *
   * @param buffer cl_mem buffer object
   * @param size_in_bytes size of data
   * @param data to be enqueued into the buffer
   * @param async flag for asynchronous operation
   * @return true if writing is successful or false otherwise
   */
  bool EnqueueWriteBuffer(cl_mem buffer, size_t size_in_bytes, const void *data,
                          bool async = false);

  /**
   * @brief Writing 1D region of a buffer object. Used from Buffer class
   *
   * @param buffer cl_mem buffer object
   * @param size_in_bytes size of data region
   * @param data pointer for the region
   * @param origin_offset offset in the memory region
   * @param async flag for asynchronous operation
   * @return true if writing is successful or false otherwise
   */
  bool EnqueueWriteBufferRegion(cl_mem buffer, size_t size_in_bytes,
                                const void *data, size_t host_origin_offset = 0,
                                size_t buffer_origin_offset = 0,
                                bool async = false);
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
  void *EnqueueMapBuffer(cl_mem buffer, size_t offset_in_bytes,
                         size_t size_in_bytes, bool read_only,
                         bool async = false, cl_event *event = nullptr);

  /**
   * @brief Un-mapping a buffer object from the host address space
   *
   * @param buffer cl_mem buffer object
   * @param mapped_ptr pointer to the mapped region
   * @param event Object that identifies this command and can be used to query
   * or wait for this command to complete
   * @return true if unmap is successful
   */
  bool EnqueueUnmapMemObject(cl_mem buffer, void *mapped_ptr,
                             cl_event *event = nullptr);

  /**
   * @brief Enqueue SVM memory map operation.
   *
   * @param svm_ptr Pointer to the SVM memory region to be mapped
   * @param size Size of the SVM memory region to be mapped
   * @param read_only Flag indicating whether the SVM memory should be mapped
   * for read-only access (true) or read-write access (false).
   * @param event Optional event object that can be used to query or wait for
   * the mapping operation to complete.
   * @param async Enqueue the map without blocking the host. Only sound on an
   * in-order queue when the region is next touched by another device
   * operation, never by the host. Defaults to the blocking behaviour.
   * @note async is appended AFTER event, not inserted before it. This header
   * is installed, so an out-of-tree caller written against the previous
   * signature passes a cl_event * in the fourth position; inserting a bool
   * there would still compile -- cl_event * converts to bool -- and silently
   * turn a blocking, event-returning map into a non-blocking one whose event
   * is discarded. Appending keeps every existing call meaning what it meant.
   * @return true if mapping is successful, false otherwise.
   */
  bool enqueueSVMMap(void *svm_ptr, size_t size, bool read_only,
                     cl_event *event = nullptr, bool async = false);

  /**
   * @brief Enqueue SVM memory unmap operation.
   *
   * This function unmaps a previously mapped SVM memory region.
   *
   * @param svm_ptr Pointer to the SVM memory region to be unmapped
   * @param event  Optional event object that can be used to query or wait for
   * the mapping operation to complete. If not provided, the mapping will be
   * blocking.
   * @return true if unmapping is successful, false otherwise.
   */
  bool enqueueSVMUnmap(void *svm_ptr, cl_event *event = nullptr);

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
  bool DispatchCommand(Kernel kernel, const int (&work_groups_count)[3],
                       const int (&work_group_size)[3],
                       cl_event *event = nullptr,
                       std::vector<cl_event> events_to_wait = {});

  /**
   * @brief Overloaded function to initiate execution of the command queue.
   *
   * @param kernel_ptr reference of OpenCL kernel shared_ptr
   * @param work_groups_count Total number of work items that will execute the
   * kernel function
   * @param work_group_size Number of work items that make up a work group
   * @param event Object that identifies this command and can be used to query
   * or wait for this command to complete
   * @return true if command queue execution is successful or false otherwise
   */
  bool DispatchCommand(const std::shared_ptr<Kernel> &kernel_ptr,
                       const int (&work_groups_count)[3],
                       const int (&work_group_size)[3],
                       cl_event *event = nullptr,
                       std::vector<cl_event> events_to_wait = {});

  /**
   * @brief Get the OpenCL Command Queue object
   *
   * @return const cl_command_queue
   */
  const cl_command_queue GetCommandQueue();

  /**
   * @brief Destroy the Command Queue Manager object
   *
   */
  ~CommandQueueManager();

  /**
   * @brief Wrapper to OpenCL function to enqueue a command to execute a kernel
   * on a device
   *
   * @param kernel OpenCL kernel
   * @param work_dim Number of dimensions used to specify the global work-items
   * and work-items in the work-group
   * @param global_work_size Total number of work items that will execute the
   * kernel function
   * @param local_work_size Number of work items that make up a work group
   * @param num_events_in_wait_list Number of events that need to complete
   * before this particular command can be executed
   * @param event_wait_list Events that need to complete before this particular
   * command can be executed
   * @param event event object that identifies this command and can be used to
   * query or wait for this command to complete
   */
  void enqueueKernel(const cl_kernel kernel, const cl_uint work_dim,
                     const size_t *global_work_size,
                     const size_t *local_work_size,
                     cl_uint num_events_in_wait_list = 0,
                     const cl_event *event_wait_list = nullptr,
                     cl_event *event = nullptr);
};
} // namespace nntrainer::opencl

#endif // __OPENCL_COMMAND_QUEUE_MANAGER_H__
