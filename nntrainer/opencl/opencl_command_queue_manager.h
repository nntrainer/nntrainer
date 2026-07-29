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
#include <string>

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

  /**
   * @brief optional suffix appended to the NEXT enqueued kernel's profile key
   * (consumed + cleared on the next enqueueKernel). Lets a caller split one
   * kernel's profile entry by shape, e.g. v8c_gemm_int8_int4 ->
   * ...:N9216_K2304. Host-only; never affects kernel behavior. Only read when
   * profiling is on.
   */
  std::string next_prof_label_;

  /**
   * @brief tri-state SVM coherence drain policy: -1 = not decided yet, 0 =
   * off, 1 = on. Decided by the owning Context after device enumeration
   * (setSvmCoherenceDrain) and never resolved here, so there is no window in
   * which this class has to guess from a device that may not exist yet.
   */
  int svm_coherence_drain_ = -1;

  /**
   * @brief Whether a dispatch that touched SVM must be drained (clFinish)
   * before the next one may consume its output.
   *
   * NNTR_XE3_SYNC overrides in both directions. Until the owning Context has
   * decided, this answers YES: a missing drain on a device that needs one
   * corrupts output silently, while an unnecessary drain only costs
   * throughput, so the unknown state must fail CLOSED.
   */
  bool needsSvmCoherenceDrain() const;

public:
  /**
   * @brief Set the SVM coherence drain policy for this queue.
   *
   * Called by the owning Context once its DeviceCaps are known
   * (svm_fine_grain + vendor), so the decision is taken where the capability
   * lives and this class stays free of device knowledge.
   *
   * @param enable true to drain after every SVM-touching dispatch
   */
  void setSvmCoherenceDrain(bool enable);

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
   * the mapping operation to complete. If not provided, the mapping will be
   * blocking.
   * @return true if mapping is successful, false otherwise.
   */
  bool enqueueSVMMap(void *svm_ptr, size_t size, bool read_only,
                     bool async = false, cl_event *event = nullptr);

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
  /**
   * @brief Get the process-wide instance (out-of-line override of
   *        Singleton<T>::Global() — one cl_command_queue set per process
   *        under shared linking; see ContextManager::Global() for the full
   *        static-vs-shared note).
   */
  static CommandQueueManager &Global();

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

  /**
   * @brief Finish the queue, then accumulate per-kernel GPU execution time
   * (from CL_PROFILING_COMMAND_START/END of events captured during
   * enqueueKernel) by kernel name and print a sorted breakdown. No-op unless
   * NNTR_OPENCL_PROFILING is set. Releases and clears captured events.
   *
   * Unlike clFinish-bracketed host stage timing (which measures out-of-order
   * queue catch-up, not real work), this reports true on-device kernel time.
   *
   * @param tag short label printed in the report header (e.g. "PREFILL").
   */
  void dumpProfile(const char *tag);

  /**
   * @brief set a suffix appended to the next enqueued kernel's profile key,
   * to split one kernel's aggregate entry by call-site/shape. No-op for kernel
   * execution; the label is consumed and cleared by the next enqueueKernel.
   */
  void setNextProfileLabel(std::string s) { next_prof_label_ = std::move(s); }

  /**
   * @brief Block until all previously enqueued commands have completed
   * (clFinish). Used as the host-coherence barrier before a host op reads a
   * GPU-resident (SVM) buffer that was written/mapped asynchronously.
   */
  void finish();
};
} // namespace nntrainer::opencl

#endif // __OPENCL_COMMAND_QUEUE_MANAGER_H__
