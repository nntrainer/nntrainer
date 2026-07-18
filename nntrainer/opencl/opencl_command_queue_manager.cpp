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

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

#include <nntrainer_error.h>
#include <nntrainer_log.h>

namespace nntrainer::opencl {

namespace {
// NNTR_XE3_SYNC, CAPS-DERIVED. The Xe3 (Panther Lake / NEO 26.22)
// in-order queue does not preserve kernel->kernel coherence for the
// COARSE-grain SVM handoffs the residency model relies on; Meteor's FINE-grain
// SVM is auto-coherent. The distinguishing factor is a queryable attribute —
// CL_DEVICE_SVM_FINE_GRAIN_BUFFER — so the per-dispatch clFinish no longer
// needs the mandatory NNTR_XE3_SYNC flag: an Intel device WITHOUT fine-grain
// SVM gets it, a fine-grain Intel (Meteor) and non-Intel devices do not. The
// env flag still overrides. Scoped to vendor=Intel so a coarse-grain Adreno
// (which is coherent without the hammer and would only lose throughput) is
// unaffected. Resolved once per process.
inline bool xe3_needs_sync() {
  static const bool sync = []() {
    const char *e = std::getenv("NNTR_XE3_SYNC");
    if (e) {
      const bool ov = std::atoi(e) != 0;
      fprintf(stderr, "[XE3_SYNC] env override NNTR_XE3_SYNC=%s -> drain %s\n",
              e, ov ? "ON" : "OFF");
      return ov; // explicit override
    }
    const auto *di = ContextManager::Global().getDeviceInfo();
    if (!di) {
      // Resolved before DeviceInfo was populated: a possible platform-specific
      // init-order gap. Fail safe to OFF but say so, since it silently disables
      // the coherence drain.
      fprintf(stderr, "[XE3_SYNC] DeviceInfo null at resolve time -> drain OFF "
                      "(set NNTR_XE3_SYNC=1 if the GPU output races)\n");
      return false;
    }
    const cl_device_svm_capabilities svm = di->getDeviceSVMCapabilities();
    const unsigned vendor = (unsigned)di->getDeviceVendorId();
    const bool fine_grain = (svm & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) != 0;
    const bool decision =
      vendor == 0x8086u && !fine_grain; // Intel coarse-grain
    // Bring-up banner (what flips Linux fine_grain=0 -> drain ON vs a
    // fine-grain driver): NNTR_GPU_VERBOSE only — it printed once per DLL
    // module (7x on Windows) and an SDK surface must stay quiet by default.
    if (std::getenv("NNTR_GPU_VERBOSE"))
      fprintf(stderr,
              "[XE3_SYNC] vendor=0x%x svm_caps=0x%x fine_grain_buffer=%d -> "
              "auto drain %s (override with NNTR_XE3_SYNC=1)\n",
              vendor, (unsigned)svm, (int)fine_grain, decision ? "ON" : "OFF");
    return decision;
  }();
  return sync;
}

/**
 * @brief Per-kernel GPU profiling registry entry, populated by
 * enqueueKernel when NNTR_OPENCL_PROFILING is set. Each entry owns one
 * cl_event reference that dumpProfile() releases. Single-threaded dispatch
 * path, so no lock needed.
 */
struct ProfRec {
  std::string name;
  cl_event evt;
};

std::vector<ProfRec> &profRecs() {
  static std::vector<ProfRec> v;
  return v;
}

bool profEnabled() {
  static const int e = std::getenv("NNTR_OPENCL_PROFILING") ? 1 : 0;
  return e != 0;
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

  // Queue ordering policy. With the SVM allocator pool (the default),
  // consecutive CL layers hand off through a shared SVM buffer with no host
  // round-trip, so the queue must execute in submission order -> in-order
  // queue. Only an explicit NNTR_GPU_SVM_POOL=0 reverts to the legacy
  // out-of-order queue (the legacy path serializes via per-layer host
  // round-trips, so OOO is harmless there). Matches the allocator default in
  // neuralnet.cpp — the two must flip together.
  cl_command_queue_properties qprops = 0;
  const char *_svm_pool_env = std::getenv("NNTR_GPU_SVM_POOL");
  if (_svm_pool_env && std::atoi(_svm_pool_env) == 0) {
    qprops |= CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
  }
  // Env-gated CL_QUEUE_PROFILING_ENABLE so v8c (and any other) callers can
  // collect per-command start/end timestamps without paying the profiling
  // tax in production runs. Set NNTR_OPENCL_PROFILING=1 to enable.
  if (std::getenv("NNTR_OPENCL_PROFILING")) {
    qprops |= CL_QUEUE_PROFILING_ENABLE;
  }
  // returns NULL with error code if fails
  command_queue_ =
    clCreateCommandQueue(context, device_id, qprops, &error_code);
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

CommandQueueManager &CommandQueueManager::Global() {
  // Out-of-line on purpose — single process-wide instance (see header note).
  static CommandQueueManager instance;
  instance.initializeOnce();
  return instance;
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

void CommandQueueManager::finish() {
  if (command_queue_)
    clFinish(command_queue_);
}

bool CommandQueueManager::enqueueSVMMap(void *svm_ptr, size_t size,
                                        bool read_only, bool async,
                                        cl_event *event) {
  // managing read/write flags
  const cl_map_flags map_flag = read_only ? CL_MAP_READ : CL_MAP_WRITE;

  // async=true => non-blocking map (CL_FALSE). Safe ONLY on an in-order queue
  // (NNTR_GPU_SVM_POOL path) where the map is ordered before the next op's
  // unmap/kernel, AND when no host access of this region happens before that
  // next GPU op. Removes the per-op host stall that otherwise drains the queue
  // to idle. Default (false) keeps the original blocking behavior.
  const cl_bool blocking = async ? CL_FALSE : CL_TRUE;

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

  // Profiling: capture a tracked event like enqueueKernel does. Without this,
  // every DispatchCommand-dispatched kernel (rmsnorm/geglu/v8c writers/...)
  // is INVISIBLE to dumpProfile and its GPU time is mis-attributed as
  // "inter-kernel idle" of the surrounding tracked kernels.
  cl_event local_evt = nullptr;
  cl_event *evt_arg = event;
  const bool track = profEnabled() && evt_arg == nullptr;
  if (track)
    evt_arg = &local_evt;

  // returns NULL with error code if fails
  const int error_code = clEnqueueNDRangeKernel(
    command_queue_, kernel_, 3, nullptr, global, local, events_to_wait.size(),
    events_to_wait.data(), evt_arg);
  if (error_code != CL_SUCCESS) {
    ml_loge("Failed to clEnqueueNDRangeKernel. OpenCL error code: %d : %s",
            error_code, OpenCLErrorCodeToString(error_code));
    return false;
  }
  if (track && local_evt != nullptr) {
    char nm[128] = {0};
    if (clGetKernelInfo(kernel_, CL_KERNEL_FUNCTION_NAME, sizeof(nm) - 1, nm,
                        nullptr) != CL_SUCCESS)
      nm[0] = '\0';
    std::string key(nm);
    if (!next_prof_label_.empty())
      key += next_prof_label_;
    profRecs().push_back({std::move(key), local_evt});
  }
  next_prof_label_.clear();

  // NNTR_XE3_SYNC: coarse-grain-SVM Intel devices do not honor in-order
  // kernel->kernel memory consistency for GPU-resident SVM handoffs. Drain
  // only after dispatches that bound an SVM pointer — the real producer->
  // consumer boundary — instead of after every dispatch. Always consume the
  // flag so it never leaks onto the next dispatch (cl_mem-only dispatches
  // read false and skip).
  const bool touched_svm = Kernel::takeDispatchTouchedSVM();
  if (xe3_needs_sync() && touched_svm)
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

  // Profiling capture: see the by-value overload above.
  cl_event local_evt = nullptr;
  cl_event *evt_arg = event;
  const bool track = profEnabled() && evt_arg == nullptr;
  if (track)
    evt_arg = &local_evt;

  // returns NULL with error code if fails
  const int error_code = clEnqueueNDRangeKernel(
    command_queue_, kernel_, 3, nullptr, global, local, events_to_wait.size(),
    events_to_wait.data(), evt_arg);
  if (error_code != CL_SUCCESS) {
    ml_loge("Failed to clEnqueueNDRangeKernel. OpenCL error code: %d : %s",
            error_code, OpenCLErrorCodeToString(error_code));
    return false;
  }
  if (track && local_evt != nullptr) {
    char nm[128] = {0};
    if (clGetKernelInfo(kernel_, CL_KERNEL_FUNCTION_NAME, sizeof(nm) - 1, nm,
                        nullptr) != CL_SUCCESS)
      nm[0] = '\0';
    std::string key(nm);
    if (!next_prof_label_.empty())
      key += next_prof_label_;
    profRecs().push_back({std::move(key), local_evt});
  }
  next_prof_label_.clear();

  // NNTR_XE3_SYNC: coarse-grain-SVM Intel devices do not honor in-order
  // kernel->kernel memory consistency for GPU-resident SVM handoffs. Drain
  // only after dispatches that bound an SVM pointer — the real producer->
  // consumer boundary — instead of after every dispatch. Always consume the
  // flag so it never leaks onto the next dispatch (cl_mem-only dispatches
  // read false and skip).
  const bool touched_svm = Kernel::takeDispatchTouchedSVM();
  if (xe3_needs_sync() && touched_svm)
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

  // When profiling and the caller did not request its own event, capture a
  // tracked event so dumpProfile() can read true per-kernel GPU time. We own
  // the single reference and release it in dumpProfile(). (Calls that pass
  // their own event — e.g. the act-quant path — are left untracked to avoid
  // event-ownership complexity; they are a negligible slice anyway.)
  cl_event local_evt = nullptr;
  cl_event *evt_arg = event;
  const bool track = profEnabled() && evt_arg == nullptr;
  if (track)
    evt_arg = &local_evt;

  const auto error_code = clEnqueueNDRangeKernel(
    command_queue_, kernel, work_dim, nullptr, global_work_size,
    local_work_size, num_events_in_wait_list, event_wait_list, evt_arg);

  NNTR_THROW_IF(error_code != CL_SUCCESS, std::runtime_error)
    << "clEnqueueNDRangeKernel failed. OpenCL error code: " << error_code
    << ", error: " << OpenCLErrorCodeToString(error_code);

  if (track && local_evt != nullptr) {
    char nm[128] = {0};
    if (clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, sizeof(nm) - 1, nm,
                        nullptr) != CL_SUCCESS)
      nm[0] = '\0';
    std::string key(nm);
    if (!next_prof_label_.empty())
      key += next_prof_label_;
    profRecs().push_back({std::move(key), local_evt});
  }
  // SVM coherence drain, same policy as DispatchCommand. Kernels that dispatch
  // through enqueueKernel (not DispatchCommand) need the same flush to keep
  // their coarse-grain-SVM producer->consumer handoffs coherent. Flush only
  // when this dispatch bound an SVM pointer; always consume the flag so it
  // never leaks.
  const bool touched_svm = Kernel::takeDispatchTouchedSVM();
  if (xe3_needs_sync() && touched_svm)
    clFinish(command_queue_);

  // consume the per-call shape label regardless of tracking, so it never
  // leaks onto a subsequent kernel's profile entry.
  next_prof_label_.clear();
}

void CommandQueueManager::dumpProfile(const char *tag) {
  if (!profEnabled())
    return;
  auto &recs = profRecs();
  if (command_queue_)
    clFinish(command_queue_);

  struct Agg {
    double total_ns = 0.0;
    unsigned long count = 0;
  };
  std::unordered_map<std::string, Agg> agg;
  double grand_ns = 0.0;
  // ordered timeline (name, start, end) for inter-kernel GPU-idle attribution
  std::vector<std::string> tl_name;
  std::vector<cl_ulong> tl_start, tl_end;
  for (auto &r : recs) {
    cl_ulong start = 0, end = 0;
    if (r.evt) {
      clGetEventProfilingInfo(r.evt, CL_PROFILING_COMMAND_START, sizeof(start),
                              &start, nullptr);
      clGetEventProfilingInfo(r.evt, CL_PROFILING_COMMAND_END, sizeof(end),
                              &end, nullptr);
      if (end > start) {
        double ns = (double)(end - start);
        agg[r.name].total_ns += ns;
        grand_ns += ns;
        tl_name.push_back(r.name);
        tl_start.push_back(start);
        tl_end.push_back(end);
      }
      agg[r.name].count++;
      clReleaseEvent(r.evt);
    }
  }
  recs.clear();

  // Inter-kernel GPU-idle: gap between kernel i's end and i+1's start = host-
  // bound dispatch overhead (the GPU waiting for the host to enqueue/prep the
  // next kernel). Attribute each gap to the "A -> B" transition (base names,
  // shape label stripped) so we see where the idle concentrates.
  auto base = [](const std::string &s) {
    auto p = s.find(':');
    return p == std::string::npos ? s : s.substr(0, p);
  };
  std::unordered_map<std::string, Agg> idle;
  // Per-transition gap list for distribution stats (first/median/max). An
  // aggregate "avg x count" can hide a one-time cost diluted across N
  // instances -- e.g. a
  // one-time first-call cost diluted across N later instances.
  std::unordered_map<std::string, std::vector<double>> idle_gaps;
  double total_idle_ns = 0.0;
  for (size_t i = 1; i < tl_start.size(); ++i) {
    if (tl_start[i] > tl_end[i - 1]) {
      double g = (double)(tl_start[i] - tl_end[i - 1]);
      std::string key = base(tl_name[i - 1]) + " -> " + base(tl_name[i]);
      idle[key].total_ns += g;
      idle[key].count++;
      idle_gaps[key].push_back(g);
      total_idle_ns += g;
    }
  }

  std::vector<std::pair<std::string, Agg>> sorted(agg.begin(), agg.end());
  std::sort(sorted.begin(), sorted.end(), [](const auto &a, const auto &b) {
    return a.second.total_ns > b.second.total_ns;
  });

  printf("\n==== GPU kernel profile [%s] : true on-device time ====\n",
         tag ? tag : "");
  printf("  %-34s %10s %8s %10s %7s\n", "kernel", "total_ms", "calls", "avg_us",
         "%%");
  for (auto &kv : sorted) {
    double tot_ms = kv.second.total_ns / 1e6;
    double avg_us = kv.second.count
                      ? (kv.second.total_ns / 1e3) / (double)kv.second.count
                      : 0.0;
    double pct = grand_ns > 0.0 ? 100.0 * kv.second.total_ns / grand_ns : 0.0;
    printf("  %-34s %10.2f %8lu %10.2f %6.1f%%\n", kv.first.c_str(), tot_ms,
           kv.second.count, avg_us, pct);
  }
  printf("  %-34s %10.2f\n", "TOTAL (sum of kernel GPU time)", grand_ns / 1e6);

  // host-bound inter-kernel idle (GPU waiting for host between dispatches)
  std::vector<std::pair<std::string, Agg>> idle_sorted(idle.begin(),
                                                       idle.end());
  std::sort(idle_sorted.begin(), idle_sorted.end(),
            [](const auto &a, const auto &b) {
              return a.second.total_ns > b.second.total_ns;
            });
  printf("\n  --- inter-kernel GPU-idle (host-bound dispatch overhead) ---\n");
  printf("  %-44s %10s %8s %9s %9s %9s %9s\n", "transition (A -> B)", "idle_ms",
         "count", "avg_us", "first_us", "p50_us", "max_us");
  size_t shown = 0;
  for (auto &kv : idle_sorted) {
    if (shown++ >= 15)
      break;
    double ms = kv.second.total_ns / 1e6;
    double avg_us = kv.second.count
                      ? (kv.second.total_ns / 1e3) / (double)kv.second.count
                      : 0.0;
    double pct =
      total_idle_ns > 0.0 ? 100.0 * kv.second.total_ns / total_idle_ns : 0.0;
    // distribution: first occurrence vs median vs max -- uniform per-layer
    // cost has first~p50~max; a one-time cost has max>>p50.
    auto &gaps = idle_gaps[kv.first];
    double first_us = gaps.empty() ? 0.0 : gaps.front() / 1e3;
    double max_us = 0.0;
    for (double g : gaps)
      max_us = std::max(max_us, g / 1e3);
    std::vector<double> tmp(gaps);
    std::nth_element(tmp.begin(), tmp.begin() + tmp.size() / 2, tmp.end());
    double p50_us = tmp.empty() ? 0.0 : tmp[tmp.size() / 2] / 1e3;
    printf("  %-44s %10.2f %8lu %9.1f %9.1f %9.1f %9.1f  (%4.1f%%)\n",
           kv.first.c_str(), ms, kv.second.count, avg_us, first_us, p50_us,
           max_us, pct);
  }
  printf("  %-44s %10.2f\n", "TOTAL inter-kernel idle", total_idle_ns / 1e6);
  printf("=========================================================\n\n");
  fflush(stdout);
}

} // namespace nntrainer::opencl
