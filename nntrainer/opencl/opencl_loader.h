// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file    opencl_loader.h
 * @date    06 Feb 2024
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Debadri Samaddar <s.debadri@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Load required OpenCL functions
 *
 */

#ifndef __OPENCL_LOADER_H__
#define __OPENCL_LOADER_H__

#include "CL/cl.h"

namespace nntrainer::opencl {

/**
 * @brief Loading OpenCL libraries and required function
 *
 * @return true if successfull or false otherwise
 */
bool LoadOpenCL();

/**
 * @brief Retrieves string representation of OpenCL status code
 *
 * @return OpenCL status code as string
 */
const char *OpenCLErrorCodeToString(const cl_int code);

typedef cl_int(CL_API_CALL *PFN_clGetPlatformIDs)(
  cl_uint /**< num_entries */, cl_platform_id * /**< platforms */,
  cl_uint * /**< num_platforms */);

typedef cl_int(CL_API_CALL *PFN_clGetDeviceIDs)(
  cl_platform_id /**< platform */, cl_device_type /**< device_type */,
  cl_uint /**< num_entries */, cl_device_id * /**< devices */,
  cl_uint * /**< num_devices */);

typedef cl_int(CL_API_CALL *PFN_clGetDeviceInfo)(
  cl_device_id /**< device */, cl_device_info /**< param_name */,
  size_t /**< param_value_size */, void * /**< param_value */,
  size_t * /**< param_value_size_ret */);

typedef cl_int(CL_API_CALL *PFN_clGetImageInfo)(
  cl_mem /**< image */, cl_image_info /**< param_name */,
  size_t /**< param_value_size */, void * /**< param_value */,
  size_t * /**< param_value_size_ret */);

typedef cl_context(CL_API_CALL *PFN_clCreateContext)(
  const cl_context_properties * /**< properties */, cl_uint /**< num_devices */,
  const cl_device_id * /**< devices */,
  void(CL_CALLBACK * /**< pfn_notify */)(const char *, const void *, size_t,
                                         void *),
  void * /**< user_data */, cl_int * /**< errcode_ret */);

typedef cl_command_queue(CL_API_CALL *PFN_clCreateCommandQueue)(
  cl_context /**< context */, cl_device_id /**< device */,
  cl_command_queue_properties /**< properties */, cl_int * /**< errcode_ret */);

typedef cl_mem(CL_API_CALL *PFN_clCreateBuffer)(cl_context /**< context */,
                                                cl_mem_flags /**< flags */,
                                                size_t /**< size */,
                                                void * /**< host_ptr */,
                                                cl_int * /**< errcode_ret */);

typedef cl_mem(CL_API_CALL *PFN_clCreateSubBuffer)(
  cl_mem /**< buffer */, cl_mem_flags /**< flags */,
  cl_buffer_create_type /**< buffer_create_type */,
  const void * /**< buffer_create_info */, cl_int * /**< errcode_ret */);

typedef cl_mem(CL_API_CALL *PFN_clCreateImage)(
  cl_context /**< context */, cl_mem_flags /**< flags */,
  const cl_image_format * /**< image_format */,
  const cl_image_desc * /**< image_desc */, void * /**<  host_ptr */,
  cl_int * /**< errcode_ret */);

typedef cl_int(CL_API_CALL *PFN_clEnqueueWriteBuffer)(
  cl_command_queue /**< command_queue */, cl_mem /**< buffer */,
  cl_bool /**< blocking_write */, size_t /**< offset */, size_t /**< size */,
  const void * /**< ptr */, cl_uint /**< num_events_in_wait_list */,
  const cl_event * /**< event_wait_list */, cl_event * /**< event */);

typedef cl_int(CL_API_CALL *PFN_clEnqueueFillBuffer)(
  cl_command_queue /**< command_queue */, cl_mem /**< buffer */,
  const void * /**< pattern */, size_t /**< pattern_size */,
  size_t /**< offset */, size_t /**< size */,
  cl_uint /**< num_events_in_wait_list */,
  const cl_event * /**< event_wait_list */, cl_event * /**< event */);

typedef cl_int(CL_API_CALL *PFN_clEnqueueReadBuffer)(
  cl_command_queue /**< command_queue */, cl_mem /**< buffer */,
  cl_bool /**< blocking_read */, size_t /**< offset */, size_t /**< size */,
  void * /**< ptr */, cl_uint /**< num_events_in_wait_list */,
  const cl_event * /**< event_wait_list */, cl_event * /**< event */);

typedef void *(CL_API_CALL *PFN_clEnqueueMapBuffer)(
  cl_command_queue /**< command_queue */, cl_mem /**< buffer */,
  cl_bool /**< blocking_map */, cl_map_flags /**< map_flags */,
  size_t /**< offset */, size_t /**< size */,
  cl_uint /**< num_events_in_wait_list */,
  const cl_event * /**< event_wait_list */, cl_event * /**< event */,
  cl_int * /**< errcode_ret */
);

typedef cl_int(CL_API_CALL *PFN_clEnqueueUnmapMemObject)(
  cl_command_queue /**< command_queue */, cl_mem /**< memobj */,
  void * /**< mapped_ptr */, cl_uint /**< num_events_in_wait_list */,
  const cl_event * /**< event_wait_list */, cl_event * /**< event */
);

typedef cl_int(CL_API_CALL *PFN_clEnqueueWriteBufferRect)(
  cl_command_queue /**< command_queue */, cl_mem /**< buffer */,
  cl_bool /**< blocking_write */, const size_t * /**< buffer_offset */,
  const size_t * /**< host_offset */, const size_t * /**< region */,
  size_t /**< buffer_row_pitch */, size_t /**< buffer_slice_pitch */,
  size_t /**< host_row_pitch */, size_t /**< host_slice_pitch */,
  const void * /**< ptr */, cl_uint /**< num_events_in_wait_list */,
  const cl_event * /**< event_wait_list */, cl_event * /**< event */);

typedef cl_int(CL_API_CALL *PFN_clEnqueueReadBufferRect)(
  cl_command_queue /**< command_queue */, cl_mem /**< buffer */,
  cl_bool /**< blocking_read */, const size_t * /**< buffer_offset */,
  const size_t * /**< host_offset */, const size_t * /**< region */,
  size_t /**< buffer_row_pitch */, size_t /**< buffer_slice_pitch */,
  size_t /**< host_row_pitch */, size_t /**< host_slice_pitch */,
  void * /**< ptr */, cl_uint /**< num_events_in_wait_list */,
  const cl_event * /**< event_wait_list */, cl_event * /**< event */);

typedef cl_program(CL_API_CALL *PFN_clCreateProgramWithSource)(
  cl_context /**< context */, cl_uint /**< count */,
  const char ** /**< strings */, const size_t * /**< lengths */,
  cl_int * /**< errcode_ret */);

typedef cl_program(CL_API_CALL *PFN_clCreateProgramWithBinary)(
  cl_context /**< context */, cl_uint /**< num_devices */,
  const cl_device_id * /**< device_list */, const size_t * /**< lengths */,
  const unsigned char ** /**< binaries */, cl_int * /**< binary_status */,
  cl_int * /**< errcode_ret */);

typedef cl_int(CL_API_CALL *PFN_clBuildProgram)(
  cl_program /**< program */, cl_uint /**< num_devices */,
  const cl_device_id * /**< device_list */, const char * /**< options */,
  void(CL_CALLBACK * /**< pfn_notify */)(cl_program /**< program */,
                                         void * /**< user_data */),
  void * /**< user_data */);

typedef cl_int(CL_API_CALL *PFN_clGetProgramInfo)(
  cl_program /**< program */, cl_program_info /**< param_name */,
  size_t /**< param_value_size */, void * /**< param_value */,
  size_t * /**< param_value_size_ret */);

typedef cl_int(CL_API_CALL *PFN_clGetProgramBuildInfo)(
  cl_program /**< program */, cl_device_id /**< device */,
  cl_program_build_info /**< param_name */, size_t /**< param_value_size */,
  void * /**< param_value */, size_t * /**< param_value_size_ret */);

typedef cl_int(CL_API_CALL *PFN_clRetainProgram)(cl_program /**< program */);

typedef cl_kernel(CL_API_CALL *PFN_clCreateKernel)(
  cl_program /**< program */, const char * /**< kernel_name */,
  cl_int * /**< errcode_ret */);

typedef cl_int(CL_API_CALL *PFN_clSetKernelArg)(cl_kernel /**< kernel */,
                                                cl_uint /**< arg_index */,
                                                size_t /**< arg_size */,
                                                const void * /**< arg_value */);

typedef cl_int(CL_API_CALL *PFN_clEnqueueNDRangeKernel)(
  cl_command_queue /**< command_queue */, cl_kernel /**< kernel */,
  cl_uint /**< work_dim */, const size_t * /**< global_work_offset */,
  const size_t * /**< global_work_size */,
  const size_t * /**< local_work_size */,
  cl_uint /**< num_events_in_wait_list */,
  const cl_event * /**< event_wait_list */, cl_event * /**< event */);

typedef cl_int(CL_API_CALL *PFN_clGetEventProfilingInfo)(
  cl_event /**< event */, cl_profiling_info /**< param_name */,
  size_t /**< param_value_size */, void * /**< param_value */,
  size_t * /**< param_value_size_ret */);

typedef cl_int(CL_API_CALL *PFN_clRetainContext)(cl_context /**< context */);

typedef cl_int(CL_API_CALL *PFN_clReleaseContext)(cl_context /**< context */);

typedef cl_int(CL_API_CALL *PFN_clRetainCommandQueue)(
  cl_command_queue /**< command_queue */);

typedef cl_int(CL_API_CALL *PFN_clReleaseCommandQueue)(
  cl_command_queue /**< command_queue */);

typedef cl_int(CL_API_CALL *PFN_clReleaseMemObject)(cl_mem /**< memobj */);

typedef cl_int(CL_API_CALL *PFN_clFlush)(
  cl_command_queue /**< command_queue */);

typedef cl_int(CL_API_CALL *PFN_clFinish)(
  cl_command_queue /**< command_queue */);

typedef void *(CL_API_CALL *PFN_clSVMAlloc)(cl_context /**< context */,
                                            cl_svm_mem_flags /**< flags */,
                                            size_t /**< size */,
                                            cl_uint /**< alignment */);

typedef void(CL_API_CALL *PFN_clSVMFree)(cl_context /**< context */,
                                         void * /**< svm_pointer */);

typedef cl_int(CL_API_CALL *PFN_clSetKernelArgSVMPointer)(
  cl_kernel /**< kernel */, cl_uint /**< arg_index */,
  const void * /**< arg_value */);

typedef cl_int(CL_API_CALL *PFN_clEnqueueSVMMap)(
  cl_command_queue /**< command_queue */, cl_bool /**< blocking_map */,
  cl_map_flags /**< flags */, void * /**< svm_ptr */, size_t /**< size */,
  cl_uint /**< num_events_in_wait_list */,
  const cl_event * /**< event_wait_list */, cl_event * /**< event */);

typedef cl_int(CL_API_CALL *PFN_clEnqueueSVMUnmap)(
  cl_command_queue /**< command_queue */, void * /**< svm_ptr */,
  cl_uint /**< num_events_in_wait_list */,
  const cl_event * /**< event_wait_list */, cl_event * /**< event */);

typedef cl_int(CL_API_CALL *PFN_clWaitForEvents)(cl_uint num_events,
                                                 const cl_event *event_list);

typedef cl_int(CL_API_CALL *PFN_clReleaseEvent)(cl_event /**< event */);

typedef cl_int(CL_API_CALL *PFN_clEnqueueBarrierWithWaitList)(
  cl_command_queue /**< command_queue */,
  cl_uint /**< num_events_in_wait_list */,
  const cl_event * /**< event_wait_list */, cl_event * /**< event */);

extern PFN_clGetPlatformIDs clGetPlatformIDs;
extern PFN_clGetDeviceIDs clGetDeviceIDs;
extern PFN_clGetDeviceInfo clGetDeviceInfo;
extern PFN_clGetImageInfo clGetImageInfo;
extern PFN_clCreateContext clCreateContext;
extern PFN_clCreateCommandQueue clCreateCommandQueue;
extern PFN_clCreateBuffer clCreateBuffer;
extern PFN_clCreateSubBuffer clCreateSubBuffer;
extern PFN_clCreateImage clCreateImage;
extern PFN_clEnqueueWriteBuffer clEnqueueWriteBuffer;
extern PFN_clEnqueueFillBuffer clEnqueueFillBuffer;
extern PFN_clEnqueueReadBuffer clEnqueueReadBuffer;
extern PFN_clEnqueueMapBuffer clEnqueueMapBuffer;
extern PFN_clEnqueueUnmapMemObject clEnqueueUnmapMemObject;
extern PFN_clEnqueueWriteBufferRect clEnqueueWriteBufferRect;
extern PFN_clEnqueueReadBufferRect clEnqueueReadBufferRect;
extern PFN_clCreateProgramWithSource clCreateProgramWithSource;
extern PFN_clCreateProgramWithBinary clCreateProgramWithBinary;
extern PFN_clBuildProgram clBuildProgram;
extern PFN_clGetProgramInfo clGetProgramInfo;
extern PFN_clGetProgramBuildInfo clGetProgramBuildInfo;
extern PFN_clRetainProgram clRetainProgram;
extern PFN_clCreateKernel clCreateKernel;
extern PFN_clSetKernelArg clSetKernelArg;
extern PFN_clEnqueueNDRangeKernel clEnqueueNDRangeKernel;
extern PFN_clGetEventProfilingInfo clGetEventProfilingInfo;
extern PFN_clRetainContext clRetainContext;
extern PFN_clReleaseContext clReleaseContext;
extern PFN_clRetainCommandQueue clRetainCommandQueue;
extern PFN_clReleaseCommandQueue clReleaseCommandQueue;
/**
 * @brief Accounting wrapper (see clMemAcct* below). Same signature and
 * semantics as the driver entry point it forwards to; it exists so that the GPU
 * memory ledger sees a release without every call site having to name a second
 * symbol.
 */
cl_int clReleaseMemObjectT(cl_mem memobj);
extern PFN_clFlush clFlush;
extern PFN_clFinish clFinish;
extern PFN_clSVMAlloc clSVMAlloc;
/**
 * @brief Accounting wrapper, as clReleaseMemObject above.
 */
void clSVMFreeT(cl_context context, void *svm_pointer);
extern PFN_clEnqueueSVMMap clEnqueueSVMMap;
extern PFN_clEnqueueSVMUnmap clEnqueueSVMUnmap;
extern PFN_clSetKernelArgSVMPointer clSetKernelArgSVMPointer;
extern PFN_clWaitForEvents clWaitForEvents;
extern PFN_clReleaseEvent clReleaseEvent;
extern PFN_clEnqueueBarrierWithWaitList clEnqueueBarrierWithWaitList;

/**
 * @brief Monotonic epoch of OpenCL memory-object handle allocation.
 *
 * Any cache that remembers a cl_mem / SVM pointer VALUE has to survive the one
 * way such a value can lie: a handle is released and the driver hands the same
 * address back for a different object. Everything that creates a memory object
 * in this tree bumps this counter, so a cache that stores the epoch alongside
 * the value can tell "same object" from "same address, new object" without
 * tracking releases. Steady-state decode creates nothing, so the epoch is
 * constant there and every such cache stays hot.
 */
unsigned long long clHandleEpoch();

/**
 * @brief Invalidate every handle-value cache. Called by the memory-object
 *        creation wrappers below.
 */
void clBumpHandleEpoch();

/**
 * @brief Handle-epoch-tracking wrappers around the creation entry points. They
 * are the ONLY way this tree should create memory objects, so that the epoch
 * above is a complete record.
 */
cl_mem clCreateBufferT(cl_context context, cl_mem_flags flags, size_t size,
                       void *host_ptr, cl_int *errcode_ret);
cl_mem clCreateSubBufferT(cl_mem buffer, cl_mem_flags flags,
                          cl_buffer_create_type type, const void *info,
                          cl_int *errcode_ret);
cl_mem clCreateImageT(cl_context context, cl_mem_flags flags,
                      const cl_image_format *format, const cl_image_desc *desc,
                      void *host_ptr, cl_int *errcode_ret);
void *clSVMAllocT(cl_context context, cl_svm_mem_flags flags, size_t size,
                  unsigned int alignment);

/**
 * @brief GPU memory ledger -- what the driver was asked for, by whom.
 *
 * @details On Adreno every clSVMAlloc / clCreateBuffer is a kgsl allocation
 * and is charged to /sys/class/kgsl/kgsl/page_alloc at the ioctl, whether or
 * not a page is ever touched. That counter is the honest GPU footprint and it
 * is also completely anonymous: it says 1 665 MiB and nothing about which of
 * the weight repack, the activation plane, the KV mirrors or a grow-only
 * scratch buffer is holding it. The ledger closes that gap by recording every
 * allocation this tree makes -- the four creation wrappers above are the only
 * way it makes one -- against a tag pushed by the call site, and every release,
 * so the dump is live bytes and not merely cumulative ones.
 *
 * The counting is ON by default -- NNTR_GPU_MEM_ACCT=0 opts out, and when it
 * is off every entry point below is a load of one bool and a return, with no
 * map, mutex or string touched on any allocation path. It has to be on by
 * default because it is the only per-process GPU byte count that exists on
 * this platform: /sys/class/kgsl/kgsl/proc/<pid>/gpumem is Permission denied
 * to the shell uid and to the application uid alike, so an app that must
 * report its own honest footprint can only count from the inside.
 *
 * The stderr dumps stay opt-in: clMemAcctDump() prints only when
 * NNTR_GPU_MEM_ACCT is set to something other than 0.
 *
 * A sub-buffer and an image created from a buffer are VIEWS -- they allocate
 * nothing -- and are recorded with zero bytes so that the count still shows
 * them without the byte column double counting the backing they share.
 */
bool clMemAcctOn();

/**
 * @brief Push/pop the tag new allocations on THIS thread are charged to. Prefer
 * the scope object below; these are for the rare site that cannot use it.
 */
void clMemAcctPush(const char *tag);
void clMemAcctPop();

/**
 * @brief Mark every allocation made in this scope as a VIEW: counted, but
 *        charged ZERO bytes.
 *
 * @details A view is memory that already exists and is already counted
 * somewhere else -- a sub-buffer, an image over a buffer, or a
 * CL_MEM_USE_HOST_PTR buffer over an allocation the ledger has seen. Charging
 * its bytes again makes the ledger disagree with the driver's own counter by
 * exactly the amount the aliasing saves, and since the application's reported
 * Peak Mem is computed FROM this ledger, that overstates the footprint by the
 * saving. Sub-buffers and images-from-buffer set the flag at their call site;
 * this scope is for the case only the caller can recognise.
 */
void clMemAcctPushView();

/**
 * @copydoc clMemAcctPushView
 */
void clMemAcctPopView();

/**
 * @brief RAII view scope. `ClMemAcctViewScope _v;` -- see clMemAcctPushView().
 */
struct ClMemAcctViewScope {
  ClMemAcctViewScope() : on_(clMemAcctOn()) {
    if (on_)
      clMemAcctPushView();
  }
  ~ClMemAcctViewScope() {
    if (on_)
      clMemAcctPopView();
  }
  ClMemAcctViewScope(const ClMemAcctViewScope &) = delete;
  ClMemAcctViewScope &operator=(const ClMemAcctViewScope &) = delete;

private:
  bool on_;
};

/**
 * @brief RAII tag scope. `ClMemAcctScope _t("kv:mirror");`
 */
struct ClMemAcctScope {
  ClMemAcctScope(const char *tag) : on_(clMemAcctOn()) {
    if (on_)
      clMemAcctPush(tag);
  }
  ~ClMemAcctScope() {
    if (on_)
      clMemAcctPop();
  }
  ClMemAcctScope(const ClMemAcctScope &) = delete;
  ClMemAcctScope &operator=(const ClMemAcctScope &) = delete;

private:
  const bool on_;
};

/**
 * @brief Print the ledger, tagged with a phase name.
 *
 * @details Written to stderr rather than the log, for the reason the pool
 * banners give: on Android ml_logi goes to logcat and never into the run's
 * captured output, and a number nobody can see in the run that produced it is
 * a claim rather than a measurement. Called at the phase boundaries a run
 * cares about (load / prefill / decode) and once more from the ledger's own
 * destructor, so a run that ends early still reports.
 */
void clMemAcctDump(const char *phase);

/**
 * @brief Bytes this process currently holds from the GPU driver, and the
 * high-water mark of that.
 *
 * @details Live, not cumulative: a grow-only cache that replaced a smaller
 * buffer counts once. Views (sub-buffers, image-from-buffer) contribute zero,
 * so the number is addable to a host RSS figure without double counting.
 * Zero when the ledger is off. Closes to 98.7 % of the kgsl page_alloc delta
 * as measured from outside the process; the residue is the driver's own
 * programs, queues and ring buffers, which no in-process ledger can see.
 */
size_t clMemAcctLiveBytes();
size_t clMemAcctPeakBytes();
} // namespace nntrainer::opencl

#endif // __OPENCL_LOADER_H__
