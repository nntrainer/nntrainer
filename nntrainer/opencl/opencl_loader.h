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
extern PFN_clReleaseMemObject clReleaseMemObject;
extern PFN_clFlush clFlush;
extern PFN_clFinish clFinish;
extern PFN_clSVMAlloc clSVMAlloc;
extern PFN_clSVMFree clSVMFree;
extern PFN_clEnqueueSVMMap clEnqueueSVMMap;
extern PFN_clEnqueueSVMUnmap clEnqueueSVMUnmap;
extern PFN_clSetKernelArgSVMPointer clSetKernelArgSVMPointer;
extern PFN_clWaitForEvents clWaitForEvents;
extern PFN_clReleaseEvent clReleaseEvent;
extern PFN_clEnqueueBarrierWithWaitList clEnqueueBarrierWithWaitList;
} // namespace nntrainer::opencl

#endif // __OPENCL_LOADER_H__
