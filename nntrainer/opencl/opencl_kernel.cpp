// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file    opencl_kernel.cpp
 * @date    06 Feb 2024
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Debadri Samaddar <s.debadri@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   OpenCL wrapper for kernel management
 *
 */

#include "opencl_kernel.h"

#include "opencl_loader.h"

#include <nntrainer_log.h>

namespace nntrainer::opencl {

// Set whenever a shared-virtual-memory pointer is bound as a kernel argument,
// read and cleared by CommandQueueManager when the kernel is enqueued, so the
// coherence drain runs only after a dispatch that actually touched shared
// memory. The dispatch path is single threaded -- a kernel is fully bound and
// then enqueued before the next one is bound -- so a file-scope flag is enough.
static bool s_bind_touched_svm = false;

/**
 * @brief Create a Kernel From Program object
 *
 * @param program
 * @param function_name the kernel string name
 * @return true if successful or false otherwise
 */
bool Kernel::CreateKernelFromProgram(Program program,
                                     const std::string &function_name) {
  int error_code;
  // get the OpenCL program
  cl_program prgm = program.GetProgram();

  // returns NULL with error code if fails
  kernel_ = clCreateKernel(prgm, function_name.c_str(), &error_code);
  if (!kernel_ || error_code != CL_SUCCESS) {
    kernel_ = nullptr;
    ml_loge("Failed to create %s. OpenCL error code: %d : %s",
            function_name.c_str(), error_code,
            OpenCLErrorCodeToString(error_code));
    return false;
  }
  // increments the program reference count.
  clRetainProgram(prgm);

  return true;
}

/**
 * @brief Set the Kernel Arguments
 *
 * @param arg_index index of the argument
 * @param arg_value value of the argument
 * @param size size of the argument
 * @return true if successful or false otherwise
 */
bool Kernel::SetKernelArguments(cl_uint arg_index, const void *arg_value,
                                size_t size) {
  int error_code;
  // returns NULL with error code if fails
  error_code = clSetKernelArg(kernel_, arg_index, size, arg_value);
  if (error_code != CL_SUCCESS) {
    ml_loge("Failed to set argument: %u = %p. OpenCL error code: %d : %s",
            arg_index, arg_value, error_code,
            OpenCLErrorCodeToString(error_code));
    return false;
  }

  return true;
}

/**
 * @brief Set the Kernel Arguments
 *
 * @param arg_index index of the argument
 * @param arg_value value of the argument
 * @param size size of the argument
 * @return true if successful or false otherwise
 */
bool Kernel::SetKernelSVMArguments(cl_uint arg_index, const void *arg_value) {
  // This dispatch reads or writes shared virtual memory, so its
  // producer-to-consumer handoff needs an explicit flush on a device without
  // fine-grain coherence. The flag accumulates until the next dispatch
  // consumes it, so the order the arguments are bound in does not matter.
  s_bind_touched_svm = true;
  int error_code;
  // returns NULL with error code if fails
  error_code = clSetKernelArgSVMPointer(kernel_, arg_index, arg_value);
  if (error_code != CL_SUCCESS) {
    ml_loge("Failed to set argument. OpenCL error code: %d : %s", error_code,
            OpenCLErrorCodeToString(error_code));
    return false;
  }

  return true;
}

/**
 * @brief Get the Kernel object
 *
 * @return const cl_kernel
 */
const cl_kernel Kernel::GetKernel() { return kernel_; }

bool Kernel::takeDispatchTouchedSVM() {
  const bool touched = s_bind_touched_svm;
  s_bind_touched_svm = false;
  return touched;
}

} // namespace nntrainer::opencl
