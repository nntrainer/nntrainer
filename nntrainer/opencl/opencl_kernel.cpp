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

// Coherence hint for the Xe3 SVM drain. Set true whenever an SVM pointer arg is
// bound, cleared at the start of each new binding cycle (arg_index == 0). Read
// by CommandQueueManager at dispatch time to drain only after SVM-touching
// dispatches. The GPU dispatch path is single-threaded, so a file-scope flag is
// race-free (one kernel is fully bound then dispatched before the next).
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
  // This dispatch binds a coarse-grain SVM pointer -> its producer/consumer
  // handoff needs an explicit flush on Xe3. The flag accumulates until the next
  // dispatch consumes it (takeDispatchTouchedSVM), so binding order is
  // irrelevant.
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
  const bool v = s_bind_touched_svm;
  s_bind_touched_svm = false;
  return v;
}

} // namespace nntrainer::opencl
