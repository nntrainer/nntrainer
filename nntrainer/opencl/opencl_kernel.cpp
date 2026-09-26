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

#include <cstdlib>
#include <cstring>

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
// NNTR_CL_ARG_CACHE=0 turns the argument cache off, so the same binary can be
// its own control arm.
static bool argCacheEnabled() {
  static const bool on = []() {
    const char *e = std::getenv("NNTR_CL_ARG_CACHE");
    return !(e != nullptr && e[0] == '0');
  }();
  return on;
}

bool Kernel::argCacheHit(cl_uint arg_index, const void *arg_value, size_t size,
                         unsigned char kind) {
  if (!argCacheEnabled() || arg_value == nullptr ||
      arg_index >= (cl_uint)kArgCacheSlots || size == 0 ||
      size > (size_t)kArgCacheBytes)
    return false;

  const unsigned long long epoch = clHandleEpoch();
  if (arg_epoch_ != epoch) {
    // Something, somewhere, created a memory object: a remembered handle value
    // may now name a different object. Drop the whole cache rather than reason
    // about which slot could be affected.
    std::memset(arg_size_, 0, sizeof(arg_size_));
    std::memset(arg_kind_, 0, sizeof(arg_kind_));
    arg_epoch_ = epoch;
  }

  if (arg_kind_[arg_index] == kind && arg_size_[arg_index] == size &&
      std::memcmp(arg_bytes_[arg_index], arg_value, size) == 0)
    return true;

  std::memcpy(arg_bytes_[arg_index], arg_value, size);
  arg_size_[arg_index] = (unsigned char)size;
  arg_kind_[arg_index] = kind;
  return false;
}

bool Kernel::SetKernelArguments(cl_uint arg_index, const void *arg_value,
                                size_t size) {
  if (argCacheHit(arg_index, arg_value, size, 1))
    return true;
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
  // The flag above is set on a cache hit too: the dispatch still reads shared
  // memory, so it still needs the coherence flush. Only the driver call is
  // skipped, never the bookkeeping that describes what the dispatch touches.
  if (argCacheHit(arg_index, &arg_value, sizeof(const void *), 2))
    return true;
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
