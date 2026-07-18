// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file    opencl_kernel.h
 * @date    06 Feb 2024
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Debadri Samaddar <s.debadri@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   OpenCL wrapper for kernel management
 *
 */

#ifndef __OPENCL_KERNEL_H__
#define __OPENCL_KERNEL_H__

#include <string>

#include "CL/cl.h"
#include "opencl_program.h"

namespace nntrainer::opencl {

/**
 * @class Kernel contains wrappers for managing OpenCL kernels
 * @brief OpenCL kernel wrapper
 *
 */
class Kernel {
  cl_kernel kernel_{nullptr};

public:
  /**
   * @brief Create a Kernel From Program object
   *
   * @param program
   * @param function_name the kernel string name
   * @return true if successful or false otherwise
   */
  bool CreateKernelFromProgram(Program program,
                               const std::string &function_name);

  /**
   * @brief Set the Kernel Arguments
   *
   * @param arg_index index of the argument
   * @param arg_value value of the argument
   * @param size size of the argument
   * @return true if successful or false otherwise
   */
  bool SetKernelArguments(cl_uint arg_index, const void *arg_value,
                          size_t size);

  /**
   * @brief Set the Kernel Arguments
   *
   * @param arg_index index of the argument
   * @param arg_value value of the argument
   * @return true if successful or false otherwise
   */
  bool SetKernelSVMArguments(cl_uint arg_index, const void *arg_value);

  /**
   * @brief Get the Kernel object
   *
   * @return const cl_kernel
   */
  const cl_kernel GetKernel();

  /**
   * @brief Read-and-clear whether an SVM pointer was bound since the previous
   *        dispatch.
   *
   * The Xe3 coherence drain (CommandQueueManager) calls this at dispatch time
   * to flush the queue ONLY after a dispatch that actually touches coarse-grain
   * SVM — the real producer->consumer boundary — instead of after every
   * dispatch. The flag is set by SetKernelSVMArguments and cleared here on
   * read, so it reflects exactly the dispatch being enqueued regardless of
   * arg-binding order. The GPU dispatch path is single-threaded, so the
   * process-wide flag is race-free (one kernel is fully bound then dispatched
   * before the next).
   */
  static bool takeDispatchTouchedSVM();
};
} // namespace nntrainer::opencl
#endif // __OPENCL_KERNEL_H__
