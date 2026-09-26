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
   * @brief Read and clear whether a shared-virtual-memory pointer was bound
   *        since the previous dispatch.
   *
   * The coherence drain in CommandQueueManager calls this when it enqueues a
   * kernel, so that it flushes the queue only after a dispatch that actually
   * touched shared memory -- the real producer-to-consumer boundary -- rather
   * than after every dispatch. The flag is set by SetKernelSVMArguments and
   * cleared here on read, so it describes exactly the dispatch being enqueued
   * whatever order the arguments were bound in. The dispatch path is single
   * threaded, so the process-wide flag is not a race.
   *
   * @return true when the dispatch being enqueued bound shared memory
   */
  static bool takeDispatchTouchedSVM();

  /**
   * @brief Declare what the NEXT dispatch writes on the device activation
   *        plane, for the producer-to-consumer handoff log below.
   *
   * A caller that knows its kernel writes exactly one device buffer (and
   * nothing else another op could be reading) calls this immediately before
   * the dispatch. A dispatch for which nothing is declared is logged as
   * UNKNOWN, and an UNKNOWN dispatch invalidates every outstanding handoff --
   * so a site that is not annotated costs a re-computation, never a wrong
   * answer.
   *
   * @param written device buffer handle (cl_mem) the next dispatch writes
   */
  static void noteDispatchWrites(void *written);

  /**
   * @brief Declare that the NEXT dispatch writes nothing that a handoff can
   *        be tracking (its outputs are private scratch).
   */
  static void noteDispatchWritesNothing();

  /**
   * @brief Append the pending declaration to the log. Called once per
   *        enqueued NDRange by CommandQueueManager, whatever the outcome of
   *        the declaration, so the log has exactly one entry per dispatch.
   */
  static void commitDispatch();

  /**
   * @brief Sequence number of the last logged dispatch.
   */
  static unsigned long long dispatchSeq();

  /**
   * @brief Whether the device buffer @a handle still holds what it held at
   *        @a since_seq.
   *
   * True only when EVERY dispatch logged after @a since_seq declared what it
   * wrote and none of them wrote @a handle. An unannotated dispatch, or a
   * window longer than the log, answers false.
   *
   * @param handle device buffer handle (cl_mem)
   * @param since_seq dispatchSeq() sampled when the content was established
   * @return true when nothing can have overwritten the buffer since
   */
  static bool bufferUnchangedSince(void *handle, unsigned long long since_seq);
};
} // namespace nntrainer::opencl
#endif // __OPENCL_KERNEL_H__
