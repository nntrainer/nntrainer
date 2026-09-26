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

#include <cstdint>
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

  /// Kernel-argument value cache. Decode re-binds ~6.6 arguments per dispatch
  /// and ~9 700 per token, almost all of them the same handle or the same
  /// scalar as the token before; each one is a host call the GPU waits behind
  /// (P2 measured host issue cost converting ~1:1 into wall time). The cache
  /// remembers the bytes last passed to clSetKernelArg for this kernel object
  /// and skips the call when they are unchanged.
  ///
  /// Soundness rests on two things. (1) The cl_kernel object holds argument
  /// state, so "unchanged bytes" really means "already bound" -- and a
  /// cl_kernel here is per (program, name, build options) and dispatched from
  /// the single-threaded forward path. (2) A remembered cl_mem/SVM VALUE can
  /// only lie if the handle was released and reissued for a different object,
  /// which is exactly what opencl::clHandleEpoch() tracks: the epoch is stored
  /// with the cache and any creation anywhere drops it.
  static constexpr int kArgCacheSlots = 20;
  static constexpr int kArgCacheBytes = 16;
  unsigned char arg_bytes_[kArgCacheSlots][kArgCacheBytes] = {};
  unsigned char arg_size_[kArgCacheSlots] = {}; ///< bound size, 0 = unset
  unsigned char arg_kind_[kArgCacheSlots] = {}; ///< 0 unset, 1 plain, 2 SVM
  unsigned long long arg_epoch_ = 0;

  /**
   * @brief Look the argument up in the cache; on a miss, record it.
   * @return true when the argument is already bound and the driver call can be
   *         skipped
   */
  bool argCacheHit(cl_uint arg_index, const void *arg_value, size_t size,
                   unsigned char kind);

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
};
} // namespace nntrainer::opencl
#endif // __OPENCL_KERNEL_H__
