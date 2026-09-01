// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_kernel.h
 * @date    22 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   CUDA wrapper for kernel (CUfunction) management. Peer of
 *          nntrainer::opencl::Kernel. Driver-API launches take an array of arg
 *          POINTERS, so SetKernelArguments copies each arg's bytes into
 * internal storage and getKernelParams() builds the void* array at launch time.
 */

#ifndef __CUDA_KERNEL_H__
#define __CUDA_KERNEL_H__

#include <cstddef>
#include <string>
#include <vector>

#include <cuda.h>

namespace nntrainer::cuda {

class Module;

/**
 * @class Kernel
 * @brief Wraps a CUfunction resolved from a Module + its bound arguments.
 */
class Kernel {
public:
  /**
   * @brief Resolve a function by name from a loaded module.
   */
  bool CreateKernelFromModule(const Module &module,
                              const std::string &function_name);

  /**
   * @brief Set (copy) the argument at @p arg_index. The bytes are copied into
   *        internal storage so the caller's value need not outlive this call.
   */
  bool SetKernelArguments(unsigned int arg_index, const void *arg_value,
                          size_t size);

  /**
   * @brief Clear all bound arguments (call before re-binding for a new launch).
   */
  void clearArguments() { arg_storage_.clear(); }

  /**
   * @brief Build the void* kernelParams array for cuLaunchKernel. The returned
   *        pointers reference internal storage and are valid until the next
   *        SetKernelArguments/clearArguments on this kernel.
   */
  std::vector<void *> getKernelParams();

  /**
   * @brief Get the underlying CUfunction
   */
  CUfunction GetFunction() const { return function_; }

  /**
   * @brief true if a function is bound
   */
  bool valid() const { return function_ != nullptr; }

private:
  CUfunction function_{nullptr};
  std::vector<std::vector<char>> arg_storage_;
};

} // namespace nntrainer::cuda

#endif // __CUDA_KERNEL_H__
