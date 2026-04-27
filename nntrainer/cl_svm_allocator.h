// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cl_svm_allocator.h
 * @date    27 Apr 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   MemAllocator subclass that routes allocate/free through the
 *          OpenCL SVM API, so MemoryPool buffers used by the GPU backend
 *          are device-visible without a separate copy step.
 */

#ifndef __CL_SVM_ALLOCATOR_H__
#define __CL_SVM_ALLOCATOR_H__

#include <mem_allocator.h>

namespace nntrainer {

namespace opencl {
class ContextManager;
}

/**
 * @brief OpenCL Shared Virtual Memory allocator.
 *
 * Wraps clSVMAlloc / clSVMFree so MemoryPool's
 * allocate/deallocate path can stay backend-agnostic. ClContext
 * installs an instance of this in its ContextData at engine init,
 * replacing the default CPU MemAllocator.
 *
 * Falls back to the base CPU implementation if the SVM allocation
 * returns nullptr (older drivers, unsupported devices) so a missing
 * SVM capability degrades to host memory rather than failing
 * outright. The fallback path uses MemAllocator::alloc/free, which
 * means the matching free() must NOT call clSVMFree on a host-
 * allocated pointer — we track ownership in a small set.
 */
class ClSVMAllocator : public MemAllocator {
public:
  /**
   * @brief Construct with an OpenCL context manager reference.
   *
   * @param ctx the global opencl::ContextManager that owns the
   *            cl_context handle clSVMAlloc needs. Held by reference,
   *            since ContextManager outlives every Tensor / pool.
   */
  explicit ClSVMAllocator(opencl::ContextManager &ctx);

  /**
   * @copydoc MemAllocator::alloc
   *
   * Tries clSVMAlloc first. If that returns nullptr (not just on
   * size==0; some drivers can fail mid-run on capacity), falls back
   * to MemAllocator::alloc and records the pointer as host-owned so
   * free() can pick the right path.
   */
  void alloc(void **ptr, size_t size, size_t alignment) override;

  /**
   * @copydoc MemAllocator::free
   *
   * Picks clSVMFree if the pointer was returned by clSVMAlloc,
   * std::free if it came from the host fallback path. Mismatching
   * the two would be undefined behaviour, so the host-owned set is
   * checked atomically.
   */
  void free(void *ptr) override;

  std::string getName() override { return "gpu-svm"; }

private:
  opencl::ContextManager &ctx_;
  // Host-owned set: pointers that came from MemAllocator::alloc
  // (fallback path) rather than clSVMAlloc. free() routes accordingly.
  // Inline header to keep the dependency graph simple — std::set is
  // already pulled in transitively. Mutex protects concurrent
  // alloc/free across pool threads.
  void track_host_owned(void *ptr);
  bool consume_host_owned(void *ptr);
};

} // namespace nntrainer

#endif // __CL_SVM_ALLOCATOR_H__
