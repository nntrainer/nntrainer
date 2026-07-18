// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cl_svm_allocator.cpp
 * @date    27 Apr 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Implementation of OpenCL SVM-backed MemAllocator subclass.
 */

#include <cl_buffer_pool.h>
#include <cl_svm_allocator.h>
#include <cstring>
#include <env_compat.h>
#include <memory_pool.h>
#include <mutex>
#include <opencl_context_manager.h>
#include <unordered_set>

namespace nntrainer {

namespace {
// Inline static state — no per-instance state needed since
// ContextManager is itself a global singleton. Keeping this hidden
// in a .cpp namespace keeps the header lean.
std::mutex host_owned_mtx;
std::unordered_set<void *> host_owned;
} // namespace

ClSVMAllocator::ClSVMAllocator(opencl::ContextManager &ctx) : ctx_(ctx) {}

std::shared_ptr<MemoryPool>
ClSVMAllocator::makePool(const std::shared_ptr<MemAllocator> &self,
                         const std::string &pool_name) {
  // NNTR_GPU_CLMEM_POOL: back the activation plane with a device cl_mem pool
  // (ClBufferPool) so activations stay GPU-resident as plain cl_mem; default
  // OFF => the SVM-backed MemoryPool. The pool-KIND decision is owned by the
  // allocator (this override) instead of a getName() string test in
  // TensorPool.
  (void)pool_name;
  if (nntr_env_on("NNTR_GPU_CLMEM_POOL"))
    return std::make_shared<ClBufferPool>(self);
  return std::make_shared<MemoryPool>(self);
}

void ClSVMAllocator::track_host_owned(void *ptr) {
  std::lock_guard<std::mutex> lk(host_owned_mtx);
  host_owned.insert(ptr);
}

bool ClSVMAllocator::consume_host_owned(void *ptr) {
  std::lock_guard<std::mutex> lk(host_owned_mtx);
  return host_owned.erase(ptr) > 0;
}

void ClSVMAllocator::alloc(void **ptr, size_t size, size_t alignment) {
  void *svm = ctx_.createSVMRegion(size);
  if (svm != nullptr) {
    // Honor MemAllocator's documented calloc semantics on the SVM path too
    // (the host path memsets; this one silently didn't). clSVMAlloc gives NO
    // zero guarantee: one driver may happen to hand kernel-zeroed pages while
    // another recycles non-zero pool pages — any consumer relying on the zero
    // contract then reads per-process garbage. One-time cost at allocation.
    std::memset(svm, 0, size);
    *ptr = svm;
    return;
  }
  // SVM not available (driver lacks support, or capacity exhausted) —
  // fall back to a host buffer so the layer still runs (correctness >
  // speed). Caller is unaware; only the matching free() needs to pick
  // the right release path.
  MemAllocator::alloc(ptr, size, alignment);
  track_host_owned(*ptr);
}

void ClSVMAllocator::free(void *ptr) {
  if (ptr == nullptr)
    return;
  if (consume_host_owned(ptr)) {
    MemAllocator::free(ptr);
    return;
  }
  ctx_.releaseSVMRegion(ptr);
}

} // namespace nntrainer
