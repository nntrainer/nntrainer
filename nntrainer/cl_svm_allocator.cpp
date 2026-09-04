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
#include <cstdlib>
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
ClSVMAllocator::makePool(const std::shared_ptr<MemAllocator> &self) {
  // NNTR_GPU_CLMEM_POOL=0: fall back to the plain SVM-backed MemoryPool, whose
  // deviceMemory() returns null, so the planner demotes every device placement
  // to the shared plane and no tensor is placed on cl_mem at all.
  //
  // The device plane stays ON by default -- this is an opt-OUT and changes no
  // default. What it buys is the A/B: the plane is the largest single variable
  // in a device run, and separating "the kernels are wrong" from "the
  // placement is wrong" otherwise costs a rebuild.
  const char *e = std::getenv("NNTR_GPU_CLMEM_POOL");
  if (e != nullptr && e[0] == '0')
    return std::make_shared<MemoryPool>(self);
  return std::make_shared<ClBufferPool>(self);
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
