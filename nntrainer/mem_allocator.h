// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    mem_allocator.h
 * @date    13 Jan 2025
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   This is memory allocator for memory pool
 *
 */
#ifndef __MEM_ALLOCATOR_H__
#define __MEM_ALLOCATOR_H__

#include <cstddef>
#include <memory>
#include <string>

namespace nntrainer {

/**
 * @brief MemAllocator, Memory allocator class
 *
 * Backend-pluggable allocator for MemoryPool. The default implementation
 * uses std::aligned_alloc (zero-initialized), so MemoryPool no longer
 * embeds calloc/SVM/rpcmem dispatch via macros. Per-vendor Contexts
 * (ClContext, QNNContext) install their own subclass through
 * ContextData::setMemAllocator(). MemoryPool then takes the allocator
 * by shared_ptr at construction and routes allocate/deallocate through
 * it — see ARCHITECTURE.md.
 */
class MemAllocator {
public:
  MemAllocator() = default;
  virtual ~MemAllocator() = default;

  /**
   * @brief Allocate aligned memory.
   * @param[out] ptr       receives the allocated address
   * @param[in]  size      bytes
   * @param[in]  alignment alignment in bytes (must be a power of two);
   *                       caller passes the page size or a smaller value
   *                       depending on the use case
   *
   * The default implementation uses std::aligned_alloc and zero-fills.
   * Subclasses (ClSVMAllocator, QNNRpcManager) override to plumb the
   * vendor allocator instead.
   */
  virtual void alloc(void **ptr, size_t size, size_t alignment);

  /**
   * @brief Free memory previously returned by alloc().
   *
   * Must match the allocator that produced ptr — never mix free() with
   * a vendor allocator's release call.
   */
  virtual void free(void *ptr);

  /**
   * @brief Backend identifier ("cpu" / "gpu-svm" / "qnn-rpc").
   *
   * MemoryPool uses this in error messages. Prefer the capability
   * predicates below for reasoning about pointer ownership — the name
   * is now log-only, not a capability signal.
   */
  virtual std::string getName() { return "cpu"; };

  /**
   * @brief Capability predicates — what KIND of memory alloc() produces,
   *        derived from what the allocator actually does rather than from
   *        its name string. MemoryPool / TensorPool reason about residency
   *        and SVM-ness through these instead of comparing getName(). The
   *        base is the plain host allocator (aligned_alloc): host-addressable
   *        and not device-visible. Vendor subclasses override.
   * @{
   */

  /**
   * @brief True if the CPU can dereference pointers from alloc() directly.
   *        Base host allocator: true. Device-only memory (cudaMalloc): false.
   */
  virtual bool isHostAddressable() const { return true; }

  /**
   * @brief True if an accelerator can read the pointer without an explicit
   *        host->device copy. Base host allocator: false.
   */
  virtual bool isDeviceVisible() const { return false; }

  /**
   * @brief CONTRACT: "this pointer may be handed to an OpenCL kernel". Every
   *        consumer of the flag in the tree is an OpenCL kernel-binding gate,
   *        so this is NOT a generic "unified memory" predicate: a non-OpenCL
   *        backend whose memory happens to be host-addressable must report
   *        false, or a unified build silently routes its tensors into the
   *        OpenCL fast paths. Derived, not stored, for the OpenCL allocators:
   *        an SVM allocation is exactly one that is both host-addressable and
   *        device-visible. Replaces the getName()=="gpu-svm" comparison in
   *        MemoryPool::getMemory(); see
   *        docs/backend_guide/ARCHITECTURE_REFACTOR.md §3.
   */
  virtual bool isSVM() const {
    return isHostAddressable() && isDeviceVisible();
  }

  /**
   * @brief True if the pointer must be registered with the backend (e.g.
   *        rpcmem/ION -> Qnn_MemHandle) before the device can use it.
   *        Base / SVM / UVM: false. QNN rpcmem: true.
   */
  virtual bool needsRegister() const { return false; }
  /** @} */
};
} // namespace nntrainer

#endif
