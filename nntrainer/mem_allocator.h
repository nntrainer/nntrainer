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

class MemoryPool;

/**
 * @brief Residency classes (defined in tensor/memory_data.h); forward-declared
 * so the allocator base does not pull in the tensor headers.
 */
enum class ResidencyClass : unsigned char;

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
   * @brief CONTRACT: "this pointer may be handed to an OpenCL kernel"
   *        (clSetKernelArgSVMPointer paths) -- every consumer in the tree is
   *        an OpenCL kernel-binding gate. NOT a generic "unified memory"
   *        predicate: CUDA UVM is unified yet MUST report false, so that a
   *        build with both OpenCL and CUDA enabled never binds a CUDA-engine
   *        tensor through the OpenCL SVM fast path.
   *        Derived, not stored, for the CL-side allocators: an OpenCL SVM
   *        allocation is exactly one that is both host-addressable and
   *        device-visible. Replaces the getName()=="gpu-svm" comparison in
   *        MemoryPool::getMemory(). Any new non-OpenCL backend MUST override
   *        this to false.
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

  /**
   * @brief True if this allocator can back a device-resident cl_mem pool
   *        (ClBufferPool) — the prerequisite for device cl_mem tensor
   *        residency. Base / CUDA UVM / rpcmem: false; ClSVMAllocator
   *        (OpenCL): true. Distinct from isSVM() because CUDA UVM is also
   *        SVM yet has no OpenCL cl_mem plane. This is the capability behind
   *        makePool's device-pool branch, asked instead of comparing
   *        getName().
   */
  virtual bool supportsDevicePool() const { return false; }

  /**
   * @brief Whether this allocator can back a tensor of the given residency
   *        class. The residency planner (and a future backend registering its
   *        memory plane) asks this instead of hard-coding the allocator
   *        identity. Default maps the classes onto the existing capability
   *        predicates (HOST always; SVM<-isSVM; GPU_CLMEM/IMAGE2D<-
   *        supportsDevicePool; RPCMEM<-needsRegister); a backend overrides to
   *        advertise IMAGE2D / RPCMEM explicitly.
   */
  virtual bool supportsResidency(ResidencyClass cls) const;
  /** @} */

  /**
   * @brief Construct the MemoryPool that backs a TensorPool for this allocator.
   *        The allocator owns the pool-KIND decision (a plain offset-planned
   *        MemoryPool vs a device cl_mem ClBufferPool) instead of TensorPool
   *        branching on getName()=="gpu-svm". The base returns a MemoryPool;
   *        ClSVMAllocator overrides to return a device pool when enabled.
   *
   * @param self shared_ptr to THIS allocator (the pool holds it for alloc/free)
   * @param pool_name identifier forwarded to pools that need one
   * @return the backing pool
   */
  virtual std::shared_ptr<MemoryPool>
  makePool(const std::shared_ptr<MemAllocator> &self,
           const std::string &pool_name = "");
};
} // namespace nntrainer

#endif
