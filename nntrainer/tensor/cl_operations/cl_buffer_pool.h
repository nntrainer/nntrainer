// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   cl_buffer_pool.h
 * @date   24 August 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  MemoryPool that can additionally back a planned tensor with a device
 *         cl_mem buffer, so a tensor the planner classified device-resident is
 *         a plain cl_mem kernel argument rather than a shared-memory pointer.
 */

#ifndef __CL_BUFFER_POOL_H__
#define __CL_BUFFER_POOL_H__

#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <mem_allocator.h>
#include <memory_pool.h>

namespace nntrainer {

/**
 * @class   ClBufferPool
 * @brief   MemoryPool with a device cl_mem plane alongside the shared one.
 *
 * @details The shared-memory plane is the base MemoryPool's, unchanged: every
 * tensor still gets its planned slice of it. On top of that, deviceMemory()
 * hands out one cl_mem per planner offset for the tensors the residency
 * planner classified GPU_CLMEM.
 *
 * Two properties matter and both come from the planner rather than from
 * runtime state:
 *
 *  - ONE handle per planner offset. Tensors the planner placed at the same
 *    offset have disjoint lifetimes, and they bind the same cl_mem — distinct
 *    handles over one region do not share a device cache line-for-line on
 *    every driver, so the reuse the planner intends has to be expressed as
 *    handle reuse.
 *  - Nothing is allocated until a tensor asks. A pool whose tensors all
 *    classify to the shared plane creates no device buffer at all, so the
 *    device plane costs exactly what it is used for.
 */
class ClBufferPool : public MemoryPool {
public:
  /**
   * @brief ClBufferPool constructor.
   * @param allocator backend allocator, forwarded to MemoryPool for the
   *        shared plane.
   */
  explicit ClBufferPool(std::shared_ptr<MemAllocator> allocator) :
    MemoryPool(std::move(allocator)) {}

  /**
   * @brief ClBufferPool destructor.
   */
  ~ClBufferPool() override;

  /**
   * @brief Allocate the shared plane and record the planner's offset map, from
   *        which deviceMemory() sizes the per-offset device buffers.
   */
  void allocate() override;

  /**
   * @brief Release the device buffers, then the shared plane.
   */
  void deallocate() override;

  /**
   * @copydoc MemoryPool::noteDeviceOnlyTokens
   *
   * Remembers the planner's answer so allocate() can leave the shared slice
   * unallocated for the offsets the device plane is about to own. An offset
   * qualifies only when EVERY token the planner put there is device-only:
   * tokens at one offset alias one region, so one host-addressable tensor
   * among them means the region has to exist on the shared plane too.
   */
  void noteDeviceOnlyTokens(const std::vector<unsigned int> &tokens) override;

  /**
   * @copydoc MemoryPool::deviceMemory
   *
   * Creates the buffer for the token's planner offset on first request, sized
   * to the largest tensor the planner placed there and zero-filled once, and
   * returns the same handle for every later token at that offset. Returns
   * nullptr if the device cannot hold it, which keeps the tensor on the shared
   * plane instead of leaving it half-placed.
   */
  void *deviceMemory(unsigned int idx) override;

private:
  /** Guards the members below, all of which deviceMemory() fills in lazily.
   *  Allocation is single-threaded today, but the OpenCL lazy getters next
   *  door (ClBufferManager) are reachable from a worker thread and are locked
   *  for it, and one file in this pair silently disagreeing about that is how
   *  a data race gets written later. The lock is taken once per planner
   *  offset, never on a kernel path. */
  std::mutex device_mtx_;
  /** planner offset of each token, indexed by token - 1 */
  std::vector<size_t> token_offset_;
  /** planner offset -> largest tensor the planner placed there */
  std::unordered_map<size_t, size_t> offset_size_;
  /** planner offset -> the one device cl_mem backing it (held as void* so the
   *  header stays free of the OpenCL types). Normally a SUB-buffer of
   *  device_plane_ below; a private buffer for any offset the sub-buffer
   *  route declined. */
  std::unordered_map<size_t, void *> offset_buffer_;
  /** The device plane: one cl_mem spanning the same bytes as the shared plane,
   *  which the per-offset handles above are windows on. A buffer per offset
   *  instead would discard the planner's reuse -- two tokens with disjoint
   *  lifetimes get offsets whose ranges overlap, and only one region can
   *  express that. Null until the first device tensor asks. */
  void *device_plane_ = nullptr;
  /** Span of device_plane_, 0 while it does not exist. */
  size_t device_plane_bytes_ = 0;
  /** The plane was attempted and refused. Sticky, so a driver that says no
   *  is asked once and every offset then takes a private buffer, rather than
   *  the pool retrying a failing allocation per tensor. */
  bool device_plane_failed_ = false;
  /** offsets whose device buffer allocate() created up front and whose shared
   *  slice it therefore skipped. Only offsets in here answer false to
   *  sharedSliceNeeded(); an offset whose device buffer failed to materialise
   *  is not in here and keeps its shared slice. */
  std::unordered_set<size_t> shared_slice_skipped_;
  /** tokens the residency planner told us it will place on the device plane,
   *  recorded by noteDeviceOnlyTokens() before allocate() runs */
  std::vector<unsigned int> device_only_tokens_;

  /**
   * @copydoc MemoryPool::sharedSliceNeeded
   *
   * False exactly for the offsets in shared_slice_skipped_.
   */
  bool sharedSliceNeeded(size_t offset) const override;

  /**
   * @brief Fill token_offset_ / offset_size_ from the planner's layout.
   * @details Split out of allocate() because the skip decision needs the
   * offset map BEFORE the shared plane is allocated, while the original call
   * site needed it after. Idempotent.
   */
  void recordPlannerLayout();

  /**
   * @brief The one device buffer the per-offset handles are windows on.
   * @param span bytes to cover -- the shared plane's span, so the two planes
   *        describe the same layout
   * @return the base cl_mem, or nullptr when the device refused it (sticky:
   *         asked once, after which every offset takes a private buffer)
   * @note Caller holds device_mtx_.
   */
  void *devicePlaneBaseLocked(size_t span);

  /**
   * @brief Register this pool's skip candidates with the plane canary.
   * @details No-op unless NNTR_CLMEM_PLANE_CANARY is set. Registers, but does
   *          not protect: clPlaneCanaryArm() does that at the first forward.
   * @note Caller holds device_mtx_.
   */
  void registerPlaneCanaryCandidates();

  /**
   * @brief Create the device buffer for one planner offset, or return nullptr.
   * @param offset planner offset
   * @return the cl_mem, or nullptr when the device cannot back it
   * @note Caller holds device_mtx_.
   */
  void *createDeviceBufferLocked(size_t offset);
};

/**
 * @brief Arm the NNTR_CLMEM_PLANE_CANARY page protection.
 *
 * @details Call once at the first forward, after every legitimate host write to
 * the plane (initialisers, weight load) has happened, so that what faults from
 * here on is a host access on the INFERENCE path -- which is the question the
 * canary asks. A no-op when NNTR_CLMEM_PLANE_CANARY is unset, when no pool
 * registered candidates, or on Windows. Diagnostic only.
 */
void clPlaneCanaryArm();

} // namespace nntrainer

#endif /** __CL_BUFFER_POOL_H__ */
