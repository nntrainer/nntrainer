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
#include <unordered_map>
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
  /** planner offset of each token, indexed by token - 1 */
  std::vector<size_t> token_offset_;
  /** planner offset -> largest tensor the planner placed there */
  std::unordered_map<size_t, size_t> offset_size_;
  /** planner offset -> the one device cl_mem backing it (held as void* so the
   *  header stays free of the OpenCL types) */
  std::unordered_map<size_t, void *> offset_buffer_;
};

} // namespace nntrainer

#endif /** __CL_BUFFER_POOL_H__ */
