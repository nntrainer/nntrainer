// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   memory_pool.h
 * @date   10 August 2021
 * @see    https://github.com/nntrainer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Memory Pool Class
 *
 * @todo   Support releaseMemory(token) - this need not release actual memory
 *         until deallocate
 * @todo   Support maximum memory size for the memory pool as an argument
 * @todo   support late memory request without optimization
 */

#ifndef __MEMORY_POOL_H__
#define __MEMORY_POOL_H__

#include <functional>
#include <memory>
#include <vector>

#include <mem_allocator.h>
#include <memory_data.h>
#include <memory_planner.h>
#include <tensor_wrap_specs.h>

namespace nntrainer {

/**
 * @class   MemoryPool
 * @brief   Memory Pool provides a common pool for all the tensor memory
 *
 * Allocation and deallocation are routed through an injected
 * MemAllocator so per-vendor backends (CPU, GPU-SVM, NPU-RPC) plug in
 * without #ifdef branches inside the pool. The default ctor uses the
 * base MemAllocator (page-aligned host memory, zero-initialised); CL
 * and QNN install ClSVMAllocator / QNNRpcManager via their Context.
 */
class MemoryPool {
public:
  /**
   * @brief MemoryPool default constructor (uses base MemAllocator).
   *
   * Kept for backward-compatibility with tests and callers that don't
   * thread an engine through.
   */
  explicit MemoryPool() : MemoryPool(std::make_shared<MemAllocator>()) {}

  /**
   * @brief MemoryPool constructor with an explicit allocator.
   *
   * @param allocator backend allocator (CPU / SVM / RPC). Held by
   *        shared_ptr so the pool keeps the allocator alive even if
   *        the originating Context unwinds early. Must be non-null.
   */
  explicit MemoryPool(std::shared_ptr<MemAllocator> allocator) :
    mem_pool(nullptr),
    pool_size(0),
    min_pool_size(0),
    n_wgrad(0),
    allocator_(std::move(allocator)) {}

  /**
   * @brief MemoryPool destructor
   *
   */
  virtual ~MemoryPool() { deallocate(); }

  /**
   * @brief Request Memory from memory pool
   *
   * @param bytes The size of the memory requested in bytes
   * @param start_time The start of the validity interval of this memory
   * @param end_time The end of the validity interval of this memory
   * @param exec_order execution orders of this memory
   * @param lifespan lifespan of memory
   * @param is_wgrad check if the tensor is weight gradient
   *
   * @return The token to get the pointer for this memory after allocation
   * @note start_time is inclusive, but end_time is exclusive
   * @note The value of the return token starts from 1.
   */
  virtual unsigned int requestMemory(
    size_t bytes, unsigned int start_time, unsigned int end_time,
    std::vector<unsigned int> exec_order = std::vector<unsigned int>(),
    TensorLifespan lifespan = TensorLifespan::MAX_LIFESPAN,
    bool is_wgrad = false);

  /**
   * @brief Plan the layout with memory planner
   *
   * @param planner The memory planner to be used for finalizing the layout
   *
   * @return The efficiency of the memory layer with the given memory planner
   *
   * @details The efficiency of the planner is calculated as the ratio of the
   * theoretical minimum memory requirement divided by the memory requirement
   * given by the memory planner.
   *
   * @details planLayout can be called multiple times as this does not perform
   * any allocation but rather just plans the layout and stores the layout.
   * Subsequent call to this function will overwrite any existing layout.
   */
  double planLayout(const MemoryPlanner &planner);

  /**
   * @brief Do the allocation of memory
   *
   */
  virtual void allocate();

  /**
   * @brief Do the allocation of memory for FSU
   *
   */
  virtual void allocateFSU();

  /**
   * @brief Get the allocated memory
   *
   * @param token The token received from the requestMemory
   *
   * @return The pointer of the memory
   *
   * @details This function will throw if called before allocation.
   */
  virtual std::shared_ptr<MemoryData> getMemory(unsigned int idx);

  /**
   * @brief Free all the allocated memory
   *
   */
  virtual void deallocate();

  /**
   * @brief Get the maximum real memory requirement
   *
   * @return The real memory requirement with this strategy in bytes
   */
  size_t size();

  /**
   * @brief Get the minimum theoretical memory requirement
   *
   * @return The theoretical memory requirement with this strategy in bytes
   */
  size_t minMemoryRequirement();

  /**
   * @brief Clear the memory pool
   *
   */
  virtual void clear();

  /**
   * @brief Is the memory pool allocated
   *
   * @return true if the memory is allocated, else false
   */
  virtual bool isAllocated() const;

  /**
   *  @brief Get memory ptrs vector from memory pool class.
   *
   * @return memory ptrs vector
   */
  std::vector<void *> getMemoryPtrs() { return memory_ptrs; }

  /**
   * @brief Get the memory pool address.
   *
   * @return MemoryPool address.
   */
  void *getMemoryPoolAddress() { return mem_pool; }

  /**
   * @brief set FSU weight path
   *
   * @param path FSU weight file path
   */
  virtual void setFsuWeightPath(std::string path){};

  /**
   * @brief set weight file offset for FSU loading
   *
   * @param offsets weight file offset
   */
  virtual void
  setWeightOffset(std::vector<std::pair<size_t, size_t>> offsets){};

  /**
   * @brief Get the allocator used by this pool.
   *
   * @return shared_ptr<MemAllocator>
   */
  std::shared_ptr<MemAllocator> getAllocator() const { return allocator_; }

protected:
  /**
   * @brief  Get memory offset
   */
  std::vector<size_t> &getMemoryOffset() { return memory_offset; }

  /**
   * @brief  Get file offset
   */
  std::vector<size_t> &getFileOffset() { return file_offset; }

  /**
   * @brief  Get memory size
   */
  std::vector<size_t> &getMemorySize() { return memory_size; }

  /**
   * @brief  Get memory execution order
   */
  std::vector<std::vector<unsigned int>> &getMemoryExecOrder() {
    return memory_exec_order;
  }

private:
  /**
   * @brief Validate the provided layout
   */
  bool validateLayout();

  /**
   * @brief Validate the provided layout does not overflow outside the given
   * size of the memory pool
   */
  bool validateOverflow();

  /**
   * @brief Validate the provided layout so that no two memories to be used at
   * overlap interval has overlapping memories
   */
  bool validateOverlap();

  /**
   * @brief Calculate the minimum memory requirement for the given memory
   * requests
   *
   * @return the minimum memory requirement in bytes
   *
   * @note This will be theoretical minimum memory requirement ensuring that the
   * memory usages at the same time do not overlap with their validity. This
   * does not consider about the fragmentation which comes from the actual
   * memory layout.
   */
  size_t calcMinMemoryRequirement();

  /**
   * @brief Get sorted permuation for the memory requests
   *
   * @return sorted permutation
   *
   * @details Performs sorting based on the memory overlap using memory offset
   * as the start and the memory offset + memory size as the end of the
   * interval.
   */
  std::vector<unsigned int> getSortedPermutation();

  std::vector<size_t> memory_size; /**< various sizes memory requested */
  std::vector<void *> memory_ptrs; /**< various pointers memory requested */

  std::vector<std::pair<unsigned int, unsigned int>>
    memory_validity; /**< validity intervals for each requested memory */
  std::vector<size_t> memory_offset; /**< offsets for the memory requested */
  std::vector<size_t> file_offset;   /**< offsets for the bin file */
  std::vector<std::vector<unsigned int>>
    memory_exec_order; /**< execution order for the requested memory */

  std::vector<bool>
    memory_is_wgrad; /**< index for identification of weight gradient */

  void *mem_pool; /**< single contiguous pool buffer (allocate path); for
                       allocateFSU() this stays null and per-element
                       buffers live in owned_buffers_ */

  size_t pool_size; /**< memory requirement for this pool */

  size_t min_pool_size; /**< minimum theoretical memory requirement */

  size_t n_wgrad;

  std::shared_ptr<MemAllocator>
    allocator_; /**< backend allocator: routes alloc/free through the
                      vendor that owns this pool. Never null. */

  std::vector<void *>
    owned_buffers_; /**< unique buffers we must release in deallocate().
                         For allocate(): one entry (mem_pool). For
                         allocateFSU(): one entry per distinct offset
                         (the multi-tenant entries in memory_ptrs alias
                         these). Tracking unique buffers fixes a
                         pre-existing leak where allocateFSU() never
                         freed its per-element ptrs. */
};

} // namespace nntrainer

#endif /** __MEMORY_POOL_H__ */
