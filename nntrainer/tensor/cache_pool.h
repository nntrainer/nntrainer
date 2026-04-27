// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 Jiho Chu <jiho.chu@samsung.com>
 *
 * @file   cache_pool.h
 * @date   01 July 2022
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jiho Chu <jiho.chu@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Cache pool class inherited from memory pool
 *
 */

#ifndef __CACHE_POOL_H__
#define __CACHE_POOL_H__

#include <list>
#include <mutex>
#include <set>
#include <unordered_set>
#include <vector>

#include <cache_elem.h>
#include <common.h>
#include <memory_pool.h>
#include <swap_device.h>

namespace nntrainer {

/**
 * @class   CachePool
 * @brief   Cache memory with fixed size utilizing swap device
 */
class CachePool : public MemoryPool {
public:
  using CacheElems =
    std::unordered_map<unsigned int,
                       std::unique_ptr<CacheElem>>; /**< cache id, cache elem */
  using CacheElemsIter = CacheElems::iterator;
  using ExecIds = std::set<unsigned int>;
  using ExecIdsIter = ExecIds::iterator;

  /**
   * @brief CachePool default constructor
   *
   * @param name name of the cache pool
   * @param allocator backend allocator forwarded to MemoryPool
   */
  explicit CachePool(
    const std::string &name,
    std::shared_ptr<MemAllocator> allocator = std::make_shared<MemAllocator>());

  /**
   * @brief CachePool constructor with cache path
   *
   */
  explicit CachePool(
    const std::string &path, const std::string &name,
    std::shared_ptr<MemAllocator> allocator = std::make_shared<MemAllocator>());

  /**
   * @brief CachePool constructor with cache path & ExecutionMode
   *
   */
  explicit CachePool(
    const std::string &path, const std::string &name,
    ml::train::ExecutionMode exec_mode = ml::train::ExecutionMode::TRAIN,
    std::shared_ptr<MemAllocator> allocator = std::make_shared<MemAllocator>());

  /**
   * @brief MemoryPool destructor
   *
   */
  virtual ~CachePool();

  /**
   * @brief inactive elements
   *
   * @param order order to inactive
   */
  void inActive(unsigned int order);

  /**
   * @brief Do the allocation of cache
   *
   */
  virtual void allocate() override;

  /**
   * @brief Free all the allocated cache
   *
   */
  virtual void deallocate() override;

  /**
   * @brief Request Memory from memory pool
   * @note start_time is inclusive, but end_time is exclusive
   */
  virtual unsigned int requestMemory(
    size_t bytes, unsigned int start_time, unsigned int end_time,
    std::vector<unsigned int> exec_order = std::vector<unsigned int>(),
    TensorLifespan lifespan = TensorLifespan::MAX_LIFESPAN,
    bool is_wgrad = false) override;
  /**
   * @brief Get the allocated cache
   *
   * @param id The token received from the requestMemory
   *
   * @return The pointer of the cache
   *
   * @details This function will throw if called before allocation.
   */
  virtual std::shared_ptr<MemoryData> getMemory(unsigned int id) override;

  /**
   * @brief Is the cache pool allocated
   *
   * @return true if the memory is allocated, else false
   */
  virtual bool isAllocated() const override;

  /**
   * @brief Flush cache data to device
   *
   * @note it must be called only when epoch ends.
   */
  virtual void flush();

  /**
   * @brief Flush cache data to device except given order
   *
   * @param order except execution order
   */
  virtual void flushExcept(unsigned int order);

  /**
   * @brief Flush cache data to device except given order
   *
   * @param order except execution order
   */
  virtual void flushExcept(std::vector<unsigned int> order);

  /**
   * @brief Clear the memory pool
   *
   */
  virtual void clear() override;

  /**
   * @brief Load cache data by execution order
   *
   * @param order execution order
   */
  virtual void loadExec(unsigned int order);

  /**
   * @brief Load Tensor
   *
   * @param order order of Tensor to load
   */
  virtual void loadTensor(unsigned int order);

  /**
   * @brief Load cache data by execution order
   *
   * @param order execution order
   */
  virtual bool loadExecOnce(unsigned int order, ExecIdsIter &iter);

  /**
   * @brief Unload cache data by execution order
   *
   * @param order execution order
   */
  virtual void unloadExec(unsigned int order);

  /**
   * @brief Unload Tensor
   *
   * @param order order of Tensor to unload
   */
  virtual void unloadTensor(unsigned int order);

  /**
   * @brief Load active cache data
   */
  virtual void loadActives();

  /**
   * @brief Unload active cache data
   */
  virtual void unloadActives();

  /**
   * @brief Get name
   *
   * @return cache pool name
   */
  virtual std::string getName() { return name; }

  /**
   * @brief Get ExecutionMode
   *
   * @return ml::train::ExecutionMode
   */
  ml::train::ExecutionMode getExecMode() const { return execution_mode_; }

  /**
   * @brief set FSU weight path
   *
   * @param path FSU weight file path
   */
  void setFsuWeightPath(std::string path) override;

  /**
   * @brief set weight file offset for FSU loading
   *
   * @param offsets weight file offset
   */
  void
  setWeightOffset(std::vector<std::pair<size_t, size_t>> offsets) override {
    swap_device->setWeightOffset(offsets);
  }

  /**
   * @brief get Tensor ID set in order
   *
   * @param order Execution order
   * @return Tensor id set
   */
  std::set<unsigned int> getExecIDs(unsigned int order) {
    return exec_ids[order];
  }

  /**
   * @brief get Active Cache Elem lists
   *
   * @return Active Cache Elem list
   */
  std::unordered_set<unsigned int> getActiveElems() { return actives; }

  /**
   * @brief get Cache Elem with id
   * @param id Tensor ID
   * @return Cache Elem
   */
  CacheElem &getCacheElem(unsigned int id) { return *elems[id]; }

  /**
   * @brief check Cache Elem with id is loaded (Active)
   * @param id Tensor ID
   * @return true if it is loaded
   */
  bool isLoaded(unsigned int id) { return elems[id]->isActive(); }

protected:
  /**
   * @brief validate cache element
   *
   * @param cache element id
   */
  virtual void validate(unsigned int id);

  /**
   * @brief invalidate cache element
   *
   * @param cache element id
   */
  virtual void invalidate(unsigned int id);

  /**
   * @brief Get cache policies
   *
   * @return Cache polices
   */
  std::vector<CachePolicy> &getCachePolicy() { return policies; }

private:
  void eraseActiveIf(const std::function<bool(unsigned int id)> &pred);

  std::string name;                         /**< pool name */
  ml::train::ExecutionMode execution_mode_; /**< execution mode */
  std::shared_ptr<SwapDevice> swap_device;  /**< swap device */
  CacheElems elems;                         /**< cache elements */
  std::unordered_set<unsigned int> actives;
  std::vector<CachePolicy> policies;
  std::unordered_map<unsigned int, ExecIds> exec_ids;

  mutable std::mutex mutex;
};

} // namespace nntrainer

#endif /** __CACHE_POOL_H__ */
