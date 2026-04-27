// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 Jiho Chu <jiho.chu@samsung.com>
 *
 * @file   cache_pool.cpp
 * @date   01 July 2022
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jiho Chu <jiho.chu@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Cache pool class inherited from memory pool
 *
 */

#include "cache_pool.h"

#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <profiler.h>

namespace nntrainer {

namespace {

/**
 * @brief convert tensor lifespan to cache policy
 *
 * @param lifespand tensor lifespan
 * @return cache policy
 * @note cache policy is defined by tensor's lifetime. If it needs to be
 * maintained its value, ALWAYS_SYNCED or ITERATION_CONSIST is proper. If not,
 * TEMPORAL doesnot keep its value.
 */
inline const CachePolicy
convertTensorLifespanToCachePolicy(const TensorLifespan lifespan) {
  CachePolicy policy;

  switch (lifespan) {
  case TensorLifespan::UNMANAGED:
    policy = CachePolicy::ALWAYS_SYNCED;
    break;
  case TensorLifespan::FORWARD_FUNC_LIFESPAN:
    policy = CachePolicy::TEMPORAL;
    break;
  case TensorLifespan::FORWARD_INFER_LIFESPAN:
    policy = CachePolicy::SYNC_ONCE;
    break;
  case TensorLifespan::CALC_DERIV_LIFESPAN:
    policy = CachePolicy::TEMPORAL;
    break;
  case TensorLifespan::CALC_GRAD_LIFESPAN:
    policy = CachePolicy::TEMPORAL;
    break;
  case TensorLifespan::CALC_AGRAD_LIFESPAN:
    policy = CachePolicy::TEMPORAL;
    break;
  case TensorLifespan::CALC_GRAD_DERIV_LIFESPAN:
    policy = CachePolicy::TEMPORAL;
    break;
  case TensorLifespan::CALC_GRAD_DERIV_AGRAD_LIFESPAN:
    policy = CachePolicy::ITERATION_CONSIST;
    break;
  case TensorLifespan::FORWARD_GRAD_LIFESPAN:
    policy = CachePolicy::ITERATION_CONSIST;
    break;
  case TensorLifespan::FORWARD_GRAD_AGRAD_LIFESPAN:
    policy = CachePolicy::ITERATION_CONSIST;
    break;
  case TensorLifespan::FORWARD_DERIV_LIFESPAN:
    policy = CachePolicy::ALWAYS_SYNCED;
    break;
  case TensorLifespan::ITERATION_LIFESPAN:
    policy = CachePolicy::ITERATION_CONSIST;
    break;
  case TensorLifespan::EPOCH_LIFESPAN:
    policy = CachePolicy::ITERATION_CONSIST;
    break;
  case TensorLifespan::MAX_LIFESPAN:
    policy = CachePolicy::ALWAYS_SYNCED;
    break;
  default:
    policy = CachePolicy::ALWAYS_SYNCED;
    break;
  }

  return policy;
}

std::atomic_int pool_id = 0;

} // namespace

CachePool::CachePool(const std::string &n,
                     std::shared_ptr<MemAllocator> allocator) :
  MemoryPool(std::move(allocator)),
  name(n),
  execution_mode_(ml::train::ExecutionMode::TRAIN),
  swap_device(std::make_shared<SwapDevice>(n + "_" + std::to_string(getpid()) +
                                           "_" + std::to_string(pool_id++))) {}

CachePool::CachePool(const std::string &path, const std::string &n,
                     std::shared_ptr<MemAllocator> allocator) :
  MemoryPool(std::move(allocator)),
  name(n),
  execution_mode_(ml::train::ExecutionMode::TRAIN) {
  if (path.empty())
    swap_device = std::make_shared<SwapDevice>(
      n + "_" + std::to_string(getpid()) + "_" + std::to_string(pool_id++));
  else
    swap_device =
      std::make_shared<SwapDevice>(path, n + "_" + std::to_string(getpid()) +
                                           "_" + std::to_string(pool_id++));
}

CachePool::CachePool(const std::string &path, const std::string &name_,
                     ml::train::ExecutionMode exec_mode_,
                     std::shared_ptr<MemAllocator> allocator) :
  MemoryPool(std::move(allocator)), name(name_), execution_mode_(exec_mode_) {
  if (path.empty())
    swap_device = std::make_shared<SwapDevice>(
      name_ + "_" + std::to_string(getpid()) + "_" + std::to_string(pool_id++));
  else
    swap_device = std::make_shared<SwapDevice>(
      path,
      name_ + "_" + std::to_string(getpid()) + "_" + std::to_string(pool_id++));
}

CachePool::~CachePool() {
  try {
    deallocate();
  } catch (...) {
    ml_loge("Failed deallocate");
  }
}

void CachePool::inActive(unsigned int order) {

  auto exec_id = exec_ids[order];
  std::lock_guard<std::mutex> lock(mutex);
  for (auto &id : exec_id) {
    actives.erase(id);
    elems[id]->inActive();
  }
}

void CachePool::allocate() {
  NNTR_THROW_IF(swap_device->isOperating(), std::runtime_error)
    << "Cache pool is already allocated";

  size_t pool_size = size();

  NNTR_THROW_IF(pool_size == 0, std::runtime_error)
    << "Allocating memory pool with size 0";
  if (execution_mode_ == ml::train::ExecutionMode::INFERENCE)
    MemoryPool::allocateFSU();
  swap_device->start(size(), execution_mode_);
}

void CachePool::deallocate() {
  MemoryPool::deallocate();
  if (!swap_device->isOperating())
    return;

  if (execution_mode_ == ml::train::ExecutionMode::INFERENCE)
    MemoryPool::deallocate();

  for (auto &[id, elem] : elems)
    invalidate(id);

  actives.clear();
  swap_device->finish();
}

void CachePool::validate(unsigned int id) {
  if (!elems[id]->isActive()) {
    elems[id]->swapIn();
    std::lock_guard<std::mutex> lock(mutex);
    actives.insert(id);
  }
}

void CachePool::invalidate(unsigned int id) {
  if (elems[id]->isActive()) {
    elems[id]->swapOut();
    std::lock_guard<std::mutex> lock(mutex);
    actives.erase(id);
  }
}

unsigned int CachePool::requestMemory(size_t bytes, unsigned int start_time,
                                      unsigned int end_time,
                                      std::vector<unsigned int> exec_order,
                                      TensorLifespan lifespan, bool is_wgrad) {
  auto id = MemoryPool::requestMemory(bytes, start_time, end_time, exec_order,
                                      lifespan, is_wgrad);

  const CachePolicy policy = convertTensorLifespanToCachePolicy(lifespan);

  policies.push_back(policy);

  NNTR_THROW_IF(id != policies.size(), std::runtime_error)
    << "Invalid requestMemory call exist";

  return id;
}

std::shared_ptr<MemoryData> CachePool::getMemory(unsigned int id) {
  NNTR_THROW_IF(!swap_device->isOperating(), std::invalid_argument)
    << "Allocate memory before allocation";

  off_t offset = getMemoryOffset().at(id - 1);
  size_t len = getMemorySize().at(id - 1);
  auto exe_order = getMemoryExecOrder().at(id - 1);
  auto policy = getCachePolicy().at(id - 1);

  void *memory_ptr = nullptr;
  if (execution_mode_ == ml::train::ExecutionMode::INFERENCE) {
    memory_ptr = getMemoryPtrs().at(id - 1);
  }

  auto mem_data = std::make_shared<MemoryData>(
    id, std::bind(&CachePool::validate, this, std::placeholders::_1),
    std::bind(&CachePool::invalidate, this, std::placeholders::_1), memory_ptr);

  elems.emplace(id, std::make_unique<CacheElem>(swap_device, id, offset, len,
                                                mem_data, policy, memory_ptr));

  std::string ords;

  if (execution_mode_ == ml::train::ExecutionMode::INFERENCE) {
    auto &o = exe_order[0];
    exec_ids[o].insert(id);
    ords.append(std::to_string(o));
  } else {
    for (auto &o : exe_order) {
      exec_ids[o].insert(id);
      ords.append(std::to_string(o));
    }
  }
  ml_logd("[%d] exe_order(%s), offset: %llu, len: %zu", id, ords.c_str(),
          (long long unsigned int)offset, len);

  return mem_data;
}

void CachePool::flush() {
  for (auto &id : actives) {
    elems[id]->swapOut(CacheElem::LAST_ACCESS);
  }

  for (auto &[id, elem] : elems)
    elem->reset();

  actives.clear();
}

void CachePool::flushExcept(unsigned int order) {
  auto exe_orders = getMemoryExecOrder();

  eraseActiveIf([&, order](const unsigned int id) -> bool {
    auto exe_order = exe_orders.at(id - 1);
    auto found = std::find(exe_order.begin(), exe_order.end(), order);
    if (found != exe_order.end()) {
      /**
       * We assumes that flushExcept will be called in front of each execution
       * order, and the order is incremental. So, we can conclude that, if the
       * order passes by the max order of the cache element, it was LAST
       * access of the element.
       */
      CacheElem::Options opt = CacheElem::NONE;
      if (*std::max_element(exe_order.begin(), exe_order.end()) < order)
        opt = CacheElem::LAST_ACCESS;
      elems[id]->swapOut(opt);
      return true;
    }
    return false;
  });
}

void CachePool::flushExcept(std::vector<unsigned int> order) {
  auto exe_orders = getMemoryExecOrder();

  eraseActiveIf([&, order](const unsigned int id) -> bool {
    auto exe_order = exe_orders.at(id - 1);
    for (auto &o : order) {
      auto found = std::find(exe_order.begin(), exe_order.end(), o);
      if (found != exe_order.end())
        return false;
    }
    /**
     * We assumes that flushExcept will be called in front of each execution
     * order, and the order is incremental. So, we can conclude that, if the
     * order passes by the max order of the cache element, it was LAST access of
     * the element.
     */
    CacheElem::Options opt = CacheElem::NONE;
    if (*std::max_element(exe_order.begin(), exe_order.end()) < order[0])
      opt = CacheElem::LAST_ACCESS;
    elems[id]->swapOut(opt);
    return true;
  });
}

void CachePool::clear() {
  flush();
  deallocate();
  policies.clear();
  MemoryPool::clear();
}

bool CachePool::isAllocated() const { return swap_device->isOperating(); }

void CachePool::loadExec(unsigned int order) {
  for (auto &id : exec_ids[order]) {
    validate(id);
  }
}

void CachePool::loadTensor(unsigned int id) { validate(id); }

bool CachePool::loadExecOnce(unsigned int order, ExecIdsIter &iter) {
  if (iter == exec_ids[order].end())
    return true;

  validate(*iter);

  iter++;
  return false;
}

void CachePool::unloadExec(unsigned int order) {
  for (auto &id : exec_ids[order]) {
    invalidate(id);
  }
  actives.clear();
}

void CachePool::unloadTensor(unsigned int order) {
  invalidate(order);
  std::lock_guard<std::mutex> lock(mutex);
  actives.erase(order);
}

void CachePool::loadActives() {
  ml_logd("load active caches");

  for (auto &id : actives) {
    elems[id]->swapIn();
  }
}

void CachePool::unloadActives() {
  ml_logd("unload active caches");
  for (auto &id : actives) {
    elems[id]->swapOut();
  }
}

void CachePool::setFsuWeightPath(std::string path) {
  auto start_with = [](const std::string &str, const std::string &prefix) {
    return str.size() >= prefix.size() &&
           str.compare(0, prefix.size(), prefix) == 0;
  };

  if (!start_with(swap_device->getDevicePath(), "weight_pool")) {
    remove(swap_device->getDevicePath().c_str());
  }

  swap_device->setFsuWeightPath(path);
  swap_device->finish();
  swap_device->start(size(), execution_mode_);
}

void CachePool::eraseActiveIf(
  const std::function<bool(unsigned int id)> &pred) {
  for (auto it = actives.begin(); it != actives.end();
       pred(*it) ? it = actives.erase(it) : ++it) {
  }
}

} // namespace nntrainer
