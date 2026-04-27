// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   memory_pool.cpp
 * @date   11 August 2021
 * @see    https://github.com/nntrainer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Memory Pool Class
 */

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory_pool.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <numeric>
#include <profiler.h>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <sysinfoapi.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace nntrainer {

namespace {

/**
 * @brief Page size used as the default alignment for pool allocations.
 *
 * Page-aligned buffers are required by RPC / SVM backends and don't
 * hurt the host case (aligned_alloc rounds up the request).
 */
size_t system_page_size() {
#if defined(_WIN32)
  SYSTEM_INFO sys_info;
  GetSystemInfo(&sys_info);
  return sys_info.dwPageSize;
#else
  return static_cast<size_t>(sysconf(_SC_PAGE_SIZE));
#endif
}

} // namespace

unsigned int MemoryPool::requestMemory(size_t bytes, unsigned int start_time,
                                       unsigned int end_time,
                                       std::vector<unsigned int> exec_order,
                                       TensorLifespan lifespan, bool is_wgrad) {
  if (bytes == 0)
    throw std::invalid_argument("Requesting memory of 0 size");

  if (mem_pool != nullptr || !owned_buffers_.empty())
    throw std::invalid_argument(
      "Deallocate memory pool before requesting more memory");

  if (end_time <= start_time)
    throw std::invalid_argument(
      "Invalid validity range for the requested memory");

  memory_size.push_back(bytes);
  memory_validity.push_back({start_time, end_time});
  memory_exec_order.push_back(exec_order);
  memory_is_wgrad.push_back(is_wgrad);
  if (is_wgrad)
    n_wgrad++;

  /** invalidate min_pool_size if already there */
  min_pool_size = 0;

  return memory_size.size();
}

double MemoryPool::planLayout(const MemoryPlanner &planner) {
  if (mem_pool != nullptr)
    /** mem_pool must be deallocated when planLayout is being called */
    throw std::runtime_error("Planning memory layout after allocation");

  if (memory_size.empty())
    throw std::runtime_error("Planning memory layout for empty pool");

  /** calculate min_pool_size if not already calculated */
  if (min_pool_size == 0)
    min_pool_size = calcMinMemoryRequirement();

  pool_size = planner.planLayout(memory_size, memory_validity, memory_offset,
                                 memory_is_wgrad, n_wgrad);
  if (pool_size < min_pool_size || !validateLayout())
    throw std::runtime_error("Planned layout is not feasible");

  return double(min_pool_size) / double(pool_size);
}

void MemoryPool::allocate() {
  if (pool_size == 0)
    throw std::runtime_error("Allocating memory pool with size 0");

  if (mem_pool != nullptr)
    throw std::runtime_error("Memory pool is already allocated");

  ml_logi("MemoryPool::allocate size: %zu allocator: %s", pool_size,
          allocator_->getName().c_str());

  // Single contiguous buffer routed through the per-vendor allocator.
  // For ClSVMAllocator this returns SVM memory directly addressable
  // by both host and device; for QNNRpcManager it returns rpcmem; the
  // base allocator returns page-aligned, zero-initialised host memory.
  allocator_->alloc(&mem_pool, pool_size, system_page_size());
  owned_buffers_.push_back(mem_pool);

  // Hand out per-token slices off the same buffer at planned offsets.
  for (size_t i = 0; i < memory_offset.size(); ++i) {
    char *ptr = static_cast<char *>(mem_pool) + memory_offset[i];
    memory_ptrs.push_back(ptr);
  }

#ifdef PROFILE
  static long long seq = 0;
  std::string msg("MemoryPool #");
  msg.append(std::to_string(seq++));
  PROFILE_MEM_ALLOC(mem_pool, pool_size, msg);
#endif
}

void MemoryPool::allocateFSU() {
  if (pool_size == 0)
    throw std::runtime_error("Allocating memory pool with size 0");

  if (mem_pool != nullptr || !owned_buffers_.empty())
    throw std::runtime_error("Memory pool is already allocated");

  ml_logi("MemoryPool::allocateFSU size: %zu allocator: %s", pool_size,
          allocator_->getName().c_str());

  // FSU path allocates a separate buffer per distinct offset and
  // shares it across all tokens that map to that offset (typical for
  // double-buffered weight reuse). Earlier code used ALIGNED_ALLOC
  // macros and never freed these — fixed here by tracking unique
  // buffers in owned_buffers_.
  std::map<size_t, void *> offset_ptr;     // offset : ptr
  std::map<size_t, size_t> allocated_size; // offset : memory size
  std::map<size_t, std::vector<int>>
    offset_indices; // offset : indices that share this offset

  const size_t alignment = system_page_size();

  int i = 0;
  for (auto &s : memory_offset) {
    size_t current_size = memory_size.at(i);
    auto it = offset_ptr.find(s);
    if (it == offset_ptr.end()) {
      void *ptr = nullptr;
      allocator_->alloc(&ptr, current_size, alignment);
      memory_ptrs.push_back(ptr);
      owned_buffers_.push_back(ptr);
      offset_ptr[s] = ptr;
      allocated_size[s] = current_size;
      offset_indices[s].push_back(i);
    } else {
      void *existing_ptr = it->second;
      size_t max_size = allocated_size[s];
      if (max_size < current_size) {
        // A larger request lands on the same offset — reallocate the
        // shared buffer and remap every aliasing entry.
        void *new_ptr = nullptr;
        allocator_->alloc(&new_ptr, current_size, alignment);
        for (int idx : offset_indices[s]) {
          memory_ptrs[idx] = new_ptr;
        }
        // Replace in owned_buffers_ so deallocate() releases the new
        // one, not the old one we're about to free.
        auto pos =
          std::find(owned_buffers_.begin(), owned_buffers_.end(), existing_ptr);
        if (pos != owned_buffers_.end())
          *pos = new_ptr;
        allocator_->free(existing_ptr);
        offset_ptr[s] = new_ptr;
        allocated_size[s] = current_size;
      }
      memory_ptrs.push_back(offset_ptr[s]);
      offset_indices[s].push_back(i);
    }
    i++;
  }
}

std::shared_ptr<MemoryData> MemoryPool::getMemory(unsigned int idx) {
  if (mem_pool == nullptr && owned_buffers_.empty())
    throw std::invalid_argument("Getting memory before allocation");

  auto mem_data = std::make_shared<MemoryData>((void *)memory_ptrs.at(idx - 1));
  // SVM-ness propagates implicitly through the allocator name now;
  // callers that need to know inspect getAllocator()->getName().
  mem_data->setSVM(allocator_->getName() == "gpu-svm");
  return mem_data;
}

void MemoryPool::deallocate() {
  // Symmetric to allocate/allocateFSU: free every uniquely-owned
  // buffer through the same allocator. Mismatching alloc/free
  // backends would corrupt the heap (e.g. clSVMFree on host memory),
  // so the allocator_ shared_ptr is the single source of truth.
  for (void *buf : owned_buffers_) {
    allocator_->free(buf);
#ifdef PROFILE
    PROFILE_MEM_DEALLOC(buf);
#endif
  }
  owned_buffers_.clear();

  mem_pool = nullptr;
  memory_size.clear();
  memory_validity.clear();
  memory_exec_order.clear();
  memory_is_wgrad.clear();
  memory_ptrs.clear();
}

size_t MemoryPool::size() { return pool_size; }

size_t MemoryPool::minMemoryRequirement() {
  if (memory_size.size() && min_pool_size == 0)
    min_pool_size = calcMinMemoryRequirement();

  return min_pool_size;
}

bool MemoryPool::validateLayout() {
  if (memory_offset.size() != memory_size.size())
    return false;

  if (memory_size.empty())
    return pool_size == 0;

  return validateOverflow() && validateOverlap();
}

bool MemoryPool::validateOverflow() {
  for (unsigned int idx = 0; idx < memory_size.size(); idx++)
    if (memory_offset[idx] + memory_size[idx] > pool_size)
      return false;

  return true;
}

template <typename T> static bool overlap(T s1, T e1, T s2, T e2) {
#if DEBUG
  if (e1 <= s1 || e2 <= s2)
    throw std::invalid_argument("Invalid range for intervals in MemoryPool");
#endif

  return !(e1 <= s2 || e2 <= s1);
}

bool MemoryPool::validateOverlap() {
  std::vector<unsigned int> perm = getSortedPermutation();

  size_t len = perm.size();
  for (unsigned int i = 0; i < len; i++) {
    unsigned int idx = perm[i];
    size_t mem_start = memory_offset[idx], mem_size = memory_size[idx];
    unsigned int valid_start = memory_validity[idx].first,
                 valid_end = memory_validity[idx].second;
    for (unsigned int match = idx + 1; match < len; match++) {
      if (overlap(mem_start, mem_start + mem_size, memory_offset[match],
                  memory_offset[match] + memory_size[match])) {
        if (overlap(valid_start, valid_end, memory_validity[match].first,
                    memory_validity[match].second))
          return false;
      } else {
        break;
      }
    }
  }

  return true;
}

std::vector<unsigned int> MemoryPool::getSortedPermutation() {
  std::vector<unsigned int> perm(memory_size.size());
  std::iota(perm.begin(), perm.end(), 0);
  std::sort(perm.begin(), perm.end(), [&](auto const &idx1, auto const &idx2) {
    if (memory_offset[idx1] == memory_offset[idx2])
      return memory_size[idx1] < memory_size[idx2];

    return memory_offset[idx1] < memory_offset[idx2];
  });

  return perm;
}

size_t MemoryPool::calcMinMemoryRequirement() {
  auto max_interval =
    *std::max_element(memory_validity.begin(), memory_validity.end(),
                      [](auto const &val1, auto const &val2) {
                        return val1.second < val2.second;
                      });
  unsigned int last_interval = max_interval.second;
  if (last_interval == (std::numeric_limits<unsigned int>::max)()) {
    max_interval = *std::max_element(
      memory_validity.begin(), memory_validity.end(),
      [last_interval](auto const &val1, auto const &val2) {
        return ((val2.second != last_interval) && (val1.second < val2.second));
      });
    last_interval = max_interval.second;
    if (last_interval == (std::numeric_limits<unsigned int>::max)())
      last_interval = 1;
  }

  std::vector<size_t> interval_req(last_interval + 1, 0);
  for (unsigned int idx = 0; idx < memory_size.size(); idx++) {
    for (unsigned int interval = memory_validity[idx].first;
         interval < std::min(memory_validity[idx].second, last_interval);
         interval++) {
      interval_req[interval] += memory_size[idx];
    }
  }

  return *std::max_element(interval_req.begin(), interval_req.end());
}

void MemoryPool::clear() {
  if (mem_pool != nullptr || !owned_buffers_.empty())
    throw std::invalid_argument("Cannot clear allocated memory pool");

  memory_size.clear();
  memory_validity.clear();
  memory_offset.clear();
  file_offset.clear();
  memory_is_wgrad.clear();

  pool_size = 0;
  min_pool_size = 0;
  n_wgrad = 0;
}

bool MemoryPool::isAllocated() const {
  return mem_pool != nullptr || !owned_buffers_.empty();
}

} // namespace nntrainer
