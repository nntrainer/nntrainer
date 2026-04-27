// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    mem_allocator.cpp
 * @date    13 Jan 2025
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   This is memory allocator for memory pool
 *
 */

#include <cstdlib>
#include <cstring>
#include <mem_allocator.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>

#if defined(_WIN32)
#include <malloc.h>
#endif

namespace nntrainer {

namespace {

/**
 * @brief Round size up to a multiple of alignment.
 *
 * std::aligned_alloc requires size to be an integer multiple of
 * alignment; otherwise behaviour is implementation-defined and on
 * glibc it returns nullptr. Page-aligned allocations are common
 * for MemoryPool, so this fix-up matters.
 */
size_t round_up(size_t size, size_t alignment) {
  return (size + alignment - 1) & ~(alignment - 1);
}

} // namespace

void MemAllocator::alloc(void **ptr, size_t size, size_t alignment) {
  NNTR_THROW_IF(size == 0, std::invalid_argument)
    << "MemAllocator::alloc: zero-size allocation rejected";
  NNTR_THROW_IF(alignment == 0 || (alignment & (alignment - 1)) != 0,
                std::invalid_argument)
    << "MemAllocator::alloc: alignment must be a non-zero power of two";

  const size_t aligned_size = round_up(size, alignment);

#if defined(_WIN32)
  *ptr = _aligned_malloc(aligned_size, alignment);
#else
  *ptr = std::aligned_alloc(alignment, aligned_size);
#endif

  NNTR_THROW_IF(*ptr == nullptr, std::runtime_error)
    << "MemAllocator::alloc: aligned_alloc(" << alignment << ", "
    << aligned_size << ") failed";

  // MemoryPool callers historically expected zeroed buffers (calloc
  // semantics). Preserve that — kernels that read uninitialised
  // gradient slots would otherwise see garbage.
  std::memset(*ptr, 0, aligned_size);
}

void MemAllocator::free(void *ptr) {
  if (ptr == nullptr)
    return;
#if defined(_WIN32)
  _aligned_free(ptr);
#else
  std::free(ptr);
#endif
}

} // namespace nntrainer
