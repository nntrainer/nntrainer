// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    neuron_dma_allocator.cpp
 * @date    30 Jul 2026
 * @see     https://github.com/nnstreamer/nntrainer
 * @brief   MemAllocator for buffers handed to the MediaTek Neuron Runtime.
 */
#include "neuron_dma_allocator.h"

#include <cstdlib>
#include <nntrainer_log.h>

namespace nntrainer {

void NeuronDmaAllocator::alloc(void **ptr, size_t size, size_t alignment) {
  // Same contract as the base MemAllocator::alloc (aligned, zero-filled);
  // see MemAllocator::alloc's doc for the round-up-to-alignment rationale.
  MemAllocator base;
  base.alloc(ptr, size, alignment);
  owned_[*ptr] = size;
}

void NeuronDmaAllocator::free(void *ptr) {
  if (ptr == nullptr) {
    return;
  }
  auto it = owned_.find(ptr);
  if (it == owned_.end()) {
    ml_logw("NeuronDmaAllocator::free: pointer %p was not allocated by this "
            "allocator",
            ptr);
  } else {
    owned_.erase(it);
  }
  // Delegate to the base implementation so the Windows _aligned_free /
  // POSIX std::free split stays in exactly one place.
  MemAllocator base;
  base.free(ptr);
}

} // namespace nntrainer
