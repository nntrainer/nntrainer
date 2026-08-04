// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    neuron_dma_allocator.h
 * @date    30 Jul 2026
 * @see     https://github.com/nnstreamer/nntrainer
 * @brief   MemAllocator for buffers handed to the MediaTek Neuron Runtime.
 *
 * @details Milestone 1 scope: allocate plain page-aligned host memory and
 * describe every buffer to the runtime as a non-ION buffer
 * (`BufferAttribute{ -1 }`), which `neuron/api/Types.h` documents as an
 * explicitly supported, correct path ("-1: Non-ION buffer"). This trades
 * away true zero-copy DMA-BUF sharing with the NPU for a allocator that is
 * simple to get right end-to-end first.
 *
 * Wiring a real ION / DMA-BUF heap (via libion.so or /dev/dma_heap/<name>,
 * both of which vary by Android version and vendor image) is deliberately
 * deferred: unlike Qualcomm's libcdsprpc.so, there is no single documented
 * MediaTek allocator entry point in this SDK to target blind, and getting
 * the heap name wrong is silent-corruption risk, not a load-time failure.
 * Confirm the right heap on-device before implementing that path.
 */
#ifndef __NEURON_DMA_ALLOCATOR_H__
#define __NEURON_DMA_ALLOCATOR_H__

#include "neuron/api/Types.h"

#include <mem_allocator.h>
#include <unordered_map>

namespace nntrainer {

/** @brief MemAllocator subclass for the "neuron" compute engine. */
class NeuronDmaAllocator : public MemAllocator {
public:
  NeuronDmaAllocator() = default;
  ~NeuronDmaAllocator() override = default;

  void alloc(void **ptr, size_t size, size_t alignment) override;
  void free(void *ptr) override;

  std::string getName() override { return "neuron"; }

  /**
   * @brief Buffer attribute to pass to NeuronRuntime_setInput/setOutput for
   * a pointer previously returned by alloc(). Always non-ION for now (see
   * file-level note); returns `BufferAttribute{-1}` for any pointer,
   * including ones not allocated by this allocator, since that is a safe
   * default for the runtime regardless of origin.
   */
  BufferAttribute attributeFor(void *ptr) const {
    (void)ptr;
    return BufferAttribute{-1};
  }

private:
  // Tracks allocations made through this allocator purely so free() can
  // assert it is not asked to release a foreign pointer; not used for any
  // ION fd bookkeeping yet (see class comment).
  std::unordered_map<void *, size_t> owned_;
};

} // namespace nntrainer

#endif /* __NEURON_DMA_ALLOCATOR_H__ */
