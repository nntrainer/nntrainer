// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_mem_allocator.h
 * @date    22 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   MemAllocator subclass that routes MemoryPool allocations through
 * CUDA Unified Memory (cudaMallocManaged), the direct analogue of the OpenCL
 * SVM path (ClSVMAllocator): one pointer that is both host- addressable and
 * device-accessible, so engine=cuda tensors are device-resident without a
 * separate copy step. CudaContext installs an instance at engine init. Falls
 * back to host memory if the managed allocation fails, so a missing-UVM device
 * degrades rather than fails.
 */

#ifndef __CUDA_MEM_ALLOCATOR_H__
#define __CUDA_MEM_ALLOCATOR_H__

#include <string>

#include <mem_allocator.h>

namespace nntrainer {

/**
 * @brief CUDA Unified Memory (managed) allocator.
 */
class CudaMemAllocator : public MemAllocator {
public:
  /**
   * @brief Ensure the CUDA device + primary context exist before any
   *        managed allocation.
   *
   * @param device_only when true, alloc() uses cudaMalloc (real device memory,
   *        NOT host-addressable) instead of cudaMallocManaged (UVM). Used for
   *        the activation pool so the CPU never touches it -> no host<->device
   *        page migration under async (the UVM thrash). Weights keep UVM
   * (false) because the host writes them at load. The OpenCL cl_mem analog.
   */
  explicit CudaMemAllocator(bool device_only = false);

  /**
   * @copydoc MemAllocator::alloc
   *
   * Tries cudaMallocManaged first; on failure (or size==0) falls back to
   * MemAllocator::alloc and records the pointer as host-owned so free() picks
   * the right release path.
   */
  void alloc(void **ptr, size_t size, size_t alignment) override;

  /**
   * @copydoc MemAllocator::free
   *
   * cudaFree for managed pointers, std::free for host-fallback pointers.
   */
  void free(void *ptr) override;

  std::string getName() override {
    return device_only_ ? "cuda-dev" : "cuda-uvm";
  }

  // UVM (device_only_=false) is unified host+device memory (the SVM analogue)
  // -> host-addressable; a device_only_ allocator's CONTRACT is "device
  // memory, the host must not touch it" (manager.h's activation pool). NOTE:
  // on an integrated GPU (Tegra/Orin) alloc() upgrades a device_only request
  // to cudaMallocManaged (host-coherent); that runtime upgrade does not change
  // the contract, so the predicate stays static here.
  bool isHostAddressable() const override { return !device_only_; }
  bool isDeviceVisible() const override { return true; }

  // [field 2026-07-09, Windows unified-binary hijack] Every isSVM() CONSUMER
  // in the tree is an OpenCL kernel-binding gate (clSetKernelArgSVMPointer
  // paths: reshaped_rms_norm / rms_norm_gpu / rms_reverse_norm / mha_core /
  // per_layer_slice / float_tensor gemv-int4 / ...). Deriving true for UVM
  // (base isSVM = host-addressable && device-visible, "the SVM analogue")
  // made the UNIFIED build's OpenCL fast paths hijack CUDA-engine tensors:
  // the Intel CL kernel silently fails on a non-OpenCL pointer, gpu_done is
  // set unconditionally, the layer output is NEVER WRITTEN, and the planner's
  // previous tenant leaks downstream (measured: layer0_gated_norm r0
  // bit-equal to layer0_wq -> whole-model deterministic garbage in every
  // CUDA-on-Windows config). Linux never sees it: build_cl / build_cuda are
  // SEPARATE binaries there -- only the Windows unified build compiles both
  // engines in. To every consumer this flag means "OpenCL SVM", so CUDA
  // memory must report false; CUDA-side device checks use
  // cuda::dev_accessible(), not this flag.
  bool isSVM() const override { return false; }

private:
  bool device_only_;
  void track_host_owned(void *ptr);
  bool consume_host_owned(void *ptr);
};

} // namespace nntrainer

#endif // __CUDA_MEM_ALLOCATOR_H__
