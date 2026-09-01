// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_mem_allocator.cpp
 * @date    22 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   CUDA Unified Memory allocator implementation.
 */

#include <cuda_mem_allocator.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_set>

#include <cuda_runtime.h>

#include <cuda_context_manager.h>

namespace nntrainer {

namespace {
// Host-fallback ownership set (pointers from MemAllocator::alloc, not
// cudaMallocManaged). ContextManager is a global singleton so no per-instance
// state is needed; keep this hidden in the .cpp.
std::mutex host_owned_mtx;
std::unordered_set<void *> host_owned;
// cudaHostAlloc(Mapped) ownership -- must be freed with cudaFreeHost, not
// cudaFree.
std::mutex pinned_owned_mtx;
std::unordered_set<void *> pinned_owned;

// [WDDM coherence, field 2026-07-09] On the RTX 5070 Laptop (sm_120, WDDM,
// concurrentManagedAccess=0) managed memory is empirically incoherent in BOTH
// directions even with a per-op cudaDeviceSynchronize drain: host reads of
// kernel-written managed pages return stale "previous tenant" bytes
// (DUMP_STATS bisect), and host-written managed (e.g. the fp16 scale table
// every FC reads) is suspect in the same way -- deterministic garbage output
// invariant across cuBLAS/dp4a. Replace the managed pool with PINNED
// HOST-MAPPED (zero-copy) memory on such devices: pages never migrate, host
// R/W is always current, kernels access over the bus at the SAME pointer
// (UVA). Derived device caches (dp4a pack / v8c / workspaces) stay cudaMalloc
// so GEMM bandwidth is unaffected; the zero-copy cost is one bus read per
// staging pass. NNTR_CUDA_HOST_MAPPED overrides both ways (=1 force on any
// box, =0 restore managed).
bool use_host_mapped() {
  static const bool on = []() {
    const char *e = std::getenv("NNTR_CUDA_HOST_MAPPED");
    if (e != nullptr)
      return e[0] == '1';
    return !nntrainer::cuda::ContextManager::Global().concurrentManagedAccess();
  }();
  return on;
}
} // namespace

CudaMemAllocator::CudaMemAllocator(bool device_only) :
  device_only_(device_only) {
  // bring up the device + primary context once (idempotent)
  cuda::ContextManager::Global();
}

void CudaMemAllocator::track_host_owned(void *ptr) {
  std::lock_guard<std::mutex> lk(host_owned_mtx);
  host_owned.insert(ptr);
}

bool CudaMemAllocator::consume_host_owned(void *ptr) {
  std::lock_guard<std::mutex> lk(host_owned_mtx);
  return host_owned.erase(ptr) > 0;
}

void CudaMemAllocator::alloc(void **ptr, size_t size, size_t alignment) {
  static const bool dbg = std::getenv("NNTR_UVM_DEBUG") != nullptr;
  if (size > 0) {
    void *dptr = nullptr;
    // device_only -> real device memory (cudaMalloc, NOT host-addressable);
    // else UVM (cudaMallocManaged, host-coherent). The activation pool uses
    // device_only so the CPU never migrates its pages (the async thrash);
    // weights stay UVM (host writes them at load).
    //
    // INTEGRATED GPU (Tegra/Jetson Orin): host and device share one physical
    // memory pool, so a device-only cudaMalloc gives ZERO bandwidth benefit
    // yet makes the buffer host-INCOHERENT -- the host RoPE/short-attention/
    // norm fallbacks then dereference a device pointer and fault/garbage. Force
    // the managed (host-coherent) pool there regardless of the device_only_
    // request, so the same binary stays coherent on Orin even if a launch
    // script leaks NNTR_CUDA_DEV_ACT. The dev()/dev_ok() GPU gates still pass
    // because they accept Managed too. Discrete GPUs are unaffected.
    const bool integrated = cuda::ContextManager::Global().isIntegrated();
    const bool dev_only = device_only_ && !integrated;
    // Pinned host-mapped (zero-copy) pool replaces managed on cMA==0 devices
    // (see use_host_mapped above). Falls through to managed on any failure.
    if (!dev_only && use_host_mapped()) {
      void *hp = nullptr;
      if (cudaHostAlloc(&hp, size, cudaHostAllocMapped) == cudaSuccess &&
          hp != nullptr) {
        void *dp = nullptr;
        const bool uva_same =
          cudaHostGetDevicePointer(&dp, hp, 0) == cudaSuccess && dp == hp;
        if (uva_same) {
          {
            std::lock_guard<std::mutex> lk(pinned_owned_mtx);
            pinned_owned.insert(hp);
          }
          // Honor MemAllocator's calloc contract: cudaHostAlloc gives no zero
          // guarantee -- the runtime may recycle pinned blocks, and a consumer
          // that reads before it writes then sees a previous tenant's bytes
          // instead of zeros.
          std::memset(hp, 0, size);
          *ptr = hp;
          if (dbg)
            fprintf(stderr,
                    "[UVMDBG] cudaHostAlloc(mapped) %zu bytes -> %p OK\n", size,
                    hp);
          return;
        }
        // no UVA same-pointer guarantee: every call site hands the host
        // pointer to kernels, so a distinct device pointer cannot work --
        // warn once and stay on managed.
        static bool warned = false;
        if (!warned) {
          warned = true;
          fprintf(stderr,
                  "[UVMDBG] host-mapped devicePointer differs (host=%p dev=%p)"
                  " -- pinned pool disabled, using managed\n",
                  hp, dp);
        }
        cudaFreeHost(hp);
      }
      cudaGetLastError(); // benign: fall back to managed below
      if (dbg)
        fprintf(stderr, "[UVMDBG] cudaHostAlloc %zu bytes FAILED -> managed\n",
                size);
    }
    const cudaError_t e =
      dev_only ? cudaMalloc(&dptr, size) : cudaMallocManaged(&dptr, size);
    if (e == cudaSuccess && dptr != nullptr) {
      if (!dev_only && !integrated) {
        // Optionally pin the managed pages to the device. Opt-in
        // (NNTR_CUDA_UVM_DEVICE); a partial, weaker alternative to device_only
        // (kept for A/B). Meaningless for real device memory AND for an
        // integrated GPU (no distinct device location to migrate to; can
        // mis-pin on some Tegra drivers), so it is skipped when integrated.
        static const bool pin_device =
          std::getenv("NNTR_CUDA_UVM_DEVICE") != nullptr;
        if (pin_device) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12020
          int dev = 0;
          cudaGetDevice(&dev);
          cudaMemLocation loc;
          loc.type = cudaMemLocationTypeDevice;
          loc.id = dev;
          cudaMemAdvise(dptr, size, cudaMemAdviseSetPreferredLocation, loc);
          cudaMemAdvise(dptr, size, cudaMemAdviseSetAccessedBy, loc);
          cudaGetLastError(); // clear any benign advise error
#endif
        }
      }
      // Calloc contract: cudaMalloc and
      // cudaMallocManaged contents are undefined. Device-side memset covers
      // both (legal on managed under WDDM cMA==0 — it is a DEVICE access);
      // weights are fully overwritten at load so the only lasting cost is a
      // one-time device fill at allocation.
      cudaMemset(dptr, 0, size);
      *ptr = dptr;
      if (dbg)
        fprintf(stderr, "[UVMDBG] %s %zu bytes -> %p OK\n",
                dev_only ? "cudaMalloc" : "cudaMallocManaged", size, dptr);
      return;
    }
    if (dbg)
      fprintf(stderr, "[UVMDBG] %s %zu bytes FAILED -> host\n",
              dev_only ? "cudaMalloc" : "cudaMallocManaged", size);
    // a failed device alloc leaves the runtime error state set; clear it so a
    // subsequent real CUDA op does not see this benign fallback as an error.
    cudaGetLastError();
  }
  // size==0 or managed alloc failed -> host buffer (correctness > speed). The
  // matching free() consults host_owned to pick std::free vs cudaFree.
  MemAllocator::alloc(ptr, size, alignment);
  track_host_owned(*ptr);
}

void CudaMemAllocator::free(void *ptr) {
  if (ptr == nullptr)
    return;
  if (consume_host_owned(ptr)) {
    MemAllocator::free(ptr);
    return;
  }
  {
    std::lock_guard<std::mutex> lk(pinned_owned_mtx);
    if (pinned_owned.erase(ptr) > 0) {
      cudaFreeHost(ptr);
      return;
    }
  }
  cudaFree(ptr);
}

} // namespace nntrainer
