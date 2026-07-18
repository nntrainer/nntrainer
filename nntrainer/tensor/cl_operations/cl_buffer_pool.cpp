// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   cl_buffer_pool.cpp
 * @date   10 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  ClBufferPool implementation.
 */

#include "cl_buffer_pool.h"

#include <stdexcept>

#include <cl_context.h>
#include <engine.h>
#include <nntrainer_log.h>

namespace nntrainer {

ClBufferPool::~ClBufferPool() { ClBufferPool::deallocate(); }

void ClBufferPool::allocate() {
  auto *cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_context ctx = cc->context_inst_.GetContext();
  cl_device_id dev = cc->context_inst_.GetDeviceId();

  // (1) Allocate the host/SVM plane first. Every existing consumer that has not
  //     yet been moved onto the cl_mem still reads its SVM pointer, so with the
  //     cl_mem plane unbound the output is byte-identical to the SVM pool.
  //
  //     The base MemoryPool::allocate() requests ONE contiguous clSVMAlloc of
  //     the whole plane. When pool_size exceeds CL_DEVICE_MAX_MEM_ALLOC_SIZE,
  //     clSVMAlloc returns null and ClSVMAllocator SILENTLY falls back to host
  //     memory -- which the device cannot use as SVM: map/unmap return
  //     CL_INVALID_VALUE and every GPU kernel reads zero/garbage (measured:
  //     Gemma2-2B with Q6_K embed/lm_head -> plane 1428 MB > 1383 MB cap ->
  //     all-<pad> collapse, while Q4_0 plane 1283 MB < cap stays coherent).
  //     The per-offset cl_mem buffers below already dodge this cap; the SVM
  //     plane must too. Use the per-offset (shared-objects, Pisarchyk & Lee,
  //     arXiv:2001.03288) allocateFSU() path when the plane exceeds the cap so
  //     every SVM buffer is a single tensor wide (far under the cap) and stays
  //     REAL SVM. Small planes keep the single-buffer path -- byte-identical to
  //     before (Qwen3, Gemma2 Q4_0 unaffected).
  cl_ulong max_alloc_pre = 0;
  clGetDeviceInfo(dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc_pre),
                  &max_alloc_pre, nullptr);
  if (max_alloc_pre > 0 && static_cast<cl_ulong>(size()) > max_alloc_pre) {
    ml_logi(
      "ClBufferPool: SVM plane %zu B > MAX_MEM_ALLOC_SIZE %llu B -- using "
      "per-offset (shared-objects) SVM allocation to avoid the host "
      "fallback",
      size(), (unsigned long long)max_alloc_pre);
    MemoryPool::allocateFSU();
  } else {
    MemoryPool::allocate();
  }

  // (2) Resolve CL_DEVICE_MEM_BASE_ADDR_ALIGN (reported in BITS by the spec).
  //     Later stages pad each per-tensor sub-buffer offset up to this so
  //     clCreateSubBuffer origins are valid; log it now to settle the value on
  //     the live device.
  cl_uint align_bits = 0;
  if (clGetDeviceInfo(dev, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(align_bits),
                      &align_bits, nullptr) == CL_SUCCESS)
    base_addr_align_ = align_bits / 8;
  ml_logi("ClBufferPool: CL_DEVICE_MEM_BASE_ADDR_ALIGN = %u bits (%zu bytes)",
          align_bits, base_addr_align_);

  // (3) Pad each per-token planner offset up to base_addr_align_ so a
  //     clCreateSubBuffer origin is valid. Liveness reuse is preserved: tokens
  //     the planner gave the SAME offset (non-overlapping lifetimes) round to
  //     the same padded offset and so share one sub-buffer region. The cl_mem
  //     plane is sized to the furthest padded slice.
  const size_t align = base_addr_align_ ? base_addr_align_ : 128;
  auto &offsets = getMemoryOffset();
  auto &sizes = getMemorySize();
  padded_offsets_.resize(offsets.size());
  offset_maxsize_.clear();
  cl_pool_bytes_ = 0;
  for (size_t i = 0; i < offsets.size(); ++i) {
    const size_t po = ((offsets[i] + align - 1) / align) * align;
    padded_offsets_[i] = po;
    const size_t sz = (i < sizes.size()) ? sizes[i] : 0;
    // One shared sub-buffer per padded offset must cover the LARGEST tensor the
    // planner reused there (disjoint lifetimes, possibly different sizes).
    auto it = offset_maxsize_.find(po);
    if (it == offset_maxsize_.end() || sz > it->second)
      offset_maxsize_[po] = sz;
    const size_t end = po + sz;
    if (end > cl_pool_bytes_)
      cl_pool_bytes_ = end;
  }
  if (cl_pool_bytes_ == 0)
    cl_pool_bytes_ = size();

  // (4) Query the device single-allocation cap and total memory: they gate
  //     the per-offset sizing below and are reported in the summary log.
  cl_ulong max_alloc = 0, global_mem = 0;
  clGetDeviceInfo(dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc),
                  &max_alloc, nullptr);
  clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem),
                  &global_mem, nullptr);

  // (5) Allocate ONE device cl_mem per DISTINCT padded offset, each sized to
  // the
  //     largest tensor the planner reused at that offset. A single whole-plane
  //     cl_mem would exceed CL_DEVICE_MAX_MEM_ALLOC_SIZE for large models on
  //     Adreno -- a Qwen3-4B layer-graph plane is ~2 GB, past the 1.38 GB
  //     single-allocation cap, and clCreateBuffer fails with
  //     CL_INVALID_BUFFER_SIZE (-61). Per-offset buffers cap each allocation at
  //     a single tensor (far under the limit), while preserving the EXACT
  //     one-handle-per-offset sharing the sub-buffer scheme relied on: tokens
  //     the planner reused at one offset (disjoint lifetimes) still bind the
  //     SAME cl_mem (coherent on Adreno's per-handle cache). The only behaviour
  //     change vs the single-plane is that offsets the planner overlapped for
  //     reuse no longer alias one parent region -- correct (disjoint
  //     lifetimes), at the cost of not sharing those overlapped bytes
  //     (alloc_total below reports the real cost vs the plane extent).
  //     Hard-fail (no SVM fallback) so a cl_mem-mode run never silently
  //     degrades to the corrupting hybrid.
  size_t alloc_total = 0, alloc_max = 0;
  for (const auto &kv : offset_maxsize_) {
    const size_t sz = kv.second;
    if (sz == 0)
      continue;
    // A single tensor larger than the device single-allocation cap cannot be a
    // cl_mem at all (clCreateBuffer -> CL_INVALID_BUFFER_SIZE). The only tensor
    // that hits this is a host-resident quantized weight plane (e.g. Gemma4's
    // Q6_K per-layer-embedding table, ~1.84 GB > Adreno's 1.45 GB cap) which is
    // dequantized ON THE HOST and never bound as a device cl_mem -- it already
    // has an SVM-backed MemoryData (host pointer). Skip the (unused) cl_mem for
    // such offsets: getMemory() then leaves device_mem unstamped and the tensor
    // falls back to its SVM backing. This is NOT the corrupting hybrid (that
    // was about activations silently losing their cl_mem); an oversized buffer
    // physically cannot be cl_mem, so SVM is the only correct home.
    if (sz > (size_t)max_alloc) {
      ml_logw("ClBufferPool: per-offset buffer %.1f MB exceeds device "
              "MAX_MEM_ALLOC_SIZE %.1f MB -> keep on SVM (host-resident plane, "
              "e.g. quantized embedding table), no device cl_mem",
              sz / 1048576.0, max_alloc / 1048576.0);
      continue;
    }
    cl_int err = CL_SUCCESS;
    cl_mem buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sz, nullptr, &err);
    if (err != CL_SUCCESS || buf == nullptr)
      throw std::runtime_error(
        "ClBufferPool: clCreateBuffer (per-offset activation buffer) failed "
        "(hard-fail, no SVM fallback): requested " +
        std::to_string(sz) + " bytes, CL err " + std::to_string(err) +
        ", device MAX_MEM_ALLOC_SIZE " + std::to_string(max_alloc));
    // Zero-init is deferred to ensureZeroFilled(): the pool consumer
    // zero-fills only the offsets a device-resident tensor actually binds.
    // An untouched cl_mem never becomes resident, while an eager fill here
    // would commit every buffer up front (hundreds of MB of resident
    // all-zero memory for never-bound offsets).
    offset_sub_[kv.first] = static_cast<void *>(buf);
    alloc_total += sz;
    if (sz > alloc_max)
      alloc_max = sz;
  }
  cl_pool_ = nullptr; // no single parent plane in the per-offset scheme

  ml_logi("ClBufferPool: allocated %zu per-offset device cl_mem buffers "
          "(total %.1f MB, plane extent %.1f MB, largest single %.1f MB; "
          "device MAX_MEM_ALLOC_SIZE %.1f MB, GLOBAL_MEM_SIZE %.1f MB)",
          offset_sub_.size(), alloc_total / 1048576.0,
          cl_pool_bytes_ / 1048576.0, alloc_max / 1048576.0,
          max_alloc / 1048576.0, global_mem / 1048576.0);
}

void ClBufferPool::ensureZeroFilled(unsigned int idx) {
  const size_t i = idx - 1;
  if (i >= padded_offsets_.size())
    return;
  const size_t po = padded_offsets_[i];
  if (zero_filled_.count(po))
    return;
  auto hit = offset_sub_.find(po);
  auto szit = offset_maxsize_.find(po);
  if (hit == offset_sub_.end() || hit->second == nullptr ||
      szit == offset_maxsize_.end() || szit->second == 0)
    return;
  // Same zero-init contract as the old allocate()-time fill: producers write
  // only the M valid rows of an M_pad-row activation, downstream ops (geglu)
  // read all M_pad rows, and uninitialized padded rows were a measured
  // corruption source. The in-order queue orders this fill before any kernel
  // (TensorPool calls this during allocate(), before the first forward).
  auto *cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_command_queue q = cc->command_queue_inst_.GetCommandQueue();
  const cl_uchar zero = 0;
  if (clEnqueueFillBuffer(q, static_cast<cl_mem>(hit->second), &zero,
                          sizeof(zero), 0, szit->second, 0, nullptr,
                          nullptr) != CL_SUCCESS)
    throw std::runtime_error(
      "ClBufferPool: clEnqueueFillBuffer (lazy zero-init) failed");
  zero_filled_.insert(po);
}

std::shared_ptr<MemoryData> ClBufferPool::getMemory(unsigned int idx) {
  // SVM-backed MemoryData: not-yet-converted consumers read it
  // byte-identically.
  auto md = MemoryPool::getMemory(idx);

  // Stamp the per-offset cl_mem (allocated up-front in allocate()) onto
  // device_mem. A GPU producer can write this buffer and a GPU consumer bind it
  // (no SVM map). ONE handle per padded offset: tokens the planner reused at
  // the same offset bind the SAME cl_mem handle (coherent on Adreno's
  // per-handle cache). Stamp it but leave device_valid FALSE -- a tensor uses
  // this cl_mem only when its static residency class is GPU_CLMEM (isClMem());
  // device_valid stays the runtime "producer wrote it this forward" signal.
  // SVM/HOST tensors get a handle stamped too but never bind it.
  const size_t i = idx - 1;
  if (i < padded_offsets_.size()) {
    const size_t po = padded_offsets_[i];
    auto hit = offset_sub_.find(po);
    if (hit != offset_sub_.end() && hit->second != nullptr)
      md->setDeviceValid(false, hit->second);
  }
  return md;
}

void ClBufferPool::deallocate() {
  // Release the one-per-offset shared sub-buffers before their parent cl_mem.
  for (auto &kv : offset_sub_)
    if (kv.second != nullptr)
      clReleaseMemObject(static_cast<cl_mem>(kv.second));
  offset_sub_.clear();
  offset_maxsize_.clear();
  zero_filled_.clear();
  if (cl_pool_ != nullptr) {
    clReleaseMemObject(static_cast<cl_mem>(cl_pool_));
    cl_pool_ = nullptr;
  }
  padded_offsets_.clear();
  cl_pool_bytes_ = 0;
  MemoryPool::deallocate();
}

} // namespace nntrainer
