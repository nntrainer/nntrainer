// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   cl_buffer_pool.cpp
 * @date   24 August 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Implementation of the device cl_mem plane.
 */

#include "cl_buffer_pool.h"

#include <CL/cl.h>

#include <cl_context.h>
#include <engine.h>
#include <nntrainer_log.h>
#include <opencl_loader.h>

namespace nntrainer {

namespace {

/**
 * @brief the OpenCL context this process' gpu Context owns, or nullptr when
 *        none is registered.
 *
 * @details getRegisteredContext() throws when the name is absent, and absent
 * is reachable: a build with OpenCL compiled in still declines to bring the
 * backend up when no device answers. That exception would otherwise leave
 * TensorPool::allocate() -- not an OpenCL call site, and with no reason to
 * catch it -- and abort a model that only ever wanted the shared plane. A null
 * answer joins the other "cannot place this on the device" answers below and
 * leaves the tensor where it already is.
 *
 * @note The static_cast is what reaching context_inst_ / command_queue_inst_
 * costs: they are ClContext members with no accessor on Context, so there is
 * no virtual seam to go through yet. It is confined to this one function so
 * that introducing one later is a single edit.
 *
 * @return the OpenCL context, or nullptr when the gpu backend is not
 *         registered in this process
 */
ClContext *clContext() {
  try {
    return static_cast<ClContext *>(
      Engine::Global().getRegisteredContext("gpu"));
  } catch (const std::exception &e) {
    ml_logw("ClBufferPool: no OpenCL context is registered (%s); the tensor "
            "stays on the shared plane",
            e.what());
    return nullptr;
  }
}

} // namespace

ClBufferPool::~ClBufferPool() { ClBufferPool::deallocate(); }

void ClBufferPool::allocate() {
  /** The base MemoryPool::allocate() requests ONE contiguous clSVMAlloc of the
   *  whole plane. When that plane exceeds CL_DEVICE_MAX_MEM_ALLOC_SIZE,
   *  clSVMAlloc returns null and ClSVMAllocator falls back to plain host
   *  memory -- which the device cannot use as SVM at all: map/unmap return
   *  CL_INVALID_VALUE and every kernel reads zeros. The fallback is silent by
   *  design (correctness over speed for a host-only run), so on a GPU run it
   *  surfaces only as output collapsing to a single repeated token.
   *
   *  The per-offset cl_mem buffers below already dodge this cap; the SVM plane
   *  has to as well. Above the cap, take the per-offset (shared-objects)
   *  allocateFSU() path so every SVM buffer is a single tensor wide -- far
   *  under the cap -- and stays REAL SVM. A plane under the cap keeps the
   *  single-buffer path and is byte-identical to before.
   *
   *  A process with no registered gpu Context reports no cap and keeps the
   *  single-buffer path, which is the same answer it gave before: on such a
   *  run the SVM allocator is already the host fallback and there is nothing
   *  to protect. */
  cl_ulong plane_cap = 0;
  if (auto *cc = clContext())
    opencl::clGetDeviceInfo(cc->context_inst_.GetDeviceId(),
                            CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(plane_cap),
                            &plane_cap, nullptr);
  if (plane_cap > 0 && static_cast<cl_ulong>(size()) > plane_cap) {
    ml_logi("ClBufferPool: the %.1f MB SVM plane exceeds the device maximum "
            "allocation %.1f MB; allocating it per offset (shared objects) so "
            "every buffer stays real SVM",
            size() / 1048576.0, plane_cap / 1048576.0);
    MemoryPool::allocateFSU();
  } else {
    MemoryPool::allocate();
  }

  /** Record where the planner put each token. Tokens that share an offset have
   *  disjoint lifetimes, so one buffer sized to the largest of them backs all
   *  of them -- the device-side expression of the planner's reuse. */
  const auto &offsets = getMemoryOffset();
  const auto &sizes = getMemorySize();

  std::lock_guard<std::mutex> lk(device_mtx_);
  token_offset_.assign(offsets.begin(), offsets.end());
  offset_size_.clear();
  for (size_t i = 0; i < offsets.size(); ++i) {
    const size_t bytes = (i < sizes.size()) ? sizes[i] : 0;
    auto it = offset_size_.find(offsets[i]);
    if (it == offset_size_.end() || bytes > it->second)
      offset_size_[offsets[i]] = bytes;
  }
}

void *ClBufferPool::deviceMemory(unsigned int idx) {
  std::lock_guard<std::mutex> lk(device_mtx_);

  const size_t i = idx - 1;
  if (i >= token_offset_.size())
    return nullptr;

  const size_t offset = token_offset_[i];
  auto hit = offset_buffer_.find(offset);
  if (hit != offset_buffer_.end())
    return hit->second;

  auto sit = offset_size_.find(offset);
  if (sit == offset_size_.end() || sit->second == 0)
    return nullptr;
  const size_t bytes = sit->second;

  auto *cc = clContext();
  if (cc == nullptr)
    return nullptr;

  cl_device_id dev = cc->context_inst_.GetDeviceId();

  /** A single allocation larger than the device can hold is not a device
   *  buffer at all. Report it and leave the tensor on the shared plane, which
   *  is where a buffer this size (a host-dequantized weight table) belongs
   *  anyway -- placing it nowhere would be worse than placing it there. */
  cl_ulong max_alloc = 0;
  opencl::clGetDeviceInfo(dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc),
                          &max_alloc, nullptr);
  if (max_alloc > 0 && static_cast<cl_ulong>(bytes) > max_alloc) {
    ml_logw("ClBufferPool: %.1f MB exceeds the device maximum allocation "
            "%.1f MB; the tensor stays on the shared plane",
            bytes / 1048576.0, max_alloc / 1048576.0);
    return nullptr;
  }

  cl_int err = CL_SUCCESS;
  cl_mem buf = opencl::clCreateBuffer(cc->context_inst_.GetContext(),
                                      CL_MEM_READ_WRITE, bytes, nullptr, &err);
  if (err != CL_SUCCESS || buf == nullptr) {
    ml_logw("ClBufferPool: clCreateBuffer for %zu bytes failed with %d; the "
            "tensor stays on the shared plane",
            bytes, err);
    return nullptr;
  }

  /** Match the shared plane's zero-initialisation: a producer writes only the
   *  rows it has, and an element-wise consumer reads the padded rows too. The
   *  in-order queue orders this fill ahead of every kernel, since allocation
   *  precedes the first forward. */
  const cl_uchar zero = 0;
  if (opencl::clEnqueueFillBuffer(cc->command_queue_inst_.GetCommandQueue(),
                                  buf, &zero, sizeof(zero), 0, bytes, 0,
                                  nullptr, nullptr) != CL_SUCCESS) {
    opencl::clReleaseMemObject(buf);
    ml_logw("ClBufferPool: zero-filling a %zu byte device buffer failed; the "
            "tensor stays on the shared plane",
            bytes);
    return nullptr;
  }

  offset_buffer_[offset] = static_cast<void *>(buf);
  return offset_buffer_[offset];
}

void ClBufferPool::deallocate() {
  {
    std::lock_guard<std::mutex> lk(device_mtx_);
    for (auto &entry : offset_buffer_)
      if (entry.second != nullptr)
        opencl::clReleaseMemObject(static_cast<cl_mem>(entry.second));
    offset_buffer_.clear();
    offset_size_.clear();
    token_offset_.clear();
  }
  MemoryPool::deallocate();
}

} // namespace nntrainer
