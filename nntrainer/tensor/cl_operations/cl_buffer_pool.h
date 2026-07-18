// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   cl_buffer_pool.h
 * @date   10 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  MemoryPool variant that backs the planned activation plane with a
 *         single device cl_mem (in addition to the host/SVM plane), so
 *         activations can be made GPU-resident as plain cl_mem -- the
 *         residency model gpu_native uses (no per-op SVM map/unmap, in-order
 *         queue coherence). This is the foundation of the comprehensive
 *         (whole-plane) cl_mem activation port; converting edges incrementally
 *         on top of SVM created an unstable hybrid that corrupted the FFN
 *         sub-chain (device-measured), whereas a planner-allocated cl_mem plane
 *         removes the SVM map model entirely by construction.
 */

#ifndef __CL_BUFFER_POOL_H__
#define __CL_BUFFER_POOL_H__

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <mem_allocator.h>
#include <memory_data.h>
#include <memory_pool.h>

namespace nntrainer {

/**
 * @class   ClBufferPool
 * @brief   MemoryPool that ALSO allocates the planned activation plane as one
 *          device cl_mem. Selected by the TensorPool factory when
 *          NNTR_GPU_CLMEM_POOL is set and the backend allocator is the GPU SVM
 *          allocator. Stage S0 is INERT: it allocates the cl_mem plane and
 *          resolves CL_DEVICE_MEM_BASE_ADDR_ALIGN but no consumer binds it yet,
 *          so the output is token-identical to the SVM pool (and the OFF path
 *          is byte-identical, since this class is then never constructed).
 *
 * @note  The cl_mem handle is held as a void* so this header stays OpenCL-free
 *        and CPU-build-safe, mirroring MemoryData.device_mem.
 */
class ClBufferPool : public MemoryPool {
public:
  /**
   * @brief ClBufferPool constructor.
   * @param allocator backend allocator (the GPU SVM allocator) forwarded to
   *        MemoryPool so the host/SVM plane still backs every existing
   *        (not-yet-converted) consumer.
   */
  explicit ClBufferPool(std::shared_ptr<MemAllocator> allocator) :
    MemoryPool(std::move(allocator)),
    cl_pool_(nullptr),
    base_addr_align_(0),
    cl_pool_bytes_(0) {}

  /**
   * @brief ClBufferPool destructor.
   */
  ~ClBufferPool() override;

  /**
   * @brief Allocate the host/SVM plane (base) AND the parallel device cl_mem
   *        plane. Hard-fails (no SVM fallback) if the cl_mem allocation fails
   * -- a silent fallback would re-introduce the corrupting SVM/cl_mem hybrid.
   */
  void allocate() override;

  /**
   * @brief Release the per-tensor sub-buffers and the device cl_mem plane, then
   *        the base plane.
   */
  void deallocate() override;

  /**
   * @brief Get the allocated memory for token idx. Returns the SVM-backed
   *        MemoryData (so not-yet-converted consumers stay byte-identical) AND
   *        stamps its device_mem with a clCreateSubBuffer slice of the cl_mem
   *        plane at the tensor's base-addr-aligned offset. Stage S1: the
   *        sub-buffer is the per-tensor cl_mem a GPU producer writes and a GPU
   *        consumer binds; until a consumer reads device_mem (gated separately)
   *        this is inert and the output stays byte-identical.
   */
  std::shared_ptr<MemoryData> getMemory(unsigned int idx) override;

  /**
   * @brief Queried CL_DEVICE_MEM_BASE_ADDR_ALIGN in bytes (0 until allocated).
   */
  size_t baseAddrAlign() const { return base_addr_align_; }

  /**
   * @brief Zero-fill token idx's per-offset cl_mem ONCE (idempotent per
   *        offset). allocate() does not eagerly FillBuffer every per-offset
   *        buffer: an untouched cl_mem never becomes resident, while an
   *        eager zero-fill would commit every buffer up front (hundreds of
   *        MB of resident all-zero memory for offsets no device-resident
   *        tensor ever binds). The pool consumer calls this for each tensor
   *        classified device-resident right after stamping, preserving the
   *        zero-init contract (padded rows read by downstream ops) exactly
   *        where it matters.
   */
  void ensureZeroFilled(unsigned int idx);

private:
  void *cl_pool_;          /**< single device cl_mem over the planned plane */
  size_t base_addr_align_; /**< CL_DEVICE_MEM_BASE_ADDR_ALIGN, bytes */
  size_t cl_pool_bytes_;   /**< size of the padded cl_mem plane */
  std::vector<size_t>
    padded_offsets_; /**< per-token offsets rounded up to base_addr_align_ so
                          clCreateSubBuffer origins are valid (liveness reuse is
                          preserved: equal planner offsets stay equal) */
  std::unordered_map<size_t, size_t>
    offset_maxsize_; /**< padded offset -> max tensor size placed there
                          (precomputed in allocate so the one shared sub-buffer
                          covers the largest tensor reusing that offset) */
  std::unordered_map<size_t, void *>
    offset_sub_; /**< ONE shared cl_mem sub-buffer per padded offset. Every
                      token the planner reused at that offset (disjoint
                      lifetimes) binds the SAME handle, so distinct handles
                      never alias one region -- distinct handles over a reused
                      offset break Adreno per-handle image/buffer cache
                      coherence (measured). Released in deallocate(). */
  std::unordered_set<size_t>
    zero_filled_; /**< padded offsets already zero-filled (ensureZeroFilled
                       idempotence) */
};

} // namespace nntrainer

#endif /** __CL_BUFFER_POOL_H__ */
