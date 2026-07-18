// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 Jiho Chu <jiho.chu@samsung.com>
 *
 * @file   memory_data.h
 * @date   14 Oct 2022
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jiho Chu <jiho.chu@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  MemoryData class
 *
 */

#ifndef __MEMORY_DATA_H__
#define __MEMORY_DATA_H__

#include <functional>

namespace nntrainer {

using MemoryDataValidateCallback = std::function<void(unsigned int)>;

/**
 * @brief  GPU residency class of a tensor's backing memory.
 * @details Decided STATICALLY by the memory planner / pool at allocation time
 *          (a tensor property), NOT a per-edge runtime flip: the engine and
 *          role of each tensor are known before execution, so "this tensor
 *          lives in cl_mem" is an allocation decision applied uniformly to all
 *          of that tensor's producers and consumers.
 *          - HOST: host-only (CPU) memory.
 *          - SVM: shared virtual memory (device-visible AND host-addressable).
 *          - GPU_CLMEM: device cl_mem, NOT host-addressable; layers bind it as
 *            a cl_mem kernel argument (see Tensor::isClMem / getClMem).
 *          - IMAGE2D: device image2d (texture-cached); reserved for the
 *            role-driven KV->image crossover.
 *          - RPCMEM: ION/rpcmem shared buffer for an NPU backend (DSP-
 *            visible); reserved.
 */
enum class ResidencyClass : unsigned char {
  HOST = 0,      /**< host-only (CPU) memory */
  SVM = 1,       /**< shared virtual memory (device + host addressable) */
  GPU_CLMEM = 2, /**< device cl_mem (not host-addressable) */
  IMAGE2D = 3,   /**< device image2d texture; reserved */
  RPCMEM = 4,    /**< ION/rpcmem NPU shared buffer; reserved */
};

/**
 * @brief  MemoryData Class
 */
class MemoryData {
  /**
   * @brief MemoryPool is granted friend access to call setSVM()
   * @details This restricts the ability to modify the SVM allocation flag
   *          to only MemoryPool::getMemory(), preventing malicious or
   *          accidental modification from other parts of the codebase.
   */
  friend class MemoryPool;

public:
  /**
   * @brief  Constructor of Memory Data
   * @param[in] addr Memory data
   */
  explicit MemoryData(void *addr) :
    valid(true),
    id(0),
    address(addr),
    validate_cb([](unsigned int) {}),
    invalidate_cb([](unsigned int) {}),
    svm_allocation(false),
    device_valid(false),
    device_mem(nullptr),
    residency_(ResidencyClass::HOST) {}

  /**
   * @brief  Constructor of Memory Data
   * @param[in] mem_id validate callback.
   * @param[in] v_cb validate callback.
   * @param[in] i_cb invalidate callback.
   */
  explicit MemoryData(unsigned int mem_id, MemoryDataValidateCallback v_cb,
                      MemoryDataValidateCallback i_cb,
                      void *memory_ptr = nullptr) :
    valid(false),
    id(mem_id),
    address(memory_ptr),
    validate_cb(v_cb),
    invalidate_cb(i_cb),
    svm_allocation(false),
    device_valid(false),
    device_mem(nullptr),
    residency_(ResidencyClass::HOST) {}

  /**
   * @brief  Deleted constructor of Memory Data
   */
  explicit MemoryData() = delete;

  /**
   * @brief  Constructor of MemoryData
   */
  explicit MemoryData(MemoryDataValidateCallback v_cb,
                      MemoryDataValidateCallback i_cb) = delete;
  /**
   * @brief  Constructor of MemoryData
   */
  explicit MemoryData(void *addr, MemoryDataValidateCallback v_cb,
                      MemoryDataValidateCallback i_cb) = delete;

  /**
   * @brief  Destructor of Memory Data
   */
  virtual ~MemoryData() = default;

  /**
   * @brief  Set address
   */
  void setAddr(void *addr) { address = addr; }

  /**
   * @brief  Get address
   */
  template <typename T = float> T *getAddr() const {
    return static_cast<T *>(address);
  }

  /**
   * @brief  Validate memory data
   */
  void validate() {
    if (valid)
      return;
    if (validate_cb != nullptr)
      validate_cb(id);
  }

  /**
   * @brief  Invalidate memory data
   */
  void invalidate() {
    if (!valid)
      return;
    if (invalidate_cb != nullptr)
      invalidate_cb(id);
  }

  /**
   * @brief  Set valid
   */
  void setValid(bool v) { valid = v; }

  /**
   * @brief   Check if data is a shared virtual memory
   */
  bool isSVM() const { return svm_allocation; }

  /**
   * @brief  True unless this memory is DEVICE-ONLY (e.g. cudaMalloc): the host
   *         must not dereference the pointer and every host read/write has to
   *         stage. Stamped from MemAllocator::isHostAddressable() at pool bind
   *         (same pattern as the SVM stamp) so consumers ask the tensor, not
   *         the driver -- replaces per-call driver probes such as
   *         cudaPointerGetAttributes (layering rule: capability flows up
   *         through the allocator; no consumer queries the driver directly).
   *         Defaults true: plain host buffers / Tensor::Map are host memory.
   */
  bool isHostAddressable() const { return host_addressable; }

  /**
   * @brief  Device-residency bit of this memory.
   * @details device_valid means "the freshest copy of this data lives in the
   *          device buffer device_mem", DISTINCT from `valid` (which means
   *          host-resident and is toggled by CachePool swapOut via setAddr).
   *          A producer device op sets it after writing device_mem; a host
   *          consumer clears it after syncing down. Default false => the bit
   *          stays inert and every consumer falls through to the existing
   *          SVM/host path until a device pool stamps it. device_mem is held
   *          as a non-owning void* so this header stays OpenCL-free (CPU
   *          build safe).
   */
  bool isDeviceValid() const { return device_valid; }

  /**
   * @brief  Set the device-residency bit and (optionally) the device buffer.
   */
  void setDeviceValid(bool v, void *dev = nullptr) {
    device_valid = v;
    if (dev != nullptr)
      device_mem = dev;
  }

  /**
   * @brief  Get the resident device buffer (cl_mem as void*), or null.
   */
  void *deviceMem() const { return device_mem; }

  /**
   * @brief  Get the static residency class assigned by the planner/pool.
   */
  ResidencyClass residency() const { return residency_; }

  /**
   * @brief  Set the static residency class (planner/pool allocation decision).
   * @note   Distinct from the per-edge device_valid runtime bit: residency_ is
   *         a static tensor property set once at allocation; device_valid is
   *         the runtime overlay. Storing it alone is inert (byte-identical)
   *         until layers bind by class.
   */
  void setResidency(ResidencyClass r) { residency_ = r; }

  /**
   * @brief  True if this memory lives in device cl_mem (not host-addressable).
   * @details Layers use this to decide HOW to bind: a cl_mem kernel argument
   *          (SetKernelArguments) for GPU_CLMEM vs an SVM/host pointer
   *          (SetKernelSVMArguments) otherwise. Host pointer arithmetic on a
   *          GPU_CLMEM tensor is a bug by construction.
   */
  bool isClMem() const { return residency_ == ResidencyClass::GPU_CLMEM; }

private:
  /**
   * @brief  Set SVM allocation flag (private - only accessible by MemoryPool)
   * @param[in] is_svm True if this memory is a shared virtual memory
   * @note This method is intentionally private to prevent modification of the
   *       SVM flag after MemoryData creation. Only MemoryPool (friend class)
   *       can call this during memory allocation to ensure data integrity.
   */
  void setSVM(bool is_svm) { svm_allocation = is_svm; }

  /**
   * @brief  Set host-addressability (private -- MemoryPool stamps it at bind,
   *         mirroring setSVM).
   */
  void setHostAddressable(bool v) { host_addressable = v; }

  bool valid;
  unsigned int id;
  void *address;
  MemoryDataValidateCallback validate_cb;
  MemoryDataValidateCallback invalidate_cb;
  bool svm_allocation;
  bool host_addressable = true;
  bool device_valid; /**< device residency: freshest copy is in device_mem */
  void *device_mem;  /**< resident device buffer (non-owning, void*) */
  ResidencyClass residency_; /**< static residency class (planner decision) */
};

} // namespace nntrainer

#endif /* __MEMORY_DATA_H__ */
