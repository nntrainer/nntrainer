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
 * @brief  Where a tensor's backing memory lives.
 *
 * @details Decided once, at allocation, by the memory planner, and therefore a
 * property of the tensor rather than of an edge between two layers. The
 * planner already knows every tensor's producer, its consumers and its
 * lifetime, so it can place a tensor and have all of them agree; a decision
 * made per edge at execution time cannot, and leaves a producer writing one
 * plane while a consumer reads another.
 *
 *   - HOST: host memory. The only class a CPU backend produces.
 *   - SVM: shared virtual memory, addressable by both the host and the device.
 *   - GPU_CLMEM: device memory, addressable only by the device. A layer binds
 *     it as a buffer kernel argument; see Tensor::isClMem / Tensor::getClMem.
 */
enum class ResidencyClass : unsigned char {
  HOST = 0,      /**< host memory */
  SVM = 1,       /**< shared virtual memory (host and device addressable) */
  GPU_CLMEM = 2, /**< device memory (device addressable only) */
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

  /**
   * @brief TensorPool is granted friend access to call setResidency()
   * @details Where a tensor lives is the memory planner's decision, taken
   *          once at allocation; TensorPool is the only place that runs the
   *          planner, so it is the only place allowed to record the result.
   */
  friend class TensorPool;

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
   * @brief   The residency class the planner assigned this memory.
   */
  ResidencyClass residency() const { return residency_; }

  /**
   * @brief   True if this memory lives in device memory the host cannot
   *          address, so a layer has to bind it as a buffer argument rather
   *          than pass a pointer. Host pointer arithmetic on it is a bug by
   *          construction.
   */
  bool isClMem() const { return residency_ == ResidencyClass::GPU_CLMEM; }

  /**
   * @brief   The device buffer backing this memory, or null when the planner
   *          left it on the shared plane.
   * @note    Held as void* so this header stays free of the OpenCL types and
   *          keeps compiling in a CPU-only build. Non-owning: the pool that
   *          handed it out owns it.
   */
  void *deviceMem() const { return device_mem; }

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
   * @brief  Record the planner's placement and, for a device-resident tensor,
   *         the buffer that backs it (private - only the pool decides).
   */
  void setResidency(ResidencyClass r, void *dev = nullptr) {
    residency_ = r;
    device_mem = dev;
  }

  bool valid;
  unsigned int id;
  void *address;
  MemoryDataValidateCallback validate_cb;
  MemoryDataValidateCallback invalidate_cb;
  bool svm_allocation;
  void *device_mem;          /**< device buffer, when device-resident */
  ResidencyClass residency_; /**< the planner's placement */
};

} // namespace nntrainer

#endif /* __MEMORY_DATA_H__ */
