// SPDX-License-Identifier: Apache-2.0
/** @file kv_storage.h
 * @brief CPU allocation ownership that survives every exported Tensor alias.
 * @author Codex (OpenAI)
 * @bug Device-visible allocators and asynchronous access are unsupported.
 */
#ifndef NNTRAINER_KV_STORAGE_H
#define NNTRAINER_KV_STORAGE_H

#include <mem_allocator.h>
#include <memory_data.h>

#include <cstdint>
#include <memory>

namespace nntrainer::kv {

/** One aligned host allocation, using the existing allocator abstraction.
 * Device allocators are rejected until an allocation-result adapter exists.
 */
class KvStorage : public std::enable_shared_from_this<KvStorage> {
public:
  static std::shared_ptr<KvStorage>
  create(size_t bytes, std::shared_ptr<MemAllocator> allocator);
  ~KvStorage();
  KvStorage(const KvStorage &) = delete;
  KvStorage &operator=(const KvStorage &) = delete;

  size_t bytes() const { return bytes_; }
  uint64_t identity() const { return identity_; }
  /** Stable allocation base, independent of exported wrapper metadata. */
  void *data() const { return allocation_; }
  /** Aliasing shared_ptr: the MemoryData itself owns no allocation.
   * Its address and metadata must not be changed by consumers. Rebind a Tensor
   * with Tensor::setData(), rather than mutating this shared wrapper.
   */
  std::shared_ptr<MemoryData> memory();

private:
  KvStorage(size_t bytes, std::shared_ptr<MemAllocator> allocator);
  std::shared_ptr<MemAllocator> allocator_;
  void *allocation_ = nullptr; /**< Immutable after successful allocation. */
  MemoryData memory_;
  size_t bytes_;
  uint64_t identity_;
};

} // namespace nntrainer::kv
#endif
