// SPDX-License-Identifier: Apache-2.0
/**
 * @file kv_storage.cpp
 * @brief Own aligned CPU allocations shared by cache views and Tensor aliases.
 * @author Codex (OpenAI)
 * @bug Device-visible allocators and asynchronous access are unsupported.
 */
#include <kv_cache/kv_cache_spec.h>
#include <kv_cache/kv_storage.h>

#include <atomic>
#include <cstring>
#include <utility>

namespace nntrainer::kv {

KvStorage::KvStorage(size_t bytes, std::shared_ptr<MemAllocator> allocator) :
  allocator_(std::move(allocator)), memory_(nullptr), bytes_(bytes) {
  if (!bytes || !allocator_ || !allocator_->isHostAddressable() ||
      allocator_->isDeviceVisible() || allocator_->isSVM() ||
      allocator_->needsRegister())
    throw std::invalid_argument("KV storage requires a plain host allocator");
  checkedAdd(bytes, 63);
  void *ptr = nullptr;
  allocator_->alloc(&ptr, bytes, 64);
  if (!ptr)
    throw std::bad_alloc();
  allocation_ = ptr;
  memory_.setAddr(ptr);
  std::memset(ptr, 0, bytes);
  static std::atomic<uint64_t> next{1};
  identity_ = next.fetch_add(1);
}

std::shared_ptr<KvStorage>
KvStorage::create(size_t bytes, std::shared_ptr<MemAllocator> allocator) {
  return std::shared_ptr<KvStorage>(new KvStorage(bytes, std::move(allocator)));
}

KvStorage::~KvStorage() { allocator_->free(allocation_); }

std::shared_ptr<MemoryData> KvStorage::memory() {
  return std::shared_ptr<MemoryData>(shared_from_this(), &memory_);
}

} // namespace nntrainer::kv
