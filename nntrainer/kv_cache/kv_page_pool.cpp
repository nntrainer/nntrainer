// SPDX-License-Identifier: Apache-2.0
/** @file kv_page_pool.cpp
 * @brief Budgeted page reuse with allocation ownership held by every view.
 * @author Codex (OpenAI)
 * @bug None known
 */
#include <cstring>
#include <kv_cache/kv_cache_spec.h>
#include <kv_cache/kv_page_pool.h>
#include <utility>

namespace nntrainer::kv {

namespace {
size_t roundedBytes(size_t bytes) { return (bytes + 63) & ~size_t{63}; }
} // namespace

KvPagePool::KvPagePool(size_t budget_bytes,
                       std::shared_ptr<MemAllocator> allocator) :
  core_(std::make_shared<Core>()) {
  if (!allocator || !allocator->isHostAddressable() ||
      allocator->isDeviceVisible() || allocator->isSVM() ||
      allocator->needsRegister())
    throw std::invalid_argument("KV page pool requires a plain host allocator");
  core_->allocator = std::move(allocator);
  core_->budget_bytes = budget_bytes;
}

std::shared_ptr<KvStorage> KvPagePool::acquire(size_t bytes) {
  if (!bytes)
    throw std::invalid_argument("KV page size must be positive");
  auto &core = *core_;
  std::shared_ptr<KvStorage> page;
  auto bucket = core.idle.find(bytes);
  if (bucket != core.idle.end() && !bucket->second.empty()) {
    page = std::move(bucket->second.back());
    bucket->second.pop_back();
    std::memset(page->data(), 0, bytes);
  } else {
    const size_t aligned = checkedAdd(bytes, 63) & ~size_t{63};
    if (core.budget_bytes &&
        checkedAdd(core.reserved_bytes, aligned) > core.budget_bytes)
      core.trim(); // Idle pages of other geometries must not block admission.
    const size_t next = checkedAdd(core.reserved_bytes, aligned);
    if (core.budget_bytes && next > core.budget_bytes)
      throw std::bad_alloc();
    page = KvStorage::create(bytes, core.allocator);
    core.reserved_bytes = next;
  }

  // The returned handle has its own control block: once the session table and
  // every exported view drop it, the owning reference goes back to the idle
  // list. The page stays out of the list while any handle copy is alive.
  KvStorage *raw = page.get();
  std::weak_ptr<Core> weak = core_;
  return std::shared_ptr<KvStorage>(
    raw, [weak, owner = std::move(page)](KvStorage *) mutable noexcept {
      auto pool = weak.lock();
      if (!pool)
        return; // The pool is gone; the allocation dies with this deleter.
      const size_t size = owner->bytes();
      try {
        pool->idle[size].push_back(std::move(owner));
      } catch (...) {
        pool->reserved_bytes -= roundedBytes(size);
        owner.reset();
      }
    });
}

size_t KvPagePool::reservedBytes() const { return core_->reserved_bytes; }

void KvPagePool::trim() { core_->trim(); }

void KvPagePool::Core::trim() noexcept {
  for (auto &bucket : idle) {
    reserved_bytes -= bucket.second.size() * roundedBytes(bucket.first);
    bucket.second.clear();
  }
  idle.clear();
}

} // namespace nntrainer::kv
