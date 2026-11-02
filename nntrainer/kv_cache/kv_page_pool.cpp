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
  std::shared_ptr<KvStorage> page = core.takeIdle(bytes);
  if (!page) {
    const size_t aligned = checkedAdd(bytes, 63) & ~size_t{63};
    auto fits = [&core, aligned] {
      return !core.budget_bytes ||
             checkedAdd(core.reserved_bytes, aligned) <= core.budget_bytes;
    };
    if (!fits())
      core.trim(); // Idle pages of other geometries must not block admission.
    // Reclaimers release pages only they reference (e.g. cached prefixes).
    // A released page of this size is reused; any other size is freed.
    while (!fits()) {
      if (!core.reclaim())
        throw std::bad_alloc();
      if ((page = core.takeIdle(bytes)))
        break;
      core.trim();
    }
    if (!page) {
      const size_t next = checkedAdd(core.reserved_bytes, aligned);
      page = KvStorage::create(bytes, core.allocator);
      core.reserved_bytes = next;
    }
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

size_t KvPagePool::budgetBytes() const { return core_->budget_bytes; }

void KvPagePool::trim() { core_->trim(); }

uint64_t KvPagePool::addReclaimer(Reclaimer reclaimer) {
  if (!reclaimer)
    throw std::invalid_argument("KV page reclaimer must be callable");
  const uint64_t token = core_->next_reclaimer++;
  core_->reclaimers.emplace_back(token, std::move(reclaimer));
  return token;
}

void KvPagePool::removeReclaimer(uint64_t token) noexcept {
  auto &list = core_->reclaimers;
  for (auto it = list.begin(); it != list.end(); ++it)
    if (it->first == token) {
      list.erase(it);
      return;
    }
}

std::shared_ptr<KvStorage> KvPagePool::Core::takeIdle(size_t bytes) {
  auto bucket = idle.find(bytes);
  if (bucket == idle.end() || bucket->second.empty())
    return nullptr;
  auto page = std::move(bucket->second.back());
  bucket->second.pop_back();
  std::memset(page->data(), 0, bytes);
  return page;
}

bool KvPagePool::Core::reclaim() noexcept {
  // Copy the callables: a reclaimer may unregister itself or another one.
  std::vector<Reclaimer> hooks;
  try {
    hooks.reserve(reclaimers.size());
    for (const auto &entry : reclaimers)
      hooks.push_back(entry.second);
  } catch (...) {
    return false;
  }
  for (auto &hook : hooks) {
    try {
      if (hook())
        return true;
    } catch (...) {
      // A failing reclaimer only means nothing was released by it.
    }
  }
  return false;
}

void KvPagePool::Core::trim() noexcept {
  for (auto &bucket : idle) {
    reserved_bytes -= bucket.second.size() * roundedBytes(bucket.first);
    bucket.second.clear();
  }
  idle.clear();
}

} // namespace nntrainer::kv
