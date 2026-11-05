// SPDX-License-Identifier: Apache-2.0
/** @file kv_page_pool.h
 * @brief Thread-confined host page pool shared by synchronous KV sessions.
 * @author Codex (OpenAI)
 * @bug None known
 */
#ifndef NNTRAINER_KV_PAGE_POOL_H
#define NNTRAINER_KV_PAGE_POOL_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <kv_cache/kv_storage.h>
#include <map>
#include <memory>
#include <utility>
#include <vector>

namespace nntrainer::kv {

/** Every acquired page is a handle whose last copy (session table or exported
 * view) returns the allocation to a per-size idle list, so acquisition and
 * release are O(1) instead of scanning every pooled page. No device completion
 * is implied. The optional budget includes idle pages and 64-byte allocation
 * rounding; when it would be exceeded, idle pages are freed first, then
 * registered reclaimers (for example a prefix cache) are asked to release
 * pages they alone hold, before failing.
 * Handles may outlive the pool; their allocations are then freed directly.
 * Page handles must not be bridged into Tensors through KvStorage::memory().
 */
class KvPagePool {
public:
  /** Releases at least one page handle and returns true, or returns false
   * when it has nothing left to release. It must not acquire pages. */
  using Reclaimer = std::function<bool()>;

  explicit KvPagePool(
    size_t budget_bytes = 0,
    std::shared_ptr<MemAllocator> allocator = std::make_shared<MemAllocator>());
  std::shared_ptr<KvStorage> acquire(size_t bytes);
  size_t reservedBytes() const;
  size_t budgetBytes() const;
  /** Free unused pages, for example before admitting a different geometry. */
  void trim();
  /** Best-effort release of idle and reclaimable pages to a byte target.
   * Pinned session/view allocations can prevent reaching the target. */
  void reclaimTo(size_t target_bytes);
  /** Register a budget-pressure hook; returns a token for removeReclaimer(). */
  uint64_t addReclaimer(Reclaimer reclaimer);
  void removeReclaimer(uint64_t token) noexcept;

private:
  struct Core {
    std::shared_ptr<MemAllocator> allocator;
    std::map<size_t, std::vector<std::shared_ptr<KvStorage>>> idle;
    std::vector<std::pair<uint64_t, Reclaimer>> reclaimers;
    uint64_t next_reclaimer = 1;
    size_t budget_bytes = 0;
    size_t reserved_bytes = 0;
    void trim() noexcept;
    std::shared_ptr<KvStorage> takeIdle(size_t bytes);
    bool reclaim() noexcept;
  };
  std::shared_ptr<Core> core_;
};

} // namespace nntrainer::kv
#endif
