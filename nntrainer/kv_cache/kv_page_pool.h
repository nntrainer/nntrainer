// SPDX-License-Identifier: Apache-2.0
/** @file kv_page_pool.h
 * @brief Thread-confined host page pool shared by synchronous KV sessions.
 * @author Codex (OpenAI)
 * @bug None known
 */
#ifndef NNTRAINER_KV_PAGE_POOL_H
#define NNTRAINER_KV_PAGE_POOL_H

#include <cstddef>
#include <kv_cache/kv_storage.h>
#include <map>
#include <memory>
#include <vector>

namespace nntrainer::kv {

/** Every acquired page is a handle whose last copy (session table or exported
 * view) returns the allocation to a per-size idle list, so acquisition and
 * release are O(1) instead of scanning every pooled page. No device completion
 * is implied. The optional budget includes idle pages and 64-byte allocation
 * rounding; when it would be exceeded, idle pages are freed before failing.
 * Handles may outlive the pool; their allocations are then freed directly.
 * Page handles must not be bridged into Tensors through KvStorage::memory().
 */
class KvPagePool {
public:
  explicit KvPagePool(
    size_t budget_bytes = 0,
    std::shared_ptr<MemAllocator> allocator = std::make_shared<MemAllocator>());
  std::shared_ptr<KvStorage> acquire(size_t bytes);
  size_t reservedBytes() const;
  /** Free unused pages, for example before admitting a different geometry. */
  void trim();

private:
  struct Core {
    std::shared_ptr<MemAllocator> allocator;
    std::map<size_t, std::vector<std::shared_ptr<KvStorage>>> idle;
    size_t budget_bytes = 0;
    size_t reserved_bytes = 0;
    void trim() noexcept;
  };
  std::shared_ptr<Core> core_;
};

} // namespace nntrainer::kv
#endif
