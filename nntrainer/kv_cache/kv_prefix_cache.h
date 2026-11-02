// SPDX-License-Identifier: Apache-2.0
/** @file kv_prefix_cache.h
 * @brief Token-addressed full KV pages shared between paged KV sessions.
 * @author SeungBaek <sb92.hong@samsung.com>
 * @bug Thread confined like KvPagePool; no asynchronous device access.
 */
#ifndef NNTRAINER_KV_PREFIX_CACHE_H
#define NNTRAINER_KV_PREFIX_CACHE_H

#include <kv_cache/kv_cache_spec.h>
#include <kv_cache/kv_page_pool.h>
#include <kv_cache/kv_storage.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace nntrainer::kv {

/** Token id recorded per committed KV row for prefix matching. */
using TokenId = uint32_t;

struct PrefixCacheStats {
  size_t blocks = 0;          /**< Cached full blocks right now. */
  size_t lookups = 0;         /**< Prefix lookups that consulted the cache. */
  size_t hit_blocks = 0;      /**< Blocks adopted by sessions. */
  size_t inserted_blocks = 0; /**< Blocks published by sessions. */
  size_t evicted_blocks = 0;  /**< Blocks dropped by LRU, budget or clear. */
};

/** Content-addressed cache of full KV pages, keyed by the token chain that
 * produced them.
 *
 * A block covers page_tokens consecutive positions [i * page_tokens,
 * (i + 1) * page_tokens) of a sequence and holds one page per owner layer for
 * both K and V. Blocks form a tree: a block is identified by its parent block
 * and its own token ids, which are compared exactly, so a match is never a
 * hash collision. Only full pages are cached; a partial tail stays private to
 * its session. A cached page is never written again: a session that has to
 * write into a shared page copies it first.
 *
 * The cache holds page handles from one KvPagePool. Under pool budget pressure
 * (and when max_blocks is reached) it drops least recently used leaf blocks
 * that no session references, returning their pages to the pool. Blocks a
 * session still uses are kept; dropping them would not free memory.
 *
 * The first KvCacheManager attached to the cache binds its page layout; later
 * managers must use the same layout. The layout check cannot see weights: one
 * cache must only be shared by sessions of the same model. Prefix caching
 * requires batch 1, because block identity is a single token sequence.
 * Thread confined: all users must run on one thread or be externally
 * serialized, like the page pool.
 */
class KvPrefixCache {
public:
  /** @param max_blocks 0 bounds the cache only by the pool budget. */
  explicit KvPrefixCache(std::shared_ptr<KvPagePool> pool,
                         size_t max_blocks = 0);
  ~KvPrefixCache();
  KvPrefixCache(const KvPrefixCache &) = delete;
  KvPrefixCache &operator=(const KvPrefixCache &) = delete;

  const std::shared_ptr<KvPagePool> &pool() const { return pool_; }
  size_t maxBlocks() const;
  PrefixCacheStats stats() const;
  /** Drop every block that no session references; returns the count. */
  size_t evictUnused();
  /** Forget all blocks. Sessions keep the pages they already reference. */
  void clear();

private:
  friend class KvCacheManager;
  struct Core;

  /** Per-layer page handles of one block; null for alias layers. */
  using LayerPages = std::vector<std::shared_ptr<KvStorage>>;

  /** Bind the page layout on first use, or reject a different one. */
  void bind(const CacheSpec &spec);
  /** Longest cached chain of full blocks covering a prefix of tokens. */
  size_t match(const TokenId *tokens, size_t count, std::vector<uint64_t> &ids,
               std::vector<LayerPages> &keys, std::vector<LayerPages> &values);
  /** Insert the block after parent (0 for the root), or return the existing
   * equal block. Returns 0 when the parent is gone or the cache is full. */
  uint64_t insert(uint64_t parent, const TokenId *tokens,
                  const LayerPages &keys, const LayerPages &values);
  bool contains(uint64_t id) const;
  /** Changes whenever blocks are dropped, so sessions revalidate their ids. */
  uint64_t epoch() const;
  void recordHit(size_t blocks);

  std::shared_ptr<Core> core_;
  std::shared_ptr<KvPagePool> pool_;
  uint64_t reclaimer_ = 0;
};

} // namespace nntrainer::kv
#endif
