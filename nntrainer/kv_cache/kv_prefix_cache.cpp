// SPDX-License-Identifier: Apache-2.0
/**
 * @file kv_prefix_cache.cpp
 * @brief Token-chain block index with LRU eviction of unreferenced pages.
 * @author SeungBaek <sb92.hong@samsung.com>
 * @bug Thread confined like KvPagePool; no asynchronous device access.
 */
#include <kv_cache/kv_prefix_cache.h>

#include <algorithm>
#include <list>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace nntrainer::kv {

namespace {

uint64_t splitmix(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

/** Bucket key only; matches always compare the parent and tokens exactly. */
uint64_t blockDigest(uint64_t parent, const TokenId *tokens, size_t count) {
  uint64_t hash = splitmix(parent);
  for (size_t i = 0; i < count; ++i)
    hash = splitmix(hash ^ tokens[i]);
  return hash;
}

bool sameLayout(const LayerSpec &a, const LayerSpec &b) {
  return a.batch == b.batch && a.num_kv_heads == b.num_kv_heads &&
         a.head_dim == b.head_dim && a.encoding == b.encoding &&
         a.storage_type == b.storage_type && a.source == b.source &&
         a.has_producer == b.has_producer;
}

} // namespace

struct KvPrefixCache::Core {
  struct Block {
    uint64_t parent = 0;
    uint64_t digest = 0;
    std::vector<TokenId> tokens;
    LayerPages keys, values;
    size_t children = 0;
    std::list<uint64_t>::iterator lru;
  };

  bool bound = false;
  size_t page_tokens = 0;
  std::vector<LayerSpec> layers;
  size_t max_blocks = 0;
  uint64_t next_id = 1;
  uint64_t epoch = 1;
  std::unordered_map<uint64_t, Block> blocks;
  std::unordered_multimap<uint64_t, uint64_t> index;
  std::list<uint64_t> lru; /**< Front is the most recently used block. */
  PrefixCacheStats stats;

  uint64_t find(uint64_t parent, const TokenId *tokens) const {
    const auto range =
      index.equal_range(blockDigest(parent, tokens, page_tokens));
    for (auto it = range.first; it != range.second; ++it) {
      const auto found = blocks.find(it->second);
      if (found != blocks.end() && found->second.parent == parent &&
          std::equal(found->second.tokens.begin(), found->second.tokens.end(),
                     tokens))
        return it->second;
    }
    return 0;
  }

  void touch(uint64_t id) noexcept {
    auto found = blocks.find(id);
    if (found != blocks.end())
      lru.splice(lru.begin(), lru, found->second.lru);
  }

  /** True when the cache holds the only handle to every page of the block. */
  static bool unused(const Block &block) noexcept {
    for (const auto *pages : {&block.keys, &block.values})
      for (const auto &page : *pages)
        if (page && page.use_count() > 1)
          return false;
    return true;
  }

  void erase(uint64_t id) noexcept {
    auto found = blocks.find(id);
    if (found == blocks.end())
      return;
    auto range = index.equal_range(found->second.digest);
    for (auto it = range.first; it != range.second; ++it)
      if (it->second == id) {
        index.erase(it);
        break;
      }
    if (found->second.parent) {
      auto parent = blocks.find(found->second.parent);
      if (parent != blocks.end() && parent->second.children)
        --parent->second.children;
    }
    lru.erase(found->second.lru);
    blocks.erase(found); // Returns the pages to the pool's idle lists.
    ++epoch;
    ++stats.evicted_blocks;
  }

  /** Drop the least recently used leaf whose pages no session references.
   * Leaves go first, so every cached block stays reachable from the root. */
  bool evictOne() noexcept {
    for (auto it = lru.rbegin(); it != lru.rend(); ++it) {
      const auto found = blocks.find(*it);
      if (found == blocks.end() || found->second.children ||
          !unused(found->second))
        continue;
      erase(found->first);
      return true;
    }
    return false;
  }
};

KvPrefixCache::KvPrefixCache(std::shared_ptr<KvPagePool> pool,
                             size_t max_blocks) :
  core_(std::make_shared<Core>()), pool_(std::move(pool)) {
  if (!pool_)
    throw std::invalid_argument("KV prefix cache requires a page pool");
  core_->max_blocks = max_blocks;
  std::weak_ptr<Core> weak = core_;
  reclaimer_ = pool_->addReclaimer([weak]() {
    auto core = weak.lock();
    return core && core->evictOne();
  });
}

KvPrefixCache::~KvPrefixCache() { pool_->removeReclaimer(reclaimer_); }

size_t KvPrefixCache::maxBlocks() const { return core_->max_blocks; }

PrefixCacheStats KvPrefixCache::stats() const {
  auto result = core_->stats;
  result.blocks = core_->blocks.size();
  return result;
}

size_t KvPrefixCache::evictUnused() {
  size_t count = 0;
  while (core_->evictOne())
    ++count;
  return count;
}

void KvPrefixCache::clear() {
  auto &core = *core_;
  core.stats.evicted_blocks += core.blocks.size();
  core.index.clear();
  core.lru.clear();
  core.blocks.clear();
  ++core.epoch;
}

void KvPrefixCache::bind(const CacheSpec &spec) {
  if (!spec.page_tokens)
    throw std::invalid_argument("KV prefix cache requires paged geometry");
  for (const auto &layer : spec.layers)
    if (layer.batch != 1)
      throw std::invalid_argument("KV prefix cache requires batch-1 sessions");
  auto &core = *core_;
  if (!core.bound) {
    core.page_tokens = spec.page_tokens;
    core.layers = spec.layers;
    core.bound = true;
    return;
  }
  bool same = core.page_tokens == spec.page_tokens &&
              core.layers.size() == spec.layers.size();
  for (size_t i = 0; same && i < spec.layers.size(); ++i)
    same = sameLayout(core.layers[i], spec.layers[i]);
  if (!same)
    throw std::invalid_argument(
      "KV prefix cache is bound to a different page layout");
}

size_t KvPrefixCache::match(const TokenId *tokens, size_t count,
                            std::vector<uint64_t> &ids,
                            std::vector<LayerPages> &keys,
                            std::vector<LayerPages> &values) {
  ids.clear();
  keys.clear();
  values.clear();
  auto &core = *core_;
  if (!core.bound || !tokens || !count)
    return 0;
  ++core.stats.lookups;
  const size_t page_tokens = core.page_tokens;
  uint64_t parent = 0;
  for (size_t at = 0; count - at >= page_tokens; at += page_tokens) {
    const uint64_t id = core.find(parent, tokens + at);
    if (!id)
      break;
    const auto &block = core.blocks.at(id);
    ids.push_back(id);
    keys.push_back(block.keys);
    values.push_back(block.values);
    parent = id;
  }
  // Touch leaf first, so the shared root ends up most recently used.
  for (auto it = ids.rbegin(); it != ids.rend(); ++it)
    core.touch(*it);
  return ids.size() * page_tokens;
}

uint64_t KvPrefixCache::insert(uint64_t parent, const TokenId *tokens,
                               const LayerPages &keys,
                               const LayerPages &values) {
  auto &core = *core_;
  if (!core.bound || !tokens || keys.size() != core.layers.size() ||
      values.size() != keys.size())
    throw std::logic_error("KV prefix block does not match the bound layout");
  if (parent && !core.blocks.count(parent))
    return 0; // The chain was evicted; an orphan block would be unreachable.
  if (const uint64_t existing = core.find(parent, tokens)) {
    core.touch(existing);
    return existing;
  }
  if (core.max_blocks && core.blocks.size() >= core.max_blocks) {
    if (!core.evictOne())
      return 0;
    if (parent && !core.blocks.count(parent))
      return 0;
  }

  Core::Block block;
  block.parent = parent;
  block.tokens.assign(tokens, tokens + core.page_tokens);
  block.digest = blockDigest(parent, tokens, core.page_tokens);
  block.keys = keys;
  block.values = values;
  const uint64_t id = core.next_id;
  // Link all three containers or none of them.
  core.lru.push_front(id);
  try {
    block.lru = core.lru.begin();
    auto placed = core.blocks.emplace(id, std::move(block)).first;
    try {
      core.index.emplace(placed->second.digest, id);
    } catch (...) {
      core.blocks.erase(placed);
      throw;
    }
  } catch (...) {
    core.lru.pop_front();
    throw;
  }
  ++core.next_id;
  if (parent)
    ++core.blocks.at(parent).children;
  ++core.stats.inserted_blocks;
  return id;
}

bool KvPrefixCache::contains(uint64_t id) const {
  return core_->blocks.count(id) != 0;
}

uint64_t KvPrefixCache::epoch() const { return core_->epoch; }

void KvPrefixCache::recordHit(size_t blocks) {
  core_->stats.hit_blocks += blocks;
}

} // namespace nntrainer::kv
