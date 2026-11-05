// SPDX-License-Identifier: Apache-2.0
/**
 * @file kv_prefix_cache.cpp
 * @brief Compressed token radix tree with LRU eviction of KV pages.
 * @author SeungBaek <sb92.hong@samsung.com>
 * @bug Thread confined like KvPagePool; no asynchronous device access.
 */
#include <kv_cache/kv_prefix_cache.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <list>
#include <map>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace nntrainer::kv {

namespace {

namespace fs = std::filesystem;
constexpr uint64_t flash_magic = 0x4e4e54524b563031ull;
using FlashHeader = std::array<uint64_t, 4>; // magic, block id, bytes, checksum

uint64_t checksum(uint64_t hash, const void *data, size_t bytes) {
  const auto *p = static_cast<const unsigned char *>(data);
  for (size_t i = 0; i < bytes; ++i)
    hash = (hash ^ p[i]) * 1099511628211ull;
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
  /** Edges hold compressed token runs. Only full-page boundaries own blocks;
   * interior branch nodes index tokens without owning or copying KV pages.
   * Node addresses stay stable through splits and merges for Block::node.
   */
  struct RadixNode {
    std::vector<TokenId> edge;
    std::map<TokenId, std::unique_ptr<RadixNode>> children;
    RadixNode *parent = nullptr;
    uint64_t block = 0;
  };

  struct Block {
    uint64_t parent = 0;
    RadixNode *node = nullptr;
    LayerPages keys, values;
    size_t children = 0;
    size_t flash_bytes = 0;
    bool temporary_file = false;
    bool protected_block = false; // A match/load is in progress.
    std::list<uint64_t>::iterator lru;
  };

  bool bound = false;
  size_t page_tokens = 0;
  std::vector<LayerSpec> layers;
  size_t max_blocks = 0;
  uint64_t next_id = 1;
  uint64_t epoch = 1;
  std::unordered_map<uint64_t, Block> blocks;
  RadixNode root;
  size_t radix_nodes = 0;  /**< Excludes the root and includes branch nodes. */
  std::list<uint64_t> lru; /**< Front is the most recently used block. */
  PrefixCacheStats stats;
  FlashCacheOptions flash;
  fs::path directory;
  size_t resident_blocks = 0;
  size_t block_bytes = 0; // Rounded pool bytes for all K/V owner pages.
  size_t file_bytes = 0;

  ~Core() {
    if (!directory.empty()) {
      std::error_code error;
      fs::remove_all(directory, error);
    }
  }

  fs::path path(uint64_t id) const {
    return directory / (std::to_string(id) + ".kv");
  }

  bool resident(const Block &block) const { return !block.keys.empty(); }

  void removeFile(uint64_t id, Block &block) noexcept {
    if (!block.flash_bytes)
      return;
    try {
      std::error_code error;
      auto file = path(id);
      if (block.temporary_file)
        file += ".tmp";
      fs::remove(file, error);
      if (error == std::errc::not_a_directory ||
          error == std::errc::no_such_file_or_directory)
        error
          .clear(); // No file can exist below an absent/non-directory parent.
      if (error) {
        ++stats.flash_errors;
        return; // Keep accounting until a later retry succeeds.
      }
      stats.flash_bytes -= block.flash_bytes;
      block.flash_bytes = 0;
      block.temporary_file = false;
    } catch (...) {
      ++stats.flash_errors;
    }
  }

  /** Make room using cold LRU leaves, preserving token-chain reachability. */
  bool makeDiskRoom(size_t bytes) {
    if (bytes > flash.budget_bytes)
      return false;
    while (stats.flash_bytes > flash.budget_bytes - bytes) {
      uint64_t victim = 0;
      for (auto it = lru.rbegin(); it != lru.rend(); ++it) {
        const auto &block = blocks.at(*it);
        if (block.flash_bytes && !resident(block) && !block.children &&
            !block.protected_block) {
          victim = *it;
          break;
        }
      }
      if (!victim)
        return false;
      auto &block = blocks.at(victim);
      removeFile(victim, block);
      if (block.flash_bytes)
        return false;
      erase(victim);
    }
    return true;
  }

  /** Stream directly from pages: no full-block staging allocation. */
  bool spill(uint64_t id) {
    auto &block = blocks.at(id);
    if (!flash.budget_bytes || !makeDiskRoom(file_bytes))
      return false;
    const auto destination = path(id);
    auto temporary = destination;
    temporary += ".tmp";
    // Charge the entire prospective file before writing, including failed
    // temporary files whose removal might also fail (e.g. device removal).
    block.flash_bytes = file_bytes;
    block.temporary_file = true;
    stats.flash_bytes += file_bytes;
    try {
      std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
      out.exceptions(std::ios::failbit | std::ios::badbit);
      FlashHeader header{flash_magic, id, file_bytes, 14695981039346656037ull};
      out.write(reinterpret_cast<const char *>(header.data()), sizeof(header));
      for (const auto *pages : {&block.keys, &block.values})
        for (const auto &page : *pages)
          if (page) {
            header[3] = checksum(header[3], page->data(), page->bytes());
            out.write(static_cast<const char *>(page->data()), page->bytes());
          }
      out.seekp(0);
      out.write(reinterpret_cast<const char *>(header.data()), sizeof(header));
      out.close();
      fs::rename(temporary, destination);
      block.temporary_file = false;
      ++stats.flash_writes;
      block.keys.clear();
      block.values.clear();
      --resident_blocks;
      return true;
    } catch (...) {
      ++stats.flash_errors;
      removeFile(id, block);
      return false;
    }
  }

  /** Load transactionally. Reclaimers skip this block and matched ancestors. */
  bool load(uint64_t id, KvPagePool &pool) {
    auto &block = blocks.at(id);
    if (resident(block))
      return true;
    try {
      std::ifstream in(path(id), std::ios::binary);
      in.exceptions(std::ios::failbit | std::ios::badbit);
      FlashHeader header{};
      in.read(reinterpret_cast<char *>(header.data()), sizeof(header));
      if (header[0] != flash_magic || header[1] != id ||
          header[2] != file_bytes || fs::file_size(path(id)) != file_bytes)
        throw std::runtime_error("Invalid flash KV header");
      // Free a whole block's worth before the first allocation. Temporary
      // page handles pin each successful allocation until the load commits.
      if (pool.budgetBytes())
        pool.reclaimTo(pool.budgetBytes() - block_bytes);
      LayerPages keys(layers.size()), values(layers.size());
      uint64_t hash = 14695981039346656037ull;
      for (auto *pages : {&keys, &values})
        for (size_t i = 0; i < layers.size(); ++i) {
          if (layers[i].source >= 0)
            continue;
          const size_t bytes =
            checkedMultiply(page_tokens, layers[i].rowBytes());
          auto page = pool.acquire(bytes);
          in.read(static_cast<char *>(page->data()), bytes);
          hash = checksum(hash, page->data(), bytes);
          (*pages)[i] = std::move(page);
        }
      if (hash != header[3])
        throw std::runtime_error("Invalid flash KV checksum");
      // Keep the cold copy accounted if removing it fails. It can be retried
      // later; never overwrite an unaccounted file during another spill.
      removeFile(id, block);
      block.keys.swap(keys);
      block.values.swap(values);
      ++resident_blocks;
      ++stats.flash_hits;
      return true;
    } catch (const std::bad_alloc &) {
      return false; // Capacity miss; keep a valid cold block for a later hit.
    } catch (...) {
      ++stats.flash_errors;
      // Forget the corrupt chain so recomputation can publish healthy pages.
      // Existing sessions retain their own handles; epoch invalidates ids.
      try {
        std::vector<uint64_t> descendants{id};
        for (size_t i = 0; i < descendants.size(); ++i)
          for (const auto &entry : blocks)
            if (entry.second.parent == descendants[i])
              descendants.push_back(entry.first);
        for (auto it = descendants.rbegin(); it != descendants.rend(); ++it)
          if (!blocks.at(*it).children)
            erase(*it);
      } catch (...) {
        // A cleanup failure must not expose any bytes from this load.
      }
      return false; // Corrupt/missing cache data is never used by attention.
    }
  }

  uint64_t find(uint64_t parent, const TokenId *tokens) const {
    const RadixNode *node = parent ? blocks.at(parent).node : &root;
    size_t at = 0;
    while (at < page_tokens) {
      const auto child = node->children.find(tokens[at]);
      if (child == node->children.end())
        return 0;
      node = child->second.get();
      if (node->edge.size() > page_tokens - at ||
          !std::equal(node->edge.begin(), node->edge.end(), tokens + at))
        return 0;
      at += node->edge.size();
    }
    return node->block;
  }

  /** Prepare every allocation before linking a new edge or splitting one.
   * Failure leaves the existing tree and its block pointers unchanged.
   */
  RadixNode *insertIndex(uint64_t parent, const TokenId *tokens, uint64_t id) {
    RadixNode *node = parent ? blocks.at(parent).node : &root;
    size_t at = 0;
    while (at < page_tokens) {
      auto found = node->children.find(tokens[at]);
      if (found == node->children.end()) {
        auto leaf = std::make_unique<RadixNode>();
        leaf->edge.assign(tokens + at, tokens + page_tokens);
        leaf->parent = node;
        leaf->block = id;
        auto *result = leaf.get();
        node->children.emplace(tokens[at], std::move(leaf));
        ++radix_nodes;
        return result;
      }
      auto &child = found->second;
      size_t common = 0;
      while (common < child->edge.size() && at + common < page_tokens &&
             child->edge[common] == tokens[at + common])
        ++common;
      if (common == child->edge.size()) {
        at += common;
        node = child.get();
        continue;
      }

      auto branch = std::make_unique<RadixNode>();
      branch->edge.assign(child->edge.begin(), child->edge.begin() + common);
      branch->parent = node;
      std::vector<TokenId> suffix(child->edge.begin() + common,
                                  child->edge.end());
      // Reserve the old child's slot before mutating its edge or ownership.
      auto old = branch->children.emplace(suffix.front(), nullptr).first;
      RadixNode *result = branch.get();
      size_t added = 1;
      if (at + common == page_tokens) {
        branch->block = id;
      } else {
        auto leaf = std::make_unique<RadixNode>();
        leaf->edge.assign(tokens + at + common, tokens + page_tokens);
        leaf->parent = branch.get();
        leaf->block = id;
        result = leaf.get();
        branch->children.emplace(leaf->edge.front(), std::move(leaf));
        ++added;
      }
      child->edge.swap(suffix);
      child->parent = branch.get();
      old->second = std::move(child);
      child = std::move(branch);
      radix_nodes += added;
      return result;
    }
    node->block = id;
    return node;
  }

  /** Remove a terminal leaf and recompress token-only unary ancestors.
   * If allocating a merged label fails, the unary node remains searchable.
   * Reclamation itself must still release KV pages under memory pressure.
   */
  void eraseIndex(RadixNode *node) noexcept {
    node->block = 0;
    while (node != &root && !node->block) {
      auto *parent = node->parent;
      const TokenId first = node->edge.front();
      if (node->children.empty()) {
        parent->children.erase(first);
        --radix_nodes;
        node = parent;
        continue;
      }
      if (node->children.size() == 1) {
        auto &child = node->children.begin()->second;
        try {
          auto edge = node->edge;
          edge.insert(edge.end(), child->edge.begin(), child->edge.end());
          child->edge.swap(edge);
        } catch (...) {
          return;
        }
        child->parent = parent;
        auto replacement = std::move(child);
        parent->children.find(first)->second = std::move(replacement);
        --radix_nodes;
      }
      return;
    }
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
    removeFile(id, found->second);
    if (found->second.flash_bytes)
      return; // Cannot forget a file still charged to the disk budget.
    if (resident(found->second))
      --resident_blocks;
    eraseIndex(found->second.node);
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
  bool evictOne(bool allow_spill = true) {
    // Demotion can include a parent of cold children: its metadata remains.
    if (allow_spill && flash.budget_bytes)
      for (uint64_t id : std::vector<uint64_t>(lru.rbegin(), lru.rend())) {
        auto found = blocks.find(id);
        if (found == blocks.end())
          continue;
        auto &block = found->second;
        if (resident(block) && !block.protected_block && unused(block)) {
          if (block.flash_bytes) {
            removeFile(id, block);
            if (block.flash_bytes)
              continue;
          }
          if (spill(id))
            return true;
        }
      }
    for (auto it = lru.rbegin(); it != lru.rend(); ++it) {
      const auto found = blocks.find(*it);
      if (found == blocks.end() || found->second.children ||
          found->second.protected_block ||
          (allow_spill && !resident(found->second)) || !unused(found->second))
        continue;
      const auto before = blocks.size();
      erase(found->first);
      return blocks.size() != before;
    }
    return false;
  }
};

KvPrefixCache::KvPrefixCache(std::shared_ptr<KvPagePool> pool,
                             size_t max_blocks, FlashCacheOptions flash) :
  core_(std::make_shared<Core>()), pool_(std::move(pool)) {
  if (!pool_)
    throw std::invalid_argument("KV prefix cache requires a page pool");
  core_->max_blocks = max_blocks;
  core_->flash = std::move(flash);
  if (core_->flash.budget_bytes) {
    if (core_->flash.directory.empty() || !pool_->budgetBytes())
      throw std::invalid_argument(
        "Flash KV requires a directory and finite DRAM budget");
    fs::create_directories(core_->flash.directory);
    std::random_device random;
    for (size_t attempt = 0; attempt < 32; ++attempt) {
      auto dir = fs::path(core_->flash.directory) /
                 ("nntr-kv-" + std::to_string(random()) + "-" +
                  std::to_string(random()));
      if (fs::create_directory(dir)) {
        core_->directory = std::move(dir);
        fs::permissions(core_->directory, fs::perms::owner_all,
                        fs::perm_options::replace);
        break;
      }
    }
    if (core_->directory.empty())
      throw std::runtime_error("Cannot create private flash KV directory");
  } else if (core_->flash.reserve_bytes) {
    throw std::invalid_argument("Flash KV reserve requires the flash tier");
  }
  std::weak_ptr<Core> weak = core_;
  reclaimer_ = pool_->addReclaimer([weak]() {
    auto core = weak.lock();
    return core && core->evictOne();
  });
}

KvPrefixCache::~KvPrefixCache() { pool_->removeReclaimer(reclaimer_); }

size_t KvPrefixCache::maxBlocks() const { return core_->max_blocks; }

bool KvPrefixCache::flashEnabled() const {
  return core_->flash.budget_bytes != 0;
}

PrefixCacheStats KvPrefixCache::stats() const {
  auto result = core_->stats;
  result.blocks = core_->blocks.size();
  result.resident_blocks = core_->resident_blocks;
  result.radix_nodes = core_->radix_nodes;
  for (const auto &entry : core_->blocks)
    result.flash_blocks += entry.second.flash_bytes != 0;
  return result;
}

size_t KvPrefixCache::evictUnused() {
  size_t count = 0;
  while (core_->evictOne(false))
    ++count;
  return count;
}

void KvPrefixCache::trim() noexcept {
  auto &core = *core_;
  try {
    while (core.max_blocks && core.resident_blocks > core.max_blocks &&
           core.evictOne()) {
    }
  } catch (...) {
    // Trimming is optional; a completed generation/session release stays valid.
  }
  if (core.flash.budget_bytes && core.bound)
    pool_->reclaimTo(pool_->budgetBytes() - core.flash.reserve_bytes);
}

void KvPrefixCache::clear() {
  auto &core = *core_;
  for (auto &entry : core.blocks)
    core.removeFile(entry.first, entry.second);
  // Removal failure must retain budget accounting and metadata for retry.
  for (const auto &entry : core.blocks)
    if (entry.second.flash_bytes)
      throw std::runtime_error("Cannot remove Flash KV cache files");
  core.stats.evicted_blocks += core.blocks.size();
  core.root.children.clear();
  core.radix_nodes = 0;
  core.lru.clear();
  core.blocks.clear();
  core.resident_blocks = 0;
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
    size_t block_bytes = 0, payload = 0;
    for (const auto &layer : spec.layers) {
      if (layer.source >= 0)
        continue;
      const size_t bytes = checkedMultiply(spec.page_tokens, layer.rowBytes());
      payload = checkedAdd(payload, checkedMultiply(2, bytes));
      block_bytes = checkedAdd(
        block_bytes, checkedMultiply(2, checkedAdd(bytes, 63) & ~size_t{63}));
    }
    if (core.flash.budget_bytes) {
      const size_t reserve =
        core.flash.reserve_bytes ? core.flash.reserve_bytes : block_bytes;
      if (block_bytes > pool_->budgetBytes() ||
          reserve > pool_->budgetBytes() - block_bytes)
        throw std::invalid_argument(
          "Flash KV DRAM budget must fit a block plus restore reserve");
      core.flash.reserve_bytes = reserve;
    }
    core.block_bytes = block_bytes;
    core.file_bytes = checkedAdd(payload, sizeof(FlashHeader));
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
  uint64_t current = 0;
  struct Unprotect {
    Core &core;
    const std::vector<uint64_t> &ids;
    const uint64_t &current;
    ~Unprotect() {
      for (uint64_t id : ids) {
        auto found = core.blocks.find(id);
        if (found != core.blocks.end())
          found->second.protected_block = false;
      }
      auto found = core.blocks.find(current);
      if (found != core.blocks.end())
        found->second.protected_block = false;
    }
  } unprotect{core, ids, current};
  uint64_t parent = 0;
  for (size_t at = 0; count - at >= page_tokens; at += page_tokens) {
    const uint64_t id = core.find(parent, tokens + at);
    if (!id)
      break;
    current = id;
    auto &block = core.blocks.at(id);
    block.protected_block = true;
    if (!core.load(id, *pool_))
      break;
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
  while (core.max_blocks && core.resident_blocks >= core.max_blocks) {
    if (!core.evictOne())
      return 0;
    if (parent && !core.blocks.count(parent))
      return 0;
  }

  Core::Block block;
  block.parent = parent;
  block.keys = keys;
  block.values = values;
  const uint64_t id = core.next_id;
  // Link all three containers or none of them.
  core.lru.push_front(id);
  try {
    block.lru = core.lru.begin();
    auto placed = core.blocks.emplace(id, std::move(block)).first;
    try {
      placed->second.node = core.insertIndex(parent, tokens, id);
    } catch (...) {
      core.blocks.erase(placed);
      throw;
    }
  } catch (...) {
    core.lru.pop_front();
    throw;
  }
  ++core.next_id;
  ++core.resident_blocks;
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
