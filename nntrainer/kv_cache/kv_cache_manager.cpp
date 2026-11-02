// SPDX-License-Identifier: Apache-2.0
/**
 * @file kv_cache_manager.cpp
 * @brief Synchronous CPU KV sessions, views, publication and prefix reuse.
 * @author Codex (OpenAI)
 * @bug Sessions are thread confined; asynchronous device access is unsupported.
 */
#include <kv_cache/kv_cache_manager.h>

#include <algorithm>
#include <cstring>
#include <istream>
#include <ostream>
#include <utility>

namespace nntrainer::kv {

struct ViewEpoch {
  uint64_t generation = 1;
  uint64_t active_step = 0;
};

namespace {
size_t roundedPageBytes(size_t bytes) { return (bytes + 63) & ~size_t{63}; }
} // namespace

struct SessionState {
  CacheSpec spec;
  /** Contiguous planes, or the shared one-row graph identities when paged. */
  std::vector<std::shared_ptr<KvStorage>> keys, values;
  std::vector<std::vector<std::shared_ptr<KvStorage>>> key_pages, value_pages;
  std::shared_ptr<KvPagePool> pool;
  std::shared_ptr<KvPrefixCache> prefix;
  std::vector<bool> ready;
  std::shared_ptr<ViewEpoch> epoch = std::make_shared<ViewEpoch>();
  size_t position = 0;
  uint64_t next_step = 0;
  bool valid = true;
  bool overwrites = false;
  CacheStats stats;
  /** Known ids of committed rows; always a prefix of [0, position). */
  std::vector<TokenId> tokens;
  /** Ids of the live step, appended on commit when they extend tokens. */
  std::vector<TokenId> pending_tokens;
  bool pending_known = false;
  /** Prefix-cache block ids of the leading full pages, in page order. */
  std::vector<uint64_t> block_ids;
  /** Cache epoch at which block_ids were last checked against evictions. */
  uint64_t prefix_epoch = 0;
  /** Leading logical pages that the prefix cache or another session may also
   * reference. They are never written in place: a step that has to append to
   * one of them copies it first. */
  size_t shared_pages = 0;

  void forgetPrefix() noexcept {
    tokens.clear();
    pending_tokens.clear();
    pending_known = false;
    block_ids.clear();
    shared_pages = 0;
  }
  // Shrinking never allocates, so lease destruction can return uncommitted
  // pages.
  void releaseTail(size_t position) noexcept {
    if (!spec.page_tokens)
      return;
    const size_t keep = position ? 1 + (position - 1) / spec.page_tokens : 0;
    for (size_t i = 0; i < spec.layers.size(); ++i) {
      const auto &s = spec.layers[i];
      const size_t stride = 1 + (s.max_tokens - 1) / spec.page_tokens;
      for (size_t b = 0; b < s.batch; ++b)
        for (size_t p = keep;
             p < stride && b * stride + p < key_pages[i].size(); ++p) {
          const size_t index = b * stride + p;
          if (key_pages[i][index]) {
            const size_t bytes = key_pages[i][index]->bytes();
            stats.backing_bytes -= 2 * roundedPageBytes(bytes);
            key_pages[i][index].reset();
            value_pages[i][index].reset();
          }
        }
    }
    // A page that is no longer full is no longer a complete cached block.
    const size_t full = position / spec.page_tokens;
    if (block_ids.size() > full)
      block_ids.erase(block_ids.begin() + full, block_ids.end());
    shared_pages = std::min(shared_pages, keep);
  }
};

void KvView::validate(bool write) const {
  if (!storage_ || epoch_->generation != generation_ ||
      (write && (!step_ || epoch_->active_step != step_)))
    throw std::logic_error("Stale or non-writable KV view");
}

const void *KvView::data() const {
  validate(false);
  return static_cast<const char *>(storage_->data()) + offset_;
}

void *KvView::writableData() const {
  validate(true);
  return static_cast<char *>(storage_->data()) + offset_;
}

StepLease::StepLease(std::shared_ptr<SessionState> state, uint64_t id,
                     size_t from, size_t to) :
  state_(std::move(state)), id_(id), from_(from), to_(to) {}

StepLease::StepLease(StepLease &&other) noexcept :
  state_(std::move(other.state_)),
  id_(other.id_),
  from_(other.from_),
  to_(other.to_) {}

StepLease &StepLease::operator=(StepLease &&other) noexcept {
  if (this != &other) {
    abort();
    state_ = std::move(other.state_);
    id_ = other.id_;
    from_ = other.from_;
    to_ = other.to_;
  }
  return *this;
}

void StepLease::abort() noexcept {
  if (state_ && state_->epoch->active_step == id_) {
    if (state_->overwrites)
      state_->valid = false;
    state_->pending_tokens.clear();
    state_->pending_known = false;
    state_->releaseTail(state_->position);
    state_->epoch->active_step = 0;
    ++state_->epoch->generation;
  }
  state_.reset();
}

StepLease::~StepLease() { abort(); }

KvCacheManager::KvCacheManager() = default;
KvCacheManager::~KvCacheManager() = default;

void KvCacheManager::initialize(const CacheSpec &spec,
                                std::shared_ptr<MemAllocator> allocator,
                                std::shared_ptr<KvPagePool> page_pool,
                                std::shared_ptr<KvPrefixCache> prefix_cache) {
  const auto bytes = spec.validate();
  if (state_ && state_->epoch->active_step)
    throw std::logic_error("KV step is in flight");
  if (page_pool && !spec.page_tokens)
    throw std::invalid_argument("Page pool requires paged KV geometry");
  if (prefix_cache) {
    if (page_pool && page_pool != prefix_cache->pool())
      throw std::invalid_argument(
        "KV prefix cache and its sessions must share one page pool");
    prefix_cache->bind(spec); // Requires paged batch-1 geometry.
    page_pool = prefix_cache->pool();
  }
  auto next = std::make_shared<SessionState>();
  next->spec = spec;
  next->stats.backing_bytes = spec.page_tokens ? 0 : bytes;
  if (spec.page_tokens)
    next->pool = page_pool ? std::move(page_pool)
                           : std::make_shared<KvPagePool>(0, allocator);
  next->prefix = std::move(prefix_cache);
  next->key_pages.resize(spec.layers.size());
  next->value_pages.resize(spec.layers.size());
  for (const auto &s : spec.layers) {
    if (spec.page_tokens) {
      // One-row graph identities, created up front so that every session of
      // this manager shares them and a graph bound once can run any session.
      // They never hold history and stay outside the page pool.
      const size_t row = checkedMultiply(s.batch, s.rowBytes());
      next->keys.push_back(
        KvStorage::create(row, std::make_shared<MemAllocator>()));
      next->values.push_back(
        KvStorage::create(row, std::make_shared<MemAllocator>()));
      if (s.source < 0)
        ++next->stats.owner_layers;
    } else if (s.source >= 0) {
      next->keys.push_back(next->keys[s.source]);
      next->values.push_back(next->values[s.source]);
    } else {
      next->keys.push_back(KvStorage::create(s.planeBytes(), allocator));
      next->values.push_back(KvStorage::create(s.planeBytes(), allocator));
      ++next->stats.owner_layers;
    }
  }
  next->ready.resize(spec.layers.size());
  std::map<SessionId, std::shared_ptr<SessionState>> sessions;
  sessions.emplace(kDefaultSession, next);
  // Every view of every replaced session becomes stale.
  for (auto &entry : sessions_)
    ++entry.second->epoch->generation;
  sessions_.swap(sessions);
  state_ = std::move(next);
  active_ = kDefaultSession;
  next_session_ = kDefaultSession + 1;
}

SessionId KvCacheManager::createSession() {
  if (!state_)
    throw std::logic_error("KV cache is uninitialized");
  if (!state_->spec.page_tokens)
    throw std::logic_error("Additional KV sessions require paged storage");
  auto next = std::make_shared<SessionState>();
  next->spec = state_->spec;
  next->keys = state_->keys; // Shared graph identities.
  next->values = state_->values;
  next->pool = state_->pool;
  next->prefix = state_->prefix;
  next->key_pages.resize(next->spec.layers.size());
  next->value_pages.resize(next->spec.layers.size());
  next->ready.resize(next->spec.layers.size());
  next->stats.owner_layers = state_->stats.owner_layers;
  const SessionId id = next_session_;
  sessions_.emplace(id, std::move(next));
  ++next_session_;
  return id;
}

void KvCacheManager::selectSession(SessionId session) {
  auto it = sessions_.find(session);
  if (it == sessions_.end())
    throw std::out_of_range("Unknown KV session");
  if (state_ && state_->epoch->active_step)
    throw std::logic_error("Cannot switch KV sessions during a step");
  state_ = it->second;
  active_ = session;
}

void KvCacheManager::destroySession(SessionId session) {
  auto it = sessions_.find(session);
  if (it == sessions_.end())
    throw std::out_of_range("Unknown KV session");
  if (session == active_)
    throw std::logic_error("Cannot destroy the selected KV session");
  // Exported views keep their pages alive but can no longer be accessed.
  ++it->second->epoch->generation;
  sessions_.erase(it);
}

SessionId KvCacheManager::activeSession() const { return active_; }

size_t KvCacheManager::sessionCount() const { return sessions_.size(); }

StepLease KvCacheManager::beginStep(size_t count, const TokenId *tokens) {
  if (!state_ || !state_->valid || state_->epoch->active_step)
    throw std::logic_error("KV session unavailable or step already active");
  if (!count ||
      count > state_->spec.layers.front().max_tokens - state_->position)
    throw std::out_of_range("KV step exceeds logical capacity");
  for (const auto &s : state_->spec.layers)
    if (count > s.capacity)
      throw std::out_of_range("KV step exceeds physical capacity; split it");
  // Copy the step's ids first, so a failed copy leaves the session unchanged.
  std::vector<TokenId> pending;
  if (tokens)
    pending.assign(tokens, tokens + count);
  const size_t to = state_->position + count;
  if (state_->spec.page_tokens) {
    const size_t page_tokens = state_->spec.page_tokens;
    const size_t pages = 1 + (to - 1) / page_tokens;
    // Committed history [0, position) always has its pages, so only the
    // partial tail page and pages beyond it can be missing. Newly acquired
    // pages are recorded and released again if any later acquisition fails,
    // leaving the session unchanged without copying the whole page table.
    const size_t first_page = state_->position / page_tokens;
    // Copy-on-write: after a truncation into a page that may be shared, the
    // step would append into a page that other sessions or the prefix cache
    // still read. Give this session a private copy of its committed rows
    // first. Whole shared pages are never written, so no other page needs it.
    const size_t kept_rows = state_->position % page_tokens;
    const bool copy_first = kept_rows && first_page < state_->shared_pages;
    struct Replaced {
      size_t layer;
      size_t index;
      std::shared_ptr<KvStorage> key, value;
    };
    std::vector<std::pair<size_t, size_t>> added;
    std::vector<Replaced> replaced;
    size_t backing = state_->stats.backing_bytes;
    try {
      for (size_t i = 0; i < state_->spec.layers.size(); ++i) {
        const auto &s = state_->spec.layers[i];
        if (s.source >= 0)
          continue;
        const size_t page_bytes = checkedMultiply(page_tokens, s.rowBytes());
        const size_t aligned = checkedAdd(page_bytes, 63) & ~size_t{63};
        // A fixed stride keeps existing batch page indices stable.
        const size_t stride = 1 + (s.max_tokens - 1) / page_tokens;
        const size_t slots = checkedMultiply(s.batch, stride);
        auto &keys = state_->key_pages[i];
        auto &values = state_->value_pages[i];
        if (keys.size() != slots) {
          keys.resize(slots);
          values.resize(slots);
        }
        if (copy_first)
          replaced.reserve(replaced.size() + s.batch);
        for (size_t b = 0; b < s.batch; ++b)
          for (size_t p = first_page; p < pages; ++p) {
            const size_t index = b * stride + p;
            if (keys[index]) {
              if (!copy_first || p != first_page)
                continue;
              auto key = state_->pool->acquire(page_bytes);
              auto value = state_->pool->acquire(page_bytes);
              const size_t used = kept_rows * s.rowBytes();
              std::memcpy(key->data(), keys[index]->data(), used);
              std::memcpy(value->data(), values[index]->data(), used);
              // Capacity was reserved above, so recording cannot throw.
              replaced.push_back(
                {i, index, std::move(keys[index]), std::move(values[index])});
              keys[index] = std::move(key);
              values[index] = std::move(value);
              continue;
            }
            // Reserve first so recording the slot below cannot throw.
            if (added.size() == added.capacity())
              added.reserve(std::max<size_t>(16, 2 * added.capacity()));
            auto key = state_->pool->acquire(page_bytes);
            auto value = state_->pool->acquire(page_bytes);
            backing = checkedAdd(backing, checkedMultiply(aligned, 2));
            keys[index] = std::move(key);
            values[index] = std::move(value);
            added.emplace_back(i, index);
          }
      }
    } catch (...) {
      for (const auto &slot : added) {
        state_->key_pages[slot.first][slot.second].reset();
        state_->value_pages[slot.first][slot.second].reset();
      }
      for (auto it = replaced.rbegin(); it != replaced.rend(); ++it) {
        state_->key_pages[it->layer][it->index] = std::move(it->key);
        state_->value_pages[it->layer][it->index] = std::move(it->value);
      }
      throw;
    }
    state_->stats.backing_bytes = backing;
    if (copy_first)
      state_->shared_pages = first_page;
  }
  state_->overwrites = false;
  for (const auto &s : state_->spec.layers)
    state_->overwrites |= to > s.capacity;
  std::fill(state_->ready.begin(), state_->ready.end(), false);
  // Old read ranges must expire before a new step can overwrite ring rows.
  ++state_->epoch->generation;
  state_->epoch->active_step = ++state_->next_step;
  state_->pending_tokens = std::move(pending);
  state_->pending_known = tokens != nullptr;
  return StepLease(state_, state_->epoch->active_step, state_->position, to);
}

void KvCacheManager::checkStep(const StepLease &step) const {
  if (!state_ || step.state_ != state_ || !step.id_ ||
      state_->epoch->active_step != step.id_ || !state_->valid)
    throw std::logic_error("Invalid KV step lease");
}

std::vector<KvView> KvCacheManager::views(const StepLease &step, size_t layer,
                                          size_t batch, bool key, bool write) {
  checkStep(step);
  const auto &s = state_->spec.layers.at(layer);
  if (!s.has_producer || batch >= s.batch || (write && s.source >= 0))
    throw std::out_of_range("Invalid KV batch or alias write");
  if (!write) {
    size_t owner = layer;
    while (state_->spec.layers[owner].source >= 0)
      owner = state_->spec.layers[owner].source;
    if (!state_->ready[owner])
      throw std::logic_error("KV producer is not ready");
  }
  const size_t from =
    write ? step.from_ : (step.to_ > s.capacity ? step.to_ - s.capacity : 0);
  const size_t count = step.to_ - from;
  if (state_->spec.page_tokens) {
    const size_t page_tokens = state_->spec.page_tokens;
    const size_t stride = 1 + (s.max_tokens - 1) / page_tokens;
    size_t owner = layer;
    while (state_->spec.layers[owner].source >= 0)
      owner = state_->spec.layers[owner].source;
    const auto &pages =
      key ? state_->key_pages[owner] : state_->value_pages[owner];
    std::vector<KvView> result;
    for (size_t pos = from; pos < step.to_;) {
      const size_t page = pos / page_tokens;
      const size_t offset = pos % page_tokens;
      const size_t rows = std::min(step.to_ - pos, page_tokens - offset);
      KvView v;
      v.storage_ = pages.at(batch * stride + page);
      v.epoch_ = state_->epoch;
      v.generation_ = state_->epoch->generation;
      v.step_ = write ? step.id_ : 0;
      v.offset_ = offset * s.rowBytes();
      v.bytes_ = rows * s.rowBytes();
      result.push_back(std::move(v));
      pos += rows;
    }
    return result;
  }
  const size_t row = from % s.capacity;
  const size_t first = std::min(count, s.capacity - row);
  std::vector<KvView> result;
  auto append = [&](size_t at, size_t rows) {
    KvView v;
    v.storage_ = key ? state_->keys[layer] : state_->values[layer];
    v.epoch_ = state_->epoch;
    v.generation_ = state_->epoch->generation;
    v.step_ = write ? step.id_ : 0;
    v.offset_ = (batch * s.capacity + at) * s.rowBytes();
    v.bytes_ = rows * s.rowBytes();
    result.push_back(std::move(v));
  };
  append(row, first);
  if (first < count)
    append(0, count - first);
  return result;
}

std::vector<KvView> KvCacheManager::writeKey(const StepLease &step, size_t l,
                                             size_t b) {
  return views(step, l, b, true, true);
}
std::vector<KvView> KvCacheManager::writeValue(const StepLease &step, size_t l,
                                               size_t b) {
  return views(step, l, b, false, true);
}
std::vector<KvView> KvCacheManager::readKey(const StepLease &step, size_t l,
                                            size_t b) {
  return views(step, l, b, true, false);
}
std::vector<KvView> KvCacheManager::readValue(const StepLease &step, size_t l,
                                              size_t b) {
  return views(step, l, b, false, false);
}

void KvCacheManager::append(const StepLease &step, size_t layer,
                            const void *keys, const void *values, size_t bytes,
                            size_t batch) {
  checkStep(step);
  if (!keys || !values ||
      bytes !=
        checkedMultiply(step.to() - step.from(), layerSpec(layer).rowBytes()))
    throw std::invalid_argument("KV append shape or pointer is invalid");
  auto key_views = writeKey(step, layer, batch);
  auto value_views = writeValue(step, layer, batch);
  size_t offset = 0;
  for (size_t i = 0; i < key_views.size(); ++i) {
    const size_t count = key_views[i].sizeBytes();
    std::memcpy(key_views[i].writableData(),
                static_cast<const char *>(keys) + offset, count);
    std::memcpy(value_views[i].writableData(),
                static_cast<const char *>(values) + offset, count);
    offset += count;
  }
}

void KvCacheManager::markReady(const StepLease &step, size_t layer) {
  checkStep(step);
  if (state_->spec.layers.at(layer).source >= 0 ||
      !state_->spec.layers.at(layer).has_producer)
    throw std::invalid_argument("KV alias has no producer");
  state_->ready[layer] = true;
}

void KvCacheManager::commit(StepLease &&step) {
  checkStep(step);
  for (size_t i = 0; i < state_->ready.size(); ++i)
    if (state_->spec.layers[i].source < 0 &&
        state_->spec.layers[i].has_producer && !state_->ready[i])
      throw std::logic_error("KV commit before all producers completed");
  // Ids extend the known prefix only when it already reaches this step.
  bool record = state_->pending_known && state_->tokens.size() == step.from_;
  if (record) {
    try {
      state_->tokens.insert(state_->tokens.end(),
                            state_->pending_tokens.begin(),
                            state_->pending_tokens.end());
    } catch (...) {
      record = false; // Unknown ids only disable reuse; the KV is committed.
    }
  }
  state_->position = step.to_;
  state_->epoch->active_step = 0;
  step.state_.reset();
  state_->pending_tokens.clear();
  state_->pending_known = false;
  if (record)
    publishPrefix();
}

void KvCacheManager::publishPrefix() noexcept {
  auto &s = *state_;
  if (!s.prefix || !s.spec.page_tokens)
    return;
  try {
    auto &cache = *s.prefix;
    if (s.prefix_epoch != cache.epoch()) {
      // Blocks were dropped since the last check; keep the linked part only.
      size_t linked = 0;
      while (linked < s.block_ids.size() && cache.contains(s.block_ids[linked]))
        ++linked;
      s.block_ids.erase(s.block_ids.begin() + linked, s.block_ids.end());
      s.prefix_epoch = cache.epoch();
    }
    const size_t page_tokens = s.spec.page_tokens;
    const size_t full = std::min(s.tokens.size(), s.position) / page_tokens;
    if (s.block_ids.size() >= full)
      return;
    std::vector<std::shared_ptr<KvStorage>> keys(s.spec.layers.size()),
      values(s.spec.layers.size());
    while (s.block_ids.size() < full) {
      const size_t page = s.block_ids.size(); // Batch 1: slot == page.
      for (size_t i = 0; i < s.spec.layers.size(); ++i) {
        keys[i] = page < s.key_pages[i].size() ? s.key_pages[i][page] : nullptr;
        values[i] =
          page < s.value_pages[i].size() ? s.value_pages[i][page] : nullptr;
      }
      const uint64_t parent = page ? s.block_ids[page - 1] : 0;
      const uint64_t id = cache.insert(
        parent, s.tokens.data() + page * page_tokens, keys, values);
      if (!id)
        break; // Cache full of referenced blocks, or the chain was evicted.
      // Mark the page shared before anything else can fail.
      s.shared_pages = std::max(s.shared_pages, page + 1);
      s.block_ids.push_back(id);
    }
  } catch (...) {
    // Publication is an optimization; the committed session is unaffected.
  }
}

void KvCacheManager::abort(StepLease &&step) {
  checkStep(step);
  step.abort();
}

void KvCacheManager::reset() {
  if (!state_ || state_->epoch->active_step)
    throw std::logic_error("Cannot reset an uninitialized or active KV cache");
  state_->valid = true;
  state_->position = 0;
  state_->forgetPrefix();
  if (state_->spec.page_tokens) {
    for (auto &pages : state_->key_pages)
      pages.clear();
    for (auto &pages : state_->value_pages)
      pages.clear();
    state_->stats.backing_bytes = 0;
  }
  ++state_->epoch->generation;
}

void KvCacheManager::setPosition(size_t position) {
  if (!state_ || !state_->valid || state_->epoch->active_step)
    throw std::logic_error("Cannot seek an unavailable KV session");
  if (position > state_->spec.layers.front().max_tokens)
    throw std::out_of_range("KV position exceeds capacity");
  if (state_->spec.page_tokens && position > state_->position)
    throw std::logic_error("Cannot seek beyond committed paged KV");
  state_->releaseTail(position);
  state_->position = position;
  if (state_->tokens.size() > position)
    state_->tokens.resize(position);
  ++state_->epoch->generation;
}

void KvCacheManager::requireIdle() const {
  if (!state_ || !state_->valid || state_->epoch->active_step)
    throw std::logic_error("KV session is unavailable or has an active step");
}

size_t KvCacheManager::reusePrefix(const TokenId *tokens, size_t count) {
  requireIdle();
  if (!state_->spec.page_tokens)
    throw std::logic_error("KV prefix reuse requires paged storage");
  if (count && !tokens)
    throw std::invalid_argument("KV prefix reuse requires token ids");
  auto &s = *state_;
  count = std::min(count, s.spec.layers.front().max_tokens);

  // Rows this session already holds for the same leading tokens.
  size_t own = 0;
  const size_t known = std::min(s.tokens.size(), count);
  while (own < known && s.tokens[own] == tokens[own])
    ++own;

  std::vector<uint64_t> ids;
  std::vector<std::vector<std::shared_ptr<KvStorage>>> keys, values;
  if (s.prefix)
    s.prefix->match(tokens, count, ids, keys, values);
  const size_t page_tokens = s.spec.page_tokens;
  const size_t cached = ids.size() * page_tokens;

  if (cached > own) {
    // Stage the adopted page table and publish it only once complete.
    const size_t layers = s.spec.layers.size();
    std::vector<std::vector<std::shared_ptr<KvStorage>>> key_pages(layers),
      value_pages(layers);
    size_t backing = 0;
    for (size_t i = 0; i < layers; ++i) {
      const auto &l = s.spec.layers[i];
      if (l.source >= 0)
        continue;
      const size_t stride = 1 + (l.max_tokens - 1) / page_tokens;
      const size_t page_bytes = checkedMultiply(page_tokens, l.rowBytes());
      key_pages[i].resize(checkedMultiply(l.batch, stride));
      value_pages[i].resize(key_pages[i].size());
      for (size_t p = 0; p < ids.size(); ++p) {
        const auto &key = keys[p].at(i);
        const auto &value = values[p].at(i);
        if (!key || !value || key->bytes() != page_bytes ||
            value->bytes() != page_bytes)
          throw std::logic_error("Cached KV block lacks an owner page");
        backing = checkedAdd(backing, 2 * roundedPageBytes(page_bytes));
        key_pages[i][p] = key; // Batch 1: slot == page.
        value_pages[i][p] = value;
      }
    }
    std::vector<TokenId> adopted(tokens, tokens + cached);
    // Nothing below throws. Replaced private pages return to the pool.
    s.key_pages.swap(key_pages);
    s.value_pages.swap(value_pages);
    s.tokens.swap(adopted);
    s.block_ids.swap(ids);
    s.prefix_epoch = s.prefix->epoch();
    s.shared_pages = s.block_ids.size();
    s.position = cached;
    s.stats.backing_bytes = backing;
    s.prefix->recordHit(s.block_ids.size());
  } else if (own < s.position) {
    s.releaseTail(own);
    s.position = own;
    s.tokens.resize(own);
  }
  ++s.epoch->generation;
  return s.position;
}

std::vector<TokenId> KvCacheManager::committedTokens() const {
  return state_ ? state_->tokens : std::vector<TokenId>{};
}

size_t KvCacheManager::knownTokenCount() const {
  return state_ ? state_->tokens.size() : 0;
}

std::shared_ptr<KvPrefixCache> KvCacheManager::prefixCache() const {
  return state_ ? state_->prefix : nullptr;
}

size_t KvCacheManager::committedLength() const {
  return state_ ? state_->position : 0;
}
bool KvCacheManager::valid() const { return state_ && state_->valid; }
bool KvCacheManager::hasActiveStep() const {
  return state_ && state_->epoch->active_step != 0;
}
CacheStats KvCacheManager::stats() const {
  return state_ ? state_->stats : CacheStats{};
}
const LayerSpec &KvCacheManager::layerSpec(size_t layer) const {
  if (!state_)
    throw std::logic_error("KV cache is uninitialized");
  return state_->spec.layers.at(layer);
}
std::shared_ptr<MemoryData> KvCacheManager::keyMemory(size_t layer) {
  layerSpec(layer);
  if (state_->spec.page_tokens)
    throw std::logic_error("Paged KV has no contiguous Tensor bridge");
  return state_->keys[layer]->memory();
}
std::shared_ptr<MemoryData> KvCacheManager::valueMemory(size_t layer) {
  layerSpec(layer);
  if (state_->spec.page_tokens)
    throw std::logic_error("Paged KV has no contiguous Tensor bridge");
  return state_->values[layer]->memory();
}

thread_local KvStepScope *KvStepScope::current_ = nullptr;

std::unique_ptr<KvStepScope> KvCacheManager::bindStep(const StepLease &step) {
  checkStep(step);
  return std::unique_ptr<KvStepScope>(new KvStepScope(*this, step));
}

KvStepScope::KvStepScope(KvCacheManager &manager, const StepLease &step) :
  manager_(manager), step_(step) {
  if (current_)
    throw std::logic_error("Nested CPU KV graph scopes are unsupported");

  for (const auto &layer : manager_.state_->spec.layers)
    if (layer.batch != 1 || layer.capacity != layer.max_tokens)
      throw std::invalid_argument(
        "CPU graph KV views require batch-1 non-ring storage");
  current_ = this;
}

KvStepScope::~KvStepScope() { current_ = nullptr; }

bool KvStepScope::pagedActive() {
  return current_ && current_->manager_.paged();
}

void KvStepScope::validate() const {
  if (current_ != this)
    throw std::logic_error("CPU KV access escaped its graph scope");
  manager_.checkStep(step_);
}

std::unique_ptr<KvLayerAccess> KvStepScope::access(const void *key,
                                                   const void *value,
                                                   size_t batch, size_t from,
                                                   size_t to) {
  if (!current_)
    return nullptr; // Existing unmanaged Tensor paths remain compatible.
  auto &scope = *current_;
  scope.validate();
  if (batch || from != scope.step_.from() || to != scope.step_.to())
    throw std::logic_error("CPU MHA interval differs from its KV lease");
  const auto &state = *scope.manager_.state_;
  for (size_t i = 0; i < state.spec.layers.size(); ++i)
    if ((state.spec.page_tokens || state.spec.layers[i].source < 0) &&
        state.keys[i] && state.values[i] && state.keys[i]->data() == key &&
        state.values[i]->data() == value)
      return std::unique_ptr<KvLayerAccess>(new KvLayerAccess(scope, i));
  throw std::logic_error("CPU MHA cache does not belong to the active KV step");
}

KvLayerAccess::KvLayerAccess(KvStepScope &scope, size_t layer) :
  scope_(&scope), scope_lifetime_(scope.lifetime_), layer_(layer) {
  if (!scope.manager_.paged()) {
    key_write_ = scope.manager_.writeKey(scope.step_, layer);
    value_write_ = scope.manager_.writeValue(scope.step_, layer);
  }
}

KvStepScope &KvLayerAccess::checkedScope() const {
  if (scope_lifetime_.expired())
    throw std::logic_error("CPU KV access outlived its graph scope");
  scope_->validate();
  return *scope_;
}

bool KvCacheManager::paged() const {
  return state_ && state_->spec.page_tokens;
}

size_t KvCacheManager::pageTokens() const {
  return state_ ? state_->spec.page_tokens : 0;
}

std::shared_ptr<MemoryData> KvCacheManager::graphKeyMemory(size_t layer) {
  layerSpec(layer);
  if (!paged())
    return keyMemory(layer);
  return state_->keys[layer]->memory(); // Shared by every session.
}
std::shared_ptr<MemoryData> KvCacheManager::graphValueMemory(size_t layer) {
  layerSpec(layer);
  if (!paged())
    return valueMemory(layer);
  return state_->values[layer]->memory();
}

bool KvLayerAccess::paged() const { return checkedScope().manager_.paged(); }
bool KvLayerAccess::alias() const { return spec().source >= 0; }
const LayerSpec &KvLayerAccess::spec() const {
  return checkedScope().manager_.layerSpec(layer_);
}
void KvLayerAccess::append(const void *key, const void *value, size_t bytes) {
  auto &scope = checkedScope();
  scope.manager_.append(scope.step_, layer_, key, value, bytes);
  scope.manager_.markReady(scope.step_, layer_);
}
std::vector<float> KvLayerAccess::attention(const std::vector<float> &q,
                                            const AttentionOptions &options) {
  auto &scope = checkedScope();
  return scope.manager_.attention(scope.step_, layer_, q, options);
}

size_t KvLayerAccess::pageTokens() const {
  return checkedScope().manager_.pageTokens();
}

std::vector<KvView> KvLayerAccess::readKeyViews(size_t batch) const {
  auto &scope = checkedScope();
  return scope.manager_.readKey(scope.step_, layer_, batch);
}

std::vector<KvView> KvLayerAccess::readValueViews(size_t batch) const {
  auto &scope = checkedScope();
  return scope.manager_.readValue(scope.step_, layer_, batch);
}

void KvCacheManager::savePlane(std::ostream &out, size_t layer, bool key,
                               size_t length) const {
  if (!paged() || hasActiveStep() || !valid() || length > committedLength())
    throw std::logic_error(
      "Paged KV snapshot requires an idle committed range");
  const auto &s = layerSpec(layer);
  size_t owner = layer;
  while (state_->spec.layers[owner].source >= 0)
    owner = state_->spec.layers[owner].source;
  const auto &pages =
    key ? state_->key_pages[owner] : state_->value_pages[owner];
  const size_t stride = 1 + (s.max_tokens - 1) / state_->spec.page_tokens;
  for (size_t b = 0; b < s.batch; ++b)
    for (size_t pos = 0; pos < length;) {
      const size_t count = std::min(length - pos, state_->spec.page_tokens);
      const auto &page = pages.at(b * stride + pos / state_->spec.page_tokens);
      out.write(static_cast<const char *>(page->data()), count * s.rowBytes());
      if (!out)
        throw std::runtime_error("Failed to write paged KV snapshot");
      pos += count;
    }
}

void KvCacheManager::restorePlanes(std::istream &in, size_t length) {
  if (!paged() || hasActiveStep() || !valid())
    throw std::logic_error("Paged KV restore requires an idle session");
  if (!length) {
    reset();
    return;
  }
  KvCacheManager next;
  next.initialize(state_->spec, std::make_shared<MemAllocator>(), state_->pool);
  auto step = next.beginStep(length);
  for (size_t i = 0; i < state_->spec.layers.size(); ++i) {
    const auto &s = state_->spec.layers[i];
    for (bool key : {true, false})
      for (size_t b = 0; b < s.batch; ++b) {
        if (s.source >= 0 || !s.has_producer) {
          size_t left = checkedMultiply(length, s.rowBytes());
          char scratch[4096];
          while (left) {
            const size_t n = std::min(left, sizeof(scratch));
            in.read(scratch, n);
            if (!in)
              throw std::runtime_error("Truncated paged KV snapshot");
            left -= n;
          }
        } else {
          auto segments =
            key ? next.writeKey(step, i, b) : next.writeValue(step, i, b);
          for (auto &segment : segments) {
            in.read(static_cast<char *>(segment.writableData()),
                    segment.sizeBytes());
            if (!in)
              throw std::runtime_error("Truncated paged KV snapshot");
          }
        }
      }
    if (s.source < 0 && s.has_producer)
      next.markReady(step, i);
  }
  next.commit(std::move(step));
  state_->key_pages.swap(next.state_->key_pages);
  state_->value_pages.swap(next.state_->value_pages);
  state_->stats = next.state_->stats;
  state_->position = length;
  state_->forgetPrefix(); // Snapshots carry no ids; the new pages are private.
  ++state_->epoch->generation;
}

void KvLayerAccess::validateWrite(const void *key, const void *value,
                                  size_t bytes) const {
  checkedScope();
  if (key_write_.size() != 1 || value_write_.size() != 1 ||
      key_write_[0].writableData() != key ||
      value_write_[0].writableData() != value ||
      key_write_[0].sizeBytes() != bytes ||
      value_write_[0].sizeBytes() != bytes)
    throw std::logic_error(
      "CPU producer Tensor is outside its checked KV write view");
}

void KvLayerAccess::producerReady() {
  auto &scope = checkedScope();
  scope.manager_.markReady(scope.step_, layer_);
  key_read_ = scope.manager_.readKey(scope.step_, layer_);
  value_read_ = scope.manager_.readValue(scope.step_, layer_);
}

void KvLayerAccess::validateRead(const void *key, const void *value,
                                 size_t bytes) const {
  checkedScope();
  if (key_read_.size() != 1 || value_read_.size() != 1 ||
      key_read_[0].data() != key || value_read_[0].data() != value ||
      key_read_[0].sizeBytes() != bytes || value_read_[0].sizeBytes() != bytes)
    throw std::logic_error(
      "CPU consumer Tensor is outside its checked KV read view");
}

} // namespace nntrainer::kv
