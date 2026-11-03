// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   htp_expert_pool.h
 * @date   22 September 2026
 * @brief  LRU of resident MoE experts for the HTP path (doc 52).
 * @see    https://github.com/nnstreamer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * One pool shared by every MoE layer, sized as capacity_per_layer x layers:
 * the HTP layer call needs a whole layer's active experts resident at once
 * (32 of 32 at prefill), which a per-layer bound below 32 cannot give but a
 * shared pool of 22 x 2 can. Keys are opaque (the layer uses the weight
 * tensor's address). The caller does the loading and releasing; this only
 * decides which key goes when, the same rule Lfm2CachedSlimMoELayer uses:
 * evict the least recently used, refresh recency from the routing's
 * extended top-k so the likely-next experts move to the back.
 *
 * moe.lrfu_half_life=<H> (tokens) evicts by LRFU instead: each key's
 * score is its use count decayed by half every H tokens (H x layers
 * acquire() calls). The lowest score goes, ties least recent first.
 * The score outlives the key's residency.
 */

#ifndef NNTR_HTP_EXPERT_POOL_H
#define NNTR_HTP_EXPERT_POOL_H
#ifdef __cplusplus

#include "htp_runtime_config.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <compute_ops.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <functional>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nntrainer {

/**
 * @brief Total experts to read at startup, across all layers.
 *
 * Unset defaults to 128, capped at the cache capacity. Zero reserves the
 * arena but leaves all expert weight reads to prefill. An explicit value
 * at least as large as the capacity restores a full-cache preload.
 */
inline size_t expertPreloadLimit(size_t capacity, unsigned count = 128u) {
  return std::min(capacity, static_cast<size_t>(count));
}

class ExpertLru {
public:
  using Key = const void *;

  /** Forget an unloaded model after its readers and DSP slots are released. */
  void reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    capacity_ = held_ = 0;
    clock_ = 0;
    pos_.clear();
    order_.clear();
    score_.clear();
    layers_.clear();
    lrfu_ = 0.0;
    lrfu_configured_ = false;
  }

  /** @brief Registers one layer's share of the pool. Idempotent per layer,
   *  so a layer finalized twice does not double its share. */
  void addLayer(const void *layer, size_t per_layer) {
    std::lock_guard<std::mutex> lock(mutex_);
    // The engine can construct this pool before the model config is applied.
    if (!lrfu_configured_) {
      lrfu_ = lrfuTokens();
      lrfu_configured_ = true;
    }
    if (layers_.insert(layer).second)
      capacity_ += per_layer;
  }

  /** @brief Raises the capacity by @a n keys at run time (the
   *  backend, once the prefill's buffers are gone: the DSP's address
   *  space has room for more expert slots then). */
  void addCapacity(size_t n) {
    std::lock_guard<std::mutex> lock(mutex_);
    capacity_ += n;
  }

  /** @brief LRFU with a half-life of @a tokens (0: LRU); the
   *  default is the runtime configuration. Set before the first acquire(). */
  void setLrfu(double tokens) {
    lrfu_ = tokens;
    lrfu_configured_ = true;
  }

  size_t capacity() const { return capacity_; }
  size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return order_.size();
  }
  bool resident(Key k) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pos_.count(k) != 0;
  }

  /**
   * @brief Counts @a n slots as taken by keys that
   *        are not in the LRU yet -- read-ahead still in flight -- so that
   *        neither makeRoom() nor acquire() hands those slots out again.
   *        unhold() gives them back, just before the keys are acquire()d.
   */
  void hold(size_t n) {
    std::lock_guard<std::mutex> lock(mutex_);
    held_ += n;
  }
  void unhold(size_t n) {
    std::lock_guard<std::mutex> lock(mutex_);
    held_ -= std::min(n, held_);
  }

  /** @brief How many keys makeRoom() can make room for with @a pinned. */
  size_t room(const std::vector<Key> &pinned) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t cap = capacity_ - std::min(held_, capacity_);
    std::unordered_set<Key> pin(pinned.begin(), pinned.end());
    size_t pinned_resident = 0;
    for (Key k : pin)
      pinned_resident += pos_.count(k);
    return cap - std::min(pinned_resident, cap);
  }

  /**
   * @brief Evicts least recently used keys outside @a pinned until @a n more
   *        keys fit -- room for keys the caller will bring in itself and
   *        then hand to acquire() with a load that does nothing.
   * @return false, evicting nothing, when the pinned keys already resident
   *         leave no room for @a n.
   */
  bool makeRoom(size_t n, const std::vector<Key> &pinned,
                const std::function<void(Key)> &evict) {
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t cap = capacity_ - std::min(held_, capacity_);
    std::unordered_set<Key> pin(pinned.begin(), pinned.end());
    size_t pinned_resident = 0;
    for (Key k : pin)
      pinned_resident += pos_.count(k);
    if (pinned_resident + n > cap)
      return false;
    size_t excess = order_.size() + n > cap ? order_.size() + n - cap : 0;
    evictFor(excess, pin, evict);
    return true;
  }

  /**
   * @brief Makes every key in @a need resident, evicting the least recently
   *        used keys not in @a need as required.
   * @param load  called for each key that has to be brought in
   * @param evict called for each key that leaves, before any load
   * @return number of keys loaded (the misses)
   * @throw std::runtime_error when @a need alone exceeds the capacity: the
   *        pool was sized below one call's working set.
   */
  size_t acquire(const std::vector<Key> &need,
                 const std::function<void(Key)> &load,
                 const std::function<void(Key)> &evict) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (need.size() > capacity_) {
      throw std::runtime_error(
        "ExpertLru: one call needs " + std::to_string(need.size()) +
        " experts resident but the pool holds " + std::to_string(capacity_) +
        " (moe.cache_experts x MoE layers); raise it");
    }
    const size_t cap = capacity_ - std::min(held_, capacity_);
    if (need.size() > cap) {
      throw std::logic_error("ExpertLru: read-ahead holds " +
                             std::to_string(held_) +
                             " slots, too many for a call that needs " +
                             std::to_string(need.size()));
    }
    std::unordered_set<Key> pinned(need.begin(), need.end());
    std::vector<Key> misses;
    const double half = halfLife();
    if (half > 0.0) {
      ++clock_;
      for (Key k : need) {
        Score &c = score_[k];
        c.crf = 1.0 + decayed(c, half);
        c.last = clock_;
      }
    }
    for (Key k : need) {
      auto it = pos_.find(k);
      if (it != pos_.end())
        touch(it);
      else
        misses.push_back(k);
    }
    // Evictions first, all of them: a slot has to be free before its
    // replacement is read, and the caller's release/register pair is
    // cheapest back to back.
    size_t excess = order_.size() + misses.size() > cap
                      ? order_.size() + misses.size() - cap
                      : 0;
    evictFor(excess, pinned, evict);
    for (Key k : misses) {
      load(k);
      order_.emplace_back(k, &score_[k]);
      pos_[k] = std::prev(order_.end());
    }
    return misses.size();
  }

  /** @brief Moves each resident key of @a recency to the back, so that the
   *  last element ends up most recent; absent keys are skipped. */
  void refresh(const std::vector<Key> &recency) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (Key k : recency) {
      auto it = pos_.find(k);
      if (it != pos_.end())
        touch(it);
    }
  }

  /** @brief Front-to-back (least to most recent) snapshot, for tests. */
  std::vector<Key> order() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<Key> keys;
    for (const auto &e : order_)
      keys.push_back(e.first);
    return keys;
  }

  /** @brief Configured half-life in tokens; zero selects LRU. */
  static double lrfuTokens() { return htpRuntimeConfig().moe.lrfu_half_life; }

private:
  struct Score {
    double crf = 0.0;  /**< decayed use count as of call last */
    uint64_t last = 0; /**< acquire() call of the last use */
  };
  /** A resident key and its score (score_'s element, which does not move):
   *  the eviction pass reads it without a lookup. */
  using Order = std::list<std::pair<Key, const Score *>>;

  /** @brief The half-life in acquire() calls, 0 under LRU. */
  double halfLife() const { return lrfu_ * layers_.size(); }

  double decayed(const Score &c, double half) const {
    return c.crf * std::exp2(-static_cast<double>(clock_ - c.last) / half);
  }

  /** @brief Evicts @a excess keys outside @a pin: under LRU from the front,
   *  under LRFU the lowest scores, lowest first, the least recent on a tie.
   *  The scores do not move within a call, so one pass reads them all (a
   *  pass a victim cost the S26 Ultra's miss server 82 us a round). */
  void evictFor(size_t excess, const std::unordered_set<Key> &pin,
                const std::function<void(Key)> &evict) {
    const double half = halfLife();
    std::vector<Order::iterator> victims;
    if (half <= 0.0) {
      for (auto it = order_.begin();
           it != order_.end() && victims.size() < excess; ++it)
        if (pin.count(it->first) == 0)
          victims.push_back(it);
    } else if (excess != 0) {
      // (score, place in the order): the place breaks a tie and keeps the
      // iterators out of the comparison
      std::vector<std::pair<double, size_t>> by_score;
      std::vector<Order::iterator> at;
      by_score.reserve(order_.size());
      at.reserve(order_.size());
      for (auto it = order_.begin(); it != order_.end(); ++it) {
        if (pin.count(it->first) != 0)
          continue;
        by_score.emplace_back(decayed(*it->second, half), at.size());
        at.push_back(it);
      }
      excess = std::min(excess, by_score.size());
      std::partial_sort(by_score.begin(), by_score.begin() + excess,
                        by_score.end());
      for (size_t i = 0; i < excess; ++i)
        victims.push_back(at[by_score[i].second]);
    }
    for (Order::iterator victim : victims) {
      const Key k = victim->first;
      order_.erase(victim);
      pos_.erase(k);
      evict(k);
    }
  }

  void touch(std::unordered_map<Key, Order::iterator>::iterator it) {
    order_.splice(order_.end(), order_, it->second);
    it->second = std::prev(order_.end());
  }

  mutable std::mutex mutex_;
  size_t capacity_ = 0;
  size_t held_ = 0; /**< slots taken by keys still being read (hold()) */
  std::set<const void *> layers_;
  Order order_; /**< front = least recently used */
  std::unordered_map<Key, Order::iterator> pos_;
  double lrfu_ = 0.0; /**< half-life, tokens */
  bool lrfu_configured_ = false;
  uint64_t clock_ = 0;                   /**< acquire() calls */
  std::unordered_map<Key, Score> score_; /**< LRFU, every key seen */
};

/**
 * @brief One layer's warm list from a prefill call: the last token's
 *        extended top-k in rank order (@a ranked, @a per_token ids a token,
 *        token after token), then the experts the call routed most rows
 *        to (@a rows, by expert; ties to the lower id). At most @a n,
 *        distinct.
 */
inline std::vector<unsigned> expertWarmList(const std::vector<int> &ranked,
                                            size_t per_token,
                                            const std::vector<size_t> &rows,
                                            size_t n) {
  std::vector<unsigned> out;
  std::vector<char> taken(rows.size(), 0);
  const size_t from =
    per_token && ranked.size() >= per_token ? ranked.size() - per_token : 0;
  for (size_t i = from; i < ranked.size() && out.size() < n; ++i) {
    const int e = ranked[i];
    if (e >= 0 && static_cast<size_t>(e) < rows.size() && !taken[e]) {
      taken[e] = 1;
      out.push_back(static_cast<unsigned>(e));
    }
  }
  std::vector<unsigned> by_rows;
  for (size_t e = 0; e < rows.size(); ++e)
    if (!taken[e] && rows[e] != 0)
      by_rows.push_back(static_cast<unsigned>(e));
  std::stable_sort(by_rows.begin(), by_rows.end(),
                   [&](unsigned a, unsigned b) { return rows[a] > rows[b]; });
  for (size_t i = 0; i < by_rows.size() && out.size() < n; ++i)
    out.push_back(by_rows[i]);
  return out;
}

/**
 * @brief The decode pool's warm-up from the prefill's routing
 *        (moe.warm_experts a layer): each layer's list from its last
 *        prefill call, read into the pool while the first token's setup
 *        runs. S26 Ultra, Gemma 4 26B-A4B at 16: the first token's misses
 *        222 -> ~115.
 */
class ExpertWarm {
public:
  using Desc = ComputeOps::ExpertFileDesc;

  /** @brief Experts a layer, read when used: the pool can exist before the
   *         model's configuration is applied (0: off). */
  static size_t count() { return htpRuntimeConfig().moe.warm_experts; }

  /** @brief A prefill call of layer @a slot: its warm list from the
   *         routing (@a ranked, the extended top-k token by token) and
   *         each expert's rows. A no-op when off. */
  void note(int slot, const std::vector<int> &ranked,
            const std::vector<std::vector<std::pair<unsigned, float>>> &rows,
            unsigned int total_tokens) {
    if (count() == 0 || slot < 0 || total_tokens < 2)
      return;
    if (lists_.size() <= static_cast<size_t>(slot))
      lists_.resize(slot + 1);
    std::vector<size_t> n(rows.size());
    for (size_t e = 0; e < rows.size(); ++e)
      n[e] = rows[e].size();
    lists_[slot] =
      expertWarmList(ranked, ranked.size() / total_tokens, n, count());
  }

  /** @brief Hands the warm-up to the backend (set_decode_pool_warm) when
   *         on. @a layers: every layer slot's experts, by expert. */
  void install(ComputeOps *ops, ExpertLru &lru,
               const std::vector<std::vector<Desc>> &layers) {
    if (count() == 0)
      return;
    ops->set_decode_pool_warm(
      [this, ops, &lru, &layers] { return begin(ops, lru, layers); },
      [this, &lru](const std::vector<const void *> &keys) {
        lru.unhold(held_);
        held_ = 0;
        lru.acquire(
          keys, [](ExpertLru::Key) {},
          [](ExpertLru::Key) {
            throw std::logic_error("expert warm-up overfilled the LRU");
          });
      });
  }

private:
  /** Room for the warm experts not resident, rank-major over the layers
   *  (a short room cuts every layer's tail, not the last layers), the
   *  slots held, the experts to read. */
  std::vector<Desc> begin(ComputeOps *ops, ExpertLru &lru,
                          const std::vector<std::vector<Desc>> &layers) {
    std::vector<Desc> want;
    std::vector<ExpertLru::Key> pinned;
    size_t depth = 0;
    for (const auto &w : lists_)
      depth = std::max(depth, w.size());
    for (size_t r = 0; r < depth; ++r)
      for (size_t l = 0; l < lists_.size() && l < layers.size(); ++l) {
        if (r >= lists_[l].size() || lists_[l][r] >= layers[l].size())
          continue;
        const Desc &d = layers[l][lists_[l][r]];
        if (lru.resident(d.key_gu))
          pinned.push_back(d.key_gu);
        else
          want.push_back(d);
      }
    want.resize(std::min(want.size(), lru.room(pinned)));
    auto release = [ops](ExpertLru::Key k) {
      if (!ops->release_qs4cx_wh_expert(k))
        throw std::logic_error(
          "expert warm-up evicted an expert the accelerator does not hold");
    };
    if (want.empty() || !lru.makeRoom(want.size(), pinned, release))
      return {};
    lru.hold(want.size());
    held_ = want.size();
    return want;
  }

  std::vector<std::vector<unsigned>> lists_;
  size_t held_ = 0;
};

/** Runtime-owned host policy and prefill registry for the accelerator slots.
 * Readers and decode callbacks must stop before this object is destroyed.
 * Only the LRU is accessed by worker callbacks; registry/queue mutation stays
 * on the serial model-loading or generation thread, as before.
 */
struct HtpExpertPool {
  struct FileCloser {
    void operator()(std::FILE *file) const { std::fclose(file); }
  };
  using ExpertFileDesc = ComputeOps::ExpertFileDesc;
  using SlotClass = std::array<unsigned int, 4>;
  ExpertLru lru;
  std::unique_ptr<std::FILE, FileCloser> trace_file;
  bool trace_opened = false;
  std::vector<std::vector<ExpertFileDesc>> layers;
  std::map<SlotClass, size_t> class_capacity;
  std::set<SlotClass> class_reserved;
  size_t slots_reserved = 0;
  std::deque<std::pair<int, size_t>> prefetch_batches;
  int prefetch_next = 0;
  std::unordered_set<ExpertLru::Key> prefetch_inflight;
  std::unordered_set<const void *> handed_layers;
  ComputeOps *owner = nullptr;
  ComputeOps *early_owner = nullptr;
  ExpertWarm warm; /**< moe.warm_experts: the decode pool's warm-up */

  static SlotClass slotClass(const ExpertFileDesc &d) {
    return {d.K, d.inter, d.N_out, d.w_bits * 2u + (d.affine ? 1u : 0u)};
  }

  // Diagnostics include CPU layers too; retain their process-wide numbering
  // and reset it only at HTP runtime release, after model destruction.
  static unsigned nextLayerOrdinal() { return trace_layers_.fetch_add(1); }
  static void resetLayerOrdinals() { trace_layers_.store(0); }

private:
  inline static std::atomic<unsigned> trace_layers_{0};
};

inline HtpExpertPool &requireHtpExpertPool(ComputeOps *ops) {
  auto *pool = ops ? ops->htpExpertPool() : nullptr;
  if (!pool)
    throw std::runtime_error(
      "virtual MoE experts require a backend-owned pool");
  return *pool;
}

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* NNTR_HTP_EXPERT_POOL_H */
