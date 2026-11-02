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
 * NNTR_MOE_LRFU=<H> (tokens) evicts by LRFU instead: each key's
 * score is its use count decayed by half every H tokens (H x layers
 * acquire() calls). The lowest score goes, ties least recent first.
 * The score outlives the key's residency.
 */

#ifndef NNTR_HTP_EXPERT_POOL_H
#define NNTR_HTP_EXPERT_POOL_H
#ifdef __cplusplus

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <compute_ops.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <functional>
#include <limits>
#include <list>
#include <map>
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
inline size_t expertPreloadLimit(size_t capacity, const char *value) {
  if (!value)
    return std::min(capacity, size_t{128});
  if (*value < '0' || *value > '9')
    throw std::invalid_argument("NNTR_MOE_PRELOAD_EXPERTS must be nonnegative");
  char *end = nullptr;
  errno = 0;
  const auto count = std::strtoull(value, &end, 10);
  if (errno || *end || count > std::numeric_limits<size_t>::max())
    throw std::invalid_argument("Invalid NNTR_MOE_PRELOAD_EXPERTS");
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
    lrfu_ = lrfuTokens();
  }

  /** @brief Registers one layer's share of the pool. Idempotent per layer,
   *  so a layer finalized twice does not double its share. */
  void addLayer(const void *layer, size_t per_layer) {
    std::lock_guard<std::mutex> lock(mutex_);
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
   *  default is NNTR_MOE_LRFU's. Set before the first acquire(). */
  void setLrfu(double tokens) { lrfu_ = tokens; }

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
        " (NNTR_MOE_CACHE_EXPERTS x MoE layers); raise it");
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

  /** @brief NNTR_MOE_LRFU's half-life in tokens; "0" is LRU, unset is 32.
   *  Read once per process. */
  static double lrfuTokens() {
    static const double h = [] {
      const char *v = std::getenv("NNTR_MOE_LRFU");
      if (v == nullptr || *v == '\0')
        return 32.0;
      char *end = nullptr;
      const double x = std::strtod(v, &end);
      if (*end != '\0' || !(x >= 0.0))
        throw std::invalid_argument(std::string("NNTR_MOE_LRFU=") + v +
                                    ": want a half-life in tokens");
      return x;
    }();
    return h;
  }

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
  double lrfu_ = lrfuTokens();           /**< half-life, tokens */
  uint64_t clock_ = 0;                   /**< acquire() calls */
  std::unordered_map<Key, Score> score_; /**< LRFU, every key seen */
};

/** Runtime-owned host policy and prefill registry for the accelerator slots.
 * Readers and decode callbacks must stop before this object is destroyed.
 * Only the LRU is accessed by worker callbacks; registry/queue mutation stays
 * on the serial model-loading or generation thread, as before.
 */
struct HtpExpertPool {
  using ExpertFileDesc = ComputeOps::ExpertFileDesc;
  using SlotClass = std::array<unsigned int, 4>;
  ExpertLru lru;
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
