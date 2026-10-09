// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   expert_lru.h
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
 * [#289] NNTR_MOE_LRFU=<H> (tokens) evicts by LRFU instead: each key's
 * score is its use count decayed by half every H tokens (H x layers
 * acquire() calls), the lowest score goes, ties least recent first --
 * tools/moe_expert_cache_sim.py's lrfu:H, which cuts the #266 p1024 trace's
 * misses at C = 24 by 20 %. The score outlives the key's residency.
 */

#ifndef __CAUSALLM_EXPERT_LRU_H__
#define __CAUSALLM_EXPERT_LRU_H__
#ifdef __cplusplus

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <list>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace causallm {

class ExpertLru {
public:
  using Key = const void *;

  /** @brief Registers one layer's share of the pool. Idempotent per layer,
   *  so a layer finalized twice does not double its share. */
  void addLayer(const void *layer, size_t per_layer) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (layers_.insert(layer).second)
      capacity_ += per_layer;
  }

  /** @brief [#289] LRFU with a half-life of @a tokens (0: LRU); the
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
   * @brief [doc 52 section 10.20] Counts @a n slots as taken by keys that
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
      order_.push_back(k);
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
    return std::vector<Key>(order_.begin(), order_.end());
  }

  /** @brief [#289] NNTR_MOE_LRFU's half-life in tokens; 0 (unset or "0")
   *  is LRU. Read once per process. */
  static double lrfuTokens() {
    static const double h = [] {
      const char *v = std::getenv("NNTR_MOE_LRFU");
      if (v == nullptr || *v == '\0')
        return 0.0;
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
  using Order = std::list<Key>;

  struct Score {
    double crf = 0.0;  /**< decayed use count as of call last */
    uint64_t last = 0; /**< acquire() call of the last use */
  };

  /** @brief The half-life in acquire() calls, 0 under LRU. */
  double halfLife() const { return lrfu_ * layers_.size(); }

  double decayed(const Score &c, double half) const {
    return c.crf * std::exp2(-static_cast<double>(clock_ - c.last) / half);
  }

  /** @brief Evicts @a excess keys outside @a pin: under LRU from the front,
   *  under LRFU the lowest score each time, the least recent on a tie. */
  void evictFor(size_t excess, const std::unordered_set<Key> &pin,
                const std::function<void(Key)> &evict) {
    const double half = halfLife();
    while (excess != 0) {
      auto victim = order_.end();
      double best = 0.0;
      for (auto it = order_.begin(); it != order_.end(); ++it) {
        if (pin.count(*it) != 0)
          continue;
        if (half <= 0.0) {
          victim = it;
          break;
        }
        const double v = decayed(score_[*it], half);
        if (victim == order_.end() || v < best) {
          victim = it;
          best = v;
        }
      }
      if (victim == order_.end())
        return; // all pinned: the callers checked the room first
      const Key k = *victim;
      order_.erase(victim);
      pos_.erase(k);
      evict(k);
      --excess;
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
  double lrfu_ = lrfuTokens();           /**< [#289] half-life, tokens */
  uint64_t clock_ = 0;                   /**< [#289] acquire() calls */
  std::unordered_map<Key, Score> score_; /**< [#289] LRFU, every key seen */
};

} // namespace causallm

#endif /* __cplusplus */
#endif /* __CAUSALLM_EXPERT_LRU_H__ */
