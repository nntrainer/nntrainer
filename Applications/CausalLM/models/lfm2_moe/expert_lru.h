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
 */

#ifndef __CAUSALLM_EXPERT_LRU_H__
#define __CAUSALLM_EXPERT_LRU_H__
#ifdef __cplusplus

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

  size_t capacity() const { return capacity_; }
  size_t size() const { return order_.size(); }
  bool resident(Key k) const { return pos_.count(k) != 0; }

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
    std::unordered_set<Key> pinned(need.begin(), need.end());
    std::vector<Key> misses;
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
    size_t excess = order_.size() + misses.size() > capacity_
                      ? order_.size() + misses.size() - capacity_
                      : 0;
    for (auto it = order_.begin(); excess != 0 && it != order_.end();) {
      if (pinned.count(*it) != 0) {
        ++it;
        continue;
      }
      Key victim = *it;
      it = order_.erase(it);
      pos_.erase(victim);
      evict(victim);
      --excess;
    }
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

private:
  using Order = std::list<Key>;

  void touch(std::unordered_map<Key, Order::iterator>::iterator it) {
    order_.splice(order_.end(), order_, it->second);
    it->second = std::prev(order_.end());
  }

  mutable std::mutex mutex_;
  size_t capacity_ = 0;
  std::set<const void *> layers_;
  Order order_; /**< front = least recently used */
  std::unordered_map<Key, Order::iterator> pos_;
};

} // namespace causallm

#endif /* __cplusplus */
#endif /* __CAUSALLM_EXPERT_LRU_H__ */
