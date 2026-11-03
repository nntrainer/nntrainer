// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   expert_residency.h
 * @date   08 October 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Which experts of each MoE layer sit in which device slot: a
 *         per-layer LRU, LFU or LRFU over a fixed partition of slots.
 *
 * @details A MoE layer whose experts are streamed from the weight file keeps
 * only C of its E experts in device memory, in C slots that belong to that
 * layer (its partition). Every layer is touched once per token, so recency
 * across layers carries no information; the partitions are independent and
 * C may differ per layer (addLayer takes it per layer).
 *
 * The victim policy (setPolicy, default LRU) picks among the resident
 * experts that are neither needed by the call nor held:
 *  - LRU: the least recent;
 *  - LFU: the fewest accesses so far (history: kept for evicted experts),
 *    ties to the least recently accessed, then the least recent;
 *  - LRFU(lambda): the smallest combined recency-frequency value
 *    CRF = sum over the expert's accesses of 2^(-lambda * age), age in
 *    acquires of the layer (history kept across evictions), same ties.
 *    lambda -> 0 is LFU, large lambda LRU.
 * An access is an acquire() that needs the expert (one per call however
 * often it is listed). refresh() reorders recency only. prime() seeds the
 * frequency history (LFU / LRFU) of a layer, e.g. from a prefill's routing
 * histogram, without loading anything.
 *
 * This class only decides; the caller moves the bytes:
 *
 *  - acquire(layer, need) returns, for each needed expert, its slot, and the
 *    misses: (expert, slot, victim) for every expert that has to be read into
 *    its slot first. A miss takes a free slot when there is one, else the
 *    slot of the least recently used expert that is neither needed by this
 *    call nor held. The needed experts become the most recent, in the order
 *    given.
 *  - hold(layer, slot) / unhold(layer, slot) pin a slot whose read is still in
 *    flight (or whose bytes a queued kernel still reads): it is never chosen
 *    as a victim while held. Holds nest.
 *  - refresh(layer, recency) moves the resident experts of @a recency to the
 *    most recent end, in order (the last one ends up most recent): a hint
 *    from the router's extended top-k, it never loads or evicts.
 *  - forget(layer, expert) drops an expert whose slot does not hold its
 *    bytes after all (its read failed): the slot becomes free.
 *  - clear(layer) forgets every expert of the layer (its slots were lent to
 *    another use, e.g. a prefill that streams whole layers through them).
 *  - reserve(layer, expert, avoid, keep) / release(layer, expert, admit):
 *    a read ahead of need (a prediction of the layer's next call) takes a
 *    slot before the call: a free slot, else the policy's victim among the
 *    resident experts outside @a avoid (evicted at once). The slot is
 *    reserved for the expert: it is not resident, it is never a victim and
 *    never a free slot for another expert until the reservation ends, and
 *    nothing is counted as an access (no recency, no frequency). It ends
 *    when an acquire needs the expert (the expert becomes resident in that
 *    slot: a miss whose bytes the reader already put there, Miss::reserved)
 *    or by release(): the slot becomes free, or with @a admit (its bytes
 *    were written) the expert stays in it as a resident expert with no
 *    access counted -- the least recent one, its frequency history as it
 *    was. At most capacity - @a keep slots are reserved at once, so a call
 *    of @a keep experts always finds room.
 *  - victimOrder(layer, exclude) / place(layer, need, placed) /
 *    admit(layer, expert, slot): for a caller that places misses itself (a
 *    device routing its own decode step from a victim queue published ahead
 *    of the step, NNTR_G4MOE_FSU_FLAGS). victimOrder lists the slots in the
 *    order successive acquires would take them; place() replays a call
 *    whose misses went to the given slots (acquire's history and recency);
 *    admit() replays a read ahead that put an expert into a slot (no access,
 *    least recent, as release() with admit).
 *
 * Deterministic: the same call sequence gives the same slots. Thread-safe
 * (one mutex). Modelled on the ExpertLru of the HTP MoE layers, partitioned
 * per layer instead of shared.
 */

#ifndef __CAUSALLM_EXPERT_RESIDENCY_H__
#define __CAUSALLM_EXPERT_RESIDENCY_H__

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <list>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace causallm {

class ExpertResidency {
public:
  static constexpr unsigned int kNone =
    std::numeric_limits<unsigned int>::max();

  /** @brief one expert to bring into a slot */
  struct Miss {
    unsigned int expert; /**< the expert to read */
    unsigned int slot;   /**< the slot it goes to */
    unsigned int victim; /**< the expert that left the slot, or kNone */
    /** the slot was reserved for the expert (reserve()): its reader may
     *  have written it already; victim is kNone (evicted at the reserve) */
    bool reserved = false;
  };

  /** @brief victim policy (see the file comment) */
  enum class Policy { lru, lfu, lrfu };

  /** @brief the answer of acquire() */
  struct Acquired {
    std::vector<unsigned int> slot; /**< slot of need[i] */
    std::vector<Miss> misses;       /**< in the order of need */
    unsigned int hits = 0;          /**< distinct needed experts resident */
  };

  /**
   * @brief Register layer @a layer with @a capacity slots over @a experts
   *        experts. Re-registering with the same shape is a no-op.
   * @throw std::invalid_argument for a zero capacity or expert count, or a
   *        different shape for a known layer
   */
  void addLayer(unsigned int layer, unsigned int capacity,
                unsigned int experts) {
    std::lock_guard<std::mutex> lk(mu_);
    if (capacity == 0 || experts == 0)
      throw std::invalid_argument(
        "ExpertResidency: layer " + std::to_string(layer) +
        " needs a non-zero capacity and expert count");
    auto it = layers_.find(layer);
    if (it != layers_.end()) {
      if (it->second.capacity != capacity ||
          it->second.slot_of.size() != experts)
        throw std::invalid_argument("ExpertResidency: layer " +
                                    std::to_string(layer) +
                                    " registered again with another shape");
      return;
    }
    Layer &l = layers_[layer];
    l.capacity = capacity;
    l.slot_of.assign(experts, kNone);
    l.expert_in.assign(capacity, kNone);
    l.held.assign(capacity, 0);
    l.reserved_for.assign(capacity, kNone);
    l.reserved_slot.assign(experts, kNone);
    l.where.assign(experts, l.order.end());
    l.score.assign(experts, 0.0);
    l.stamp.assign(experts, 0);
    l.last.assign(experts, 0);
  }

  /**
   * @brief Victim policy of every layer (see the file comment); @a lambda
   *        is LRFU's decay per acquire (ignored otherwise; > 0). Set before
   *        the first acquire: the history of a policy is not converted.
   * @throw std::invalid_argument for LRFU with lambda <= 0
   */
  void setPolicy(Policy p, double lambda = 0.1) {
    std::lock_guard<std::mutex> lk(mu_);
    if (p == Policy::lrfu && !(lambda > 0.0))
      throw std::invalid_argument("ExpertResidency: LRFU needs lambda > 0");
    policy_ = p;
    lambda_ = p == Policy::lrfu ? lambda : 0.0;
  }
  Policy policy() const {
    std::lock_guard<std::mutex> lk(mu_);
    return policy_;
  }
  double lambda() const {
    std::lock_guard<std::mutex> lk(mu_);
    return lambda_;
  }

  /**
   * @brief Seed @a layer's frequency history: expert e gets @a weight[e]
   *        accesses' worth (LFU: added to its count; LRFU: added to its CRF
   *        now). Nothing is loaded or evicted and recency is unchanged; no-op
   *        for LRU. Used to prime the decode from the prefill's histogram.
   */
  void prime(unsigned int layer, const std::vector<double> &weight) {
    std::lock_guard<std::mutex> lk(mu_);
    Layer &l = get(layer);
    if (policy_ == Policy::lru)
      return;
    for (size_t e = 0; e < weight.size() && e < l.score.size(); ++e) {
      if (!(weight[e] > 0.0))
        continue;
      l.score[e] = scoreNow(l, (unsigned int)e) + weight[e];
      l.stamp[e] = l.clock;
    }
  }

  bool hasLayer(unsigned int layer) const {
    std::lock_guard<std::mutex> lk(mu_);
    return layers_.count(layer) != 0;
  }

  unsigned int capacity(unsigned int layer) const {
    std::lock_guard<std::mutex> lk(mu_);
    return get(layer).capacity;
  }

  /** @brief resident experts of @a layer */
  unsigned int size(unsigned int layer) const {
    std::lock_guard<std::mutex> lk(mu_);
    return static_cast<unsigned int>(get(layer).order.size());
  }

  /** @brief slot of @a expert, or kNone */
  unsigned int slotOf(unsigned int layer, unsigned int expert) const {
    std::lock_guard<std::mutex> lk(mu_);
    const Layer &l = get(layer);
    return expert < l.slot_of.size() ? l.slot_of[expert] : kNone;
  }

  /** @brief expert -> slot map of @a layer (kNone when not resident) */
  std::vector<unsigned int> slotMap(unsigned int layer) const {
    std::lock_guard<std::mutex> lk(mu_);
    return get(layer).slot_of;
  }

  /**
   * @brief Make every expert of @a need resident (see the file comment).
   * @throw std::runtime_error when the distinct experts of @a need exceed the
   *        slots that are not held by other experts
   * @throw std::out_of_range for an expert id outside the layer
   */
  Acquired acquire(unsigned int layer, const std::vector<unsigned int> &need) {
    std::lock_guard<std::mutex> lk(mu_);
    Layer &l = get(layer);
    std::unordered_set<unsigned int> pinned;
    for (unsigned int e : need) {
      if (e >= l.slot_of.size())
        throw std::out_of_range("ExpertResidency: expert " + std::to_string(e) +
                                " of layer " + std::to_string(layer));
      pinned.insert(e);
    }
    // room: slots that are not held by an expert outside this call, nor
    // reserved for one
    unsigned int blocked = 0;
    for (unsigned int s = 0; s < l.capacity; ++s)
      if ((l.held[s] != 0 &&
           (l.expert_in[s] == kNone || pinned.count(l.expert_in[s]) == 0)) ||
          (l.reserved_for[s] != kNone && pinned.count(l.reserved_for[s]) == 0))
        ++blocked;
    if (pinned.size() + blocked > l.capacity)
      throw std::runtime_error(
        "ExpertResidency: layer " + std::to_string(layer) + " needs " +
        std::to_string(pinned.size()) + " experts resident at once but has " +
        std::to_string(l.capacity) + " slots (" + std::to_string(blocked) +
        " held); raise the cache size");

    Acquired out;
    out.slot.resize(need.size(), kNone);
    // the access history: one step per acquire, every distinct needed
    // expert accessed once
    ++l.clock;
    for (unsigned int e : pinned) {
      if (policy_ != Policy::lru) {
        l.score[e] = scoreNow(l, e) + 1.0;
        l.stamp[e] = l.clock;
      }
      l.last[e] = l.clock;
    }
    std::unordered_set<unsigned int> seen;
    for (size_t i = 0; i < need.size(); ++i) {
      const unsigned int e = need[i];
      const bool first = seen.insert(e).second;
      if (l.slot_of[e] != kNone) {
        if (first)
          ++out.hits;
        touch(l, e);
        out.slot[i] = l.slot_of[e];
        continue;
      }
      // a miss: its reserved slot; else the lowest free slot that is neither
      // held nor reserved, else the policy's victim
      unsigned int slot = kNone, victim = kNone;
      const bool was_reserved = l.reserved_slot[e] != kNone;
      if (was_reserved) {
        slot = l.reserved_slot[e];
        l.reserved_for[slot] = kNone;
        l.reserved_slot[e] = kNone;
        ++reserved_used_;
      }
      for (unsigned int s = 0; s < l.capacity && slot == kNone; ++s)
        if (freeSlot(l, s))
          slot = s;
      if (slot == kNone) {
        victim = pickVictim(l, pinned);
        if (victim != kNone)
          slot = l.slot_of[victim];
      }
      if (slot == kNone) // the room check above makes this unreachable
        throw std::logic_error("ExpertResidency: no slot for a miss");
      if (victim != kNone) {
        l.order.erase(l.where[victim]);
        l.where[victim] = l.order.end();
        l.slot_of[victim] = kNone;
      }
      l.expert_in[slot] = e;
      l.slot_of[e] = slot;
      l.order.push_back(e);
      l.where[e] = std::prev(l.order.end());
      out.slot[i] = slot;
      out.misses.push_back({e, slot, victim, was_reserved});
    }
    hits_ += out.hits;
    misses_ += out.misses.size();
    return out;
  }

  /**
   * @brief The slots of @a layer in the order successive acquire() calls
   *        would take them for misses: the free slots (neither held nor
   *        reserved), lowest first, then the resident experts' slots, the
   *        policy's victim first (LRU: least recent; LFU / LRFU: smallest
   *        score, ties to the least recently accessed, then the least
   *        recent), skipping held and reserved slots and @a exclude.
   * @details A caller that places a call's misses itself takes, for each
   * miss, the first slot of this list it has not taken yet whose expert the
   * call does not need: acquire() would have picked the same slots (the
   * scores of the experts it may evict all decay by the same factor between
   * this list and the call; only exact score ties may order differently
   * after that rounding, which moves an eviction, never a result).
   */
  std::vector<unsigned int>
  victimOrder(unsigned int layer,
              const std::vector<unsigned int> &exclude = {}) const {
    std::lock_guard<std::mutex> lk(mu_);
    const Layer &l = get(layer);
    std::vector<bool> skip(l.capacity, false);
    for (unsigned int s : exclude)
      if (s < l.capacity)
        skip[s] = true;
    std::vector<unsigned int> out;
    out.reserve(l.capacity);
    for (unsigned int s = 0; s < l.capacity; ++s)
      if (!skip[s] && freeSlot(l, s))
        out.push_back(s);
    struct Cand {
      unsigned int slot;
      double score;
      uint64_t last;
    };
    std::vector<Cand> cand;
    for (unsigned int e : l.order) {
      const unsigned int s = l.slot_of[e];
      if (skip[s] || l.held[s] != 0 || l.reserved_for[s] != kNone)
        continue;
      cand.push_back({s, policy_ == Policy::lru ? 0.0 : scoreNow(l, e),
                      policy_ == Policy::lru ? 0 : l.last[e]});
    }
    // stable: equal keys keep the order list's order (pickVictim's ties)
    std::stable_sort(
      cand.begin(), cand.end(), [](const Cand &a, const Cand &b) {
        return a.score < b.score || (a.score == b.score && a.last < b.last);
      });
    for (const Cand &c : cand)
      out.push_back(c.slot);
    return out;
  }

  /**
   * @brief Replay a call of @a need whose misses a caller placed itself
   *        (victimOrder): every expert of @a placed ((expert, slot), in the
   *        order of @a need) goes to its slot, evicting the slot's expert;
   *        the access history and recency are acquire()'s.
   * @throw std::logic_error when the replay contradicts the state: a placed
   *        expert that is resident or not needed, a needed expert neither
   *        resident nor placed, a slot held, reserved, placed twice or
   *        holding a needed expert (nothing is changed then)
   * @throw std::out_of_range for an expert or slot outside the layer
   */
  Acquired
  place(unsigned int layer, const std::vector<unsigned int> &need,
        const std::vector<std::pair<unsigned int, unsigned int>> &placed) {
    std::lock_guard<std::mutex> lk(mu_);
    Layer &l = get(layer);
    std::unordered_set<unsigned int> pinned;
    for (unsigned int e : need) {
      if (e >= l.slot_of.size())
        throw std::out_of_range("ExpertResidency: expert " + std::to_string(e) +
                                " of layer " + std::to_string(layer));
      pinned.insert(e);
    }
    std::unordered_map<unsigned int, unsigned int> to; // expert -> slot
    std::unordered_set<unsigned int> slots;
    for (const auto &p : placed) {
      if (p.first >= l.slot_of.size())
        throw std::out_of_range("ExpertResidency: expert " +
                                std::to_string(p.first) + " of layer " +
                                std::to_string(layer));
      checkSlot(l, layer, p.second);
      const unsigned int occ = l.expert_in[p.second];
      if (pinned.count(p.first) == 0 || l.slot_of[p.first] != kNone ||
          !to.emplace(p.first, p.second).second ||
          !slots.insert(p.second).second || l.held[p.second] != 0 ||
          l.reserved_for[p.second] != kNone ||
          (occ != kNone && pinned.count(occ) != 0))
        throw std::logic_error("ExpertResidency: layer " +
                               std::to_string(layer) + " cannot place expert " +
                               std::to_string(p.first) + " in slot " +
                               std::to_string(p.second));
    }
    for (unsigned int e : pinned)
      if (l.slot_of[e] == kNone && to.count(e) == 0)
        throw std::logic_error("ExpertResidency: layer " +
                               std::to_string(layer) + " needs expert " +
                               std::to_string(e) +
                               ", neither resident nor "
                               "placed");
    Acquired out;
    out.slot.resize(need.size(), kNone);
    ++l.clock;
    for (unsigned int e : pinned) {
      if (policy_ != Policy::lru) {
        l.score[e] = scoreNow(l, e) + 1.0;
        l.stamp[e] = l.clock;
      }
      l.last[e] = l.clock;
    }
    std::unordered_set<unsigned int> seen;
    for (size_t i = 0; i < need.size(); ++i) {
      const unsigned int e = need[i];
      const bool first = seen.insert(e).second;
      if (l.slot_of[e] != kNone) {
        if (first)
          ++out.hits;
        touch(l, e);
        out.slot[i] = l.slot_of[e];
        continue;
      }
      const unsigned int slot = to.at(e);
      const unsigned int victim = l.expert_in[slot];
      if (victim != kNone) {
        l.order.erase(l.where[victim]);
        l.where[victim] = l.order.end();
        l.slot_of[victim] = kNone;
      }
      l.expert_in[slot] = e;
      l.slot_of[e] = slot;
      l.order.push_back(e);
      l.where[e] = std::prev(l.order.end());
      out.slot[i] = slot;
      out.misses.push_back({e, slot, victim, false});
    }
    hits_ += out.hits;
    misses_ += out.misses.size();
    return out;
  }

  /**
   * @brief Replay a read ahead that put @a expert into @a slot of @a layer:
   *        the slot's expert (if any) is evicted and @a expert becomes
   *        resident there with no access counted, as the least recent
   *        expert (release() with admit).
   * @return the evicted expert, or kNone
   * @throw std::logic_error when @a expert is resident already or the slot
   *        is held or reserved (nothing is changed then)
   */
  unsigned int admit(unsigned int layer, unsigned int expert,
                     unsigned int slot) {
    std::lock_guard<std::mutex> lk(mu_);
    Layer &l = get(layer);
    if (expert >= l.slot_of.size())
      throw std::out_of_range("ExpertResidency: expert " +
                              std::to_string(expert) + " of layer " +
                              std::to_string(layer));
    checkSlot(l, layer, slot);
    if (l.slot_of[expert] != kNone || l.reserved_slot[expert] != kNone ||
        l.held[slot] != 0 || l.reserved_for[slot] != kNone)
      throw std::logic_error("ExpertResidency: layer " + std::to_string(layer) +
                             " cannot admit expert " + std::to_string(expert) +
                             " into slot " + std::to_string(slot));
    const unsigned int victim = l.expert_in[slot];
    if (victim != kNone) {
      l.order.erase(l.where[victim]);
      l.where[victim] = l.order.end();
      l.slot_of[victim] = kNone;
    }
    l.expert_in[slot] = expert;
    l.slot_of[expert] = slot;
    l.order.push_front(expert);
    l.where[expert] = l.order.begin();
    ++admitted_;
    return victim;
  }

  /** @brief pin @a slot of @a layer against eviction (nests) */
  void hold(unsigned int layer, unsigned int slot) {
    std::lock_guard<std::mutex> lk(mu_);
    Layer &l = get(layer);
    checkSlot(l, layer, slot);
    ++l.held[slot];
  }

  /** @brief undo one hold() of @a slot (no-op when not held) */
  void unhold(unsigned int layer, unsigned int slot) {
    std::lock_guard<std::mutex> lk(mu_);
    Layer &l = get(layer);
    checkSlot(l, layer, slot);
    if (l.held[slot] != 0)
      --l.held[slot];
  }

  /**
   * @brief Drop @a expert from @a layer (no-op when it is not resident): its
   *        slot is free again. A hold on the slot stays.
   */
  void forget(unsigned int layer, unsigned int expert) {
    std::lock_guard<std::mutex> lk(mu_);
    Layer &l = get(layer);
    if (expert >= l.slot_of.size() || l.slot_of[expert] == kNone)
      return;
    l.expert_in[l.slot_of[expert]] = kNone;
    l.slot_of[expert] = kNone;
    l.order.erase(l.where[expert]);
    l.where[expert] = l.order.end();
  }

  /**
   * @brief Forget every expert of @a layer: all its slots are free. Holds
   *        stay with their slots.
   */
  void clear(unsigned int layer) {
    std::lock_guard<std::mutex> lk(mu_);
    Layer &l = get(layer);
    std::fill(l.slot_of.begin(), l.slot_of.end(), kNone);
    std::fill(l.expert_in.begin(), l.expert_in.end(), kNone);
    std::fill(l.reserved_for.begin(), l.reserved_for.end(), kNone);
    std::fill(l.reserved_slot.begin(), l.reserved_slot.end(), kNone);
    l.order.clear();
    std::fill(l.where.begin(), l.where.end(), l.order.end());
  }

  /**
   * @brief Reserve a slot of @a layer for @a expert ahead of a call that
   *        may need it (see the file comment).
   * @param avoid experts not to evict (the prediction's other experts)
   * @param keep slots a call needs at once (the top-k): at most capacity -
   *        keep slots are reserved
   * @return the slot, or kNone: the expert is resident or reserved already,
   *         the reservations are at their limit, or every resident expert
   *         is held or avoided
   */
  unsigned int reserve(unsigned int layer, unsigned int expert,
                       const std::vector<unsigned int> &avoid,
                       unsigned int keep) {
    std::lock_guard<std::mutex> lk(mu_);
    Layer &l = get(layer);
    if (expert >= l.slot_of.size())
      throw std::out_of_range("ExpertResidency: expert " +
                              std::to_string(expert) + " of layer " +
                              std::to_string(layer));
    if (l.slot_of[expert] != kNone || l.reserved_slot[expert] != kNone)
      return kNone;
    unsigned int n = 0;
    for (unsigned int s = 0; s < l.capacity; ++s)
      n += l.reserved_for[s] != kNone;
    if (n + keep >= l.capacity)
      return kNone;
    unsigned int slot = kNone;
    for (unsigned int s = 0; s < l.capacity && slot == kNone; ++s)
      if (freeSlot(l, s))
        slot = s;
    if (slot == kNone) {
      std::unordered_set<unsigned int> pinned(avoid.begin(), avoid.end());
      pinned.insert(expert);
      const unsigned int victim = pickVictim(l, pinned);
      if (victim == kNone)
        return kNone;
      slot = l.slot_of[victim];
      l.order.erase(l.where[victim]);
      l.where[victim] = l.order.end();
      l.slot_of[victim] = kNone;
      l.expert_in[slot] = kNone;
      ++reserve_evictions_;
    }
    l.reserved_for[slot] = expert;
    l.reserved_slot[expert] = slot;
    ++reserves_;
    return slot;
  }

  /**
   * @brief End @a expert's reservation in @a layer without a call using it
   *        (no-op when it has none, e.g. an acquire took it or clear()
   *        dropped it): the slot becomes free, or with @a admit (the bytes
   *        are in the slot) the expert becomes resident there with no
   *        access counted, as the least recent expert.
   */
  void release(unsigned int layer, unsigned int expert, bool admit) {
    std::lock_guard<std::mutex> lk(mu_);
    Layer &l = get(layer);
    if (expert >= l.slot_of.size() || l.reserved_slot[expert] == kNone)
      return;
    const unsigned int slot = l.reserved_slot[expert];
    l.reserved_for[slot] = kNone;
    l.reserved_slot[expert] = kNone;
    if (admit) {
      l.expert_in[slot] = expert;
      l.slot_of[expert] = slot;
      l.order.push_front(expert);
      l.where[expert] = l.order.begin();
      ++admitted_;
    } else {
      ++released_;
    }
  }

  /** @brief the slot reserved for @a expert, or kNone */
  unsigned int reservedSlot(unsigned int layer, unsigned int expert) const {
    std::lock_guard<std::mutex> lk(mu_);
    const Layer &l = get(layer);
    return expert < l.reserved_slot.size() ? l.reserved_slot[expert] : kNone;
  }

  /** @brief slots of @a layer reserved now */
  unsigned int reserved(unsigned int layer) const {
    std::lock_guard<std::mutex> lk(mu_);
    const Layer &l = get(layer);
    unsigned int n = 0;
    for (unsigned int x : l.reserved_for)
      n += x != kNone;
    return n;
  }

  /** @brief reservation totals: made, evictions they caused, used by an
   *         acquire, released free, admitted */
  struct ReserveStats {
    uint64_t reserves = 0, evictions = 0, used = 0, released = 0, admitted = 0;
  };
  ReserveStats reserveStats() const {
    std::lock_guard<std::mutex> lk(mu_);
    return {reserves_, reserve_evictions_, reserved_used_, released_,
            admitted_};
  }

  bool held(unsigned int layer, unsigned int slot) const {
    std::lock_guard<std::mutex> lk(mu_);
    const Layer &l = get(layer);
    checkSlot(l, layer, slot);
    return l.held[slot] != 0;
  }

  /** @brief move the resident experts of @a recency to the most recent end,
   *         in order; others are skipped */
  void refresh(unsigned int layer, const std::vector<unsigned int> &recency) {
    std::lock_guard<std::mutex> lk(mu_);
    Layer &l = get(layer);
    for (unsigned int e : recency)
      if (e < l.slot_of.size() && l.slot_of[e] != kNone)
        touch(l, e);
  }

  /** @brief resident experts of @a layer, least to most recent (tests) */
  std::vector<unsigned int> order(unsigned int layer) const {
    std::lock_guard<std::mutex> lk(mu_);
    const Layer &l = get(layer);
    return std::vector<unsigned int>(l.order.begin(), l.order.end());
  }

  /** @brief every expert of @a layer resident (C == E and all read) */
  bool complete(unsigned int layer) const {
    std::lock_guard<std::mutex> lk(mu_);
    const Layer &l = get(layer);
    return l.order.size() == l.slot_of.size();
  }

  uint64_t hits() const {
    std::lock_guard<std::mutex> lk(mu_);
    return hits_;
  }
  uint64_t misses() const {
    std::lock_guard<std::mutex> lk(mu_);
    return misses_;
  }

private:
  struct Layer {
    unsigned int capacity = 0;
    std::vector<unsigned int> slot_of;       /**< expert -> slot */
    std::vector<unsigned int> expert_in;     /**< slot -> expert */
    std::vector<unsigned int> held;          /**< slot -> hold count */
    std::vector<unsigned int> reserved_for;  /**< slot -> expert, or kNone */
    std::vector<unsigned int> reserved_slot; /**< expert -> slot, or kNone */
    std::list<unsigned int> order;           /**< front = least recent */
    std::vector<std::list<unsigned int>::iterator> where; /**< in order */
    std::vector<double> score;   /**< LFU count / LRFU CRF at stamp */
    std::vector<uint64_t> stamp; /**< clock of score */
    std::vector<uint64_t> last;  /**< clock of the last access */
    uint64_t clock = 0;          /**< acquires so far */
  };

  /** LFU count or LRFU CRF of @a e now */
  double scoreNow(const Layer &l, unsigned int e) const {
    if (policy_ != Policy::lrfu)
      return l.score[e];
    return l.score[e] * std::exp2(-lambda_ * (double)(l.clock - l.stamp[e]));
  }

  /**
   * @brief The victim among the resident experts outside @a pinned and not
   *        held, or kNone: LRU the least recent; LFU / LRFU the smallest
   *        score, ties to the least recently accessed, then the least
   *        recent (the order list).
   */
  unsigned int
  pickVictim(const Layer &l,
             const std::unordered_set<unsigned int> &pinned) const {
    unsigned int best = kNone;
    double best_score = 0.0;
    uint64_t best_last = 0;
    for (auto it = l.order.begin(); it != l.order.end(); ++it) {
      const unsigned int cand = *it;
      if (pinned.count(cand) != 0 || l.held[l.slot_of[cand]] != 0)
        continue;
      if (policy_ == Policy::lru)
        return cand;
      const double sc = scoreNow(l, cand);
      if (best == kNone || sc < best_score ||
          (sc == best_score && l.last[cand] < best_last)) {
        best = cand;
        best_score = sc;
        best_last = l.last[cand];
      }
    }
    return best;
  }

  const Layer &get(unsigned int layer) const {
    auto it = layers_.find(layer);
    if (it == layers_.end())
      throw std::out_of_range("ExpertResidency: unknown layer " +
                              std::to_string(layer));
    return it->second;
  }
  Layer &get(unsigned int layer) {
    return const_cast<Layer &>(
      static_cast<const ExpertResidency *>(this)->get(layer));
  }
  static void checkSlot(const Layer &l, unsigned int layer, unsigned int slot) {
    if (slot >= l.capacity)
      throw std::out_of_range("ExpertResidency: slot " + std::to_string(slot) +
                              " of layer " + std::to_string(layer));
  }
  /** @a s holds no expert and is neither held nor reserved */
  static bool freeSlot(const Layer &l, unsigned int s) {
    return l.expert_in[s] == kNone && l.held[s] == 0 &&
           l.reserved_for[s] == kNone;
  }
  static void touch(Layer &l, unsigned int e) {
    l.order.splice(l.order.end(), l.order, l.where[e]);
    l.where[e] = std::prev(l.order.end());
  }

  mutable std::mutex mu_;
  std::unordered_map<unsigned int, Layer> layers_;
  uint64_t hits_ = 0, misses_ = 0;
  uint64_t reserves_ = 0, reserve_evictions_ = 0, reserved_used_ = 0,
           released_ = 0, admitted_ = 0;
  Policy policy_ = Policy::lru;
  double lambda_ = 0.0;
};

} // namespace causallm

#endif // __CAUSALLM_EXPERT_RESIDENCY_H__
