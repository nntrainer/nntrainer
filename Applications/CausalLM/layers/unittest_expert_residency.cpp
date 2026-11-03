// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   unittest_expert_residency.cpp
 * @date   08 October 2026
 * @brief  Host tests for the per-layer expert slot residency
 *         (expert_residency.h)
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * @details The residency decides which device slot an expert's bytes go to.
 * A wrong answer is a kernel reading another expert's weights (a wrong token)
 * or a read overwriting a slot a queued kernel still uses, so these tests pin
 * the rules: free slots first, LRU victims outside the call and outside the
 * holds, recency order, partitions independent per layer, and the throws.
 * No GPU needed.
 */

#include <expert_residency.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

using causallm::ExpertResidency;

namespace {

/** slots of @a a are distinct and inside the capacity */
void expectDistinctSlots(const ExpertResidency::Acquired &a, unsigned int cap) {
  std::set<unsigned int> s(a.slot.begin(), a.slot.end());
  EXPECT_EQ(s.size(), a.slot.size());
  for (unsigned int x : a.slot)
    EXPECT_LT(x, cap);
}

/**
 * The misses of @a need (ascending ids) placed the way a device does it from
 * victimOrder(): each miss takes the first slot of the order it has not
 * taken whose expert the call does not need.
 */
std::vector<std::pair<unsigned int, unsigned int>>
placeFromOrder(const ExpertResidency &r, unsigned int layer,
               const std::vector<unsigned int> &need,
               const std::vector<unsigned int> &order) {
  const std::vector<unsigned int> map = r.slotMap(layer);
  std::vector<unsigned int> in(r.capacity(layer), ExpertResidency::kNone);
  for (unsigned int e = 0; e < map.size(); ++e)
    if (map[e] != ExpertResidency::kNone)
      in[map[e]] = e;
  std::vector<bool> taken(order.size(), false);
  std::vector<std::pair<unsigned int, unsigned int>> placed;
  for (unsigned int e : need) {
    if (map[e] != ExpertResidency::kNone)
      continue;
    for (size_t i = 0; i < order.size(); ++i) {
      const unsigned int occ = in[order[i]];
      if (taken[i] || (occ != ExpertResidency::kNone &&
                       std::find(need.begin(), need.end(), occ) != need.end()))
        continue;
      taken[i] = true;
      placed.emplace_back(e, order[i]);
      break;
    }
  }
  return placed;
}

} // namespace

TEST(ExpertResidency, FirstAcquireFillsFreeSlotsInOrder) {
  ExpertResidency r;
  r.addLayer(0, 4, 16);
  auto a = r.acquire(0, {7, 3, 9});
  EXPECT_EQ(a.hits, 0u);
  ASSERT_EQ(a.misses.size(), 3u);
  EXPECT_EQ(a.slot, (std::vector<unsigned int>{0, 1, 2}));
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(a.misses[i].slot, a.slot[i]);
    EXPECT_EQ(a.misses[i].victim, ExpertResidency::kNone);
  }
  EXPECT_EQ(a.misses[0].expert, 7u);
  EXPECT_EQ(r.slotOf(0, 3), 1u);
  EXPECT_EQ(r.slotOf(0, 5), ExpertResidency::kNone);
  EXPECT_EQ(r.order(0), (std::vector<unsigned int>{7, 3, 9}));
}

TEST(ExpertResidency, HitsKeepTheirSlotAndBecomeRecent) {
  ExpertResidency r;
  r.addLayer(0, 4, 16);
  r.acquire(0, {1, 2, 3, 4});
  auto a = r.acquire(0, {2, 4});
  EXPECT_EQ(a.hits, 2u);
  EXPECT_TRUE(a.misses.empty());
  EXPECT_EQ(a.slot, (std::vector<unsigned int>{1, 3}));
  EXPECT_EQ(r.order(0), (std::vector<unsigned int>{1, 3, 2, 4}));
}

TEST(ExpertResidency, MissEvictsLeastRecentOutsideTheCall) {
  ExpertResidency r;
  r.addLayer(0, 4, 16);
  r.acquire(0, {1, 2, 3, 4}); // order 1 2 3 4
  // 1 is least recent but needed: the victim must be 2
  auto a = r.acquire(0, {1, 10});
  ASSERT_EQ(a.misses.size(), 1u);
  EXPECT_EQ(a.misses[0].expert, 10u);
  EXPECT_EQ(a.misses[0].victim, 2u);
  EXPECT_EQ(a.misses[0].slot, 1u); // 2's slot
  EXPECT_EQ(r.slotOf(0, 2), ExpertResidency::kNone);
  EXPECT_EQ(r.slotOf(0, 10), 1u);
  EXPECT_EQ(r.order(0), (std::vector<unsigned int>{3, 4, 1, 10}));
}

TEST(ExpertResidency, HeldSlotIsNeverAVictim) {
  ExpertResidency r;
  r.addLayer(0, 3, 32);
  r.acquire(0, {1, 2, 3}); // slots 0 1 2
  r.hold(0, 0);            // 1's read still in flight
  auto a = r.acquire(0, {20});
  ASSERT_EQ(a.misses.size(), 1u);
  EXPECT_EQ(a.misses[0].victim, 2u); // not 1 (held)
  EXPECT_EQ(a.misses[0].slot, 1u);
  EXPECT_TRUE(r.held(0, 0));
  r.unhold(0, 0);
  EXPECT_FALSE(r.held(0, 0));
  auto b = r.acquire(0, {21});
  EXPECT_EQ(b.misses[0].victim, 1u);
}

TEST(ExpertResidency, HoldsNest) {
  ExpertResidency r;
  r.addLayer(0, 2, 8);
  r.acquire(0, {0, 1});
  r.hold(0, 0);
  r.hold(0, 0);
  r.unhold(0, 0);
  EXPECT_TRUE(r.held(0, 0));
  r.unhold(0, 0);
  EXPECT_FALSE(r.held(0, 0));
  r.unhold(0, 0); // no-op
  EXPECT_FALSE(r.held(0, 0));
}

TEST(ExpertResidency, TooManyAtOnceThrowsAndChangesNothing) {
  ExpertResidency r;
  r.addLayer(0, 4, 16);
  r.acquire(0, {1, 2});
  EXPECT_THROW(r.acquire(0, {1, 2, 3, 4, 5}), std::runtime_error);
  EXPECT_EQ(r.order(0), (std::vector<unsigned int>{1, 2}));
  // holds on other experts' slots shrink the room
  r.acquire(0, {3, 4});
  r.hold(0, r.slotOf(0, 3));
  r.hold(0, r.slotOf(0, 4));
  EXPECT_THROW(r.acquire(0, {5, 6, 7}), std::runtime_error);
  EXPECT_NO_THROW(r.acquire(0, {5, 6}));
}

TEST(ExpertResidency, DuplicatesInOneCallShareTheSlot) {
  ExpertResidency r;
  r.addLayer(0, 2, 8);
  auto a = r.acquire(0, {5, 5, 6});
  EXPECT_EQ(a.slot[0], a.slot[1]);
  EXPECT_EQ(a.misses.size(), 2u);
  EXPECT_EQ(a.hits, 0u);
}

TEST(ExpertResidency, RefreshReordersResidentOnly) {
  ExpertResidency r;
  r.addLayer(0, 4, 16);
  r.acquire(0, {1, 2, 3, 4});
  r.refresh(0, {2, 3, 1}); // the last one ends up most recent
  EXPECT_EQ(r.order(0), (std::vector<unsigned int>{4, 2, 3, 1}));
  r.refresh(0, {15}); // not resident: skipped
  EXPECT_EQ(r.order(0), (std::vector<unsigned int>{4, 2, 3, 1}));
  auto a = r.acquire(0, {8});
  EXPECT_EQ(a.misses[0].victim, 4u);
}

TEST(ExpertResidency, LayersAreIndependentAndMayDifferInSize) {
  ExpertResidency r;
  r.addLayer(0, 2, 8);
  r.addLayer(1, 5, 8);
  auto a0 = r.acquire(0, {1, 2});
  auto a1 = r.acquire(1, {1, 2, 3, 4, 5});
  expectDistinctSlots(a0, 2);
  expectDistinctSlots(a1, 5);
  r.acquire(0, {3}); // evicts in layer 0 only
  EXPECT_EQ(r.slotOf(0, 1), ExpertResidency::kNone);
  EXPECT_EQ(r.slotOf(1, 1), 0u);
  EXPECT_EQ(r.capacity(0), 2u);
  EXPECT_EQ(r.capacity(1), 5u);
  EXPECT_EQ(r.size(1), 5u);
}

TEST(ExpertResidency, CompleteWhenEveryExpertIsResident) {
  ExpertResidency r;
  r.addLayer(3, 4, 4);
  r.acquire(3, {2, 0});
  EXPECT_FALSE(r.complete(3));
  r.acquire(3, {1, 3});
  EXPECT_TRUE(r.complete(3));
  auto m = r.slotMap(3);
  EXPECT_EQ(m, (std::vector<unsigned int>{1, 2, 0, 3}));
}

TEST(ExpertResidency, CountsHitsAndMisses) {
  ExpertResidency r;
  r.addLayer(0, 2, 8);
  r.acquire(0, {1, 2});
  r.acquire(0, {1, 3});
  EXPECT_EQ(r.hits(), 1u);
  EXPECT_EQ(r.misses(), 3u);
}

TEST(ExpertResidency, Deterministic) {
  auto run = []() {
    ExpertResidency r;
    r.addLayer(0, 8, 128);
    std::vector<unsigned int> slots;
    unsigned int x = 12345;
    for (int step = 0; step < 200; ++step) {
      std::vector<unsigned int> need;
      for (int k = 0; k < 8; ++k) {
        x = x * 1103515245u + 12345u;
        unsigned int e = (x >> 8) % 128;
        if (std::find(need.begin(), need.end(), e) == need.end())
          need.push_back(e);
      }
      auto a = r.acquire(0, need);
      slots.insert(slots.end(), a.slot.begin(), a.slot.end());
      r.refresh(0, need);
    }
    return slots;
  };
  EXPECT_EQ(run(), run());
}

TEST(ExpertResidency, ForgetFreesTheSlotForTheNextMiss) {
  ExpertResidency r;
  r.addLayer(0, 2, 8);
  r.acquire(0, {4, 5});
  r.hold(0, 0); // the in-flight read of expert 4 failed: forgotten, held
  r.forget(0, 4);
  EXPECT_EQ(r.slotOf(0, 4), ExpertResidency::kNone);
  EXPECT_EQ(r.size(0), 1u);
  EXPECT_EQ(r.order(0), (std::vector<unsigned int>{5}));
  EXPECT_TRUE(r.held(0, 0)); // the hold stays with the slot
  r.unhold(0, 0);
  // the freed slot takes the next miss; 5 stays where it is
  auto a = r.acquire(0, {6});
  ASSERT_EQ(a.misses.size(), 1u);
  EXPECT_EQ(a.misses[0].slot, 0u);
  EXPECT_EQ(a.misses[0].victim, ExpertResidency::kNone);
  EXPECT_EQ(r.slotOf(0, 5), 1u);
  // forgetting what is not resident (or out of range) is a no-op
  r.forget(0, 4);
  r.forget(0, 100);
  EXPECT_EQ(r.size(0), 2u);
}

TEST(ExpertResidency, ClearEmptiesOneLayerOnly) {
  ExpertResidency r;
  r.addLayer(0, 3, 8);
  r.addLayer(1, 3, 8);
  r.acquire(0, {1, 2, 3});
  r.acquire(1, {4});
  r.clear(0);
  EXPECT_EQ(r.size(0), 0u);
  EXPECT_TRUE(r.order(0).empty());
  for (unsigned int e = 0; e < 8; ++e)
    EXPECT_EQ(r.slotOf(0, e), ExpertResidency::kNone);
  EXPECT_EQ(r.slotOf(1, 4), 0u);
  // the cleared layer fills from slot 0 again, with no victims
  auto a = r.acquire(0, {3, 7});
  ASSERT_EQ(a.misses.size(), 2u);
  EXPECT_EQ(a.slot, (std::vector<unsigned int>{0, 1}));
  EXPECT_EQ(a.misses[0].victim, ExpertResidency::kNone);
  EXPECT_EQ(a.misses[1].victim, ExpertResidency::kNone);
}

// ---- LFU / LRFU victim policy ----

TEST(ExpertResidency, LrfuAndLfuKeepTheFrequentOverTheRecentOnce) {
  ExpertResidency lru, lfu, f;
  lfu.setPolicy(ExpertResidency::Policy::lfu);
  f.setPolicy(ExpertResidency::Policy::lrfu, 0.1);
  for (ExpertResidency *r : {&lru, &lfu, &f}) {
    r->addLayer(0, 3, 16);
    // 1 is used three steps running, then 2 once, then 3 now
    r->acquire(0, {1});
    r->acquire(0, {1});
    r->acquire(0, {1});
    r->acquire(0, {2});
    r->acquire(0, {3});
  }
  auto a = lru.acquire(0, {9});
  ASSERT_EQ(a.misses.size(), 1u);
  EXPECT_EQ(a.misses[0].victim, 1u); // LRU: the least recent
  for (ExpertResidency *r : {&lfu, &f}) {
    a = r->acquire(0, {9});
    ASSERT_EQ(a.misses.size(), 1u);
    EXPECT_EQ(a.misses[0].victim, 2u); // 1 has the larger count / CRF
  }
  EXPECT_DOUBLE_EQ(f.lambda(), 0.1);
  EXPECT_EQ(f.policy(), ExpertResidency::Policy::lrfu);
  EXPECT_EQ(lru.policy(), ExpertResidency::Policy::lru);
  EXPECT_THROW(f.setPolicy(ExpertResidency::Policy::lrfu, 0.0),
               std::invalid_argument);
}

TEST(ExpertResidency, LrfuForgetsOldFrequencyLfuDoesNot) {
  ExpertResidency lfu, f;
  lfu.setPolicy(ExpertResidency::Policy::lfu);
  f.setPolicy(ExpertResidency::Policy::lrfu, 0.5);
  for (ExpertResidency *r : {&lfu, &f}) {
    r->addLayer(0, 3, 16);
    for (int i = 0; i < 4; ++i)
      r->acquire(0, {1}); // 1: four accesses, long ago
    for (int i = 0; i < 12; ++i)
      r->acquire(0, {5}); // 5: hot now
    r->acquire(0, {2});   // 2: once, just now (the last free slot)
  }
  // a miss must evict 1 or 2 ({5} is not needed now but it is the hottest):
  // LFU keeps 1 (4 accesses vs 1), LRFU with a fast decay keeps 2
  auto a = lfu.acquire(0, {7, 5});
  ASSERT_EQ(a.misses.size(), 1u);
  EXPECT_EQ(a.misses[0].victim, 2u);
  a = f.acquire(0, {7, 5});
  ASSERT_EQ(a.misses.size(), 1u);
  EXPECT_EQ(a.misses[0].victim, 1u);
}

TEST(ExpertResidency, PolicyNeverEvictsAHeldOrNeededExpert) {
  for (auto p : {ExpertResidency::Policy::lru, ExpertResidency::Policy::lfu,
                 ExpertResidency::Policy::lrfu}) {
    ExpertResidency r;
    r.setPolicy(p, 0.1);
    r.addLayer(0, 3, 16);
    r.acquire(0, {4});
    r.acquire(0, {6});
    r.acquire(0, {8});
    r.acquire(0, {8}); // 4 is the coldest by any measure
    r.hold(0, r.slotOf(0, 4));
    auto a = r.acquire(0, {6, 11});
    ASSERT_EQ(a.misses.size(), 1u);
    EXPECT_EQ(a.misses[0].victim, 8u); // 4 is held, 6 needed
  }
}

TEST(ExpertResidency, PrimeSeedsTheHistoryOnly) {
  ExpertResidency r, plain;
  r.setPolicy(ExpertResidency::Policy::lrfu, 0.1);
  plain.setPolicy(ExpertResidency::Policy::lrfu, 0.1);
  for (ExpertResidency *x : {&r, &plain})
    x->addLayer(0, 2, 16);
  std::vector<double> w(16, 0.0);
  w[3] = 8.0; // the prompt used 3 a lot
  r.prime(0, w);
  EXPECT_EQ(r.size(0), 0u); // nothing loaded
  for (ExpertResidency *x : {&r, &plain}) {
    x->acquire(0, {3});
    x->acquire(0, {9});
  }
  // 3 is the least recent; primed, it outweighs 9
  auto a = r.acquire(0, {12});
  ASSERT_EQ(a.misses.size(), 1u);
  EXPECT_EQ(a.misses[0].victim, 9u);
  a = plain.acquire(0, {12});
  ASSERT_EQ(a.misses.size(), 1u);
  EXPECT_EQ(a.misses[0].victim, 3u);
  // LRU ignores priming
  ExpertResidency l;
  l.addLayer(0, 2, 16);
  l.prime(0, w);
  l.acquire(0, {3});
  l.acquire(0, {9});
  a = l.acquire(0, {12});
  ASSERT_EQ(a.misses.size(), 1u);
  EXPECT_EQ(a.misses[0].victim, 3u);
}

TEST(ExpertResidency, BadArgumentsThrow) {
  ExpertResidency r;
  EXPECT_THROW(r.addLayer(0, 0, 8), std::invalid_argument);
  r.addLayer(0, 4, 8);
  EXPECT_NO_THROW(r.addLayer(0, 4, 8));
  EXPECT_THROW(r.addLayer(0, 5, 8), std::invalid_argument);
  EXPECT_THROW(r.acquire(0, {8}), std::out_of_range);
  EXPECT_THROW(r.acquire(1, {0}), std::out_of_range);
  EXPECT_THROW(r.hold(0, 4), std::out_of_range);
  EXPECT_FALSE(r.hasLayer(1));
}

TEST(ExpertResidency, AReservedSlotIsNeverEvictedNorGivenToAnother) {
  for (auto p : {ExpertResidency::Policy::lru, ExpertResidency::Policy::lfu,
                 ExpertResidency::Policy::lrfu}) {
    ExpertResidency r;
    r.setPolicy(p, 0.1);
    r.addLayer(0, 4, 32);
    r.acquire(0, {1, 2, 3}); // slots 0 1 2, slot 3 free
    // a free slot is taken first: nothing is evicted
    EXPECT_EQ(r.reserve(0, 20, {}, 2), 3u);
    EXPECT_EQ(r.reservedSlot(0, 20), 3u);
    EXPECT_EQ(r.slotOf(0, 20), ExpertResidency::kNone); // not resident
    EXPECT_EQ(r.reserveStats().evictions, 0u);
    // many misses of other experts, one at a time: slot 3 is never chosen
    for (unsigned int e = 4; e < 32; ++e) {
      if (e == 20)
        continue;
      auto a = r.acquire(0, {e});
      ASSERT_EQ(a.misses.size(), 1u);
      EXPECT_NE(a.misses[0].slot, 3u);
      EXPECT_FALSE(a.misses[0].reserved);
    }
    EXPECT_EQ(r.reservedSlot(0, 20), 3u);
    // a call that needs the reserved slot's room as well as every other slot
    // does not fit: the reservation is not taken away
    EXPECT_THROW(r.acquire(0, {4, 5, 6, 7}), std::runtime_error);
    EXPECT_EQ(r.reservedSlot(0, 20), 3u);
  }
}

TEST(ExpertResidency, AReservationCountsNoAccessUntilUsed) {
  // LRFU: a reserved expert gains no frequency and no recency; the call that
  // needs it makes it resident in its slot and counts that access
  ExpertResidency r, ref;
  for (ExpertResidency *x : {&r, &ref}) {
    x->setPolicy(ExpertResidency::Policy::lrfu, 0.1);
    x->addLayer(0, 3, 16);
    x->acquire(0, {1, 2, 3});
  }
  // reserve 9 three times over (reserve, release, reserve, ...): no access
  for (int i = 0; i < 3; ++i) {
    const unsigned int s = r.reserve(0, 9, {1}, 1);
    ASSERT_NE(s, ExpertResidency::kNone);
    r.release(0, 9, /*admit=*/false); // freed: the slot is empty now
    EXPECT_EQ(r.slotOf(0, 9), ExpertResidency::kNone);
    EXPECT_EQ(r.reserved(0), 0u);
  }
  EXPECT_EQ(r.hits(), ref.hits());
  EXPECT_EQ(r.misses(), ref.misses());
  // the first reserve evicted the coldest outside {1} (2, the least recent
  // of 2 and 3 with equal history); later ones took the freed slot
  EXPECT_EQ(r.reserveStats().evictions, 1u);
  EXPECT_EQ(r.reserveStats().released, 3u);
  // 9 is then needed: a miss into the reserved slot, flagged
  const unsigned int s = r.reserve(0, 9, {}, 1);
  auto a = r.acquire(0, {9});
  ASSERT_EQ(a.misses.size(), 1u);
  EXPECT_TRUE(a.misses[0].reserved);
  EXPECT_EQ(a.misses[0].slot, s);
  EXPECT_EQ(a.misses[0].victim, ExpertResidency::kNone);
  EXPECT_EQ(r.slotOf(0, 9), s);
  EXPECT_EQ(r.reservedSlot(0, 9), ExpertResidency::kNone);
  EXPECT_EQ(r.order(0).back(), 9u); // used: the most recent
  EXPECT_EQ(r.reserveStats().used, 1u);
}

TEST(ExpertResidency, ReleaseFreesOrAdmitsWithoutAnAccess) {
  ExpertResidency r;
  r.setPolicy(ExpertResidency::Policy::lfu);
  r.addLayer(0, 3, 16);
  r.acquire(0, {1});
  r.acquire(0, {1});
  r.acquire(0, {2, 3}); // 1 has 2 accesses, 2 and 3 one each
  const unsigned int s = r.reserve(0, 7, {}, 1);
  ASSERT_NE(s, ExpertResidency::kNone);
  EXPECT_EQ(r.reserveStats().evictions, 1u); // 2: fewest, least recent
  EXPECT_EQ(r.slotOf(0, 2), ExpertResidency::kNone);
  // admitted: resident in its slot, the least recent, no access counted --
  // LFU therefore evicts it first (0 accesses)
  r.release(0, 7, /*admit=*/true);
  EXPECT_EQ(r.slotOf(0, 7), s);
  EXPECT_EQ(r.order(0).front(), 7u);
  auto a = r.acquire(0, {11});
  ASSERT_EQ(a.misses.size(), 1u);
  EXPECT_EQ(a.misses[0].victim, 7u);
  // a release of nothing reserved is a no-op
  r.release(0, 7, true);
  r.release(0, 12, false);
  EXPECT_EQ(r.reserveStats().admitted, 1u);
}

TEST(ExpertResidency, ReservationsLeaveRoomForACallAndAvoidThePrediction) {
  ExpertResidency r;
  r.addLayer(0, 4, 32);
  r.acquire(0, {1, 2, 3, 4}); // LRU order 1 2 3 4
  // keep 2: at most 2 reservations
  EXPECT_EQ(r.reserve(0, 10, {1, 2, 10, 11}, 2), 2u);         // 1, 2 avoided: 3
  EXPECT_EQ(r.reserve(0, 11, {1, 2, 10, 11}, 2), 3u);         // then 4
  EXPECT_EQ(r.reserve(0, 12, {}, 2), ExpertResidency::kNone); // limit
  EXPECT_EQ(r.reserve(0, 10, {}, 0), ExpertResidency::kNone); // reserved
  EXPECT_EQ(r.reserve(0, 1, {}, 0), ExpertResidency::kNone);  // resident
  // a 2-expert call still fits next to the 2 reservations
  auto a = r.acquire(0, {20, 21});
  EXPECT_EQ(a.misses.size(), 2u);
  for (const auto &m : a.misses) {
    EXPECT_NE(m.slot, 2u);
    EXPECT_NE(m.slot, 3u);
  }
  // clear drops the reservations with the residents
  r.clear(0);
  EXPECT_EQ(r.reserved(0), 0u);
  EXPECT_EQ(r.reservedSlot(0, 10), ExpertResidency::kNone);
  r.release(0, 10, true); // no-op after clear
  EXPECT_EQ(r.slotOf(0, 10), ExpertResidency::kNone);
}

TEST(ExpertResidency, VictimOrderListsFreeSlotsThenThePolicysVictims) {
  ExpertResidency r;
  r.addLayer(0, 5, 32);
  r.acquire(0, {4, 5, 6}); // slots 0 1 2, LRU order 4 5 6
  r.acquire(0, {4});       // order 5 6 4
  EXPECT_EQ(r.victimOrder(0), (std::vector<unsigned int>{3, 4, 1, 2, 0}));
  EXPECT_EQ(r.victimOrder(0, {4, 1}), (std::vector<unsigned int>{3, 2, 0}));
  r.hold(0, 2);
  EXPECT_EQ(r.victimOrder(0), (std::vector<unsigned int>{3, 4, 1, 0}));
  r.unhold(0, 2);
  EXPECT_EQ(r.reserve(0, 9, {}, 0), 3u); // a free slot, now reserved
  EXPECT_EQ(r.victimOrder(0), (std::vector<unsigned int>{4, 1, 2, 0}));
}

TEST(ExpertResidency, PlaceFromVictimOrderReplaysAcquire) {
  // A device that places its own misses from victimOrder() and a host that
  // replays them with place() end where acquire() ends, step by step: the
  // same slots, the same recency, the same history (LRU, LFU, LRFU with an
  // exact decay and with a rounding one), from cold and from a primed state.
  struct Arm {
    ExpertResidency::Policy p;
    double lambda;
    bool prime;
  };
  const Arm arms[] = {{ExpertResidency::Policy::lru, 0.1, false},
                      {ExpertResidency::Policy::lfu, 0.1, false},
                      {ExpertResidency::Policy::lrfu, 1.0, false},
                      {ExpertResidency::Policy::lrfu, 0.1, true}};
  for (const Arm &arm : arms) {
    ExpertResidency a, b;
    for (ExpertResidency *r : {&a, &b}) {
      r->addLayer(0, 8, 40);
      r->setPolicy(arm.p, arm.lambda);
      if (arm.prime) {
        std::vector<double> w(40, 0.0);
        for (unsigned int e = 0; e < 40; e += 3)
          w[e] = 0.25 * (e % 7);
        r->prime(0, w);
      }
    }
    std::mt19937 rng(1234);
    std::uniform_int_distribution<unsigned int> pick(0, 39);
    unsigned int misses = 0;
    for (int step = 0; step < 400; ++step) {
      std::set<unsigned int> s;
      // a skewed draw: low ids often, so hits and evictions both occur
      while (s.size() < 4)
        s.insert(std::min(pick(rng), pick(rng)));
      const std::vector<unsigned int> need(s.begin(), s.end());
      const auto placed = placeFromOrder(b, 0, need, b.victimOrder(0));
      const auto ra = a.acquire(0, need);
      const auto rb = b.place(0, need, placed);
      misses += (unsigned int)ra.misses.size();
      ASSERT_EQ(ra.slot, rb.slot) << "step " << step;
      ASSERT_EQ(ra.hits, rb.hits) << "step " << step;
      ASSERT_EQ(a.slotMap(0), b.slotMap(0)) << "step " << step;
      ASSERT_EQ(a.order(0), b.order(0)) << "step " << step;
    }
    EXPECT_GT(misses, 100u);
    EXPECT_EQ(a.hits(), b.hits());
    EXPECT_EQ(a.misses(), b.misses());
  }
}

TEST(ExpertResidency, AdmitIsAnUnaccessedLeastRecentResident) {
  ExpertResidency r;
  r.addLayer(0, 3, 16);
  r.acquire(0, {1, 2, 3});         // order 1 2 3
  EXPECT_EQ(r.admit(0, 9, 1), 2u); // evicts 2 from slot 1
  EXPECT_EQ(r.slotOf(0, 9), 1u);
  EXPECT_EQ(r.slotOf(0, 2), ExpertResidency::kNone);
  EXPECT_EQ(r.order(0), (std::vector<unsigned int>{9, 1, 3}));
  EXPECT_EQ(r.victimOrder(0).front(), 1u); // 9 is the first victim
  // a later call that needs it is a hit
  auto a = r.acquire(0, {9});
  EXPECT_EQ(a.hits, 1u);
  EXPECT_EQ(r.order(0), (std::vector<unsigned int>{1, 3, 9}));
  // into a free slot: no victim
  r.forget(0, 3);
  EXPECT_EQ(r.admit(0, 7, 2), ExpertResidency::kNone);
  EXPECT_THROW(r.admit(0, 7, 0), std::logic_error); // resident already
  r.hold(0, 0);
  EXPECT_THROW(r.admit(0, 8, 0), std::logic_error); // held
  r.unhold(0, 0);
  EXPECT_THROW(r.admit(0, 8, 3), std::out_of_range);
}

TEST(ExpertResidency, PlaceRejectsAReplayThatContradictsTheState) {
  ExpertResidency r;
  r.addLayer(0, 4, 16);
  r.acquire(0, {1, 2, 3, 4});
  const auto map = r.slotMap(0);
  const auto ord = r.order(0);
  using P = std::vector<std::pair<unsigned int, unsigned int>>;
  // a resident expert placed again
  EXPECT_THROW(r.place(0, {1, 5}, P{{1, 0}, {5, 1}}), std::logic_error);
  // into the slot of an expert the call needs
  EXPECT_THROW(r.place(0, {1, 5}, P{{5, map[1]}}), std::logic_error);
  // a needed expert neither resident nor placed
  EXPECT_THROW(r.place(0, {5, 6}, P{{5, 0}}), std::logic_error);
  // one slot twice
  EXPECT_THROW(r.place(0, {5, 6}, P{{5, 0}, {6, 0}}), std::logic_error);
  // a placed expert the call does not need
  EXPECT_THROW(r.place(0, {5}, P{{5, 0}, {6, 1}}), std::logic_error);
  EXPECT_EQ(r.slotMap(0), map);
  EXPECT_EQ(r.order(0), ord);
  // a consistent one
  auto a = r.place(0, {2, 5}, P{{5, map[1]}});
  EXPECT_EQ(a.hits, 1u);
  ASSERT_EQ(a.misses.size(), 1u);
  EXPECT_EQ(a.misses[0].victim, 1u);
  EXPECT_EQ(r.slotOf(0, 5), map[1]);
  EXPECT_EQ(r.order(0), (std::vector<unsigned int>{3, 4, 2, 5}));
}
