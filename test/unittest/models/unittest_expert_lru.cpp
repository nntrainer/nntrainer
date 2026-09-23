// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_expert_lru.cpp
 * @date   22 September 2026
 * @brief  The expert LRU behind the HTP MoE path's flash streaming (doc 52):
 *         capacity, eviction order, pinning of one call's working set, and
 *         the extended-top-k recency refresh. Pure data structure, so this
 *         is the one check that runs without a device.
 * @see    https://github.com/nnstreamer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <gtest/gtest.h>

#include <expert_lru.h>

#include <vector>

namespace {

using causallm::ExpertLru;

/** Keys are opaque addresses; these are as good as any. */
static const char kExperts[64] = {};
static ExpertLru::Key key(int e) { return &kExperts[e]; }

struct Recorder {
  std::vector<int> loaded, evicted;
  std::function<void(ExpertLru::Key)> load = [this](ExpertLru::Key k) {
    loaded.push_back(static_cast<int>(static_cast<const char *>(k) - kExperts));
  };
  std::function<void(ExpertLru::Key)> evict = [this](ExpertLru::Key k) {
    evicted.push_back(
      static_cast<int>(static_cast<const char *>(k) - kExperts));
  };
};

static std::vector<int> ids(const std::vector<ExpertLru::Key> &keys) {
  std::vector<int> out;
  for (auto k : keys)
    out.push_back(static_cast<int>(static_cast<const char *>(k) - kExperts));
  return out;
}

TEST(ExpertLru, CapacityIsPerLayerTimesLayersAndIdempotentPerLayer) {
  ExpertLru lru;
  int a, b;
  lru.addLayer(&a, 2);
  lru.addLayer(&b, 2);
  lru.addLayer(&a, 2); // a second finalize of the same layer adds nothing
  EXPECT_EQ(lru.capacity(), 4u);
}

TEST(ExpertLru, MissesLoadHitsDoNot) {
  ExpertLru lru;
  int layer;
  lru.addLayer(&layer, 4);
  Recorder r;
  EXPECT_EQ(lru.acquire({key(0), key(1)}, r.load, r.evict), 2u);
  EXPECT_EQ(lru.acquire({key(1), key(2)}, r.load, r.evict), 1u);
  EXPECT_EQ(r.loaded, (std::vector<int>{0, 1, 2}));
  EXPECT_TRUE(r.evicted.empty());
  EXPECT_EQ(lru.size(), 3u);
  EXPECT_TRUE(lru.resident(key(0)));
  EXPECT_FALSE(lru.resident(key(3)));
}

TEST(ExpertLru, EvictsLeastRecentlyUsedNeverThisCallsExperts) {
  ExpertLru lru;
  int layer;
  lru.addLayer(&layer, 3);
  Recorder r;
  lru.acquire({key(0), key(1), key(2)}, r.load, r.evict); // order 0 1 2
  lru.acquire({key(0)}, r.load, r.evict);                 // order 1 2 0
  // Needs 3 and 4 with 0 pinned: 1 then 2 go, in that order, before any load.
  EXPECT_EQ(lru.acquire({key(0), key(3), key(4)}, r.load, r.evict), 2u);
  EXPECT_EQ(r.evicted, (std::vector<int>{1, 2}));
  EXPECT_EQ(r.loaded, (std::vector<int>{0, 1, 2, 3, 4}));
  EXPECT_EQ(ids(lru.order()), (std::vector<int>{0, 3, 4}));
  EXPECT_EQ(lru.size(), 3u);
}

TEST(ExpertLru, EvictionsPrecedeLoads) {
  // A slot has to be free before its replacement is read into it.
  ExpertLru lru;
  int layer;
  lru.addLayer(&layer, 1);
  std::vector<std::string> trace;
  auto load = [&](ExpertLru::Key) { trace.push_back("load"); };
  auto evict = [&](ExpertLru::Key) { trace.push_back("evict"); };
  lru.acquire({key(0)}, load, evict);
  lru.acquire({key(1)}, load, evict);
  EXPECT_EQ(trace, (std::vector<std::string>{"load", "evict", "load"}));
}

TEST(ExpertLru, RefusesAWorkingSetLargerThanThePool) {
  ExpertLru lru;
  int layer;
  lru.addLayer(&layer, 2);
  Recorder r;
  EXPECT_THROW(lru.acquire({key(0), key(1), key(2)}, r.load, r.evict),
               std::runtime_error);
  EXPECT_TRUE(r.loaded.empty());
}

TEST(ExpertLru, RefreshMovesResidentKeysToTheBackInOrder) {
  ExpertLru lru;
  int layer;
  lru.addLayer(&layer, 4);
  Recorder r;
  lru.acquire({key(0), key(1), key(2), key(3)}, r.load, r.evict);
  // Extended top-k of one token: 2 then 0 most recent; 9 is not resident.
  lru.refresh({key(9), key(2), key(0)});
  EXPECT_EQ(ids(lru.order()), (std::vector<int>{1, 3, 2, 0}));
  // The next miss evicts 1, the least recent after the refresh.
  lru.acquire({key(5)}, r.load, r.evict);
  EXPECT_EQ(r.evicted, (std::vector<int>{1}));
}

TEST(ExpertLru, MakeRoomEvictsOldestUnpinnedUntilNFit) {
  ExpertLru lru;
  int layer;
  lru.addLayer(&layer, 5);
  Recorder r;
  lru.acquire({key(0), key(1), key(2), key(3), key(4)}, r.load, r.evict);
  // Room for 3 with 0 and 2 pinned: 1, 3, 4 go in LRU order.
  EXPECT_TRUE(lru.makeRoom(3, {key(0), key(2)}, r.evict));
  EXPECT_EQ(r.evicted, (std::vector<int>{1, 3, 4}));
  EXPECT_EQ(ids(lru.order()), (std::vector<int>{0, 2}));
  // The caller brings them in itself; acquire files them with a no-op load.
  size_t loads = 0;
  lru.acquire(
    {key(5), key(6), key(7)}, [&](ExpertLru::Key) { ++loads; }, r.evict);
  EXPECT_EQ(loads, 3u);
  EXPECT_EQ(lru.size(), 5u);
  EXPECT_EQ(r.evicted.size(), 3u); // no further eviction
}

TEST(ExpertLru, MakeRoomRefusesWithoutEvictingWhenPinsFillThePool) {
  ExpertLru lru;
  int layer;
  lru.addLayer(&layer, 4);
  Recorder r;
  lru.acquire({key(0), key(1), key(2), key(3)}, r.load, r.evict);
  // 3 pinned resident + 2 wanted > 4: nothing may go.
  EXPECT_FALSE(lru.makeRoom(2, {key(0), key(1), key(2), key(9)}, r.evict));
  EXPECT_TRUE(r.evicted.empty());
  EXPECT_EQ(lru.size(), 4u);
  // Room that already exists costs nothing.
  EXPECT_TRUE(lru.makeRoom(0, {}, r.evict));
  EXPECT_TRUE(r.evicted.empty());
}

} // namespace
