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

#include <htp_backend/htp_expert_pool.h>

#include <vector>

namespace {

using nntrainer::ExpertLru;

TEST(ExpertPreload, BudgetDoesNotChangeCacheCapacity_p) {
  ExpertLru lru;
  int layer = 0;
  lru.addLayer(&layer, 480);
  EXPECT_EQ(nntrainer::expertPreloadLimit(lru.capacity()), 128u);
  EXPECT_EQ(nntrainer::expertPreloadLimit(0), 0u);
  EXPECT_EQ(nntrainer::expertPreloadLimit(64), 64u);
  EXPECT_EQ(nntrainer::expertPreloadLimit(128), 128u);
  EXPECT_EQ(nntrainer::expertPreloadLimit(lru.capacity(), 0u), 0u);
  EXPECT_EQ(nntrainer::expertPreloadLimit(lru.capacity(), 64u), 64u);
  EXPECT_EQ(nntrainer::expertPreloadLimit(lru.capacity(), 128u), 128u);
  EXPECT_EQ(nntrainer::expertPreloadLimit(lru.capacity(), 480u), 480u);
  EXPECT_EQ(nntrainer::expertPreloadLimit(lru.capacity(), 999u), 480u);
  EXPECT_EQ(lru.capacity(), 480u);
}

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
  int a = 0, b = 0;
  lru.addLayer(&a, 2);
  lru.addLayer(&b, 2);
  lru.addLayer(&a, 2); // a second finalize of the same layer adds nothing
  EXPECT_EQ(lru.capacity(), 4u);
}

TEST(ExpertLru, MissesLoadHitsDoNot) {
  ExpertLru lru;
  int layer = 0;
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

TEST(ExpertLru, ResetAllowsAnUnloadedModelToReuseLayerAndExpertAddresses) {
  ExpertLru lru;
  int layer = 0;
  Recorder first;
  lru.addLayer(&layer, 4);
  lru.setLrfu(8);
  lru.acquire({key(0), key(1)}, first.load, first.evict);
  lru.hold(2);
  lru.reset();
  EXPECT_EQ(lru.capacity(), 0u);
  EXPECT_EQ(lru.size(), 0u);
  EXPECT_FALSE(lru.resident(key(0)));

  // A new model may use the same allocator addresses with a different pool.
  lru.addLayer(&layer, 2);
  Recorder second;
  EXPECT_EQ(lru.capacity(), 2u);
  EXPECT_EQ(lru.acquire({key(0), key(1)}, second.load, second.evict), 2u);
  EXPECT_EQ(second.loaded, (std::vector<int>{0, 1}));
  EXPECT_TRUE(second.evicted.empty());
  EXPECT_EQ(lru.acquire({key(2)}, second.load, second.evict), 1u);
  EXPECT_EQ(second.evicted, (std::vector<int>{0}));
}

TEST(ExpertLru, EvictsLeastRecentlyUsedNeverThisCallsExperts) {
  ExpertLru lru;
  int layer = 0;
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
  int layer = 0;
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
  int layer = 0;
  lru.addLayer(&layer, 2);
  Recorder r;
  EXPECT_THROW(lru.acquire({key(0), key(1), key(2)}, r.load, r.evict),
               std::runtime_error);
  EXPECT_TRUE(r.loaded.empty());
}

TEST(ExpertLru, RefreshMovesResidentKeysToTheBackInOrder) {
  ExpertLru lru;
  int layer = 0;
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
  int layer = 0;
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
  int layer = 0;
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

TEST(ExpertLru, RoomIsWhatMakeRoomWillGrant) {
  ExpertLru lru;
  int layer = 0;
  lru.addLayer(&layer, 6);
  Recorder r;
  lru.acquire({key(0), key(1), key(2), key(3)}, r.load, r.evict);
  // 2 pinned resident (9 is not), 1 held: 6 - 1 - 2 = 3.
  lru.hold(1);
  EXPECT_EQ(lru.room({key(0), key(1), key(9)}), 3u);
  EXPECT_TRUE(lru.makeRoom(3, {key(0), key(1), key(9)}, r.evict));
  EXPECT_EQ(r.evicted, (std::vector<int>{2, 3}));
  EXPECT_EQ(lru.room({key(0), key(1)}), 3u);
  EXPECT_FALSE(lru.makeRoom(4, {key(0), key(1)}, r.evict));
}

TEST(ExpertLru, HeldSlotsAreNotHandedOutUntilUnheld) {
  // Doc 52 section 10.20: read-ahead in flight owns its slots before its
  // keys are in the LRU, so a miss meanwhile must evict instead.
  ExpertLru lru;
  int layer = 0;
  lru.addLayer(&layer, 6);
  Recorder r;
  lru.acquire({key(0), key(1), key(2), key(3)}, r.load, r.evict);
  EXPECT_TRUE(lru.makeRoom(2, {}, r.evict)); // 4 + 2 fit: nothing goes
  EXPECT_TRUE(r.evicted.empty());
  lru.hold(2);
  // Two misses with two slots held: 0 and 1 go, not the held slots.
  EXPECT_EQ(lru.acquire({key(4), key(5)}, r.load, r.evict), 2u);
  EXPECT_EQ(r.evicted, (std::vector<int>{0, 1}));
  EXPECT_EQ(lru.size(), 4u);
  // Pins filling what the hold leaves refuse more room.
  EXPECT_FALSE(lru.makeRoom(1, {key(2), key(3), key(4), key(5)}, r.evict));
  // A call needing more than the hold leaves is a caller bug.
  EXPECT_THROW(
    lru.acquire({key(10), key(11), key(12), key(13), key(14)}, r.load, r.evict),
    std::logic_error);
  // The reads land: unhold, then file them with no eviction.
  lru.unhold(2);
  lru.acquire({key(8), key(9)}, [](ExpertLru::Key) {}, r.evict);
  EXPECT_EQ(r.evicted.size(), 2u);
  EXPECT_EQ(lru.size(), 6u);
}

TEST(ExpertLru, LrfuKeepsTheFrequentKeyWhereLruDropsIt) {
  // A three times, then B, then C into a pool of 2: LRU drops A (least
  // recent), LRFU drops B (one use against A's three, decayed by a long
  // half-life) -- tools/moe_expert_cache_sim.py's lrfu on the same calls
  for (const double h : {0.0, 32.0}) {
    ExpertLru lru;
    int layer = 0;
    lru.setLrfu(h);
    lru.addLayer(&layer, 2);
    Recorder r;
    for (int e : {0, 0, 0, 1, 2})
      lru.acquire({key(e)}, r.load, r.evict);
    EXPECT_EQ(r.evicted, std::vector<int>{h > 0.0 ? 1 : 0}) << "h=" << h;
  }
}

TEST(ExpertLru, BindsModelConfigAtLayerRegistrationAndAfterReset_p) {
  nntrainer::resetHtpRuntimeConfig();
  ExpertLru lru; // HtpContext can construct its pool before model JSON is read.
  bool first = true;
  for (double half_life : {0.0, 1000.0, 0.0}) {
    if (!first)
      lru.reset(); // Reset also precedes the next model's configuration.
    first = false;
    nntrainer::HtpRuntimeConfig config;
    config.moe.lrfu_half_life = half_life;
    nntrainer::configureHtpRuntime(config);
    int layer = 0;
    lru.addLayer(&layer, 2);
    Recorder r;
    for (int e : {0, 0, 0, 1, 2})
      lru.acquire({key(e)}, r.load, r.evict);
    EXPECT_EQ(r.evicted, std::vector<int>{half_life == 0 ? 0 : 1});
  }
  nntrainer::resetHtpRuntimeConfig();
}

TEST(ExpertLru, LrfuEvictsSeveralLowestFirstLeastRecentOnATie) {
  // Uses: 0 x3, 1 x1, 2 x2, 3 x1 in a pool of 4. A call that brings three
  // new keys in evicts three: the two single-use keys (a tie: 1 before 3,
  // the least recent first), then 2 -- the order one victim at a time gave.
  ExpertLru lru;
  int layer = 0;
  lru.setLrfu(1000.0);
  lru.addLayer(&layer, 4);
  Recorder r;
  for (int e : {0, 0, 0, 1, 2, 2, 3})
    lru.acquire({key(e)}, r.load, r.evict);
  lru.acquire({key(4), key(5), key(6)}, r.load, r.evict);
  EXPECT_EQ(r.evicted, (std::vector<int>{1, 3, 2}));
  EXPECT_EQ(ids(lru.order()), (std::vector<int>{0, 4, 5, 6}));
}

TEST(ExpertLru, LrfuForgetsOldUsesWithAShortHalfLife) {
  // A used three times long ago, B four times since: with a half-life of
  // one call A's score has decayed below B's, so C evicts A, not B
  ExpertLru lru;
  int layer = 0;
  lru.setLrfu(1.0);
  lru.addLayer(&layer, 2);
  Recorder r;
  for (int e : {0, 0, 0, 1, 1, 1, 1})
    lru.acquire({key(e)}, r.load, r.evict);
  lru.acquire({key(2)}, r.load, r.evict);
  EXPECT_EQ(r.evicted, std::vector<int>{0});
}

TEST(ExpertLru, WarmUpReplacesThePrefillTailWithoutGrowingThePool) {
  // The NNTR_MOE_WARM sequence (lfm2_moe_pool_layer.cpp): a pool of 6 the
  // prefill left with 0..5, a warm list 4, 7, 8 (4 resident, pinned).
  // makeRoom frees two of the oldest, the held slots keep a decode miss
  // off them while the reads run, and acquire files the registered keys
  // with nothing loaded or evicted.
  ExpertLru lru;
  int layer;
  lru.addLayer(&layer, 6);
  Recorder r;
  lru.acquire({key(0), key(1), key(2), key(3), key(4), key(5)}, r.load,
              r.evict);
  r.loaded.clear();
  const std::vector<ExpertLru::Key> pinned = {key(4)};
  ASSERT_EQ(lru.room(pinned), 5u);
  ASSERT_TRUE(lru.makeRoom(2, pinned, r.evict));
  EXPECT_EQ(r.evicted, (std::vector<int>{0, 1}));
  lru.hold(2);
  EXPECT_THROW(
    lru.acquire({key(10), key(11), key(12), key(13), key(14)}, r.load, r.evict),
    std::logic_error);
  lru.unhold(2);
  r.evicted.clear();
  lru.acquire(
    {key(7), key(8)}, [](ExpertLru::Key) {},
    [](ExpertLru::Key) { FAIL() << "the warm-up evicted"; });
  EXPECT_TRUE(r.evicted.empty());
  EXPECT_EQ(lru.size(), 6u);
  EXPECT_EQ(lru.capacity(), 6u);
  EXPECT_TRUE(lru.resident(key(7)) && lru.resident(key(8)) &&
              lru.resident(key(4)));
}

/** moe.warm_experts for one test, the defaults again after it. */
struct WarmCount {
  explicit WarmCount(unsigned n) {
    nntrainer::HtpRuntimeConfig config;
    config.moe.warm_experts = n;
    nntrainer::configureHtpRuntime(config);
  }
  ~WarmCount() { nntrainer::resetHtpRuntimeConfig(); }
};

TEST(ExpertWarm, ListIsTheLastTokensRankThenTheMostRoutedRows_p) {
  // two tokens of three ranked ids; the last token's 5, 2, 7 lead in rank
  // order, then the rest by rows (3 before 0 before 1, 4 on a tie with 1
  // goes after it: the lower id first), never an unrouted expert (6)
  const std::vector<int> ranked = {0, 1, 3, 5, 2, 7};
  const std::vector<size_t> rows = {4, 2, 1, 9, 2, 1, 0, 1};
  EXPECT_EQ(nntrainer::expertWarmList(ranked, 3, rows, 8),
            (std::vector<unsigned>{5, 2, 7, 3, 0, 1, 4}));
  EXPECT_EQ(nntrainer::expertWarmList(ranked, 3, rows, 2),
            (std::vector<unsigned>{5, 2}));
  EXPECT_EQ(nntrainer::expertWarmList(ranked, 3, rows, 4),
            (std::vector<unsigned>{5, 2, 7, 3}));
  // a repeated or out-of-range id in the rank list is skipped
  EXPECT_EQ(nntrainer::expertWarmList({9, 2, 2}, 3, rows, 3),
            (std::vector<unsigned>{2, 3, 0}));
  EXPECT_TRUE(nntrainer::expertWarmList(ranked, 3, rows, 0).empty());
}

/** The backend side of set_decode_pool_warm: keeps the pair, releases. */
class WarmOps : public nntrainer::ComputeOps {
public:
  std::function<std::vector<ExpertFileDesc>()> begin;
  std::function<void(const std::vector<const void *> &)> end;
  std::vector<const void *> released;
  void set_decode_pool_warm(
    const std::function<std::vector<ExpertFileDesc>()> &b,
    const std::function<void(const std::vector<const void *> &)> &e) override {
    begin = b;
    end = e;
  }
  bool release_qs4cx_wh_expert(const void *key_gu) override {
    released.push_back(key_gu);
    return true;
  }
};

TEST(ExpertWarm, PrefillListsBecomeTheReadsAndTheLruTakesThem) {
  // Two warm experts a layer, two layers of 4 experts (layer l's expert e
  // is key 4l + e) in a pool of 4 the prefill left with layer 1's. Layer
  // 0's last token ranked 2, 1; layer 1's 1, 2 (keys 5, 6, resident:
  // pinned). The reads are keys 2, 1, the two oldest outside the pins
  // (4, 7) make room, and the LRU files the reads without evicting.
  ExpertLru lru;
  int layer0, layer1;
  lru.addLayer(&layer0, 2);
  lru.addLayer(&layer1, 2);
  Recorder r;
  lru.acquire({key(4), key(5), key(6), key(7)}, r.load, r.evict);
  using Desc = nntrainer::ComputeOps::ExpertFileDesc;
  std::vector<std::vector<Desc>> layers(2, std::vector<Desc>(4));
  for (int l = 0; l < 2; ++l)
    for (int e = 0; e < 4; ++e)
      layers[l][e].key_gu = key(4 * l + e);
  using Rows = std::vector<std::vector<std::pair<unsigned, float>>>;
  WarmCount two(2);
  nntrainer::ExpertWarm warm;
  warm.note(0, {0, 3, 2, 1}, Rows{{{0, 1.f}}, {{1, 1.f}}, {{1, 1.f}}, {}}, 2);
  warm.note(1, {0, 3, 1, 2}, Rows(4), 2);
  WarmOps ops;
  warm.install(&ops, lru, layers);
  ASSERT_TRUE(ops.begin && ops.end);
  const std::vector<Desc> want = ops.begin();
  std::vector<const void *> keys;
  for (const Desc &d : want)
    keys.push_back(d.key_gu);
  EXPECT_EQ(ids(keys), (std::vector<int>{2, 1}));
  EXPECT_EQ(ids(ops.released), (std::vector<int>{4, 7}));
  ops.end(keys); // what the backend registered
  EXPECT_EQ(lru.size(), 4u);
  for (int e : {1, 2, 5, 6})
    EXPECT_TRUE(lru.resident(key(e))) << e;
  // the slots are no longer held: a call of four fits
  lru.acquire({key(1), key(2), key(5), key(6)}, r.load, r.evict);
}

TEST(ExpertWarm, OnAtSixteenByDefaultAndOffAtZero) {
  // moe.warm_experts is 16 unless configured; 0 installs nothing
  EXPECT_EQ(nntrainer::ExpertWarm::count(), 16u);
  ExpertLru lru;
  int layer0;
  lru.addLayer(&layer0, 4);
  using Desc = nntrainer::ComputeOps::ExpertFileDesc;
  std::vector<std::vector<Desc>> layers(1, std::vector<Desc>(4));
  for (int e = 0; e < 4; ++e)
    layers[0][e].key_gu = key(e);
  using Rows = std::vector<std::vector<std::pair<unsigned, float>>>;
  {
    nntrainer::ExpertWarm on;
    WarmOps ops;
    on.note(0, {3, 1, 2, 0}, Rows(4), 2);
    on.install(&ops, lru, layers);
    ASSERT_TRUE(ops.begin);
    EXPECT_EQ(ops.begin().size(), 2u); // the last token's two
  }
  WarmCount zero(0);
  nntrainer::ExpertWarm off;
  WarmOps ops;
  off.note(0, {3, 1, 2, 0}, Rows(4), 2);
  off.install(&ops, lru, layers);
  EXPECT_FALSE(ops.begin);
}

TEST(ExpertWarm, ReadsCancelledAtTheFirstTokenLeaveTheirSlotsFree) {
  // As above, but the first token's deadline cancelled the read of key 1
  // (warmEnd): the LRU files key 2 alone and every slot is free again, so
  // the token's miss of key 1 fills the cancelled one without evicting.
  ExpertLru lru;
  int layer0, layer1;
  lru.addLayer(&layer0, 2);
  lru.addLayer(&layer1, 2);
  Recorder r;
  lru.acquire({key(4), key(5), key(6), key(7)}, r.load, r.evict);
  using Desc = nntrainer::ComputeOps::ExpertFileDesc;
  std::vector<std::vector<Desc>> layers(2, std::vector<Desc>(4));
  for (int l = 0; l < 2; ++l)
    for (int e = 0; e < 4; ++e)
      layers[l][e].key_gu = key(4 * l + e);
  using Rows = std::vector<std::vector<std::pair<unsigned, float>>>;
  WarmCount two(2);
  nntrainer::ExpertWarm warm;
  warm.note(0, {0, 3, 2, 1}, Rows{{{0, 1.f}}, {{1, 1.f}}, {{1, 1.f}}, {}}, 2);
  warm.note(1, {0, 3, 1, 2}, Rows(4), 2);
  WarmOps ops;
  warm.install(&ops, lru, layers);
  ASSERT_EQ(ops.begin().size(), 2u);
  ops.end({key(2)});
  EXPECT_EQ(lru.size(), 3u);
  EXPECT_TRUE(lru.resident(key(2)) && !lru.resident(key(1)));
  EXPECT_EQ(lru.room({key(2), key(5), key(6)}), 1u);
  r.evicted.clear();
  EXPECT_EQ(lru.acquire({key(1), key(2), key(5), key(6)}, r.load, r.evict), 1u);
  EXPECT_TRUE(r.evicted.empty());
}

} // namespace
