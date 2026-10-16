// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file        unittest_expert_fsu_flags.cpp
 * @date        10 October 2026
 * @brief       Host tests of the streamed-expert decode protocol without a
 *              host round trip (expert_fsu_flags.h): the miss ring, the
 *              victim queue, the ready words, the gate, the replay of the
 *              residency policy. The device side is the reference form of
 *              the kernels on a thread of its own that runs commands in
 *              order, like an in-order queue; the reads are fake readers
 *              that write a record's identity into its slot. No GPU.
 * @see         https://github.com/nntrainer/nntrainer
 * @author      Jijoong Moon <jijoong.moon@samsung.com>
 * @bug         No known bugs
 */

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#if defined(__linux__)
#include <sys/prctl.h>
#endif

#include <cl_context.h>
#include <cl_kernels/tq2moe_fsu_flags.h>
#include <engine.h>
#include <expert_fsu_flags.h>
#include <opencl_loader.h>
#include <tq2cx_cl.h>

// the residency policy the protocol replays (header-only, host-only; the
// layer that uses it lives with the CausalLM application)
#include "../../Applications/CausalLM/layers/expert_residency.h"

using namespace nntrainer::fsuflags;
using causallm::ExpertResidency;

namespace {

/** words of a fake expert record in its slot */
constexpr uint32_t kRec = 64;

/** what a fake reader writes into a slot: the record's identity */
uint32_t ident(uint32_t layer, uint32_t expert) {
  return 0x5A000000u | layer << 12 | expert;
}

/**
 * @brief The device: runs commands in order on a thread of its own (an
 *        in-order queue), each the reference form of a kernel.
 */
class Device {
public:
  Device() : th_([this] { loop(); }) {}
  ~Device() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    th_.join();
  }
  void push(std::function<void()> f) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      q_.push_back(std::move(f));
    }
    cv_.notify_all();
  }
  /** wait until every command pushed so far ran (the token's end) */
  void finish() {
    auto p = std::make_shared<std::promise<void>>();
    auto f = p->get_future();
    push([p] { p->set_value(); });
    ASSERT_EQ(f.wait_for(std::chrono::seconds(30)), std::future_status::ready);
  }

private:
  void loop() {
    for (;;) {
      std::function<void()> f;
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [this] { return stop_ || !q_.empty(); });
        if (q_.empty())
          return;
        f = std::move(q_.front());
        q_.pop_front();
      }
      f();
    }
  }
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<std::function<void()>> q_;
  bool stop_ = false;
  std::thread th_;
};

/** fake readers: write the record's identity into its slot after a delay */
class Readers {
public:
  explicit Readers(unsigned int n, unsigned int max_delay_us) :
    max_delay_us_(max_delay_us) {
    for (unsigned int i = 0; i < n; ++i)
      th_.emplace_back([this, i] { loop(i); });
  }
  ~Readers() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto &t : th_)
      t.join();
  }
  struct Job {
    FsuFlagsHost::Read r;
    std::shared_ptr<std::atomic<bool>> once;
    bool low;
  };
  /** the memory of every layer's slots and the host to report to */
  void bind(FsuFlagsHost *h, std::vector<std::vector<uint32_t>> *mem) {
    host_ = h;
    mem_ = mem;
  }
  void issue(const FsuFlagsHost::Read &r,
             std::shared_ptr<std::atomic<bool>> once, bool low) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      (low ? low_ : high_).push_back({r, std::move(once), low});
    }
    cv_.notify_one();
  }
  void pause(bool p) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      paused_ = p;
    }
    cv_.notify_all();
  }
  std::atomic<bool> fail_next{false};
  std::atomic<uint64_t> done{0};

private:
  void loop(unsigned int seed) {
    std::mt19937 rng(seed + 77);
    for (;;) {
      Job j;
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [this] {
          return stop_ || (!paused_ && (!high_.empty() || !low_.empty()));
        });
        if (stop_)
          return;
        auto &q = !high_.empty() ? high_ : low_;
        j = q.front();
        q.pop_front();
      }
      if (j.once->exchange(true)) {
        host_->readDone(j.r, ECANCELED);
        continue;
      }
      if (max_delay_us_ != 0)
        std::this_thread::sleep_for(std::chrono::microseconds(
          std::uniform_int_distribution<unsigned int>(0, max_delay_us_)(rng)));
      int err = 0;
      if (fail_next.exchange(false)) {
        err = EIO;
      } else {
        uint32_t *p = (*mem_)[j.r.layer].data() + (size_t)j.r.slot * kRec;
        for (uint32_t i = 0; i < kRec; ++i)
          p[i] = ident(j.r.layer, j.r.expert);
      }
      host_->readDone(j.r, err);
      ++done;
    }
  }
  unsigned int max_delay_us_;
  FsuFlagsHost *host_ = nullptr;
  std::vector<std::vector<uint32_t>> *mem_ = nullptr;
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<Job> high_, low_;
  bool paused_ = false, stop_ = false;
  std::vector<std::thread> th_;
};

/** hooks of @a r's layer 0 for the host */
FsuFlagsHost::Hooks hooksOf(ExpertResidency &r) {
  FsuFlagsHost::Hooks h;
  h.victim_order = [&r](const std::vector<unsigned int> &ex) {
    return r.victimOrder(0, ex);
  };
  h.place = [&r](const std::vector<unsigned int> &need,
                 const std::vector<std::pair<unsigned int, unsigned int>> &p) {
    r.place(0, need, p);
  };
  h.admit = [&r](unsigned int e, unsigned int s) { r.admit(0, e, s); };
  h.forget = [&r](unsigned int e) { r.forget(0, e); };
  h.refresh = [&r](const std::vector<unsigned int> &rec) { r.refresh(0, rec); };
  h.slot_map = [&r]() { return r.slotMap(0); };
  return h;
}

/**
 * @brief A decode of @a layers streamed layers through the protocol: the
 *        host thread prepares / publishes like the backend, the device
 *        thread runs publish -> (read ahead of the next layer) -> gate ->
 *        a data check like the expert GEMVs would read, the poller and the
 *        fake readers serve the misses.
 */
struct Sim {
  unsigned int layers, C, E, K, X;
  std::vector<std::vector<uint32_t>> ctl_mem;
  std::vector<std::vector<uint32_t>> blocks, mem;
  std::vector<std::unique_ptr<ExpertResidency>> res;
  // destroyed in reverse: the device, then the readers (they report to the
  // host), then the host (its poller is stopped in ~Sim first)
  std::unique_ptr<FsuFlagsHost> host;
  std::unique_ptr<Readers> readers;
  Device dev;
  /** per layer: what the last publish gave the gate */
  std::vector<std::vector<uint32_t>> wait_slot, wait_tag;
  std::atomic<uint64_t> bad_data{0}, gate_errors{0}, publish_errors{0};
  std::vector<float> w;

  Sim(unsigned int nl, unsigned int c, unsigned int e, unsigned int k,
      unsigned int n_readers, unsigned int delay_us) :
    layers(nl), C(c), E(e), K(k), X(k + 4) {
    ctl_mem.assign(1, std::vector<uint32_t>(kCtlWords, 0));
    readers = std::make_unique<Readers>(n_readers, delay_us);
    host = std::make_unique<FsuFlagsHost>(
      ctl_mem[0].data(),
      [this](const FsuFlagsHost::Read &r,
             std::shared_ptr<std::atomic<bool>> once,
             bool low) { readers->issue(r, std::move(once), low); },
      /*check=*/true);
    blocks.resize(nl);
    mem.assign(nl, std::vector<uint32_t>((size_t)c * kRec, 0));
    wait_slot.assign(nl, std::vector<uint32_t>(k, kNone));
    wait_tag.assign(nl, std::vector<uint32_t>(k, 0));
    for (unsigned int l = 0; l < nl; ++l) {
      res.push_back(std::make_unique<ExpertResidency>());
      res[l]->addLayer(0, c, e);
      res[l]->setPolicy(ExpertResidency::Policy::lrfu, 0.25);
      blocks[l].assign(LayerLayout(c, e).words, 0);
      host->addLayer(l, blocks[l].data(), c, e, k, hooksOf(*res[l]));
    }
    readers->bind(host.get(), &mem);
    host->startPoller(500);
  }
  ~Sim() { host->stopPoller(); }

  uint32_t *ctl() { return ctl_mem[0].data(); }

  /** one layer's step: @a ids ascending, @a pred the next layer's
   *  prediction (empty: none) */
  void step(unsigned int l, const std::vector<uint32_t> &ids,
            const std::vector<uint32_t> &pred) {
    if (host->needsPrepare(l))
      host->prepare(l);
    const uint32_t seq = host->beginPublish(l, false);
    std::vector<uint32_t> ext(ids);
    while (ext.size() < X)
      ext.push_back((ext.back() + 7) % E);
    const LayerLayout L(C, E);
    dev.push([this, l, seq, ids, ext, L] {
      if (refPublish(ctl(), blocks[l].data(), L, seq, l, ids.data(),
                     (uint32_t)ids.size(), ext.data(), (uint32_t)ext.size(),
                     wait_slot[l].data(), wait_tag[l].data(),
                     /*cap=*/2000000) != kOk)
        ++publish_errors;
    });
    if (!pred.empty() && l + 1 < layers) {
      if (host->needsPrepare(l + 1))
        host->prepare(l + 1);
      const uint32_t s2 = host->beginPublish(l + 1, true);
      dev.push([this, l, s2, pred, L] {
        if (refSpec(ctl(), blocks[l + 1].data(), L, s2, l + 1, pred.data(),
                    (uint32_t)pred.size(), K, 2000000) != kOk)
          ++publish_errors;
      });
    }
    // the gate, then what the GEMVs read: every selected expert's record
    dev.push([this, l, ids, L] {
      w.assign(K, 1.0f);
      for (uint32_t a = 0; a < ids.size(); ++a)
        if (refGateLane(ctl(), blocks[l].data(), L, l, a, wait_slot[l][a],
                        wait_tag[l][a], /*cap=*/200000000ull, w.data()) != kOk)
          ++gate_errors;
      for (uint32_t a = 0; a < ids.size(); ++a) {
        const uint32_t *p = mem[l].data() + (size_t)wait_slot[l][a] * kRec;
        for (uint32_t i = 0; i < kRec; ++i)
          if (p[i] != ident(l, ids[a])) {
            ++bad_data;
            break;
          }
      }
    });
  }
};

/** @a k distinct experts of @a e, skewed toward low ids, ascending */
std::vector<uint32_t> route(std::mt19937 &rng, unsigned int k, unsigned int e) {
  std::uniform_int_distribution<unsigned int> d(0, e - 1);
  std::set<uint32_t> s;
  while (s.size() < k)
    s.insert(std::min(d(rng), d(rng)));
  return std::vector<uint32_t>(s.begin(), s.end());
}

/** a one-layer host with a manual reader (the test completes reads) */
struct Manual {
  std::vector<uint32_t> ctl_mem = std::vector<uint32_t>(kCtlWords, 0);
  std::vector<uint32_t> block;
  ExpertResidency res;
  std::mutex mu;
  std::vector<std::pair<FsuFlagsHost::Read, std::shared_ptr<std::atomic<bool>>>>
    reads;
  std::unique_ptr<FsuFlagsHost> host;
  LayerLayout L;
  Manual(unsigned int C, unsigned int E, unsigned int K) : L(C, E) {
    block.assign(L.words, 0);
    res.addLayer(0, C, E);
    host = std::make_unique<FsuFlagsHost>(
      ctl_mem.data(),
      [this](const FsuFlagsHost::Read &r,
             std::shared_ptr<std::atomic<bool>> once, bool) {
        std::lock_guard<std::mutex> lk(mu);
        reads.emplace_back(r, std::move(once));
      },
      true);
    host->addLayer(0, block.data(), C, E, K, hooksOf(res));
  }
  uint32_t *ctl() { return ctl_mem.data(); }
  /** complete every read issued so far with @a err */
  void completeAll(int err = 0) {
    std::vector<
      std::pair<FsuFlagsHost::Read, std::shared_ptr<std::atomic<bool>>>>
      v;
    {
      std::lock_guard<std::mutex> lk(mu);
      v.swap(reads);
    }
    for (auto &r : v)
      host->readDone(r.first, r.second->exchange(true) ? ECANCELED : err);
  }
};

} // namespace

TEST(FsuFlags, LayoutFitsAndTagsNeverCollideWithNothingOrHostTags) {
  EXPECT_EQ(kRingEntries & (kRingEntries - 1), 0u);
  EXPECT_GE(kEntryWords, kEnItems + kItemWords * kMaxItems);
  const LayerLayout L(32, 128);
  EXPECT_EQ(L.hdr, 32u * kHaveStride);
  EXPECT_EQ(L.words, L.hdr + kHdrWords + 128 + 3 * 32);
  std::set<uint32_t> tags;
  for (uint32_t seq = 1; seq < 3000; ++seq)
    for (uint32_t i = 0; i < kMaxItems; ++i) {
      const uint32_t t = tagOf(seq, i);
      EXPECT_NE(t, 0u);
      EXPECT_EQ(t & 0x80000000u, 0u);
      EXPECT_TRUE(tags.insert(t).second);
    }
  EXPECT_NE(hostTag(1) & 0x80000000u, 0u);
  // the kernels get the layout from the host's constants
  const std::string d = clDefines();
  EXPECT_NE(d.find("-DFSU_RING=128"), std::string::npos);
  EXPECT_NE(d.find("-DFSU_EN_ITEMS=40"), std::string::npos);
}

TEST(FsuFlags, TagsStayDeviceTagsAcrossThe27BitWrap) {
  // seq is reduced to 27 bits for its tags: around every multiple of 2^27
  // (where a plain mask would give item 0 the tag 0 and, from 2^27 on with
  // a 28-bit mask, set bit 31) and up to the end of the 32-bit sequence,
  // a tag is never 0, never a host tag, and the tags of 4000 consecutive
  // publishes across the wrap are all different
  EXPECT_EQ(kTagSeqPeriod, (1u << 27) - 1u);
  for (uint32_t base :
       {1u << 27, 1u << 28, 3u << 27, 1u << 31, 0xFFFFFFFFu - 2000u}) {
    std::set<uint32_t> tags;
    for (uint32_t d = 0; d < 4000; ++d) {
      const uint32_t seq = base - 2000u + d;
      EXPECT_GE(tagSeq(seq), 1u);
      EXPECT_LT(tagSeq(seq), 1u << 27);
      for (uint32_t i = 0; i < kMaxItems; ++i) {
        const uint32_t t = tagOf(seq, i);
        EXPECT_NE(t, 0u) << "seq " << seq << " item " << i;
        EXPECT_EQ(t & 0x80000000u, 0u) << "seq " << seq << " item " << i;
        EXPECT_TRUE(tags.insert(t).second) << "seq " << seq << " item " << i;
      }
    }
  }
  // the period: the same tag again only 2^27 - 1 publishes later
  EXPECT_EQ(tagOf(5, 3), tagOf(5 + kTagSeqPeriod, 3));
  EXPECT_NE(tagOf(5, 3), tagOf(5 + kTagSeqPeriod - 1, 3));
  // the kernels get the period from the host's constants
  EXPECT_NE(clDefines().find("-DFSU_TAG_PERIOD=0x7ffffffu"), std::string::npos);
}

TEST(FsuFlags, TheGateSpinAccountSumsTheSlowestLanePerGate) {
  // refGateAccount (the gate kernel's lane 0): the slowest lane's polls
  // summed over gates in 64 bits, the gates, the gates that waited, the
  // largest gate; spinAccount reads it back
  std::vector<uint32_t> ctl(kCtlWords, 0u);
  refGateAccount(ctl.data(), 0);
  refGateAccount(ctl.data(), 700);
  refGateAccount(ctl.data(), 0xFFFFFFF0ull);
  refGateAccount(ctl.data(), 0x1FFFFFFFFull); // saturates at 2^32 - 1
  const SpinAccount a = spinAccount(ctl.data());
  EXPECT_EQ(a.gates, 4u);
  EXPECT_EQ(a.waited, 3u);
  EXPECT_EQ(a.polls, 700ull + 0xFFFFFFF0ull + 0xFFFFFFFFull);
  EXPECT_EQ(a.max_polls, 0xFFFFFFFFu);
  // a lane's polls: 0 when ready at once, cap + 1 when it timed out
  const LayerLayout L(2, 4);
  std::vector<uint32_t> lay(L.words, 0u), c2(kCtlWords, 0u);
  lay[L.have(0)] = 77;
  uint64_t n = 99;
  EXPECT_EQ(
    refGateLane(c2.data(), lay.data(), L, 0, 0, 0, 77, 500, nullptr, &n), kOk);
  EXPECT_EQ(n, 0u);
  EXPECT_EQ(
    refGateLane(c2.data(), lay.data(), L, 0, 1, 1, 78, 500, nullptr, &n),
    kErrTimeout);
  EXPECT_EQ(n, 501u);
}

TEST(FsuFlags, TheRingDeliversEveryEntryInOrderAcrossWrapAround) {
  // 1000 publishes (7.8 times around the ring) as fast as the device can,
  // the poller on its own thread: every entry taken once, in order, the
  // device never ahead of the tail by more than the ring
  Manual m(4, 16, 2);
  m.host->prepare(0);
  m.host->startPoller(200);
  std::vector<uint32_t> seqs;
  for (int i = 0; i < 1000; ++i)
    seqs.push_back(m.host->beginPublish(0, true));
  std::thread devt([&] {
    const uint32_t none[1] = {0};
    for (uint32_t s : seqs)
      ASSERT_EQ(
        refSpec(m.ctl(), m.block.data(), m.L, s, 0, none, 0, 2, 100000000ull),
        kOk);
  });
  devt.join();
  m.host->drain();
  m.host->stopPoller();
  const auto st = m.host->stats();
  EXPECT_EQ(st.specs, 1000u);
  EXPECT_EQ(loadAcquire(m.ctl() + kCtlTail), 1000u);
  EXPECT_EQ(m.host->errorCode(), (uint32_t)kOk);
}

TEST(FsuFlags, APublishWaitsForTheTailAndReportsAFullRing) {
  Manual m(4, 16, 2);
  m.host->prepare(0);
  const uint32_t none[1] = {0};
  // nobody consumes: kRingEntries entries fit, the next one waits, bounded
  for (uint32_t i = 0; i < kRingEntries; ++i)
    ASSERT_EQ(refSpec(m.ctl(), m.block.data(), m.L,
                      m.host->beginPublish(0, true), 0, none, 0, 2, 1000),
              kOk);
  EXPECT_EQ(refSpec(m.ctl(), m.block.data(), m.L, m.host->beginPublish(0, true),
                    0, none, 0, 2, 1000),
            (uint32_t)kErrRingFull);
  EXPECT_EQ(m.host->errorCode(), (uint32_t)kErrRingFull);
  EXPECT_THROW(m.host->throwIfFailed(), std::runtime_error);
}

TEST(FsuFlags, AStaleEntryIsNeverTakenForANewOne) {
  Manual m(4, 16, 2);
  m.host->prepare(0);
  const uint32_t none[1] = {0};
  for (uint32_t i = 0; i < kRingEntries; ++i) {
    ASSERT_EQ(refSpec(m.ctl(), m.block.data(), m.L,
                      m.host->beginPublish(0, true), 0, none, 0, 2, 1000),
              kOk);
    ASSERT_TRUE(m.host->pollOnce());
  }
  // ring slot of seq kRingEntries + 1 still holds seq 1: not the expected
  EXPECT_FALSE(m.host->pollOnce());
  const uint32_t s = m.host->beginPublish(0, true);
  EXPECT_EQ(s, kRingEntries + 1);
  ASSERT_EQ(refSpec(m.ctl(), m.block.data(), m.L, s, 0, none, 0, 2, 1000), kOk);
  EXPECT_TRUE(m.host->pollOnce());
  EXPECT_FALSE(m.host->pollOnce());
}

TEST(FsuFlags, AGateWaitsForItsTagNotForAnOlderReadyWord) {
  // one slot reused: expert 0 read into it (ready = tag A), then expert 1
  // placed into the same slot (tag B): its gate must not pass on A
  Manual m(1, 4, 1);
  uint32_t slot[1], tag[1];
  const uint32_t id0[1] = {0}, id1[1] = {1};
  m.host->prepare(0);
  ASSERT_EQ(refPublish(m.ctl(), m.block.data(), m.L,
                       m.host->beginPublish(0, false), 0, id0, 1, id0, 1, slot,
                       tag, 1000),
            kOk);
  ASSERT_TRUE(m.host->pollOnce());
  m.completeAll();
  float w[1] = {1.0f};
  EXPECT_EQ(
    refGateLane(m.ctl(), m.block.data(), m.L, 0, 0, slot[0], tag[0], 1000, w),
    kOk);
  const uint32_t tag_a = tag[0];
  m.host->prepare(0);
  ASSERT_EQ(refPublish(m.ctl(), m.block.data(), m.L,
                       m.host->beginPublish(0, false), 0, id1, 1, id1, 1, slot,
                       tag, 1000),
            kOk);
  ASSERT_EQ(slot[0], 0u);
  ASSERT_NE(tag[0], tag_a);
  ASSERT_EQ(loadAcquire(m.block.data() + m.L.have(0)), tag_a);
  ASSERT_TRUE(m.host->pollOnce());
  std::atomic<uint32_t> rc{99};
  std::thread gate([&] {
    rc = refGateLane(m.ctl(), m.block.data(), m.L, 0, 0, slot[0], tag[0],
                     4000000000ull, w);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  EXPECT_EQ(rc.load(), 99u); // still waiting on the new tag
  m.completeAll();
  gate.join();
  EXPECT_EQ(rc.load(), (uint32_t)kOk);
  EXPECT_FALSE(std::isnan(w[0]));
}

TEST(FsuFlags, AGateTimesOutWithTheErrorWordAndNaNWeights) {
  Manual m(2, 4, 2);
  uint32_t slot[2], tag[2];
  const uint32_t ids[2] = {1, 3};
  m.host->prepare(0);
  ASSERT_EQ(refPublish(m.ctl(), m.block.data(), m.L,
                       m.host->beginPublish(0, false), 0, ids, 2, ids, 2, slot,
                       tag, 1000),
            kOk);
  ASSERT_TRUE(m.host->pollOnce()); // the reads are issued, never completed
  float w[2] = {1.0f, 1.0f};
  const auto t0 = std::chrono::steady_clock::now();
  EXPECT_EQ(
    refGateLane(m.ctl(), m.block.data(), m.L, 7, 1, slot[1], tag[1], 100000, w),
    (uint32_t)kErrTimeout);
  EXPECT_LT(std::chrono::steady_clock::now() - t0, std::chrono::seconds(5));
  EXPECT_TRUE(std::isnan(w[1]));
  EXPECT_FALSE(std::isnan(w[0]));
  EXPECT_EQ(loadAcquire(m.ctl() + kCtlError), (uint32_t)kErrTimeout);
  EXPECT_EQ(loadAcquire(m.ctl() + kCtlErrInfo), 7u << 16 | 1u);
  // a second failure keeps the first
  EXPECT_EQ(
    refGateLane(m.ctl(), m.block.data(), m.L, 7, 0, slot[0], tag[0], 1000, w),
    (uint32_t)kErrFailed); // the error word stops it (64 polls)
  EXPECT_EQ(loadAcquire(m.ctl() + kCtlError), (uint32_t)kErrTimeout);
  EXPECT_THROW(m.host->throwIfFailed(), std::runtime_error);
  EXPECT_THROW(m.host->prepare(0), std::runtime_error);
}

TEST(FsuFlags, AFailedReadStopsEveryGate) {
  Manual m(2, 4, 2);
  uint32_t slot[2], tag[2];
  const uint32_t ids[2] = {0, 2};
  m.host->prepare(0);
  ASSERT_EQ(refPublish(m.ctl(), m.block.data(), m.L,
                       m.host->beginPublish(0, false), 0, ids, 2, ids, 2, slot,
                       tag, 1000),
            kOk);
  ASSERT_TRUE(m.host->pollOnce());
  std::atomic<uint32_t> rc{99};
  float w[2] = {1.0f, 1.0f};
  std::thread gate([&] {
    // a budget of minutes: only the failed word can end it in time
    rc = refGateLane(m.ctl(), m.block.data(), m.L, 0, 0, slot[0], tag[0], ~0ull,
                     w);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  m.completeAll(EIO);
  gate.join();
  EXPECT_EQ(rc.load(), (uint32_t)kErrFailed);
  EXPECT_TRUE(std::isnan(w[0]));
  EXPECT_EQ(m.host->errorCode(), (uint32_t)kErrFailed);
  try {
    m.host->throwIfFailed();
    FAIL() << "no throw";
  } catch (const std::runtime_error &e) {
    EXPECT_NE(std::string(e.what()).find("failed"), std::string::npos);
  }
}

TEST(FsuFlags, ADecodeThroughTheRingIsTheAcquireDecodeWithTheRightBytes) {
  // 3 layers x 300 tokens, no read ahead: the policy the host replays ends
  // where plain acquire() calls end (slots, recency, hits), every gate
  // passes, and every expert the GEMVs read holds its own record
  Sim sim(3, 6, 24, 3, 2, 300);
  std::vector<std::unique_ptr<ExpertResidency>> ref;
  for (unsigned int l = 0; l < 3; ++l) {
    ref.push_back(std::make_unique<ExpertResidency>());
    ref[l]->addLayer(0, 6, 24);
    ref[l]->setPolicy(ExpertResidency::Policy::lrfu, 0.25);
  }
  std::mt19937 rng(42);
  for (int t = 0; t < 300; ++t) {
    for (unsigned int l = 0; l < 3; ++l) {
      const auto ids = route(rng, 3, 24);
      std::vector<unsigned int> need(ids.begin(), ids.end());
      ref[l]->acquire(0, need);
      std::vector<unsigned int> rec(need.begin(), need.end());
      while (rec.size() < sim.X)
        rec.push_back((rec.back() + 7) % 24);
      ref[l]->refresh(0, std::vector<unsigned int>(rec.rbegin(), rec.rend()));
      sim.step(l, ids, {});
    }
    sim.dev.finish(); // the token's logits read
  }
  sim.host->drain();
  EXPECT_EQ(sim.publish_errors.load(), 0u);
  EXPECT_EQ(sim.gate_errors.load(), 0u);
  EXPECT_EQ(sim.bad_data.load(), 0u);
  EXPECT_EQ(sim.host->errorCode(), (uint32_t)kOk);
  for (unsigned int l = 0; l < 3; ++l) {
    EXPECT_EQ(sim.res[l]->slotMap(0), ref[l]->slotMap(0)) << "layer " << l;
    EXPECT_EQ(sim.res[l]->order(0), ref[l]->order(0)) << "layer " << l;
    EXPECT_EQ(sim.res[l]->hits(), ref[l]->hits()) << "layer " << l;
    EXPECT_EQ(sim.res[l]->misses(), ref[l]->misses()) << "layer " << l;
  }
  const auto st = sim.host->stats();
  EXPECT_EQ(st.steps, 900u);
  EXPECT_EQ(st.misses,
            sim.res[0]->misses() + sim.res[1]->misses() + sim.res[2]->misses());
  EXPECT_GT(st.misses, 300u);
  EXPECT_EQ(st.reads_failed, 0u);
  EXPECT_EQ(sim.host->inflight(), 0u);
}

TEST(FsuFlags, ReadsAheadAreServedPromotedOrDroppedAndTheTablesAgree) {
  // with a noisy prediction of the next layer and slow, paused readers:
  // reads ahead are hit, promoted when the step needs one nobody started,
  // dropped (and their placements undone) when it does not; the device
  // tables equal the replayed mirrors at every prepare (check mode), every
  // gate passes and every expert read holds its own record
  Sim sim(4, 8, 32, 4, 2, 400);
  std::mt19937 rng(7);
  std::uniform_int_distribution<unsigned int> coin(0, 9);
  for (int t = 0; t < 200; ++t) {
    std::vector<std::vector<uint32_t>> ids(4);
    for (unsigned int l = 0; l < 4; ++l)
      ids[l] = route(rng, 4, 32);
    if (t % 5 == 0)
      sim.readers->pause(true); // reads ahead queue up unstarted
    for (unsigned int l = 0; l < 4; ++l) {
      std::vector<uint32_t> pred;
      if (l + 1 < 4) {
        // 4 of the next layer's experts, some right, some wrong
        for (uint32_t e : ids[l + 1])
          pred.push_back(coin(rng) < 6 ? e : (e + 11) % 32);
        std::set<uint32_t> u(pred.begin(), pred.end());
        pred.assign(u.begin(), u.end());
      }
      sim.step(l, ids[l], pred);
      if (t % 5 == 0 && l == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        sim.readers->pause(false);
      }
    }
    sim.dev.finish();
  }
  sim.host->drain();
  EXPECT_EQ(sim.publish_errors.load(), 0u);
  EXPECT_EQ(sim.gate_errors.load(), 0u);
  EXPECT_EQ(sim.bad_data.load(), 0u);
  EXPECT_EQ(sim.host->errorCode(), (uint32_t)kOk);
  const auto st = sim.host->stats();
  EXPECT_EQ(st.steps, 800u);
  EXPECT_EQ(st.specs, 600u);
  EXPECT_GT(st.spec_placed, 0u);
  EXPECT_GT(st.spec_hits, 0u);
  EXPECT_GT(st.spec_cancelled, 0u);
  EXPECT_EQ(st.reads_failed, 0u);
  EXPECT_EQ(sim.host->inflight(), 0u);
  std::printf(
    "[ INFO     ] spec placed %llu hits %llu promoted %llu "
    "cancelled %llu misses %llu\n",
    (unsigned long long)st.spec_placed, (unsigned long long)st.spec_hits,
    (unsigned long long)st.spec_promoted, (unsigned long long)st.spec_cancelled,
    (unsigned long long)st.misses);
  // the mirrors and the policies agree after the drain
  for (unsigned int l = 0; l < 4; ++l) {
    const auto mir = sim.host->mirror(l);
    const auto map = sim.res[l]->slotMap(0);
    ASSERT_EQ(mir.size(), map.size());
    for (size_t e = 0; e < map.size(); ++e)
      EXPECT_EQ(mir[e], map[e]) << "layer " << l << " expert " << e;
  }
}

namespace {

/** what pollBench measured */
struct PollBench {
  double p50_us = 0, p90_us = 0, max_us = 0, cpu_share = 0, gap_p50_us = 0;
  size_t taken = 0, published = 0;
};

/**
 * @brief The poller against a device that publishes like a phone's decode
 *        (32 slots: ~1.7 ms per layer): per layer a step entry, a read
 *        ahead 60 us later, the next step 1.55 ms after that and, every
 *        7th layer, 2 ms more (a gate that waited); the host enqueues each
 *        token's publishes ahead, as the backend does. The latency is
 *        publish -> taken.
 */
PollBench pollBench(FsuFlagsHost::PollConfig c, int tokens) {
  using clk = std::chrono::steady_clock;
  std::vector<uint32_t> ctl(kCtlWords, 0);
  std::vector<uint32_t> block(LayerLayout(4, 16).words, 0);
  ExpertResidency res;
  res.addLayer(0, 4, 16);
  FsuFlagsHost host(
    ctl.data(),
    [](const FsuFlagsHost::Read &, std::shared_ptr<std::atomic<bool>>, bool) {},
    false);
  host.addLayer(0, block.data(), 4, 16, 2, hooksOf(res));
  constexpr int kLayers = 24;
  const size_t n = (size_t)tokens * kLayers * 2 + 2;
  std::vector<clk::time_point> pub(n), take(n);
  std::vector<char> got(n, 0);
  host.onTake([&](uint32_t seq) {
    if (seq < n) {
      take[seq] = clk::now();
      got[seq] = 1;
    }
  });
  host.startPoller(c);
#if defined(__linux__)
  (void)prctl(PR_SET_TIMERSLACK, 1UL, 0UL, 0UL, 0UL);
#endif
  auto spin_until = [](clk::time_point t) {
    while (clk::now() < t) {
    }
  };
  auto publish = [&](uint32_t seq, uint32_t kind) {
    while ((uint32_t)(seq - loadAcquire(ctl.data() + kCtlTail)) >
           kRingEntries) {
    }
    uint32_t *en = ctl.data() + kCtlRing + (seq % kRingEntries) * kEntryWords;
    en[kEnKind] = kind;
    en[kEnLayer] = 0;
    en[kEnCount] = 0;
    en[kEnX] = 0;
    pub[seq] = clk::now();
    storeRelease(en, seq);
  };
  for (int t = 0; t < tokens; ++t) {
    std::vector<uint32_t> seqs;
    for (int l = 0; l < kLayers; ++l) {
      seqs.push_back(host.beginPublish(0, false));
      seqs.push_back(host.beginPublish(0, true));
    }
    auto at = clk::now();
    for (int l = 0; l < kLayers; ++l) {
      publish(seqs[2 * l], kKindStep);
      spin_until(at += std::chrono::microseconds(60));
      publish(seqs[2 * l + 1], kKindSpec);
      at += std::chrono::microseconds(1550 + (l % 7 == 6 ? 2000 : 0));
      spin_until(at);
    }
    // the token's end: sampling and the next token's enqueue
    while (!host.consumed(seqs.back()))
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    std::this_thread::sleep_for(std::chrono::milliseconds(3));
  }
  host.stopPoller();
  PollBench b;
  std::vector<double> lat;
  for (size_t i = 1; i < n; ++i)
    if (pub[i] != clk::time_point()) {
      ++b.published;
      if (got[i]) {
        ++b.taken;
        lat.push_back(
          std::chrono::duration<double, std::micro>(take[i] - pub[i]).count());
      }
    }
  std::sort(lat.begin(), lat.end());
  if (!lat.empty()) {
    b.p50_us = lat[lat.size() / 2];
    b.p90_us = lat[lat.size() * 9 / 10];
    b.max_us = lat.back();
  }
  const auto st = host.stats();
  b.cpu_share =
    st.poller_wall_us ? (double)st.poller_cpu_us / st.poller_wall_us : 0.0;
  std::vector<uint32_t> g = st.poll_gap_us;
  std::sort(g.begin(), g.end());
  b.gap_p50_us = g.empty() ? 0.0 : g[g.size() / 2];
  return b;
}

} // namespace

TEST(FsuFlags, TheAdaptivePollerTakesEveryEntrySoonWithLessCpu) {
  // the spin poller (the earlier default) against the adaptive one at a
  // phone's publish cadence: every entry taken, the adaptive one at a
  // fraction of the CPU, and soon (target: median <= 10 us)
  FsuFlagsHost::PollConfig spin;
  spin.spin_us = 2000;
  FsuFlagsHost::PollConfig adaptive;
  adaptive.adaptive = true;
  adaptive.spin_us = 100;
  adaptive.max_sleep_us = 50;
  const PollBench a = pollBench(spin, 30);
  const PollBench b = pollBench(adaptive, 30);
  for (const auto &x :
       {std::make_pair("spin", a), std::make_pair("adaptive", b)})
    std::printf("[ INFO     ] poller %s: entries %zu/%zu latency p50 %.1f us "
                "p90 %.1f us max %.1f us, last-empty-poll gap p50 %.1f us, "
                "cpu %.1f %%\n",
                x.first, x.second.taken, x.second.published, x.second.p50_us,
                x.second.p90_us, x.second.max_us, x.second.gap_p50_us,
                100.0 * x.second.cpu_share);
  EXPECT_EQ(a.taken, a.published);
  EXPECT_EQ(b.taken, b.published);
  EXPECT_GT(a.cpu_share, 0.0);
  EXPECT_LT(b.cpu_share, a.cpu_share);
  // generous: a loaded CI host; the figures are printed above
  EXPECT_LT(b.p50_us, 60.0);
}

TEST(FsuFlags, TheLastTokensStepsAreTalliedNotReplayed) {
  // a step is replayed into the policy at its layer's next prepare, so the
  // last token's never is; at exit tallyUnreplayed counts it through the
  // tally hook (the model's totals) without touching the policy
  std::vector<uint32_t> ctl_mem(kCtlWords, 0);
  LayerLayout L(4, 16);
  std::vector<uint32_t> block(L.words, 0);
  ExpertResidency res;
  res.addLayer(0, 4, 16);
  unsigned int t_steps = 0, t_hits = 0, t_misses = 0;
  FsuFlagsHost::Hooks h = hooksOf(res);
  h.tally = [&](unsigned int hits, unsigned int misses) {
    ++t_steps;
    t_hits += hits;
    t_misses += misses;
  };
  std::vector<std::pair<FsuFlagsHost::Read, std::shared_ptr<std::atomic<bool>>>>
    reads;
  FsuFlagsHost host(
    ctl_mem.data(),
    [&](const FsuFlagsHost::Read &r, std::shared_ptr<std::atomic<bool>> once,
        bool) { reads.emplace_back(r, std::move(once)); },
    true);
  host.addLayer(0, block.data(), 4, 16, 2, std::move(h));
  auto step = [&](uint32_t a, uint32_t b) {
    if (host.needsPrepare(0))
      host.prepare(0);
    const uint32_t seq = host.beginPublish(0, false);
    const uint32_t ids[2] = {a, b}, ext[2] = {a, b};
    uint32_t slot[2], tag[2];
    ASSERT_EQ(refPublish(ctl_mem.data(), block.data(), L, seq, 0, ids, 2, ext,
                         2, slot, tag, 1000),
              (uint32_t)kOk);
    ASSERT_TRUE(host.pollOnce());
    for (auto &r : reads)
      host.readDone(r.first, r.second->exchange(true) ? ECANCELED : 0);
    reads.clear();
  };
  step(1, 2); // 2 misses
  step(1, 3); // replays the first; 1 hit, 1 miss
  EXPECT_EQ(res.misses(), 2u);
  EXPECT_EQ(host.tallyUnreplayed(), 1u);
  EXPECT_EQ(t_steps, 1u);
  EXPECT_EQ(t_hits, 1u);
  EXPECT_EQ(t_misses, 1u);
  EXPECT_EQ(res.misses(), 2u);           // the policy is untouched
  EXPECT_EQ(host.tallyUnreplayed(), 0u); // once
  EXPECT_EQ(host.stats().steps, 2u);
  EXPECT_EQ(host.stats().misses, 3u); // = replayed 2 + tallied 1
}

TEST(FsuFlags, AfterADrainAPolicyChangedOutsideIsAdopted) {
  // a prefill between decodes: drain, the policy forgets everything and the
  // host fills some slots itself; the next prepare adopts that (host tags,
  // ready at once), and a step hitting those experts passes its gate with
  // no read
  Sim sim(1, 4, 16, 2, 1, 0);
  std::mt19937 rng(3);
  for (int t = 0; t < 20; ++t) {
    sim.step(0, route(rng, 2, 16), {});
    sim.dev.finish();
  }
  sim.host->drain();
  EXPECT_TRUE(sim.host->needsPrepare(0));
  sim.res[0]->clear(0);
  auto a = sim.res[0]->acquire(0, {9, 12});
  for (const auto &mi : a.misses) // the host path's own (synchronous) fill
    for (uint32_t i = 0; i < kRec; ++i)
      sim.mem[0][(size_t)mi.slot * kRec + i] = ident(0, mi.expert);
  const uint64_t reads_before = sim.readers->done.load();
  sim.step(0, {9, 12}, {});
  sim.dev.finish();
  EXPECT_EQ(sim.readers->done.load(), reads_before); // no read: hits
  EXPECT_EQ(sim.gate_errors.load(), 0u);
  EXPECT_EQ(sim.bad_data.load(), 0u);
  EXPECT_EQ(sim.host->stats().adopts, 1u);
  // and the decode goes on from there
  for (int t = 0; t < 20; ++t) {
    sim.step(0, route(rng, 2, 16), {});
    sim.dev.finish();
  }
  sim.host->drain();
  EXPECT_EQ(sim.gate_errors.load(), 0u);
  EXPECT_EQ(sim.bad_data.load(), 0u);
  EXPECT_EQ(sim.host->errorCode(), (uint32_t)kOk);
}

namespace {

/** the flags kernel @a name built as the backend builds it: OpenCL C 2.0,
 *  else 3.0 (a device without 2.0, e.g. Xe3) */
nntrainer::ClContext::SharedPtrClKernel flagsKernel(nntrainer::ClContext *cc,
                                                    const char *name) {
  for (const char *std : {"-cl-std=CL2.0", "-cl-std=CL3.0"}) {
    auto k = cc->registerClKernel(nntrainer::tq2moe_fsu_flags_kernel, name,
                                  std + clDefines());
    if (k)
      return k;
  }
  return nullptr;
}

/** an SVM allocation of @a words the host reads and writes between
 *  launches (coarse-grain: mapped around every host access) */
struct SvmWords {
  cl_context ctx;
  cl_command_queue q;
  uint32_t *p = nullptr;
  size_t n;
  SvmWords(cl_context c, cl_command_queue qq, size_t words) :
    ctx(c), q(qq), n(words) {
    p = static_cast<uint32_t *>(nntrainer::opencl::clSVMAllocT(
      ctx, CL_MEM_READ_WRITE, n * sizeof(uint32_t), 0));
  }
  ~SvmWords() {
    if (p != nullptr)
      nntrainer::opencl::clSVMFreeT(ctx, p);
  }
  void map() {
    nntrainer::opencl::clEnqueueSVMMap(q, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE,
                                       p, n * sizeof(uint32_t), 0, nullptr,
                                       nullptr);
  }
  void unmap() {
    nntrainer::opencl::clEnqueueSVMUnmap(q, p, 0, nullptr, nullptr);
    nntrainer::opencl::clFinish(q);
  }
};

/** a device buffer of @a v */
cl_mem buf(cl_context ctx, std::vector<uint32_t> v) {
  cl_int err = CL_SUCCESS;
  return nntrainer::opencl::clCreateBufferT(
    ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, v.size() * sizeof(uint32_t),
    v.data(), &err);
}

std::vector<uint32_t> readBack(cl_command_queue q, cl_mem m, size_t n) {
  std::vector<uint32_t> v(n);
  nntrainer::opencl::clEnqueueReadBuffer(q, m, CL_TRUE, 0, n * sizeof(uint32_t),
                                         v.data(), 0, nullptr, nullptr);
  return v;
}

} // namespace

/**
 * The .cl kernels against their reference forms, one launch at a time (no
 * host concurrency: any device with SVM, coarse-grain included): a read
 * ahead (publish @a s0), a step with hits and misses (publish s0 + 1), a
 * gate that passes and one that times out -- the same words in the control
 * and layer blocks, the same offsets, wait list (tags) and weights.
 */
void kernelsVsReference(uint32_t s0) {
  auto *cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));
  if (cc == nullptr || nntrainer::opencl::clSVMAlloc == nullptr ||
      nntrainer::opencl::clSetKernelArgSVMPointer == nullptr)
    GTEST_SKIP() << "no OpenCL device with SVM";
  auto kpub = flagsKernel(cc, "tq2moe_fsu_publish");
  auto kspec = flagsKernel(cc, "tq2moe_fsu_spec");
  auto kgate = flagsKernel(cc, "tq2moe_fsu_gate");
  ASSERT_TRUE(kpub && kspec && kgate)
    << "tq2moe_fsu_flags.cl does not build here";
  cl_context ctx = cc->context_inst_.GetContextNoRetain();
  cl_command_queue q = cc->command_queue_inst_.GetCommandQueue();
  const uint32_t C = 8, E = 32, K = 4, X = 6, keep = 4;
  const LayerLayout L(C, E);
  SvmWords ctl(ctx, q, kCtlWords), lay(ctx, q, L.words);
  if (ctl.p == nullptr || lay.p == nullptr)
    GTEST_SKIP() << "no SVM allocation";
  // a table the host would write: experts 1, 5, 9 resident (host tags),
  // the rest free, through the host's own prepare
  ExpertResidency res;
  res.addLayer(0, C, E);
  res.acquire(0, {1, 5, 9});
  ctl.map();
  lay.map();
  std::fill(ctl.p, ctl.p + kCtlWords, 0u);
  // the host consumed every publish before s0
  ctl.p[kCtlTail] = s0 - 1u;
  {
    FsuFlagsHost host(ctl.p, [](const FsuFlagsHost::Read &,
                                std::shared_ptr<std::atomic<bool>>, bool) {});
    host.addLayer(0, lay.p, C, E, K, hooksOf(res));
    host.prepare(0);
  }
  std::vector<uint32_t> rctl(ctl.p, ctl.p + kCtlWords),
    rlay(lay.p, lay.p + L.words);
  ctl.unmap();
  lay.unmap();

  // the reference
  const std::vector<uint32_t> pred = {2, 5, 3, 7}; // 5 resident
  const std::vector<uint32_t> ids = {1, 3, 9, 20}, ext = {9, 3, 1, 20, 6, 0};
  std::vector<uint32_t> shift(E);
  for (uint32_t e = 0; e < E; ++e)
    shift[e] = (e * 16) % 1024;
  const uint32_t stride_w = 1000, gu_w = 300;
  ASSERT_EQ(
    refSpec(rctl.data(), rlay.data(), L, s0, 0, pred.data(), 4, keep, 1000),
    kOk);
  uint32_t rs[K], rt[K];
  ASSERT_EQ(refPublish(rctl.data(), rlay.data(), L, s0 + 1u, 0, ids.data(), K,
                       ext.data(), X, rs, rt, 1000),
            kOk);
  std::vector<uint32_t> roff_gu(2 * K), roff_d(K), rwait(2 * K);
  for (uint32_t a = 0; a < K; ++a) {
    const uint32_t base = rs[a] * stride_w + shift[ids[a]];
    roff_gu[a] = base;
    roff_gu[K + a] = base + gu_w;
    roff_d[a] = base + 2 * gu_w;
    rwait[a] = rs[a];
    rwait[K + a] = rt[a];
  }

  // the device
  cl_mem d_pred = buf(ctx, pred), d_ids = buf(ctx, ids), d_ext = buf(ctx, ext),
         d_shift = buf(ctx, shift),
         d_ogu = buf(ctx, std::vector<uint32_t>(2 * K, 0)),
         d_od = buf(ctx, std::vector<uint32_t>(K, 0)),
         d_wait = buf(ctx, std::vector<uint32_t>(2 * K, 0));
  const cl_kernel ks = kspec->GetKernel(), kp = kpub->GetKernel(),
                  kg = kgate->GetKernel();
  using nntrainer::opencl::clSetKernelArg;
  using nntrainer::opencl::clSetKernelArgSVMPointer;
  const cl_ulong cap = 100000;
  uint32_t seq = s0, layer = 0, m = 4;
  ASSERT_EQ(clSetKernelArgSVMPointer(ks, 0, ctl.p), CL_SUCCESS);
  clSetKernelArgSVMPointer(ks, 1, lay.p);
  clSetKernelArg(ks, 2, sizeof(cl_mem), &d_pred);
  clSetKernelArg(ks, 3, sizeof(uint32_t), &seq);
  clSetKernelArg(ks, 4, sizeof(uint32_t), &layer);
  clSetKernelArg(ks, 5, sizeof(uint32_t), &C);
  clSetKernelArg(ks, 6, sizeof(uint32_t), &E);
  clSetKernelArg(ks, 7, sizeof(uint32_t), &m);
  clSetKernelArg(ks, 8, sizeof(uint32_t), &keep);
  clSetKernelArg(ks, 9, sizeof(cl_ulong), &cap);
  const size_t one = 1;
  ASSERT_EQ(nntrainer::opencl::clEnqueueNDRangeKernel(
              q, ks, 1, nullptr, &one, &one, 0, nullptr, nullptr),
            CL_SUCCESS);
  seq = s0 + 1u;
  clSetKernelArgSVMPointer(kp, 0, ctl.p);
  clSetKernelArgSVMPointer(kp, 1, lay.p);
  clSetKernelArg(kp, 2, sizeof(cl_mem), &d_ids);
  clSetKernelArg(kp, 3, sizeof(cl_mem), &d_ext);
  clSetKernelArg(kp, 4, sizeof(cl_mem), &d_ogu);
  clSetKernelArg(kp, 5, sizeof(cl_mem), &d_od);
  clSetKernelArg(kp, 6, sizeof(cl_mem), &d_shift);
  clSetKernelArg(kp, 7, sizeof(cl_mem), &d_wait);
  clSetKernelArg(kp, 8, sizeof(uint32_t), &seq);
  clSetKernelArg(kp, 9, sizeof(uint32_t), &layer);
  clSetKernelArg(kp, 10, sizeof(uint32_t), &C);
  clSetKernelArg(kp, 11, sizeof(uint32_t), &E);
  clSetKernelArg(kp, 12, sizeof(uint32_t), &K);
  clSetKernelArg(kp, 13, sizeof(uint32_t), &X);
  clSetKernelArg(kp, 14, sizeof(uint32_t), &stride_w);
  clSetKernelArg(kp, 15, sizeof(uint32_t), &gu_w);
  clSetKernelArg(kp, 16, sizeof(cl_ulong), &cap);
  ASSERT_EQ(nntrainer::opencl::clEnqueueNDRangeKernel(
              q, kp, 1, nullptr, &one, &one, 0, nullptr, nullptr),
            CL_SUCCESS);
  nntrainer::opencl::clFinish(q);
  EXPECT_EQ(readBack(q, d_ogu, 2 * K), roff_gu);
  EXPECT_EQ(readBack(q, d_od, K), roff_d);
  EXPECT_EQ(readBack(q, d_wait, 2 * K), rwait);
  ctl.map();
  lay.map();
  EXPECT_TRUE(std::equal(rctl.begin(), rctl.end(), ctl.p)) << "control block";
  EXPECT_TRUE(std::equal(rlay.begin(), rlay.end(), lay.p)) << "layer block";
  // the gate: lanes 0 and 2 (hits on host-filled slots) are ready; the
  // misses' ready words are set for lane 1 only: lane 3 times out
  lay.p[L.have(rs[1])] = rt[1];
  rlay[L.have(rs[1])] = rt[1];
  ctl.unmap();
  lay.unmap();
  std::vector<float> rw = {0.5f, 0.25f, 0.125f, 0.0625f};
  uint64_t slowest = 0;
  for (uint32_t a = 0; a < K; ++a) {
    uint64_t n = 0;
    refGateLane(rctl.data(), rlay.data(), L, layer, a, rwait[a], rwait[K + a],
                cap, rw.data(), &n);
    slowest = std::max(slowest, n);
  }
  refGateAccount(rctl.data(), slowest);
  cl_int err = CL_SUCCESS;
  std::vector<float> w0 = {0.5f, 0.25f, 0.125f, 0.0625f};
  cl_mem d_w = nntrainer::opencl::clCreateBufferT(
    ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, K * sizeof(float), w0.data(),
    &err);
  clSetKernelArgSVMPointer(kg, 0, ctl.p);
  clSetKernelArgSVMPointer(kg, 1, lay.p);
  clSetKernelArg(kg, 2, sizeof(cl_mem), &d_wait);
  clSetKernelArg(kg, 3, sizeof(cl_mem), &d_w);
  clSetKernelArg(kg, 4, sizeof(uint32_t), &layer);
  clSetKernelArg(kg, 5, sizeof(uint32_t), &C);
  clSetKernelArg(kg, 6, sizeof(uint32_t), &K);
  clSetKernelArg(kg, 7, sizeof(cl_ulong), &cap);
  const size_t gk = 16;
  ASSERT_EQ(nntrainer::opencl::clEnqueueNDRangeKernel(q, kg, 1, nullptr, &gk,
                                                      &gk, 0, nullptr, nullptr),
            CL_SUCCESS);
  nntrainer::opencl::clFinish(q);
  std::vector<float> w(K);
  nntrainer::opencl::clEnqueueReadBuffer(q, d_w, CL_TRUE, 0, K * sizeof(float),
                                         w.data(), 0, nullptr, nullptr);
  for (uint32_t a = 0; a < K; ++a)
    EXPECT_EQ(std::isnan(w[a]), std::isnan(rw[a])) << "lane " << a;
  EXPECT_TRUE(std::isnan(w[3]));
  EXPECT_FALSE(std::isnan(w[0]) || std::isnan(w[1]) || std::isnan(w[2]));
  ctl.map();
  EXPECT_EQ(ctl.p[kCtlError], (uint32_t)kErrTimeout);
  EXPECT_EQ(ctl.p[kCtlErrInfo], 3u);
  // the spin account: lanes 0-2 ready at once, lane 3 spun to the cap
  for (uint32_t i :
       {kCtlSpinLo, kCtlSpinHi, kCtlGates, kCtlSpinGates, kCtlSpinMax})
    EXPECT_EQ(ctl.p[i], rctl[i]) << "spin account word " << i;
  EXPECT_EQ(spinAccount(ctl.p).polls, cap + 1);
  EXPECT_EQ(spinAccount(ctl.p).gates, 1u);
  ctl.unmap();
  for (cl_mem mm : {d_pred, d_ids, d_ext, d_shift, d_ogu, d_od, d_wait, d_w})
    nntrainer::opencl::clReleaseMemObjectT(mm);
}

TEST(FsuFlags, TheKernelsDoWhatTheReferenceDoesOnThisDevice) {
  kernelsVsReference(1);
}

TEST(FsuFlags, TheGatePollIsMeasuredOnThisDevice) {
  // the poll time the stats turn the spin account into ms with: the gate
  // kernel on scratch blocks at two caps (coarse-grain SVM on Xe3)
  auto *cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));
  if (cc == nullptr || nntrainer::opencl::clSVMAlloc == nullptr)
    GTEST_SKIP() << "no OpenCL device with SVM";
  const double ns = nntrainer::tq2cx_fsu_gate_poll_ns_cl();
  std::printf("gate poll: %.1f ns\n", ns);
  EXPECT_GT(ns, 0.0);
  EXPECT_LT(ns, 100000.0);
}

TEST(FsuFlags, TheKernelsTagLikeTheHostAcrossThe27BitWrap) {
  // the publish at 2^27 (a plain 27-bit mask: tag 0 for item 0) and one
  // past 2^31 (a 28-bit mask: bit 31 set): the device's tags are the host's
  kernelsVsReference(kTagSeqPeriod);
  kernelsVsReference(0x80000010u);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
