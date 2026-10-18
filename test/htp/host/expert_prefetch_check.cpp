// SPDX-License-Identifier: Apache-2.0
/**
 * @file expert_prefetch_check.cpp
 * @brief Concurrent ownership, cancellation and zero-copy adoption checks.
 * @author NNTrainer contributors
 * @bug No known bugs
 */
#include "htp_expert_prefetch.h"
#include "htp_router_predict.h"
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <future>
#include <limits>
#include <memory>
#include <stdexcept>

struct Buffer {
  std::shared_ptr<std::array<unsigned, 1024>> data;
  unsigned value = 0, handle = 0;
};
using Queue = nntrainer::HtpExpertPrefetch<Buffer>;

static void completed(Queue &q, uint64_t count) {
  const auto deadline =
    std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (q.stats().completed < count) {
    assert(std::chrono::steady_clock::now() < deadline);
    std::this_thread::yield();
  }
}

/** Wait for a controlled reader checkpoint instead of relying on sleeps. */
static void checkpoint(std::future<void> &event) {
  assert(event.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
}

/** A stale read skips its second half but retains its slot until it exits. */
static void cooperativeCancellation_p() {
  int keys[2]{};
  auto data = std::make_shared<std::array<unsigned, 1024>>();
  data->fill(99);
  std::promise<void> gu_done, check_release, cancel_seen, exit_release;
  auto gu_ready = gu_done.get_future();
  auto check_allowed = check_release.get_future().share();
  auto cancel_ready = cancel_seen.get_future();
  auto exit_allowed = exit_release.get_future().share();
  std::atomic<unsigned> down_reads{0};
  Queue q({{data}}, 1, [&](Buffer &b, const Queue::CancelCheck &cancelled) {
    auto middle = b.data->begin() + b.data->size() / 2;
    std::fill(b.data->begin(), middle, b.value);
    if (b.value == 1) {
      gu_done.set_value();
      check_allowed.wait();
    }
    if (cancelled()) {
      cancel_seen.set_value();
      exit_allowed.wait();
      return Queue::ReadResult::Cancelled;
    }
    ++down_reads;
    std::fill(middle, b.data->end(), b.value);
    return Queue::ReadResult::Complete;
  });
  assert(q.submit(keys, 1, [](Buffer &b) { b.value = 1; }));
  checkpoint(gu_ready);
  q.expireBefore(2);
  check_release.set_value();
  checkpoint(cancel_ready);
  // The canceled callback still owns the buffer, including its partial data.
  assert(!q.claim(keys));
  assert(!q.take(keys, [](Buffer &) { assert(false); }));
  assert(!q.submit(keys + 1, 2, [](Buffer &) { assert(false); }));
  assert(q.stats().completed == 0 && q.stats().cancelled == 0);
  assert(down_reads == 0);
  for (size_t i = 0; i < data->size(); ++i)
    assert((*data)[i] == (i < data->size() / 2 ? 1u : 99u));
  exit_release.set_value();
  q.drain();
  assert(q.stats().cancelled == 1 && q.stats().failed == 0 &&
         q.stats().completed == 0 && q.stats().discarded == 1);
  assert(!q.take(keys, [](Buffer &) { assert(false); }));
  assert(q.submit(keys + 1, 2, [&](Buffer &b) {
    assert(b.data == data);
    b.value = 2;
  }));
  completed(q, 1);
  assert(q.take(keys + 1, [](Buffer &b) {
    for (unsigned value : *b.data)
      assert(value == 2);
  }));
  assert(down_reads == 1 && q.stats().used == 1 && q.stats().cancelled == 1);
}

/** Claimed in-flight and queued reads survive expiry and finish both halves. */
static void claimedCancellationGuard_p() {
  int keys[2]{};
  std::vector<Buffer> buffers;
  for (unsigned i = 0; i < 2; ++i)
    buffers.push_back({std::make_shared<std::array<unsigned, 1024>>()});
  std::promise<void> gu_done, release;
  auto gu_ready = gu_done.get_future();
  auto released = release.get_future().share();
  std::atomic<unsigned> down_reads{0};
  Queue q(std::move(buffers), 1,
          [&](Buffer &b, const Queue::CancelCheck &cancelled) {
            auto middle = b.data->begin() + b.data->size() / 2;
            std::fill(b.data->begin(), middle, b.value);
            if (b.value == 1) {
              gu_done.set_value();
              released.wait();
            }
            assert(!cancelled());
            ++down_reads;
            std::fill(middle, b.data->end(), b.value);
            return Queue::ReadResult::Complete;
          });
  assert(q.submit(keys, 1, [](Buffer &b) { b.value = 1; }));
  checkpoint(gu_ready);
  assert(q.submit(keys + 1, 1, [](Buffer &b) { b.value = 2; }));
  assert(q.claim(keys) && q.claim(keys + 1));
  q.expireBefore(2);
  release.set_value();
  assert(q.takeClaimed(keys + 1, [](Buffer &b) {
    for (unsigned value : *b.data)
      assert(value == 2);
  }));
  assert(q.takeClaimed(keys, [](Buffer &b) {
    for (unsigned value : *b.data)
      assert(value == 1);
  }));
  assert(down_reads == 2 && q.stats().completed == 2 && q.stats().used == 2 &&
         q.stats().cancelled == 0 && q.stats().discarded == 0 &&
         q.stats().joined == 2);
}

/** Expiry after the last cancel check must still prevent stale adoption. */
static void lateCompletion_n() {
  int key{};
  auto data = std::make_shared<std::array<unsigned, 1024>>();
  std::promise<void> checked, release;
  auto checked_ready = checked.get_future();
  auto released = release.get_future().share();
  Queue q({{data}}, 1, [&](Buffer &b, const Queue::CancelCheck &cancelled) {
    assert(!cancelled());
    checked.set_value();
    released.wait();
    b.data->fill(1);
    return Queue::ReadResult::Complete;
  });
  assert(q.submit(&key, 1, [](Buffer &) {}));
  checkpoint(checked_ready);
  q.expireBefore(2);
  release.set_value();
  q.drain();
  assert(!q.take(&key, [](Buffer &) { assert(false); }));
  assert(q.stats().completed == 1 && q.stats().cancelled == 0 &&
         q.stats().failed == 0 && q.stats().discarded == 1);
}

/** Read errors and exceptions wake a claimed demand and remain failures. */
static void cancellableFailures_n() {
  int key{};
  for (bool throws : {false, true}) {
    auto data = std::make_shared<std::array<unsigned, 1024>>();
    std::promise<void> entered, release;
    auto entered_ready = entered.get_future();
    auto released = release.get_future().share();
    Queue q({{data}}, 1, [&](Buffer &, const Queue::CancelCheck &cancelled) {
      entered.set_value();
      released.wait();
      assert(!cancelled());
      if (throws)
        throw std::runtime_error("cancellable read failure");
      return Queue::ReadResult::Failed;
    });
    assert(q.submit(&key, 1, [](Buffer &) {}));
    checkpoint(entered_ready);
    assert(q.claim(&key));
    q.expireBefore(2);
    release.set_value();
    assert(!q.takeClaimed(&key, [](Buffer &) { assert(false); }));
    assert(q.stats().completed == 1 && q.stats().failed == 1 &&
           q.stats().cancelled == 0 && q.stats().used == 0);
  }
}

/** Only unread queued work moves to demand; no buffer is borrowed. */
static void queuedDemandHandoff_p() {
  int keys[3]{};
  auto active = std::make_shared<std::array<unsigned, 1024>>();
  auto queued = std::make_shared<std::array<unsigned, 1024>>();
  queued->fill(99);
  std::promise<void> entered, release;
  auto started = entered.get_future();
  auto allowed = release.get_future().share();
  Queue q({{active}, {queued}}, 1, [&](Buffer &b) {
    if (b.value == 1) {
      entered.set_value();
      allowed.wait();
    }
    b.data->fill(b.value);
    return true;
  });
  assert(q.submit(keys, 1, [](Buffer &b) { b.value = 1; }));
  checkpoint(started);
  assert(q.submit(keys + 1, 1, [](Buffer &b) { b.value = 2; }));
  assert(q.claim(keys, true)); // Already Reading: keep the ordinary claim.
  assert(!q.claim(keys + 1, true));
  assert(!q.claim(keys + 1, true)); // No second removal of the queued index.
  assert(!q.take(keys + 1, [](Buffer &) { assert(false); }));
  assert(!q.takeClaimed(keys + 1, [](Buffer &) { assert(false); }));
  // The tombstone rejects a late predictor until the miss batch expires.
  assert(!q.submit(keys + 1, 1, [](Buffer &) { assert(false); }));
  assert(!q.submit(keys + 2, 2, [](Buffer &) { assert(false); }));
  for (unsigned value : *queued)
    assert(value == 99);
  q.expireBefore(2);
  assert(!q.submit(keys + 1, 1, [](Buffer &) { assert(false); }));
  assert(q.submit(keys + 2, 2, [&](Buffer &b) {
    assert(b.data == queued);
    b.value = 3;
  }));
  release.set_value();
  completed(q, 2);
  assert(q.takeClaimed(keys, [](Buffer &b) {
    for (unsigned value : *b.data)
      assert(value == 1);
  }));
  assert(q.claim(keys + 2, true)); // Ready is also consumed normally.
  assert(q.takeClaimed(keys + 2, [](Buffer &b) {
    for (unsigned value : *b.data)
      assert(value == 3);
  }));
  const auto stats = q.stats();
  assert(stats.handed_off == 1 && stats.discarded == 1 && stats.joined == 1 &&
         stats.completed == 2 && stats.used == 2 && stats.failed == 0);
}

/** An error before demand dispatch can drain a removed queued key safely. */
static void abandonedHandoffDrain_p() {
  int keys[2]{};
  auto a = std::make_shared<std::array<unsigned, 1024>>();
  auto b = std::make_shared<std::array<unsigned, 1024>>();
  std::promise<void> entered, release;
  auto started = entered.get_future();
  auto allowed = release.get_future().share();
  Queue q({{a}, {b}}, 1, [&](Buffer &buffer) {
    if (buffer.value == 1) {
      entered.set_value();
      allowed.wait();
    }
    return true;
  });
  assert(q.submit(keys, 1, [](Buffer &buffer) { buffer.value = 1; }));
  checkpoint(started);
  assert(q.submit(keys + 1, 1, [](Buffer &buffer) { buffer.value = 2; }));
  assert(!q.claim(keys + 1, true));
  release.set_value();
  q.drain();
  assert(q.stats().completed == 1 && q.stats().handed_off == 1 &&
         q.stats().discarded == 2);
  assert(q.submit(keys + 1, 2, [](Buffer &buffer) { buffer.value = 3; }));
  completed(q, 2);
  assert(q.take(keys + 1, [](Buffer &) {}));
}

int main() {
  queuedDemandHandoff_p();
  abandonedHandoffDrain_p();
  cooperativeCancellation_p();
  claimedCancellationGuard_p();
  lateCompletion_n();
  cancellableFailures_n();
  int keys[8]{};
  std::promise<void> entered, release;
  auto released = release.get_future().share();
  auto data = std::make_shared<std::array<unsigned, 1024>>();
  Queue q({{data}}, 1, [&](Buffer &b) {
    if (b.value == 1) {
      entered.set_value();
      released.wait();
    }
    if (b.value == 3)
      return false;
    if (b.value == 4)
      throw std::runtime_error("injected read failure");
    b.data->fill(b.value);
    return true;
  });
  assert(q.submit(keys, 1, [](Buffer &b) { b.value = 1; }));
  entered.get_future().wait();
  assert(!q.take(keys, [](Buffer &) { assert(false); }));
  assert(!q.submit(keys, 1, [](Buffer &) { assert(false); }));
  q.expireBefore(2);
  assert(!q.submit(keys + 3, 1, [](Buffer &) { assert(false); }));
  // Cancellation cannot release the buffer while its reader is writing.
  assert(!q.submit(keys + 1, 2, [](Buffer &) { assert(false); }));
  release.set_value();
  q.drain();
  assert(q.submit(keys + 1, 2, [](Buffer &b) { b.value = 2; }));
  completed(q, 2);
  Buffer victim{std::make_shared<std::array<unsigned, 1024>>(), 99, 123};
  auto retired = victim.data;
  assert(q.take(keys + 1, [&](Buffer &ready) {
    // Same handle transfer as the HTP miss answer; only buffers exchange.
    ready.handle = victim.handle;
    victim.handle = 0;
    std::swap(ready, victim);
  }));
  assert(victim.data == data && victim.handle == 123);
  for (auto value : *victim.data)
    assert(value == 2);
  assert(!q.take(keys + 1, [](Buffer &) { assert(false); }));
  assert(q.submit(keys + 2, 3, [&](Buffer &b) {
    assert(b.data == retired && b.handle == 0);
    b.value = 3;
  }));
  completed(q, 3);
  assert(!q.take(keys + 2, [](Buffer &) { assert(false); }));
  assert(q.submit(keys + 3, 4, [](Buffer &b) { b.value = 4; }));
  completed(q, 4);
  assert(q.stats().failed == 2 && q.stats().used == 1 && q.stats().late == 1);

  // A canceled queued index must not later execute a replacement job twice.
  std::promise<void> entered2, release2;
  auto released2 = release2.get_future().share();
  Queue queued({{data}, {retired}}, 1, [&](Buffer &b) {
    if (b.value == 1) {
      entered2.set_value();
      released2.wait();
    }
    b.data->fill(b.value);
    return true;
  });
  assert(queued.submit(keys, 10, [](Buffer &b) { b.value = 1; }));
  entered2.get_future().wait();
  assert(queued.submit(keys + 1, 10, [](Buffer &b) { b.value = 10; }));
  queued.expireBefore(11);
  assert(queued.submit(keys + 2, 11, [](Buffer &b) { b.value = 11; }));
  release2.set_value();
  completed(queued, 2);
  assert(queued.take(keys + 2, [](Buffer &b) {
    for (auto value : *b.data)
      assert(value == 11);
  }));
  queued.drain();
  assert(queued.stats().completed == 2);

  // A real miss keeps in-flight/queued guesses alive across expiry. Other
  // guesses are canceled, and no consumer can steal a claimed buffer.
  std::promise<void> claim_entered, claim_release;
  auto claim_released = claim_release.get_future().share();
  Queue claimed(
    {{data}, {retired}, {std::make_shared<std::array<unsigned, 1024>>()}}, 1,
    [&](Buffer &b) {
      if (b.value == 1) {
        claim_entered.set_value();
        claim_released.wait();
      }
      assert(b.value != 2); // This queued wrong guess must never read.
      b.data->fill(b.value);
      return true;
    });
  assert(claimed.submit(keys, 20, [](Buffer &b) { b.value = 1; }));
  claim_entered.get_future().wait();
  assert(claimed.submit(keys + 1, 20, [](Buffer &b) { b.value = 2; }));
  assert(claimed.submit(keys + 2, 20, [](Buffer &b) { b.value = 3; }));
  assert(claimed.claim(keys) && claimed.claim(keys + 2));
  assert(!claimed.claim(keys));
  assert(!claimed.take(keys, [](Buffer &) { assert(false); }));
  claimed.expireBefore(21);
  claim_release.set_value();
  assert(claimed.takeClaimed(keys + 2, [](Buffer &b) {
    for (auto v : *b.data)
      assert(v == 3);
  }));
  assert(claimed.takeClaimed(keys, [](Buffer &b) {
    for (auto v : *b.data)
      assert(v == 1);
  }));
  claimed.drain();
  assert(claimed.stats().completed == 2 && claimed.stats().used == 2 &&
         claimed.stats().discarded == 1 && claimed.stats().joined == 2);
  assert(!claimed.takeClaimed(keys, [](Buffer &) { assert(false); }));
  // Failures must wake a joined demand read, which can then fall back.
  for (bool throws : {false, true}) {
    std::promise<void> failure_entered, failure_release;
    auto failure_released = failure_release.get_future().share();
    Queue failure({{data}}, 1, [&](Buffer &) -> bool {
      failure_entered.set_value();
      failure_released.wait();
      if (throws)
        throw std::runtime_error("claimed failure");
      return false;
    });
    assert(failure.submit(keys, 1, [](Buffer &) {}));
    failure_entered.get_future().wait();
    assert(failure.claim(keys));
    failure.expireBefore(2);
    failure_release.set_value();
    assert(!failure.takeClaimed(keys, [](Buffer &) { assert(false); }));
    failure.drain();
    assert(failure.stats().failed == 1);
  }

  // Repeated reuse across workers, tokens and failed predictions.
  std::vector<Buffer> buffers;
  for (unsigned i = 0; i < 8; ++i)
    buffers.push_back({std::make_shared<std::array<unsigned, 1024>>()});
  Queue stress(std::move(buffers), 4, [](Buffer &b) {
    b.data->fill(b.value);
    return true;
  });
  for (uint64_t epoch = 1; epoch <= 1000; ++epoch) {
    for (unsigned i = 0; i < 8; ++i)
      assert(stress.submit(keys + i, epoch, [epoch, i](Buffer &b) {
        b.value = static_cast<unsigned>(epoch * 8 + i);
      }));
    completed(stress, epoch * 8);
    for (unsigned i = 0; i < 4; ++i)
      assert(stress.take(keys + i, [epoch, i](Buffer &b) {
        for (auto value : *b.data)
          assert(value == epoch * 8 + i);
      }));
    stress.expireBefore(epoch + 1);
  }
  stress.drain();
  assert(stress.stats().used == 4000 && stress.stats().discarded == 4000);
  // Learn only adjacent layers in one token; apply only to the next token.
  nntrainer::HtpRouterResidual residual(4, 3);
  const float first[] = {1, 2, 3}, second[] = {3, 1, 5};
  const float current[] = {5, 6, 7};
  float corrected[3]{};
  residual.observe(0, 10, first);
  residual.observe(1, 10, second);
  residual.apply(1, 11, current, 0.5f, corrected);
  assert(corrected[0] == 6 && corrected[1] == 5.5f && corrected[2] == 8);
  residual.apply(1, 12, current, 1, corrected);
  assert(std::equal(corrected, corrected + 3, current));
  residual.observe(3, 10, first); // Skipped layer must not become a residual.
  residual.apply(3, 11, current, 1, corrected);
  assert(std::equal(corrected, corrected + 3, current));
  residual.observe(0, 11, first);
  residual.observe(1, 12, second); // Different tokens must not be joined.
  residual.apply(1, 13, current, 1, corrected);
  assert(std::equal(corrected, corrected + 3, current));
  nntrainer::HtpPrefetchConfidence confidence(2, 0.6f);
  const uint32_t actual[] = {1, 3};
  assert(confidence.allow(1, 0) && confidence.allow(1, 1));
  confidence.record(1, 10, {1, 2}, {true, true});
  confidence.observe(1, 9, actual, 2); // No future/stale feedback.
  assert(confidence.allow(1, 1));
  confidence.observe(1, 10, actual, 2);
  assert(confidence.allow(1, 0) && !confidence.allow(1, 1));
  // Observe a filtered candidate too, allowing it to recover.
  confidence.record(1, 11, {1, 3}, {true, true});
  confidence.observe(1, 11, actual, 2);
  assert(confidence.allow(1, 1));
  const float input[] = {2, -1, 3};
  // Matrix already includes the target router gamma, [hidden, expert].
  const float gate[] = {1, 0, 2, -1, 0, 4, 0, 3, 0, 0, 0, 1};
  assert((nntrainer::htpPredictExperts(input, gate, 3, 4, 2) ==
          std::vector<unsigned>{2, 0}));
  const float zero[] = {0, 0, 0};
  assert((nntrainer::htpPredictExperts(zero, gate, 3, 4, 2) ==
          std::vector<unsigned>{0, 1}));
  const float nan[] = {std::numeric_limits<float>::quiet_NaN(), 0, 0};
  assert(nntrainer::htpPredictExperts(nan, gate, 3, 4, 2).empty());
  assert(nntrainer::htpPredictExperts(input, gate, 3, 4, 0).empty());
  // LFM2 ranks sigmoid(logit) plus bias, not the logit plus bias. Here
  // a smaller logit wins after the sigmoid saturates the two positive ones.
  const float sigmoid_bias[] = {0.25f, 0.0f, 0.0f, 0.5f};
  assert((nntrainer::htpPredictExperts(input, gate, 3, 4, 2, sigmoid_bias) ==
          std::vector<unsigned>{0, 2}));
  assert((nntrainer::htpPredictExperts(zero, gate, 3, 4, 4, sigmoid_bias) ==
          std::vector<unsigned>{3, 0, 1, 2}));
  const float extreme_gate[] = {1000, -1000, 0, 0};
  const float unit[] = {1};
  const float tie_bias[] = {0, 0, 0.5f, -0.5f};
  assert((nntrainer::htpPredictExperts(unit, extreme_gate, 1, 4, 4, tie_bias) ==
          std::vector<unsigned>{0, 2, 1, 3}));
  const float bad_bias[] = {0, std::numeric_limits<float>::infinity(), 0, 0};
  assert(nntrainer::htpPredictExperts(input, gate, 3, 4, 2, bad_bias).empty());
  std::puts("EXPERT PREFETCH OWNERSHIP / CANCELLATION / REBIND / PREDICTOR OK");
}
