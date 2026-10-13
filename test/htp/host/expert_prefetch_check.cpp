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

int main() {
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
  std::puts("EXPERT PREFETCH OWNERSHIP / CANCELLATION / REBIND / PREDICTOR OK");
}
