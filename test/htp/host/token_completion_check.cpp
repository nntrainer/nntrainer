// SPDX-License-Identifier: Apache-2.0
/**
 * @file token_completion_check.cpp
 * @brief Mailbox completion validation and server lifecycle checks.
 * @author NNTrainer contributors
 * @bug No known bugs
 */
#include "htp_dspq_wire.h"
#include "htp_token_completion.h"

#include <array>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <thread>

using nntrainer::HtpTokenPollState;

static void records() {
  std::array<uint32_t, 2> words{};
  auto load = [&](unsigned w) { return words[w]; };
  for (uint32_t tok : {0u, 1u, 17u, UINT32_MAX}) {
    words = {0u, 0u};
    assert(!HtpTokenPollState::isDone(tok, load));
    words = {tok - 1u, ~(tok - 1u)};
    assert(!HtpTokenPollState::isDone(tok, load));
    words = {tok, ~(tok - 1u)};
    assert(!HtpTokenPollState::isDone(tok, load));
    words = {tok, ~tok};
    assert(HtpTokenPollState::isDone(tok, load));
    unsigned reads = 0;
    assert(!HtpTokenPollState::isDone(tok, [&](unsigned w) {
      const uint32_t value = words[w];
      if (++reads == 2u)
        words[0] = tok + 1u;
      return value;
    }));
  }
  HtpTokenPollState state;
  assert(!state.canStart(false, 0u));
  assert(state.canStart(true, 0u));
  state.begin(0u);
  assert(!state.canStart(true, 0u));
  assert(state.canStart(true, UINT32_MAX));
  state.begin(UINT32_MAX);
  assert(state.canStart(true, 0u));
  static_assert(HTP_MBOX_TOKEN_DONE + HTP_MBOX_TOKEN_DONE_BYTES <=
                HTP_MBOX_MISS_REQ);
}

/** Mirrors the pool server's CV/active ownership, using atomic words in place
 * of uncached ARM/DSP shared memory. No volatile C++ data races in the test. */
class Server {
public:
  Server() : worker_([this] { serve(); }) {
    std::unique_lock<std::mutex> lock(mu_);
    wait(lock, [&] { return idle_; });
  }
  ~Server() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      active_ = false;
      stop_ = true;
      cv_.notify_all();
    }
    worker_.join();
  }
  void arm(uint32_t tok) {
    std::unique_lock<std::mutex> lock(mu_);
    assert(idle_ && !active_);
    seq_ = 0u;
    inverse_ = 0u;
    tok_ = tok;
    const unsigned before = starts_;
    active_ = true;
    cv_.notify_all();
    wait(lock, [&] { return starts_ == before + 1u; });
  }
  void publish(uint32_t tok) {
    inverse_.store(~tok, std::memory_order_relaxed);
    seq_.store(tok, std::memory_order_release);
  }
  void assertPolling() {
    std::lock_guard<std::mutex> lock(mu_);
    assert(!idle_ && active_);
  }
  void parkWhileActive() {
    std::unique_lock<std::mutex> lock(mu_);
    wait(lock, [&] { return idle_; });
    assert(active_);
    // A spurious wake and the still-true active flag cannot serve this token
    // twice. The same predicate is used by the production server.
    assert(!state_.canStart(active_, tok_));
    cv_.notify_all();
  }
  void disarm() {
    std::unique_lock<std::mutex> lock(mu_);
    active_ = false;
    wait(lock, [&] { return idle_; });
  }
  unsigned starts() {
    std::lock_guard<std::mutex> lock(mu_);
    return starts_;
  }

private:
  template <typename Predicate>
  void wait(std::unique_lock<std::mutex> &lock, Predicate predicate) {
    assert(cv_.wait_for(lock, std::chrono::seconds(5), predicate));
  }
  void serve() {
    for (;;) {
      uint32_t tok;
      {
        std::unique_lock<std::mutex> lock(mu_);
        idle_ = true;
        cv_.notify_all();
        cv_.wait(lock, [&] { return stop_ || state_.canStart(active_, tok_); });
        if (stop_)
          return;
        idle_ = false;
        tok = tok_;
        state_.begin(tok);
        ++starts_;
        cv_.notify_all();
      }
      while (active_) {
        if (HtpTokenPollState::isDone(tok, [&](unsigned w) {
              return w == 0u ? seq_.load(std::memory_order_acquire)
                             : inverse_.load(std::memory_order_relaxed);
            }))
          break;
        std::this_thread::yield();
      }
    }
  }

  std::mutex mu_;
  std::condition_variable cv_;
  std::atomic<bool> active_{false};
  std::atomic<uint32_t> seq_{0}, inverse_{0};
  bool idle_ = false, stop_ = false;
  uint32_t tok_ = 0;
  unsigned starts_ = 0;
  HtpTokenPollState state_;
  std::thread worker_;
};

static void lifecycle() {
  Server server;
  unsigned submitted = 0;
  // Success and failure publish the same hint. The server does not infer a
  // return code; the caller must still receive the independent DSPQ response.
  for (uint32_t tok : {0u, 1u, UINT32_MAX, 0u}) {
    server.arm(tok);
    ++submitted;
    server.publish(tok - 1u);
    server.assertPolling();
    server.publish(tok);
    server.parkWhileActive();
    server.disarm();
  }
  // Older DSP skeletons and failures before driver start publish no hint.
  server.arm(1u);
  ++submitted;
  server.disarm();
  // Completion and DSPQ-driven disarm can race in either order. Reuse after
  // every join verifies neither stale records nor a missed wake stop progress.
  for (uint32_t tok = 2u; tok < 502u; ++tok) {
    server.arm(tok);
    ++submitted;
    std::thread dsp([&server, tok] { server.publish(tok); });
    server.disarm();
    dsp.join();
  }
  server.arm(502u);
  ++submitted;
  server.publish(502u);
  server.parkWhileActive();
  server.disarm();
  assert(server.starts() == submitted);
}

int main() {
  records();
  lifecycle();
  std::puts("token_completion_check: PASS (stale/torn/wrap, failure fallback, "
            "next token, 500 disarm races)");
}
