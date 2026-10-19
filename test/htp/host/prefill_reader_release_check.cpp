// SPDX-License-Identifier: Apache-2.0
/**
 * @file prefill_reader_release_check.cpp
 * @brief Actual prefill reader start/retire methods with deterministic TLS.
 * @author NNTrainer contributors
 * @bug No known bugs except for NYI items
 */
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>
#if defined(_WIN32)
static constexpr int _SC_NPROCESSORS_CONF = 0;
static long sysconf(int) { return 2; }
#else
#include <unistd.h>
#endif

static std::atomic<unsigned> destroyed{0};

struct Fixture {
  struct PrefetchKnobs {
    size_t readers = 2;
    std::vector<int> cpus{0};
  };
  static const PrefetchKnobs &prefetchKnobs() {
    static const PrefetchKnobs knobs;
    return knobs;
  }
  static uint32_t cpuBit() { return 1; }
  static void pinToCpus(const std::vector<int> &) {}

  std::mutex prefetch_mutex_;
  std::condition_variable prefetch_cv_, entered_cv;
  std::deque<int> prefetch_batches_, prefetch_jobs_;
  bool prefetch_stop_ = false;
  std::vector<std::thread> prefetch_readers_;
  uint32_t prefetch_caller_cpus_ = 0;
  unsigned entered = 0;

#include "releaseIdlePrefillReaders.inc"
#include "startPrefetchReaders.inc"

  void prefetchReaderLoop() {
    struct Scratch {
      ~Scratch() { ++destroyed; }
    };
    thread_local Scratch scratch;
    (void)scratch;
    std::unique_lock<std::mutex> lock(prefetch_mutex_);
    ++entered;
    entered_cv.notify_all();
    // Payload ownership is represented by the real method's batch/job guards.
    // Deterministic workers let the test observe TLS destruction on join.
    prefetch_cv_.wait(lock, [this] { return prefetch_stop_; });
  }

  void waitEntered(unsigned count) {
    std::unique_lock<std::mutex> lock(prefetch_mutex_);
    assert(entered_cv.wait_for(lock, std::chrono::seconds(5),
                               [this, count] { return entered == count; }));
  }
};

static void enable(bool enabled) {
#if defined(_WIN32)
  _putenv_s("NNTR_HTP_RELEASE_PREFILL_READERS", enabled ? "1" : "0");
#else
  assert(setenv("NNTR_HTP_RELEASE_PREFILL_READERS", enabled ? "1" : "0", 1) ==
         0);
#endif
}

int main() {
  Fixture f;
  for (unsigned prompt = 0; prompt < 3; ++prompt) {
    f.startPrefetchReaders();
    f.waitEntered(2 * (prompt + 1));
    f.startPrefetchReaders(); // Idempotent start must not spawn more readers.
    assert(f.prefetch_readers_.size() == 2 && !f.prefetch_stop_);
    enable(false);
    f.releaseIdlePrefillReaders();
    assert(f.prefetch_readers_.size() == 2 && destroyed == 2 * prompt);
    enable(true);
    f.prefetch_batches_.push_back(1);
    f.releaseIdlePrefillReaders(); // A running batch cannot be cancelled.
    assert(f.prefetch_readers_.size() == 2 && !f.prefetch_stop_);
    f.prefetch_batches_.clear();
    f.prefetch_jobs_.push_back(1);
    f.releaseIdlePrefillReaders(); // Queued payloads cannot be abandoned.
    assert(f.prefetch_readers_.size() == 2 && !f.prefetch_stop_);
    f.prefetch_jobs_.clear();
    f.releaseIdlePrefillReaders();
    assert(f.prefetch_readers_.empty() && !f.prefetch_stop_);
    assert(destroyed == 2 * (prompt + 1)); // Join includes TLS destruction.
    f.releaseIdlePrefillReaders();         // Safe before destructor and reuse.
  }
  std::puts("PREFILL READER IDLE RELEASE / TLS DESTRUCTION / RESTART OK");
}
