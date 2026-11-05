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
#include <htp_runtime_config.h>
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

struct WorkerPolicy {
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
};
#include <htp_prefill_readers.h>
struct Fixture : nntrainer::HtpPrefillReaders<int, WorkerPolicy> {
  std::condition_variable entered_cv;
  unsigned entered = 0;
  void startPrefetchReaders() {
    HtpPrefillReaders::startPrefetchReaders([this] { prefetchReaderLoop(); });
  }
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
  nntrainer::HtpRuntimeConfig config;
  config.htp.release_prefill_readers = enabled;
  nntrainer::configureHtpRuntime(config);
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
    f.prefetch_batches_.push_back(std::make_unique<Fixture::PrefetchBatch>());
    f.releaseIdlePrefillReaders(); // A running batch cannot be cancelled.
    assert(f.prefetch_readers_.size() == 2 && !f.prefetch_stop_);
    f.prefetch_batches_.clear();
    f.prefetch_jobs_.emplace_back(nullptr, 0);
    f.releaseIdlePrefillReaders(); // Queued payloads cannot be abandoned.
    assert(f.prefetch_readers_.size() == 2 && !f.prefetch_stop_);
    f.prefetch_jobs_.clear();
    f.releaseIdlePrefillReaders();
    assert(f.prefetch_readers_.empty() && !f.prefetch_stop_);
    assert(destroyed == 2 * (prompt + 1)); // Join includes TLS destruction.
    f.releaseIdlePrefillReaders();         // Safe before destructor and reuse.
  }
  nntrainer::resetHtpRuntimeConfig();
  std::puts("PREFILL READER IDLE RELEASE / TLS DESTRUCTION / RESTART OK");
}
