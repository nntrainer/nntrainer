// SPDX-License-Identifier: Apache-2.0
/** @file htp_prefill_readers.h
 * @brief Owns prefill read batches, worker synchronization and retirement.
 * @author NNTrainer contributors
 * @bug No known bugs except for NYI items
 */
#ifndef NNTR_HTP_PREFILL_READERS_H
#define NNTR_HTP_PREFILL_READERS_H
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <functional>
#include <htp_worker_policy.h>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <weight_file_reader.h>
namespace nntrainer {
template <typename StagedExpert, typename Policy = HtpWorkerPolicy>
class HtpPrefillReaders {
public:
  ~HtpPrefillReaders() { stop(); }
  std::vector<StagedExpert> prefetch_;
  struct PrefetchReadTrace {
    uint64_t start_us = 0, end_us = 0;
    WeightReadTiming timing;
    uint32_t cpu = 0;
  };
  /** One _begin's experts; done counts the ones read. */
  struct PrefetchBatch {
    std::vector<StagedExpert> experts;
    std::vector<bool> cancelled;
    size_t done = 0;
    uint64_t trace_id = 0, queued_us = 0;
    size_t jobs_ahead = 0;
    std::vector<PrefetchReadTrace> read_trace;
  };
  uint64_t prefetch_trace_next_ = 0;
  /** Queued batches, oldest first, and their unread experts, all under
   *  prefetch_mutex_. A batch's address is stable until _end takes it. */
  std::mutex prefetch_mutex_;
  std::condition_variable prefetch_cv_, prefetch_done_cv_;
  std::deque<std::unique_ptr<PrefetchBatch>> prefetch_batches_;
  std::deque<std::pair<PrefetchBatch *, size_t>> prefetch_jobs_;
  bool prefetch_stop_ = false;
  std::vector<std::thread> prefetch_readers_;
  std::atomic<uint32_t> prefetch_reader_cpus_{0};
  uint32_t prefetch_caller_cpus_ = 0;
  void releaseIdlePrefillReaders() {
    const char *v = std::getenv("NNTR_HTP_RELEASE_PREFILL_READERS");
    if (!v || std::strcmp(v, "1") != 0)
      return;
    size_t count;
    {
      std::lock_guard<std::mutex> lock(prefetch_mutex_);
      if (prefetch_readers_.empty())
        return;
      if (!prefetch_batches_.empty() || !prefetch_jobs_.empty()) {
        std::fprintf(stderr,
                     "[HTP-MEM] prefill_readers_release skipped=busy\n");
        return;
      }
      count = prefetch_readers_.size();
      prefetch_stop_ = true;
    }
    prefetch_cv_.notify_all();
    for (auto &t : prefetch_readers_)
      if (t.joinable())
        t.join();
    prefetch_readers_.clear();
    {
      std::lock_guard<std::mutex> lock(prefetch_mutex_);
      prefetch_stop_ = false;
    }
    std::fprintf(stderr, "[HTP-MEM] prefill_readers_released=%zu\n", count);
  }
  void startPrefetchReaders(const std::function<void()> &reader) {
    if (!prefetch_readers_.empty())
      return;
    const auto &knobs = Policy::prefetchKnobs();
    prefetch_caller_cpus_ = Policy::cpuBit();
    std::vector<int> cpus = knobs.cpus;
#if defined(__linux__)
    if (cpus.empty() && prefetch_caller_cpus_ != 0) {
      const long n = sysconf(_SC_NPROCESSORS_CONF);
      for (int c = 0; c < n && c < 32; ++c)
        if ((prefetch_caller_cpus_ & (1u << c)) == 0)
          cpus.push_back(c);
    }
#endif
    for (size_t t = 0; t < knobs.readers; ++t)
      prefetch_readers_.emplace_back([this, cpus, reader] {
        Policy::pinToCpus(cpus);
        reader();
      });
  }
  void stop() {
    {
      std::lock_guard<std::mutex> lock(prefetch_mutex_);
      prefetch_stop_ = true;
    }
    prefetch_cv_.notify_all();
    for (std::thread &t : prefetch_readers_)
      if (t.joinable())
        t.join();
  }
};
} // namespace nntrainer
#endif
