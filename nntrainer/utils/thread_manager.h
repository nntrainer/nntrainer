// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   thread_manager.h
 * @date   20 March 2026
 * @brief  Unified thread manager for compute and I/O operations
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __NNTRAINER_THREAD_MANAGER_H__
#define __NNTRAINER_THREAD_MANAGER_H__

#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <completion_token.h>
#include <singleton.h>

namespace nntrainer {

struct ThreadManagerConfig {
  /**
   * @brief Number of compute worker threads.
   * Default uses NNTR_NUM_THREADS if set > 0, otherwise
   * std::thread::hardware_concurrency() / 2
   */
  unsigned int compute_threads = defaultComputeThreads();
  unsigned int io_threads = 0;

  /**
   * @brief Enable CPU affinity pinning.
   * When true, workers are pinned to cores and use GGML-style spin-wait
   * barrier for minimal latency. When false (default), uses condvar-based
   * barrier which is safe without dedicated cores.
   */
  bool enable_affinity = true;

private:
  static unsigned int defaultComputeThreads() {
    // priority
    // 1. environment variable
    // 2. compile flag
    // 3. std::thread::hardware_concurrency()

    auto nntr_num_threads = std::getenv("NNTR_NUM_THREADS");
    if (nntr_num_threads) {
      return static_cast<unsigned int>(std::stoul(nntr_num_threads));
    }

#if defined(NNTR_NUM_THREADS) && NNTR_NUM_THREADS > 0
    return NNTR_NUM_THREADS;
#else
    /// @todo use performance core only for x86
    unsigned int hw = std::thread::hardware_concurrency();
    return hw > 0 ? hw / 2 : 1;
#endif
  }
};

/**
 * @class ThreadManager
 * @brief Hybrid thread pool: spin-wait (with affinity) or condvar (without).
 *
 * With enable_affinity=true:
 *   Workers are pinned to cores and use GGML-style spin-wait + atomic barrier.
 *   Minimal dispatch latency (~0.1us), but requires dedicated cores.
 *
 * With enable_affinity=false (default):
 *   Workers use condvar for dispatch and barrier.
 *   Safe without dedicated cores, slightly higher latency (~1-2us).
 */
class ThreadManager : public Singleton<ThreadManager> {
  friend class Singleton<ThreadManager>;

public:
  ~ThreadManager();

  template <typename F> void parallel_for(size_t begin, size_t end, F &&fn) {
    if (begin >= end)
      return;
    if (end - begin == 1 || compute_workers_.empty()) {
      for (size_t i = begin; i < end; ++i)
        fn(i);
      return;
    }
    dispatchAndJoin(begin, end, std::forward<F>(fn));
  }

  template <typename F>
  void parallel_for(size_t begin, size_t end, unsigned int n_workers, F &&fn) {
    if (begin >= end)
      return;
    unsigned int total = static_cast<unsigned int>(compute_workers_.size());
    if (n_workers > total)
      n_workers = total;
    if (end - begin == 1 || n_workers == 0 || compute_workers_.empty()) {
      for (size_t i = begin; i < end; ++i)
        fn(i);
      return;
    }
    dispatchAndJoin(begin, end, std::forward<F>(fn), n_workers);
  }

  template <typename F> void parallel_for_chunked(size_t n_threads, F &&fn) {
    if (n_threads <= 1) {
      fn(0);
      return;
    }
    parallel_for(0, n_threads, std::forward<F>(fn));
  }

  CompletionToken submit(std::function<void()> task);

  unsigned int getComputeThreadCount() const {
    // main thread is not in compute_workers
    return static_cast<unsigned int>(compute_workers_.size()) + 1;
  }

  unsigned int getIOThreadCount() const {
    return static_cast<unsigned int>(io_workers_.size());
  }

  bool isSpinMode() const { return spin_mode_; }

  static void setConfig(const ThreadManagerConfig &config) {
    config_ = config;
  }

protected:
  ThreadManager();
  void initialize() noexcept override;

private:
  // ─── Spin-wait helpers (GGML-style, used when affinity=true) ────
  static inline void cpuRelax() {
#if defined(__x86_64__) || defined(_M_X64)
    __builtin_ia32_pause();
#elif defined(__aarch64__) || defined(__arm__)
    asm volatile("yield" ::: "memory");
#endif
  }

  void spinBarrier(bool sense) {
    int n_threads = spin_active_threads_.load(std::memory_order_acquire);
    int n = spin_n_barrier_.fetch_add(1, std::memory_order_acq_rel);
    if (n == n_threads - 1) {
      spin_n_barrier_.store(0, std::memory_order_release);
      spin_barrier_sense_.store(sense, std::memory_order_release);
      return;
    }
    while (spin_barrier_sense_.load(std::memory_order_acquire) != sense) {
      cpuRelax();
    }
  }

  // ─── Dispatch (branching on spin_mode_) ─────────────────────────
  template <typename F>
  void dispatchAndJoin(size_t begin, size_t end, F &&fn,
                       unsigned int n_workers = 0) {
    unsigned int total = static_cast<unsigned int>(compute_workers_.size());
    if (n_workers == 0 || n_workers > total)
      n_workers = total;

    if (spin_mode_) {
      // ── SPIN-WAIT PATH (affinity=true) ──
      spin_active_threads_.store(static_cast<int>(n_workers + 1),
                                 std::memory_order_release);
      bool sense = !spin_barrier_sense_.load(std::memory_order_acquire);
      spin_current_sense_.store(sense, std::memory_order_release);

      current_task_ = [&fn](size_t i) { fn(i); };
      task_end_ = end;
      current_chunk_.store(begin, std::memory_order_relaxed);
      spin_active_workers_.store(n_workers, std::memory_order_release);

      // wake workers via generation bump
      spin_generation_.fetch_add(1, std::memory_order_seq_cst);

      // caller does work
      while (true) {
        size_t idx = current_chunk_.fetch_add(1, std::memory_order_relaxed);
        if (idx >= end)
          break;
        fn(idx);
      }

      spinBarrier(sense);
      current_task_ = nullptr;

    } else {
      // ── CONDVAR PATH (affinity=false, default) ──
      unsigned int my_barrier_gen;
      {
        std::lock_guard<std::mutex> lock(dispatch_mutex_);
        current_task_ = [&fn](size_t i) { fn(i); };
        task_end_ = end;
        current_chunk_.store(begin, std::memory_order_relaxed);
        cv_active_workers_ = n_workers;
        barrier_target_ = static_cast<int>(n_workers + 1);
        ++dispatch_gen_;
        ++barrier_gen_;
        my_barrier_gen = barrier_gen_;
        barrier_arrived_ = 0;
      }
      dispatch_cv_.notify_all();

      // caller does work
      while (true) {
        size_t idx = current_chunk_.fetch_add(1, std::memory_order_relaxed);
        if (idx >= end)
          break;
        fn(idx);
      }

      // caller arrives at barrier
      {
        std::unique_lock<std::mutex> lock(barrier_mutex_);
        ++barrier_arrived_;
        if (barrier_arrived_ >= barrier_target_) {
          barrier_done_gen_ = my_barrier_gen;
          barrier_cv_.notify_all();
        } else {
          barrier_cv_.wait(lock, [this, my_barrier_gen] {
            return barrier_done_gen_ >= my_barrier_gen;
          });
        }
      }
      current_task_ = nullptr;
    }
  }

  void computeWorkerLoopSpin(unsigned int worker_id);
  void computeWorkerLoopCondvar(unsigned int worker_id);
  void ioWorkerLoop();

  // ─── Mode ───────────────────────────────────────────
  bool spin_mode_{false};

  // ─── Shared ─────────────────────────────────────────
  std::vector<std::thread> compute_workers_;
  std::function<void(size_t)> current_task_;
  size_t task_end_{0};
  alignas(64) std::atomic<size_t> current_chunk_{0};
  alignas(64) std::atomic<bool> stop_{false};

  // ─── Spin-wait mode state ───────────────────────────
  alignas(64) std::atomic<unsigned int> spin_generation_{0};
  alignas(64) std::atomic<int> spin_n_barrier_{0};
  alignas(64) std::atomic<bool> spin_barrier_sense_{false};
  alignas(64) std::atomic<bool> spin_current_sense_{false};
  alignas(64) std::atomic<unsigned int> spin_active_workers_{0};
  alignas(64) std::atomic<int> spin_active_threads_{1};
  alignas(64) std::atomic<unsigned int> spin_workers_ready_{0};

  // ─── Condvar mode state ─────────────────────────────
  std::mutex dispatch_mutex_;
  std::condition_variable dispatch_cv_;
  unsigned int dispatch_gen_{0};
  unsigned int cv_active_workers_{0};

  std::mutex barrier_mutex_;
  std::condition_variable barrier_cv_;
  int barrier_arrived_{0};
  int barrier_target_{0};
  unsigned int barrier_gen_{0};
  unsigned int barrier_done_gen_{0};

  // ─── I/O ────────────────────────────────────────────
  std::vector<std::thread> io_workers_;
  std::queue<std::pair<std::function<void()>,
                       std::shared_ptr<CompletionToken::SharedState>>>
    io_queue_;
  std::mutex io_mutex_;
  std::condition_variable io_cv_;

  // ─── Config ─────────────────────────────────────────
  static ThreadManagerConfig config_;
};

} // namespace nntrainer

#endif // __NNTRAINER_THREAD_MANAGER_H__
