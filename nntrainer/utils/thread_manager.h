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
#include <climits>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <linux/futex.h>
#include <mutex>
#include <queue>
#include <string>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include <iostream>

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

#if defined(__GNUC__)
#define CACHELINE_ALIGNED __attribute__((__aligned__(64)))
#elif defined(_MSC_VER)
#define CACHELINE_ALIGNED __declspec(align(64))
#else
#error "Platform-specific implementation of CACHELINE_ALIGNED required"
#endif

struct CACHELINE_ALIGNED thread_info {
  CACHELINE_ALIGNED std::atomic<size_t> range_start;
  CACHELINE_ALIGNED std::atomic<size_t> range_end;
  CACHELINE_ALIGNED std::atomic<size_t> range_length;
};

static_assert(sizeof(thread_info) % 64 == 0);

enum threadpool_command {
  INIT,
  RUN,
  SHUTDOWN,
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

  static inline int futex_wait(std::atomic<uint32_t> *addr, uint32_t val) {
    return syscall(SYS_futex, addr, FUTEX_WAIT | FUTEX_PRIVATE_FLAG, val, NULL);
  }

  static inline int futex_wake(std::atomic<uint32_t> *addr) {
    return syscall(SYS_futex, addr, FUTEX_WAKE | FUTEX_PRIVATE_FLAG, INT_MAX);
  }

  static inline void yield() {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_pause();
#elif defined(__aarch64__) || defined(__arm__)
    __asm__ volatile("yield");
#endif
  }

  static inline size_t modulo_decrement(size_t i, size_t n) {
    if (i == 0)
      i = n;
    return i - 1;
  }

  void checkin() {
    size_t t = active_threads_.fetch_sub(1, std::memory_order_acq_rel) - 1;
    if (t == 0) {
      has_active_threads_.store(false, std::memory_order_release);
      futex_wake(&has_active_threads_);
    }
  }

  uint32_t wait_for_new_command(uint32_t last_command) {
    uint32_t command = command_.load(std::memory_order_acquire);
    if (command != last_command)
      return command;

    for (uint32_t i = 1000000; i != 0; i--) {
      yield();

      command = command_.load(std::memory_order_acquire);
      if (command != last_command)
        return command;
    }

    do {
      futex_wait(&command_, last_command);
      command = command_.load(std::memory_order_acquire);
    } while (command == last_command);

    return command;
  }

  void wait_worker_threads() {
    uint32_t has_active_threads =
      has_active_threads_.load(std::memory_order_acquire);

    if (has_active_threads == 0)
      return;

    for (uint32_t i = 1000000; i != 0; i--) {
      yield();

      has_active_threads = has_active_threads_.load(std::memory_order_acquire);
      if (has_active_threads == 0)
        return;
    }

    while ((has_active_threads =
              has_active_threads_.load(std::memory_order_acquire)) != 0) {
      futex_wait(&has_active_threads_, 1);
    }
  }

  void print(const std::string &str) {
    std::lock_guard<std::mutex> lock(barrier_mutex_);
    int c = counter.fetch_add(1);
    std::cout << "[" << c << "]" << str << std::endl;
  }

  void thread_main(size_t tid) {
    uint32_t last_command = threadpool_command::INIT;
    // print(std::string{"entering checkin "} + std::to_string(tid));
    checkin();
    // print(std::string{"finish checkin "} + std::to_string(tid));
    while (true) {
      // print(std::string{"entering command "} + std::to_string(tid));
      uint32_t command = wait_for_new_command(last_command);
      // print(std::string{"finish command "} + std::to_string(tid) + std::string{" "} +
      //       std::to_string(command));
      std::atomic_thread_fence(std::memory_order_acquire);

      switch (command & 0x7FFFFFFF) {
      case RUN: {
        thread_function_(tid);
        break;
      }
      case SHUTDOWN:
        return;
      case INIT:
        break;
      }
      // print(std::string{"entering checkin "} + std::to_string(tid));

      checkin();
      // print(std::string{"finish checkin "} + std::to_string(tid));

      last_command = command;
    }
  }

  static inline bool try_decrement(std::atomic<size_t> &value) {
    size_t actual_value = value.load(std::memory_order_relaxed);
    while (actual_value != 0) {
      if (value.compare_exchange_weak(actual_value, actual_value - 1,
                                      std::memory_order_relaxed))
        return true;
    }
    return false;
  }

  void thread_parallelize_1d(size_t my_tid) {
    // print(std::string{"start "} + std::to_string(my_tid));

    /* Process thread's own range of items */

    size_t range_start =
      thread_infos_[my_tid].range_start.load(std::memory_order_relaxed);

    while (try_decrement(thread_infos_[my_tid].range_length)) {
      // print(std::string{"s "} + std::to_string(range_start));
      task_(range_start++);
    }

    // print(std::string{"mid "} + std::to_string(my_tid));

    /* There still may be other threads with work */
    size_t threads_count = compute_workers_.size() + 1;
    for (size_t tid = modulo_decrement(my_tid, threads_count); tid != my_tid;
         tid = modulo_decrement(tid, threads_count)) {
      while (try_decrement(thread_infos_[tid].range_length)) {
        size_t index =
          thread_infos_[tid].range_end.fetch_sub(1, std::memory_order_relaxed) -
          1;
        // print(std::string{"e "} + std::to_string(index));
        task_(index);
      }
      // print(std::string{"mytid "} + std::to_string(my_tid) +
      //       std::string{" tid "} + std::to_string(tid));
    }

    // print(std::string{"finish "} + std::to_string(my_tid));

    /* Make changes by this thread visible to other threads */
    std::atomic_thread_fence(std::memory_order_release);
  }

  template <typename F>
  void parallelize_1d(size_t begin, size_t end, F &&task) {
    if (end - begin == 1 || compute_workers_.empty()) {
      for (size_t i = begin; i < end; i++)
        task(i);
      return;
    }

    std::function<void(size_t)> thread_function = [this](size_t i) {
      this->thread_parallelize_1d(i);
    };

    parallelize(thread_function, begin, end, std::forward<F>(task));
  }

  template <typename F>
  void parallelize(std::function<void(size_t)> thread_function, size_t begin,
                   size_t end, F &&task) {
    std::lock_guard<std::mutex> lock(execution_mutex_);

    task_ = std::move(task);
    thread_function_ = thread_function;
    // todo context

    size_t threads_count = compute_workers_.size() + 1;
    active_threads_.store(threads_count - 1, std::memory_order_relaxed);
    has_active_threads_.store(1, std::memory_order_relaxed);

    size_t range_quotient = (end - begin) / threads_count;
    size_t range_remainder = (end - begin) % threads_count;

    size_t range_start = begin;
    for (size_t tid = 0; tid < threads_count; tid++) {
      size_t range_length = range_quotient + (size_t)(tid < range_remainder);
      size_t range_end = range_start + range_length;
      thread_infos_[tid].range_start.store(range_start,
                                           std::memory_order_relaxed);
      thread_infos_[tid].range_end.store(range_end, std::memory_order_relaxed);
      thread_infos_[tid].range_length.store(range_length,
                                            std::memory_order_relaxed);

      // print(std::to_string(range_start) + std::string{" "} +
      //       std::to_string(range_end));
      // std::cout << range_start << " " << range_end << "\n";
      range_start = range_end;
    }

    uint32_t old_command = command_.load(std::memory_order_relaxed);
    uint32_t new_command =
      (~(old_command | 0x7FFFFFFF)) | threadpool_command::RUN;

    // print("START");

    command_.store(new_command, std::memory_order_release);
    futex_wake(&command_);

    thread_function_(0);

    // print("main mid");

    wait_worker_threads();
    // print("main finish");

    std::atomic_thread_fence(std::memory_order_acquire);
  }

  template <typename F> void parallel_for(size_t begin, size_t end, F &&fn) {
    if (begin >= end) {
      return;
    }
    parallelize_1d(begin, end, std::forward<F>(fn));
    // if (begin >= end)
    //   return;
    // if (end - begin == 1 || compute_workers_.empty()) {
    //   for (size_t i = begin; i < end; ++i)
    //     fn(i);
    //   return;
    // }
    // dispatchAndJoin(begin, end, std::forward<F>(fn));
  }

  template <typename F>
  void parallel_for(size_t begin, size_t end, unsigned int n_workers, F &&fn) {
    if (begin >= end) {
      return;
    }
    parallelize_1d(begin, end, std::forward<F>(fn));
    // if (begin >= end)
    //   return;
    // unsigned int total = static_cast<unsigned int>(compute_workers_.size());
    // if (n_workers > total)
    //   n_workers = total;
    // if (end - begin == 1 || n_workers == 0 || compute_workers_.empty()) {
    //   for (size_t i = begin; i < end; ++i)
    //     fn(i);
    //   return;
    // }
    // dispatchAndJoin(begin, end, std::forward<F>(fn), n_workers);
  }

  template <typename F> void parallel_for_chunked(size_t n_threads, F &&fn) {
    parallelize_1d(0, n_threads, std::forward<F>(fn));
    // if (n_threads <= 1) {
    //   fn(0);
    //   return;
    // }
    // parallel_for(0, n_threads, std::forward<F>(fn));
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

  static void setConfig(const ThreadManagerConfig &config) { config_ = config; }

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
  CACHELINE_ALIGNED std::atomic<size_t> current_chunk_{0};
  CACHELINE_ALIGNED std::atomic<bool> stop_{false};

  // ─── Spin-wait mode state ───────────────────────────
  CACHELINE_ALIGNED std::atomic<unsigned int> spin_generation_{0};
  CACHELINE_ALIGNED std::atomic<int> spin_n_barrier_{0};
  CACHELINE_ALIGNED std::atomic<bool> spin_barrier_sense_{false};
  CACHELINE_ALIGNED std::atomic<bool> spin_current_sense_{false};
  CACHELINE_ALIGNED std::atomic<unsigned int> spin_active_workers_{0};
  CACHELINE_ALIGNED std::atomic<int> spin_active_threads_{1};
  CACHELINE_ALIGNED std::atomic<unsigned int> spin_workers_ready_{0};

  std::unique_ptr<thread_info[]> thread_infos_;
  std::mutex execution_mutex_;
  std::function<void(size_t)> thread_function_;
  std::function<void(size_t)> task_;
  CACHELINE_ALIGNED std::atomic<uint32_t> command_;
  CACHELINE_ALIGNED std::atomic<uint32_t> flags_;
  CACHELINE_ALIGNED std::atomic<size_t> active_threads_;
  CACHELINE_ALIGNED std::atomic<uint32_t> has_active_threads_;

  std::atomic<uint32_t> counter{0};

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
