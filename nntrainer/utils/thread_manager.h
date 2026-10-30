// SPDX-License-Identifier: Apache-2.0, BSD 2-Clause "Simplified" License
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 * Copyright (C) 2025 Jaemin Shin <jaemin2.shin@samsung.com>
 *
 * @file   thread_manager.h
 * @date   23 April 2026
 * @brief  Unified thread manager for compute, inspired by pthreadpool
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Jaemin Shin <jaemin2.shin@samsung.com>
 * @bug    No known bugs except for NYI items
 */

/**
Copyright 2019 Google LLC
Copyright (c) 2017 Facebook Inc.
Copyright (c) 2015-2017 Georgia Institute of Technology
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
 list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef __NNTRAINER_THREAD_MANAGER_H__
#define __NNTRAINER_THREAD_MANAGER_H__

#include <atomic>
#include <climits>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__) || defined(__ANDROID__)
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) ||             \
  defined(_M_IX86)
#include <xmmintrin.h>
#endif

#include <singleton.h>

#include "thread_manager_util.h"

namespace nntrainer {

/** @brief Forward declaration of ThreadManager. */
class ThreadManager;
/** @brief Return the ThreadManager singleton. */
template <>
NNTRAINER_SINGLETON_API ThreadManager &Singleton<ThreadManager>::Global();

#if defined(__GNUC__)
#define CACHELINE_ALIGNED __attribute__((__aligned__(64)))
#elif defined(_MSC_VER)
#define CACHELINE_ALIGNED __declspec(align(64))
#else
#error "Platform-specific implementation of CACHELINE_ALIGNED required"
#endif

/**
 * @brief Configuration for ThreadManager: compute threads and affinity settings
 */
struct ThreadManagerConfig {
  uint32_t compute_threads = defaultComputeThreads();
  bool enable_affinity = true;

private:
  /**
   * @brief return number of compute worker threads
   * priority order:
   *   1. environment variable NNTR_NUM_THREADS
   *   2. compile flag NNTR_NUM_THREADS, when > 0 (-Dnntr-num-threads)
   *   3. getDefaultComputeThreadCount(): on x86 the usable physical cores
   *      (capped at 16), elsewhere std::thread::hardware_concurrency() / 2
   */
  static uint32_t defaultComputeThreads() {
    auto nntr_num_threads = std::getenv("NNTR_NUM_THREADS");
    if (nntr_num_threads) {
      return static_cast<uint32_t>(std::stoul(nntr_num_threads));
    }

#if defined(NNTR_NUM_THREADS) && NNTR_NUM_THREADS > 0
    return NNTR_NUM_THREADS;
#else
    return getDefaultComputeThreadCount();
#endif
  }
};

/**
 * @brief Per-thread range information for parallel work distribution
 */
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
 * @brief Hybrid thread pool: gthreadpool-style threadpool with affinity
 * Use futex for linux, android, mutex and conditional variable for windows.
 *
 * With enable_affinity=true(default):
 *   Workers are pinned to cores.
 *   If core uses SMT, only pin 1 thread per 1 physical core.
 *
 * With enable_affinity=false:
 *   Workers are not pinned to cores.
 *   Let OS scheduler control cores.
 *
 * How to use?
 *   Just wrap the loop body with parallel_for(). Check it for details.
 */
class ThreadManager : public Singleton<ThreadManager> {
public:
  ThreadManager();
  ~ThreadManager();

  /** @brief Return the process-wide ThreadManager singleton. */
  static NNTRAINER_SINGLETON_API ThreadManager &Global();

  /**
   * @brief parallize loop for given function
   * @param begin loop start index
   * @param end loop end index
   * @param fn callback to run in parallel
   * @details This method parallelizes the following for loop.
   * for(i = begin ; i < end ; i++) { fn(i); }
   * @warning callback should not contain another parallel_for()
   * i.e. nested parallel_for is forbidden.
   */
  template <typename F> void parallel_for(size_t begin, size_t end, F &&fn) {
    if (begin >= end) {
      return;
    }

    if (end - begin == 1 || compute_workers_.empty()) {
      for (size_t i = begin; i < end; i++)
        fn(i);
      return;
    }

    parallelize(begin, end, std::forward<F>(fn));
  }

  /**
   * @brief Run a loop only if the pool is available, otherwise return false.
   * @details Nested callbacks also return false, including callbacks on the
   * calling thread. A false result runs no iterations; callers may fall back
   * to serial work. This does not wait for another pool user.
   */
  bool try_parallel_for(size_t begin, size_t end,
                        std::function<void(size_t)> task);

  /**
   * @brief return the number of threads including main thread
   */
  uint32_t getComputeThreadCount() const {
    // main thread is not in compute_workers
    return static_cast<uint32_t>(compute_workers_.size()) + 1;
  }

  /**
   * @brief set config with given parameter
   * @warning It should be called before first call of thread manager.
   */
  static void setConfig(const ThreadManagerConfig &config) { config_ = config; }

private:
#if defined(__linux__) || defined(__ANDROID__)
  /**
   * @brief wrapper for futex wait system call
   */
  inline int futex_wait(std::atomic<uint32_t> *addr, uint32_t val) {
    return syscall(SYS_futex, addr, FUTEX_WAIT | FUTEX_PRIVATE_FLAG, val, NULL);
  }

  /**
   * @brief wrapper for futex wake system call
   */
  inline int futex_wake(std::atomic<uint32_t> *addr) {
    return syscall(SYS_futex, addr, FUTEX_WAKE | FUTEX_PRIVATE_FLAG, INT_MAX);
  }
#endif

  /**
   * @brief wrapper for hw specific yield
   */
  inline void yield() {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_pause();
#elif defined(__aarch64__) || defined(__arm__)
    __asm__ volatile("yield");
#endif
  }

  /**
   * @brief checkin current thread for initialization and finishing job
   */
  void checkin();

  /**
   * @brief wait until command updates with new value
   * It spin-waits for a specific counts, and then falls back to mutex/futex
   * wait.
   */
  uint32_t wait_for_new_command(uint32_t last_command);

  /**
   * @brief wait until every thread finishes job
   */
  void wait_worker_threads();

  /**
   * @brief try fetch and decrement for a given atomic value
   * return false if value is 0
   */
  inline bool try_decrement(std::atomic<size_t> &value);

  /**
   * @brief main function for worker threads
   */
  void thread_main(size_t tid);

  /**
   * @brief callback function for each thread
   */
  void thread_parallelize(size_t my_tid);

  /**
   * @brief do parallelize and compute with workers
   */
  template <typename F> void parallelize(size_t begin, size_t end, F &&task) {
    std::lock_guard<std::mutex> lock(execution_mutex_);
    parallelize_locked(begin, end, std::forward<F>(task));
  }

  /** @brief Dispatch with execution_mutex_ already held. */
  void parallelize_locked(size_t begin, size_t end,
                          std::function<void(size_t)> task);

protected:
  void initialize() noexcept override;

private:
  std::vector<std::thread> compute_workers_;

  std::unique_ptr<thread_info[]> thread_infos_;
  std::mutex execution_mutex_;
  std::function<void(size_t)> task_;

  CACHELINE_ALIGNED std::atomic<uint32_t> command_{threadpool_command::INIT};
  CACHELINE_ALIGNED std::atomic<size_t> active_threads_;
#if defined(__linux__) || defined(__ANDROID__)
  CACHELINE_ALIGNED std::atomic<uint32_t> has_active_threads_;
#elif defined(_WIN32)
  std::mutex command_mutex_;
  std::condition_variable command_cv_;
  std::mutex completion_mutex_;
  std::condition_variable completion_cv_;
#endif

  const uint32_t SPIN_COUNT = 1000000;
  const uint32_t COMMAND_MASK = 0x7FFFFFFF;

  // ─── Config ─────────────────────────────────────────
  static ThreadManagerConfig config_;
};

} // namespace nntrainer

#endif // __NNTRAINER_THREAD_MANAGER_H__
