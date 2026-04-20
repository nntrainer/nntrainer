// SPDX-License-Identifier: Apache-2.0, BSD 2-Clause "Simplified" License
/*
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
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 * Copyright (C) 2025 Jaemin Shin <jaemin2.shin@samsung.com>
 *
 * @file   thread_manager.h
 * @date   20 March 2026
 * @brief  Unified thread manager for compute, inspired by pthreadpool
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Jaemin Shin <jaemin2.shin@samsung.com>
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

#if defined(__GNUC__)
#define CACHELINE_ALIGNED __attribute__((__aligned__(64)))
#elif defined(_MSC_VER)
#define CACHELINE_ALIGNED __declspec(align(64))
#else
#error "Platform-specific implementation of CACHELINE_ALIGNED required"
#endif

struct ThreadManagerConfig {
  uint32_t compute_threads = defaultComputeThreads();
  bool enable_affinity = true;

private:
  /**
   * @brief return number of compute worker threads
   * priority order:
   *   1. environment variable NNTR_NUM_THREADS
   *   2. compile flag NNTR_NUM_THREADS
   *   3. std::thread::hardware_concurrency() / 2
   */
  static uint32_t defaultComputeThreads() {
    auto nntr_num_threads = std::getenv("NNTR_NUM_THREADS");
    if (nntr_num_threads) {
      return static_cast<uint32_t>(std::stoul(nntr_num_threads));
    }

#if defined(NNTR_NUM_THREADS) && NNTR_NUM_THREADS > 0
    return NNTR_NUM_THREADS;
#else
    /// @todo use performance core only for x86
    uint32_t hw = std::thread::hardware_concurrency();
    return hw > 0 ? hw / 2 : 1;
#endif
  }
};

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
 */
class ThreadManager : public Singleton<ThreadManager> {
public:
  ThreadManager();
  ~ThreadManager();

  /**
   * @brief parallize loop for given function
   */
  template <typename F> void parallel_for(size_t begin, size_t end, F &&fn) {
    if (begin >= end) {
      return;
    }
    parallelize_1d(begin, end, std::forward<F>(fn));
  }

  /**
   * @brief parallize loop for given function
   */
  template <typename F>
  void parallel_for(size_t begin, size_t end, size_t n_workers, F &&fn) {
    parallel_for(begin, end, std::forward<F>(fn));
  }

  /**
   * @brief parallize loop for given function
   */
  template <typename F> void parallel_for_chunked(size_t n_threads, F &&fn) {
    // todo: divie chunk inside this method
    parallelize_1d(0, n_threads, std::forward<F>(fn));
  }

  /**
   * @brief return the number of threads including main thread
   */
  uint32_t getComputeThreadCount() const {
    // main thread is not in compute_workers
    return static_cast<uint32_t>(compute_workers_.size()) + 1;
  }

  /**
   * @brief set config with given parameter
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
  void thread_parallelize_1d(size_t my_tid);

  /**
   * @brief
   */
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

  /**
   * @brief do parallelize and compute with workers
   */
  template <typename F>
  void parallelize(std::function<void(size_t)> thread_function, size_t begin,
                   size_t end, F &&task) {
    std::lock_guard<std::mutex> lock(execution_mutex_);

#if defined(_WIN32)
    std::unique_lock<std::mutex> command_lock(command_mutex_);
#endif

    task_ = std::move(task);
    thread_function_ = thread_function;

    // set active thread numbers
    size_t threads_count = compute_workers_.size() + 1;
    active_threads_.store(threads_count - 1, std::memory_order_relaxed);
#if defined(__linux__) || defined(__ANDROID__)
    has_active_threads_.store(1, std::memory_order_relaxed);
#endif
    // distribute loops
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

      range_start = range_end;
    }

    // always flipping MSB makes workers notice new command arrives even if last
    // command was same command
    uint32_t old_command = command_.load(std::memory_order_relaxed);
    uint32_t new_command =
      (~(old_command | COMMAND_MASK)) | threadpool_command::RUN;

    command_.store(new_command, std::memory_order_release);

#if defined(__linux__) || defined(__ANDROID__)
    futex_wake(&command_);
#elif defined(_WIN32)
    command_lock.unlock();
    command_cv_.notify_all();
#endif

    // main thread also works
    thread_function_(0);

    wait_worker_threads();

    std::atomic_thread_fence(std::memory_order_acquire);
  }

protected:
  void initialize() noexcept override;

private:
  std::vector<std::thread> compute_workers_;

  std::unique_ptr<thread_info[]> thread_infos_;
  std::mutex execution_mutex_;
  std::function<void(size_t)> thread_function_;
  std::function<void(size_t)> task_;

  CACHELINE_ALIGNED std::atomic<uint32_t> command_;
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
