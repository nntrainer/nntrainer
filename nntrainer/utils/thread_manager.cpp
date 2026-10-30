// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 * Copyright (C) 2025 Jaemin Shin <jaemin2.shin@samsung.com>
 * @file   thread_manager.cpp
 * @date   23 April 2026
 * @brief  Unified thread manager: spin-wait (affinity) or condvar (default)
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Jaemin Shin <jaemin2.shin@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

#include "thread_manager.h"

#if defined(__linux__) || defined(__ANDROID__)
#include <sched.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

namespace {

// Every dispatched callback, including the caller's own share, passes through
// thread_parallelize. This also covers callers compiled with the old inline
// parallelize body and avoids try_lock on a mutex owned by the current thread.
thread_local bool in_parallel_callback = false;

struct ParallelCallbackScope {
  bool previous = in_parallel_callback;
  ParallelCallbackScope() { in_parallel_callback = true; }
  ~ParallelCallbackScope() { in_parallel_callback = previous; }
};

uint32_t workerIdleSpins() {
  static const uint32_t spins = [] {
    constexpr uint32_t limit = 1000000;
    const char *value = std::getenv("NNTR_WORKER_IDLE_SPINS");
    if (!value || !*value)
      return limit;
    uint32_t count = 0;
    for (const char *p = value; *p; ++p) {
      if (*p < '0' || *p > '9')
        return limit;
      const uint32_t digit = static_cast<uint32_t>(*p - '0');
      if (count > (limit - digit) / 10)
        return limit;
      count = count * 10 + digit;
    }
    return count;
  }();
  return spins;
}

inline size_t modulo_decrement(size_t i, size_t n) {
  if (i == 0)
    i = n;
  return i - 1;
}

} // namespace

namespace nntrainer {

ThreadManagerConfig ThreadManager::config_ = {};

/** @brief Return the ThreadManager singleton. */
template <>
NNTRAINER_SINGLETON_API ThreadManager &Singleton<ThreadManager>::Global() {
  static ThreadManager instance;
  instance.initializeOnce();
  return instance;
}

/** @brief Return the process-wide ThreadManager singleton. */
NNTRAINER_SINGLETON_API ThreadManager &ThreadManager::Global() {
  return Singleton<ThreadManager>::Global();
}

ThreadManager::ThreadManager() {}

ThreadManager::~ThreadManager() {
#if defined(__linux__) || defined(__ANDROID__)
  active_threads_.store(compute_workers_.size(), std::memory_order_relaxed);
  has_active_threads_.store(1, std::memory_order_relaxed);

  command_.store(threadpool_command::SHUTDOWN, std::memory_order_release);
  futex_wake(&command_);
#elif defined(_WIN32)
  std::unique_lock<std::mutex> lock(command_mutex_);
  active_threads_.store(compute_workers_.size(), std::memory_order_relaxed);
  command_.store(threadpool_command::SHUTDOWN, std::memory_order_release);
  command_cv_.notify_all();
  lock.unlock();

#endif

  for (auto &t : compute_workers_) {
    t.join();
  }
}

void ThreadManager::initialize() noexcept {
  auto config = config_;

  // Don't initialize for 1 thread(0 worker)
  if (config.compute_threads <= 1u)
    return;

  uint32_t hw_threads = getPhysicalCoreCount();

  // Don't initialize for single core system
  if (hw_threads <= 1)
    return;

  // adjust configuration if thread number overs
  if (hw_threads < config.compute_threads) {
    std::cerr << "Too many threads!\n"
              << "  available threads: " << hw_threads << "\n"
              << "  compute_threads: " << config.compute_threads << "\n";

    config.compute_threads = hw_threads;
  }

  // compute core assignment map (sorted by performance for big.LITTLE)
  std::vector<uint32_t> core_map;
  if (config.enable_affinity) {
    core_map = getCoresByPerformance();
  }

  // pin main thread first
  if (config.enable_affinity) {
    pinSelfToCore(core_map[0]);
  }

  // total compute_threads = one main thread + compute_workers
  uint32_t num_workers = config.compute_threads - 1;

  thread_infos_ = std::make_unique<thread_info[]>(config.compute_threads);
#if defined(__linux__) || defined(__ANDROID__)
  has_active_threads_.store(1, std::memory_order_relaxed);
#endif
  active_threads_.store(num_workers, std::memory_order_relaxed);

  compute_workers_.reserve(num_workers);

  // start compute workers with appropriate loop
  for (uint32_t i = 1; i <= num_workers; ++i) {
    int core_id = config.enable_affinity ? static_cast<int>(core_map[i]) : -1;
    compute_workers_.emplace_back([this, i, core_id] {
      if (core_id >= 0)
        pinSelfToCore(static_cast<uint32_t>(core_id));
      thread_main((size_t)i);
    });
  }

  wait_worker_threads();
}

void ThreadManager::checkin() {
#if defined(__linux__) || defined(__ANDROID__)
  size_t t = active_threads_.fetch_sub(1, std::memory_order_acq_rel) - 1;
  if (t == 0) {
    has_active_threads_.store(0, std::memory_order_release);
    futex_wake(&has_active_threads_);
  }
#elif defined(_WIN32)
  std::lock_guard<std::mutex> lock(completion_mutex_);
  size_t t = active_threads_.fetch_sub(1, std::memory_order_acq_rel) - 1;
  if (t == 0) {
    completion_cv_.notify_all();
  }

#endif
}

uint32_t ThreadManager::wait_for_new_command(uint32_t last_command) {
  // try once
  uint32_t command = command_.load(std::memory_order_acquire);
  if (command != last_command)
    return command;

  // spin wait
  for (uint32_t i = workerIdleSpins(); i != 0; i--) {
    yield();

    command = command_.load(std::memory_order_acquire);
    if (command != last_command)
      return command;
  }

// fall back to futex/mutex wait
#if defined(__linux__) || defined(__ANDROID__)
  do {
    futex_wait(&command_, last_command);
    command = command_.load(std::memory_order_acquire);
  } while (command == last_command);
#elif defined(_WIN32)
  std::unique_lock<std::mutex> lock(command_mutex_);
  while ((command = command_.load(std::memory_order_acquire)) == last_command) {
    command_cv_.wait(lock);
  }
#endif

  return command;
}

void ThreadManager::wait_worker_threads() {
#if defined(__linux__) || defined(__ANDROID__)
  uint32_t has_active_threads =
    has_active_threads_.load(std::memory_order_acquire);

  if (has_active_threads == 0)
    return;
#elif defined(_WIN32)
  size_t active_threads = active_threads_.load(std::memory_order_acquire);

  if (active_threads == 0)
    return;
#endif

  for (uint32_t i = SPIN_COUNT; i != 0; i--) {
    yield();

#if defined(__linux__) || defined(__ANDROID__)
    has_active_threads = has_active_threads_.load(std::memory_order_acquire);

    if (has_active_threads == 0)
      return;
#elif defined(_WIN32)
    active_threads = active_threads_.load(std::memory_order_acquire);

    if (active_threads == 0)
      return;
#endif
  }
#if defined(__linux__) || defined(__ANDROID__)
  while ((has_active_threads =
            has_active_threads_.load(std::memory_order_acquire)) != 0) {
    futex_wait(&has_active_threads_, 1);
  }
#elif defined(_WIN32)
  std::unique_lock<std::mutex> lock(completion_mutex_);
  while ((active_threads = active_threads_.load(std::memory_order_acquire)) !=
         0) {
    completion_cv_.wait(lock);
  }
#endif
}

bool ThreadManager::try_parallel_for(size_t begin, size_t end,
                                     std::function<void(size_t)> task) {
  if (in_parallel_callback)
    return false;
  std::unique_lock<std::mutex> lock(execution_mutex_, std::try_to_lock);
  if (!lock.owns_lock())
    return false;
  if (begin >= end)
    return true;
  if (end - begin == 1 || compute_workers_.empty()) {
    ParallelCallbackScope scope;
    for (size_t i = begin; i < end; ++i)
      task(i);
    return true;
  }
  parallelize_locked(begin, end, std::move(task));
  return true;
}

void ThreadManager::parallelize_locked(size_t begin, size_t end,
                                       std::function<void(size_t)> task) {
#if defined(_WIN32)
  std::unique_lock<std::mutex> command_lock(command_mutex_);
#endif

  task_ = std::move(task);

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
  thread_parallelize(0);

  wait_worker_threads();

  std::atomic_thread_fence(std::memory_order_acquire);
}

void ThreadManager::thread_main(size_t tid) {
  uint32_t last_command = threadpool_command::INIT;
  checkin();
  while (true) {
    uint32_t command = wait_for_new_command(last_command);
    std::atomic_thread_fence(std::memory_order_acquire);

    switch (command & COMMAND_MASK) {
    case RUN: {
      thread_parallelize(tid);
      break;
    }
    case SHUTDOWN:
      return;
    case INIT:
      break;
    }

    checkin();

    last_command = command;
  }
}

inline bool ThreadManager::try_decrement(std::atomic<size_t> &value) {
  size_t actual_value = value.load(std::memory_order_relaxed);
  while (actual_value != 0) {
    if (value.compare_exchange_weak(actual_value, actual_value - 1,
                                    std::memory_order_relaxed))
      return true;
  }
  return false;
}

void ThreadManager::thread_parallelize(size_t my_tid) {
  ParallelCallbackScope scope;

  // process my job
  size_t range_start =
    thread_infos_[my_tid].range_start.load(std::memory_order_relaxed);

  // fetching range_length resolves conflict with other thread
  while (try_decrement(thread_infos_[my_tid].range_length)) {
    task_(range_start++);
  }

  // steal other thread's job
  size_t threads_count = compute_workers_.size() + 1;
  for (size_t tid = modulo_decrement(my_tid, threads_count); tid != my_tid;
       tid = modulo_decrement(tid, threads_count)) {
    while (try_decrement(thread_infos_[tid].range_length)) {
      size_t index =
        thread_infos_[tid].range_end.fetch_sub(1, std::memory_order_relaxed) -
        1;
      task_(index);
    }
  }

  std::atomic_thread_fence(std::memory_order_release);
}

} // namespace nntrainer
