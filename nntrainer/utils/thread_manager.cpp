// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 * Copyright (C) 2025 Jaemin Shin <jaemin2.shin@samsung.com>
 * @file   thread_manager.cpp
 * @date   20 March 2026
 * @brief  Unified thread manager: spin-wait (affinity) or condvar (default)
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Jaemin Shin <jaemin2.shin@samsung.com>
 * @bug    No known bugs except for NYI items
 */

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

inline size_t modulo_decrement(size_t i, size_t n) {
  if (i == 0)
    i = n;
  return i - 1;
}

} // namespace

namespace nntrainer {

ThreadManagerConfig ThreadManager::config_ = {};

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
  for (uint32_t i = SPIN_COUNT; i != 0; i--) {
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
