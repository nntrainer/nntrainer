// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   thread_manager.cpp
 * @date   20 March 2026
 * @brief  Unified thread manager: spin-wait (affinity) or condvar (default)
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
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
constexpr bool is_x86 =
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) ||             \
  defined(_M_IX86)
  true;
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) ||         \
  defined(_M_ARM)
  false;
#endif

unsigned int readUInt(const std::string &path) {
  std::ifstream f(path);
  unsigned int v = 0;
  if (f)
    f >> v;
  return v;
}

std::string readStr(const std::string &path) {
  std::ifstream f(path);
  std::string s;
  if (f)
    f >> s;
  return s;
}

// "0-3,8" -> {0,1,2,3,8}
std::vector<unsigned int> parseCpuList(const std::string &s) {
  std::vector<unsigned int> out;
  std::stringstream ss(s);
  std::string token;

  while (std::getline(ss, token, ',')) {
    size_t dash = token.find('-');
    if (dash == std::string::npos) {
      out.push_back(std::stoi(token));
    } else {
      unsigned int a = std::stoi(token.substr(0, dash));
      unsigned int b = std::stoi(token.substr(dash + 1));
      for (int i = a; i <= b; i++) {
        out.push_back(i);
      }
    }
  }
  return out;
}

} // namespace

namespace nntrainer {

/**
 * @brief Pin the CALLING thread to a specific CPU core.
 * Used on Android where affinity must be set from within the target thread.
 */
static bool pinSelfToCore(unsigned int core_id) {
#if defined(__linux__) || defined(__ANDROID__) || defined(__TIZEN__)
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core_id, &cpuset);
  int ret = sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
  if (ret != 0) {
    std::cerr << "Warning: pinning thread on cpu" << core_id << " failed!"
              << std::endl;
  }
  return ret == 0;
#elif defined(_WIN32)
  /// @todo support windows
  (void)core_id;
  return true;
#else
  (void)core_id;
  return true;
#endif
}

static unsigned int getPhysicalCoreCount() {
  if constexpr (is_x86) {
    unsigned int smt = readUInt("/sys/devices/system/cpu/smt/active");
    bool is_smt = (smt == 1);
    bool is_hybrid = std::filesystem::exists("/sys/devices/cpu_core");

    if (is_hybrid) {
      std::string p_cores = readStr("/sys/devices/cpu_core/cpus");
      auto p_list = parseCpuList(p_cores);
      std::string e_cores = readStr("/sys/devices/cpu_atom/cpus");
      auto e_list = parseCpuList(e_cores);

      if (is_smt) {
        return p_list.size() / 2 + e_list.size();
      } else {
        return p_list.size() + e_list.size();
      }
    } else {
      unsigned int hw = std::thread::hardware_concurrency();
      return is_smt ? (hw / 2) : hw;
    }
  } else {
    // ARM doesn't support SMT
    return std::thread::hardware_concurrency();
  }
}

static std::vector<unsigned int> getCoresByPerformance() {
#if defined(__linux__) || defined(__ANDROID__)
  if constexpr (is_x86) {
    unsigned int hw_threads = std::thread::hardware_concurrency();
    std::vector<std::pair<unsigned int, unsigned int>> freq_core;
    freq_core.reserve(getPhysicalCoreCount());

    for (unsigned int i = 0; i < hw_threads; ++i) {
      std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(i);

      std::string siblings = readStr(base + "/topology/thread_siblings_list");
      if (!siblings.empty()) {
        auto list = parseCpuList(siblings);
        // use first core only
        if (!list.empty() && list[0] != i)
          continue;
      }

      unsigned int freq = readUInt(base + "/cpufreq/cpuinfo_max_freq");
      freq_core.push_back({freq, i});
    }

    bool has_freq = false;
    for (auto &p : freq_core)
      if (p.first > 0) {
        has_freq = true;
        break;
      }

    if (!has_freq) {
      std::vector<unsigned int> cores(freq_core.size());
      for (unsigned int i = 0; i < freq_core.size(); ++i)
        cores[i] = i;
      return cores;
    }

    std::sort(freq_core.begin(), freq_core.end(),
              [](const auto &a, const auto &b) { return a.first > b.first; });

    std::vector<unsigned int> cores;
    cores.reserve(freq_core.size());
    for (auto &p : freq_core)
      cores.push_back(p.second);
    return cores;
  } else {
    // ARM cores does not have SMT
    unsigned int hw_threads = std::thread::hardware_concurrency();
    std::vector<std::pair<unsigned int, unsigned int>> freq_core;
    freq_core.reserve(hw_threads);
    for (unsigned int i = 0; i < hw_threads; ++i) {
      std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(i) +
                         "/cpufreq/cpuinfo_max_freq";

      unsigned int freq = readUInt(path);
      freq_core.push_back({freq, i});
    }
    bool has_freq = false;
    for (auto &p : freq_core)
      if (p.first > 0) {
        has_freq = true;
        break;
      }

    if (!has_freq) {
      std::vector<unsigned int> cores(hw_threads);
      for (unsigned int i = 0; i < hw_threads; ++i)
        cores[i] = i;
      return cores;
    }

    std::sort(freq_core.begin(), freq_core.end(),
              [](const auto &a, const auto &b) { return a.first > b.first; });

    std::vector<unsigned int> cores;
    cores.reserve(hw_threads);
    for (auto &p : freq_core)
      cores.push_back(p.second);
    return cores;
  }

#elif defined(_WIN32)
  /// @todo support windows
  unsigned int hw_threads = std::thread::hardware_concurrency();
  std::vector<unsigned int> cores(hw_threads);
  for (unsigned int i = 0; i < hw_threads; ++i)
    cores[i] = i;
  return cores;
#endif
}

ThreadManagerConfig ThreadManager::config_ = {};

ThreadManager::ThreadManager() {}

ThreadManager::~ThreadManager() {

  active_threads_.store(compute_workers_.size(), std::memory_order_relaxed);
  has_active_threads_.store(1, std::memory_order_relaxed);

  command_.store(threadpool_command::SHUTDOWN, std::memory_order_release);
  futex_wake(&command_);

  for(auto &t : compute_workers_){
    t.join();
  }

  return;

  stop_.store(true, std::memory_order_release);

  if (spin_mode_) {
    // bump generation to wake spin-waiting workers
    spin_generation_.fetch_add(1, std::memory_order_seq_cst);
  } else {
    dispatch_cv_.notify_all();
  }

  for (auto &t : compute_workers_)
    if (t.joinable())
      t.join();

  io_cv_.notify_all();
  for (auto &t : io_workers_)
    if (t.joinable())
      t.join();
}

void ThreadManager::initialize() noexcept {
  auto config = config_;

  unsigned int hw_threads = getPhysicalCoreCount();

  // Don't initialize for single core system
  if (hw_threads <= 1)
    return;

  // adjust configuration if thread number overs
  if (hw_threads < config.compute_threads + config.io_threads) {
    std::cerr << "Too many threads!\n"
              << "  available threads: " << hw_threads << "\n"
              << "  compute_threads: " << config.compute_threads << "\n"
              << "  io_threads: " << config.io_threads << std::endl;

    // use at most 1 io thread
    if (config.io_threads > 1)
      config.io_threads = 1;

    // check if it still overs
    if (hw_threads < config.compute_threads + config.io_threads)
      config.compute_threads = hw_threads - config.io_threads;
  }

  // set mode based on affinity setting
  spin_mode_ = config.enable_affinity;

  // compute core assignment map (sorted by performance for big.LITTLE)
  std::vector<unsigned int> core_map;
  if (config.enable_affinity) {
    core_map = getCoresByPerformance();
  }

  // pin main thread first
  unsigned int idx = 0;
  if (config.enable_affinity) {
    pinSelfToCore(core_map[idx++]);
  }

  // total compute_threads = one main thread + compute_workers
  unsigned int num_additional_threads = config.compute_threads - 1;

  thread_infos_ = std::make_unique<thread_info[]>(config.compute_threads);
  has_active_threads_.store(1, std::memory_order_relaxed);
  active_threads_.store(config.compute_threads - 1, std::memory_order_relaxed);

  compute_workers_.reserve(num_additional_threads);

  // start compute workers with appropriate loop
  for (unsigned int i = 0; i < num_additional_threads; ++i) {
    int core_id = config.enable_affinity
                    ? static_cast<int>(core_map[(idx++) % core_map.size()])
                    : -1;
    compute_workers_.emplace_back([this, i, core_id] {
      if (core_id >= 0)
        pinSelfToCore(static_cast<unsigned int>(core_id));
      thread_main((size_t)i + 1);
    });
  }

  wait_worker_threads();

  return;

  // wait for all spin workers to be ready before allowing dispatches
  if (spin_mode_) {
    while (spin_workers_ready_.load(std::memory_order_acquire) <
           num_additional_threads) {
      std::this_thread::yield();
    }
  }

  // start I/O workers
  io_workers_.reserve(config.io_threads);
  for (unsigned int i = 0; i < config.io_threads; ++i) {
    int core_id = config.enable_affinity
                    ? static_cast<int>(core_map[(idx++) % core_map.size()])
                    : -1;
    io_workers_.emplace_back([this, core_id] {
      if (core_id >= 0)
        pinSelfToCore(static_cast<unsigned int>(core_id));
      ioWorkerLoop();
    });
  }
}

// ─── SPIN-WAIT WORKER (GGML-style, used when affinity=true) ──────

void ThreadManager::computeWorkerLoopSpin(unsigned int worker_id) {
  unsigned int my_gen = spin_generation_.load(std::memory_order_acquire);

  // signal that this worker is ready in the spin loop
  spin_workers_ready_.fetch_add(1, std::memory_order_release);

  while (true) {
    // spin-wait for new generation
    while (spin_generation_.load(std::memory_order_acquire) == my_gen) {
      if (stop_.load(std::memory_order_acquire))
        return;
      cpuRelax();
    }
    my_gen = spin_generation_.load(std::memory_order_acquire);

    if (stop_.load(std::memory_order_acquire))
      return;

    // only active workers do work + barrier
    if (worker_id < spin_active_workers_.load(std::memory_order_acquire)) {
      bool sense = spin_current_sense_.load(std::memory_order_acquire);
      size_t end = task_end_;
      while (true) {
        size_t idx = current_chunk_.fetch_add(1, std::memory_order_relaxed);
        if (idx >= end)
          break;
        current_task_(idx);
      }

      spinBarrier(sense);
    }
    // inactive workers loop back to generation spin
  }
}

// ─── CONDVAR WORKER (safe without affinity, default) ─────────────

void ThreadManager::computeWorkerLoopCondvar(unsigned int worker_id) {
  unsigned int my_gen = 0;

  while (true) {
    // wait for new dispatch via condvar
    unsigned int my_barrier_gen;
    {
      std::unique_lock<std::mutex> lock(dispatch_mutex_);
      dispatch_cv_.wait(lock, [this, &my_gen] {
        return dispatch_gen_ != my_gen || stop_.load(std::memory_order_acquire);
      });
      if (stop_.load(std::memory_order_acquire))
        return;
      my_gen = dispatch_gen_;
      my_barrier_gen = barrier_gen_;
    }

    // only active workers do work + barrier
    if (worker_id < cv_active_workers_) {
      size_t end = task_end_;
      while (true) {
        size_t idx = current_chunk_.fetch_add(1, std::memory_order_relaxed);
        if (idx >= end)
          break;
        current_task_(idx);
      }

      // arrive at barrier (generation-based: no reset race)
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
    }
    // inactive workers skip both work and barrier
  }
}

// ─── I/O WORKER ──────────────────────────────────────────────────

void ThreadManager::ioWorkerLoop() {
  while (true) {
    std::pair<std::function<void()>,
              std::shared_ptr<CompletionToken::SharedState>>
      item;
    {
      std::unique_lock<std::mutex> lock(io_mutex_);
      io_cv_.wait(lock, [this] {
        return !io_queue_.empty() || stop_.load(std::memory_order_acquire);
      });
      if (stop_.load(std::memory_order_acquire) && io_queue_.empty())
        return;
      item = std::move(io_queue_.front());
      io_queue_.pop();
    }

    auto &task = item.first;
    auto &state = item.second;

    try {
      task();
      {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->done.store(true, std::memory_order_release);
      }
      state->cv.notify_all();
    } catch (...) {
      {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->exception = std::current_exception();
        state->done.store(true, std::memory_order_release);
      }
      state->cv.notify_all();
    }
  }
}

CompletionToken ThreadManager::submit(std::function<void()> task) {
  if (io_workers_.empty())
    throw std::runtime_error{"io thread is not enabled."};

  CompletionToken token = CompletionToken::create();
  {
    std::lock_guard<std::mutex> lock(io_mutex_);
    io_queue_.push({std::move(task), token.getState()});
  }
  io_cv_.notify_one();
  return token;
}

} // namespace nntrainer
