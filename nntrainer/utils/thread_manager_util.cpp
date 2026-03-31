// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jaemin Shin <jaemin2.shin@samsung.com>
 *
 * @file   thread_manager_util.cpp
 * @date   20 April 2026
 * @brief  Utils for unified thread manager
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jaemin Shin <jaemin2.shin@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "thread_manager_util.h"

namespace {
constexpr static bool is_x86 =
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) ||             \
  defined(_M_IX86)
  true;
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) ||         \
  defined(_M_ARM)
  false;
#endif
} // namespace

namespace nntrainer {

uint32_t readUInt(const std::string &path) {
  std::ifstream f(path);
  uint32_t v = 0;
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

std::vector<uint32_t> parseCpuList(const std::string &s) {
  std::vector<uint32_t> out;
  std::stringstream ss(s);
  std::string token;

  while (std::getline(ss, token, ',')) {
    size_t dash = token.find('-');
    if (dash == std::string::npos) {
      out.push_back(std::stoi(token));
    } else {
      uint32_t a = std::stoi(token.substr(0, dash));
      uint32_t b = std::stoi(token.substr(dash + 1));
      for (int i = a; i <= b; i++) {
        out.push_back(i);
      }
    }
  }
  return out;
}

uint32_t getPhysicalCoreCount() {
#if defined(__linux__) || defined(__ANDROID__)
  if constexpr (is_x86) {
    uint32_t smt = readUInt("/sys/devices/system/cpu/smt/active");
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
      uint32_t hw = std::thread::hardware_concurrency();
      return is_smt ? (hw / 2) : hw;
    }
  } else {
    // ARM doesn't support SMT
    return std::thread::hardware_concurrency();
  }
#elif defined(_WIN32)
  // todo support windows
  return std::thread::hardware_concurrency();
#endif
}

std::vector<uint32_t> getCoresByPerformance() {
#if defined(__linux__) || defined(__ANDROID__)
  if constexpr (is_x86) {
    uint32_t hw_threads = std::thread::hardware_concurrency();
    std::vector<std::pair<uint32_t, uint32_t>> freq_core;
    freq_core.reserve(getPhysicalCoreCount());

    for (uint32_t i = 0; i < hw_threads; ++i) {
      std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(i);

      std::string siblings = readStr(base + "/topology/thread_siblings_list");
      if (!siblings.empty()) {
        auto list = parseCpuList(siblings);
        // use first core only
        if (!list.empty() && list[0] != i)
          continue;
      }

      uint32_t freq = readUInt(base + "/cpufreq/cpuinfo_max_freq");
      freq_core.push_back({freq, i});
    }

    bool has_freq = false;
    for (auto &p : freq_core)
      if (p.first > 0) {
        has_freq = true;
        break;
      }

    if (!has_freq) {
      std::vector<uint32_t> cores(freq_core.size());
      for (uint32_t i = 0; i < freq_core.size(); ++i)
        cores[i] = i;
      return cores;
    }

    std::sort(freq_core.begin(), freq_core.end(),
              [](const auto &a, const auto &b) { return a.first > b.first; });

    std::vector<uint32_t> cores;
    cores.reserve(freq_core.size());
    for (auto &p : freq_core)
      cores.push_back(p.second);
    return cores;
  } else {
    // ARM cores does not have SMT
    uint32_t hw_threads = std::thread::hardware_concurrency();
    std::vector<std::pair<uint32_t, uint32_t>> freq_core;
    freq_core.reserve(hw_threads);
    for (uint32_t i = 0; i < hw_threads; ++i) {
      std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(i) +
                         "/cpufreq/cpuinfo_max_freq";

      uint32_t freq = readUInt(path);
      freq_core.push_back({freq, i});
    }
    bool has_freq = false;
    for (auto &p : freq_core)
      if (p.first > 0) {
        has_freq = true;
        break;
      }

    if (!has_freq) {
      std::vector<uint32_t> cores(hw_threads);
      for (uint32_t i = 0; i < hw_threads; ++i)
        cores[i] = i;
      return cores;
    }

    std::sort(freq_core.begin(), freq_core.end(),
              [](const auto &a, const auto &b) { return a.first > b.first; });

    std::vector<uint32_t> cores;
    cores.reserve(hw_threads);
    for (auto &p : freq_core)
      cores.push_back(p.second);
    return cores;
  }

#elif defined(_WIN32)
  /// @todo support windows
  uint32_t hw_threads = std::thread::hardware_concurrency();
  std::vector<uint32_t> cores(hw_threads);
  for (uint32_t i = 0; i < hw_threads; ++i)
    cores[i] = i;
  return cores;
#endif
}

/**
 * @brief Pin the CALLING thread to a specific CPU core.
 * Used on Android where affinity must be set from within the target thread.
 */
bool pinSelfToCore(uint32_t core_id) {
#if defined(__linux__) || defined(__ANDROID__)
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
#endif
}

} // namespace nntrainer
