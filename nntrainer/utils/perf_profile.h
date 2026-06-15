// SPDX-License-Identifier: Apache-2.0
// Minimal RAII timer + global bucket registry for FP16 vs FP32 path profiling.
// Single-process, lock-protected (cheap: only outer-layer scopes are timed,
// not inner parallel_for callbacks). Compiled into causallm_core; main.cpp
// calls PerfBucket::dump() once at the end of generation to print sorted
// totals.

#ifndef __CAUSALLM_PERF_PROFILE_H__
#define __CAUSALLM_PERF_PROFILE_H__

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace causallm {

class PerfBucket {
public:
  static PerfBucket &global() {
    static PerfBucket inst;
    return inst;
  }

  void add(const std::string &name, double ms) {
    std::lock_guard<std::mutex> g(m_);
    auto &e = totals_[name];
    e.total_ms += ms;
    e.calls += 1;
  }

  void dump(const std::string &header = "PERF") {
    std::lock_guard<std::mutex> g(m_);
    if (totals_.empty())
      return;
    std::vector<std::pair<double, std::string>> sorted;
    for (const auto &kv : totals_)
      sorted.emplace_back(kv.second.total_ms, kv.first);
    std::sort(sorted.begin(), sorted.end(),
              [](const std::pair<double, std::string> &a,
                 const std::pair<double, std::string> &b) {
                return a.first > b.first;
              });
    std::cerr << "=== " << header << " ===" << std::endl;
    for (const auto &p : sorted) {
      const auto &e = totals_.at(p.second);
      std::cerr << "  " << std::setw(40) << std::left << p.second
                << " total=" << std::setw(8) << std::right << std::fixed
                << std::setprecision(2) << e.total_ms << " ms"
                << "  calls=" << std::setw(6) << e.calls
                << "  avg=" << std::setw(8) << std::setprecision(2)
                << (e.total_ms * 1000.0 / e.calls) << " us" << std::endl;
    }
    std::cerr << "==============" << std::endl;
  }

  void reset() {
    std::lock_guard<std::mutex> g(m_);
    totals_.clear();
  }

private:
  struct Entry {
    double total_ms = 0.0;
    long long calls = 0;
  };
  std::mutex m_;
  std::map<std::string, Entry> totals_;
};

class PerfScope {
public:
  explicit PerfScope(const char *name)
    : name_(name), start_(std::chrono::steady_clock::now()) {}
  ~PerfScope() {
    auto end = std::chrono::steady_clock::now();
    double ms =
      std::chrono::duration<double, std::milli>(end - start_).count();
    PerfBucket::global().add(name_, ms);
  }

private:
  const char *name_;
  std::chrono::steady_clock::time_point start_;
};

} // namespace causallm

#define PERF_SCOPE(name) ::causallm::PerfScope _perf_scope_##__LINE__(name)

#endif
