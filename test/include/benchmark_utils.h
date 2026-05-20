// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Yonghyeon Cho
 *
 * @file  benchmark_utils.h
 * @date  24 March 2026
 * @brief General-purpose benchmark utilities for nntrainer
 * @see   https://github.com/nntrainer/nntrainer
 * @author Yonghyeon Cho
 * @bug   No known bugs except for NYI items
 *
 * @details Provides a minimal, reusable benchmark framework for measuring
 * function latency, throughput, GFLOPS, and memory bandwidth. Any test in
 * nntrainer can include this header to benchmark backend functions.
 *
 * Usage:
 *   #include "benchmark_utils.h"
 *
 *   auto stats = bench::measure([&]() { nntrainer::softmax(N, X, Y); });
 *   bench::report("softmax", "FP32", "N=1024", stats);
 *
 *   bench::compare("ele_sub", "FP32", "N=1024", baseline, optimized);
 */

#ifndef __BENCHMARK_UTILS_H__
#define __BENCHMARK_UTILS_H__

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

namespace bench {

/**
 * @brief Timing statistics from benchmark measurements
 */
struct Stats {
  double avg_ns; /**< average latency in nanoseconds */
  double min_ns; /**< minimum latency in nanoseconds */
  double max_ns; /**< maximum latency in nanoseconds */
};

/**
 * @brief Optional derived metrics for benchmark reporting
 *
 * Set only the fields relevant to your benchmark. Unset fields (0) are
 * omitted from the output.
 */
struct Metrics {
  size_t num_elements = 0; /**< element count → throughput (elem/s) */
  size_t total_bytes = 0; /**< bytes read+written → memory bandwidth (GB/s) */
  double flop_count = 0.0; /**< FP operations → GFLOPS */
};

/**
 * @brief Format nanoseconds to human-readable string with auto-scaled unit
 *
 * @param ns time in nanoseconds
 * @return formatted string (e.g., "312 ns", "4.12 us", "1.50 ms")
 */
inline std::string format_time(double ns) {
  std::ostringstream oss;
  if (ns < 1e3) {
    oss << std::fixed << std::setprecision(0) << ns << " ns";
  } else if (ns < 1e6) {
    oss << std::fixed << std::setprecision(2) << ns / 1e3 << " us";
  } else if (ns < 1e9) {
    oss << std::fixed << std::setprecision(2) << ns / 1e6 << " ms";
  } else {
    oss << std::fixed << std::setprecision(3) << ns / 1e9 << " s";
  }
  return oss.str();
}

/**
 * @brief Format throughput to human-readable string
 *
 * @param elem_per_sec elements per second
 * @return formatted string (e.g., "3.28 Gelem/s", "332 Melem/s")
 */
inline std::string format_throughput(double elem_per_sec) {
  std::ostringstream oss;
  if (elem_per_sec >= 1e9) {
    oss << std::fixed << std::setprecision(2) << elem_per_sec / 1e9
        << " Gelem/s";
  } else if (elem_per_sec >= 1e6) {
    oss << std::fixed << std::setprecision(0) << elem_per_sec / 1e6
        << " Melem/s";
  } else if (elem_per_sec >= 1e3) {
    oss << std::fixed << std::setprecision(0) << elem_per_sec / 1e3
        << " Kelem/s";
  } else {
    oss << std::fixed << std::setprecision(0) << elem_per_sec << " elem/s";
  }
  return oss.str();
}

/**
 * @brief Measure execution time of a callable over multiple iterations
 *
 * @tparam Func callable type (lambda, function pointer, etc.)
 * @param func function to benchmark
 * @param warmup number of warmup iterations (not measured)
 * @param iters number of measured iterations
 * @return Stats average, min, max latency in nanoseconds
 */
template <typename Func>
Stats measure(Func &&func, unsigned int warmup = 10,
              unsigned int iters = 1000) {
  using clock = std::chrono::steady_clock;

  if (iters == 0)
    iters = 1;

  for (unsigned int i = 0; i < warmup; ++i) {
    func();
  }

  double total_ns = 0.0;
  double min_ns = std::numeric_limits<double>::max();
  double max_ns = 0.0;

  for (unsigned int i = 0; i < iters; ++i) {
    auto start = clock::now();
    func();
    auto end = clock::now();
    double ns = static_cast<double>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
        .count());
    total_ns += ns;
    min_ns = std::min(min_ns, ns);
    max_ns = std::max(max_ns, ns);
  }

  return {total_ns / iters, min_ns, max_ns};
}

/**
 * @brief Print a table header for benchmark reports
 *
 * Call once before a group of report() calls for aligned output.
 */
inline void print_header() {
  std::cout << "  " << std::left << std::setw(22) << "Function" << std::setw(7)
            << "Type" << std::setw(18) << "Size" << std::right << std::setw(10)
            << "Avg" << std::setw(12) << "Min" << std::setw(12) << "Max"
            << std::setw(15) << "Throughput" << std::setw(12) << "Mem BW"
            << std::endl;
  std::cout << "  " << std::string(20, '-') << "  " << std::string(5, '-')
            << "  " << std::string(16, '-') << "  " << std::string(8, '-')
            << "  " << std::string(10, '-') << "  " << std::string(10, '-')
            << "  " << std::string(13, '-') << "  " << std::string(10, '-')
            << std::endl;
}

/**
 * @brief Print a single benchmark result row
 *
 * @param name function name (e.g., "ele_sub", "sgemm")
 * @param type data type (e.g., "FP32", "FP16")
 * @param size size description (e.g., "N=1024", "256x1024x1024")
 * @param stats timing statistics from measure()
 * @param metrics optional derived metrics (throughput, bandwidth, GFLOPS)
 */
inline void report(const std::string &name, const std::string &type,
                   const std::string &size, const Stats &stats,
                   const Metrics &metrics = {}) {
  // Function | Type | Size
  std::cout << "  " << std::left << std::setw(22) << name << std::setw(7)
            << type << std::setw(18) << size;

  // Avg | Min | Max
  std::cout << std::right << std::setw(10) << format_time(stats.avg_ns)
            << std::setw(12) << format_time(stats.min_ns) << std::setw(12)
            << format_time(stats.max_ns);

  // Throughput
  if (metrics.flop_count > 0) {
    double gflops = metrics.flop_count / stats.avg_ns;
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << gflops << " GFLOPS";
    std::cout << std::setw(15) << oss.str();
  } else if (metrics.num_elements > 0) {
    double tput =
      static_cast<double>(metrics.num_elements) / (stats.avg_ns * 1e-9);
    std::cout << std::setw(15) << format_throughput(tput);
  } else {
    std::cout << std::setw(15) << "--";
  }

  // Memory Bandwidth
  if (metrics.total_bytes > 0) {
    double bw = static_cast<double>(metrics.total_bytes) / stats.avg_ns;
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << bw << " GB/s";
    std::cout << std::setw(12) << oss.str();
  } else {
    std::cout << std::setw(12) << "--";
  }

  std::cout << std::endl;
}

/**
 * @brief Print a comparison row showing baseline vs optimized with speedup
 *
 * @param name function name
 * @param type data type
 * @param size size description
 * @param baseline baseline timing stats
 * @param optimized optimized timing stats
 */
inline void compare(const std::string &name, const std::string &type,
                    const std::string &size, const Stats &baseline,
                    const Stats &optimized) {
  double speedup =
    (optimized.avg_ns > 0) ? baseline.avg_ns / optimized.avg_ns : 0.0;

  std::cout << "  " << std::left << std::setw(22) << name << std::setw(7)
            << type << std::setw(18) << size << std::right
            << "  baseline=" << format_time(baseline.avg_ns)
            << "  optimized=" << format_time(optimized.avg_ns)
            << "  speedup=" << std::fixed << std::setprecision(2) << speedup
            << "x" << std::endl;
}

/**
 * @brief Print a separator line for grouping benchmark results
 *
 * @param title optional section title (e.g., "Element-wise Operations")
 */
inline void print_separator(const std::string &title = "") {
  std::cout << "  " << std::string(108, '=') << std::endl;
  if (!title.empty()) {
    std::cout << "  " << title << std::endl;
    std::cout << "  " << std::string(108, '-') << std::endl;
  }
}

} // namespace bench

#endif // __BENCHMARK_UTILS_H__
