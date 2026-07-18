// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   layer_prof.h
 * @date   18 July 2026
 * @brief  Tiny env-gated per-layer wall-clock profiler for prefill/decode
 *         breakdown. Activated via NNTR_LAYER_PROFILE=1. Dumps an aggregated
 *         table at process exit. Promoted from
 *         Applications/CausalLM/layers/_layer_prof.h to core.
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>

namespace nntrainer {

/**
 * @brief Per-layer accumulated wall time and call count, split into the
 *        prefill (M>1) and decode (M==1) regimes.
 */
struct LayerProfAgg {
  long long ns_pre = 0; // M > 1
  long long ns_dec = 0; // M == 1
  long long calls_pre = 0;
  long long calls_dec = 0;
};

inline std::unordered_map<std::string, LayerProfAgg> &layer_prof_table() {
  static std::unordered_map<std::string, LayerProfAgg> t;
  return t;
}

inline std::mutex &layer_prof_mtx() {
  static std::mutex m;
  return m;
}

inline bool layer_prof_enabled() {
  static const bool on = std::getenv("NNTR_LAYER_PROFILE") != nullptr;
  return on;
}

/**
 * @brief Process-exit hook that prints the aggregated per-layer profile
 *        table to stderr when NNTR_LAYER_PROFILE is set.
 */
struct LayerProfDumper {
  ~LayerProfDumper() {
    if (!layer_prof_enabled())
      return;
    std::lock_guard<std::mutex> lock(layer_prof_mtx());
    auto &t = layer_prof_table();
    std::fprintf(stderr,
                 "\n[LAYER-PROF]  prefill (M>1) / decode (M=1) per layer\n");
    std::fprintf(stderr, "%-28s  %10s %10s     %10s %10s\n", "layer", "pre_ms",
                 "pre_calls", "dec_ms", "dec_calls");
    long long sum_pre = 0, sum_dec = 0;
    for (auto &kv : t) {
      double pre_ms = kv.second.ns_pre / 1.0e6;
      double dec_ms = kv.second.ns_dec / 1.0e6;
      sum_pre += kv.second.ns_pre;
      sum_dec += kv.second.ns_dec;
      std::fprintf(stderr, "%-28s  %10.2f %10lld     %10.2f %10lld\n",
                   kv.first.c_str(), pre_ms, kv.second.calls_pre, dec_ms,
                   kv.second.calls_dec);
    }
    std::fprintf(stderr, "%-28s  %10.2f                  %10.2f\n",
                 "(sum non-FC)", sum_pre / 1.0e6, sum_dec / 1.0e6);
  }
};

inline LayerProfDumper &layer_prof_dumper() {
  static LayerProfDumper d;
  return d;
}

/**
 * @brief RAII scope timer: accumulates its lifetime into the named layer's
 *        LayerProfAgg entry on destruction.
 */
struct LayerProfScope {
  const char *name;
  bool is_decode;
  std::chrono::steady_clock::time_point t0;
  LayerProfScope(const char *n, bool dec) : name(n), is_decode(dec) {
    if (layer_prof_enabled()) {
      // Force construction order so the dumper (which uses mutex+table in
      // its destructor) is constructed AFTER mutex+table, hence destroyed
      // BEFORE them at process exit. Otherwise the dtor would touch an
      // already-destroyed mutex (FORTIFY: pthread_mutex_lock on dead mtx).
      (void)layer_prof_mtx();
      (void)layer_prof_table();
      (void)layer_prof_dumper();
      t0 = std::chrono::steady_clock::now();
    }
  }
  ~LayerProfScope() {
    if (!layer_prof_enabled())
      return;
    auto t1 = std::chrono::steady_clock::now();
    long long ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    std::lock_guard<std::mutex> lock(layer_prof_mtx());
    auto &agg = layer_prof_table()[name];
    if (is_decode) {
      agg.ns_dec += ns;
      ++agg.calls_dec;
    } else {
      agg.ns_pre += ns;
      ++agg.calls_pre;
    }
  }
};

} // namespace nntrainer
