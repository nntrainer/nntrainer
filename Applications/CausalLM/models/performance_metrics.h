// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    performance_metrics.h
 * @date    24 Mar 2026
 * @brief   Performance metrics definitions shared between models and API layers
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Eunju Yang <ej.yang@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#ifndef __CAUSAL_LM_PERFORMANCE_METRICS_H__
#define __CAUSAL_LM_PERFORMANCE_METRICS_H__

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Performance Metrics
 */
typedef struct {
  unsigned int prefill_tokens;
  double prefill_duration_ms;
  unsigned int generation_tokens;
  double generation_duration_ms;
  double total_duration_ms;
  double initialization_duration_ms;
  /** The honest footprint of this run: max over the run of RssAnon plus the
   *  accelerator bytes this process holds (footprint_sampler.h). Falls back to
   *  peak_rss_kb where /proc is not available. */
  size_t peak_memory_kb;
  /** getrusage ru_maxrss -- the worker's lifetime host high-water, no
   *  accelerator, never reset. Kept for debugging; it is what peak_memory_kb
   *  used to be. */
  size_t peak_rss_kb;
} TransformerPerformanceMetrics;

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <psapi.h>
#else
#include <sys/resource.h>
#endif

/**
 * @brief Get peak host RSS in KB, as the kernel's lifetime high-water.
 *
 * @note This is not the number to show a user: it excludes the accelerator
 * (kgsl on Adreno, dma-buf on the NPU, which together are the bulk of an LLM's
 * footprint) and it never comes back down, so from the second request onwards
 * it reports the worker's history rather than the request's peak. See
 * footprint_sampler.h for the per-run honest footprint.
 */
inline size_t getPeakMemoryKb() {
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS pmc;
  if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
    return (size_t)(pmc.PeakWorkingSetSize / 1024);
  }
  return 0;
#else
  struct rusage rusage;
  if (getrusage(RUSAGE_SELF, &rusage) == 0) {
    return (size_t)(rusage.ru_maxrss);
  }
  return 0;
#endif
}

#endif // __cplusplus

#endif // __CAUSAL_LM_PERFORMANCE_METRICS_H__
