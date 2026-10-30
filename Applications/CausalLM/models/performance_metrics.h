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
  /** Prompt tokens DROPPED from the tail of the last request because the
   *  prompt did not fit the prefill window (0 = the whole prompt was fed). A
   *  truncated run still returns success and fluent text, so this is the only
   *  way a caller can tell that the model never saw the end of its prompt. */
  unsigned int prompt_truncated_tokens;
  /** ---- [perf-split] honest prefill/decode split -----------------------
   * The two duration fields above take the prefill->decode boundary where the
   * HOST loop leaves prefill, which on an asynchronous backend is BEFORE the
   * GPU has retired the work prefill submitted. A larger prefill block leaves
   * more queued, so prefill_duration_ms is short by exactly what the first
   * decode step then blocks on: "prefill TPS" reads high and
   * generation_duration_ms/generation_tokens reads low, by the same amount.
   * Measured, 0.3B/16383 tokens, same build: block 4096 -> prefill 3671 ms +
   * first token 1.90 s; block 1024 -> prefill 4989 ms + first token 0.53 s.
   * Sum 5.55 s either way, and 128-token end-to-end 7.68 vs 7.63 s.
   *
   * The fields below name the boundary instead of guessing at it:
   *  - prefill_duration_ms is corrected IN PLACE (queue drained at the
   *    boundary) only when the measurement drain is armed
   *    (NNTR_PERF_SPLIT_DRAIN=1, which only a measurement run sets);
   *    prefill_drain_ms says how much of it the drain accounted for, and is
   *    0.0 when the drain was not armed, i.e. on the production path.
   *  - ttft_ms and the per-step numbers are ALWAYS filled: they are host
   *    timestamps taken around the streaming callback and cost nothing.
   * Totals are what to compare across arms: ttft_ms and total_duration_ms are
   * drain-invariant by construction, the split is not.
   */
  /** Prompt submit (run() entry; the model is already loaded, that is
   *  initialization_duration_ms) -> the first token handed to the caller.
   *  Its OWN row: it is not prefill_duration_ms plus anything. */
  double ttft_ms;
  /** The first decode step alone: prefill boundary -> first token out. */
  double first_token_ms;
  /** Median ms/token over tokens 2..N -- the steady state, with the first
   *  token (which absorbs whatever prefill left in flight) excluded. 0.0 when
   *  fewer than 2 tokens were generated. */
  double decode_steady_ms;
  /** How many steps decode_steady_ms is a median over (generation_tokens-1,
   *  0 when generation_tokens < 2). */
  unsigned int decode_steady_tokens;
  /** Wall time the boundary drain blocked for = the GPU work prefill had left
   *  in flight. 0.0 when the drain was not armed, in which case that much time
   *  is still hiding inside first_token_ms. */
  double prefill_drain_ms;
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
