// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   HAP_perf.h
 * @date   27 Sep 2026
 * @brief  Host stand-in for the SDK's HAP_perf.h, whose qtimer and pcycle
 *         readers are Hexagon register reads (in-process HTP build)
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * Monotonic host nanoseconds stand in for both counters, so a stage's
 * timer pair still brackets the stage and a "pcycles" bracket of an op
 * that ran reads > 0. None of these numbers means anything about the
 * device; the profile's dsp / transport columns on the host are noise.
 */
#pragma once
#include <stdint.h>
#include <time.h>

static inline uint64_t hap_perf_host_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}
/** @brief The 19.2 MHz qtimer: one host nanosecond per tick here. */
static inline uint64_t HAP_perf_get_qtimer_count(void) {
  return hap_perf_host_ns();
}
static inline uint64_t HAP_perf_qtimer_count_to_us(uint64_t count) {
  return count / 1000u;
}
static inline uint64_t HAP_perf_get_pcycles(void) { return hap_perf_host_ns(); }
static inline uint64_t HAP_perf_get_time_us(void) {
  return hap_perf_host_ns() / 1000u;
}
