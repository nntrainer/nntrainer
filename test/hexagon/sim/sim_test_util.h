// SPDX-License-Identifier: Apache-2.0
/**
 * @file	sim_test_util.h
 * @date	18 August 2026
 * @brief	Shared deterministic PRNG and float comparison for hexagon-sim tests
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#ifndef SIM_TEST_UTIL_H
#define SIM_TEST_UTIL_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>

static uint32_t rng_state = 12345u;
static inline float frand(void) { /* [-1, 1) */
  rng_state = rng_state * 1664525u + 1013904223u;
  return (((rng_state >> 8) & 0xFFFF) / 32768.0f) - 1.0f;
}
static inline int cmp_f(const char *tag, const float *ref, const float *got,
                        uint32_t n, float rtol, float atol) {
  float worst = 0.f;
  uint32_t wi = 0;
  float max_abs = 0.f, max_rel = 0.f;
  for (uint32_t i = 0; i < n; ++i) {
    float d = fabsf(ref[i] - got[i]), t = atol + rtol * fabsf(ref[i]);
    if (d - t > worst) {
      worst = d - t;
      wi = i;
    }
    if (d > max_abs)
      max_abs = d;
    if (fabsf(ref[i]) > 0.f) {
      float rel = d / fabsf(ref[i]);
      if (rel > max_rel)
        max_rel = rel;
    }
  }
  printf("SIM_TEST %s STAT max_abs=%g max_rel=%g\n", tag, (double)max_abs,
         (double)max_rel);
  if (worst > 0.f) {
    printf("SIM_TEST %s FAIL i=%u ref=%f got=%f\n", tag, (unsigned)wi, ref[wi],
           got[wi]);
    return 1;
  }
  return 0;
}

#endif
