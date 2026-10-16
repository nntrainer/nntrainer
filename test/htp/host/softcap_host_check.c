// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   softcap_host_check.c
 * @date   7 October 2026
 * @brief  Host check: the real hvx_softcap_f32.c on the lane emulation
 *         against cap * tanh(x / cap) in double
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "hvx_softcap_f32.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
  /* logits over the range a 262144-wide vocabulary spans, cap 30 */
  const uint32_t n = 262144u + 7u; /* a scalar tail too */
  const float cap = 30.0f;
  float *y = malloc(sizeof(float) * n), *x = malloc(sizeof(float) * n);
  unsigned s = 1u;
  for (uint32_t i = 0; i < n; ++i) {
    s = s * 1664525u + 1013904223u;
    x[i] = ((float)((s >> 8) & 0xFFFF) / 65535.0f - 0.5f) * 400.0f;
    y[i] = x[i];
  }
  y[0] = x[0] = 0.0f;
  y[1] = x[1] = -0.0f;
  y[2] = x[2] = 1e-6f;
  int fail = hvx_softcap_f32(y, n, cap) != 0;
  double worst = 0.0;
  for (uint32_t i = 0; i < n; ++i) {
    const double ref = cap * tanh((double)x[i] / cap);
    const double d = fabs(y[i] - ref) / cap; /* against the output's range */
    if (d > worst)
      worst = d;
  }
  float z[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  const int r = hvx_softcap_f32(z, 4, 0.0f);
  printf("softcap n=%u cap=%g: worst |err|/cap=%g; cap 0 rejected=%d "
         "untouched=%d\n",
         n, cap, worst, r == -1, z[0] == 1.0f);
  fail |= worst > 1e-5 || r != -1 || z[0] != 1.0f;
  printf(fail ? "SOFTCAP WRONG\n" : "SOFTCAP OK\n");
  free(x);
  free(y);
  return fail;
}
