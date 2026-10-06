// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   router_rows_host_check.c
 * @date   6 October 2026
 * @brief  Host check: the real hvx_router_rows_f32.c on the lane emulation
 *         against a double reference
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "hvx_router_rows_f32.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The pool, in place of the QuRT one: min(n_units, 3) lanes run one after
   another on the caller, so the row split across lanes is exercised. */
void hvx_worker_pool_run(hvx_worker_pool *pool, hvx_worker_pool_func func,
                         void *ctx, uint32_t n_units) {
  const uint32_t n = n_units < 3u ? n_units : 3u;
  (void)pool;
  for (uint32_t i = 0; i < n; ++i) {
    func(n, i, ctx);
  }
}

static float frand(unsigned *s) {
  *s = *s * 1664525u + 1013904223u;
  return ((float)((*s >> 8) & 0xFFFF) / 65535.0f - 0.5f) * 2.0f;
}

/* worst |got - ref| / (sum |x_k w_ke| + floor): the dot cancels at places */
static double run(uint32_t M, uint32_t K, uint32_t E) {
  float *x = malloc(sizeof(float) * M * K), *w = malloc(sizeof(float) * K * E);
  float *y = malloc(sizeof(float) * M * E);
  unsigned seed = 3u + M + K + E;
  for (uint32_t i = 0; i < M * K; ++i)
    x[i] = frand(&seed) * 3.0f;
  for (uint32_t i = 0; i < K * E; ++i)
    w[i] = frand(&seed) * 0.05f;
  const int rc = hvx_router_rows_f32(x, w, y, M, K, E, NULL);
  if (rc != 0) {
    printf("router rows M=%u K=%u E=%u: rc=%d\n", M, K, E, rc);
    return 1e9;
  }
  double worst = 0.0;
  for (uint32_t r = 0; r < M; ++r) {
    for (uint32_t e = 0; e < E; ++e) {
      double ref = 0.0, mag = 0.0;
      for (uint32_t k = 0; k < K; ++k) {
        const double t = (double)x[(size_t)r * K + k] * w[(size_t)k * E + e];
        ref += t;
        mag += fabs(t);
      }
      const double d = fabs(y[(size_t)r * E + e] - ref) / (mag + 1e-6);
      if (d > worst)
        worst = d;
    }
  }
  printf("router rows M=%u K=%u E=%u: worst_rel=%g\n", M, K, E, worst);
  free(x);
  free(w);
  free(y);
  return worst;
}

int main(void) {
  int fail = 0;
  const double tol = 2e-6; /* f32 partial sums every 256 of 2816 terms */
  fail |= run(7, 2816, 128) > tol;  /* the softmax router: 7 rows, 3 lanes */
  fail |= run(5, 2048, 32) > tol;   /* the sigmoid router's width */
  fail |= run(9, 300, 64) > tol;    /* K not a multiple of the chunk */
  fail |= run(1, 256, 128) > tol;   /* one row, one chunk */
  {
    float x[32], w[32 * 32], y[32];
    memset(y, 0, sizeof y);
    const int r1 = hvx_router_rows_f32(x, w, y, 1, 32, 48, NULL);
    const int r2 = hvx_router_rows_f32(x, w, y, 1, 32, 160, NULL);
    const int r3 = hvx_router_rows_f32(x, w, y, 0, 32, 32, NULL);
    printf("rejections: E%%32=%d E>128=%d M=0:%d untouched=%d\n", r1, r2, r3,
           y[0] == 0.0f);
    fail |= !(r1 == -1 && r2 == -1 && r3 == -1 && y[0] == 0.0f);
  }
  printf(fail ? "ROUTER ROWS WRONG\n" : "ROUTER ROWS OK\n");
  return fail;
}
