// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 * @file   rmsnorm_rows_host_check.c
 * @date   6 October 2026
 * @brief  Host check: the real hvx_rmsnorm_rows_f32.c on the lane emulation
 *         against a double reference, at the hidden width and per head
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * What it proves: the chunk and row indexing, the lane reduction, gamma
 * and the gamma-less form, in-place operation, and the shape rejections.
 * What it rests on: one Vsf op = one IEEE op on the device (rule 24).
 */

#include "hvx_rmsnorm_rows_f32.h"

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
  *s = *s * 1103515245u + 12345u;
  return ((float)((*s >> 8) & 0xFFFF) / 65535.0f - 0.5f) * 4.0f;
}

/* worst |got - ref| / (|ref| + floor) over one shape */
static double run(uint32_t M, uint32_t n, uint32_t chunk, int with_gamma,
                  int in_place, float eps) {
  float *x = malloc(sizeof(float) * M * n), *y = malloc(sizeof(float) * M * n);
  float *g = malloc(sizeof(float) * chunk);
  unsigned seed = 7u + M + n + chunk;
  for (uint32_t i = 0; i < M * n; ++i)
    x[i] = frand(&seed) * (1.0f + (float)(i % 7));
  for (uint32_t j = 0; j < chunk; ++j)
    g[j] = 1.0f + 0.25f * frand(&seed);
  float *src = x, *dst = y;
  float *xcopy = malloc(sizeof(float) * M * n);
  memcpy(xcopy, x, sizeof(float) * M * n);
  if (in_place)
    dst = x;
  const int rc = hvx_rmsnorm_rows_f32(src, dst, M, n, chunk, with_gamma ? g : NULL,
                                      eps, NULL);
  if (rc != 0) {
    printf("rmsnorm rows M=%u n=%u chunk=%u: rc=%d\n", M, n, chunk, rc);
    return 1e9;
  }
  double worst = 0.0;
  for (uint32_t r = 0; r < M; ++r) {
    for (uint32_t k = 0; k < n / chunk; ++k) {
      const float *xr = xcopy + (size_t)r * n + (size_t)k * chunk;
      double ss = 0.0;
      for (uint32_t j = 0; j < chunk; ++j)
        ss += (double)xr[j] * xr[j];
      const double rs = 1.0 / sqrt(ss / chunk + eps);
      for (uint32_t j = 0; j < chunk; ++j) {
        const double ref = xr[j] * rs * (with_gamma ? g[j] : 1.0);
        const double got = dst[(size_t)r * n + (size_t)k * chunk + j];
        const double d = fabs(got - ref) / (fabs(ref) + 1e-6);
        if (d > worst)
          worst = d;
      }
    }
  }
  printf("rmsnorm rows M=%u n=%u chunk=%u gamma=%d in_place=%d: worst_rel=%g\n",
         M, n, chunk, with_gamma, in_place, worst);
  free(x);
  free(y);
  free(g);
  free(xcopy);
  return worst;
}

/* The epilogue against a double reference: worst relative error. */
static double run_add(uint32_t M, uint32_t n, int two, int with_gamma,
                      float scale) {
  const size_t len = (size_t)M * n;
  float *out = malloc(sizeof(float) * len), *x = malloc(sizeof(float) * len);
  float *x2 = malloc(sizeof(float) * len), *g = malloc(sizeof(float) * n);
  float *resid = malloc(sizeof(float) * len);
  unsigned seed = 11u + M + n;
  for (size_t i = 0; i < len; ++i) {
    x[i] = frand(&seed) * (1.0f + (float)(i % 5));
    x2[i] = frand(&seed);
    resid[i] = frand(&seed) * 3.0f;
    out[i] = resid[i];
  }
  for (uint32_t j = 0; j < n; ++j)
    g[j] = 1.0f + 0.25f * frand(&seed);
  const int rc = hvx_rmsnorm_add_f32(out, x, two ? x2 : NULL, M, n,
                                     with_gamma ? g : NULL, 1e-6f, scale, NULL);
  if (rc != 0) {
    printf("rmsnorm add M=%u n=%u: rc=%d\n", M, n, rc);
    return 1e9;
  }
  double worst = 0.0;
  for (uint32_t r = 0; r < M; ++r) {
    double ss = 0.0;
    for (uint32_t j = 0; j < n; ++j) {
      const double a = x[(size_t)r * n + j] + (two ? x2[(size_t)r * n + j] : 0);
      ss += a * a;
    }
    const double rs = 1.0 / sqrt(ss / n + 1e-6);
    for (uint32_t j = 0; j < n; ++j) {
      const size_t i = (size_t)r * n + j;
      const double a = x[i] + (two ? x2[i] : 0);
      const double t = a * rs * (with_gamma ? g[j] : 1.0);
      const double ref = scale * (resid[i] + t);
      /* relative to the terms, not the sum: the add cancels at places */
      const double d = fabs(out[i] - ref) / (fabs(resid[i]) + fabs(t) + 1e-6);
      if (d > worst)
        worst = d;
    }
  }
  printf("rmsnorm add M=%u n=%u two=%d gamma=%d scale=%g: worst_rel=%g\n", M, n,
         two, with_gamma, scale, worst);
  free(out);
  free(x);
  free(x2);
  free(g);
  free(resid);
  return worst;
}

int main(void) {
  int fail = 0;
  const double tol = 2e-6; /* f32 sum of squares over 2816 terms, sqrtf */
  fail |= run(7, 2816, 2816, 1, 0, 1e-6f) > tol;  /* the hidden norm */
  fail |= run(5, 2816, 2816, 1, 1, 1e-6f) > tol;  /* in place */
  fail |= run(3, 4096, 256, 1, 0, 1e-6f) > tol;   /* per head, 16 x 256 */
  fail |= run(3, 1024, 512, 1, 1, 1e-6f) > tol;   /* per head, 2 x 512 */
  fail |= run(4, 2048, 256, 0, 0, 1e-6f) > tol;   /* gamma-less */
  fail |= run(1, 64, 32, 1, 0, 1e-6f) > tol;      /* one vector a chunk */
  /* the epilogue: post norm + residual (+ second addend, scalar) */
  fail |= run_add(7, 2816, 0, 1, 1.0f) > tol;  /* post_attention_norm */
  fail |= run_add(5, 2816, 1, 1, 0.75f) > tol; /* ffn + moe, layer scalar */
  fail |= run_add(2, 64, 1, 0, 2.0f) > tol;    /* tiny, gamma-less */
  {
    float out[32], x[32];
    memset(out, 0, sizeof out);
    const int r =
      hvx_rmsnorm_add_f32(out, x, NULL, 1, 48, NULL, 1e-6f, 1.0f, NULL);
    printf("add rejection: n%%32=%d untouched=%d\n", r, out[0] == 0.0f);
    fail |= !(r == -1 && out[0] == 0.0f);
  }
  /* rejections: nothing written */
  {
    float x[64], y[64];
    memset(y, 0, sizeof y);
    for (int i = 0; i < 64; ++i)
      x[i] = (float)i;
    int r1 = hvx_rmsnorm_rows_f32(x, y, 1, 64, 48, NULL, 1e-6f, NULL);
    int r2 = hvx_rmsnorm_rows_f32(x, y, 1, 64, 128, NULL, 1e-6f, NULL);
    int r3 = hvx_rmsnorm_rows_f32(x, y, 0, 64, 32, NULL, 1e-6f, NULL);
    int untouched = 1;
    for (int i = 0; i < 64; ++i)
      untouched &= y[i] == 0.0f;
    printf("rejections: chunk%%32=%d chunk>n=%d M=0:%d untouched=%d\n", r1, r2,
           r3, untouched);
    fail |= !(r1 == -1 && r2 == -1 && r3 == -1 && untouched);
  }
  printf(fail ? "RMSNORM ROWS WRONG\n" : "RMSNORM ROWS OK\n");
  return fail;
}
