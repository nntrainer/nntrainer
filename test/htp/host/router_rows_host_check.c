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

#include "hvx_rmsnorm_rows_f32.h"
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

/* The selection against the CPU layer's own (std::exp softmax in float,
   partial_sort with the lower index winning a tie): the same experts in
   the same order, the weights within tol of the CPU's. */
static int run_topk(uint32_t M, uint32_t E, uint32_t top_k, uint32_t n_sel,
                    int ties) {
  float *p = malloc(sizeof(float) * M * E), *lg = malloc(sizeof(float) * M * E);
  float *scale = malloc(sizeof(float) * E);
  uint32_t *sel = malloc(sizeof(uint32_t) * M * n_sel);
  float *w = malloc(sizeof(float) * M * top_k);
  unsigned seed = 17u + M + E;
  for (uint32_t i = 0; i < M * E; ++i)
    lg[i] = ties ? (float)((int)(frand(&seed) * 3.0f)) : frand(&seed) * 6.0f;
  for (uint32_t e = 0; e < E; ++e)
    scale[e] = 0.5f + 0.5f * (frand(&seed) + 1.0f);
  memcpy(p, lg, sizeof(float) * M * E);
  if (hvx_router_topk_rows_f32(p, scale, sel, w, M, E, top_k, n_sel, NULL)) {
    printf("router topk M=%u E=%u: rejected\n", M, E);
    return 1;
  }
  int bad_sel = 0;
  double worst = 0.0;
  float *pr = malloc(sizeof(float) * E);
  uint32_t *idx = malloc(sizeof(uint32_t) * E);
  for (uint32_t r = 0; r < M; ++r) {
    const float *l = lg + (size_t)r * E;
    float mx = l[0];
    for (uint32_t e = 1; e < E; ++e)
      mx = l[e] > mx ? l[e] : mx;
    float sum = 0.0f;
    for (uint32_t e = 0; e < E; ++e)
      sum += pr[e] = expf(l[e] - mx);
    for (uint32_t e = 0; e < E; ++e) {
      pr[e] /= sum;
      idx[e] = e;
    }
    for (uint32_t k = 0; k < n_sel; ++k) /* selection sort, CPU comparator */
      for (uint32_t j = k + 1; j < E; ++j)
        if (pr[idx[j]] > pr[idx[k]] ||
            (pr[idx[j]] == pr[idx[k]] && idx[j] < idx[k])) {
          const uint32_t t = idx[k];
          idx[k] = idx[j];
          idx[j] = t;
        }
    float wsum = 0.0f;
    for (uint32_t k = 0; k < top_k; ++k)
      wsum += pr[idx[k]];
    for (uint32_t k = 0; k < n_sel; ++k)
      bad_sel += sel[(size_t)r * n_sel + k] != idx[k];
    for (uint32_t k = 0; k < top_k; ++k) {
      const double ref = pr[idx[k]] * (1.0f / wsum) * scale[idx[k]];
      const double d = fabs(w[(size_t)r * top_k + k] - ref) / fabs(ref);
      if (d > worst)
        worst = d;
    }
  }
  printf("router topk M=%u E=%u top_k=%u n_sel=%u ties=%d: sel mismatches=%d "
         "weight worst_rel=%g\n",
         M, E, top_k, n_sel, ties, bad_sel, worst);
  free(p);
  free(lg);
  free(scale);
  free(sel);
  free(w);
  free(pr);
  free(idx);
  return bad_sel != 0 || worst > 1e-5;
}

/* The fused norm against the two passes it replaces (hvx_rmsnorm_rows_f32
   then hvx_router_rows_f32), bit for bit; x has a few outliers, the way
   Gemma-4's activations do. */
static int run_norm(uint32_t M, uint32_t K, uint32_t E) {
  float *x = malloc(sizeof(float) * M * K), *n = malloc(sizeof(float) * M * K);
  float *w = malloc(sizeof(float) * K * E), *g = malloc(sizeof(float) * K);
  float *y1 = malloc(sizeof(float) * M * E),
        *y2 = malloc(sizeof(float) * M * E);
  unsigned seed = 11u + M + K + E;
  for (uint32_t i = 0; i < M * K; ++i)
    x[i] = frand(&seed) * ((i % 97u) == 5u ? 300.0f : 2.0f);
  for (uint32_t i = 0; i < K * E; ++i)
    w[i] = frand(&seed) * 0.05f;
  for (uint32_t i = 0; i < K; ++i)
    g[i] = 1.0f + frand(&seed) * 0.5f;
  const float eps = 1e-6f;
  const int r0 = hvx_rmsnorm_rows_f32(x, n, M, K, K, g, eps, NULL);
  const int r1 = hvx_router_rows_f32(n, w, y1, M, K, E, NULL);
  const int r2 = hvx_router_rows_norm_f32(x, g, eps, w, y2, M, K, E, NULL);
  const int same =
    r0 == 0 && r1 == 0 && r2 == 0 && memcmp(y1, y2, sizeof(float) * M * E) == 0;
  printf("router rows fused norm M=%u K=%u E=%u: rc=%d/%d/%d %s\n", M, K, E, r0,
         r1, r2, same ? "bit-identical" : "DIFFERENT");
  free(x), free(n), free(w), free(g), free(y1), free(y2);
  return !same;
}

int main(void) {
  int fail = 0;
  const double tol = 2e-6; /* f32 partial sums every 256 of 2816 terms */
  fail |= run(7, 2816, 128) > tol; /* the softmax router: 7 rows, 3 lanes */
  fail |= run(5, 2048, 32) > tol;  /* the sigmoid router's width */
  fail |= run(9, 300, 64) > tol;   /* K not a multiple of the chunk */
  fail |= run(1, 256, 128) > tol;  /* one row, one chunk */
  fail |= run_norm(900, 320, 128); /* 300 rows a lane: two segments */
  fail |= run_norm(7, 2816, 128);  /* the Gemma-4 router */
  fail |= run_norm(5, 2048, 32);
  /* the softmax router's selection: top-8 plus the 5 prefetch hints */
  fail |= run_topk(7, 128, 8, 13, 0);
  fail |= run_topk(5, 128, 8, 8, 1); /* integer logits: exact ties */
  fail |= run_topk(3, 32, 4, 6, 0);
  {
    float p[32], sc[32], w[4];
    uint32_t sl[4];
    const int r1 = hvx_router_topk_rows_f32(p, sc, sl, w, 1, 32, 4, 3, NULL);
    const int r2 = hvx_router_topk_rows_f32(p, sc, sl, w, 1, 48, 4, 4, NULL);
    printf("topk rejections: n_sel<top_k=%d E%%32=%d\n", r1, r2);
    fail |= !(r1 == -1 && r2 == -1);
  }
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
