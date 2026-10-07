// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   rope_rows_host_check.c
 * @date   6 October 2026
 * @brief  Host check: the real hvx_rope_rows_f32.c on the lane emulation
 *         against the CPU kernel's arithmetic, bit for bit
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "hvx_rope_rows_f32.h"

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
  return ((float)((*s >> 8) & 0xFFFF) / 65535.0f - 0.5f) * 4.0f;
}

/* The CPU table's rows (calc_trigonometric_vals_dup): cos and sin of
   pos * theta_j for j < half, each half duplicated; theta_j = 0 past
   rot/2 for a partial rotary (cos 1, sin 0: the pair is left alone). */
static void table_row(float *cs, uint32_t hd, uint32_t rot, float theta,
                      unsigned pos) {
  const uint32_t half = hd / 2u;
  for (uint32_t j = 0; j < half; ++j) {
    const float t =
      j < rot / 2u ? 1.0f / powf(theta, (2.0f * j) / (float)hd) : 0.0f;
    const float a = (float)pos * t;
    cs[j] = cs[j + half] = cosf(a);
    cs[hd + j] = cs[hd + j + half] = sinf(a);
  }
}

/* mismatches against the CPU kernel's order: a*c - b*s, a*s + b*c */
static int run(uint32_t M, uint32_t n, uint32_t hd, uint32_t rot, unsigned from) {
  const size_t len = (size_t)M * n;
  float *x = malloc(sizeof(float) * len), *ref = malloc(sizeof(float) * len);
  float *cs = malloc(sizeof(float) * M * 2u * hd);
  unsigned seed = 5u + M + n + hd;
  for (size_t i = 0; i < len; ++i)
    x[i] = frand(&seed);
  memcpy(ref, x, sizeof(float) * len);
  for (uint32_t r = 0; r < M; ++r)
    table_row(cs + (size_t)r * 2u * hd, hd, rot, 10000.0f, from + r);
  const int rc = hvx_rope_rows_f32(x, M, n, hd, cs, NULL);
  if (rc != 0) {
    printf("rope rows M=%u n=%u hd=%u: rc=%d\n", M, n, hd, rc);
    return 1;
  }
  const uint32_t half = hd / 2u;
  int bad = 0;
  for (uint32_t r = 0; r < M; ++r) {
    const float *c = cs + (size_t)r * 2u * hd, *s = c + hd;
    for (uint32_t h = 0; h < n / hd; ++h) {
      float *head = ref + (size_t)r * n + (size_t)h * hd;
      for (uint32_t j = 0; j < half; ++j) {
        const float a = head[j], b = head[j + half];
        const float o0 = a * c[j] - b * s[j];
        const float o1 = a * s[j] + b * c[j];
        head[j] = o0;
        head[j + half] = o1;
      }
    }
  }
  for (size_t i = 0; i < len; ++i)
    bad += x[i] != ref[i];
  /* the untouched pairs of a partial rotary stay exactly what they were */
  printf("rope rows M=%u n=%u hd=%u rot=%u from=%u: mismatches=%d\n", M, n, hd,
         rot, from, bad);
  free(x);
  free(ref);
  free(cs);
  return bad != 0;
}

int main(void) {
  int fail = 0;
  fail |= run(7, 4096, 256, 256, 0);   /* sliding: 16 heads of 256 */
  fail |= run(5, 2048, 256, 256, 500); /* k: 8 heads, later positions */
  fail |= run(4, 1024, 512, 128, 3);   /* full: hd 512, partial 0.25 */
  fail |= run(3, 192, 64, 64, 1);      /* the decode graph's head dim */
  {
    float x[64], cs[128];
    memset(cs, 0, sizeof cs);
    for (int i = 0; i < 64; ++i)
      x[i] = (float)i;
    const int r1 = hvx_rope_rows_f32(x, 1, 64, 48, cs, NULL);
    const int r2 = hvx_rope_rows_f32(x, 1, 96, 64, cs, NULL);
    const int r3 = hvx_rope_rows_f32(x, 0, 64, 64, cs, NULL);
    printf("rejections: half%%32=%d n%%hd=%d M=0:%d untouched=%d\n", r1, r2,
           r3, x[5] == 5.0f);
    fail |= !(r1 == -1 && r2 == -1 && r3 == -1 && x[5] == 5.0f);
  }
  printf(fail ? "ROPE ROWS WRONG\n" : "ROPE ROWS OK\n");
  return fail;
}
