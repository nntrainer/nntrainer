// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hvx_router_rows_f32.c
 * @date   6 October 2026
 * @brief  The MoE router's logits over M rows of f32 on HVX
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "hvx_router_rows_f32.h"

#include <stddef.h>

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include "hvx_convert.h"
#include "hvx_softmax_f32.h"

/** @brief f32 lanes per HVX vector at 128B. */
#define LANES 32u
/** @brief Vectors a row of logits spans at the widest E taken. */
#define EV_MAX 4u
/** @brief Rows of w per pass: 256 x 128 floats is 128 KiB, L2-resident. */
#define KB 256u
/** @brief Rows of x per pass: RB x EV_MAX accumulators, one w read each. */
#define RB 4u

typedef struct {
  const float *x, *w;
  float *logits;
  uint32_t M, K, E;
} rows_ctx;

/* One (rows, logit-vectors) block over w rows [k0, k1): NR x EV
 * accumulators and the EV vectors of one w row as compile-time constants,
 * so they live in registers and a w row is loaded once per k for all NR
 * rows. The runtime-indexed acc[RB][EV_MAX] this replaces spilled to the
 * stack, which made every (k, row, e) step a load, a multiply-add and a
 * store (doc 57 section 9.10): 9.7 ms a call at 446 rows against the CPU
 * dot's 6. The k terms go in order, each product and sum an IEEE sf op.
 * They were qf32 with a sf round trip at each KB chunk boundary: the host
 * lane emulation passed that, but on the V79 three rows of every four came
 * back as 1e37 or NaN, and two variants moved which row survived (doc 57
 * section 9.21). The sf ops agree with the CPU dot to 2.3e-5 on device. */
#define ROUTER_ROWS_BLOCK(EV, NR)                                              \
  static void rows_block_##EV##x##NR(const float *x, const float *w,           \
                                     float *logits, uint32_t K, uint32_t E,    \
                                     uint32_t k0, uint32_t k1) {               \
    const HVX_Vector zero = Q6_V_vzero();                                      \
    HVX_Vector acc[NR][EV];                                                    \
    for (uint32_t r = 0; r < (NR); ++r) {                                      \
      const HVX_UVector *lr = (const HVX_UVector *)(logits + (size_t)r * E);   \
      for (uint32_t e = 0; e < (EV); ++e)                                      \
        acc[r][e] = k0 ? lr[e] : zero;                                         \
    }                                                                          \
    for (uint32_t k = k0; k < k1; ++k) {                                       \
      const HVX_UVector *wk = (const HVX_UVector *)(w + (size_t)k * E);        \
      HVX_Vector wv[EV];                                                       \
      for (uint32_t e = 0; e < (EV); ++e)                                      \
        wv[e] = wk[e];                                                         \
      for (uint32_t r = 0; r < (NR); ++r) {                                    \
        const HVX_Vector xs = hvx_splat_sf(x[(size_t)r * K + k]);              \
        for (uint32_t e = 0; e < (EV); ++e)                                    \
          acc[r][e] =                                                          \
            Q6_Vsf_vadd_VsfVsf(acc[r][e], Q6_Vsf_vmpy_VsfVsf(xs, wv[e]));      \
      }                                                                        \
    }                                                                          \
    for (uint32_t r = 0; r < (NR); ++r) {                                      \
      HVX_UVector *lr = (HVX_UVector *)(logits + (size_t)r * E);               \
      for (uint32_t e = 0; e < (EV); ++e)                                      \
        lr[e] = acc[r][e];                                                     \
    }                                                                          \
  }
#define ROUTER_ROWS_EV(EV)                                                     \
  ROUTER_ROWS_BLOCK(EV, 1)                                                     \
  ROUTER_ROWS_BLOCK(EV, 2)                                                     \
  ROUTER_ROWS_BLOCK(EV, 3)                                                     \
  ROUTER_ROWS_BLOCK(EV, 4)
ROUTER_ROWS_EV(1)
ROUTER_ROWS_EV(2)
ROUTER_ROWS_EV(3)
ROUTER_ROWS_EV(4)
#undef ROUTER_ROWS_EV
#undef ROUTER_ROWS_BLOCK

typedef void (*rows_block_fn)(const float *, const float *, float *, uint32_t,
                              uint32_t, uint32_t, uint32_t);
/** [ev - 1][nr - 1] */
static const rows_block_fn kRowsBlock[EV_MAX][RB] = {
  {rows_block_1x1, rows_block_1x2, rows_block_1x3, rows_block_1x4},
  {rows_block_2x1, rows_block_2x2, rows_block_2x3, rows_block_2x4},
  {rows_block_3x1, rows_block_3x2, rows_block_3x3, rows_block_3x4},
  {rows_block_4x1, rows_block_4x2, rows_block_4x3, rows_block_4x4},
};

static void rows_worker(uint32_t n_threads, uint32_t i, void *v) {
  const rows_ctx *c = (const rows_ctx *)v;
  const uint32_t lo = (uint32_t)(((uint64_t)c->M * i) / n_threads);
  const uint32_t hi = (uint32_t)(((uint64_t)c->M * (i + 1u)) / n_threads);
  const uint32_t ev = c->E / LANES;
  for (uint32_t k0 = 0; k0 < c->K; k0 += KB) {
    const uint32_t k1 = k0 + KB < c->K ? k0 + KB : c->K;
    for (uint32_t r0 = lo; r0 < hi; r0 += RB) {
      const uint32_t nr = r0 + RB < hi ? RB : hi - r0;
      kRowsBlock[ev - 1u][nr - 1u](c->x + (size_t)r0 * c->K, c->w,
                                   c->logits + (size_t)r0 * c->E, c->K, c->E,
                                   k0, k1);
    }
  }
}

int hvx_router_rows_f32(const float *x, const float *w, float *logits,
                        uint32_t M, uint32_t K, uint32_t E,
                        hvx_worker_pool *pool) {
  if (!x || !w || !logits || M == 0u || K == 0u || E == 0u ||
      E % LANES != 0u || E > EV_MAX * LANES) {
    return -1;
  }
  rows_ctx c = {x, w, logits, M, K, E};
  hvx_worker_pool_run(pool, rows_worker, &c, M);
  return 0;
}

typedef struct {
  float *p;
  const float *scale;
  uint32_t *sel;
  float *weight;
  uint32_t M, E, top_k, n_sel;
} topk_ctx;

static void topk_worker(uint32_t n_threads, uint32_t i, void *v) {
  const topk_ctx *c = (const topk_ctx *)v;
  const uint32_t lo = (uint32_t)(((uint64_t)c->M * i) / n_threads);
  const uint32_t hi = (uint32_t)(((uint64_t)c->M * (i + 1u)) / n_threads);
  if (lo >= hi)
    return;
  hvx_softmax_rows_f32(c->p, c->p, lo, hi, c->E, 1.0f);
  for (uint32_t r = lo; r < hi; ++r) {
    const float *pr = c->p + (size_t)r * c->E;
    uint32_t *sr = c->sel + (size_t)r * c->n_sel;
    uint32_t taken[4] = {0u, 0u, 0u, 0u}; /* E <= 128 */
    /* n_sel passes of a first-maximum scan: strict > in index order is
       the CPU comparator's tie rule (the lower index wins) */
    for (uint32_t k = 0; k < c->n_sel; ++k) {
      uint32_t best = c->E;
      for (uint32_t e = 0; e < c->E; ++e) {
        if (taken[e >> 5] & (1u << (e & 31u)))
          continue;
        if (best == c->E || pr[e] > pr[best])
          best = e;
      }
      taken[best >> 5] |= 1u << (best & 31u);
      sr[k] = best;
    }
    float wsum = 0.0f;
    for (uint32_t k = 0; k < c->top_k; ++k)
      wsum += pr[sr[k]];
    const float inv = 1.0f / wsum;
    float *wr = c->weight + (size_t)r * c->top_k;
    for (uint32_t k = 0; k < c->top_k; ++k)
      wr[k] = pr[sr[k]] * inv * c->scale[sr[k]];
  }
}

int hvx_router_topk_rows_f32(float *p, const float *scale, uint32_t *sel,
                             float *weight, uint32_t M, uint32_t E,
                             uint32_t top_k, uint32_t n_sel,
                             hvx_worker_pool *pool) {
  if (!p || !scale || !sel || !weight || M == 0u || E == 0u ||
      E % LANES != 0u || E > EV_MAX * LANES || top_k == 0u || n_sel < top_k ||
      n_sel > E) {
    return -1;
  }
  topk_ctx c = {p, scale, sel, weight, M, E, top_k, n_sel};
  hvx_worker_pool_run(pool, topk_worker, &c, M);
  return 0;
}
