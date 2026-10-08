// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hvx_rope_rows_f32.c
 * @date   6 October 2026
 * @brief  RoPE over M rows of heads of f32 on HVX, in place
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "hvx_rope_rows_f32.h"

#include <stddef.h>

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

/** @brief f32 lanes per HVX vector at 128B. */
#define LANES 32u

typedef struct {
  float *x;
  const float *cs;
  uint32_t M, n, ld, hd;
} rows_ctx;

static void rows_worker(uint32_t n_threads, uint32_t i, void *v) {
  const rows_ctx *c = (const rows_ctx *)v;
  const uint32_t lo = (uint32_t)(((uint64_t)c->M * i) / n_threads);
  const uint32_t hi = (uint32_t)(((uint64_t)c->M * (i + 1u)) / n_threads);
  const uint32_t half = c->hd / 2u;
  const uint32_t nvec = half / LANES;
  const uint32_t n_heads = c->n / c->hd;
  for (uint32_t r = lo; r < hi; ++r) {
    const HVX_UVector *vc = (const HVX_UVector *)(c->cs + (size_t)r * 2u * c->hd);
    const HVX_UVector *vs =
      (const HVX_UVector *)(c->cs + (size_t)r * 2u * c->hd + c->hd);
    for (uint32_t h = 0; h < n_heads; ++h) {
      float *head = c->x + (size_t)r * c->ld + (size_t)h * c->hd;
      HVX_UVector *va = (HVX_UVector *)head;
      HVX_UVector *vb = (HVX_UVector *)(head + half);
      for (uint32_t k = 0; k < nvec; ++k) {
        const HVX_Vector a = va[k], b = vb[k];
        /* a*cos - b*sin and a*sin + b*cos, each product rounded once as
           the CPU's (no fused multiply-add there either) */
        va[k] = Q6_Vsf_vsub_VsfVsf(Q6_Vsf_vmpy_VsfVsf(a, vc[k]),
                                   Q6_Vsf_vmpy_VsfVsf(b, vs[k]));
        vb[k] = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(a, vs[k]),
                                   Q6_Vsf_vmpy_VsfVsf(b, vc[k]));
      }
    }
  }
}

int hvx_rope_rows_f32(float *x, uint32_t M, uint32_t n, uint32_t hd,
                      const float *cs, hvx_worker_pool *pool) {
  return hvx_rope_rows_ld_f32(x, M, n, n, hd, cs, pool);
}

int hvx_rope_rows_ld_f32(float *x, uint32_t M, uint32_t n, uint32_t ld,
                         uint32_t hd, const float *cs, hvx_worker_pool *pool) {
  if (!x || !cs || M == 0u || hd == 0u || (hd / 2u) % LANES != 0u || n == 0u ||
      n % hd != 0u || ld < n) {
    return -1;
  }
  rows_ctx c = {x, cs, M, n, ld, hd};
  hvx_worker_pool_run(pool, rows_worker, &c, M);
  return 0;
}
