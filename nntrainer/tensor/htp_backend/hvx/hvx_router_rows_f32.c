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

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include "hvx_convert.h"

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

static void rows_worker(uint32_t n_threads, uint32_t i, void *v) {
  const rows_ctx *c = (const rows_ctx *)v;
  const uint32_t lo = (uint32_t)(((uint64_t)c->M * i) / n_threads);
  const uint32_t hi = (uint32_t)(((uint64_t)c->M * (i + 1u)) / n_threads);
  const uint32_t ev = c->E / LANES;
  const HVX_Vector zero = Q6_V_vzero();
  for (uint32_t k0 = 0; k0 < c->K; k0 += KB) {
    const uint32_t k1 = k0 + KB < c->K ? k0 + KB : c->K;
    for (uint32_t r0 = lo; r0 < hi; r0 += RB) {
      const uint32_t nr = r0 + RB < hi ? RB : hi - r0;
      /* ponytail: the accumulators are indexed by two runtime counts, so
         the compiler may keep them on the stack (2 KiB); a per-E
         specialization is the upgrade if this shows in a profile. */
      HVX_Vector acc[RB][EV_MAX];
      for (uint32_t r = 0; r < nr; ++r) {
        const HVX_UVector *lr =
          (const HVX_UVector *)(c->logits + (size_t)(r0 + r) * c->E);
        for (uint32_t e = 0; e < ev; ++e)
          acc[r][e] = k0 ? Q6_Vqf32_vadd_VsfVsf(lr[e], zero) : zero;
      }
      for (uint32_t k = k0; k < k1; ++k) {
        const HVX_UVector *wk = (const HVX_UVector *)(c->w + (size_t)k * c->E);
        for (uint32_t r = 0; r < nr; ++r) {
          const HVX_Vector xs = hvx_splat_sf(c->x[(size_t)(r0 + r) * c->K + k]);
          for (uint32_t e = 0; e < ev; ++e)
            acc[r][e] = Q6_Vqf32_vadd_Vqf32Vqf32(
              acc[r][e], Q6_Vqf32_vmpy_VsfVsf(xs, wk[e]));
        }
      }
      for (uint32_t r = 0; r < nr; ++r) {
        HVX_UVector *lr = (HVX_UVector *)(c->logits + (size_t)(r0 + r) * c->E);
        for (uint32_t e = 0; e < ev; ++e)
          lr[e] = Q6_Vsf_equals_Vqf32(acc[r][e]);
      }
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
