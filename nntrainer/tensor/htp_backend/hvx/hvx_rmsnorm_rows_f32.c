// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 * @file   hvx_rmsnorm_rows_f32.c
 * @date   6 October 2026
 * @brief  RMSNorm over M rows of f32 on HVX, in chunks, with an optional gamma
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "hvx_rmsnorm_rows_f32.h"

#include <math.h>

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include "hvx_convert.h"

/** @brief f32 lanes per HVX vector at 128B. */
#define LANES 32u

typedef struct {
  const float *x;
  float *y;
  const float *gamma;
  uint32_t M, n, chunk;
  float eps;
} rows_ctx;

/** @brief The sum of a vector's 32 lanes: five rotate-and-add steps, then
 *         lane 0. The rotation pairs lanes in a fixed order, so the sum is
 *         the same whatever the row (no data-dependent reduction tree). */
static inline float lanes_sum(HVX_Vector v) {
  v = Q6_Vsf_vadd_VsfVsf(v, Q6_V_vror_VR(v, 64));
  v = Q6_Vsf_vadd_VsfVsf(v, Q6_V_vror_VR(v, 32));
  v = Q6_Vsf_vadd_VsfVsf(v, Q6_V_vror_VR(v, 16));
  v = Q6_Vsf_vadd_VsfVsf(v, Q6_V_vror_VR(v, 8));
  v = Q6_Vsf_vadd_VsfVsf(v, Q6_V_vror_VR(v, 4));
  const int32_t bits = Q6_R_vextract_VR(v, 0);
  float f;
  __builtin_memcpy(&f, &bits, sizeof(f));
  return f;
}

static void norm_chunk(const float *x, float *y, const float *gamma,
                       uint32_t chunk, float eps) {
  const uint32_t nvec = chunk / LANES;
  const HVX_UVector *vx = (const HVX_UVector *)x;
  HVX_UVector *vy = (HVX_UVector *)y;
  /* Sum of squares: per-lane accumulation in qf32 (one rounding per
     multiply-add), the lanes summed once at the end. */
  HVX_Vector acc = Q6_V_vzero();
  for (uint32_t i = 0; i < nvec; ++i) {
    acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(vx[i], vx[i]));
  }
  const float ss = lanes_sum(Q6_Vsf_equals_Vqf32(acc));
  const float rs = 1.0f / sqrtf(ss / (float)chunk + eps);
  const HVX_Vector r = hvx_splat_sf(rs);
  if (gamma) {
    const HVX_UVector *vg = (const HVX_UVector *)gamma;
    for (uint32_t i = 0; i < nvec; ++i) {
      vy[i] = Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vmpy_VsfVsf(vx[i], r), vg[i]);
    }
  } else {
    for (uint32_t i = 0; i < nvec; ++i) {
      vy[i] = Q6_Vsf_vmpy_VsfVsf(vx[i], r);
    }
  }
}

static void rows_worker(uint32_t n_threads, uint32_t i, void *v) {
  const rows_ctx *c = (const rows_ctx *)v;
  const uint32_t lo = (uint32_t)(((uint64_t)c->M * i) / n_threads);
  const uint32_t hi = (uint32_t)(((uint64_t)c->M * (i + 1u)) / n_threads);
  const uint32_t n_chunks = c->n / c->chunk;
  for (uint32_t r = lo; r < hi; ++r) {
    const float *x = c->x + (size_t)r * c->n;
    float *y = c->y + (size_t)r * c->n;
    for (uint32_t k = 0; k < n_chunks; ++k) {
      norm_chunk(x + (size_t)k * c->chunk, y + (size_t)k * c->chunk, c->gamma,
                 c->chunk, c->eps);
    }
  }
}

int hvx_rmsnorm_rows_f32(const float *x, float *y, uint32_t M, uint32_t n,
                         uint32_t chunk, const float *gamma, float eps,
                         hvx_worker_pool *pool) {
  if (!x || !y || M == 0u || chunk == 0u || chunk % LANES != 0u ||
      n % chunk != 0u) {
    return -1;
  }
  rows_ctx c = {x, y, gamma, M, n, chunk, eps};
  hvx_worker_pool_run(pool, rows_worker, &c, M);
  return 0;
}
