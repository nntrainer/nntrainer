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

#include <stddef.h>

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

typedef struct {
  float *out;
  const float *res; /**< the residual, or NULL when it is out itself */
  const float *x, *x2, *gamma;
  uint32_t M, n;
  float eps, scale;
} add_ctx;

static void add_worker(uint32_t n_threads, uint32_t i, void *v) {
  const add_ctx *c = (const add_ctx *)v;
  const uint32_t lo = (uint32_t)(((uint64_t)c->M * i) / n_threads);
  const uint32_t hi = (uint32_t)(((uint64_t)c->M * (i + 1u)) / n_threads);
  const uint32_t nvec = c->n / LANES;
  const HVX_Vector vs = hvx_splat_sf(c->scale);
  for (uint32_t r = lo; r < hi; ++r) {
    const HVX_UVector *vx = (const HVX_UVector *)(c->x + (size_t)r * c->n);
    const HVX_UVector *vx2 =
      c->x2 ? (const HVX_UVector *)(c->x2 + (size_t)r * c->n) : NULL;
    HVX_UVector *vo = (HVX_UVector *)(c->out + (size_t)r * c->n);
    const HVX_UVector *vres =
      c->res ? (const HVX_UVector *)(c->res + (size_t)r * c->n) : vo;
    /* Pass 1: the sum of squares of the summed addend, as norm_chunk. */
    HVX_Vector acc = Q6_V_vzero();
    for (uint32_t k = 0; k < nvec; ++k) {
      const HVX_Vector a = vx2 ? Q6_Vsf_vadd_VsfVsf(vx[k], vx2[k]) : vx[k];
      acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(a, a));
    }
    const float ss = lanes_sum(Q6_Vsf_equals_Vqf32(acc));
    const HVX_Vector vr = hvx_splat_sf(1.0f / sqrtf(ss / (float)c->n + c->eps));
    /* Pass 2: out = scale * (out + a * r * gamma). */
    const HVX_UVector *vg = (const HVX_UVector *)c->gamma;
    for (uint32_t k = 0; k < nvec; ++k) {
      HVX_Vector a = vx2 ? Q6_Vsf_vadd_VsfVsf(vx[k], vx2[k]) : vx[k];
      a = Q6_Vsf_vmpy_VsfVsf(a, vr);
      if (vg)
        a = Q6_Vsf_vmpy_VsfVsf(a, vg[k]);
      vo[k] = Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vadd_VsfVsf(vres[k], a), vs);
    }
  }
}

int hvx_rmsnorm_add_f32(float *out, const float *x, const float *x2, uint32_t M,
                        uint32_t n, const float *gamma, float eps, float scale,
                        hvx_worker_pool *pool) {
  return hvx_rmsnorm_add_res_f32(out, NULL, x, x2, M, n, gamma, eps, scale,
                                 pool);
}

int hvx_rmsnorm_add_res_f32(float *out, const float *res, const float *x,
                            const float *x2, uint32_t M, uint32_t n,
                            const float *gamma, float eps, float scale,
                            hvx_worker_pool *pool) {
  if (!out || !x || M == 0u || n == 0u || n % LANES != 0u) {
    return -1;
  }
  add_ctx c = {out, res, x, x2, gamma, M, n, eps, scale};
  hvx_worker_pool_run(pool, add_worker, &c, M);
  return 0;
}

typedef struct {
  float *out;
  const float *res;
  const float *const *xb; /**< column blocks, each M x cols[b] row-major */
  const uint32_t *cols;
  uint32_t n_blk;
  const float *gamma;
  uint32_t M, n;
  float eps, scale;
} addb_ctx;

/* add_worker over column blocks: the same squares summed in the same
   column order, the same second pass, so the same values as a row-major
   x would give. */
static void addb_worker(uint32_t n_threads, uint32_t i, void *v) {
  const addb_ctx *c = (const addb_ctx *)v;
  const uint32_t lo = (uint32_t)(((uint64_t)c->M * i) / n_threads);
  const uint32_t hi = (uint32_t)(((uint64_t)c->M * (i + 1u)) / n_threads);
  const HVX_Vector vs = hvx_splat_sf(c->scale);
  for (uint32_t r = lo; r < hi; ++r) {
    HVX_Vector acc = Q6_V_vzero();
    for (uint32_t b = 0; b < c->n_blk; ++b) {
      const HVX_UVector *vx =
        (const HVX_UVector *)(c->xb[b] + (size_t)r * c->cols[b]);
      for (uint32_t k = 0; k < c->cols[b] / LANES; ++k)
        acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(vx[k], vx[k]));
    }
    const float ss = lanes_sum(Q6_Vsf_equals_Vqf32(acc));
    const HVX_Vector vr = hvx_splat_sf(1.0f / sqrtf(ss / (float)c->n + c->eps));
    uint32_t col0 = 0;
    for (uint32_t b = 0; b < c->n_blk; ++b) {
      const HVX_UVector *vx =
        (const HVX_UVector *)(c->xb[b] + (size_t)r * c->cols[b]);
      const HVX_UVector *vres =
        (const HVX_UVector *)(c->res + (size_t)r * c->n + col0);
      const HVX_UVector *vg =
        c->gamma ? (const HVX_UVector *)(c->gamma + col0) : NULL;
      HVX_UVector *vo = (HVX_UVector *)(c->out + (size_t)r * c->n + col0);
      for (uint32_t k = 0; k < c->cols[b] / LANES; ++k) {
        HVX_Vector a = Q6_Vsf_vmpy_VsfVsf(vx[k], vr);
        if (vg)
          a = Q6_Vsf_vmpy_VsfVsf(a, vg[k]);
        vo[k] = Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vadd_VsfVsf(vres[k], a), vs);
      }
      col0 += c->cols[b];
    }
  }
}

int hvx_rmsnorm_add_blocks_f32(float *out, const float *res,
                               const float *const *xb, const uint32_t *cols,
                               uint32_t n_blk, uint32_t M, const float *gamma,
                               float eps, float scale, hvx_worker_pool *pool) {
  uint32_t n = 0;
  if (!out || !res || !xb || !cols || n_blk == 0u || M == 0u) {
    return -1;
  }
  for (uint32_t b = 0; b < n_blk; ++b) {
    if (!xb[b] || cols[b] == 0u || cols[b] % LANES != 0u) {
      return -1;
    }
    n += cols[b];
  }
  addb_ctx c = {out, res, xb, cols, n_blk, gamma, M, n, eps, scale};
  hvx_worker_pool_run(pool, addb_worker, &c, M);
  return 0;
}
