// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hvx_dequant_i32.c
 * @date   03 Aug 2026
 * @brief  int32 accumulator to f32 dequantization for the A8W4 path
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 */

#include <stddef.h>

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include "hvx_convert.h"
#include "hvx_dequant_i32.h"

/** @brief HVX vector width in bytes (128B mode). */
#define VLEN 128u
/** @brief f32 or int32 lanes per HVX vector. */
#define LANES (VLEN / 4u)

/** @brief One row of K3: out's m-th row from acc's m-th row. Touches only
 *         row m of acc and out, plus the three per-channel tables it shares
 *         read-only with every other row -- which is what makes splitting
 *         this by row range across worker threads safe, the same argument
 *         K1 rests on. */
static void dequant_row_one(const int32_t *acc, uint32_t m, uint32_t n,
                            const float *act_scale, const int32_t *act_zp,
                            const int32_t *colsum_w, const float *w_scale,
                            const float *bias, float *out) {
  const int32_t *arow = acc + (size_t)m * n;
  float *orow = out + (size_t)m * n;
  const float s = act_scale[m];
  const int32_t z = act_zp[m];

  const uint32_t n_vec = n / LANES;
  const HVX_Vector vs = hvx_splat_sf(s);
  const HVX_Vector vzf = Q6_Vsf_equals_Vw(Q6_V_vsplat_R(z));

  const HVX_UVector *vacc = (const HVX_UVector *)arow;
  const HVX_UVector *vcs = (const HVX_UVector *)colsum_w;
  const HVX_UVector *vws = (const HVX_UVector *)w_scale;
  const HVX_UVector *vb = (const HVX_UVector *)bias;
  HVX_UVector *vout = (HVX_UVector *)orow;

  for (uint32_t v = 0; v < n_vec; ++v) {
    const HVX_Vector af = Q6_Vsf_equals_Vw(vacc[v]);
    const HVX_Vector csf = Q6_Vsf_equals_Vw(vcs[v]);
    const HVX_Vector corrected =
      Q6_Vsf_vsub_VsfVsf(af, Q6_Vsf_vmpy_VsfVsf(vzf, csf));
    const HVX_Vector scaled =
      Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vmpy_VsfVsf(corrected, vs), vws[v]);
    vout[v] = Q6_Vsf_vadd_VsfVsf(scaled, vb[v]);
  }

  for (uint32_t j = n_vec * LANES; j < n; ++j) {
    const int32_t corrected = arow[j] - z * colsum_w[j];
    orow[j] = (float)corrected * s * w_scale[j] + bias[j];
  }
}

typedef struct {
  const int32_t *acc;
  uint32_t m_valid, n;
  const float *act_scale;
  const int32_t *act_zp;
  const int32_t *colsum_w;
  const float *w_scale;
  const float *bias;
  float *out;
} dequant_rows_ctx;

static void dequant_rows_worker(uint32_t n_threads, uint32_t i, void *ctx_) {
  dequant_rows_ctx *ctx = (dequant_rows_ctx *)ctx_;
  const uint32_t lo = (uint32_t)((uint64_t)ctx->m_valid * i / n_threads);
  const uint32_t hi = (uint32_t)((uint64_t)ctx->m_valid * (i + 1) / n_threads);
  for (uint32_t m = lo; m < hi; ++m) {
    dequant_row_one(ctx->acc, m, ctx->n, ctx->act_scale, ctx->act_zp,
                    ctx->colsum_w, ctx->w_scale, ctx->bias, ctx->out);
  }
}

void hvx_dequant_i32_to_f32(const int32_t *acc, uint32_t m_valid,
                            uint32_t m_pad, uint32_t n, const float *act_scale,
                            const int32_t *act_zp, const int32_t *colsum_w,
                            const float *w_scale, const float *bias, float *out,
                            hvx_worker_pool *pool) {
  (void)m_pad;

  // Row split, not a column split: a row's arithmetic is untouched by this,
  // so every output float is bit-identical to what the serial loop produced
  // and the host reference still matches exactly. Splitting the n dimension
  // instead would have divided each row's reduction-free work into pieces
  // sharing a cache line at the seams for no gain.
  dequant_rows_ctx ctx = {acc,      m_valid, n,    act_scale, act_zp,
                          colsum_w, w_scale, bias, out};
  hvx_worker_pool_run(pool, dequant_rows_worker, &ctx, m_valid);
}
