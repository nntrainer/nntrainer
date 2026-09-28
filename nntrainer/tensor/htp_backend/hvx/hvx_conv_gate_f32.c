// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hvx_conv_gate_f32.c
 * @date   22 Sep 2026
 * @brief  Causal depthwise conv1d (L=3) folded into the conv block's gate
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "hvx_conv_gate_f32.h"

#include <stddef.h>

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include "hvx_convert.h"

#define LANES 32u

typedef struct {
  float *z;
  const float *g;
  const float *w;
  uint32_t z_stride, g_stride, t0, m_count, C;
} conv_gate_ctx;

/** @brief One row: 32 columns per iteration, the three taps' vectors
 *         reloaded per row -- they sit in VTCM beside z, and a row is 64
 *         vector iterations at C = 2048, so the loads are not the cost. */
static void conv_gate_row(const conv_gate_ctx *c, uint32_t r) {
  const uint32_t t = c->t0 + r;
  float *zr = c->z + (size_t)r * c->z_stride;
  const float *g0 = c->g + (size_t)t * c->g_stride;
  const float *g1 = (t >= 1u) ? c->g + (size_t)(t - 1u) * c->g_stride : NULL;
  const float *g2 = (t >= 2u) ? c->g + (size_t)(t - 2u) * c->g_stride : NULL;
  const float *w0 = c->w, *w1 = c->w + c->C, *w2 = c->w + 2u * c->C;
  const HVX_Vector zero = Q6_V_vzero();

  for (uint32_t j = 0; j < c->C; j += LANES) {
    const HVX_Vector x0 = ((const HVX_UVector *)(g0 + j))[0];
    const HVX_Vector x1 = g1 ? ((const HVX_UVector *)(g1 + j))[0] : zero;
    const HVX_Vector x2 = g2 ? ((const HVX_UVector *)(g2 + j))[0] : zero;
    const HVX_Vector k0 = ((const HVX_UVector *)(w0 + j))[0];
    const HVX_Vector k1 = ((const HVX_UVector *)(w1 + j))[0];
    const HVX_Vector k2 = ((const HVX_UVector *)(w2 + j))[0];
    /* (w0*x0 + w1*x1) + w2*x2, then the gate: five separate operations,
       the order the header promises. */
    HVX_Vector y = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(k0, x0),
                                      Q6_Vsf_vmpy_VsfVsf(k1, x1));
    y = Q6_Vsf_vadd_VsfVsf(y, Q6_Vsf_vmpy_VsfVsf(k2, x2));
    HVX_UVector *zo = (HVX_UVector *)(zr + j);
    zo[0] = Q6_Vsf_vmpy_VsfVsf(zo[0], y);
  }
}

static void conv_gate_worker(uint32_t n_threads, uint32_t i, void *vctx) {
  const conv_gate_ctx *c = (const conv_gate_ctx *)vctx;
  const uint32_t lo = (uint32_t)((uint64_t)c->m_count * i / n_threads);
  const uint32_t hi = (uint32_t)((uint64_t)c->m_count * (i + 1) / n_threads);
  for (uint32_t r = lo; r < hi; ++r) {
    conv_gate_row(c, r);
  }
}

void hvx_conv_gate_f32(float *z, uint32_t z_stride, const float *g,
                       uint32_t g_stride, uint32_t t0, uint32_t m_count,
                       uint32_t C, const float *conv_w, hvx_worker_pool *pool) {
  if (!z || !g || !conv_w || m_count == 0u || C == 0u) {
    return;
  }
  conv_gate_ctx c = {z, g, conv_w, z_stride, g_stride, t0, m_count, C};
  hvx_worker_pool_run(pool, conv_gate_worker, &c, m_count);
}
