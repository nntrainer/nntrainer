// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hvx_m1_ops_f32.c
 * @date   27 Sep 2026
 * @brief  The M=1 small ops on HVX, bit-identical to m1_ops_det.h
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * Every arithmetic step is a Vsf multiply, add or subtract on whole
 * vectors -- including the per-chunk scalars (mean + eps, the rsqrt): the
 * reduced sum is already splat across the lanes, so it stays there rather
 * than round-tripping through the scalar FPU, whose sffma the compiler may
 * contract into and whose subnormal handling this file need not know.
 * Rule 24: no flush-to-zero anywhere, no qf32 (v75/v79 differ), and no
 * exp, so LEDGER section 4's argument clamp has nothing to clamp; the
 * domain is m1_ops_det.h's (d >= eps keeps the rsqrt seed normal).
 */

#include "hvx_m1_ops_f32.h"

#include <string.h>

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include "hvx_conv_gate_f32.h"
#include "hvx_convert.h"

#define LANES 32u

/** @brief 1/sqrt(d) per lane: the seed and three Newton-Raphson steps of
 *         m1_rsqrt_det, in its order (t = h*y; t = t*y; t = 1.5 - t;
 *         y = y*t). */
static inline HVX_Vector hvx_rsqrt_det_sf(HVX_Vector d) {
  HVX_Vector y =
    Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x5F3759DFu), Q6_Vuw_vlsr_VuwR(d, 1));
  const HVX_Vector h = Q6_Vsf_vmpy_VsfVsf(d, hvx_splat_sf(0.5f));
  const HVX_Vector three_halves = hvx_splat_sf(1.5f);
  for (int it = 0; it < 3; ++it) {
    HVX_Vector t = Q6_Vsf_vmpy_VsfVsf(h, y);
    t = Q6_Vsf_vmpy_VsfVsf(t, y);
    t = Q6_Vsf_vsub_VsfVsf(three_halves, t);
    y = Q6_Vsf_vmpy_VsfVsf(y, t);
  }
  return y;
}

/** @brief Sum of all 32 lanes into every lane: five rotate-and-add steps.
 *         IEEE add commutes bit for bit, so the rotation direction does
 *         not matter and every lane ends equal (m1_sumsq_det's tree). */
static inline HVX_Vector hvx_reduce_add_sf(HVX_Vector v) {
  v = Q6_Vsf_vadd_VsfVsf(v, Q6_V_vror_VR(v, 64));
  v = Q6_Vsf_vadd_VsfVsf(v, Q6_V_vror_VR(v, 32));
  v = Q6_Vsf_vadd_VsfVsf(v, Q6_V_vror_VR(v, 16));
  v = Q6_Vsf_vadd_VsfVsf(v, Q6_V_vror_VR(v, 8));
  v = Q6_Vsf_vadd_VsfVsf(v, Q6_V_vror_VR(v, 4));
  return v;
}

void hvx_rmsnorm_f32(const float *x, const float *gamma, float *y, uint32_t n,
                     uint32_t chunk, float eps, float *row_scale_out) {
  if (!x || !gamma || !y || chunk == 0u || chunk % LANES != 0u ||
      (chunk & (chunk - 1u)) != 0u || n % chunk != 0u) {
    return;
  }
  const HVX_Vector inv_chunk = hvx_splat_sf(1.0f / (float)chunk);
  const HVX_Vector veps = hvx_splat_sf(eps);
  const uint32_t nvec = chunk / LANES;

  for (uint32_t c = 0; c < n / chunk; ++c) {
    const HVX_UVector *vx = (const HVX_UVector *)(x + (size_t)c * chunk);
    const HVX_UVector *vg = (const HVX_UVector *)gamma;
    HVX_UVector *vy = (HVX_UVector *)(y + (size_t)c * chunk);

    HVX_Vector acc = Q6_V_vzero();
    for (uint32_t i = 0; i < nvec; ++i) {
      const HVX_Vector xv = vx[i];
      acc = Q6_Vsf_vadd_VsfVsf(acc, Q6_Vsf_vmpy_VsfVsf(xv, xv));
    }
    const HVX_Vector sum = hvx_reduce_add_sf(acc);
    const HVX_Vector d =
      Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(sum, inv_chunk), veps);
    const HVX_Vector r = hvx_rsqrt_det_sf(d);
    if (row_scale_out) {
      const int32_t rb = Q6_R_vextract_VR(r, 0);
      memcpy(row_scale_out + c, &rb, sizeof(float));
    }
    for (uint32_t i = 0; i < nvec; ++i) {
      vy[i] = Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vmpy_VsfVsf(vx[i], r), vg[i]);
    }
  }
}

/** @brief One head: a | b halves, (a*c) - (b*s) and (a*s) + (b*c). */
static inline void hvx_rope64_head(float *x, HVX_Vector c, HVX_Vector s) {
  HVX_UVector *v = (HVX_UVector *)x;
  const HVX_Vector a = v[0], b = v[1];
  v[0] = Q6_Vsf_vsub_VsfVsf(Q6_Vsf_vmpy_VsfVsf(a, c), Q6_Vsf_vmpy_VsfVsf(b, s));
  v[1] = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(a, s), Q6_Vsf_vmpy_VsfVsf(b, c));
}

void hvx_rope64_f32(float *q, uint32_t n_q, float *k, uint32_t n_k,
                    const float *cs) {
  if (!cs || (n_q && !q) || (n_k && !k)) {
    return;
  }
  const HVX_Vector c = ((const HVX_UVector *)cs)[0];
  const HVX_Vector s = ((const HVX_UVector *)cs)[1];
  for (uint32_t h = 0; h < n_q; ++h) {
    hvx_rope64_head(q + (size_t)h * 2u * LANES, c, s);
  }
  for (uint32_t h = 0; h < n_k; ++h) {
    hvx_rope64_head(k + (size_t)h * 2u * LANES, c, s);
  }
}

void hvx_conv_gate_m1_f32(const float *abc, float *state3, const float *conv_w,
                          float *out, uint32_t C) {
  if (!abc || !state3 || !conv_w || !out || C == 0u || C % LANES != 0u) {
    return;
  }
  const HVX_UVector *va = (const HVX_UVector *)abc;
  const HVX_UVector *vb = (const HVX_UVector *)(abc + C);
  const HVX_UVector *vc = (const HVX_UVector *)(abc + 2u * C);
  HVX_UVector *vg = (HVX_UVector *)(state3 + 2u * C);
  HVX_UVector *vo = (HVX_UVector *)out;
  const uint32_t nvec = C / LANES;

  /* Row 2 <- a*c (the pre-gate the prefill path fuses into hvx_dq_mul);
     out <- b, which the kernel multiplies in place. */
  for (uint32_t i = 0; i < nvec; ++i) {
    vg[i] = Q6_Vsf_vmpy_VsfVsf(va[i], vc[i]);
    vo[i] = vb[i];
  }
  /* The prefill kernel at m_count = 1 on row t = 2 of the 3-row state:
     the same five operations in the same order as a prefill-shape call,
     so an M=1 chain is bit-identical to the prefill over the same rows. */
  hvx_conv_gate_f32(out, C, state3, C, 2u, 1u, C, conv_w, NULL);

  /* state <- x_{t-1} | g. ponytail: two 8 KiB copies per token at C = 2048
     (x 18 conv layers, roughly 1-2 us); a ring of three row pointers
     removes them if #85's per-op pcycles ever show it. */
  memcpy(state3, state3 + C, (size_t)C * sizeof(float));
  memcpy(state3 + C, state3 + 2u * C, (size_t)C * sizeof(float));
}
