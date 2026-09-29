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
 * Rule 24: no flush-to-zero anywhere, no qf32 (v75/v79 differ). The one
 * exp, the router's sigmoid (#132), is hvx_swiglu_det.h's exp_det with its
 * own clamp; the domain is m1_ops_det.h's (d >= eps keeps the rsqrt seed
 * normal).
 */

#include "hvx_m1_ops_f32.h"

#include <string.h>

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include "hvx_conv_gate_f32.h"
#include "hvx_convert.h"
#include "hvx_swiglu_det.h"

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

/** @brief l2fetch of the next weight block: 128 rows of 128 B (16 KiB) at
 *         a time, a hint only -- the bits do not depend on it. */
#define ROUTER_PF_ROWS 128u

/** @brief One vector seen as its 32 lanes, for the scalar selection. */
typedef union {
  HVX_Vector v;
  float f[LANES];
} router_lanes;

void hvx_router_topk_f32(const float *x, const float *w32, const float *bias,
                         uint32_t K, uint32_t E, uint32_t top_k, float *logits,
                         uint32_t *sel, float *weight) {
  HVX_Vector a0 = Q6_V_vzero(), a1 = Q6_V_vzero(), a2 = Q6_V_vzero(),
             a3 = Q6_V_vzero();
  const HVX_UVector *vw = (const HVX_UVector *)w32;
  HVX_Vector wsum = Q6_V_vzero();
  router_lanes lg, sg, sc;
  uint32_t taken = 0u, e, r, k;
  if (!x || !w32 || !bias || !logits || !sel || !weight || K == 0u ||
      K % 4u != 0u || E == 0u || E > LANES || top_k == 0u || top_k > E) {
    return;
  }
  /* logit: four lane accumulators over k = 0, 1, 2, 3 (mod 4), then
     (a0 + a1) + (a2 + a3) -- m1_router_topk_det's order per lane */
  for (k = 0; k < K; k += 4u) {
#if defined(__hexagon__)
    if (k % ROUTER_PF_ROWS == 0u && k + ROUTER_PF_ROWS < K) {
      /* Rtt: [47:32] stride, [31:16] width, [15:0] height */
      Q6_l2fetch_AP((void *)(w32 + (size_t)(k + ROUTER_PF_ROWS) * LANES),
                    ((uint64_t)(LANES * 4u) << 32) |
                      ((uint64_t)(LANES * 4u) << 16) |
                      (uint64_t)ROUTER_PF_ROWS);
    }
#endif
    a0 = Q6_Vsf_vadd_VsfVsf(a0, Q6_Vsf_vmpy_VsfVsf(hvx_splat_sf(x[k]), vw[k]));
    a1 = Q6_Vsf_vadd_VsfVsf(
      a1, Q6_Vsf_vmpy_VsfVsf(hvx_splat_sf(x[k + 1u]), vw[k + 1u]));
    a2 = Q6_Vsf_vadd_VsfVsf(
      a2, Q6_Vsf_vmpy_VsfVsf(hvx_splat_sf(x[k + 2u]), vw[k + 2u]));
    a3 = Q6_Vsf_vadd_VsfVsf(
      a3, Q6_Vsf_vmpy_VsfVsf(hvx_splat_sf(x[k + 3u]), vw[k + 3u]));
  }
  lg.v =
    Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vadd_VsfVsf(a0, a1), Q6_Vsf_vadd_VsfVsf(a2, a3));
  /* sigmoid = recip_det(1 + exp_det(0 - logit)); score = sigmoid + bias */
  sg.v = hvx_recip_det_sf(
    Q6_Vsf_vadd_VsfVsf(hvx_splat_sf(1.0f),
                       hvx_exp_det_sf(Q6_Vsf_vsub_VsfVsf(Q6_V_vzero(), lg.v))));
  memset(&sc, 0, sizeof(sc));
  memcpy(sc.f, bias, (size_t)E * sizeof(float));
  sc.v = Q6_Vsf_vadd_VsfVsf(sg.v, sc.v);
  memcpy(logits, lg.f, (size_t)E * sizeof(float));
  /* top_k rounds of argmax over the unchosen, the lowest index on a tie: a
     compare moves no bits, so the scalar unit may do it */
  for (r = 0; r < top_k; ++r) {
    uint32_t best = E;
    for (e = 0; e < E; ++e) {
      if (!(taken & (1u << e)) && (best == E || sc.f[e] > sc.f[best])) {
        best = e;
      }
    }
    taken |= 1u << best;
    sel[r] = best;
    wsum = Q6_Vsf_vadd_VsfVsf(wsum, hvx_splat_sf(sg.f[best]));
  }
  /* weight = (sig * recip_det(wsum + 1e-6)) * 1.0 on splats, so no scalar
     FPU op (and no sffma the compiler could contract into) touches the
     arithmetic */
  {
    const HVX_Vector inv =
      hvx_recip_det_sf(Q6_Vsf_vadd_VsfVsf(wsum, hvx_splat_sf(1e-6f)));
    for (r = 0; r < top_k; ++r) {
      lg.f[r] = sg.f[sel[r]];
    }
    lg.v =
      Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vmpy_VsfVsf(lg.v, inv), hvx_splat_sf(1.0f));
    memcpy(weight, lg.f, (size_t)top_k * sizeof(float));
  }
}
