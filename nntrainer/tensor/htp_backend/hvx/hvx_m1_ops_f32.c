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
 * vectors, with one exception: the RMSNorm row scale r (#164) is the
 * Android CPU's, computed on the scalar core -- 16 sffma chains held in
 * registers (IEEE fused multiply-add, the CPU's fmla), m1_ops_det.h's
 * reduction order, and its integer sqrt / reciprocal, each correctly
 * rounded. HVX has no sf FMA, and an emulated one costs more than the
 * whole kernel (plan 164 section 3.2); r is one scalar per chunk, splat
 * once for the (x * r) * gamma loop.
 * RoPE is the other exception to "f32 inside": it is the Android CPU's
 * fp16 RoPE (#152), every step rounded with hvx_rne16_sf, so the resident
 * attention sees the CPU's q and k bit for bit.
 * Rule 24: no flush-to-zero anywhere, no qf32 (v75/v79 differ); the
 * norms' domain is m1_ops_det.h's (d >= eps, normal). The router is the
 * third exception (#132 PR 2): the Android CPU's order, E fused sffma
 * chains on the scalar core and the spec's own sigmoid and selection
 * (m1_router_cpu_sigmoid / _pick).
 */

#include "hvx_m1_ops_f32.h"

#include <string.h>

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include "hvx_conv_gate_f32.h"
#include "hvx_convert.h"
#include "hvx_swiglu_det.h"
#include "m1_ops_det.h"

#define LANES 32u

/** @brief The CPU's row scale r of one chunk (m1_norm_scale_det of
 *         m1_sumsq_cpu_det): the 16 chains are sffma (Q6_R_sfmpyacc_RR,
 *         one rounding, what fmaf is), in registers; the reduction and the
 *         integer sqrt / reciprocal are the spec's own functions. */
static inline float hvx_rmsnorm_scale(const float *x, uint32_t chunk,
                                      float eps) {
  float acc[M1_DET_NORM_CHAINS];
  for (uint32_t j = 0; j < M1_DET_NORM_CHAINS; ++j) {
    acc[j] = 0.0f;
  }
  for (uint32_t i = 0; i < chunk; i += M1_DET_NORM_CHAINS) {
    for (uint32_t j = 0; j < M1_DET_NORM_CHAINS; ++j) {
      acc[j] = Q6_R_sfmpyacc_RR(acc[j], x[i + j], x[i + j]);
    }
  }
  return m1_norm_scale_det(m1_norm_reduce_det(acc), chunk, eps);
}

void hvx_rmsnorm_f32(const float *x, const float *gamma, float *y, uint32_t n,
                     uint32_t chunk, float eps, float *row_scale_out) {
  if (!x || !gamma || !y || chunk == 0u || chunk % LANES != 0u ||
      (chunk & (chunk - 1u)) != 0u || n % chunk != 0u) {
    return;
  }
  const uint32_t nvec = chunk / LANES;

  for (uint32_t c = 0; c < n / chunk; ++c) {
    const HVX_UVector *vx = (const HVX_UVector *)(x + (size_t)c * chunk);
    const HVX_UVector *vg = (const HVX_UVector *)gamma;
    HVX_UVector *vy = (HVX_UVector *)(y + (size_t)c * chunk);

    const float rs = hvx_rmsnorm_scale(x + (size_t)c * chunk, chunk, eps);
    if (row_scale_out) {
      row_scale_out[c] = rs;
    }
    const HVX_Vector r = hvx_splat_sf(rs);
    for (uint32_t i = 0; i < nvec; ++i) {
      vy[i] = Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vmpy_VsfVsf(vx[i], r), vg[i]);
    }
  }
}

/** @brief One head in fp16 (m1_rope64_det): a | b halves rounded, then
 *         rne16(rne16(a*c) - rne16(b*s)) and rne16(rne16(a*s) + rne16(b*c)),
 *         the CPU's fmul / fsub / fadd .8h. */
static inline void hvx_rope64_head(float *x, HVX_Vector c, HVX_Vector s) {
  HVX_UVector *v = (HVX_UVector *)x;
  const HVX_Vector a = hvx_rne16_sf(v[0]), b = hvx_rne16_sf(v[1]);
  v[0] =
    hvx_rne16_sf(Q6_Vsf_vsub_VsfVsf(hvx_rne16_sf(Q6_Vsf_vmpy_VsfVsf(a, c)),
                                    hvx_rne16_sf(Q6_Vsf_vmpy_VsfVsf(b, s))));
  v[1] =
    hvx_rne16_sf(Q6_Vsf_vadd_VsfVsf(hvx_rne16_sf(Q6_Vsf_vmpy_VsfVsf(a, s)),
                                    hvx_rne16_sf(Q6_Vsf_vmpy_VsfVsf(b, c))));
}

void hvx_rope64_f32(float *q, uint32_t n_q, float *k, uint32_t n_k,
                    const float *cs) {
  if (!cs || (n_q && !q) || (n_k && !k)) {
    return;
  }
  /* The CPU's table is (_FP16) of the same f32 values. */
  const HVX_Vector c = hvx_rne16_sf(((const HVX_UVector *)cs)[0]);
  const HVX_Vector s = hvx_rne16_sf(((const HVX_UVector *)cs)[1]);
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

/** @brief Router chains run at once: ROUTER_CHAINS accumulators stay in
 *  registers across the k loop (the first sitting's 32-entry array spilled
 *  to memory, 427784 pcycles/op against the vector router's 34100). */
#ifndef ROUTER_CHAINS
#define ROUTER_CHAINS 8u
#endif

/** @brief Experts e0 .. e0 + ROUTER_CHAINS - 1 of the w32 rows: one fused
 *         chain each over k in order (the CPU's sgemv_n fmadd). */
static inline __attribute__((always_inline)) void
router_chains(const float *x, const float *w32, uint32_t K, uint32_t e0,
              float *out) {
  float acc[ROUTER_CHAINS];
  for (uint32_t j = 0; j < ROUTER_CHAINS; ++j) {
    acc[j] = 0.0f;
  }
  for (uint32_t k0 = 0; k0 < K; k0 += ROUTER_PF_ROWS) {
    const uint32_t k1 = k0 + ROUTER_PF_ROWS < K ? k0 + ROUTER_PF_ROWS : K;
#if defined(__hexagon__)
    if (e0 == 0u && k1 < K) {
      /* the next 16 KiB; Rtt: [47:32] stride, [31:16] width, [15:0] height */
      Q6_l2fetch_AP((void *)(w32 + (size_t)k1 * LANES),
                    ((uint64_t)(LANES * 4u) << 32) |
                      ((uint64_t)(LANES * 4u) << 16) |
                      (uint64_t)ROUTER_PF_ROWS);
    }
#endif
    for (uint32_t k = k0; k < k1; ++k) {
      const float xk = x[k];
      /* w32 is at least 8-byte aligned (malloc's, the graph's memalign),
         a row 128 bytes, e0 a multiple of 8 floats: paired loads */
      const float *row = (const float *)__builtin_assume_aligned(
        w32 + (size_t)k * LANES + e0, 8);
      for (uint32_t j = 0; j < ROUTER_CHAINS; ++j) {
        acc[j] = Q6_R_sfmpyacc_RR(acc[j], xk, row[j]);
      }
    }
  }
  for (uint32_t j = 0; j < ROUTER_CHAINS; ++j) {
    out[j] = acc[j];
  }
}

typedef struct {
  const float *x, *w32, *bias;
  uint32_t K, E;
  float *acc, *sig, *score;
} router_ctx;

/** @brief Pool lane i: chain groups i, i + n, ... of ROUTER_CHAINS, then
 *         the sigmoids of the experts it owns (the expf port and the
 *         integer divide are the costly scalar part, so they spread too). */
static void router_lane(uint32_t n, uint32_t i, void *v) {
  const router_ctx *c = (const router_ctx *)v;
  for (uint32_t e0 = i * ROUTER_CHAINS; e0 < c->E; e0 += n * ROUTER_CHAINS) {
    router_chains(c->x, c->w32, c->K, e0, c->acc + e0);
    for (uint32_t e = e0; e < e0 + ROUTER_CHAINS && e < c->E; ++e) {
      m1_router_cpu_sigmoid(c->acc[e], c->bias[e], &c->sig[e], &c->score[e]);
    }
  }
}

void hvx_router_topk_f32(const float *x, const float *w32, const float *bias,
                         uint32_t K, uint32_t E, uint32_t top_k, float *logits,
                         uint32_t *sel, float *weight, hvx_worker_pool *pool) {
  float acc[LANES], sig[LANES], score[LANES];
  if (!x || !w32 || !bias || !logits || !sel || !weight || K == 0u || E == 0u ||
      E > LANES || top_k == 0u || top_k > E) {
    return;
  }
  /* [#132 PR 2] the CPU's sgemv_n: one fused chain per expert in k order
     (m1_router_cpu_det), ROUTER_CHAINS of them per pool lane on the scalar
     cores; lanes >= E are the padding's zero columns, computed and dropped.
     The bits do not depend on the lane count. */
  router_ctx c = {x, w32, bias, K, E, acc, sig, score};
  hvx_worker_pool_run(pool, router_lane, &c,
                      (E + ROUTER_CHAINS - 1u) / ROUTER_CHAINS);
  memcpy(logits, acc, (size_t)E * sizeof(float));
  m1_router_cpu_pick(sig, score, E, top_k, sel, weight);
}
