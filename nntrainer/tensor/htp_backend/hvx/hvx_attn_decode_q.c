// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file   hvx_attn_decode_q.c
 * @date   28 Sep 2026
 * @brief  Pure-HVX decode attention over the int8 / int4 KV cache
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Per (query row, head), per block of 32 cache rows r:
 *
 *   acc[r] = sum_g vrmpy(K4[g][r] + 128, Qu[4g..4g+3])      uint8 x uint8
 *   S[r]   = (acc[r] - 128*sum(Qu) - zp*colsum_k[r])
 *            * s_q * log2e/sqrt(hd) * s_k[r]
 *   P      = online softmax of S in f32 (lane mask; m, l, a as vectors)
 *   P'_g   = uint8 of P * s_v[r][g], scale = block row max / 255
 *   acc_g  = sum_j vrmpy(V4[j][32g..] + 128, P'_g[4j..4j+3])
 *   O_g    = a*O_g + scale_g * (acc_g - 128*sum(P'_g))
 *   out    = O / l
 *
 * The masters are offset-binary (hexkl_kv_q.h) because vrmpy's vector
 * operand is unsigned bytes; the +128 falls out as one integer correction
 * per block from the sum of the scalar side's bytes, which the P'
 * quantization has anyway. The scalar-side words (Q per 4 dims, P' per 4
 * rows) sit in cached stack memory; vrmpy takes them from a register.
 * Nothing scalar touches VTCM -- nothing here touches VTCM at all.
 */

#include "hvx_attn_decode_q.h"

#include <math.h>
#include <string.h>

#include <AEEStdErr.h>
#include <HAP_perf.h>

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include "hexkl_kv_q.h"
#include "hvx_attn_math_f32.h"
#include "hvx_attn_softmax_f16.h"
#include "hvx_tile_f16.h"

#define LOG2E 1.4426950408889634f
/** @brief Cache rows per block: one f32 vector of scores. */
#define BLOCK 32u
/** @brief Largest head_dim this path takes (as the fp16 decode). */
#define MAX_HD 128u
/** @brief 4-dim groups of Q at MAX_HD. */
#define MAX_G (MAX_HD / 4u)
/** @brief 32-dim V groups at MAX_HD. */
#define MAX_DT (MAX_HD / 32u)

typedef struct {
  const hexkl_attn_f16_shape *s;
  const hexkl_attn_q_io *io;
  const hexkl_kv_q *kv;
  float scale;             /**< log2e / sqrt(hd) */
  HVX_Vector cap_sf, k_sf; /**< softcap constants (f32 domain) */
  int softcap_on;
  HVX_Vector idx32; /**< {0..31} as int32 */
} dec_ctx;

/** @brief Per-row asymmetric uint8 of one Q row, hvx_quant_u8's rules
 *         (range [min(x,0), max(x,0)], RNE), plus the byte sum. */
static void quant_q_u8(const float *x, uint32_t hd, uint8_t *u, float *scale,
                       int32_t *zp, int32_t *sum) {
  float mn = 0.0f, mx = 0.0f;
  for (uint32_t d = 0; d < hd; ++d) {
    mn = x[d] < mn ? x[d] : mn;
    mx = x[d] > mx ? x[d] : mx;
  }
  *scale = 1.0f;
  *zp = 0;
  *sum = 0;
  if (mn == mx) {
    memset(u, 0, hd);
    return;
  }
  const float s = (mx - mn) / 255.0f;
  int32_t z = (int32_t)rintf(-mn / s);
  z = z < 0 ? 0 : (z > 255 ? 255 : z);
  const float inv = 1.0f / s;
  int32_t acc = 0;
  for (uint32_t d = 0; d < hd; ++d) {
    int32_t v = (int32_t)rintf(x[d] * inv) + z;
    v = v < 0 ? 0 : (v > 255 ? 255 : v);
    u[d] = (uint8_t)v;
    acc += v;
  }
  *scale = s;
  *zp = z;
  *sum = acc;
}

/** @brief One (query row, head) unit. */
static void decode_unit(const dec_ctx *c, uint32_t q, uint32_t h) {
  const hexkl_attn_f16_shape *s = c->s;
  const hexkl_attn_q_io *io = c->io;
  const hexkl_kv_q *kv = c->kv;
  const uint32_t hd = s->head_dim;
  const uint32_t dt = hd / 32u;
  const uint32_t ng = hd / 4u;
  const uint32_t G = s->n_head_q / s->n_head_kv;
  const uint32_t n = h / G;
  const HVX_Vector zero = Q6_V_vzero();
  const HVX_Vector neg_inf = Q6_V_vsplat_R((int)HVX_ATTN_SF_NEG_INF);
  const HVX_Vector one_sf = Q6_V_vsplat_R((int)HVX_ATTN_SF_ONE);
  const HVX_Vector inv255 = hvx_splat_sf(1.0f / 255.0f);

  // Q -> uint8, its 4-byte words for vrmpy, and the dequant constants.
  uint8_t qu[MAX_HD];
  float s_q;
  int32_t zp, qsum;
  quant_q_u8(io->q + (size_t)q * io->q_stride + (size_t)h * hd, hd, qu, &s_q,
             &zp, &qsum);
  uint32_t qw[MAX_G];
  memcpy(qw, qu, hd);
  const HVX_Vector qcorr = Q6_V_vsplat_R(HEXKL_KV_Q_BIAS * qsum);
  const HVX_Vector zpf = hvx_splat_sf((float)zp);
  const HVX_Vector sq = hvx_splat_sf(s_q * c->scale);

  uint32_t lo, hi;
  hexkl_attn_f16_row_range(s, q, &lo, &hi);
  const HVX_Vector vlo = Q6_V_vsplat_R((int)lo);
  const HVX_Vector vhi = Q6_V_vsplat_R((int)hi);

  // Softmax state as all-lanes-equal vectors; the sink is the first key.
  HVX_Vector m = neg_inf;
  HVX_Vector l = zero;
  if (io->sinks) {
    m = hvx_splat_sf(io->sinks[h] * LOG2E);
    l = one_sf;
  }
  HVX_Vector o[MAX_DT];
  for (uint32_t g = 0; g < MAX_DT; ++g) {
    o[g] = zero;
  }

  const uint8_t *kbase = kv->kt4 + hexkl_kv_q_kt4_index(kv, n, 0, 0);
  const size_t kgroup = (size_t)kv->max_rows * 4u; /* bytes per 4-dim group */
  const uint8_t *vbase = kv->v4 + hexkl_kv_q_v4_index(kv, n, 0, 0);
  const float *skb = kv->s_k + hexkl_kv_q_sk_index(kv, n, 0);
  const int32_t *csb = kv->colsum_k + hexkl_kv_q_sk_index(kv, n, 0);

  for (uint32_t k0 = lo & ~(BLOCK - 1u); k0 < hi; k0 += BLOCK) {
    // Scores: one vrmpy per 4-dim group over the 32 rows.
    HVX_Vector acc = zero;
    const uint8_t *kp = kbase + (size_t)k0 * 4u;
    for (uint32_t g = 0; g < ng; ++g) {
      acc =
        Q6_Vuw_vrmpyacc_VuwVubRub(acc, hvx_tile_load_u(kp + g * kgroup), qw[g]);
    }
    const HVX_Vector af = Q6_Vsf_equals_Vw(Q6_Vw_vsub_VwVw(acc, qcorr));
    const HVX_Vector cs = Q6_Vsf_equals_Vw(hvx_tile_load_u(csb + k0));
    const HVX_Vector sk = hvx_tile_load_u(skb + k0);
    HVX_Vector sv = Q6_Vsf_vmpy_VsfVsf(
      Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vsub_VsfVsf(af, Q6_Vsf_vmpy_VsfVsf(zpf, cs)),
                         sq),
      sk);
    if (c->softcap_on) {
      sv = hvx_attn_softcap_f32(sv, c->cap_sf, c->k_sf);
    }
    // Visibility: lo <= k0 + lane < hi.
    const HVX_Vector col = Q6_Vw_vadd_VwVw(c->idx32, Q6_V_vsplat_R((int)k0));
    const HVX_VectorPred vis = Q6_Q_and_QQ(
      Q6_Q_vcmp_gt_VwVw(vhi, col), Q6_Q_not_Q(Q6_Q_vcmp_gt_VwVw(vlo, col)));
    sv = Q6_V_vmux_QVV(vis, sv, neg_inf);

    // Online update; m = -inf guard as the other kernels.
    const HVX_Vector m_new = Q6_Vsf_vmax_VsfVsf(m, hvx_attn_max32_sf(sv));
    const HVX_VectorPred unseen = Q6_Q_vcmp_eq_VwVw(m_new, neg_inf);
    const HVX_Vector m_use = Q6_V_vmux_QVV(unseen, zero, m_new);
    const HVX_Vector a = hvx_attn_exp2_f32(Q6_Vsf_vsub_VsfVsf(m, m_use));
    HVX_Vector p = hvx_attn_exp2_f32(Q6_Vsf_vsub_VsfVsf(sv, m_use));
    p = Q6_V_vmux_QVV(vis, p, zero);
    l = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(l, a), hvx_attn_sum32_sf(p));
    m = m_new;

    // P' per V group: uint8 with the block's row max, packed to 32 bytes.
    uint8_t pbytes[MAX_DT][128];
    HVX_Vector pscale[MAX_DT], pcorr[MAX_DT];
    for (uint32_t g = 0; g < dt; ++g) {
      const HVX_Vector svg =
        hvx_tile_load_u(kv->s_v + hexkl_kv_q_sv_index(kv, n, k0, g));
      const HVX_Vector pp = Q6_Vsf_vmpy_VsfVsf(p, svg);
      const HVX_Vector pm = hvx_attn_max32_sf(pp);
      const HVX_VectorPred pos = Q6_Q_vcmp_gt_VwVw(pm, zero);
      pscale[g] = Q6_V_vmux_QVV(pos, Q6_Vsf_vmpy_VsfVsf(pm, inv255), one_sf);
      const HVX_Vector pq = hvx_sf_to_w_rne(
        Q6_Vsf_vmpy_VsfVsf(pp, hvx_attn_recip_pos_f32(pscale[g])));
      pcorr[g] = Q6_Vw_vasl_VwR(hvx_attn_sum32_w(pq), 7); /* 128 * sum */
      const HVX_Vector ph = Q6_Vh_vpack_VwVw_sat(pq, pq);
      hvx_tile_store_u(pbytes[g], Q6_Vub_vpack_VhVh_sat(ph, ph));
    }
    // P'.V: one vrmpy per (4 rows, 32 dims).
    HVX_Vector vacc[MAX_DT];
    for (uint32_t g = 0; g < dt; ++g) {
      vacc[g] = zero;
    }
    const uint8_t *vp = vbase + (size_t)k0 * hd; /* quad k0/4: (k0/4)*hd*4 */
    for (uint32_t j = 0; j < BLOCK / 4u; ++j) {
      const uint8_t *vrow = vp + (size_t)j * hd * 4u;
      for (uint32_t g = 0; g < dt; ++g) {
        uint32_t pw;
        memcpy(&pw, pbytes[g] + 4u * j, sizeof(pw));
        vacc[g] = Q6_Vuw_vrmpyacc_VuwVubRub(
          vacc[g], hvx_tile_load_u(vrow + 128u * g), pw);
      }
    }
    for (uint32_t g = 0; g < dt; ++g) {
      const HVX_Vector t = Q6_Vsf_vmpy_VsfVsf(
        Q6_Vsf_equals_Vw(Q6_Vw_vsub_VwVw(vacc[g], pcorr[g])), pscale[g]);
      o[g] = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(o[g], a), t);
    }
  }

  // out = O / l.
  const HVX_Vector inv_l = hvx_attn_recip_pos_f32(l);
  float *orow = io->out + (size_t)q * io->out_stride + (size_t)h * hd;
  for (uint32_t g = 0; g < dt; ++g) {
    hvx_tile_store_u(orow + 32u * g, Q6_Vsf_vmpy_VsfVsf(o[g], inv_l));
  }
}

static void decode_worker(uint32_t n_threads, uint32_t i, void *ctx_) {
  const dec_ctx *c = (const dec_ctx *)ctx_;
  const uint32_t n_units = c->s->n_q * c->s->n_head_q;
  const uint32_t lo = (uint32_t)((uint64_t)n_units * i / n_threads);
  const uint32_t hi = (uint32_t)((uint64_t)n_units * (i + 1) / n_threads);
  for (uint32_t u = lo; u < hi; ++u) {
    decode_unit(c, u / c->s->n_head_q, u % c->s->n_head_q);
  }
}

int hvx_attn_decode_q(const hexkl_attn_f16_shape *s, const hexkl_attn_q_io *io,
                      hvx_worker_pool *pool, uint64_t *us) {
  if (!s || !io || !io->q || !io->out || !io->kv) {
    return AEE_EBADPARM;
  }
  const hexkl_kv_q *kv = io->kv;
  if (s->n_q == 0 || s->n_head_kv == 0 || (s->n_head_q % s->n_head_kv) != 0 ||
      s->head_dim == 0 || s->head_dim > MAX_HD || (s->head_dim % 32u) != 0 ||
      s->cache_to < s->cache_from + s->n_q || s->softcap < 0.0f ||
      kv->n_head_kv != s->n_head_kv || kv->head_dim != s->head_dim ||
      kv->max_rows < s->cache_to) {
    return AEE_EBADPARM;
  }
  const uint64_t t0 = HAP_perf_get_time_us();

  dec_ctx c;
  memset(&c, 0, sizeof(c));
  c.s = s;
  c.io = io;
  c.kv = kv;
  c.scale = LOG2E / sqrtf((float)s->head_dim);
  {
    int32_t idx[32];
    for (int32_t i = 0; i < 32; ++i) {
      idx[i] = i;
    }
    c.idx32 = hvx_tile_load_u(idx);
  }
  if (s->softcap > 0.0f) {
    const float cap = s->softcap * LOG2E;
    c.cap_sf = hvx_splat_sf(cap);
    c.k_sf = hvx_splat_sf(2.0f * LOG2E / cap);
    c.softcap_on = 1;
  }

  hvx_worker_pool_run(pool, decode_worker, &c, s->n_q * s->n_head_q);

  if (us) {
    *us = HAP_perf_get_time_us() - t0;
  }
  return AEE_SUCCESS;
}
