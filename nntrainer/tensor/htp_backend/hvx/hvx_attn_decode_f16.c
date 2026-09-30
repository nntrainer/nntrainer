// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file   hvx_attn_decode_f16.c
 * @date   14 Sep 2026
 * @brief  Pure-HVX flash attention for a handful of query rows (decode)
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Per (query row, head): walk the visible cache in blocks of 64 rows,
 *   s_j = q . k_j          fp16 dot on HVX, summed in qf32
 *   m'  = max(m, max s);  P = 2^(s - m');  l = l 2^(m-m') + sum P
 *   O   = O 2^(m-m') + sum_j P_j v_j      f32 accumulate
 * and finally out = O / l. Q carries log2e/sqrt(hd) so the softmax is
 * base-2, as in the HMX kernel.
 *
 * Lane order note: the fp16 -> qf32 widening multiply splits a 64-lane
 * vector into even lanes (low result) and odd lanes (high result). The Q
 * vector is built in NATURAL fp16 order (by dealing the f32 input into
 * even/odd words before narrowing) so its lanes line up with the K rows
 * read from DDR; and the O accumulators are kept in that dealt (even,
 * odd) order until the store shuffles them back. Nothing here depends on
 * the HMX tile layout.
 */

#include "hvx_attn_decode_f16.h"

#include <math.h>
#include <string.h>

#include <AEEStdErr.h>
#include <HAP_perf.h>

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include "hvx_attn_softmax_f16.h"
#include "hvx_tile_f16.h"

#define LOG2E 1.4426950408889634f
/** @brief Cache rows per block: one fp16 vector of scores. */
#define BLOCK 64u
/** @brief Largest head_dim: two 64-lane fp16 vectors. */
#define MAX_HD 128u
/** @brief fp16 vectors per head row at MAX_HD. */
#define MAX_CHUNKS (MAX_HD / 64u)

typedef struct {
  const hexkl_attn_f16_shape *s;
  const hexkl_attn_f16_io *io;
  float scale;
  HVX_Vector cap_hf, k_hf; /**< softcap constants, see hexkl_attn_f16.c */
  int softcap_on;
  HVX_Vector col_idx; /**< {0,1,...,63} as uint16 */
} dec_ctx;

/** @brief Column index per lane, uint16 {0..63}. */
static HVX_Vector col_index64(void) {
  uint16_t idx[64];
  for (uint32_t l = 0; l < 64u; ++l) {
    idx[l] = (uint16_t)l;
  }
  return hvx_tile_load_u(idx);
}

#define max64_hf hvx_attn_max64_hf
#define f32x64_to_hf hvx_attn_f32x64_to_hf

/**
 * @brief One (query row, head) unit.
 */
static void decode_unit(const dec_ctx *c, uint32_t q, uint32_t h) {
  const hexkl_attn_f16_shape *s = c->s;
  const hexkl_attn_f16_io *io = c->io;
  const uint32_t hd = s->head_dim;
  const uint32_t n_chunks = (hd + 63u) / 64u;
  const uint32_t G = s->n_head_q / s->n_head_kv;
  const uint32_t n = h / G;
  const HVX_Vector one_hf = Q6_Vh_vsplat_R((int)HVX_ATTN_HF_ONE);
  const HVX_Vector neg_inf = Q6_Vh_vsplat_R((int)HVX_ATTN_HF_NEG_INF);
  const HVX_Vector zero = Q6_V_vzero();

  // Q row -> scaled fp16, natural lane order. head_dim 96 leaves the top
  // 32 lanes of the second chunk zero, matching the zero-padded K load.
  HVX_Vector q_hf[MAX_CHUNKS];
  {
    int bits;
    memcpy(&bits, &c->scale, sizeof(bits));
    const HVX_Vector vscale = Q6_V_vsplat_R(bits);
    const float *qrow = io->q + (size_t)q * io->q_stride + (size_t)h * hd;
    for (uint32_t ch = 0; ch < n_chunks; ++ch) {
      float tmp[64];
      const uint32_t n_el = (hd - 64u * ch) < 64u ? (hd - 64u * ch) : 64u;
      memset(tmp, 0, sizeof(tmp));
      memcpy(tmp, qrow + 64u * ch, n_el * sizeof(float));
      const HVX_Vector a =
        Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(hvx_tile_load_u(tmp), vscale));
      const HVX_Vector b = Q6_Vsf_equals_Vqf32(
        Q6_Vqf32_vmpy_VsfVsf(hvx_tile_load_u(tmp + 32), vscale));
      q_hf[ch] = f32x64_to_hf(a, b);
    }
  }

  uint32_t lo, hi;
  hexkl_attn_f16_row_range(s, q, &lo, &hi);

  // Softmax state, sink included as the first "key" if present.
  uint16_t m_hf = (uint16_t)HVX_ATTN_HF_NEG_INF;
  float l = 0.0f;
  if (io->sinks) {
    m_hf = hvx_attn_f32_to_hf(io->sinks[h] * LOG2E);
    l = 1.0f;
  }
  // O in dealt (even, odd) order per chunk, held as IEEE f32 (sf), not
  // qf32: the per-block rescale O *= 2^(m-m') is then an sf*sf multiply,
  // which is exact about zeros. A qf32*qf32 multiply is not -- with a
  // sink, whose nonzero first-block factor met the still-zero O, every
  // lane came back ~1e30 on device (the all-zero pattern and a qf32 zero
  // made by adding 0.0f + 0.0f both did). Each accumulation step widens
  // p*v to qf32, adds the sf running sum, and rounds back to sf; that is
  // an ordinary f32 accumulator.
  HVX_Vector o_even[MAX_CHUNKS], o_odd[MAX_CHUNKS];
  for (uint32_t ch = 0; ch < MAX_CHUNKS; ++ch) {
    o_even[ch] = zero;
    o_odd[ch] = zero;
  }

  const uint16_t *kbase = io->k + (size_t)n * hd;
  const uint16_t *vbase = io->v + (size_t)n * hd;
  const uint32_t row_bytes = hd * 2u;

  for (uint32_t k0 = lo & ~(BLOCK - 1u); k0 < hi; k0 += BLOCK) {
    const uint32_t k1 = (k0 + BLOCK < hi) ? k0 + BLOCK : hi;
    const uint32_t j0 = (k0 < lo) ? lo - k0 : 0u; // first valid lane
    const uint32_t j1 = k1 - k0;                  // one past last valid

    // Scores for the block: fp16 dot per row, summed in qf32.
    float sc[BLOCK];
    for (uint32_t j = 0; j < BLOCK; ++j) {
      if (j < j0 || j >= j1) {
        sc[j] = 0.0f; // masked below; keep it finite
        continue;
      }
      const uint16_t *krow = kbase + (size_t)(k0 + j) * io->kv_stride;
      HVX_Vector acc = zero;
      for (uint32_t ch = 0; ch < n_chunks; ++ch) {
        HVX_Vector kv;
        if (row_bytes - 128u * ch >= 128u) {
          kv = hvx_tile_load_u(krow + 64u * ch);
        } else {
          kv = zero;
          memcpy(&kv, krow + 64u * ch, row_bytes - 128u * ch);
        }
        const HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(kv, q_hf[ch]);
        acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_V_lo_W(p));
        acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_V_hi_W(p));
      }
      sc[j] = hvx_attn_lane0_f32(hvx_attn_sum32_sf(Q6_Vsf_equals_Vqf32(acc)));
    }
    // sc was just filled by scalar stores: load it through memcpy (see
    // hvx_tile_load_u), never a punned vector load.
    HVX_Vector sv = f32x64_to_hf(hvx_tile_load_u(sc), hvx_tile_load_u(sc + 32));
    if (c->softcap_on) {
      sv = hvx_attn_softcap_hf(sv, c->cap_hf, c->k_hf);
    }
    if (j0 != 0 || j1 != BLOCK) {
      const HVX_VectorPred below_hi =
        Q6_Q_vcmp_gt_VuhVuh(Q6_Vh_vsplat_R((int)j1), c->col_idx);
      const HVX_VectorPred below_lo =
        Q6_Q_vcmp_gt_VuhVuh(Q6_Vh_vsplat_R((int)j0), c->col_idx);
      sv =
        Q6_V_vmux_QVV(Q6_Q_and_QQ(below_hi, Q6_Q_not_Q(below_lo)), sv, neg_inf);
    }

    // Online update. The m = -inf guard mirrors hexkl_attn_f16.c.
    const HVX_Vector m_old = Q6_Vh_vsplat_R((int)m_hf);
    const HVX_Vector m_new = Q6_Vhf_vmax_VhfVhf(m_old, max64_hf(sv));
    const HVX_VectorPred unseen = Q6_Q_vcmp_eq_VhVh(m_new, neg_inf);
    const HVX_Vector m_use = Q6_V_vmux_QVV(unseen, zero, m_new);
    const HVX_Vector a_v = hvx_attn_exp2_hf(Q6_Vhf_vsub_VhfVhf(m_old, m_use));
    const uint16_t a_hf = (uint16_t)(hvx_attn_word0(a_v) & 0xFFFFu);
    m_hf = (uint16_t)(hvx_attn_word0(m_new) & 0xFFFFu);
    const float a = hvx_attn_hf_to_f32(a_hf);

    const HVX_Vector p = hvx_attn_exp2_hf(Q6_Vhf_vsub_VhfVhf(sv, m_use));
    const HVX_VectorPair pw = Q6_Wqf32_vmpy_VhfVhf(p, one_hf);
    const float rowsum =
      hvx_attn_lane0_f32(hvx_attn_sum32_sf(Q6_Vsf_equals_Vqf32(
        Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(pw), Q6_V_hi_W(pw)))));
    l = l * a + rowsum;

    // O = a*O + sum_j p_j v_j
    int abits;
    memcpy(&abits, &a, sizeof(abits));
    const HVX_Vector a_sf = Q6_V_vsplat_R(abits);
    for (uint32_t ch = 0; ch < n_chunks; ++ch) {
      o_even[ch] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(o_even[ch], a_sf));
      o_odd[ch] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(o_odd[ch], a_sf));
    }
    uint16_t pb[BLOCK];
    hvx_tile_store_u(pb, p);
    for (uint32_t j = j0; j < j1; ++j) {
      const HVX_Vector pj = Q6_Vh_vsplat_R((int)pb[j]);
      const uint16_t *vrow = vbase + (size_t)(k0 + j) * io->kv_stride;
      for (uint32_t ch = 0; ch < n_chunks; ++ch) {
        HVX_Vector vv;
        if (row_bytes - 128u * ch >= 128u) {
          vv = hvx_tile_load_u(vrow + 64u * ch);
        } else {
          vv = zero;
          memcpy(&vv, vrow + 64u * ch, row_bytes - 128u * ch);
        }
        const HVX_VectorPair w = Q6_Wqf32_vmpy_VhfVhf(vv, pj);
        o_even[ch] =
          Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_lo_W(w), o_even[ch]));
        o_odd[ch] =
          Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_hi_W(w), o_odd[ch]));
      }
    }
  }

  // out = O / l, shuffled back to natural order.
  const float inv_l = 1.0f / l;
  int ibits;
  memcpy(&ibits, &inv_l, sizeof(ibits));
  const HVX_Vector vinv = Q6_V_vsplat_R(ibits);
  float *orow = io->out + (size_t)q * io->out_stride + (size_t)h * hd;
  for (uint32_t ch = 0; ch < n_chunks; ++ch) {
    const HVX_Vector e = Q6_Vsf_vmpy_VsfVsf(o_even[ch], vinv);
    const HVX_Vector o = Q6_Vsf_vmpy_VsfVsf(o_odd[ch], vinv);
    const HVX_VectorPair nat = Q6_W_vshuff_VVR(o, e, -4);
    const uint32_t n_el = (hd - 64u * ch) < 64u ? (hd - 64u * ch) : 64u;
    float tmp[64];
    hvx_tile_store_u(tmp, Q6_V_lo_W(nat));
    hvx_tile_store_u(tmp + 32, Q6_V_hi_W(nat));
    memcpy(orow + 64u * ch, tmp, n_el * sizeof(float));
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

int hvx_attn_decode_f16(const hexkl_attn_f16_shape *s,
                        const hexkl_attn_f16_io *io, hvx_worker_pool *pool,
                        uint64_t *us) {
  if (!s || !io || !io->q || !io->k || !io->v || !io->out) {
    return AEE_EBADPARM;
  }
  if (s->n_q == 0 || s->n_head_kv == 0 || (s->n_head_q % s->n_head_kv) != 0 ||
      s->head_dim == 0 || s->head_dim > MAX_HD || (s->head_dim % 32u) != 0 ||
      s->cache_to < s->cache_from + s->n_q || s->softcap < 0.0f) {
    return AEE_EBADPARM;
  }
  const uint64_t t0 = HAP_perf_get_time_us();

  dec_ctx c;
  memset(&c, 0, sizeof(c));
  c.s = s;
  c.io = io;
  c.scale = LOG2E / sqrtf((float)s->head_dim);
  c.col_idx = col_index64();
  if (s->softcap > 0.0f) {
    const float cap = s->softcap * LOG2E;
    c.cap_hf = Q6_Vh_vsplat_R((int)hvx_attn_f32_to_hf(cap));
    c.k_hf = Q6_Vh_vsplat_R((int)hvx_attn_f32_to_hf(2.0f * LOG2E / cap));
    c.softcap_on = 1;
  }

  hvx_worker_pool_run(pool, decode_worker, &c, s->n_q * s->n_head_q);

  if (us) {
    *us = HAP_perf_get_time_us() - t0;
  }
  return AEE_SUCCESS;
}
