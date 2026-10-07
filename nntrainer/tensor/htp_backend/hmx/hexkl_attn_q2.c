// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file   hexkl_attn_q2.c
 * @date   07 Oct 2026
 * @brief  Row-blocked A8W8 attention on HMX over a fixed-scale KV cache
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "hexkl_attn_q2.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <AEEStdErr.h>
#include <HAP_perf.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include "hexkl_cvt.h"
#include "hexkl_dma_ring.h"
#include "hexkl_hmx_mm.h"
#include "hexkl_micro.h"
#include "hvx_attn_softmax_f16.h"
#include "hvx_convert.h"
#include "hvx_softmax_q.h"
#include "hvx_tile_f16.h"

#define LOG2E 1.4426950408889634f
#define FRAC_BITS 8u

static inline uint64_t now_us(void) { return HAP_perf_get_time_us(); }

typedef struct {
  uint8_t *vb;
  const hexkl_attn_f16_shape *s;
  const hexkl_attn_q2_io *io;
  hexkl_attn_q2_layout L;
  hexkl_attn_q2_range res; /**< resident column tiles */
  uint32_t hd, dt, G;
  int16_t *corr_heap; /**< [(n_res+1)*32] staging in cached memory */
  int32_t *rs_heap;   /**< HVX_SOFTMAX_Q_ROWSUM_WORDS, cached copy */
  hexkl_attn_q2_stats st;
} ctx;

/* ---- small helpers ----------------------------------------------------- */

/** @brief @a dst must be 128-byte aligned (VTCM regions, or memalign'd
 *         heap); @a src may be anything. An aligned vector store to an
 *         unaligned address is a fault, not a slow path. */
static inline void vcopy(void *dst, const void *src, uint32_t bytes) {
  HVX_Vector *d = (HVX_Vector *)dst;
  for (uint32_t i = 0; i < bytes / 128u; ++i) {
    d[i] = hvx_tile_load_u((const uint8_t *)src + 128u * i);
  }
}

/** @brief Builds a uniform bias block in cached memory, copies it to VTCM. */
static void set_cvt_block(uint8_t *vtcm_block, uint16_t scale_hf) {
  uint32_t blk[HEXKL_CVT_BLOCK_BYTES / 4u] __attribute__((aligned(128)));
  hexkl_cvt_block_set_uniform(blk, scale_hf);
  vcopy(vtcm_block, blk, HEXKL_CVT_BLOCK_BYTES);
}

/* ---- stage 0: Q --------------------------------------------------------- */

/** @brief max |q| over the n_q rows of head h. */
static float q_amax(const ctx *c, uint32_t h) {
  const HVX_Vector mask = Q6_V_vsplat_R(0x7FFFFFFF);
  HVX_Vector m = Q6_V_vzero();
  for (uint32_t r = 0; r < c->s->n_q; ++r) {
    const float *row = c->io->q + (size_t)r * c->io->q_stride + h * c->hd;
    for (uint32_t d = 0; d < c->dt; ++d) {
      m = Q6_Vsf_vmax_VsfVsf(
        m, Q6_V_vand_VV(hvx_tile_load_u(row + 32u * d), mask));
    }
  }
  return hvx_attn_lane0_f32(hvx_attn_max32_sf(m));
}

/** @brief Rows [q0, q0 + n_rows) of head h -> uint8 tiles q_ah[dt], offset
 *         binary at scale s_q; rows past n_rows read as 0 (128). */
static void q_quant_block(const ctx *c, uint32_t h, uint32_t q0,
                          uint32_t n_rows, float s_q) {
  const HVX_Vector inv = hvx_splat_sf(1.0f / s_q);
  const HVX_Vector lo = Q6_V_vsplat_R(-127), hi = Q6_V_vsplat_R(127);
  const HVX_Vector bias = Q6_V_vsplat_R(128);
  for (uint32_t d = 0; d < c->dt; ++d) {
    HVX_Vector *tile =
      (HVX_Vector *)(c->vb + c->L.q_ah + d * HEXKL_ATTN_Q2_TILE);
    for (uint32_t r = 0; r < HEXKL_ATTN_Q2_ROWS; r += 4u) {
      HVX_Vector w[4];
      for (uint32_t i = 0; i < 4u; ++i) {
        const uint32_t rr = r + i;
        if (rr < n_rows) {
          const float *p = c->io->q + (size_t)(q0 + rr) * c->io->q_stride +
                           h * c->hd + 32u * d;
          HVX_Vector q =
            hvx_sf_to_w_rne(Q6_Vsf_vmpy_VsfVsf(hvx_tile_load_u(p), inv));
          q = Q6_Vw_vmin_VwVw(Q6_Vw_vmax_VwVw(q, lo), hi);
          w[i] = Q6_Vw_vadd_VwVw(q, bias);
        } else {
          w[i] = bias;
        }
      }
      const HVX_Vector h01 = Q6_Vh_vpack_VwVw_sat(w[1], w[0]);
      const HVX_Vector h23 = Q6_Vh_vpack_VwVw_sat(w[3], w[2]);
      tile[r / 4u] = Q6_Vub_vpack_VhVh_sat(h23, h01);
    }
  }
}

/* ---- resident K/V and the per-head constants ---------------------------- */

static void dma_resident(ctx *c, uint32_t n) {
  const hexkl_kv_q *kv = c->io->kv;
  const uint32_t row_bytes = c->dt * 1024u;
  const size_t src = hexkl_kv_q_tile_off(kv, n, c->res.lo, 0);
  hexkl_dma_ring_push2d(c->vb + c->L.kt_wh, kv->kt + src, row_bytes, row_bytes,
                        row_bytes, c->res.n, 0, 1);
  hexkl_dma_ring_push2d(c->vb + c->L.v_wh, kv->v + src, row_bytes, row_bytes,
                        row_bytes, c->res.n, 0, 1);
  hexkl_dma_ring_drain();
}

/** @brief corr[c][j] = rint(128 * colsum_k[32c + j] * 2^k / 512) mod 2^16
 *         for the resident tiles, built in cached memory, copied to VTCM. */
static void build_corr(ctx *c, uint32_t n, int k) {
  const hexkl_kv_q *kv = c->io->kv;
  const float f = ldexpf(128.0f, k - 9);
  const uint32_t n_vals = c->res.n * 32u;
  const int32_t *cs =
    kv->colsum_k + hexkl_kv_q_sk_index(kv, n, 32u * c->res.lo);
  for (uint32_t i = 0; i < n_vals; ++i) {
    const float v = rintf((float)cs[i] * f);
    const int32_t vi = (int32_t)v;
    c->corr_heap[i] = (int16_t)(uint16_t)(vi & 0xffff);
  }
  memset(c->corr_heap + n_vals, 0, 32u * sizeof(int16_t));
  vcopy(c->vb + c->L.corr, c->corr_heap,
        hexkl_attn_round_up((c->res.n + 1u) * 64u, 128u));
}

/* ---- stages 1 and 3: HMX with the convert readout ------------------------ */

/** @brief The current accumulator -> int16 tile @a dst through blocks
 *         blk_lo / blk_hi (scales s and s/256). */
static inline void acc_to_i16(ctx *c, const uint8_t *blk_lo,
                              const uint8_t *blk_hi, int16_t *dst) {
  uint8_t *p0 = c->vb + c->L.planes;
  uint8_t *p1 = p0 + HEXKL_CVT_PLANE_BYTES;
  hexkl_cvt_issue(blk_lo, p0);
  hexkl_cvt_issue(blk_hi, p1);
  hexkl_cvt_zip_i16(p0, p1, dst);
}

static void stage_qk(ctx *c, const hexkl_attn_q2_range *blk) {
  const uint8_t *q_ah = c->vb + c->L.q_ah;
  const uint8_t *blk_lo = c->vb + c->L.cvt;
  const uint8_t *blk_hi = blk_lo + HEXKL_CVT_BLOCK_BYTES;
  for (uint32_t t = 0; t < blk->n; ++t) {
    const uint32_t ri = blk->lo + t - c->res.lo;
    const uint8_t *kt = c->vb + c->L.kt_wh + (size_t)ri * c->dt * 1024u;
    hexkl_micro_hmx_acc_clear_int32();
    for (uint32_t d = 0; d < c->dt; ++d) {
      hexkl_hmx_mm_u8i8(q_ah + d * HEXKL_ATTN_Q2_TILE, kt + d * 1024u);
    }
    acc_to_i16(
      c, blk_lo, blk_hi,
      (int16_t *)(c->vb + c->L.s16 + (size_t)t * HEXKL_ATTN_Q2_TILE16));
  }
}

static void stage_pv(ctx *c, const hexkl_attn_q2_range *blk) {
  const uint8_t *p_ah = c->vb + c->L.p_ah;
  const uint8_t *blk_lo = c->vb + c->L.cvt + 2u * HEXKL_CVT_BLOCK_BYTES;
  const uint8_t *blk_hi = blk_lo + HEXKL_CVT_BLOCK_BYTES;
  const uint8_t *v0 =
    c->vb + c->L.v_wh + (size_t)(blk->lo - c->res.lo) * c->dt * 1024u;
  for (uint32_t d = 0; d < c->dt; ++d) {
    hexkl_micro_hmx_acc_clear_int32();
    for (uint32_t t = 0; t < blk->n; ++t) {
      hexkl_hmx_mm_u8i8(p_ah + t * HEXKL_ATTN_Q2_TILE,
                        v0 + ((size_t)t * c->dt + d) * 1024u);
    }
    acc_to_i16(
      c, blk_lo, blk_hi,
      (int16_t *)(c->vb + c->L.o16 + (size_t)d * HEXKL_ATTN_Q2_TILE16));
  }
}

/* ---- stage 4: epilogue --------------------------------------------------- */

/** @brief Largest row sum of the block, from the vector layout. */
static uint32_t rowsum_max(ctx *c) {
  const HVX_Vector *rs = (const HVX_Vector *)(c->vb + c->L.rowsum);
  HVX_Vector m = rs[0];
  for (uint32_t w = 1; w < 16u; ++w) {
    m = Q6_Vw_vmax_VwVw(m, rs[w]);
  }
  for (int rot = 4; rot <= 64; rot <<= 1) {
    m = Q6_Vw_vmax_VwVw(m, Q6_V_vror_VR(m, rot));
  }
  return hvx_attn_word0(m);
}

/** @brief o16[d][r][j] * (512 / 2^kv) * s_v[d][j] / rowsum[r] -> out. */
static void stage_epilogue(ctx *c, uint32_t h, uint32_t n, uint32_t q0,
                           uint32_t n_rows, int kv_exp) {
  // Row sums to cached memory: 16 vector stores, then cheap scalar reads.
  vcopy(c->rs_heap, c->vb + c->L.rowsum, HVX_SOFTMAX_Q_ROWSUM_WORDS * 4u);
  const float back = ldexpf(512.0f, -kv_exp);
  const float *sv = c->io->kv->fs_v + (size_t)n * c->hd;
  for (uint32_t d = 0; d < c->dt; ++d) {
    const HVX_Vector svv = hvx_tile_load_u(sv + 32u * d);
    const HVX_Vector *o =
      (const HVX_Vector *)(c->vb + c->L.o16 + (size_t)d * HEXKL_ATTN_Q2_TILE16);
    for (uint32_t v = 0; v < 32u; ++v) {
      const uint32_t r0 = 2u * v;
      if (r0 >= n_rows) {
        break;
      }
      const HVX_VectorPair w = Q6_Ww_vunpack_Vh(o[v]);
      for (uint32_t i = 0; i < 2u && r0 + i < n_rows; ++i) {
        const int32_t rsum = c->rs_heap[hvx_softmax_q_rowsum_index(r0 + i)];
        const float inv = rsum > 0 ? back / (float)rsum : 0.0f;
        HVX_Vector f = Q6_Vsf_equals_Vw(i == 0 ? Q6_V_lo_W(w) : Q6_V_hi_W(w));
        f = Q6_Vsf_vmpy_VsfVsf(f, hvx_splat_sf(inv));
        f = Q6_Vsf_vmpy_VsfVsf(f, svv);
        float *dst = c->io->out + (size_t)(q0 + r0 + i) * c->io->out_stride +
                     h * c->hd + 32u * d;
        hvx_tile_store_u(dst, f);
      }
    }
  }
}

/* ---- the kernel ---------------------------------------------------------- */

int hexkl_attn_q2_prefill(uint8_t *vtcm_base, uint32_t arena_top,
                          const hexkl_attn_f16_shape *s,
                          const hexkl_attn_q2_io *io, hvx_worker_pool *pool,
                          hexkl_attn_q2_stats *st) {
  (void)pool;
  if (!vtcm_base || !s || !io || !io->q || !io->out || !io->kv) {
    return AEE_EBADPARM;
  }
  const hexkl_kv_q *kv = io->kv;
  if (!kv->fixed || kv->kind != HEXKL_KV_Q8 || kv->head_dim != s->head_dim ||
      kv->n_head_kv != s->n_head_kv || s->cache_to > kv->max_rows) {
    return AEE_EBADPARM;
  }
  ctx c;
  memset(&c, 0, sizeof(c));
  c.vb = vtcm_base;
  c.s = s;
  c.io = io;
  if (hexkl_attn_q2_plan(s, arena_top, &c.L) != HEXKL_ATTN_OK) {
    return AEE_ENOMEMORY;
  }
  c.hd = s->head_dim;
  c.dt = c.hd / 32u;
  c.G = s->n_head_q / s->n_head_kv;
  c.res = hexkl_attn_q2_cols(s->cache_from, s->cache_from + s->n_q - 1u,
                             s->cache_to, s->window);
  // 128-byte aligned: rs_heap is written with vector stores.
  c.corr_heap = (int16_t *)memalign(128u, ((size_t)c.res.n + 1u) * 64u + 128u);
  c.rs_heap = (int32_t *)memalign(128u, HVX_SOFTMAX_Q_ROWSUM_WORDS * 4u);
  if (!c.corr_heap || !c.rs_heap) {
    free(c.corr_heap);
    free(c.rs_heap);
    return AEE_ENOMEMORY;
  }
  const uint64_t t_start = now_us();
  const uint64_t c_start = HAP_perf_get_pcycles();
  const float logit_scale = 1.0f / sqrtf((float)c.hd);
  int rc = AEE_SUCCESS;

  hexkl_dma_ring_reset();
  for (uint32_t n = 0; n < s->n_head_kv && rc == AEE_SUCCESS; ++n) {
    uint64_t t0 = now_us();
    dma_resident(&c, n);
    c.st.us_dma += now_us() - t0;
    for (uint32_t g = 0; g < c.G && rc == AEE_SUCCESS; ++g) {
      const uint32_t h = n * c.G + g;
      // Per-head Q scale, convert scale 2^k and residual rho.
      t0 = now_us();
      float amax = q_amax(&c, h);
      const float s_q = amax > 0.0f ? amax / 127.0f : 1.0f;
      c.st.us_qprep += now_us() - t0;
      const float alpha = s_q * kv->fs_k[n] * LOG2E * logit_scale;
      int k = 0;
      uint16_t rho = 0;
      if (hvx_softmax_q_scale(alpha, FRAC_BITS, &k, &rho) != 0) {
        rc = AEE_EBADPARM;
        break;
      }
      t0 = now_us();
      set_cvt_block(c.vb + c.L.cvt, hexkl_cvt_f32_to_hf(ldexpf(1.0f, k)));
      set_cvt_block(c.vb + c.L.cvt + HEXKL_CVT_BLOCK_BYTES,
                    hexkl_cvt_f32_to_hf(ldexpf(1.0f, k - 8)));
      build_corr(&c, n, k);
      c.st.us_dma += now_us() - t0;

      for (uint32_t q0 = 0; q0 < s->n_q; q0 += HEXKL_ATTN_Q2_ROWS) {
        const uint32_t n_rows =
          s->n_q - q0 < HEXKL_ATTN_Q2_ROWS ? s->n_q - q0 : HEXKL_ATTN_Q2_ROWS;
        const uint32_t row0 = s->cache_from + q0;
        const hexkl_attn_q2_range blk = hexkl_attn_q2_cols(
          row0, row0 + HEXKL_ATTN_Q2_ROWS - 1u, s->cache_to, s->window);

        t0 = now_us();
        q_quant_block(&c, h, q0, n_rows, s_q);
        uint64_t t1 = now_us();
        c.st.us_qprep += t1 - t0;

        stage_qk(&c, &blk);
        t0 = now_us();
        c.st.us_qk += t0 - t1;

        hvx_softmax_q_block b;
        b.n_col_tiles = blk.n;
        b.col0 = 32u * blk.lo;
        b.n_cols = s->cache_to - 32u * blk.lo;
        b.row0 = row0;
        b.n_rows = n_rows;
        b.window = s->window;
        b.rho_q15 = rho;
        b.frac_bits = FRAC_BITS;
        hvx_softmax_q(&b, (const int16_t *)(c.vb + c.L.s16),
                      (const int16_t *)(c.vb + c.L.corr) +
                        (size_t)(blk.lo - c.res.lo) * 32u,
                      c.vb + c.L.p_ah, (int32_t *)(c.vb + c.L.rowsum),
                      c.vb + c.L.smx);
        t1 = now_us();
        c.st.us_softmax += t1 - t0;

        // 2^kv / 512 * 127 * rowsum_max <= 32767.
        const uint32_t rsmax = rowsum_max(&c);
        int kv_exp = 0;
        if (rsmax > 0) {
          kv_exp =
            (int)floorf(log2f(32767.0f * 512.0f / (127.0f * (float)rsmax)));
        }
        set_cvt_block(c.vb + c.L.cvt + 2u * HEXKL_CVT_BLOCK_BYTES,
                      hexkl_cvt_f32_to_hf(ldexpf(1.0f, kv_exp)));
        set_cvt_block(c.vb + c.L.cvt + 3u * HEXKL_CVT_BLOCK_BYTES,
                      hexkl_cvt_f32_to_hf(ldexpf(1.0f, kv_exp - 8)));
        stage_pv(&c, &blk);
        t0 = now_us();
        c.st.us_pv += t0 - t1;

        stage_epilogue(&c, h, n, q0, n_rows, kv_exp);
        c.st.us_epi += now_us() - t0;
        ++c.st.n_blocks;
      }
    }
  }
  c.st.us_total = now_us() - t_start;
  c.st.pcycles = HAP_perf_get_pcycles() - c_start;
  if (st) {
    *st = c.st;
  }
  free(c.corr_heap);
  free(c.rs_heap);
  return rc;
}
