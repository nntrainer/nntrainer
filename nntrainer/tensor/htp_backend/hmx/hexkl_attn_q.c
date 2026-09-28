// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file   hexkl_attn_q.c
 * @date   28 Sep 2026
 * @brief  A8W8 / A8W4 flash attention on HMX over a quantized KV cache
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * hexkl_attn_f16's FA-2 pipeline with the operands on HMX's 8-bit ports:
 *
 *   Qu     = per-row asymmetric uint8 of Q         (hvx_quant_u8)
 *   S      = (Qu.Kq^T - zp*colsum_k) * s_q*s_k     int32 acc -> f32 -> fp16
 *   P      = online softmax of S                    (shared fp16 routine)
 *   P'_d   = uint8 of P * s_v[k][d], per row, per 32-dim group d
 *   O_d    = a*O_d + s_p[d] * (P'_d.Vq_d)           int32 acc -> f32, on HVX
 *   out    = O / l
 *
 * The V scale is per cache row -- the reduction axis of P.V -- so it
 * cannot be a weight-side dequant constant; folding it into P before P's
 * own quantization is what lets V carry per-token, per-group scales on an
 * int accumulator. Everything the accumulator produces is read in place
 * (hexkl_acc_tile: a plain row-major 64x32 int32 tile on every part
 * measured) and never leaves VTCM until the final store. S / P tiles are
 * the fp16 kernel's layout so its softmax runs unmodified. See
 * docs/backend_guide/htp_backend/21_quantized_attention_plan.md.
 *
 * Pipeline per (kv head, q block), as the fp16 kernel: with block k's
 * scores in S[k&1], the pool runs its softmax while the calling thread
 * lands block k+1's tiles, runs its Q.K^T and dequantizes into S[(k+1)&1];
 * then P' quantization on the pool, and P'.V plus the f32 O update on the
 * calling thread.
 */

#include "hexkl_attn_q.h"

#include <math.h>
#include <string.h>

#include <AEEStdErr.h>
#include <HAP_perf.h>

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include "hexkl_acc_tile.h"
#include "hexkl_dma_ring.h"
#include "hexkl_micro.h"
#include "hvx_attn_softmax_f16.h"
#include "hvx_convert.h"
#include "hvx_quant_u8.h"
#include "hvx_tile_f16.h"

#define TB HEXKL_ATTN_TILE_BYTES
#define LOG2E 1.4426950408889634f
/** @brief Row cap; the meta region is sized from g_br, this bounds the
 *         per-block stack scratch. */
#define MAX_G_BR 256u
/** @brief Vectors per 2 KiB tile. */
#define TILE_VECS (TB / sizeof(HVX_Vector))

static int map_plan_rc(int rc) {
  switch (rc) {
  case HEXKL_ATTN_OK:
    return AEE_SUCCESS;
  case HEXKL_ATTN_ENOMEM:
    return AEE_ENOMEMORY;
  default:
    return AEE_EBADPARM;
  }
}

typedef struct {
  uint8_t *vb;
  uint32_t cfg;
  const hexkl_attn_f16_shape *s;
  const hexkl_attn_f16_tiling *t;
  const hexkl_attn_q_io *io;
  const hexkl_kv_q *kv;
  hvx_worker_pool *pool;
  hexkl_attn_q_layout L;
  const hexkl_acc_layout *acc_layout;
  float scale;    /**< log2e / sqrt(hd), folded into the S dequant */
  float *q_scale; /**< [g_br] uint8 Q scale (times c->scale below) */
  int32_t *q_zp;  /**< [g_br] */
  float *l_f32;   /**< [g_br] running sum */
  float *a_f32;   /**< [g_br] this block's rescale, f32 */
  uint16_t *m_hf; /**< [g_br] running max, fp16 bits */
  uint16_t *a_hf; /**< [g_br] this block's rescale, fp16 bits */
  float *p_scale; /**< [dt][g_br] uint8 P' scale per V group */
  HVX_Vector col_idx;
  HVX_Vector cap_hf, inv_cap_hf;
  int softcap_on;
  /** per-block parameters handed to the pool workers; frozen while a job
      is in flight */
  uint32_t w_qb, w_kb, w_buf, w_sbuf;
  int w_masked;
  hexkl_attn_q_stats *st;
} attn_ctx;

/* --- addressing ------------------------------------------------------- */

static inline uint32_t off_of(const attn_ctx *c, const void *p) {
  return (uint32_t)((const uint8_t *)p - c->vb);
}
static inline uint8_t *q_tile(const attn_ctx *c, uint32_t i, uint32_t d) {
  return c->vb + c->L.q_ah + (i * c->L.n_dot_tiles + d) * TB;
}
static inline uint8_t *kt_tile(const attn_ctx *c, uint32_t buf, uint32_t col,
                               uint32_t d) {
  return c->vb + c->L.kt_wh[buf] +
         (col * c->L.n_dot_tiles + d) * c->L.tile_bytes;
}
static inline uint8_t *v_tile(const attn_ctx *c, uint32_t buf, uint32_t col,
                              uint32_t d) {
  return c->vb + c->L.v_wh[buf] +
         (col * c->L.n_dot_tiles + d) * c->L.tile_bytes;
}
/** @brief fp16 score tile (i32 in [0, 2*rt)) of S[sbuf]. */
static inline HVX_Vector *s_tile(const attn_ctx *c, uint32_t sbuf, uint32_t i32,
                                 uint32_t col) {
  return (HVX_Vector *)(c->vb + c->L.s_hf[sbuf] +
                        (i32 * c->L.n_col_tiles + col) * TB);
}
static inline uint8_t *p_tile(const attn_ctx *c, uint32_t d, uint32_t i,
                              uint32_t col) {
  return c->vb + c->L.p_ah +
         ((d * c->L.n_row_tiles + i) * c->L.n_col_tiles + col) * TB;
}
static inline int32_t *acc_tile(const attn_ctx *c) {
  return (int32_t *)(c->vb + c->L.acc);
}
static inline float *o_row(const attn_ctx *c, uint32_t r) {
  return (float *)(c->vb + c->L.o_f32) + (size_t)r * c->s->head_dim;
}
static inline float *qx_row(const attn_ctx *c, uint32_t r) {
  return (float *)(c->vb + c->L.qx_f32) + (size_t)r * c->s->head_dim;
}
static inline float *sv_blk(const attn_ctx *c, uint32_t buf, uint32_t d) {
  return (float *)(c->vb + c->L.sv_blk[buf]) + (size_t)d * c->t->bc;
}

static inline uint64_t now_us(void) { return HAP_perf_get_time_us(); }

static inline void unit_range(uint32_t n, uint32_t n_threads, uint32_t i,
                              uint32_t *lo, uint32_t *hi) {
  *lo = (uint32_t)((uint64_t)n * i / n_threads);
  *hi = (uint32_t)((uint64_t)n * (i + 1) / n_threads);
}

/** @brief Column tiles of the registry that block kb actually covers. */
static inline uint32_t resident_cols(const attn_ctx *c, uint32_t kb) {
  const uint32_t ct = c->L.n_col_tiles;
  const uint32_t c0 = kb * ct;
  if (c0 >= c->kv->n_col_tiles) {
    return 0;
  }
  const uint32_t left = c->kv->n_col_tiles - c0;
  return left < ct ? left : ct;
}

/* --- phases ----------------------------------------------------------- */

/**
 * @brief Q rows of (kv head n, query block qb) -> uint8 AH tiles with
 *        per-row scale and zero point, and the softmax state reset.
 *
 * Tile row r is query row qb*br + r/G, head n*G + r%G, gathered into a
 * contiguous f32 block so hvx_quant_u8's row quantizer runs as it does for
 * the FC layers; pad rows are zero, so they quantize to 0 with scale 1.
 */
static int phase_qprep(attn_ctx *c, uint32_t n, uint32_t qb) {
  const uint32_t hd = c->s->head_dim;
  const uint32_t g_br = c->t->g_br;
  for (uint32_t r = 0; r < g_br; ++r) {
    uint32_t q, g;
    float *dst = qx_row(c, r);
    if (hexkl_attn_f16_tile_row_to_qg(c->s, c->t, qb, r, &q, &g)) {
      memcpy(dst,
             c->io->q + (size_t)q * c->io->q_stride +
               (size_t)(n * c->t->g + g) * hd,
             (size_t)hd * sizeof(float));
    } else {
      memset(dst, 0, (size_t)hd * sizeof(float));
    }
  }
  hvx_quant_rows_u8_params(qx_row(c, 0), g_br, g_br, hd, c->q_scale, c->q_zp,
                           c->pool);
  int rc = hvx_quant_pack_u8_ah(qx_row(c, 0), g_br, g_br, hd, c->q_scale,
                                c->q_zp, q_tile(c, 0, 0), c->pool);
  if (rc != AEE_SUCCESS) {
    return rc;
  }
  // Fold the logit scale into the dequant multiplier.
  for (uint32_t r = 0; r < g_br; ++r) {
    c->q_scale[r] *= c->scale;
  }

  for (uint32_t r = 0; r < g_br; ++r) {
    uint32_t q, g;
    if (c->io->sinks &&
        hexkl_attn_f16_tile_row_to_qg(c->s, c->t, qb, r, &q, &g)) {
      c->m_hf[r] = hvx_attn_f32_to_hf(c->io->sinks[n * c->t->g + g] * LOG2E);
      c->l_f32[r] = 1.0f;
    } else {
      c->m_hf[r] = (uint16_t)HVX_ATTN_HF_NEG_INF;
      c->l_f32[r] = 0.0f;
    }
  }
  return AEE_SUCCESS;
}

/** @brief Queues the DMA of the ready tiles of block kb of head n. */
static void dma_issue(attn_ctx *c, uint32_t n, uint32_t kb, uint32_t buf) {
  const uint32_t ncol = resident_cols(c, kb);
  if (ncol == 0) {
    return;
  }
  const uint32_t dt = c->L.n_dot_tiles;
  const uint32_t tb = c->L.tile_bytes;
  const size_t src = hexkl_kv_q_tile_off(c->kv, n, kb * c->L.n_col_tiles, 0);
  hexkl_dma_ring_push2d(kt_tile(c, buf, 0, 0), c->kv->kt + src, tb, tb, tb,
                        ncol * dt, /*src_vtcm=*/0, /*dst_vtcm=*/1);
  hexkl_dma_ring_push2d(v_tile(c, buf, 0, 0), c->kv->v + src, tb, tb, tb,
                        ncol * dt, /*src_vtcm=*/0, /*dst_vtcm=*/1);
}

/**
 * @brief Waits for the DMA, zeroes the tiles past the registry's last
 *        column (finite zeros for HMX), and gathers the block's per-row
 *        V scales into [dt][bc] so the P' quantizer reads them as vectors.
 */
static void dma_land(attn_ctx *c, uint32_t n, uint32_t kb, uint32_t buf) {
  hexkl_dma_ring_drain();
  const uint32_t ncol = resident_cols(c, kb);
  const uint32_t ct = c->L.n_col_tiles;
  const uint32_t dt = c->L.n_dot_tiles;
  if (ncol < ct) {
    const size_t bytes = (size_t)(ct - ncol) * dt * c->L.tile_bytes;
    memset(kt_tile(c, buf, ncol, 0), 0, bytes);
    memset(v_tile(c, buf, ncol, 0), 0, bytes);
  }
  const uint32_t k0 = kb * c->t->bc;
  const uint32_t rows = ncol * 32u;
  for (uint32_t d = 0; d < dt; ++d) {
    float *sv = sv_blk(c, buf, d);
    for (uint32_t j = 0; j < rows; ++j) {
      sv[j] = c->kv->s_v[hexkl_kv_q_sv_index(c->kv, n, k0 + j, d)];
    }
    for (uint32_t j = rows; j < c->t->bc; ++j) {
      sv[j] = 1.0f;
    }
  }
}

/**
 * @brief The int32 accumulator of output tile (i, col) -> fp16 score tiles
 *        (2i, col) and (2i+1, col) of S[sbuf], dequantized per row:
 *        (acc - zp*colsum_k) * (s_q*log2e/sqrt(hd)) * s_k.
 *
 * Exact up to the f32 products: |acc| and |zp*colsum| are below 2^24 for
 * head_dim <= 256. Two rows go into one tile vector through the widening
 * narrow (low vector -> even lanes), as the fp16 tile builders do.
 */
static void dequant_s(attn_ctx *c, uint32_t n, uint32_t kb, uint32_t i,
                      uint32_t col, uint32_t sbuf) {
  const int32_t *acc = acc_tile(c);
  const hexkl_acc_layout *AL = c->acc_layout;
  const uint32_t k0 = kb * c->t->bc + 32u * col;
  HVX_Vector sk, cs;
  if (k0 + 32u <= c->kv->max_rows) {
    sk = hvx_tile_load_u(c->kv->s_k + hexkl_kv_q_sk_index(c->kv, n, k0));
    cs = Q6_Vsf_equals_Vw(
      hvx_tile_load_u(c->kv->colsum_k + hexkl_kv_q_sk_index(c->kv, n, k0)));
  } else {
    // Past the registry: the tiles were zeroed, the lanes will be masked;
    // keep the arithmetic finite.
    sk = Q6_V_vzero();
    cs = Q6_V_vzero();
  }
  for (uint32_t h = 0; h < 2u; ++h) {
    HVX_Vector *dst = s_tile(c, sbuf, 2u * i + h, col);
    for (uint32_t v = 0; v < TILE_VECS; ++v) {
      HVX_Vector q[2];
      for (uint32_t p = 0; p < 2u; ++p) {
        const uint32_t r = 32u * h + 2u * v + p;
        const uint32_t R = HEXKL_ATTN_Q_ROWS * i + r;
        const HVX_Vector a = Q6_Vsf_equals_Vw(
          hvx_tile_load_u(acc + AL->base + r * AL->row_stride));
        const HVX_Vector zc =
          Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_equals_Vw(Q6_V_vsplat_R(c->q_zp[R])), cs);
        const HVX_Vector corr = Q6_Vsf_vsub_VsfVsf(a, zc);
        const HVX_Vector scaled =
          Q6_Vsf_vmpy_VsfVsf(corr, hvx_splat_sf(c->q_scale[R]));
        q[p] = Q6_Vqf32_vmpy_VsfVsf(scaled, sk);
      }
      dst[v] = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(q[1], q[0]));
    }
  }
}

/** @brief S[sbuf] = Q.K^T for the block in @a buf, dequantized. */
static int phase_qk(attn_ctx *c, uint32_t n, uint32_t kb, uint32_t buf,
                    uint32_t sbuf) {
  const int q4 = c->kv->kind == HEXKL_KV_Q4;
  for (uint32_t i = 0; i < c->L.n_row_tiles; ++i) {
    for (uint32_t col = 0; col < c->L.n_col_tiles; ++col) {
      uint64_t t0 = now_us();
      hexkl_micro_hmx_acc_clear_int32();
      for (uint32_t d = 0; d < c->L.n_dot_tiles; ++d) {
        const uint32_t ao = off_of(c, q_tile(c, i, d));
        const uint32_t wo = off_of(c, kt_tile(c, buf, col, d));
        int rc = q4 ? hexkl_micro_hmx_mm_u8i4(c->vb, ao, wo)
                    : hexkl_micro_hmx_mm_u8i8(c->vb, ao, wo);
        if (rc != AEE_SUCCESS) {
          return rc;
        }
      }
      int rc = hexkl_micro_hmx_acc_read_int32(c->vb, c->cfg, c->L.acc);
      if (rc != AEE_SUCCESS) {
        return rc;
      }
      uint64_t t1 = now_us();
      c->st->us_qk += t1 - t0;
      dequant_s(c, n, kb, i, col, sbuf);
      c->st->us_dequant += now_us() - t1;
    }
  }
  return AEE_SUCCESS;
}

/** @brief One fp16 row pair of S[w_sbuf] through the shared online softmax. */
static void softmax_row_pair(attn_ctx *c, uint32_t i32, uint32_t v) {
  const uint32_t r0 = 32u * i32 + 2u * v;
  hvx_attn_softmax_pair_args a;
  a.n_col_tiles = c->L.n_col_tiles;
  a.k0 = c->w_kb * c->t->bc;
  a.masked = c->w_masked;
  a.lo_pair = Q6_V_vzero();
  a.hi_pair = Q6_V_vzero();
  if (a.masked) {
    uint32_t lo[2] = {0, 0}, hi[2] = {0xFFFFu, 0xFFFFu};
    for (uint32_t p = 0; p < 2u; ++p) {
      uint32_t q, g;
      if (hexkl_attn_f16_tile_row_to_qg(c->s, c->t, c->w_qb, r0 + p, &q, &g)) {
        hexkl_attn_f16_row_range(c->s, q, &lo[p], &hi[p]);
      }
    }
    a.lo_pair = hvx_attn_splat_pair_hf(lo[0], lo[1]);
    a.hi_pair = hvx_attn_splat_pair_hf(hi[0], hi[1]);
  }
  a.col_idx = c->col_idx;
  a.softcap_on = c->softcap_on;
  a.cap_hf = c->cap_hf;
  a.inv_cap_hf = c->inv_cap_hf;
  hvx_attn_online_softmax_pair(s_tile(c, c->w_sbuf, i32, 0) + v, TILE_VECS, &a,
                               c->m_hf + r0, c->a_hf + r0, c->l_f32 + r0);
}

static void softmax_worker(uint32_t n_threads, uint32_t i, void *ctx_) {
  attn_ctx *c = (attn_ctx *)ctx_;
  const uint32_t n = 2u * c->L.n_row_tiles * TILE_VECS;
  uint32_t lo, hi;
  unit_range(n, n_threads, i, &lo, &hi);
  for (uint32_t u = lo; u < hi; ++u) {
    softmax_row_pair(c, u / TILE_VECS, u % TILE_VECS);
  }
}

static int softmax_start(attn_ctx *c, uint32_t qb, uint32_t kb, uint32_t sbuf,
                         int masked) {
  c->w_qb = qb;
  c->w_kb = kb;
  c->w_sbuf = sbuf;
  c->w_masked = masked;
  const uint32_t n = 2u * c->L.n_row_tiles * TILE_VECS;
  return hvx_worker_pool_submit(c->pool, softmax_worker, c, n);
}

/**
 * @brief P' tiles of V group d for the 32 rows of fp16 row tile i32:
 *        P'[r][k] = P[r][k] * s_v[k][d], per-row scale = row max / 255,
 *        uint8 round-to-nearest-even, stored flat into the 64-row tile.
 *
 * Four rows (two fp16 vectors) at a time: their 4 x 32 bytes are one
 * vector store, the same packing hvx_quant_pack_u8_ah uses.
 */
static void pquant_unit(attn_ctx *c, uint32_t i32, uint32_t d) {
  const HVX_Vector one_hf = Q6_Vh_vsplat_R((int)HVX_ATTN_HF_ONE);
  const HVX_Vector zero = Q6_V_vzero();
  const uint32_t ct = c->L.n_col_tiles;
  const uint32_t i64 = i32 / 2u;
  const uint32_t rbase = 32u * (i32 & 1u);
  const float *sv = sv_blk(c, c->w_buf, d);
  float *ps = c->p_scale + (size_t)d * c->t->g_br + HEXKL_ATTN_Q_ROWS * i64;

  for (uint32_t v = 0; v < TILE_VECS; v += 2u) {
    // Pass 1: row max of P * s_v over the block, rows 2v..2v+3.
    HVX_Vector mx[4] = {zero, zero, zero, zero};
    for (uint32_t col = 0; col < ct; ++col) {
      const HVX_Vector svv = hvx_tile_load_u(sv + 32u * col);
      for (uint32_t j = 0; j < 2u; ++j) {
        const HVX_VectorPair w = Q6_Wqf32_vmpy_VhfVhf(
          *(s_tile(c, c->w_sbuf, i32, col) + v + j), one_hf);
        const HVX_Vector e =
          Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(Q6_V_lo_W(w)), svv);
        const HVX_Vector o =
          Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(Q6_V_hi_W(w)), svv);
        mx[2u * j] = Q6_Vsf_vmax_VsfVsf(mx[2u * j], e);
        mx[2u * j + 1u] = Q6_Vsf_vmax_VsfVsf(mx[2u * j + 1u], o);
      }
    }
    // A row whose P' is all zero (fully masked so far) keeps scale 1 and
    // quantizes to zeros; the max lane of any other row maps to 255.
    HVX_Vector inv[4];
    for (uint32_t j = 0; j < 4u; ++j) {
      const float m = hvx_attn_lane0_f32(hvx_attn_max32_sf(mx[j]));
      const float scale = m > 0.0f ? m * (1.0f / 255.0f) : 1.0f;
      ps[rbase + 2u * v + j] = scale;
      inv[j] = hvx_splat_sf(1.0f / scale);
    }
    // Pass 2: quantize and pack.
    for (uint32_t col = 0; col < ct; ++col) {
      const HVX_Vector svv = hvx_tile_load_u(sv + 32u * col);
      HVX_Vector vq[4];
      for (uint32_t j = 0; j < 2u; ++j) {
        const HVX_VectorPair w = Q6_Wqf32_vmpy_VhfVhf(
          *(s_tile(c, c->w_sbuf, i32, col) + v + j), one_hf);
        const HVX_Vector e =
          Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(Q6_V_lo_W(w)), svv);
        const HVX_Vector o =
          Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(Q6_V_hi_W(w)), svv);
        vq[2u * j] = hvx_sf_to_w_rne(Q6_Vsf_vmpy_VsfVsf(e, inv[2u * j]));
        vq[2u * j + 1u] =
          hvx_sf_to_w_rne(Q6_Vsf_vmpy_VsfVsf(o, inv[2u * j + 1u]));
      }
      const HVX_Vector h01 = Q6_Vh_vpack_VwVw_sat(vq[1], vq[0]);
      const HVX_Vector h23 = Q6_Vh_vpack_VwVw_sat(vq[3], vq[2]);
      const HVX_Vector b = Q6_Vub_vpack_VhVh_sat(h23, h01);
      hvx_tile_store_u(p_tile(c, d, i64, col) + (rbase + 2u * v) * 32u, b);
    }
  }
}

static void pquant_worker(uint32_t n_threads, uint32_t i, void *ctx_) {
  attn_ctx *c = (attn_ctx *)ctx_;
  const uint32_t n = 2u * c->L.n_row_tiles * c->L.n_dot_tiles;
  uint32_t lo, hi;
  unit_range(n, n_threads, i, &lo, &hi);
  for (uint32_t u = lo; u < hi; ++u) {
    pquant_unit(c, u / c->L.n_dot_tiles, u % c->L.n_dot_tiles);
  }
}

/** @brief Joins the softmax (exposed time), then P -> P' across the pool. */
static void softmax_finish_pquant(attn_ctx *c, uint32_t buf) {
  uint64_t t0 = now_us();
  hvx_worker_pool_wait(c->pool);
  uint64_t t1 = now_us();
  c->st->us_softmax += t1 - t0;
  for (uint32_t r = 0; r < c->t->g_br; ++r) {
    c->a_f32[r] = hvx_attn_hf_to_f32(c->a_hf[r]);
  }
  c->w_buf = buf;
  hvx_worker_pool_run(c->pool, pquant_worker, c,
                      2u * c->L.n_row_tiles * c->L.n_dot_tiles);
  c->st->us_pquant += now_us() - t1;
}

/**
 * @brief O_d = a*O_d + s_p[d] * (P'_d . V_d) for every output tile, the
 *        rescale and accumulate in f32 on HVX from the int32 accumulator.
 *        On the first visited block O has no past and is assigned.
 */
static int phase_pv(attn_ctx *c, uint32_t buf, int first) {
  const int q4 = c->kv->kind == HEXKL_KV_Q4;
  const int32_t *acc = acc_tile(c);
  const hexkl_acc_layout *AL = c->acc_layout;
  for (uint32_t i = 0; i < c->L.n_row_tiles; ++i) {
    for (uint32_t d = 0; d < c->L.n_dot_tiles; ++d) {
      uint64_t t0 = now_us();
      hexkl_micro_hmx_acc_clear_int32();
      for (uint32_t col = 0; col < c->L.n_col_tiles; ++col) {
        const uint32_t ao = off_of(c, p_tile(c, d, i, col));
        const uint32_t wo = off_of(c, v_tile(c, buf, col, d));
        int rc = q4 ? hexkl_micro_hmx_mm_u8i4(c->vb, ao, wo)
                    : hexkl_micro_hmx_mm_u8i8(c->vb, ao, wo);
        if (rc != AEE_SUCCESS) {
          return rc;
        }
      }
      int rc = hexkl_micro_hmx_acc_read_int32(c->vb, c->cfg, c->L.acc);
      if (rc != AEE_SUCCESS) {
        return rc;
      }
      uint64_t t1 = now_us();
      c->st->us_pv += t1 - t0;
      const float *ps = c->p_scale + (size_t)d * c->t->g_br;
      for (uint32_t r = 0; r < HEXKL_ATTN_Q_ROWS; ++r) {
        const uint32_t R = HEXKL_ATTN_Q_ROWS * i + r;
        float *o = o_row(c, R) + 32u * d;
        const HVX_Vector af = Q6_Vsf_equals_Vw(
          hvx_tile_load_u(acc + AL->base + r * AL->row_stride));
        HVX_Vector t = Q6_Vsf_vmpy_VsfVsf(af, hvx_splat_sf(ps[R]));
        if (!first) {
          t = Q6_Vsf_vadd_VsfVsf(
            t,
            Q6_Vsf_vmpy_VsfVsf(hvx_tile_load_u(o), hvx_splat_sf(c->a_f32[R])));
        }
        hvx_tile_store_u(o, t);
      }
      c->st->us_oupd += now_us() - t1;
    }
  }
  return AEE_SUCCESS;
}

/** @brief out rows = O / l, f32. */
static void phase_store(attn_ctx *c, uint32_t n, uint32_t qb) {
  const uint32_t hd = c->s->head_dim;
  for (uint32_t r = 0; r < c->t->g_br; ++r) {
    uint32_t q, g;
    if (!hexkl_attn_f16_tile_row_to_qg(c->s, c->t, qb, r, &q, &g)) {
      continue;
    }
    const HVX_Vector inv_l = hvx_splat_sf(1.0f / c->l_f32[r]);
    const float *o = o_row(c, r);
    float *dst = c->io->out + (size_t)q * c->io->out_stride +
                 (size_t)(n * c->t->g + g) * hd;
    for (uint32_t d = 0; d < c->L.n_dot_tiles; ++d) {
      hvx_tile_store_u(dst + 32u * d,
                       Q6_Vsf_vmpy_VsfVsf(hvx_tile_load_u(o + 32u * d), inv_l));
    }
  }
}

/* --- the block pipeline for one (kv head, query block) ---------------- */

static int run_q_block(attn_ctx *c, uint32_t n, uint32_t qb) {
  hexkl_attn_q_stats *st = c->st;
  uint32_t kb_lo, kb_hi;
  hexkl_attn_f16_kv_block_range(c->s, c->t, qb, &kb_lo, &kb_hi);
  if (kb_lo >= kb_hi) {
    return AEE_SUCCESS;
  }

  uint64_t t0 = now_us();
  dma_issue(c, n, kb_lo, kb_lo & 1u);
  dma_land(c, n, kb_lo, kb_lo & 1u);
  st->us_dma += now_us() - t0;
  int rc = phase_qk(c, n, kb_lo, kb_lo & 1u, 0);
  if (rc != AEE_SUCCESS) {
    return rc;
  }

  int first = 1;
  for (uint32_t kb = kb_lo; kb < kb_hi; ++kb) {
    const uint32_t buf = kb & 1u;
    const uint32_t sbuf = (kb - kb_lo) & 1u;
    const int has_next = (kb + 1u < kb_hi);

    t0 = now_us();
    if (has_next) {
      dma_issue(c, n, kb + 1u, buf ^ 1u);
    }
    const int masked = !hexkl_attn_f16_kv_block_unmasked(c->s, c->t, qb, kb);
    softmax_start(c, qb, kb, sbuf, masked);
    uint64_t t1 = now_us();
    st->us_softmax += t1 - t0;

    if (has_next) {
      dma_land(c, n, kb + 1u, buf ^ 1u);
      uint64_t t2 = now_us();
      st->us_dma += t2 - t1;
      rc = phase_qk(c, n, kb + 1u, buf ^ 1u, sbuf ^ 1u);
      if (rc != AEE_SUCCESS) {
        hvx_worker_pool_wait(c->pool);
        return rc;
      }
      t1 = now_us();
    }

    // The softmax of this block has to finish writing P before pquant
    // reads it; both happen here, workers joined in between.
    softmax_finish_pquant(c, buf);

    rc = phase_pv(c, buf, first);
    if (rc != AEE_SUCCESS) {
      return rc;
    }
    first = 0;
    st->n_blocks++;
  }

  t0 = now_us();
  phase_store(c, n, qb);
  st->us_store += now_us() - t0;
  return AEE_SUCCESS;
}

/* --- entry ------------------------------------------------------------ */

int hexkl_attn_q_prefill(uint8_t *vtcm_base, uint32_t config_off,
                         const hexkl_attn_f16_shape *s,
                         const hexkl_attn_f16_tiling *t,
                         const hexkl_attn_q_io *io, hvx_worker_pool *pool,
                         hexkl_attn_q_stats *st) {
  if (!vtcm_base || !s || !t || !io || !io->q || !io->out || !io->kv) {
    return AEE_EBADPARM;
  }
  const hexkl_kv_q *kv = io->kv;
  if (kv->n_head_kv != s->n_head_kv || kv->head_dim != s->head_dim ||
      kv->max_rows < s->cache_to) {
    return AEE_EBADPARM;
  }
  if (s->cache_to > 0xFFFFu || t->g_br > MAX_G_BR || s->softcap < 0.0f) {
    return AEE_EBADPARM;
  }

  attn_ctx c;
  memset(&c, 0, sizeof(c));
  c.vb = vtcm_base;
  c.cfg = config_off;
  c.s = s;
  c.t = t;
  c.io = io;
  c.kv = kv;
  c.pool = pool;
  int rc =
    map_plan_rc(hexkl_attn_q_plan(s, t, config_off, kv->tile_bytes, &c.L));
  if (rc != AEE_SUCCESS) {
    return rc;
  }
  c.acc_layout = hexkl_acc_layout_get(vtcm_base, c.L.acc);
  if (!c.acc_layout->usable) {
    return AEE_EUNSUPPORTED;
  }
  c.scale = LOG2E / sqrtf((float)s->head_dim);

  uint8_t *meta = vtcm_base + c.L.meta;
  const uint32_t g_br = t->g_br;
  c.q_scale = (float *)meta;
  c.q_zp = (int32_t *)(meta + 4u * g_br);
  c.l_f32 = (float *)(meta + 8u * g_br);
  c.a_f32 = (float *)(meta + 12u * g_br);
  c.m_hf = (uint16_t *)(meta + 16u * g_br);
  c.a_hf = (uint16_t *)(meta + 18u * g_br);
  c.p_scale = (float *)(meta + 20u * g_br);

  c.col_idx = hvx_attn_col_index_pairs();
  if (s->softcap > 0.0f) {
    const float cap = s->softcap * LOG2E;
    c.cap_hf = Q6_Vh_vsplat_R((int)hvx_attn_f32_to_hf(cap));
    c.inv_cap_hf = Q6_Vh_vsplat_R((int)hvx_attn_f32_to_hf(2.0f * LOG2E / cap));
    c.softcap_on = 1;
  }
  hexkl_dma_ring_reset();

  hexkl_attn_q_stats local;
  memset(&local, 0, sizeof(local));
  c.st = st ? st : &local;
  memset(c.st, 0, sizeof(*c.st));

  const uint32_t n_qb = hexkl_attn_f16_n_q_blocks(s, t);
  for (uint32_t n = 0; n < s->n_head_kv; ++n) {
    for (uint32_t qb = 0; qb < n_qb; ++qb) {
      const uint64_t t0 = now_us();
      rc = phase_qprep(&c, n, qb);
      if (rc != AEE_SUCCESS) {
        return rc;
      }
      c.st->us_qprep += now_us() - t0;
      rc = run_q_block(&c, n, qb);
      if (rc != AEE_SUCCESS) {
        return rc;
      }
    }
  }
  return AEE_SUCCESS;
}
