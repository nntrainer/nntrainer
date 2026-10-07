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
 *
 * Pipeline. The work items are (query head, 64-row block) in order. The
 * calling thread owns HMX and the DMA queue and runs the two matmul
 * stages plus the Q quantization (a VTCM pass over rows the DMA staged);
 * the pool workers run the softmax and the epilogue. Iteration i:
 *
 *   submit W(i) = { softmax(i), epilogue(i-1) rows 0-31, rows 32-63 }
 *   drain DMA; quantize Q(i+2) from its staging; push DMA for Q(i+3)
 *   and for the next KV head's K/V, a slice at a time
 *   QK(i+1)                                                   on HMX
 *   wait W(i)
 *   PV(i)                                                     on HMX
 *
 * Everything pushed to the DMA queue is consumed one iteration later, so
 * the drain never waits. Nothing is shared between a reader and a writer
 * in flight: Q tiles have three slots, Q staging two, scores, P', output
 * and row sums two, the per-head constants two by head parity. The next
 * KV head's K/V arrive under the current head when the plan found room
 * for a second buffer; otherwise the pipeline drains at the boundary.
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

/** @brief One (query head, row block). */
typedef struct {
  uint32_t n, h;   /**< kv head, query head */
  uint32_t q0;     /**< first query row, relative to cache_from */
  uint32_t n_rows; /**< valid rows, <= 64 */
  uint32_t row0;   /**< absolute position of row 0 */
  hexkl_attn_q2_range blk;
  uint32_t slot;  /**< i % 2: scores, P', output, row sums */
  uint32_t qslot; /**< i % 3: Q tiles */
  int kv_exp;     /**< PV convert exponent, set after the softmax */
  uint32_t us_qprep, us_softmax, us_epi; /**< worker times */
} item_t;

/** @brief Per query head. */
typedef struct {
  float s_q; /**< Q scale, by the worker preparing the head's first block */
  int ready;
  int k; /**< QK convert exponent, by the HMX thread */
  uint16_t rho;
} head_t;

typedef struct {
  uint8_t *vb;
  const hexkl_attn_f16_shape *s;
  const hexkl_attn_q2_io *io;
  hexkl_attn_q2_layout L;
  hexkl_attn_q2_range res; /**< resident column tiles */
  uint32_t hd, dt, G;
  item_t *items;
  uint32_t n_items;
  head_t *heads;
  int32_t *rs_heap[HEXKL_ATTN_Q2_SLOTS]; /**< row sums, cached copies */
  uint32_t kv_next;   /**< kv head whose K/V slices are being pushed */
  uint32_t kv_next_c; /**< resident tiles of it pushed so far */
  uint32_t kv_slice;  /**< tiles per slice */
  hexkl_attn_q2_stats st;
} ctx;

typedef struct {
  ctx *c;
  uint32_t i;
} job_t;

/* ---- small helpers ----------------------------------------------------- */

/** @brief A uniform bias block straight into VTCM: 64 words of the scale
 *         (two vectors), zeros to 1 KiB. Eight vector stores. */
static void set_cvt_block(uint8_t *vtcm_block, uint16_t scale_hf) {
  HVX_Vector *d = (HVX_Vector *)vtcm_block;
  const HVX_Vector w = Q6_V_vsplat_R((int)scale_hf);
  d[0] = w;
  d[1] = w;
  for (uint32_t i = 2; i < HEXKL_CVT_BLOCK_BYTES / 128u; ++i) {
    d[i] = Q6_V_vzero();
  }
}

/** @brief floor(log2(x)) for x >= 1. */
static inline int ilog2_u32(uint32_t x) { return 31 - __builtin_clz(x); }

static inline uint8_t *q_tiles(ctx *c, uint32_t qslot) {
  return c->vb + c->L.q_ah + qslot * c->dt * HEXKL_ATTN_Q2_TILE;
}
static inline float *q_stage(ctx *c, uint32_t i) {
  return (float *)(c->vb + c->L.q_f32 +
                   (i & 1u) * HEXKL_ATTN_Q2_ROWS * c->hd * 4u);
}
static inline uint8_t *s16_tiles(ctx *c, uint32_t slot) {
  return c->vb + c->L.s16 + slot * c->L.n_blk * HEXKL_ATTN_Q2_TILE16;
}
static inline uint8_t *p_tiles(ctx *c, uint32_t slot) {
  return c->vb + c->L.p_ah + slot * c->L.n_blk * HEXKL_ATTN_Q2_TILE;
}
static inline uint8_t *o16_tiles(ctx *c, uint32_t slot) {
  return c->vb + c->L.o16 + slot * c->dt * HEXKL_ATTN_Q2_TILE16;
}
static inline int32_t *rowsum_v(ctx *c, uint32_t slot) {
  return (int32_t *)(c->vb + c->L.rowsum + slot * HEXKL_ATTN_Q2_TILE);
}
static inline int16_t *corr_v(ctx *c, uint32_t h) {
  return (int16_t *)(c->vb + c->L.corr + (h & 1u) * c->L.corr_bytes);
}
static inline uint8_t *cvt_qk(ctx *c, uint32_t h) {
  return c->vb + c->L.cvt + (h & 1u) * 2u * HEXKL_CVT_BLOCK_BYTES;
}
static inline uint8_t *cvt_pv(ctx *c) {
  return c->vb + c->L.cvt + 4u * HEXKL_CVT_BLOCK_BYTES;
}
static inline uint8_t *kt_buf(ctx *c, uint32_t n) {
  return c->vb + c->L.kt_wh + (n % c->L.n_kv_bufs) * c->L.kv_bytes;
}
static inline uint8_t *v_buf(ctx *c, uint32_t n) {
  return c->vb + c->L.v_wh + (n % c->L.n_kv_bufs) * c->L.kv_bytes;
}

/* ---- worker stages ------------------------------------------------------- */

/** @brief max |q| over the n_q rows of head h (only when the caller gave
 *         no scale: a pass over DDR). */
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

static void head_scale(ctx *c, uint32_t h) {
  head_t *hd = &c->heads[h];
  if (hd->ready) {
    return;
  }
  if (c->io->q_scale) {
    hd->s_q = c->io->q_scale[h] > 0.0f ? c->io->q_scale[h] : 1.0f;
  } else {
    const float amax = q_amax(c, h);
    hd->s_q = amax > 0.0f ? amax / 127.0f : 1.0f;
  }
  hd->ready = 1;
}

/** @brief Pushes the DMA of item i's Q rows into its staging half. */
static void q_dma_push(ctx *c, uint32_t i) {
  const item_t *it = &c->items[i];
  const float *src =
    c->io->q + (size_t)it->q0 * c->io->q_stride + it->h * c->hd;
  hexkl_dma_ring_push2d(q_stage(c, i), src, c->hd * 4u, c->io->q_stride * 4u,
                        c->hd * 4u, it->n_rows, 0, 1);
}

/** @brief Stage 0: the staged f32 rows of item i -> uint8 tiles, offset
 *         binary at the head's scale; rows past n_rows read as 0 (128). */
static void stage_qprep(ctx *c, uint32_t i) {
  const uint64_t t0 = now_us();
  item_t *it = &c->items[i];
  head_scale(c, it->h);
  const HVX_Vector inv = hvx_splat_sf(1.0f / c->heads[it->h].s_q);
  const HVX_Vector lo = Q6_V_vsplat_R(-127), hi = Q6_V_vsplat_R(127);
  const HVX_Vector bias = Q6_V_vsplat_R(128);
  const float *stage = q_stage(c, i);
  uint8_t *qt = q_tiles(c, it->qslot);
  for (uint32_t d = 0; d < c->dt; ++d) {
    HVX_Vector *tile = (HVX_Vector *)(qt + d * HEXKL_ATTN_Q2_TILE);
    for (uint32_t r = 0; r < HEXKL_ATTN_Q2_ROWS; r += 4u) {
      HVX_Vector w[4];
      for (uint32_t k = 0; k < 4u; ++k) {
        const uint32_t rr = r + k;
        if (rr < it->n_rows) {
          const HVX_Vector x =
            *(const HVX_Vector *)(stage + (size_t)rr * c->hd + 32u * d);
          HVX_Vector q = hvx_sf_to_w_rne(Q6_Vsf_vmpy_VsfVsf(x, inv));
          q = Q6_Vw_vmin_VwVw(Q6_Vw_vmax_VwVw(q, lo), hi);
          w[k] = Q6_Vw_vadd_VwVw(q, bias);
        } else {
          w[k] = bias;
        }
      }
      const HVX_Vector h01 = Q6_Vh_vpack_VwVw_sat(w[1], w[0]);
      const HVX_Vector h23 = Q6_Vh_vpack_VwVw_sat(w[3], w[2]);
      tile[r / 4u] = Q6_Vub_vpack_VhVh_sat(h23, h01);
    }
  }
  it->us_qprep = (uint32_t)(now_us() - t0);
}

/** @brief Stage 2: the integer softmax over the block's score tiles. */
static void stage_softmax(ctx *c, item_t *it) {
  const uint64_t t0 = now_us();
  const head_t *hd = &c->heads[it->h];
  hvx_softmax_q_block b;
  b.n_col_tiles = it->blk.n;
  b.col0 = 32u * it->blk.lo;
  b.n_cols = c->s->cache_to - 32u * it->blk.lo;
  b.row0 = it->row0;
  b.n_rows = it->n_rows;
  b.window = c->s->window;
  b.rho_q15 = hd->rho;
  b.frac_bits = FRAC_BITS;
  hvx_softmax_q(&b, (const int16_t *)s16_tiles(c, it->slot),
                corr_v(c, it->h) + (size_t)(it->blk.lo - c->res.lo) * 32u,
                p_tiles(c, it->slot), rowsum_v(c, it->slot), c->vb + c->L.smx);
  it->us_softmax = (uint32_t)(now_us() - t0);
}

/** @brief Stage 4: o16 * (512 / 2^kv) * s_v[d] / rowsum[r] -> f32 rows,
 *         for rows [r_lo, r_hi) of the block. */
static void stage_epilogue(ctx *c, item_t *it, uint32_t r_lo, uint32_t r_hi) {
  const uint64_t t0 = now_us();
  // Each caller copies the row sums it needs; the two halves write
  // disjoint words of the cached copy.
  int32_t *rs = c->rs_heap[it->slot];
  {
    HVX_Vector *d = (HVX_Vector *)rs;
    const HVX_Vector *v = (const HVX_Vector *)rowsum_v(c, it->slot);
    for (uint32_t w = r_lo / 4u; w < (r_hi + 3u) / 4u; ++w) {
      d[w] = v[w];
    }
  }
  const float back = ldexpf(512.0f, -it->kv_exp);
  const float *sv = c->io->kv->fs_v + (size_t)it->n * c->hd;
  const uint8_t *o16 = o16_tiles(c, it->slot);
  for (uint32_t d = 0; d < c->dt; ++d) {
    const HVX_Vector svv = hvx_tile_load_u(sv + 32u * d);
    const HVX_Vector *o =
      (const HVX_Vector *)(o16 + (size_t)d * HEXKL_ATTN_Q2_TILE16);
    for (uint32_t v = r_lo / 2u; v < r_hi / 2u; ++v) {
      const uint32_t r0 = 2u * v;
      if (r0 >= it->n_rows) {
        break;
      }
      const HVX_VectorPair w = Q6_Ww_vunpack_Vh(o[v]);
      for (uint32_t i = 0; i < 2u && r0 + i < it->n_rows; ++i) {
        const int32_t rsum = rs[hvx_softmax_q_rowsum_index(r0 + i)];
        const float inv = rsum > 0 ? back / (float)rsum : 0.0f;
        HVX_Vector f = Q6_Vsf_equals_Vw(i == 0 ? Q6_V_lo_W(w) : Q6_V_hi_W(w));
        f = Q6_Vsf_vmpy_VsfVsf(f, hvx_splat_sf(inv));
        f = Q6_Vsf_vmpy_VsfVsf(f, svv);
        float *dst = c->io->out +
                     (size_t)(it->q0 + r0 + i) * c->io->out_stride +
                     it->h * c->hd + 32u * d;
        hvx_tile_store_u(dst, f);
      }
    }
  }
  if (r_lo == 0) {
    it->us_epi = (uint32_t)(now_us() - t0);
  }
}

/** @brief W(i): softmax(i) and the two halves of epilogue(i-1), spread
 *         over the workers by task index. */
static void worker_job(uint32_t n_threads, uint32_t idx, void *arg) {
  job_t *j = (job_t *)arg;
  ctx *c = j->c;
  const uint32_t i = j->i;
  for (uint32_t task = idx; task < 3u; task += n_threads) {
    if (task == 0u && i < c->n_items) {
      stage_softmax(c, &c->items[i]);
    } else if (task == 1u && i >= 1u) {
      stage_epilogue(c, &c->items[i - 1u], 0u, HEXKL_ATTN_Q2_ROWS / 2u);
    } else if (task == 2u && i >= 1u) {
      stage_epilogue(c, &c->items[i - 1u], HEXKL_ATTN_Q2_ROWS / 2u,
                     HEXKL_ATTN_Q2_ROWS);
    }
  }
}

/* ---- HMX-thread stages --------------------------------------------------- */

/** @brief Pushes resident column tiles [c0, c0 + n_tiles) of kv head n's
 *         K^T and V into its buffer. */
static void dma_push_tiles(ctx *c, uint32_t n, uint32_t c0, uint32_t n_tiles) {
  const hexkl_kv_q *kv = c->io->kv;
  const uint32_t row_bytes = c->dt * 1024u;
  const size_t src = hexkl_kv_q_tile_off(kv, n, c->res.lo + c0, 0);
  const size_t dst = (size_t)c0 * row_bytes;
  hexkl_dma_ring_push2d(kt_buf(c, n) + dst, kv->kt + src, row_bytes, row_bytes,
                        row_bytes, n_tiles, 0, 1);
  hexkl_dma_ring_push2d(v_buf(c, n) + dst, kv->v + src, row_bytes, row_bytes,
                        row_bytes, n_tiles, 0, 1);
}

static void dma_push(ctx *c, uint32_t n) { dma_push_tiles(c, n, 0, c->res.n); }

/** @brief Starts streaming kv head n's K/V in slices, one per iteration. */
static void kv_stream_begin(ctx *c, uint32_t n, uint32_t items_per_head) {
  c->kv_next = n;
  c->kv_next_c = 0;
  // Finish within the first half of the current head's items.
  const uint32_t steps = items_per_head > 1u ? items_per_head / 2u : 1u;
  c->kv_slice = (c->res.n + steps - 1u) / steps;
}

static void kv_stream_step(ctx *c) {
  if (c->kv_next_c >= c->res.n) {
    return;
  }
  uint32_t n_tiles = c->kv_slice;
  if (c->kv_next_c + n_tiles > c->res.n) {
    n_tiles = c->res.n - c->kv_next_c;
  }
  dma_push_tiles(c, c->kv_next, c->kv_next_c, n_tiles);
  c->kv_next_c += n_tiles;
}

/** @brief Per-head constants once the head's Q scale is known: the QK
 *         convert blocks and the zero-point correction, into the head's
 *         parity slots. */
static int setup_head(ctx *c, uint32_t h) {
  head_t *hd = &c->heads[h];
  const uint32_t n = h / c->G;
  const hexkl_kv_q *kv = c->io->kv;
  const float alpha = hd->s_q * kv->fs_k[n] * LOG2E / sqrtf((float)c->hd);
  if (hvx_softmax_q_scale(alpha, FRAC_BITS, &hd->k, &hd->rho) != 0) {
    return AEE_EBADPARM;
  }
  uint8_t *blk = cvt_qk(c, h);
  set_cvt_block(blk, hexkl_cvt_f32_to_hf(ldexpf(1.0f, hd->k)));
  set_cvt_block(blk + HEXKL_CVT_BLOCK_BYTES,
                hexkl_cvt_f32_to_hf(ldexpf(1.0f, hd->k - 8)));
  // corr[c][j] = rint(128 * colsum_k[32c + j] * 2^k / 512) mod 2^16
  //            = colsum << (k - 2), or rounded >> (2 - k): one vector of
  // 32 column sums per tile, packed to the low 16 bits (no saturation:
  // the correction is modular like the convert's truncation).
  const int sh = hd->k - 2;
  const int32_t *cs =
    kv->colsum_k + hexkl_kv_q_sk_index(kv, n, 32u * c->res.lo);
  HVX_Vector *dst = (HVX_Vector *)corr_v(c, h);
  for (uint32_t t = 0; t < c->res.n; t += 2u) {
    HVX_Vector a = hvx_tile_load_u(cs + 32u * t);
    HVX_Vector b =
      t + 1u < c->res.n ? hvx_tile_load_u(cs + 32u * (t + 1u)) : Q6_V_vzero();
    if (sh >= 0) {
      a = Q6_Vw_vasl_VwR(a, sh);
      b = Q6_Vw_vasl_VwR(b, sh);
    } else {
      // rint(x / 2^-sh): add half, arithmetic shift (ties away from zero
      // for positive, towards for negative; a one-step tie difference in a
      // per-column constant is immaterial).
      const HVX_Vector half = Q6_V_vsplat_R(1 << (-sh - 1));
      a = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(a, half), -sh);
      b = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(b, half), -sh);
    }
    // Low halves of 64 words -> 64 halfwords: tiles t and t+1.
    dst[t / 2u] = Q6_Vh_vpacke_VwVw(b, a);
  }
  // The padding tile past the last one reads as zero.
  dst[(c->res.n + 1u) / 2u] = Q6_V_vzero();
  return AEE_SUCCESS;
}

/** @brief The current accumulator -> int16 tile @a dst through two convert
 *         blocks (scales s and s/256). */
static inline void acc_to_i16(ctx *c, const uint8_t *blk_lo,
                              const uint8_t *blk_hi, int16_t *dst) {
  uint8_t *p0 = c->vb + c->L.planes;
  uint8_t *p1 = p0 + HEXKL_CVT_PLANE_BYTES;
  hexkl_cvt_issue(blk_lo, p0);
  hexkl_cvt_issue(blk_hi, p1);
  hexkl_cvt_zip_i16(p0, p1, dst);
}

static void stage_qk(ctx *c, const item_t *it) {
  const uint64_t t0 = now_us();
  const uint8_t *q_ah = q_tiles(c, it->qslot);
  const uint8_t *blk_lo = cvt_qk(c, it->h);
  const uint8_t *blk_hi = blk_lo + HEXKL_CVT_BLOCK_BYTES;
  const uint8_t *kt0 = kt_buf(c, it->n);
  uint8_t *s16 = s16_tiles(c, it->slot);
  for (uint32_t t = 0; t < it->blk.n; ++t) {
    const uint32_t ri = it->blk.lo + t - c->res.lo;
    const uint8_t *kt = kt0 + (size_t)ri * c->dt * 1024u;
    hexkl_micro_hmx_acc_clear_int32();
    for (uint32_t d = 0; d < c->dt; ++d) {
      hexkl_hmx_mm_u8i8(q_ah + d * HEXKL_ATTN_Q2_TILE, kt + d * 1024u);
    }
    acc_to_i16(c, blk_lo, blk_hi,
               (int16_t *)(s16 + (size_t)t * HEXKL_ATTN_Q2_TILE16));
  }
  c->st.us_qk += now_us() - t0;
}

/** @brief Largest row sum of the block, from the vector layout. */
static uint32_t rowsum_max(ctx *c, uint32_t slot) {
  const HVX_Vector *rs = (const HVX_Vector *)rowsum_v(c, slot);
  HVX_Vector m = rs[0];
  for (uint32_t w = 1; w < 16u; ++w) {
    m = Q6_Vw_vmax_VwVw(m, rs[w]);
  }
  for (int rot = 4; rot <= 64; rot <<= 1) {
    m = Q6_Vw_vmax_VwVw(m, Q6_V_vror_VR(m, rot));
  }
  return hvx_attn_word0(m);
}

static void stage_pv(ctx *c, item_t *it) {
  const uint64_t t0 = now_us();
  // 2^kv / 512 * 127 * rowsum_max <= 32767.
  const uint32_t rsmax = rowsum_max(c, it->slot);
  // 32767 * 512 / 127 = 132100.4: floor(log2(132100 / rsmax)).
  it->kv_exp = rsmax > 0 ? ilog2_u32(132100u / rsmax) : 0;
  uint8_t *blk_lo = cvt_pv(c);
  uint8_t *blk_hi = blk_lo + HEXKL_CVT_BLOCK_BYTES;
  set_cvt_block(blk_lo, hexkl_cvt_f32_to_hf(ldexpf(1.0f, it->kv_exp)));
  set_cvt_block(blk_hi, hexkl_cvt_f32_to_hf(ldexpf(1.0f, it->kv_exp - 8)));
  const uint8_t *p_ah = p_tiles(c, it->slot);
  const uint8_t *v0 =
    v_buf(c, it->n) + (size_t)(it->blk.lo - c->res.lo) * c->dt * 1024u;
  uint8_t *o16 = o16_tiles(c, it->slot);
  for (uint32_t d = 0; d < c->dt; ++d) {
    hexkl_micro_hmx_acc_clear_int32();
    for (uint32_t t = 0; t < it->blk.n; ++t) {
      hexkl_hmx_mm_u8i8(p_ah + t * HEXKL_ATTN_Q2_TILE,
                        v0 + ((size_t)t * c->dt + d) * 1024u);
    }
    acc_to_i16(c, blk_lo, blk_hi,
               (int16_t *)(o16 + (size_t)d * HEXKL_ATTN_Q2_TILE16));
  }
  c->st.us_pv += now_us() - t0;
}

/* ---- the kernel ---------------------------------------------------------- */

static void free_ctx(ctx *c) {
  free(c->items);
  free(c->heads);
  for (uint32_t i = 0; i < HEXKL_ATTN_Q2_SLOTS; ++i) {
    free(c->rs_heap[i]);
  }
}

int hexkl_attn_q2_prefill(uint8_t *vtcm_base, uint32_t arena_top,
                          const hexkl_attn_f16_shape *s,
                          const hexkl_attn_q2_io *io, hvx_worker_pool *pool,
                          hexkl_attn_q2_stats *st) {
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
  const uint32_t n_blocks =
    (s->n_q + HEXKL_ATTN_Q2_ROWS - 1u) / HEXKL_ATTN_Q2_ROWS;
  c.n_items = s->n_head_q * n_blocks;
  c.items = (item_t *)calloc(c.n_items, sizeof(item_t));
  c.heads = (head_t *)calloc(s->n_head_q, sizeof(head_t));
  // 128-byte aligned: written with vector stores.
  for (uint32_t i = 0; i < HEXKL_ATTN_Q2_SLOTS; ++i) {
    c.rs_heap[i] = (int32_t *)memalign(128u, HVX_SOFTMAX_Q_ROWSUM_WORDS * 4u);
  }
  if (!c.items || !c.heads || !c.rs_heap[0] || !c.rs_heap[1]) {
    free_ctx(&c);
    return AEE_ENOMEMORY;
  }
  for (uint32_t h = 0, i = 0; h < s->n_head_q; ++h) {
    for (uint32_t b = 0; b < n_blocks; ++b, ++i) {
      item_t *it = &c.items[i];
      it->n = h / c.G;
      it->h = h;
      it->q0 = b * HEXKL_ATTN_Q2_ROWS;
      it->n_rows = s->n_q - it->q0 < HEXKL_ATTN_Q2_ROWS ? s->n_q - it->q0
                                                        : HEXKL_ATTN_Q2_ROWS;
      it->row0 = s->cache_from + it->q0;
      it->blk = hexkl_attn_q2_cols(it->row0, it->row0 + HEXKL_ATTN_Q2_ROWS - 1u,
                                   s->cache_to, s->window);
      it->slot = i % HEXKL_ATTN_Q2_SLOTS;
      it->qslot = i % HEXKL_ATTN_Q2_Q_SLOTS;
    }
  }

  const uint64_t t_start = now_us();
  const uint64_t c_start = HAP_perf_get_pcycles();
  int rc = AEE_SUCCESS;
  const uint32_t N = c.n_items;
  job_t job;
  job.c = &c;

  // Prologue: K/V of head 0, Q of blocks 0 and 1 staged and quantized,
  // Q of block 2 in flight, head 0's constants, QK(0). With a second K/V
  // buffer, head 1 starts streaming in slices.
  const uint32_t items_per_head = n_blocks * c.G;
  hexkl_dma_ring_reset();
  uint64_t t0 = now_us();
  dma_push(&c, 0);
  q_dma_push(&c, 0);
  if (N > 1u) {
    q_dma_push(&c, 1);
  }
  hexkl_dma_ring_drain();
  c.st.us_dma += now_us() - t0;
  stage_qprep(&c, 0);
  if (N > 1u) {
    stage_qprep(&c, 1);
  }
  if (N > 2u) {
    q_dma_push(&c, 2);
  }
  c.kv_next_c = c.res.n; // nothing streaming yet
  if (c.L.n_kv_bufs == 2u && s->n_head_kv > 1u) {
    kv_stream_begin(&c, 1, items_per_head);
  }
  rc = setup_head(&c, c.items[0].h);
  if (rc == AEE_SUCCESS) {
    stage_qk(&c, &c.items[0]);
  }

  for (uint32_t i = 0; i < N && rc == AEE_SUCCESS; ++i) {
    item_t *it = &c.items[i];
    job.i = i;
    t0 = now_us();
    const int in_flight = hvx_worker_pool_submit(pool, worker_job, &job, 3u);
    c.st.us_submit += now_us() - t0;

    // Everything pushed last iteration has had an iteration to land.
    t0 = now_us();
    hexkl_dma_ring_drain();
    c.st.us_dma += now_us() - t0;
    if (i + 2u < N) {
      head_scale(&c, c.items[i + 2u].h);
      stage_qprep(&c, i + 2u);
    }
    if (i + 3u < N) {
      q_dma_push(&c, i + 3u);
    }
    kv_stream_step(&c);

    // QK(i+1), unless it needs the single K/V buffer PV(i) still reads.
    int defer_qk = 0;
    if (i + 1u < N) {
      item_t *nx = &c.items[i + 1u];
      if (nx->h != it->h) {
        t0 = now_us();
        head_scale(&c, nx->h);
        rc = setup_head(&c, nx->h);
        c.st.us_head += now_us() - t0;
      }
      if (nx->n != it->n) {
        if (c.L.n_kv_bufs == 2u) {
          // Its slices were pushed over head it->n; the last one at least
          // an iteration ago unless the head was very short.
          t0 = now_us();
          hexkl_dma_ring_drain();
          c.st.us_dma += now_us() - t0;
        } else {
          defer_qk = 1;
        }
      }
      if (rc == AEE_SUCCESS && !defer_qk) {
        stage_qk(&c, nx);
      }
    }

    t0 = now_us();
    if (in_flight) {
      hvx_worker_pool_wait(pool);
    }
    c.st.us_wait += now_us() - t0;
    if (rc != AEE_SUCCESS) {
      break;
    }
    stage_pv(&c, it);

    if (i + 1u < N) {
      item_t *nx = &c.items[i + 1u];
      if (nx->n != it->n) {
        // Head it->n is finished on HMX: its buffer is free.
        if (defer_qk) {
          t0 = now_us();
          dma_push(&c, nx->n);
          hexkl_dma_ring_drain();
          c.st.us_dma += now_us() - t0;
          stage_qk(&c, nx);
        } else if (nx->n + 1u < s->n_head_kv) {
          kv_stream_begin(&c, nx->n + 1u, items_per_head);
        }
      }
    }
  }
  if (rc == AEE_SUCCESS) {
    stage_epilogue(&c, &c.items[N - 1u], 0u, HEXKL_ATTN_Q2_ROWS);
  }

  c.st.us_total = now_us() - t_start;
  c.st.pcycles = HAP_perf_get_pcycles() - c_start;
  c.st.n_blocks = N;
  for (uint32_t i = 0; i < N; ++i) {
    c.st.us_qprep += c.items[i].us_qprep;
    c.st.us_softmax += c.items[i].us_softmax;
    c.st.us_epi += c.items[i].us_epi;
  }
  if (st) {
    *st = c.st;
  }
  free_ctx(&c);
  return rc;
}
