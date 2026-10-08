// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file   hexkl_attn_q2.c
 * @date   07 Oct 2026
 * @brief  Row-blocked a16 / kv8 attention on HMX over a fixed-scale KV cache
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Pipeline. The work items are (query head, 64-row block) in order. The
 * calling thread owns HMX and the DMA queue and runs the two matmul
 * stages and the Q byte split (a VTCM pass over rows the DMA staged); the
 * pool workers run the softmax (one block shared by all of them, split by
 * column range) and the epilogue. Iteration i, with softmax(i) and QK(i+1)
 * already done:
 *
 *   submit W(i) = { softmax(i+1), epilogue(i-1) rows 0-31, rows 32-63 }
 *   drain DMA; PV(i); split Q(i+2) from its staging; push DMA for Q(i+3)
 *   and for the next KV head's K/V, a slice at a time
 *   QK(i+2)                                                   on HMX
 *   wait W(i)
 *
 * so a block's softmax overlaps the previous block's PV and the next
 * block's QK, and the per-block time is the longer of the two sides, not
 * their sum. Everything pushed to the DMA queue is consumed one iteration
 * later, so the drain never waits. Nothing is shared between a reader and
 * a writer in flight: Q tiles have three slots, Q staging two, scores, P
 * and output two, the per-head constants two by head parity. The next KV
 * head's K/V arrive under the current head when the plan found room for a
 * second buffer; otherwise QK of the new head waits for the last PV of the
 * old one (try_qk) and the pipeline drains at the boundary.
 *
 * 16-bit activations on a u8 x i8 array. A = 256 A_hi + A_lo, so
 * A . W = 256 (A_hi . W) + (A_lo . W): two accumulations, each read out as
 * int16 through two convert passes (exact at power-of-two scales, D9) and
 * added modulo 2^16. The combined value is made to fit int16 by the choice
 * of scale: for QK through 2^k from hvx_softmax_q_scale_k (k in [-6, 7] so
 * 2^(k-8) is still an fp16 normal), for PV by the normalized P, whose row
 * sum is at most 65535 + n/2, so |acc| <= 127 * 67584 and floor(acc / 512)
 * stays below 2^15 with a factor of two to spare (2^kv = 2^0; 2^1 would
 * give the output one more bit but overflow on a pathological row).
 */

#include "hexkl_attn_q2.h"

#include <math.h>
#include <stdatomic.h>
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
#define FRAC_BITS 9u
/** QK convert exponent range: 2^(k-8) and 2^(k+8) must be fp16 normals. */
#define QK_K_MIN (-6)
#define QK_K_MAX 7
/** PV convert exponent (see the file comment). */
#define PV_KV 0

static inline uint64_t now_us(void) { return HAP_perf_get_time_us(); }

/** @brief One (query head, row block). */
typedef struct {
  uint32_t n, h;   /**< kv head, query head */
  uint32_t q0;     /**< first query row, relative to cache_from */
  uint32_t n_rows; /**< valid rows, <= 64 */
  uint32_t row0;   /**< absolute position of row 0 */
  hexkl_attn_q2_range blk;
  uint32_t slot;                         /**< i % 2: scores, P, output */
  uint32_t qslot;                        /**< i % 3: Q tiles */
  uint32_t us_qprep, us_softmax, us_epi; /**< worker times */
} item_t;

/** @brief Per query head, by the HMX thread in setup_head. */
typedef struct {
  int k;      /**< QK convert exponent, in [QK_K_MIN, QK_K_MAX] */
  uint32_t F; /**< fraction bits of the log2 score, FRAC_BITS unless k had
                   to be moved into range (one bit per step) */
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
  uint32_t kv_next;     /**< kv head whose K/V slices are being pushed */
  uint32_t kv_next_c;   /**< resident tiles of it pushed so far */
  uint32_t kv_slice;    /**< tiles per slice */
  uint32_t kv_resident; /**< highest kv head whose K/V are all in VTCM */
  uint32_t kv_in_use;   /**< lowest kv head a pending PV still reads */
  hexkl_attn_q2_stats st;
} ctx;

/** @brief One pool job: W(i), with the two barriers of the shared softmax
 *         (reset by the HMX thread before each submit). */
#define NO_ITEM 0xFFFFFFFFu
typedef struct {
  ctx *c;
  uint32_t smx_item; /**< softmax of this item, or NO_ITEM */
  uint32_t epi_item; /**< epilogue of this item, or NO_ITEM */
  atomic_uint bar[2];
} job_t;

/** @brief Waits until all @a n participants of job @a j reached barrier
 *         @a k; the same SMT pause hint as the worker pool's spin. */
static inline void job_barrier(job_t *j, uint32_t k, uint32_t n) {
  if (n <= 1u) {
    return;
  }
  atomic_fetch_add_explicit(&j->bar[k], 1u, memory_order_acq_rel);
  while (atomic_load_explicit(&j->bar[k], memory_order_acquire) < n) {
    asm volatile(" pause(#255)\n");
  }
}

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

static inline uint16_t pow2_hf(int e) {
  return hexkl_cvt_f32_to_hf(ldexpf(1.0f, e));
}

/** @brief Q tiles of a slot: [2][dt], low bytes then high bytes. */
static inline uint8_t *q_tiles_lo(ctx *c, uint32_t qslot) {
  return c->vb + c->L.q_ah + qslot * 2u * c->dt * HEXKL_ATTN_Q2_TILE;
}
static inline uint8_t *q_tiles_hi(ctx *c, uint32_t qslot) {
  return q_tiles_lo(c, qslot) + c->dt * HEXKL_ATTN_Q2_TILE;
}
static inline uint16_t *q_stage(ctx *c, uint32_t i) {
  return (uint16_t *)(c->vb + c->L.q_u16 +
                      (i & 1u) * HEXKL_ATTN_Q2_ROWS * c->hd * 2u);
}
static inline uint8_t *s16_tiles(ctx *c, uint32_t slot) {
  return c->vb + c->L.s16 + slot * c->L.n_blk * HEXKL_ATTN_Q2_TILE16;
}
/** @brief P tiles of a slot: [2][n_blk], low bytes then high bytes. */
static inline uint8_t *p_tiles_lo(ctx *c, uint32_t slot) {
  return c->vb + c->L.p_ah + slot * 2u * c->L.n_blk * HEXKL_ATTN_Q2_TILE;
}
static inline uint8_t *p_tiles_hi(ctx *c, uint32_t slot) {
  return p_tiles_lo(c, slot) + c->L.n_blk * HEXKL_ATTN_Q2_TILE;
}
static inline uint8_t *o16_tiles(ctx *c, uint32_t slot) {
  return c->vb + c->L.o16 + slot * c->dt * HEXKL_ATTN_Q2_TILE16;
}
static inline int16_t *corr_v(ctx *c, uint32_t h) {
  return (int16_t *)(c->vb + c->L.corr + (h & 1u) * c->L.corr_bytes);
}
/** @brief The head's three QK bias blocks: 2^(k+8), 2^k, 2^(k-8). */
static inline uint8_t *cvt_qk(ctx *c, uint32_t h) {
  return c->vb + c->L.cvt + (h & 1u) * 3u * HEXKL_CVT_BLOCK_BYTES;
}
/** @brief The three PV bias blocks: 2^(kv+8), 2^kv, 2^(kv-8). */
static inline uint8_t *cvt_pv(ctx *c) {
  return c->vb + c->L.cvt + 6u * HEXKL_CVT_BLOCK_BYTES;
}
static inline uint8_t *planes(ctx *c) { return c->vb + c->L.planes; }
static inline void *smx(ctx *c, uint32_t part) {
  return c->vb + c->L.smx + part * HEXKL_ATTN_Q2_SMX_BYTES;
}
static inline uint8_t *kt_buf(ctx *c, uint32_t n) {
  return c->vb + c->L.kt_wh + (n % c->L.n_kv_bufs) * c->L.kv_bytes;
}
static inline uint8_t *v_buf(ctx *c, uint32_t n) {
  return c->vb + c->L.v_wh + (n % c->L.n_kv_bufs) * c->L.kv_bytes;
}
static inline float q_scale_of(const ctx *c, uint32_t h) {
  return c->io->q_enc[2u * h];
}
static inline uint32_t q_zp_of(const ctx *c, uint32_t h) {
  const float zp = c->io->q_enc[2u * h + 1u];
  return zp <= 0.0f ? 0u : zp >= 65535.0f ? 65535u : (uint32_t)(zp + 0.5f);
}

/* ---- worker stages ------------------------------------------------------- */

/** @brief Pushes the DMA of item i's Q rows into its staging half. */
static void q_dma_push(ctx *c, uint32_t i) {
  const item_t *it = &c->items[i];
  const uint16_t *src =
    c->io->q + (size_t)it->q0 * c->io->q_stride + it->h * c->hd;
  hexkl_dma_ring_push2d(q_stage(c, i), src, c->hd * 2u, c->io->q_stride * 2u,
                        c->hd * 2u, it->n_rows, 0, 1);
}

/**
 * @brief Stage 0: the staged u16 rows of item i -> low-byte and high-byte
 *        uint8 tiles. A vector of a row holds 64 u16 = two dim tiles;
 *        vdeal(-1) over a row pair separates the bytes, vdeal(-32) over
 *        two such pairs regroups four rows of one tile. Rows past n_rows
 *        read as the zero point (they are masked anyway).
 */
static void stage_qprep(ctx *c, uint32_t i) {
  const uint64_t t0 = now_us();
  item_t *it = &c->items[i];
  const uint32_t zp = q_zp_of(c, it->h);
  const HVX_Vector pad = Q6_Vh_vsplat_R((int)zp);
  const uint16_t *stage = q_stage(c, i);
  uint8_t *qlo = q_tiles_lo(c, it->qslot);
  uint8_t *qhi = q_tiles_hi(c, it->qslot);
  for (uint32_t d = 0; d < c->dt; d += 2u) {
    const int two = d + 1u < c->dt;
    HVX_Vector *tl0 = (HVX_Vector *)(qlo + d * HEXKL_ATTN_Q2_TILE);
    HVX_Vector *th0 = (HVX_Vector *)(qhi + d * HEXKL_ATTN_Q2_TILE);
    HVX_Vector *tl1 = tl0 + HEXKL_ATTN_Q2_TILE / 128u;
    HVX_Vector *th1 = th0 + HEXKL_ATTN_Q2_TILE / 128u;
    for (uint32_t r = 0; r < HEXKL_ATTN_Q2_ROWS; r += 4u) {
      HVX_Vector x[4];
      for (uint32_t k = 0; k < 4u; ++k) {
        const uint32_t rr = r + k;
        x[k] = rr < it->n_rows
                 ? *(const HVX_UVector *)(stage + (size_t)rr * c->hd + 32u * d)
                 : pad;
      }
      /* Bytes: even (low) to lo_W, odd (high) to hi_W; row k then k+1. */
      const HVX_VectorPair b01 = Q6_W_vdeal_VVR(x[1], x[0], -1);
      const HVX_VectorPair b23 = Q6_W_vdeal_VVR(x[3], x[2], -1);
      /* 32-byte groups (row, tile): even -> tile d, odd -> tile d + 1. */
      const HVX_VectorPair lo =
        Q6_W_vdeal_VVR(Q6_V_lo_W(b23), Q6_V_lo_W(b01), -32);
      const HVX_VectorPair hi =
        Q6_W_vdeal_VVR(Q6_V_hi_W(b23), Q6_V_hi_W(b01), -32);
      tl0[r / 4u] = Q6_V_lo_W(lo);
      th0[r / 4u] = Q6_V_lo_W(hi);
      if (two) {
        tl1[r / 4u] = Q6_V_hi_W(lo);
        th1[r / 4u] = Q6_V_hi_W(hi);
      }
    }
  }
  it->us_qprep = (uint32_t)(now_us() - t0);
}

/**
 * @brief Stage 2: the integer softmax over the block's score tiles, shared
 *        by the @a n participants of job @a j: part @a idx takes the
 *        column tiles [ct idx / n, ct (idx + 1) / n), the two merges
 *        happen behind the job's barriers.
 */
static void stage_softmax(ctx *c, item_t *it, job_t *j, uint32_t idx,
                          uint32_t n) {
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
  b.frac_bits = (uint8_t)hd->F;
  const int16_t *s16 = (const int16_t *)s16_tiles(c, it->slot);
  const int16_t *corr =
    corr_v(c, it->h) + (size_t)(it->blk.lo - c->res.lo) * 32u;
  uint8_t *plo = p_tiles_lo(c, it->slot);
  uint8_t *phi = p_tiles_hi(c, it->slot);
  /** An even split by tile count; weighting partial tiles and the epilogue
   * halves was tried and measured no better. */
  const uint32_t c0 = it->blk.n * idx / n;
  const uint32_t c1 = it->blk.n * (idx + 1u) / n;
  void *parts[HEXKL_ATTN_Q2_SMX_PARTS];
  for (uint32_t p = 0; p < n; ++p) {
    parts[p] = smx(c, p);
  }
  void *mine = smx(c, idx);
  hvx_softmax_q16_part_max(&b, s16, corr, c0, c1, mine);
  job_barrier(j, 0, n);
  hvx_softmax_q16_merge_max(mine, parts, n);
  hvx_softmax_q16_part_exp(&b, s16, c0, c1, plo, phi, mine);
  job_barrier(j, 1, n);
  hvx_softmax_q16_merge_sum(mine, parts, n, NULL);
  hvx_softmax_q16_part_norm(&b, c0, c1, plo, phi, mine);
  if (idx == 0) {
    it->us_softmax = (uint32_t)(now_us() - t0);
  }
}

/**
 * @brief Stage 4: out = sat_u16(round(o16 * c[d] + zp_o)) with
 *        c[d] = 512 / 2^kv * s_v[d] / (65535 * s_o), rows [r_lo, r_hi) of
 *        the block, two dim tiles (64 u16, one vector) per store.
 */
static void stage_epilogue(ctx *c, item_t *it, uint32_t r_lo, uint32_t r_hi) {
  const uint64_t t0 = now_us();
  const float s_o = c->io->out_enc[2u * it->h];
  const float zp_o = c->io->out_enc[2u * it->h + 1u];
  const float back =
    ldexpf(512.0f, -PV_KV) / (65535.0f * (s_o > 0.0f ? s_o : 1.0f));
  const HVX_Vector vback = hvx_splat_sf(back);
  const HVX_Vector vzp = hvx_splat_sf(zp_o);
  const float *sv = c->io->kv->fs_v + (size_t)it->n * c->hd;
  const uint8_t *o16 = o16_tiles(c, it->slot);
  const uint32_t r_end = r_hi < it->n_rows ? r_hi : it->n_rows;
  for (uint32_t d = 0; d < c->dt; d += 2u) {
    const int two = d + 1u < c->dt;
    const HVX_Vector c0 =
      Q6_Vsf_vmpy_VsfVsf(hvx_tile_load_u(sv + 32u * d), vback);
    const HVX_Vector c1 =
      two ? Q6_Vsf_vmpy_VsfVsf(hvx_tile_load_u(sv + 32u * (d + 1u)), vback)
          : c0;
    const HVX_Vector *o0 =
      (const HVX_Vector *)(o16 + (size_t)d * HEXKL_ATTN_Q2_TILE16);
    const HVX_Vector *o1 = o0 + (two ? HEXKL_ATTN_Q2_TILE16 / 128u : 0u);
    for (uint32_t v = r_lo / 2u; 2u * v < r_end; ++v) {
      const uint32_t r0 = 2u * v;
      /* Row-pair vectors -> words by halves: lo_W row r0, hi_W row r0+1. */
      const HVX_VectorPair w0 = Q6_Ww_vunpack_Vh(o0[v]);
      const HVX_VectorPair w1 = Q6_Ww_vunpack_Vh(o1[v]);
      for (uint32_t i = 0; i < 2u && r0 + i < r_end; ++i) {
        HVX_Vector f0 =
          Q6_Vsf_equals_Vw(i == 0 ? Q6_V_lo_W(w0) : Q6_V_hi_W(w0));
        HVX_Vector f1 =
          Q6_Vsf_equals_Vw(i == 0 ? Q6_V_lo_W(w1) : Q6_V_hi_W(w1));
        f0 = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(f0, c0), vzp);
        f1 = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(f1, c1), vzp);
        const HVX_Vector u =
          Q6_Vuh_vpack_VwVw_sat(hvx_sf_to_w_rne(f1), hvx_sf_to_w_rne(f0));
        uint16_t *dst = c->io->out +
                        (size_t)(it->q0 + r0 + i) * c->io->out_stride +
                        it->h * c->hd + 32u * d;
        if (two) {
          *(HVX_UVector *)dst = u;
        } else {
          /* A lone last tile: its 32 values are the low half. */
          HVX_Vector tmp[1] __attribute__((aligned(128)));
          tmp[0] = u;
          memcpy(dst, tmp, 64u);
        }
      }
    }
  }
  if (r_lo == 0) {
    it->us_epi = (uint32_t)(now_us() - t0);
  }
}

/** @brief One job: the softmax of smx_item shared by every participant,
 *         then the two halves of epi_item's epilogue spread by index. */
static void worker_job(uint32_t n_threads, uint32_t idx, void *arg) {
  job_t *j = (job_t *)arg;
  ctx *c = j->c;
  const uint32_t n =
    n_threads < HEXKL_ATTN_Q2_SMX_PARTS ? n_threads : HEXKL_ATTN_Q2_SMX_PARTS;
  if (j->smx_item != NO_ITEM && idx < n) {
    stage_softmax(c, &c->items[j->smx_item], j, idx, n);
  }
  if (j->epi_item != NO_ITEM) {
    for (uint32_t task = idx; task < 2u; task += n_threads) {
      stage_epilogue(c, &c->items[j->epi_item],
                     task * (HEXKL_ATTN_Q2_ROWS / 2u),
                     (task + 1u) * (HEXKL_ATTN_Q2_ROWS / 2u));
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

/**
 * @brief Per-head constants: the QK convert exponent and residual, the
 *        three bias blocks and the zero-point correction, into the head's
 *        parity slots.
 *
 * corr[c][j] = rint(zp * colsum_k[32c + j] * 2^k / 512) mod 2^16. The
 * product zp * colsum reaches 2^32, so with zp = 256 zh + zl it is formed
 * as 256 A + B, A = zh * colsum and B = zl * colsum (both within int32),
 * then shifted right by sh = 9 - k (>= 2 for k <= 7): for sh >= 8,
 * A / 2^(sh-8) = a1 + a0 / 2^(sh-8) with a0 = A mod 2^(sh-8), and
 * corr = a1 + rint((256 a0 + B) / 2^sh), everything within int32; for
 * sh < 8, corr = A * 2^(8-sh) + rint(B / 2^sh). Modular like the
 * convert's truncation, no saturation.
 */
static int setup_head(ctx *c, uint32_t h) {
  head_t *hd = &c->heads[h];
  const uint32_t n = h / c->G;
  const hexkl_kv_q *kv = c->io->kv;
  const float s_q = q_scale_of(c, h);
  if (!(s_q > 0.0f)) {
    return AEE_EBADPARM;
  }
  const float alpha = s_q * kv->fs_k[n] * LOG2E / sqrtf((float)c->hd);
  /** k moves one for one with F, so a k outside the fp16-normal window of
   * the three converts is brought back by trading fraction bits: fewer
   * for a large alpha (coarser P, as the a8 kernel had at F = 8), more for
   * a tiny one (the scores are then small and the wider F cannot
   * overflow them). The checkpoint's heads all sit at F = 9. */
  const double target9 = (double)alpha * 512.0 * 512.0; /* 2^F * 512 at 9 */
  const int k9 = target9 > 0.0 ? (int)ceil(log2(target9)) : -1000;
  int F = (int)FRAC_BITS;
  if (k9 > QK_K_MAX) {
    F -= k9 - QK_K_MAX;
  } else if (k9 < QK_K_MIN) {
    F += QK_K_MIN - k9;
  }
  if (F < 1) {
    return AEE_EBADPARM; /* scores would overflow int16 at any F */
  }
  if (F > 15) {
    /** alpha below 2^-30: the logits are numerically zero (an all-zero
     * Q or K, e.g. an uninitialized bench model). Any k gives a zero
     * corrected score; take the smallest with rho = 0 -> uniform P. */
    hd->F = 15;
    hd->k = QK_K_MIN;
    hd->rho = 0;
  } else {
    if (hvx_softmax_q_scale_k(alpha, (uint32_t)F, QK_K_MIN, QK_K_MAX, &hd->k,
                              &hd->rho) != 0) {
      return AEE_EBADPARM;
    }
    hd->F = (uint32_t)F;
  }
  uint8_t *blk = cvt_qk(c, h);
  set_cvt_block(blk, pow2_hf(hd->k + 8));
  set_cvt_block(blk + HEXKL_CVT_BLOCK_BYTES, pow2_hf(hd->k));
  set_cvt_block(blk + 2u * HEXKL_CVT_BLOCK_BYTES, pow2_hf(hd->k - 8));

  const uint32_t zp = q_zp_of(c, h);
  // vmpyi(Vw, Rub) picks a byte of the scalar per lane: replicate it.
  const uint32_t zh4 = (zp >> 8) * 0x01010101u;
  const uint32_t zl4 = (zp & 0xffu) * 0x01010101u;
  const int sh = 9 - hd->k;
  const int32_t *cs =
    kv->colsum_k + hexkl_kv_q_sk_index(kv, n, 32u * c->res.lo);
  HVX_Vector *dst = (HVX_Vector *)corr_v(c, h);
  for (uint32_t t = 0; t < c->res.n; t += 2u) {
    HVX_Vector out[2];
    for (uint32_t u = 0; u < 2u; ++u) {
      if (t + u >= c->res.n) {
        out[u] = Q6_V_vzero();
        continue;
      }
      const HVX_Vector col = hvx_tile_load_u(cs + 32u * (t + u));
      const HVX_Vector A = Q6_Vw_vmpyi_VwRub(col, zh4);
      const HVX_Vector B = Q6_Vw_vmpyi_VwRub(col, zl4);
      HVX_Vector r;
      if (sh >= 8) {
        const int m = sh - 8;
        const HVX_Vector a1 = Q6_Vw_vasr_VwR(A, m);
        const HVX_Vector a0 =
          Q6_Vw_vsub_VwVw(A, Q6_Vw_vasl_VwR(a1, m)); /* A mod 2^m */
        HVX_Vector s = Q6_Vw_vadd_VwVw(Q6_Vw_vasl_VwR(a0, 8), B);
        s = Q6_Vw_vadd_VwVw(s, Q6_V_vsplat_R(1 << (sh - 1)));
        r = Q6_Vw_vadd_VwVw(a1, Q6_Vw_vasr_VwR(s, sh));
      } else {
        const HVX_Vector s = Q6_Vw_vadd_VwVw(B, Q6_V_vsplat_R(1 << (sh - 1)));
        r = Q6_Vw_vadd_VwVw(Q6_Vw_vasl_VwR(A, 8 - sh), Q6_Vw_vasr_VwR(s, sh));
      }
      out[u] = r;
    }
    // Low halves of 64 words -> 64 halfwords: tiles t and t+1.
    dst[t / 2u] = Q6_Vh_vpacke_VwVw(out[1], out[0]);
  }
  // The padding tile past the last one reads as zero.
  dst[(c->res.n + 1u) / 2u] = Q6_V_vzero();
  return AEE_SUCCESS;
}

/**
 * @brief Waits for the convert that wrote plane @a p: a load of its last
 *        vector returns only once the unit has written it, i.e. after it
 *        has read the accumulator, so the clear that follows is safe.
 */
static inline void cvt_fence(const uint8_t *p) {
  const volatile HVX_Vector *v =
    (const volatile HVX_Vector *)(p + HEXKL_CVT_PLANE_BYTES - 128u);
  (void)*v;
}

/** @brief The accumulator -> two byte planes through the blocks @a blk_lo
 *         (low byte: scale s) and @a blk_hi (high byte: s / 256). */
static inline void acc_to_planes(const uint8_t *blk_lo, const uint8_t *blk_hi,
                                 uint8_t *p_lo, uint8_t *p_hi) {
  hexkl_cvt_issue(blk_lo, p_lo);
  hexkl_cvt_issue(blk_hi, p_hi);
}

/** @brief dst (int16 tile) = zip(A) + zip(B) modulo 2^16: the two
 *         sweeps' readouts combined. */
static inline void zip_add_i16(const uint8_t *a_lo, const uint8_t *a_hi,
                               const uint8_t *b_lo, const uint8_t *b_hi,
                               int16_t *dst) {
  const HVX_Vector *al = (const HVX_Vector *)a_lo;
  const HVX_Vector *ah = (const HVX_Vector *)a_hi;
  const HVX_Vector *bl = (const HVX_Vector *)b_lo;
  const HVX_Vector *bh = (const HVX_Vector *)b_hi;
  HVX_Vector *d = (HVX_Vector *)dst;
  for (uint32_t i = 0; i < HEXKL_CVT_PLANE_BYTES / 128u; ++i) {
    const HVX_VectorPair a = Q6_W_vshuff_VVR(ah[i], al[i], -1);
    const HVX_VectorPair b = Q6_W_vshuff_VVR(bh[i], bl[i], -1);
    d[2u * i] = Q6_Vh_vadd_VhVh(Q6_V_lo_W(a), Q6_V_lo_W(b));
    d[2u * i + 1u] = Q6_Vh_vadd_VhVh(Q6_V_hi_W(a), Q6_V_hi_W(b));
  }
}

/**
 * @brief One int16 output tile from two u8 sweeps over @a n_pk packets:
 *        activation tiles @a act_hi / @a act_lo (stride @a act_step),
 *        weight tiles from @a wt (stride @a wt_step); blocks[0..2] are the
 *        scales 2^(e+8), 2^e, 2^(e-8).
 */
static inline void two_sweeps(ctx *c, const uint8_t *act_hi,
                              const uint8_t *act_lo, uint32_t act_step,
                              const uint8_t *wt, uint32_t wt_step,
                              uint32_t n_pk, const uint8_t *blocks,
                              int16_t *dst) {
  uint8_t *pl = planes(c);
  uint8_t *a_lo = pl, *a_hi = pl + HEXKL_CVT_PLANE_BYTES;
  uint8_t *b_lo = a_hi + HEXKL_CVT_PLANE_BYTES,
          *b_hi = b_lo + HEXKL_CVT_PLANE_BYTES;
  const uint8_t *blk8 = blocks;
  const uint8_t *blk0 = blocks + HEXKL_CVT_BLOCK_BYTES;
  const uint8_t *blkm8 = blocks + 2u * HEXKL_CVT_BLOCK_BYTES;
  hexkl_micro_hmx_acc_clear_int32();
  for (uint32_t p = 0; p < n_pk; ++p) {
    hexkl_hmx_mm_u8i8(act_hi + (size_t)p * act_step, wt + (size_t)p * wt_step);
  }
  acc_to_planes(blk8, blk0, a_lo, a_hi);
  cvt_fence(a_hi);
  hexkl_micro_hmx_acc_clear_int32();
  for (uint32_t p = 0; p < n_pk; ++p) {
    hexkl_hmx_mm_u8i8(act_lo + (size_t)p * act_step, wt + (size_t)p * wt_step);
  }
  acc_to_planes(blk0, blkm8, b_lo, b_hi);
  zip_add_i16(a_lo, a_hi, b_lo, b_hi, dst);
}

/** @brief Stage 1: the block's score tiles. Per column tile the packets
 *         run over the dim tiles: Q tile d x K^T tile (col, d). */
static void stage_qk(ctx *c, const item_t *it) {
  const uint64_t t0 = now_us();
  const uint8_t *q_lo = q_tiles_lo(c, it->qslot);
  const uint8_t *q_hi = q_tiles_hi(c, it->qslot);
  const uint8_t *blocks = cvt_qk(c, it->h);
  const uint8_t *kt0 = kt_buf(c, it->n);
  uint8_t *s16 = s16_tiles(c, it->slot);
  for (uint32_t t = 0; t < it->blk.n; ++t) {
    const uint32_t ri = it->blk.lo + t - c->res.lo;
    const uint8_t *kt = kt0 + (size_t)ri * c->dt * 1024u;
    two_sweeps(c, q_hi, q_lo, HEXKL_ATTN_Q2_TILE, kt, 1024u, c->dt, blocks,
               (int16_t *)(s16 + (size_t)t * HEXKL_ATTN_Q2_TILE16));
  }
  c->st.us_qk += now_us() - t0;
}

/** @brief Stage 3: the block's output tiles. Per dim tile the packets run
 *         over the column tiles: P tile t x V tile (t, d). */
static void stage_pv(ctx *c, item_t *it) {
  const uint64_t t0 = now_us();
  const uint8_t *p_lo = p_tiles_lo(c, it->slot);
  const uint8_t *p_hi = p_tiles_hi(c, it->slot);
  const uint8_t *blocks = cvt_pv(c);
  const uint8_t *v0 =
    v_buf(c, it->n) + (size_t)(it->blk.lo - c->res.lo) * c->dt * 1024u;
  uint8_t *o16 = o16_tiles(c, it->slot);
  for (uint32_t d = 0; d < c->dt; ++d) {
    two_sweeps(c, p_hi, p_lo, HEXKL_ATTN_Q2_TILE, v0 + d * 1024u, c->dt * 1024u,
               it->blk.n, blocks,
               (int16_t *)(o16 + (size_t)d * HEXKL_ATTN_Q2_TILE16));
  }
  c->st.us_pv += now_us() - t0;
}

static void free_ctx(ctx *c) {
  free(c->items);
  free(c->heads);
}

/**
 * @brief Submits a job and returns whether one is in flight (nothing to
 *        do: no submit, no wait).
 */
static int submit_job(ctx *c, hvx_worker_pool *pool, job_t *job,
                      uint32_t smx_item, uint32_t epi_item) {
  if (smx_item == NO_ITEM && epi_item == NO_ITEM) {
    return 0;
  }
  job->c = c;
  job->smx_item = smx_item;
  job->epi_item = epi_item;
  atomic_store_explicit(&job->bar[0], 0u, memory_order_relaxed);
  atomic_store_explicit(&job->bar[1], 0u, memory_order_relaxed);
  const uint64_t t0 = now_us();
  // This branch's submit returns nothing: it runs the job inline when the
  // pool has no workers, and wait is then a no-op, so a submitted job is
  // always safe to wait for.
  hvx_worker_pool_submit(pool, worker_job, job, 3u);
  c->st.us_submit += now_us() - t0;
  return 1;
}

/**
 * @brief QK of item k on HMX, once its kv head's K/V are resident and its
 *        head constants are set. The K/V buffer of head n is the one head
 *        n - n_kv_bufs used; it is free once no pending PV reads that head
 *        (kv_in_use past it). A head being streamed in slices is finished
 *        synchronously; one not yet begun is fetched whole when its buffer
 *        is free. Returns 0 when the buffer is still busy: the caller
 *        retries after its PV.
 */
static int try_qk(ctx *c, uint32_t k, int *rc) {
  const item_t *it = &c->items[k];
  const uint32_t n = it->n;
  if (n > c->kv_resident) {
    uint64_t t0 = now_us();
    if (c->kv_next == n && c->kv_next_c > 0u && c->kv_next_c <= c->res.n) {
      while (c->kv_next_c < c->res.n) {
        kv_stream_step(c);
      }
    } else {
      const uint32_t prev = n - c->L.n_kv_bufs; /* n >= n_kv_bufs here? */
      const int free_buf = n < c->L.n_kv_bufs || c->kv_in_use > prev;
      if (!free_buf) {
        return 0;
      }
      dma_push(c, n);
      c->kv_next = n;
      c->kv_next_c = c->res.n; /* nothing left to stream for it */
    }
    hexkl_dma_ring_drain();
    c->kv_resident = n;
    c->st.us_dma += now_us() - t0;
  }
  if (k == 0 || it->h != c->items[k - 1u].h) {
    const uint64_t t0 = now_us();
    *rc = setup_head(c, it->h);
    c->st.us_head += now_us() - t0;
    if (*rc != AEE_SUCCESS) {
      return 1;
    }
  }
  stage_qk(c, it);
  return 1;
}

/**
 * Schedule. At the top of iteration i the softmax of item i is done and
 * QK(i+1) is done unless its kv head's buffer was busy. The job of
 * iteration i is { softmax(i+1), epilogue(i-1) }; under it the HMX thread
 * runs PV(i), the Q split of item i+2 and QK(i+2), so the softmax of one
 * block overlaps the PV of the previous one and the QK of the next: the
 * per-block critical path is the longer of the softmax span and the HMX
 * work rather than their sum.
 */
int hexkl_attn_q2_prefill(uint8_t *vtcm_base, uint32_t arena_top,
                          const hexkl_attn_f16_shape *s,
                          const hexkl_attn_q2_io *io, hvx_worker_pool *pool,
                          hexkl_attn_q2_stats *st) {
  if (!vtcm_base || !s || !io || !io->q || !io->out || !io->kv || !io->q_enc ||
      !io->out_enc) {
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
  if (!c.items || !c.heads) {
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
  // The PV blocks are constants of the kernel.
  {
    uint8_t *blk = cvt_pv(&c);
    set_cvt_block(blk, pow2_hf(PV_KV + 8));
    set_cvt_block(blk + HEXKL_CVT_BLOCK_BYTES, pow2_hf(PV_KV));
    set_cvt_block(blk + 2u * HEXKL_CVT_BLOCK_BYTES, pow2_hf(PV_KV - 8));
  }

  const uint64_t t_start = now_us();
  const uint64_t c_start = HAP_perf_get_pcycles();
  int rc = AEE_SUCCESS;
  const uint32_t N = c.n_items;
  const uint32_t items_per_head = n_blocks * c.G;
  job_t job;
  memset(&job, 0, sizeof(job));

  // Prologue: K/V of head 0, Q of items 0 and 1 staged and split, Q of
  // item 2 in flight, QK(0), then softmax(0) with QK(1) under it. With a
  // second K/V buffer, kv head 1 starts streaming in slices.
  hexkl_dma_ring_reset();
  uint64_t t0 = now_us();
  dma_push(&c, 0);
  q_dma_push(&c, 0);
  if (N > 1u) {
    q_dma_push(&c, 1);
  }
  hexkl_dma_ring_drain();
  c.st.us_dma += now_us() - t0;
  c.kv_resident = 0;
  c.kv_in_use = 0;
  c.kv_next = 0;
  c.kv_next_c = c.res.n; // head 0 is whole; nothing streaming yet
  stage_qprep(&c, 0);
  if (N > 1u) {
    stage_qprep(&c, 1);
  }
  if (N > 2u) {
    q_dma_push(&c, 2);
  }
  if (c.L.n_kv_bufs == 2u && s->n_head_kv > 1u) {
    kv_stream_begin(&c, 1, items_per_head);
  }
  uint32_t qk_done = 0; // items [0, qk_done) have their scores
  if (try_qk(&c, 0, &rc)) {
    qk_done = 1;
  }
  if (rc == AEE_SUCCESS) {
    const int in_flight = submit_job(&c, pool, &job, 0, NO_ITEM);
    if (N > 1u && try_qk(&c, 1, &rc)) {
      qk_done = 2;
    }
    t0 = now_us();
    if (in_flight) {
      hvx_worker_pool_wait(pool);
    }
    c.st.us_wait += now_us() - t0;
  }

  for (uint32_t i = 0; i < N && rc == AEE_SUCCESS; ++i) {
    item_t *it = &c.items[i];
    const int deferred = (i + 1u < N) && qk_done == i + 1u;
    const uint32_t next_kv = i + 1u < N ? c.items[i + 1u].n : it->n;

    if (deferred) {
      // QK(i+1) waits for this PV to release the single K/V buffer.
      stage_pv(&c, it);
      c.kv_in_use = next_kv;
      if (try_qk(&c, i + 1u, &rc)) {
        qk_done = i + 2u;
      }
      if (rc != AEE_SUCCESS) {
        break;
      }
    }
    const int in_flight =
      submit_job(&c, pool, &job, i + 1u < N ? i + 1u : NO_ITEM,
                 i >= 1u ? i - 1u : NO_ITEM);

    // Everything pushed last iteration has had an iteration to land.
    t0 = now_us();
    hexkl_dma_ring_drain();
    c.st.us_dma += now_us() - t0;

    if (!deferred) {
      stage_pv(&c, it);
      if (next_kv != it->n) {
        c.kv_in_use = next_kv;
        // Head it->n is finished on HMX: with two buffers its buffer takes
        // the head after next, streamed under the next head's items.
        if (c.L.n_kv_bufs == 2u && next_kv + 1u < s->n_head_kv &&
            c.kv_next < next_kv + 1u) {
          kv_stream_begin(&c, next_kv + 1u, items_per_head);
        }
      }
    }
    if (i + 2u < N) {
      stage_qprep(&c, i + 2u);
    }
    if (i + 3u < N) {
      q_dma_push(&c, i + 3u);
    }
    kv_stream_step(&c);
    if (i + 2u < N && qk_done == i + 2u) {
      if (try_qk(&c, i + 2u, &rc)) {
        qk_done = i + 3u;
      }
    }

    t0 = now_us();
    if (in_flight) {
      hvx_worker_pool_wait(pool);
    }
    c.st.us_wait += now_us() - t0;
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
