// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file   hexkl_attn_q2_plan.h
 * @date   07 Oct 2026
 * @brief  VTCM layout for the row-blocked A8W8 attention kernel
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * 23_attention_v2_plan.md. The kernel keeps one KV head's K^T and V tiles
 * resident -- every column tile any query row of this call can see -- and
 * walks 64-row blocks of query rows: Q.K^T into int16 score tiles through
 * the convert unit, one integer softmax, P'.V, epilogue.
 *
 * The block stages run as a pipeline: the HMX stages on the calling
 * thread, Q preparation, softmax and epilogue on pool workers, so the
 * block-level regions come in slots. Three blocks of Q tiles can be live
 * (the one being multiplied, the next, the one being prepared), two of
 * everything else (the block on HMX and the block on the workers). The
 * per-head constants (zero-point correction, QK convert blocks) come in
 * two slots as well, indexed by query-head parity, because the softmax of
 * one head's last block overlaps the next head's first multiply. K/V get
 * two buffers when they fit, so the next KV head's DMA runs under the
 * current head's blocks.
 *
 * When one KV head's K^T and V do not fit (a 512-wide head past ~4K rows:
 * 4 MiB each at 8192), they stream instead, per block: K^T in chunks of
 * column tiles for Q.K^T and V one dim tile at a time, V[:, d] over every
 * column the block sees, for P'.V. Each score tile sums over the dim tiles
 * of its own column tile and each output tile over every column tile of
 * its dim tile, so both splits leave the accumulation order -- and the
 * result -- as it is resident. The score and P' rows of a block stay
 * whole: the softmax is the same two-pass one, nothing is rescaled.
 *
 * Streamed in pairs (pairs != 0, when an even number of query heads share
 * each KV head): two items of one block and two query heads of one KV head
 * share every K^T chunk and V slice, so the K/V traffic halves. The pipeline
 * then runs a pair a stage: four score slots (the pair on HMX and the pair
 * in the softmax), one of P', output and Q tiles, four of the per-head
 * constants (a pair's heads and the next pair's).
 *
 * Plain arithmetic.
 */

#ifndef __NNTRAINER_HEXKL_ATTN_Q2_PLAN_H__
#define __NNTRAINER_HEXKL_ATTN_Q2_PLAN_H__

#include <stdint.h>

#include "hexkl_attn_f16_plan.h"
#include "hexkl_cvt.h"
#include "hvx_softmax_q.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Rows per block: one uint8 activation tile. */
#define HEXKL_ATTN_Q2_ROWS 64u
/** @brief Bytes of one uint8 tile (Q, P', convert plane). */
#define HEXKL_ATTN_Q2_TILE 2048u
/** @brief Bytes of one int16 tile (scores). */
#define HEXKL_ATTN_Q2_TILE16 4096u
/** @brief Bytes of one f32 tile (O, unnormalized). */
#define HEXKL_ATTN_Q2_TILE32 8192u
/** @brief Bytes of one block's P' row sums (HVX_SOFTMAX_Q_ROWSUM_WORDS). */
#define HEXKL_ATTN_Q2_RS_BYTES 2048u
/** @brief Convert planes: two sweeps of the QK readout (int16, 2 + 2) or of
 *         the PV readout (int24, 3 + 3). */
#define HEXKL_ATTN_Q2_PLANES 6u
/** @brief Largest PV readout shift: its third convert scale, 2^(-7 - s),
 *         must stay an fp16 normal. */
#define HEXKL_ATTN_Q2_PV_SHIFT_MAX 7u
/** @brief Largest head_dim / 32. */
#define HEXKL_ATTN_Q2_MAX_DT 16u
/** @brief Q tile slots and block slots of the pipeline. */
#define HEXKL_ATTN_Q2_Q_SLOTS 3u
#define HEXKL_ATTN_Q2_SLOTS 2u

/** @brief Column tile range [lo, hi] (inclusive) plus the count. */
typedef struct {
  uint32_t lo, hi, n;
} hexkl_attn_q2_range;

/**
 * @brief Column tiles rows [row_lo, row_hi] of a causal, optionally
 *        windowed attention can see, over a cache of cache_to rows.
 */
static inline hexkl_attn_q2_range hexkl_attn_q2_cols(uint32_t row_lo,
                                                     uint32_t row_hi,
                                                     uint32_t cache_to,
                                                     uint32_t window) {
  hexkl_attn_q2_range r;
  uint32_t c_lo = 0;
  if (window != 0 && row_lo + 1u > window) {
    c_lo = (row_lo + 1u - window) / 32u;
  }
  uint32_t last = row_hi;
  if (last + 1u > cache_to) {
    last = cache_to - 1u;
  }
  r.lo = c_lo;
  r.hi = last / 32u;
  r.n = r.hi >= r.lo ? r.hi - r.lo + 1u : 0u;
  return r;
}

/**
 * @brief Byte offsets from vtcm_base of every region.
 *
 * Regions start on 2 KiB boundaries (the activation alignment).
 */
typedef struct {
  uint32_t kt_wh;      /**< [n_kv_bufs][n_res][dt] K^T weight tiles, 1 KiB;
                            streamed: [2][stream][dt] K^T chunks */
  uint32_t v_wh;       /**< [n_kv_bufs][n_res][dt] V weight tiles;
                            streamed: [2][n_blk] tiles of one V[:, d] */
  uint32_t kv_bytes;   /**< bytes of one K^T (or V) buffer; streamed: of
                            one K^T chunk */
  uint32_t n_kv_bufs;  /**< 2 when the next head's K/V fit alongside */
  uint32_t stream;     /**< 0: K/V resident; else column tiles of a K^T
                            chunk, K^T and V streamed per block */
  uint32_t v_slice;    /**< streamed: bytes of one V[:, d] buffer */
  uint32_t q_u16;      /**< [2][64][hd] u16 Q rows staged by DMA */
  uint32_t q_ah;       /**< [3][2][dt] uint8 Q tiles: low bytes, high bytes */
  uint32_t s16;        /**< [2][n_blk][64][32] int16 scores */
  uint32_t planes;     /**< HEXKL_ATTN_Q2_PLANES convert planes, 2 KiB each
                            (HMX thread only) */
  uint32_t p_ah;       /**< [2][2][n_blk] uint8 P' tiles: low, high bytes */
  uint32_t o32;        /**< [2][dt][64][32] f32 output, sum_j P'_j V_j */
  uint32_t rs;         /**< [3][HVX_SOFTMAX_Q_ROWSUM_WORDS] int32 P' row
                            sums, by Q slot: softmax(i + 2) shares a job
                            with epilogue(i) */
  uint32_t corr;       /**< [2][n_res + 1][32] int16 zero-point correction */
  uint32_t corr_bytes; /**< bytes of one corr slot */
  uint32_t cvt;        /**< bias blocks: QK 2^(k+8), 2^k, 2^(k-8) x 2 slots,
                            PV 2^(9-s), 2^(1-s), 2^(-7-s)
                            (HEXKL_ATTN_Q2_CVT_BLOCKS) */
  uint32_t smx;        /**< [HEXKL_ATTN_Q2_SMX_PARTS] softmax scratches */
  uint32_t total;      /**< first byte past the last region */
  uint32_t n_res;      /**< resident column tiles */
  uint32_t n_blk;      /**< most column tiles one block sees */
  uint32_t dt;         /**< head_dim / 32 */
  uint32_t pv_shift;   /**< s: the PV readout is floor(acc / 2^s) as int24,
                            so 127 * 255 * 32 n_blk >> s fits */
  uint32_t pairs;      /**< streamed: two items share each K/V transfer */
} hexkl_attn_q2_layout;

/** @brief Bias blocks in the cvt region: 3 QK blocks per head parity and
 *         3 PV blocks, padded to a 2 KiB multiple. */
#define HEXKL_ATTN_Q2_CVT_BLOCKS 10u
/** @brief Per-head constant slots (QK bias blocks, zero-point corrections)
 *         of the layout: by head parity, or by head mod 4 in pairs. */
#define HEXKL_ATTN_Q2_HEAD_SLOTS(L) ((L)->pairs ? 4u : 2u)
/** @brief Softmax scratches: one per worker sharing a block's softmax. */
#define HEXKL_ATTN_Q2_SMX_PARTS 3u
#define HEXKL_ATTN_Q2_SMX_BYTES                                                \
  hexkl_attn_round_up(HVX_SOFTMAX_Q_SCRATCH_BYTES, HEXKL_ATTN_Q2_TILE)

/**
 * @brief Validates the shape for this kernel: head_dim up to 512, the
 *        cache covering every query row, no softcap or sinks (Gemma-4 has
 *        neither; both are the f16 kernel's business).
 */
static inline int hexkl_attn_q2_check(const hexkl_attn_f16_shape *s) {
  if (!s || s->n_q == 0 || s->n_head_q == 0 || s->n_head_kv == 0 ||
      s->head_dim == 0 || (s->head_dim % 32u) != 0 ||
      s->head_dim > 32u * HEXKL_ATTN_Q2_MAX_DT ||
      (s->n_head_q % s->n_head_kv) != 0 ||
      s->cache_to < s->cache_from + s->n_q || s->softcap > 0.0f) {
    return HEXKL_ATTN_EBADPARM;
  }
  return HEXKL_ATTN_OK;
}

/** @brief Column tiles of the largest streamed K^T chunk (1 MiB at a
 *         512-wide head); the plan halves it until the layout fits. */
#define HEXKL_ATTN_Q2_KC_MAX 64u

/**
 * @brief The shape's tile counts into @a L: resident column tiles, the
 *        most one block sees, dim tiles, and the per-head region sizes.
 *        Checks that each of @a n_parts softmax parts stays within the
 *        softmax's staged tiles (it splits a block's tiles evenly).
 */
static inline int hexkl_attn_q2_geometry(const hexkl_attn_f16_shape *s,
                                         uint32_t n_parts,
                                         hexkl_attn_q2_layout *L) {
  if (hexkl_attn_q2_check(s) != HEXKL_ATTN_OK || !L || n_parts == 0) {
    return HEXKL_ATTN_EBADPARM;
  }
  const hexkl_attn_q2_range res = hexkl_attn_q2_cols(
    s->cache_from, s->cache_from + s->n_q - 1u, s->cache_to, s->window);
  uint32_t n_blk = 0;
  for (uint32_t r0 = s->cache_from; r0 < s->cache_from + s->n_q;
       r0 += HEXKL_ATTN_Q2_ROWS) {
    const hexkl_attn_q2_range b = hexkl_attn_q2_cols(
      r0, r0 + HEXKL_ATTN_Q2_ROWS - 1u, s->cache_to, s->window);
    if (b.n > n_blk) {
      n_blk = b.n;
    }
  }
  if ((n_blk + n_parts - 1u) / n_parts > HVX_SOFTMAX_Q_MAX_TILES) {
    return HEXKL_ATTN_EBADPARM;
  }
  /* One PV sweep sums u8 x i8 over the block's columns: |acc| <=
     127 * 255 * 32 n_blk, read out as floor(acc / 2^s) in 24 bits. */
  const uint64_t acc_max = 127ull * 255ull * 32ull * n_blk;
  uint32_t shift = 0;
  while ((acc_max >> shift) >= (1ull << 23)) {
    ++shift;
  }
  if (shift > HEXKL_ATTN_Q2_PV_SHIFT_MAX) {
    return HEXKL_ATTN_EBADPARM;
  }
  L->pv_shift = shift;
  L->n_res = res.n;
  L->n_blk = n_blk;
  L->dt = s->head_dim / 32u;
  L->corr_bytes = hexkl_attn_round_up((res.n + 1u) * 64u, HEXKL_ATTN_Q2_TILE);
  return HEXKL_ATTN_OK;
}

/** @brief The regions after K/V, from @a off; returns the end. */
static inline uint32_t hexkl_attn_q2_layout_rest(const hexkl_attn_f16_shape *s,
                                                 hexkl_attn_q2_layout *L,
                                                 uint32_t off) {
  const uint32_t TB = HEXKL_ATTN_Q2_TILE;
  const uint32_t dt = L->dt, n_blk = L->n_blk;
  /** pairs: Q tiles of one pair, scores of two pairs (slot 2 (u % 2) + e),
     row sums likewise */
  const uint32_t q_slots = L->pairs ? 2u : HEXKL_ATTN_Q2_Q_SLOTS;
  const uint32_t s_slots = L->pairs ? 4u : HEXKL_ATTN_Q2_SLOTS;
  const uint32_t rs_slots = L->pairs ? 4u : HEXKL_ATTN_Q2_Q_SLOTS;
  const uint32_t h_slots = HEXKL_ATTN_Q2_HEAD_SLOTS(L);
  L->q_u16 = off;
  off += 2u * HEXKL_ATTN_Q2_ROWS * s->head_dim * 2u;
  L->q_ah = off;
  off += q_slots * 2u * dt * TB;
  L->s16 = off;
  off += s_slots * n_blk * HEXKL_ATTN_Q2_TILE16;
  L->planes = off;
  off += HEXKL_ATTN_Q2_PLANES * HEXKL_CVT_PLANE_BYTES;
  L->p_ah = off;
  off += HEXKL_ATTN_Q2_SLOTS * 2u * n_blk * TB;
  L->o32 = off;
  off += HEXKL_ATTN_Q2_SLOTS * dt * HEXKL_ATTN_Q2_TILE32;
  L->rs = off;
  off += rs_slots * HEXKL_ATTN_Q2_RS_BYTES;
  L->corr = off;
  off += h_slots * L->corr_bytes;
  L->cvt = off;
  off += hexkl_attn_round_up((3u * h_slots + 3u) * HEXKL_CVT_BLOCK_BYTES, TB);
  L->smx = off;
  off += HEXKL_ATTN_Q2_SMX_PARTS * HEXKL_ATTN_Q2_SMX_BYTES;
  L->total = off;
  return off;
}

/** @brief Resident layout with @a bufs K/V buffers (after the geometry);
 *         returns its total bytes. */
static inline uint32_t
hexkl_attn_q2_layout_resident(const hexkl_attn_f16_shape *s,
                              hexkl_attn_q2_layout *L, uint32_t bufs) {
  L->stream = 0;
  L->v_slice = 0;
  L->pairs = 0;
  L->n_kv_bufs = bufs;
  L->kv_bytes =
    hexkl_attn_round_up(L->n_res * L->dt * 1024u, HEXKL_ATTN_Q2_TILE);
  L->kt_wh = 0;
  L->v_wh = bufs * L->kv_bytes;
  return hexkl_attn_q2_layout_rest(s, L, 2u * bufs * L->kv_bytes);
}

/** @brief Streamed layout with K^T chunks of @a kc column tiles, in pairs
 *         when @a pairs (after the geometry); returns its total bytes. */
static inline uint32_t
hexkl_attn_q2_layout_stream(const hexkl_attn_f16_shape *s,
                            hexkl_attn_q2_layout *L, uint32_t kc,
                            uint32_t pairs) {
  const uint32_t TB = HEXKL_ATTN_Q2_TILE;
  L->stream = kc;
  L->pairs = pairs;
  L->n_kv_bufs = 1;
  L->kv_bytes = hexkl_attn_round_up(kc * L->dt * 1024u, TB);
  L->v_slice = hexkl_attn_round_up(L->n_blk * 1024u, TB);
  L->kt_wh = 0;
  L->v_wh = 2u * L->kv_bytes;
  return hexkl_attn_q2_layout_rest(s, L, L->v_wh + 2u * L->v_slice);
}

/**
 * @brief Lays the regions out and checks them against the arena. Tries
 *        two resident K/V buffers first, then one, then streamed K/V with
 *        the largest K^T chunk that fits: in pairs when a KV head has an
 *        even number of query heads and the call more than one row (a
 *        decode row folds its heads instead), else one item at a time.
 *
 * @param arena_top  first byte NOT available
 * @param n_parts    softmax parts the kernel will run (>= 1)
 */
static inline int hexkl_attn_q2_plan(const hexkl_attn_f16_shape *s,
                                     uint32_t arena_top, uint32_t n_parts,
                                     hexkl_attn_q2_layout *L) {
  const int rc = hexkl_attn_q2_geometry(s, n_parts, L);
  if (rc != HEXKL_ATTN_OK) {
    return rc;
  }
  for (uint32_t bufs = 2; bufs >= 1; --bufs) {
    if (hexkl_attn_q2_layout_resident(s, L, bufs) <= arena_top) {
      return HEXKL_ATTN_OK;
    }
  }
  const uint32_t kc0 =
    L->n_res < HEXKL_ATTN_Q2_KC_MAX ? L->n_res : HEXKL_ATTN_Q2_KC_MAX;
  const uint32_t G = s->n_head_q / s->n_head_kv;
  for (uint32_t pairs = s->n_q > 1u && G % 2u == 0u ? 1u : 0u;; --pairs) {
    for (uint32_t kc = kc0; kc != 0; kc /= 2u) {
      if (hexkl_attn_q2_layout_stream(s, L, kc, pairs) <= arena_top) {
        return HEXKL_ATTN_OK;
      }
    }
    if (pairs == 0u) {
      return HEXKL_ATTN_ENOMEM;
    }
  }
}

#ifdef __cplusplus
}
#endif

#endif /* __NNTRAINER_HEXKL_ATTN_Q2_PLAN_H__ */
