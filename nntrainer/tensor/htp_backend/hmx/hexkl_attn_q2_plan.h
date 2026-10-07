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
 * the convert unit, one integer softmax, P'.V, epilogue. The regions below
 * are what one call needs; the block-level regions hold one block.
 *
 * Plain arithmetic, host-tested.
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
/** @brief Bytes of one int16 tile (scores, O). */
#define HEXKL_ATTN_Q2_TILE16 4096u
/** @brief Largest head_dim / 32. */
#define HEXKL_ATTN_Q2_MAX_DT 16u

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
  uint32_t kt_wh;  /**< [n_res][dt] K^T weight tiles, 1 KiB each */
  uint32_t v_wh;   /**< [n_res][dt] V weight tiles */
  uint32_t q_ah;   /**< [dt] uint8 Q tiles of the current block */
  uint32_t s16;    /**< [n_blk][64][32] int16 scores */
  uint32_t planes; /**< 2 convert planes, 2 KiB each */
  uint32_t p_ah;   /**< [n_blk] uint8 P' tiles */
  uint32_t o16;    /**< [dt][64][32] int16 output */
  uint32_t corr;   /**< [n_res + 1][32] int16 zero-point correction */
  uint32_t cvt;    /**< 4 bias blocks: QK lo/hi, PV lo/hi */
  uint32_t smx;    /**< softmax scratch */
  uint32_t rowsum; /**< HVX_SOFTMAX_Q_ROWSUM_WORDS int32 */
  uint32_t total;  /**< first byte past the last region */
  uint32_t n_res;  /**< resident column tiles */
  uint32_t n_blk;  /**< most column tiles one block sees */
  uint32_t dt;     /**< head_dim / 32 */
} hexkl_attn_q2_layout;

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

/**
 * @brief Lays the regions out and checks them against the arena.
 *
 * @param arena_top  first byte NOT available
 */
static inline int hexkl_attn_q2_plan(const hexkl_attn_f16_shape *s,
                                     uint32_t arena_top,
                                     hexkl_attn_q2_layout *L) {
  if (hexkl_attn_q2_check(s) != HEXKL_ATTN_OK || !L) {
    return HEXKL_ATTN_EBADPARM;
  }
  const uint32_t TB = HEXKL_ATTN_Q2_TILE;
  const uint32_t dt = s->head_dim / 32u;
  const hexkl_attn_q2_range res = hexkl_attn_q2_cols(
    s->cache_from, s->cache_from + s->n_q - 1u, s->cache_to, s->window);
  // The widest block: the last one, or any full one under a window.
  uint32_t n_blk = 0;
  for (uint32_t r0 = s->cache_from; r0 < s->cache_from + s->n_q;
       r0 += HEXKL_ATTN_Q2_ROWS) {
    const hexkl_attn_q2_range b = hexkl_attn_q2_cols(
      r0, r0 + HEXKL_ATTN_Q2_ROWS - 1u, s->cache_to, s->window);
    if (b.n > n_blk) {
      n_blk = b.n;
    }
  }
  L->n_res = res.n;
  L->n_blk = n_blk;
  L->dt = dt;

  uint32_t off = 0;
  L->kt_wh = off;
  off += hexkl_attn_round_up(res.n * dt * 1024u, TB);
  L->v_wh = off;
  off += hexkl_attn_round_up(res.n * dt * 1024u, TB);
  L->q_ah = off;
  off += dt * TB;
  L->s16 = off;
  off += n_blk * HEXKL_ATTN_Q2_TILE16;
  L->planes = off;
  off += 2u * HEXKL_CVT_PLANE_BYTES;
  L->p_ah = off;
  off += n_blk * TB;
  L->o16 = off;
  off += dt * HEXKL_ATTN_Q2_TILE16;
  L->corr = off;
  off += hexkl_attn_round_up((res.n + 1u) * 64u, TB);
  L->cvt = off;
  off += 4u * HEXKL_CVT_BLOCK_BYTES;
  L->smx = off;
  off += hexkl_attn_round_up(HVX_SOFTMAX_Q_SCRATCH_BYTES, TB);
  L->rowsum = off;
  off += TB;
  L->total = off;
  if (off > arena_top) {
    return HEXKL_ATTN_ENOMEM;
  }
  return HEXKL_ATTN_OK;
}

#ifdef __cplusplus
}
#endif

#endif /* __NNTRAINER_HEXKL_ATTN_Q2_PLAN_H__ */
