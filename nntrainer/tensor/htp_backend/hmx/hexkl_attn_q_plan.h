// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file   hexkl_attn_q_plan.h
 * @date   28 Sep 2026
 * @brief  VTCM layout for A8W8 / A8W4 flash attention over a quantized cache
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The causal block geometry is the fp16 planner's (hexkl_attn_f16_plan.h:
 * shape, tiling, row ranges, block ranges, tile-row mapping); only the
 * regions differ, because the operands do. HMX's 8-bit activation tile is
 * 64x32 (2 KiB, flat row-major) against fp16's 32x32, so g_br is padded to
 * 64 and one uint8 row tile spans two fp16 score tiles; weight tiles are
 * 1024 B (int8) or 512 B (int4); the accumulator comes out as int32 and is
 * dequantized on HVX into fp16 score tiles for the shared softmax, and the
 * partial output lives as f32 in VTCM instead of an fp16 HMX operand.
 * Plain arithmetic, host-tested (unittest_hexkl_attn_q_plan.cpp).
 */

#ifndef __NNTRAINER_HEXKL_ATTN_Q_PLAN_H__
#define __NNTRAINER_HEXKL_ATTN_Q_PLAN_H__

#include <stdint.h>

#include "hexkl_attn_f16_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Rows in one uint8 activation tile and in one int32 accumulator. */
#define HEXKL_ATTN_Q_ROWS 64u
/** @brief One int32 accumulator readout tile, 64x32x4. */
#define HEXKL_ATTN_Q_ACC_BYTES 8192u

/**
 * @brief Byte offsets from vtcm_base for every region one call uses.
 *
 * Every region starts on a 2 KiB boundary (the activation alignment; the
 * weight alignment of 128 follows). Regions indexed [0]/[1] are the
 * double buffers of the block pipeline: block kb uses kb & 1.
 */
typedef struct {
  uint32_t q_ah;        /**< [n_qb_chunk][rt][dt] uint8 Q tiles */
  uint32_t kt_wh[2];    /**< [ct][dt] K^T weight tiles of tile_bytes */
  uint32_t v_wh[2];     /**< [ct][dt] V weight tiles */
  uint32_t acc;         /**< one int32 readout tile */
  uint32_t s_hf[2];     /**< [2*rt][ct] fp16 score tiles (softmax layout) */
  uint32_t p_ah;        /**< [dt][rt][ct] uint8 P' tiles, one set per V group */
  uint32_t o_f32;       /**< [g_br][hd] f32 partial output */
  uint32_t qx_f32;      /**< [n_qb_chunk*g_br][hd] f32 gathered Q rows */
  uint32_t total;       /**< first byte past the last region */
  uint32_t n_row_tiles; /**< g_br / 64 */
  uint32_t n_col_tiles; /**< bc / 32 */
  uint32_t n_dot_tiles; /**< hd / 32 */
  uint32_t n_qb_chunk;  /**< q blocks whose Q is quantized in one pass */
  uint32_t tile_bytes;  /**< 1024 or 512 */
} hexkl_attn_q_layout;

/**
 * @brief Bytes of per-row state the kernel keeps in cached heap memory
 *        (never VTCM: the scalar unit reads and writes it): q_scale f32 and
 *        q_zp i32 for every row of the chunk, then l f32, a f32, m_hf u16,
 *        a_hf u16 for the current q block, and one 128-byte splat vector
 *        of the P' scale per (V group, row).
 */
static inline uint32_t hexkl_attn_q_meta_bytes(uint32_t g_br, uint32_t dt,
                                               uint32_t n_qb_chunk) {
  return n_qb_chunk * g_br * 8u + g_br * (4u + 4u + 2u + 2u) + dt * g_br * 128u;
}

/** @brief hexkl_attn_f16_tiling_init at the uint8 tile height. */
static inline int hexkl_attn_q_tiling_init(const hexkl_attn_f16_shape *s,
                                           hexkl_attn_f16_tiling *t) {
  return hexkl_attn_f16_tiling_init_aligned(s, t, HEXKL_ATTN_Q_ROWS);
}

/**
 * @brief Lays the regions out and checks them against the arena.
 *
 * @param arena_top   first byte NOT available: the session's int32 HMX
 *                    config offset, which this kernel reads through
 * @param tile_bytes  1024 (int8) or 512 (int4)
 * @param n_qb_chunk  q blocks per Q quantization pass (>= 1); the kernel
 *                    picks it from a byte budget and falls back to 1
 */
static inline int hexkl_attn_q_plan(const hexkl_attn_f16_shape *s,
                                    const hexkl_attn_f16_tiling *t,
                                    uint32_t arena_top, uint32_t tile_bytes,
                                    uint32_t n_qb_chunk,
                                    hexkl_attn_q_layout *L) {
  if (!s || !t || !L || t->g_br == 0 || (t->g_br % HEXKL_ATTN_Q_ROWS) != 0 ||
      (tile_bytes != 1024u && tile_bytes != 512u) || n_qb_chunk == 0) {
    return HEXKL_ATTN_EBADPARM;
  }
  const uint32_t TB = HEXKL_ATTN_TILE_BYTES;
  const uint32_t rt = t->g_br / HEXKL_ATTN_Q_ROWS;
  const uint32_t ct = t->bc / HEXKL_ATTN_TILE;
  const uint32_t dt = s->head_dim / HEXKL_ATTN_TILE;
  L->n_row_tiles = rt;
  L->n_col_tiles = ct;
  L->n_dot_tiles = dt;
  L->n_qb_chunk = n_qb_chunk;
  L->tile_bytes = tile_bytes;

  uint32_t off = 0;
  L->q_ah = off;
  off += n_qb_chunk * rt * dt * TB;
  for (int i = 0; i < 2; ++i) {
    L->kt_wh[i] = off;
    off += hexkl_attn_round_up(ct * dt * tile_bytes, TB);
  }
  for (int i = 0; i < 2; ++i) {
    L->v_wh[i] = off;
    off += hexkl_attn_round_up(ct * dt * tile_bytes, TB);
  }
  L->acc = off;
  off += HEXKL_ATTN_Q_ACC_BYTES;
  for (int i = 0; i < 2; ++i) {
    L->s_hf[i] = off;
    off += 2u * rt * ct * TB;
  }
  L->p_ah = off;
  off += dt * rt * ct * TB;
  L->o_f32 = off;
  off += hexkl_attn_round_up(t->g_br * s->head_dim * 4u, TB);
  L->qx_f32 = off;
  off += hexkl_attn_round_up(n_qb_chunk * t->g_br * s->head_dim * 4u, TB);
  L->total = off;
  if (off > arena_top) {
    return HEXKL_ATTN_ENOMEM;
  }
  return HEXKL_ATTN_OK;
}

#ifdef __cplusplus
}
#endif

#endif /* __NNTRAINER_HEXKL_ATTN_Q_PLAN_H__ */
