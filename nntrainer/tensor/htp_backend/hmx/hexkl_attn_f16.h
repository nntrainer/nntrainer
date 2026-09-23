// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file   hexkl_attn_f16.h
 * @date   14 Sep 2026
 * @brief  fp16 flash attention on HMX over nntrainer's fp16 KV cache
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __NNTRAINER_HEXKL_ATTN_F16_H__
#define __NNTRAINER_HEXKL_ATTN_F16_H__

#include <stdint.h>

#include "hexkl_attn_f16_plan.h"
#include "hvx_worker_pool.h"

/**
 * @brief Where the operands live in DDR. Strides are in elements.
 *
 * Matches MHACoreLayer: Q is f32 [n_q][n_head_q*head_dim] post-RoPE; K and
 * V are the fp16 cache [cache_to][n_head_kv*head_dim] with the fp16 bit
 * patterns in uint16 containers (K post-RoPE, V raw); out is f32
 * [n_q][n_head_q*head_dim] with query head n*G+g at column (n*G+g)*head_dim.
 * sinks is NULL or n_head_q f32 per-head sink logits (MHACoreLayer's
 * use_sink): each joins its row's softmax as one more logit with no value.
 */
typedef struct {
  const float *q;
  uint32_t q_stride;
  const uint16_t *k;
  const uint16_t *v;
  uint32_t kv_stride;
  const float *sinks;
  float *out;
  uint32_t out_stride;
  /** Resident tiles (hexkl_kv_tiles_f16): when kt_tiles is non-NULL the
      kernel DMAs ready K^T / V weight tiles from here instead of raw rows
      from k / v, and skips the tile-building phase. kv_tile_cols is the
      registry's column-tile count (max_rows / 32), which sets the per-head
      stride between tiles. k and v may then be NULL. */
  const uint8_t *kt_tiles;
  const uint8_t *v_tiles;
  uint32_t kv_tile_cols;
} hexkl_attn_f16_io;

/** @brief On-DSP wall time per phase, microseconds, summed over the call. */
typedef struct {
  uint64_t us_qprep;   /**< f32 -> scaled f16 Q tiles */
  uint64_t us_dma;     /**< K/V block DMA incl. the drain */
  uint64_t us_tile;    /**< K^T / V tile builds from the landing buffers */
  uint64_t us_qk;      /**< HMX Q.K^T incl. accumulator reads */
  uint64_t us_softmax; /**< HVX online softmax + D tiles */
  uint64_t us_pv;      /**< HMX D.O_prev + P.V incl. accumulator reads */
  uint64_t us_store;   /**< O/l -> f32 rows in DDR */
  uint32_t n_blocks;   /**< (kv_head, q_block, kv_block) iterations run */
} hexkl_attn_f16_stats;

/**
 * @brief Runs causal (optionally windowed) attention for one step.
 *
 * out[q][(n*G+g)*hd + d] = sum_k softmax_k(Q.K^T / sqrt(hd)) V, over the
 * cache rows hexkl_attn_f16_row_range() allows for row q -- the same math
 * MHACoreLayer's compute_kcaches/softmax_row/compute_fp16vcache do, minus
 * the materialized logits. With s->softcap > 0 the logits pass through
 * tanh(s/softcap)*softcap first; with io->sinks each row's denominator
 * gains exp(sink - m).
 *
 * Requires the HMX lock to be held by the caller (the session takes it in
 * open()) and writes the fp16 accumulator config into the arena itself on
 * every call. HMX work and the DMA drain stay on the calling thread; the
 * HVX phases (tile builds, softmax) split across @a pool when it is given.
 *
 * @param arena_top  first VTCM byte this call may NOT use (the session's
 *                   int32 config offset)
 * @param pool       may be NULL: everything runs on the calling thread
 * @param st         may be NULL
 * @return AEE_SUCCESS, AEE_EBADPARM, AEE_ENOMEMORY, AEE_EUNSUPPORTED (no
 *         fp16 HMX), or a HexKL error passed through
 */
int hexkl_attn_f16_prefill(uint8_t *vtcm_base, uint32_t arena_top,
                           uint32_t hmx_fp16_rate,
                           const hexkl_attn_f16_shape *s,
                           const hexkl_attn_f16_tiling *t,
                           const hexkl_attn_f16_io *io, hvx_worker_pool *pool,
                           hexkl_attn_f16_stats *st);

#endif /* __NNTRAINER_HEXKL_ATTN_F16_H__ */
