// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file   hexkl_attn_q2.h
 * @date   07 Oct 2026
 * @brief  Row-blocked A8W8 attention on HMX over a fixed-scale KV cache
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The kernel 23_attention_v2_plan.md describes. Per KV head its K^T and V
 * tiles stay resident in VTCM; per 64-row block of one query head:
 *
 *   1. Q.K^T on HMX, read out through the convert unit as int16 scores
 *      (two passes at power-of-two scales, D9), one tile per column tile
 *   2. one integer softmax over the whole row (hvx_softmax_q)
 *   3. P'.V on HMX, read out as int16 with a per-block power-of-two scale
 *   4. epilogue: int16 * s_v[d] / rowsum[r] -> f32 rows in DDR
 *
 * Q is quantized here, symmetric per query head (one scale for all rows of
 * the call), stored offset-binary; the zero-point term 128 * colsum_k is a
 * per-column int16 constant the softmax subtracts. The cache must be in
 * fixed-scale mode (hexkl_kv_q_set_fixed_scales): K's scale joins Q's in
 * the softmax's residual factor, V's per-dim scale is applied in the
 * epilogue. Logits are scaled by 1/sqrt(head_dim) as MHACoreLayer does.
 */

#ifndef __NNTRAINER_HEXKL_ATTN_Q2_H__
#define __NNTRAINER_HEXKL_ATTN_Q2_H__

#include <stdint.h>

#include "hexkl_attn_q2_plan.h"
#include "hexkl_kv_q.h"
#include "hvx_worker_pool.h"

typedef struct {
  const float *q; /**< [n_q][q_stride] f32, head h at column h*hd */
  uint32_t q_stride;
  float *out; /**< [n_q][out_stride] f32 */
  uint32_t out_stride;
  const hexkl_kv_q *kv; /**< fixed-scale int8 cache, >= cache_to rows */
} hexkl_attn_q2_io;

/** @brief On-DSP wall time per stage, microseconds, summed over the call. */
typedef struct {
  uint64_t us_qprep;   /**< Q amax, quantization, tiles */
  uint64_t us_dma;     /**< resident K^T / V tiles incl. the drain, corr */
  uint64_t us_qk;      /**< HMX Q.K^T incl. converts and zips */
  uint64_t us_softmax; /**< hvx_softmax_q */
  uint64_t us_pv;      /**< HMX P'.V incl. converts and zips */
  uint64_t us_epi;     /**< int16 -> f32 rows in DDR */
  uint64_t us_total;
  uint64_t pcycles;
  uint32_t n_blocks; /**< (q head, row block) pairs run */
} hexkl_attn_q2_stats;

/**
 * @brief Runs causal (optionally windowed) attention for one step.
 *
 * Requires the HMX lock. @a pool is accepted for the threaded version and
 * ignored for now. @a st may be NULL.
 *
 * @return AEE_SUCCESS, AEE_EBADPARM, AEE_ENOMEMORY
 */
int hexkl_attn_q2_prefill(uint8_t *vtcm_base, uint32_t arena_top,
                          const hexkl_attn_f16_shape *s,
                          const hexkl_attn_q2_io *io, hvx_worker_pool *pool,
                          hexkl_attn_q2_stats *st);

#endif /* __NNTRAINER_HEXKL_ATTN_Q2_H__ */
