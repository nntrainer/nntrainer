// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file   hexkl_attn_q2.h
 * @date   07 Oct 2026
 * @brief  Row-blocked a16 / kv8 attention on HMX over a fixed-scale KV cache
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The kernel of 24_attention_a16_plan.md (the a8 version was plan 23). Per
 * KV head its K^T and V tiles stay resident in VTCM; per 64-row block of
 * one query head:
 *
 *   1. Q.K^T on HMX in two u8 sweeps (Q's high and low bytes), each read
 *      out through the convert unit as int16 (two passes at power-of-two
 *      scales, D9) and added: one int16 score tile per column tile
 *   2. one integer softmax over the whole row with a normalized u16 P
 *      (hvx_softmax_q16), left as its low and high byte tiles
 *   3. P.V on HMX in two u8 sweeps, read out as int16 at the constant
 *      scale 2^1 / 512 (sum_j P_j <= 65535 bounds the accumulator)
 *   4. epilogue: int16 * c[d] + zp -> u16 rows in DDR
 *
 * Q arrives as the model's 16-bit asymmetric tensor (per-head scale and
 * zero point, q = s_q (u - zp)); the zero-point term zp * colsum_k is a
 * per-column int16 constant the softmax subtracts. The cache must be in
 * fixed-scale mode (hexkl_kv_q_set_fixed_scales): K's scale joins Q's in
 * the softmax's residual factor, V's per-dim scale and the output's
 * per-head scale and zero point are applied in the epilogue. Logits are
 * scaled by 1/sqrt(head_dim) as MHACoreLayer does.
 */

#ifndef __NNTRAINER_HEXKL_ATTN_Q2_H__
#define __NNTRAINER_HEXKL_ATTN_Q2_H__

#include <stdint.h>

#include "hexkl_attn_q2_plan.h"
#include "hexkl_kv_q.h"
#include "hvx_worker_pool.h"

typedef struct {
  const uint16_t *q;  /**< [n_q][q_stride] u16, head h at column h*hd */
  uint32_t q_stride;  /**< elements */
  const float *q_enc; /**< [n_head_q][2]: scale, zero point of Q per head,
                           as the model's encodings give them */
  uint16_t *out;      /**< [n_q][out_stride] u16 */
  uint32_t out_stride;
  const float *out_enc; /**< [n_head_q][2]: scale, zero point of the
                             context output per head */
  const hexkl_kv_q *kv; /**< fixed-scale int8 cache, >= cache_to rows */
} hexkl_attn_q2_io;

/**
 * @brief On-DSP time per stage, microseconds, summed over the call. The
 *        worker stages (qprep, softmax, epi) are summed over the blocks as
 *        the workers measured them and overlap each other and the HMX
 *        stages; us_wait is what the HMX thread spent waiting for them,
 *        i.e. the exposed HVX time.
 */
typedef struct {
  uint64_t us_qprep;   /**< Q byte split into tiles (workers) */
  uint64_t us_dma;     /**< exposed DMA: drains the HMX thread waited on */
  uint64_t us_qk;      /**< HMX Q.K^T incl. converts and zips */
  uint64_t us_softmax; /**< hvx_softmax_q16 (workers) */
  uint64_t us_pv;      /**< HMX P.V incl. converts and zips */
  uint64_t us_epi;     /**< int16 -> u16 rows in DDR (workers) */
  uint64_t us_wait;    /**< HMX thread waiting for the workers */
  uint64_t us_head;    /**< per-head constants (HMX thread) */
  uint64_t us_submit;  /**< handing jobs to the pool (HMX thread) */
  uint64_t us_total;
  uint64_t pcycles;
  uint32_t n_blocks; /**< (q head, row block) pairs run */
} hexkl_attn_q2_stats;

/**
 * @brief Runs causal (optionally windowed) attention for one step.
 *
 * Requires the HMX lock. The calling thread runs the HMX stages; @a pool's
 * workers run the rest (NULL or an empty pool runs everything inline).
 * @a st may be NULL.
 *
 * @return AEE_SUCCESS, AEE_EBADPARM, AEE_ENOMEMORY
 */
int hexkl_attn_q2_prefill(uint8_t *vtcm_base, uint32_t arena_top,
                          const hexkl_attn_f16_shape *s,
                          const hexkl_attn_q2_io *io, hvx_worker_pool *pool,
                          hexkl_attn_q2_stats *st);

#endif /* __NNTRAINER_HEXKL_ATTN_Q2_H__ */
