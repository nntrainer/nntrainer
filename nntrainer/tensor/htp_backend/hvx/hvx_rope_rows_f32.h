// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hvx_rope_rows_f32.h
 * @date   6 October 2026
 * @brief  RoPE over M rows of heads of f32 on HVX, in place
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The CPU's rotary embedding (compute_rotary_emb_value) over prefill rows:
 * each head of hd floats rotates its pairs (j, j + hd/2) by the position's
 * cos / sin, so the q and k projections leave the accelerator rotated and
 * the attention core has nothing left to do with them but the fp16 copy
 * into the cache (doc 57 section 5 step 4). hvx_rope64_f32 is the one-row,
 * head_dim 64 version for the resident decode graph.
 */

#ifndef __NNTRAINER_HVX_ROPE_ROWS_F32_H__
#define __NNTRAINER_HVX_ROPE_ROWS_F32_H__

#include <stdint.h>

#include "hvx_worker_pool.h"

/**
 * @brief For every row r and head, for j < hd/2:
 *        a = x[j], b = x[j + hd/2];
 *        x[j] = a * cos[j] - b * sin[j]; x[j + hd/2] = a * sin[j] + b * cos[j]
 *        -- the CPU kernel's operation order, so the rows match it.
 *
 * @param x    M rows of n floats, n a multiple of hd; rotated in place
 * @param hd   the head dim; hd/2 a multiple of 32
 * @param cs   M rows of 2*hd floats: the CPU table's cos row (hd floats,
 *             halves duplicated) then its sin row, for the row's position;
 *             only the first hd/2 of each are read
 * @param pool the worker pool rows are split across; NULL = inline
 * @return 0, or -1 for a shape the kernel does not take (nothing written)
 */
int hvx_rope_rows_f32(float *x, uint32_t M, uint32_t n, uint32_t hd,
                      const float *cs, hvx_worker_pool *pool);

/** @brief hvx_rope_rows_f32 over rows @a ld floats apart (ld >= n). */
int hvx_rope_rows_ld_f32(float *x, uint32_t M, uint32_t n, uint32_t ld,
                         uint32_t hd, const float *cs, hvx_worker_pool *pool);

/**
 * @brief hvx_rmsnorm_rows_ld_f32(x, x, M, n, ld, chunk, gamma, eps) then
 *        hvx_rope_rows_ld_f32(x, M, n, ld, hd, cs), row by row in one pass:
 *        the same floats as the two calls, without reading x twice.
 */
int hvx_norm_rope_rows_ld_f32(float *x, uint32_t M, uint32_t n, uint32_t ld,
                              uint32_t chunk, const float *gamma, float eps,
                              uint32_t hd, const float *cs,
                              hvx_worker_pool *pool);

#endif /* __NNTRAINER_HVX_ROPE_ROWS_F32_H__ */
