// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hvx_router_rows_f32.h
 * @date   6 October 2026
 * @brief  The MoE router's logits over M rows of f32 on HVX
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * logits[M][E] = x[M][K] . w[K][E] in f32, the matmul a MoE layer runs
 * before its top-k. hvx_router_topk_f32 (hvx_m1_ops_f32.h) is the one-row,
 * E <= 32, sigmoid version for the resident decode graph; this one takes
 * prefill's rows, E up to 128 (the softmax router's width), and leaves the
 * softmax and the selection to the caller, so the selection rule stays the
 * CPU's (doc 57 section 5 step 5). The weight is read once per 256-row
 * chunk for every row of x, so the K x E read stays in L2 instead of
 * streaming from DDR once per row.
 */

#ifndef __NNTRAINER_HVX_ROUTER_ROWS_F32_H__
#define __NNTRAINER_HVX_ROUTER_ROWS_F32_H__

#include <stdint.h>

#include "hvx_worker_pool.h"

/**
 * @brief logits[r][e] = sum_k x[r][k] * w[k][e], qf32 accumulation within
 *        a 256-row chunk of w, f32 between chunks.
 *
 * @param x       M rows of K floats
 * @param w       K rows of E floats (the [K][E] gate weight, row-major)
 * @param logits  M rows of E floats out
 * @param E       a multiple of 32, at most 128
 * @param pool    the worker pool rows are split across; NULL = inline
 * @return 0, or -1 for a shape the kernel does not take (nothing written)
 */
int hvx_router_rows_f32(const float *x, const float *w, float *logits,
                        uint32_t M, uint32_t K, uint32_t E,
                        hvx_worker_pool *pool);

#endif /* __NNTRAINER_HVX_ROUTER_ROWS_F32_H__ */
