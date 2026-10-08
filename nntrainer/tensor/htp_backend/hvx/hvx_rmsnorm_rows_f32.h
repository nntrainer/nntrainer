// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 * @file   hvx_rmsnorm_rows_f32.h
 * @date   6 October 2026
 * @brief  RMSNorm over M rows of f32 on HVX, in chunks, with an optional gamma
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The prefill-shape twin of hvx_m1_ops_f32.h's hvx_rmsnorm_f32: that one is
 * the bit-exact M=1 spec at a power-of-two chunk; this one takes any chunk
 * that is a multiple of 32 (a 2816-wide hidden norm, a 256- or 512-wide
 * per-head norm), M rows across the worker pool, and plain f32 arithmetic
 * (a qf32 sum of squares, sqrtf on the scalar core). It exists so a layer
 * call can fold the norms around its projections into the same FastRPC
 * call (doc 57 section 5 step 4) rather than round-trip every row twice.
 */

#ifndef __NNTRAINER_HVX_RMSNORM_ROWS_F32_H__
#define __NNTRAINER_HVX_RMSNORM_ROWS_F32_H__

#include <stdint.h>

#include "hvx_worker_pool.h"

/**
 * @brief y[r][c*chunk + j] = x[r][c*chunk + j] * rs(r, c) * gamma[j], with
 *        rs = 1 / sqrt(mean of squares over the chunk + eps).
 *
 * @param x, y     M rows of n floats; y may be x (each chunk is read in
 *                 full before it is written)
 * @param chunk    a multiple of 32 dividing n; n for a whole-row norm
 * @param gamma    chunk floats shared by every chunk and row; NULL = 1
 * @param pool     the worker pool rows are split across; NULL = inline
 * @return 0, or -1 for a shape the kernel does not take (nothing written)
 */
int hvx_rmsnorm_rows_f32(const float *x, float *y, uint32_t M, uint32_t n,
                         uint32_t chunk, const float *gamma, float eps,
                         hvx_worker_pool *pool);

/**
 * @brief The decoder block's epilogue, streamed: out[r][j] = scale *
 *        (out[r][j] + (x[r][j] + x2[r][j]) * rs(r) * gamma[j]), with rs the
 *        whole-row RMSNorm scale of x + x2.
 *
 * The residual comes in through out and the result replaces it, so the
 * post-attention / post-FFN norm, the residual add and the block's scalar
 * (and the dense + MoE sum, x2) leave the CPU as one call (doc 57 section
 * 5 step 4). Two passes over x and x2 per row and no scratch.
 *
 * @param out      M rows of n floats, the residual in and the result out
 * @param x, x2    M rows of n floats each; x2 NULL for one addend
 * @param n        a multiple of 32
 * @param gamma    n floats; NULL = 1
 * @param pool     the worker pool rows are split across; NULL = inline
 * @return 0, or -1 for a shape the kernel does not take (nothing written)
 */
int hvx_rmsnorm_add_f32(float *out, const float *x, const float *x2, uint32_t M,
                        uint32_t n, const float *gamma, float eps, float scale,
                        hvx_worker_pool *pool);

#endif /* __NNTRAINER_HVX_RMSNORM_ROWS_F32_H__ */
