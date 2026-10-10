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

/** @brief hvx_rmsnorm_rows_f32 over rows @a ld floats apart (ld >= n). */
int hvx_rmsnorm_rows_ld_f32(const float *x, float *y, uint32_t M, uint32_t n,
                            uint32_t ld, uint32_t chunk, const float *gamma,
                            float eps, hvx_worker_pool *pool);

/** @brief One row of hvx_rmsnorm_rows_f32 with chunk = n, inline on the
 *         caller: the same arithmetic, so the same floats. */
void hvx_rmsnorm_row_f32(const float *x, float *y, const float *gamma,
                         uint32_t n, float eps);

/** @brief The scale one row of hvx_rmsnorm_row_f32 multiplies by,
 *         1 / sqrt(mean(x^2) + eps): the same arithmetic, so the same float.
 *         The fused consumers (the router logits) apply it themselves. */
float hvx_rmsnorm_rs_f32(const float *x, uint32_t n, float eps);

/** @brief One row of hvx_rmsnorm_rows_ld_f32: n floats normed @a chunk at a
 *         time (the same norm_chunk per chunk), inline on the caller. */
void hvx_rmsnorm_row_chunks_f32(const float *x, float *y, const float *gamma,
                                uint32_t n, uint32_t chunk, float eps);

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

/**
 * @brief hvx_rmsnorm_add_f32 with the residual read from @a res instead of
 *        @a out: out = scale * (res + rmsnorm(x [+ x2]) * gamma). @a res
 *        NULL is hvx_rmsnorm_add_f32. @a res may be @a out; it must not
 *        overlap it otherwise.
 */
int hvx_rmsnorm_add_res_f32(float *out, const float *res, const float *x,
                            const float *x2, uint32_t M, uint32_t n,
                            const float *gamma, float eps, float scale,
                            hvx_worker_pool *pool);

/**
 * @brief hvx_rmsnorm_add_res_f32 with x given as column blocks: block b is
 *        an M x cols[b] row-major matrix holding columns [sum of the
 *        earlier cols, + cols[b]) -- the per-handle layout
 *        hexkl_mm_u8i4_layer_run writes. Every cols[b] a multiple of 32;
 *        out and res are M x sum(cols) row-major and must not overlap x.
 */
int hvx_rmsnorm_add_blocks_f32(float *out, const float *res,
                               const float *const *xb, const uint32_t *cols,
                               uint32_t n_blk, uint32_t M, const float *gamma,
                               float eps, float scale, hvx_worker_pool *pool);

#endif /* __NNTRAINER_HVX_RMSNORM_ROWS_F32_H__ */
