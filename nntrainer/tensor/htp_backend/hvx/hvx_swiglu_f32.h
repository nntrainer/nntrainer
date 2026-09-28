// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hvx_swiglu_f32.h
 * @date   08 Sep 2026
 * @brief  In-place SwiGLU over f32 rows for the fused MoE FFN path
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __NNTRAINER_HVX_SWIGLU_F32_H__
#define __NNTRAINER_HVX_SWIGLU_F32_H__

#include <stdint.h>

#include "hvx_worker_pool.h"

/**
 * @brief gate[r][j] = silu(gate[r][j]) * up[r][j], in place, for every row.
 *
 * silu(x) = x / (1 + exp(-x)), matching avx2_impl.cpp's swiglu reference
 * element for element up to the HVX exp/reciprocal's ~1e-6 relative error.
 * The result feeds a uint8 requantization downstream, so the fused layer is
 * gated on SNR, not bit equality (doc 43 §[L2]).
 *
 * Rows are independent, so @a pool splits by row range; NULL runs
 * single-threaded. @a gate and @a up must not overlap.
 *
 * @param[in,out] gate  m_valid rows by n_out columns, row-major f32
 * @param[in]     up    same shape; read-only
 * @param[in]     m_valid  rows to process
 * @param[in]     n_out    columns per row
 */
void hvx_swiglu_inplace_f32(float *gate, const float *up, uint32_t m_valid,
                            uint32_t n_out, hvx_worker_pool *pool);

#endif /* __NNTRAINER_HVX_SWIGLU_F32_H__ */
