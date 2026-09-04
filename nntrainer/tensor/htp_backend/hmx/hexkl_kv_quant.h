// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file hexkl_kv_quant.h
 * @date 06 Aug 2026
 * @brief Per-KV-block symmetric quantizer for the micro u8i4 / u8i8 attention
 * path. HOST-only arithmetic: no HVX, no HexKL calls, no DSP headers.
 * @see https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug No known bugs.
 *
 * Turns appended fp16 K and V rows into exactly the inputs
 * hexkl_weight_u8i4_register / hexkl_weight_u8i8_register want, one KV block at
 * a time. The width is a runtime parameter; ONE code path covers both.
 *
 * CPU cache layout (mha_core.cpp:61, :95) is [kv_position][nch][head_dim] fp16:
 * row r of head h starts at (r*nch + h)*head_dim.
 *
 * out_rm is one int8 container per value for BOTH widths rm_to_wh_i4 and
 * rm_to_wh_i8 both take that. The 4-bit pack happens inside the WH bake.
 *
 * Quantization scheme:
 * - Kt block [head_dim][T], N = kv position -> symmetric per-N (per kv
 * position) quantization.
 * - V block [T][head_dim], N = head_dim column -> symmetric per-N (per
 * column) over THIS BLOCK's T rows only (blocking fixes the scale axis,
 * see ).
 *
 * Symmetric ranges:
 * i4 -> [-8, 7] scale denominator = 7
 * i8 -> [-127, 127] scale denominator = 127 (not -128: symmetric scale
 * stays exact)
 * Round to nearest, ties away from zero.
 *
 * Partial tail (n_rows_valid < T): zero-fill every unused element of out_rm,
 * set the corresponding out_scale to 1.0f and out_colsum to 0. A garbage tail
 * is a silent wrong answer later; the mask alone will not save it. */

#ifndef __NNTRAINER_HEXKL_KV_QUANT_H__
#define __NNTRAINER_HEXKL_KV_QUANT_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Weight width. Value doubles as the per-tile clamp target selector. */
typedef enum { HEXKL_W_I4 = 4, HEXKL_W_I8 = 8 } hexkl_w_width;

/**
 * @brief Pack one Kt block: logical [head_dim][T], N = kv position.
 *
 * Symmetric per-N (per kv position) quantization. For each kv position r in
 * [0, n_rows_valid): amax over that position's head_dim values, scale =
 * amax / denom (denom=7 i4, 127 i8), q = round(x/scale) clamped to the width's
 * range. For r in [n_rows_valid, T): zero-fill out_rm, out_scale[r]=1.0f,
 * out_colsum[r]=0.
 *
 * @param k_rows_f16 [n_rows_valid][nch][head_dim] fp16 (CPU cache layout);
 * entries past n_rows_valid are not read.
 * @param n_rows_valid number of valid kv positions in this block (<= T).
 * @param T block size (kv positions per block); must be > 0.
 * @param head_dim per-head dimension.
 * @param nch number of cache heads.
 * @param head which cache head (0..nch-1).
 * @param w width (HEXKL_W_I4 or HEXKL_W_I8).
 * @param[out] out_rm head_dim * T int8 containers, row-major [head_dim][T]
 * (row d, col r at d*T + r).
 * @param[out] out_scale T floats, one per kv position (per-N weight scale).
 * @param[out] out_colsum T int32s; colsum_w[n] = sum_d out_rm[d*T + n]. */
void hexkl_kvq_pack_kt_block(const uint16_t *k_rows_f16, uint32_t n_rows_valid,
                             uint32_t T, uint32_t head_dim, uint32_t nch,
                             uint32_t head, hexkl_w_width w, int8_t *out_rm,
                             float *out_scale, int32_t *out_colsum);

/**
 * @brief Pack one V block: logical [T][head_dim], N = head_dim column.
 *
 * Symmetric per-N (per head_dim column) quantization over THIS BLOCK's T rows
 * only. For each column d in [0, head_dim): amax over the n_rows_valid values
 * in that column, scale = amax / denom, q = round(x/scale) clamped. For rows
 * in [n_rows_valid, T): zero-fill (column sums ignore them).
 *
 * @param v_rows_f16 [n_rows_valid][nch][head_dim] fp16.
 * @param[out] out_rm T * head_dim int8 containers, row-major [T][head_dim]
 * (row t, col d at t*head_dim + d).
 * @param[out] out_scale head_dim floats, one per column (per-N weight scale).
 * @param[out] out_colsum head_dim int32s; colsum_w[n] = sum_t out_rm[t*head_dim
 * + n]. */
void hexkl_kvq_pack_v_block(const uint16_t *v_rows_f16, uint32_t n_rows_valid,
                            uint32_t T, uint32_t head_dim, uint32_t nch,
                            uint32_t head, hexkl_w_width w, int8_t *out_rm,
                            float *out_scale, int32_t *out_colsum);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* __NNTRAINER_HEXKL_KV_QUANT_H__ */
