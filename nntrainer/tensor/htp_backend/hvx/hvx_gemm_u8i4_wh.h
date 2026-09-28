// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hvx_gemm_u8i4_wh.h
 * @date   16 Sep 2026
 * @brief  HVX u8 x i4 matmul over HMX-layout tiles, bit-identical to HMX
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __NNTRAINER_HVX_GEMM_U8I4_WH_H__
#define __NNTRAINER_HVX_GEMM_U8I4_WH_H__

#include <stdint.h>

/** @brief Most activation rows one call handles. The MoE kernel's tail
 *         blocks are at most this many rows (hexkl_mm_u8i4_moe.c); the
 *         HMX unit's own block is 64 and that is what a bigger block goes
 *         to. */
#define HVX_GEMM_U8I4_MAX_ROWS 16u

/**
 * @brief One n-tile column of C = A x W for @a m rows, exact int32 --
 *        the same numbers the HMX unit produces for the same tiles, since
 *        both are plain integer sums of u8 x i4 products (no rounding
 *        anywhere), so a row computed here and a row computed by the HMX
 *        dequantize to identical bytes.
 *
 * Reads the operands as the HMX sees them: @a act_ah in AH tiles (64 rows
 * x 32 bytes per k-tile at a 2048-byte stride, row r at r*32) and @a wh in
 * WH tiles (32x32 i4, 512 bytes, tile (kt, nt) at (kt*n_col + nt)*512,
 * nibble layout per htp_wh_layout.h: byte (r/8)*128 + c*4 + r%4, low
 * nibble for rows 8g..8g+3, high for 8g+4..8g+7). That layout is what
 * makes this cheap: each 128-byte quarter of a tile is one HVX vector
 * whose lane c already holds the four consecutive k values of column c,
 * which is exactly what vrmpy multiplies against four activation bytes.
 *
 * @param act_ah   the row block's AH tiles, k_tiles of them
 * @param m        rows to compute, 1..HVX_GEMM_U8I4_MAX_ROWS
 * @param k_tiles  reduction length in 32-wide tiles
 * @param wh       the weight's WH bytes (DDR is fine; a column is 2D
 *                 prefetched)
 * @param n_col    the weight's n-tile count (tile row stride)
 * @param nt       the n-tile column to compute
 * @param out      m x 32 int32, row-major, row stride 32 -- the same shape
 *                 hexkl_micro_hmx_acc_read_int32 lands with row_stride 32,
 *                 so the existing dequant passes read it unchanged
 */
void hvx_gemm_u8i4_wh_col(const uint8_t *act_ah, uint32_t m, uint32_t k_tiles,
                          const uint8_t *wh, uint32_t n_col, uint32_t nt,
                          int32_t *out);

#endif /* __NNTRAINER_HVX_GEMM_U8I4_WH_H__ */
