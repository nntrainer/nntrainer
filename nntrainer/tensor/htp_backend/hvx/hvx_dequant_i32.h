// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hvx_dequant_i32.h
 * @date   03 Aug 2026
 * @brief  int32 accumulator to f32 dequantization for the A8W4 path
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __NNTRAINER_HVX_DEQUANT_I32_H__
#define __NNTRAINER_HVX_DEQUANT_I32_H__

#include <stdint.h>

/**
 * @brief Turns HMX int32 accumulators back into f32 (K3).
 *
 * out[m][n] = (acc[m][n] - act_zp[m]*colsum_w[n]) * act_scale[m]
 *             * w_scale[n] + bias[n]
 *
 * The zp*colsum term corrects for HMX taking unsigned activations: with
 * x = s*(u - zp), the dot product expands to s*d*(sum u*q_w - zp*sum q_w),
 * and sum q_w over the reduction depends only on the weights, so it is a
 * precomputed table rather than runtime work.
 *
 * HexKL micro exposes no bias, scale or zero-point registers, which is why
 * this pass exists at all.
 *
 * @param[in]  acc      m_pad by n int32, row-major
 * @param[in]  m_valid  rows to emit; padded rows are skipped because their
 *                      quantization parameters are synthetic
 * @param[in]  colsum_w per-channel sum of the int4 weights, n entries
 * @param[in]  w_scale  per-channel dequantization multiplier, n entries
 * @param[out] out      m_valid by n f32, row-major
 */
void hvx_dequant_i32_to_f32(const int32_t *acc, uint32_t m_valid,
                            uint32_t m_pad, uint32_t n, const float *act_scale,
                            const int32_t *act_zp, const int32_t *colsum_w,
                            const float *w_scale, const float *bias,
                            float *out);

/**
 * @brief Dequantizes one 64x32 accumulator tile straight out of VTCM.
 *
 * Same formula as hvx_dequant_i32_to_f32, applied to one HMX output tile
 * instead of the whole matrix, so the int32 accumulator never has to go
 * to DDR and come back.
 *
 * A tile row is exactly 32 lanes, which is exactly one 128-byte HVX
 * vector, so there is no tail to handle and every store is a full vector.
 * The column operands are the same for every row of the tile and are
 * loaded once outside the row loop.
 *
 * The caller offsets every pointer, so this function knows nothing about
 * tile indices. @a tile must be 128-byte aligned (VTCM); the DDR-side
 * pointers need not be.
 *
 * @param[in]  tile      n_rows by 32 int32, row-major, 128-byte aligned
 * @param[in]  n_rows    rows to emit, 1 to 64; padded rows are excluded by
 *                       the caller so their synthetic scale never applies
 * @param[in]  act_scale per row, already offset to this tile's first row
 * @param[in]  act_zp    per row, already offset to this tile's first row
 * @param[in]  colsum_w  32 entries, already offset to this tile's column
 * @param[in]  w_scale   32 entries, already offset to this tile's column
 * @param[in]  bias      32 entries, already offset to this tile's column
 * @param[out] out       already offset to out_f32 + row0*stride + col0
 * @param[in]  out_stride elements between consecutive rows of @a out
 */
void hvx_dequant_tile_i32_to_f32(const int32_t *tile, uint32_t n_rows,
                                 const float *act_scale, const int32_t *act_zp,
                                 const int32_t *colsum_w, const float *w_scale,
                                 const float *bias, float *out,
                                 uint32_t out_stride);

#endif /* __NNTRAINER_HVX_DEQUANT_I32_H__ */
