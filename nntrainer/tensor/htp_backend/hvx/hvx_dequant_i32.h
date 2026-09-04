// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file hvx_dequant_i32.h
 * @date 03 Aug 2026
 * @brief int32 accumulator to f32 dequantization for the A8W4 path
 * @see https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug No known bugs except for NYI items */

#ifndef __NNTRAINER_HVX_DEQUANT_I32_H__
#define __NNTRAINER_HVX_DEQUANT_I32_H__

#include <stdint.h>

/**
 * @brief Turns HMX int32 accumulators back into f32 (K3).
 *
 * out[m][n] = (acc[m][n] - act_zp[m]*colsum_w[n]) * act_scale[m]
 * * w_scale[n] + bias[n]
 *
 * The zp*colsum term corrects for HMX taking unsigned activations: with
 * x = s*(u - zp), the dot product expands to s*d*(sum u*q_w - zp*sum q_w),
 * and sum q_w over the reduction depends only on the weights, so it is a
 * precomputed table rather than runtime work.
 *
 * HexKL micro exposes no bias, scale or zero-point registers, which is why
 * this pass exists at all.
 *
 * @param[in] acc m_pad by n int32, row-major
 * @param[in] m_valid rows to emit; padded rows are skipped because their
 * quantization parameters are synthetic
 * @param[in] colsum_w per-channel sum of the int4 weights, n entries
 * @param[in] w_scale per-channel dequantization multiplier, n entries
 * @param[out] out m_valid by n f32, row-major */
void hvx_dequant_i32_to_f32(const int32_t *acc, uint32_t m_valid,
                            uint32_t m_pad, uint32_t n, const float *act_scale,
                            const int32_t *act_zp, const int32_t *colsum_w,
                            const float *w_scale, const float *bias, float *out,
                            int accumulate);

/**
 * @brief Same dequantization, applied to ONE 64x32 accumulator tile still
 * sitting in VTCM.
 *
 * The formula, the intrinsics and their order are hvx_dequant_i32_to_f32's,
 * element for element this is where the tile comes from and where the
 * result goes that differ, never the arithmetic. Results are therefore
 * bitwise identical to dequantizing the same tile after a round trip through
 * a DDR scratch, which is what makes the existing bit-exactness gates a valid
 * check on this path (hexkl_acc_tile.h explains why the round trip existed).
 *
 * A tile column count of 32 is exactly one HVX f32 vector, so each row is a
 * single load, a single store and no tail loop.
 *
 * @param[in] tile first element of the tile's row 0, i.e. the VTCM
 * result tile advanced by hexkl_acc_layout::base
 * @param[in] row_stride int32 elements between tile rows (from the probe)
 * @param[in] m_count tile rows carrying real data; padded rows are
 * skipped, same rule as m_valid above
 * @param[in] act_scale m_count entries, already offset to this row block
 * @param[in] act_zp m_count entries, already offset to this row block
 * @param[in] colsum_w 32 entries, already offset to this column tile
 * @param[in] w_scale 32 entries, already offset to this column tile
 * @param[in] bias 32 entries, already offset to this column tile
 * @param[out] out first output element of this tile
 * @param[in] out_stride f32 elements between output rows (the full N)
 * @param[in] accumulate nonzero adds into @a out instead of storing; one
 * add per element per call either way, so this is
 * bitwise the staged add it replaces */
void hvx_dequant_acc_tile_to_f32(const int32_t *tile, uint32_t row_stride,
                                 uint32_t m_count, const float *act_scale,
                                 const int32_t *act_zp, const int32_t *colsum_w,
                                 const float *w_scale, const float *bias,
                                 float *out, uint32_t out_stride,
                                 int accumulate);

#endif /* __NNTRAINER_HVX_DEQUANT_I32_H__ */
