// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hvx_quant_u8.h
 * @date   03 Aug 2026
 * @brief  Per-row asymmetric uint8 dynamic activation quantization
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __NNTRAINER_HVX_QUANT_U8_H__
#define __NNTRAINER_HVX_QUANT_U8_H__

#include <stdint.h>

/**
 * @brief Computes the per-row scale and zero point (K1).
 *
 * x[m][k] is recovered as scale[m] * (u[m][k] - zp[m]).
 *
 * Rows at or past @a m_valid are padding: they get scale 1 and zp 0 so a
 * host reference can reproduce them without special cases.
 *
 * @param[in]  x        activation, m_valid rows by k columns, row-major f32
 * @param[in]  m_valid  rows carrying real data
 * @param[in]  m_pad    rows after padding up to a multiple of 64
 * @param[out] scale    m_pad entries
 * @param[out] zp       m_pad entries, each in [0, 255]
 */
void hvx_quant_rows_u8_params(const float *x, uint32_t m_valid, uint32_t m_pad,
                              uint32_t k, float *scale, int32_t *zp);

/**
 * @brief Quantizes to uint8 and writes AH tiles (K2).
 *
 * Writes directly in the layout the HMX activation port expects: 64x32
 * tiles, flat row-major inside a tile, tiles in (row_block, inner_tile)
 * order at a 2048-byte stride. No separate layout pass is needed for
 * 8-bit activations.
 *
 * Rounding is round-to-nearest-even, matching hvx_sf_to_w_rne.
 *
 * @param[in]  out_ah  destination, m_pad * k bytes. Usually VTCM, and then
 *                     it must be 2048-byte aligned.
 */
void hvx_quant_pack_u8_ah(const float *x, uint32_t m_valid, uint32_t m_pad,
                          uint32_t k, const float *scale, const int32_t *zp,
                          uint8_t *out_ah);

#endif /* __NNTRAINER_HVX_QUANT_U8_H__ */
