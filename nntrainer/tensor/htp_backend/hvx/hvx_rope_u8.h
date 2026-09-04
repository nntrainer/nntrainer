// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hvx_rope_u8.h
 * @brief  Row-major uint8-in/uint8-out NeoX RoPE kernel.
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs.
 */

#ifndef __NNTRAINER_HVX_ROPE_U8_H__
#define __NNTRAINER_HVX_ROPE_U8_H__

#include <stdint.h>

/**
 * @brief Applies NeoX half-rotation and requantizes each row to uint8.
 *
 * cos_u8 and sin_u8 contain [n_rows][dim / 2] uint8 values indexed by row
 * m (not m - m_first). Each carries its own per-tensor (scale, zero-point).
 * s_in/zp_in are read as s_in[m * s_in_stride] and
 * zp_in[m * zp_in_stride] pass stride 0 for a single value broadcast
 * to every row, or 1 for one entry per row. y may alias x.
 *
 * Channels past the last whole `dim` segment (when width % dim != 0) are
 * copied unrotated, but only when y != x. width % dim == 0 in every real
 * caller (width = n_heads * head_dim).
 *
 * ponytail: the u8 widen and the u8 pack around the vector core are scalar
 * loops. Upgrade path: Q6_Wuh_vunpack_Vub in, Q6_Vh_vpack_VwVw_sat ->
 * Q6_Vub_vpack_VhVh_sat out, and a vector compare against [0,255] for the
 * saturation count instead of the scalar one. */
void hvx_rope_u8_rows(const uint8_t *x, uint8_t *y, uint32_t m_first,
                      uint32_t m_last, uint32_t width, uint32_t dim,
                      const float *s_in, uint32_t s_in_stride,
                      const int32_t *zp_in, uint32_t zp_in_stride,
                      const uint8_t *cos_u8, const uint8_t *sin_u8,
                      float cos_scale, int32_t cos_zp, float sin_scale,
                      int32_t sin_zp, float s_out, int32_t zp_out,
                      uint32_t *n_saturated);

#endif /* __NNTRAINER_HVX_ROPE_U8_H__ */
