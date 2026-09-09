// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   htp_act_quant.h
 * @date   08 Sep 2026
 * @brief  ARM-side reproduction of the DSP's per-row u8 activation
 *         quantize + AH-tile pack (K1+K2 of hexkl_mm_u8i4_layer_run)
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * hexkl_mm_u8i4_layer_run (nntrainer/tensor/htp_backend/hmx/
 * hexkl_mm_u8i4_dma.c, DSP-side) does this same math on every call: per-row
 * min/max -> scale/zp (hvx_quant_rows_u8_params), then quantize + permute
 * into HMX's AH tile layout (hvx_quant_pack_u8_ah). Doing it there costs
 * DSP time that the docs/htp_attention measurement pins at 14-22% of
 * dsp_total for the MoE FFN's prefill shapes, AND it forces the activation
 * to cross FastRPC as f32 (4 bytes/value) instead of the u8 AH bytes this
 * function produces (1 byte/value plus a handful of scale/zp floats).
 *
 * Doing the identical arithmetic here, once, on the ARM side that already
 * holds the f32 activation, removes both costs for any caller willing to
 * hand the DSP pre-packed AH bytes instead of raw f32 -- see
 * hexkl_mm_u8i4_layer_run's act_ah_prepacked option.
 *
 * The tile geometry (64 rows x 32 columns, 2048 bytes/tile, tiles ordered
 * (row_block, k_tile)) and the quantization math (rmin/rmax folding in 0,
 * scale = range/255, round-to-nearest-even, zp clamped to [0,255]) are
 * restated here bit-for-bit from hvx_quant_u8.c's comments and hexkl_
 * micro.h's HEXKL_HMX_INT8_BLOCK_N_ROW/_COL/HEXKL_HMX_ACTIVATION_ALIGNMENT
 * -- this file cannot include either (one is DSP-only C, the other is an
 * SDK header not on the host build path), so the constants are kept in
 * sync by inspection, the same convention htp_compute_ops.cpp's HTP_T_*
 * enum and hexkl_mm_u8i4_dma.c's WEIGHT_TILE_BYTES_U8I4 already use.
 *
 * Plain host code, no Hexagon SDK dependency -- like htp_q4_0_convert.h
 * next to it, this is not gated behind ENABLE_HEXKL and its unit test runs
 * in the default host build.
 */

#ifndef __NNTRAINER_HTP_ACT_QUANT_H__
#define __NNTRAINER_HTP_ACT_QUANT_H__
#ifdef __cplusplus

#include <cstdint>

namespace nntrainer {

/** @brief HMX activation tile geometry -- see this file's header comment
 *  for where these are pinned. */
constexpr uint32_t HTP_ACT_TILE_ROW = 64u;
constexpr uint32_t HTP_ACT_TILE_INNER = 32u;
constexpr uint32_t HTP_ACT_TILE_BYTES = 2048u;

/**
 * @brief Rounds @a m up to a multiple of ::HTP_ACT_TILE_ROW -- the same
 *        m_pad every caller of hexkl_mm_u8i4_layer_run already computes,
 *        restated here so callers of htp_quant_pack_u8_ah don't have to
 *        duplicate the rounding formula themselves.
 */
inline uint32_t htp_act_m_pad(uint32_t m) {
  return ((m + HTP_ACT_TILE_ROW - 1) / HTP_ACT_TILE_ROW) * HTP_ACT_TILE_ROW;
}

/**
 * @brief Quantizes an f32 activation to u8 and packs it into HMX's AH tile
 *        layout, bit-for-bit matching hvx_quant_rows_u8_params +
 *        hvx_quant_pack_u8_ah's combined output.
 *
 * @param[in]  x        activation, M rows by K columns, row-major f32
 * @param[in]  M         rows carrying real data
 * @param[in]  K         columns; must be a multiple of ::HTP_ACT_TILE_INNER
 *                       (HexKL's own K-divisibility requirement, already
 *                       enforced wherever this weight shape was accepted)
 * @param[out] out_ah    m_pad(M) * K bytes -- m_pad's padding rows come out
 *                       zeroed, matching the DSP path's memset
 * @param[out] act_scale m_pad(M) entries; x[m][k] ~= act_scale[m] *
 *                       (out_ah's dequantized value - act_zp[m])
 * @param[out] act_zp    m_pad(M) entries, each in [0, 255]
 */
void htp_quant_pack_u8_ah(const float *x, uint32_t M, uint32_t K,
                          uint8_t *out_ah, float *act_scale, int32_t *act_zp);

} // namespace nntrainer

#endif // __cplusplus
#endif // __NNTRAINER_HTP_ACT_QUANT_H__
