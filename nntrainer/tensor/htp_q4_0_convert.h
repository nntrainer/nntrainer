// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   htp_q4_0_convert.h
 * @date   04 Sep 2026
 * @brief  Q4_0x4 (ARM-repacked) weight -> HexKL qs4cx conversion
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * HtpComputeOps::gemm_q4_0_accel_fp32 (nntrainer/tensor/htp_backend/, built
 * only when ENABLE_HEXKL is defined) is the only caller, and receives
 * whatever bytes the model file stored -- Q4_0x4-repacked
 * (Q4_0Utils::dequantizeQ4_0x4's format), because ENABLE_HEXKL only ever
 * builds for Android/ARM64 (Qualcomm Snapdragon HTP), and on that target
 * nntrainer::repack_q4_0's ARM and DEFAULT branches both choose the x4
 * (4-wide) layout -- see arm_compute_backend.cpp. There is no x8 (X86) case
 * to handle here; if this file is ever compiled into a build that isn't
 * Android/ARM64, that assumption needs revisiting first.
 *
 * This file itself has no Hexagon SDK dependency -- it is plain host code
 * against Q4_0Utils, same as q4_0_utils.cpp next to it -- so it is not
 * gated behind ENABLE_HEXKL and its unit test runs in the default host
 * build, no device or SDK required.
 */

#ifndef __NNTRAINER_HTP_Q4_0_CONVERT_H__
#define __NNTRAINER_HTP_Q4_0_CONVERT_H__
#ifdef __cplusplus

#include <cstdint>

namespace nntrainer {

/**
 * @brief Requantizes a Q4_0x4-repacked weight into HexKL's qs4cx-shaped
 *        registry input: int4 values in int8 containers, K rows by N
 *        columns row-major, plus one dequant scale and one colsum per
 *        output channel (N).
 *
 * This is a real requantization, not a bit reshuffle: Q4_0 keeps one scale
 * per 32-wide block along K, qs4cx keeps one scale for the whole column, so
 * going from one to the other has to pass back through f32. The accuracy
 * cost of that is measured by this file's unit test, not assumed.
 *
 * @param[in]  q4_0x4_repacked Q4_0x4-packed weight, N rows of K values each
 *             (Q4_0Utils::dequantizeQ4_0x4's own convention -- this is the
 *             transposed-at-save-time layout quantize_q4_0 + repack_q4_0
 *             produce; see Lfm2MoELayer::save for where that transpose
 *             happens). data_size = N * ceil(K/128) * sizeof(block_q4_0x4).
 * @param[in]  K HexKL's K (input dimension, matches the FC weight's height)
 * @param[in]  N HexKL's N (output dimension, matches the FC weight's width)
 * @param[out] q_w4_i8 K*N int4 values in [-8, 7] stored as int8, row-major
 * @param[out] w_scale N entries: value[k][n] ~= q_w4_i8[k][n] * w_scale[n]
 * @param[out] colsum_w N entries: sum over k of q_w4_i8[k][n], HexKL's
 *             dequant correction term for unsigned activations
 */
void htp_qs4cx_from_q4_0x4(const void *q4_0x4_repacked, uint32_t K, uint32_t N,
                           int8_t *q_w4_i8, float *w_scale, int32_t *colsum_w);

/**
 * @brief Rearranges an on-disk QS4CX weight into HexKL's registry input.
 *
 * Unlike htp_qs4cx_from_q4_0x4, this is *not* a requantization -- it is a
 * bit rearrangement plus one integer sum, and the int4 values it produces
 * are exactly the ones quant_qs4cx_f32 already chose. The two quantizers
 * agree bit for bit on the arithmetic: both take rmin/rmax over the
 * channel with 0 folded in, both use scale = 15/(rmax-rmin), both round
 * and clamp to [-8, 7], and both store 1/scale (compare
 * __fallback_quant_nxk_qs4cx_f32 in fallback_internal.cpp with
 * htp_qs4cx_from_q4_0x4's body). Only the representation differs:
 *
 *   QS4CX on disk  nibbles q+8, two per byte, one channel per row:
 *                  packed[n * ((K+1)/2) + k/2], low nibble for even k
 *   HexKL registry raw signed q in an int8 container, K-major:
 *                  q_w4_i8[k * N + n], plus a per-channel colsum
 *
 * So a model quantized straight from FP32 to QS4CX loses nothing on the
 * way to the DSP, where routing it through Q4_0 costs a second
 * quantization (measured at ~0.7 dB SNR, 22.8 vs 23.5, in this branch's
 * Stage 3 notes) and a full dequantize-to-f32 pass at load.
 *
 * @param[in]  qs4cx_packed N rows of ceil(K/2) nibble bytes, as
 *             quant_qs4cx_f32(N, K, ..., is_nxk=true) writes them and as
 *             QS4CX_Tensor::getData() returns them
 * @param[in]  qs4cx_scales N floats, QS4CX_Tensor::getScale()'s region
 * @param[in]  K input dimension (the QS4CX tensor's height)
 * @param[in]  N output dimension (the QS4CX tensor's width)
 * @param[out] q_w4_i8 K*N int4 values in [-8, 7] as int8, row-major
 * @param[out] w_scale N entries, copied through unchanged
 * @param[out] colsum_w N entries: sum over k of q_w4_i8[k][n]
 */
void htp_qs4cx_from_packed(const void *qs4cx_packed, const float *qs4cx_scales,
                           uint32_t K, uint32_t N, int8_t *q_w4_i8,
                           float *w_scale, int32_t *colsum_w);

} // namespace nntrainer

#endif // __cplusplus
#endif // __NNTRAINER_HTP_Q4_0_CONVERT_H__
