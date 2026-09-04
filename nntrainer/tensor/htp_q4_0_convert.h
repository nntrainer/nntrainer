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

} // namespace nntrainer

#endif // __cplusplus
#endif // __NNTRAINER_HTP_Q4_0_CONVERT_H__
