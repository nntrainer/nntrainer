// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file   hvx_attn_math_f32.h
 * @date   28 Sep 2026
 * @brief  f32 exp2 / floor / softcap on 32 HVX lanes for the decode softmax
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The quantized decode kernel scores 32 cache rows per vector as f32 and
 * keeps its softmax in f32: fewer conversions than the fp16 route when the
 * scores are dequantized from int32 anyway, and every running quantity (m,
 * l, the rescale) is a vector with all lanes equal, so nothing is ever
 * extracted to a scalar. exp2 follows llama.cpp's hvx_vec_exp_f32: split
 * off the integer part, Taylor series on the fraction times ln 2, integer
 * add into the exponent field.
 */

#ifndef __NNTRAINER_HVX_ATTN_MATH_F32_H__
#define __NNTRAINER_HVX_ATTN_MATH_F32_H__

#include <stdint.h>

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include "hvx_attn_softmax_f16.h"
#include "hvx_convert.h"

/** @brief f32 bit patterns. */
#define HVX_ATTN_SF_NEG_INF 0xFF800000u
#define HVX_ATTN_SF_ONE 0x3F800000u

/** @brief floor(x) for |x| < 2^22: nearest-even, then back off by one where
 *         that rounded up. */
static inline HVX_Vector hvx_attn_floor_f32(HVX_Vector x, HVX_Vector *k_int) {
  HVX_Vector t = hvx_sf_to_w_rne(x);
  const HVX_Vector tf = Q6_Vsf_equals_Vw(t);
  const HVX_VectorPred up = Q6_Q_vcmp_gt_VsfVsf(tf, x);
  t = Q6_Vw_vsub_VwVw(t, Q6_V_vmux_QVV(up, Q6_V_vsplat_R(1), Q6_V_vzero()));
  *k_int = t;
  return Q6_Vsf_equals_Vw(t);
}

/**
 * @brief 2^x for 32 f32 lanes, x <= 0 expected (softmax arguments).
 *
 * x is clamped at -126 so the result stays a normal number (2^-126 at the
 * floor, not 0: callers that need an exact 0 for a masked lane mux it in
 * afterwards). With k = floor(x), f = x - k in [0, 1) and t = f ln 2,
 * 2^f = e^t = 1 + t + t^2 (1/2! + t/3! + ... + t^5/7!), then k goes into
 * the exponent field. Relative error a few 1e-7.
 */
static inline HVX_Vector hvx_attn_exp2_f32(HVX_Vector x) {
  const HVX_Vector lo = Q6_V_vsplat_R(0xC2FC0000); // -126.0f
  x = Q6_Vsf_vmax_VsfVsf(x, lo);
  HVX_Vector k;
  const HVX_Vector kf = hvx_attn_floor_f32(x, &k);
  const HVX_Vector f = Q6_Vsf_vsub_VsfVsf(x, kf);
  const HVX_Vector t = Q6_Vsf_vmpy_VsfVsf(f, Q6_V_vsplat_R(0x3F317218)); // ln2
  const HVX_Vector t2 = Q6_Vsf_vmpy_VsfVsf(t, t);
  // Horner on the tail: 1/7!, 1/6!, 1/5!, 1/4!, 1/3!, 1/2!
  HVX_Vector y = Q6_Vsf_vmpy_VsfVsf(t, Q6_V_vsplat_R(0x39506967));
  y = Q6_Vsf_vadd_VsfVsf(y, Q6_V_vsplat_R(0x3AB743CE));
  y = Q6_Vsf_vmpy_VsfVsf(y, t);
  y = Q6_Vsf_vadd_VsfVsf(y, Q6_V_vsplat_R(0x3C088908));
  y = Q6_Vsf_vmpy_VsfVsf(y, t);
  y = Q6_Vsf_vadd_VsfVsf(y, Q6_V_vsplat_R(0x3D2AA9C1));
  y = Q6_Vsf_vmpy_VsfVsf(y, t);
  y = Q6_Vsf_vadd_VsfVsf(y, Q6_V_vsplat_R(0x3E2AAAAA));
  y = Q6_Vsf_vmpy_VsfVsf(y, t);
  y = Q6_Vsf_vadd_VsfVsf(y, Q6_V_vsplat_R(0x3F000000));
  y = Q6_Vsf_vmpy_VsfVsf(y, t2);
  y = Q6_Vsf_vadd_VsfVsf(y, t);
  y = Q6_Vsf_vadd_VsfVsf(y, Q6_V_vsplat_R((int)HVX_ATTN_SF_ONE));
  // y in [1, 2): adding k to the exponent field is exact and stays normal
  // for k >= -126.
  return Q6_Vw_vadd_VwVw(y, Q6_Vw_vasl_VwR(k, 23));
}

/** @brief Sum over all 32 int32 lanes, result in every lane. */
static inline HVX_Vector hvx_attn_sum32_w(HVX_Vector v) {
  for (int rot = 4; rot <= 64; rot <<= 1) {
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, rot));
  }
  return v;
}

/**
 * @brief Logit softcap on 32 f32 lanes: cap * tanh(s / cap), for scores
 *        that carry log2e (cap' = cap*log2e, k = 2*log2e/cap').
 *
 *   tanh(|u|) = (1 - y) / (1 + y),  y = 2^(-2|u| log2e)
 *
 * as the fp16 version; the divide is the Newton reciprocal on (1, 2].
 */
static inline HVX_Vector hvx_attn_softcap_f32(HVX_Vector s, HVX_Vector cap_sf,
                                              HVX_Vector k_sf) {
  const HVX_Vector one = Q6_V_vsplat_R((int)HVX_ATTN_SF_ONE);
  const HVX_Vector sign = Q6_V_vand_VV(s, Q6_V_vsplat_R((int)0x80000000u));
  const HVX_Vector mag = Q6_V_vand_VV(s, Q6_V_vsplat_R(0x7FFFFFFF));
  const HVX_Vector e = Q6_Vsf_vmpy_VsfVsf(mag, k_sf);
  const HVX_Vector y =
    hvx_attn_exp2_f32(Q6_V_vxor_VV(e, Q6_V_vsplat_R((int)0x80000000u)));
  const HVX_Vector num = Q6_Vsf_vsub_VsfVsf(one, y);
  const HVX_Vector den = Q6_Vsf_vadd_VsfVsf(one, y);
  const HVX_Vector t = Q6_Vsf_vmpy_VsfVsf(num, hvx_attn_recip_f32(den));
  return Q6_V_vor_VV(Q6_Vsf_vmpy_VsfVsf(t, cap_sf), sign);
}

#endif /* __NNTRAINER_HVX_ATTN_MATH_F32_H__ */
