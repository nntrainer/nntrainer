// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file   hvx_attn_softmax_f16.h
 * @date   14 Sep 2026
 * @brief  HVX pieces of the online softmax over interleaved fp16 score tiles
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The scores come out of HMX in the tile layout hvx_tile_f16.h describes:
 * one vector holds two rows interleaved, even lanes row 2v and odd lanes
 * row 2v+1. Rather than de-interleave, every helper here works on that
 * "row pair" vector directly -- a max or a sum that only combines lanes of
 * equal parity is a per-row reduction, and a value splatted as a 32-bit
 * word (even half, odd half) is a per-row broadcast. The widening
 * fp16 -> qf32 multiply even splits a vector by parity for free (even lanes
 * to the low result, odd to the high), which is how the row sums leave
 * fp16 without a shuffle.
 *
 * Softmax runs base-2 on scores pre-multiplied by log2(e), as llama.cpp's
 * HTP flash attention does; exp2 is a straight port of its device-proven
 * fp16 polynomial.
 */

#ifndef __NNTRAINER_HVX_ATTN_SOFTMAX_F16_H__
#define __NNTRAINER_HVX_ATTN_SOFTMAX_F16_H__

#include <stdint.h>
#include <string.h>

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

/** @brief fp16 bit patterns used as constants. */
#define HVX_ATTN_HF_NEG_INF 0xFC00u
#define HVX_ATTN_HF_ONE 0x3C00u
#define HVX_ATTN_HF_ZERO 0x0000u

/** @brief Broadcasts (even, odd) fp16 bit patterns to every lane pair. */
static inline HVX_Vector hvx_attn_splat_pair_hf(uint32_t even, uint32_t odd) {
  return Q6_V_vsplat_R((int)((odd << 16) | (even & 0xFFFFu)));
}

/** @brief Word 0 of a vector: fp16 lanes 0 (low half) and 1 (high half). */
static inline uint32_t hvx_attn_word0(HVX_Vector v) {
  return (uint32_t)Q6_R_vextract_VR(v, 0);
}

/**
 * @brief Per-parity max over all 64 fp16 lanes.
 *
 * Rotating by 4, 8, 16, 32 and 64 bytes only ever pairs a lane with one of
 * the same parity (every step is a multiple of two fp16 lanes), so after
 * the five steps each even lane holds the max of all even lanes and each
 * odd lane the max of all odd lanes.
 */
static inline HVX_Vector hvx_attn_pairmax_hf(HVX_Vector v) {
  for (int rot = 4; rot <= 64; rot <<= 1) {
    v = Q6_Vhf_vmax_VhfVhf(v, Q6_V_vror_VR(v, rot));
  }
  return v;
}

/** @brief Sum over all 32 f32 lanes, result in every lane. */
static inline HVX_Vector hvx_attn_sum32_sf(HVX_Vector v) {
  for (int rot = 4; rot <= 64; rot <<= 1) {
    v = Q6_Vsf_vadd_VsfVsf(v, Q6_V_vror_VR(v, rot));
  }
  return v;
}

/**
 * @brief 2^x for 64 fp16 lanes, x <= 0 expected (the softmax argument).
 *
 * Ported from llama.cpp ggml-hexagon hvx-exp.h hvx_vec_exp2_f16. Input is
 * clamped at -24; the integer part goes into the exponent field and a
 * degree-6 polynomial in qf16 handles the fraction. Any result whose
 * exponent would underflow is forced to exactly zero -- which is what makes
 * a -inf score (a masked lane) come out as an exact 0 probability with no
 * separate fixup.
 */
static inline HVX_Vector hvx_attn_exp2_hf(HVX_Vector x) {
  const HVX_Vector zero = Q6_V_vzero();
  const HVX_Vector half = Q6_Vh_vsplat_R(0x3800);      // 0.5
  const HVX_Vector clamp_min = Q6_Vh_vsplat_R(0xCE00); // -24.0

  x = Q6_Vhf_vmax_VhfVhf(clamp_min, x);

  // k = round(x) via trunc(x - 0.5) on the (non-positive) input; f = x - k.
  const HVX_Vector x_minus_half =
    Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(x, half));
  const HVX_Vector k = Q6_Vh_equals_Vhf(x_minus_half);
  const HVX_Vector f = Q6_Vhf_equals_Vh(k);
  const HVX_Vector frac = Q6_Vqf16_vsub_VhfVhf(x, f);

  // Horner: y = (((((E5*f + E4)*f + E3)*f + E2)*f + E1)*f + E0)*f + 1
  //
  // E5 = ln2^6/720 enters as an fp16 constant (0x090c) through the mixed
  // qf16*hf multiply, NOT as llama.cpp's hand-encoded qf16 constant
  // (0x5082) through the qf16*qf16 one: hexagon-clang 19 re-types a
  // spilled splat constant as hf and selects the mixed instruction on its
  // own, which read 0x5082 as 36.06 and added ~36*f^6 to every
  // probability (found on-device: bit-exact scores, 20 dB outputs). With
  // the operand types stated explicitly there is nothing to re-type.
  HVX_Vector y = Q6_Vqf16_vmpy_Vqf16Vhf(frac, Q6_Vh_vsplat_R(0x090c));
  y = Q6_Vqf16_vadd_Vqf16Vhf(y, Q6_Vh_vsplat_R(0x157d));
  y = Q6_Vqf16_vmpy_Vqf16Vqf16(y, frac);
  y = Q6_Vqf16_vadd_Vqf16Vhf(y, Q6_Vh_vsplat_R(0x20ed));
  y = Q6_Vqf16_vmpy_Vqf16Vqf16(y, frac);
  y = Q6_Vqf16_vadd_Vqf16Vhf(y, Q6_Vh_vsplat_R(0x2b1b));
  y = Q6_Vqf16_vmpy_Vqf16Vqf16(y, frac);
  y = Q6_Vqf16_vadd_Vqf16Vhf(y, Q6_Vh_vsplat_R(0x33b0));
  y = Q6_Vqf16_vmpy_Vqf16Vqf16(y, frac);
  y = Q6_Vqf16_vadd_Vqf16Vhf(y, Q6_Vh_vsplat_R(0x398c));
  y = Q6_Vqf16_vmpy_Vqf16Vqf16(y, frac);
  y = Q6_Vqf16_vadd_Vqf16Vhf(y, Q6_Vh_vsplat_R(0x3c00));

  // Scale by 2^k: add k to the exponent field; underflow -> 0.
  y = Q6_Vhf_equals_Vqf16(y);
  HVX_Vector y_exp = Q6_Vuh_vlsr_VuhR(Q6_Vh_vasl_VhR(y, 1), 11);
  y_exp = Q6_Vh_vadd_VhVh(k, y_exp);
  const HVX_VectorPred underflow = Q6_Q_vcmp_gt_VhVh(zero, y_exp);
  y = Q6_Vh_vaslacc_VhVhR(y, k, 10);
  return Q6_V_vmux_QVV(underflow, zero, y);
}

/**
 * @brief Lane -> column index within a tile, as uint16: {0,0,1,1,...,31,31}.
 *
 * Built once per call on the stack and loaded; a per-lane constant HVX has
 * no single instruction for.
 */
static inline HVX_Vector hvx_attn_col_index_pairs(void) {
  uint16_t idx[64];
  for (uint32_t l = 0; l < 64u; ++l) {
    idx[l] = (uint16_t)(l >> 1);
  }
  // memcpy, not a punned vector load: see hvx_tile_load_u.
  HVX_Vector v;
  memcpy(&v, idx, sizeof(v));
  return v;
}

/**
 * @brief Visibility predicate for one score vector.
 *
 * True where column (col_base + lane/2) lies in the lane's row range
 * [lo, hi), with the two rows' bounds arriving as word splats (even half
 * for the even row). Unsigned 16-bit compares, so cache positions up to
 * 65535 are representable; the kernel rejects anything larger.
 */
static inline HVX_VectorPred hvx_attn_visible(HVX_Vector col_idx_pairs,
                                              uint32_t col_base,
                                              HVX_Vector lo_pair,
                                              HVX_Vector hi_pair) {
  const HVX_Vector col =
    Q6_Vh_vadd_VhVh(col_idx_pairs, Q6_Vh_vsplat_R((int)col_base));
  const HVX_VectorPred below_hi = Q6_Q_vcmp_gt_VuhVuh(hi_pair, col);
  const HVX_VectorPred below_lo = Q6_Q_vcmp_gt_VuhVuh(lo_pair, col);
  return Q6_Q_and_QQ(below_hi, Q6_Q_not_Q(below_lo));
}

/**
 * @brief 64 f32 (two vectors, natural order) -> one fp16 vector, natural
 *        order. Deals into even/odd words first so the narrowing
 *        conversion's even/odd lane placement lands each element back at
 *        its own index.
 */
static inline HVX_Vector hvx_attn_f32x64_to_hf(HVX_Vector lo32,
                                               HVX_Vector hi32) {
  const HVX_VectorPair d = Q6_W_vdeal_VVR(hi32, lo32, -4);
  const HVX_Vector one_sf = Q6_V_vsplat_R(0x3F800000);
  return Q6_Vhf_equals_Wqf32(
    Q6_W_vcombine_VV(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(d), one_sf),
                     Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(d), one_sf)));
}

/** @brief Max over all 64 fp16 lanes, result in every lane. */
static inline HVX_Vector hvx_attn_max64_hf(HVX_Vector v) {
  for (int rot = 2; rot <= 64; rot <<= 1) {
    v = Q6_Vhf_vmax_VhfVhf(v, Q6_V_vror_VR(v, rot));
  }
  return v;
}

/** @brief Scalar fp16 bits -> f32. */
static inline float hvx_attn_hf_to_f32(uint16_t bits) {
  _Float16 h;
  memcpy(&h, &bits, sizeof(h));
  return (float)h;
}

/** @brief Scalar f32 -> fp16 bits (round to nearest even). */
static inline uint16_t hvx_attn_f32_to_hf(float f) {
  const _Float16 h = (_Float16)f;
  uint16_t bits;
  memcpy(&bits, &h, sizeof(bits));
  return bits;
}

/** @brief Lane 0 of an f32 vector as a scalar. */
static inline float hvx_attn_lane0_f32(HVX_Vector v) {
  const uint32_t w = hvx_attn_word0(v);
  float f;
  memcpy(&f, &w, sizeof(f));
  return f;
}

/**
 * @brief 1/v for 32 f32 lanes, v in a sane positive range (here (1, 2]).
 *
 * llama.cpp's hvx_vec_inverse_f32: the classic magic-constant seed
 * followed by two Newton steps i = i*(2 - i*v). Plenty for a softcap
 * denominator; not a general-purpose divide (no inf/nan/zero handling).
 */
static inline HVX_Vector hvx_attn_recip_f32(HVX_Vector v) {
  const HVX_Vector two = Q6_V_vsplat_R(0x40000000); // 2.0f
  HVX_Vector i = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x7EEEEBB3), v);
  for (int n = 0; n < 2; ++n) {
    const HVX_Vector iv = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(i, v));
    const HVX_Vector corr = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(two, iv));
    i = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(i, corr));
  }
  return i;
}

/**
 * @brief Logit softcap on 64 fp16 lanes: cap * tanh(s / cap).
 *
 * MHACoreLayer's attn_logit_softcapping, applied to scores that already
 * carry a log2e factor, so the caller passes cap' = cap*log2e and
 * k = 2*log2e/cap' (the constant that turns s' into the exponent below).
 *
 *   tanh(|u|) = (1 - y) / (1 + y),  y = 2^(-2|u| log2e)
 *
 * keeps the exp2 argument non-positive (the range hvx_attn_exp2_hf is fit
 * for), never overflows (y in (0, 1]), and saturates to exactly 1 once y
 * underflows to 0. The one divide runs as an f32 reciprocal on the widened
 * pair, then narrows back; the sign is reapplied from the input.
 */
static inline HVX_Vector hvx_attn_softcap_hf(HVX_Vector s, HVX_Vector cap_hf,
                                             HVX_Vector k_hf) {
  const HVX_Vector one_hf = Q6_Vh_vsplat_R((int)HVX_ATTN_HF_ONE);
  const HVX_Vector one_f32 = Q6_V_vsplat_R(0x3F800000);
  const HVX_Vector sign = Q6_V_vand_VV(s, Q6_Vh_vsplat_R(0x8000));
  const HVX_Vector mag = Q6_V_vand_VV(s, Q6_Vh_vsplat_R(0x7FFF));

  // y = 2^(-|s'| * k)
  const HVX_Vector e = Q6_Vhf_vmpy_VhfVhf(mag, k_hf);
  const HVX_Vector y =
    hvx_attn_exp2_hf(Q6_V_vxor_VV(e, Q6_Vh_vsplat_R(0x8000)));
  const HVX_Vector num = Q6_Vhf_vsub_VhfVhf(one_hf, y);
  const HVX_Vector den = Q6_Vhf_vadd_VhfVhf(one_hf, y);

  // 1/den in f32 on the (even, odd) halves, back to the interleaved hf.
  const HVX_VectorPair dw = Q6_Wqf32_vmpy_VhfVhf(den, one_hf);
  const HVX_Vector r_lo =
    hvx_attn_recip_f32(Q6_Vsf_equals_Vqf32(Q6_V_lo_W(dw)));
  const HVX_Vector r_hi =
    hvx_attn_recip_f32(Q6_Vsf_equals_Vqf32(Q6_V_hi_W(dw)));
  const HVX_Vector recip = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(
    Q6_Vqf32_vmpy_VsfVsf(r_hi, one_f32), Q6_Vqf32_vmpy_VsfVsf(r_lo, one_f32)));

  const HVX_Vector t = Q6_Vhf_vmpy_VhfVhf(num, recip);
  return Q6_V_vor_VV(Q6_Vhf_vmpy_VhfVhf(t, cap_hf), sign);
}

#endif /* __NNTRAINER_HVX_ATTN_SOFTMAX_F16_H__ */
