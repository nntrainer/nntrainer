// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hvx_swiglu_det.h
 * @date   10 Sep 2026
 * @brief  SwiGLU specified tightly enough to reproduce bit-for-bit off HVX
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * WHY THIS EXISTS
 *
 * The fused MoE FFN path computes SwiGLU on the DSP and requantizes the
 * result to uint8 for the down matmul; the working path computes it on ARM
 * and lets the down matmul's own quantizer do the same thing. Both are
 * accurate to spec and they still disagree, because they are different
 * approximations: a value that lands within half a quantization step of a
 * bin boundary can round to different levels on the two sides. Measured on
 * device (doc 43 section 7): one element of 1792 flips on 5 of 32 expert
 * calls, and the down matmul spreads that one element across all 2048
 * output columns. Twenty-two layers of that is a different token stream.
 *
 * No amount of extra accuracy fixes it -- two independently-rounded
 * pipelines will always disagree somewhere. The fix is for both sides to
 * compute the SAME sequence of IEEE-754 f32 operations, so they produce the
 * same bits and the flip count is exactly zero.
 *
 * THE SPECIFICATION
 *
 * Every operation below is a plain f32 multiply, add, subtract, min or max,
 * each rounding to f32 on its own. No fused multiply-add (it rounds once
 * where this rounds twice), no qf32 (it carries bits f32 does not), no
 * divide (HVX has none, and a reciprocal-multiply does not round like one).
 * The constants and the ORDER of operations are as much part of the
 * contract as the formula: reassociating a Horner step changes the result.
 *
 *   exp_det(x):
 *     x  = max(min(x, DET_EXP_MAX), DET_EXP_MIN)
 *     kf = x * LOG2E
 *     k  = round_to_nearest_even_int(kf)
 *     kl = (float)k
 *     r  = (x - kl*LN2_HI) - kl*LN2_LO
 *     p  = C7
 *     p  = p*r + C6   ... down to ...   p = p*r + C0
 *     y  = bitcast_f32(bitcast_i32(p) + (k << 23))
 *     y  = 0 when k + exponent_field(p) <= 0
 *
 *   recip_det(d):                     (d >= 1 here, so the seed is in range)
 *     y = bitcast_f32(0x7EF311C2 - bitcast_i32(d))
 *     y = y * (2 - d*y)               three times
 *
 *   swiglu_det(g, u) = (g * recip_det(1 + exp_det(-g))) * u
 *
 * ACCURACY, for the record -- this is not a precision compromise:
 *   |r| <= ln2/2, so the degree-7 truncation is r^8/8! <= 5.2e-9 relative,
 *   well under f32's 1.2e-7 epsilon. Three Newton-Raphson steps take the
 *   magic seed's ~6% to ~1.7e-10. Both are below the last bit.
 *
 * DET_EXP_MAX is 85, not 88.7 where hvx_exp_sf gives out. recip_det's magic
 * seed is a word subtract on the raw bit pattern and goes negative once
 * bits(d) > 0x7EF311C2, i.e. d > 1.615e38 -- and Newton-Raphson on a
 * negative seed diverges to NaN rather than to a merely wrong number. At
 * x = 85 the reciprocal's relative error is 1.55e-8 with 36.5M ULPs of
 * headroom to that edge (host-derived, doc 43 section 7). What the clamp
 * discards is nothing: the largest silu term it can suppress is
 * 85*exp(-85) = 1.03e-35 per unit of up.
 *
 * THE ASSUMPTION THIS RESTS ON, AND HOW IT IS CHECKED
 *
 * That HVX's Vsf multiply and add round exactly the way ARM's f32 multiply
 * and add do. hvx_exp_f32.h's own comment says chained Vsf multiply-adds
 * "lose precision that qf32 retains", which is either the ordinary
 * observation that f32 rounds at every step where an extended format does
 * not -- in which case this works -- or a statement that Vsf is not
 * IEEE-correctly-rounded, in which case bit identity is impossible here and
 * this whole approach is dead.
 *
 * That is not something to assume. nntr_hvx_swiglu_det_f32 exposes this
 * function's output AND its two intermediates, and
 * unittest_hvx_softmax.cpp's HvxSwigluDet.MatchesScalarBitExact compares all
 * three against a scalar reference built from the same spec, bit for bit.
 * If it fails, its per-stage counts say whether exp_det, recip_det or the
 * final multiply is where the two arithmetics part company.
 */

#ifndef __NNTRAINER_HVX_SWIGLU_DET_H__
#define __NNTRAINER_HVX_SWIGLU_DET_H__

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include "hvx_convert.h"

/** @brief Upper clamp on exp_det's argument -- see the header comment for
 *         why 85 and not hvx_exp_sf's 88.7. */
#define DET_EXP_MAX 85.0f
/** @brief Lower clamp. exp(-88) is already below f32's smallest normal, so
 *         nothing representable is lost and k stays inside the domain of
 *         the float-to-int conversion. */
#define DET_EXP_MIN (-88.0f)

/**
 * @brief exp(x) for every f32 lane, in plain Vsf only.
 *
 * The qf32 sibling hvx_exp_sf is more accurate and stays where it is; this
 * one exists to be reproducible on a different machine, which qf32 cannot
 * be. See the header comment for the operation-by-operation contract.
 */
static inline HVX_Vector hvx_exp_det_sf(HVX_Vector x) {
  const HVX_Vector zero = Q6_V_vzero();
  x = Q6_Vsf_vmin_VsfVsf(x, hvx_splat_sf(DET_EXP_MAX));
  x = Q6_Vsf_vmax_VsfVsf(x, hvx_splat_sf(DET_EXP_MIN));

  const HVX_Vector k = hvx_sf_to_w_rne(
    Q6_Vsf_vmpy_VsfVsf(x, hvx_splat_sf(1.44269504f))); /* x/ln2, rounded */
  const HVX_Vector kl = Q6_Vsf_equals_Vw(k);

  /** ln2 split so that kl*LN2_HI is exact in f32 (LN2_HI's low mantissa
     bits are zero) and LN2_LO carries the remainder -- the Cephes/fdlibm
     technique. One f32 constant cannot hold ln2 to enough bits once it is
     multiplied by a k as large as 122. */
  HVX_Vector r =
    Q6_Vsf_vsub_VsfVsf(x, Q6_Vsf_vmpy_VsfVsf(kl, hvx_splat_sf(0.693359375f)));
  r = Q6_Vsf_vsub_VsfVsf(r,
                         Q6_Vsf_vmpy_VsfVsf(kl, hvx_splat_sf(-2.12194440e-4f)));

  /** Horner on sum r^n/n!, n = 0..7. Written as 1/n! rather than hex bit
     patterns so the value's origin stays in the source; the compiler folds
     them. Each step is a separate multiply and a separate add on purpose --
     see the header comment on fused multiply-add. */
  HVX_Vector p = hvx_splat_sf(1.0f / 5040.0f);
  p = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(p, r), hvx_splat_sf(1.0f / 720.0f));
  p = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(p, r), hvx_splat_sf(1.0f / 120.0f));
  p = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(p, r), hvx_splat_sf(1.0f / 24.0f));
  p = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(p, r), hvx_splat_sf(1.0f / 6.0f));
  p = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(p, r), hvx_splat_sf(0.5f));
  p = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(p, r), hvx_splat_sf(1.0f));
  p = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(p, r), hvx_splat_sf(1.0f));

  /** p is in [0.707, 1.414], so its exponent field is 126 or 127 and adding
     k there multiplies by 2^k -- one instruction, no 2^k constant. */
  const HVX_Vector y = Q6_Vw_vaslacc_VwVwR(p, k, 23);

  /** If k plus that exponent lands at or below zero the true result is
     subnormal or smaller, and the bit add produced garbage rather than a
     small number. Shift off the sign bit, pull the 8 exponent bits down. */
  const HVX_Vector p_exp = Q6_Vuw_vlsr_VuwR(Q6_Vw_vasl_VwR(p, 1), 24);
  const HVX_VectorPred underflow =
    Q6_Q_vcmp_gt_VwVw(zero, Q6_Vw_vadd_VwVw(k, p_exp));
  return Q6_V_vmux_QVV(underflow, zero, y);
}

/**
 * @brief 1/d for every f32 lane, in plain Vsf only.
 *
 * The magic seed is the reciprocal counterpart of the fast inverse square
 * root constant: it lands within ~6% for any normal input whose bit pattern
 * is at or below it, and each Newton-Raphson step y <- y*(2 - d*y) doubles
 * the correct bits.
 *
 * DOMAIN: bits(d) <= 0x7EF311C2, i.e. d <= 1.6154730e38. Past that the
 * subtraction goes negative, the seed reinterprets as a negative float, and
 * three steps overflow to NaN -- not to a wrong-but-finite reciprocal.
 * hvx_swiglu_det_sf's caller-side clamp is what keeps d inside it.
 */
static inline HVX_Vector hvx_recip_det_sf(HVX_Vector d) {
  const HVX_Vector two = hvx_splat_sf(2.0f);
  HVX_Vector y = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x7EF311C2u), d);
  for (int it = 0; it < 3; ++it) {
    y =
      Q6_Vsf_vmpy_VsfVsf(y, Q6_Vsf_vsub_VsfVsf(two, Q6_Vsf_vmpy_VsfVsf(d, y)));
  }
  return y;
}

/** @brief silu(g)*u = (g * (1/(1 + exp(-g)))) * u, for every f32 lane. */
static inline HVX_Vector hvx_swiglu_det_sf(HVX_Vector g, HVX_Vector u) {
  const HVX_Vector e = hvx_exp_det_sf(Q6_Vsf_vsub_VsfVsf(Q6_V_vzero(), g));
  const HVX_Vector d = Q6_Vsf_vadd_VsfVsf(hvx_splat_sf(1.0f), e);
  const HVX_Vector s = hvx_recip_det_sf(d);
  return Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vmpy_VsfVsf(g, s), u);
}

#endif /* __NNTRAINER_HVX_SWIGLU_DET_H__ */
