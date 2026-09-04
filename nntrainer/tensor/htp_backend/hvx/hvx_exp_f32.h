// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file hvx_exp_f32.h
 * @date 05 Aug 2026
 * @brief Vector exp over f32 lanes for HVX
 * @see https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug No known bugs except for NYI items */

#ifndef __NNTRAINER_HVX_EXP_F32_H__
#define __NNTRAINER_HVX_EXP_F32_H__

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include "hvx_convert.h"

/**
 * @brief exp(x) for every f32 lane.
 *
 * exp(x) = 2^k * exp(r) with k an integer and r near zero. 2^k is not
 * computed; k is added to the exponent field, one instruction. So the
 * polynomial only has to cover r.
 *
 * k comes from round-to-nearest, not floor. That halves r's range to
 * [-ln2/2, ln2/2], which cuts the residue of a given degree by 2^(degree+1).
 * It also means the reduction is hvx_sf_to_w_rne, which already exists * HVX
 * has no floor instruction, so floor would need its own bit-twiddling helper.
 *
 * The polynomial is 7th order to keep the worst-case relative error under
 * 1e-6 across the domain below. Both the range reduction (x - kf*ln2) and
 * the polynomial's Horner recurrence are computed in the qf32
 * extended-precision format rather than plain Vsf: on real HVX hardware,
 * chained Vsf multiply-adds lose precision that qf32 retains, so every
 * vmpy/vadd in both steps below routes through qf32 and only drops to Vsf
 * once, at the very end.
 *
 * Domain:
 * x >= -87.3 relative error <= 1e-6.
 * x <~ -87.7 returns 0. The true value is subnormal there and this
 * flushes it, which is what softmax wants: it feeds
 * x - max, and a term that small cannot move a sum of 1.
 * x > 88.7 UNDEFINED, and not an infinity there is no overflow
 * clamp. softmax feeds x - max <= 0, so a clamp would
 * cost instructions on an unreachable path.
 * NaN / Inf undefined.
 *
 * x == 0 returns exactly 1.0f. */
static inline HVX_Vector hvx_exp_sf(HVX_Vector x) {
  const HVX_Vector log2e = hvx_splat_sf(1.44269504f); /* 1/ln(2) */
  /** ln2_hi + ln2_lo == ln2, split so that k*ln2_hi is exact (see the
  range-reduction comment below for why one f32 constant is not enough
  and how this split fixes it). */
  const HVX_Vector ln2_hi = hvx_splat_sf(0.693359375f);
  const HVX_Vector ln2_lo = hvx_splat_sf(-2.12194440e-4f);
  const HVX_Vector zero = Q6_V_vzero;

  /** Clamp the bottom. Without it a -1e30 input pushes x/ln2 past the
  |v| <= 2^22 domain of hvx_sf_to_w_rne, k comes out as garbage, and the
  underflow guard below can be fooled into passing it through. -120
  underflows to zero anyway, so nothing representable is lost. */
  x = Q6_Vsf_vmax_VsfVsf(x, hvx_splat_sf(-120.0f));

  /** k = round(x / ln2), kf = (float)k.
  The two directions are asymmetric: Q6_Vw_equals_Vsf rounds toward
  zero, so f32 -> int32 goes through the magic-number identity in
  hvx_convert.h, while int32 -> f32 is a single numeric convert (the
  same one hvx_dequant_i32.c uses). */
  const HVX_Vector k = hvx_sf_to_w_rne(Q6_Vsf_vmpy_VsfVsf(x, log2e));
  const HVX_Vector kf = Q6_Vsf_equals_Vw(k);

  /** r = x - kf*ln2, in [-ln2/2, ln2/2]. Computed in qf32 (dropping to Vsf
  only once, at the end) because x and kf*ln2 are close in magnitude
  here, and a plain-Vsf multiply would round away precision before the
  subtraction's cancellation amplifies it.

  ln2 itself is split into ln2_hi + ln2_lo (the Cephes/fdlibm/SLEEF/
  glibc technique): a single f32 constant only carries ~24 mantissa
  bits, too few once k*ln2 is computed for |k| up to ~170 over this
  domain. ln2_hi's low mantissa bits are zero, so k*ln2_hi is exact in
  f32; ln2_lo is the tiny remainder that recovers the rest. */
  const HVX_Vector kf_ln2_hi_qf32 = Q6_Vqf32_vmpy_VsfVsf(kf, ln2_hi);
  const HVX_Vector x_qf32 = Q6_Vqf32_vadd_VsfVsf(x, zero);
  const HVX_Vector r_hi_qf32 = Q6_Vqf32_vsub_Vqf32Vqf32(x_qf32, kf_ln2_hi_qf32);
  const HVX_Vector kf_ln2_lo_qf32 = Q6_Vqf32_vmpy_VsfVsf(kf, ln2_lo);
  const HVX_Vector r =
    Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(r_hi_qf32, kf_ln2_lo_qf32));

  /** p = 1 + r + r^2 * (1/2! + r*(1/3! + r*(1/4! + r*(1/5! + r*(1/6! +
  * r/7!)))))
  Coefficients are written as 1/n! rather than hex bit patterns so the
  value's origin stays in the source; the compiler folds them.

  This Horner recurrence runs in qf32, for the same reason the
  reduction above does: chained plain-Vsf multiply/adds don't hold
  full precision through real HVX hardware, so every vmpy and vadd
  here is qf32. Q6_Vqf32_vadd_Vqf32Vsf's own output is already a valid
  operand to Q6_Vqf32_vmpy_Vqf32Vqf32, so no extra renormalizing add is
  needed between steps. r itself is lifted to qf32 once, up front, and
  reused. p only comes back to Vsf at the very end, right before the
  exponent injection below, which still operates on plain Vsf. */
  const HVX_Vector r_qf32 = Q6_Vqf32_vadd_VsfVsf(r, zero);

  HVX_Vector p_qf32 = Q6_Vqf32_vmpy_VsfVsf(hvx_splat_sf(1.0f / 5040.0f), r);
  p_qf32 = Q6_Vqf32_vadd_Vqf32Vsf(p_qf32, hvx_splat_sf(1.0f / 720.0f));

  p_qf32 = Q6_Vqf32_vmpy_Vqf32Vqf32(p_qf32, r_qf32);
  p_qf32 = Q6_Vqf32_vadd_Vqf32Vsf(p_qf32, hvx_splat_sf(1.0f / 120.0f));

  p_qf32 = Q6_Vqf32_vmpy_Vqf32Vqf32(p_qf32, r_qf32);
  p_qf32 = Q6_Vqf32_vadd_Vqf32Vsf(p_qf32, hvx_splat_sf(1.0f / 24.0f));

  p_qf32 = Q6_Vqf32_vmpy_Vqf32Vqf32(p_qf32, r_qf32);
  p_qf32 = Q6_Vqf32_vadd_Vqf32Vsf(p_qf32, hvx_splat_sf(1.0f / 6.0f));

  p_qf32 = Q6_Vqf32_vmpy_Vqf32Vqf32(p_qf32, r_qf32);
  p_qf32 = Q6_Vqf32_vadd_Vqf32Vsf(p_qf32, hvx_splat_sf(0.5f));

  p_qf32 = Q6_Vqf32_vmpy_Vqf32Vqf32(p_qf32, r_qf32);
  p_qf32 = Q6_Vqf32_vmpy_Vqf32Vqf32(p_qf32, r_qf32);

  const HVX_Vector r_plus_1_qf32 = Q6_Vqf32_vadd_VsfVsf(r, hvx_splat_sf(1.0f));
  p_qf32 = Q6_Vqf32_vadd_Vqf32Vqf32(p_qf32, r_plus_1_qf32);

  const HVX_Vector p = Q6_Vsf_equals_Vqf32(p_qf32);

  /** p is in [0.707, 1.414], so its exponent field is 126 or 127. Adding
  k there multiplies by 2^k. */
  const HVX_Vector y = Q6_Vw_vaslacc_VwVwR(p, k, 23);

  /** If k plus that exponent lands at zero or below, the true result is
  subnormal or smaller and the bit add produced garbage, not a small
  number. Shift off the sign bit, pull the 8 exponent bits down. */
  const HVX_Vector p_exp = Q6_Vuw_vlsr_VuwR(Q6_Vw_vasl_VwR(p, 1), 24);
  const HVX_VectorPred underflow =
    Q6_Q_vcmp_gt_VwVw(zero, Q6_Vw_vadd_VwVw(k, p_exp));

  return Q6_V_vmux_QVV(underflow, zero, y);
}

#endif /* __NNTRAINER_HVX_EXP_F32_H__ */
