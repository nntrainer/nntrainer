// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hvx_convert.h
 * @date   03 Aug 2026
 * @brief  int32 <-> f32 vector conversion for HVX, which has no such
 *         instruction, and fp16-grid rounding, fused multiply-add and
 *         divide kept in f32 (attn_m1_det.h's rne16 / fma16 / divide)
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __NNTRAINER_HVX_CONVERT_H__
#define __NNTRAINER_HVX_CONVERT_H__

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

/**
 * @brief Bit pattern of 1.5 * 2^23 (12582912.0f).
 *
 * HVX has no numeric conversion between int32 and f32. Q6_Vw_equals_Vsf
 * and Q6_Vsf_equals_Vw are bit reinterpretations, and the vcvt family
 * only covers hf<->h/uh/b/ub and sf<->hf/bf. The magic-number identity
 * below is the way across.
 *
 * Adding an integer x to the bit pattern of 1.5 * 2^23 lands x in the low
 * mantissa bits, because that float's exponent puts the mantissa's least
 * significant bit at value 1. Subtracting the same constant as a float
 * then leaves exactly (float)x.
 *
 * Valid for |x| <= 2^22 = 4194304. Our accumulators are bounded by
 * 255 * 8 * K, which is 2088960 at K=1024, so the margin is about 2x.
 * Nothing here is safe for larger K without revisiting this bound.
 */
#define HVX_CONVERT_MAGIC_BITS 0x4B400000u

/**
 * @brief Converts int32 lanes to f32 lanes. Exact for |x| <= 2^22.
 *
 * HVX_Vector is type-agnostic: passing it directly to an f32 intrinsic
 * reinterprets the bits without conversion. Q6_Vsf_equals_Vw /
 * Q6_Vw_equals_Vsf look like reinterpret but are in fact NUMERIC
 * conversions (V79 HVX Programmer's Reference Manual: "Convert IEEE
 * floating point to integer... round to zero"), which would destroy the
 * magic-number identity.
 */
static inline HVX_Vector hvx_w_to_sf(HVX_Vector v) {
  const HVX_Vector magic = Q6_V_vsplat_R(HVX_CONVERT_MAGIC_BITS);
  const HVX_Vector bits = Q6_Vw_vadd_VwVw(v, magic);
  return Q6_Vsf_vsub_VsfVsf(bits, magic);
}

/**
 * @brief Converts f32 lanes to int32 lanes, rounding to nearest even.
 *
 * The rounding comes from the float add, so it is round-to-nearest-even.
 * Host references that compare against this must use nearbyint, not round.
 * Valid for results in |x| <= 2^22.
 *
 * Same reinterpret note as hvx_w_to_sf: the cross-type handoff is implicit.
 */
static inline HVX_Vector hvx_sf_to_w_rne(HVX_Vector v) {
  const HVX_Vector magic = Q6_V_vsplat_R(HVX_CONVERT_MAGIC_BITS);
  const HVX_Vector biased = Q6_Vsf_vadd_VsfVsf(v, magic);
  return Q6_Vw_vsub_VwVw(biased, magic);
}

/**
 * @brief Broadcasts a float to every lane.
 *
 * Q6_V_vsplat_R takes a word, so the float's bits have to get there
 * somehow. memcpy is the way that does not punt on strict aliasing --
 * casting through int* is what -Wstrict-aliasing objects to, and this
 * build uses -Wall -Werror. The compiler folds the memcpy away.
 */
static inline HVX_Vector hvx_splat_sf(float f) {
  int bits;
  __builtin_memcpy(&bits, &f, sizeof(bits));
  return Q6_V_vsplat_R(bits);
}

/**
 * @brief Rounds f32 lanes to the fp16 grid, ties to even, fp16 subnormals
 *        kept, in f32: attn_m1_det_rne16 per lane, the same operations.
 *
 * (x + C) - C with C = 1.5 * 2^(max(E, -14) + 13), E = x's exponent: C's
 * f32 ulp is 2^(E - 10), the fp16 ulp at E (2^-24 below fp16's smallest
 * normal), and x + C stays in C's binade, so the one f32 add rounds x to
 * that grid and the subtraction is exact. x's sign is ORed back so a value
 * that rounds to zero keeps it. The exponent fields: 0x38800000 is 2^-14's,
 * 0x06C00000 adds 13 to the exponent and the 0.5 mantissa bit. No
 * saturation: past 65520 the result is 65536, outside the callers' domain
 * (attn_m1_det.h). Plain Vsf add / sub and word ops only (rule 24).
 */
static inline HVX_Vector hvx_rne16_sf(HVX_Vector x) {
  HVX_Vector ef = Q6_V_vand_VV(x, Q6_V_vsplat_R(0x7F800000));
  ef = Q6_Vw_vmax_VwVw(ef, Q6_V_vsplat_R(0x38800000));
  const HVX_Vector c = Q6_Vw_vadd_VwVw(ef, Q6_V_vsplat_R(0x06C00000));
  const HVX_Vector r = Q6_Vsf_vsub_VsfVsf(Q6_Vsf_vadd_VsfVsf(x, c), c);
  return Q6_V_vor_VV(r, Q6_V_vand_VV(x, Q6_V_vsplat_R((int)0x80000000u)));
}

/**
 * @brief attn_m1_det_fma16 per lane: RN16(c + a * b) rounded once, for
 *        fp16 values. p = a * b is exact; s = c + p and TwoSum's err; where
 *        err != 0, s steps one ulp toward zero if err's sign differs and
 *        its last bit is set (round to odd); then rne16.
 */
static inline HVX_Vector hvx_fma16_sf(HVX_Vector c, HVX_Vector a,
                                      HVX_Vector b) {
  const HVX_Vector p = Q6_Vsf_vmpy_VsfVsf(a, b);
  const HVX_Vector s = Q6_Vsf_vadd_VsfVsf(c, p);
  const HVX_Vector bv = Q6_Vsf_vsub_VsfVsf(s, c);
  const HVX_Vector err =
    Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vsub_VsfVsf(c, Q6_Vsf_vsub_VsfVsf(s, bv)),
                       Q6_Vsf_vsub_VsfVsf(p, bv));
  /* (err ^ s) >> 31 is -1 where the signs differ: s - 1 toward zero. */
  HVX_Vector t = Q6_Vw_vadd_VwVw(s, Q6_Vw_vasr_VwR(Q6_V_vxor_VV(err, s), 31));
  t = Q6_V_vor_VV(t, Q6_V_vsplat_R(1));
  const HVX_VectorPred exact = Q6_Q_vcmp_eq_VwVw(err, Q6_V_vzero());
  return hvx_rne16_sf(Q6_V_vmux_QVV(exact, s, t));
}

/**
 * @brief rne16(e / l) per lane without a divider, for fp16 values e >= 0
 *        and l >= 1 (lv splat), r = hvx_recip_det_sf(l) or any f32 value
 *        within a few ulps of 1/l.
 *
 * c0 = rne16(e * r) is the correctly rounded quotient or a neighbour: r is
 * within a few f32 ulps of 1/l, far inside half an fp16 ulp. The two
 * midpoints next to c0 have at most 12 significant bits and l has 11, so
 * mid * l is exact in f32 and comparing it with e decides the rounding
 * exactly: above the upper midpoint -> the next value up, below the lower
 * one -> the next down. A tie (e == mid * l) needs an odd 12-bit midpoint
 * times l to fit fp16's 11 bits, which only a midpoint of the subnormal
 * grid (2^-24 steps, an odd factor as small as 1) allows; there c0's
 * parity is whether (c0 + 1.5) - 1.5 == c0, and the tie goes to the even
 * side. Above c0 the half step is hu = 2^(max(E, -14) - 11); below it is
 * hu / 2 when c0 is a power of two above 2^-14 (the grid halves there).
 * attn_m1_host_check.c runs every quotient of the attention domain (l in
 * [1, 2048] -- an fp16 sum of values <= 1 cannot pass 2048 -- and e in
 * [0, l]) whose c0 is off or which is a tie: 7881 and 27049 of them. The
 * power-of-two branch is never taken there (no c0 = 2^E is one too high);
 * it stays so the function is right for any fp16 e and l.
 */
static inline HVX_Vector hvx_div16_sf(HVX_Vector e, HVX_Vector lv,
                                      HVX_Vector r) {
  const HVX_Vector c0 = hvx_rne16_sf(Q6_Vsf_vmpy_VsfVsf(e, r));
  const HVX_Vector ef = Q6_V_vand_VV(c0, Q6_V_vsplat_R(0x7F800000));
  const HVX_Vector ef_min = Q6_V_vsplat_R(0x38800000);
  const HVX_Vector hu =
    Q6_Vw_vsub_VwVw(Q6_Vw_vmax_VwVw(ef, ef_min), Q6_V_vsplat_R(11 << 23));
  /* A power of two above 2^-14: c0 >= 0 has no mantissa bits (c0 == ef). */
  const HVX_VectorPred pow2 =
    Q6_Q_and_QQ(Q6_Q_vcmp_gt_VwVw(ef, ef_min), Q6_Q_vcmp_eq_VwVw(ef, c0));
  const HVX_Vector hd =
    Q6_V_vmux_QVV(pow2, Q6_Vw_vsub_VwVw(hu, Q6_V_vsplat_R(1 << 23)), hu);
  const HVX_Vector mid_hi = Q6_Vsf_vadd_VsfVsf(c0, hu);
  const HVX_Vector mid_lo = Q6_Vsf_vsub_VsfVsf(c0, hd);
  const HVX_Vector c_up = Q6_Vsf_vadd_VsfVsf(mid_hi, hu);
  const HVX_Vector c_dn = Q6_Vsf_vsub_VsfVsf(mid_lo, hd);
  const HVX_Vector p_hi = Q6_Vsf_vmpy_VsfVsf(mid_hi, lv);
  const HVX_Vector p_lo = Q6_Vsf_vmpy_VsfVsf(mid_lo, lv);
  const HVX_Vector h15 = hvx_splat_sf(1.5f);
  const HVX_VectorPred even =
    Q6_Q_vcmp_eq_VwVw(Q6_Vsf_vsub_VsfVsf(Q6_Vsf_vadd_VsfVsf(c0, h15), h15), c0);
  /* e >= 0 and p_hi > 0, so the word compares are the float compares; a
     negative p_lo (c0 == 0) reads as a negative word, below every e. */
  const HVX_VectorPred up = Q6_Q_or_QQ(
    Q6_Q_vcmp_gt_VwVw(e, p_hi), Q6_Q_and_QQn(Q6_Q_vcmp_eq_VwVw(e, p_hi), even));
  const HVX_VectorPred dn = Q6_Q_or_QQ(
    Q6_Q_vcmp_gt_VwVw(p_lo, e), Q6_Q_and_QQn(Q6_Q_vcmp_eq_VwVw(e, p_lo), even));
  return Q6_V_vmux_QVV(up, c_up, Q6_V_vmux_QVV(dn, c_dn, c0));
}

#endif /* __NNTRAINER_HVX_CONVERT_H__ */
