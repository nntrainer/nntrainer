// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hvx_attn_m1_hf.h
 * @date   29 Sep 2026
 * @brief  fp16-lane (hf) primitives of the m=1 decode attention, each equal
 *         to one step of attn_m1_det.h: the one-rounding FMA built from
 *         qf32 ops, the score tree, exp16 and the divide
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * Plan 170 section 3. One header for the kernel, the probe entry
 * (test/htp/nntr_hvx_attn_m1_probe.c) and the host check, so the probe on
 * silicon tests the code the kernel ships. An hf vector is 64 fp16 lanes;
 * a widened pair holds the even lanes in lo and the odd lanes in hi.
 *
 * WHY EXPLICIT qf32. hexagon-clang 19 -mv79 lowers every IEEE sf / hf
 * intrinsic to a qf32 / qf16 op plus a convert and, without
 * -mhvx-ieee-fp (the skel does not pass it), contracts an sf multiply into
 * a following sf add with no rounding between them (plan 170 section 0).
 * So every f32 rounding point of the spec below is written as the qf32 op
 * and its own Vsf_equals_Vqf32, and the compiler has nothing to fuse. The
 * hf add / sub / mpy / max intrinsics are single fp16 operations already.
 *
 *   hvx_hf_fma      RN16(c + a*b) rounded once (attn_m1_det_fma16, the
 *                   CPU's fmla .8h): the exact products of a*b and c*1.0
 *                   in qf32, one qf32 add, one conversion to hf. The v79
 *                   ISS matched the fused spec on every real and
 *                   adversarial case, where the same sum narrowed through
 *                   sf failed every double-rounding hazard; silicon is
 *                   HvxAttnM1Probe.Semantics' question (G1).
 *   hvx_hf_score    step 2's tail: the vpaddq tree of the eight
 *                   accumulators, 0 + t, * scale, each one hf op.
 *   hvx_hf_exp16    exp16 on hf lanes d <= 0: neon_mathfun's exp_ps, one
 *                   explicit rounding per f32 step, fx = x * LOG2EF + 0.5
 *                   in two roundings (equal to the fused one at every fp16
 *                   d <= 0: the host check's exhaustive exp16 row).
 *   hvx_hf_div16    rne16(e / l): the lanes widened to sf, hvx_div16_sf
 *                   (the kernel's proven divide), narrowed back.
 *                   ponytail: an hf-native divide (the midpoint test on
 *                   qf32 products) would drop the widen / narrow; worth it
 *                   only if the phase words show the divide matters.
 *   hvx_hf_round_row / hvx_hf_store_sf  (#170 round 2) the kernel's q / k
 *                   / v rounding and its output widening as vectors:
 *                   hvx_hf_bits_rne and hf_float per lane in integer ops
 *                   plus exact f32 steps, so the sign of a zero and the
 *                   subnormals come through (no qf32 conversion on the
 *                   result; qf32 has no -0).
 *   hvx_hf_round_row_lut / hvx_hf_splat_lut  (#170 round 3) P1's q
 *                   operand without 64 stored splat vectors per head: the
 *                   rounded row stored zipped (vshuffe for vpacke), q[d]
 *                   and q[d + 1] splatted from it by one vlut16 -- a
 *                   permute, no memory, no scalar -> vector transfer.
 *   hvx_hf_exp16_fill / hvx_hf_exp16_idx  (#170 round 3) exp16 as a lookup:
 *                   attn_m1_det_exp_table's values as fp16 bits, and the
 *                   spec's attn_m1_det_exp_index as three integer ops on
 *                   the hf lanes. The kernel gathers tab[idx(d)] with
 *                   scalar loads; the host check proves tab[idx(d)] is
 *                   exp16(d) bit for bit at every fp16 d <= 0.
 *
 * Widening and narrowing values already on the fp16 grid are exact both
 * ways. The host emulation (test/htp/host/hvx_emu) checks all of this
 * against the spec (ATTN M1 HF PRIM); what it cannot see is a silicon
 * qf32 that narrows differently, which is why S1 runs first.
 */

#ifndef __NNTRAINER_HVX_ATTN_M1_HF_H__
#define __NNTRAINER_HVX_ATTN_M1_HF_H__

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include "attn_m1_det.h"
#include "hvx_convert.h"

/** @brief fp16 bits of 1.0, the qf32 widening multiplier. */
#define HVX_HF_ONE 0x3C00

/**
 * @brief fp16 bits of rne16(x) for any finite f32 x with |x| < 65520 (the
 *        spec's domain), in integer ops: the kernel's q / k / v rounding,
 *        one scalar step per value instead of attn_m1_det_rne16's volatile
 *        f32 ops. Round to nearest even on the dropped bits; below 2^-14
 *        onto the subnormal grid (x = m * 2^(e - 150), grid 2^-24), so an f32
 *        subnormal or anything under 2^-25 gives a signed zero, as rne16's
 *        sign restore does. attn_m1_host_check.c compares it with
 *        rne16 for every f32 below 65520 (ATTN M1 HF PRIM).
 */
static inline uint16_t hvx_hf_bits_rne(float x) {
  uint32_t u;
  __builtin_memcpy(&u, &x, sizeof(u));
  const uint32_t s = (u >> 16) & 0x8000u, a = u & 0x7FFFFFFFu;
  uint32_t r, rem, half;
  if (a >= 0x38800000u) { /* normal fp16: drop 13 bits */
    r = (a >> 13) - (112u << 10);
    rem = a & 0x1FFFu;
    half = 0x1000u;
  } else if (a >= 0x33000000u) { /* [2^-25, 2^-14): shift 14..24 */
    const uint32_t sh = 126u - (a >> 23);
    const uint32_t m = (a & 0x7FFFFFu) | 0x800000u;
    r = m >> sh;
    rem = m & ((1u << sh) - 1u);
    half = 1u << (sh - 1u);
  } else {
    return (uint16_t)s;
  }
  r += (rem > half || (rem == half && (r & 1u))) ? 1u : 0u;
  return (uint16_t)(s | r);
}

/** @brief RN16(c + a*b), rounded once; @a one = Q6_Vh_vsplat_R(HVX_HF_ONE). */
static inline HVX_Vector hvx_hf_fma(HVX_Vector c, HVX_Vector a, HVX_Vector b,
                                    HVX_Vector one) {
  const HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(a, b);
  const HVX_VectorPair cq = Q6_Wqf32_vmpy_VhfVhf(c, one);
  return Q6_Vhf_equals_Wqf32(
    Q6_W_vcombine_VV(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_hi_W(p), Q6_V_hi_W(cq)),
                     Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p), Q6_V_lo_W(cq))));
}

/** @brief The spec's score from its eight accumulators: (((a0 + a1) +
 *         (a2 + a3)) + ((a4 + a5) + (a6 + a7))), 0 + t, * scale (an fp16
 *         value, splat). */
static inline HVX_Vector hvx_hf_score(const HVX_Vector acc[ATTN_M1_DET_ACC],
                                      HVX_Vector scale) {
  const HVX_Vector s03 = Q6_Vhf_vadd_VhfVhf(Q6_Vhf_vadd_VhfVhf(acc[0], acc[1]),
                                            Q6_Vhf_vadd_VhfVhf(acc[2], acc[3]));
  const HVX_Vector s47 = Q6_Vhf_vadd_VhfVhf(Q6_Vhf_vadd_VhfVhf(acc[4], acc[5]),
                                            Q6_Vhf_vadd_VhfVhf(acc[6], acc[7]));
  const HVX_Vector t =
    Q6_Vhf_vadd_VhfVhf(Q6_V_vzero(), Q6_Vhf_vadd_VhfVhf(s03, s47));
  return Q6_Vhf_vmpy_VhfVhf(t, scale);
}

/** @brief hf lanes to sf, exact: lo = the even lanes, hi = the odd. */
static inline HVX_VectorPair hvx_hf_widen(HVX_Vector h, HVX_Vector one) {
  const HVX_VectorPair q = Q6_Wqf32_vmpy_VhfVhf(h, one);
  return Q6_W_vcombine_VV(Q6_Vsf_equals_Vqf32(Q6_V_hi_W(q)),
                          Q6_Vsf_equals_Vqf32(Q6_V_lo_W(q)));
}

/** @brief sf lanes already on the fp16 grid back to hf, exact: each half
 *         to qf32 as x * 1.0 (v79 has no plain sf -> qf32 conversion;
 *         Q6_Vqf32_equals_Vsf is v81), then the pair narrowed. */
static inline HVX_Vector hvx_hf_narrow(HVX_VectorPair w) {
  const HVX_Vector one = hvx_splat_sf(1.0f);
  return Q6_Vhf_equals_Wqf32(
    Q6_W_vcombine_VV(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(w), one),
                     Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(w), one)));
}

/** @brief x unchanged, but opaque to the optimiser: hexagon-clang 19 -O3
 *         folds a Vsf_equals_Vqf32 into the qf32 op that consumes it
 *         (vadd(Vu.qf32, Vv.sf), vmpy(Vu.qf32, Vv.qf32)) even when both
 *         are written as intrinsics, which skips that f32 rounding. */
static inline HVX_Vector hvx_hf_pin(HVX_Vector x) {
#if defined(__hexagon__)
  __asm__("" : "+v"(x));
#endif
  return x;
}

/** @brief One IEEE f32 multiply / add / subtract, rounded on its own. */
static inline HVX_Vector hvx_hf_mul_rn(HVX_Vector a, HVX_Vector b) {
  return hvx_hf_pin(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b)));
}
static inline HVX_Vector hvx_hf_add_rn(HVX_Vector a, HVX_Vector b) {
  return hvx_hf_pin(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b)));
}
static inline HVX_Vector hvx_hf_sub_rn(HVX_Vector a, HVX_Vector b) {
  return hvx_hf_pin(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(a, b)));
}

/** @brief hvx_rne16_sf with each rounding explicit (its input here comes
 *         straight from a qf32 multiply, which the compiler must not fuse
 *         into the add). */
static inline HVX_Vector hvx_hf_rne16_rn(HVX_Vector x) {
  HVX_Vector ef = Q6_V_vand_VV(x, Q6_V_vsplat_R(0x7F800000));
  ef = Q6_Vw_vmax_VwVw(ef, Q6_V_vsplat_R((int)ATTN_M1_DET_EF_MIN16));
  const HVX_Vector c = Q6_Vw_vadd_VwVw(ef, Q6_V_vsplat_R(ATTN_M1_DET_RNE16_C));
  const HVX_Vector r = hvx_hf_sub_rn(hvx_hf_add_rn(x, c), c);
  return Q6_V_vor_VV(r, Q6_V_vand_VV(x, Q6_V_vsplat_R((int)0x80000000u)));
}

/**
 * @brief attn_m1_det_exp16 on 32 sf lanes holding fp16 values d <= 0 (the
 *        softmax's rne16(s - m)): the spec's exp_ps step for step. The
 *        upper clamp is dropped (d <= 0); the trunc-and-correct of fx is
 *        computed as floor directly (the same value: an integer in
 *        [-128, 0], exact in every form); (int)fx comes from the 1.5 * 2^23
 *        add, exact for |fx| < 2^22.
 */
static inline HVX_Vector hvx_hf_exp16_sf(HVX_Vector x) {
  const HVX_Vector magic = hvx_splat_sf(12582912.0f); /* 1.5 * 2^23 */
  const HVX_Vector one = hvx_splat_sf(1.0f);
  x = Q6_Vsf_vmax_VsfVsf(x, hvx_splat_sf(ATTN_M1_DET_EXP_LO));
  const HVX_Vector fx = hvx_hf_add_rn(
    hvx_hf_mul_rn(x, hvx_splat_sf(ATTN_M1_DET_LOG2EF)), hvx_splat_sf(0.5f));
  /* floor(fx): the nearest integer, minus 1 where it is above fx (fx - r
     is exact and never -0: fx is not an integer near 0). */
  const HVX_Vector r = hvx_hf_sub_rn(hvx_hf_add_rn(fx, magic), magic);
  const HVX_VectorPred above =
    Q6_Q_vcmp_gt_VwVw(Q6_V_vzero(), hvx_hf_sub_rn(fx, r));
  const HVX_Vector fl =
    hvx_hf_sub_rn(r, Q6_V_vmux_QVV(above, one, Q6_V_vzero()));
  const HVX_Vector ki =
    Q6_Vw_vsub_VwVw(hvx_hf_add_rn(fl, magic), Q6_V_vsplat_R(0x4B400000));
  const HVX_Vector mm =
    Q6_Vw_vasl_VwR(Q6_Vw_vadd_VwVw(ki, Q6_V_vsplat_R(0x7F)), 23);
  const HVX_Vector t1 = hvx_hf_mul_rn(fl, hvx_splat_sf(ATTN_M1_DET_EXP_C1));
  HVX_Vector z = hvx_hf_mul_rn(fl, hvx_splat_sf(ATTN_M1_DET_EXP_C2));
  x = hvx_hf_sub_rn(hvx_hf_sub_rn(x, t1), z);
  /* neon_mathfun's cephes coefficients, as attn_m1_det_exp_ps holds them */
  HVX_Vector y = hvx_hf_mul_rn(hvx_splat_sf((float)1.9875691500E-4), x);
  z = hvx_hf_mul_rn(x, x);
  y = hvx_hf_add_rn(y, hvx_splat_sf((float)1.3981999507E-3));
  y = hvx_hf_mul_rn(y, x);
  y = hvx_hf_add_rn(y, hvx_splat_sf((float)8.3334519073E-3));
  y = hvx_hf_mul_rn(y, x);
  y = hvx_hf_add_rn(y, hvx_splat_sf((float)4.1665795894E-2));
  y = hvx_hf_mul_rn(y, x);
  y = hvx_hf_add_rn(y, hvx_splat_sf((float)1.6666665459E-1));
  y = hvx_hf_mul_rn(y, x);
  y = hvx_hf_add_rn(y, hvx_splat_sf((float)5.0000001201E-1));
  y = hvx_hf_mul_rn(y, z);
  y = hvx_hf_add_rn(y, x);
  y = hvx_hf_add_rn(y, one);
  return hvx_hf_rne16_rn(hvx_hf_mul_rn(y, mm));
}

/** @brief exp16 on 64 hf lanes, d <= 0. */
static inline HVX_Vector hvx_hf_exp16(HVX_Vector d, HVX_Vector one) {
  const HVX_VectorPair w = hvx_hf_widen(d, one);
  return hvx_hf_narrow(Q6_W_vcombine_VV(hvx_hf_exp16_sf(Q6_V_hi_W(w)),
                                        hvx_hf_exp16_sf(Q6_V_lo_W(w))));
}

/** @brief Fills ATTN_M1_DET_EXP_N fp16 bits of attn_m1_det_exp_table (the
 *         entries are fp16 values, so the rounding is exact); @a tmp holds
 *         ATTN_M1_DET_EXP_N floats. */
static inline void hvx_hf_exp16_fill(uint16_t *tab, float *tmp) {
  attn_m1_det_exp_table(tmp);
  for (uint32_t i = 0; i < ATTN_M1_DET_EXP_N; ++i) {
    tab[i] = hvx_hf_bits_rne(tmp[i]);
  }
}

/**
 * @brief attn_m1_det_exp_index on 64 hf lanes d <= 0. For a normal fp16
 *        magnitude m (bits >= 0x400) the spec's (f32 bits >> 13) - BIAS is
 *        m + 0x1C000 - 0x1C3FF = m - 1023; a zero or subnormal saturates to
 *        0 (exp16 = 1), anything from 17.5 on clamps to LAST + 1 (0.0).
 *        The sign bit is dropped (the domain is d <= 0, -0 gives 0).
 */
static inline HVX_Vector hvx_hf_exp16_idx(HVX_Vector d) {
  const HVX_Vector mag = Q6_V_vand_VV(d, Q6_Vh_vsplat_R(0x7FFF));
  return Q6_Vuh_vmin_VuhVuh(Q6_Vuh_vsub_VuhVuh_sat(mag, Q6_Vh_vsplat_R(1023)),
                            Q6_Vh_vsplat_R((int)(ATTN_M1_DET_EXP_N - 1u)));
}

/**
 * @brief rne16(e / l) on 64 hf lanes, e >= 0 and l in [1, 2048] fp16 (the
 *        softmax's domain, hvx_div16_sf's comment): @a l the divisor
 *        widened like e (a splat pair for the kernel), @a r =
 *        hvx_recip_det_sf of each half.
 */
static inline HVX_Vector hvx_hf_div16(HVX_Vector e, HVX_VectorPair l,
                                      HVX_VectorPair r, HVX_Vector one) {
  const HVX_VectorPair w = hvx_hf_widen(e, one);
  return hvx_hf_narrow(
    Q6_W_vcombine_VV(hvx_div16_sf(Q6_V_hi_W(w), Q6_V_hi_W(l), Q6_V_hi_W(r)),
                     hvx_div16_sf(Q6_V_lo_W(w), Q6_V_lo_W(l), Q6_V_lo_W(r))));
}

/**
 * @brief hvx_hf_bits_rne on 32 f32 lanes: the fp16 bits of rne16(x) in the
 *        low halfword of each word (the high one is 0).
 *
 * The same steps as the scalar function, so the same bits by construction
 * (ATTN M1 HF PRIM walks every f32 below 65520): at or above 2^-14 the
 * normal fp16 is the magnitude's bits shifted down 13 with round to
 * nearest even (+ 0xFFF + the kept lsb, then the shift) and the exponent
 * rebased; below it the subnormal grid 2^-24, i.e. the integer nearest to
 * |x| * 2^24 (exact) by the 1.5 * 2^23 add of hvx_hf_exp16_sf (one
 * rounding, ties to even; 1024 is the smallest normal's bits, as it should
 * be). The sign is ORed back as bits, so a zero result keeps it -- no qf32
 * conversion touches the result (hvx_hf_narrow would, and qf32 has no -0).
 */
static inline HVX_Vector hvx_hf_bits_rne_w(HVX_Vector x) {
  const HVX_Vector a = Q6_V_vand_VV(x, Q6_V_vsplat_R(0x7FFFFFFF));
  const HVX_Vector s =
    Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(x, 16), Q6_V_vsplat_R(0x8000));
  const HVX_Vector lsb =
    Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(a, 13), Q6_V_vsplat_R(1));
  const HVX_Vector rn = Q6_Vw_vsub_VwVw(
    Q6_Vuw_vlsr_VuwR(
      Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(a, Q6_V_vsplat_R(0xFFF)), lsb), 13),
    Q6_V_vsplat_R(112 << 10));
  const HVX_Vector rs =
    Q6_Vw_vsub_VwVw(hvx_hf_add_rn(hvx_hf_mul_rn(a, hvx_splat_sf(16777216.0f)),
                                  hvx_splat_sf(12582912.0f)),
                    Q6_V_vsplat_R(0x4B400000));
  const HVX_VectorPred normal =
    Q6_Q_vcmp_gt_VwVw(a, Q6_V_vsplat_R(ATTN_M1_DET_EF_MIN16 - 1u));
  return Q6_V_vor_VV(Q6_V_vmux_QVV(normal, rn, rs), s);
}

/** @brief One head row: 64 f32 at @a x (any 4-byte alignment) rounded to
 *         64 fp16 lanes, lane i = hvx_hf_bits_rne(x[i]). */
static inline HVX_Vector hvx_hf_round_row(const float *x) {
  return Q6_Vh_vpacke_VwVw(hvx_hf_bits_rne_w(*(const HVX_UVector *)(x + 32)),
                           hvx_hf_bits_rne_w(*(const HVX_UVector *)x));
}

/**
 * @brief hvx_hf_round_row's 64 fp16 lanes in vlut16's order: halfword
 *        2 (d & 31) + (d >> 5) holds lane d. vshuffe of the two word
 *        vectors is that zip (Vv's low halfwords to the even lanes, Vu's
 *        to the odd), where vpacke would lay them end to end.
 */
static inline HVX_Vector hvx_hf_round_row_lut(const float *x) {
  return Q6_Vh_vshuffe_VhVh(hvx_hf_bits_rne_w(*(const HVX_UVector *)(x + 32)),
                            hvx_hf_bits_rne_w(*(const HVX_UVector *)x));
}

/**
 * @brief Lanes d and d + 1 (d even) of a zipped row (hvx_hf_round_row_lut),
 *        each in all 64 lanes of one half of the pair: @a idx holds byte d
 *        in its even bytes and d + 1 in its odd ones, @a rt = d >> 4.
 *        vlut16 in 128-byte mode reads, for index byte i, halfword 2 (i &
 *        15) + 32 (Rt & 1) + ((Rt >> 1) & 1) of the row when i >> 4 == (Rt
 *        & 15), else 0; output lane j of lo takes byte 2j, of hi byte 2j +
 *        1 (hexagon-sim -mv79; hvx_emu is diffed against it). With i < 64
 *        and Rt = i >> 4 that is halfword 2 (i & 31) + (i >> 5): lane i. So
 *        one permute gives two splats (d and d + 1 share Rt for d even).
 */
static inline HVX_VectorPair hvx_hf_splat2_lut(HVX_Vector idx, HVX_Vector row,
                                               int rt) {
  return Q6_Wh_vlut16_VbVhR(idx, row, rt);
}

/**
 * @brief 32 fp16 values (zero-extended words) to f32, exactly and with the
 *        sign of a zero kept: the kernel's scalar hf_float per lane. A
 *        normal is its bits rebased; a subnormal m * 2^-24 (m < 1024 made
 *        f32 by hvx_w_to_sf, then one exact multiply); the sign ORed last.
 */
static inline HVX_Vector hvx_hf_bits_to_sf(HVX_Vector h) {
  const HVX_Vector a = Q6_V_vand_VV(h, Q6_V_vsplat_R(0x7FFF));
  const HVX_Vector s =
    Q6_Vw_vasl_VwR(Q6_V_vand_VV(h, Q6_V_vsplat_R(0x8000)), 16);
  const HVX_Vector nrm =
    Q6_Vw_vadd_VwVw(Q6_Vw_vasl_VwR(a, 13), Q6_V_vsplat_R(112 << 23));
  const HVX_Vector sub =
    hvx_hf_mul_rn(hvx_w_to_sf(a), hvx_splat_sf(1.0f / 16777216.0f));
  const HVX_VectorPred normal = Q6_Q_vcmp_gt_VwVw(a, Q6_V_vsplat_R(0x3FF));
  return Q6_V_vor_VV(Q6_V_vmux_QVV(normal, nrm, sub), s);
}

/** @brief 64 fp16 lanes to 64 f32 at @a out (any 4-byte alignment). */
static inline void hvx_hf_store_sf(float *out, HVX_Vector h) {
  const HVX_VectorPair w = Q6_Wuw_vunpack_Vuh(h);
  *(HVX_UVector *)out = hvx_hf_bits_to_sf(Q6_V_lo_W(w));
  *(HVX_UVector *)(out + 32) = hvx_hf_bits_to_sf(Q6_V_hi_W(w));
}

#endif /* __NNTRAINER_HVX_ATTN_M1_HF_H__ */
