// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hvx_q4_gemv_f32.c
 * @date   29 Sep 2026
 * @brief  The Android CPU's M=1 Q4_0 GEMV on the DSP, bit for bit
 *         (hvx_q4_gemv_f32.h; spec q4_gemv_cpu_det.h)
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * Why the HVX form is exact, per 32 columns and one block b:
 *   T  = isum * m_w (int32; |isum| <= 32 * 8 * 127, m_w < 2^11, so
 *        |T| < 2^26), signed by d_w
 *   T  = Th 2^13 + Tl, 0 <= Tl < 2^13; Th * m_a and Tl * m_a are < 2^24
 *   P1 = f32(Th m_a) 2^(e + 13), P2 = f32(Tl m_a) 2^e: exact, and
 *        P1 + P2 = isum * d_w * d_a = the CPU's isum * s exactly
 *   Fast2Sum(P1, P2) = uh + ul exactly (|P1| > |P2| unless Th = 0, when
 *        P1 = 0); TwoSum(acc, uh) = th + tl exactly;
 *   acc' = RN(th + RO(tl + ul)), RO = round to odd, is RN(acc + P1 + P2)
 *        (Boldo and Melquiond, "Emulation of FMA and correctly rounded
 *        sums: proved algorithms using rounding to odd", 2008)
 * Every step is one f32 add, sub or mul rounded to nearest, which is what
 * the proof needs of the vector unit (G1 checks it on silicon). The
 * integer part is exact by construction. No subnormal can arise in P1 /
 * P2 (their exponents are >= -48 - 26); acc and the sums may be
 * subnormal, which TwoSum and the round-to-odd step handle as IEEE does.
 */

#include "hvx_q4_gemv_f32.h"

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <string.h>

#include "q4_gemv_cpu_det.h"

#define Q4M1_INLINE static inline __attribute__((always_inline))

/* One f32 op per lane: the Q6_Vsf_* intrinsics, which the compiler
   lowers to a qf32 op and a conversion. That pair is the IEEE op on
   silicon (the 2026-09-29 sitting: bit-identical on every G1 row); the
   inline-asm IEEE .sf instructions the ISS also accepts returned 0 on the
   phone (measurement 132-pr2 section G1), so they are not used. */
#define sf_add Q6_Vsf_vadd_VsfVsf
#define sf_sub Q6_Vsf_vsub_VsfVsf
#define sf_mpy Q6_Vsf_vmpy_VsfVsf

/** @brief The error of s = RN(a + b): a + b - s, exact (TwoSum). */
Q4M1_INLINE HVX_Vector two_sum_err(HVX_Vector a, HVX_Vector b, HVX_Vector s) {
  const HVX_Vector bb = sf_sub(s, a);
  const HVX_Vector ab = sf_sub(s, bb);
  return sf_add(sf_sub(a, ab), sf_sub(b, bb));
}

/** @brief amax / 127 and 1 / d below this take the spec's own path: a
 *  block whose amax is under 2^-100 keeps every x id product a subnormal x
 *  could reach below 2^-26 (so q = 0 whether the vector unit flushes it or
 *  not) and 1 / d finite. No decode activation block is that small. */
#define Q4M1_PREP_TINY 0x0d800000u /* 2^-100 */

/** @brief The per-block terms of an f16 d the kernels read; @a s the sum
 *         of the block's quants. */
static inline void q4m1_block_terms(hvx_q4m1_act *a, uint32_t b, int32_t s) {
  const uint32_t h = a->d[b], e = (h >> 10) & 31u;
  a->s8[b] = -8 * s;
  a->ma[b] = (int32_t)((h & 1023u) | (e ? 1024u : 0u));
  a->ea[b] = (int32_t)(e ? e : 1u) - 50 + 127;
  a->df[b] = (e != 0u && e != 31u)
               ? cpu_det_float(((h & 0x8000u) << 16) |
                               (((h & 0x7fffu) << 13) + (112u << 23)))
               : cpu_det_f16_to_f32((uint16_t)h);
}

/** @brief cpu_det_f32_to_f16 with the f16-normal case inline (RN on the 13
 *         dropped bits; a carry into the exponent, or to inf, is right). */
static inline uint16_t q4m1_f32_to_f16(float f) {
  const uint32_t u = cpu_det_bits(f), a = u & 0x7fffffffu;
  if (a < 0x38800000u || a >= 0x47800000u) { /* f16 subnormal, or >= 2^16 */
    return cpu_det_f32_to_f16(f);
  }
  uint32_t h = ((a >> 23) - 112u) << 10 | ((a >> 13) & 0x3ffu);
  const uint32_t rem = a & 0x1fffu;
  h += (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ? 1u : 0u;
  return (uint16_t)(((u >> 16) & 0x8000u) | h);
}

/** @brief Blocks per call the quantizer's static scratch serves (K <=
 *  8192, the entry's own limit). */
#define Q4M1_PREP_MAX_NB 256u

/**
 * q8_0_quant_cpu_det on the vector unit (K % 128 == 0, K <= 8192), one
 * 32-float block per vector, in
 * three passes so the vector and scalar units meet twice per call, not
 * twice per block:
 *   1 (vector) amax = max |x|: word max of the sign-cleared bits (exact,
 *              no FP op), stored per block
 *   2 (scalar) d = RN(amax / 127), id = RN(1 / d): the Hexagon's scalar
 *              divide (sfrecipa / sffixup; IEEE RN, checked against
 *              cpu_det_div_rn over every mantissa on the ISS and on
 *              silicon, HvxFcQ4.ScalarDivide), then d's f16 and terms
 *   3 (vector) q = RN-even(x * id) by the 1.5 * 2^23 magic add (|x id| <=
 *              127.5, well inside its 2^22 range; the add rounds ties to
 *              even, as fcvtns does); four blocks' low bytes per 128-byte
 *              store (vpacke), and the block sums by a rotate-add tree
 * Both Vsf ops are the silicon-checked qf32 pairs. A block under
 * Q4M1_PREP_TINY, or not finite, runs the scalar spec in pass 2.
 * ponytail: static scratch, one call at a time (as the entry file's).
 */
void hvx_q4m1_prep(const float *x, uint32_t K, hvx_q4m1_act *a) {
  /* lane j of amax_v[b / 32] / sum_v[b / 32] is block b = 32 (b / 32) + j:
     one vector store per 32 blocks, read back as consecutive words */
  static HVX_Vector amax_v[Q4M1_PREP_MAX_NB / 32u],
    sum_v[Q4M1_PREP_MAX_NB / 32u];
  static int32_t id_bits[Q4M1_PREP_MAX_NB];
  static const int32_t lane_idx[32] __attribute__((aligned(128))) = {
    0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
  const HVX_Vector lanes = *(const HVX_Vector *)lane_idx;
  static uint8_t scalar_blk[Q4M1_PREP_MAX_NB];
  const HVX_Vector absm = Q6_V_vsplat_R(0x7FFFFFFF);
  const HVX_Vector magic = Q6_V_vsplat_R(0x4B400000); /* 1.5 * 2^23 */
  const uint32_t nb = K / Q4_CPU_QK;
  const HVX_UVector *xv = (const HVX_UVector *)x;
  for (uint32_t b = 0; b < nb; b += 4u) { /* 4 independent trees */
    HVX_Vector m[4];
    for (uint32_t j = 0; j < 4u; ++j) {
      m[j] = Q6_V_vand_VV(xv[b + j], absm);
    }
    for (int r = 64; r >= 4; r >>= 1) {
      for (uint32_t j = 0; j < 4u; ++j) {
        m[j] = Q6_Vw_vmax_VwVw(m[j], Q6_V_vror_VR(m[j], r));
      }
    }
    for (uint32_t j = 0; j < 4u; ++j) { /* amax in every lane of m[j] */
      amax_v[(b + j) / 32u] = Q6_V_vmux_QVV(
        Q6_Q_vcmp_eq_VwVw(lanes, Q6_V_vsplat_R((int32_t)((b + j) % 32u))), m[j],
        amax_v[(b + j) / 32u]);
    }
  }
  const uint32_t *amax_w = (const uint32_t *)amax_v;
  const int32_t *sum_w = (const int32_t *)sum_v;
  for (uint32_t b = 0; b < nb; ++b) {
    const uint32_t amax_bits = amax_w[b];
    scalar_blk[b] = amax_bits < Q4M1_PREP_TINY || amax_bits >= 0x7f800000u;
    if (scalar_blk[b]) {
      id_bits[b] = 0;
      continue;
    }
    const float d = cpu_det_float(amax_bits) / 127.0f;
    const float id = 1.0f / d;
    memcpy(&id_bits[b], &id, sizeof(float));
    a->d[b] = q4m1_f32_to_f16(d);
  }
  for (uint32_t b = 0; b < nb; b += 4u) {
    HVX_Vector qw[4], sv[4];
    for (uint32_t j = 0; j < 4u; ++j) {
      qw[j] = Q6_Vw_vsub_VwVw(
        Q6_Vsf_vadd_VsfVsf(
          Q6_Vsf_vmpy_VsfVsf(xv[b + j], Q6_V_vsplat_R(id_bits[b + j])), magic),
        magic);
      sv[j] = qw[j];
    }
    for (int r = 64; r >= 4; r >>= 1) {
      for (uint32_t j = 0; j < 4u; ++j) {
        sv[j] = Q6_Vw_vadd_VwVw(sv[j], Q6_V_vror_VR(sv[j], r));
      }
    }
    for (uint32_t j = 0; j < 4u; ++j) {
      sum_v[(b + j) / 32u] = Q6_V_vmux_QVV(
        Q6_Q_vcmp_eq_VwVw(lanes, Q6_V_vsplat_R((int32_t)((b + j) % 32u))),
        sv[j], sum_v[(b + j) / 32u]);
    }
    *(HVX_UVector *)(a->q + (size_t)b * Q4_CPU_QK) = Q6_Vb_vpacke_VhVh(
      Q6_Vh_vpacke_VwVw(qw[3], qw[2]), Q6_Vh_vpacke_VwVw(qw[1], qw[0]));
  }
  for (uint32_t b = 0; b < nb; ++b) {
    int32_t s;
    if (scalar_blk[b]) {
      int8_t *qb = a->q + (size_t)b * Q4_CPU_QK;
      q8_0_quant_cpu_det(x + (size_t)b * Q4_CPU_QK, Q4_CPU_QK, qb, &a->d[b]);
      s = 0;
      for (uint32_t j = 0; j < Q4_CPU_QK; ++j) {
        s += qb[j];
      }
    } else {
      s = sum_w[b];
    }
    q4m1_block_terms(a, b, s);
  }
}

/** @brief isum of block b for the 32 columns of one 512-byte nibble
 *         block: sum q w - 8 sum q, int32 per lane (vrmpy: unsigned
 *         nibble bytes times the four signed quants of Rt). */
Q4M1_INLINE HVX_Vector q4m1_isum(const HVX_Vector *wv, const int8_t *q,
                                 int32_t s8) {
  const HVX_Vector m0f = Q6_V_vsplat_R(0x0F0F0F0F);
  const uint32_t *a = (const uint32_t *)q;
  HVX_Vector is = Q6_V_vsplat_R(s8);
  for (uint32_t j = 0; j < 4u; ++j) {
    const HVX_Vector v = wv[j];
    is = Q6_Vw_vrmpyacc_VwVubRb(is, Q6_V_vand_VV(v, m0f), (int)a[2u * j]);
    is = Q6_Vw_vrmpyacc_VwVubRb(is, Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(v, 4), m0f),
                                (int)a[2u * j + 1u]);
  }
  return is;
}

/** @brief One group, all blocks, on the vector unit (NATIVE / INTRIN). */
static void q4m1_group_hvx(const uint8_t *wg, uint32_t np,
                           const hvx_q4m1_act *a, float *y) {
  const HVX_Vector zero = Q6_V_vzero(), one = Q6_V_vsplat_R(1);
  const HVX_Vector m13 = Q6_V_vsplat_R(0x1FFF), m3ff = Q6_V_vsplat_R(0x3FF);
  const HVX_Vector m1f = Q6_V_vsplat_R(31), c400 = Q6_V_vsplat_R(0x400);
  const HVX_Vector e13 = Q6_V_vsplat_R(13 << 23);
  const HVX_Vector absm = Q6_V_vsplat_R(0x7FFFFFFF);
  HVX_Vector acc = zero;
  for (uint32_t p = 0; p < np; ++p) {
    const uint8_t *unit = wg + (size_t)p * Q4M1_PAIR_BYTES;
    const HVX_VectorPair dwp =
      Q6_Ww_vunpack_Vh(*(const HVX_Vector *)(unit + Q4M1_D_OFF));
    for (uint32_t hb = 0; hb < 2u; ++hb) {
      const uint32_t b = 2u * p + hb;
      const HVX_Vector is = q4m1_isum((const HVX_Vector *)(unit + 512u * hb),
                                      a->q + (size_t)b * Q4_CPU_QK, a->s8[b]);
      /* d_w: sign, exponent, integer mantissa; T = isum * m_w signed */
      const HVX_Vector h = hb ? Q6_V_hi_W(dwp) : Q6_V_lo_W(dwp);
      const HVX_Vector E = Q6_V_vand_VV(Q6_Vw_vasr_VwR(h, 10), m1f);
      HVX_Vector mw = Q6_V_vand_VV(h, m3ff);
      mw = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(E, zero), mw, Q6_V_vor_VV(mw, c400));
      const HVX_Vector En = Q6_Vw_vmax_VwVw(E, one);
      const HVX_Vector sg = Q6_Vw_vasr_VwR(h, 31); /* 0 or -1 */
      HVX_Vector T = Q6_Vw_vmpyie_VwVuh(is, mw);
      T = Q6_Vw_vsub_VwVw(Q6_V_vxor_VV(T, sg), sg);
      const HVX_Vector Th = Q6_Vw_vasr_VwR(T, 13), Tl = Q6_V_vand_VV(T, m13);
      const HVX_Vector vma = Q6_V_vsplat_R(a->ma[b]);
      const HVX_Vector F1 = Q6_Vsf_equals_Vw(Q6_Vw_vmpyie_VwVuh(Th, vma));
      const HVX_Vector F2 = Q6_Vsf_equals_Vw(Q6_Vw_vmpyie_VwVuh(Tl, vma));
      /* 2^(En - 25 + max(e_a, 1) - 25) and 2^13 times it */
      const HVX_Vector S2 =
        Q6_Vw_vasl_VwR(Q6_Vw_vadd_VwVw(En, Q6_V_vsplat_R(a->ea[b])), 23);
      const HVX_Vector S1 = Q6_Vw_vadd_VwVw(S2, e13);
      const HVX_Vector P1 = sf_mpy(F1, S1);
      const HVX_Vector P2 = sf_mpy(F2, S2);
      /* Fast2Sum(P1, P2) */
      const HVX_Vector uh = sf_add(P1, P2);
      const HVX_Vector ul = sf_sub(P2, sf_sub(uh, P1));
      /* TwoSum(acc, uh) */
      const HVX_Vector th = sf_add(acc, uh);
      const HVX_Vector tl = two_sum_err(acc, uh, th);
      /* w = RO(tl + ul): RN, then one ulp toward the error when inexact
         and even */
      const HVX_Vector w = sf_add(tl, ul);
      const HVX_Vector er = two_sum_err(tl, ul, w);
      const HVX_Vector dir =
        Q6_V_vor_VV(Q6_Vw_vasr_VwR(Q6_V_vxor_VV(w, er), 31), one);
      const HVX_VectorPred keep =
        Q6_Q_or_QQ(Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(w, one), one),
                   Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(er, absm), zero));
      const HVX_Vector wo = Q6_V_vmux_QVV(keep, w, Q6_Vw_vadd_VwVw(w, dir));
      acc = sf_add(th, wo);
    }
  }
  *(HVX_UVector *)y = acc;
}

void hvx_q4m1_gemv_groups(const uint8_t *w, uint32_t K, uint32_t ngroups,
                          const hvx_q4m1_act *a, float *y) {
  const uint32_t np = K / 64u;
  const size_t gbytes = (size_t)np * Q4M1_PAIR_BYTES;
  for (uint32_t g = 0; g < ngroups; ++g) {
    q4m1_group_hvx(w + g * gbytes, np, a, y + (size_t)g * Q4M1_GROUP);
  }
}
