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

/* One f32 op per lane: the IEEE instruction when @a native (DSP only),
   else the Q6_Vsf_* intrinsic. Every caller passes a constant, so the
   branch folds away. */
Q4M1_INLINE HVX_Vector sf_add(HVX_Vector a, HVX_Vector b, const int native) {
#if defined(__hexagon__)
  if (native) {
    HVX_Vector r;
    __asm__("%0.sf = vadd(%1.sf,%2.sf)" : "=v"(r) : "v"(a), "v"(b));
    return r;
  }
#endif
  (void)native;
  return Q6_Vsf_vadd_VsfVsf(a, b);
}
Q4M1_INLINE HVX_Vector sf_sub(HVX_Vector a, HVX_Vector b, const int native) {
#if defined(__hexagon__)
  if (native) {
    HVX_Vector r;
    __asm__("%0.sf = vsub(%1.sf,%2.sf)" : "=v"(r) : "v"(a), "v"(b));
    return r;
  }
#endif
  (void)native;
  return Q6_Vsf_vsub_VsfVsf(a, b);
}
Q4M1_INLINE HVX_Vector sf_mpy(HVX_Vector a, HVX_Vector b, const int native) {
#if defined(__hexagon__)
  if (native) {
    HVX_Vector r;
    __asm__("%0.sf = vmpy(%1.sf,%2.sf)" : "=v"(r) : "v"(a), "v"(b));
    return r;
  }
#endif
  (void)native;
  return Q6_Vsf_vmpy_VsfVsf(a, b);
}

/** @brief The error of s = RN(a + b): a + b - s, exact (TwoSum). */
Q4M1_INLINE HVX_Vector two_sum_err(HVX_Vector a, HVX_Vector b, HVX_Vector s,
                                   const int native) {
  const HVX_Vector bb = sf_sub(s, a, native);
  const HVX_Vector ab = sf_sub(s, bb, native);
  return sf_add(sf_sub(a, ab, native), sf_sub(b, bb, native), native);
}

void hvx_q4m1_prep(const float *x, uint32_t K, hvx_q4m1_act *a) {
  q8_0_quant_cpu_det(x, K, a->q, a->d);
  for (uint32_t b = 0; b < K / Q4_CPU_QK; ++b) {
    int32_t s = 0;
    for (uint32_t j = 0; j < Q4_CPU_QK; ++j) {
      s += a->q[b * Q4_CPU_QK + j];
    }
    const uint32_t h = a->d[b], e = (h >> 10) & 31u;
    a->s8[b] = -8 * s;
    a->ma[b] = (int32_t)((h & 1023u) | (e ? 1024u : 0u));
    a->ea[b] = (int32_t)(e ? e : 1u) - 50 + 127;
    a->df[b] = cpu_det_f16_to_f32((uint16_t)h);
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
Q4M1_INLINE void q4m1_group_hvx(const uint8_t *wg, uint32_t np,
                                const hvx_q4m1_act *a, float *y,
                                const int native) {
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
      const HVX_Vector P1 = sf_mpy(F1, S1, native);
      const HVX_Vector P2 = sf_mpy(F2, S2, native);
      /* Fast2Sum(P1, P2) */
      const HVX_Vector uh = sf_add(P1, P2, native);
      const HVX_Vector ul = sf_sub(P2, sf_sub(uh, P1, native), native);
      /* TwoSum(acc, uh) */
      const HVX_Vector th = sf_add(acc, uh, native);
      const HVX_Vector tl = two_sum_err(acc, uh, th, native);
      /* w = RO(tl + ul): RN, then one ulp toward the error when inexact
         and even */
      const HVX_Vector w = sf_add(tl, ul, native);
      const HVX_Vector er = two_sum_err(tl, ul, w, native);
      const HVX_Vector dir =
        Q6_V_vor_VV(Q6_Vw_vasr_VwR(Q6_V_vxor_VV(w, er), 31), one);
      const HVX_VectorPred keep =
        Q6_Q_or_QQ(Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(w, one), one),
                   Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(er, absm), zero));
      const HVX_Vector wo = Q6_V_vmux_QVV(keep, w, Q6_Vw_vadd_VwVw(w, dir));
      acc = sf_add(th, wo, native);
    }
  }
  *(HVX_UVector *)y = acc;
}

/** @brief One group's isum (as f32) and s = d_w d_a for every block, into
 *         fs[64 b .. 64 b + 31] and fs[64 b + 32 ..]. Both exact. */
static void q4m1_group_terms(const uint8_t *wg, uint32_t np,
                             const hvx_q4m1_act *a, float *fs) {
  const HVX_Vector zero = Q6_V_vzero(), one = Q6_V_vsplat_R(1);
  const HVX_Vector m3ff = Q6_V_vsplat_R(0x3FF), m1f = Q6_V_vsplat_R(31);
  const HVX_Vector c400 = Q6_V_vsplat_R(0x400);
  for (uint32_t p = 0; p < np; ++p) {
    const uint8_t *unit = wg + (size_t)p * Q4M1_PAIR_BYTES;
    const HVX_VectorPair dwp =
      Q6_Ww_vunpack_Vh(*(const HVX_Vector *)(unit + Q4M1_D_OFF));
    for (uint32_t hb = 0; hb < 2u; ++hb) {
      const uint32_t b = 2u * p + hb;
      const HVX_Vector is = q4m1_isum((const HVX_Vector *)(unit + 512u * hb),
                                      a->q + (size_t)b * Q4_CPU_QK, a->s8[b]);
      const HVX_Vector h = hb ? Q6_V_hi_W(dwp) : Q6_V_lo_W(dwp);
      const HVX_Vector E = Q6_V_vand_VV(Q6_Vw_vasr_VwR(h, 10), m1f);
      HVX_Vector mw = Q6_V_vand_VV(h, m3ff);
      mw = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(E, zero), mw, Q6_V_vor_VV(mw, c400));
      const HVX_Vector sg = Q6_Vw_vasr_VwR(h, 31);
      mw = Q6_Vw_vsub_VwVw(Q6_V_vxor_VV(mw, sg), sg);
      /* d_w = m_w 2^(max(E, 1) - 25), then s = d_w * d_a: both exact */
      const HVX_Vector pw = Q6_Vw_vasl_VwR(
        Q6_Vw_vadd_VwVw(Q6_Vw_vmax_VwVw(E, one), Q6_V_vsplat_R(127 - 25)), 23);
      int32_t dab;
      memcpy(&dab, &a->df[b], sizeof(dab));
      const HVX_Vector daf = Q6_V_vsplat_R(dab);
      const HVX_Vector dw = sf_mpy(Q6_Vsf_equals_Vw(mw), pw, 1);
      *(HVX_Vector *)(fs + 64u * b) = Q6_Vsf_equals_Vw(is);
      *(HVX_Vector *)(fs + 64u * b + 32u) = sf_mpy(dw, daf, 1);
    }
  }
}

/** @brief The CPU's chains on the scalar core, @a C columns in flight. */
Q4M1_INLINE void q4m1_chains(const float *fs, uint32_t nb, float *y,
                             const uint32_t C) {
  for (uint32_t c0 = 0; c0 < Q4M1_GROUP; c0 += C) {
    float acc[Q4M1_GROUP];
    for (uint32_t c = 0; c < C; ++c) {
      acc[c] = 0.0f;
    }
    for (uint32_t b = 0; b < nb; ++b) {
      const float *f = fs + 64u * b + c0, *s = f + 32u;
      for (uint32_t c = 0; c < C; ++c) {
        acc[c] = Q6_R_sfmpyacc_RR(acc[c], f[c], s[c]); /* sffma */
      }
    }
    for (uint32_t c = 0; c < C; ++c) {
      y[c0 + c] = acc[c];
    }
  }
}

void hvx_q4m1_gemv_groups(const uint8_t *w, uint32_t K, uint32_t ngroups,
                          const hvx_q4m1_act *a, float *y, uint32_t variant,
                          uint32_t cif, float *fs) {
  const uint32_t np = K / 64u, nb = K / Q4_CPU_QK;
  const size_t gbytes = (size_t)np * Q4M1_PAIR_BYTES;
  for (uint32_t g = 0; g < ngroups; ++g) {
    const uint8_t *wg = w + g * gbytes;
    float *yg = y + (size_t)g * Q4M1_GROUP;
    if (variant == HVX_Q4M1_NATIVE) {
      q4m1_group_hvx(wg, np, a, yg, 1);
    } else if (variant == HVX_Q4M1_INTRIN) {
      q4m1_group_hvx(wg, np, a, yg, 0);
    } else {
      q4m1_group_terms(wg, np, a, fs);
      if (cif == 8u) {
        q4m1_chains(fs, nb, yg, 8u);
      } else if (cif == 16u) {
        q4m1_chains(fs, nb, yg, 16u);
      } else {
        q4m1_chains(fs, nb, yg, 32u);
      }
    }
  }
}
