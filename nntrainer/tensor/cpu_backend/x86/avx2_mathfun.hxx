/**
 * @file   avx2_mathfun.hxx
 * @date   03 Apr 2026
 * @brief  This is collection of sin, cos function with AVX2 SIMD
 * @see    https://github.com/nntrainer/nntrainer
 * @author Julien Pommier (original cephes-based algorithm)
 * @bug    No known bugs except for NYI items
 *
 */

/* AVX2 implementation of sin, cos

   Inspired by Intel Approximate Math library, and based on the
   corresponding algorithms of the cephes math library.

   Ported from the NEON implementation (neon_mathfun.hxx) to AVX2
   intrinsics, processing 8 floats at a time instead of 4.
*/

/* Copyright (C) 2011  Julien Pommier

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  (this is the zlib license)
*/

#define c_minus_cephes_DP1 -0.78515625
#define c_minus_cephes_DP2 -2.4187564849853515625e-4
#define c_minus_cephes_DP3 -3.77489497744594108e-8
#define c_sincof_p0 -1.9515295891E-4
#define c_sincof_p1 8.3321608736E-3
#define c_sincof_p2 -1.6666654611E-1
#define c_coscof_p0 2.443315711809948E-005
#define c_coscof_p1 -1.388731625493765E-003
#define c_coscof_p2 4.166664568298827E-002
#define c_cephes_FOPI 1.27323954473516 // 4 / M_PI

/**
 * @brief AVX2 equivalent of NEON vtstq_u32 : test bits
 * @note  returns all 1s per lane where (a & b) != 0
 *
 * @param a first operand
 * @param b second operand
 * @return __m256i mask with all 1s where (a & b) != 0
 */
static inline __m256i _mm256_tst_epi32(__m256i a, __m256i b) {
  __m256i t = _mm256_and_si256(a, b);
  __m256i zero = _mm256_setzero_si256();
  __m256i eq_zero = _mm256_cmpeq_epi32(t, zero);
  return _mm256_xor_si256(eq_zero, _mm256_cmpeq_epi32(zero, zero));
}

/* evaluation of 8 sines & cosines at once.

   The code is the exact rewriting of the cephes sinf function.
   Precision is excellent as long as x < 8192 (I did not bother to
   take into account the special handling they have for greater values
   -- it does not return garbage for arguments over 8192, though, but
   the extra precision is missing).

   Note that it is such that sinf((float)M_PI) = 8.74e-8, which is the
   surprising but correct result.

   Note also that when you compute sin(x), cos(x) is available at
   almost no extra price so both sin256_ps and cos256_ps make use of
   sincos256_ps..
  */
/**
 * @brief sincos function with AVX2 SIMD
 *
 * @param x input register variable (__m256, 8 floats)
 * @param ysin output sin register variable
 * @param ycos output cos register variable
 */
inline void sincos256_ps(__m256 x, __m256 *ysin, __m256 *ycos) {
  __m256 xmm1, xmm2, xmm3, y;

  __m256i emm2;

  __m256i sign_mask_sin, sign_mask_cos;
  sign_mask_sin =
    _mm256_castps_si256(_mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LT_OQ));
  x = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x); /* abs(x) */

  /* scale by 4/Pi */
  y = _mm256_mul_ps(x, _mm256_set1_ps(c_cephes_FOPI));

  /* store the integer part of y in emm2 */
  emm2 = _mm256_cvttps_epi32(y);
  /* j=(j+1) & (~1) (see the cephes sources) */
  emm2 = _mm256_add_epi32(emm2, _mm256_set1_epi32(1));
  emm2 = _mm256_and_si256(emm2, _mm256_set1_epi32(~1));
  y = _mm256_cvtepi32_ps(emm2);

  /* get the polynom selection mask
     there is one polynom for 0 <= x <= Pi/4
     and another one for Pi/4<x<=Pi/2

     Both branches will be computed.
  */
  __m256i poly_mask = _mm256_tst_epi32(emm2, _mm256_set1_epi32(2));

  /* The magic pass: "Extended precision modular arithmetic"
     x = ((x - y * DP1) - y * DP2) - y * DP3; */
  xmm1 = _mm256_mul_ps(y, _mm256_set1_ps(c_minus_cephes_DP1));
  xmm2 = _mm256_mul_ps(y, _mm256_set1_ps(c_minus_cephes_DP2));
  xmm3 = _mm256_mul_ps(y, _mm256_set1_ps(c_minus_cephes_DP3));
  x = _mm256_add_ps(x, xmm1);
  x = _mm256_add_ps(x, xmm2);
  x = _mm256_add_ps(x, xmm3);

  sign_mask_sin = _mm256_xor_si256(
    sign_mask_sin, _mm256_tst_epi32(emm2, _mm256_set1_epi32(4)));
  sign_mask_cos = _mm256_tst_epi32(_mm256_sub_epi32(emm2, _mm256_set1_epi32(2)),
                                   _mm256_set1_epi32(4));

  /* Evaluate the first polynom  (0 <= x <= Pi/4) in y1,
     and the second polynom      (Pi/4 <= x <= 0) in y2 */
  __m256 z = _mm256_mul_ps(x, x);
  __m256 y1, y2;

  y1 = _mm256_fmadd_ps(_mm256_set1_ps(c_coscof_p0), z,
                       _mm256_set1_ps(c_coscof_p1));
  y2 = _mm256_fmadd_ps(_mm256_set1_ps(c_sincof_p0), z,
                       _mm256_set1_ps(c_sincof_p1));
  y1 = _mm256_fmadd_ps(y1, z, _mm256_set1_ps(c_coscof_p2));
  y2 = _mm256_fmadd_ps(y2, z, _mm256_set1_ps(c_sincof_p2));
  y1 = _mm256_mul_ps(y1, z);
  y2 = _mm256_mul_ps(y2, z);
  y1 = _mm256_mul_ps(y1, z);
  y2 = _mm256_mul_ps(y2, x);
  y1 = _mm256_sub_ps(y1, _mm256_mul_ps(z, _mm256_set1_ps(0.5f)));
  y2 = _mm256_add_ps(y2, x);
  y1 = _mm256_add_ps(y1, _mm256_set1_ps(1));

  /* select the correct result from the two polynoms */
  __m256 pm = _mm256_castsi256_ps(poly_mask);
  __m256 ys = _mm256_blendv_ps(y2, y1, pm);
  __m256 yc = _mm256_blendv_ps(y1, y2, pm);

  /* apply sign */
  __m256 sign_bit = _mm256_set1_ps(-0.0f);
  __m256 neg_ys = _mm256_xor_ps(ys, sign_bit);
  __m256 neg_yc = _mm256_xor_ps(yc, sign_bit);
  *ysin = _mm256_blendv_ps(ys, neg_ys, _mm256_castsi256_ps(sign_mask_sin));
  *ycos = _mm256_blendv_ps(neg_yc, yc, _mm256_castsi256_ps(sign_mask_cos));
}

/**
 * @brief sin function with AVX2 SIMD
 *
 * @param x input register variable
 * @return __m256 result of sin(x)
 */
inline __m256 sin256_ps(__m256 x) {
  __m256 ysin, ycos;
  sincos256_ps(x, &ysin, &ycos);
  return ysin;
}

/**
 * @brief cos function with AVX2 SIMD
 *
 * @param x input register variable
 * @return __m256 result of cos(x)
 */
inline __m256 cos256_ps(__m256 x) {
  __m256 ysin, ycos;
  sincos256_ps(x, &ysin, &ycos);
  return ycos;
}
