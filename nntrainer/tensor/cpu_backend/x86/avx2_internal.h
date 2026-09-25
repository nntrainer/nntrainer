// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Donghyeon Jeong <dhyeon.jeong@samsung.com>
 *
 * @file   avx2_internal.h
 * @date   19 Jul 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @author YongHyeon02 <dyddyd8574@snu.ac.kr>
 * @bug    No known bugs except for NYI items
 * @brief  Shared AVX2 SIMD helper primitives extracted from avx2_impl.cpp,
 *         reused by the x86 FP16 kernels
 */

#ifndef __AVX2_INTERNAL_H_
#define __AVX2_INTERNAL_H_
#ifdef __cplusplus

#include <array>
#if __has_include(<bit>)
#include <bit>
#endif
#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <limits>
#if __has_include(<numbers>)
#include <numbers>
#endif
#include <type_traits>
#if __has_include(<version>)
#include <version>
#endif

#if !defined(__has_constexpr_builtin)
#define __has_constexpr_builtin(x) (0)
#define _nnt_UNDEF_HAS_CONSTEXPR_BUILTIN
#endif

#if !defined(__has_cpp_attribute)
#define __has_cpp_attribute(x) (0)
#define _nnt_UNDEF_HAS_CPP_ATTRIBUTE
#endif

// VECTORCALL calling-conv (default for x86_64-linux-gnu)
#if _MSC_VER >= 1700
#define _nnt_CC_VECTORCALL __vectorcall
#else
#define _nnt_CC_VECTORCALL
#endif

// Flatten attribute
#if _MSC_VER >= 1700 || __has_cpp_attribute(msvc::flatten)
#define _nnt_ATTR_FLATTEN [[msvc::flatten]]
#elif __has_cpp_attribute(gnu::flatten)
#define _nnt_ATTR_FLATTEN [[gnu::flatten]]
#else
#define _nnt_ATTR_FLATTEN
#endif

#if _MSC_VER >= 1700 || __has_cpp_attribute(msvc::forceinline)
#define _nnt_ATTR_ALWAYS_INLINE [[msvc::forceinline]]
#elif __has_cpp_attribute(gnu::always_inline)
#define _nnt_ATTR_ALWAYS_INLINE [[gnu::always_inline]]
#else
#define _nnt_ATTR_ALWAYS_INLINE
#endif

namespace nntrainer::avx2::internal {

// --- popcount / power-of-two helpers ---

[[nodiscard]] constexpr inline unsigned
constexpr_popcount(uint32_t v) noexcept {
#if __cpp_lib_bitops >= 201907L
  return std::popcount(v);
#else
  v = v - ((v >> 1) & 0x55555555);
  v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
  auto c = (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
  return c;
#endif
}

template <unsigned I_>
constexpr inline bool concept17_PowerOfTwo = (constexpr_popcount(I_) == 1);

namespace numbers {
#if __has_include(<numbers>) && __cpp_lib_math_constants >= 201907L
using std::numbers::ln2_v;
using std::numbers::log2e_v;
#else
template <typename Float_> constexpr inline auto ln2_v = Float_{M_LN2};
template <typename Float_> constexpr inline auto log2e_v = Float_{M_LOG2E};
#endif
} // namespace numbers

// --- Exp constants ---

constexpr inline float EXP_ARG_MIN = -87.0;
constexpr inline float EXP_ARG_MAX = +88.3762626647949f;

// --- Exp2 lookup table ---

/** @brief Exp2 lookup table for fast vectorized exponential approximation */
template <unsigned N_, typename Ty_ = uint32_t, typename Float_ = float,
          typename = std::enable_if_t<concept17_PowerOfTwo<N_>>>
struct Exp2Table {
  constexpr static inline auto MANTISSA_BITS =
    std::numeric_limits<Float_>::digits - 1;

#if __cpp_consteval >= 201811L && __has_constexpr_builtin(__builtin_exp2)
  [[nodiscard]] static consteval auto calculate() noexcept {
    std::array<Ty_, N_> t;
    for (unsigned i = 0; i < N_; ++i)
      t[i] = std::bit_cast<Ty_>(std::exp2(Float_{1.0} * i / N_)) -
             ((i << MANTISSA_BITS) / N_);
    return t;
  }
#endif
};

#if !__has_constexpr_builtin(__builtin_exp2) || !(__cpp_consteval >= 201811L)
/** @brief Exp2 lookup table explicit specialization for 8-entry float table on
 * platforms lacking consteval */
template <> struct Exp2Table<8, uint32_t, float> {
  [[nodiscard]] static constexpr auto calculate() noexcept {
    std::array<uint32_t, 8> t = {0x3f800000U, 0x3f7b95c2U, 0x3f7837f0U,
                                 0x3f75fed7U, 0x3f7504f3U, 0x3f75672aU,
                                 0x3f7744fdU, 0x3f7ac0c7U};
    return t;
  }
};
#endif

template <unsigned N_>
alignas(__m256) inline constexpr auto exp2_table_v = Exp2Table<N_>::calculate();

// --- Vectorized exp (lookup-based, used by swiglu) ---

template <unsigned N_, typename = std::enable_if_t<concept17_PowerOfTwo<N_>>>
_nnt_ATTR_ALWAYS_INLINE _nnt_ATTR_FLATTEN inline auto _nnt_CC_VECTORCALL
avx2_approx_exp_e2lookup(__m256 xs) noexcept {
  constexpr static uint32_t N_MASK = uint32_t(N_ - 1U);
  alignas(64) constexpr static auto EXP2_TBL = exp2_table_v<N_>;
  constexpr static unsigned MANTISSA_BITS =
    std::numeric_limits<float>::digits - 1;

  xs = _mm256_max_ps(xs, _mm256_set1_ps(EXP_ARG_MIN));
  xs =
    _mm256_mul_ps(xs, _mm256_set1_ps(float(1.0 / numbers::ln2_v<double> * N_)));
  xs = _mm256_min_ps(
    xs, _mm256_set1_ps(float(EXP_ARG_MAX / numbers::ln2_v<double> * N_)));

  auto xs_int = _mm256_add_ps(xs, _mm256_set1_ps(0x1.8p23f));
  auto xs_int_as_u32 = _mm256_castps_si256(xs_int);
  xs_int = _mm256_sub_ps(xs_int, _mm256_set1_ps(0x1.8p23f));

  auto xs_frac = _mm256_sub_ps(xs, xs_int);
  auto exp2_idxs = _mm256_and_si256(xs_int_as_u32, _mm256_set1_epi32(N_MASK));

  __m256i s_ints;
  if constexpr (N_ == 8) {
    auto tbl = _mm256_load_si256((__m256i *)EXP2_TBL.data());
    s_ints = _mm256_permutevar8x32_epi32(tbl, exp2_idxs);
  } else {
    s_ints = _mm256_i32gather_epi32(EXP2_TBL.data(), exp2_idxs, 1);
  }

  auto xs_uint_shifted = _mm256_slli_epi32(
    xs_int_as_u32, MANTISSA_BITS - constexpr_popcount(N_MASK));
  auto s_ints_2 = _mm256_add_epi32(s_ints, xs_uint_shifted);
  auto s_floats = _mm256_castsi256_ps(s_ints_2);

  static constexpr float poly_d4[] = {0x1.c6af84b912394p-5f / N_ / N_ / N_,
                                      0x1.ebfce50fac4f3p-3f / N_ / N_,
                                      0x1.62e42ff0c52d6p-1f / N_};

  const auto C0 = _mm256_set1_ps(poly_d4[0]);
  const auto C1 = _mm256_set1_ps(poly_d4[1]);
  const auto C2 = _mm256_set1_ps(poly_d4[2]);

  auto qs0 = _mm256_fmadd_ps(xs_frac, C0, C1);
  auto xs_frac_pow2 = _mm256_mul_ps(xs_frac, xs_frac);
  auto qs2 = _mm256_mul_ps(xs_frac, C2);

  xs = _mm256_fmadd_ps(qs0, xs_frac_pow2, qs2);

  return _mm256_fmadd_ps(xs, s_floats, s_floats);
}

// --- Negate ---

_nnt_ATTR_ALWAYS_INLINE _nnt_ATTR_FLATTEN inline auto _nnt_CC_VECTORCALL
avx2_negate_ps(__m256 x) noexcept -> __m256 {
  constexpr auto SIGN_SHIFT = sizeof(float) * 8 - 1;
  const auto UNDEF = _mm256_undefined_si256();
  const auto sign_bit =
    _mm256_slli_epi32(_mm256_cmpeq_epi16(UNDEF, UNDEF), SIGN_SHIFT);
  auto flt_sign_bit = _mm256_castsi256_ps(sign_bit);
  auto neg_x = _mm256_xor_ps(x, flt_sign_bit);
  return neg_x;
}

// --- SwiGLU helpers ---

_nnt_ATTR_ALWAYS_INLINE _nnt_ATTR_FLATTEN inline auto _nnt_CC_VECTORCALL
avx2_approx_swiglu(__m256 x, __m256 s) noexcept -> __m256 {
  auto neg_x = avx2_negate_ps(x);
  auto inv_sigmoid =
    _mm256_add_ps(avx2_approx_exp_e2lookup<8>(neg_x), _mm256_set1_ps(1.0f));
  auto swiglu_nonscaled = _mm256_div_ps(x, inv_sigmoid);
  return _mm256_mul_ps(swiglu_nonscaled, s);
}

_nnt_ATTR_ALWAYS_INLINE _nnt_ATTR_FLATTEN inline auto _nnt_CC_VECTORCALL
avx2_approx_swiglu_alpha(__m256 x, __m256 s, __m256 alpha) noexcept -> __m256 {
  auto alpha_x = _mm256_mul_ps(alpha, x);
  auto neg_alpha_x = avx2_negate_ps(alpha_x);
  auto inv_sigmoid = _mm256_add_ps(avx2_approx_exp_e2lookup<8>(neg_alpha_x),
                                   _mm256_set1_ps(1.0f));
  auto swiglu_nonscaled = _mm256_div_ps(x, inv_sigmoid);
  return _mm256_mul_ps(swiglu_nonscaled, s);
}

// --- exp256_ps (used by softmax) ---

inline __m256 exp256_ps(__m256 x) {
  const __m256 LOG2EF = _mm256_set1_ps(1.44269504088896341f);
  const __m256 LN2 = _mm256_set1_ps(0.6931471805599453f);

  const __m256 max_x = _mm256_set1_ps(88.3762626647949f);
  const __m256 min_x = _mm256_set1_ps(-88.3762626647949f);
  x = _mm256_max_ps(min_x, _mm256_min_ps(max_x, x));

  __m256 fx = _mm256_mul_ps(x, LOG2EF);
  fx = _mm256_floor_ps(_mm256_add_ps(fx, _mm256_set1_ps(0.5f)));

  __m256 tmp = _mm256_mul_ps(fx, LN2);
  __m256 r = _mm256_sub_ps(x, tmp);

  const __m256 c0 = _mm256_set1_ps(1.0f);
  const __m256 c1 = _mm256_set1_ps(1.0f);
  const __m256 c2 = _mm256_set1_ps(0.5f);
  const __m256 c3 = _mm256_set1_ps(1.0f / 6.0f);
  const __m256 c4 = _mm256_set1_ps(1.0f / 24.0f);
  const __m256 c5 = _mm256_set1_ps(1.0f / 120.0f);
  const __m256 c6 = _mm256_set1_ps(1.0f / 720.0f);
  const __m256 c7 = _mm256_set1_ps(1.0f / 5040.0f);
  const __m256 c8 = _mm256_set1_ps(1.0f / 40320.0f);
  const __m256 c9 = _mm256_set1_ps(1.0f / 362880.0f);
  const __m256 c10 = _mm256_set1_ps(1.0f / 3628800.0f);

  __m256 y = c10;
  y = _mm256_fmadd_ps(y, r, c9);
  y = _mm256_fmadd_ps(y, r, c8);
  y = _mm256_fmadd_ps(y, r, c7);
  y = _mm256_fmadd_ps(y, r, c6);
  y = _mm256_fmadd_ps(y, r, c5);
  y = _mm256_fmadd_ps(y, r, c4);
  y = _mm256_fmadd_ps(y, r, c3);
  y = _mm256_fmadd_ps(y, r, c2);
  y = _mm256_fmadd_ps(y, r, c1);
  y = _mm256_fmadd_ps(y, r, c0);

  __m256i emm0 = _mm256_cvtps_epi32(fx);
  emm0 = _mm256_add_epi32(emm0, _mm256_set1_epi32(127));
  emm0 = _mm256_slli_epi32(emm0, 23);
  __m256 pow2n = _mm256_castsi256_ps(emm0);

  return _mm256_mul_ps(y, pow2n);
}

// --- rcp_ps (Newton-Raphson reciprocal) ---

inline __m256 rcp_ps(__m256 x) {
  __m256 rcp = _mm256_rcp_ps(x);
  __m256 two = _mm256_set1_ps(2.0f);
  return _mm256_mul_ps(rcp, _mm256_fnmadd_ps(x, rcp, two));
}

// --- hsum_avx (horizontal sum) ---

inline float hsum_avx(__m256 v) {
  __m128 vlow = _mm256_castps256_ps128(v);
  __m128 vhigh = _mm256_extractf128_ps(v, 1);
  vlow = _mm_add_ps(vlow, vhigh);
  __m128 shuf = _mm_movehdup_ps(vlow);
  __m128 sums = _mm_add_ps(vlow, shuf);
  shuf = _mm_movehl_ps(shuf, sums);
  sums = _mm_add_ss(sums, shuf);
  return _mm_cvtss_f32(sums);
}

// --- GELU polynomial approximation (tanh variant) ---

constexpr inline float gelu_start_tanh = -4.38086284326899f;
constexpr inline float gelu_end_tanh = 4.38086284326899f;

constexpr inline float tanh_c_gelu_p0 = 5.91303808e-6f;
constexpr inline float tanh_c_gelu_p1 = 5.00000000e-1f;
constexpr inline float tanh_c_gelu_p2 = 3.98865869e-1f;
constexpr inline float tanh_c_gelu_p4 = -6.66574676e-2f;
constexpr inline float tanh_c_gelu_p6 = 1.00712610e-2f;
constexpr inline float tanh_c_gelu_p8 = -1.19336340e-3f;
constexpr inline float tanh_c_gelu_p10 = 1.09543224e-4f;
constexpr inline float tanh_c_gelu_p12 = -7.55788500e-6f;
constexpr inline float tanh_c_gelu_p14 = 3.73374142e-7f;
constexpr inline float tanh_c_gelu_p16 = -1.23162678e-8f;
constexpr inline float tanh_c_gelu_p18 = 2.40940960e-10f;
constexpr inline float tanh_c_gelu_p20 = -2.10237709e-12f;

inline __m256 poly_gelu_tanh_avx2(__m256 x) {
  const __m256 x2 = _mm256_mul_ps(x, x);

  __m256 y = _mm256_mul_ps(x2, _mm256_set1_ps(tanh_c_gelu_p20));
  y = _mm256_add_ps(y, _mm256_set1_ps(tanh_c_gelu_p18));
  y = _mm256_mul_ps(x2, y);
  y = _mm256_add_ps(y, _mm256_set1_ps(tanh_c_gelu_p16));
  y = _mm256_mul_ps(x2, y);
  y = _mm256_add_ps(y, _mm256_set1_ps(tanh_c_gelu_p14));
  y = _mm256_mul_ps(x2, y);
  y = _mm256_add_ps(y, _mm256_set1_ps(tanh_c_gelu_p12));
  y = _mm256_mul_ps(x2, y);
  y = _mm256_add_ps(y, _mm256_set1_ps(tanh_c_gelu_p10));
  y = _mm256_mul_ps(x2, y);
  y = _mm256_add_ps(y, _mm256_set1_ps(tanh_c_gelu_p8));
  y = _mm256_mul_ps(x2, y);
  y = _mm256_add_ps(y, _mm256_set1_ps(tanh_c_gelu_p6));
  y = _mm256_mul_ps(x2, y);
  y = _mm256_add_ps(y, _mm256_set1_ps(tanh_c_gelu_p4));
  y = _mm256_mul_ps(x2, y);
  y = _mm256_add_ps(y, _mm256_set1_ps(tanh_c_gelu_p2));
  y = _mm256_mul_ps(x2, y);

  __m256 z = _mm256_mul_ps(x, _mm256_set1_ps(tanh_c_gelu_p1));
  z = _mm256_add_ps(z, _mm256_set1_ps(tanh_c_gelu_p0));

  y = _mm256_add_ps(y, z);

  const __m256 gt_start =
    _mm256_cmp_ps(x, _mm256_set1_ps(gelu_start_tanh), _CMP_GT_OQ);
  const __m256 le_end =
    _mm256_cmp_ps(x, _mm256_set1_ps(gelu_end_tanh), _CMP_LE_OQ);
  const __m256 gt_end =
    _mm256_cmp_ps(x, _mm256_set1_ps(gelu_end_tanh), _CMP_GT_OQ);

  y = _mm256_and_ps(y, gt_start);
  y = _mm256_and_ps(y, le_end);
  __m256 x_hi = _mm256_and_ps(x, gt_end);

  return _mm256_add_ps(y, x_hi);
}

// --- GELU polynomial approximation (erf variant) ---

constexpr inline float gelu_start_erf = -4.59373833108583f;
constexpr inline float gelu_end_erf = 4.59373833108583f;

constexpr inline float erf_c_gelu_p0 = 8.70757509e-06f;
constexpr inline float erf_c_gelu_p1 = 5.00000000e-1f;
constexpr inline float erf_c_gelu_p2 = 3.98833088e-01f;
constexpr inline float erf_c_gelu_p4 = -6.62633808e-02f;
constexpr inline float erf_c_gelu_p6 = 9.78776282e-03f;
constexpr inline float erf_c_gelu_p8 = -1.10798998e-03f;
constexpr inline float erf_c_gelu_p10 = 9.51056006e-05f;
constexpr inline float erf_c_gelu_p12 = -6.04633051e-06f;
constexpr inline float erf_c_gelu_p14 = 2.73076070e-07f;
constexpr inline float erf_c_gelu_p16 = -8.20707325e-09f;
constexpr inline float erf_c_gelu_p18 = 1.46115955e-10f;
constexpr inline float erf_c_gelu_p20 = -1.16009840e-12f;

inline __m256 poly_gelu_erf_avx2(__m256 x) {
  const __m256 x2 = _mm256_mul_ps(x, x);

  __m256 y = _mm256_mul_ps(x2, _mm256_set1_ps(erf_c_gelu_p20));
  y = _mm256_add_ps(y, _mm256_set1_ps(erf_c_gelu_p18));
  y = _mm256_mul_ps(x2, y);
  y = _mm256_add_ps(y, _mm256_set1_ps(erf_c_gelu_p16));
  y = _mm256_mul_ps(x2, y);
  y = _mm256_add_ps(y, _mm256_set1_ps(erf_c_gelu_p14));
  y = _mm256_mul_ps(x2, y);
  y = _mm256_add_ps(y, _mm256_set1_ps(erf_c_gelu_p12));
  y = _mm256_mul_ps(x2, y);
  y = _mm256_add_ps(y, _mm256_set1_ps(erf_c_gelu_p10));
  y = _mm256_mul_ps(x2, y);
  y = _mm256_add_ps(y, _mm256_set1_ps(erf_c_gelu_p8));
  y = _mm256_mul_ps(x2, y);
  y = _mm256_add_ps(y, _mm256_set1_ps(erf_c_gelu_p6));
  y = _mm256_mul_ps(x2, y);
  y = _mm256_add_ps(y, _mm256_set1_ps(erf_c_gelu_p4));
  y = _mm256_mul_ps(x2, y);
  y = _mm256_add_ps(y, _mm256_set1_ps(erf_c_gelu_p2));
  y = _mm256_mul_ps(x2, y);

  __m256 z = _mm256_mul_ps(x, _mm256_set1_ps(erf_c_gelu_p1));
  z = _mm256_add_ps(z, _mm256_set1_ps(erf_c_gelu_p0));

  y = _mm256_add_ps(y, z);

  const __m256 gt_start =
    _mm256_cmp_ps(x, _mm256_set1_ps(gelu_start_erf), _CMP_GT_OQ);
  const __m256 le_end =
    _mm256_cmp_ps(x, _mm256_set1_ps(gelu_end_erf), _CMP_LE_OQ);
  const __m256 gt_end =
    _mm256_cmp_ps(x, _mm256_set1_ps(gelu_end_erf), _CMP_GT_OQ);

  y = _mm256_and_ps(y, gt_start);
  y = _mm256_and_ps(y, le_end);
  __m256 x_hi = _mm256_and_ps(x, gt_end);

  return _mm256_add_ps(y, x_hi);
}

} // namespace nntrainer::avx2::internal

// Scrub the file-local compiler shims so they do not leak into includers.
#undef _nnt_CC_VECTORCALL
#undef _nnt_ATTR_FLATTEN
#undef _nnt_ATTR_ALWAYS_INLINE
#if defined(_nnt_UNDEF_HAS_CONSTEXPR_BUILTIN)
#undef __has_constexpr_builtin
#undef _nnt_UNDEF_HAS_CONSTEXPR_BUILTIN
#endif
#if defined(_nnt_UNDEF_HAS_CPP_ATTRIBUTE)
#undef __has_cpp_attribute
#undef _nnt_UNDEF_HAS_CPP_ATTRIBUTE
#endif

#endif /* __cplusplus */
#endif /* __AVX2_INTERNAL_H_ */
