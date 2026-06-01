// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Donghyeon Jeong <dhyeon.jeong@samsung.com>
 *
 * @file   avx2_impl_fp16.cpp
 * @date   20 Feb 2024
 * @see    https://github.com/nntrainer/nntrainer
 * @author Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is a source for AVX implementation
 *
 */

#include "avx2_internal.h"
#include <avx2_impl.h>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <util_func.h>

using namespace nntrainer::avx2::internal;

namespace nntrainer::avx2 {

void vcvt_f16_f32(unsigned int N, const _Float16 *input, float *output) {
  assert(N != 0);
  assert(input != NULL);
  assert(output != NULL);

  unsigned int idx = 0;
  const _Float16 *data = (const _Float16 *)input;

  // 16 half-precision floating point values to single-precision values
  for (; N - idx >= 16; idx += 16) {
    const __m256 vec0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)data));
    const __m256 vec1 =
      _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(data + 8)));
    data += 16;

    _mm256_storeu_ps(output, vec0);
    _mm256_storeu_ps(output + 8, vec1);
    output += 16;
  }
  // 8 half-precision floating point values to single-precision values
  for (; N - idx >= 8; idx += 8) {
    const __m256 vec = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)data));
    data += 8;

    _mm256_storeu_ps(output, vec);
    output += 8;
  }
  // remaining half-precision floating point values to single-precision values
  while (idx < N) {
    *output = static_cast<float>(*data);
    ++output;
    ++data;
    ++idx;
  }
}

void vcvt_f32_f16(unsigned int N, const float *input, _Float16 *output) {
  assert(N != 0);
  assert(input != NULL);
  assert(output != NULL);

  unsigned int idx = 0;
  _Float16 *out_data = (_Float16 *)output;

  // 16 single-precision floating point values to half-precision values
  for (; N - idx >= 16; idx += 16) {
    const __m256 vec0 = _mm256_loadu_ps(input);
    const __m256 vec1 = _mm256_loadu_ps(input + 8);
    input += 16;

    _mm_storeu_si128((__m128i *)out_data,
                     _mm256_cvtps_ph(vec0, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i *)(out_data + 8),
                     _mm256_cvtps_ph(vec1, _MM_FROUND_TO_NEAREST_INT));
    out_data += 16;
  }
  // 8 single-precision floating point values to half-precision values
  for (; N - idx >= 8; idx += 8) {
    const __m256 vec = _mm256_loadu_ps(input);
    input += 8;

    _mm_storeu_si128((__m128i *)out_data,
                     _mm256_cvtps_ph(vec, _MM_FROUND_TO_NEAREST_INT));
    out_data += 8;
  }
  // 4 single-precision floating point values to half-precision values
  for (; N - idx >= 4; idx += 4) {
    const __m128 vec = _mm_loadu_ps(input);
    input += 4;

    _mm_storeu_si64((__m128i *)out_data,
                    _mm_cvtps_ph(vec, _MM_FROUND_TO_NEAREST_INT));
    out_data += 4;
  }
  // remaining single-precision floating point values to half-precision values
  while (idx < N) {
    *out_data = static_cast<_Float16>(*input);
    ++out_data;
    ++input;
    ++idx;
  }
}

bool is_valid(const unsigned int N, const _Float16 *input) {
  assert(N != 0);
  assert(input != NULL);

  int temp = 0;
  unsigned int idx = 0;

  const __m256 SIGN_MASK = _mm256_set1_ps(-0.0);
  const __m256 INF = _mm256_set1_ps(std::numeric_limits<float>::infinity());

  // 16 single-precision check : ( X != X )
  for (; N - idx >= 16; idx += 16) {
    __m256 vec0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)input));
    __m256 vec1 =
      _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(input + 8)));

    input += 16;

    // check NaN in vec0
    __m256 res = _mm256_cmp_ps(vec0, vec0, _CMP_NEQ_UQ);
    temp = temp | _mm256_movemask_ps(res);
    if (temp)
      return false;

    // check infinity in vec0
    vec0 = _mm256_andnot_ps(SIGN_MASK, vec0);
    vec0 = _mm256_cmp_ps(vec0, INF, _CMP_EQ_OQ);

    temp = temp | _mm256_movemask_ps(vec0);
    if (temp)
      return false;

    // check NaN in vec1
    __m256 res1 = _mm256_cmp_ps(vec1, vec1, _CMP_NEQ_UQ);
    temp = temp | _mm256_movemask_ps(res1);

    if (temp)
      return false;

    // check infinity in vec1
    vec1 = _mm256_andnot_ps(SIGN_MASK, vec1);
    vec1 = _mm256_cmp_ps(vec1, INF, _CMP_EQ_OQ);

    temp = temp | _mm256_movemask_ps(vec1);

    if (temp)
      return false;
  }

  // 8 single-precision check : ( X != X )
  for (; N - idx >= 8; idx += 8) {
    __m256 vec = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)input));
    input += 8;
    __m256 res = _mm256_cmp_ps(vec, vec, _CMP_NEQ_UQ);
    temp = temp | _mm256_movemask_ps(res);

    if (temp)
      return false;

    // check infinity in vec1
    vec = _mm256_andnot_ps(SIGN_MASK, vec);
    vec = _mm256_cmp_ps(vec, INF, _CMP_EQ_OQ);

    temp = temp | _mm256_movemask_ps(vec);

    if (temp)
      return false;
  }

  while (idx < N) {
    if (!isFloatValid(*input)) {
      return false;
    }
    ++input;
    ++idx;
  }

  return true;
}

// ============================================================
// FP16 elementwise operations
// ============================================================

void ele_mul(const unsigned int N, const _Float16 *X, const _Float16 *Y,
             _Float16 *Z, float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  if (alpha == 1.0f && beta == 0.0f && o_stride == 1) {
    unsigned int i = 0;
    if (i_stride == 0) {
      float y0_f32 = static_cast<float>(Y[0]);
      __m256 vy = _mm256_set1_ps(y0_f32);
      for (; i + 16 <= N; i += 16) {
        __m256i xd = _mm256_loadu_si256((const __m256i *)(X + i));
        __m256 x_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(xd));
        __m256 x_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(xd, 1));
        __m256 z_lo = _mm256_mul_ps(x_lo, vy);
        __m256 z_hi = _mm256_mul_ps(x_hi, vy);
        __m128i r_lo = _mm256_cvtps_ph(z_lo, _MM_FROUND_TO_NEAREST_INT);
        __m128i r_hi = _mm256_cvtps_ph(z_hi, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256(
          (__m256i *)(Z + i),
          _mm256_inserti128_si256(_mm256_castsi128_si256(r_lo), r_hi, 1));
      }
      for (; i + 8 <= N; i += 8) {
        __m256 x = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(X + i)));
        __m256 z = _mm256_mul_ps(x, vy);
        _mm_storeu_si128((__m128i *)(Z + i),
                         _mm256_cvtps_ph(z, _MM_FROUND_TO_NEAREST_INT));
      }
      for (; i < N; ++i) {
        Z[i] = static_cast<_Float16>(static_cast<float>(X[i]) * y0_f32);
      }
    } else if (i_stride == 1) {
      for (; i + 16 <= N; i += 16) {
        __m256i xd = _mm256_loadu_si256((const __m256i *)(X + i));
        __m256i yd = _mm256_loadu_si256((const __m256i *)(Y + i));
        __m256 x_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(xd));
        __m256 x_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(xd, 1));
        __m256 y_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(yd));
        __m256 y_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(yd, 1));
        __m256 z_lo = _mm256_mul_ps(x_lo, y_lo);
        __m256 z_hi = _mm256_mul_ps(x_hi, y_hi);
        __m128i r_lo = _mm256_cvtps_ph(z_lo, _MM_FROUND_TO_NEAREST_INT);
        __m128i r_hi = _mm256_cvtps_ph(z_hi, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256(
          (__m256i *)(Z + i),
          _mm256_inserti128_si256(_mm256_castsi128_si256(r_lo), r_hi, 1));
      }
      for (; i + 8 <= N; i += 8) {
        __m256 x = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(X + i)));
        __m256 y = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(Y + i)));
        __m256 z = _mm256_mul_ps(x, y);
        _mm_storeu_si128((__m128i *)(Z + i),
                         _mm256_cvtps_ph(z, _MM_FROUND_TO_NEAREST_INT));
      }
      for (; i < N; ++i) {
        Z[i] = static_cast<_Float16>(static_cast<float>(X[i]) *
                                     static_cast<float>(Y[i]));
      }
    } else {
      for (unsigned int i = 0; i < N; ++i) {
        Z[i] = static_cast<_Float16>(static_cast<float>(X[i]) *
                                     static_cast<float>(Y[i * i_stride]));
      }
    }
  } else if (o_stride == 1 && (i_stride == 0 || i_stride == 1)) {
    __m256 alpha_v = _mm256_set1_ps(alpha);
    __m256 beta_v = _mm256_set1_ps(beta);
    unsigned int i = 0;

    if (i_stride == 0) {
      __m256 vy = _mm256_set1_ps(static_cast<float>(Y[0]));
      for (; i + 16 <= N; i += 16) {
        __m256i xd = _mm256_loadu_si256((const __m256i *)(X + i));
        __m256 x_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(xd));
        __m256 x_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(xd, 1));
        __m256 z_lo = _mm256_mul_ps(_mm256_mul_ps(x_lo, vy), alpha_v);
        __m256 z_hi = _mm256_mul_ps(_mm256_mul_ps(x_hi, vy), alpha_v);
        if (beta != 0.0f) {
          __m256i zd = _mm256_loadu_si256((const __m256i *)(Z + i));
          __m256 zo_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(zd));
          __m256 zo_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(zd, 1));
          z_lo = _mm256_fmadd_ps(beta_v, zo_lo, z_lo);
          z_hi = _mm256_fmadd_ps(beta_v, zo_hi, z_hi);
        }
        __m128i r_lo = _mm256_cvtps_ph(z_lo, _MM_FROUND_TO_NEAREST_INT);
        __m128i r_hi = _mm256_cvtps_ph(z_hi, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256(
          (__m256i *)(Z + i),
          _mm256_inserti128_si256(_mm256_castsi128_si256(r_lo), r_hi, 1));
      }
      for (; i + 8 <= N; i += 8) {
        __m256 x = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(X + i)));
        __m256 z = _mm256_mul_ps(_mm256_mul_ps(x, vy), alpha_v);
        if (beta != 0.0f) {
          __m256 z_old =
            _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(Z + i)));
          z = _mm256_fmadd_ps(beta_v, z_old, z);
        }
        _mm_storeu_si128((__m128i *)(Z + i),
                         _mm256_cvtps_ph(z, _MM_FROUND_TO_NEAREST_INT));
      }
    } else {
      for (; i + 16 <= N; i += 16) {
        __m256i xd = _mm256_loadu_si256((const __m256i *)(X + i));
        __m256i yd = _mm256_loadu_si256((const __m256i *)(Y + i));
        __m256 x_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(xd));
        __m256 x_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(xd, 1));
        __m256 y_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(yd));
        __m256 y_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(yd, 1));
        __m256 z_lo = _mm256_mul_ps(_mm256_mul_ps(x_lo, y_lo), alpha_v);
        __m256 z_hi = _mm256_mul_ps(_mm256_mul_ps(x_hi, y_hi), alpha_v);
        if (beta != 0.0f) {
          __m256i zd = _mm256_loadu_si256((const __m256i *)(Z + i));
          __m256 zo_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(zd));
          __m256 zo_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(zd, 1));
          z_lo = _mm256_fmadd_ps(beta_v, zo_lo, z_lo);
          z_hi = _mm256_fmadd_ps(beta_v, zo_hi, z_hi);
        }
        __m128i r_lo = _mm256_cvtps_ph(z_lo, _MM_FROUND_TO_NEAREST_INT);
        __m128i r_hi = _mm256_cvtps_ph(z_hi, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256(
          (__m256i *)(Z + i),
          _mm256_inserti128_si256(_mm256_castsi128_si256(r_lo), r_hi, 1));
      }
      for (; i + 8 <= N; i += 8) {
        __m256 x = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(X + i)));
        __m256 y = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(Y + i)));
        __m256 z = _mm256_mul_ps(_mm256_mul_ps(x, y), alpha_v);
        if (beta != 0.0f) {
          __m256 z_old =
            _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(Z + i)));
          z = _mm256_fmadd_ps(beta_v, z_old, z);
        }
        _mm_storeu_si128((__m128i *)(Z + i),
                         _mm256_cvtps_ph(z, _MM_FROUND_TO_NEAREST_INT));
      }
    }
    for (; i < N; ++i) {
      float xf = static_cast<float>(X[i]);
      float yf = static_cast<float>(Y[i * i_stride]);
      float zf = xf * alpha * yf +
                 ((0.0f == beta) ? 0.0f : beta * static_cast<float>(Z[i]));
      Z[i] = static_cast<_Float16>(zf);
    }
  } else {
    for (unsigned int i = 0; i < N; ++i) {
      float xf = static_cast<float>(*X);
      float yf = static_cast<float>(*Y);
      float zf = xf * alpha * yf +
                 ((0.0f == beta) ? 0.0f : beta * static_cast<float>(*Z));
      *Z = static_cast<_Float16>(zf);
      X += o_stride;
      Y += i_stride;
      Z += o_stride;
    }
  }
}

void ele_add(const unsigned int N, const _Float16 *X, const _Float16 *Y,
             _Float16 *Z, float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  if (alpha == 1.0f && beta == 0.0f && o_stride == 1) {
    unsigned int i = 0;
    if (i_stride == 0) {
      __m256 vy = _mm256_set1_ps(static_cast<float>(Y[0]));
      for (; i + 16 <= N; i += 16) {
        __m256i xd = _mm256_loadu_si256((const __m256i *)(X + i));
        __m256 x_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(xd));
        __m256 x_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(xd, 1));
        __m256 z_lo = _mm256_add_ps(x_lo, vy);
        __m256 z_hi = _mm256_add_ps(x_hi, vy);
        __m128i r_lo = _mm256_cvtps_ph(z_lo, _MM_FROUND_TO_NEAREST_INT);
        __m128i r_hi = _mm256_cvtps_ph(z_hi, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256(
          (__m256i *)(Z + i),
          _mm256_inserti128_si256(_mm256_castsi128_si256(r_lo), r_hi, 1));
      }
      for (; i + 8 <= N; i += 8) {
        __m256 x = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(X + i)));
        __m256 z = _mm256_add_ps(x, vy);
        _mm_storeu_si128((__m128i *)(Z + i),
                         _mm256_cvtps_ph(z, _MM_FROUND_TO_NEAREST_INT));
      }
      for (; i < N; ++i) {
        Z[i] = static_cast<_Float16>(static_cast<float>(X[i]) +
                                     static_cast<float>(Y[0]));
      }
    } else if (i_stride == 1) {
      for (; i + 16 <= N; i += 16) {
        __m256i xd = _mm256_loadu_si256((const __m256i *)(X + i));
        __m256i yd = _mm256_loadu_si256((const __m256i *)(Y + i));
        __m256 x_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(xd));
        __m256 x_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(xd, 1));
        __m256 y_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(yd));
        __m256 y_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(yd, 1));
        __m256 z_lo = _mm256_add_ps(x_lo, y_lo);
        __m256 z_hi = _mm256_add_ps(x_hi, y_hi);
        __m128i r_lo = _mm256_cvtps_ph(z_lo, _MM_FROUND_TO_NEAREST_INT);
        __m128i r_hi = _mm256_cvtps_ph(z_hi, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256(
          (__m256i *)(Z + i),
          _mm256_inserti128_si256(_mm256_castsi128_si256(r_lo), r_hi, 1));
      }
      for (; i + 8 <= N; i += 8) {
        __m256 x = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(X + i)));
        __m256 y = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(Y + i)));
        __m256 z = _mm256_add_ps(x, y);
        _mm_storeu_si128((__m128i *)(Z + i),
                         _mm256_cvtps_ph(z, _MM_FROUND_TO_NEAREST_INT));
      }
      for (; i < N; ++i) {
        Z[i] = static_cast<_Float16>(static_cast<float>(X[i]) +
                                     static_cast<float>(Y[i]));
      }
    } else {
      for (unsigned int i = 0; i < N; ++i) {
        Z[i] = static_cast<_Float16>(static_cast<float>(X[i]) +
                                     static_cast<float>(Y[i * i_stride]));
      }
    }
  } else if (o_stride == 1 && (i_stride == 0 || i_stride == 1)) {
    __m256 alpha_v = _mm256_set1_ps(alpha);
    __m256 beta_v = _mm256_set1_ps(beta);
    unsigned int i = 0;

    if (i_stride == 0) {
      __m256 vy = _mm256_set1_ps(static_cast<float>(Y[0]));
      for (; i + 16 <= N; i += 16) {
        __m256i xd = _mm256_loadu_si256((const __m256i *)(X + i));
        __m256 x_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(xd));
        __m256 x_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(xd, 1));
        __m256 z_lo = _mm256_fmadd_ps(alpha_v, vy, x_lo);
        __m256 z_hi = _mm256_fmadd_ps(alpha_v, vy, x_hi);
        if (beta != 0.0f) {
          __m256i zd = _mm256_loadu_si256((const __m256i *)(Z + i));
          __m256 zo_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(zd));
          __m256 zo_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(zd, 1));
          z_lo = _mm256_fmadd_ps(beta_v, zo_lo, z_lo);
          z_hi = _mm256_fmadd_ps(beta_v, zo_hi, z_hi);
        }
        __m128i r_lo = _mm256_cvtps_ph(z_lo, _MM_FROUND_TO_NEAREST_INT);
        __m128i r_hi = _mm256_cvtps_ph(z_hi, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256(
          (__m256i *)(Z + i),
          _mm256_inserti128_si256(_mm256_castsi128_si256(r_lo), r_hi, 1));
      }
      for (; i + 8 <= N; i += 8) {
        __m256 x = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(X + i)));
        __m256 z = _mm256_fmadd_ps(alpha_v, vy, x);
        if (beta != 0.0f) {
          __m256 z_old =
            _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(Z + i)));
          z = _mm256_fmadd_ps(beta_v, z_old, z);
        }
        _mm_storeu_si128((__m128i *)(Z + i),
                         _mm256_cvtps_ph(z, _MM_FROUND_TO_NEAREST_INT));
      }
    } else {
      for (; i + 16 <= N; i += 16) {
        __m256i xd = _mm256_loadu_si256((const __m256i *)(X + i));
        __m256i yd = _mm256_loadu_si256((const __m256i *)(Y + i));
        __m256 x_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(xd));
        __m256 x_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(xd, 1));
        __m256 y_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(yd));
        __m256 y_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(yd, 1));
        __m256 z_lo = _mm256_fmadd_ps(alpha_v, y_lo, x_lo);
        __m256 z_hi = _mm256_fmadd_ps(alpha_v, y_hi, x_hi);
        if (beta != 0.0f) {
          __m256i zd = _mm256_loadu_si256((const __m256i *)(Z + i));
          __m256 zo_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(zd));
          __m256 zo_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(zd, 1));
          z_lo = _mm256_fmadd_ps(beta_v, zo_lo, z_lo);
          z_hi = _mm256_fmadd_ps(beta_v, zo_hi, z_hi);
        }
        __m128i r_lo = _mm256_cvtps_ph(z_lo, _MM_FROUND_TO_NEAREST_INT);
        __m128i r_hi = _mm256_cvtps_ph(z_hi, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256(
          (__m256i *)(Z + i),
          _mm256_inserti128_si256(_mm256_castsi128_si256(r_lo), r_hi, 1));
      }
      for (; i + 8 <= N; i += 8) {
        __m256 x = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(X + i)));
        __m256 y = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(Y + i)));
        __m256 z = _mm256_fmadd_ps(alpha_v, y, x);
        if (beta != 0.0f) {
          __m256 z_old =
            _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(Z + i)));
          z = _mm256_fmadd_ps(beta_v, z_old, z);
        }
        _mm_storeu_si128((__m128i *)(Z + i),
                         _mm256_cvtps_ph(z, _MM_FROUND_TO_NEAREST_INT));
      }
    }
    for (; i < N; ++i) {
      float xf = static_cast<float>(X[i]);
      float yf = static_cast<float>(Y[i * i_stride]);
      float zf = xf + alpha * yf +
                 ((0.0f == beta) ? 0.0f : beta * static_cast<float>(Z[i]));
      Z[i] = static_cast<_Float16>(zf);
    }
  } else {
    for (unsigned int i = 0; i < N; ++i) {
      float xf = static_cast<float>(*X);
      float yf = static_cast<float>(*Y);
      float zf = xf + alpha * yf +
                 ((0.0f == beta) ? 0.0f : beta * static_cast<float>(*Z));
      *Z = static_cast<_Float16>(zf);
      X += o_stride;
      Y += i_stride;
      Z += o_stride;
    }
  }
}

void ele_sub(const unsigned int N, const _Float16 *X, const _Float16 *Y,
             _Float16 *Z, float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  if (alpha == 1.0f && beta == 0.0f && o_stride == 1) {
    unsigned int i = 0;
    if (i_stride == 0) {
      __m256 vy = _mm256_set1_ps(static_cast<float>(Y[0]));
      for (; i + 16 <= N; i += 16) {
        __m256i xd = _mm256_loadu_si256((const __m256i *)(X + i));
        __m256 x_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(xd));
        __m256 x_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(xd, 1));
        __m256 z_lo = _mm256_sub_ps(x_lo, vy);
        __m256 z_hi = _mm256_sub_ps(x_hi, vy);
        __m128i r_lo = _mm256_cvtps_ph(z_lo, _MM_FROUND_TO_NEAREST_INT);
        __m128i r_hi = _mm256_cvtps_ph(z_hi, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256(
          (__m256i *)(Z + i),
          _mm256_inserti128_si256(_mm256_castsi128_si256(r_lo), r_hi, 1));
      }
      for (; i + 8 <= N; i += 8) {
        __m256 x = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(X + i)));
        __m256 z = _mm256_sub_ps(x, vy);
        _mm_storeu_si128((__m128i *)(Z + i),
                         _mm256_cvtps_ph(z, _MM_FROUND_TO_NEAREST_INT));
      }
      for (; i < N; ++i) {
        Z[i] = static_cast<_Float16>(static_cast<float>(X[i]) -
                                     static_cast<float>(Y[0]));
      }
    } else if (i_stride == 1) {
      for (; i + 16 <= N; i += 16) {
        __m256i xd = _mm256_loadu_si256((const __m256i *)(X + i));
        __m256i yd = _mm256_loadu_si256((const __m256i *)(Y + i));
        __m256 x_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(xd));
        __m256 x_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(xd, 1));
        __m256 y_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(yd));
        __m256 y_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(yd, 1));
        __m256 z_lo = _mm256_sub_ps(x_lo, y_lo);
        __m256 z_hi = _mm256_sub_ps(x_hi, y_hi);
        __m128i r_lo = _mm256_cvtps_ph(z_lo, _MM_FROUND_TO_NEAREST_INT);
        __m128i r_hi = _mm256_cvtps_ph(z_hi, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256(
          (__m256i *)(Z + i),
          _mm256_inserti128_si256(_mm256_castsi128_si256(r_lo), r_hi, 1));
      }
      for (; i + 8 <= N; i += 8) {
        __m256 x = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(X + i)));
        __m256 y = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(Y + i)));
        __m256 z = _mm256_sub_ps(x, y);
        _mm_storeu_si128((__m128i *)(Z + i),
                         _mm256_cvtps_ph(z, _MM_FROUND_TO_NEAREST_INT));
      }
      for (; i < N; ++i) {
        Z[i] = static_cast<_Float16>(static_cast<float>(X[i]) -
                                     static_cast<float>(Y[i]));
      }
    } else {
      for (unsigned int i = 0; i < N; ++i) {
        Z[i] = static_cast<_Float16>(static_cast<float>(X[i]) -
                                     static_cast<float>(Y[i * i_stride]));
      }
    }
  } else if (o_stride == 1 && (i_stride == 0 || i_stride == 1)) {
    __m256 alpha_v = _mm256_set1_ps(alpha);
    __m256 beta_v = _mm256_set1_ps(beta);
    unsigned int i = 0;

    if (i_stride == 0) {
      __m256 vy = _mm256_set1_ps(static_cast<float>(Y[0]));
      for (; i + 16 <= N; i += 16) {
        __m256i xd = _mm256_loadu_si256((const __m256i *)(X + i));
        __m256 x_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(xd));
        __m256 x_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(xd, 1));
        __m256 z_lo = _mm256_fnmadd_ps(alpha_v, vy, x_lo);
        __m256 z_hi = _mm256_fnmadd_ps(alpha_v, vy, x_hi);
        if (beta != 0.0f) {
          __m256i zd = _mm256_loadu_si256((const __m256i *)(Z + i));
          __m256 zo_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(zd));
          __m256 zo_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(zd, 1));
          z_lo = _mm256_fmadd_ps(beta_v, zo_lo, z_lo);
          z_hi = _mm256_fmadd_ps(beta_v, zo_hi, z_hi);
        }
        __m128i r_lo = _mm256_cvtps_ph(z_lo, _MM_FROUND_TO_NEAREST_INT);
        __m128i r_hi = _mm256_cvtps_ph(z_hi, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256(
          (__m256i *)(Z + i),
          _mm256_inserti128_si256(_mm256_castsi128_si256(r_lo), r_hi, 1));
      }
      for (; i + 8 <= N; i += 8) {
        __m256 x = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(X + i)));
        __m256 z = _mm256_fnmadd_ps(alpha_v, vy, x);
        if (beta != 0.0f) {
          __m256 z_old =
            _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(Z + i)));
          z = _mm256_fmadd_ps(beta_v, z_old, z);
        }
        _mm_storeu_si128((__m128i *)(Z + i),
                         _mm256_cvtps_ph(z, _MM_FROUND_TO_NEAREST_INT));
      }
    } else {
      for (; i + 16 <= N; i += 16) {
        __m256i xd = _mm256_loadu_si256((const __m256i *)(X + i));
        __m256i yd = _mm256_loadu_si256((const __m256i *)(Y + i));
        __m256 x_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(xd));
        __m256 x_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(xd, 1));
        __m256 y_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(yd));
        __m256 y_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(yd, 1));
        __m256 z_lo = _mm256_fnmadd_ps(alpha_v, y_lo, x_lo);
        __m256 z_hi = _mm256_fnmadd_ps(alpha_v, y_hi, x_hi);
        if (beta != 0.0f) {
          __m256i zd = _mm256_loadu_si256((const __m256i *)(Z + i));
          __m256 zo_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(zd));
          __m256 zo_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(zd, 1));
          z_lo = _mm256_fmadd_ps(beta_v, zo_lo, z_lo);
          z_hi = _mm256_fmadd_ps(beta_v, zo_hi, z_hi);
        }
        __m128i r_lo = _mm256_cvtps_ph(z_lo, _MM_FROUND_TO_NEAREST_INT);
        __m128i r_hi = _mm256_cvtps_ph(z_hi, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256(
          (__m256i *)(Z + i),
          _mm256_inserti128_si256(_mm256_castsi128_si256(r_lo), r_hi, 1));
      }
      for (; i + 8 <= N; i += 8) {
        __m256 x = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(X + i)));
        __m256 y = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(Y + i)));
        __m256 z = _mm256_fnmadd_ps(alpha_v, y, x);
        if (beta != 0.0f) {
          __m256 z_old =
            _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(Z + i)));
          z = _mm256_fmadd_ps(beta_v, z_old, z);
        }
        _mm_storeu_si128((__m128i *)(Z + i),
                         _mm256_cvtps_ph(z, _MM_FROUND_TO_NEAREST_INT));
      }
    }
    for (; i < N; ++i) {
      float xf = static_cast<float>(X[i]);
      float yf = static_cast<float>(Y[i * i_stride]);
      float zf = xf - alpha * yf +
                 ((0.0f == beta) ? 0.0f : beta * static_cast<float>(Z[i]));
      Z[i] = static_cast<_Float16>(zf);
    }
  } else {
    for (unsigned int i = 0; i < N; ++i) {
      float xf = static_cast<float>(*X);
      float yf = static_cast<float>(*Y);
      float zf = xf - alpha * yf +
                 ((0.0f == beta) ? 0.0f : beta * static_cast<float>(*Z));
      *Z = static_cast<_Float16>(zf);
      X += o_stride;
      Y += i_stride;
      Z += o_stride;
    }
  }
}

void ele_div(const unsigned int N, const _Float16 *X, const _Float16 *Y,
             _Float16 *Z, float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  // Newton-Raphson reciprocal helper constant
  const __m256 two = _mm256_set1_ps(2.0f);

  if (alpha == 1.0f && beta == 0.0f && o_stride == 1) {
    unsigned int i = 0;
    if (i_stride == 0) {
      // Precompute refined reciprocal of Y[0] once (rcp + Newton-Raphson)
      __m256 vy = _mm256_set1_ps(static_cast<float>(Y[0]));
      __m256 rcp_est = _mm256_rcp_ps(vy);
      __m256 vy_inv =
        _mm256_mul_ps(rcp_est, _mm256_fnmadd_ps(vy, rcp_est, two));

      for (; i + 16 <= N; i += 16) {
        __m256i xd = _mm256_loadu_si256((const __m256i *)(X + i));
        __m256 x_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(xd));
        __m256 x_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(xd, 1));
        __m256 z_lo = _mm256_mul_ps(x_lo, vy_inv);
        __m256 z_hi = _mm256_mul_ps(x_hi, vy_inv);
        __m128i r_lo = _mm256_cvtps_ph(z_lo, _MM_FROUND_TO_NEAREST_INT);
        __m128i r_hi = _mm256_cvtps_ph(z_hi, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256(
          (__m256i *)(Z + i),
          _mm256_inserti128_si256(_mm256_castsi128_si256(r_lo), r_hi, 1));
      }
      for (; i + 8 <= N; i += 8) {
        __m256 x = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(X + i)));
        __m256 z = _mm256_mul_ps(x, vy_inv);
        _mm_storeu_si128((__m128i *)(Z + i),
                         _mm256_cvtps_ph(z, _MM_FROUND_TO_NEAREST_INT));
      }
      for (; i < N; ++i) {
        Z[i] = static_cast<_Float16>(static_cast<float>(X[i]) /
                                     static_cast<float>(Y[0]));
      }
    } else if (i_stride == 1) {
      for (; i + 16 <= N; i += 16) {
        __m256i xd = _mm256_loadu_si256((const __m256i *)(X + i));
        __m256i yd = _mm256_loadu_si256((const __m256i *)(Y + i));
        __m256 x_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(xd));
        __m256 x_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(xd, 1));
        __m256 y_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(yd));
        __m256 y_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(yd, 1));
        // rcp + Newton-Raphson for per-element reciprocal
        __m256 rcp_lo = _mm256_rcp_ps(y_lo);
        __m256 inv_lo =
          _mm256_mul_ps(rcp_lo, _mm256_fnmadd_ps(y_lo, rcp_lo, two));
        __m256 rcp_hi = _mm256_rcp_ps(y_hi);
        __m256 inv_hi =
          _mm256_mul_ps(rcp_hi, _mm256_fnmadd_ps(y_hi, rcp_hi, two));
        __m256 z_lo = _mm256_mul_ps(x_lo, inv_lo);
        __m256 z_hi = _mm256_mul_ps(x_hi, inv_hi);
        __m128i r_lo = _mm256_cvtps_ph(z_lo, _MM_FROUND_TO_NEAREST_INT);
        __m128i r_hi = _mm256_cvtps_ph(z_hi, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256(
          (__m256i *)(Z + i),
          _mm256_inserti128_si256(_mm256_castsi128_si256(r_lo), r_hi, 1));
      }
      for (; i + 8 <= N; i += 8) {
        __m256 x = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(X + i)));
        __m256 y = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(Y + i)));
        __m256 rcp_y = _mm256_rcp_ps(y);
        __m256 inv_y = _mm256_mul_ps(rcp_y, _mm256_fnmadd_ps(y, rcp_y, two));
        __m256 z = _mm256_mul_ps(x, inv_y);
        _mm_storeu_si128((__m128i *)(Z + i),
                         _mm256_cvtps_ph(z, _MM_FROUND_TO_NEAREST_INT));
      }
      for (; i < N; ++i) {
        Z[i] = static_cast<_Float16>(static_cast<float>(X[i]) /
                                     static_cast<float>(Y[i]));
      }
    } else {
      for (unsigned int i = 0; i < N; ++i) {
        Z[i] = static_cast<_Float16>(static_cast<float>(X[i]) /
                                     static_cast<float>(Y[i * i_stride]));
      }
    }
  } else if (o_stride == 1 && (i_stride == 0 || i_stride == 1)) {
    __m256 alpha_v = _mm256_set1_ps(alpha);
    __m256 beta_v = _mm256_set1_ps(beta);
    unsigned int i = 0;

    if (i_stride == 0) {
      // Precompute refined reciprocal of (alpha * Y[0])
      __m256 denom =
        _mm256_mul_ps(alpha_v, _mm256_set1_ps(static_cast<float>(Y[0])));
      __m256 rcp_d = _mm256_rcp_ps(denom);
      __m256 inv_d = _mm256_mul_ps(rcp_d, _mm256_fnmadd_ps(denom, rcp_d, two));

      for (; i + 16 <= N; i += 16) {
        __m256i xd = _mm256_loadu_si256((const __m256i *)(X + i));
        __m256 x_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(xd));
        __m256 x_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(xd, 1));
        __m256 z_lo = _mm256_mul_ps(x_lo, inv_d);
        __m256 z_hi = _mm256_mul_ps(x_hi, inv_d);
        if (beta != 0.0f) {
          __m256i zd = _mm256_loadu_si256((const __m256i *)(Z + i));
          __m256 zo_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(zd));
          __m256 zo_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(zd, 1));
          z_lo = _mm256_fmadd_ps(beta_v, zo_lo, z_lo);
          z_hi = _mm256_fmadd_ps(beta_v, zo_hi, z_hi);
        }
        __m128i r_lo = _mm256_cvtps_ph(z_lo, _MM_FROUND_TO_NEAREST_INT);
        __m128i r_hi = _mm256_cvtps_ph(z_hi, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256(
          (__m256i *)(Z + i),
          _mm256_inserti128_si256(_mm256_castsi128_si256(r_lo), r_hi, 1));
      }
      for (; i + 8 <= N; i += 8) {
        __m256 x = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(X + i)));
        __m256 z = _mm256_mul_ps(x, inv_d);
        if (beta != 0.0f) {
          __m256 z_old =
            _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(Z + i)));
          z = _mm256_fmadd_ps(beta_v, z_old, z);
        }
        _mm_storeu_si128((__m128i *)(Z + i),
                         _mm256_cvtps_ph(z, _MM_FROUND_TO_NEAREST_INT));
      }
    } else {
      for (; i + 16 <= N; i += 16) {
        __m256i xd = _mm256_loadu_si256((const __m256i *)(X + i));
        __m256i yd = _mm256_loadu_si256((const __m256i *)(Y + i));
        __m256 x_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(xd));
        __m256 x_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(xd, 1));
        __m256 y_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(yd));
        __m256 y_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(yd, 1));
        __m256 d_lo = _mm256_mul_ps(alpha_v, y_lo);
        __m256 d_hi = _mm256_mul_ps(alpha_v, y_hi);
        // rcp + Newton-Raphson for per-element reciprocal
        __m256 rcp_lo = _mm256_rcp_ps(d_lo);
        __m256 inv_lo =
          _mm256_mul_ps(rcp_lo, _mm256_fnmadd_ps(d_lo, rcp_lo, two));
        __m256 rcp_hi = _mm256_rcp_ps(d_hi);
        __m256 inv_hi =
          _mm256_mul_ps(rcp_hi, _mm256_fnmadd_ps(d_hi, rcp_hi, two));
        __m256 z_lo = _mm256_mul_ps(x_lo, inv_lo);
        __m256 z_hi = _mm256_mul_ps(x_hi, inv_hi);
        if (beta != 0.0f) {
          __m256i zd = _mm256_loadu_si256((const __m256i *)(Z + i));
          __m256 zo_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(zd));
          __m256 zo_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(zd, 1));
          z_lo = _mm256_fmadd_ps(beta_v, zo_lo, z_lo);
          z_hi = _mm256_fmadd_ps(beta_v, zo_hi, z_hi);
        }
        __m128i r_lo = _mm256_cvtps_ph(z_lo, _MM_FROUND_TO_NEAREST_INT);
        __m128i r_hi = _mm256_cvtps_ph(z_hi, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256(
          (__m256i *)(Z + i),
          _mm256_inserti128_si256(_mm256_castsi128_si256(r_lo), r_hi, 1));
      }
      for (; i + 8 <= N; i += 8) {
        __m256 x = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(X + i)));
        __m256 y = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(Y + i)));
        __m256 d = _mm256_mul_ps(alpha_v, y);
        __m256 rcp_d = _mm256_rcp_ps(d);
        __m256 inv_d = _mm256_mul_ps(rcp_d, _mm256_fnmadd_ps(d, rcp_d, two));
        __m256 z = _mm256_mul_ps(x, inv_d);
        if (beta != 0.0f) {
          __m256 z_old =
            _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(Z + i)));
          z = _mm256_fmadd_ps(beta_v, z_old, z);
        }
        _mm_storeu_si128((__m128i *)(Z + i),
                         _mm256_cvtps_ph(z, _MM_FROUND_TO_NEAREST_INT));
      }
    }
    for (; i < N; ++i) {
      float xf = static_cast<float>(X[i]);
      float yf = static_cast<float>(Y[i * i_stride]);
      float zf = xf / (alpha * yf) +
                 ((0.0f == beta) ? 0.0f : beta * static_cast<float>(Z[i]));
      Z[i] = static_cast<_Float16>(zf);
    }
  } else {
    for (unsigned int i = 0; i < N; ++i) {
      float xf = static_cast<float>(*X);
      float yf = static_cast<float>(*Y);
      float zf = xf / (alpha * yf) +
                 ((0.0f == beta) ? 0.0f : beta * static_cast<float>(*Z));
      *Z = static_cast<_Float16>(zf);
      X += o_stride;
      Y += i_stride;
      Z += o_stride;
    }
  }
}

// ============================================================
// FP16 BLAS-like operations
// ============================================================

void saxpy(const unsigned int N, const float alpha, const _Float16 *X,
           const unsigned int incX, _Float16 *Y, const unsigned int incY) {
  if (incX == 1 && incY == 1) {
    unsigned int i = 0;
    unsigned int N16 = (N & ~15u);
    __m256 alpha_v = _mm256_set1_ps(alpha);

    // Main loop: process 16 elements per iteration via 256-bit loads
    for (; i < N16; i += 16) {
      __m256i x_raw = _mm256_loadu_si256((const __m256i *)(X + i));
      __m256i y_raw = _mm256_loadu_si256((const __m256i *)(Y + i));

      __m128i x_lo = _mm256_castsi256_si128(x_raw);
      __m128i x_hi = _mm256_extracti128_si256(x_raw, 1);
      __m128i y_lo = _mm256_castsi256_si128(y_raw);
      __m128i y_hi = _mm256_extracti128_si256(y_raw, 1);

      __m256 xf0 = _mm256_cvtph_ps(x_lo);
      __m256 xf1 = _mm256_cvtph_ps(x_hi);
      __m256 yf0 = _mm256_cvtph_ps(y_lo);
      __m256 yf1 = _mm256_cvtph_ps(y_hi);

      __m256 r0 = _mm256_fmadd_ps(alpha_v, xf0, yf0);
      __m256 r1 = _mm256_fmadd_ps(alpha_v, xf1, yf1);

      __m128i out_lo = _mm256_cvtps_ph(r0, _MM_FROUND_TO_NEAREST_INT);
      __m128i out_hi = _mm256_cvtps_ph(r1, _MM_FROUND_TO_NEAREST_INT);
      __m256i out =
        _mm256_inserti128_si256(_mm256_castsi128_si256(out_lo), out_hi, 1);
      _mm256_storeu_si256((__m256i *)(Y + i), out);
    }

    // Tail: process remaining 8 elements
    if (i + 8 <= N) {
      __m256 x = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(X + i)));
      __m256 y = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(Y + i)));
      __m256 result = _mm256_fmadd_ps(alpha_v, x, y);
      _mm_storeu_si128((__m128i *)(Y + i),
                       _mm256_cvtps_ph(result, _MM_FROUND_TO_NEAREST_INT));
      i += 8;
    }

    // Scalar tail
    for (; i < N; ++i) {
      Y[i] = static_cast<_Float16>(static_cast<float>(Y[i]) +
                                   alpha * static_cast<float>(X[i]));
    }
  } else {
    for (unsigned int i = 0; i < N; ++i) {
      Y[i * incY] =
        static_cast<_Float16>(static_cast<float>(Y[i * incY]) +
                              alpha * static_cast<float>(X[i * incX]));
    }
  }
}

_Float16 sdot(const unsigned int N, const _Float16 *X, const unsigned int incX,
              const _Float16 *Y, const unsigned int incY) {
  assert(incX > 0 && incY > 0);
  if (incX == 1 && incY == 1) {
    unsigned int i = 0;
    unsigned int N16 = (N & ~15u);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    // Main loop: 16 elements with dual accumulators to break dependency
    for (; i < N16; i += 16) {
      __m256i x_raw = _mm256_loadu_si256((const __m256i *)(X + i));
      __m256i y_raw = _mm256_loadu_si256((const __m256i *)(Y + i));

      __m256 xf0 = _mm256_cvtph_ps(_mm256_castsi256_si128(x_raw));
      __m256 xf1 = _mm256_cvtph_ps(_mm256_extracti128_si256(x_raw, 1));
      __m256 yf0 = _mm256_cvtph_ps(_mm256_castsi256_si128(y_raw));
      __m256 yf1 = _mm256_cvtph_ps(_mm256_extracti128_si256(y_raw, 1));

      acc0 = _mm256_fmadd_ps(xf0, yf0, acc0);
      acc1 = _mm256_fmadd_ps(xf1, yf1, acc1);
    }

    // Merge accumulators and reduce
    __m256 acc = _mm256_add_ps(acc0, acc1);

    // Tail: process remaining 8 elements
    if (i + 8 <= N) {
      __m256 x = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(X + i)));
      __m256 y = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(Y + i)));
      acc = _mm256_fmadd_ps(x, y, acc);
      i += 8;
    }

    float sum = hsum_avx(acc);

    // Scalar tail
    for (; i < N; ++i) {
      sum += static_cast<float>(X[i]) * static_cast<float>(Y[i]);
    }
    return static_cast<_Float16>(sum);
  } else {
    float sum = 0.0f;
    for (unsigned int i = 0; i < N; ++i) {
      sum += static_cast<float>(X[i * incX]) * static_cast<float>(Y[i * incY]);
    }
    return static_cast<_Float16>(sum);
  }
}

_Float16 snrm2(const unsigned int N, const _Float16 *X,
               const unsigned int incX) {
  if (incX == 1) {
    unsigned int i = 0;
    unsigned int N16 = (N & ~15u);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    // Main loop: 16 elements with dual accumulators
    for (; i < N16; i += 16) {
      __m256i x_raw = _mm256_loadu_si256((const __m256i *)(X + i));

      __m256 xf0 = _mm256_cvtph_ps(_mm256_castsi256_si128(x_raw));
      __m256 xf1 = _mm256_cvtph_ps(_mm256_extracti128_si256(x_raw, 1));

      acc0 = _mm256_fmadd_ps(xf0, xf0, acc0);
      acc1 = _mm256_fmadd_ps(xf1, xf1, acc1);
    }

    // Merge accumulators
    __m256 acc = _mm256_add_ps(acc0, acc1);

    // Tail: process remaining 8 elements
    if (i + 8 <= N) {
      __m256 x = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(X + i)));
      acc = _mm256_fmadd_ps(x, x, acc);
      i += 8;
    }

    float sum = hsum_avx(acc);

    // Scalar tail
    for (; i < N; ++i) {
      float xf = static_cast<float>(X[i]);
      sum += xf * xf;
    }
    return static_cast<_Float16>(std::sqrt(sum));
  } else {
    float sum = 0.0f;
    for (unsigned int i = 0; i < N; ++i) {
      float xf = static_cast<float>(X[i * incX]);
      sum += xf * xf;
    }
    return static_cast<_Float16>(std::sqrt(sum));
  }
}

void sscal(const unsigned int N, const float alpha, _Float16 *X,
           const unsigned int incX) {
  if (incX == 1) {
    unsigned int i = 0;
    unsigned int N16 = (N & ~15u);
    __m256 alpha_v = _mm256_set1_ps(alpha);

    // Main loop: process 16 elements per iteration via 256-bit loads
    for (; i < N16; i += 16) {
      __m256i x_raw = _mm256_loadu_si256((const __m256i *)(X + i));

      __m128i x_lo = _mm256_castsi256_si128(x_raw);
      __m128i x_hi = _mm256_extracti128_si256(x_raw, 1);

      __m256 xf0 = _mm256_cvtph_ps(x_lo);
      __m256 xf1 = _mm256_cvtph_ps(x_hi);

      __m256 r0 = _mm256_mul_ps(alpha_v, xf0);
      __m256 r1 = _mm256_mul_ps(alpha_v, xf1);

      __m128i out_lo = _mm256_cvtps_ph(r0, _MM_FROUND_TO_NEAREST_INT);
      __m128i out_hi = _mm256_cvtps_ph(r1, _MM_FROUND_TO_NEAREST_INT);
      __m256i out =
        _mm256_inserti128_si256(_mm256_castsi128_si256(out_lo), out_hi, 1);
      _mm256_storeu_si256((__m256i *)(X + i), out);
    }

    // Tail: process remaining 8 elements
    if (i + 8 <= N) {
      __m256 x = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(X + i)));
      __m256 result = _mm256_mul_ps(alpha_v, x);
      _mm_storeu_si128((__m128i *)(X + i),
                       _mm256_cvtps_ph(result, _MM_FROUND_TO_NEAREST_INT));
      i += 8;
    }

    // Scalar tail
    for (; i < N; ++i) {
      X[i] = static_cast<_Float16>(alpha * static_cast<float>(X[i]));
    }
  } else {
    for (unsigned int i = 0; i < N; ++i) {
      X[i * incX] =
        static_cast<_Float16>(alpha * static_cast<float>(X[i * incX]));
    }
  }
}

void custom_scopy(const unsigned int N, const _Float16 *X,
                  const unsigned int incX, _Float16 *Y,
                  const unsigned int incY) {
  if (incX == 1 && incY == 1) {
    unsigned int i = 0;
    unsigned int N16 = (N & ~15u);
    for (; i < N16; i += 16) {
      __m256i data = _mm256_loadu_si256((const __m256i *)(X + i));
      _mm256_storeu_si256((__m256i *)(Y + i), data);
    }
    for (; i < N; ++i) {
      Y[i] = X[i];
    }
  } else {
    for (unsigned int i = 0; i < N; ++i) {
      Y[i * incY] = X[i * incX];
    }
  }
}

// ============================================================
// FP16 activation/norm functions
// ============================================================

_Float16 max_val(const unsigned int N, _Float16 *X) {
  unsigned int i = 0;
  __m256 vmax0 = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
  __m256 vmax1 = _mm256_set1_ps(-std::numeric_limits<float>::infinity());

  // Main loop: 16 elements per iteration with dual max accumulators
  unsigned int N16 = (N & ~15u);
  for (; i < N16; i += 16) {
    __m256i raw = _mm256_loadu_si256((const __m256i *)(X + i));
    __m256 x0 = _mm256_cvtph_ps(_mm256_castsi256_si128(raw));
    __m256 x1 = _mm256_cvtph_ps(_mm256_extracti128_si256(raw, 1));
    vmax0 = _mm256_max_ps(vmax0, x0);
    vmax1 = _mm256_max_ps(vmax1, x1);
  }

  // Merge dual accumulators
  __m256 vmax = _mm256_max_ps(vmax0, vmax1);

  // Tail: 8 elements
  if (i + 8 <= N) {
    __m256 x = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(X + i)));
    vmax = _mm256_max_ps(vmax, x);
    i += 8;
  }

  // Horizontal max reduction
  __m128 hi = _mm256_extractf128_ps(vmax, 1);
  __m128 lo = _mm256_castps256_ps128(vmax);
  lo = _mm_max_ps(lo, hi);
  __m128 shuf = _mm_movehl_ps(lo, lo);
  lo = _mm_max_ps(lo, shuf);
  shuf = _mm_movehdup_ps(lo);
  lo = _mm_max_ss(lo, shuf);
  float result = _mm_cvtss_f32(lo);

  for (; i < N; ++i) {
    result = std::max(result, static_cast<float>(X[i]));
  }
  return static_cast<_Float16>(result);
}

void softmax(const unsigned int N, _Float16 *X, _Float16 *Y) {
  // Step 1: find max
  float max_x = static_cast<float>(max_val(N, X));
  __m256 vmax = _mm256_set1_ps(max_x);

  // Step 2: exp(x - max) and accumulate sum with 16-wide loop
  unsigned int i = 0;
  __m256 vsum0 = _mm256_setzero_ps();
  __m256 vsum1 = _mm256_setzero_ps();

  for (; i + 16 <= N; i += 16) {
    __m256i raw = _mm256_loadu_si256((const __m256i *)(X + i));
    __m256 x0 = _mm256_cvtph_ps(_mm256_castsi256_si128(raw));
    __m256 x1 = _mm256_cvtph_ps(_mm256_extracti128_si256(raw, 1));
    __m256 e0 = exp256_ps(_mm256_sub_ps(x0, vmax));
    __m256 e1 = exp256_ps(_mm256_sub_ps(x1, vmax));
    __m128i out_lo = _mm256_cvtps_ph(e0, _MM_FROUND_TO_NEAREST_INT);
    __m128i out_hi = _mm256_cvtps_ph(e1, _MM_FROUND_TO_NEAREST_INT);
    _mm256_storeu_si256(
      (__m256i *)(Y + i),
      _mm256_inserti128_si256(_mm256_castsi128_si256(out_lo), out_hi, 1));
    vsum0 = _mm256_add_ps(vsum0, e0);
    vsum1 = _mm256_add_ps(vsum1, e1);
  }

  // Tail: 8 elements
  if (i + 8 <= N) {
    __m256 x = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(X + i)));
    __m256 e = exp256_ps(_mm256_sub_ps(x, vmax));
    _mm_storeu_si128((__m128i *)(Y + i),
                     _mm256_cvtps_ph(e, _MM_FROUND_TO_NEAREST_INT));
    vsum0 = _mm256_add_ps(vsum0, e);
    i += 8;
  }

  float sum = hsum_avx(_mm256_add_ps(vsum0, vsum1));
  for (; i < N; ++i) {
    float e = std::exp(static_cast<float>(X[i]) - max_x);
    Y[i] = static_cast<_Float16>(e);
    sum += e;
  }

  // Step 3: normalize with 16-wide loop
  float inv_sum = 1.0f / sum;
  __m256 vinv = _mm256_set1_ps(inv_sum);

  i = 0;
  for (; i + 16 <= N; i += 16) {
    __m256i raw = _mm256_loadu_si256((const __m256i *)(Y + i));
    __m256 y0 = _mm256_cvtph_ps(_mm256_castsi256_si128(raw));
    __m256 y1 = _mm256_cvtph_ps(_mm256_extracti128_si256(raw, 1));
    __m256 r0 = _mm256_mul_ps(y0, vinv);
    __m256 r1 = _mm256_mul_ps(y1, vinv);
    __m128i out_lo = _mm256_cvtps_ph(r0, _MM_FROUND_TO_NEAREST_INT);
    __m128i out_hi = _mm256_cvtps_ph(r1, _MM_FROUND_TO_NEAREST_INT);
    _mm256_storeu_si256(
      (__m256i *)(Y + i),
      _mm256_inserti128_si256(_mm256_castsi128_si256(out_lo), out_hi, 1));
  }
  if (i + 8 <= N) {
    __m256 y = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(Y + i)));
    __m256 result = _mm256_mul_ps(y, vinv);
    _mm_storeu_si128((__m128i *)(Y + i),
                     _mm256_cvtps_ph(result, _MM_FROUND_TO_NEAREST_INT));
    i += 8;
  }
  for (; i < N; ++i) {
    Y[i] = static_cast<_Float16>(static_cast<float>(Y[i]) * inv_sum);
  }
}

void inv_sqrt_inplace(const unsigned int N, _Float16 *X) {
  unsigned int i = 0;
  const __m256 zero = _mm256_setzero_ps();
  const __m256 inf_val = _mm256_set1_ps(INFINITY);
  const __m256 three_half = _mm256_set1_ps(1.5f);
  const __m256 half = _mm256_set1_ps(0.5f);

  // Main loop: 16 elements per iteration
  for (; i + 16 <= N; i += 16) {
    __m256i raw = _mm256_loadu_si256((const __m256i *)(X + i));
    __m256 x0 = _mm256_cvtph_ps(_mm256_castsi256_si128(raw));
    __m256 x1 = _mm256_cvtph_ps(_mm256_extracti128_si256(raw, 1));

    __m256 is_zero0 = _mm256_cmp_ps(x0, zero, _CMP_EQ_OQ);
    __m256 is_zero1 = _mm256_cmp_ps(x1, zero, _CMP_EQ_OQ);

    __m256 est0 = _mm256_rsqrt_ps(x0);
    __m256 est1 = _mm256_rsqrt_ps(x1);

    // Newton-Raphson: y = y * (1.5 - 0.5 * x * y * y)
    __m256 half_x0 = _mm256_mul_ps(half, x0);
    __m256 half_x1 = _mm256_mul_ps(half, x1);
    __m256 yy0 = _mm256_mul_ps(est0, est0);
    __m256 yy1 = _mm256_mul_ps(est1, est1);
    __m256 ref0 =
      _mm256_mul_ps(est0, _mm256_fnmadd_ps(half_x0, yy0, three_half));
    __m256 ref1 =
      _mm256_mul_ps(est1, _mm256_fnmadd_ps(half_x1, yy1, three_half));

    ref0 = _mm256_blendv_ps(ref0, inf_val, is_zero0);
    ref1 = _mm256_blendv_ps(ref1, inf_val, is_zero1);

    __m128i out_lo = _mm256_cvtps_ph(ref0, _MM_FROUND_TO_NEAREST_INT);
    __m128i out_hi = _mm256_cvtps_ph(ref1, _MM_FROUND_TO_NEAREST_INT);
    _mm256_storeu_si256(
      (__m256i *)(X + i),
      _mm256_inserti128_si256(_mm256_castsi128_si256(out_lo), out_hi, 1));
  }

  // Tail: 8 elements
  if (i + 8 <= N) {
    __m256 x = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(X + i)));
    __m256 is_zero = _mm256_cmp_ps(x, zero, _CMP_EQ_OQ);
    __m256 est = _mm256_rsqrt_ps(x);
    __m256 half_x = _mm256_mul_ps(half, x);
    __m256 yy = _mm256_mul_ps(est, est);
    __m256 refined =
      _mm256_mul_ps(est, _mm256_fnmadd_ps(half_x, yy, three_half));
    refined = _mm256_blendv_ps(refined, inf_val, is_zero);
    _mm_storeu_si128((__m128i *)(X + i),
                     _mm256_cvtps_ph(refined, _MM_FROUND_TO_NEAREST_INT));
    i += 8;
  }

  // Scalar tail
  for (; i < N; ++i) {
    X[i] = static_cast<_Float16>(1.0f / std::sqrt(static_cast<float>(X[i])));
  }
}

void swiglu(const unsigned int N, _Float16 *X, _Float16 *Y, _Float16 *Z) {
  unsigned int i = 0;

  const auto oldcsr = _mm_getcsr();
  _mm_setcsr(oldcsr | 0x8040); // DAZ | FTZ

  // 16-wide blocks with 256-bit loads/stores
  for (; i + 16 <= N; i += 16) {
    __m256i y_raw = _mm256_loadu_si256((const __m256i *)(Y + i));
    __m256i z_raw = _mm256_loadu_si256((const __m256i *)(Z + i));

    __m256 y0 = _mm256_cvtph_ps(_mm256_castsi256_si128(y_raw));
    __m256 y1 = _mm256_cvtph_ps(_mm256_extracti128_si256(y_raw, 1));
    __m256 z0 = _mm256_cvtph_ps(_mm256_castsi256_si128(z_raw));
    __m256 z1 = _mm256_cvtph_ps(_mm256_extracti128_si256(z_raw, 1));

    __m128i out_lo =
      _mm256_cvtps_ph(avx2_approx_swiglu(y0, z0), _MM_FROUND_TO_NEAREST_INT);
    __m128i out_hi =
      _mm256_cvtps_ph(avx2_approx_swiglu(y1, z1), _MM_FROUND_TO_NEAREST_INT);
    _mm256_storeu_si256(
      (__m256i *)(X + i),
      _mm256_inserti128_si256(_mm256_castsi128_si256(out_lo), out_hi, 1));
  }

  // One 8-wide block if available
  if (i + 8 <= N) {
    __m256 y0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(Y + i)));
    __m256 z0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(Z + i)));
    _mm_storeu_si128(
      (__m128i *)(X + i),
      _mm256_cvtps_ph(avx2_approx_swiglu(y0, z0), _MM_FROUND_TO_NEAREST_INT));
    i += 8;
  }

  // Scalar remainder
  for (; i < N; ++i) {
    float yf = static_cast<float>(Y[i]);
    float zf = static_cast<float>(Z[i]);
    X[i] = static_cast<_Float16>((yf / (1.0f + std::exp(-yf))) * zf);
  }

  _mm_setcsr(oldcsr);
}

// ============================================================
// FP16 rms_norm and rotary embedding
// ============================================================

void rms_norm_wrt_width_fp16(const _Float16 *__restrict X,
                             _Float16 *__restrict Y, size_t H, size_t W,
                             float epsilon) {
  for (size_t h = 0; h < H; ++h) {
    const _Float16 *rowX = X + h * W;
    _Float16 *rowY = Y + h * W;

    size_t i = 0;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    // Sum-of-squares: 32-wide loop with 256-bit loads
    for (; i + 32 <= W; i += 32) {
      __m256i raw0 = _mm256_loadu_si256((const __m256i *)(rowX + i));
      __m256i raw1 = _mm256_loadu_si256((const __m256i *)(rowX + i + 16));
      __m256 x0 = _mm256_cvtph_ps(_mm256_castsi256_si128(raw0));
      __m256 x1 = _mm256_cvtph_ps(_mm256_extracti128_si256(raw0, 1));
      __m256 x2 = _mm256_cvtph_ps(_mm256_castsi256_si128(raw1));
      __m256 x3 = _mm256_cvtph_ps(_mm256_extracti128_si256(raw1, 1));
      acc0 = _mm256_fmadd_ps(x0, x0, acc0);
      acc1 = _mm256_fmadd_ps(x1, x1, acc1);
      acc2 = _mm256_fmadd_ps(x2, x2, acc2);
      acc3 = _mm256_fmadd_ps(x3, x3, acc3);
    }
    // 16-wide tail
    if (i + 16 <= W) {
      __m256i raw = _mm256_loadu_si256((const __m256i *)(rowX + i));
      __m256 x0 = _mm256_cvtph_ps(_mm256_castsi256_si128(raw));
      __m256 x1 = _mm256_cvtph_ps(_mm256_extracti128_si256(raw, 1));
      acc0 = _mm256_fmadd_ps(x0, x0, acc0);
      acc1 = _mm256_fmadd_ps(x1, x1, acc1);
      i += 16;
    }
    // 8-wide tail
    if (i + 8 <= W) {
      __m256 x = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(rowX + i)));
      acc0 = _mm256_fmadd_ps(x, x, acc0);
      i += 8;
    }
    float sumsq =
      hsum_avx(acc0) + hsum_avx(acc1) + hsum_avx(acc2) + hsum_avx(acc3);
    for (; i < W; ++i) {
      float v = static_cast<float>(rowX[i]);
      sumsq += v * v;
    }

    float mean = sumsq / static_cast<float>(W);
    float scale = 1.0f / std::sqrt(mean + epsilon);
    __m256 vscale = _mm256_set1_ps(scale);

    // Scaling pass: 32-wide loop with 256-bit loads/stores
    i = 0;
    for (; i + 32 <= W; i += 32) {
      __m256i raw0 = _mm256_loadu_si256((const __m256i *)(rowX + i));
      __m256i raw1 = _mm256_loadu_si256((const __m256i *)(rowX + i + 16));
      __m256 x0 = _mm256_cvtph_ps(_mm256_castsi256_si128(raw0));
      __m256 x1 = _mm256_cvtph_ps(_mm256_extracti128_si256(raw0, 1));
      __m256 x2 = _mm256_cvtph_ps(_mm256_castsi256_si128(raw1));
      __m256 x3 = _mm256_cvtph_ps(_mm256_extracti128_si256(raw1, 1));
      __m128i r0 =
        _mm256_cvtps_ph(_mm256_mul_ps(x0, vscale), _MM_FROUND_TO_NEAREST_INT);
      __m128i r1 =
        _mm256_cvtps_ph(_mm256_mul_ps(x1, vscale), _MM_FROUND_TO_NEAREST_INT);
      __m128i r2 =
        _mm256_cvtps_ph(_mm256_mul_ps(x2, vscale), _MM_FROUND_TO_NEAREST_INT);
      __m128i r3 =
        _mm256_cvtps_ph(_mm256_mul_ps(x3, vscale), _MM_FROUND_TO_NEAREST_INT);
      _mm256_storeu_si256(
        (__m256i *)(rowY + i),
        _mm256_inserti128_si256(_mm256_castsi128_si256(r0), r1, 1));
      _mm256_storeu_si256(
        (__m256i *)(rowY + i + 16),
        _mm256_inserti128_si256(_mm256_castsi128_si256(r2), r3, 1));
    }
    // 16-wide tail
    if (i + 16 <= W) {
      __m256i raw = _mm256_loadu_si256((const __m256i *)(rowX + i));
      __m256 x0 = _mm256_cvtph_ps(_mm256_castsi256_si128(raw));
      __m256 x1 = _mm256_cvtph_ps(_mm256_extracti128_si256(raw, 1));
      __m128i r0 =
        _mm256_cvtps_ph(_mm256_mul_ps(x0, vscale), _MM_FROUND_TO_NEAREST_INT);
      __m128i r1 =
        _mm256_cvtps_ph(_mm256_mul_ps(x1, vscale), _MM_FROUND_TO_NEAREST_INT);
      _mm256_storeu_si256(
        (__m256i *)(rowY + i),
        _mm256_inserti128_si256(_mm256_castsi128_si256(r0), r1, 1));
      i += 16;
    }
    // 8-wide tail
    if (i + 8 <= W) {
      __m256 x = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(rowX + i)));
      _mm_storeu_si128(
        (__m128i *)(rowY + i),
        _mm256_cvtps_ph(_mm256_mul_ps(x, vscale), _MM_FROUND_TO_NEAREST_INT));
      i += 8;
    }
    for (; i < W; ++i) {
      rowY[i] = static_cast<_Float16>(static_cast<float>(rowX[i]) * scale);
    }
  }
}

void compute_rotary_embedding_value(unsigned int dim, unsigned int half_,
                                    unsigned int w, _Float16 *in, _Float16 *out,
                                    float *cos_, float *sin_) {
  unsigned int k = 0;
  for (; k + 7 < half_; k += 8) {
    unsigned int i0 = w + k;
    unsigned int i1 = w + k + half_;

    __m256 a = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(in + i0)));
    __m256 b = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(in + i1)));
    __m256 cos_v = _mm256_loadu_ps(&cos_[k]);
    __m256 sin_v = _mm256_loadu_ps(&sin_[k]);

    // FMA: out0 = a*cos - b*sin, out1 = a*sin + b*cos
    __m256 out0 = _mm256_fmsub_ps(a, cos_v, _mm256_mul_ps(b, sin_v));
    __m256 out1 = _mm256_fmadd_ps(a, sin_v, _mm256_mul_ps(b, cos_v));

    _mm_storeu_si128((__m128i *)(out + i0),
                     _mm256_cvtps_ph(out0, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i *)(out + i1),
                     _mm256_cvtps_ph(out1, _MM_FROUND_TO_NEAREST_INT));
  }

  for (; k < dim; ++k) {
    unsigned int span = w + k;
    float value = static_cast<float>(in[span]);
    float transformed_value;
    if (k < half_) {
      transformed_value = -1.0f * static_cast<float>(in[w + k + half_]);
    } else {
      transformed_value = static_cast<float>(in[w + k - half_]);
    }
    out[span] =
      static_cast<_Float16>(value * cos_[k] + transformed_value * sin_[k]);
  }
}

// ============================================================
// FP16 Group B functions
// ============================================================

unsigned int isamax(const unsigned int N, const _Float16 *X,
                    const unsigned int incX) {
  if (incX == 1 && N >= 8) {
    unsigned int N8 = (N & ~7u);
    __m256 sign_mask = _mm256_set1_ps(-0.0f);
    unsigned int max_idx = 0;
    float max_abs = 0.0f;

    for (unsigned int i = 0; i < N8; i += 8) {
      __m256 x = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(X + i)));
      __m256 abs_x = _mm256_andnot_ps(sign_mask, x);
      // Check each element individually for max tracking
      alignas(32) float buf[8];
      _mm256_storeu_ps(buf, abs_x);
      for (int j = 0; j < 8; ++j) {
        if (buf[j] > max_abs) {
          max_abs = buf[j];
          max_idx = i + j;
        }
      }
    }
    for (unsigned int i = N8; i < N; ++i) {
      float cur = std::abs(static_cast<float>(X[i]));
      if (cur > max_abs) {
        max_abs = cur;
        max_idx = i;
      }
    }
    return max_idx;
  } else {
    unsigned int max_idx = 0;
    float max_val = 0.0f;
    for (unsigned int i = 0; i < N; ++i) {
      float cur = std::abs(static_cast<float>(X[i * incX]));
      if (cur > max_val) {
        max_val = cur;
        max_idx = i;
      }
    }
    return max_idx;
  }
}

void transpose_matrix(const unsigned int M, const unsigned int N,
                      const _Float16 *src, unsigned int ld_src, _Float16 *dst,
                      unsigned int ld_dst) {
  for (unsigned int i = 0; i < M; i++) {
    for (unsigned int j = 0; j < N; j++) {
      dst[i + j * ld_dst] = src[i * ld_src + j];
    }
  }
}

void scopy_int4_to_float16(const unsigned int N, const uint8_t *X,
                           const unsigned int incX, _Float16 *Y,
                           const unsigned int incY) {
  if (incX == 1 && incY == 1) {
    unsigned int i = 0;
    unsigned int N8 = (N & ~7u);
    for (; i < N8; i += 8) {
      alignas(32) float buf[16];
      for (int j = 0; j < 8; ++j) {
        buf[2 * j] = static_cast<float>(X[i + j] >> 4);
        buf[2 * j + 1] = static_cast<float>(X[i + j] & 0x0f);
      }
      __m256 v0 = _mm256_loadu_ps(buf);
      __m256 v1 = _mm256_loadu_ps(buf + 8);
      _mm_storeu_si128((__m128i *)(Y + 2 * i),
                       _mm256_cvtps_ph(v0, _MM_FROUND_TO_NEAREST_INT));
      _mm_storeu_si128((__m128i *)(Y + 2 * i + 8),
                       _mm256_cvtps_ph(v1, _MM_FROUND_TO_NEAREST_INT));
    }
    for (; i < N; ++i) {
      Y[2 * i] = static_cast<_Float16>(X[i] >> 4);
      Y[2 * i + 1] = static_cast<_Float16>(X[i] & 0x0f);
    }
  } else {
    for (unsigned int idx = 0; idx < N; idx++) {
      Y[2 * idx] = static_cast<_Float16>(X[idx] >> 4);
      Y[2 * idx + 1] = static_cast<_Float16>(X[idx] & 0x0f);
    }
  }
}

void scopy_int8_to_float16(const unsigned int N, const uint8_t *X,
                           const unsigned int incX, _Float16 *Y,
                           const unsigned int incY) {
  if (incX == 1 && incY == 1) {
    unsigned int i = 0;
    unsigned int N8 = (N & ~7u);
    for (; i < N8; i += 8) {
      __m256i xi =
        _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(X + i)));
      __m256 xf = _mm256_cvtepi32_ps(xi);
      _mm_storeu_si128((__m128i *)(Y + i),
                       _mm256_cvtps_ph(xf, _MM_FROUND_TO_NEAREST_INT));
    }
    for (; i < N; ++i) {
      Y[i] = static_cast<_Float16>(X[i]);
    }
  } else {
    for (unsigned int idx = 0; idx < N; idx++) {
      Y[idx * incY] = static_cast<_Float16>(X[idx * incX]);
    }
  }
}

void scopy_int8_to_float16(const unsigned int N, const int8_t *X,
                           const unsigned int incX, _Float16 *Y,
                           const unsigned int incY) {
  if (incX == 1 && incY == 1) {
    unsigned int i = 0;
    unsigned int N8 = (N & ~7u);
    for (; i < N8; i += 8) {
      __m256i xi =
        _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(X + i)));
      __m256 xf = _mm256_cvtepi32_ps(xi);
      _mm_storeu_si128((__m128i *)(Y + i),
                       _mm256_cvtps_ph(xf, _MM_FROUND_TO_NEAREST_INT));
    }
    for (; i < N; ++i) {
      Y[i] = static_cast<_Float16>(X[i]);
    }
  } else {
    for (unsigned int idx = 0; idx < N; idx++) {
      Y[idx * incY] = static_cast<_Float16>(X[idx * incX]);
    }
  }
}

} // namespace nntrainer::avx2
