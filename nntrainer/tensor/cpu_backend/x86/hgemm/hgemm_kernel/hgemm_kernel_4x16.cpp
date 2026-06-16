// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Yonghyeon Cho <dyddyd8574@gmail.com>
 *
 * @file   hgemm_kernel_4x16.cpp
 * @date   30 May 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Yonghyeon Cho <dyddyd8574@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  4x16 AVX2+FMA micro-kernel for x86 FP16 GEMM
 */

#include "hgemm_kernel.h"

#include <immintrin.h>

namespace nntrainer::hgemm::internal {

void hgemm_kernel_4x16(unsigned int K, const float *packed_A,
                       const float *packed_B, float *C, unsigned int c_stride) {
  float *c0 = C + 0 * c_stride;
  float *c1 = C + 1 * c_stride;
  float *c2 = C + 2 * c_stride;
  float *c3 = C + 3 * c_stride;

  __m256 c00 = _mm256_loadu_ps(c0 + 0);
  __m256 c01 = _mm256_loadu_ps(c0 + 8);
  __m256 c10 = _mm256_loadu_ps(c1 + 0);
  __m256 c11 = _mm256_loadu_ps(c1 + 8);
  __m256 c20 = _mm256_loadu_ps(c2 + 0);
  __m256 c21 = _mm256_loadu_ps(c2 + 8);
  __m256 c30 = _mm256_loadu_ps(c3 + 0);
  __m256 c31 = _mm256_loadu_ps(c3 + 8);

  for (unsigned int k = 0; k < K; ++k) {
    const float *pa = packed_A + k * 6;
    const float *pb = packed_B + k * 16;

    const __m256 b0 = _mm256_loadu_ps(pb + 0);
    const __m256 b1 = _mm256_loadu_ps(pb + 8);

    const __m256 a0 = _mm256_broadcast_ss(pa + 0);
    c00 = _mm256_fmadd_ps(a0, b0, c00);
    c01 = _mm256_fmadd_ps(a0, b1, c01);

    const __m256 a1 = _mm256_broadcast_ss(pa + 1);
    c10 = _mm256_fmadd_ps(a1, b0, c10);
    c11 = _mm256_fmadd_ps(a1, b1, c11);

    const __m256 a2 = _mm256_broadcast_ss(pa + 2);
    c20 = _mm256_fmadd_ps(a2, b0, c20);
    c21 = _mm256_fmadd_ps(a2, b1, c21);

    const __m256 a3 = _mm256_broadcast_ss(pa + 3);
    c30 = _mm256_fmadd_ps(a3, b0, c30);
    c31 = _mm256_fmadd_ps(a3, b1, c31);
  }

  _mm256_storeu_ps(c0 + 0, c00);
  _mm256_storeu_ps(c0 + 8, c01);
  _mm256_storeu_ps(c1 + 0, c10);
  _mm256_storeu_ps(c1 + 8, c11);
  _mm256_storeu_ps(c2 + 0, c20);
  _mm256_storeu_ps(c2 + 8, c21);
  _mm256_storeu_ps(c3 + 0, c30);
  _mm256_storeu_ps(c3 + 8, c31);
}

} /* namespace nntrainer::hgemm::internal */
