// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Yonghyeon Cho <dyddyd8574@gmail.com>
 *
 * @file   hgemm_kernel_2x8.cpp
 * @date   30 May 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Yonghyeon Cho <dyddyd8574@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  2x8 AVX2+FMA micro-kernel for x86 FP16 GEMM
 */

#include "hgemm_kernel.h"

#include <immintrin.h>

namespace nntrainer::avx2::internal {

void hgemm_kernel_2x8(unsigned int K, const float *packed_A,
                      const float *packed_B, float *C, unsigned int c_stride) {
  float *c0 = C + 0 * c_stride;
  float *c1 = C + 1 * c_stride;

  __m256 c00 = _mm256_loadu_ps(c0);
  __m256 c10 = _mm256_loadu_ps(c1);

  for (unsigned int k = 0; k < K; ++k) {
    const float *pa = packed_A + k * 6;
    const float *pb = packed_B + k * 16;

    const __m256 b0 = _mm256_loadu_ps(pb);

    const __m256 a0 = _mm256_broadcast_ss(pa + 0);
    c00 = _mm256_fmadd_ps(a0, b0, c00);

    const __m256 a1 = _mm256_broadcast_ss(pa + 1);
    c10 = _mm256_fmadd_ps(a1, b0, c10);
  }

  _mm256_storeu_ps(c0, c00);
  _mm256_storeu_ps(c1, c10);
}

} /* namespace nntrainer::avx2::internal */
