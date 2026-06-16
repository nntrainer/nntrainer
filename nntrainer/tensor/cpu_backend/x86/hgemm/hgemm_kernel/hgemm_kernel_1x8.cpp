// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Yonghyeon Cho <dyddyd8574@gmail.com>
 *
 * @file   hgemm_kernel_1x8.cpp
 * @date   30 May 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Yonghyeon Cho <dyddyd8574@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  1x8 AVX2+FMA micro-kernel for x86 FP16 GEMM
 */

#include "hgemm_kernel.h"

#include <immintrin.h>

namespace nntrainer::hgemm::internal {

void hgemm_kernel_1x8(unsigned int K, const float *packed_A,
                      const float *packed_B, float *C, unsigned int c_stride) {
  (void)c_stride;

  __m256 c00 = _mm256_loadu_ps(C);

  for (unsigned int k = 0; k < K; ++k) {
    const float *pa = packed_A + k * 6;
    const float *pb = packed_B + k * 16;

    const __m256 b0 = _mm256_loadu_ps(pb);
    const __m256 a0 = _mm256_broadcast_ss(pa + 0);
    c00 = _mm256_fmadd_ps(a0, b0, c00);
  }

  _mm256_storeu_ps(C, c00);
}

} /* namespace nntrainer::hgemm::internal */
