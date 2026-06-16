// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Yonghyeon Cho <dyddyd8574@gmail.com>
 *
 * @file   hgemm_kernel_mxn.cpp
 * @date   30 May 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Yonghyeon Cho <dyddyd8574@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  Micro-kernel shape dispatcher for x86 FP16 GEMM
 */

#include "hgemm_kernel.h"

namespace nntrainer::hgemm::internal {

void hgemm_kernel_mxn(unsigned int M, unsigned int N, unsigned int K,
                      const float *packed_A, const float *packed_B, float *C,
                      unsigned int c_stride) {
  if (M <= 1) {
    if (N <= 8) {
      hgemm_kernel_1x8(K, packed_A, packed_B, C, c_stride);
    } else {
      hgemm_kernel_1x16(K, packed_A, packed_B, C, c_stride);
    }
  } else if (M <= 2) {
    if (N <= 8) {
      hgemm_kernel_2x8(K, packed_A, packed_B, C, c_stride);
    } else {
      hgemm_kernel_2x16(K, packed_A, packed_B, C, c_stride);
    }
  } else if (M <= 4) {
    if (N <= 8) {
      hgemm_kernel_4x8(K, packed_A, packed_B, C, c_stride);
    } else {
      hgemm_kernel_4x16(K, packed_A, packed_B, C, c_stride);
    }
  } else if (N <= 8) {
    hgemm_kernel_6x8(K, packed_A, packed_B, C, c_stride);
  } else {
    hgemm_kernel_6x16(K, packed_A, packed_B, C, c_stride);
  }
}

} /* namespace nntrainer::hgemm::internal */
