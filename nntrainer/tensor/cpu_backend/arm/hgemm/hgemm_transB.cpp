// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   hgemm_transB.cpp
 * @date   10 July 2024
 * @see    https://github.com/nntrainer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is half-precision GEMM interface of transposed B case
 *
 */

#include <cmath>
#include <hgemm_common.h>
#include <hgemm_kernel.h>
#include <hgemm_noTrans.h>
#include <hgemm_pack.h>
#include <hgemm_transB.h>
#include <hgemm_util.h>
#include <limits>
#include <matrix_transpose_neon.h>

void hgemm_transB_8x16(unsigned int M, unsigned int N, unsigned int K,
                       const __fp16 *A, unsigned int lda, const __fp16 *B,
                       unsigned int ldb, float *C, unsigned int ldc,
                       float alpha, float beta) {

  __fp16 *sA = alignedMalloc(M * K);
  __fp16 *sB = alignedMalloc(K * N);

  unsigned int ms, ms2, ns, ks;
  unsigned int m_min, m2_min, n_min, k_min;
  unsigned int stride_l1 = 1;

  for (ms = 0; ms < M; ms += M_BLOCKING) {
    m_min = M - ms;
    if (m_min > M_BLOCKING) {
      m_min = M_BLOCKING;
    }

    for (ks = 0; ks < K; ks += k_min) {
      k_min = K - ks;
      if (k_min >= (K_BLOCKING << 1)) {
        k_min = K_BLOCKING;
      } else if (k_min > K_BLOCKING) {
        k_min = (k_min / 2 + GEMM_UNROLLING_4 - 1) & ~(GEMM_UNROLLING_4 - 1);
      }

      n_min = N;
      if (N >= N_BLOCKING * 2) {
        n_min = N_BLOCKING;
      } else if (N > N_BLOCKING) {
        n_min = ((n_min / 2 + GEMM_UNROLLING_16 - 1) / GEMM_UNROLLING_16) *
                GEMM_UNROLLING_16;
      } else {
        stride_l1 = 0;
      }
      packing_transB16(k_min, n_min, B + (ks), ldb, sB);

      for (ms2 = ms; ms2 < ms + m_min; ms2 += m2_min) {
        m2_min = (ms + m_min) - ms2;
        if (m2_min >= 3 * GEMM_UNROLLING_8) {
          m2_min = 3 * GEMM_UNROLLING_8;
        } else if (m2_min >= 2 * GEMM_UNROLLING_8) {
          m2_min = 2 * GEMM_UNROLLING_8;
        } else if (m2_min > GEMM_UNROLLING_8) {
          m2_min = GEMM_UNROLLING_8;
        }
        packing_A8(m2_min, k_min, A + ms2 * lda + ks, lda,
                   sA + k_min * (ms2 - ms) * stride_l1);
        hgemm_kernel_8x16(m2_min, n_min, k_min,
                          sA + k_min * (ms2 - ms) * stride_l1, sB,
                          C + ms2 * ldc, ldc);
      }

      for (ns = n_min; ns < N; ns += n_min) {
        n_min = N - ns;
        if (n_min >= N_BLOCKING * 2) {
          n_min = N_BLOCKING;
        } else if (n_min > N_BLOCKING) {
          n_min =
            (n_min / 2 + GEMM_UNROLLING_16 - 1) & ~(GEMM_UNROLLING_16 - 1);
        }
        packing_transB16(k_min, n_min, B + ks + ldb * ns, ldb, sB);
        hgemm_kernel_8x16(m_min, n_min, k_min, sA, sB, C + ms * ldc + ns, ldc);
      }
    }
  }

  free(sA);
  free(sB);
}

void hgemm_transB(const __fp16 *A, const __fp16 *B, float *C, unsigned int M,
                  unsigned int N, unsigned int K, float alpha, float beta) {
  const float eps = std::numeric_limits<float>::epsilon();
  if (((M & 0x7) == 0 && (N & 0xF) == 0 && (K & 0x7) == 0 &&
       (std::abs(alpha - 1.F) < eps))) {
    return hgemm_transB_8x16(M, N, K, A, K, B, K, C, N, alpha, beta);
  } else {
    return hgemm_transB_fallback(A, B, C, M, N, K, alpha, beta);
  }
}

void hgemm_transB_fallback(const __fp16 *A, const __fp16 *B, float *C,
                           unsigned int M, unsigned int N, unsigned int K,
                           float alpha, float beta) {
  __fp16 *B_T = alignedMalloc(K * N);

  transpose_neon<__fp16>(N, K, B, K, B_T, N);

  hgemm_noTrans(A, B_T, C, M, N, K, alpha, beta);

  free(B_T);
}
