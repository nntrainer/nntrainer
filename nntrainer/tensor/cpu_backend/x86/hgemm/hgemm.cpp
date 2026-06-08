// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Yonghyeon Cho <dyddyd8574@gmail.com>
 *
 * @file   hgemm.cpp
 * @date   15 May 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Yonghyeon Cho <dyddyd8574@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  Entry point for the x86 cache-blocked FP16 GEMM
 */

#include "hgemm.h"

#include "hgemm_blocked.h"
#include "hgemm_fast_path.h"
#include "hgemm_util.h"
#include "hgemm_workspace.h"

#include <cmath>

namespace nntrainer::avx2 {

using namespace internal;

namespace {

template <typename AType, typename BType, typename CType>
void hgemm_compute(bool TransA, bool TransB, unsigned int M, unsigned int N,
                   unsigned int K, float alpha, const AType *A,
                   unsigned int a_stride, const BType *B, unsigned int b_stride,
                   float beta, CType *C, unsigned int c_stride) {
  if (M == 0 || N == 0) {
    return;
  }

  if (K == 0 || std::fpclassify(alpha) == FP_ZERO) {
    apply_beta_to_C<CType>(C, M, N, c_stride, beta);
    return;
  }

  HgemmWorkspace &workspace = get_hgemm_workspace();

  if (try_hgemm_fast_path<AType, BType, CType>(TransA, TransB, M, N, K, alpha,
                                               A, a_stride, B, b_stride, beta,
                                               C, c_stride, workspace)) {
    return;
  }

  run_hgemm_blocked<AType, BType, CType>(TransA, TransB, M, N, K, alpha, A,
                                         a_stride, B, b_stride, beta, C,
                                         c_stride, workspace);
}

} // namespace

void hgemm(const _FP16 *A, const _FP16 *B, _FP16 *C, unsigned int M,
           unsigned int N, unsigned int K, unsigned int lda, unsigned int ldb,
           unsigned int ldc, float alpha, float beta, bool TransA,
           bool TransB) {
  hgemm_compute<_FP16, _FP16, _FP16>(TransA, TransB, M, N, K, alpha, A, lda, B,
                                     ldb, beta, C, ldc);
}

void shgemm(const float *A, const _FP16 *B, float *C, unsigned int M,
            unsigned int N, unsigned int K, unsigned int lda, unsigned int ldb,
            unsigned int ldc, float alpha, float beta, bool TransA,
            bool TransB) {
  hgemm_compute<float, _FP16, float>(TransA, TransB, M, N, K, alpha, A, lda, B,
                                     ldb, beta, C, ldc);
}

void hsgemm(const _FP16 *A, const float *B, float *C, unsigned int M,
            unsigned int N, unsigned int K, unsigned int lda, unsigned int ldb,
            unsigned int ldc, float alpha, float beta, bool TransA,
            bool TransB) {
  hgemm_compute<_FP16, float, float>(TransA, TransB, M, N, K, alpha, A, lda, B,
                                     ldb, beta, C, ldc);
}

} /* namespace nntrainer::avx2 */
