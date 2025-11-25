// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file cblas_interface.cpp
 * @date   23 April 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Function interface to use cblas lib from cpu_backend
 *
 */

#include <cblas.h>
#include <cblas_interface.h>

namespace nntrainer {
void __cblas_saxpy(const unsigned int N, const float alpha, const float *X,
                   const unsigned int incX, float *Y, const unsigned int incY) {
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#endif
  cblas_saxpy(N, alpha, X, incX, Y, incY);
}

void __cblas_sgemv(const unsigned int TStorageOrder, bool TransA,
                   const unsigned int M, const unsigned int N,
                   const float alpha, const float *A, const unsigned int lda,
                   const float *X, const unsigned int incX, const float beta,
                   float *Y, const unsigned int incY) {
  CBLAS_TRANSPOSE transA = TransA ? CblasTrans : CblasNoTrans;
  CBLAS_ORDER order = TStorageOrder ? CblasColMajor : CblasRowMajor;
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#endif
  cblas_sgemv(order, transA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
}

float __cblas_sdot(const unsigned int N, const float *X,
                   const unsigned int incX, const float *Y,
                   const unsigned int incY) {
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#endif
  return cblas_sdot(N, X, incX, Y, incY);
}

void __cblas_scopy(const unsigned int N, const float *X,
                   const unsigned int incX, float *Y, const unsigned int incY) {
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#endif
  cblas_scopy(N, X, incX, Y, incY);
}

void __cblas_sscal(const unsigned int N, const float alpha, float *X,
                   const unsigned int incX) {
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#endif
  cblas_sscal(N, alpha, X, incX);
}

float __cblas_snrm2(const unsigned int N, const float *X,
                    const unsigned int incX) {
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#endif
  return cblas_snrm2(N, X, incX);
}

double __cblas_dnrm2(const unsigned int N, const double *X,
                    const unsigned int incX) {
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#endif
  return cblas_dnrm2(N, X, incX);
}

void __cblas_sgemm(const unsigned int TStorageOrder, bool TransA, bool TransB,
                   const unsigned int M, const unsigned int N,
                   const unsigned int K, const float alpha, const float *A,
                   const unsigned int lda, const float *B,
                   const unsigned int ldb, const float beta, float *C,
                   const unsigned int ldc) {
  CBLAS_TRANSPOSE transA = TransA ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE transB = TransB ? CblasTrans : CblasNoTrans;
  CBLAS_ORDER order = TStorageOrder ? CblasColMajor : CblasRowMajor;
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#endif
  cblas_sgemm(order, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C,
              ldc);
}

unsigned int __cblas_isamax(const unsigned int N, const float *X,
                            const unsigned int incX) {
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#endif
  return cblas_isamax(N, X, incX);
}
} // namespace nntrainer
