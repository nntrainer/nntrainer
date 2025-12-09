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

/**
 * @brief Set openblas #thread.
 * @param [in] num_threads Use BLAS_NUM_THREADS if < 0.
 *                         Otherwise, directly call openblas API.
 */
void __openblas_set_num_threads(int num_threads) {
  if (num_threads < 0) {
  #ifdef BLAS_NUM_THREADS
    openblas_set_num_threads(BLAS_NUM_THREADS);
  #else
    /// Without openblas_set_num_threads,
    /// it's set std::thread::hardware_concurrency()
    /// It can be too high especially when the given blas function is small
    /// or if there are other threads already created (nntrainer thread pool)
    /// With big-little & threadboost, hardware_concurrency might be not good.
    /// @todo configure this! (4? num of big cores? ...)
  #endif
  } else {
    openblas_set_num_threads(num_threads);
  }
}
void __cblas_saxpy(const unsigned int N, const float alpha, const float *X,
                   const unsigned int incX, float *Y, const unsigned int incY) {
  cblas_saxpy(N, alpha, X, incX, Y, incY);
}

void __cblas_sgemv(const unsigned int TStorageOrder, bool TransA,
                   const unsigned int M, const unsigned int N,
                   const float alpha, const float *A, const unsigned int lda,
                   const float *X, const unsigned int incX, const float beta,
                   float *Y, const unsigned int incY) {
  CBLAS_TRANSPOSE transA = TransA ? CblasTrans : CblasNoTrans;
  CBLAS_ORDER order = TStorageOrder ? CblasColMajor : CblasRowMajor;
  cblas_sgemv(order, transA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
}

float __cblas_sdot(const unsigned int N, const float *X,
                   const unsigned int incX, const float *Y,
                   const unsigned int incY) {
  return cblas_sdot(N, X, incX, Y, incY);
}

void __cblas_scopy(const unsigned int N, const float *X,
                   const unsigned int incX, float *Y, const unsigned int incY) {
  cblas_scopy(N, X, incX, Y, incY);
}

void __cblas_sscal(const unsigned int N, const float alpha, float *X,
                   const unsigned int incX) {
  cblas_sscal(N, alpha, X, incX);
}

float __cblas_snrm2(const unsigned int N, const float *X,
                    const unsigned int incX) {
  return cblas_snrm2(N, X, incX);
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
  cblas_sgemm(order, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C,
              ldc);
}

unsigned int __cblas_isamax(const unsigned int N, const float *X,
                            const unsigned int incX) {
  return cblas_isamax(N, X, incX);
}
} // namespace nntrainer
