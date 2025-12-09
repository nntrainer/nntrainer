// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file cblas_interface.h
 * @date   23 April 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Function interface to use cblas lib from cpu_backend
 *
 */

#ifndef __CBLAS_INTERFACE_H__
#define __CBLAS_INTERFACE_H__
#ifdef __cplusplus

namespace nntrainer {

#ifdef USE_BLAS
/**
 * @brief     Wrapper for openblas_set_num_threads()
 * @param[in] num_threads The number of threads for openblas op.
 *                        Set -1 for default policy (BLAS_NUM_THREADS)
 */
void __openblas_set_num_threads(int num_threads);
/**
 * @brief     saxpy computation : Y = alpha*X + Y
 * @param[in] N number of elements in Y
 * @param[in] alpha float number
 * @param[in] X __fp16 * for Vector X
 * @param[in] Y __fp16 * for Vector Y
 */
void __cblas_saxpy(const unsigned int N, const float alpha, const float *X,
                   const unsigned int incX, float *Y, const unsigned int incY);
/**
 * @brief     sgemv computation : Y = alpha*A*X + beta*Y
 * @param[in] TStorageOrder Row major / Col major
 * @param[in] A float * for Matrix A
 * @param[in] X float * for Vector X
 * @param[in] Y float * for Vector Y
 * @param[in] rows number of A's row
 * @param[in] cols number of A's columns
 * @param[in] alpha float number
 * @param[in] beta float number
 */
void __cblas_sgemv(const unsigned int TStorageOrder, bool TransA,
                   const unsigned int M, const unsigned int N,
                   const float alpha, const float *A, const unsigned int lda,
                   const float *X, const unsigned int incX, const float beta,
                   float *Y, const unsigned int incY);
/**
 * @brief     sdot computation : sum of all X * Y
 * @param[in] N number of elements in Y
 * @param[in] X float * for Vector X
 * @param[in] Y float * for Vector Y
 */
float __cblas_sdot(const unsigned int N, const float *X,
                   const unsigned int incX, const float *Y,
                   const unsigned int incY);
/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] Y float * for Vector Y
 */
void __cblas_scopy(const unsigned int N, const float *X,
                   const unsigned int incX, float *Y, const unsigned int incY);
/**
 * @brief     sscal computation : X = alpha * X
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] alpha float number
 */
void __cblas_sscal(const unsigned int N, const float alpha, float *X,
                   const unsigned int incX);
/**
 * @brief     snrm2 computation : Euclidean norm
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 */
float __cblas_snrm2(const unsigned int N, const float *X,
                    const unsigned int incX);
/**
 * @brief     sgemm computation  : Y = alpha*op(A)*op(B) + beta*C,
 * where op(X) is one of X or X**T
 * @param[in] TStorageOrder Row major / Col major
 * @param[in] A float * for Matrix A
 * @param[in] B float * for Matrix B
 * @param[in] C float * for Matrix C
 * @param[in] M number of op(A)'s and C's row
 * @param[in] N number of op(B)'s and C's columns
 * @param[in] K number of op(A)'s and columns and op(B)'s rows
 * @param[in] alpha float number
 * @param[in] beta float number
 */
void __cblas_sgemm(const unsigned int TStorageOrder, bool TransA, bool TransB,
                   const unsigned int M, const unsigned int N,
                   const unsigned int K, const float alpha, const float *A,
                   const unsigned int lda, const float *B,
                   const unsigned int ldb, const float beta, float *C,
                   const unsigned int ldc);
/**
 * @brief     isamax function : index of first maxima
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 */
unsigned int __cblas_isamax(const unsigned int N, const float *X,
                            const unsigned int incX);
#else  // USE_BLAS
#error "cblas_interface.h is referred without enabling USE_BLAS (-Denable-blas=true with meson). Enable blas to use cblas_interface.h"
#endif // USE_BLAS
} // namespace nntrainer

#endif
#endif
