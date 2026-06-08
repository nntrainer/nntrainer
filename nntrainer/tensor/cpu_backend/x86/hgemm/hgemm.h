// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Yonghyeon Cho <dyddyd8574@gmail.com>
 *
 * @file   hgemm.h
 * @date   15 May 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Yonghyeon Cho <dyddyd8574@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  Entry point for the x86 cache-blocked FP16 GEMM
 */

#ifndef __X86_HGEMM_H_
#define __X86_HGEMM_H_

#include <tensor_dim.h>

namespace nntrainer::avx2 {

/**
 * @brief Compute C = alpha * op(A) * op(B) + beta * C in FP16.
 *
 * Replaces the legacy FP16 sgemm triple-conversion + CBLAS path with a
 * cache-blocked GEMM that converts FP16->FP32 during packing in L2-sized
 * blocks. Internal compute is done in FP32; the result is converted back to
 * FP16 on writeback.
 *
 * @param A    FP16 source matrix, row-major
 * @param B    FP16 source matrix, row-major
 * @param C    FP16 destination, row-major
 * @param M    rows of op(A) / C
 * @param N    cols of op(B) / C
 * @param K    inner dimension
 * @param lda  leading dimension of A (row stride in elements). Must be >=
 *             (TransA ? M : K).
 * @param ldb  leading dimension of B (row stride in elements). Must be >=
 *             (TransB ? K : N).
 * @param ldc  leading dimension of C (row stride in elements). Must be >= N.
 * @param alpha scalar applied to op(A) * op(B)
 * @param beta scalar applied to C before accumulation
 * @param TransA whether A is transposed
 * @param TransB whether B is transposed
 */
void hgemm(const _FP16 *A, const _FP16 *B, _FP16 *C, unsigned int M,
           unsigned int N, unsigned int K, unsigned int lda, unsigned int ldb,
           unsigned int ldc, float alpha, float beta, bool TransA, bool TransB);

/**
 * @brief Mixed-precision GEMM C = alpha * op(A) * op(B) + beta * C with FP32 A,
 * FP16 B and FP32 C (shgemm).
 *
 * Shares the cache-blocked path with hgemm(): A is re-laid out without
 * conversion during packing, B is widened from FP16, and the FP32 accumulator
 * is copied straight back into C with no narrowing.
 *
 * @param A    FP32 source matrix, row-major
 * @param B    FP16 source matrix, row-major
 * @param C    FP32 destination, row-major
 * @param M    rows of op(A) / C
 * @param N    cols of op(B) / C
 * @param K    inner dimension
 * @param lda  leading dimension of A (>= TransA ? M : K)
 * @param ldb  leading dimension of B (>= TransB ? K : N)
 * @param ldc  leading dimension of C (>= N)
 * @param alpha scalar applied to op(A) * op(B)
 * @param beta scalar applied to C before accumulation
 * @param TransA whether A is transposed
 * @param TransB whether B is transposed
 */
void shgemm(const float *A, const _FP16 *B, float *C, unsigned int M,
            unsigned int N, unsigned int K, unsigned int lda, unsigned int ldb,
            unsigned int ldc, float alpha, float beta, bool TransA,
            bool TransB);

/**
 * @brief Mixed-precision GEMM C = alpha * op(A) * op(B) + beta * C with FP16 A,
 * FP32 B and FP32 C (hsgemm).
 *
 * Mirror of shgemm() with the converted operand swapped: A is widened from
 * FP16, B is re-laid out from FP32.
 *
 * @param A    FP16 source matrix, row-major
 * @param B    FP32 source matrix, row-major
 * @param C    FP32 destination, row-major
 * @param M    rows of op(A) / C
 * @param N    cols of op(B) / C
 * @param K    inner dimension
 * @param lda  leading dimension of A (>= TransA ? M : K)
 * @param ldb  leading dimension of B (>= TransB ? K : N)
 * @param ldc  leading dimension of C (>= N)
 * @param alpha scalar applied to op(A) * op(B)
 * @param beta scalar applied to C before accumulation
 * @param TransA whether A is transposed
 * @param TransB whether B is transposed
 */
void hsgemm(const _FP16 *A, const float *B, float *C, unsigned int M,
            unsigned int N, unsigned int K, unsigned int lda, unsigned int ldb,
            unsigned int ldc, float alpha, float beta, bool TransA,
            bool TransB);

} /* namespace nntrainer::avx2 */

#endif /* __X86_HGEMM_H_ */
