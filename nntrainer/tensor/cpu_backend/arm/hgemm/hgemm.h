// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   hgemm.h
 * @date   01 April 2024
 * @see    https://github.com/nntrainer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @author Debadri Samaddar <s.debadri@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is half-precision GEMM interface
 *
 */

#ifndef __HGEMM_H__
#define __HGEMM_H__

/**
 * @brief     hgemm computation with neon : Y = alpha*op(A)*op(B) + beta*C,
 * where op(X) is one of X or X**T
 * @param[in] A __fp16 * for Matrix A
 * @param[in] B __fp16 * for Matrix B
 * @param[in] C __fp16 * for Matrix C
 * @param[in] M number of op(A)'s and C's row
 * @param[in] N number of op(B)'s and C's columns
 * @param[in] K number of op(A)'s and columns and op(B)'s rows
 * @param[in] alpha float number
 * @param[in] beta float number
 * @param[in] TransA bool transpose info of lhs matrix
 * @param[in] TransB bool transpose info of rhs matrix
 */
void hgemm(const __fp16 *A, const __fp16 *B, __fp16 *C, unsigned int M,
           unsigned int N, unsigned int K, float alpha, float beta, bool TransA,
           bool TransB);

/**
 * @brief     hgemm computation with neon but with small dim without padding : Y
 * = alpha*op(A)*op(B) + beta*C, where op(X) is one of X or X**T
 * @param[in] A __fp16 * for Matrix A
 * @param[in] B __fp16 * for Matrix B
 * @param[in] C __fp16 * for Matrix C
 * @param[in] M number of op(A)'s and C's row
 * @param[in] N number of op(B)'s and C's columns
 * @param[in] K number of op(A)'s and columns and op(B)'s rows
 * @param[in] alpha float number
 * @param[in] beta float number
 * @param[in] TransA bool transpose info of lhs matrix
 * @param[in] TransB bool transpose info of rhs matrix
 */
void hgemm_small(const __fp16 *A, const __fp16 *B, __fp16 *C, unsigned int M,
                 unsigned int N, unsigned int K, float alpha, float beta,
                 bool TransA, bool TransB);

/**
 * @brief     Checking function for whether matrix A or B needs padding for
 * optimal performance of fixed blocking-kernel sequence
 * @param[in] A __fp16 * for Matrix A
 * @param[in] B __fp16 * for Matrix B
 * @param[in] C float * for Matrix C
 * @param[in] M number of op(A)'s and C's row
 * @param[in] N number of op(B)'s and C's columns
 * @param[in] K number of op(A)'s and columns and op(B)'s rows
 * @param[in] alpha float number
 * @param[in] beta float number
 * @param[in] TransA bool transpose info of lhs matrix
 * @param[in] TransB bool transpose info of rhs matrix
 */
void hgemm_ensure_divisibility(const __fp16 *A, const __fp16 *B, float *C32,
                               unsigned int M, unsigned int N, unsigned int K,
                               float alpha = 1.F, float beta = 0.F,
                               bool TransA = false, bool TransB = false);

/**
 * @brief     Classifying function for GEMM computation case for noTrans,
 * transA, transB, transAB
 * @param[in] A __fp16 * for Matrix A
 * @param[in] B __fp16 * for Matrix B
 * @param[in] C __fp16 * for Matrix C
 * @param[in] M number of op(A)'s and C's row
 * @param[in] N number of op(B)'s and C's columns
 * @param[in] K number of op(A)'s and columns and op(B)'s rows
 * @param[in] alpha float number
 * @param[in] beta float number
 * @param[in] TransA bool transpose info of lhs matrix
 * @param[in] TransB bool transpose info of rhs matrix
 */
void hgemm_classify(const __fp16 *A, const __fp16 *B, float *C32,
                    unsigned int M, unsigned int N, unsigned int K,
                    float alpha = 1.F, float beta = 0.F, bool TransA = false,
                    bool TransB = false);
/**
 * @brief     hgemm computation when K = 1. Transpose is mathematically no use
 * for here, and partial accumulation is also not needed.
 * @param[in] A __fp16 * for Matrix A
 * @param[in] B __fp16 * for Matrix B
 * @param[in] C __fp16 * for Matrix C
 * @param[in] M number of op(A)'s and C's row
 * @param[in] N number of op(B)'s and C's columns
 * @param[in] K number of op(A)'s and columns and op(B)'s rows
 * @param[in] alpha float number
 * @param[in] beta float number
 * @param[in] TransA bool transpose info of lhs matrix
 * @param[in] TransB bool transpose info of rhs matrix
 */
void hgemm_K1(const __fp16 *A, const __fp16 *B, __fp16 *C, unsigned int M,
              unsigned int N, unsigned int K, float alpha, float beta,
              bool TransA, bool TransB);

/**
 * @brief     FP16 x FP16 -> FP32 QK GEMM via ARMv8.2-A FMLAL widening.
 * Computes C[m,n] = alpha * sum_k A[m,k] * B[n,k] (TransB-style dot). Each pair
 * of vfmlalq_low/high_f16 widens 8 FP16 products into FP32 accumulators, so
 * products accumulate in FP32 from the start (no FP16-product overflow on wide
 * logits). Row-major; lda/ldb/ldc are row strides; K is the dot length.
 * @param[in] A __fp16 * lhs, row-major, lda columns
 * @param[in] B __fp16 * rhs, row-major, ldb columns
 * @param[in] C float * output, row-major, ldc columns
 * @param[in] M number of rows of A and C
 * @param[in] N number of rows of B (columns of C)
 * @param[in] K dot length
 * @param[in] alpha scaling factor applied to the result
 * @param[in] lda row stride of A
 * @param[in] ldb row stride of B
 * @param[in] ldc row stride of C
 */
void hgemm_f16xf16_f32_fmlal(const __fp16 *A, const __fp16 *B, float *C,
                             unsigned int M, unsigned int N, unsigned int K,
                             float alpha, unsigned int lda, unsigned int ldb,
                             unsigned int ldc);

/**
 * @brief     Runtime check for FEAT_FHM (ARMv8.2-A FMLAL widening, asimdfhm)
 * support on the current CPU. hgemm_f16xf16_f32_fmlal() is compiled with
 * "+fp16fml", which is an optional extension on top of nntrainer's baseline
 * march (armv8.2-a+fp16+dotprod+i8mm) -- some ARMv8.2 FP16 devices lack it,
 * so callers must check this before dispatching to that kernel and fall back
 * otherwise (e.g. to shgemm) to avoid SIGILL. Result is cached after the
 * first call.
 * @return    true if the CPU supports FEAT_FHM
 */
bool hgemm_fp16fml_supported();

#endif /* __HGEMM_H__ */
