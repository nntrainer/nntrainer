// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   nntr_ggml_impl.h
 * @date   13 August 2025
 * @see    https://github.com/nntrainer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Custom-implemented functions to support ggml functions for internal
 * uses in nntrainer
 */

#ifndef __NNTR_GGML_IMPL__
#define __NNTR_GGML_IMPL__

#include <stddef.h>
#include <stdint.h>

/**
 * @brief Initialize ggml implementation state
 */
void nntr_ggml_init();

/**
 * @brief Compute Q4_0 4x8 weights by Q8_0 activations GEMM
 */
void nntr_gemm_q4_0_4x8_q8_0(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc);

#ifdef ENABLE_FP16
// Pick the half type the same way tensor_dim.h does, so this header stays
// self-contained even if a caller pulls it in without tensor_dim.h. On
// ARM/Android (USE__FP16) it is __fp16; on x86_64 fp16 builds it is _Float16.
#ifdef USE__FP16
#define NNTR_GGML_FP16 __fp16
#else
#define NNTR_GGML_FP16 _Float16
#endif
void nntr_gemm_q4_0_4x8_q8_0_fp16(int n, NNTR_GGML_FP16 *__restrict s,
                                  size_t bs, const void *__restrict vx,
                                  const void *__restrict vy, int nr, int nc);

/**
 * @brief Compute Q8_0 weights by Q8_0 activations GEMM, FP16 output (SMMLA)
 */
void nntr_gemm_q8_0_q8_0_fp16(int n, NNTR_GGML_FP16 *__restrict s, size_t bs,
                              const void *__restrict vx,
                              const void *__restrict vy, int nr, int nc);

/**
 * @brief Compute interleaved (q8_0x4) Q8_0 weights by q8_0x4 activations GEMM,
 *        FP16 output (register-blocked 4x4 SMMLA). Both operands pre-packed.
 */
void nntr_gemm_q8_0_q8_0_4x4_fp16(int n, NNTR_GGML_FP16 *__restrict s,
                                  size_t bs, const void *__restrict vx,
                                  const void *__restrict vy, int nr, int nc);

/**
 * @brief Hand-written ARM64 SMMLA asm port of nntr_gemm_q4_0_4x8_q8_0_fp16
 *        for Q8_0x4 weights (no nibble-unpack) by Q8_0x4 activations GEMM,
 *        FP16 output. Same 4x8-interleaved block-loop structure/register
 *        schedule as the Q4_0 kernel, with 8 direct block loads (stride
 *        0x88) replacing the 4-load+unpack sequence and no /16 scvtf shift.
 */
void nntr_gemm_q8_0_4x8_q8_0_fp16(int n, NNTR_GGML_FP16 *__restrict s,
                                  size_t bs, const void *__restrict vx,
                                  const void *__restrict vy, int nr, int nc);

void nntr_gemv_q4_0_4x8_q8_0_fp16(int n, NNTR_GGML_FP16 *__restrict s,
                                  size_t bs, const void *__restrict vx,
                                  const void *__restrict vy, int nr, int nc);
#endif

/**
 * @brief Compute Q4_0 8x8 weights by Q8_0 activations GEMM
 */
void nntr_gemm_q4_0_8x8_q8_0(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc);

/**
 * @brief Compute Q4_K 8x8 weights by Q8_K activations GEMM
 */
void nntr_gemm_q4_K_8x8_q8_K(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc);

/**
 * @brief Compute Q4_0 4x8 weights by Q8_0 activations GEMV
 */
void nntr_gemv_q4_0_4x8_q8_0(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc);

/**
 * @brief Compute Q4_0 8x8 weights by Q8_0 activations GEMV
 */
void nntr_gemv_q4_0_8x8_q8_0(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc);

/**
 * @brief Compute Q8_0 weights by Q8_0 activations GEMM
 */
void nntr_gemm_q8_0_q8_0(int n, float *__restrict s, size_t bs,
                         const void *__restrict vx, const void *__restrict vy,
                         int nr, int nc);

/**
 * @brief Compute Q8_0 weights by Q8_0 activations GEMV
 */
void nntr_gemv_q8_0_q8_0(int n, float *__restrict s, size_t bs,
                         const void *__restrict vx, const void *__restrict vy,
                         int nr, int nc);

/**
 * @brief Compute Q4_K 8x8 weights by Q8_K activations GEMV
 */
void nntr_gemv_q4_K_8x8_q8_K(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc);

/**
 * @brief Quantize FP32 matrix rows to Q8_0 4x8 layout
 */
void nntr_quantize_mat_q8_0_4x8(const float *__restrict x, void *__restrict vy,
                                int64_t k);

/**
 * @brief Quantize FP32 matrix rows to Q8_K 4x8 layout
 */
void nntr_quantize_mat_q8_K_4x8(const float *__restrict x, void *__restrict vy,
                                int64_t k);

/**
 * @brief Repack Q4_0 data to Q4_0 4-block layout
 */
int nntr_repack_q4_0_to_q4_0_4_bl(void *__restrict dst, int interleave_block,
                                  const void *__restrict data, size_t data_size,
                                  size_t nrow, size_t k);

/**
 * @brief Repack Q4_0 data to Q4_0 8-block layout
 */
int nntr_repack_q4_0_to_q4_0_8_bl(void *__restrict dst, int interleave_block,
                                  const void *__restrict data, size_t data_size,
                                  size_t nrow, size_t k);

/**
 * @brief Repack Q4_K data to Q4_K 8-block layout
 */
int nntr_repack_q4_K_to_q4_K_8_bl(void *__restrict dst, int interleave_block,
                                  const void *__restrict data, size_t data_size,
                                  size_t nrow, size_t k);

/**
 * @brief Quantize FP32 rows to Q4_0
 */
size_t nntr_quantize_q4_0(const float *__restrict src, void *__restrict dst,
                          int64_t nrows, int64_t n_per_row,
                          const float *imatrix);

/**
 * @brief Quantize FP32 rows to Q4_K
 */
size_t nntr_quantize_q4_K(const float *__restrict src, void *__restrict dst,
                          int64_t nrows, int64_t n_per_row,
                          const float *imatrix);

/**
 * @brief Quantize FP32 rows to Q6_K
 */
size_t nntr_quantize_q6_K(const float *__restrict src, void *__restrict dst,
                          int64_t nrows, int64_t n_per_row,
                          const float *imatrix);

/**
 * @brief Quantize FP32 rows to Q8_0
 */
size_t nntr_quantize_q8_0(const float *__restrict src, void *__restrict dst,
                          int64_t nrows, int64_t n_per_row,
                          const float *imatrix);

/**
 * @brief Quantize one FP32 row to Q8_0
 */
void nntr_quantize_row_q8_0(const float *__restrict x, void *__restrict y,
                            int64_t k);

/**
 * @brief Quantize one FP32 row to Q8_K
 */
void nntr_quantize_row_q8_K(const float *__restrict x, void *__restrict y,
                            int64_t k);

/**
 * @brief Dequantize one Q4_0 row to FP32
 */
void nntr_dequantize_row_q4_0(const void *__restrict x, float *__restrict y,
                              int64_t k);

/**
 * @brief Dequantize one Q4_K row to FP32
 */
void nntr_dequantize_row_q4_K(const void *__restrict x, float *__restrict y,
                              int64_t k);

/**
 * @brief Dequantize one Q6_K row to FP32
 */
void nntr_dequantize_row_q6_K(const void *__restrict x, float *__restrict y,
                              int64_t k);

/**
 * @brief Dequantize one Q8_0 row to FP32
 */
void nntr_dequantize_row_q8_0(const void *__restrict x, float *__restrict y,
                              int64_t k);

/**
 * @brief Dequantize one Q8_K row to FP32
 */
void nntr_dequantize_row_q8_K(const void *__restrict x, float *__restrict y,
                              int64_t k);

/**
 * @brief Compute dot product between Q6_K and Q8_K vectors
 */
void nntr_vec_dot_q6_K_q8_K(int n, float *__restrict s, size_t bs,
                            const void *__restrict vx, size_t bx,
                            const void *__restrict vy, size_t by, int nrc);

#endif
