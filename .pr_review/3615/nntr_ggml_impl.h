// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   nntr_ggml_impl.h
 * @date   13 August 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Custom-implemented functions to support ggml functions for internal
 * uses in nntrainer
 */

#ifndef __NNTR_GGML_IMPL__
#define __NNTR_GGML_IMPL__

#include <stddef.h>
#include <stdint.h>

void nntr_ggml_init();

void nntr_gemm_q4_0_4x8_q8_0(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc);

void nntr_gemm_q4_0_8x8_q8_0(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc);

void nntr_gemm_q4_K_8x8_q8_K(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc);

void nntr_gemv_q4_0_4x8_q8_0(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc);

void nntr_gemv_q4_0_8x8_q8_0(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc);

void nntr_gemv_q4_K_8x8_q8_K(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc);

void nntr_quantize_mat_q8_0_4x8(const float *__restrict x, void *__restrict vy,
                                int64_t k);

void nntr_quantize_mat_q8_K_4x8(const float *__restrict x, void *__restrict vy,
                                int64_t k);

int nntr_repack_q4_0_to_q4_0_4_bl(void *__restrict dst, int interleave_block,
                                  const void *__restrict data, size_t data_size,
                                  size_t nrow, size_t k);

int nntr_repack_q4_0_to_q4_0_8_bl(void *__restrict dst, int interleave_block,
                                  const void *__restrict data, size_t data_size,
                                  size_t nrow, size_t k);

int nntr_repack_q4_K_to_q4_K_8_bl(void *__restrict dst, int interleave_block,
                                  const void *__restrict data, size_t data_size,
                                  size_t nrow, size_t k);

size_t nntr_quantize_q4_0(const float *__restrict src, void *__restrict dst,
                          int64_t nrows, int64_t n_per_row,
                          const float *imatrix);

size_t nntr_quantize_q4_K(const float *__restrict src, void *__restrict dst,
                          int64_t nrows, int64_t n_per_row,
                          const float *imatrix);

size_t nntr_quantize_q6_K(const float *__restrict src, void *__restrict dst,
                          int64_t nrows, int64_t n_per_row,
                          const float *imatrix);

size_t nntr_quantize_q8_0(const float *__restrict src, void *__restrict dst,
                          int64_t nrows, int64_t n_per_row,
                          const float *imatrix);

void nntr_quantize_row_q8_0(const float *__restrict x, void *__restrict y,
                            int64_t k);

void nntr_quantize_row_q8_K(const float *__restrict x, void *__restrict y,
                            int64_t k);

void nntr_dequantize_row_q4_0(const void *__restrict x, float *__restrict y,
                              int64_t k);

void nntr_dequantize_row_q4_K(const void *__restrict x, float *__restrict y,
                              int64_t k);

void nntr_dequantize_row_q6_K(const void *__restrict x, float *__restrict y,
                              int64_t k);

void nntr_dequantize_row_q8_0(const void *__restrict x, float *__restrict y,
                              int64_t k);

void nntr_dequantize_row_q8_K(const void *__restrict x, float *__restrict y,
                              int64_t k);

void nntr_vec_dot_q6_K_q8_K(int n, float *__restrict s, size_t bs,
                            const void *__restrict vx, size_t bx,
                            const void *__restrict vy, size_t by, int nrc);

#endif
