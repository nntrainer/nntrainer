// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file fallback.cpp
 * @date   23 April 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Fallback interface (Raw implementations)
 *
 */

#include <assert.h>
#include <fallback_internal.h>
#include <nntrainer_error.h>

namespace nntrainer {

void scopy_int4_to_float32(const unsigned int N, const uint8_t *X,
                           const unsigned int incX, float *Y,
                           const unsigned int incY) {
  __fallback_scopy_int4_to_float32(N, X, incX, Y, incY);
}

void scopy_int8_to_float32(const unsigned int N, const uint8_t *X,
                           const unsigned int incX, float *Y,
                           const unsigned int incY) {
  __fallback_scopy_uint8_to_float32(N, X, incX, Y, incY);
}

template <>
void sine(const unsigned int N, float *X, float *Y, float alpha, float beta) {
  __fallback_sine(N, X, Y, alpha, beta);
}

template <>
void cosine(const unsigned int N, float *X, float *Y, float alpha, float beta) {
  __fallback_cosine(N, X, Y, alpha, beta);
}

void inv_sqrt_inplace(const unsigned int N, float *X) {
  __fallback_inv_sqrt_inplace(N, X);
}

void ele_mul(const unsigned int N, const float *X, const float *Y, float *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  __fallback_ele_mul(N, X, Y, Z, alpha, beta, i_stride, o_stride);
}

void ele_add(const unsigned int N, const float *X, const float *Y, float *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  __fallback_ele_add(N, X, Y, Z, alpha, beta, i_stride, o_stride);
}

void ele_sub(const unsigned N, const float *X, const float *Y, float *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  __fallback_ele_sub(N, X, Y, Z, alpha, beta, i_stride, o_stride);
}

void ele_div(const unsigned N, const float *X, const float *Y, float *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  __fallback_ele_div(N, X, Y, Z, alpha, beta, i_stride, o_stride);
}

void saxpy(const unsigned int N, const float alpha, const float *X,
           const unsigned int incX, float *Y, const unsigned int incY) {
  __fallback_saxpy(N, alpha, X, incX, Y, incY);
}

void sgemv(const unsigned int TStorageOrder, bool TransA, const unsigned int M,
           const unsigned int N, const float alpha, const float *A,
           const unsigned int lda, const float *X, const unsigned int incX,
           const float beta, float *Y, const unsigned int incY) {
  __fallback_sgemv(TStorageOrder, TransA, M, N, alpha, A, lda, Y, incX, beta, Y,
                   incY);
}

float sdot(const unsigned int N, const float *X, const unsigned int incX,
           const float *Y, const unsigned int incY) {
  return __fallback_sdot(N, X, incX, Y, incY);
}

void scopy(const unsigned int N, const uint8_t *X, const unsigned int incX,
           uint8_t *Y, const unsigned int incY) {
  __fallback_scopy(N, X, incX, Y, incY);
}

void scopy(const unsigned int N, const int8_t *X, const unsigned int incX,
           int8_t *Y, const unsigned int incY) {
  __fallback_scopy(N, X, incX, Y, incY);
}

void scopy(const unsigned int N, const float *X, const unsigned int incX,
           float *Y, const unsigned int incY) {
  __fallback_scopy(N, X, incX, Y, incY);
}

void sscal(const unsigned int N, const float alpha, float *X,
           const unsigned int incX) {
  __fallback_sscal(N, alpha, X, incX);
}

float snrm2(const unsigned int N, const float *X, const unsigned int incX) {
  return __fallback_snrm2(N, X, incX);
}

void sgemm(const unsigned int TStorageOrder, bool TransA, bool TransB,
           const unsigned int M, const unsigned int N, const unsigned int K,
           const float alpha, const float *A, const unsigned int lda,
           const float *B, const unsigned int ldb, const float beta, float *C,
           const unsigned int ldc) {
  __fallback_sgemm(TStorageOrder, TransA, TransB, M, N, K, alpha, A, lda, B,
                   ldb, beta, C, ldc);
}

unsigned int isamax(const unsigned int N, const float *X,
                    const unsigned int incX) {
  return __fallback_isamax(N, X, incX);
}

void transpose_matrix(const unsigned int M, const unsigned int N,
                      const float *src, unsigned int ld_src, float *dst,
                      unsigned int ld_dst) {
  __fallback_transpose_matrix(M, N, src, ld_src, dst, ld_dst);
}

bool is_valid(const unsigned int N, const float *X) {
  return __fallback_isValid(N, X);
}

void scopy_int8_to_float32(const unsigned int N, const int8_t *X,
                           const unsigned int incX, float *Y,
                           const unsigned int incY) {
  __fallback_scopy_int8_to_float32(N, X, incX, Y, incY);
}

void copy_s16_fp32(const unsigned int N, const int16_t *X, float *Y) {
  __fallback_copy_s16_fp32(N, X, Y);
}

void copy_u16_fp32(const unsigned int N, const uint16_t *X, float *Y) {
  __fallback_copy_u16_fp32(N, X, Y);
}

void copy_fp32_u32(const unsigned int N, const float *X, uint32_t *Y) {
  __fallback_copy_fp32_u32(N, X, Y);
}

void copy_fp32_u16(const unsigned int N, const float *X, uint16_t *Y) {
  __fallback_copy_fp32_u16(N, X, Y);
}

void copy_fp32_u8(const unsigned int N, const float *X, uint8_t *Y) {
  __fallback_copy_fp32_u8(N, X, Y);
}

void copy_fp32_s16(const unsigned int N, const float *X, int16_t *Y) {
  __fallback_copy_fp32_s16(N, X, Y);
}

void copy_fp32_s8(const unsigned int N, const float *X, int8_t *Y) {
  __fallback_copy_fp32_s8(N, X, Y);
}

void copy_s16(const unsigned int N, const int16_t *X, int16_t *Y) {
  __fallback_copy_s16(N, X, Y);
}

void copy_u16(const unsigned int N, const uint16_t *X, uint16_t *Y) {
  __fallback_copy_u16(N, X, Y);
}

void unpack_q4_0x8_transpose16(const void *src, uint16_t *d_out,
                               uint16_t *qs_out, int N, int K) {
  __fallback_unpack_q4_0x8_transpose16(src, d_out, qs_out, N, K);
}

template <>
void calc_trigonometric_vals_dup(unsigned int N_half, float *angle, float *cos_,
                                 float *sin_, unsigned int from,
                                 float attention_scaling) {
  __fallback_calc_trigonometric_vals_dup(N_half, angle, cos_, sin_, from,
                                         attention_scaling);
}

void swiglu(const unsigned int N, float *X, float *Y, float *Z) {
  __fallback_swiglu(N, X, Y, Z);
}

void swiglu(const unsigned int N, float *X, float *Y, float *Z, float alpha) {
  __fallback_swiglu(N, X, Y, Z, alpha);
}

float max_val(const unsigned int N, float *X) { return __fallback_max(N, X); }

void softmax(const unsigned int N, float *X, float *Y) {
  __fallback_softmax(N, X, Y);
}

template <>
void gemm_q4_0(const unsigned int M, const unsigned int N, const unsigned int K,
               const float *A, const unsigned int lda, const void *B,
               const unsigned int ldb, float *C, const unsigned int ldc) {
  return __fallback_gemm_q4_0<float>(M, N, K, A, lda, B, ldb, C, ldc);
}

void gemm_q4_K(const unsigned int M, const unsigned int N, const unsigned int K,
               const float *A, const unsigned int lda, const void *B,
               const unsigned int ldb, float *C, const unsigned int ldc) {
  return __fallback_gemm_q4_K(M, N, K, A, lda, B, ldb, C, ldc);
}

template <>
void gemm_q6_K(const unsigned int M, const unsigned int N, const unsigned int K,
               const float *A, const unsigned int lda, const void *B,
               const unsigned int ldb, float *C, const unsigned int ldc) {
  return __fallback_gemm_q6_K(M, N, K, A, lda, B, ldb, C, ldc);
}

float dot_q6_K_q8_K(const unsigned int K, const void *v_q6_K,
                    const void *v_q8_K) {
  return __fallback_dot_q6_K_q8_K(K, v_q6_K, v_q8_K);
}

float dot_q6_K_f32(const unsigned int K, const void *v_q6_K, const float *f) {
  return __fallback_dot_q6_K_f32(K, v_q6_K, f);
}

size_t quantize_q4_0(const float *src, void *dst, int64_t nrow,
                     int64_t n_per_row, const float *quant_weights) {
  return __fallback_quantize_q4_0(src, dst, nrow, n_per_row, quant_weights);
}

size_t quantize_q4_K(const float *src, void *dst, int64_t nrow,
                     int64_t n_per_row, const float *quant_weights) {
  return __fallback_quantize_q4_K(src, dst, nrow, n_per_row, quant_weights);
}

size_t quantize_q6_K(const float *src, void *dst, int64_t nrow,
                     int64_t n_per_row, const float *quant_weights) {
  return __fallback_quantize_q6_K(src, dst, nrow, n_per_row, quant_weights);
}

void dequantize_row_q4_K(const void *x_raw, float *y, int64_t k) {
  return __fallback_dequantize_row_q4_K(x_raw, y, k);
}

void dequantize_row_q4_0(const void *x_raw, float *y, int64_t k) {
  return __fallback_dequantize_row_q4_0(x_raw, y, k);
}

void dequantize_row_q6_K(const void *x, float *y, int64_t k) {
  return __fallback_dequantize_row_q6_K(x, y, k);
}

template <> void dequantize_row_q8_K(const void *x, float *y, int64_t k) {
  return __fallback_dequantize_row_q8_K(x, y, k);
}

template <> void quantize_row_q8_K(const void *x, float *y, int64_t k) {
  return __fallback_quantize_row_q8_K(x, y, k);
}

void repack_q4_0(void *W, void *repacked_W, size_t data_size,
                 const unsigned int M, const unsigned int N) {
  return __fallback_repack_q4_0_to_q4_0_8(W, repacked_W, data_size, M, N);
}

void repack_q4_0_to_q4_0_8(void *W, void *repacked_W, size_t data_size,
                           const unsigned int M, const unsigned int N) {
  return __fallback_repack_q4_0_to_q4_0_8(W, repacked_W, data_size, M, N);
}

void repack_q4_K_to_q4_K_8(void *W, void *repacked_W, size_t data_size,
                           const unsigned int M, const unsigned int N) {
  return __fallback_repack_q4_K_to_q4_K_8(W, repacked_W, data_size, M, N);
}

void unpack_q4_0(const void *in_q4_0x, void *out_q4_0, size_t data_size,
                 const unsigned int M, const unsigned int N) {
  __fallback_unpack_q4_0_8_to_q4_0(in_q4_0x, out_q4_0, data_size, M, N);
}

template <>
void softmax_row_inplace(float *qk_out, size_t start_row, size_t end_row,
                         size_t num_heads, float *sink) {
  __fallback_softmax_row_inplace(qk_out, start_row, end_row, num_heads);
}

template <>
void softmax_row(float *qk_out, size_t start_row, size_t end_row,
                 size_t num_heads, float *sink) {
  __fallback_softmax_row(qk_out, start_row, end_row, num_heads);
}

void compute_fp16vcache_fp32_transposed(int row_num, const float *in,
                                        const uint16_t *vcache, float *output,
                                        int num_cache_head, int gqa_size,
                                        int head_dim,
                                        size_t local_window_size) {
  __fallback_compute_fp16vcache_fp32_transposed(row_num, in, vcache, output,
                                                num_cache_head, gqa_size,
                                                head_dim, local_window_size);
}

template <>
void compute_kcaches(const float *in, const uint16_t *kcache, float *output,
                     int num_rows, int num_cache_head, int head_dim,
                     int gqa_size, int tile_size, size_t local_window_size) {
  __fallback_compute_kcaches<uint16_t>(in, kcache, output, num_rows,
                                       num_cache_head, head_dim, gqa_size,
                                       tile_size, local_window_size);
}

void compute_rotary_emb_value(unsigned int width, unsigned int dim,
                              unsigned int half_, float *inout, void *output,
                              const float *cos_, const float *sin_,
                              bool only_convert_to_fp16) {
  __fallback_compute_rotary_emb_value(width, dim, half_, inout, output, cos_,
                                      sin_, only_convert_to_fp16);
}

void rms_norm_wrt_width_fp32_intrinsic(const float *__restrict X,
                                       float *__restrict Y, size_t H, size_t W,
                                       float epsilon) {
  __fallback_rms_norm_wrt_width_fp32_intrinsic(X, Y, H, W, epsilon);
}

template <>
void rms_norm_wrt_width_fp16_intrinsic(const float *__restrict X,
                                       float *__restrict Y, size_t H, size_t W,
                                       float epsilon) {
  __fallback_rms_norm_wrt_width_fp16_intrinsic(X, Y, H, W, epsilon);
}

template <>
void clamp(const float *input, float *output, size_t length, float lower_bound,
           float upper_bound) {
  __fallback_clamp(input, output, length, lower_bound, upper_bound);
}

void create_q4_0_weights(const uint8_t *int4_weight, uint8_t *q4_0_weight) {
  __fallback_create_q4_0_weights(int4_weight, q4_0_weight);
}

void transform_int4_osv32_isv2_to_q4_0(size_t N, size_t K,
                                       const uint8_t *osv32_weights,
                                       const uint16_t *osv32_scales,
                                       size_t scale_group_size,
                                       void *dst_q4_0x) {
  __fallback_transform_int4_osv32_isv2_to_q4_0(
    N, K, osv32_weights, osv32_scales, scale_group_size, dst_q4_0x);
}

} /* namespace nntrainer */
