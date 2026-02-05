// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file fallback.cpp
 * @date   23 April 2024
 * @see    https://github.com/nntrainer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Fallback interface (Raw implementations)
 *
 */

#include <assert.h>
#include <fallback_internal.h>
#include <nntrainer_error.h>

namespace nntrainer {
void shgemm(const unsigned int TStorageOrder, bool TransA, bool TransB,
            const unsigned int M, const unsigned int N, const unsigned int K,
            const float alpha, const float *A, const unsigned int lda,
            const _FP16 *B, const unsigned int ldb, const float beta, float *C,
            const unsigned int ldc) {
  __fallback_shgemm(TStorageOrder, TransA, TransB, M, N, K, alpha, A, lda, B,
                    ldb, beta, C, ldc);
}

void shgemv(const unsigned int TStorageOrder, bool TransA, const unsigned int M,
            const unsigned int N, const float alpha, const float *A,
            const unsigned int lda, const _FP16 *X, const unsigned int incX,
            const float beta, float *Y, const unsigned int incY) {
  __fallback_shgemv(TStorageOrder, TransA, M, N, alpha, A, lda, X, incX, beta,
                    Y, incY);
}

void hsgemm(const unsigned int TStorageOrder, bool TransA, bool TransB,
            const unsigned int M, const unsigned int N, const unsigned int K,
            const float alpha, const _FP16 *A, const unsigned int lda,
            const float *B, const unsigned int ldb, const float beta, float *C,
            const unsigned int ldc) {
  __fallback_hsgemm(TStorageOrder, TransA, TransB, M, N, K, alpha, A, lda, B,
                    ldb, beta, C, ldc);
}

void hsgemv(const unsigned int TStorageOrder, bool TransA, const unsigned int M,
            const unsigned int N, const float alpha, const _FP16 *A,
            const unsigned int lda, const float *X, const unsigned int incX,
            const float beta, float *Y, const unsigned int incY) {
  __fallback_hsgemv(TStorageOrder, TransA, M, N, alpha, A, lda, X, incX, beta,
                    Y, incY);
}

void quantize_row_q8_0(const _FP16 *__restrict src, void *__restrict dst,
                       int64_t k) {
  __fallback_quantize_row_q8_0(src, dst, k);
}

size_t quantize_q8_0(const _FP16 *src, void *dst, int64_t nrow,
                     int64_t n_per_row, const float *quant_weights) {
  return __fallback_quantize_q8_0(src, dst, nrow, n_per_row, quant_weights);
}

void dequantize_row_q8_0(const void *x_raw, _FP16 *y, int64_t k) {
  __fallback_dequantize_row_q8_0(x_raw, y, k);
}

void sscal(const unsigned int N, const float alpha, _FP16 *X,
           const unsigned int incX) {
  __fallback_sscal(N, alpha, X, incX);
}

_FP16 snrm2(const unsigned int N, const _FP16 *X, const unsigned int incX) {
  assert(incX > 0);
  _FP16 sum = 0;
  _FP16 tmp;
  sum = __fallback_snrm2(N, X, incX);
  return sum;
}

void scopy(const unsigned int N, const _FP16 *X, const unsigned int incX,
           _FP16 *Y, const unsigned int incY) {
  __fallback_scopy(N, X, incX, Y, incY);
}

void scopy(const unsigned int N, const float *X, const unsigned int incX,
           _FP16 *Y, const unsigned int incY) {
  __fallback_scopy(N, X, incX, Y, incY);
}

void scopy(const unsigned int N, const _FP16 *X, const unsigned int incX,
           float *Y, const unsigned int incY) {
  __fallback_scopy(N, X, incX, Y, incY);
}

void scopy_int4_to_float16(const unsigned int N, const uint8_t *X,
                           const unsigned int incX, _FP16 *Y,
                           const unsigned int incY) {
  __fallback_scopy_int4_to_float16(N, X, incX, Y, incY);
}

void scopy_int8_to_float16(const unsigned int N, const uint8_t *X,
                           const unsigned int incX, _FP16 *Y,
                           const unsigned int incY) {
  __fallback_scopy_int8_to_float16(N, X, incX, Y, incY);
}

void scopy_int8_to_float16(const unsigned int N, const int8_t *X,
                           const unsigned int incX, _FP16 *Y,
                           const unsigned int incY) {
  __fallback_scopy_int8_to_float16(N, X, incX, Y, incY);
}

_FP16 sdot(const unsigned int N, const _FP16 *X, const unsigned int incX,
           const _FP16 *Y, const unsigned int incY) {
  assert(incX > 0 && incY > 0);
  _FP16 ret = 0;
  return __fallback_sdot(N, X, incX, Y, incY);
}

void saxpy(const unsigned int N, const float alpha, const _FP16 *X,
           const unsigned int incX, _FP16 *Y, const unsigned int incY) {
  __fallback_saxpy(N, alpha, X, incX, Y, incY);
}

void sgemm(const unsigned int TStorageOrder, bool TransA, bool TransB,
           const unsigned int M, const unsigned int N, const unsigned int K,
           const float alpha, const _FP16 *A, const unsigned int lda,
           const _FP16 *B, const unsigned int ldb, const float beta, _FP16 *C,
           const unsigned int ldc) {
  __fallback_sgemm(TStorageOrder, TransA, TransB, M, N, K, alpha, A, lda, B,
                   ldb, beta, C, ldc);
}

void sgemv(const unsigned int TStorageOrder, bool TransA, const unsigned int M,
           const unsigned int N, const float alpha, const _FP16 *A,
           const unsigned int lda, const _FP16 *X, const unsigned int incX,
           const float beta, _FP16 *Y, const unsigned int incY) {
  __fallback_sgemv(TStorageOrder, TransA, M, N, alpha, A, lda, X, incX, beta, Y,
                   incY);
}

void ele_mul(const unsigned int N, const _FP16 *X, const _FP16 *Y, _FP16 *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  __fallback_ele_mul(N, X, Y, Z, alpha, beta, i_stride, o_stride);
}

void ele_add(const unsigned int N, const _FP16 *X, const _FP16 *Y, _FP16 *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  __fallback_ele_add(N, X, Y, Z, alpha, beta, i_stride, o_stride);
}

void ele_sub(const unsigned N, const _FP16 *X, const _FP16 *Y, _FP16 *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  __fallback_ele_sub(N, X, Y, Z, alpha, beta, i_stride, o_stride);
}

void ele_div(const unsigned N, const _FP16 *X, const _FP16 *Y, _FP16 *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  __fallback_ele_div(N, X, Y, Z, alpha, beta, i_stride, o_stride);
}

unsigned int isamax(const unsigned int N, const _FP16 *X,
                    const unsigned int incX) {
  unsigned int max_idx = 0;
  max_idx = __fallback_isamax(N, X, incX);
  return max_idx;
}

void inv_sqrt_inplace(const unsigned int N, _FP16 *X) {
  __fallback_inv_sqrt_inplace(N, X);
}

void transpose_matrix(const unsigned int M, const unsigned int N,
                      const _FP16 *src, unsigned int ld_src, _FP16 *dst,
                      unsigned int ld_dst) {
  __fallback_transpose_matrix(M, N, src, ld_src, dst, ld_dst);
}

bool is_valid(const unsigned int N, const _FP16 *X) {
  return __fallback_isValid(N, X);
}

void compute_rotary_embedding_value(unsigned int dim, unsigned int half_,
                                    unsigned int w, _FP16 *in, _FP16 *out,
                                    float *cos_, float *sin_) {
  __fallback_compute_rotary_embedding_value(dim, half_, w, in, out, cos_, sin_);
}

void swiglu(const unsigned int N, _FP16 *X, _FP16 *Y, _FP16 *Z) {
  __fallback_swiglu(N, X, Y, Z);
}

void tanh_gelu(const unsigned int N, const _FP16 *X, _FP16 *Y) {
  __fallback_tanh_gelu(N, X, Y);
}

_FP16 max_val(const unsigned int N, _FP16 *X) { return __fallback_max(N, X); }

void softmax(const unsigned int N, _FP16 *X, _FP16 *Y) {
  __fallback_softmax(N, X, Y);
}

template <>
void gemm_q4_0(const unsigned int M, const unsigned int N, const unsigned int K,
               const _FP16 *A, const unsigned int lda, const void *B,
               const unsigned int ldb, _FP16 *C, const unsigned int ldc) {
  return __fallback_gemm_q4_0<_FP16>(M, N, K, A, lda, B, ldb, C, ldc);
}

template <> void dequantize_row_q8_K(const void *x, _FP16 *y, int64_t k) {
  return __fallback_dequantize_row_q8_K(x, y, k);
}

template <> void quantize_row_q8_K(const void *x, _FP16 *y, int64_t k) {
  return __fallback_quantize_row_q8_K(x, y, k);
}

template <>
void gemm_q6_K(const unsigned int M, const unsigned int N, const unsigned int K,
               const _FP16 *A, const unsigned int lda, const void *B,
               const unsigned int ldb, _FP16 *C, const unsigned int ldc) {
  return __fallback_gemm_q6_K<_FP16>(M, N, K, A, lda, B, ldb, C, ldc);
}

template <>
void rms_norm_wrt_width_fp16_intrinsic(const _FP16 *__restrict X,
                                       _FP16 *__restrict Y, size_t H, size_t W,
                                       float epsilon) {
  __fallback_rms_norm_wrt_width_fp16_intrinsic<_FP16>(X, Y, H, W, epsilon);
}

template <>
void clamp(const _FP16 *input, _FP16 *output, size_t length, _FP16 lower_bound,
           _FP16 upper_bound) {
  __fallback_clamp<_FP16>(input, output, length, lower_bound, upper_bound);
}

void nntr_quant_qs4cx_f32(size_t n, size_t k, void *rhs_native_mtx_f32,
                          void *rhs_native_mtx_qs4cx, void *rhs_scales_f32,
                          bool transB) {
  __fallback_nntr_quant_qs4cx_f32(n, k, rhs_native_mtx_f32,
                                  rhs_native_mtx_qs4cx, rhs_scales_f32, transB);
}

template <>
uint32_t nntr_gemm_qai8dxp_qsi4cxp_unpacked(
  size_t m, size_t n, size_t k, void *lhs_native_mtx_f32,
  void *rhs_native_mtx_qs4cx, void *rhs_scales_f32, float *dst_mtx_f32,
  bool transB, float lower_bound, float upper_bound) {
  return __fallback_nntr_gemm_qai8dxp_qsi4cxp_unpacked(
    m, n, k, lhs_native_mtx_f32, rhs_native_mtx_qs4cx, rhs_scales_f32,
    dst_mtx_f32, transB, lower_bound, upper_bound);
}

void nntr_quant_qs4cx_f32(size_t n, size_t k, void *rhs_native_mtx_f32,
                          void *rhs_native_mtx_qs4cx, void *rhs_scales_f32,
                          bool transB = true) {
  __fallback_nntr_quant_qs4cx_f32(n, k, rhs_native_mtx_f32,
                                  rhs_native_mtx_qs4cx, rhs_scales_f32, transB);
}

size_t nntr_get_rhs_packed_size_qsi4cxp_qs4cxs1s0(size_t n, size_t k,
                                                  uint32_t idx_variant,
                                                  bool transB) {
  return __fallback_nntr_get_rhs_packed_size_qsi4cxp_qs4cxs1s0(
    n, k, idx_variant, transB);
}

void nntr_qsi4cxp_qs4cxs1s0_rhs_pack(size_t n, size_t k,
                                     void *rhs_packed_mtx_qs4cx,
                                     void *rhs_native_mtx_qs4cx,
                                     void *rhs_scales_f32, uint32_t idx_variant,
                                     bool transB) {
  __fallback_nntr_qsi4cxp_qs4cxs1s0_rhs_pack(
    n, k, rhs_packed_mtx_qs4cx, rhs_native_mtx_qs4cx, rhs_scales_f32,
    idx_variant, transB);
}

template <>
void nntr_gemm_qai8dxp_qsi4cxp_packed(size_t m, size_t n, size_t k,
                                      void *lhs_native_mtx_f32,
                                      void *rhs_packed_mtx_qs4cx,
                                      float *dst_act_mtx_f32,
                                      uint32_t idx_variant, bool transB,
                                      float lower_bound, float upper_bound) {
  __fallback_nntr_gemm_qai8dxp_qsi4cxp_packed(
    m, n, k, lhs_native_mtx_f32, rhs_packed_mtx_qs4cx, dst_act_mtx_f32,
    idx_variant, transB, lower_bound, upper_bound);
}

} /* namespace nntrainer */
