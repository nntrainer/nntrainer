// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   x86_compute_backend_fp16.cpp
 * @date   23 April 2024
 * @see    https://github.com/nntrainer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Compute backend for x86
 *
 */

#include <assert.h>
#include <avx2_impl.h>
#include <fallback_internal.h>
#include <nntrainer_error.h>
#include <tensor_dim.h>
#include <x86_compute_backend.h>
#ifdef USE_BLAS
#include <cblas_interface.h>
#endif
#include <ggml_interface.h>

#define ROW_MAJOR 0
#define COL_MAJOR 1

namespace nntrainer {

void shgemm(const unsigned int TStorageOrder, bool TransA, bool TransB,
            const unsigned int M, const unsigned int N, const unsigned int K,
            const float alpha, const float *A, const unsigned int lda,
            const _FP16 *B, const unsigned int ldb, const float beta, float *C,
            const unsigned int ldc) {
  float *B_ = new float[N * K];
  scopy(N * K, B, 1, B_, 1);

#ifdef USE_BLAS
  __cblas_sgemm(TStorageOrder, TransA, TransB, M, N, K, alpha, A, lda, B_, ldb,
                beta, C, ldc);
#else
  __fallback_sgemm(TStorageOrder, TransA, TransB, M, N, K, alpha, A, lda, B_,
                   ldb, beta, C, ldc);
#endif

  delete[] B_;
}

void shgemv(const unsigned int TStorageOrder, bool TransA, const unsigned int M,
            const unsigned int N, const float alpha, const float *A,
            const unsigned int lda, const _FP16 *X, const unsigned int incX,
            const float beta, float *Y, const unsigned int incY) {
  unsigned int lenX = (TransA) ? 1 + (M - 1) * (incX) : 1 + (N - 1) * (incX);
  unsigned int lenY = (TransA) ? 1 + (N - 1) * (incY) : 1 + (M - 1) * (incY);

  float *X_ = new float[lenX];

  scopy(lenX, X, 1, X_, 1);

#ifdef USE_BLAS
  __cblas_sgemv(TStorageOrder, TransA, M, N, alpha, A, lda, X_, incX, beta, Y,
                incY);
#else
  __fallback_sgemv(TStorageOrder, TransA, M, N, alpha, A, lda, X_, incX, beta,
                   Y, incY);
#endif

  delete[] X_;
}

void hsgemm(const unsigned int TStorageOrder, bool TransA, bool TransB,
            const unsigned int M, const unsigned int N, const unsigned int K,
            const float alpha, const _FP16 *A, const unsigned int lda,
            const float *B, const unsigned int ldb, const float beta, float *C,
            const unsigned int ldc) {
  float *A_ = new float[M * K];

  scopy(M * K, A, 1, A_, 1);

#ifdef USE_BLAS
  __cblas_sgemm(TStorageOrder, TransA, TransB, M, N, K, alpha, A_, lda, B, ldb,
                beta, C, ldc);
#else
  __fallback_sgemm(TStorageOrder, TransA, TransB, M, N, K, alpha, A_, lda, B,
                   ldb, beta, C, ldc);
#endif

  delete[] A_;
}

void hsgemv(const unsigned int TStorageOrder, bool TransA, const unsigned int M,
            const unsigned int N, const float alpha, const _FP16 *A,
            const unsigned int lda, const float *X, const unsigned int incX,
            const float beta, float *Y, const unsigned int incY) {
  unsigned int lenX = (TransA) ? 1 + (M - 1) * (incX) : 1 + (N - 1) * (incX);
  unsigned int lenY = (TransA) ? 1 + (N - 1) * (incY) : 1 + (M - 1) * (incY);

  float *A_ = new float[M * N];

  scopy(M * N, A, 1, A_, 1);

#ifdef USE_BLAS
  __cblas_sgemv(TStorageOrder, TransA, M, N, alpha, A_, lda, X, incX, beta, Y,
                incY);
#else
  __fallback_sgemv(TStorageOrder, TransA, M, N, alpha, A_, lda, X, incX, beta,
                   Y, incY);
#endif

  delete[] A_;
}

void sscal(const unsigned int N, const float alpha, _FP16 *X,
           const unsigned int incX) {
  __fallback_sscal(N, alpha, X, incX);
}

_FP16 snrm2(const unsigned int N, const _FP16 *X, const unsigned int incX) {
  assert(incX > 0);
  _FP16 sum = __fallback_snrm2(N, X, incX);
  return sum;
}

void scopy(const unsigned int N, const _FP16 *X, const unsigned int incX,
           _FP16 *Y, const unsigned int incY) {
  if (incX == 1 && incY == 1) {
    __fallback_scopy(N, X, incX, Y, incY);
  }
}

void scopy(const unsigned int N, const float *X, const unsigned int incX,
           _FP16 *Y, const unsigned int incY) {
  if (incX == 1 && incY == 1) {
    nntrainer::avx2::vcvt_f32_f16(N, X, Y);

  } else {
    __fallback_scopy(N, X, incX, Y, incY);
  }
}

void scopy(const unsigned int N, const _FP16 *X, const unsigned int incX,
           float *Y, const unsigned int incY) {
  if (incX == 1 && incY == 1) {
    nntrainer::avx2::vcvt_f16_f32(N, X, Y);
  } else {
    __fallback_scopy(N, X, incX, Y, incY);
  }
}

void scopy_int4_to_float16(const unsigned int N, const uint8_t *X,
                           const unsigned int incX, _FP16 *Y,
                           const unsigned int incY) {
  if (incX == 1 && incY == 1) {
    __fallback_scopy_int4_to_float16(N, X, incX, Y, incY);
  }
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
#ifdef USE_BLAS

  float *A_ = new float[M * K];
  float *B_ = new float[N * K];
  float *C_ = new float[M * N];

  scopy(M * K, A, 1, A_, 1);
  scopy(N * K, B, 1, B_, 1);
  scopy(M * N, C, 1, C_, 1);

  __cblas_sgemm(TStorageOrder, TransA, TransB, M, N, K, alpha, A_, lda, B_, ldb,
                beta, C_, ldc);

  scopy(M * N, C_, 1, C, 1);

  delete[] A_;
  delete[] B_;
  delete[] C_;
#else
  __fallback_sgemm(TStorageOrder, TransA, TransB, M, N, K, alpha, A, lda, B,
                   ldb, beta, C, ldc);
#endif
}

void sgemv(const unsigned int TStorageOrder, bool TransA, const unsigned int M,
           const unsigned int N, const float alpha, const _FP16 *A,
           const unsigned int lda, const _FP16 *X, const unsigned int incX,
           const float beta, _FP16 *Y, const unsigned int incY) {
#ifdef USE_BLAS
  unsigned int lenX = (TransA) ? 1 + (M - 1) * (incX) : 1 + (N - 1) * (incX);
  unsigned int lenY = (TransA) ? 1 + (N - 1) * (incY) : 1 + (M - 1) * (incY);

  float *A_ = new float[M * N];
  float *X_ = new float[lenX];
  float *Y_ = new float[lenY];

  scopy(M * N, A, 1, A_, 1);
  scopy(lenX, X, 1, X_, 1);
  scopy(lenY, Y, 1, Y_, 1);

  __cblas_sgemv(TStorageOrder, TransA, M, N, alpha, A_, lda, X_, incX, beta, Y_,
                incY);

  scopy(lenY, Y_, 1, Y, 1);

  delete[] A_;
  delete[] X_;
  delete[] Y_;

#else
  __fallback_sgemv(TStorageOrder, TransA, M, N, alpha, A, lda, X, incX, beta, Y,
                   incY);
#endif
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

bool is_valid(const unsigned int N, const _FP16 *input) {
  return nntrainer::avx2::is_valid(N, input);
}

void compute_rotary_embedding_value(unsigned int dim, unsigned int half_,
                                    unsigned int w, _FP16 *in, _FP16 *out,
                                    float *cos_, float *sin_) {
  __fallback_compute_rotary_embedding_value(dim, half_, w, in, out, cos_, sin_);
}

void swiglu(const unsigned int N, _FP16 *X, _FP16 *Y, _FP16 *Z) {
  __fallback_swiglu(N, X, Y, Z);
}

_FP16 max_val(const unsigned int N, _FP16 *X) { return __fallback_max(N, X); }

void softmax(const unsigned int N, _FP16 *X, _FP16 *Y) {
  __fallback_softmax(N, X, Y);
}

template <> void dequantize_row_q8_0(const void *x_raw, _FP16 *y, int64_t k) {
  __ggml_dequantize_row_q8_0(x_raw, y, k);
}

template <>
size_t quantize_q8_0(const _FP16 *src, void *dst, int64_t nrow,
                     int64_t n_per_row, const float *quant_weights) {
  return __ggml_quantize_q8_0(src, dst, nrow, n_per_row, quant_weights);
}

template <>
void gemm_q4_0(const unsigned int M, const unsigned int N, const unsigned int K,
               const _FP16 *A, const unsigned int lda, const void *B,
               const unsigned int ldb, _FP16 *C, const unsigned int ldc) {
  return __fallback_gemm_q4_0(M, N, K, A, lda, B, ldb, C, ldc);
}

template <> void quantize_row_q8_K(const _FP16 *src, void *dst, int64_t k) {
  __ggml_quantize_row_q8_K(src, dst, k);
}

template <> void dequantize_row_q8_K(const void *x, _FP16 *y, int64_t k) {
  __ggml_dequantize_row_q8_K(x, y, k);
}

template <>
void gemm_q6_K(const unsigned int M, const unsigned int N, const unsigned int K,
               const _FP16 *A, const unsigned int lda, const void *B,
               const unsigned int ldb, _FP16 *C, const unsigned int ldc) {
  return __ggml_gemm_q6_K<_FP16>(M, N, K, A, lda, B, ldb, C, ldc);
}

template <>
void rms_norm_wrt_width_fp16_intrinsic(const _FP16 *__restrict X,
                                       _FP16 *__restrict Y, size_t H, size_t W,
                                       float epsilon) {
  __fallback_rms_norm_wrt_width_fp16_intrinsic<_FP16>(X, Y, H, W, epsilon);
}

} /* namespace nntrainer */
