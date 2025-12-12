// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file arm_compute_backend_fp16.cpp
 * @date   23 April 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Compute backend for arm
 *
 */
#include <arm_compute_backend.h>
#include <assert.h>
#include <fallback_internal.h>
#include <fallback_kleidiai.h>
#include <ggml_interface.h>
#include <kleidiai_interface.h>
#include <neon_impl.h>
#include <nntrainer_error.h>
#ifdef USE_BLAS
#include <cblas_interface.h>
#endif

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
  assert(incX > 0);

  if (incX == 1) {
    nntrainer::neon::hscal(N, X, alpha);
  } else {
    __fallback_sscal(N, alpha, X, incX);
  }
}

_FP16 snrm2(const unsigned int N, const _FP16 *X, const unsigned int incX) {
  assert(incX > 0);
  _FP16 sum;
  if (incX == 1) {
    sum = nntrainer::neon::hnrm2(N, X);
  } else {
    sum = __fallback_snrm2(N, X, incX);
  }
  return sum;
}

void scopy(const unsigned int N, const _FP16 *X, const unsigned int incX,
           _FP16 *Y, const unsigned int incY) {
  if (incX == 1 && incY == 1) {
    nntrainer::neon::hcopy(N, X, Y);
  } else {
    __fallback_scopy(N, X, incX, Y, incY);
  }
}

void scopy(const unsigned int N, const float *X, const unsigned int incX,
           _FP16 *Y, const unsigned int incY) {
  if (incX == 1 && incY == 1) {
    nntrainer::neon::copy_fp32_to_fp16(N, X, Y);
  } else {
    __fallback_scopy(N, X, incX, Y, incY);
  }
}

void scopy(const unsigned int N, const _FP16 *X, const unsigned int incX,
           float *Y, const unsigned int incY) {
  if (incX == 1 && incY == 1) {
    nntrainer::neon::copy_fp16_to_fp32(N, X, Y);
  } else {
    __fallback_scopy(N, X, incX, Y, incY);
  }
}

void scopy_int4_to_float16(const unsigned int N, const uint8_t *X,
                           const unsigned int incX, _FP16 *Y,
                           const unsigned int incY) {
  if (incX == 1 && incY == 1) {
    nntrainer::neon::copy_int4_to_fp16(N, X, Y);
  } else {
    __fallback_scopy_int4_to_float16(N, X, incX, Y, incY);
  }
}

void scopy_int8_to_float16(const unsigned int N, const uint8_t *X,
                           const unsigned int incX, _FP16 *Y,
                           const unsigned int incY) {
  if (incX == 1 && incY == 1) {
    nntrainer::neon::copy_int8_to_fp16(N, X, Y);
  } else {
    __fallback_scopy_int8_to_float16(N, X, incX, Y, incY);
  }
}

void scopy_int8_to_float16(const unsigned int N, const int8_t *X,
                           const unsigned int incX, _FP16 *Y,
                           const unsigned int incY) {
  if (incX == 1 && incY == 1) {
    nntrainer::neon::copy_int8_to_fp16(N, X, Y);
  } else {
    __fallback_scopy_int8_to_float16(N, X, incX, Y, incY);
  }
}

_FP16 sdot(const unsigned int N, const _FP16 *X, const unsigned int incX,
           const _FP16 *Y, const unsigned int incY) {
  _FP16 ret = 0;
  assert(incX > 0 && incY > 0);
  if (incX == 1 && incY == 1) {
    ret = nntrainer::neon::hdot(N, X, Y);
  } else {
    __fallback_sdot(N, X, incX, Y, incY);
  }
  return ret;
}

void saxpy(const unsigned int N, const float alpha, const _FP16 *X,
           const unsigned int incX, _FP16 *Y, const unsigned int incY) {
  if (incX == 1 && incY == 1) {
    nntrainer::neon::haxpy(N, alpha, X, Y);
  } else {
    __fallback_saxpy(N, alpha, X, incX, Y, incY);
  }
}

void sgemm(const unsigned int TStorageOrder, bool TransA, bool TransB,
           const unsigned int M, const unsigned int N, const unsigned int K,
           const float alpha, const _FP16 *A, const unsigned int lda,
           const _FP16 *B, const unsigned int ldb, const float beta, _FP16 *C,
           const unsigned int ldc) {
  if (TStorageOrder) {
    __fallback_sgemm(TStorageOrder, TransA, TransB, M, N, K, alpha, A, lda, B,
                     ldb, beta, C, ldc);
  } else {
    nntrainer::neon::custom_hgemm(A, B, C, M, N, K, alpha, beta, TransA,
                                  TransB);
  }
}

void sgemv(const unsigned int TStorageOrder, bool TransA, const unsigned int M,
           const unsigned int N, const float alpha, const _FP16 *A,
           const unsigned int lda, const _FP16 *X, const unsigned int incX,
           const float beta, _FP16 *Y, const unsigned int incY) {
  if (TStorageOrder) {
    __fallback_sgemv(TStorageOrder, TransA, M, N, alpha, A, lda, X, incX, beta,
                     Y, incY);
  } else {
    if (TransA) {
      nntrainer::neon::hgemv_transpose(A, X, Y, M, N, alpha, beta);
    } else {
      nntrainer::neon::hgemv(A, X, Y, M, N, alpha, beta);
    }
  }
}

void ele_mul(const unsigned int N, const _FP16 *X, const _FP16 *Y, _FP16 *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  if (i_stride == 1 && o_stride == 1) {
    nntrainer::neon::ele_mul(N, X, Y, Z, alpha, beta);
  } else
    __fallback_ele_mul(N, X, Y, Z, alpha, beta, i_stride, o_stride);
}

void ele_add(const unsigned int N, const _FP16 *X, const _FP16 *Y, _FP16 *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  if (i_stride == 1 && o_stride == 1) {
    nntrainer::neon::ele_add(N, X, Y, Z, alpha, beta);
  } else
    __fallback_ele_add(N, X, Y, Z, alpha, beta, i_stride, o_stride);
}

void ele_sub(const unsigned N, const _FP16 *X, const _FP16 *Y, _FP16 *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  if (i_stride == 1 && o_stride == 1) {
    nntrainer::neon::ele_sub(N, X, Y, Z, alpha, beta);
  } else
    __fallback_ele_sub(N, X, Y, Z, alpha, beta, i_stride, o_stride);
}

void ele_div(const unsigned N, const _FP16 *X, const _FP16 *Y, _FP16 *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  if (i_stride == 1 && o_stride == 1) {
    nntrainer::neon::ele_div(N, X, Y, Z, alpha, beta);
  } else
    __fallback_ele_div(N, X, Y, Z, alpha, beta, i_stride, o_stride);
}

unsigned int isamax(const unsigned int N, const _FP16 *X,
                    const unsigned int incX) {
  unsigned int max_idx = 0;
  if (incX == 1 && N >= 8) {
    max_idx = nntrainer::neon::isamax(N, X);
  } else {
    max_idx = __fallback_isamax(N, X, incX);
  }
  return max_idx;
}

void inv_sqrt_inplace(const unsigned int N, _FP16 *X) {
  nntrainer::neon::inv_sqrt_inplace(N, X);
}

void transpose_matrix(const unsigned int M, const unsigned int N,
                      const __fp16 *src, unsigned int ld_src, __fp16 *dst,
                      unsigned int ld_dst) {
  nntrainer::neon::transpose_matrix(M, N, src, ld_src, dst, ld_dst);
}

bool is_valid(const unsigned int N, const __fp16 *input) {
  return nntrainer::neon::is_valid(N, input);
}

void compute_rotary_embedding_value(unsigned int dim, unsigned int half_,
                                    unsigned int w, _FP16 *in, _FP16 *out,
                                    float *cos_, float *sin_) {
  nntrainer::neon::compute_rotary_embedding_value(dim, half_, w, in, out, cos_,
                                                  sin_);
}

void swiglu(const unsigned int N, _FP16 *X, _FP16 *Y, _FP16 *Z) {
  nntrainer::neon::swiglu(N, X, Y, Z);
}

_FP16 max_val(const unsigned int N, _FP16 *X) {
  return nntrainer::neon::max_val(N, X);
}

void softmax(const unsigned int N, _FP16 *X, _FP16 *Y) {
  nntrainer::neon::softmax(N, X, Y);
}

template <>
size_t quantize_q8_0(const _FP16 *src, void *dst, int64_t nrow,
                     int64_t n_per_row, const float *quant_weights) {
  return __ggml_quantize_q8_0(src, dst, nrow, n_per_row, quant_weights);
}

template <> void dequantize_row_q8_0(const void *x_raw, _FP16 *y, int64_t k) {
  __ggml_dequantize_row_q8_0(x_raw, y, k);
}

template <>
void gemm_q4_0(const unsigned int M, const unsigned int N, const unsigned int K,
               const _FP16 *A, const unsigned int lda, const void *B,
               const unsigned int ldb, _FP16 *C, const unsigned int ldc) {
  return __ggml_q4_0_4x8_q8_0_GEMM<_FP16>(M, N, K, A, lda, B, ldb, C, ldc);
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
void softmax_row_inplace(_FP16 *qk_out, size_t start_row, size_t end_row,
                         size_t num_heads, _FP16 *sink) {
  nntrainer::neon::softmax_row_inplace<_FP16>(qk_out, start_row, end_row,
                                              num_heads, sink);
}

template <>
void softmax_row(_FP16 *qk_out, size_t start_row, size_t end_row,
                 size_t num_heads, _FP16 *sink) {
  nntrainer::neon::softmax_row<_FP16>(qk_out, start_row, end_row, num_heads,
                                      sink);
}

// Overloaded function for __fp16 with float sink (mixed precision)
void softmax_row_inplace(__fp16 *qk_out, size_t start_row, size_t end_row,
                         size_t num_heads, float *sink) {
  neon::softmax_row_inplace(qk_out, start_row, end_row, num_heads, sink);
}

void softmax_row(__fp16 *qk_out, size_t start_row, size_t end_row,
                 size_t num_heads, float *sink) {
  neon::softmax_row(qk_out, start_row, end_row, num_heads, sink);
}

void compute_fp16vcache_transposed(int row_num, const _FP16 *in,
                                   const _FP16 *vcache, _FP16 *output,
                                   int num_cache_head, int gqa_size,
                                   int head_dim, int chunk_size,
                                   size_t local_window_size) {
  neon::compute_fp16vcache_transposed(row_num, in, vcache, output,
                                      num_cache_head, gqa_size, head_dim,
                                      chunk_size, local_window_size);
}

void compute_kcaches(const _FP16 *in, const _FP16 *kcache, _FP16 *output,
                     int num_rows, int num_cache_head, int head_dim,
                     int gqa_size, int tile_off, int tile_size,
                     size_t local_window_size) {
  nntrainer::neon::compute_kcaches(in, kcache, output, num_rows, num_cache_head,
                                   head_dim, gqa_size, tile_off, tile_size,
                                   local_window_size);
}

void compute_rotary_emb_value(unsigned int width, unsigned int dim,
                              unsigned int half_, _FP16 *inout, _FP16 *output,
                              const _FP16 *cos_, const _FP16 *sin_) {
  neon::compute_rotary_emb_value(width, dim, half_, inout, output, cos_, sin_);
}

template <>
void rms_norm_wrt_width_fp16_intrinsic(const _FP16 *__restrict X,
                                       _FP16 *__restrict Y, size_t H, size_t W,
                                       float epsilon) {
  __fallback_rms_norm_wrt_width_fp16_intrinsic<_FP16>(X, Y, H, W, epsilon);
}

template <>
void sine(const unsigned int N, _FP16 *X, _FP16 *Y, float alpha, float beta) {
  nntrainer::neon::sine<_FP16>(N, X, Y, alpha, beta);
}

template <>
void cosine(const unsigned int N, _FP16 *X, _FP16 *Y, float alpha, float beta) {
  nntrainer::neon::cosine<_FP16>(N, X, Y, alpha, beta);
}

template <>
void calc_trigonometric_vals_dup(unsigned int N_half, _FP16 *angle, _FP16 *cos_,
                                 _FP16 *sin_, unsigned int from,
                                 float attention_scaling) {
  nntrainer::neon::calc_trigonometric_vals_dup<_FP16>(N_half, angle, cos_, sin_,
                                                      from, attention_scaling);
}

template <>
void rms_norm_wrt_width_fp16_intrinsic(const float *__restrict X,
                                       float *__restrict Y, size_t H, size_t W,
                                       float epsilon) {
  neon::rms_norm_wrt_width_fp16_intrinsic(X, Y, H, W, epsilon);
}

void nntr_quant_qs4cx_f32(size_t n, size_t k, void *rhs_native_mtx_f32,
                          void *rhs_native_mtx_qs4cx, void *rhs_scales_f32,
                          bool transB) {
  __fallback_nntr_quant_qs4cx_f32(n, k, rhs_native_mtx_f32,
                                  rhs_native_mtx_qs4cx, rhs_scales_f32, transB);
}

void nntr_quant_qs4c32_f32(size_t n, size_t k, size_t bl,
                           void *rhs_native_mtx_f32,
                           void *rhs_native_mtx_qs4c32) {
  nntr_kai_quant_qs4c32_f32(n, k, bl, (const float *)rhs_native_mtx_f32,
                            (uint8_t *)rhs_native_mtx_qs4c32);
}

template <>
uint32_t nntr_gemm_qai8dxp_qsi4cxp_unpacked(
  size_t m, size_t n, size_t k, void *lhs_native_mtx_f32,
  void *rhs_native_mtx_qs4cx, void *rhs_scales_f32, float *dst_mtx_f32,
  bool transB, float lower_bound, float upper_bound) {
  return nntr_kai_gemm_qai8dxp_qsi4cxp_rtp(
    m, n, k, lhs_native_mtx_f32, rhs_native_mtx_qs4cx, rhs_scales_f32,
    dst_mtx_f32, transB, lower_bound, upper_bound);
}

size_t nntr_get_rhs_packed_size_qsi4cxp_qs4cxs1s0(size_t n, size_t k,
                                                  uint32_t idx_variant,
                                                  bool transB) {
  return nntr_kai_get_rhs_packed_size_qsi4cxp_qs4cxs1s0(n, k, idx_variant,
                                                        transB);
}

void nntr_qsi4cxp_qs4cxs1s0_rhs_pack(size_t n, size_t k,
                                     void *rhs_packed_mtx_qs4cx,
                                     void *rhs_native_mtx_qs4cx,
                                     void *rhs_scales_f32, uint32_t idx_variant,
                                     bool transB) {
  nntr_kai_qsi4cxp_qs4cxs1s0_rhs_pack(n, k, rhs_packed_mtx_qs4cx,
                                      rhs_native_mtx_qs4cx, rhs_scales_f32,
                                      idx_variant, transB);
}

template <>
void nntr_gemm_qai8dxp_qsi4cxp_packed(size_t m, size_t n, size_t k,
                                      void *lhs_native_mtx_f32,
                                      void *rhs_packed_mtx_qs4cx,
                                      float *dst_act_mtx_f32,
                                      uint32_t idx_variant, bool transB,
                                      float lower_bound, float upper_bound) {
  nntr_kai_gemm_qai8dxp_qsi4cxp_olp(
    m, n, k, lhs_native_mtx_f32, rhs_packed_mtx_qs4cx, dst_act_mtx_f32,
    idx_variant, transB, lower_bound, upper_bound);
}

} /* namespace nntrainer */

namespace nntrainer {

size_t nntr_get_rhs_packed_size_qsi8d32p_qsi4c32p(size_t n, size_t k,
                                                  uint32_t idx_variant,
                                                  bool transB) {
  return nntr_kai_get_rhs_packed_size_qsi8d32p_qsi4c32p(n, k, idx_variant,
                                                        transB);
}

void nntr_qsi8d32p_qsi4c32p_rhs_pack(size_t n, size_t k,
                                     void *rhs_packed_mtx_qs4cx,
                                     void *rhs_native_mtx_qs4cx,
                                     void *rhs_scales_f32, uint32_t idx_variant,
                                     bool transB) {
  nntr_kai_qsi8d32p_qsi4c32p_rhs_pack(n, k, rhs_packed_mtx_qs4cx,
                                      rhs_native_mtx_qs4cx, rhs_scales_f32,
                                      idx_variant, transB);
}

template <>
uint32_t nntr_gemm_qsi8d32p_qsi4c32p_unpacked(
  size_t m, size_t n, size_t k, void *lhs_native_mtx_f32,
  void *rhs_native_mtx_qs4cx, void *rhs_scales_f32, float *dst_mtx_f32,
  bool transB, float lower_bound, float upper_bound) {
  return nntr_kai_gemm_qsi8d32p_qsi4c32p_rtp(
    m, n, k, lhs_native_mtx_f32, rhs_native_mtx_qs4cx, rhs_scales_f32,
    dst_mtx_f32, transB, lower_bound, upper_bound);
}

template <>
void nntr_gemm_qsi8d32p_qsi4c32p_packed(size_t m, size_t n, size_t k,
                                        void *lhs_native_mtx_f32,
                                        void *rhs_packed_mtx_qs4cx,
                                        float *dst_act_mtx_f32,
                                        uint32_t idx_variant, bool transB,
                                        float lower_bound, float upper_bound) {
  nntr_kai_gemm_qsi8d32p_qsi4c32p_olp(
    m, n, k, lhs_native_mtx_f32, rhs_packed_mtx_qs4cx, dst_act_mtx_f32,
    idx_variant, transB, lower_bound, upper_bound);
}

} /* namespace nntrainer */
