// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file arm_compute_backend.cpp
 * @date   23 April 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Compute backend for arm
 *
 */
#include <arm_compute_backend.h>
#include <assert.h>
#ifdef USE_BLAS
#include <cblas_interface.h>
#endif
#include <fallback_internal.h>
#include <ggml_interface.h>
#include <neon_impl.h>
#include <nntrainer_error.h>
#include <q4_0_utils.h>

namespace nntrainer {

void init_backend() { __ggml_init(); }

void unpack_q4_0x8_transpose16(const void *src, uint16_t *d_out,
                               uint16_t *qs_out, int N, int K) {
  __fallback_unpack_q4_0x8_transpose16(src, d_out, qs_out, N, K);
  // Do not repeatedly call set_num_threads. It's a global config.
  __openblas_set_num_threads(-1); // -1 = BLAS_NUM_THREADS if defined.
}

template <>
void calc_trigonometric_vals_dup(unsigned int N_half, float *angle, float *cos_,
                                 float *sin_, unsigned int from,
                                 float attention_scaling) {
  nntrainer::neon::calc_trigonometric_vals_dup(N_half, angle, cos_, sin_, from,
                                               attention_scaling);
}

void swiglu(const unsigned int N, float *X, float *Y, float *Z) {
  nntrainer::neon::swiglu(N, X, Y, Z);
}

void swiglu(const unsigned int N, float *X, float *Y, float *Z, float alpha) {
  nntrainer::neon::swiglu(N, X, Y, Z, alpha);
}

float max_val(const unsigned int N, float *X) {
  return nntrainer::neon::max_val(N, X);
}

void softmax(const unsigned int N, float *X, float *Y) {
  nntrainer::neon::softmax(N, X, Y);
}

void scopy(const unsigned int N, const uint8_t *X, const unsigned int incX,
           uint8_t *Y, const unsigned int incY) {
  if (incX == 1 && incY == 1) {
    nntrainer::neon::copy_int8_or_int4(N, X, Y);
  } else {
    __fallback_scopy(N, X, incX, Y, incY);
  }
}

void scopy(const unsigned int N, const int8_t *X, const unsigned int incX,
           int8_t *Y, const unsigned int incY) {
  if (incX == 1 && incY == 1) {
    nntrainer::neon::copy_s8(N, X, Y);
  } else {
    __fallback_scopy(N, X, incX, Y, incY);
  }
}

void scopy_int4_to_float32(const unsigned int N, const uint8_t *X,
                           const unsigned int incX, float *Y,
                           const unsigned int incY) {
  if (incX == 1 && incY == 1) {
    nntrainer::neon::copy_int4_to_fp32(N, X, Y);
  } else {
    __fallback_scopy_int4_to_float32(N, X, incX, Y, incY);
  }
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

/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] Y uint32_t * for Vector Y
 */
template <> void copy_fp32(const unsigned int N, const float *X, uint32_t *Y) {
  copy_fp32_u32(N, X, Y);
}

/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] Y uint16_t * for Vector Y
 */
template <> void copy_fp32(const unsigned int N, const float *X, uint16_t *Y) {
  copy_fp32_u16(N, X, Y);
}

/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] Y uint16_t * for Vector Y
 */
template <> void copy_fp32(const unsigned int N, const float *X, uint8_t *Y) {
  copy_fp32_u8(N, X, Y);
}

/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] Y int16_t * for Vector Y
 */
template <> void copy_fp32(const unsigned int N, const float *X, int16_t *Y) {
  copy_fp32_s16(N, X, Y);
}

/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] Y int8_t * for Vector Y
 */
template <> void copy_fp32(const unsigned int N, const float *X, int8_t *Y) {
  copy_fp32_s8(N, X, Y);
}

void copy_s16_fp32(const unsigned int N, const int16_t *X, float *Y) {
  nntrainer::neon::copy_s16_fp32(N, X, Y);
}

void copy_u16_fp32(const unsigned int N, const uint16_t *X, float *Y) {
  nntrainer::neon::copy_u16_fp32(N, X, Y);
}

void copy_s16(const unsigned int N, const int16_t *X, int16_t *Y) {
  nntrainer::neon::copy_s16(N, X, Y);
}

void copy_u16(const unsigned int N, const uint16_t *X, uint16_t *Y) {
  nntrainer::neon::copy_u16(N, X, Y);
}

void scopy_int8_to_float32(const unsigned int N, const uint8_t *X,
                           const unsigned int incX, float *Y,
                           const unsigned int incY) {

  if (incX == 1 && incY == 1) {
    nntrainer::neon::copy_int8_to_fp32(N, X, Y);
  } else {
    __fallback_scopy_uint8_to_float32(N, X, incX, Y, incY);
  }
}

void scopy_int8_to_float32(const unsigned int N, const int8_t *X,
                           const unsigned int incX, float *Y,
                           const unsigned int incY) {

  if (incX == 1 && incY == 1) {
    nntrainer::neon::copy_int8_to_fp32(N, X, Y);
  } else {
    __fallback_scopy_int8_to_float32(N, X, incX, Y, incY);
  }
}

template <>
void sine(const unsigned int N, float *X, float *Y, float alpha, float beta) {
  nntrainer::neon::sine(N, X, Y, alpha, beta);
}

template <>
void cosine(const unsigned int N, float *X, float *Y, float alpha, float beta) {
  nntrainer::neon::cosine(N, X, Y, alpha, beta);
}

void inv_sqrt_inplace(const unsigned int N, float *X) {
  nntrainer::neon::inv_sqrt_inplace(N, X);
}

void ele_mul(const unsigned int N, const float *X, const float *Y, float *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  if (i_stride == 1 && o_stride == 1) {
    nntrainer::neon::ele_mul(N, X, Y, Z, alpha, beta);
  } else
    __fallback_ele_mul(N, X, Y, Z, alpha, beta, i_stride, o_stride);
}

void ele_add(const unsigned int N, const float *X, const float *Y, float *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  if (i_stride == 1 && o_stride == 1) {
    nntrainer::neon::ele_add(N, X, Y, Z, alpha, beta);
  } else
    __fallback_ele_add(N, X, Y, Z, alpha, beta, i_stride, o_stride);
}

void ele_sub(const unsigned N, const float *X, const float *Y, float *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  if (i_stride == 1 && o_stride == 1) {
    nntrainer::neon::ele_sub(N, X, Y, Z, alpha, beta);
  } else
    __fallback_ele_sub(N, X, Y, Z, alpha, beta, i_stride, o_stride);
}

void ele_div(const unsigned N, const float *X, const float *Y, float *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  if (i_stride == 1 && o_stride == 1) {
    nntrainer::neon::ele_div(N, X, Y, Z, alpha, beta);
  } else
    __fallback_ele_div(N, X, Y, Z, alpha, beta, i_stride, o_stride);
}

void saxpy(const unsigned int N, const float alpha, const float *X,
           const unsigned int incX, float *Y, const unsigned int incY) {
#ifdef USE_BLAS
  __cblas_saxpy(N, alpha, X, incX, Y, incY);
#else
  __fallback_saxpy(N, alpha, X, incX, Y, incY);
#endif
}

void sgemv(const unsigned int TStorageOrder, bool TransA, const unsigned int M,
           const unsigned int N, const float alpha, const float *A,
           const unsigned int lda, const float *X, const unsigned int incX,
           const float beta, float *Y, const unsigned int incY) {
#ifdef USE_BLAS
  __cblas_sgemv(TStorageOrder, TransA, M, N, alpha, A, lda, X, incX, beta, Y,
                incY);
#else
  __fallback_sgemv(TStorageOrder, TransA, M, N, alpha, A, lda, X, incX, beta, Y,
                   incY);
#endif
}

float sdot(const unsigned int N, const float *X, const unsigned int incX,
           const float *Y, const unsigned int incY) {
#ifdef USE_BLAS
  return __cblas_sdot(N, X, incX, Y, incY);
#else
  return __fallback_sdot(N, X, incX, Y, incY);
#endif
}

void scopy(const unsigned int N, const float *X, const unsigned int incX,
           float *Y, const unsigned int incY) {
  /// @note cblas_scopy is evoking SIGSEGV for some reason. Use custom
  /// implementation instead.
  // __cblas_scopy(N, X, incX, Y, incY);
  nntrainer::neon::custom_scopy(N, X, incX, Y, incY);
}

void sscal(const unsigned int N, const float alpha, float *X,
           const unsigned int incX) {
#ifdef USE_BLAS
  __cblas_sscal(N, alpha, X, incX);
#else
  __fallback_sscal(N, alpha, X, incX);
#endif
}

float snrm2(const unsigned int N, const float *X, const unsigned int incX) {
#ifdef USE_BLAS
  return __cblas_snrm2(N, X, incX);
#else
  return __fallback_snrm2(N, X, incX);
#endif
}

void sgemm(const unsigned int TStorageOrder, bool TransA, bool TransB,
           const unsigned int M, const unsigned int N, const unsigned int K,
           const float alpha, const float *A, const unsigned int lda,
           const float *B, const unsigned int ldb, const float beta, float *C,
           const unsigned int ldc) {
#ifdef USE_BLAS
  __cblas_sgemm(TStorageOrder, TransA, TransB, M, N, K, alpha, A, lda, B, ldb,
                beta, C, ldc);
#else
  __fallback_sgemm(TStorageOrder, TransA, TransB, M, N, K, alpha, A, lda, B,
                   ldb, beta, C, ldc);
#endif
}

unsigned int isamax(const unsigned int N, const float *X,
                    const unsigned int incX) {
#ifdef USE_BLAS
  return __cblas_isamax(N, X, incX);
#else
  return __fallback_isamax(N, X, incX);
#endif
}

void transpose_matrix(const unsigned int M, const unsigned int N,
                      const float *src, unsigned int ld_src, float *dst,
                      unsigned int ld_dst) {
  nntrainer::neon::transpose_matrix(M, N, src, ld_src, dst, ld_dst);
}

bool is_valid(const unsigned int N, const float *input) {
  return nntrainer::neon::is_valid(N, input);
}

template <>
void gemm_q4_0(const unsigned int M, const unsigned int N, const unsigned int K,
               const float *A, const unsigned int lda, const void *B,
               const unsigned int ldb, float *C, const unsigned int ldc) {
  return __ggml_q4_0_4x8_q8_0_GEMM<float>(M, N, K, A, lda, B, ldb, C, ldc);
}

void gemm_q4_0(const unsigned int M, std::vector<unsigned int> Ns,
               const unsigned int K, const float *A, const unsigned int lda,
               std::vector<void *> Bs, std::vector<unsigned int> ldbs,
               std::vector<float *> Cs, std::vector<unsigned int> ldcs) {
  return __ggml_q4_0_4x8_q8_0_GEMM<float>(M, Ns, K, A, lda, Bs, ldbs, Cs, ldcs);
}

void gemm_q4_K(const unsigned int M, const unsigned int N, const unsigned int K,
               const float *A, const unsigned int lda, const void *B,
               const unsigned int ldb, float *C, const unsigned int ldc) {
  return __ggml_q4_K_8x8_q8_K_GEMM(M, N, K, A, lda, B, ldb, C, ldc);
}

void gemm_q4_K(const unsigned int M, std::vector<unsigned int> Ns,
               const unsigned int K, const float *A, const unsigned int lda,
               std::vector<void *> Bs, std::vector<unsigned int> ldbs,
               std::vector<float *> Cs, std::vector<unsigned int> ldcs) {
  return __ggml_q4_K_8x8_q8_K_GEMM(M, Ns, K, A, lda, Bs, ldbs, Cs, ldcs);
}

template <>
void gemm_q6_K(const unsigned int M, const unsigned int N, const unsigned int K,
               const float *A, const unsigned int lda, const void *B,
               const unsigned int ldb, float *C, const unsigned int ldc) {
  return __ggml_gemm_q6_K<float>(M, N, K, A, lda, B, ldb, C, ldc);
}

float dot_q6_K_q8_K(const unsigned int K, const void *v_q6_K,
                    const void *v_q8_K) {
  return __ggml_vec_dot_q6_K_q8_K(K, v_q6_K, v_q8_K);
}

float dot_q6_K_f32(const unsigned int K, const void *v_q6_K, const float *f) {
  return __ggml_vec_dot_q6_K_f32(K, v_q6_K, f);
}

size_t quantize_q4_0(const float *src, void *dst, int64_t nrow,
                     int64_t n_per_row, const float *quant_weights) {
  return __ggml_quantize_q4_0(src, dst, nrow, n_per_row, quant_weights);
}

size_t quantize_q4_K(const float *src, void *dst, int64_t nrow,
                     int64_t n_per_row, const float *quant_weights) {
  return __ggml_quantize_q4_K(src, dst, nrow, n_per_row, quant_weights);
}

size_t quantize_q6_K(const float *src, void *dst, int64_t nrow,
                     int64_t n_per_row, const float *quant_weights) {
  return __ggml_quantize_q6_K(src, dst, nrow, n_per_row, quant_weights);

  return __fallback_quantize_q6_K(src, dst, nrow, n_per_row, quant_weights);
}

void quantize_row_q6_K(const float *src, void *dst, int64_t k) {
  __ggml_quantize_row_q6_K(src, dst, k);
}

template <> void quantize_row_q8_K(const float *src, void *dst, int64_t k) {
  __ggml_quantize_row_q8_K(src, dst, k);
}

void dequantize_row_q4_K(const void *x_raw, float *y, int64_t k) {
  __ggml_dequantize_row_q4_K(x_raw, y, k);
}

void dequantize_row_q4_0(const void *x_raw, float *y, int64_t k) {
  __ggml_dequantize_row_q4_0(x_raw, y, k);
}

void dequantize_row_q6_K(const void *x, float *y, int64_t k) {
  __ggml_dequantize_row_q6_K(x, y, k);
}

template <> void dequantize_row_q8_K(const void *x, float *y, int64_t k) {
  __ggml_dequantize_row_q8_K(x, y, k);
}

void repack_q4_0_to_q4_0_8(void *W, void *repacked_W, size_t data_size,
                           const unsigned int M, const unsigned int N) {
  __ggml_repack_q4_0_to_q4_0_8(W, repacked_W, data_size, M, N);
}

void repack_q4_0(void *W, void *repacked_W, size_t data_size,
                 const unsigned int M, const unsigned int N) {
  __ggml_repack_q4_0_to_q4_0_4(W, repacked_W, data_size, M, N);
}

void repack_q4_K(void *W, void *repacked_W, size_t data_size,
                 const unsigned int M, const unsigned int N) {
  __ggml_repack_q4_K_to_q4_K_8(W, repacked_W, data_size, M, N);
}

void unpack_q4_0(const void *in_q4_0x, void *out_q4_0, size_t data_size,
                 const unsigned int M, const unsigned int N) {
  Q4_0Utils::unpackBlocksQ4_0x4((const block_q4_0x4 *)in_q4_0x, data_size, M, N,
                                (block_q4_0 *)out_q4_0);
}

template <>
void softmax_row_inplace(float *qk_out, size_t start_row, size_t end_row,
                         size_t num_heads, float *sink) {
  neon::softmax_row_inplace(qk_out, start_row, end_row, num_heads, sink);
}

template <>
void softmax_row(float *qk_out, size_t start_row, size_t end_row,
                 size_t num_heads, float *sink) {
  neon::softmax_row(qk_out, start_row, end_row, num_heads, sink);
}

void rms_norm_wrt_width_fp32_intrinsic(const float *__restrict X,
                                       float *__restrict Y, size_t H, size_t W,
                                       float epsilon) {
  neon::rms_norm_wrt_width_fp32_intrinsic(X, Y, H, W, epsilon);
}

template <>
void clamp(const float *input, float *output, size_t length, float lower_bound,
           float upper_bound) {
  neon::clamp(input, output, length, lower_bound, upper_bound);
}

template <>
void compute_kcaches(const float *in, const uint16_t *kcache, float *output,
                     int num_rows, int num_cache_head, int head_dim,
                     int gqa_size, int tile_size, size_t local_window_size) {
#ifdef ENABLE_FP16
  neon::compute_kcaches<_FP16>(in, reinterpret_cast<const _FP16 *>(kcache),
                               output, num_rows, num_cache_head, head_dim,
                               gqa_size, tile_size, local_window_size);
#else
/// @note float16x4_t and related FP16 NEON are available
#if defined(__aarch64__) || defined(_M_ARM64)
  neon::compute_kcaches_uint16(in, kcache, output, num_rows, num_cache_head,
                               head_dim, gqa_size, tile_size,
                               local_window_size);
#else
  __fallback_compute_kcaches(in, kcache, output, num_rows, num_cache_head,
                             head_dim, gqa_size, tile_size, local_window_size);
#endif
#endif
}

void compute_fp16vcache_fp32_transposed(int row_num, const float *in,
                                        const uint16_t *vcache, float *output,
                                        int num_cache_head, int gqa_size,
                                        int head_dim,
                                        size_t local_window_size) {
#ifdef ENABLE_FP16
  neon::compute_fp16vcache_fp32_transposed(
    row_num, in, reinterpret_cast<const _FP16 *>(vcache), output,
    num_cache_head, gqa_size, head_dim, local_window_size);
#else
/// @note float16x4_t and related FP16 NEON are available
#if defined(__aarch64__) || defined(_M_ARM64)
  neon::compute_fp16vcache_fp32_transposed(row_num, in, vcache, output,
                                           num_cache_head, gqa_size, head_dim,
                                           local_window_size);
#else
  __fallback_compute_fp16vcache_fp32_transposed(row_num, in, vcache, output,
                                                num_cache_head, gqa_size,
                                                head_dim, local_window_size);
#endif
#endif
}

void compute_rotary_emb_value(unsigned int width, unsigned int dim,
                              unsigned int half_, float *inout, void *output,
                              const float *cos_, const float *sin_,
                              bool only_convert_to_fp16) {
#ifdef ENABLE_FP16
  neon::compute_rotary_emb_value(width, dim, half_, inout, output, cos_, sin_,
                                 only_convert_to_fp16);
#else
/// @note float16x4_t and related FP16 NEON are available
#if defined(__aarch64__) || defined(_M_ARM64)
  neon::compute_rotary_emb_value_uint16(width, dim, half_, inout, output, cos_,
                                        sin_, only_convert_to_fp16);
#else
  __fallback_compute_rotary_emb_value(width, dim, half_, inout, output, cos_,
                                      sin_, only_convert_to_fp16);
#endif
#endif
}

void create_q4_0_weights(const uint8_t *int4_weight, uint8_t *q4_0_weight) {
  __fallback_create_q4_0_weights(int4_weight, q4_0_weight);
}

/// @todo rename it to `transform_int4_osv32_isv2_to_q4_0`
void transform_q4_0x_from_int4(size_t N, size_t K, const uint8_t *osv32_weights,
                               const uint16_t *osv32_scales,
                               size_t scale_group_size, void *dst_q4_0x) {
#if defined(__aarch64__) || defined(_M_ARM64)
  neon::transform_int4_osv32_isv2_to_q4_0x4(N, K, osv32_weights, osv32_scales,
                                            scale_group_size, dst_q4_0x);
#else
  Q4_0Utils::transformQ4_0x_FromInt4(N, K, osv32_weights, osv32_scales,
                                     scale_group_size, 4, dst_q4_0x);
#endif
}

} /* namespace nntrainer */
