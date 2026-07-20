// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   x86_compute_backend.cpp
 * @date   23 April 2024
 * @see    https://github.com/nntrainer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Compute backend for x86
 *
 */

#include <assert.h>

#include <avx2_impl.h>
#include <nntr_ggml_impl.h>
#ifdef USE_BLAS
#include <cblas_interface.h>
#endif
#include <compute_ops.h>
#include <fallback_internal.h>
#include <ggml_interface.h>
#include <nntrainer_error.h>
#include <q4_0_utils.h>
#include <x86_compute_backend.h>

#define ROW_MAJOR 0
#define COL_MAJOR 1

namespace nntrainer {

void init_backend() {
  __ggml_init();
#ifdef USE_BLAS
  // Do not repeatedly call set_num_threads. It's a global config.
  __openblas_set_num_threads(-1); // -1 = BLAS_NUM_THREADS if defined.
#endif
  g_compute_ops = get_cpu_ops();
}

void scopy_int4_to_float32(const unsigned int N, const uint8_t *X,
                           const unsigned int incX, float *Y,
                           const unsigned int incY) {
  __fallback_scopy_int4_to_float32(N, X, incX, Y, incY);
}

void copy_s16(const unsigned int N, const int16_t *X, int16_t *Y) {
  __fallback_copy_s16(N, X, Y);
}

void copy_u16(const unsigned int N, const uint16_t *X, uint16_t *Y) {
  __fallback_copy_u16(N, X, Y);
}

void copy_s16_fp32(const unsigned int N, const int16_t *X, float *Y) {
  __fallback_copy_s16_fp32(N, X, Y);
}

void copy_u16_fp32(const unsigned int N, const uint16_t *X, float *Y) {
  nntrainer::avx2::copy_f16_f32(N, X, Y);
}

void copy_fp32_u32(const unsigned int N, const float *X, uint32_t *Y) {
  __fallback_copy_fp32_u32(N, X, Y);
}

void copy_fp32_u16(const unsigned int N, const float *X, uint16_t *Y) {
  nntrainer::avx2::copy_f32_f16(N, X, Y);
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

void scopy_int8_to_float32(const unsigned int N, const uint8_t *X,
                           const unsigned int incX, float *Y,
                           const unsigned int incY) {
  __fallback_scopy_uint8_to_float32(N, X, incX, Y, incY);
}

void scopy_int8_to_float32(const unsigned int N, const int8_t *X,
                           const unsigned int incX, float *Y,
                           const unsigned int incY) {
  __fallback_scopy_int8_to_float32(N, X, incX, Y, incY);
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
  nntrainer::avx2::ele_mul(N, X, Y, Z, alpha, beta, i_stride, o_stride);
}

void ele_add(const unsigned int N, const float *X, const float *Y, float *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  nntrainer::avx2::ele_add(N, X, Y, Z, alpha, beta, i_stride, o_stride);
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
  /// @note cblas_scopy is evoking SIGSEGV for some reason. Use custom
  /// implementation instead.
  // __cblas_scopy(N, X, incX, Y, incY);
  nntrainer::avx2::custom_scopy(N, X, incX, Y, incY);
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
  nntrainer::avx2::transpose_matrix(M, N, src, ld_src, dst, ld_dst);
}

bool is_valid(const unsigned int N, const float *input) {
  return nntrainer::avx2::is_valid(N, input);
}

void unpack_q4_0x8_transpose16(const void *src, uint16_t *d_out,
                               uint16_t *qs_out, int N, int K) {
  return nntrainer::avx2::unpack_q4_0x8_transpose16(src, d_out, qs_out, N, K);
}

template <>
void calc_trigonometric_vals_dup(unsigned int N_half, float *angle, float *cos_,
                                 float *sin_, unsigned int from,
                                 float attention_scaling) {
  __fallback_calc_trigonometric_vals_dup(N_half, angle, cos_, sin_, from,
                                         attention_scaling);
}

void swiglu(const unsigned int N, float *X, float *Y, float *Z) {
  nntrainer::avx2::swiglu(N, X, Y, Z);
}

void swiglu(const unsigned int N, float *X, float *Y, float *Z, float alpha) {
  nntrainer::avx2::swiglu(N, X, Y, Z, alpha);
}

void tanh_gelu(const unsigned int N, const float *X, float *Y) {
  // AVX implmenetation will be implemented, now fallback instead
  __fallback_tanh_gelu(N, X, Y);
}

void tanh_gelu_v2(const unsigned int N, const float *X, float *Y) {
  nntrainer::avx2::tanh_gelu_v2(N, X, Y);
}

void gelu_v2(const unsigned int N, const float *X, float *Y) {
  nntrainer::avx2::gelu_v2(N, X, Y);
}

void tanh_gelu_mul(const unsigned int N, float *X, float *Y, float *Z) {
  __fallback_tanh_gelu_mul(N, X, Y, Z);
}

void tanh_gelu_v2_mul(const unsigned int N, float *X, float *Y, float *Z) {
  __fallback_tanh_gelu_mul(N, X, Y, Z);
}

float max_val(const unsigned int N, float *X) { return __fallback_max(N, X); }

void softmax(const unsigned int N, float *X, float *Y) {
  __fallback_softmax(N, X, Y);
}

template <>
void gemm_q4_0(const unsigned int M, const unsigned int N, const unsigned int K,
               const float *A, const unsigned int lda, const void *B,
               const unsigned int ldb, float *C, const unsigned int ldc) {
  return __ggml_q4_0_8x8_q8_0_GEMM(M, N, K, A, lda, B, ldb, C, ldc);
}

template <>
void gemm_q8_0(const unsigned int M, const unsigned int N, const unsigned int K,
               const float *A, const unsigned int lda, const void *B,
               const unsigned int ldb, float *C, const unsigned int ldc) {
  return __ggml_q8_0_q8_0_GEMM(M, N, K, A, lda, B, ldb, C, ldc);
}

void gemm_q4_0(const unsigned int M, std::vector<unsigned int> Ns,
               const unsigned int K, const float *A, const unsigned int lda,
               std::vector<void *> Bs, std::vector<unsigned int> ldbs,
               std::vector<float *> Cs, std::vector<unsigned int> ldcs) {
  throw std::runtime_error("Error: NYI for gemm_q4_0 with vectored weights");
}

void gemm_q4_0_indirect_conv(const unsigned int M, const unsigned int N,
                             const unsigned int K, const float *in,
                             const ConvGatherParams &geom, const void *B,
                             const unsigned int ldb, float *C,
                             const unsigned int ldc) {
  throw std::runtime_error("Error: NYI for gemm_q4_0_indirect_conv on x86");
}

#ifdef ENABLE_FP16
void gemm_q4_0_indirect_conv_fp16(const unsigned int M, const unsigned int N,
                                  const unsigned int K, const _FP16 *in,
                                  const ConvGatherParams &geom, const void *B,
                                  const unsigned int ldb, _FP16 *C,
                                  const unsigned int ldc) {
  throw std::runtime_error(
    "Error: NYI for gemm_q4_0_indirect_conv_fp16 on x86");
}
void gemm_q4_0_indirect_conv_q8_0(const unsigned int M, const unsigned int N,
                                  const unsigned int K, const void *in,
                                  const ConvGatherParams &geom, const void *B,
                                  const unsigned int ldb, _FP16 *C,
                                  const unsigned int ldc) {
  throw std::runtime_error(
    "Error: NYI for gemm_q4_0_indirect_conv_q8_0 on x86");
}
void gemm_q8_0_indirect_conv_q8_0(const unsigned int M, const unsigned int N,
                                  const unsigned int K, const void *in,
                                  const ConvGatherParams &geom, const void *B,
                                  const unsigned int ldb, _FP16 *C,
                                  const unsigned int ldc) {
  throw std::runtime_error(
    "Error: NYI for gemm_q8_0_indirect_conv_q8_0 on x86");
}
void gemm_q8_0_indirect_conv_fp16(const unsigned int M, const unsigned int N,
                                  const unsigned int K, const _FP16 *in,
                                  const ConvGatherParams &geom, const void *B,
                                  const unsigned int ldb, _FP16 *C,
                                  const unsigned int ldc) {
  throw std::runtime_error(
    "Error: NYI for gemm_q8_0_indirect_conv_fp16 on x86");
}
#endif

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

float dot_q6_K_q8_K(const unsigned int K, const void *v_q6_K,
                    const void *v_q8_K) {
  return __ggml_vec_dot_q6_K_q8_K(K, v_q6_K, v_q8_K);
}

float dot_q6_K_f32(const unsigned int K, const void *v_q6_K, const float *f) {
  return __ggml_vec_dot_q6_K_f32(K, v_q6_K, f);
}

template <>
void gemm_q6_K(const unsigned int M, const unsigned int N, const unsigned int K,
               const float *A, const unsigned int lda, const void *B,
               const unsigned int ldb, float *C, const unsigned int ldc) {
  return __ggml_gemm_q6_K(M, N, K, A, lda, B, ldb, C, ldc);
}

size_t quantize_q4_0(const float *src, void *dst, int64_t nrow,
                     int64_t n_per_row, const float *quant_weights) {
  return __ggml_quantize_q4_0(src, dst, nrow, n_per_row, quant_weights);
}

size_t quantize_q8_0(const float *src, void *dst, int64_t nrow,
                     int64_t n_per_row, const float *quant_weights) {
  return __ggml_quantize_q8_0(src, dst, nrow, n_per_row, quant_weights);
}

size_t quantize_q4_K(const float *src, void *dst, int64_t nrow,
                     int64_t n_per_row, const float *quant_weights) {
  return __ggml_quantize_q4_K(src, dst, nrow, n_per_row, quant_weights);
}

size_t quantize_q6_K(const float *src, void *dst, int64_t nrow,
                     int64_t n_per_row, const float *quant_weights) {
  return __ggml_quantize_q6_K(src, dst, nrow, n_per_row, quant_weights);
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

void dequantize_row_q8_0(const void *x_raw, float *y, int64_t k) {
  __ggml_dequantize_row_q8_0(x_raw, y, k);
}

void quantize_row_q8_0(const float *src, void *dst, int64_t k) {
  nntr_quantize_row_q8_0(src, dst, k);
}

void dequantize_row_q6_K(const void *x, float *y, int64_t k) {
  __ggml_dequantize_row_q6_K(x, y, k);
}

template <> void dequantize_row_q8_K(const void *x, float *y, int64_t k) {
  __ggml_dequantize_row_q8_K(x, y, k);
}

void repack_q4_0(void *dst, void *src, size_t data_size, const unsigned int M,
                 const unsigned int N, ml::train::ISA target) {

  switch (target) {
  case ml::train::ISA::X86:
    // Use x86 format (q4_0x8)
    __ggml_repack_q4_0_to_q4_0_8(dst, src, data_size, M, N);
    break;
  case ml::train::ISA::ARM:
    // Use ARM format (q4_0x4) for cross-platform quantization
    __ggml_repack_q4_0_to_q4_0_4(dst, src, data_size, M, N);
    break;
  case ml::train::ISA::DEFAULT:
    // Use x86 format (q4_0x8)
    __ggml_repack_q4_0_to_q4_0_8(dst, src, data_size, M, N);
    break;
  default:
    break;
  }
}

void repack_q4_0_to_q4_0_8(void *dst, void *src, size_t data_size,
                           const unsigned int M, const unsigned int N) {
  __ggml_repack_q4_0_to_q4_0_8(dst, src, data_size, M, N);
}

void repack_q4_K(void *dst, void *src, size_t data_size, const unsigned int M,
                 const unsigned int N) {
  __ggml_repack_q4_K_to_q4_K_8(dst, src, data_size, M, N);
}

void unpack_q4_0(const void *in_q4_0x, void *out_q4_0, size_t data_size,
                 const unsigned int M, const unsigned int N) {
  Q4_0Utils::unpackBlocksQ4_0x8((const block_q4_0x8 *)in_q4_0x, data_size, M, N,
                                (block_q4_0 *)out_q4_0);
}

template <>
void softmax_row_inplace(float *qk_out, size_t start_row, size_t end_row,
                         size_t num_heads, float *sink) {
  nntrainer::avx2::softmax_row_inplace<float>(qk_out, start_row, end_row,
                                              num_heads, sink);
}

template <>
void softmax_row(float *qk_out, size_t start_row, size_t end_row,
                 size_t num_heads, float *sink) {
  nntrainer::avx2::softmax_row<float>(qk_out, start_row, end_row, num_heads,
                                      sink);
}

void compute_fp16vcache_fp32_transposed(int row_num, const float *in,
                                        const uint16_t *vcache, float *output,
                                        int num_cache_head, int gqa_size,
                                        int head_dim, size_t local_window_size,
                                        int head_start, int head_end) {
  nntrainer::avx2::compute_fp16vcache_fp32_transposed(
    row_num, in, vcache, output, num_cache_head, gqa_size, head_dim,
    local_window_size, head_start, head_end);
}

template <>
void compute_kcaches(const float *in, const uint16_t *kcache, float *output,
                     int num_rows, int num_cache_head, int head_dim,
                     int gqa_size, int tile_size, size_t local_window_size,
                     int head_start, int head_end) {
  nntrainer::avx2::compute_kcaches<uint16_t>(
    in, kcache, output, num_rows, num_cache_head, head_dim, gqa_size, tile_size,
    local_window_size, head_start, head_end);
}

void compute_rotary_emb_value(unsigned int width, unsigned int dim,
                              unsigned int half_, float *inout, void *output,
                              const float *cos_, const float *sin_,
                              bool only_convert_to_fp16) {
  nntrainer::avx2::compute_rotary_emb_value(width, dim, half_, inout, output,
                                            cos_, sin_, only_convert_to_fp16);
}

void rms_norm_wrt_width_fp32_intrinsic(const float *__restrict X,
                                       float *__restrict Y, size_t H, size_t W,
                                       float epsilon) {
  nntrainer::avx2::rms_norm_wrt_width_fp32_intrinsic(X, Y, H, W, epsilon);
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
  nntrainer::avx2::clamp(input, output, length, lower_bound, upper_bound);
}

void depthwise_conv2d_fp32(const float *input, const float *kernel,
                           float *output, unsigned int batch,
                           unsigned int channels, unsigned int in_h,
                           unsigned int in_w, unsigned int out_h,
                           unsigned int out_w, unsigned int kh, unsigned int kw,
                           unsigned int stride_h, unsigned int stride_w,
                           unsigned int pad_top, unsigned int pad_left,
                           unsigned int dilation_h, unsigned int dilation_w) {
  // TODO: AVX specialization
  __fallback_depthwise_conv2d_fp32(
    input, kernel, output, batch, channels, in_h, in_w, out_h, out_w, kh, kw,
    stride_h, stride_w, pad_top, pad_left, dilation_h, dilation_w);
}

#ifdef ENABLE_FP16
void depthwise_conv2d_fp16(const _FP16 *input, const float *kernel,
                           _FP16 *output, unsigned int batch,
                           unsigned int channels, unsigned int in_h,
                           unsigned int in_w, unsigned int out_h,
                           unsigned int out_w, unsigned int kh, unsigned int kw,
                           unsigned int stride_h, unsigned int stride_w,
                           unsigned int pad_top, unsigned int pad_left,
                           unsigned int dilation_h, unsigned int dilation_w) {
  // TODO: AVX fp16 specialization
  __fallback_depthwise_conv2d_fp16(
    input, kernel, output, batch, channels, in_h, in_w, out_h, out_w, kh, kw,
    stride_h, stride_w, pad_top, pad_left, dilation_h, dilation_w);
}
#endif

void create_q4_0_weights(const uint8_t *int4_weight, uint8_t *q4_0_weight) {
  nntrainer::avx2::create_q4_0_weights(int4_weight, q4_0_weight);
}

void transform_int4_osv32_isv2_to_q4_0(size_t N, size_t K,
                                       const uint8_t *osv32_weights,
                                       const uint16_t *osv32_scales,
                                       size_t scale_group_size,
                                       void *dst_q4_0x) {
#ifdef __AVX2__
  nntrainer::avx2::transform_int4_osv32_isv2_to_q4_0x8(
    N, K, osv32_weights, osv32_scales, scale_group_size, dst_q4_0x);
#else
  __fallback_transform_int4_osv32_isv2_to_q4_0(
    N, K, osv32_weights, osv32_scales, scale_group_size, 8, dst_q4_0x);
#endif
}

void quant_qs4cx_f32(size_t n, size_t k, void *rhs_native_mtx_f32,
                     void *rhs_native_mtx_qs4cx, void *rhs_scales_f32,
                     bool is_nxk) {
  if (is_nxk) {
    __fallback_quant_nxk_qs4cx_f32(n, k, (const float *)rhs_native_mtx_f32,
                                   (uint8_t *)rhs_native_mtx_qs4cx,
                                   (float *)rhs_scales_f32);
  } else {
    __fallback_quant_kxn_qs4cx_f32(n, k, (const float *)rhs_native_mtx_f32,
                                   (uint8_t *)rhs_native_mtx_qs4cx,
                                   (float *)rhs_scales_f32);
  }
}

size_t get_rhs_packed_size_qsi4cxp_qs4cxs1s0(size_t n, size_t k,
                                             size_t idx_variant, bool is_nxk) {
  return __fallback_get_rhs_packed_size_qsi4cxp_qs4cxs1s0(n, k, idx_variant,
                                                          is_nxk);
}

void rhs_pack_qsi4cxp_qs4cxs1s0(size_t n, size_t k, void *rhs_packed_mtx_qs4cx,
                                void *rhs_native_mtx_qs4cx,
                                void *rhs_scales_f32, size_t idx_variant,
                                bool is_nxk) {
  __fallback_rhs_pack_qsi4cxp_qs4cxs1s0(n, k, rhs_packed_mtx_qs4cx,
                                        rhs_native_mtx_qs4cx, rhs_scales_f32,
                                        idx_variant, is_nxk);
}

void gemm_qai8dxp_qsi4cxp_rhs_unpacked(
  size_t m, size_t n, size_t k, void *lhs_native_mtx_f32,
  void *rhs_native_mtx_qs4cx, void *rhs_scales_f32, float *dst_act_mtx_f32,
  size_t idx_variant, bool is_nxk, float lower_bound, float upper_bound) {
  // online quant lhs
  const size_t lhs_ref_size_qa8dx = m * (k + sizeof(int32_t) + sizeof(float));

  std::vector<uint8_t> lhs_qa8dx(lhs_ref_size_qa8dx);

  __fallback_quant_qa8dx_f32(m, k, (const float *)lhs_native_mtx_f32,
                             (int8_t *)lhs_qa8dx.data());

  // do matmul
  if (is_nxk) {
    __fallback_matmul_mxn_mxk_nxk_f32_qa8dx_qs4cx(
      m, n, k, (const int8_t *)lhs_qa8dx.data(),
      (const uint8_t *)rhs_native_mtx_qs4cx, (const float *)rhs_scales_f32,
      dst_act_mtx_f32, lower_bound, upper_bound);
  } else {
    __fallback_matmul_mxn_mxk_kxn_f32_qa8dx_qs4cx(
      m, n, k, (const int8_t *)lhs_qa8dx.data(),
      (const uint8_t *)rhs_native_mtx_qs4cx, (const float *)rhs_scales_f32,
      dst_act_mtx_f32, lower_bound, upper_bound);
  }
}

void gemm_qai8dxp_qsi4cxp(size_t m, size_t n, size_t k,
                          void *lhs_native_mtx_f32, void *rhs_packed_mtx_qs4cx,
                          float *dst_act_mtx_f32, size_t idx_variant,
                          float lower_bound, float upper_bound) {
  __fallback_gemm_qai8dxp_qsi4cxp_packed(m, n, k, lhs_native_mtx_f32,
                                         rhs_packed_mtx_qs4cx, dst_act_mtx_f32,
                                         idx_variant, lower_bound, upper_bound);
}

// Dispatches to the AVX2 causal depthwise Conv1D prefill kernel.
void causal_depthwise_conv1d_k3(const float *input, const float *packed_weight,
                                const float *bias, float *output,
                                unsigned int B, unsigned int H,
                                unsigned int W) {
  nntrainer::avx2::causal_depthwise_conv1d_k3(input, packed_weight, bias,
                                              output, B, H, W);
}

// Dispatches to the AVX2 single-token decode kernel.
void causal_depthwise_conv1d_k3_decode(const float *x_cur,
                                       const float *packed_weight, float *state,
                                       float *y_cur, unsigned int W) {
  nntrainer::avx2::causal_depthwise_conv1d_k3_decode(x_cur, packed_weight,
                                                     state, y_cur, W);
}

} /* namespace nntrainer */
