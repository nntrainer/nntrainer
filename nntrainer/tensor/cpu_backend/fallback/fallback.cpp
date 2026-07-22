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
#include <cmath>
#include <compute_ops.h>
#include <fallback.h>
#include <fallback_internal.h>
#include <nntrainer_error.h>

namespace nntrainer {

void init_backend() {
  // Fallback build has no GGML / OpenBLAS to set up — bind the CPU
  // ops table directly. Same shape as the ARM / x86 init_backend
  // entry points so callers can use ensureComputeOps() uniformly.
  g_compute_ops = get_cpu_ops();
}

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

void tanh_gelu(const unsigned int N, const float *X, float *Y) {
  __fallback_tanh_gelu(N, X, Y);
}

void tanh_gelu_v2(const unsigned int N, const float *X, float *Y) {
  __fallback_tanh_gelu(N, X, Y);
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
  return __fallback_gemm_q4_0<float>(M, N, K, A, lda, B, ldb, C, ldc);
}

template <>
void gemm_q8_0(const unsigned int M, const unsigned int N, const unsigned int K,
               const float *A, const unsigned int lda, const void *B,
               const unsigned int ldb, float *C, const unsigned int ldc) {
  return __fallback_gemm_q8_0<float>(M, N, K, A, lda, B, ldb, C, ldc);
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
                 const unsigned int M, const unsigned int N,
                 ml::train::ISA target) {
  switch (target) {
  case ml::train::ISA::ARM:
    return __fallback_repack_q4_0_to_q4_0_4(W, repacked_W, data_size, M, N);
  case ml::train::ISA::X86:
  case ml::train::ISA::DEFAULT:
  default:
    return __fallback_repack_q4_0_to_q4_0_8(W, repacked_W, data_size, M, N);
  }
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
                                        int head_dim, size_t local_window_size,
                                        int head_start, int head_end) {
  __fallback_compute_fp16vcache_fp32_transposed(
    row_num, in, vcache, output, num_cache_head, gqa_size, head_dim,
    local_window_size, head_start, head_end);
}

template <>
void compute_kcaches(const float *in, const uint16_t *kcache, float *output,
                     int num_rows, int num_cache_head, int head_dim,
                     int gqa_size, int tile_size, size_t local_window_size,
                     int head_start, int head_end) {
  __fallback_compute_kcaches<uint16_t>(
    in, kcache, output, num_rows, num_cache_head, head_dim, gqa_size, tile_size,
    local_window_size, head_start, head_end);
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

void depthwise_conv2d_fp32(const float *input, const float *kernel,
                           float *output, unsigned int batch,
                           unsigned int channels, unsigned int in_h,
                           unsigned int in_w, unsigned int out_h,
                           unsigned int out_w, unsigned int kh, unsigned int kw,
                           unsigned int stride_h, unsigned int stride_w,
                           unsigned int pad_top, unsigned int pad_left,
                           unsigned int dilation_h, unsigned int dilation_w) {
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
  __fallback_depthwise_conv2d_fp16(
    input, kernel, output, batch, channels, in_h, in_w, out_h, out_w, kh, kw,
    stride_h, stride_w, pad_top, pad_left, dilation_h, dilation_w);
}
#endif

void create_q4_0_weights(const uint8_t *int4_weight, uint8_t *q4_0_weight) {
  __fallback_create_q4_0_weights(int4_weight, q4_0_weight);
}

void transform_int4_osv32_isv2_to_q4_0(size_t N, size_t K,
                                       const uint8_t *osv32_weights,
                                       const uint16_t *osv32_scales,
                                       size_t scale_group_size,
                                       void *dst_q4_0x) {
  __fallback_transform_int4_osv32_isv2_to_q4_0(
    N, K, osv32_weights, osv32_scales, scale_group_size, 8, dst_q4_0x);
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

} /* namespace nntrainer */
