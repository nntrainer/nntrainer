// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file fallback_internal.cpp
 * @date   23 April 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Fallback computation functions (raw implementation)
 *
 */

#include <algorithm>
#include <assert.h>
#include <climits>
#include <cmath>
#include <cstdint>
#include <fallback_internal.h>
#include <limits>
#include <q4_0_utils.h>
#include <stdexcept>
#include <tensor_dim.h>
#include <util_func.h>

#define sgemv_loop(ci, cj, cM, cN)                                             \
  do {                                                                         \
    float y0;                                                                  \
    unsigned int i, j;                                                         \
    for (ci = 0; ci != cM; ci++) {                                             \
      y0 = 0.0f;                                                               \
      if (beta != 0.0f) {                                                      \
        y0 = Y[ci * incY] * beta;                                              \
      }                                                                        \
      for (cj = 0; cj != cN; cj++)                                             \
        y0 += A[i + j * lda] * X[cj * incX];                                   \
      Y[ci * incY] = y0;                                                       \
    }                                                                          \
  } while (0);
namespace nntrainer {

/**
 * @brief struct of q4_0x8 block
 */
struct block_q4_0x8 {
  uint16_t d[8];   // 16B
  uint8_t qs[128]; // 16 x u64
};

void __fallback_sscal(const unsigned int N, const float alpha, float *X,
                      const unsigned int incX) {
  assert(incX > 0);
  for (unsigned int i = 0; i < N; ++i)
    X[i * incX] = alpha * X[i * incX];
}

float __fallback_snrm2(const unsigned int N, const float *X,
                       const unsigned int incX) {
  assert(incX > 0);
  float sum = 0.0f;
  float tmp;

  for (unsigned int i = 0; i < N; i++) {
    tmp = X[i * incX];
    sum += tmp * tmp;
  }
  return sqrt(sum);
}

void __fallback_copy_s16_fp32(const unsigned int N, const int16_t *X,
                              float *Y) {
  for (unsigned int i = 0; i < N; ++i) {
    Y[i] = X[i];
  }
}

void __fallback_copy_u16_fp32(const unsigned int N, const uint16_t *X,
                              float *Y) {
  for (unsigned int i = 0; i < N; ++i) {
    Y[i] = X[i];
  }
}

void __fallback_copy_fp32_u32(const unsigned int N, const float *X,
                              uint32_t *Y) {
  for (unsigned int i = 0; i < N; ++i) {
    Y[i] = static_cast<uint32_t>(X[i]);
  }
}

void __fallback_copy_fp32_u16(const unsigned int N, const float *X,
                              uint16_t *Y) {
  for (unsigned int i = 0; i < N; ++i) {
    Y[i] = static_cast<uint16_t>(X[i]);
  }
}

void __fallback_copy_fp32_u8(const unsigned int N, const float *X, uint8_t *Y) {
  for (unsigned int i = 0; i < N; ++i) {
    Y[i] = static_cast<uint8_t>(X[i]);
  }
}

void __fallback_copy_fp32_s16(const unsigned int N, const float *X,
                              int16_t *Y) {
  for (unsigned int i = 0; i < N; ++i) {
    Y[i] = static_cast<int16_t>(X[i]);
  }
}

void __fallback_copy_fp32_s8(const unsigned int N, const float *X, int8_t *Y) {
  for (unsigned int i = 0; i < N; ++i) {
    Y[i] = static_cast<int8_t>(X[i]);
  }
}

void __fallback_copy_s16(const unsigned int N, const int16_t *X, int16_t *Y) {
  for (unsigned int i = 0; i < N; ++i) {
    Y[i] = X[i];
  }
}

void __fallback_copy_u16(const unsigned int N, const uint16_t *X, uint16_t *Y) {
  for (unsigned int i = 0; i < N; ++i) {
    Y[i] = X[i];
  }
}

void __fallback_scopy(const unsigned int N, const float *X,
                      const unsigned int incX, float *Y,
                      const unsigned int incY) {
  assert(incX > 0 && incY > 0);
  for (unsigned int i = 0; i < N; ++i)
    Y[i * incY] = X[i * incX];
}

void __fallback_scopy(const unsigned int N, const uint8_t *X,
                      const unsigned int incX, uint8_t *Y,
                      const unsigned int incY) {
  for (unsigned int idx = 0; idx < N; idx++) {
    Y[idx * incX] = X[idx * incY];
  }
}

void __fallback_scopy(const unsigned int N, const int8_t *X,
                      const unsigned int incX, int8_t *Y,
                      const unsigned int incY) {
  for (unsigned int idx = 0; idx < N; idx++) {
    Y[idx * incX] = X[idx * incY];
  }
}

void __fallback_scopy_int4_to_float32(const unsigned int N, const uint8_t *X,
                                      const unsigned int incX, float *Y,
                                      const unsigned int incY) {
  for (unsigned int idx = 0; idx < N; idx++) {
    Y[2 * idx] = static_cast<float>(X[idx] >> 4);
    Y[2 * idx + 1] = static_cast<float>(X[idx] & 0x0f);
  }
}

/// @todo function with the same internal representation should be merged.
void __fallback_scopy_uint8_to_float32(const unsigned int N, const uint8_t *X,
                                       const unsigned int incX, float *Y,
                                       const unsigned int incY) {
  for (unsigned int idx = 0; idx < N; idx++) {
    Y[idx * incX] = X[idx * incY];
  }
}

void __fallback_scopy_int8_to_float32(const unsigned int N, const int8_t *X,
                                      const unsigned int incX, float *Y,
                                      const unsigned int incY) {
  for (unsigned int idx = 0; idx < N; idx++) {
    Y[idx * incX] = X[idx * incY];
  }
}

float __fallback_sdot(const unsigned int N, const float *X,
                      const unsigned int incX, const float *Y,
                      const unsigned int incY) {
  float ret = 0;
  for (unsigned int i = 0; i < N; ++i) {
    ret += X[i * incX] * Y[i * incY];
  }
  return ret;
}

void __fallback_saxpy(const unsigned int N, const float alpha, const float *X,
                      const unsigned int incX, float *Y,
                      const unsigned int incY) {
  assert(incX > 0 && incY > 0);
  for (unsigned int i = 0; i < N; ++i)
    Y[i * incY] = Y[i * incY] + X[i * incX] * alpha;
}

void __fallback_sgemm(const unsigned int TStorageOrder, bool TransA,
                      bool TransB, const unsigned int M, const unsigned int N,
                      const unsigned int K, const float alpha, const float *A,
                      const unsigned int lda, const float *B,
                      const unsigned int ldb, const float beta, float *C,
                      const unsigned int ldc) {
  for (unsigned int m = 0; m < M; ++m) {
    for (unsigned int n = 0; n < N; ++n) {
      double c = 0.0;
      float c_old = C[m * ldc + n];
      for (unsigned int k = 0; k < K; ++k) {
        float a, b;
        a = ((TransA == true) ? A[k * lda + m] : A[m * lda + k]);
        b = ((TransB == true) ? B[n * ldb + k] : B[k * ldb + n]);
        c += a * b;
      }
      C[m * ldc + n] = alpha * c;
      if (beta != 0.0f) {
        C[m * ldc + n] += beta * c_old;
      }
    }
  }
}

void __fallback_sgemv(const unsigned int TStorageOrder, bool TransA,
                      const unsigned int M, const unsigned int N,
                      const float alpha, const float *A, const unsigned int lda,
                      const float *X, const unsigned int incX, const float beta,
                      float *Y, const unsigned int incY) {

  if (TransA == true) {
    sgemv_loop(i, j, N, M);
  } else {
    sgemv_loop(j, i, M, N);
  }
}

unsigned int __fallback_isamax(const unsigned int N, const float *X,
                               const unsigned int incX) {
  unsigned int max_idx = 0;
  float max_val = X[0];
  for (unsigned int n = 1; n < N; n += incX) {
    float cur_val = std::abs(X[n]);
    if (cur_val > max_val) {
      max_val = cur_val;
      max_idx = n;
    }
  }

  return max_idx;
}

template <>
void __fallback_sine(const unsigned int N, float *X, float *Y, float alpha,
                     float beta) {
  unsigned int i = 0;
  while (i < N) {
    Y[i] = std::sin(alpha * X[i]) * beta;
    ++i;
  }
}

template <>
void __fallback_cosine(const unsigned int N, float *X, float *Y, float alpha,
                       float beta) {
  unsigned int i = 0;
  while (i < N) {
    Y[i] = std::cos(alpha * X[i]) * beta;
    ++i;
  }
}

void __fallback_inv_sqrt_inplace(const unsigned int N, float *X) {
  for (unsigned int i = 0; i < N; ++i) {
    X[i] = 1 / std::sqrt(static_cast<float>(X[i]));
  }
}

void __fallback_ele_mul(const unsigned int N, const float *X, const float *Y,
                        float *Z, float alpha, float beta,
                        unsigned int i_stride, unsigned int o_stride) {
  for (unsigned int i = 0; i < N; ++i) {
    *Z = *X * alpha * *Y + ((0.0f == beta) ? 0.0f : beta * *Z);
    X += o_stride;
    Y += i_stride;
    Z += o_stride;
  }
}

void __fallback_ele_add(const unsigned int N, const float *X, const float *Y,
                        float *Z, float alpha, float beta,
                        unsigned int i_stride, unsigned int o_stride) {
  for (unsigned int i = 0; i < N; ++i) {
    *Z = *X + alpha * *Y + ((0.0f == beta) ? 0.0f : beta * *Z);
    X += o_stride;
    Y += i_stride;
    Z += o_stride;
  }
}

void __fallback_ele_sub(const unsigned N, const float *X, const float *Y,
                        float *Z, float alpha, float beta,
                        unsigned int i_stride, unsigned int o_stride) {
  for (unsigned int i = 0; i < N; ++i) {
    *Z = *X - alpha * *Y + ((0.0f == beta) ? 0.0f : beta * *Z);
    X += o_stride;
    Y += i_stride;
    Z += o_stride;
  }
}

void __fallback_ele_div(const unsigned N, const float *X, const float *Y,
                        float *Z, float alpha, float beta,
                        unsigned int i_stride, unsigned int o_stride) {
  for (unsigned int i = 0; i < N; ++i) {
    *Z = *X / (alpha * *Y) + ((0.0f == beta) ? 0.0f : beta * *Z);
    X += o_stride;
    Y += i_stride;
    Z += o_stride;
  }
}

void __fallback_transpose_matrix(const unsigned int M, const unsigned int N,
                                 const float *src, unsigned int ld_src,
                                 float *dst, unsigned int ld_dst) {
  for (unsigned int i = 0; i < M; i++) {
    for (unsigned int j = 0; j < N; j++) {
      dst[i + j * ld_dst] = src[i * ld_src + j];
    }
  }
}

bool __fallback_isValid(const unsigned int N, const float *X) {
  for (size_t i = 0; i < N; ++i) {
    if (!isFloatValid(*X)) {
      return false;
    }
    ++X;
  }

  return true;
}

void __fallback_unpack_q4_0x8_transpose16(const void *src,
                                          uint16_t *__restrict dT,
                                          uint16_t *__restrict qsT, int N,
                                          int K, int CT) {
  const auto *x = static_cast<const block_q4_0x8 *>(src);

  const int groups_N8 = N / 8;    // # of 8-row groups
  const int cols_scales = K / 32; // # subblocks along K (scales columns)
  const uint64_t mask = 0x8888888888888888ULL; // flip MSB of each nibble

  // Tile over columns to keep working set small.
  for (int c0 = 0; c0 < cols_scales; c0 += CT) {
    const int c1 = std::min(c0 + CT, cols_scales);

    // Process rows in natural 8-row groups for source-friendly access
    for (int b = 0; b < groups_N8; ++b) {
      // For each column in the tile, read the source block contiguously
      for (int c = c0; c < c1; ++c) {
        const block_q4_0x8 &blk = x[b * cols_scales + c];

        // Precompute column bases in the transposed outputs
        unsigned short *__restrict dT_c = dT + c * N; // column c in dT
        unsigned short *__restrict qsT_c0 =
          qsT + (c * 8) * N; // first of 8 columns for this subblock

        // Walk the 8 rows inside this block group
        for (int off = 0; off < 8; ++off) {
          const int r = b * 8 + off; // absolute row index in [0..N-1]

          // ---------- SCALES (fp16), transposed on the fly ----------
          dT_c[r] = blk.d[off];

          // ---------- QUANTS (bytes → XOR → swizzle → 8×u16), transposed
          // ---------- load two u64 chunks for this row
          uint64_t v0, v1;
          std::memcpy(&v0, blk.qs + 8 * off, 8);
          std::memcpy(&v1, blk.qs + 8 * (off + 8), 8);
          v0 ^= mask;
          v1 ^= mask;

          unsigned char in[16];
          std::memcpy(in + 0, &v0, 8);
          std::memcpy(in + 8, &v1, 8);

          // nibble-lane swizzle (identical to your reference)
          unsigned char out[16];
          for (int i = 0; i < 8; ++i) {
            const unsigned char x0 = in[2 * i + 0];
            const unsigned char x1 = in[2 * i + 1];
            out[i + 0] = (unsigned char)((x0 & 0x0F) | ((x1 & 0x0F) << 4));
            out[i + 8] = (unsigned char)(((x0 & 0xF0) >> 4) | (x1 & 0xF0));
          }

          // pack to 8×u16 and store to transposed columns j = c*8 .. c*8+7 at
          // row r
          for (int t = 0; t < 8; ++t) {
            const unsigned short w =
              (unsigned short)((unsigned short)out[2 * t + 0] |
                               ((unsigned short)out[2 * t + 1] << 8));
            qsT_c0[t * N + r] = w; // column (c*8 + t), row r
          }
        } // off
      }   // c in tile
    }     // b
  }       // c0 tiles
}

template <>
void __fallback_calc_trigonometric_vals_dup(unsigned int N_half, float *angle,
                                            float *cos_, float *sin_,
                                            unsigned int from,
                                            float attention_scaling) {
  throw std::runtime_error(
    "Error: No implementation of rotary embedding layer incremental_forwarding "
    "with SIMD acceleration except for NEON!");
}

void __fallback_swiglu(const unsigned int N, float *X, float *Y, float *Z) {
  unsigned int i = 0;
  while (i < N) {
    X[i] = (Y[i] / (1.f + std::exp(-Y[i]))) * Z[i];
    ++i;
  }
}

void __fallback_swiglu(const unsigned int N, float *X, float *Y, float *Z,
                       float alpha) {
  unsigned int i = 0;
  while (i < N) {
    X[i] = (Y[i] / (1.f + std::exp(-alpha * Y[i]))) * Z[i];
    ++i;
  }
}

float __fallback_max(const unsigned int N, float *X) {
  std::vector<float> v(X, X + N);
  return *std::max_element(v.begin(), v.end());
}

void __fallback_softmax(const unsigned int N, float *X, float *Y) {
  unsigned int i = 0;
  float sum = 0.f;
  float max_x = __fallback_max(N, X);
  while (i < N) {
    sum += std::exp(X[i] - max_x);
    ++i;
  }
  i = 0;
  while (i < N) {
    Y[i] = std::exp(X[i] - max_x) / sum;
    ++i;
  }
}

template <>
void __fallback_gemm_q4_0(const unsigned int M, const unsigned int N,
                          const unsigned int K, const float *A,
                          const unsigned int lda, const void *B,
                          const unsigned int ldb, float *C,
                          const unsigned int ldc) {
  throw std::runtime_error("NYI : __fallback_gemm_q4_0");
}

void __fallback_gemm_q4_K(const unsigned int M, const unsigned int N,
                          const unsigned int K, const float *A,
                          const unsigned int lda, const void *B,
                          const unsigned int ldb, float *C,
                          const unsigned int ldc) {
  throw std::runtime_error("NYI : __fallback_gemm_q4_K");
}

float __fallback_dot_q6_K_q8_K(const unsigned int K, const void *v_q6_K,
                               const void *v_q8_K) {
  throw std::runtime_error("NYI : __fallback_dot_q6_K_q8_K");
  return 0;
}

float __fallback_dot_q6_K_f32(const unsigned int K, const void *v_q6_K,
                              const float *f) {
  throw std::runtime_error("NYI : __fallback_dot_q6_K_f32");
  return 0;
}

template <>
void __fallback_gemm_q6_K(const unsigned int M, const unsigned int N,
                          const unsigned int K, const float *A,
                          const unsigned int lda, const void *B,
                          const unsigned int ldb, float *C,
                          const unsigned int ldc) {
  throw std::runtime_error("NYI : __fallback_gemm_q6_K");
}

size_t __fallback_quantize_q4_0(const float *src, void *dst, int64_t nrow,
                                int64_t n_per_row, const float *quant_weights) {
  throw std::runtime_error("NYI : __fallback_quantize_q4_0");
  return 1;
}

size_t __fallback_quantize_q4_K(const float *src, void *dst, int64_t nrow,
                                int64_t n_per_row, const float *quant_weights) {
  throw std::runtime_error("NYI : __fallback_quantize_q4_K");
  return 1;
}

size_t __fallback_quantize_q6_K(const float *src, void *dst, int64_t nrow,
                                int64_t n_per_row, const float *quant_weights) {
  throw std::runtime_error("NYI : __fallback_quantize_q4_K");
  return 1;
}

void __fallback_dequantize_row_q4_K(const void *x_raw, float *y, int64_t k) {
  throw std::runtime_error("NYI : __fallback_dequantize_row_q4_K");
}

void __fallback_dequantize_row_q4_0(const void *x_raw, float *y, int64_t k) {
  throw std::runtime_error("NYI : __fallback_dequantize_row_q4_0");
}

void __fallback_dequantize_row_q6_K(const void *x, float *y, int64_t k) {
  throw std::runtime_error("NYI : __fallback_dequantize_row_q6_K");
}

void __fallback_quantize_row_q6_K(const float *src, void *dst, int64_t k) {
  throw std::runtime_error("NYI : __fallback_quantize_row_q6_K");
}

template <>
void __fallback_quantize_row_q8_K(const float *src, void *dst, int64_t k) {
  throw std::runtime_error("NYI : __fallback_quantize_row_q8_K");
}

template <>
void __fallback_dequantize_row_q8_K(const void *x, float *y, int64_t k) {
  throw std::runtime_error("NYI : __fallback_dequantize_row_q8_K");
}

void __fallback_repack_q4_0_to_q4_0_4(void *W, void *repacked_W,
                                      size_t data_size, const unsigned int M,
                                      const unsigned int N) {
  throw std::runtime_error("NYI : __fallback_repack_q4_0_to_q4_0_4");
}

void __fallback_repack_q4_0_to_q4_0_8(void *W, void *repacked_W,
                                      size_t data_size, const unsigned int M,
                                      const unsigned int N) {
  throw std::runtime_error("NYI : __fallback_repack_q4_0_to_q4_0_8");
}

void __fallback_repack_q4_K_to_q4_K_8(void *W, void *repacked_W,
                                      size_t data_size, const unsigned int M,
                                      const unsigned int N) {
  throw std::runtime_error("NYI : __fallback_repack_q4_K_to_q4_K_8");
}

void __fallback_unpack_q4_0_8_to_q4_0(const void *in_q4_0x, void *out_q4_0,
                                      size_t data_size, const unsigned int M,
                                      const unsigned int N) {
  throw std::runtime_error("NYI : __fallback_unpack_q4_0_8_to_q4_0");
}

void __fallback_softmax_row_inplace(float *qk_out, size_t start_row,
                                    size_t end_row, size_t num_heads) {
  throw std::runtime_error("NYI : __fallback_softmax_row_inplace");
}

void __fallback_softmax_row(float *qk_out, size_t start_row, size_t end_row,
                            size_t num_heads) {
  throw std::runtime_error("NYI : __fallback_softmax_row");
}

void __fallback_compute_fp16vcache_fp32_transposed(
  int row_num, const float *in, const uint16_t *vcache, float *output,
  int num_cache_head, int gqa_size, int head_dim, size_t local_window_size) {
  throw std::runtime_error(
    "NYI : __fallback_compute_fp16vcache_fp32_transposed");
}

template <>
void __fallback_compute_kcaches(const float *in, const uint16_t *kcache,
                                float *output, int num_rows, int num_cache_head,
                                int head_dim, int gqa_size, int tile_size,
                                size_t local_window_size) {
  throw std::runtime_error("NYI : __fallback_compute_kcaches");
}

void __fallback_compute_rotary_emb_value(unsigned int width, unsigned int dim,
                                         unsigned int half_, float *inout,
                                         void *output, const float *cos_,
                                         const float *sin_,
                                         bool only_convert_to_fp16) {
  throw std::runtime_error("NYI : __fallback_compute_rotary_emb_value");
}

void __fallback_rms_norm_wrt_width_fp32_intrinsic(const float *__restrict X,
                                                  float *__restrict Y, size_t H,
                                                  size_t W, float epsilon) {
  throw std::runtime_error(
    "NYI : __fallback_rms_norm_wrt_width_fp32_intrinsic");
}

template <>
void __fallback_rms_norm_wrt_width_fp16_intrinsic(const float *__restrict X,
                                                  float *__restrict Y, size_t H,
                                                  size_t W, float epsilon) {
  throw std::runtime_error(
    "NYI : __fallback_rms_norm_wrt_width_fp16_intrinsic");
}

template <>
void __fallback_clamp(const float *input, float *output, size_t length,
                      float lower_bound, float upper_bound) {
  for (int i = 0; i < length; ++i) {
    output[i] = std::clamp(input[i], lower_bound, upper_bound);
  }
}

void __fallback_create_q4_0_weights(const uint8_t *int4_weight,
                                    uint8_t *q4_0_weight) {
  for (int i = 0; i < 8; i++) {
    char v0 = int4_weight[i] & 0xF;
    char v1 = (int4_weight[i] >> 4) & 0xF;
    char v2 = int4_weight[8 + i] & 0xF;
    char v3 = (int4_weight[8 + i] >> 4) & 0xF;
    q4_0_weight[2 * i] = (v0 | (v2 << 4));
    q4_0_weight[2 * i + 1] = (v1 | (v3 << 4));
  }
}

void __fallback_transform_int4_osv32_isv2_to_q4_0(size_t N, size_t K,
                                                  const uint8_t *osv32_weights,
                                                  const uint16_t *osv32_scales,
                                                  size_t scale_group_size,
                                                  void *dst_q4_0x) {
  throw std::runtime_error(
    "NYI : __fallback_transform_int4_osv32_isv2_to_q4_0");
}

} // namespace nntrainer
