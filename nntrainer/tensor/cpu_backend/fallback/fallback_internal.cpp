// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file fallback_internal.cpp
 * @date   23 April 2024
 * @see    https://github.com/nntrainer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Fallback computation functions (raw implementation)
 *
 */

#include <algorithm>
#include <assert.h>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fallback_internal.h>
#include <fp16.h>
#include <limits>
#include <nntr_ggml_impl.h>
#include <q4_0_utils.h>
#include <q8_0_tensor.h>
#include <stdexcept>
#include <tensor_dim.h>
#include <thread_manager.h>
#include <util_func.h>
#include <vector>

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
  for (unsigned int i = 0; i < N_half; ++i) {
    float angle_val = from * angle[i] * attention_scaling;
    cos_[i] = std::cos(angle_val);
    cos_[i + N_half] = std::cos(angle_val);

    sin_[i] = std::sin(angle_val);
    sin_[i + N_half] = std::sin(angle_val);
  }
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

void __fallback_tanh_gelu(const unsigned int N, const float *X, float *Y) {
  for (unsigned int i = 0; i < N; ++i) {
    float x = X[i];
    Y[i] = 0.5f * x *
           (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
  }
}

void __fallback_tanh_gelu_mul(const unsigned int N, float *X, float *Y,
                              float *Z) {
  for (unsigned int i = 0; i < N; ++i) {
    float y = Y[i];
    float z = Z[i];
    X[i] = 0.5f * y *
           (1.0f + std::tanh(0.7978845608f * (y + 0.044715f * y * y * y))) * z;
  }
}

void __fallback_gelu_v2(const unsigned int N, const float *X, float *Y) {
  for (unsigned int i = 0; i < N; ++i) {
    float x = X[i];

    Y[i] = 0.5f * x * (1.0f + std::erf(x * 0.7071067811f));
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

namespace {
// IEEE 754 half-precision (fp16) -> single-precision (fp32) bit cast.
static inline float fp16_bits_to_fp32(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  uint32_t exp = (h & 0x7C00u) >> 10;
  uint32_t mant = h & 0x03FFu;
  uint32_t f;
  if (exp == 0u) {
    if (mant == 0u) {
      f = sign;
    } else {
      while ((mant & 0x0400u) == 0u) {
        mant <<= 1;
        exp -= 1u;
      }
      exp += 1u;
      mant &= ~0x0400u;
      f = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
    }
  } else if (exp == 0x1Fu) {
    f = sign | 0x7F800000u | (mant << 13);
  } else {
    f = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
  }
  float out;
  std::memcpy(&out, &f, sizeof(out));
  return out;
}
} // namespace

template <>
void __fallback_gemm_q8_0(const unsigned int M, const unsigned int N,
                          const unsigned int K, const float *A,
                          const unsigned int lda, const void *B,
                          const unsigned int ldb, float *C,
                          const unsigned int ldc) {
  if (K % QK8_0 != 0) {
    throw std::runtime_error("gemm_q8_0: K must be a multiple of 32 (got K=" +
                             std::to_string(K) + ")");
  }
  const unsigned int n_blocks_k = K / QK8_0;

  std::vector<nntrainer::block_q8_0> A_q8(static_cast<size_t>(M) * n_blocks_k);
  for (unsigned int m = 0; m < M; ++m) {
    nntr_quantize_row_q8_0(A + static_cast<size_t>(m) * lda,
                           A_q8.data() + static_cast<size_t>(m) * n_blocks_k,
                           static_cast<int64_t>(K));
  }

  const auto *B_blocks = reinterpret_cast<const nntrainer::block_q8_0 *>(B);
  const unsigned int b_row_blocks = ldb / QK8_0;

  for (unsigned int m = 0; m < M; ++m) {
    const nntrainer::block_q8_0 *a_row =
      A_q8.data() + static_cast<size_t>(m) * n_blocks_k;
    for (unsigned int n = 0; n < N; ++n) {
      const nntrainer::block_q8_0 *b_row =
        B_blocks + static_cast<size_t>(n) * b_row_blocks;
      float acc = 0.0f;
      for (unsigned int blk = 0; blk < n_blocks_k; ++blk) {
        const nntrainer::block_q8_0 &a = a_row[blk];
        const nntrainer::block_q8_0 &b = b_row[blk];
        int32_t sumi = 0;
        for (int l = 0; l < QK8_0; ++l) {
          sumi += static_cast<int32_t>(a.qs[l]) * static_cast<int32_t>(b.qs[l]);
        }
        const float a_d = fp16_bits_to_fp32(a.d);
        const float b_d = fp16_bits_to_fp32(b.d);
        acc += a_d * b_d * static_cast<float>(sumi);
      }
      C[static_cast<size_t>(m) * ldc + n] = acc;
    }
  }
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
  int num_cache_head, int gqa_size, int head_dim, size_t local_window_size,
  int head_start, int head_end) {
  throw std::runtime_error(
    "NYI : __fallback_compute_fp16vcache_fp32_transposed");
}

template <>
void __fallback_compute_kcaches(const float *in, const uint16_t *kcache,
                                float *output, int num_rows, int num_cache_head,
                                int head_dim, int gqa_size, int tile_size,
                                size_t local_window_size, int head_start,
                                int head_end) {
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
  for (size_t h = 0; h < H; ++h) {
    const float *rowX = X + h * W;
    float *rowY = Y + h * W;

    // Use FP32 accumulator to avoid overflow
    float sum_sq = 0.F;
    for (size_t i = 0; i < W; ++i) {
      sum_sq += rowX[i] * rowX[i];
    }

    float mean_single = sum_sq / W;
    float scale_single = 1.F / std::sqrt(mean_single + epsilon);

    for (size_t i = 0; i < W; ++i) {
      rowY[i] = rowX[i] * scale_single;
    }
  }
}

template <>
void __fallback_rms_norm_wrt_width_fp16_intrinsic(const float *__restrict X,
                                                  float *__restrict Y, size_t H,
                                                  size_t W, float epsilon) {
  throw std::runtime_error("ERROR : rms_norm_wrt_width_fp16_intrinsic(float *) "
                           "is deprecated due to overflow in fp16");
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
                                                  int q4_0x_block_size,
                                                  void *dst_q4_0x) {
  Q4_0Utils::transformQ4_0x_FromInt4(N, K, osv32_weights, osv32_scales,
                                     scale_group_size, q4_0x_block_size,
                                     dst_q4_0x);
}

// anonymous namespace for kai fallback
namespace {

constexpr int INT4_MIN = -8;
constexpr int INT4_MAX = 7;

size_t roundup(size_t a, size_t b) { return ((a + b - 1) / b) * b; }

inline size_t num_blocks_per_row(size_t k, size_t bl) { return k / bl; }

inline size_t num_bytes_per_block_qs4c32(size_t bl) {
  return (bl / 2) + sizeof(int16_t);
}

inline size_t num_bytes_per_block_qs8c32(size_t bl) {
  return bl + sizeof(int16_t);
}

} // namespace

void __fallback_quant_nxk_qs4cx_f32(size_t n, size_t k, const float *rhs_f32,
                                    uint8_t *rhs_qs4cx, float *rhs_scales_f32) {
  const size_t rhs_qs4cx_stride = (roundup(k, 2) / 2);

  // Make sure the output is filled with zeros
  std::memset(rhs_qs4cx, 0, n * rhs_qs4cx_stride);

  for (size_t n_idx = 0; n_idx < n; ++n_idx) {
    const float *src_ptr = rhs_f32 + n_idx * k;

    float max0 = -FLT_MAX;
    float min0 = FLT_MAX;

    // Find min/max for each channel
    for (size_t k_idx = 0; k_idx < k; ++k_idx) {
      const float src0_0 = src_ptr[k_idx];

      max0 = std::max(src0_0, max0);
      min0 = std::min(src0_0, min0);
    }

    // Maximum/minimum int8 values
    const float qmin = (float)INT4_MIN;
    const float qmax = (float)INT4_MAX;

    const float rmin0 = std::min(0.0f, min0);
    const float rmax0 = std::max(0.0f, max0);

    const float scale0 = rmin0 == rmax0 ? 1.f : (qmax - qmin) / (rmax0 - rmin0);

    // Reciprocal to quantize
    const float recip_scale0 = scale0 ? 1.0f / scale0 : 0.0f;

    // Quantize the channels
    for (size_t k_idx = 0; k_idx < k; ++k_idx) {
      const float src0_0 = src_ptr[k_idx];

      // Scale the values
      int32_t v0_s32 = (int32_t)(std::round(src0_0 * scale0));

      // Maximum/minimum int4 values
      v0_s32 = std::max(v0_s32, INT4_MIN);
      v0_s32 = std::min(v0_s32, INT4_MAX);

      const uint8_t v0_u8 = (uint8_t)(v0_s32 + 8);

      const size_t dst_addr = (k_idx / 2) + n_idx * rhs_qs4cx_stride;
      uint8_t rhs_v0 = rhs_qs4cx[dst_addr];

      if ((k_idx % 2) == 0) {
        rhs_v0 |= v0_u8;
      } else {
        rhs_v0 |= (v0_u8 << 4);
      }
      rhs_qs4cx[dst_addr] = rhs_v0;
    }

    rhs_scales_f32[n_idx] = recip_scale0;
  }
}

void __fallback_quant_kxn_qs4cx_f32(size_t n, size_t k, const float *rhs_f32,
                                    uint8_t *rhs_qs4cx, float *rhs_scales_f32) {
  const size_t rhs_qs4cx_stride = (roundup(n, 2) / 2);

  // Make sure the output is filled with zeros
  std::memset(rhs_qs4cx, 0, k * rhs_qs4cx_stride);

  for (size_t n_idx = 0; n_idx < n; ++n_idx) {
    const float *src_ptr = rhs_f32 + n_idx * k;

    float max0 = -FLT_MAX;
    float min0 = FLT_MAX;

    // Find min/max for each channel
    for (size_t k_idx = 0; k_idx < k; ++k_idx) {
      const float src0_0 = src_ptr[k_idx];

      max0 = std::max(src0_0, max0);
      min0 = std::min(src0_0, min0);
    }

    // Maximum/minimum int8 values
    const float qmin = (float)INT4_MIN;
    const float qmax = (float)INT4_MAX;

    const float rmin0 = std::min(0.0f, min0);
    const float rmax0 = std::max(0.0f, max0);

    const float scale0 = rmin0 == rmax0 ? 1.f : (qmax - qmin) / (rmax0 - rmin0);

    // Reciprocal to quantize
    const float recip_scale0 = scale0 ? 1.0f / scale0 : 0.0f;

    // Quantize the channels
    for (size_t k_idx = 0; k_idx < k; ++k_idx) {
      const float src0_0 = src_ptr[k_idx];

      // Scale the values
      int32_t v0_s32 = (int32_t)(std::round(src0_0 * scale0));

      // Maximum/minimum int4 values
      v0_s32 = std::max(v0_s32, INT4_MIN);
      v0_s32 = std::min(v0_s32, INT4_MAX);

      const uint8_t v0_u8 = (uint8_t)(v0_s32 + 8);

      const size_t dst_addr = (n_idx / 2) + k_idx * rhs_qs4cx_stride;
      uint8_t rhs_v0 = rhs_qs4cx[dst_addr];

      if ((n_idx % 2) == 0) {
        rhs_v0 |= v0_u8;
      } else {
        rhs_v0 |= (v0_u8 << 4);
      }
      rhs_qs4cx[dst_addr] = rhs_v0;
    }

    rhs_scales_f32[n_idx] = recip_scale0;
  }
}

void __fallback_quant_qa8dx_f32(size_t m, size_t k, const float *lhs_f32,
                                int8_t *lhs_qa8dx) {
  const size_t dst_stride =
    (k * sizeof(int8_t) + sizeof(float) + sizeof(int32_t));

  const size_t lhs_qa8dx_stride = k;

  for (size_t m_idx = 0; m_idx < m; ++m_idx) {
    const float *src_ptr = lhs_f32 + m_idx * lhs_qa8dx_stride;

    float max0 = -FLT_MAX;
    float min0 = FLT_MAX;

    // Find min/max for each channel
    for (size_t k_idx = 0; k_idx < k; ++k_idx) {
      const float src0_0 = src_ptr[k_idx];

      max0 = std::max(src0_0, max0);
      min0 = std::min(src0_0, min0);
    }

    // Maximum/minimum int8 values
    const float qmin = (float)INT8_MIN;
    const float qmax = (float)INT8_MAX;

    const float rmin0 = std::min(0.0f, min0);
    const float rmax0 = std::max(0.0f, max0);

    const float scale0 = rmin0 == rmax0 ? 1.f : (qmax - qmin) / (rmax0 - rmin0);

    // Reciprocal to quantize
    const float recip_scale0 = scale0 ? 1.0f / scale0 : 0.0f;

    const float descaled_min0 = rmin0 * scale0;
    const float descaled_max0 = rmax0 * scale0;

    const float zero_point_from_min_error0 = qmin + descaled_min0;
    const float zero_point_from_max_error0 = qmax + descaled_max0;

    float zero_point0 =
      zero_point_from_min_error0 + zero_point_from_max_error0 > 0
        ? qmin - descaled_min0
        : qmax - descaled_max0;

    zero_point0 = std::max(zero_point0, qmin);
    zero_point0 = std::min(zero_point0, qmax);

    // Round to nearest integer
    const int32_t nudged_zero_point0 = lrintf(zero_point0);

    int8_t *dst_ptr = (int8_t *)lhs_qa8dx + m_idx * dst_stride;

    // LHS offset at the beginning of the row
    *((float *)(dst_ptr)) = recip_scale0;
    dst_ptr += sizeof(float);
    *((int32_t *)(dst_ptr)) = -nudged_zero_point0;
    dst_ptr += sizeof(int32_t);

    // Quantize the channels
    for (size_t k_idx = 0; k_idx < k; ++k_idx) {
      const float src0_0 = src_ptr[k_idx];

      // Scale the values
      int32_t v0_s32 = (int32_t)(std::round(src0_0 * scale0));

      v0_s32 = v0_s32 + nudged_zero_point0;
      v0_s32 = std::max(v0_s32, static_cast<int32_t>(INT8_MIN));
      v0_s32 = std::min(v0_s32, static_cast<int32_t>(INT8_MAX));
      dst_ptr[0] = (int8_t)v0_s32;
      dst_ptr += sizeof(int8_t);
    }
  }
}

size_t __fallback_get_rhs_packed_size_qsi4cxp_qs4cxs1s0(size_t n, size_t k,
                                                        uint32_t idx_variant,
                                                        bool is_nxk) {
  throw std::runtime_error(
    "NYI : __fallback_get_rhs_packed_size_qsi4cxp_qs4cxs1s0");
  return 1;
}

void __fallback_rhs_pack_qsi4cxp_qs4cxs1s0(size_t n, size_t k, void *rhs_packed,
                                           void *rhs, void *rhs_scales_f32,
                                           uint32_t idx_variant, bool is_nxk) {
  throw std::runtime_error("NYI : __fallback_rhs_pack_qsi4cxp_qs4cxs1s0");
}

void __fallback_matmul_mxn_mxk_nxk_f32_qa8dx_qs4cx(
  size_t m, size_t n, size_t k, const int8_t *lhs_qa8dx,
  const uint8_t *rhs_qs4cx, const float *rhs_scales_f32, float *dst_f32,
  float scalar_min, float scalar_max) {
  const size_t lhs_stride =
    k * sizeof(int8_t) + sizeof(float) + sizeof(int32_t);

  const size_t rhs_qs4cx_stride = (roundup(k, 2) / 2);

  for (size_t m_idx = 0; m_idx < m; ++m_idx) {
    const int8_t *lhs_ptr_start = lhs_qa8dx + m_idx * lhs_stride;

    for (size_t n_idx = 0; n_idx < n; ++n_idx) {
      // Main f32 accumulator
      int32_t iacc = 0;

      const int8_t *lhs_ptr = lhs_ptr_start;
      const uint8_t *rhs_ptr = rhs_qs4cx + n_idx * rhs_qs4cx_stride;

      // Get the LHS quantization parameters stored at the
      // beginning of each row
      const float lhs_scale = *(const float *)lhs_ptr;
      lhs_ptr += sizeof(float);

      const int32_t lhs_offset = *(const int32_t *)lhs_ptr;
      lhs_ptr += sizeof(int32_t);

      for (size_t k_idx = 0; k_idx < k; ++k_idx) {
        // Get the LHS values
        const int32_t lhs_v0 = (int32_t)lhs_ptr[0];

        // Get the RHS values
        const uint8_t rhs_byte = rhs_ptr[0];

        // Unpack the RHS values
        int32_t rhs_v0 = 0;
        if ((k_idx % 2) == 0) {
          rhs_v0 = (((int32_t)(rhs_byte & 0x0F)) - 8);
        } else {
          rhs_v0 = (((int32_t)(rhs_byte >> 4)) - 8);
        }

        iacc += lhs_v0 * rhs_v0;
        iacc += lhs_offset * rhs_v0;

        lhs_ptr += 1;

        // Increment only when k_idx is not a multiple of 2
        rhs_ptr += k_idx % 2;
      }

      // Get the RHS scale
      const float rhs_scale = rhs_scales_f32[n_idx];

      float main_acc = iacc * rhs_scale;

      main_acc = main_acc * lhs_scale;

      // Clamp (min-max) operation
      main_acc = std::max(main_acc, scalar_min);
      main_acc = std::min(main_acc, scalar_max);

      dst_f32[0] = main_acc;
      dst_f32 += 1;
    }
  }
}

void __fallback_matmul_mxn_mxk_kxn_f32_qa8dx_qs4cx(
  size_t m, size_t n, size_t k, const int8_t *lhs_qa8dx,
  const uint8_t *rhs_qs4cx, const float *rhs_scales_f32, float *dst_f32,
  float scalar_min, float scalar_max) {
  const size_t lhs_stride =
    k * sizeof(int8_t) + sizeof(float) + sizeof(int32_t);

  const size_t rhs_qs4cx_stride = (roundup(n, 2) / 2);

  for (size_t m_idx = 0; m_idx < m; ++m_idx) {
    const int8_t *lhs_ptr_start = lhs_qa8dx + m_idx * lhs_stride;

    for (size_t n_idx = 0; n_idx < n; ++n_idx) {
      // Main f32 accumulator
      int32_t iacc = 0;

      const int8_t *lhs_ptr = lhs_ptr_start;
      const uint8_t *rhs_ptr = rhs_qs4cx + (n_idx / 2);

      // Get the LHS quantization parameters stored at the
      // beginning of each row
      const float lhs_scale = *(const float *)lhs_ptr;
      lhs_ptr += sizeof(float);

      const int32_t lhs_offset = *(const int32_t *)lhs_ptr;
      lhs_ptr += sizeof(int32_t);

      for (size_t k_idx = 0; k_idx < k; ++k_idx) {
        // Get the LHS values
        const int32_t lhs_v0 = (int32_t)lhs_ptr[0];

        // Get the RHS values
        const uint8_t rhs_byte = rhs_ptr[0];

        // Unpack the RHS values
        int32_t rhs_v0 = 0;
        if ((n_idx % 2) == 0) {
          rhs_v0 = (((int32_t)(rhs_byte & 0x0F)) - 8);
        } else {
          rhs_v0 = (((int32_t)(rhs_byte >> 4)) - 8);
        }

        iacc += lhs_v0 * rhs_v0;
        iacc += lhs_offset * rhs_v0;

        lhs_ptr += 1;

        // Increment only when k_idx is not a multiple of 2
        rhs_ptr += rhs_qs4cx_stride;
      }

      // Get the RHS scale
      const float rhs_scale = rhs_scales_f32[n_idx];

      float main_acc = iacc * rhs_scale;

      main_acc = main_acc * lhs_scale;

      // Clamp (min-max) operation
      main_acc = std::max(main_acc, scalar_min);
      main_acc = std::min(main_acc, scalar_max);

      dst_f32[0] = main_acc;
      dst_f32 += 1;
    }
  }
};

void __fallback_gemm_qai8dxp_qsi4cxp_packed(size_t m, size_t n, size_t k,
                                            void *lhs, void *rhs_packed_qsi4cxp,
                                            float *dst, uint32_t idx_variant,
                                            float lower_bound,
                                            float upper_bound) {
  throw std::runtime_error("NYI : __fallback_gemm_qai8dxp_qsi4cxp_packed");
}

void __fallback_quant_qs4c32_f32(size_t n, size_t k, size_t bl,
                                 const float *rhs_f32, uint8_t *rhs_qs4c32) {
  const size_t num_blocks_row = num_blocks_per_row(k, bl);
  const size_t num_bytes_block = num_bytes_per_block_qs4c32(bl);
  const size_t dst_stride = num_blocks_row * num_bytes_block;

  for (size_t row_idx = 0; row_idx < n; ++row_idx) {
    const float *src_ptr = rhs_f32 + row_idx * k;

    uint8_t *dst_ptr = (uint8_t *)rhs_qs4c32 + row_idx * dst_stride;

    for (size_t block_idx = 0; block_idx < num_blocks_row; ++block_idx) {
      float amax = 0.0f;
      float max = 0.0f;

      for (size_t b = 0; b < bl; ++b) {
        const float src0_0 = src_ptr[block_idx * bl + b];
        const float asrc0_0 = fabsf(src0_0);

        if (amax < asrc0_0) {
          amax = asrc0_0;
          max = src0_0;
        }
      }

      const float scale = max / -8.0f;
      const float recip_scale = scale ? 1.0f / scale : 0.0f;

      // Store the scale at the beginning of the block
      *((uint16_t *)dst_ptr) = compute_fp32_to_fp16(scale);
      dst_ptr += sizeof(uint16_t);

      const size_t block_size = 32;
      const size_t num_subblocks = bl / 32;

      for (size_t subblock_idx = 0; subblock_idx < num_subblocks;
           ++subblock_idx) {
        for (size_t i = 0; i < block_size / 2; ++i) {
          const size_t src_base_addr =
            block_idx * bl + i + subblock_idx * block_size;
          float v0_f32 = src_ptr[src_base_addr];
          float v1_f32 = src_ptr[src_base_addr + block_size / 2];

          v0_f32 *= recip_scale;
          v1_f32 *= recip_scale;

          const uint8_t v0_u8 =
            (uint8_t)std::min((int8_t)15, (int8_t)(v0_f32 + 8.5f));
          const uint8_t v1_u8 =
            (uint8_t)std::min((int8_t)15, (int8_t)(v1_f32 + 8.5f));

          const uint8_t rhs_v0 = (v1_u8 << 4) | v0_u8;

          dst_ptr[0] = rhs_v0;
          dst_ptr += sizeof(uint8_t);
        }
      }
    }
  }
}

void __fallback_quant_qs8d32_f32(size_t m, size_t k, size_t bl,
                                 const float *lhs_f32, uint8_t *lhs_qs8c32) {
  if (k % bl != 0) {
    throw std::invalid_argument{"k must be a multiple of bl"};
  }
  const size_t num_blocks_row = num_blocks_per_row(k, bl);
  const size_t num_bytes_block = num_bytes_per_block_qs8c32(bl);
  const size_t dst_stride = num_blocks_row * num_bytes_block;

  for (size_t row_idx = 0; row_idx < m; ++row_idx) {
    const float *src_ptr = lhs_f32 + row_idx * k;

    int8_t *dst_ptr = (int8_t *)lhs_qs8c32 + row_idx * dst_stride;

    for (size_t block_idx = 0; block_idx < num_blocks_row; ++block_idx) {
      float amax = 0.0f;

      for (size_t b = 0; b < bl; ++b) {
        const float src0_0 = src_ptr[block_idx * bl + b];
        const float asrc0_0 = fabsf(src0_0);

        if (amax < asrc0_0) {
          amax = asrc0_0;
        }
      }

      const float scale = amax / ((1 << 7) - 1);
      const float recip_scale = scale ? 1.0f / scale : 0.0f;

      // Store the scale at the beginning of the block
      *((uint16_t *)dst_ptr) = compute_fp32_to_fp16(scale);
      dst_ptr += sizeof(uint16_t);

      const size_t block_size = 32;
      const size_t num_subblocks = bl / 32;

      for (size_t subblock_idx = 0; subblock_idx < num_subblocks;
           ++subblock_idx) {
        for (size_t i = 0; i < block_size; ++i) {
          const size_t src_base_addr =
            block_idx * bl + i + subblock_idx * block_size;
          float v0_f32 = src_ptr[src_base_addr];

          v0_f32 *= recip_scale;

          dst_ptr[0] = roundf(v0_f32);
          dst_ptr += sizeof(int8_t);
        }
      }
    }
  }
}

size_t __fallback_get_rhs_packed_size_qsi4c32pscalef16_qsu4c32s16s0(
  size_t n, size_t k, uint32_t idx_variant, bool transB) {
  throw std::runtime_error(
    "NYI : __fallback_get_rhs_packed_size_qsi4c32pscalef16_qsu4c32s16s0");
  return 1;
}

void __fallback_rhs_pack_qsi4c32pscalef16_qsu4c32s16s0(
  size_t n, size_t k, void *rhs_packed, void *rhs, void *rhs_scales_f32,
  uint32_t idx_variant, bool transB) {
  throw std::runtime_error(
    "NYI : __fallback_rhs_pack_qsi4c32pscalef16_qsu4c32s16s0");
}

void __fallback_matmul_f32_qs8d32_qs4c32(size_t m, size_t n, size_t k,
                                         size_t bl, const int8_t *lhs_qa8d32,
                                         const uint8_t *rhs_qs4c32,
                                         float *dst_f32, float scalar_min,
                                         float scalar_max) {
  const size_t num_blocks_row = num_blocks_per_row(k, bl);
  const size_t num_bytes_block_qs4c32 = num_bytes_per_block_qs4c32(bl);
  const size_t num_bytes_block_qs8c32 = num_bytes_per_block_qs8c32(bl);

  const size_t lhs_stride = num_blocks_row * num_bytes_block_qs8c32;
  const size_t rhs_stride = num_blocks_row * num_bytes_block_qs4c32;

  for (size_t row_idx = 0; row_idx < m; ++row_idx) {
    const int8_t *lhs_ptr_start = lhs_qa8d32 + row_idx * lhs_stride;
    for (size_t col_idx = 0; col_idx < n; ++col_idx) {
      // Main f32 accumulator
      float main_acc = 0.0f;

      const size_t block_size = 32;
      const size_t num_subblocks = bl / 32;

      for (size_t block_idx = 0; block_idx < num_blocks_row; ++block_idx) {
        const int8_t *lhs_ptr = lhs_ptr_start;
        const uint8_t *rhs_ptr = rhs_qs4c32 + col_idx * rhs_stride;

        lhs_ptr += block_idx * num_bytes_block_qs8c32;
        rhs_ptr += block_idx * num_bytes_block_qs4c32;

        for (size_t subblock_idx = 0; subblock_idx < num_subblocks;
             ++subblock_idx) {
          int32_t temp_acc = 0;

          // Get the LHS/RHS quantization scale stored at the
          // beginning of each block
          const float lhs_scale =
            compute_fp16_to_fp32(*(const uint16_t *)lhs_ptr);
          const float rhs_scale =
            compute_fp16_to_fp32(*(const uint16_t *)rhs_ptr);

          lhs_ptr += sizeof(uint16_t);
          rhs_ptr += sizeof(uint16_t);

          for (size_t i = 0; i < block_size / 2; ++i) {
            // Get the LHS values
            const int32_t lhs_v0 = (int32_t)lhs_ptr[0];
            const int32_t lhs_v1 = (int32_t)lhs_ptr[block_size / 2];

            // Get the RHS values
            const uint8_t rhs_byte = rhs_ptr[0];

            // Unpack the RHS values
            const int32_t rhs_v0 = (((int32_t)(rhs_byte & 0x0F)) - 8);
            const int32_t rhs_v1 = (((int32_t)(rhs_byte >> 4)) - 8);

            temp_acc += lhs_v0 * rhs_v0;
            temp_acc += lhs_v1 * rhs_v1;

            lhs_ptr += 1;
            rhs_ptr += 1;
          }

          main_acc += temp_acc * lhs_scale * rhs_scale;
        }
      }

      main_acc = std::max(main_acc, scalar_min);
      main_acc = std::min(main_acc, scalar_max);

      dst_f32[0] = main_acc;
      dst_f32 += 1;
    }
  }
}

template <>
void __fallback_gemm_qs8d32p_qs4c32p_packed(size_t m, size_t n, size_t k,
                                            void *lhs, void *rhs_packed_qs4c32p,
                                            float *dst, uint32_t idx_variant,
                                            bool transB, float lower_bound,
                                            float upper_bound) {
  throw std::runtime_error("NYI : __fallback_gemm_qs8d32p_qs4c32p_packed");
}

void __fallback_depthwise_conv2d_fp32(
  const float *input, const float *kernel, float *output, unsigned int batch,
  unsigned int channels, unsigned int in_h, unsigned int in_w,
  unsigned int out_h, unsigned int out_w, unsigned int kh, unsigned int kw,
  unsigned int stride_h, unsigned int stride_w, unsigned int pad_top,
  unsigned int pad_left, unsigned int dilation_h, unsigned int dilation_w) {
  // Scalar depthwise conv — reproduces the fast path from conv2d_layer.cpp.
  // Layout: input/output are NCHW contiguous; kernel is [C,1,kh,kw].
  // Parallelized over (batch*channels) via the nntr thread manager (the same
  // primitive ggml_interface_omp uses), so each channel's small convolution
  // runs independently — this preserves the channel-parallel speedup the old
  // in-layer ParallelBatch fast path had.
  // TODO: NEON/AVX inner-loop specialization in the arch backends.
  auto &tm = ThreadManager::Global();
  tm.parallel_for(0, (size_t)batch * channels, [&](size_t bc) {
    const unsigned int c = (unsigned int)(bc % channels);
    const float *inc = input + bc * in_h * in_w;
    const float *fc = kernel + (size_t)c * kh * kw;
    float *outc = output + bc * out_h * out_w;
    for (unsigned int oh = 0; oh < out_h; ++oh) {
      const int ih0 = (int)oh * (int)stride_h - (int)pad_top;
      for (unsigned int ow = 0; ow < out_w; ++ow) {
        const int iw0 = (int)ow * (int)stride_w - (int)pad_left;
        float acc = 0.0f;
        for (unsigned int ki = 0; ki < kh; ++ki) {
          const int ih = ih0 + (int)ki * (int)dilation_h;
          if (ih < 0 || ih >= (int)in_h)
            continue;
          for (unsigned int kj = 0; kj < kw; ++kj) {
            const int iw = iw0 + (int)kj * (int)dilation_w;
            if (iw < 0 || iw >= (int)in_w)
              continue;
            acc += inc[(size_t)ih * in_w + iw] * fc[ki * kw + kj];
          }
        }
        outc[(size_t)oh * out_w + ow] = acc;
      }
    }
  });
}

#ifdef ENABLE_FP16
void __fallback_depthwise_conv2d_fp16(
  const _FP16 *input, const float *kernel, _FP16 *output, unsigned int batch,
  unsigned int channels, unsigned int in_h, unsigned int in_w,
  unsigned int out_h, unsigned int out_w, unsigned int kh, unsigned int kw,
  unsigned int stride_h, unsigned int stride_w, unsigned int pad_top,
  unsigned int pad_left, unsigned int dilation_h, unsigned int dilation_w) {
  // FP16 mirror of __fallback_depthwise_conv2d_fp32: same channel-parallel
  // direct loop, but accumulate in float (the kernel/input are widened per MAC)
  // so a long kh*kw reduction does not lose precision in FP16. Output narrowed
  // back to _FP16. Keeps depthwise off the slow per-channel im2col + FP16 GEMV
  // path the grouped conv else-branch would otherwise take.
  auto &tm = ThreadManager::Global();
  tm.parallel_for(0, (size_t)batch * channels, [&](size_t bc) {
    const unsigned int c = (unsigned int)(bc % channels);
    const _FP16 *inc = input + bc * in_h * in_w;
    const float *fc = kernel + (size_t)c * kh * kw;
    _FP16 *outc = output + bc * out_h * out_w;
    for (unsigned int oh = 0; oh < out_h; ++oh) {
      const int ih0 = (int)oh * (int)stride_h - (int)pad_top;
      for (unsigned int ow = 0; ow < out_w; ++ow) {
        const int iw0 = (int)ow * (int)stride_w - (int)pad_left;
        float acc = 0.0f;
        for (unsigned int ki = 0; ki < kh; ++ki) {
          const int ih = ih0 + (int)ki * (int)dilation_h;
          if (ih < 0 || ih >= (int)in_h)
            continue;
          for (unsigned int kj = 0; kj < kw; ++kj) {
            const int iw = iw0 + (int)kj * (int)dilation_w;
            if (iw < 0 || iw >= (int)in_w)
              continue;
            acc += static_cast<float>(inc[(size_t)ih * in_w + iw]) *
                   static_cast<float>(fc[ki * kw + kj]);
          }
        }
        outc[(size_t)oh * out_w + ow] = static_cast<_FP16>(acc);
      }
    }
  });
}
#endif

} // namespace nntrainer
