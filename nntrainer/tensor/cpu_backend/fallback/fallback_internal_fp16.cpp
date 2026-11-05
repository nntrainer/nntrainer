// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file fallback_internal.cpp
 * @date   23 April 2024
 * @see    https://github.com/nntrainer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Fallback half-precision computation functions (raw implementation)
 * This file is differentiated because there are some targets that do not
 * support half-precision dataType fundamentally.
 *
 */

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstdint>
#include <fallback_internal.h>
#include <stdexcept>
#include <tensor_dim.h>
#include <vector>

#define hgemv_loop(ci, cj, cM, cN)                                             \
  do {                                                                         \
    float y0;                                                                  \
    unsigned int i, j;                                                         \
    for (ci = 0; ci != cM; ci++) {                                             \
      y0 = 0.0f;                                                               \
      if (beta != 0.0f) {                                                      \
        y0 = static_cast<float>(Y[ci * incY] * static_cast<_FP16>(beta));      \
      }                                                                        \
      for (cj = 0; cj != cN; cj++)                                             \
        y0 += static_cast<float>(A[i + j * lda] * X[cj * incX]);               \
      Y[ci * incY] = static_cast<_FP16>(y0);                                   \
    }                                                                          \
  } while (0);

#define hgemm_loop()                                                           \
  do {                                                                         \
    for (unsigned int m = 0; m < M; ++m) {                                     \
      for (unsigned int n = 0; n < N; ++n) {                                   \
        float c = 0;                                                           \
        _FP16 c_old = C[m * ldc + n];                                          \
        for (unsigned int k = 0; k < K; ++k) {                                 \
          _FP16 a, b;                                                          \
          a = ((TransA) ? A[k * lda + m] : A[m * lda + k]);                    \
          b = ((TransB) ? B[n * ldb + k] : B[k * ldb + n]);                    \
          c += static_cast<float>(a * b);                                      \
        }                                                                      \
        C[m * ldc + n] = static_cast<_FP16>(alpha * c);                        \
        if (beta != 0.0f) {                                                    \
          C[m * ldc + n] += static_cast<_FP16>(beta) * c_old;                  \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  } while (0);

#define haxpy_loop()                                                           \
  do {                                                                         \
    unsigned int i;                                                            \
    for (i = 0; i < N; ++i)                                                    \
      Y[i * incY] = Y[i * incY] + static_cast<_FP16>(alpha) * X[i * incX];     \
  } while (0);
#define hsgemv_loop(ci, cj, cM, cN)                                            \
  do {                                                                         \
    float y0;                                                                  \
    unsigned int i, j;                                                         \
    for (ci = 0; ci != cM; ci++) {                                             \
      y0 = 0.0f;                                                               \
      if (beta != 0.0f) {                                                      \
        y0 = Y[ci * incY] * beta;                                              \
      }                                                                        \
      for (cj = 0; cj != cN; cj++)                                             \
        y0 += static_cast<float>(A[i + j * lda]) * X[cj * incX];               \
      Y[ci * incY] = y0;                                                       \
    }                                                                          \
  } while (0);

#define hsgemm_loop()                                                          \
  do {                                                                         \
    for (unsigned int m = 0; m < M; ++m) {                                     \
      for (unsigned int n = 0; n < N; ++n) {                                   \
        float c = 0;                                                           \
        float c_old = C[m * ldc + n];                                          \
        for (unsigned int k = 0; k < K; ++k) {                                 \
          _FP16 a;                                                             \
          float b;                                                             \
          a = ((TransA) ? A[k * lda + m] : A[m * lda + k]);                    \
          b = ((TransB) ? B[n * ldb + k] : B[k * ldb + n]);                    \
          c += static_cast<float>(a) * b;                                      \
        }                                                                      \
        C[m * ldc + n] = (alpha * c);                                          \
        if (beta != 0.0f) {                                                    \
          C[m * ldc + n] += (beta)*c_old;                                      \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  } while (0);
#define shgemv_loop(ci, cj, cM, cN)                                            \
  do {                                                                         \
    float y0;                                                                  \
    unsigned int i, j;                                                         \
    for (ci = 0; ci != cM; ci++) {                                             \
      y0 = 0.0f;                                                               \
      if (beta != 0.0f) {                                                      \
        y0 = static_cast<float>(Y[ci * incY] * beta);                          \
      }                                                                        \
      for (cj = 0; cj != cN; cj++)                                             \
        y0 += (A[i + j * lda] * static_cast<float>(X[cj * incX]));             \
      Y[ci * incY] = y0;                                                       \
    }                                                                          \
  } while (0);

#define shgemm_loop()                                                          \
  do {                                                                         \
    for (unsigned int m = 0; m < M; ++m) {                                     \
      for (unsigned int n = 0; n < N; ++n) {                                   \
        float c = 0;                                                           \
        float c_old = C[m * ldc + n];                                          \
        for (unsigned int k = 0; k < K; ++k) {                                 \
          float a;                                                             \
          _FP16 b;                                                             \
          a = ((TransA) ? A[k * lda + m] : A[m * lda + k]);                    \
          b = ((TransB) ? B[n * ldb + k] : B[k * ldb + n]);                    \
          c += static_cast<float>(a * b);                                      \
        }                                                                      \
        C[m * ldc + n] = (alpha * c);                                          \
        if (beta != 0.0f) {                                                    \
          C[m * ldc + n] += (beta)*c_old;                                      \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  } while (0);

namespace nntrainer {
void __fallback_shgemm(const unsigned int TStorageOrder, bool TransA,
                       bool TransB, const unsigned int M, const unsigned int N,
                       const unsigned int K, const float alpha, const float *A,
                       const unsigned int lda, const _FP16 *B,
                       const unsigned int ldb, const float beta, float *C,
                       const unsigned int ldc) {
  shgemm_loop();
}

void __fallback_shgemv(const unsigned int TStorageOrder, bool TransA,
                       const unsigned int M, const unsigned int N,
                       const float alpha, const float *A,
                       const unsigned int lda, const _FP16 *X,
                       const unsigned int incX, const float beta, float *Y,
                       const unsigned int incY) {
  if (TransA == true) {
    shgemv_loop(i, j, N, M);
  } else {
    shgemv_loop(j, i, M, N);
  }
}

void __fallback_hsgemm(const unsigned int TStorageOrder, bool TransA,
                       bool TransB, const unsigned int M, const unsigned int N,
                       const unsigned int K, const float alpha, const _FP16 *A,
                       const unsigned int lda, const float *B,
                       const unsigned int ldb, const float beta, float *C,
                       const unsigned int ldc) {
  hsgemm_loop();
}
void __fallback_hsgemv(const unsigned int TStorageOrder, bool TransA,
                       const unsigned int M, const unsigned int N,
                       const float alpha, const _FP16 *A,
                       const unsigned int lda, const float *X,
                       const unsigned int incX, const float beta, float *Y,
                       const unsigned int incY) {
  if (TransA == true) {
    hsgemv_loop(i, j, N, M);
  } else {
    hsgemv_loop(j, i, M, N);
  }
}

void __fallback_quantize_row_q8_0(const _FP16 *__restrict src,
                                  void *__restrict dst, int64_t k) {
  throw std::runtime_error("NYI : __fallback_quantize_row_q8_0");
}

size_t __fallback_quantize_q8_0(const _FP16 *src, void *dst, int64_t nrow,
                                int64_t n_per_row, const float *quant_weights) {
  throw std::runtime_error("NYI : __fallback_quantize_q8_0");
  return 1;
}

void __fallback_dequantize_row_q8_0(const void *x_raw, _FP16 *y, int64_t k) {
  throw std::runtime_error("NYI : __fallback_dequantize_row_q8_0");
}

void __fallback_compute_rotary_embedding_value(unsigned int dim,
                                               unsigned int half_,
                                               unsigned int w, _FP16 *in,
                                               _FP16 *out, float *cos_,
                                               float *sin_) {
  for (unsigned int k = 0; k < dim; k++) {
    unsigned int span = w + k;
    float transformed_value = 0.F;
    float value = static_cast<float>(in[span]);
    if (k < half_) {
      transformed_value = -1.0 * static_cast<float>(in[half_ + span]);
    } else {
      transformed_value = static_cast<float>(in[span - half_]);
    }
    out[span] =
      static_cast<_FP16>(value * cos_[k] + transformed_value * sin_[k]);
  }
}

void __fallback_compute_rotary_emb_value(unsigned int width, unsigned int dim,
                                         unsigned int half_, _FP16 *inout,
                                         _FP16 *output, const _FP16 *cos_,
                                         const _FP16 *sin_) {
  // Same pairing as the NEON kernel; the products are taken in fp32 and
  // rounded once, where NEON rounds each fp16 product.
  _FP16 *dst = output != nullptr ? output : inout;
  for (unsigned int w = 0; w < width; w += dim) {
    for (unsigned int k = 0; k < half_; ++k) {
      const float a = static_cast<float>(inout[w + k]);
      const float b = static_cast<float>(inout[w + k + half_]);
      const float c = static_cast<float>(cos_[k]);
      const float s = static_cast<float>(sin_[k]);
      dst[w + k] = static_cast<_FP16>(a * c - b * s);
      dst[w + k + half_] = static_cast<_FP16>(a * s + b * c);
    }
  }
}

void __fallback_compute_kcaches(const _FP16 *in, const _FP16 *kcache,
                                _FP16 *output, int num_rows, int num_cache_head,
                                int head_dim, int gqa_size, int tile_size,
                                size_t local_window_size, int head_start,
                                int head_end) {
  // The NEON kernel's arithmetic without its tiling, accumulating in fp32.
  (void)tile_size;
  const int actual_head_end = (head_end < 0) ? num_cache_head : head_end;
  const int start_row = ((size_t)num_rows < local_window_size)
                          ? 0
                          : num_rows - (int)local_window_size;
  const int row_cnt =
    ((size_t)num_rows < local_window_size) ? num_rows : (int)local_window_size;
  const float inv = 1.0f / std::sqrt((float)head_dim);
  for (int n = head_start; n < actual_head_end; ++n) {
    for (int r = 0; r < row_cnt; ++r) {
      const int row = start_row + r;
      const _FP16 *k_row =
        kcache + (size_t)(row * num_cache_head + n) * head_dim;
      for (int g = 0; g < gqa_size; ++g) {
        const _FP16 *in_ptr = in + (size_t)(n * gqa_size + g) * head_dim;
        float sum = 0.0f;
        for (int i = 0; i < head_dim; ++i)
          sum += (float)in_ptr[i] * (float)k_row[i];
        output[(size_t)r * num_cache_head * gqa_size + n * gqa_size + g] =
          (_FP16)(sum * inv);
      }
    }
  }
}

void __fallback_compute_fp16vcache_transposed(int row_num, const _FP16 *in,
                                              const _FP16 *vcache,
                                              _FP16 *output, int num_cache_head,
                                              int gqa_size, int head_dim,
                                              size_t local_window_size,
                                              int head_start, int head_end) {
  const int actual_head_end = (head_end < 0) ? num_cache_head : head_end;
  const int j_start = ((size_t)row_num < local_window_size)
                        ? 0
                        : row_num + 1 - (int)local_window_size;
  std::vector<float> acc((size_t)gqa_size * head_dim);
  for (int n = head_start; n < actual_head_end; ++n) {
    std::fill(acc.begin(), acc.end(), 0.0f);
    for (int j = j_start; j <= row_num; ++j) {
      const _FP16 *vptr = vcache + (size_t)(j * num_cache_head + n) * head_dim;
      const int score_row =
        ((size_t)row_num < local_window_size) ? j : j - j_start;
      for (int h = 0; h < gqa_size; ++h) {
        const float a = (float)
          in[(size_t)score_row * gqa_size * num_cache_head + n * gqa_size + h];
        float *acc_h = acc.data() + (size_t)h * head_dim;
        for (int d = 0; d < head_dim; ++d)
          acc_h[d] += a * (float)vptr[d];
      }
    }
    for (int h = 0; h < gqa_size; ++h) {
      _FP16 *out_h = output + (size_t)(n * gqa_size + h) * head_dim;
      const float *acc_h = acc.data() + (size_t)h * head_dim;
      for (int d = 0; d < head_dim; ++d)
        out_h[d] = (_FP16)acc_h[d];
    }
  }
}

void __fallback_softmax_row_inplace(_FP16 *qk_out, size_t start_row,
                                    size_t end_row, size_t num_heads,
                                    _FP16 *sink) {
  for (size_t c = 0; c < num_heads; ++c) {
    float mx = sink ? (float)sink[c] : -INFINITY;
    for (size_t r = start_row; r < end_row; ++r)
      mx = std::max(mx, (float)qk_out[r * num_heads + c]);
    float sum = sink ? std::exp((float)sink[c] - mx) : 0.0f;
    for (size_t r = start_row; r < end_row; ++r) {
      const float e = std::exp((float)qk_out[r * num_heads + c] - mx);
      qk_out[r * num_heads + c] = (_FP16)e;
      sum += e;
    }
    const float invsum = (sum > 0.0f) ? 1.0f / sum : 0.0f;
    for (size_t r = start_row; r < end_row; ++r)
      qk_out[r * num_heads + c] =
        (_FP16)((float)qk_out[r * num_heads + c] * invsum);
  }
}

void __fallback_swiglu(const unsigned int N, _FP16 *X, _FP16 *Y, _FP16 *Z) {
  unsigned int i = 0;
  while (i < N) {
    X[i] =
      (Y[i] / static_cast<_FP16>(1.f + std::exp(static_cast<float>(-Y[i])))) *
      Z[i];
    ++i;
  }
}

void __fallback_tanh_gelu(const unsigned int N, const _FP16 *X, _FP16 *Y) {
  for (unsigned int i = 0; i < N; ++i) {
    float x = static_cast<float>(X[i]);
    Y[i] = static_cast<_FP16>(
      0.5f * x *
      (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x))));
  }
}

void __fallback_gelu_v2(const unsigned int N, const _FP16 *X, _FP16 *Y) {
  for (unsigned int i = 0; i < N; ++i) {
    float x = static_cast<float>(X[i]);

    Y[i] = static_cast<_FP16>(0.5f * x * (1.0f + std::erf(x * 0.7071067811f)));
  }
}

void __fallback_tanh_gelu_mul(const unsigned int N, _FP16 *X, _FP16 *Y,
                              _FP16 *Z) {
  for (unsigned int i = 0; i < N; ++i) {
    float y = static_cast<float>(Y[i]);
    float z = static_cast<float>(Z[i]);
    X[i] = static_cast<_FP16>(
      0.5f * y *
      (1.0f + std::tanh(0.7978845608f * (y + 0.044715f * y * y * y))) * z);
  }
}

_FP16 __fallback_max(const unsigned int N, _FP16 *X) {
  std::vector<_FP16> v(X, X + N);
  return *std::max_element(v.begin(), v.end());
}

void __fallback_softmax(const unsigned int N, _FP16 *X, _FP16 *Y) {
  _FP16 max_x = __fallback_max(N, X);
  unsigned int i = 0;
  float sum = 0.f;
  while (i < N) {
    sum += std::exp(static_cast<float>(X[i] - max_x));
    ++i;
  }
  i = 0;
  while (i < N) {
    Y[i] = static_cast<_FP16>(std::exp(static_cast<float>(X[i] - max_x)) / sum);
    ++i;
  }
}

void __fallback_sscal(const unsigned int N, const float alpha, _FP16 *X,
                      const unsigned int incX) {
  for (unsigned int i = 0; i < N; ++i)
    X[i * incX] = static_cast<_FP16>(alpha) * X[i * incX];
}

_FP16 __fallback_snrm2(const unsigned int N, const _FP16 *X,
                       const unsigned int incX) {
  float sum = 0;
  float tmp;
  for (unsigned int i = 0; i < N; i++) {
    tmp = static_cast<float>(X[i * incX]);
    sum += tmp * tmp;
  }
  return static_cast<_FP16>(sqrt(sum));
}

void __fallback_scopy(const unsigned int N, const _FP16 *X,
                      const unsigned int incX, _FP16 *Y,
                      const unsigned int incY) {
  for (unsigned int i = 0; i < N; ++i)
    Y[i * incY] = X[i * incX];
}

void __fallback_scopy(const unsigned int N, const float *X,
                      const unsigned int incX, _FP16 *Y,
                      const unsigned int incY) {
  for (unsigned int i = 0; i < N; ++i)
    Y[i * incY] = static_cast<_FP16>(X[i * incX]);
}

void __fallback_scopy(const unsigned int N, const _FP16 *X,
                      const unsigned int incX, float *Y,
                      const unsigned int incY) {
  for (unsigned int i = 0; i < N; ++i)
    Y[i * incY] = static_cast<float>(X[i * incX]);
}

void __fallback_scopy_int4_to_float16(const unsigned int N, const uint8_t *X,
                                      const unsigned int incX, _FP16 *Y,
                                      const unsigned int incY) {
  for (unsigned int idx = 0; idx < N; idx++) {
    Y[2 * idx] = X[idx] >> 4;
    Y[2 * idx + 1] = X[idx] & 0x0f;
  }
}

void __fallback_scopy_int8_to_float16(const unsigned int N, const uint8_t *X,
                                      const unsigned int incX, _FP16 *Y,
                                      const unsigned int incY) {
  for (unsigned int idx = 0; idx < N; idx++) {
    Y[idx] = X[idx];
  }
}

void __fallback_scopy_int8_to_float16(const unsigned int N, const int8_t *X,
                                      const unsigned int incX, _FP16 *Y,
                                      const unsigned int incY) {
  for (unsigned int idx = 0; idx < N; idx++) {
    Y[idx] = X[idx];
  }
}

_FP16 __fallback_sdot(const unsigned int N, const _FP16 *X,
                      const unsigned int incX, const _FP16 *Y,
                      const unsigned int incY) {
  assert(incX > 0 && incY > 0);
  float ret = 0;
  for (unsigned int i = 0; i < N; ++i) {
    ret += static_cast<float>(X[i * incX]) * static_cast<float>(Y[i * incY]);
  }
  return static_cast<_FP16>(ret);
}

void __fallback_saxpy(const unsigned int N, const float alpha, const _FP16 *X,
                      const unsigned int incX, _FP16 *Y,
                      const unsigned int incY) {
  haxpy_loop();
}

void __fallback_sgemm(const unsigned int TStorageOrder, bool TransA,
                      bool TransB, const unsigned int M, const unsigned int N,
                      const unsigned int K, const float alpha, const _FP16 *A,
                      const unsigned int lda, const _FP16 *B,
                      const unsigned int ldb, const float beta, _FP16 *C,
                      const unsigned int ldc) {
  hgemm_loop();
}

void __fallback_sgemv(const unsigned int TStorageOrder, bool TransA,
                      const unsigned int M, const unsigned int N,
                      const float alpha, const _FP16 *A, const unsigned int lda,
                      const _FP16 *X, const unsigned int incX, const float beta,
                      _FP16 *Y, const unsigned int incY) {

  if (TransA == true) {
    hgemv_loop(i, j, N, M);
  } else {
    hgemv_loop(j, i, M, N);
  }
}

void __fallback_ele_mul(const unsigned int N, const _FP16 *X, const _FP16 *Y,
                        _FP16 *Z, float alpha, float beta,
                        unsigned int i_stride, unsigned int o_stride) {
  for (unsigned int i = 0; i < N; ++i) {
    *Z = *X * static_cast<_FP16>(alpha) * *Y +
         ((0.0f == beta) ? static_cast<_FP16>(0.0f)
                         : static_cast<_FP16>(beta) * *Z);
    X += o_stride;
    Y += i_stride;
    Z += o_stride;
  }
}

void __fallback_ele_add(const unsigned int N, const _FP16 *X, const _FP16 *Y,
                        _FP16 *Z, float alpha, float beta,
                        unsigned int i_stride, unsigned int o_stride) {
  for (unsigned int i = 0; i < N; ++i) {
    *Z = *X + static_cast<_FP16>(alpha) * *Y +
         ((0.0f == beta) ? static_cast<_FP16>(0.0f)
                         : static_cast<_FP16>(beta) * *Z);
    X += o_stride;
    Y += i_stride;
    Z += o_stride;
  }
}

void __fallback_ele_sub(const unsigned N, const _FP16 *X, const _FP16 *Y,
                        _FP16 *Z, float alpha, float beta,
                        unsigned int i_stride, unsigned int o_stride) {
  for (unsigned int i = 0; i < N; ++i) {
    *Z = *X - static_cast<_FP16>(alpha) * *Y +
         ((0.0f == beta) ? static_cast<_FP16>(0.0f)
                         : static_cast<_FP16>(beta) * *Z);
    X += o_stride;
    Y += i_stride;
    Z += o_stride;
  }
}

void __fallback_ele_div(const unsigned N, const _FP16 *X, const _FP16 *Y,
                        _FP16 *Z, float alpha, float beta,
                        unsigned int i_stride, unsigned int o_stride) {
  for (unsigned int i = 0; i < N; ++i) {
    *Z = *X / (static_cast<_FP16>(alpha) * *Y) +
         ((0.0f == beta) ? static_cast<_FP16>(0.0f)
                         : static_cast<_FP16>(beta) * *Z);
    X += o_stride;
    Y += i_stride;
    Z += o_stride;
  }
}

unsigned int __fallback_isamax(const unsigned int N, const _FP16 *X,
                               const unsigned int incX) {
  unsigned int max_idx = 0;
  _FP16 max_val = X[0];
  for (unsigned int n = 1; n < N; n += incX) {
    _FP16 cur_val = (X[n] >= 0) ? X[n] : -1 * X[n];
    if (cur_val > max_val) {
      max_val = cur_val;
      max_idx = n;
    }
  }
  return max_idx;
}

void __fallback_inv_sqrt_inplace(const unsigned int N, _FP16 *X) {
  for (unsigned int i = 0; i < N; ++i) {
    X[i] = static_cast<_FP16>(1 / std::sqrt(static_cast<float>(X[i])));
  }
}

void __fallback_transpose_matrix(const unsigned int M, const unsigned int N,
                                 const _FP16 *src, unsigned int ld_src,
                                 _FP16 *dst, unsigned int ld_dst) {
  for (unsigned int i = 0; i < M; i++) {
    for (unsigned int j = 0; j < N; j++) {
      dst[i + j * ld_dst] = src[i * ld_src + j];
    }
  }
}

bool __fallback_isValid(const unsigned int N, const _FP16 *X) {
  for (size_t i = 0; i < N; ++i) {
    /// @todo check for inf
    if (*X != *X)
      return false;
    ++X;
  }

  return true;
}

template <>
void __fallback_gemm_q4_0(const unsigned int M, const unsigned int N,
                          const unsigned int K, const _FP16 *A,
                          const unsigned int lda, const void *B,
                          const unsigned int ldb, _FP16 *C,
                          const unsigned int ldc) {
  throw std::runtime_error("NYI : __fallback_gemm_q4_0");
}

template <>
void __fallback_dequantize_row_q8_K(const void *x, _FP16 *y, int64_t k) {
  throw std::runtime_error("NYI : __fallback_dequantize_row_q8_K");
}

template <>
void __fallback_quantize_row_q8_K(const _FP16 *src, void *dst, int64_t k) {
  throw std::runtime_error("NYI : __fallback_quantize_row_q8_K");
}

template <>
void __fallback_gemm_q6_K(const unsigned int M, const unsigned int N,
                          const unsigned int K, const _FP16 *A,
                          const unsigned int lda, const void *B,
                          const unsigned int ldb, _FP16 *C,
                          const unsigned int ldc) {
  throw std::runtime_error("NYI : __fallback_gemm_q6_K");
}

template <>
void __fallback_rms_norm_wrt_width_fp16_intrinsic(const _FP16 *__restrict X,
                                                  _FP16 *__restrict Y, size_t H,
                                                  size_t W, float epsilon) {

  throw std::runtime_error(
    "NYI : __fallback_rms_norm_wrt_width_fp16_intrinsic with FP16 type input");
}

template <>
void __fallback_clamp(const _FP16 *input, _FP16 *output, size_t length,
                      _FP16 lower_bound, _FP16 upper_bound) {
  for (int i = 0; i < length; ++i) {
    output[i] = std::clamp(input[i], lower_bound, upper_bound);
  }
}

void __fallback_gemm_qs4cx_fp16(size_t M, size_t N, size_t K, const _FP16 *lhs,
                                const uint8_t *rhs_qs4cx,
                                const float *rhs_scales, _FP16 *dst,
                                float beta) {
  const size_t row_bytes = (K + 1) / 2;
  for (size_t n = 0; n < N; ++n) {
    const uint8_t *wrow = rhs_qs4cx + n * row_bytes;
    const float s = rhs_scales[n];
    for (size_t m = 0; m < M; ++m) {
      const _FP16 *xr = lhs + m * K;
      _FP16 *yr = dst + m * N;
      float acc = 0.f;
      for (size_t k = 0; k < K; ++k) {
        const uint8_t byte = wrow[k >> 1];
        const uint8_t u4 = (k & 1) ? (uint8_t)(byte >> 4) : (byte & 0x0F);
        acc += (float)xr[k] * (float)((int32_t)u4 - 8);
      }
      float v = acc * s;
      if (beta != 0.0f)
        v += beta * (float)yr[n];
      v = std::min(std::max(v, -65504.f), 65504.f);
      yr[n] = (_FP16)v;
    }
  }
}

} // namespace nntrainer
