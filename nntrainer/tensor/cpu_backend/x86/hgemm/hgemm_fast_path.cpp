// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Yonghyeon Cho <dyddyd8574@gmail.com>
 *
 * @file   hgemm_fast_path.cpp
 * @date   01 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Yonghyeon Cho <dyddyd8574@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  Small/skinny x86 FP16 GEMM fast paths
 */

#include "hgemm_fast_path.h"

#include "hgemm_util.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <immintrin.h>
#include <thread_manager.h>

namespace nntrainer::hgemm::internal {

namespace {

bool should_parallelize_fast_path(unsigned int M, unsigned int N,
                                  unsigned int K) {
  constexpr std::size_t min_parallel_work = 1u * 1024u * 1024u;
  const std::size_t work = static_cast<std::size_t>(M) * N * K;
  return work >= min_parallel_work;
}

template <typename AType, typename BType, typename CType>
void direct_small_gemm(bool TransA, bool TransB, unsigned int M, unsigned int N,
                       unsigned int K, float alpha, const AType *A,
                       unsigned int a_stride, const BType *B,
                       unsigned int b_stride, float beta, CType *C,
                       unsigned int c_stride) {
  for (unsigned int m = 0; m < M; ++m) {
    for (unsigned int n = 0; n < N; ++n) {
      float sum = 0.0F;
      for (unsigned int k = 0; k < K; ++k) {
        const float a =
          TransA ? read_f32(A + static_cast<std::size_t>(k) * a_stride + m)
                 : read_f32(A + static_cast<std::size_t>(m) * a_stride + k);
        const float b =
          TransB ? read_f32(B + static_cast<std::size_t>(n) * b_stride + k)
                 : read_f32(B + static_cast<std::size_t>(k) * b_stride + n);
        sum += a * b;
      }

      const std::size_t c_idx = static_cast<std::size_t>(m) * c_stride + n;
      float out = alpha * sum;
      if (std::fpclassify(beta) != FP_ZERO) {
        out += beta * static_cast<float>(C[c_idx]);
      }
      C[c_idx] = static_cast<CType>(out);
    }
  }
}

template <typename AType, typename BType, typename CType>
void small_k_notrans_fast_path(bool TransA, unsigned int M, unsigned int N,
                               unsigned int K, float alpha, const AType *A,
                               unsigned int a_stride, const BType *B,
                               unsigned int b_stride, float beta, CType *C,
                               unsigned int c_stride) {
  const bool beta_zero = std::fpclassify(beta) == FP_ZERO;
  const __m256 vbeta = _mm256_set1_ps(beta);

  auto compute_row = [&](unsigned int m) {
    CType *c_row = C + static_cast<std::size_t>(m) * c_stride;
    unsigned int n = 0;
    for (; n + 8 <= N; n += 8) {
      __m256 acc = beta_zero
                     ? _mm256_setzero_ps()
                     : _mm256_mul_ps(load8_to_f32<CType>(c_row + n), vbeta);
      for (unsigned int k = 0; k < K; ++k) {
        const AType *a_ptr = TransA
                               ? A + static_cast<std::size_t>(k) * a_stride + m
                               : A + static_cast<std::size_t>(m) * a_stride + k;
        const float scale = alpha * static_cast<float>(*a_ptr);
        if (std::fpclassify(scale) == FP_ZERO) {
          continue;
        }
        const __m256 vscale = _mm256_set1_ps(scale);
        const BType *b_row = B + static_cast<std::size_t>(k) * b_stride;
        acc = _mm256_fmadd_ps(vscale, load8_to_f32<BType>(b_row + n), acc);
      }
      store8_from_f32<CType>(c_row + n, acc);
    }

    for (; n < N; ++n) {
      float out = beta_zero ? 0.0F : beta * static_cast<float>(c_row[n]);
      for (unsigned int k = 0; k < K; ++k) {
        const AType *a_ptr = TransA
                               ? A + static_cast<std::size_t>(k) * a_stride + m
                               : A + static_cast<std::size_t>(m) * a_stride + k;
        out +=
          alpha * static_cast<float>(*a_ptr) *
          static_cast<float>(B[static_cast<std::size_t>(k) * b_stride + n]);
      }
      c_row[n] = static_cast<CType>(out);
    }
  };

  if (should_parallelize_fast_path(M, N, K)) {
    auto &tm = ThreadManager::Global();
    tm.parallel_for(0, static_cast<std::size_t>(M), [&](std::size_t m) {
      compute_row(static_cast<unsigned int>(m));
    });
    return;
  }

  for (unsigned int m = 0; m < M; ++m) {
    compute_row(m);
  }
}

template <typename AType, typename BType, typename CType>
void skinny_n_fast_path(bool TransA, bool TransB, unsigned int M,
                        unsigned int N, unsigned int K, float alpha,
                        const AType *A, unsigned int a_stride, const BType *B,
                        unsigned int b_stride, float beta, CType *C,
                        unsigned int c_stride) {
  const bool beta_zero = std::fpclassify(beta) == FP_ZERO;

  auto compute_row = [&](unsigned int m) {
    float acc0 = 0.0F;
    float acc1 = 0.0F;
    for (unsigned int k = 0; k < K; ++k) {
      const float a =
        TransA ? read_f32(A + static_cast<std::size_t>(k) * a_stride + m)
               : read_f32(A + static_cast<std::size_t>(m) * a_stride + k);
      if (TransB) {
        acc0 += a * read_f32(B + k);
        if (N == 2) {
          acc1 += a * read_f32(B + b_stride + k);
        }
      } else {
        const BType *b_row = B + static_cast<std::size_t>(k) * b_stride;
        acc0 += a * read_f32(b_row);
        if (N == 2) {
          acc1 += a * read_f32(b_row + 1);
        }
      }
    }

    CType *c_row = C + static_cast<std::size_t>(m) * c_stride;
    float out0 = alpha * acc0;
    if (!beta_zero) {
      out0 += beta * static_cast<float>(c_row[0]);
    }
    c_row[0] = static_cast<CType>(out0);

    if (N == 2) {
      float out1 = alpha * acc1;
      if (!beta_zero) {
        out1 += beta * static_cast<float>(c_row[1]);
      }
      c_row[1] = static_cast<CType>(out1);
    }
  };

  if (should_parallelize_fast_path(M, N, K)) {
    auto &tm = ThreadManager::Global();
    tm.parallel_for(0, static_cast<std::size_t>(M), [&](std::size_t m) {
      compute_row(static_cast<unsigned int>(m));
    });
    return;
  }

  for (unsigned int m = 0; m < M; ++m) {
    compute_row(m);
  }
}

template <typename AType, typename BType, typename CType>
void row_gemm_fast_path(bool TransA, unsigned int N, unsigned int K,
                        float alpha, const AType *A, unsigned int a_stride,
                        const BType *B, unsigned int b_stride, float beta,
                        CType *C, float *scratch) {
  const bool beta_zero = std::fpclassify(beta) == FP_ZERO;

  auto compute_columns = [&](unsigned int n_begin, unsigned int n_end) {
    if (beta_zero) {
      std::memset(scratch + n_begin, 0,
                  static_cast<std::size_t>(n_end - n_begin) * sizeof(float));
    } else {
      const __m256 vbeta = _mm256_set1_ps(beta);
      unsigned int n = n_begin;
      for (; n + 8 <= n_end; n += 8) {
        __m256 c32 = load8_to_f32<CType>(C + n);
        _mm256_storeu_ps(scratch + n, _mm256_mul_ps(c32, vbeta));
      }
      for (; n < n_end; ++n) {
        scratch[n] = beta * static_cast<float>(C[n]);
      }
    }

    for (unsigned int k = 0; k < K; ++k) {
      const AType *a_ptr =
        TransA ? A + static_cast<std::size_t>(k) * a_stride : A + k;
      const float scale = alpha * static_cast<float>(*a_ptr);
      if (std::fpclassify(scale) == FP_ZERO) {
        continue;
      }

      const __m256 vscale = _mm256_set1_ps(scale);
      const BType *b_row = B + static_cast<std::size_t>(k) * b_stride;
      unsigned int n = n_begin;
      for (; n + 8 <= n_end; n += 8) {
        __m256 b32 = load8_to_f32<BType>(b_row + n);
        __m256 acc = _mm256_loadu_ps(scratch + n);
        acc = _mm256_fmadd_ps(vscale, b32, acc);
        _mm256_storeu_ps(scratch + n, acc);
      }
      for (; n < n_end; ++n) {
        scratch[n] += scale * static_cast<float>(b_row[n]);
      }
    }

    unsigned int n = n_begin;
    for (; n + 8 <= n_end; n += 8) {
      store8_from_f32<CType>(C + n, _mm256_loadu_ps(scratch + n));
    }
    for (; n < n_end; ++n) {
      C[n] = static_cast<CType>(scratch[n]);
    }
  };

  if (should_parallelize_fast_path(1, N, K)) {
    constexpr unsigned int chunk_size = 256;
    const unsigned int chunks = (N + chunk_size - 1) / chunk_size;
    auto &tm = ThreadManager::Global();
    tm.parallel_for(0, static_cast<std::size_t>(chunks),
                    [&](std::size_t chunk_idx) {
                      const unsigned int n_begin =
                        static_cast<unsigned int>(chunk_idx) * chunk_size;
                      const unsigned int n_end =
                        std::min<unsigned int>(N, n_begin + chunk_size);
                      compute_columns(n_begin, n_end);
                    });
    return;
  }

  compute_columns(0, N);
}

template <typename AType, typename BType, typename CType>
void row_gemm_transB_fast_path(bool TransA, unsigned int N, unsigned int K,
                               float alpha, const AType *A,
                               unsigned int a_stride, const BType *B,
                               unsigned int b_stride, float beta, CType *C,
                               float *scratch) {
  for (unsigned int k = 0; k < K; ++k) {
    const AType *a_ptr =
      TransA ? A + static_cast<std::size_t>(k) * a_stride : A + k;
    scratch[k] = alpha * static_cast<float>(*a_ptr);
  }

  const bool beta_zero = std::fpclassify(beta) == FP_ZERO;

  auto compute_columns = [&](unsigned int n_begin, unsigned int n_end) {
    for (unsigned int n = n_begin; n < n_end; ++n) {
      const BType *b_row = B + static_cast<std::size_t>(n) * b_stride;
      __m256 vacc = _mm256_setzero_ps();
      unsigned int k = 0;
      for (; k + 8 <= K; k += 8) {
        vacc = _mm256_fmadd_ps(_mm256_loadu_ps(scratch + k),
                               load8_to_f32<BType>(b_row + k), vacc);
      }

      float out = hsum8_f32(vacc);
      for (; k < K; ++k) {
        out += scratch[k] * static_cast<float>(b_row[k]);
      }
      if (!beta_zero) {
        out += beta * static_cast<float>(C[n]);
      }
      C[n] = static_cast<CType>(out);
    }
  };

  if (should_parallelize_fast_path(1, N, K)) {
    constexpr unsigned int chunk_size = 32;
    const unsigned int chunks = (N + chunk_size - 1) / chunk_size;
    auto &tm = ThreadManager::Global();
    tm.parallel_for(0, static_cast<std::size_t>(chunks),
                    [&](std::size_t chunk_idx) {
                      const unsigned int n_begin =
                        static_cast<unsigned int>(chunk_idx) * chunk_size;
                      const unsigned int n_end =
                        std::min<unsigned int>(N, n_begin + chunk_size);
                      compute_columns(n_begin, n_end);
                    });
    return;
  }

  compute_columns(0, N);
}

bool should_use_direct_small_path(bool TransB, unsigned int M, unsigned int N,
                                  unsigned int K) {
  const std::size_t mn = static_cast<std::size_t>(M) * N;
  const bool small_k_without_fast_path = K <= 4 && (TransB || N < 8);
  return small_k_without_fast_path || mn <= 512 || (M <= 2 && N <= 512);
}

} // namespace

template <typename AType, typename BType, typename CType>
bool try_hgemm_fast_path(bool TransA, bool TransB, unsigned int M,
                         unsigned int N, unsigned int K, float alpha,
                         const AType *A, unsigned int a_stride, const BType *B,
                         unsigned int b_stride, float beta, CType *C,
                         unsigned int c_stride, HgemmWorkspace &workspace) {
  if (K <= 4 && !TransB && N >= 8) {
    small_k_notrans_fast_path<AType, BType, CType>(
      TransA, M, N, K, alpha, A, a_stride, B, b_stride, beta, C, c_stride);
    return true;
  }

  if (M == 1 && TransB) {
    float *scratch = workspace.ensure_scratch(K);
    row_gemm_transB_fast_path<AType, BType, CType>(
      TransA, N, K, alpha, A, a_stride, B, b_stride, beta, C, scratch);
    return true;
  }

  if (N <= 2) {
    skinny_n_fast_path<AType, BType, CType>(TransA, TransB, M, N, K, alpha, A,
                                            a_stride, B, b_stride, beta, C,
                                            c_stride);
    return true;
  }

  if (M == 1 && !TransB) {
    float *scratch = workspace.ensure_scratch(N);
    row_gemm_fast_path<AType, BType, CType>(TransA, N, K, alpha, A, a_stride, B,
                                            b_stride, beta, C, scratch);
    return true;
  }

  if (should_use_direct_small_path(TransB, M, N, K)) {
    direct_small_gemm<AType, BType, CType>(TransA, TransB, M, N, K, alpha, A,
                                           a_stride, B, b_stride, beta, C,
                                           c_stride);
    return true;
  }

  return false;
}

template bool try_hgemm_fast_path<_FP16, _FP16, _FP16>(
  bool, bool, unsigned int, unsigned int, unsigned int, float, const _FP16 *,
  unsigned int, const _FP16 *, unsigned int, float, _FP16 *, unsigned int,
  HgemmWorkspace &);
template bool try_hgemm_fast_path<float, _FP16, float>(
  bool, bool, unsigned int, unsigned int, unsigned int, float, const float *,
  unsigned int, const _FP16 *, unsigned int, float, float *, unsigned int,
  HgemmWorkspace &);
template bool try_hgemm_fast_path<_FP16, float, float>(
  bool, bool, unsigned int, unsigned int, unsigned int, float, const _FP16 *,
  unsigned int, const float *, unsigned int, float, float *, unsigned int,
  HgemmWorkspace &);

} /* namespace nntrainer::hgemm::internal */
