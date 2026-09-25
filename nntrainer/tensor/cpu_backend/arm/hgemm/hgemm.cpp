// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   hgemm.cpp
 * @date   03 April 2024
 * @see    https://github.com/nntrainer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @author Debadri Samaddar <s.debadri@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is half-precision GEMM interface
 *
 */

#include <arm_neon.h>
#include <cmath>
#include <hgemm.h>
#include <hgemm_common.h>
#include <hgemm_noTrans.h>
#include <hgemm_padding.h>
#include <hgemm_transA.h>
#include <hgemm_transAB.h>
#include <hgemm_transB.h>
#include <hgemm_util.h>
#include <limits>

void hgemm(const __fp16 *A, const __fp16 *B, __fp16 *C, unsigned int M,
           unsigned int N, unsigned int K, float alpha, float beta, bool TransA,
           bool TransB) {
  if (K == 1) {
    return hgemm_K1(A, B, C, M, N, K, alpha, beta, TransA, TransB);
  } else if (M < 8 && K < 16 && N < 16) {
    return hgemm_small(A, B, C, M, N, K, alpha, beta, TransA, TransB);
  }

  const unsigned int M8_high = get_next_mltpl_of_n(M, 8);
  const unsigned int K8_high = get_next_mltpl_of_n(K, 8);
  const unsigned int N16_high = get_next_mltpl_of_n(N, 16);
  const unsigned int N8_low = get_prev_mltpl_of_2p_n(N, 3);

  float32x4_t ZEROS = vmovq_n_f32(0.F);

  float *C32 = (float *)malloc(M8_high * N16_high * sizeof(float));

  unsigned int size = M8_high * N16_high;
  unsigned int size8 = get_prev_mltpl_of_2p_n(size, 3);
  unsigned int size4 = get_prev_mltpl_of_2p_n(size, 2);

  if (std::fpclassify(beta) != FP_ZERO) {
    for (unsigned int m = 0; m < M; ++m) {
      for (unsigned int n = 0; n < N8_low; n += 8) {
        float16x8_t c =
          vmulq_n_f16(vld1q_f16(&C[m * N + n]), static_cast<__fp16>(beta));
        vst1q_f32(&C32[m * N16_high + n], vcvt_f32_f16(vget_low_f16(c)));
        vst1q_f32(&C32[m * N16_high + n + 4], vcvt_f32_f16(vget_high_f16(c)));
      }
      for (unsigned int n = N8_low; n < N; ++n) {
        C32[m * N16_high + n] = beta * C[m * N + n];
      }
      for (unsigned int n = N; n < N16_high; ++n) {
        C32[m * N16_high + n] = 0.F;
      }
    }
    for (unsigned m = M; m < M8_high; ++m) {
      for (unsigned int n = 0; n < N16_high; n += 4) {
        vst1q_f32(&C32[m * N16_high + n], ZEROS);
      }
    }
  } else {
    for (unsigned int idx = 0; idx < size4; idx += 4) {
      vst1q_f32(&C32[idx], ZEROS);
    }
    for (unsigned int idx = size4; idx < size; idx++) {
      C32[idx] = 0.F;
    }
  }

  hgemm_ensure_divisibility(A, B, C32, M, N, K, alpha, beta, TransA, TransB);

  for (unsigned int m = 0; m < M; ++m) {
    for (unsigned int n = 0; n < N8_low; n += 8) {
      float32x4_t x1 = vld1q_f32(&C32[m * N16_high + n]);
      float32x4_t x2 = vld1q_f32(&C32[m * N16_high + n + 4]);
      vst1q_f16(&C[m * N + n],
                vcombine_f16(vcvt_f16_f32(x1), vcvt_f16_f32(x2)));
    }
    for (unsigned int n = N8_low; n < N; ++n) {
      C[m * N + n] = C32[m * N16_high + n];
    }
  }

  free(C32);
}

void hgemm_small(const __fp16 *A, const __fp16 *B, __fp16 *C, unsigned int M,
                 unsigned int N, unsigned int K, float alpha, float beta,
                 bool TransA, bool TransB) {
  float *C32 = (float *)malloc(M * N * sizeof(float));

  copy_C_to_C32(C, C32, M, N, beta);

  hgemm_classify(A, B, C32, M, N, K, alpha, beta, TransA, TransB);

  copy_C32_to_C(C32, C, M, N, beta);

  free(C32);
}

void hgemm_ensure_divisibility(const __fp16 *A, const __fp16 *B, float *C32,
                               unsigned int M, unsigned int N, unsigned int K,
                               float alpha, float beta, bool TransA,
                               bool TransB) {
  /// @note Padding standard : 8x16 is the only KERNEL that outperforms single
  /// precision GEMM 'so far'. Padding will forcibly make every GEMM cases to
  /// use it. Note that padding is not an optimal way here, but just an option
  /// that is easier to implement. Fine-grained packing, blocking, and
  /// corresponding kernels should be supported in the future for optimal
  /// performance in terms of both latency and memory.

  __fp16 *A_ = (__fp16 *)A, *B_ = (__fp16 *)B;
  unsigned int M_ = M, N_ = N, K_ = K;
  bool pad_A = false, pad_B = false;

  __fp16 *Ap;
  __fp16 *Bp;

  const unsigned int M8_high = ((M - 1) / 8 + 1) * 8;
  const unsigned int K8_high = ((K - 1) / 16 + 1) * 16;
  const unsigned int N16_high = ((N - 1) / 16 + 1) * 16;

  if ((M8_high != M) || (K8_high != K)) {
    pad_A = true;
    Ap = alignedMalloc(M8_high * K8_high);
    hgemm_padding_A(A, Ap, M, K, M8_high, K8_high, TransA);
    A_ = Ap;
    M_ = M8_high;
    K_ = K8_high;
  }
  if ((K8_high != K) || (N16_high != N)) {
    pad_B = true;
    Bp = alignedMalloc(K8_high * N16_high);
    hgemm_padding_B(B, Bp, K, N, K8_high, N16_high, TransB);
    B_ = Bp;
    K_ = K8_high;
    N_ = N16_high;
  }

  hgemm_classify(A_, B_, C32, M_, N_, K_, alpha, beta, TransA, TransB);

  if (pad_A)
    free(Ap);
  if (pad_B)
    free(Bp);
}

void hgemm_classify(const __fp16 *A, const __fp16 *B, float *C32,
                    unsigned int M, unsigned int N, unsigned int K, float alpha,
                    float beta, bool TransA, bool TransB) {
  if (!TransA && !TransB) {
    hgemm_noTrans(A, B, C32, M, N, K, alpha, beta);
  } else if (TransA && !TransB) {
    hgemm_transA(A, B, C32, M, N, K, alpha, beta);
  } else if (!TransA && TransB) {
    hgemm_transB(A, B, C32, M, N, K, alpha, beta);
  } else { // TransA && TransB
    hgemm_transAB(A, B, C32, M, N, K, alpha, beta);
  }
}

void hgemm_K1(const __fp16 *A, const __fp16 *B, __fp16 *C, unsigned int M,
              unsigned int N, unsigned int K, float alpha, float beta,
              bool TransA, bool TransB) {
  unsigned int lda = (TransA) ? M : K;
  unsigned int ldb = (TransB) ? K : N;
  unsigned int ldc = N;

  const float eps = std::numeric_limits<float>::epsilon();
  float16x8_t a_vec;
  unsigned int N8 = (N >> 3) << 3;
  for (unsigned int m = 0; m < M; ++m) {
    a_vec = vmovq_n_f16(alpha * A[m]);
    if (std::fpclassify(beta) != FP_ZERO) {
      for (unsigned int n = 0; n < N8; n += 8) {
        vst1q_f16(&C[m * ldc + n],
                  vaddq_f16(vmulq_f16(a_vec, vld1q_f16(&B[n])),
                            vmulq_n_f16(vld1q_f16(&C[m * ldc + n]), beta)));
      }
    } else {
      for (unsigned int n = 0; n < N8; n += 8) {
        vst1q_f16(&C[m * ldc + n], vmulq_f16(a_vec, vld1q_f16(&B[n])));
      }
    }
    for (unsigned int n = N8; n < N; ++n) {
      C[m * ldc + n] = alpha * A[m] * B[n] + beta * C[m * ldc + n];
    }
  }
}

// QK micro-kernel for the FP16-query + FP32-score attention path.
// Computes S[m, n] = alpha * sum_k A[m, k] * B[n, k]   (TransB-style dot)
// using ARMv8.2-A FMLAL (vfmlalq_low/high_f16): each pair of intrinsics
// widens 8 FP16 products into two FP32 accumulators, so the per-element
// product is computed in FP32 from the start. This avoids the FP16-product
// overflow that an FP16-accumulating kernel hits when packing wide encoder
// logits (Q,K magnitudes ~400 -> products ~160k > FP16 max 65504), and unlike
// a cast-up-Q+shgemm path it never materialises an FP32 copy of Q.
//
// Layout: A row-major (M rows, lda cols), B row-major (N rows, ldb cols),
// C row-major (M rows, ldc cols). Inner length K is the dot length and
// must match both A's and B's stride argument (i.e. lda == ldb == K when
// the rows are contiguous).
//
// Requires FEAT_FHM (asimdfhm in /proc/cpuinfo). The "+feature" target
// attribute form adds these ISA extensions on top of whatever -march the TU
// is built with, rather than pinning the function to "arch=armv8.2-a" (which
// would silently downgrade codegen -- and break builds targeting a stricter
// baseline like armv9.2-a).
// Single-output FP16xFP16->FP32 dot, used for the M/N tails of the blocked
// kernel below. The 4x2 block reproduces this exact accumulation order per
// output, so blocked and tail results are bit-identical (no token drift).
__attribute__((target("+fp16+fp16fml+dotprod+i8mm"))) static inline float
fmlal_dot_one(const __fp16 *a_row, const __fp16 *b_row, unsigned int K) {
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);
  unsigned int k = 0;
  for (; k + 16 <= K; k += 16) {
    float16x8_t a0 = vld1q_f16(a_row + k);
    float16x8_t b0 = vld1q_f16(b_row + k);
    float16x8_t a1 = vld1q_f16(a_row + k + 8);
    float16x8_t b1 = vld1q_f16(b_row + k + 8);
    acc0 = vfmlalq_low_f16(acc0, a0, b0);
    acc1 = vfmlalq_high_f16(acc1, a0, b0);
    acc0 = vfmlalq_low_f16(acc0, a1, b1);
    acc1 = vfmlalq_high_f16(acc1, a1, b1);
  }
  if (k + 8 <= K) {
    float16x8_t a0 = vld1q_f16(a_row + k);
    float16x8_t b0 = vld1q_f16(b_row + k);
    acc0 = vfmlalq_low_f16(acc0, a0, b0);
    acc1 = vfmlalq_high_f16(acc1, a0, b0);
    k += 8;
  }
  float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
  for (; k < K; ++k)
    sum += (float)a_row[k] * (float)b_row[k];
  return sum;
}

// QK GEMM: C[m,n] = alpha * sum_k A[m,k]*B[n,k] (FP16 in, FP32 out).
// 4x2 (M x N) register-blocked: each k-step loads 4 A-rows and 2 B-rows once
// and reuses them across the 8 outputs of the tile, cutting the naive kernel's
// per-(m,n) reloads of B by 4x and of A by 2x. Per-output accumulation order
// is identical to fmlal_dot_one(), so output is bit-identical to the previous
// naive triple-loop. M/N remainders fall back to fmlal_dot_one().
__attribute__((target("+fp16+fp16fml+dotprod+i8mm"))) void
hgemm_f16xf16_f32_fmlal(const __fp16 *A, const __fp16 *B, float *C,
                        unsigned int M, unsigned int N, unsigned int K,
                        float alpha, unsigned int lda, unsigned int ldb,
                        unsigned int ldc) {
  unsigned int m = 0;
  for (; m + 4 <= M; m += 4) {
    const __fp16 *ar[4] = {A + (size_t)(m + 0) * lda, A + (size_t)(m + 1) * lda,
                           A + (size_t)(m + 2) * lda,
                           A + (size_t)(m + 3) * lda};
    float *cr[4] = {C + (size_t)(m + 0) * ldc, C + (size_t)(m + 1) * ldc,
                    C + (size_t)(m + 2) * ldc, C + (size_t)(m + 3) * ldc};
    unsigned int n = 0;
    for (; n + 2 <= N; n += 2) {
      const __fp16 *br[2] = {B + (size_t)(n + 0) * ldb,
                             B + (size_t)(n + 1) * ldb};
      float32x4_t lo[4][2], hi[4][2];
      for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 2; ++j) {
          lo[i][j] = vdupq_n_f32(0.0f);
          hi[i][j] = vdupq_n_f32(0.0f);
        }
      unsigned int k = 0;
      for (; k + 16 <= K; k += 16) {
        float16x8_t Alo[4], Ahi[4], Blo[2], Bhi[2];
        for (int i = 0; i < 4; ++i) {
          Alo[i] = vld1q_f16(ar[i] + k);
          Ahi[i] = vld1q_f16(ar[i] + k + 8);
        }
        for (int j = 0; j < 2; ++j) {
          Blo[j] = vld1q_f16(br[j] + k);
          Bhi[j] = vld1q_f16(br[j] + k + 8);
        }
        for (int i = 0; i < 4; ++i)
          for (int j = 0; j < 2; ++j) {
            lo[i][j] = vfmlalq_low_f16(lo[i][j], Alo[i], Blo[j]);
            hi[i][j] = vfmlalq_high_f16(hi[i][j], Alo[i], Blo[j]);
            lo[i][j] = vfmlalq_low_f16(lo[i][j], Ahi[i], Bhi[j]);
            hi[i][j] = vfmlalq_high_f16(hi[i][j], Ahi[i], Bhi[j]);
          }
      }
      if (k + 8 <= K) {
        float16x8_t Alo[4], Blo[2];
        for (int i = 0; i < 4; ++i)
          Alo[i] = vld1q_f16(ar[i] + k);
        for (int j = 0; j < 2; ++j)
          Blo[j] = vld1q_f16(br[j] + k);
        for (int i = 0; i < 4; ++i)
          for (int j = 0; j < 2; ++j) {
            lo[i][j] = vfmlalq_low_f16(lo[i][j], Alo[i], Blo[j]);
            hi[i][j] = vfmlalq_high_f16(hi[i][j], Alo[i], Blo[j]);
          }
        k += 8;
      }
      for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 2; ++j) {
          float sum = vaddvq_f32(vaddq_f32(lo[i][j], hi[i][j]));
          for (unsigned int kk = k; kk < K; ++kk)
            sum += (float)ar[i][kk] * (float)br[j][kk];
          cr[i][n + j] = alpha * sum;
        }
    }
    // N remainder (n < N): 4 outputs per column via the scalar-order helper.
    for (; n < N; ++n) {
      const __fp16 *b_row = B + (size_t)n * ldb;
      for (int i = 0; i < 4; ++i)
        cr[i][n] = alpha * fmlal_dot_one(ar[i], b_row, K);
    }
  }
  // M remainder (m < M): naive per (m, n).
  for (; m < M; ++m) {
    const __fp16 *a_row = A + (size_t)m * lda;
    float *c_row = C + (size_t)m * ldc;
    for (unsigned int n = 0; n < N; ++n)
      c_row[n] = alpha * fmlal_dot_one(a_row, B + (size_t)n * ldb, K);
  }
}

#include <sys/auxv.h>
#ifndef HWCAP_ASIMDFHM
#define HWCAP_ASIMDFHM (1 << 23)
#endif

bool hgemm_fp16fml_supported() {
  static const bool supported = (getauxval(AT_HWCAP) & HWCAP_ASIMDFHM) != 0;
  return supported;
}
