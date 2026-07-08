// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   ggml_impl_fp16.cpp
 * @date   23 July 2025
 * @see    https://github.com/nntrainer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  GGML kernels for FP16 activation flow
 */

#include <engine.h>
#include <ggml_interface.h>
#include <nntr_ggml_impl.h>
#include <nntr_ggml_impl_common.h>

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <math.h>
#include <stdint.h>
#include <thread_manager.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace nntrainer {

static inline void __copy_f16_from_f32(const float *src, _FP16 *dst,
                                       int64_t k) {
#if defined(__ARM_NEON)
  for (int i = 0; i < k; i += 8) {
    vst1q_f16((dst), vcombine_f16(vcvt_f16_f32(vld1q_f32((float *)(src))),
                                  vcvt_f16_f32(vld1q_f32((float *)(src + 4)))));
    src += 8;
    dst += 8;
  }
#else
  for (unsigned int i = 0; i < k; i++) {
    dst[i] = static_cast<_FP16>(src[i]);
  }
#endif
}

void __ggml_quantize_row_q8_0(const _FP16 *__restrict x, void *vy, int64_t k) {
  assert(QK8_0 == 32);
  assert(k % QK8_0 == 0);
  const int nb = k / QK8_0;

  block_q8_0 *__restrict y = (block_q8_0 *__restrict)vy;

#if defined(__ARM_NEON)
  for (int i = 0; i < nb; i++) {
    float16x8_t srcv[4];  // loaded source
    float16x8_t asrcv[4]; // absolute value of source
    float16x8_t amaxv[2]; // absolute max buffer

    for (int j = 0; j < 4; j++) {
      srcv[j] = vld1q_f16(x + i * 32 + 8 * j);
    }
    for (int j = 0; j < 4; j++) {
      asrcv[j] = vabsq_f16(srcv[j]);
    }

    for (int j = 0; j < 2; j++) {
      amaxv[j] =
        vmaxq_f16(asrcv[2 * j], asrcv[2 * j + 1]); // 0, 1 <- 0, 1 VS 2, 3
    }
    amaxv[0] = vmaxq_f16(amaxv[0], amaxv[1]); // 0 <- 0, 1

    const float amax = static_cast<float>(vmaxvq_f16(amaxv[0]));

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f / d : 0.0f;

    y[i].d = nntr_compute_fp32_to_fp16(d);

    for (int j = 0; j < 4; j++) {
      const float16x8_t v = vmulq_n_f16(srcv[j], id);
      const int16x8_t vi = vcvtnq_s16_f16(v);

      y[i].qs[8 * j + 0] = vgetq_lane_s16(vi, 0);
      y[i].qs[8 * j + 1] = vgetq_lane_s16(vi, 1);
      y[i].qs[8 * j + 2] = vgetq_lane_s16(vi, 2);
      y[i].qs[8 * j + 3] = vgetq_lane_s16(vi, 3);
      y[i].qs[8 * j + 4] = vgetq_lane_s16(vi, 4);
      y[i].qs[8 * j + 5] = vgetq_lane_s16(vi, 5);
      y[i].qs[8 * j + 6] = vgetq_lane_s16(vi, 6);
      y[i].qs[8 * j + 7] = vgetq_lane_s16(vi, 7);
    }
  }
#else
  for (int i = 0; i < nb; i++) {
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK8_0; j++) {
      const float v = x[i * QK8_0 + j];
      amax = std::max(amax, fabsf(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f / d : 0.0f;

    y[i].d = nntr_compute_fp32_to_fp16(d);

    for (int j = 0; j < QK8_0; ++j) {
      const float x0 = x[i * QK8_0 + j] * id;

      y[i].qs[j] = std::roundf(x0);
    }
  }
#endif
}

size_t __ggml_quantize_q8_0(const _FP16 *src, void *dst, int64_t nrow,
                            int64_t n_per_row, const float *quant_weights) {
  const size_t row_size = ggml_row_size(GGML_TYPE_Q8_0, n_per_row);
  __ggml_quantize_row_q8_0(src, dst, (int64_t)nrow * n_per_row);
  return nrow * row_size;
}

void __ggml_dequantize_row_q8_0(const void *_x, _FP16 *__restrict y,
                                int64_t k) {
  static const int qk = QK8_0;
  const block_q8_0 *__restrict x = (const block_q8_0 *__restrict)_x;

  assert(k % qk == 0);

  const int nb = k / qk;

  for (int i = 0; i < nb; i++) {
    // const _FP16 d = (x[i].d); ///@todo check if this works
    const float d = nntr_compute_fp16_to_fp32(x[i].d);

    for (int j = 0; j < qk; ++j) {
      y[i * qk + j] = static_cast<_FP16>(x[i].qs[j] * d);
    }
  }
}

static void __ggml_quantize_mat_q8_0_4x8(const _FP16 *__restrict x,
                                         void *__restrict vy, int64_t k) {
  assert(QK8_0 == 32);
  assert(k % QK8_0 == 0);
  const int nb = k / QK8_0;

  block_q8_0x4 *__restrict y = (block_q8_0x4 *)vy;

#if defined(__ARM_NEON)
  float16x8_t srcv[4][4];
  float id[4];

  for (int i = 0; i < nb; i++) {
    float16x8_t asrcv[4];
    float16x8_t amaxv[2];

    for (int row_iter = 0; row_iter < 4; row_iter++) {
      for (int j = 0; j < 4; j++)
        srcv[row_iter][j] = vld1q_f16(x + row_iter * k + i * 32 + 8 * j);
      for (int j = 0; j < 4; j++)
        asrcv[j] = vabsq_f16(srcv[row_iter][j]);

      for (int j = 0; j < 2; j++) {
        amaxv[j] =
          vmaxq_f16(asrcv[2 * j], asrcv[2 * j + 1]); // 0, 1 <- 0, 1 VS 2, 3
      }
      amaxv[0] = vmaxq_f16(amaxv[0], amaxv[1]); // 0 <- 0, 1

      const float amax = vmaxvq_f16(amaxv[0]);

      const float d = amax / ((1 << 7) - 1);
      id[row_iter] = d ? 1.0f / d : 0.0f;

      y[i].d[row_iter] = nntr_compute_fp32_to_fp16(d);
    }

    for (int j = 0; j < 4; j++) {
      float16x8_t v = vmulq_n_f16(srcv[0][j], id[0]);
      int16x8_t vi = vcvtnq_s16_f16(v);
      y[i].qs[32 * j + 0] = vgetq_lane_s16(vi, 0);
      y[i].qs[32 * j + 1] = vgetq_lane_s16(vi, 1);
      y[i].qs[32 * j + 2] = vgetq_lane_s16(vi, 2);
      y[i].qs[32 * j + 3] = vgetq_lane_s16(vi, 3);
      y[i].qs[32 * j + 4] = vgetq_lane_s16(vi, 4);
      y[i].qs[32 * j + 5] = vgetq_lane_s16(vi, 5);
      y[i].qs[32 * j + 6] = vgetq_lane_s16(vi, 6);
      y[i].qs[32 * j + 7] = vgetq_lane_s16(vi, 7);

      v = vmulq_n_f16(srcv[1][j], id[1]);
      vi = vcvtnq_s16_f16(v);
      y[i].qs[32 * j + 8] = vgetq_lane_s16(vi, 0);
      y[i].qs[32 * j + 9] = vgetq_lane_s16(vi, 1);
      y[i].qs[32 * j + 10] = vgetq_lane_s16(vi, 2);
      y[i].qs[32 * j + 11] = vgetq_lane_s16(vi, 3);
      y[i].qs[32 * j + 12] = vgetq_lane_s16(vi, 4);
      y[i].qs[32 * j + 13] = vgetq_lane_s16(vi, 5);
      y[i].qs[32 * j + 14] = vgetq_lane_s16(vi, 6);
      y[i].qs[32 * j + 15] = vgetq_lane_s16(vi, 7);

      v = vmulq_n_f16(srcv[2][j], id[2]);
      vi = vcvtnq_s16_f16(v);
      y[i].qs[32 * j + 16] = vgetq_lane_s16(vi, 0);
      y[i].qs[32 * j + 17] = vgetq_lane_s16(vi, 1);
      y[i].qs[32 * j + 18] = vgetq_lane_s16(vi, 2);
      y[i].qs[32 * j + 19] = vgetq_lane_s16(vi, 3);
      y[i].qs[32 * j + 20] = vgetq_lane_s16(vi, 4);
      y[i].qs[32 * j + 21] = vgetq_lane_s16(vi, 5);
      y[i].qs[32 * j + 22] = vgetq_lane_s16(vi, 6);
      y[i].qs[32 * j + 23] = vgetq_lane_s16(vi, 7);

      v = vmulq_n_f16(srcv[3][j], id[3]);
      vi = vcvtnq_s16_f16(v);
      y[i].qs[32 * j + 24] = vgetq_lane_s16(vi, 0);
      y[i].qs[32 * j + 25] = vgetq_lane_s16(vi, 1);
      y[i].qs[32 * j + 26] = vgetq_lane_s16(vi, 2);
      y[i].qs[32 * j + 27] = vgetq_lane_s16(vi, 3);
      y[i].qs[32 * j + 28] = vgetq_lane_s16(vi, 4);
      y[i].qs[32 * j + 29] = vgetq_lane_s16(vi, 5);
      y[i].qs[32 * j + 30] = vgetq_lane_s16(vi, 6);
      y[i].qs[32 * j + 31] = vgetq_lane_s16(vi, 7);
    }
  }
#else
  // scalar
  const int blck_size_interleave = 8;
  _FP16 srcv[4][QK8_0];
  float id[4];

  for (int i = 0; i < nb; i++) {
    for (int row_iter = 0; row_iter < 4; row_iter++) {
      float amax = 0.0f; // absolute max

      for (int j = 0; j < QK8_0; j++) {
        srcv[row_iter][j] = x[row_iter * k + i * QK8_0 + j];
        amax = MAX(amax, fabsf(srcv[row_iter][j]));
      }

      const float d = amax / ((1 << 7) - 1);
      id[row_iter] = d ? 1.0f / d : 0.0f;

      y[i].d[row_iter] = nntr_compute_fp32_to_fp16(d);
    }

    for (int j = 0; j < QK8_0 * 4; j++) {
      int src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
      int src_id = (j % (4 * blck_size_interleave)) / blck_size_interleave;
      src_offset += (j % blck_size_interleave);

      float x0 = srcv[src_id][src_offset] * id[src_id];
      y[i].qs[j] = roundf(x0);
    }
  }
#endif
}

template <>
void __ggml_dequantize_row_q8_K(const void *__restrict _x, _FP16 *__restrict y,
                                int64_t k) {
  assert(k % QK_K == 0);
  const int64_t nb = k / QK_K;
  const block_q8_K *__restrict x = (const block_q8_K *__restrict)_x;

  for (int i = 0; i < nb; i++) {
    for (int j = 0; j < QK_K; ++j) {
      *y++ = static_cast<_FP16>(x[i].d * x[i].qs[j]);
    }
  }
}

void __ggml_quantize_row_q8_K_ref(const _FP16 *__restrict x,
                                  void *__restrict _y, int64_t k) {
  assert(k % QK_K == 0);
  const int64_t nb = k / QK_K;
  block_q8_K *__restrict y = (block_q8_K *__restrict)_y;

  for (int i = 0; i < nb; i++) {

    float max = 0;
    float amax = 0;
    for (int j = 0; j < QK_K; ++j) {
      float ax = fabsf(x[j]);
      if (ax > amax) {
        amax = ax;
        max = x[j];
      }
    }
    if (!amax) {
      y[i].d = 0;
      memset(y[i].qs, 0, QK_K);
      x += QK_K;
      continue;
    }
    // const float iscale = -128.f/max;
    //  We need this change for IQ2_XXS, else the AVX implementation becomes
    //  very awkward
    const float iscale = -127.f / max;
    for (int j = 0; j < QK_K; ++j) {
      int v = nearest_int(iscale * x[j]);
      y[i].qs[j] = MIN(127, v);
    }
    for (int j = 0; j < QK_K / 16; ++j) {
      int sum = 0;
      for (int ii = 0; ii < 16; ++ii) {
        sum += y[i].qs[j * 16 + ii];
      }
      y[i].bsums[j] = sum;
    }
    y[i].d = 1 / iscale;
    x += QK_K;
  }
}

template <>
void __ggml_quantize_row_q8_K(const _FP16 *__restrict x, void *__restrict y,
                              int64_t k) {
  __ggml_quantize_row_q8_K_ref(x, y, k);
}

template <>
void __ggml_gemm_q6_K(const unsigned int M, const unsigned int N,
                      const unsigned int K, const _FP16 *A,
                      const unsigned int lda, const void *B,
                      const unsigned int ldb, _FP16 *C,
                      const unsigned int ldc) {
  auto &tm = ThreadManager::Global();

  static constexpr const int32_t bs = 1;
  static constexpr const int32_t bx = 1;
  static constexpr const int32_t by = 1;
  static constexpr const int32_t nrc = 1;

  const int32_t blocks_per_row = (K + QK_K - 1) / QK_K;
  const int32_t A_row_size = sizeof(block_q8_K) * blocks_per_row;
  const int32_t B_row_size = sizeof(block_q6_K) * blocks_per_row;

  // GEMV. The inner kernel writes one FP32 result per call; convert to FP16
  // and store into C directly so we avoid the M*N FP32 scratch + final cast
  // that the original FP16 wrapper used.
  if (M == 1) {
    std::vector<char> quantized_A(A_row_size);
    __ggml_quantize_row_q8_K(A, (void *)quantized_A.data(), K);

    const void *const quantized_A_data = quantized_A.data();

    tm.parallel_for(0, static_cast<size_t>(N), [&](size_t thread_job) {
      const int32_t B_row_data_offset = B_row_size * thread_job;

      const void *const B_data = (void *)((char *)B + B_row_data_offset);

      float result;
      nntr_vec_dot_q6_K_q8_K(K, &result, bs, B_data, bx, quantized_A_data, by,
                             nrc);
      C[thread_job] = (_FP16)result;
    });
  } else { // GEMM. Same idea per (row,col): one FP32 result, cast to FP16,
           // store directly. No M*N FP32 scratch needed.
    const int32_t A_total_size = A_row_size * M;
    std::vector<char> quantized_A(A_total_size);

    tm.parallel_for(0, static_cast<size_t>(M), [&](size_t thread_job) {
      const int32_t A_row_data_offset = A_row_size * thread_job;
      void *A_data = (void *)((char *)quantized_A.data() + A_row_data_offset);
      __ggml_quantize_row_q8_K(A + thread_job * K, A_data, K);
    });

    tm.parallel_for(0, static_cast<size_t>(M), [&](size_t thread_job) {
      const int32_t A_row_data_offset = A_row_size * thread_job;
      void *A_data = (void *)((char *)quantized_A.data() + A_row_data_offset);

      for (uint32_t j = 0; j < N; j++) {
        const int32_t B_row_data_offset = B_row_size * j;
        const void *const B_data = (void *)((char *)B + B_row_data_offset);

        float result;
        nntr_vec_dot_q6_K_q8_K(K, &result, bs, B_data, bx, A_data, by, nrc);
        C[thread_job * ldc + j] = (_FP16)result;
      }
    });
  }
}

static inline void __ggml_q4_0_4x8_q8_0_GEMM_BSTP(
  const unsigned int M, const unsigned int N, const unsigned int K,
  const _FP16 *A, const unsigned int lda, const void *B, const unsigned int ldb,
  _FP16 *C16, const unsigned int ldc) {
  // Mirrors the FP32 OMP impl's 2D row+col chunking. The previous 1D
  // column-only split gave each of 4 threads a full M-row strip per
  // call, which hurt cache locality and load balance at long prefill
  // lengths. Using 16x16 (row x col) chunks restores the parallelism
  // granularity of the FP32 path while keeping FP16-direct stores.
  auto &tm = ThreadManager::Global();
  unsigned int blocks_per_4_rows = (K + QK8_0 - 1) / QK8_0;
  unsigned int qa_4_rows_size = sizeof(block_q8_0x4) * blocks_per_4_rows;
  const size_t qa_row_size = (sizeof(block_q8_0) * K) / QK8_0;
  unsigned int M4 = ((M - M % 4) / 4);
  int B_step = sizeof(block_q4_0) * (K / QK4_0);

  unsigned int qa_size = qa_4_rows_size * (((M >> 2) << 2) / 4 + 1);
  std::vector<char> QA = std::vector<char>(qa_size);

  tm.parallel_for(0, static_cast<size_t>(M4), [=, &QA](size_t i) {
    __ggml_quantize_mat_q8_0_4x8(A + 4 * i * K, QA.data() + i * qa_4_rows_size,
                                 K);
  });
  for (unsigned int i = M4 * 4; i < M; i++) {
    __ggml_quantize_row_q8_0(
      (_FP16 *)A + i * K,
      (QA.data() + (M4 * qa_4_rows_size) + (i - M4 * 4) * qa_row_size), K);
  }

  unsigned int row_chunk_size = 16;
  size_t row_loop = (M4 * 4 + row_chunk_size - 1) / row_chunk_size;
  unsigned int A_step = sizeof(block_q8_0) * (K / QK8_0);

  unsigned int col_chunk_size = 16;
  size_t col_loop = (N + col_chunk_size - 1) / col_chunk_size;

  tm.parallel_for(0, col_loop * row_loop, [=](size_t i) {
    unsigned int r = i / col_loop;
    unsigned int c = i % col_loop;

    unsigned int r_start = r * row_chunk_size;
    unsigned int r_end = std::min(row_chunk_size * (r + 1), M4 * 4);

    unsigned int c_start = c * col_chunk_size;
    unsigned int c_end = std::min(col_chunk_size * (c + 1), N);

#if defined(__ARM_NEON)
    nntr_gemm_q4_0_4x8_q8_0_fp16(K, (_FP16 *)(C16 + r_start * N + c_start), ldc,
                                 (void *)((char *)B + c_start * B_step),
                                 (void *)(QA.data() + r_start * A_step),
                                 r_end - r_start, c_end - c_start);
#else
    unsigned int t_rows = r_end - r_start;
    unsigned int t_cols = c_end - c_start;
    std::vector<float> tile(t_rows * t_cols);
    nntr_gemm_q4_0_4x8_q8_0(K, tile.data(), t_cols,
                             (void *)((char *)B + c_start * B_step),
                             (void *)(QA.data() + r_start * A_step), t_rows,
                             t_cols);
    for (unsigned int ii = 0; ii < t_rows; ++ii)
      __copy_f16_from_f32(&tile[ii * t_cols],
                          C16 + (r_start + ii) * N + c_start, t_cols);
#endif
  });

  // Leftover 1..3 rows still go through the FP32-output GEMV kernel into a
  // small per-(M%4) scratch, then we cast just that tail back into C16.
  unsigned int leftover_rows = M - M4 * 4;
  if (leftover_rows > 0) {
    std::vector<float> tail32(leftover_rows * (size_t)N);
    unsigned int chunk_size = 16;
    unsigned int loop = (N + chunk_size - 1) / chunk_size;
    for (unsigned int pb = M4 * 4; pb < M; pb++) {
      tm.parallel_for(0, loop, [=, &tail32](size_t idx) {
        unsigned int M_step_start = chunk_size * idx;
        unsigned int M_step_end = std::min(chunk_size * (idx + 1), (size_t)N);

        nntr_gemv_q4_0_4x8_q8_0(
          K, (float *)(tail32.data() + (pb - M4 * 4) * N) + M_step_start, N,
          (void *)((char *)B + M_step_start * B_step),
          QA.data() + (M4 * qa_4_rows_size) + (pb - M4 * 4) * qa_row_size, 1,
          M_step_end - M_step_start);
      });
    }
    __copy_f16_from_f32(tail32.data(), C16 + (size_t)M4 * 4 * N,
                        (size_t)leftover_rows * N);
  }
}

template <>
void __ggml_q4_0_4x8_q8_0_GEMM(const unsigned int M, const unsigned int N,
                               const unsigned int K, const _FP16 *A,
                               const unsigned int lda, const void *B,
                               const unsigned int ldb, _FP16 *C,
                               const unsigned int ldc) {
  auto &tm = ThreadManager::Global();

  // GEMV: M=1, used by per-token decode and small prefill. Mirror the FP32
  // OMP path's chunked parallel_for so the work is split across threads in
  // chunk_size=16 column tiles instead of one giant kernel call.
  if (M == 1) {
    unsigned int B_step = sizeof(block_q4_0) * (K / QK4_0);
    unsigned int blocks_per_row = (K + QK8_0 - 1) / QK8_0;
    unsigned int qa_size = sizeof(block_q8_0) * blocks_per_row;
    thread_local std::vector<char> QA;
    QA.resize(qa_size);
    __ggml_quantize_row_q8_0(A, (void *)QA.data(), K);
    auto qa_data = QA.data();

    unsigned int chunk_size = 16;
    unsigned int loop = (N + chunk_size - 1) / chunk_size;

    tm.parallel_for(0, loop, [=](size_t idx) {
      unsigned int M_step_start = chunk_size * idx;
      unsigned int M_step_end = std::min(chunk_size * (idx + 1), (size_t)N);

#if defined(__ARM_NEON)
      nntr_gemv_q4_0_4x8_q8_0_fp16(K, (_FP16 *)(C + M_step_start), N,
                                   (void *)((char *)B + M_step_start * B_step),
                                   qa_data, M, M_step_end - M_step_start);
#else
      unsigned int n_cols = M_step_end - M_step_start;
      std::vector<float> out(n_cols);
      nntr_gemv_q4_0_4x8_q8_0(K, out.data(), N,
                               (void *)((char *)B + M_step_start * B_step),
                               qa_data, M, n_cols);
      __copy_f16_from_f32(out.data(), C + M_step_start, n_cols);
#endif
    });
    return;
  }
  return __ggml_q4_0_4x8_q8_0_GEMM_BSTP(M, N, K, A, lda, B, ldb, C, ldc);
}

} // namespace nntrainer
