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

#include <conv_indirect.h>

#if defined(_WIN32) || defined(_MSC_VER)
#include <malloc.h>
#else
#include <alloca.h>
#endif
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

#if defined(__ARM_NEON)
/**
 * @brief Scale 8 FP16 activations by the q8_0 inverse-scale `id` and round to
 * int8, doing the multiply in the FP32 domain.
 *
 * `id` is the inverse scale `1/d` and can be very large for a block whose amax
 * is small (e.g. amax 1e-3 -> id ~1.3e5). Applying it with vmulq_n_f16 narrows
 * `id` to FP16, where anything above 65504 becomes +inf; src*inf then yields
 * inf/NaN in the quantized activation and poisons the GEMM (observed as
 * widespread NaN in W4A16 inference). The FP32 GEMM path and the scalar
 * fallback both multiply in float, so mirror them: widen src to FP32, multiply
 * by the float `id`, then convert to int8.
 */
static inline int16x8_t __q8_0_scale_f16x8(float16x8_t src, float id) {
  float32x4_t lo = vmulq_n_f32(vcvt_f32_f16(vget_low_f16(src)), id);
  float32x4_t hi = vmulq_n_f32(vcvt_f32_f16(vget_high_f16(src)), id);
  return vcombine_s16(vmovn_s32(vcvtnq_s32_f32(lo)),
                      vmovn_s32(vcvtnq_s32_f32(hi)));
}
#endif

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
      const int16x8_t vi = __q8_0_scale_f16x8(srcv[j], id);

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
      int16x8_t vi = __q8_0_scale_f16x8(srcv[0][j], id[0]);
      y[i].qs[32 * j + 0] = vgetq_lane_s16(vi, 0);
      y[i].qs[32 * j + 1] = vgetq_lane_s16(vi, 1);
      y[i].qs[32 * j + 2] = vgetq_lane_s16(vi, 2);
      y[i].qs[32 * j + 3] = vgetq_lane_s16(vi, 3);
      y[i].qs[32 * j + 4] = vgetq_lane_s16(vi, 4);
      y[i].qs[32 * j + 5] = vgetq_lane_s16(vi, 5);
      y[i].qs[32 * j + 6] = vgetq_lane_s16(vi, 6);
      y[i].qs[32 * j + 7] = vgetq_lane_s16(vi, 7);

      vi = __q8_0_scale_f16x8(srcv[1][j], id[1]);
      y[i].qs[32 * j + 8] = vgetq_lane_s16(vi, 0);
      y[i].qs[32 * j + 9] = vgetq_lane_s16(vi, 1);
      y[i].qs[32 * j + 10] = vgetq_lane_s16(vi, 2);
      y[i].qs[32 * j + 11] = vgetq_lane_s16(vi, 3);
      y[i].qs[32 * j + 12] = vgetq_lane_s16(vi, 4);
      y[i].qs[32 * j + 13] = vgetq_lane_s16(vi, 5);
      y[i].qs[32 * j + 14] = vgetq_lane_s16(vi, 6);
      y[i].qs[32 * j + 15] = vgetq_lane_s16(vi, 7);

      vi = __q8_0_scale_f16x8(srcv[2][j], id[2]);
      y[i].qs[32 * j + 16] = vgetq_lane_s16(vi, 0);
      y[i].qs[32 * j + 17] = vgetq_lane_s16(vi, 1);
      y[i].qs[32 * j + 18] = vgetq_lane_s16(vi, 2);
      y[i].qs[32 * j + 19] = vgetq_lane_s16(vi, 3);
      y[i].qs[32 * j + 20] = vgetq_lane_s16(vi, 4);
      y[i].qs[32 * j + 21] = vgetq_lane_s16(vi, 5);
      y[i].qs[32 * j + 22] = vgetq_lane_s16(vi, 6);
      y[i].qs[32 * j + 23] = vgetq_lane_s16(vi, 7);

      vi = __q8_0_scale_f16x8(srcv[3][j], id[3]);
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

#ifdef ENABLE_FP16
void __ggml_q4_0_4x8_q8_0_indirect_GEMM_fp16(
  const unsigned int M, const unsigned int N, const unsigned int K,
  const _FP16 *in, const ConvGatherParams &geom, const void *B,
  const unsigned int ldb, _FP16 *C, const unsigned int ldc) {
  /// FP16-activation indirect (im2col-fused) variant of
  /// __ggml_q4_0_4x8_q8_0_GEMM<_FP16>. The activation matrix A = [M=OH*OW,
  /// K=CRS] is never materialized. Each Q8_0 activation tile is gathered
  /// directly from the NCHW _FP16 input via gather_conv_act_rows_fp16
  /// (byte-identical im2col rows, _FP16 typed — no FP32 staging copy, so the
  /// indirect path's no-col-materialization memory win is preserved for FP16
  /// activations) and quantized on the fly with the FP16-input quantizers, then
  /// fed to the SAME FP16-output micro-kernels the non-indirect FP16 GEMM
  /// (BSTP) uses. The FP16 Q8_0 quantizer here is the id-FP32-narrowing-fixed
  /// variant (see __q8_0_scale_f16x8), so small-amax blocks no longer poison
  /// the GEMM with Inf/NaN.
  auto &tm = ThreadManager::Global();
  (void)ldb;

  const unsigned int blocks_per_4_rows = (K + QK8_0 - 1) / QK8_0;
  const unsigned int qa_4_rows_size = sizeof(block_q8_0x4) * blocks_per_4_rows;
  const size_t qa_row_size =
    (sizeof(block_q8_0) * K) / QK8_0; // ignore remainder
  const unsigned int M4 = M / 4;
  const unsigned int rem = M % 4;

  const unsigned int qa_size =
    qa_4_rows_size * M4 + static_cast<unsigned int>(qa_row_size) * rem;
  std::vector<char> QA(qa_size);
  char *QA_ptr = QA.data();

  /// Fused gather + Q8_0 quantize, parallel over chunks. Gather feeds the
  /// FP16-input quantizer __ggml_quantize_mat_q8_0_4x8(const _FP16*) one
  /// 4-row tile at a time through a small L1/L2-resident _FP16 buffer (no FP32
  /// staging). Byte-identical QA layout to the materialized FP16 path.
  const unsigned int QCHUNK = 64; // multiple of 4
  if (M4 > 0) {
    const unsigned int rows4 = M4 * 4;
    const size_t qloops = (rows4 + QCHUNK - 1) / QCHUNK;
    tm.parallel_for(0, qloops, [=](size_t q) {
      const unsigned int r0 = static_cast<unsigned int>(q) * QCHUNK;
      const unsigned int r1 = std::min(r0 + QCHUNK, rows4);
      _FP16 *tile = (_FP16 *)alloca(
        4 * K * sizeof(_FP16)); // zero heap overhead stack buffer
      for (unsigned int r = r0; r < r1; r += 4) {
        gather_conv_act_rows_fp16(tile, in, geom, (int)r, 4);
        __ggml_quantize_mat_q8_0_4x8(tile, QA_ptr + (r / 4) * qa_4_rows_size,
                                     K);
      }
    });
  }
  /// Remainder rows (M % 4): single-row gather + quantize.
  for (unsigned int i = M4 * 4; i < M; ++i) {
    _FP16 *staging =
      (_FP16 *)alloca(K * sizeof(_FP16)); // zero heap overhead stack buffer
    gather_conv_act_rows_fp16(staging, in, geom, (int)i, 1);
    __ggml_quantize_row_q8_0(
      staging, QA_ptr + (M4 * qa_4_rows_size) + (i - M4 * 4) * qa_row_size, K);
  }

  /// GEMM over the 4-row-divisible rows + GEMV over the remainder, mirroring
  /// __ggml_q4_0_4x8_q8_0_GEMM_BSTP's output block (same QA addressing, same
  /// FP16-output micro-kernels) — only the activation source changed above.
  const unsigned int A_step = sizeof(block_q8_0) * (K / QK8_0);
  const unsigned int B_step = sizeof(block_q4_0) * (K / QK4_0);

  if (M4 > 0) {
    const unsigned int row_chunk_size = 16;
    const size_t row_loop = (M4 * 4 + row_chunk_size - 1) / row_chunk_size;
    const unsigned int col_chunk_size = 16;
    const size_t col_loop = (N + col_chunk_size - 1) / col_chunk_size;

    tm.parallel_for(0, col_loop * row_loop, [=](size_t i) {
      unsigned int r = i / col_loop;
      unsigned int c = i % col_loop;

      unsigned int r_start = r * row_chunk_size;
      unsigned int r_end = std::min(row_chunk_size * (r + 1), M4 * 4);

      unsigned int c_start = c * col_chunk_size;
      unsigned int c_end = std::min(col_chunk_size * (c + 1), N);

#if defined(__ARM_NEON)
      nntr_gemm_q4_0_4x8_q8_0_fp16(K, (_FP16 *)(C + r_start * N + c_start), ldc,
                                   (void *)((char *)B + c_start * B_step),
                                   (void *)(QA_ptr + r_start * A_step),
                                   r_end - r_start, c_end - c_start);
#else
      unsigned int t_rows = r_end - r_start;
      unsigned int t_cols = c_end - c_start;
      std::vector<float> tile(t_rows * t_cols);
      nntr_gemm_q4_0_4x8_q8_0(K, tile.data(), t_cols,
                              (void *)((char *)B + c_start * B_step),
                              (void *)(QA_ptr + r_start * A_step), t_rows,
                              t_cols);
      for (unsigned int ii = 0; ii < t_rows; ++ii)
        __copy_f16_from_f32(&tile[ii * t_cols],
                            C + (r_start + ii) * N + c_start, t_cols);
#endif
    });
  }

  /// Leftover 1..3 rows: FP32-output GEMV into a small per-(M%4) scratch, then
  /// cast just that tail back into C (same as BSTP's leftover block).
  if (rem > 0) {
    std::vector<float> tail32(rem * (size_t)N);
    unsigned int chunk_size = 16;
    unsigned int loop = (N + chunk_size - 1) / chunk_size;
    for (unsigned int pb = M4 * 4; pb < M; pb++) {
      tm.parallel_for(0, loop, [=, &tail32](size_t idx) {
        unsigned int M_step_start = chunk_size * idx;
        unsigned int M_step_end = std::min(chunk_size * (idx + 1), (size_t)N);

        nntr_gemv_q4_0_4x8_q8_0(
          K, (float *)(tail32.data() + (pb - M4 * 4) * N) + M_step_start, N,
          (void *)((char *)B + M_step_start * B_step),
          QA_ptr + (M4 * qa_4_rows_size) + (pb - M4 * 4) * qa_row_size, 1,
          M_step_end - M_step_start);
      });
    }
    __copy_f16_from_f32(tail32.data(), C + (size_t)M4 * 4 * N, (size_t)rem * N);
  }
}

void __ggml_q4_0_4x8_q8_0_indirect_GEMM_q8_0(
  const unsigned int M, const unsigned int N, const unsigned int K,
  const void *in, const ConvGatherParams &geom, const void *B,
  const unsigned int ldb, _FP16 *C, const unsigned int ldc) {
  auto &tm = ThreadManager::Global();
  (void)ldb;

  const unsigned int blocks_per_4_rows = (K + QK8_0 - 1) / QK8_0;
  const unsigned int qa_4_rows_size = sizeof(block_q8_0x4) * blocks_per_4_rows;
  const size_t qa_row_size =
    (sizeof(block_q8_0) * K) / QK8_0; // ignore remainder
  const unsigned int M4 = M / 4;
  const unsigned int rem = M % 4;

  const unsigned int qa_size =
    qa_4_rows_size * M4 + static_cast<unsigned int>(qa_row_size) * rem;
  std::vector<char> QA(qa_size);
  char *QA_ptr = QA.data();

  const unsigned int QCHUNK = 64; // multiple of 4
  if (M4 > 0) {
    const unsigned int rows4 = M4 * 4;
    const size_t qloops = (rows4 + QCHUNK - 1) / QCHUNK;
    tm.parallel_for(0, qloops, [=](size_t q) {
      const unsigned int r0 = static_cast<unsigned int>(q) * QCHUNK;
      const unsigned int r1 = std::min(r0 + QCHUNK, rows4);
      for (unsigned int r = r0; r < r1; r += 4) {
        gather_conv_act_rows_q8_0(QA_ptr + (r / 4) * qa_4_rows_size, in, geom,
                                  (int)r, 4);
      }
    });
  }

  for (unsigned int i = M4 * 4; i < M; ++i) {
    gather_conv_act_rows_q8_0_single(QA_ptr + (M4 * qa_4_rows_size) +
                                       (i - M4 * 4) * qa_row_size,
                                     in, geom, (int)i);
  }

  const unsigned int A_step = sizeof(block_q8_0) * (K / QK8_0);
  const unsigned int B_step = sizeof(block_q4_0) * (K / QK4_0);

  if (M4 > 0) {
    const unsigned int row_chunk_size = 16;
    const size_t row_loop = (M4 * 4 + row_chunk_size - 1) / row_chunk_size;
    const unsigned int col_chunk_size = 16;
    const size_t col_loop = (N + col_chunk_size - 1) / col_chunk_size;

    tm.parallel_for(0, col_loop * row_loop, [=](size_t i) {
      unsigned int r = i / col_loop;
      unsigned int c = i % col_loop;

      unsigned int r_start = r * row_chunk_size;
      unsigned int r_end = std::min(row_chunk_size * (r + 1), M4 * 4);

      unsigned int c_start = c * col_chunk_size;
      unsigned int c_end = std::min(col_chunk_size * (c + 1), N);

#if defined(__ARM_NEON)
      nntr_gemm_q4_0_4x8_q8_0_fp16(K, (_FP16 *)(C + r_start * N + c_start), ldc,
                                   (void *)((char *)B + c_start * B_step),
                                   (void *)(QA_ptr + r_start * A_step),
                                   r_end - r_start, c_end - c_start);
#else
      unsigned int t_rows = r_end - r_start;
      unsigned int t_cols = c_end - c_start;
      std::vector<float> tile(t_rows * t_cols);
      nntr_gemm_q4_0_4x8_q8_0(K, tile.data(), t_cols,
                              (void *)((char *)B + c_start * B_step),
                              (void *)(QA_ptr + r_start * A_step), t_rows,
                              t_cols);
      for (unsigned int ii = 0; ii < t_rows; ++ii)
        __copy_f16_from_f32(&tile[ii * t_cols],
                            C + (r_start + ii) * N + c_start, t_cols);
#endif
    });
  }

  if (rem > 0) {
    std::vector<float> tail32(rem * (size_t)N);
    unsigned int chunk_size = 16;
    unsigned int loop = (N + chunk_size - 1) / chunk_size;
    for (unsigned int pb = M4 * 4; pb < M; pb++) {
      tm.parallel_for(0, loop, [=, &tail32](size_t idx) {
        unsigned int M_step_start = chunk_size * idx;
        unsigned int M_step_end = std::min(chunk_size * (idx + 1), (size_t)N);

        nntr_gemv_q4_0_4x8_q8_0(
          K, (float *)(tail32.data() + (pb - M4 * 4) * N) + M_step_start, N,
          (void *)((char *)B + M_step_start * B_step),
          QA_ptr + (M4 * qa_4_rows_size) + (pb - M4 * 4) * qa_row_size, 1,
          M_step_end - M_step_start);
      });
    }
    __copy_f16_from_f32(tail32.data(), C + (size_t)M4 * 4 * N, (size_t)rem * N);
  }
}

/**
 * @brief Q8_0-weight variant of the indirect conv GEMM.
 *
 * Mirrors __ggml_q4_0_4x8_q8_0_indirect_GEMM_q8_0 but the weight operand is a
 * plain (non-interleaved) block_q8_0 array [N, K/32] instead of Q4_0x8. Both
 * activation and weight are int8 Q8_0, so we reuse the proven plain-block
 * primitive nntr_gemm_q8_0_q8_0 (int8 SDOT core, no nibble unpack, no repack)
 * rather than a bespoke SMMLA kernel. The activation is gathered row-by-row as
 * plain block_q8_0 via gather_conv_act_rows_q8_0_single (same gather the Q4_0
 * path uses for its remainder rows). Output C is FP16.
 */
void __ggml_q8_0_q8_0_indirect_GEMM_q8_0(const unsigned int M,
                                         const unsigned int N,
                                         const unsigned int K, const void *in,
                                         const ConvGatherParams &geom,
                                         const void *B, const unsigned int ldb,
                                         _FP16 *C, const unsigned int ldc) {
  auto &tm = ThreadManager::Global();
  (void)ldb;
  (void)ldc;

  const unsigned int nb = K / QK8_0; // blocks per row (K multiple of 32)
  const size_t qa_row_size = (size_t)nb * sizeof(block_q8_0);

  // 1) Gather all M activation rows as plain block_q8_0 [M, nb].
  std::vector<char> QA((size_t)M * qa_row_size);
  char *QA_ptr = QA.data();
  {
    const unsigned int QCHUNK = 64;
    const size_t qloops = (M + QCHUNK - 1) / QCHUNK;
    tm.parallel_for(0, qloops, [=](size_t q) {
      const unsigned int r0 = static_cast<unsigned int>(q) * QCHUNK;
      const unsigned int r1 = std::min(r0 + QCHUNK, M);
      for (unsigned int r = r0; r < r1; ++r) {
        gather_conv_act_rows_q8_0_single(QA_ptr + (size_t)r * qa_row_size, in,
                                         geom, (int)r);
      }
    });
  }

  // 2) Tiled GEMM: plain Q8_0 weight [N, nb] x plain Q8_0 act [M, nb] -> FP16.
  const size_t B_step = (size_t)nb * sizeof(block_q8_0);
  const unsigned int row_chunk_size = 16;
  const unsigned int col_chunk_size = 16;
  const size_t row_loop = (M + row_chunk_size - 1) / row_chunk_size;
  const size_t col_loop = (N + col_chunk_size - 1) / col_chunk_size;

  tm.parallel_for(0, row_loop * col_loop, [=](size_t i) {
    unsigned int r = static_cast<unsigned int>(i / col_loop);
    unsigned int c = static_cast<unsigned int>(i % col_loop);

    unsigned int r_start = r * row_chunk_size;
    unsigned int r_end = std::min(row_chunk_size * (r + 1), M);
    unsigned int c_start = c * col_chunk_size;
    unsigned int c_end = std::min(col_chunk_size * (c + 1), N);

    unsigned int t_rows = r_end - r_start;
    unsigned int t_cols = c_end - c_start;

    std::vector<float> tile((size_t)t_rows * t_cols);
    nntr_gemm_q8_0_q8_0(
      (int)K, tile.data(), t_cols,
      (const void *)((const char *)B + (size_t)c_start * B_step),
      (const void *)(QA_ptr + (size_t)r_start * qa_row_size), (int)t_rows,
      (int)t_cols);

    for (unsigned int ii = 0; ii < t_rows; ++ii)
      __copy_f16_from_f32(&tile[(size_t)ii * t_cols],
                          C + (size_t)(r_start + ii) * N + c_start, t_cols);
  });
}

void __ggml_q8_0_q8_0_indirect_GEMM_fp16(const unsigned int M,
                                         const unsigned int N,
                                         const unsigned int K, const _FP16 *in,
                                         const ConvGatherParams &geom,
                                         const void *B, const unsigned int ldb,
                                         _FP16 *C, const unsigned int ldc) {
  /// Interleaved (q8_0x4) SMMLA path, mirroring the proven Q4_0 x8 FP16
  /// indirect GEMM. The activation is gathered in 4-row tiles and quantized to
  /// the interleaved block_q8_0x4 layout (same __ggml_quantize_mat_q8_0_4x8 the
  /// Q4_0 path uses), and the weight is consumed pre-interleaved to q8_0x4 by
  /// the C++ quantizer (Conv2DLayer::save calls repack_q8_0 at weight-export
  /// time). A q8_0x4 super-block is exactly 4 plain block_q8_0 rows of bytes,
  /// so the per-column weight stride B_step = nb*sizeof(block_q8_0) lands on
  /// 4-col super-block boundaries -- identical addressing trick to the Q4_0
  /// path. Both operands then feed the register-blocked 4x4 SMMLA kernel with
  /// single contiguous loads (see nntr_gemm_q8_0_q8_0_4x4_fp16).
  auto &tm = ThreadManager::Global();
  (void)ldb;
  (void)ldc;

  const unsigned int nb = K / QK8_0; // blocks per row (K multiple of 32)
  const unsigned int qa_4_rows_size = sizeof(block_q8_0x4) * nb;
  const unsigned int Mfull = (M / 4) * 4; // 4-row-divisible part
  const unsigned int rem = M % 4;
  const unsigned int M4 = M / 4;
  const unsigned int M4c = (M + 3) / 4; // groups incl. padded tail

  // 1) Fused gather + Q8_0 quantize to interleaved q8_0x4, 4 rows at a time.
  //    To match the highly optimized Q4_0 path's chunked parallel_for:
  //    we parallelize over larger chunks (QCHUNK = 64 rows / 16 tiles) instead
  //    of individual 4-row groups, reducing thread manager dispatch overheads
  //    drastically, and allocating our reusable tile buffer once per thread.
  std::vector<char> QA((size_t)M4c * qa_4_rows_size);
  char *QA_ptr = QA.data();

  const unsigned int QCHUNK = 64; // multiple of 4
  if (Mfull > 0) {
    const size_t qloops = (Mfull + QCHUNK - 1) / QCHUNK;
    tm.parallel_for(0, qloops, [=](size_t q) {
      const unsigned int r0 = static_cast<unsigned int>(q) * QCHUNK;
      const unsigned int r1 = std::min(r0 + QCHUNK, Mfull);
      _FP16 *tile = (_FP16 *)alloca(
        4 * K * sizeof(_FP16)); // zero heap overhead stack buffer
      for (unsigned int r = r0; r < r1; r += 4) {
        gather_conv_act_rows_fp16(tile, in, geom, (int)r, 4);
        __ggml_quantize_mat_q8_0_4x8(
          tile, QA_ptr + (size_t)(r / 4) * qa_4_rows_size, K);
      }
    });
  }

  // Handle M-tail (rem 1..3) gather and quantization
  if (rem > 0) {
    _FP16 *tile =
      (_FP16 *)alloca(4 * K * sizeof(_FP16)); // zero heap overhead stack buffer
    std::memset(tile, 0, 4 * K * sizeof(_FP16));
    gather_conv_act_rows_fp16(tile, in, geom, (int)Mfull, (int)rem);
    __ggml_quantize_mat_q8_0_4x8(tile, QA_ptr + (size_t)M4 * qa_4_rows_size, K);
  }

  // 2) Tiled 4x4 SMMLA GEMM over the 4-row-divisible part, direct to C.
  const size_t B_step = (size_t)nb * sizeof(block_q8_0);
  const size_t A_step = (size_t)nb * sizeof(block_q8_0); // 4 plain rows / super
  const unsigned int row_chunk_size = 16;                // multiple of 4
  const unsigned int col_chunk_size = 16;                // multiple of 4

  if (Mfull > 0) {
    const size_t row_loop = (Mfull + row_chunk_size - 1) / row_chunk_size;
    const size_t col_loop = (N + col_chunk_size - 1) / col_chunk_size;
    tm.parallel_for(0, row_loop * col_loop, [=](size_t i) {
      unsigned int r = static_cast<unsigned int>(i / col_loop);
      unsigned int c = static_cast<unsigned int>(i % col_loop);
      unsigned int r_start = r * row_chunk_size;
      unsigned int r_end = std::min(row_chunk_size * (r + 1), Mfull);
      unsigned int c_start = c * col_chunk_size;
      unsigned int c_end = std::min(col_chunk_size * (c + 1), N);

      nntr_gemm_q8_0_q8_0_4x4_fp16(
        (int)K, C + (size_t)r_start * N + c_start, N,
        (const void *)((const char *)B + (size_t)c_start * B_step),
        (const void *)(QA_ptr + (size_t)(r_start / 4) * qa_4_rows_size),
        (int)(r_end - r_start), (int)(c_end - c_start));
    });
  }

  // 3) M-tail (rem 1..3): run the padded last group into a 4xN FP16 scratch,
  //    then copy only the valid rem rows into C. (A_step unused; the tail
  //    super-block is the last one in QA.)
  (void)A_step;
  if (rem > 0) {
    _FP16 *scratch =
      (_FP16 *)alloca(4 * N * sizeof(_FP16)); // zero heap overhead stack buffer
    const char *tail_a = QA_ptr + (size_t)(M / 4) * qa_4_rows_size;
    const unsigned int col_loop = (N + col_chunk_size - 1) / col_chunk_size;
    tm.parallel_for(0, col_loop, [=](size_t c) {
      unsigned int c_start = static_cast<unsigned int>(c) * col_chunk_size;
      unsigned int c_end = std::min(col_chunk_size * ((unsigned int)c + 1), N);
      nntr_gemm_q8_0_q8_0_4x4_fp16(
        (int)K, scratch + c_start, N,
        (const void *)((const char *)B + (size_t)c_start * B_step),
        (const void *)tail_a, 4, (int)(c_end - c_start));
    });
    for (unsigned int rr = 0; rr < rem; ++rr)
      std::memcpy(C + (size_t)(Mfull + rr) * N, scratch + (size_t)rr * N,
                  (size_t)N * sizeof(_FP16));
  }
}
#endif

} // namespace nntrainer
