// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file neon_fp16.cpp
 * @date   23 April 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Half-precision computation functions based on NEON
 *
 */

#include <cstring>

#include <hgemm.h>
#include <matrix_transpose_neon.h>
#include <memory>
#include <neon_impl.h>
#include <neon_setting.h>
#ifdef ARMV7
#include <armv7_neon.h>
#endif

#include <iostream>

namespace nntrainer::neon {
bool is_valid(const unsigned int N, const __fp16 *input) {
  const uint16_t inf_bits = 0x7C00;
  const uint16_t n_inf_bits = 0xFC00;
  const uint16x8_t inf_v = vdupq_n_u16(inf_bits);
  const uint16x8_t neg_inf_v = vdupq_n_u16(n_inf_bits);

  size_t i = 0;
  for (; N - i >= 8; i += 8) {
    float16x8_t vec = vld1q_f16(&input[i]);
    uint16x8_t nan_check = vceqq_f16(vec, vec);
    nan_check = vmvnq_u16(nan_check); // Invert: 1s where NaN, 0s where not NaN
    if (vaddvq_u16(nan_check)) {
      return false;
    }

    uint16x8_t inf_check = vceqq_u16(vreinterpretq_u16_f16(vec), inf_v);
    if (vaddvq_u16(inf_check)) {
      return false;
    }
    inf_check = vceqq_u16(vreinterpretq_u16_f16(vec), neg_inf_v);
    if (vaddvq_u16(inf_check)) {
      return false;
    }
  }

  while (i < N) {
    __fp16 val = input[i];
    if (val != val) {
      return false;
    }
    uint16_t val_bits;
    std::memcpy((char *)&val_bits, (const char *)&val, sizeof(__fp16));
    if (val_bits == 0x7C00 || val_bits == 0xFC00) {
      return false;
    }
    ++i;
  }

  return true;
}

void hgemv(const __fp16 *A, const __fp16 *X, __fp16 *Y, uint32_t M, uint32_t N,
           float alpha, float beta) {
  const unsigned int batch = 0;
  const __fp16 *__restrict x;
  float *Y32 = new float[M];

  unsigned int idx = 0;

  for (; M - idx >= 8; idx += 8) {
    float32x4_t y0_3 = vcvt_f32_f16(vld1_f16(&Y[idx]));
    float32x4_t y4_7 = vcvt_f32_f16(vld1_f16(&Y[idx + 4]));
    y0_3 = vmulq_n_f32(y0_3, beta);
    y4_7 = vmulq_n_f32(y4_7, beta);

    vst1q_f32(&Y32[idx], y0_3);
    vst1q_f32(&Y32[idx + 4], y4_7);
  }
  for (; M - idx >= 4; idx += 4) {
    float32x4_t y0_3_32 = vcvt_f32_f16(vld1_f16(&Y[idx]));
    y0_3_32 = vmulq_n_f32(y0_3_32, beta);

    vst1q_f32(&Y32[idx], y0_3_32);
  }
  for (; idx < M; ++idx) {
    Y32[idx] = beta * Y[idx];
  }

  idx = 0;
  if (N / 120 >= batch) {
    for (; N - idx >= 120; idx += 120) {
      float16x8_t x0_7 = vld1q_f16(&X[idx]);
      float16x8_t x8_15 = vld1q_f16(&X[idx + 8]);
      float16x8_t x16_23 = vld1q_f16(&X[idx + 16]);
      float16x8_t x24_31 = vld1q_f16(&X[idx + 24]);

      float16x8_t x32_39 = vld1q_f16(&X[idx + 32]);
      float16x8_t x40_47 = vld1q_f16(&X[idx + 40]);
      float16x8_t x48_55 = vld1q_f16(&X[idx + 48]);
      float16x8_t x56_63 = vld1q_f16(&X[idx + 56]);

      float16x8_t x64_71 = vld1q_f16(&X[idx + 64]);
      float16x8_t x72_79 = vld1q_f16(&X[idx + 72]);
      float16x8_t x80_87 = vld1q_f16(&X[idx + 80]);

      float16x8_t x88_95 = vld1q_f16(&X[idx + 88]);
      float16x8_t x96_103 = vld1q_f16(&X[idx + 96]);
      float16x8_t x104_111 = vld1q_f16(&X[idx + 104]);
      float16x8_t x112_120 = vld1q_f16(&X[idx + 112]);

      if (std::fpclassify(alpha - 1.F) != FP_ZERO) {
        x0_7 = vmulq_n_f16(x0_7, alpha);
        x8_15 = vmulq_n_f16(x8_15, alpha);
        x16_23 = vmulq_n_f16(x16_23, alpha);
        x24_31 = vmulq_n_f16(x24_31, alpha);
        x32_39 = vmulq_n_f16(x32_39, alpha);
        x40_47 = vmulq_n_f16(x40_47, alpha);
        x48_55 = vmulq_n_f16(x48_55, alpha);
        x56_63 = vmulq_n_f16(x56_63, alpha);

        x64_71 = vmulq_n_f16(x64_71, alpha);
        x72_79 = vmulq_n_f16(x72_79, alpha);
        x80_87 = vmulq_n_f16(x80_87, alpha);
        x88_95 = vmulq_n_f16(x88_95, alpha);
        x96_103 = vmulq_n_f16(x96_103, alpha);
        x104_111 = vmulq_n_f16(x104_111, alpha);
        x112_120 = vmulq_n_f16(x112_120, alpha);
      }

      const __fp16 *__restrict w;

      for (unsigned int j = 0; j < M; ++j) {
        w = &A[j * N + idx];
        float16x8_t y = vmulq_f16(vld1q_f16(&w[0]), x0_7);
        y = vfmaq_f16(y, vld1q_f16(&w[8]), x8_15);
        y = vfmaq_f16(y, vld1q_f16(&w[16]), x16_23);
        y = vfmaq_f16(y, vld1q_f16(&w[24]), x24_31);

        y = vfmaq_f16(y, vld1q_f16(&w[32]), x32_39);
        y = vfmaq_f16(y, vld1q_f16(&w[40]), x40_47);
        y = vfmaq_f16(y, vld1q_f16(&w[48]), x48_55);
        y = vfmaq_f16(y, vld1q_f16(&w[56]), x56_63);

        y = vfmaq_f16(y, vld1q_f16(&w[64]), x64_71);
        y = vfmaq_f16(y, vld1q_f16(&w[72]), x72_79);
        y = vfmaq_f16(y, vld1q_f16(&w[80]), x80_87);

        y = vfmaq_f16(y, vld1q_f16(&w[88]), x88_95);
        y = vfmaq_f16(y, vld1q_f16(&w[96]), x96_103);
        y = vfmaq_f16(y, vld1q_f16(&w[104]), x104_111);
        y = vfmaq_f16(y, vld1q_f16(&w[112]), x112_120);

        Y32[j] += vaddvq_f32(vcvt_f32_f16(vget_low_f16(y))) +
                  vaddvq_f32(vcvt_f32_f16(vget_high_f16(y)));
      }
    }
  }
  if (N / 64 >= batch) {
    for (; N - idx >= 64; idx += 64) {
      float16x8_t x0_7 = vld1q_f16(&X[idx]);
      float16x8_t x8_15 = vld1q_f16(&X[idx + 8]);
      float16x8_t x16_23 = vld1q_f16(&X[idx + 16]);
      float16x8_t x24_31 = vld1q_f16(&X[idx + 24]);

      float16x8_t x32_39 = vld1q_f16(&X[idx + 32]);
      float16x8_t x40_47 = vld1q_f16(&X[idx + 40]);
      float16x8_t x48_55 = vld1q_f16(&X[idx + 48]);
      float16x8_t x56_63 = vld1q_f16(&X[idx + 56]);

      if (std::fpclassify(alpha - 1.F) != FP_ZERO) {
        x0_7 = vmulq_n_f16(x0_7, alpha);
        x8_15 = vmulq_n_f16(x8_15, alpha);
        x16_23 = vmulq_n_f16(x16_23, alpha);
        x24_31 = vmulq_n_f16(x24_31, alpha);
        x32_39 = vmulq_n_f16(x32_39, alpha);
        x40_47 = vmulq_n_f16(x40_47, alpha);
        x48_55 = vmulq_n_f16(x48_55, alpha);
        x56_63 = vmulq_n_f16(x56_63, alpha);
      }

      const __fp16 *__restrict w;

      for (unsigned int j = 0; j < M; ++j) {
        w = &A[j * N + idx];
        float16x8_t y = vmulq_f16(vld1q_f16(&w[0]), x0_7);
        y = vfmaq_f16(y, vld1q_f16(&w[8]), x8_15);
        y = vfmaq_f16(y, vld1q_f16(&w[16]), x16_23);
        y = vfmaq_f16(y, vld1q_f16(&w[24]), x24_31);

        y = vfmaq_f16(y, vld1q_f16(&w[32]), x32_39);
        y = vfmaq_f16(y, vld1q_f16(&w[40]), x40_47);
        y = vfmaq_f16(y, vld1q_f16(&w[48]), x48_55);
        y = vfmaq_f16(y, vld1q_f16(&w[56]), x56_63);

        Y32[j] += vaddvq_f32(vcvt_f32_f16(vget_low_f16(y))) +
                  vaddvq_f32(vcvt_f32_f16(vget_high_f16(y)));
      }
    }
  }
  if (N / 32 >= batch) {
    for (; N - idx >= 32; idx += 32) {
      float16x8_t x0_7 = vld1q_f16(&X[idx]);
      float16x8_t x8_15 = vld1q_f16(&X[idx + 8]);
      float16x8_t x16_23 = vld1q_f16(&X[idx + 16]);
      float16x8_t x24_31 = vld1q_f16(&X[idx + 24]);

      if (std::fpclassify(alpha - 1.F) != FP_ZERO) {
        x0_7 = vmulq_n_f16(x0_7, alpha);
        x8_15 = vmulq_n_f16(x8_15, alpha);
        x16_23 = vmulq_n_f16(x16_23, alpha);
        x24_31 = vmulq_n_f16(x24_31, alpha);
      }

      const __fp16 *__restrict w;

      for (unsigned int j = 0; j < M; ++j) {
        w = &A[j * N + idx];
        float16x8_t y = vmulq_f16(vld1q_f16(&w[0]), x0_7);
        y = vfmaq_f16(y, vld1q_f16(&w[8]), x8_15);
        y = vfmaq_f16(y, vld1q_f16(&w[16]), x16_23);
        y = vfmaq_f16(y, vld1q_f16(&w[24]), x24_31);

        Y32[j] += vaddvq_f32(vcvt_f32_f16(vget_low_f16(y))) +
                  vaddvq_f32(vcvt_f32_f16(vget_high_f16(y)));
      }
    }
  }
  if (N / 16 >= batch) {
    for (; N - idx >= 16; idx += 16) {
      float16x8_t x0_7 = vld1q_f16(&X[idx]);
      float16x8_t x8_15 = vld1q_f16(&X[idx + 8]);

      if (std::fpclassify(alpha - 1.F) != FP_ZERO) {
        x0_7 = vmulq_n_f16(x0_7, alpha);
        x8_15 = vmulq_n_f16(x8_15, alpha);
      }

      const __fp16 *__restrict w;
      for (unsigned int j = 0; j < M; ++j) {
        w = &A[j * N + idx];
        float16x8_t y = vmulq_f16(vld1q_f16(&w[0]), x0_7);
        y = vfmaq_f16(y, vld1q_f16(&w[8]), x8_15);

        Y32[j] += vaddvq_f32(vcvt_f32_f16(vget_low_f16(y))) +
                  vaddvq_f32(vcvt_f32_f16(vget_high_f16(y)));
      }
    }
  }
  for (; N - idx >= 8; idx += 8) {
    float16x8_t x0_7 = vld1q_f16(&X[idx]);

    if (std::fpclassify(alpha - 1.F) != FP_ZERO) {
      x0_7 = vmulq_n_f16(x0_7, alpha);
    }

    const __fp16 *__restrict w;

    for (unsigned int j = 0; j < M; ++j) {
      w = &A[j * N + idx];
      float16x8_t wvec0_7 = vld1q_f16(&w[0]);
      float16x8_t y = vmulq_f16(wvec0_7, x0_7);

      Y32[j] += vaddvq_f32(vcvt_f32_f16(vget_low_f16(y))) +
                vaddvq_f32(vcvt_f32_f16(vget_high_f16(y)));
    }
  }
  for (; N - idx >= 4; idx += 4) {
    float32x4_t x0_3 = vcvt_f32_f16(vld1_f16(&X[idx]));

    if (std::fpclassify(alpha - 1.F) != FP_ZERO) {
      x0_3 = vmulq_n_f32(x0_3, alpha);
    }

    const __fp16 *__restrict w;

    for (unsigned int j = 0; j < M; ++j) {
      w = &A[j * N + idx];
      float32x4_t wvec0_3 = vcvt_f32_f16(vld1_f16(&w[0]));
      float32x4_t y0 = vmulq_f32(wvec0_3, x0_3);

      Y32[j] += vaddvq_f32(y0);
    }
  }

  // now, N - idx is under 4 : 0 1 2 3 = N - idx
  if (N != idx) {
    float32x4_t x0_3 = vcvt_f32_f16(vld1_f16(&X[idx]));
    for (unsigned int j = N - idx; j < 4; ++j) {
      x0_3[j] = 0;
    }

    if (std::fpclassify(alpha - 1.F) != FP_ZERO) {
      x0_3 = vmulq_n_f32(x0_3, alpha);
    }

    const __fp16 *__restrict w;

    __fp16 yVal;

    for (unsigned int j = 0; j < M; ++j) {
      w = &A[j * N + idx];
      float32x4_t wvec0_3 = vcvt_f32_f16(vld1_f16(&w[0]));

      for (unsigned int n = N - idx; n < 4; ++n) {
        wvec0_3[n] = 0;
      }

      float32x4_t y0 = vmulq_f32(wvec0_3, x0_3);

      for (unsigned int n = 0; n < N - idx; ++n) {
        Y32[j] += y0[n];
      }
    }
  }

  copy_fp32_to_fp16(M, Y32, Y);
  delete[] Y32;
  return;
}

void hgemv_transpose(const __fp16 *A, const __fp16 *X, __fp16 *Y, uint32_t M,
                     uint32_t N, float alpha, float beta) {
#ifdef OMP_NUM_THREADS
  set_gemv_num_threads(OMP_NUM_THREADS);
#endif
  size_t GEMV_NUM_THREADS = get_gemv_num_threads();

  float *Y32 = new float[N];
  unsigned int idx = 0;
  for (; N - idx >= 8; idx += 8) {
    float32x4_t y0_3_32 = vcvt_f32_f16(vld1_f16(&Y[idx]));
    float32x4_t y4_7_32 = vcvt_f32_f16(vld1_f16(&Y[idx + 4]));

    y0_3_32 = vmulq_n_f32(y0_3_32, beta);
    y4_7_32 = vmulq_n_f32(y4_7_32, beta);

    vst1q_f32(&Y32[idx], y0_3_32);
    vst1q_f32(&Y32[idx + 4], y4_7_32);
  }
  for (; N - idx >= 4; idx += 4) {
    float32x4_t y0_3_32 = vcvt_f32_f16(vld1_f16(&Y[idx]));
    y0_3_32 = vmulq_n_f32(y0_3_32, beta);
    vst1q_f32(&Y32[idx], y0_3_32);
  }
  for (; N - idx >= 1; idx += 1) {
    Y32[idx] = beta * Y[idx];
  }
  unsigned int i = 0;
  unsigned int N8 = (N >> 3) << 3;
  for (; M - i >= 16; i += 16) {
    __fp16 x[16];
    vst1q_f16(&x[0], vmulq_n_f16(vld1q_f16(&X[i]), alpha));
    vst1q_f16(&x[8], vmulq_n_f16(vld1q_f16(&X[i + 8]), alpha));
#pragma omp parallel for schedule(guided) num_threads(GEMV_NUM_THREADS)
    for (unsigned int idx = 0; idx < N8; idx += 8) {
      float16x8_t wvec0_7_f16 = vmulq_n_f16(vld1q_f16(&A[i * N + idx]), x[0]);
      wvec0_7_f16 =
        vfmaq_n_f16(wvec0_7_f16, vld1q_f16(&A[(i + 1) * N + idx]), x[1]);
      wvec0_7_f16 =
        vfmaq_n_f16(wvec0_7_f16, vld1q_f16(&A[(i + 2) * N + idx]), x[2]);
      wvec0_7_f16 =
        vfmaq_n_f16(wvec0_7_f16, vld1q_f16(&A[(i + 3) * N + idx]), x[3]);

      float16x8_t w2vec0_7_f16 =
        vmulq_n_f16(vld1q_f16(&A[(i + 4) * N + idx]), x[4]);
      w2vec0_7_f16 =
        vfmaq_n_f16(w2vec0_7_f16, vld1q_f16(&A[(i + 5) * N + idx]), x[5]);
      w2vec0_7_f16 =
        vfmaq_n_f16(w2vec0_7_f16, vld1q_f16(&A[(i + 6) * N + idx]), x[6]);
      w2vec0_7_f16 =
        vfmaq_n_f16(w2vec0_7_f16, vld1q_f16(&A[(i + 7) * N + idx]), x[7]);

      float16x8_t w3vec0_7_f16 =
        vmulq_n_f16(vld1q_f16(&A[(i + 8) * N + idx]), x[8]);
      w3vec0_7_f16 =
        vfmaq_n_f16(w3vec0_7_f16, vld1q_f16(&A[(i + 9) * N + idx]), x[9]);
      w3vec0_7_f16 =
        vfmaq_n_f16(w3vec0_7_f16, vld1q_f16(&A[(i + 10) * N + idx]), x[10]);
      w3vec0_7_f16 =
        vfmaq_n_f16(w3vec0_7_f16, vld1q_f16(&A[(i + 11) * N + idx]), x[11]);

      float16x8_t w4vec0_7_f16 =
        vmulq_n_f16(vld1q_f16(&A[(i + 12) * N + idx]), x[12]);
      w4vec0_7_f16 =
        vfmaq_n_f16(w4vec0_7_f16, vld1q_f16(&A[(i + 13) * N + idx]), x[13]);
      w4vec0_7_f16 =
        vfmaq_n_f16(w4vec0_7_f16, vld1q_f16(&A[(i + 14) * N + idx]), x[14]);
      w4vec0_7_f16 =
        vfmaq_n_f16(w4vec0_7_f16, vld1q_f16(&A[(i + 15) * N + idx]), x[15]);

      wvec0_7_f16 = vaddq_f16(wvec0_7_f16, w3vec0_7_f16);
      w2vec0_7_f16 = vaddq_f16(w2vec0_7_f16, w4vec0_7_f16);

      float32x4_t y0_3 = vaddq_f32(vld1q_f32(&Y32[idx]),
                                   vcvt_f32_f16(vget_low_f16(wvec0_7_f16)));
      y0_3 = vaddq_f32(y0_3, vcvt_f32_f16(vget_low_f16(w2vec0_7_f16)));
      float32x4_t y4_7 = vaddq_f32(vld1q_f32(&Y32[idx + 4]),
                                   vcvt_f32_f16(vget_high_f16(wvec0_7_f16)));
      y4_7 = vaddq_f32(y4_7, vcvt_f32_f16(vget_high_f16(w2vec0_7_f16)));

      vst1q_f32(&Y32[idx], y0_3);
      vst1q_f32(&Y32[idx + 4], y4_7);
    }

    if (N != N8) {
      unsigned int idx = N8;

      float y0_7[8];
      float v0[8], v1[8], v2[8], v3[8];
      float v4[8], v5[8], v6[8], v7[8];
      float v8[8], v9[8], v10[8], v11[8];
      float v12[8], v13[8], v14[8], v15[8];

      unsigned int n = 0;
      for (; n < N - idx; ++n) {
        y0_7[n] = Y32[idx + n];

        v0[n] = A[i * N + idx + n];
        v1[n] = A[(i + 1) * N + idx + n];
        v2[n] = A[(i + 2) * N + idx + n];
        v3[n] = A[(i + 3) * N + idx + n];
        v4[n] = A[(i + 4) * N + idx + n];
        v5[n] = A[(i + 5) * N + idx + n];
        v6[n] = A[(i + 6) * N + idx + n];
        v7[n] = A[(i + 7) * N + idx + n];
        v8[n] = A[(i + 8) * N + idx + n];
        v9[n] = A[(i + 9) * N + idx + n];
        v10[n] = A[(i + 10) * N + idx + n];
        v11[n] = A[(i + 11) * N + idx + n];
        v12[n] = A[(i + 12) * N + idx + n];
        v13[n] = A[(i + 13) * N + idx + n];
        v14[n] = A[(i + 14) * N + idx + n];
        v15[n] = A[(i + 15) * N + idx + n];
      }
      for (; n < 8; ++n) {
        y0_7[n] = 0;
        v0[n] = v1[n] = v2[n] = v3[n] = 0;
        v4[n] = v5[n] = v6[n] = v7[n] = 0;
        v8[n] = v9[n] = v10[n] = v11[n] = 0;
        v12[n] = v13[n] = v14[n] = v15[n] = 0;
      }

      float32x4_t y0_3 = vld1q_f32(&y0_7[0]);
      float32x4_t y4_7 = vld1q_f32(&y0_7[4]);

      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v0[0]), x[0]);
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v1[0]), x[1]);
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v2[0]), x[2]);
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v3[0]), x[3]);
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v4[0]), x[4]);
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v5[0]), x[5]);
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v6[0]), x[6]);
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v7[0]), x[7]);
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v8[0]), x[8]);
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v9[0]), x[9]);
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v10[0]), x[10]);
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v11[0]), x[11]);
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v12[0]), x[12]);
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v13[0]), x[13]);
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v14[0]), x[14]);
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v15[0]), x[15]);

      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v0[4]), x[0]);
      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v1[4]), x[1]);
      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v2[4]), x[2]);
      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v3[4]), x[3]);
      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v4[4]), x[4]);
      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v5[4]), x[5]);
      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v6[4]), x[6]);
      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v7[4]), x[7]);
      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v8[4]), x[8]);
      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v9[4]), x[9]);
      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v10[4]), x[10]);
      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v11[4]), x[11]);
      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v12[4]), x[12]);
      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v13[4]), x[13]);
      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v14[4]), x[14]);
      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v15[4]), x[15]);

      for (unsigned int n = 0; n < N - idx; ++n) {
        if (n < 4)
          Y32[idx + n] = y0_3[n];
        else
          Y32[idx + n] = y4_7[n];
      }
    }
  }
  for (; M - i >= 8; i += 8) {
    __fp16 x[8];
    vst1q_f16(&x[0], vmulq_n_f16(vld1q_f16(&X[i]), alpha));
#pragma omp parallel for schedule(guided) num_threads(GEMV_NUM_THREADS)
    for (unsigned int idx = 0; idx < N8; idx += 8) {
      float16x8_t wvec0_7_f16 = vmulq_n_f16(vld1q_f16(&A[i * N + idx]), x[0]);
      wvec0_7_f16 =
        vfmaq_n_f16(wvec0_7_f16, vld1q_f16(&A[(i + 1) * N + idx]), x[1]);
      wvec0_7_f16 =
        vfmaq_n_f16(wvec0_7_f16, vld1q_f16(&A[(i + 2) * N + idx]), x[2]);
      wvec0_7_f16 =
        vfmaq_n_f16(wvec0_7_f16, vld1q_f16(&A[(i + 3) * N + idx]), x[3]);

      float16x8_t w2vec0_7_f16 =
        vmulq_n_f16(vld1q_f16(&A[(i + 4) * N + idx]), x[4]);
      w2vec0_7_f16 =
        vfmaq_n_f16(w2vec0_7_f16, vld1q_f16(&A[(i + 5) * N + idx]), x[5]);
      w2vec0_7_f16 =
        vfmaq_n_f16(w2vec0_7_f16, vld1q_f16(&A[(i + 6) * N + idx]), x[6]);
      w2vec0_7_f16 =
        vfmaq_n_f16(w2vec0_7_f16, vld1q_f16(&A[(i + 7) * N + idx]), x[7]);

      float32x4_t y0_3 = vaddq_f32(vld1q_f32(&Y32[idx]),
                                   vcvt_f32_f16(vget_low_f16(wvec0_7_f16)));
      y0_3 = vaddq_f32(y0_3, vcvt_f32_f16(vget_low_f16(w2vec0_7_f16)));
      float32x4_t y4_7 = vaddq_f32(vld1q_f32(&Y32[idx + 4]),
                                   vcvt_f32_f16(vget_high_f16(wvec0_7_f16)));
      y4_7 = vaddq_f32(y4_7, vcvt_f32_f16(vget_high_f16(w2vec0_7_f16)));

      vst1q_f32(&Y32[idx], y0_3);
      vst1q_f32(&Y32[idx + 4], y4_7);
    }

    if (N != N8) {
      unsigned int idx = N8;

      float y0_7[8];
      float v0[8], v1[8], v2[8], v3[8];
      float v4[8], v5[8], v6[8], v7[8];

      unsigned int n = 0;
      for (; n < N - idx; ++n) {
        y0_7[n] = Y32[idx + n];

        v0[n] = A[i * N + idx + n];
        v1[n] = A[(i + 1) * N + idx + n];
        v2[n] = A[(i + 2) * N + idx + n];
        v3[n] = A[(i + 3) * N + idx + n];
        v4[n] = A[(i + 4) * N + idx + n];
        v5[n] = A[(i + 5) * N + idx + n];
        v6[n] = A[(i + 6) * N + idx + n];
        v7[n] = A[(i + 7) * N + idx + n];
      }
      for (; n < 8; ++n) {
        y0_7[n] = 0;
        v0[n] = v1[n] = v2[n] = v3[n] = 0;
        v4[n] = v5[n] = v6[n] = v7[n] = 0;
      }

      float32x4_t y0_3 = vld1q_f32(&y0_7[0]);
      float32x4_t y4_7 = vld1q_f32(&y0_7[4]);

      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v0[0]), x[0]);
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v1[0]), x[1]);
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v2[0]), x[2]);
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v3[0]), x[3]);
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v4[0]), x[4]);
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v5[0]), x[5]);
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v6[0]), x[6]);
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v7[0]), x[7]);

      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v0[4]), x[0]);
      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v1[4]), x[1]);
      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v2[4]), x[2]);
      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v3[4]), x[3]);
      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v4[4]), x[4]);
      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v5[4]), x[5]);
      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v6[4]), x[6]);
      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v7[4]), x[7]);

      for (unsigned int n = 0; n < N - idx; ++n) {
        if (n < 4)
          Y32[idx + n] = y0_3[n];
        else
          Y32[idx + n] = y4_7[n];
      }
    }
  }
  for (; M - i >= 4; i += 4) {
    __fp16 x[4];
    vst1_f16(&x[0], vmul_n_f16(vld1_f16(&X[i]), alpha));
#pragma omp parallel for schedule(guided) num_threads(GEMV_NUM_THREADS)
    for (unsigned int idx = 0; idx < N8; idx += 8) {
      float16x8_t wvec0_7_f16 = vmulq_n_f16(vld1q_f16(&A[i * N + idx]), x[0]);
      wvec0_7_f16 =
        vfmaq_n_f16(wvec0_7_f16, vld1q_f16(&A[(i + 1) * N + idx]), x[1]);
      float16x8_t w2vec0_7_f16 =
        vmulq_n_f16(vld1q_f16(&A[(i + 2) * N + idx]), x[2]);
      w2vec0_7_f16 =
        vfmaq_n_f16(w2vec0_7_f16, vld1q_f16(&A[(i + 3) * N + idx]), x[3]);

      float32x4_t y0_3 = vaddq_f32(vld1q_f32(&Y32[idx]),
                                   vcvt_f32_f16(vget_low_f16(wvec0_7_f16)));
      y0_3 = vaddq_f32(y0_3, vcvt_f32_f16(vget_low_f16(w2vec0_7_f16)));
      float32x4_t y4_7 = vaddq_f32(vld1q_f32(&Y32[idx + 4]),
                                   vcvt_f32_f16(vget_high_f16(wvec0_7_f16)));
      y4_7 = vaddq_f32(y4_7, vcvt_f32_f16(vget_high_f16(w2vec0_7_f16)));

      vst1q_f32(&Y32[idx], y0_3);
      vst1q_f32(&Y32[idx + 4], y4_7);
    }
    if (N != N8) {
      unsigned int idx = N8;
      float y0_3_0[8];
      float v0[8], v1[8], v2[8], v3[8];
      unsigned int n = 0;
      for (; n < N - idx; ++n) {
        y0_3_0[n] = Y32[idx + n];

        v0[n] = A[i * N + idx + n];
        v1[n] = A[(i + 1) * N + idx + n];
        v2[n] = A[(i + 2) * N + idx + n];
        v3[n] = A[(i + 3) * N + idx + n];
      }
      for (; n < 8; ++n) {
        y0_3_0[n] = 0;
        v0[n] = v1[n] = v2[n] = v3[n] = 0;
      }

      float32x4_t y0_3 = vld1q_f32(&y0_3_0[0]);
      float32x4_t y4_7 = vld1q_f32(&y0_3_0[4]);

      // we can separate mul and accum for faster compute.
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v0[0]), x[0]);
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v1[0]), x[1]);
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v2[0]), x[2]);
      y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(&v3[0]), x[3]);

      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v0[4]), x[0]);
      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v1[4]), x[1]);
      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v2[4]), x[2]);
      y4_7 = vfmaq_n_f32(y4_7, vld1q_f32(&v3[4]), x[3]);

      for (unsigned int n = 0; n < N - idx; ++n) {
        if (n < 4)
          Y32[idx + n] = y0_3[n];
        else
          Y32[idx + n] = y4_7[n];
      }
    }
  }
  for (; i < M; ++i) {
    __fp16 x = alpha * (X[i]);
#pragma omp parallel for schedule(guided) num_threads(GEMV_NUM_THREADS)
    for (unsigned int idx = 0; idx < N8; idx += 8) {
      float16x8_t wvec0_7_f16 = vmulq_n_f16(vld1q_f16(&A[i * N + idx]), x);
      float32x4_t y0_3 = vaddq_f32(vld1q_f32(&Y32[idx]),
                                   vcvt_f32_f16(vget_low_f16(wvec0_7_f16)));
      float32x4_t y4_7 = vaddq_f32(vld1q_f32(&Y32[idx + 4]),
                                   vcvt_f32_f16(vget_high_f16(wvec0_7_f16)));

      vst1q_f32(&Y32[idx], y0_3);
      vst1q_f32(&Y32[idx + 4], y4_7);
    }
    if (N != N8) {
      unsigned int idx = N8;
      float v0[8];
      for (unsigned int j = 0; j < N - idx; ++j) {
        v0[j] = A[i * N + idx + j];
      }
      for (unsigned int j = N - idx; j < 8; ++j) {
        v0[j] = 0;
      }

      for (unsigned int j = 0; j < N - idx; ++j) {
        Y32[idx + j] = Y32[idx + j] + v0[j] * x;
      }
    }
  }
  copy_fp32_to_fp16(N, Y32, Y);
  delete[] Y32;
  return;
}

void haxpy(const unsigned int N, const float alpha, const __fp16 *X,
           __fp16 *Y) {

  const float16x8_t v_alphaX8 = vmovq_n_f16(alpha);
  const float16x4_t v_alphaX4 = vmov_n_f16(alpha);

  unsigned int idx = 0;

  // processing batch of 8
  for (; (N - idx) >= 8; idx += 8) {
    float16x8_t x = vld1q_f16(&X[idx]);
    float16x8_t y = vld1q_f16(&Y[idx]);

    // alpha*X + Y -> mulacc
    float16x8_t mulacc = vfmaq_f16(y, v_alphaX8, x);
    vst1q_f16(&Y[idx], mulacc);
  }

  // processing remaining batch of 4
  for (; (N - idx) >= 4; idx += 4) {
    float16x4_t x = vld1_f16(&X[idx]);
    float16x4_t y = vld1_f16(&Y[idx]);

    // alpha*X + Y -> mulacc
    float16x4_t mulacc = vfma_f16(y, v_alphaX4, x);
    vst1_f16(&Y[idx], mulacc);
  }

  // pocessing remaining values
  for (; idx < N; idx++)
    Y[idx] = Y[idx] + alpha * X[idx];
}

__fp16 hdot(const unsigned int N, const __fp16 *X, const __fp16 *Y) {
  float32x4_t accX0_3 = vmovq_n_f32(0.F);
  float32x4_t accX4_7 = vmovq_n_f32(0.F);

  unsigned int idx = 0;
  unsigned int N8 = (N >> 3) << 3;
  float ret = 0;

  // Adaptive loop for batch size of 8
  for (; idx < N8; idx += 8) {
    float16x8_t x = vld1q_f16(&X[idx]);
    float16x8_t y = vld1q_f16(&Y[idx]);

    x = vmulq_f16(x, y);
    accX0_3 = vaddq_f32(accX0_3, vcvt_f32_f16(vget_low_f16(x)));
    accX4_7 = vaddq_f32(accX4_7, vcvt_f32_f16(vget_high_f16(x)));
  }
  ret += vaddvq_f32(accX0_3) + vaddvq_f32(accX4_7);

  // Loop for remaining indices
  for (; idx < N; idx++)
    ret += X[idx] * Y[idx];

  return static_cast<__fp16>(ret);
}

__fp16 hnrm2(const unsigned int N, const __fp16 *X) {
  float32x4_t accX0_3 = vmovq_n_f32(0.F);
  float32x4_t accX4_7 = vmovq_n_f32(0.F);

  unsigned int idx = 0;
  unsigned int N8 = (N >> 3) << 3;
  float ret = 0;

  // Adaptive loop for batch size of 8
  for (; idx < N8; idx += 8) {
    float16x8_t x0_7 = vld1q_f16(&X[idx]);

    x0_7 = vmulq_f16(x0_7, x0_7);
    accX0_3 = vaddq_f32(accX0_3, vcvt_f32_f16(vget_low_f16(x0_7)));
    accX4_7 = vaddq_f32(accX4_7, vcvt_f32_f16(vget_high_f16(x0_7)));
  }
  ret += vaddvq_f32(accX0_3) + vaddvq_f32(accX4_7);

  // Loop for remaining indices
  for (; idx < N; idx++) {
    ret += X[idx] * X[idx];
  }

  return static_cast<__fp16>(sqrt(ret));
}

void hscal(const unsigned int N, __fp16 *X, const float alpha) {
  const float16x8_t v_alphaX8 = vmovq_n_f16(alpha);
  const float16x4_t v_alphaX4 = vmov_n_f16(alpha);

  unsigned int idx = 0;

  // processing batch of 8
  for (; (N - idx) >= 8; idx += 8) {
    float16x8_t x = vld1q_f16(&X[idx]);

    // alpha*X -> X
    float16x8_t mulacc = vmulq_f16(v_alphaX8, x);
    vst1q_f16(&X[idx], mulacc);
  }

  // processing remaining batch of 4
  for (; (N - idx) >= 4; idx += 4) {
    float16x4_t x = vld1_f16(&X[idx]);

    // alpha*X -> X
    float16x4_t mulacc = vmul_f16(v_alphaX4, x);
    vst1_f16(&X[idx], mulacc);
  }

  // pocessing remaining values
  for (; idx < N; idx++)
    X[idx] = alpha * X[idx];
}

float32x4_t vcvtq_f32_u32_bitwise(uint32x4_t u32) {
  constexpr uint32_t offsetValue = 0x4b000000;
  const uint32x4_t offsetInt = vdupq_n_u32(offsetValue);
  return vsubq_f32(vreinterpretq_f32_u32(vorrq_u32(u32, offsetInt)),
                   vreinterpretq_f32_u32(offsetInt));
}

void hcopy(const unsigned int N, const __fp16 *X, __fp16 *Y) {

  unsigned int idx = 0;

  // processing batch of 8
  for (; (N - idx) >= 8; idx += 8) {
    float16x8_t batch = vld1q_f16(&X[idx]);
    vst1q_f16(&Y[idx], batch);
  }

  // processing remaining batch of 4
  for (; (N - idx) >= 4; idx += 4) {
    float16x4_t batch = vld1_f16(&X[idx]);
    vst1_f16(&Y[idx], batch);
  }

  // pocessing remaining values
  for (; idx < N; idx++)
    Y[idx] = X[idx];
}

void copy_int4_to_fp16(const unsigned int N, const uint8_t *X, __fp16 *Y) {

  unsigned int idx = 0;

  // keep in mind that : len(X) = N, and len(Y) = 2*N

  // processing batch of 16

  float16x8_t y0, y1, y2, y3;
  float16x4_t yh0, yh1;

  uint8_t low0, low1, high0, high1;

  for (; (N - idx) >= 16; idx += 16) {
    uint8x16_t batch = vld1q_u8(&X[idx]);

    uint8x8_t low = vget_low_u8(batch);
    uint8x8_t high = vget_high_u8(batch);

    for (unsigned int i = 0; i < 8; ++i) {
      low0 = low[i] >> 4;
      low1 = low[i] & 0x0f;

      high0 = high[i] >> 4;
      high1 = high[i] & 0x0f;

      if (i < 4) {
        y0[2 * i] = low0;
        y0[2 * i + 1] = low1;
      } else {
        y1[2 * (i - 4)] = low0;
        y1[2 * (i - 4) + 1] = low1;
      }

      if (i < 4) {
        y2[2 * i] = high0;
        y2[2 * i + 1] = high1;
      } else {
        y3[2 * (i - 4)] = high0;
        y3[2 * (i - 4) + 1] = high1;
      }
    }

    vst1q_f16(&Y[2 * idx], y0);
    vst1q_f16(&Y[2 * idx + 8], y1);
    vst1q_f16(&Y[2 * idx + 16], y2);
    vst1q_f16(&Y[2 * idx + 24], y3);
  }

  // processing remaining batch of 8
  for (; (N - idx) >= 8; idx += 8) {
    uint8x8_t batch = vld1_u8(&X[idx]);

    for (unsigned int i = 0; i < 8; ++i) {
      low0 = batch[i] >> 4;
      low1 = batch[i] & 0x0f;

      if (i < 4) {
        y0[2 * i] = low0;
        y0[2 * i + 1] = low1;
      } else {
        y1[2 * (i - 4)] = low0;
        y1[2 * (i - 4) + 1] = low1;
      }
    }

    vst1q_f16(&Y[2 * idx], y0);
    vst1q_f16(&Y[2 * idx + 8], y1);
  }

  // pocessing remaining values
  for (; idx < N; idx++) {
    Y[2 * idx] = X[idx] >> 4;
    Y[2 * idx + 1] = X[idx] & 0x0f;
  }
}

void copy_int8_to_fp16(const unsigned int N, const uint8_t *X, __fp16 *Y) {
  unsigned int idx = 0;
  for (; (N - idx) >= 16; idx += 16) {
    uint8x16_t batch = vld1q_u8(&X[idx]);
    uint8x8_t low = vget_low_u8(batch);
    uint8x8_t high = vget_high_u8(batch);

    // convert to u16
    uint16x8_t batch_low_u16 = vmovl_u8(low);
    uint16x8_t batch_high_u16 = vmovl_u8(high);

    // todo : experiment with vcvt_f32_u32_ bitwise operation w.r.t.
    // time/accuracy
    vst1q_f16(&Y[idx], vcvtq_f16_u16(batch_low_u16));
    vst1q_f16(&Y[idx + 8], vcvtq_f16_u16(batch_high_u16));
  }
  for (; (N - idx) >= 8; idx += 8) {
    uint8x8_t batch = vld1_u8(&X[idx]);

    // convert to u16
    uint16x8_t batch_u16 = vmovl_u8(batch);
    vst1q_f16(&Y[idx], vcvtq_f16_u16(batch_u16));
  }
  for (; (N - idx) >= 1; ++idx) {
    Y[idx] = X[idx];
  }
}

void copy_int8_to_fp16(const unsigned int N, const int8_t *X, __fp16 *Y) {
  unsigned int idx = 0;
  for (; (N - idx) >= 16; idx += 16) {
    int8x16_t batch = vld1q_s8(&X[idx]);
    int8x8_t low = vget_low_s8(batch);
    int8x8_t high = vget_high_s8(batch);

    // convert to s16
    int16x8_t batch_low_s16 = vmovl_s8(low);
    int16x8_t batch_high_s16 = vmovl_s8(high);

    // todo : experiment with vcvt_f32_s32_ bitwise operation w.r.t.
    // time/accuracy
    vst1q_f16(&Y[idx], vcvtq_f16_s16(batch_low_s16));
    vst1q_f16(&Y[idx + 8], vcvtq_f16_s16(batch_high_s16));
  }
  for (; (N - idx) >= 8; idx += 8) {
    int8x8_t batch = vld1_s8(&X[idx]);

    // convert to s16
    int16x8_t batch_s16 = vmovl_s8(batch);
    vst1q_f16(&Y[idx], vcvtq_f16_s16(batch_s16));
  }
  for (; (N - idx) >= 1; ++idx) {
    Y[idx] = X[idx];
  }
}

void copy_fp16_to_fp32(const unsigned int N, const __fp16 *X, float *Y) {
  unsigned int idx = 0;

  for (; N - idx >= 8; idx += 8) {
    float32x4_t y1 = vcvt_f32_f16(vld1_f16(&X[idx]));
    float32x4_t y2 = vcvt_f32_f16(vld1_f16(&X[idx + 4]));

    vst1q_f32(&Y[idx], y1);
    vst1q_f32(&Y[idx + 4], y2);
  }

  for (; N - idx >= 4; idx += 4) {
    float16x4_t x = vld1_f16(&X[idx]);
    float32x4_t y1 = vcvt_f32_f16(x);
    vst1q_f32(&Y[idx], y1);
  }

  for (; idx < N; ++idx) {
    Y[idx] = static_cast<float>(X[idx]);
  }
}

void copy_fp32_to_fp16(const unsigned int N, const float *X, __fp16 *Y) {
  unsigned int idx = 0;

  for (; N - idx >= 8; idx += 8) {
    float32x4_t x1 = vld1q_f32(&X[idx]);
    float32x4_t x2 = vld1q_f32(&X[idx + 4]);

    float16x8_t y1 = vcombine_f16(vcvt_f16_f32(x1), vcvt_f16_f32(x2));

    vst1q_f16(&Y[idx], y1);
  }

  for (; N - idx >= 4; idx += 4) {
    float32x4_t x1 = vld1q_f32(&X[idx]);

    float16x4_t y1 = vcvt_f16_f32(x1);

    vst1_f16(&Y[idx], y1);
  }

  for (; idx < N; ++idx) {
    Y[idx] = static_cast<__fp16>(X[idx]);
  }
}

unsigned int isamax(const unsigned int N, const __fp16 *X) {

  unsigned int retIdx;
  __fp16 maxNum;

  uint16_t indices[] = {0, 1, 2, 3, 4, 5, 6, 7};
  uint16x8_t stride = vmovq_n_u16(8);
  float16x8_t batch = vld1q_f16(&X[0]);
  uint16x8_t curr_index = vld1q_u16(indices);
  uint16x8_t max_index = curr_index;

  unsigned int idx = 8;

  // processing batch of 8
  // stop before idx reaches UNIT16_MAX
  for (; ((N - idx) >= 8) && ((idx + 8) <= UINT16_MAX); idx += 8) {
    float16x8_t values = vld1q_f16(&X[idx]);
    curr_index = vaddq_u16(curr_index, stride);

    // comparison
    uint16x8_t mask = vcgtq_f16(batch, values);

    // blend values and indices based on the mask
    batch = vbslq_f16(mask, batch, values);
    max_index = vbslq_u16(mask, max_index, curr_index);
  }

  // storing indices and max values
  __fp16 maxVal[8];
  vst1q_f16(maxVal, batch);
  vst1q_u16(indices, max_index);

  // getting the index of the maxima
  maxNum = maxVal[0];
  retIdx = indices[0];
  for (unsigned int i = 1; i < 8; i++) {
    if (maxVal[i] > maxNum) {
      maxNum = maxVal[i];
      retIdx = indices[i];
    }
  }

  // if idx is more than UNIT16_MAX
  if ((N > UINT16_MAX) && (N - idx) >= 4) {
    uint32_t indices_u32[] = {idx, idx + 1, idx + 2, idx + 3};
    uint32x4_t stride_u32 = vmovq_n_u32(4);
    float16x4_t batch_4 = vld1_f16(&X[0]);
    uint32x4_t curr_index_u32 = vld1q_u32(indices_u32);
    uint32x4_t max_index_u32 = curr_index_u32;

    idx += 4;
    // processing batch of 4
    for (; (N - idx) >= 4; idx += 4) {
      float16x4_t values_4 = vld1_f16(&X[idx]);
      curr_index_u32 = vaddq_u32(curr_index_u32, stride_u32);

      // comparison
      uint16x4_t mask_4 = vcgt_f16(batch_4, values_4);

      // converting to u32 mask as required by vbslq_u32
      uint32x4_t mask_4_u32 = vmovl_u16(mask_4);

      // blend values and indices based on the mask
      batch_4 = vbsl_f16(mask_4, batch_4, values_4);
      max_index_u32 = vbslq_u32(mask_4_u32, max_index_u32, curr_index_u32);
    }

    // storing indices and max values
    __fp16 maxVal_4[4];
    vst1_f16(maxVal_4, batch_4);
    vst1q_u32(indices_u32, max_index_u32);

    // getting the index of the maxima
    for (unsigned int i = 0; i < 4; i++) {
      if (maxVal_4[i] > maxNum) {
        maxNum = maxVal_4[i];
        retIdx = indices_u32[i];
      }
    }
  }

  // processing remaining values
  for (; idx < N; idx++) {
    if (X[idx] > maxNum) {
      maxNum = X[idx];
      retIdx = idx;
    }
  }

  return retIdx;
}

void custom_hgemm(const __fp16 *A, const __fp16 *B, __fp16 *C, uint32_t M,
                  uint32_t N, uint32_t K, float alpha, float beta, bool TransA,
                  bool TransB) {
  hgemm(A, B, C, M, N, K, alpha, beta, TransA, TransB);
}

void ele_mul(const unsigned int N, const __fp16 *X, const __fp16 *Y, __fp16 *Z,
             float alpha, float beta) {
  unsigned int i = 0;
  float16x8_t alpha_vec = vdupq_n_f16(alpha);
  float16x8_t beta_vec = vdupq_n_f16(beta);
  for (; N - i >= 8; i += 8) {
    float16x8_t x0_7 = vld1q_f16(&X[i]);
    float16x8_t y0_7 = vld1q_f16(&Y[i]);
    if (std::fpclassify(alpha - 1.F) != FP_ZERO) {
      y0_7 = vmulq_f16(y0_7, alpha_vec);
    }
    float16x8_t xy0_7 = vmulq_f16(x0_7, y0_7);
    if (std::abs(beta) > __FLT_MIN__) {
      float16x8_t z0_7 = vmulq_f16(vld1q_f16(&Z[i]), beta_vec);
      vst1q_f16(&Z[i], vaddq_f16(z0_7, xy0_7));
    } else {
      vst1q_f16(&Z[i], xy0_7);
    }
  }
  while (i < N) {
    if (std::abs(beta) > __FLT_MIN__)
      Z[i] = alpha * X[i] * Y[i] + beta * Z[i];
    else
      Z[i] = alpha * X[i] * Y[i];
    ++i;
  }
}

void ele_add(const unsigned int N, const __fp16 *X, const __fp16 *Y, __fp16 *Z,
             float alpha, float beta) {
  unsigned int i = 0;
  float16x8_t alpha_vec = vdupq_n_f16(alpha);
  float16x8_t beta_vec = vdupq_n_f16(beta);
  for (; N - i >= 8; i += 8) {
    float16x8_t x0_7 = vld1q_f16(&X[i]);
    float16x8_t y0_7 = vld1q_f16(&Y[i]);
    if (std::fpclassify(alpha - 1.F) != FP_ZERO) {
      y0_7 = vmulq_f16(y0_7, alpha_vec);
    }
    float16x8_t xy0_7 = vaddq_f16(x0_7, y0_7);
    if (std::abs(beta) > __FLT_MIN__) {
      float16x8_t z0_7 = vmulq_f16(vld1q_f16(&Z[i]), beta_vec);
      vst1q_f16(&Z[i], vaddq_f16(z0_7, xy0_7));
    } else {
      vst1q_f16(&Z[i], xy0_7);
    }
  }
  while (i < N) {
    if (std::abs(beta) > __FLT_MIN__)
      Z[i] = X[i] + alpha * Y[i] + beta * Z[i];
    else
      Z[i] = X[i] + alpha * Y[i];
    ++i;
  }
}

void ele_sub(const unsigned int N, const __fp16 *X, const __fp16 *Y, __fp16 *Z,
             float alpha, float beta) {
  unsigned int i = 0;
  float16x8_t alpha_vec = vdupq_n_f16(alpha);
  float16x8_t beta_vec = vdupq_n_f16(beta);
  for (; N - i >= 8; i += 8) {
    float16x8_t x0_7 = vld1q_f16(&X[i]);
    float16x8_t y0_7 = vld1q_f16(&Y[i]);
    if (std::fpclassify(alpha - 1.F) != FP_ZERO) {
      y0_7 = vmulq_f16(y0_7, alpha_vec);
    }
    float16x8_t xy0_7 = vsubq_f16(x0_7, y0_7);
    if (std::abs(beta) > __FLT_MIN__) {
      float16x8_t z0_7 = vmulq_f16(vld1q_f16(&Z[i]), beta_vec);
      vst1q_f16(&Z[i], vaddq_f16(z0_7, xy0_7));
    } else {
      vst1q_f16(&Z[i], xy0_7);
    }
  }
  while (i < N) {
    if (std::abs(beta) > __FLT_MIN__)
      Z[i] = X[i] - alpha * Y[i] + beta * Z[i];
    else
      Z[i] = X[i] - alpha * Y[i];
    ++i;
  }
}

void ele_div(const unsigned int N, const __fp16 *X, const __fp16 *Y, __fp16 *Z,
             float alpha, float beta) {
  unsigned int i = 0;
  float16x8_t alpha_vec = vdupq_n_f16(alpha);
  float16x8_t beta_vec = vdupq_n_f16(beta);
  for (; N - i >= 8; i += 8) {
    float16x8_t x0_7 = vld1q_f16(&X[i]);
    float16x8_t y0_7 = vld1q_f16(&Y[i]);
    if (std::fpclassify(alpha - 1.F) != FP_ZERO) {
      y0_7 = vmulq_f16(y0_7, alpha_vec);
    }
    float16x8_t xy0_7 = vdivq_f16(x0_7, y0_7);
    if (std::abs(beta) > __FLT_MIN__) {
      float16x8_t z0_7 = vmulq_f16(vld1q_f16(&Z[i]), beta_vec);
      vst1q_f16(&Z[i], vaddq_f16(z0_7, xy0_7));
    } else {
      vst1q_f16(&Z[i], xy0_7);
    }
  }
  while (i < N) {
    if (std::abs(beta) > __FLT_MIN__)
      Z[i] = X[i] / (alpha * Y[i]) + beta * Z[i];
    else
      Z[i] = X[i] / (alpha * Y[i]);
    ++i;
  }
}

void inv_sqrt_inplace(const unsigned int N, __fp16 *X) {
  unsigned int i = 0;
  for (; N - i >= 8; i += 8) {
    float16x8_t x0_7 = vld1q_f16(&X[i]);
    float16x8_t x0_7_sqrt = vsqrtq_f16(x0_7);
    float16x8_t ones = vmovq_n_f16(1);
    float16x8_t x0_7_sqrt_div = vdivq_f16(ones, x0_7_sqrt);
    vst1q_f16(&X[i], x0_7_sqrt_div);
  }
  while (i < N) {
    X[i] = (1 / std::sqrt(static_cast<float>(X[i])));
    ++i;
  }
}

void transpose_matrix(const unsigned int M, const unsigned int N,
                      const __fp16 *src, unsigned int ld_src, __fp16 *dst,
                      unsigned int ld_dst) {
  transpose_neon<__fp16>(M, N, src, ld_src, dst, ld_dst);
}

void compute_rotary_embedding_value(unsigned int dim, unsigned int half_,
                                    unsigned int w, __fp16 *in, __fp16 *out,
                                    float *cos_, float *sin_) {
  unsigned int k = 0;
  while (k < dim) {
    unsigned int span = w + k;

    if (k < half_) { // upper half
      if (half_ - k >= 8) {
        float16x8_t values0_7 = vld1q_f16(&in[span]);
        float16x8_t transformed_values0_7 =
          vmulq_n_f16(vld1q_f16(&in[span + half_]), -1);
        float32x4_t cos0_3 = vld1q_f32(&cos_[k]);
        float32x4_t cos4_7 = vld1q_f32(&cos_[k + 4]);
        float32x4_t sin0_3 = vld1q_f32(&sin_[k]);
        float32x4_t sin4_7 = vld1q_f32(&sin_[k + 4]);

        float32x4_t values0_3 = vaddq_f32(
          vmulq_f32(vcvt_f32_f16(vget_low_f16(values0_7)), cos0_3),
          vmulq_f32(vcvt_f32_f16(vget_low_f16(transformed_values0_7)), sin0_3));
        float32x4_t values4_7 = vaddq_f32(
          vmulq_f32(vcvt_f32_f16(vget_high_f16(values0_7)), cos4_7),
          vmulq_f32(vcvt_f32_f16(vget_high_f16(transformed_values0_7)),
                    sin4_7));

        vst1q_f16(&out[span], vcombine_f16(vcvt_f16_f32(values0_3),
                                           vcvt_f16_f32(values4_7)));

        k += 8;
      } else {
        float value = in[span];
        float transformed_value = -1 * in[span + half_];

        value = (value * cos_[k]) + (transformed_value * sin_[k]);

        out[span] = value;

        ++k;
      }
    } else { // lower half : k >= half_
      if (dim - k >= 8) {
        float16x8_t values0_7 = vld1q_f16(&in[span]);
        float16x8_t transformed_values0_7 = vld1q_f16(&in[span - half_]);
        float32x4_t cos0_3 = vld1q_f32(&cos_[k]);
        float32x4_t cos4_7 = vld1q_f32(&cos_[k + 4]);
        float32x4_t sin0_3 = vld1q_f32(&sin_[k]);
        float32x4_t sin4_7 = vld1q_f32(&sin_[k + 4]);

        float32x4_t values0_3 = vaddq_f32(
          vmulq_f32(vcvt_f32_f16(vget_low_f16(values0_7)), cos0_3),
          vmulq_f32(vcvt_f32_f16(vget_low_f16(transformed_values0_7)), sin0_3));
        float32x4_t values4_7 = vaddq_f32(
          vmulq_f32(vcvt_f32_f16(vget_high_f16(values0_7)), cos4_7),
          vmulq_f32(vcvt_f32_f16(vget_high_f16(transformed_values0_7)),
                    sin4_7));

        vst1q_f16(&out[span], vcombine_f16(vcvt_f16_f32(values0_3),
                                           vcvt_f16_f32(values4_7)));

        k += 8;
      } else {
        float value = in[span];
        float transformed_value = in[span - half_];

        value = (value * cos_[k]) + (transformed_value * sin_[k]);

        out[span] = value;

        ++k;
      }
    }
  }
}

void swiglu(const unsigned int N, __fp16 *X, __fp16 *Y, __fp16 *Z) {
  unsigned int i = 0;
  for (; N - i >= 8; i += 8) {
    float16x8_t y0_7 = vld1q_f16(&Y[i]);
    float16x8_t z0_7 = vld1q_f16(&Z[i]);
    float16x8_t y0_7_minus = vmulq_n_f16(y0_7, -1);

    float32x4_t exp0_3 = exp_ps(vcvt_f32_f16(vget_low_f16(y0_7_minus)));
    float32x4_t exp4_7 = exp_ps(vcvt_f32_f16(vget_high_f16(y0_7_minus)));

    float16x8_t exp0_7 =
      vcombine_f16(vcvt_f16_f32(exp0_3), vcvt_f16_f32(exp4_7));
    exp0_7 = vaddq_f16(exp0_7, vmovq_n_f16(1.f));
    exp0_7 = vdivq_f16(y0_7, exp0_7);
    exp0_7 = vmulq_f16(exp0_7, z0_7);

    vst1q_f16(&X[i], exp0_7);
  }
  while (i < N) {
    X[i] = (Y[i] / (1.f + std::exp(static_cast<float>(-Y[i])))) * Z[i];
    ++i;
  }
}

__fp16 max_val(const unsigned int N, __fp16 *X) {
  unsigned int i = 0;
  __fp16 ret = X[i];
  for (; N - i >= 8; i += 8) {
    float16x8_t x0_7 = vld1q_f16(&X[i]);
    __fp16 x_max = vmaxvq_f16(x0_7);
    ret = (ret > x_max) ? ret : x_max;
  }
  while (i < N) {
    ret = (ret > X[i]) ? ret : X[i];
    ++i;
  }
  return ret;
}

void softmax(const unsigned int N, __fp16 *X, __fp16 *Y) {
  unsigned int i = 0;
  float sum = 0.f;
  __fp16 max_x = max_val(N, X);
  float32x4_t max_x_v = vmovq_n_f32(static_cast<float>(max_x));

  for (; N - i >= 8; i += 8) {
    float16x8_t x0_7 = vld1q_f16(&X[i]);
    float32x4_t x0_3 = vcvt_f32_f16(vget_low_f16(x0_7));
    float32x4_t x4_7 = vcvt_f32_f16(vget_high_f16(x0_7));
    x0_3 = vsubq_f32(x0_3, max_x_v);
    x4_7 = vsubq_f32(x4_7, max_x_v);
    float32x4_t exp0_3 = exp_ps(x0_3);
    float32x4_t exp4_7 = exp_ps(x4_7);
    sum += vaddvq_f32(exp0_3);
    sum += vaddvq_f32(exp4_7);
  }
  while (i < N) {
    sum += std::exp(static_cast<float>(X[i] - max_x));
    ++i;
  }

  i = 0;
  float32x4_t sum_vec = vmovq_n_f32(sum);
  for (; N - i >= 8; i += 8) {
    float16x8_t x0_7 = vld1q_f16(&X[i]);
    float32x4_t x0_3 = vcvt_f32_f16(vget_low_f16(x0_7));
    float32x4_t x4_7 = vcvt_f32_f16(vget_high_f16(x0_7));
    x0_3 = vsubq_f32(x0_3, max_x_v);
    x4_7 = vsubq_f32(x4_7, max_x_v);
    float32x4_t exp0_3 = exp_ps(x0_3);
    float32x4_t exp4_7 = exp_ps(x4_7);
    float32x4_t softmax0_3 = vdivq_f32(exp0_3, sum_vec);
    float32x4_t softmax4_7 = vdivq_f32(exp4_7, sum_vec);
    vst1q_f16(&Y[i],
              vcombine_f16(vcvt_f16_f32(softmax0_3), vcvt_f16_f32(softmax4_7)));
  }
  while (i < N) {
    Y[i] = std::exp(static_cast<float>(X[i] - max_x)) / sum;
    ++i;
  }
}

// TODO: better implementation of exp_f16x8() if needed
inline static float16x8_t exp_f16x8(float16x8_t x) {
  float32x4_t x_low = vcvt_f32_f16(vget_low_f16(x));
  float32x4_t x_high = vcvt_f32_f16(vget_high_f16(x));
  float32x4_t res_low = exp_ps(x_low);
  float32x4_t res_high = exp_ps(x_high);
  return vcombine_f16(vcvt_f16_f32(res_low), vcvt_f16_f32(res_high));
}

// Static helper function for softmax_row_inplace with __fp16 sink
static void softmax_row_inplace_with_fp16_sink(__fp16 *qk_out, size_t start_row,
                                               size_t end_row, size_t num_heads,
                                               __fp16 *sink) {
  size_t row_range = end_row - start_row;
  const size_t full_blocks = (num_heads / 8) * 8;

  __fp16 *max_vals = new __fp16[num_heads];
  __fp16 *sum_vals = new __fp16[num_heads];

  // 1. max (including sink)
  for (size_t c = 0; c < num_heads; ++c) {
    __fp16 max_val = sink[c]; // Include sink in max calculation
    for (size_t r = start_row; r < end_row; ++r)
      max_val = std::max<__fp16>(max_val, qk_out[r * num_heads + c]);
    max_vals[c] = max_val;
  }

  // 2. inplace exp + sum (including sink)
  for (size_t c = 0; c < full_blocks; c += 8) {
    float16x8_t maxv = vld1q_f16(&max_vals[c]);
    float16x8_t sinkv = vld1q_f16(&sink[c]);
    float16x8_t sum = exp_f16x8(vsubq_f16(sinkv, maxv)); // Include sink in sum

    for (size_t r = 0; r < row_range; ++r) {
      __fp16 *ptr = &qk_out[(start_row + r) * num_heads + c];
      float16x8_t val = vld1q_f16(ptr);
      float16x8_t e = exp_f16x8(vsubq_f16(val, maxv));
      vst1q_f16(ptr, e); // overwrite qk_out
      sum = vaddq_f16(sum, e);
    }
    vst1q_f16(&sum_vals[c], sum);
  }

  for (size_t c = full_blocks; c < num_heads; ++c) {
    __fp16 maxv = max_vals[c];
    __fp16 sum = std::exp(sink[c] - maxv); // Include sink in sum

    for (size_t r = 0; r < row_range; ++r) {
      __fp16 &a = qk_out[(start_row + r) * num_heads + c];
      a = std::exp(a - maxv); // overwrite qk_out
      sum += a;
    }
    sum_vals[c] = sum;
  }

  // 3. softmax = exp / sum (inplace)
  for (size_t r = 0; r < row_range; ++r) {
    for (size_t c = 0; c < full_blocks; c += 8) {
      __fp16 *ptr = &qk_out[(start_row + r) * num_heads + c];
      float16x8_t val = vld1q_f16(ptr); // already exp(x - max)
      float16x8_t sumv = vld1q_f16(&sum_vals[c]);
      float16x8_t soft = vdivq_f16(val, sumv);
      vst1q_f16(ptr, soft);
    }
    for (size_t c = full_blocks; c < num_heads; ++c) {
      qk_out[(start_row + r) * num_heads + c] /= sum_vals[c];
    }
  }

  delete[] max_vals;
  delete[] sum_vals;
}

// Static helper function for softmax_row_inplace without sink
static void softmax_row_inplace_no_sink(__fp16 *qk_out, size_t start_row,
                                        size_t end_row, size_t num_heads) {
  size_t row_range = end_row - start_row;
  const size_t full_blocks = (num_heads / 8) * 8;
  // const size_t remainder = num_heads % 8;

  __fp16 *max_vals = new __fp16[num_heads];
  __fp16 *sum_vals = new __fp16[num_heads];

  // 1. max
  for (size_t c = 0; c < num_heads; ++c) {
    __fp16 max_val = -INFINITY;
    for (size_t r = start_row; r < end_row; ++r)
      max_val = std::max<__fp16>(max_val, qk_out[r * num_heads + c]);
    max_vals[c] = max_val;
  }

  // 2. inplace exp + sum
  for (size_t c = 0; c < full_blocks; c += 8) {
    float16x8_t maxv = vld1q_f16(&max_vals[c]);
    float16x8_t sum = vdupq_n_f16(0.0f);
    for (size_t r = 0; r < row_range; ++r) {
      __fp16 *ptr = &qk_out[(start_row + r) * num_heads + c];
      float16x8_t val = vld1q_f16(ptr);
      float16x8_t e = exp_f16x8(vsubq_f16(val, maxv));
      vst1q_f16(ptr, e); // overwrite qk_out
      sum = vaddq_f16(sum, e);
    }
    vst1q_f16(&sum_vals[c], sum);
  }

  for (size_t c = full_blocks; c < num_heads; ++c) {
    __fp16 sum = 0.0f;
    __fp16 maxv = max_vals[c];
    for (size_t r = 0; r < row_range; ++r) {
      __fp16 &a = qk_out[(start_row + r) * num_heads + c];
      a = std::exp(a - maxv); // overwrite qk_out
      sum += a;
    }
    sum_vals[c] = sum;
  }
  // 3. softmax = exp / sum (inplace)
  for (size_t r = 0; r < row_range; ++r) {
    for (size_t c = 0; c < full_blocks; c += 8) {
      __fp16 *ptr = &qk_out[(start_row + r) * num_heads + c];
      float16x8_t val = vld1q_f16(ptr); // already exp(x - max)
      float16x8_t sumv = vld1q_f16(&sum_vals[c]);
      float16x8_t soft = vdivq_f16(val, sumv);
      vst1q_f16(ptr, soft);
    }
    for (size_t c = full_blocks; c < num_heads; ++c) {
      qk_out[(start_row + r) * num_heads + c] /= sum_vals[c];
    }
  }

  delete[] max_vals;
  delete[] sum_vals;
}

template <>
void softmax_row_inplace(__fp16 *qk_out, size_t start_row, size_t end_row,
                         size_t num_heads, __fp16 *sink) {
  if (sink == nullptr) {
    return softmax_row_inplace_no_sink(qk_out, start_row, end_row, num_heads);
  } else {
    return softmax_row_inplace_with_fp16_sink(qk_out, start_row, end_row,
                                              num_heads, sink);
  }
}

// New function: softmax_row_inplace with __fp16 input and float sink
static void softmax_row_inplace_with_fp32_sink(__fp16 *qk_out, size_t start_row,
                                               size_t end_row, size_t num_heads,
                                               float *sink) {
  size_t row_range = end_row - start_row;
  const size_t full_blocks = (num_heads / 8) * 8;

  float *max_vals = new float[num_heads];
  float *sum_vals = new float[num_heads];

  // 1. max (including sink)
  for (size_t c = 0; c < num_heads; ++c) {
    float max_val = sink[c]; // Include sink in max calculation
    for (size_t r = start_row; r < end_row; ++r) {
      float val = static_cast<float>(qk_out[r * num_heads + c]);
      max_val = std::max(max_val, val);
    }
    max_vals[c] = max_val;
  }

  // 2. inplace exp + sum (including sink)
  for (size_t c = 0; c < full_blocks; c += 8) {
    float32x4_t maxv_low = vld1q_f32(&max_vals[c]);
    float32x4_t maxv_high = vld1q_f32(&max_vals[c + 4]);
    float32x4_t sinkv_low = vld1q_f32(&sink[c]);
    float32x4_t sinkv_high = vld1q_f32(&sink[c + 4]);

    // Calculate exp(sink - max) for sum initialization
    float32x4_t sum_low = exp_ps(vsubq_f32(sinkv_low, maxv_low));
    float32x4_t sum_high = exp_ps(vsubq_f32(sinkv_high, maxv_high));

    for (size_t r = 0; r < row_range; ++r) {
      __fp16 *ptr = &qk_out[(start_row + r) * num_heads + c];
      float16x8_t val_fp16 = vld1q_f16(ptr);

      // Convert to float32 for computation
      float32x4_t val_low = vcvt_f32_f16(vget_low_f16(val_fp16));
      float32x4_t val_high = vcvt_f32_f16(vget_high_f16(val_fp16));

      // Compute exp(val - max)
      float32x4_t e_low = exp_ps(vsubq_f32(val_low, maxv_low));
      float32x4_t e_high = exp_ps(vsubq_f32(val_high, maxv_high));

      // Convert back to fp16 and store
      float16x8_t e_fp16 =
        vcombine_f16(vcvt_f16_f32(e_low), vcvt_f16_f32(e_high));
      vst1q_f16(ptr, e_fp16);

      // Accumulate sum
      sum_low = vaddq_f32(sum_low, e_low);
      sum_high = vaddq_f32(sum_high, e_high);
    }

    vst1q_f32(&sum_vals[c], sum_low);
    vst1q_f32(&sum_vals[c + 4], sum_high);
  }

  for (size_t c = full_blocks; c < num_heads; ++c) {
    float maxv = max_vals[c];
    float sum = std::exp(sink[c] - maxv); // Include sink in sum

    for (size_t r = 0; r < row_range; ++r) {
      __fp16 &a = qk_out[(start_row + r) * num_heads + c];
      float val = static_cast<float>(a);
      float e = std::exp(val - maxv);
      a = static_cast<__fp16>(e); // overwrite qk_out with exp value
      sum += e;
    }
    sum_vals[c] = sum;
  }

  // 3. softmax = exp / sum (inplace)
  for (size_t r = 0; r < row_range; ++r) {
    for (size_t c = 0; c < full_blocks; c += 8) {
      __fp16 *ptr = &qk_out[(start_row + r) * num_heads + c];
      float16x8_t val_fp16 = vld1q_f16(ptr); // already exp(x - max)

      // Convert to float32 for division
      float32x4_t val_low = vcvt_f32_f16(vget_low_f16(val_fp16));
      float32x4_t val_high = vcvt_f32_f16(vget_high_f16(val_fp16));

      float32x4_t sumv_low = vld1q_f32(&sum_vals[c]);
      float32x4_t sumv_high = vld1q_f32(&sum_vals[c + 4]);

      float32x4_t soft_low = vdivq_f32(val_low, sumv_low);
      float32x4_t soft_high = vdivq_f32(val_high, sumv_high);

      // Convert back to fp16 and store
      float16x8_t soft_fp16 =
        vcombine_f16(vcvt_f16_f32(soft_low), vcvt_f16_f32(soft_high));
      vst1q_f16(ptr, soft_fp16);
    }
    for (size_t c = full_blocks; c < num_heads; ++c) {
      __fp16 &val = qk_out[(start_row + r) * num_heads + c];
      val = static_cast<__fp16>(static_cast<float>(val) / sum_vals[c]);
    }
  }

  delete[] max_vals;
  delete[] sum_vals;
}

// Overloaded function for __fp16 input with float sink
void softmax_row_inplace(__fp16 *qk_out, size_t start_row, size_t end_row,
                         size_t num_heads, float *sink) {
  if (sink == nullptr) {
    return softmax_row_inplace_no_sink(qk_out, start_row, end_row, num_heads);
  } else {
    return softmax_row_inplace_with_fp32_sink(qk_out, start_row, end_row,
                                              num_heads, sink);
  }
}

// Static helper function for softmax_row with __fp16 sink
static void softmax_row_with_fp16_sink(__fp16 *qk_out, size_t start_row,
                                       size_t end_row, size_t num_heads,
                                       __fp16 *sink) {
  const size_t full_block = (num_heads / 8) * 8;

  __fp16 *max_vals = new __fp16[num_heads];
  __fp16 *sum_vals = new __fp16[num_heads];

  // 1. Find Max along with col (including sink)
  for (size_t c = 0; c < num_heads; ++c) {
    __fp16 max_val = sink[c];
    for (size_t r = start_row; r < end_row; ++r) {
      max_val = std::max<__fp16>(max_val, qk_out[r * num_heads + c]);
    }
    max_vals[c] = max_val;
  }

  // 2. Compute sum along with col (exp vectorized, including sink)
  for (size_t c = 0; c < full_block; c += 8) {
    float16x8_t maxv = vld1q_f16(&max_vals[c]);
    float16x8_t sinkv = vld1q_f16(&sink[c]);
    float16x8_t sum = exp_f16x8(vsubq_f16(sinkv, maxv)); // Include sink in sum

    for (size_t r = start_row; r < end_row; ++r) {
      float16x8_t val = vld1q_f16(&qk_out[r * num_heads + c]);
      float16x8_t sub = vsubq_f16(val, maxv);
      float16x8_t e = exp_f16x8(sub);
      sum = vaddq_f16(sum, e);
    }
    vst1q_f16(&sum_vals[c], sum);
  }

  for (size_t c = full_block; c < num_heads; ++c) {
    float sum = std::exp(sink[c] - max_vals[c]); // Include sink in sum
    for (size_t r = start_row; r < end_row; ++r) {
      sum += std::exp(qk_out[r * num_heads + c] - max_vals[c]);
    }
    sum_vals[c] = sum;
  }

  // 3. apply softmax
  for (size_t r = start_row; r < end_row; ++r) {
    for (size_t c = 0; c < full_block; c += 8) {
      float16x8_t val = vld1q_f16(&qk_out[r * num_heads + c]);
      float16x8_t maxv = vld1q_f16(&max_vals[c]);
      float16x8_t sub = vsubq_f16(val, maxv);
      float16x8_t e = exp_f16x8(sub);
      float16x8_t sumv = vld1q_f16(&sum_vals[c]);
      float16x8_t softmax = vdivq_f16(e, sumv);
      vst1q_f16(&qk_out[r * num_heads + c], softmax);
    }
    for (size_t c = full_block; c < num_heads; ++c) {
      qk_out[r * num_heads + c] =
        std::exp(qk_out[r * num_heads + c] - max_vals[c]) / sum_vals[c];
    }
  }

  delete[] max_vals;
  delete[] sum_vals;
}

// Static helper function for softmax_row without sink
static void softmax_row_no_sink(__fp16 *qk_out, size_t start_row,
                                size_t end_row, size_t num_heads) {
  const size_t full_block = (num_heads / 8) * 8;

  __fp16 *max_vals = new __fp16[num_heads];
  __fp16 *sum_vals = new __fp16[num_heads];

  // 1. Find Max along with col
  for (size_t c = 0; c < num_heads; ++c) {
    __fp16 max_val = -INFINITY;
    for (size_t r = start_row; r < end_row; ++r) {
      max_val = std::max<__fp16>(max_val, qk_out[r * num_heads + c]);
    }
    max_vals[c] = max_val;
  }

  // 2. Compute sum along with col (exp vectorized)
  for (size_t c = 0; c < full_block; c += 8) {
    float16x8_t sum = vdupq_n_f16(0.0f);
    for (size_t r = start_row; r < end_row; ++r) {
      float16x8_t val = vld1q_f16(&qk_out[r * num_heads + c]);
      float16x8_t maxv = vld1q_f16(&max_vals[c]);
      float16x8_t sub = vsubq_f16(val, maxv);
      float16x8_t e = exp_f16x8(sub);
      sum = vaddq_f16(sum, e);
    }
    vst1q_f16(&sum_vals[c], sum);
  }

  for (size_t c = full_block; c < num_heads; ++c) {
    float sum = 0.0f;
    for (size_t r = start_row; r < end_row; ++r) {
      sum += std::exp(qk_out[r * num_heads + c] - max_vals[c]);
    }
    sum_vals[c] = sum;
  }

  // 3. apply softmax
  for (size_t r = start_row; r < end_row; ++r) {
    for (size_t c = 0; c < full_block; c += 8) {
      float16x8_t val = vld1q_f16(&qk_out[r * num_heads + c]);
      float16x8_t maxv = vld1q_f16(&max_vals[c]);
      float16x8_t sub = vsubq_f16(val, maxv);
      float16x8_t e = exp_f16x8(sub);
      float16x8_t sumv = vld1q_f16(&sum_vals[c]);
      float16x8_t softmax = vdivq_f16(e, sumv);
      vst1q_f16(&qk_out[r * num_heads + c], softmax);
    }
    for (size_t c = full_block; c < num_heads; ++c) {
      qk_out[r * num_heads + c] =
        std::exp(qk_out[r * num_heads + c] - max_vals[c]) / sum_vals[c];
    }
  }

  delete[] max_vals;
  delete[] sum_vals;
}

template <>
void softmax_row(__fp16 *qk_out, size_t start_row, size_t end_row,
                 size_t num_heads, __fp16 *sink) {
  if (sink == nullptr) {
    return softmax_row_no_sink(qk_out, start_row, end_row, num_heads);
  } else {
    return softmax_row_with_fp16_sink(qk_out, start_row, end_row, num_heads,
                                      sink);
  }
}

// Static helper function for softmax_row with float sink
static void softmax_row_with_fp32_sink(__fp16 *qk_out, size_t start_row,
                                       size_t end_row, size_t num_heads,
                                       float *sink) {
  const size_t full_block = (num_heads / 8) * 8;

  float *max_vals = new float[num_heads];
  float *sum_vals = new float[num_heads];

  // 1. Find Max along with col (including sink)
  for (size_t c = 0; c < num_heads; ++c) {
    float max_val = sink[c];
    for (size_t r = start_row; r < end_row; ++r) {
      float val = static_cast<float>(qk_out[r * num_heads + c]);
      max_val = std::max(max_val, val);
    }
    max_vals[c] = max_val;
  }

  // 2. Compute sum along with col (exp vectorized, including sink)
  for (size_t c = 0; c < full_block; c += 8) {
    float32x4_t maxv_low = vld1q_f32(&max_vals[c]);
    float32x4_t maxv_high = vld1q_f32(&max_vals[c + 4]);
    float32x4_t sinkv_low = vld1q_f32(&sink[c]);
    float32x4_t sinkv_high = vld1q_f32(&sink[c + 4]);

    // Calculate exp(sink - max) for sum initialization
    float32x4_t sum_low = exp_ps(vsubq_f32(sinkv_low, maxv_low));
    float32x4_t sum_high = exp_ps(vsubq_f32(sinkv_high, maxv_high));

    for (size_t r = start_row; r < end_row; ++r) {
      float16x8_t val_fp16 = vld1q_f16(&qk_out[r * num_heads + c]);

      // Convert to float32 for computation
      float32x4_t val_low = vcvt_f32_f16(vget_low_f16(val_fp16));
      float32x4_t val_high = vcvt_f32_f16(vget_high_f16(val_fp16));

      float32x4_t sub_low = vsubq_f32(val_low, maxv_low);
      float32x4_t sub_high = vsubq_f32(val_high, maxv_high);

      float32x4_t e_low = exp_ps(sub_low);
      float32x4_t e_high = exp_ps(sub_high);

      sum_low = vaddq_f32(sum_low, e_low);
      sum_high = vaddq_f32(sum_high, e_high);
    }

    vst1q_f32(&sum_vals[c], sum_low);
    vst1q_f32(&sum_vals[c + 4], sum_high);
  }

  for (size_t c = full_block; c < num_heads; ++c) {
    float maxv = max_vals[c];
    float sum = std::exp(sink[c] - maxv); // Include sink in sum
    for (size_t r = start_row; r < end_row; ++r) {
      float val = static_cast<float>(qk_out[r * num_heads + c]);
      sum += std::exp(val - maxv);
    }
    sum_vals[c] = sum;
  }

  // 3. apply softmax
  for (size_t r = start_row; r < end_row; ++r) {
    for (size_t c = 0; c < full_block; c += 8) {
      float16x8_t val_fp16 = vld1q_f16(&qk_out[r * num_heads + c]);

      // Convert to float32 for computation
      float32x4_t val_low = vcvt_f32_f16(vget_low_f16(val_fp16));
      float32x4_t val_high = vcvt_f32_f16(vget_high_f16(val_fp16));

      float32x4_t maxv_low = vld1q_f32(&max_vals[c]);
      float32x4_t maxv_high = vld1q_f32(&max_vals[c + 4]);

      float32x4_t sub_low = vsubq_f32(val_low, maxv_low);
      float32x4_t sub_high = vsubq_f32(val_high, maxv_high);

      float32x4_t e_low = exp_ps(sub_low);
      float32x4_t e_high = exp_ps(sub_high);

      float32x4_t sumv_low = vld1q_f32(&sum_vals[c]);
      float32x4_t sumv_high = vld1q_f32(&sum_vals[c + 4]);

      float32x4_t softmax_low = vdivq_f32(e_low, sumv_low);
      float32x4_t softmax_high = vdivq_f32(e_high, sumv_high);

      // Convert back to fp16 and store
      float16x8_t softmax_fp16 =
        vcombine_f16(vcvt_f16_f32(softmax_low), vcvt_f16_f32(softmax_high));
      vst1q_f16(&qk_out[r * num_heads + c], softmax_fp16);
    }
    for (size_t c = full_block; c < num_heads; ++c) {
      float val = static_cast<float>(qk_out[r * num_heads + c]);
      qk_out[r * num_heads + c] =
        static_cast<__fp16>(std::exp(val - max_vals[c]) / sum_vals[c]);
    }
  }

  delete[] max_vals;
  delete[] sum_vals;
}

// Overloaded function for softmax_row with __fp16 input and float sink
void softmax_row(__fp16 *qk_out, size_t start_row, size_t end_row,
                 size_t num_heads, float *sink) {
  if (sink == nullptr) {
    return softmax_row_no_sink(qk_out, start_row, end_row, num_heads);
  } else {
    return softmax_row_with_fp32_sink(qk_out, start_row, end_row, num_heads,
                                      sink);
  }
}

static inline void load_fp16_4_to_chunk(const __fp16 *src, float *dst,
                                        int chunk_size) {
  int i = 0;
  for (; i + 4 <= chunk_size; i += 4) {
    float16x4_t half = vld1_f16(src + i);
    float32x4_t f32 = vcvt_f32_f16(half);
    vst1q_f32(dst + i, f32);
  }
  for (; i < chunk_size; ++i) {
    dst[i] = src[i];
  }
}

void compute_fp16vcache_fp32_transposed(int row_num, const float *in,
                                        const __fp16 *vcache, float *output,
                                        int num_cache_head, int gqa_size,
                                        int head_dim,
                                        size_t local_window_size) {
  std::vector<float> tmp_fp32(head_dim);

  for (int n = 0; n < num_cache_head; ++n) {
    int num_blocks = head_dim / 4;
    int rem = head_dim % 4;

    std::vector<float32x4_t> sumVec(num_blocks * gqa_size, vdupq_n_f32(0.0f));
    std::vector<float> sumRem(gqa_size * rem, 0.0f);

    for (int j = row_num < local_window_size ? 0
                                             : row_num + 1 - local_window_size;
         j <= row_num; ++j) {
      const __fp16 *vptr = vcache + (j * num_cache_head + n) * head_dim;

      load_fp16_4_to_chunk(vptr, tmp_fp32.data(), head_dim);

      for (int h = 0; h < gqa_size; ++h) {
        float a_val = in[(row_num < local_window_size
                            ? j
                            : j - (row_num + 1 - local_window_size)) *
                           gqa_size * num_cache_head +
                         n * gqa_size + h];

        float32x4_t inVec = vdupq_n_f32(a_val);

        for (int b = 0; b < num_blocks; ++b) {
          float32x4_t bVec = vld1q_f32(&tmp_fp32[b * 4]);
          sumVec[h * num_blocks + b] =
            vfmaq_f32(sumVec[h * num_blocks + b], inVec, bVec);
        }

        float *remPtr = &sumRem.data()[h * rem];
        int base = num_blocks * 4;
        for (int r = 0; r < rem; ++r) {
          remPtr[r] += a_val * tmp_fp32[base + r];
        }
      }
    }

    for (int h = 0; h < gqa_size; ++h) {
      for (int b = 0; b < num_blocks; ++b) {
        int out_base = (n * gqa_size + h) * head_dim + b * 4;
        vst1q_f32(&output[out_base], sumVec[h * num_blocks + b]);
      }

      float *remPtr = &sumRem.data()[h * rem];
      int base = num_blocks * 4;
      for (int r = 0; r < rem; ++r) {
        int out_idx = (n * gqa_size + h) * head_dim + base + r;
        output[out_idx] = remPtr[r];
      }
    }
  }
}

void compute_fp16vcache_transposed(int row_num, const __fp16 *in,
                                   const __fp16 *vcache, __fp16 *output,
                                   int num_cache_head, int gqa_size,
                                   int head_dim, int chunk_size,
                                   size_t local_window_size) {
  int start_row =
    row_num < local_window_size ? 0 : row_num + 1 - local_window_size;

  int num_blocks = chunk_size / 8;
  int rem = chunk_size % 8;

  std::vector<float16x8_t> sumVec(num_blocks * gqa_size, vdupq_n_f16(0.0f));
  std::vector<__fp16> sumRem(gqa_size * rem, 0.0f);

  for (int j = start_row; j <= row_num; ++j) {
    const __fp16 *vptr = vcache + j * num_cache_head * head_dim;

    for (int h = 0; h < gqa_size; ++h) {
      __fp16 in_val = in[(row_num < local_window_size ? j : j - start_row) *
                           gqa_size * num_cache_head +
                         h];
      float16x8_t inVec = vdupq_n_f16(in_val);

      for (int b = 0; b < num_blocks; ++b) {
        float16x8_t bVec = vld1q_f16(&vptr[b * 8]);
        sumVec[h * num_blocks + b] =
          vfmaq_f16(sumVec[h * num_blocks + b], inVec, bVec);
      }

      __fp16 *remPtr = &sumRem.data()[h * rem];
      int base = num_blocks * 8;
      for (int r = 0; r < rem; ++r) {
        remPtr[r] += in_val * vptr[base + r];
      }
    }
  }

  for (int h = 0; h < gqa_size; ++h) {
    for (int b = 0; b < num_blocks; ++b) {
      vst1q_f16(&output[h * head_dim + b * 8], sumVec[h * num_blocks + b]);
    }

    __fp16 *remPtr = &sumRem.data()[h * rem];
    for (int r = 0; r < rem; ++r) {
      output[h * head_dim + num_blocks * 8 + r] = remPtr[r];
    }
  }
}

template <>
void compute_kcaches(const float *in, const __fp16 *kcache, float *output,
                     int num_rows, int num_cache_head, int head_dim,
                     int gqa_size, int tile_size, size_t local_window_size) {
  std::vector<float> tmp_fp32(head_dim);

  int start_row =
    num_rows < local_window_size ? 0 : num_rows - local_window_size;
  int row_cnt = num_rows < local_window_size ? num_rows : local_window_size;
  const int tile_count = (row_cnt + tile_size - 1) / tile_size;

  for (int n = 0; n < num_cache_head; ++n) {
    for (int t = 0; t < tile_count; ++t) {
      int row_tile_start = t * tile_size;
      int tile_rows = std::min(tile_size, row_cnt - row_tile_start);

      for (int g = 0; g < gqa_size; ++g) {
        const float *in_ptr = in + n * gqa_size * head_dim + g * head_dim;
        for (int t_row = 0; t_row < tile_rows; ++t_row) {
          int row = start_row + row_tile_start + t_row;
          if (t_row + 1 < tile_rows) {
            const __fp16 *next_kptr =
              kcache + ((row + 1) * num_cache_head + n) * head_dim;
            __builtin_prefetch(next_kptr, 0, 3); // Read, L1 cache
          }
          const __fp16 *kptr = kcache + (row * num_cache_head + n) * head_dim;

          load_fp16_4_to_chunk(kptr, tmp_fp32.data(), head_dim);

          const float *k_row = tmp_fp32.data();

          float sum = 0.0f;
          int i = 0;
          float32x4_t acc = vdupq_n_f32(0.0f);
          for (; i + 4 <= head_dim; i += 4) {
            float32x4_t va = vld1q_f32(in_ptr + i);
            float32x4_t vb = vld1q_f32(k_row + i);
            acc = vfmaq_f32(acc, va, vb);
          }

          acc = vpaddq_f32(acc, acc);
          acc = vpaddq_f32(acc, acc);
          sum += vgetq_lane_f32(acc, 0);

          for (; i < head_dim; ++i)
            sum += in_ptr[i] * k_row[i];

          output[(row - start_row) * num_cache_head * gqa_size + n * gqa_size +
                 g] = sum / sqrt((float)head_dim);
        }
      }
    }
  }
}

static inline __fp16 compute_kcaches_core(const __fp16 *in_ptr,
                                          const __fp16 *k_row, int head_dim) {
  int i = 0;
  float16x8_t acc = vdupq_n_f16(0.0);
  for (; i + 8 <= head_dim; i += 8) {
    float16x8_t va = vld1q_f16(in_ptr + i);
    float16x8_t vb = vld1q_f16(k_row + i);
    acc = vfmaq_f16(acc, va, vb);
  }

  acc = vpaddq_f16(acc, acc);
  acc = vpaddq_f16(acc, acc);
  acc = vpaddq_f16(acc, acc);
  __fp16 sum = vgetq_lane_f16(acc, 0);

  for (; i < head_dim; ++i)
    sum += in_ptr[i] * k_row[i];

  return sum / sqrt((float)head_dim);
}

void compute_kcaches(const __fp16 *in, const __fp16 *kcache, __fp16 *output,
                     int num_rows, int num_cache_head, int head_dim,
                     int gqa_size, int tile_off, int tile_size,
                     size_t local_window_size) {
  int start_row =
    num_rows < local_window_size ? 0 : num_rows - local_window_size;
  int row_cnt = num_rows < local_window_size ? num_rows : local_window_size;
  const int tile_count = (row_cnt + tile_size - 1) / tile_size;

  int row_tile_start = tile_off * tile_size;
  int tile_rows = std::min(tile_size, row_cnt - row_tile_start);

  for (int g = 0; g < gqa_size; ++g) {
    const __fp16 *in_ptr = in + g * head_dim;
    for (int t_row = 0; t_row < tile_rows; ++t_row) {
      int row = start_row + row_tile_start + t_row;
      if (t_row + 1 < tile_rows) {
        const __fp16 *next_kptr =
          kcache + (row + 1) * num_cache_head * head_dim;
        __builtin_prefetch(next_kptr, 0, 3); // Read, L1 cache
      }
      const __fp16 *k_row = kcache + row * num_cache_head * head_dim;

      output[(row - start_row) * num_cache_head * gqa_size + g] =
        compute_kcaches_core(in_ptr, k_row, head_dim);
    }
  }
}

void compute_rotary_emb_value(unsigned int width, unsigned int dim,
                              unsigned int half_, float *inout, void *output,
                              const float *cos_, const float *sin_,
                              bool only_convert_to_fp16) {
  enum class OutputType { FP16, FP32 };

  OutputType out_type = OutputType::FP32;
  if (output != nullptr)
    out_type = OutputType::FP16;

  for (unsigned int w = 0; w < width; w += dim) {
    unsigned int k = 0;
    for (; k + 3 < half_; k += 4) {
      unsigned int i0 = w + k;
      unsigned int i1 = w + k + half_;

      float32x4_t a = vld1q_f32(&inout[i0]);
      float32x4_t b = vld1q_f32(&inout[i1]);

      if (only_convert_to_fp16) {
        if (out_type == OutputType::FP16) {
          float16x4_t a_fp16 = vcvt_f16_f32(a);
          float16x4_t b_fp16 = vcvt_f16_f32(b);

          vst1_f16(static_cast<__fp16 *>(output) + i0, a_fp16);
          vst1_f16(static_cast<__fp16 *>(output) + i1, b_fp16);
        }

      } else {
        float32x4_t cos_v = vld1q_f32(&cos_[k]);
        float32x4_t sin_v = vld1q_f32(&sin_[k]);

        float32x4_t out0 = vsubq_f32(vmulq_f32(a, cos_v), vmulq_f32(b, sin_v));
        float32x4_t out1 = vaddq_f32(vmulq_f32(a, sin_v), vmulq_f32(b, cos_v));

        if (out_type == OutputType::FP16) {
          float16x4_t out0_fp16 = vcvt_f16_f32(out0);
          float16x4_t out1_fp16 = vcvt_f16_f32(out1);

          vst1_f16(static_cast<__fp16 *>(output) + i0, out0_fp16);
          vst1_f16(static_cast<__fp16 *>(output) + i1, out1_fp16);
        } else if (out_type == OutputType::FP32) {
          vst1q_f32(&inout[i0], out0);
          vst1q_f32(&inout[i1], out1);
        }
      }
    }

    for (; k < half_; ++k) {
      unsigned int i0 = w + k;
      unsigned int i1 = w + k + half_;
      // assert(i1 < width && "Scalar i1 overflow!");
      float a = inout[i0];
      float b = inout[i1];

      if (only_convert_to_fp16) {
        static_cast<__fp16 *>(output)[i0] = a;
        static_cast<__fp16 *>(output)[i1] = b;

      } else {

        float c = cos_[k];
        float s = sin_[k];

        float out0 = a * c - b * s;
        float out1 = a * s + b * c;

        if (out_type == OutputType::FP16) {
          static_cast<__fp16 *>(output)[i0] = out0;
          static_cast<__fp16 *>(output)[i1] = out1;
        } else if (out_type == OutputType::FP32) {
          inout[i0] = out0;
          inout[i1] = out1;
        }
      }
    }
  }
}

void compute_rotary_emb_value(unsigned int width, unsigned int dim,
                              unsigned int half_, __fp16 *inout, __fp16 *output,
                              const __fp16 *cos_, const __fp16 *sin_) {
  for (unsigned int w = 0; w < width; w += dim) {
    unsigned int k = 0;
    if (output != nullptr) {
      for (; k + 7 < half_; k += 8) {
        unsigned int i0 = w + k;
        unsigned int i1 = w + k + half_;

        float16x8_t a = vld1q_f16(&inout[i0]);
        float16x8_t b = vld1q_f16(&inout[i1]);

        float16x8_t cos_v = vld1q_f16(&cos_[k]);
        float16x8_t sin_v = vld1q_f16(&sin_[k]);

        float16x8_t out0 = vsubq_f16(vmulq_f16(a, cos_v), vmulq_f16(b, sin_v));
        float16x8_t out1 = vaddq_f16(vmulq_f16(a, sin_v), vmulq_f16(b, cos_v));

        vst1q_f16(&output[i0], out0);
        vst1q_f16(&output[i1], out1);
      }
    } else {
      for (; k + 7 < half_; k += 8) {
        unsigned int i0 = w + k;
        unsigned int i1 = w + k + half_;

        float16x8_t a = vld1q_f16(&inout[i0]);
        float16x8_t b = vld1q_f16(&inout[i1]);

        float16x8_t cos_v = vld1q_f16(&cos_[k]);
        float16x8_t sin_v = vld1q_f16(&sin_[k]);

        float16x8_t out0 = vsubq_f16(vmulq_f16(a, cos_v), vmulq_f16(b, sin_v));
        float16x8_t out1 = vaddq_f16(vmulq_f16(a, sin_v), vmulq_f16(b, cos_v));

        vst1q_f16(&inout[i0], out0);
        vst1q_f16(&inout[i1], out1);
      }
    }

    for (; k < half_; ++k) {
      unsigned int i0 = w + k;
      unsigned int i1 = w + k + half_;

      __fp16 a = inout[i0];
      __fp16 b = inout[i1];

      __fp16 c = cos_[k];
      __fp16 s = sin_[k];

      __fp16 out0 = a * c - b * s;
      __fp16 out1 = a * s + b * c;

      if (output != nullptr) {
        output[i0] = out0;
        output[i1] = out1;
      } else {
        inout[i0] = out0;
        inout[i1] = out1;
      }
    }
  }
}

static __fp16 hsumq_f16(float16x8_t v) {
  float16x4_t lo = vget_low_f16(v);
  float16x4_t hi = vget_high_f16(v);
  float16x4_t s4 = vadd_f16(lo, hi);
  float16x4_t s2 = vadd_f16(s4, vext_f16(s4, s4, 2));
  float16x4_t s1 = vadd_f16(s2, vext_f16(s2, s2, 1));
  return vget_lane_f16(s1, 0);
}

template <>
void rms_norm_wrt_width_fp16_intrinsic(const float *__restrict X,
                                       float *__restrict Y, size_t H, size_t W,
                                       float epsilon) {
  const float eps_h = (float)epsilon;

  for (size_t h = 0; h < H; ++h) {
    const float *rowX = X + h * W;
    float *rowY = Y + h * W;

    size_t i = 0;
    float16x8_t acc = vdupq_n_f16(0.F);

    for (; i + 8 <= W; i += 8) {
      float32x4_t f0 = vld1q_f32(rowX + i + 0);
      float32x4_t f1 = vld1q_f32(rowX + i + 4);
      float16x4_t h0 = vcvt_f16_f32(f0);
      float16x4_t h1 = vcvt_f16_f32(f1);
      float16x8_t h8 = vcombine_f16(h0, h1);
      acc = vfmaq_f16(acc, h8, h8);
    }

    if (i + 4 <= W) {
      float32x4_t f = vld1q_f32(rowX + i);
      float16x4_t h4 = vcvt_f16_f32(f);
      float16x4_t p4 = vmul_f16(h4, h4);
      acc = vaddq_f16(acc, vcombine_f16(p4, vdup_n_f16(0.F)));
      i += 4;
    }

    __fp16 sum_h = hsumq_f16(acc);
    for (; i < W; ++i) {
      float hx = (float)rowX[i];
      sum_h = sum_h + hx * hx;
    }

    float mean_single = sum_h / W;
    float scale_single = 1.F / std::sqrt(mean_single + eps_h);
    float16x8_t scale_v = vdupq_n_f16(scale_single);

    i = 0;
    for (; i + 8 <= W; i += 8) {
      float32x4_t f0 = vld1q_f32(rowX + i + 0);
      float32x4_t f1 = vld1q_f32(rowX + i + 4);
      float16x8_t xh = vcombine_f16(vcvt_f16_f32(f0), vcvt_f16_f32(f1));
      float16x8_t yh = vmulq_f16(xh, scale_v);

      vst1q_f32(rowY + i + 0, vcvt_f32_f16(vget_low_f16(yh)));
      vst1q_f32(rowY + i + 4, vcvt_f32_f16(vget_high_f16(yh)));
    }
    if (i + 4 <= W) {
      float32x4_t f = vld1q_f32(rowX + i);
      float16x4_t y4 = vmul_f16(vcvt_f16_f32(f), vget_low_f16(scale_v));
      vst1q_f32(rowY + i, vcvt_f32_f16(y4));
      i += 4;
    }
    for (; i < W; ++i) {
      rowY[i] = rowX[i] * scale_single;
    }
  }
}

static inline float16x8_t vbslq_f16_u16(uint16x8_t m, float16x8_t a,
                                        float16x8_t b) {
  return vbslq_f16(m, a, b);
}

// Constants as __fp16 (half). Precision is limited, but acceptable for approx
// trig.
static inline float16x8_t vdupq_n_f16_const(float c) {
  return vdupq_n_f16((__fp16)c);
}

#define H_CEPPHES_FOPI (1.27323954473516f) // 4 / pi
#define H_MINUS_DP1 (-0.78515625f)
#define H_MINUS_DP2 (-2.4187564849853515625e-4f)
#define H_MINUS_DP3 (-3.77489497744594108e-8f)
#define H_SINCOF_P0 (-1.9515295891E-4f)
#define H_SINCOF_P1 (+8.3321608736E-3f)
#define H_SINCOF_P2 (-1.6666654611E-1f)
#define H_COSCOF_P0 (+2.443315711809948E-005f)
#define H_COSCOF_P1 (-1.388731625493765E-003f)
#define H_COSCOF_P2 (+4.166664568298827E-002f)

static inline void sincos_ph(float16x8_t x, float16x8_t *ysin,
                             float16x8_t *ycos) {
  uint16x8_t sign_mask_sin = vcltq_f16(x, vdupq_n_f16(0));
  x = vabsq_f16(x);

  float16x8_t y = vmulq_f16(x, vdupq_n_f16_const(H_CEPPHES_FOPI));

  int16x8_t tmpi = vcvtq_s16_f16(y);
  float16x8_t tmpf = vcvtq_f16_s16(tmpi);
  uint16x8_t gt_mask = vcgtq_f16(tmpf, y);
  tmpi =
    vsubq_s16(tmpi, vreinterpretq_s16_u16(vandq_u16(gt_mask, vdupq_n_u16(1))));

  int16x8_t emm2_s16 = vaddq_s16(tmpi, vdupq_n_s16(1));
  emm2_s16 = vandq_s16(emm2_s16, vdupq_n_s16((int16_t)~1));
  uint16x8_t emm2 = vreinterpretq_u16_s16(emm2_s16);

  y = vcvtq_f16_s16(emm2_s16);

  float16x8_t xmm1 = vmulq_f16(y, vdupq_n_f16_const(H_MINUS_DP1));
  float16x8_t xmm2 = vmulq_f16(y, vdupq_n_f16_const(H_MINUS_DP2));
  float16x8_t xmm3 = vmulq_f16(y, vdupq_n_f16_const(H_MINUS_DP3));
  x = vaddq_f16(x, xmm1);
  x = vaddq_f16(x, xmm2);
  x = vaddq_f16(x, xmm3);

  uint16x8_t poly_mask = vtstq_u16(emm2, vdupq_n_u16(2));

  sign_mask_sin = veorq_u16(sign_mask_sin, vtstq_u16(emm2, vdupq_n_u16(4)));
  uint16x8_t sign_mask_cos = vtstq_u16(
    vreinterpretq_u16_s16(vsubq_s16(emm2_s16, vdupq_n_s16(2))), vdupq_n_u16(4));

  float16x8_t z = vmulq_f16(x, x);

  float16x8_t y1 = vmulq_f16(z, vdupq_n_f16_const(H_COSCOF_P0));
  float16x8_t y2 = vmulq_f16(z, vdupq_n_f16_const(H_SINCOF_P0));

  y1 = vaddq_f16(y1, vdupq_n_f16_const(H_COSCOF_P1));
  y2 = vaddq_f16(y2, vdupq_n_f16_const(H_SINCOF_P1));

  y1 = vmulq_f16(y1, z);
  y2 = vmulq_f16(y2, z);

  y1 = vaddq_f16(y1, vdupq_n_f16_const(H_COSCOF_P2));
  y2 = vaddq_f16(y2, vdupq_n_f16_const(H_SINCOF_P2));

  y1 = vmulq_f16(y1, z);
  y2 = vmulq_f16(y2, z);

  y1 = vmulq_f16(y1, z);
  y2 = vaddq_f16(y2, x);

  y1 = vsubq_f16(y1, vmulq_f16(z, vdupq_n_f16((__fp16)0.5f)));
  y1 = vaddq_f16(y1, vdupq_n_f16((__fp16)1.0f));

  float16x8_t ys = vbslq_f16_u16(poly_mask, y1, y2);
  float16x8_t yc = vbslq_f16_u16(poly_mask, y2, y1);

  *ysin = vbslq_f16_u16(sign_mask_sin, vnegq_f16(ys), ys);
  *ycos = vbslq_f16_u16(sign_mask_cos, yc, vnegq_f16(yc));
}

static inline float16x8_t sin_ph(float16x8_t x) {
  float16x8_t s, c;
  sincos_ph(x, &s, &c);
  return s;
}

static inline float16x8_t cos_ph(float16x8_t x) {
  float16x8_t s, c;
  sincos_ph(x, &s, &c);
  return c;
}

static inline float16x8x2_t sincosx2_ph(float16x8_t x) {
  float16x8_t s, c;
  float16x8x2_t sc;
  sincos_ph(x, &s, &c);
  sc.val[0] = s;
  sc.val[1] = c;
  return sc;
}

template <>
void sine(const unsigned int N, _FP16 *X, _FP16 *Y, float alpha, float beta) {
  unsigned int i = 0;
  for (; N - i >= 8; i += 8) {
    float16x8_t x0_3 = vld1q_f16(&X[i]);
    if (std::fpclassify(alpha - 1.F) != FP_ZERO)
      x0_3 = vmulq_n_f16(x0_3, alpha);
    float16x8_t sinx0_3 = sin_ph(x0_3);
    if (std::fpclassify(beta - 1.F) != FP_ZERO)
      sinx0_3 = vmulq_n_f16(sinx0_3, beta);
    vst1q_f16(&Y[i], sinx0_3);
  }
  while (i < N) {
    Y[i] = std::sin(alpha * X[i]) * beta;
    ++i;
  }
}

template <>
void cosine(const unsigned int N, _FP16 *X, _FP16 *Y, float alpha, float beta) {
  unsigned int i = 0;
  for (; N - i >= 8; i += 8) {
    float16x8_t x0_3 = vld1q_f16(&X[i]);
    if (std::fpclassify(alpha - 1.F) != FP_ZERO)
      x0_3 = vmulq_n_f16(x0_3, alpha);
    float16x8_t cosx0_3 = cos_ph(x0_3);
    if (std::fpclassify(beta - 1.F) != FP_ZERO)
      cosx0_3 = vmulq_n_f16(cosx0_3, beta);
    vst1q_f16(&Y[i], cosx0_3);
  }
  while (i < N) {
    Y[i] = std::cos(alpha * X[i]) * beta;
    ++i;
  }
}

inline void sinecosine(const unsigned int N, _FP16 *X, _FP16 *Ys, _FP16 *Yc,
                       float alpha, float beta) {
  unsigned int i = 0;
  for (; N - i >= 8; i += 8) {
    float16x8_t x0_3 = vld1q_f16(&X[i]);
    if (std::fpclassify(alpha - 1.F) != FP_ZERO) {
      x0_3 = vmulq_n_f16(x0_3, alpha);
    }
    float16x8x2_t sincosx0_3 = sincosx2_ph(x0_3);
    float16x8_t sin0_3 = sincosx0_3.val[0];
    float16x8_t cos0_3 = sincosx0_3.val[1];
    if (std::fpclassify(beta - 1.F) != FP_ZERO) {
      sin0_3 = vmulq_n_f16(sin0_3, beta);
      cos0_3 = vmulq_n_f16(cos0_3, beta);
    }
    vst1q_f16(&Ys[i], sin0_3);
    vst1q_f16(&Yc[i], cos0_3);
  }
  while (i < N) {
    Ys[i] = std::sin(alpha * X[i]) * beta;
    Yc[i] = std::cos(alpha * X[i]) * beta;
    ++i;
  }
}

template <>
void calc_trigonometric_vals_dup(unsigned int N_half, _FP16 *angle, _FP16 *cos_,
                                 _FP16 *sin_, unsigned int from,
                                 float attention_scaling) {
  sinecosine(N_half, angle, sin_, cos_, from, attention_scaling);

  unsigned int N = 2 * N_half;

  // Copy values to second half (duplicate)
  unsigned int i = N_half;
  unsigned int i_half = 0;

  for (; (N - i >= 8) && (N_half - i_half >= 8); i += 8, i_half += 8) {
    vst1q_f16(&cos_[i], vld1q_f16(&cos_[i_half]));
    vst1q_f16(&sin_[i], vld1q_f16(&sin_[i_half]));
  }
  while (i < N || i_half < N_half) {
    cos_[i] = cos_[i_half];
    sin_[i] = sin_[i_half];
    ++i;
    ++i_half;
  }
}
} // namespace nntrainer::neon
