// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (c) 2023-2024 The ggml authors
 *
 * Portions of this file are derived from llama.cpp
 * (https://github.com/ggml-org/llama.cpp), licensed under the MIT License.
 * Copyright (c) Contributors to llama.cpp
 *
 * Modified by Sungsik Kong, 2025: Adapted for CPU backend integration
 *
 * @file   nntr_ggml_impl_fallback.cpp
 * @date   9 December 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Fallback (scalar C) implementations of GGML functions
 *
 * This file provides portable C implementations that work on any platform.
 * For optimized SIMD implementations, use architecture-specific files:
 * - nntr_ggml_impl_neon.cpp for ARM NEON (armv8.2-a)
 * - nntr_ggml_impl_sve.cpp for ARM SVE (armv9.2-a)
 * - nntr_ggml_impl_avx.cpp for x86_64 AVX/AVX2
 */

#include <algorithm>
#include <assert.h>
#include <cstring>
#include <math.h>
#include <stddef.h>
#include <stdexcept>
#include <stdint.h>

#include <nntr_ggml_impl.h>
#include <nntr_ggml_impl_common.h>
#include <nntr_ggml_impl_utils.h>

void nntr_gemv_q4_0_4x8_q8_0(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc) {
  const int qk = Q8_0;
  const int nb = n / qk;
  const int ncols_interleaved = 4;
  const int blocklen = 8;

  assert(n % qk == 0);
  assert(nc % ncols_interleaved == 0);

  float sumf[4];
  int sumi;

  const block_q8_0 *a_ptr = (const block_q8_0 *)vy;
  for (int x = 0; x < nc / ncols_interleaved; x++) {
    const block_q4_0x4 *b_ptr = (const block_q4_0x4 *)vx + (x * nb);

    for (int j = 0; j < ncols_interleaved; j++)
      sumf[j] = 0.0;
    for (int l = 0; l < nb; l++) {
      for (int k = 0; k < (qk / (2 * blocklen)); k++) {
        for (int j = 0; j < ncols_interleaved; j++) {
          sumi = 0;
          for (int i = 0; i < blocklen; ++i) {
            const int v0 =
              (int8_t)(b_ptr[l].qs[k * ncols_interleaved * blocklen +
                                   j * blocklen + i]
                       << 4);
            const int v1 =
              (int8_t)(b_ptr[l].qs[k * ncols_interleaved * blocklen +
                                   j * blocklen + i] &
                       0xF0);
            sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) +
                     (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2])) >>
                    4;
          }
          sumf[j] += sumi * nntr_compute_fp16_to_fp32(b_ptr[l].d[j]) *
                     nntr_compute_fp16_to_fp32(a_ptr[l].d);
        }
      }
    }
    for (int j = 0; j < ncols_interleaved; j++)
      s[x * ncols_interleaved + j] = sumf[j];
  }
}

void nntr_gemm_q4_0_4x8_q8_0(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc) {
  const int qk = Q8_0;
  const int nb = n / qk;
  const int ncols_interleaved = 4;
  const int blocklen = 8;

  assert(n % qk == 0);
  assert(nr % 4 == 0);
  assert(nc % ncols_interleaved == 0);

  float sumf[4][4];
  int sumi;

  for (int y = 0; y < nr / 4; y++) {
    const block_q8_0x4 *a_ptr = (const block_q8_0x4 *)vy + (y * nb);
    for (int x = 0; x < nc / ncols_interleaved; x++) {
      const block_q4_0x4 *b_ptr = (const block_q4_0x4 *)vx + (x * nb);
      for (int m = 0; m < 4; m++) {
        for (int j = 0; j < ncols_interleaved; j++)
          sumf[m][j] = 0.0;
      }
      for (int l = 0; l < nb; l++) {
        for (int k = 0; k < (qk / (2 * blocklen)); k++) {
          for (int m = 0; m < 4; m++) {
            for (int j = 0; j < ncols_interleaved; j++) {
              sumi = 0;
              for (int i = 0; i < blocklen; ++i) {
                const int v0 =
                  (int8_t)(b_ptr[l].qs[k * ncols_interleaved * blocklen +
                                       j * blocklen + i]
                           << 4);
                const int v1 =
                  (int8_t)(b_ptr[l].qs[k * ncols_interleaved * blocklen +
                                       j * blocklen + i] &
                           0xF0);
                sumi +=
                  ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                   (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i +
                                     qk / 2 * 4])) >>
                  4;
              }
              sumf[m][j] += sumi * nntr_compute_fp16_to_fp32(b_ptr[l].d[j]) *
                            nntr_compute_fp16_to_fp32(a_ptr[l].d[m]);
            }
          }
        }
      }
      for (int m = 0; m < 4; m++) {
        for (int j = 0; j < ncols_interleaved; j++)
          s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
      }
    }
  }
}

void nntr_gemm_q4_0_8x8_q8_0(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc) {
  // NYI: Fallback implementation for 8x8 GEMM
  throw std::runtime_error("NYI: nntr_gemm_q4_0_8x8_q8_0 fallback");
}

void nntr_gemm_q4_K_8x8_q8_K(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc) {
  // NYI: Fallback implementation for q4_K GEMM
  throw std::runtime_error("NYI: nntr_gemm_q4_K_8x8_q8_K fallback");
}

void nntr_gemv_q4_0_8x8_q8_0(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc) {
  // NYI: Fallback implementation for 8x8 GEMV
  throw std::runtime_error("NYI: nntr_gemv_q4_0_8x8_q8_0 fallback");
}

void nntr_gemv_q4_K_8x8_q8_K(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc) {
  // NYI: Fallback implementation for q4_K GEMV
  throw std::runtime_error("NYI: nntr_gemv_q4_K_8x8_q8_K fallback");
}

void nntr_quantize_mat_q8_0_4x8(const float *__restrict x, void *__restrict vy,
                                int64_t k) {
  // NYI: Fallback quantization
  throw std::runtime_error("NYI: nntr_quantize_mat_q8_0_4x8 fallback");
}

void nntr_quantize_mat_q8_K_4x8(const float *__restrict x, void *__restrict vy,
                                int64_t k) {
  // NYI: Fallback quantization
  throw std::runtime_error("NYI: nntr_quantize_mat_q8_K_4x8 fallback");
}

void nntr_vec_dot_q6_K_q8_K(int n, float *__restrict s, size_t bs,
                            const void *__restrict vx, size_t bx,
                            const void *__restrict vy, size_t by, int nrc) {
  // NYI: Fallback vec_dot
  throw std::runtime_error("NYI: nntr_vec_dot_q6_K_q8_K fallback");
}

// Include common implementations from nntr_ggml_impl_quant.cpp and
// nntr_ggml_impl_fp16_fp32.cpp which are always compiled
