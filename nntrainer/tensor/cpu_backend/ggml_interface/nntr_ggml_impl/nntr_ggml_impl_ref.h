// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (c) 2023-2024 The ggml authors
 *
 * Portions of this file are derived from llama.cpp
 * (https://github.com/ggml-org/llama.cpp), licensed under the MIT License.
 * Copyright (c) Contributors to llama.cpp
 *
 * @file   nntr_ggml_impl_ref.h
 * @date   20 November 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jaemin Shin <jaemin980311@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  Scalar reference kernels shared by every ggml backend (fallback,
 * x86, and NEON builds without the extension a SIMD kernel needs)
 */

#ifndef __NNTR_GGML_IMPL_REF_H__
#define __NNTR_GGML_IMPL_REF_H__

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <nntr_ggml_impl_common.h>

/// @todo make nntr_ggml_impl_fallback.cpp the always-built scalar backend
/// (own symbol names) that neon/avx call, and move these kernels there

/**
 * @brief Scalar Q4_0 (4x8 interleaved) x Q8_0 GEMV
 * @tparam T output type (float, or an fp16 type)
 */
template <typename T>
void nntr_gemv_q4_0_4x8_q8_0_ref(int n, T *__restrict s,
                                 const void *__restrict vx,
                                 const void *__restrict vy, int nc) {
  const int qk = Q8_0;
  const int nb = n / qk;
  const int ncols_interleaved = 4;
  const int blocklen = 8;

  assert(n % qk == 0);
  assert(nc % ncols_interleaved == 0);

  const block_q8_0 *a_ptr = (const block_q8_0 *)vy;
  for (int x = 0; x < nc / ncols_interleaved; x++) {
    const block_q4_0x4 *b_ptr = (const block_q4_0x4 *)vx + (x * nb);
    float sumf[4] = {0};
    for (int l = 0; l < nb; l++) {
      for (int k = 0; k < (qk / (2 * blocklen)); k++) {
        for (int j = 0; j < ncols_interleaved; j++) {
          int sumi = 0;
          for (int i = 0; i < blocklen; ++i) {
            const uint8_t q =
              b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
            const int v0 = (int8_t)(q << 4);
            const int v1 = (int8_t)(q & 0xF0);
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
      s[x * ncols_interleaved + j] = static_cast<T>(sumf[j]);
  }
}

/**
 * @brief Scalar Q4_0 (4x8 interleaved) x Q8_0 (4-row interleaved) GEMM
 * @tparam T output type (float, or an fp16 type)
 */
template <typename T>
void nntr_gemm_q4_0_4x8_q8_0_ref(int n, T *__restrict s, size_t bs,
                                 const void *__restrict vx,
                                 const void *__restrict vy, int nr, int nc) {
  const int qk = Q8_0;
  const int nb = n / qk;
  const int ncols_interleaved = 4;
  const int blocklen = 8;

  assert(n % qk == 0);
  assert(nr % 4 == 0);
  assert(nc % ncols_interleaved == 0);

  for (int y = 0; y < nr / 4; y++) {
    const block_q8_0x4 *a_ptr = (const block_q8_0x4 *)vy + (y * nb);
    for (int x = 0; x < nc / ncols_interleaved; x++) {
      const block_q4_0x4 *b_ptr = (const block_q4_0x4 *)vx + (x * nb);
      float sumf[4][4] = {{0}};
      for (int l = 0; l < nb; l++) {
        for (int k = 0; k < (qk / (2 * blocklen)); k++) {
          for (int m = 0; m < 4; m++) {
            for (int j = 0; j < ncols_interleaved; j++) {
              int sumi = 0;
              for (int i = 0; i < blocklen; ++i) {
                const uint8_t q =
                  b_ptr[l]
                    .qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                const int v0 = (int8_t)(q << 4);
                const int v1 = (int8_t)(q & 0xF0);
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
          s[(y * 4 + m) * bs + x * ncols_interleaved + j] =
            static_cast<T>(sumf[m][j]);
      }
    }
  }
}

#endif // __NNTR_GGML_IMPL_REF_H__
