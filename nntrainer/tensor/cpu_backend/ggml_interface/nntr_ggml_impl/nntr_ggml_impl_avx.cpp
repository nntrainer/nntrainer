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
 * @file   nntr_ggml_impl_avx.cpp
 * @date   20 August 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author  Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Custom-implemented functions to support ggml functions for internal
 * uses in nntrainer
 */

#include <algorithm>
#include <assert.h>
#include <cstring>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <tensor_dim.h>

#include <nntr_ggml_impl.h>
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
  const int qk = QK8_0;
  const int nb = n / qk;
  const int ncols_interleaved = 8;
  const int blocklen = 8;

  assert(n % qk == 0);
  assert(nr % 4 == 0);
  assert(nc % ncols_interleaved == 0);

  {
    const block_q4_0x8 *b_ptr_start = (const block_q4_0x8 *)vx;
    const block_q8_0x4 *a_ptr_start = (const block_q8_0x4 *)vy;
    int64_t b_nb = n / QK4_0;
    int64_t y = 0;
    // Mask to mask out nibbles from packed bytes
    const __m256i m4b = _mm256_set1_epi8(0x0F);
    const __m128i loadMask =
      _mm_blend_epi32(_mm_setzero_si128(), _mm_set1_epi32(0xFFFFFFFF), 3);
    // Lookup table to convert signed nibbles to signed bytes
    __m256i signextendlut = _mm256_castsi128_si256(
      _mm_set_epi8(-1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0));
    signextendlut = _mm256_permute2f128_si256(signextendlut, signextendlut, 0);
    // Permute mask used for easier vector processing at later stages
    __m256i requiredOrder = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);
    int64_t xstart = 0;
    int anr = nr - nr % 16; // Used to align nr with boundary of 16

    // Take group of four block_q8_0x4 structures at each pass of the loop and
    // perform dot product operation

    for (; y < anr / 4; y += 4) {
      const block_q8_0x4 *a_ptrs[4];

      a_ptrs[0] = a_ptr_start + (y * nb);
      for (int i = 0; i < 3; ++i) {
        a_ptrs[i + 1] = a_ptrs[i] + nb;
      }

      // Take group of eight block_q4_0x8 structures at each pass of the loop
      // and perform dot product operation
      for (int64_t x = xstart; x < nc / 8; x++) {

        const block_q4_0x8 *b_ptr = b_ptr_start + (x * b_nb);

        // Master FP accumulators
        __m256 acc_rows[16];
        for (int i = 0; i < 16; i++) {
          acc_rows[i] = _mm256_setzero_ps();
        }

        for (int64_t b = 0; b < nb; b++) {
          // Load the eight block_q4_0 quantized values interleaved with each
          // other in chunks of eight - B0,B1 ....B6,B7
          const __m256i rhs_raw_mat_0123_0 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs));
          const __m256i rhs_raw_mat_4567_0 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 32));
          const __m256i rhs_raw_mat_0123_1 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 64));
          const __m256i rhs_raw_mat_4567_1 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 96));

          // Save the values in the following vectors in the formats B0B1B4B5,
          // B2B3B6B7 for further processing and storing of values
          const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(
            rhs_raw_mat_0123_0,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder),
            240);
          const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder),
            rhs_raw_mat_4567_0, 240);
          const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(
            rhs_raw_mat_0123_1,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder),
            240);
          const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder),
            rhs_raw_mat_4567_1, 240);

          // 4-bit -> 8-bit - Sign is maintained
          const __m256i rhs_mat_0145_0 = _mm256_shuffle_epi8(
            signextendlut,
            _mm256_and_si256(rhs_raw_mat_0145_0,
                             m4b)); // B0(0-7) B1(0-7) B4(0-7) B5(0-7)
          const __m256i rhs_mat_2367_0 = _mm256_shuffle_epi8(
            signextendlut,
            _mm256_and_si256(rhs_raw_mat_2367_0,
                             m4b)); // B2(0-7) B3(0-7) B6(0-7) B7(0-7)

          const __m256i rhs_mat_0145_1 = _mm256_shuffle_epi8(
            signextendlut,
            _mm256_and_si256(rhs_raw_mat_0145_1,
                             m4b)); // B0(8-15) B1(8-15) B4(8-15) B5(8-15)
          const __m256i rhs_mat_2367_1 = _mm256_shuffle_epi8(
            signextendlut,
            _mm256_and_si256(rhs_raw_mat_2367_1,
                             m4b)); // B2(8-15) B3(8-15) B6(8-15) B7(8-15)

          const __m256i rhs_mat_0145_2 = _mm256_shuffle_epi8(
            signextendlut,
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_0, 4),
                             m4b)); // B0(16-23) B1(16-23) B4(16-23) B5(16-23)
          const __m256i rhs_mat_2367_2 = _mm256_shuffle_epi8(
            signextendlut,
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_0, 4),
                             m4b)); // B2(16-23) B3(16-23) B6(16-23) B7(16-23)

          const __m256i rhs_mat_0145_3 = _mm256_shuffle_epi8(
            signextendlut,
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_1, 4),
                             m4b)); // B0(24-31) B1(24-31) B4(24-31) B5(24-31)
          const __m256i rhs_mat_2367_3 = _mm256_shuffle_epi8(
            signextendlut,
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_1, 4),
                             m4b)); // B2(24-31) B3(24-31) B6(24-31) B7(24-31)

          // Shuffle pattern one - right side input
          const __m256i rhs_mat_0145_0_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_0, 136); // B0(0-3) B1(0-3) B0(0-3) B1(0-3) B4(0-3)
                                  // B5(0-3) B4(0-3) B5(0-3)
          const __m256i rhs_mat_2367_0_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_0, 136); // B2(0-3) B3(0-3) B2(0-3) B3(0-3) B6(0-3)
                                  // B7(0-3) B6(0-3) B7(0-3)

          const __m256i rhs_mat_0145_1_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_1, 136); // B0(8-11) B1(8-11) B0(8-11) B1(8-11)
                                  // B4(8-11) B5(8-11) B4(8-11) B5(8-11)
          const __m256i rhs_mat_2367_1_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_1, 136); // B2(8-11) B3(8-11) B2(8-11) B3(8-11)
                                  // B6(8-11) B7(8-11) B6(8-11) B7(8-11)

          const __m256i rhs_mat_0145_2_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_2, 136); // B0(16-19) B1(16-19) B0(16-19) B1(16-19)
                                  // B4(16-19) B5(16-19) B4(16-19) B5(16-19)
          const __m256i rhs_mat_2367_2_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_2, 136); // B2(16-19) B3(16-19) B2(16-19) B3(16-19)
                                  // B6(16-19) B7(16-19) B6(16-19) B7(16-19)

          const __m256i rhs_mat_0145_3_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_3, 136); // B0(24-27) B1(24-27) B0(24-27) B1(24-27)
                                  // B4(24-27) B5(24-27) B4(24-27) B5(24-27)
          const __m256i rhs_mat_2367_3_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_3, 136); // B2(24-27) B3(24-27) B2(24-27) B3(24-27)
                                  // B6(24-27) B7(24-27) B6(24-27) B7(24-27)

          // Shuffle pattern two - right side input

          const __m256i rhs_mat_0145_0_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_0, 221); // B0(4-7) B1(4-7) B0(4-7) B1(4-7) B4(4-7)
                                  // B5(4-7) B4(4-7) B5(4-7)
          const __m256i rhs_mat_2367_0_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_0, 221); // B2(4-7) B3(4-7) B2(4-7) B3(4-7) B6(4-7)
                                  // B7(4-7) B6(4-7) B7(4-7)

          const __m256i rhs_mat_0145_1_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_1, 221); // B0(12-15) B1(12-15) B0(12-15) B1(12-15)
                                  // B4(12-15) B5(12-15) B4(12-15) B5(12-15)
          const __m256i rhs_mat_2367_1_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_1, 221); // B2(12-15) B3(12-15) B2(12-15) B3(12-15)
                                  // B6(12-15) B7(12-15) B6(12-15) B7(12-15)

          const __m256i rhs_mat_0145_2_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_2, 221); // B0(20-23) B1(20-23) B0(20-23) B1(20-23)
                                  // B4(20-23) B5(20-23) B4(20-23) B5(20-23)
          const __m256i rhs_mat_2367_2_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_2, 221); // B2(20-23) B3(20-23) B2(20-23) B3(20-23)
                                  // B6(20-23) B7(20-23) B6(20-23) B7(20-23)

          const __m256i rhs_mat_0145_3_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_3, 221); // B0(28-31) B1(28-31) B0(28-31) B1(28-31)
                                  // B4(28-31) B5(28-31) B4(28-31) B5(28-31)
          const __m256i rhs_mat_2367_3_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_3, 221); // B2(28-31) B3(28-31) B2(28-31) B3(28-31)
                                  // B6(28-31) B7(28-31) B6(28-31) B7(28-31)

          // Scale values - Load the wight scale values of block_q4_0x8
          const __m256 col_scale_f32 = GGML_F32Cx8_LOAD(b_ptr[b].d);

          // Process LHS in groups of four
          for (int rp = 0; rp < 4; rp++) {
            // Load the four block_q4_0 quantized values interleaved with each
            // other in chunks of eight - A0,A1,A2,A3 Loaded as set of 128 bit
            // vectors and repeated into a 256 bit vector
            __m256i lhs_mat_0123_0 =
              _mm256_loadu_si256((const __m256i *)((a_ptrs[rp][b].qs)));
            __m256i lhs_mat_01_0 =
              _mm256_permute2f128_si256(lhs_mat_0123_0, lhs_mat_0123_0, 0);
            __m256i lhs_mat_23_0 =
              _mm256_permute2f128_si256(lhs_mat_0123_0, lhs_mat_0123_0, 17);
            __m256i lhs_mat_0123_1 =
              _mm256_loadu_si256((const __m256i *)((a_ptrs[rp][b].qs + 32)));
            __m256i lhs_mat_01_1 =
              _mm256_permute2f128_si256(lhs_mat_0123_1, lhs_mat_0123_1, 0);
            __m256i lhs_mat_23_1 =
              _mm256_permute2f128_si256(lhs_mat_0123_1, lhs_mat_0123_1, 17);
            __m256i lhs_mat_0123_2 =
              _mm256_loadu_si256((const __m256i *)((a_ptrs[rp][b].qs + 64)));
            __m256i lhs_mat_01_2 =
              _mm256_permute2f128_si256(lhs_mat_0123_2, lhs_mat_0123_2, 0);
            __m256i lhs_mat_23_2 =
              _mm256_permute2f128_si256(lhs_mat_0123_2, lhs_mat_0123_2, 17);
            __m256i lhs_mat_0123_3 =
              _mm256_loadu_si256((const __m256i *)((a_ptrs[rp][b].qs + 96)));
            __m256i lhs_mat_01_3 =
              _mm256_permute2f128_si256(lhs_mat_0123_3, lhs_mat_0123_3, 0);
            __m256i lhs_mat_23_3 =
              _mm256_permute2f128_si256(lhs_mat_0123_3, lhs_mat_0123_3, 17);

            // Shuffle pattern one - left side input
            const __m256i lhs_mat_01_0_sp1 = _mm256_shuffle_epi32(
              lhs_mat_01_0, 160); // A0(0-3) A0(0-3) A1(0-3) A1(0-3) A0(0-3)
                                  // A0(0-3) A1(0-3) A1(0-3)
            const __m256i lhs_mat_23_0_sp1 = _mm256_shuffle_epi32(
              lhs_mat_23_0, 160); // A2(0-3) A2(0-3) A3(0-3) A3(0-3) A2(0-3)
                                  // A2(0-3) A3(0-3) A3(0-3)

            const __m256i lhs_mat_01_1_sp1 = _mm256_shuffle_epi32(
              lhs_mat_01_1, 160); // A0(8-11) A0(8-11) A1(8-11) A1(8-11)
                                  // A0(8-11) A0(8-11) A1(8-11) A1(8-11)
            const __m256i lhs_mat_23_1_sp1 = _mm256_shuffle_epi32(
              lhs_mat_23_1, 160); // A2(8-11) A2(8-11) A3(8-11) A3(8-11)
                                  // A2(8-11) A2(8-11) A3(8-11) A3(8-11)

            const __m256i lhs_mat_01_2_sp1 = _mm256_shuffle_epi32(
              lhs_mat_01_2, 160); // A0(16-19) A0(16-19) A1(16-19) A1(16-19)
                                  // A0(16-19) A0(16-19) A1(16-19) A1(16-19)
            const __m256i lhs_mat_23_2_sp1 = _mm256_shuffle_epi32(
              lhs_mat_23_2, 160); // A2(16-19) A2(16-19) A3(16-19) A3(16-19)
                                  // A2(16-19) A2(16-19) A3(16-19) A3(16-19)

            const __m256i lhs_mat_01_3_sp1 = _mm256_shuffle_epi32(
              lhs_mat_01_3, 160); // A0(24-27) A0(24-27) A1(24-27) A1(24-27)
                                  // A0(24-27) A0(24-27) A1(24-27) A1(24-27)
            const __m256i lhs_mat_23_3_sp1 = _mm256_shuffle_epi32(
              lhs_mat_23_3, 160); // A2(24-27) A2(24-27) A3(24-27) A3(24-27)
                                  // A2(24-27) A2(24-27) A3(24-27) A3(24-27)

            // Shuffle pattern two - left side input
            const __m256i lhs_mat_01_0_sp2 = _mm256_shuffle_epi32(
              lhs_mat_01_0, 245); // A0(4-7) A0(4-7) A1(4-7) A1(4-7) A0(4-7)
                                  // A0(4-7) A1(4-7) A1(4-7)
            const __m256i lhs_mat_23_0_sp2 = _mm256_shuffle_epi32(
              lhs_mat_23_0, 245); // A2(4-7) A2(4-7) A3(4-7) A3(4-7) A2(4-7)
                                  // A2(4-7) A3(4-7) A3(4-7)

            const __m256i lhs_mat_01_1_sp2 = _mm256_shuffle_epi32(
              lhs_mat_01_1, 245); // A0(12-15) A0(12-15) A1(12-15) A1(12-15)
                                  // A0(12-15) A0(12-15) A1(12-15) A1(12-15)
            const __m256i lhs_mat_23_1_sp2 = _mm256_shuffle_epi32(
              lhs_mat_23_1, 245); // A2(12-15) A2(12-15) A3(12-15) A3(12-15)
                                  // A2(12-15) A2(12-15) A3(12-15) A3(12-15)

            const __m256i lhs_mat_01_2_sp2 = _mm256_shuffle_epi32(
              lhs_mat_01_2, 245); // A0(20-23) A0(20-23) A1(20-23) A1(20-23)
                                  // A0(20-23) A0(20-23) A1(20-23) A1(20-23)
            const __m256i lhs_mat_23_2_sp2 = _mm256_shuffle_epi32(
              lhs_mat_23_2, 245); // A2(20-23) A2(20-23) A3(20-23) A3(20-23)
                                  // A2(20-23) A2(20-23) A3(20-23) A3(20-23)

            const __m256i lhs_mat_01_3_sp2 = _mm256_shuffle_epi32(
              lhs_mat_01_3, 245); // A0(28-31) A0(28-31) A1(28-31) A1(28-31)
                                  // A0(28-31) A0(28-31) A1(28-31) A1(28-31)
            const __m256i lhs_mat_23_3_sp2 = _mm256_shuffle_epi32(
              lhs_mat_23_3, 245); // A2(28-31) A2(28-31) A3(28-31) A3(28-31)
                                  // A2(28-31) A2(28-31) A3(28-31) A3(28-31)

            // The values arranged in shuffle patterns are operated with dot
            // product operation within 32 bit lane i.e corresponding bytes and
            // multiplied and added into 32 bit integers within 32 bit lane
            // Resembles MMLAs into 2x2 matrices in ARM Version
            const __m256i zero = _mm256_setzero_si256();
            __m256i iacc_mat_00_sp1 = mul_sum_i8_pairs_acc_int32x8(
              mul_sum_i8_pairs_acc_int32x8(
                mul_sum_i8_pairs_acc_int32x8(
                  mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_01_3_sp1,
                                               rhs_mat_0145_3_sp1),
                  lhs_mat_01_2_sp1, rhs_mat_0145_2_sp1),
                lhs_mat_01_1_sp1, rhs_mat_0145_1_sp1),
              lhs_mat_01_0_sp1, rhs_mat_0145_0_sp1);
            __m256i iacc_mat_01_sp1 = mul_sum_i8_pairs_acc_int32x8(
              mul_sum_i8_pairs_acc_int32x8(
                mul_sum_i8_pairs_acc_int32x8(
                  mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_01_3_sp1,
                                               rhs_mat_2367_3_sp1),
                  lhs_mat_01_2_sp1, rhs_mat_2367_2_sp1),
                lhs_mat_01_1_sp1, rhs_mat_2367_1_sp1),
              lhs_mat_01_0_sp1, rhs_mat_2367_0_sp1);
            __m256i iacc_mat_10_sp1 = mul_sum_i8_pairs_acc_int32x8(
              mul_sum_i8_pairs_acc_int32x8(
                mul_sum_i8_pairs_acc_int32x8(
                  mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_23_3_sp1,
                                               rhs_mat_0145_3_sp1),
                  lhs_mat_23_2_sp1, rhs_mat_0145_2_sp1),
                lhs_mat_23_1_sp1, rhs_mat_0145_1_sp1),
              lhs_mat_23_0_sp1, rhs_mat_0145_0_sp1);
            __m256i iacc_mat_11_sp1 = mul_sum_i8_pairs_acc_int32x8(
              mul_sum_i8_pairs_acc_int32x8(
                mul_sum_i8_pairs_acc_int32x8(
                  mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_23_3_sp1,
                                               rhs_mat_2367_3_sp1),
                  lhs_mat_23_2_sp1, rhs_mat_2367_2_sp1),
                lhs_mat_23_1_sp1, rhs_mat_2367_1_sp1),
              lhs_mat_23_0_sp1, rhs_mat_2367_0_sp1);
            __m256i iacc_mat_00_sp2 = mul_sum_i8_pairs_acc_int32x8(
              mul_sum_i8_pairs_acc_int32x8(
                mul_sum_i8_pairs_acc_int32x8(
                  mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_01_3_sp2,
                                               rhs_mat_0145_3_sp2),
                  lhs_mat_01_2_sp2, rhs_mat_0145_2_sp2),
                lhs_mat_01_1_sp2, rhs_mat_0145_1_sp2),
              lhs_mat_01_0_sp2, rhs_mat_0145_0_sp2);
            __m256i iacc_mat_01_sp2 = mul_sum_i8_pairs_acc_int32x8(
              mul_sum_i8_pairs_acc_int32x8(
                mul_sum_i8_pairs_acc_int32x8(
                  mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_01_3_sp2,
                                               rhs_mat_2367_3_sp2),
                  lhs_mat_01_2_sp2, rhs_mat_2367_2_sp2),
                lhs_mat_01_1_sp2, rhs_mat_2367_1_sp2),
              lhs_mat_01_0_sp2, rhs_mat_2367_0_sp2);
            __m256i iacc_mat_10_sp2 = mul_sum_i8_pairs_acc_int32x8(
              mul_sum_i8_pairs_acc_int32x8(
                mul_sum_i8_pairs_acc_int32x8(
                  mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_23_3_sp2,
                                               rhs_mat_0145_3_sp2),
                  lhs_mat_23_2_sp2, rhs_mat_0145_2_sp2),
                lhs_mat_23_1_sp2, rhs_mat_0145_1_sp2),
              lhs_mat_23_0_sp2, rhs_mat_0145_0_sp2);
            __m256i iacc_mat_11_sp2 = mul_sum_i8_pairs_acc_int32x8(
              mul_sum_i8_pairs_acc_int32x8(
                mul_sum_i8_pairs_acc_int32x8(
                  mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_23_3_sp2,
                                               rhs_mat_2367_3_sp2),
                  lhs_mat_23_2_sp2, rhs_mat_2367_2_sp2),
                lhs_mat_23_1_sp2, rhs_mat_2367_1_sp2),
              lhs_mat_23_0_sp2, rhs_mat_2367_0_sp2);

            // Output of both shuffle patterns are added in order to sum dot
            // product outputs of all 32 values in block
            __m256i iacc_mat_00 =
              _mm256_add_epi32(iacc_mat_00_sp1, iacc_mat_00_sp2);
            __m256i iacc_mat_01 =
              _mm256_add_epi32(iacc_mat_01_sp1, iacc_mat_01_sp2);
            __m256i iacc_mat_10 =
              _mm256_add_epi32(iacc_mat_10_sp1, iacc_mat_10_sp2);
            __m256i iacc_mat_11 =
              _mm256_add_epi32(iacc_mat_11_sp1, iacc_mat_11_sp2);

            // Straighten out to make 4 row vectors
            __m256i iacc_row_0 = _mm256_blend_epi32(
              iacc_mat_00, _mm256_shuffle_epi32(iacc_mat_01, 78), 204);
            __m256i iacc_row_1 = _mm256_blend_epi32(
              _mm256_shuffle_epi32(iacc_mat_00, 78), iacc_mat_01, 204);
            __m256i iacc_row_2 = _mm256_blend_epi32(
              iacc_mat_10, _mm256_shuffle_epi32(iacc_mat_11, 78), 204);
            __m256i iacc_row_3 = _mm256_blend_epi32(
              _mm256_shuffle_epi32(iacc_mat_10, 78), iacc_mat_11, 204);

            // Load the scale(d) values for all the 4 Q8_0 blocks and repeat it
            // across lanes
            const __m256 row_scale_f32 =
              GGML_F32Cx8_REPEAT_LOAD(a_ptrs[rp][b].d, loadMask);

            // Multiply with appropiate scales and accumulate
            acc_rows[rp * 4] = _mm256_fmadd_ps(
              _mm256_cvtepi32_ps(iacc_row_0),
              _mm256_mul_ps(col_scale_f32,
                            _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)),
              acc_rows[rp * 4]);
            acc_rows[rp * 4 + 1] = _mm256_fmadd_ps(
              _mm256_cvtepi32_ps(iacc_row_1),
              _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(
                                             row_scale_f32, row_scale_f32, 85)),
              acc_rows[rp * 4 + 1]);
            acc_rows[rp * 4 + 2] = _mm256_fmadd_ps(
              _mm256_cvtepi32_ps(iacc_row_2),
              _mm256_mul_ps(
                col_scale_f32,
                _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)),
              acc_rows[rp * 4 + 2]);
            acc_rows[rp * 4 + 3] = _mm256_fmadd_ps(
              _mm256_cvtepi32_ps(iacc_row_3),
              _mm256_mul_ps(
                col_scale_f32,
                _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 255)),
              acc_rows[rp * 4 + 3]);
          }
        }

        // Store the accumulated values
        for (int i = 0; i < 16; i++) {
          _mm256_storeu_ps((float *)(s + ((y * 4 + i) * bs + x * 8)),
                           acc_rows[i]);
        }
      }
    }

    // Take a block_q8_0x4 structures at each pass of the loop and perform dot
    // product operation
    for (; y < nr / 4; y++) {

      const block_q8_0x4 *a_ptr = a_ptr_start + (y * nb);

      // Load the eight block_q4_0 quantized values interleaved with each other
      // in chunks of eight - B0,B1 ....B6,B7
      for (int64_t x = xstart; x < nc / 8; x++) {

        const block_q4_0x8 *b_ptr = b_ptr_start + (x * b_nb);

        // Master FP accumulators
        __m256 acc_rows[4];
        for (int i = 0; i < 4; i++) {
          acc_rows[i] = _mm256_setzero_ps();
        }

        for (int64_t b = 0; b < nb; b++) {
          // Load the eight block_q8_0 quantized values interleaved with each
          // other in chunks of eight - B0,B1 ....B6,B7
          const __m256i rhs_raw_mat_0123_0 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs));
          const __m256i rhs_raw_mat_4567_0 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 32));
          const __m256i rhs_raw_mat_0123_1 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 64));
          const __m256i rhs_raw_mat_4567_1 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 96));

          // Save the values in the following vectors in the formats B0B1B4B5,
          // B2B3B6B7 for further processing and storing of valuess
          const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(
            rhs_raw_mat_0123_0,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder),
            240);
          const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder),
            rhs_raw_mat_4567_0, 240);
          const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(
            rhs_raw_mat_0123_1,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder),
            240);
          const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder),
            rhs_raw_mat_4567_1, 240);

          // 4-bit -> 8-bit - Sign is maintained
          const __m256i rhs_mat_0145_0 = _mm256_shuffle_epi8(
            signextendlut,
            _mm256_and_si256(rhs_raw_mat_0145_0,
                             m4b)); // B0(0-7) B1(0-7) B4(0-7) B5(0-7)
          const __m256i rhs_mat_2367_0 = _mm256_shuffle_epi8(
            signextendlut,
            _mm256_and_si256(rhs_raw_mat_2367_0,
                             m4b)); // B2(0-7) B3(0-7) B6(0-7) B7(0-7)

          const __m256i rhs_mat_0145_1 = _mm256_shuffle_epi8(
            signextendlut,
            _mm256_and_si256(rhs_raw_mat_0145_1,
                             m4b)); // B0(8-15) B1(8-15) B4(8-15) B5(8-15)
          const __m256i rhs_mat_2367_1 = _mm256_shuffle_epi8(
            signextendlut,
            _mm256_and_si256(rhs_raw_mat_2367_1,
                             m4b)); // B2(8-15) B3(8-15) B6(8-15) B7(8-15)

          const __m256i rhs_mat_0145_2 = _mm256_shuffle_epi8(
            signextendlut,
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_0, 4),
                             m4b)); // B0(16-23) B1(16-23) B4(16-23) B5(16-23)
          const __m256i rhs_mat_2367_2 = _mm256_shuffle_epi8(
            signextendlut,
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_0, 4),
                             m4b)); // B2(16-23) B3(16-23) B6(16-23) B7(16-23)

          const __m256i rhs_mat_0145_3 = _mm256_shuffle_epi8(
            signextendlut,
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_1, 4),
                             m4b)); // B0(24-31) B1(24-31) B4(24-31) B5(24-31)
          const __m256i rhs_mat_2367_3 = _mm256_shuffle_epi8(
            signextendlut,
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_1, 4),
                             m4b)); // B2(24-31) B3(24-31) B6(24-31) B7(24-31)

          // Shuffle pattern one - right side input
          const __m256i rhs_mat_0145_0_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_0, 136); // B0(0-3) B1(0-3) B0(0-3) B1(0-3) B4(0-3)
                                  // B5(0-3) B4(0-3) B5(0-3)
          const __m256i rhs_mat_2367_0_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_0, 136); // B2(0-3) B3(0-3) B2(0-3) B3(0-3) B6(0-3)
                                  // B7(0-3) B6(0-3) B7(0-3)

          const __m256i rhs_mat_0145_1_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_1, 136); // B0(8-11) B1(8-11) B0(8-11) B1(8-11)
                                  // B4(8-11) B5(8-11) B4(8-11) B5(8-11)
          const __m256i rhs_mat_2367_1_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_1, 136); // B2(8-11) B3(8-11) B2(8-11) B3(8-11)
                                  // B6(8-11) B7(8-11) B6(8-11) B7(8-11)

          const __m256i rhs_mat_0145_2_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_2, 136); // B0(16-19) B1(16-19) B0(16-19) B1(16-19)
                                  // B4(16-19) B5(16-19) B4(16-19) B5(16-19)
          const __m256i rhs_mat_2367_2_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_2, 136); // B2(16-19) B3(16-19) B2(16-19) B3(16-19)
                                  // B6(16-19) B7(16-19) B6(16-19) B7(16-19)

          const __m256i rhs_mat_0145_3_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_3, 136); // B0(24-27) B1(24-27) B0(24-27) B1(24-27)
                                  // B4(24-27) B5(24-27) B4(24-27) B5(24-27)
          const __m256i rhs_mat_2367_3_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_3, 136); // B2(24-27) B3(24-27) B2(24-27) B3(24-27)
                                  // B6(24-27) B7(24-27) B6(24-27) B7(24-27)

          // Shuffle pattern two - right side input

          const __m256i rhs_mat_0145_0_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_0, 221); // B0(4-7) B1(4-7) B0(4-7) B1(4-7) B4(4-7)
                                  // B5(4-7) B4(4-7) B5(4-7)
          const __m256i rhs_mat_2367_0_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_0, 221); // B2(4-7) B3(4-7) B2(4-7) B3(4-7) B6(4-7)
                                  // B7(4-7) B6(4-7) B7(4-7)

          const __m256i rhs_mat_0145_1_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_1, 221); // B0(12-15) B1(12-15) B0(12-15) B1(12-15)
                                  // B4(12-15) B5(12-15) B4(12-15) B5(12-15)
          const __m256i rhs_mat_2367_1_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_1, 221); // B2(12-15) B3(12-15) B2(12-15) B3(12-15)
                                  // B6(12-15) B7(12-15) B6(12-15) B7(12-15)

          const __m256i rhs_mat_0145_2_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_2, 221); // B0(20-23) B1(20-23) B0(20-23) B1(20-23)
                                  // B4(20-23) B5(20-23) B4(20-23) B5(20-23)
          const __m256i rhs_mat_2367_2_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_2, 221); // B2(20-23) B3(20-23) B2(20-23) B3(20-23)
                                  // B6(20-23) B7(20-23) B6(20-23) B7(20-23)

          const __m256i rhs_mat_0145_3_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_3, 221); // B0(28-31) B1(28-31) B0(28-31) B1(28-31)
                                  // B4(28-31) B5(28-31) B4(28-31) B5(28-31)
          const __m256i rhs_mat_2367_3_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_3, 221); // B2(28-31) B3(28-31) B2(28-31) B3(28-31)
                                  // B6(28-31) B7(28-31) B6(28-31) B7(28-31)

          // Scale values - Load the wight scale values of block_q4_0x8
          const __m256 col_scale_f32 = GGML_F32Cx8_LOAD(b_ptr[b].d);

          // Load the four block_q4_0 quantized values interleaved with each
          // other in chunks of eight - A0,A1,A2,A3 Loaded as set of 128 bit
          // vectors and repeated into a 256 bit vector
          __m256i lhs_mat_0123_0 =
            _mm256_loadu_si256((const __m256i *)((a_ptr[b].qs)));
          __m256i lhs_mat_01_0 =
            _mm256_permute2f128_si256(lhs_mat_0123_0, lhs_mat_0123_0, 0);
          __m256i lhs_mat_23_0 =
            _mm256_permute2f128_si256(lhs_mat_0123_0, lhs_mat_0123_0, 17);
          __m256i lhs_mat_0123_1 =
            _mm256_loadu_si256((const __m256i *)((a_ptr[b].qs + 32)));
          __m256i lhs_mat_01_1 =
            _mm256_permute2f128_si256(lhs_mat_0123_1, lhs_mat_0123_1, 0);
          __m256i lhs_mat_23_1 =
            _mm256_permute2f128_si256(lhs_mat_0123_1, lhs_mat_0123_1, 17);
          __m256i lhs_mat_0123_2 =
            _mm256_loadu_si256((const __m256i *)((a_ptr[b].qs + 64)));
          __m256i lhs_mat_01_2 =
            _mm256_permute2f128_si256(lhs_mat_0123_2, lhs_mat_0123_2, 0);
          __m256i lhs_mat_23_2 =
            _mm256_permute2f128_si256(lhs_mat_0123_2, lhs_mat_0123_2, 17);
          __m256i lhs_mat_0123_3 =
            _mm256_loadu_si256((const __m256i *)((a_ptr[b].qs + 96)));
          __m256i lhs_mat_01_3 =
            _mm256_permute2f128_si256(lhs_mat_0123_3, lhs_mat_0123_3, 0);
          __m256i lhs_mat_23_3 =
            _mm256_permute2f128_si256(lhs_mat_0123_3, lhs_mat_0123_3, 17);

          // Shuffle pattern one - left side input

          const __m256i lhs_mat_01_0_sp1 = _mm256_shuffle_epi32(
            lhs_mat_01_0, 160); // A0(0-3) A0(0-3) A1(0-3) A1(0-3) A0(0-3)
                                // A0(0-3) A1(0-3) A1(0-3)
          const __m256i lhs_mat_23_0_sp1 = _mm256_shuffle_epi32(
            lhs_mat_23_0, 160); // A2(0-3) A2(0-3) A3(0-3) A3(0-3) A2(0-3)
                                // A2(0-3) A3(0-3) A3(0-3)

          const __m256i lhs_mat_01_1_sp1 = _mm256_shuffle_epi32(
            lhs_mat_01_1, 160); // A0(8-11) A0(8-11) A1(8-11) A1(8-11) A0(8-11)
                                // A0(8-11) A1(8-11) A1(8-11)
          const __m256i lhs_mat_23_1_sp1 = _mm256_shuffle_epi32(
            lhs_mat_23_1, 160); // A2(8-11) A2(8-11) A3(8-11) A3(8-11) A2(8-11)
                                // A2(8-11) A3(8-11) A3(8-11)

          const __m256i lhs_mat_01_2_sp1 = _mm256_shuffle_epi32(
            lhs_mat_01_2, 160); // A0(16-19) A0(16-19) A1(16-19) A1(16-19)
                                // A0(16-19) A0(16-19) A1(16-19) A1(16-19)
          const __m256i lhs_mat_23_2_sp1 = _mm256_shuffle_epi32(
            lhs_mat_23_2, 160); // A2(16-19) A2(16-19) A3(16-19) A3(16-19)
                                // A2(16-19) A2(16-19) A3(16-19) A3(16-19)

          const __m256i lhs_mat_01_3_sp1 = _mm256_shuffle_epi32(
            lhs_mat_01_3, 160); // A0(24-27) A0(24-27) A1(24-27) A1(24-27)
                                // A0(24-27) A0(24-27) A1(24-27) A1(24-27)
          const __m256i lhs_mat_23_3_sp1 = _mm256_shuffle_epi32(
            lhs_mat_23_3, 160); // A2(24-27) A2(24-27) A3(24-27) A3(24-27)
                                // A2(24-27) A2(24-27) A3(24-27) A3(24-27)

          // Shuffle pattern two - left side input

          const __m256i lhs_mat_01_0_sp2 = _mm256_shuffle_epi32(
            lhs_mat_01_0, 245); // A0(4-7) A0(4-7) A1(4-7) A1(4-7) A0(4-7)
                                // A0(4-7) A1(4-7) A1(4-7)
          const __m256i lhs_mat_23_0_sp2 = _mm256_shuffle_epi32(
            lhs_mat_23_0, 245); // A2(4-7) A2(4-7) A3(4-7) A3(4-7) A2(4-7)
                                // A2(4-7) A3(4-7) A3(4-7)

          const __m256i lhs_mat_01_1_sp2 = _mm256_shuffle_epi32(
            lhs_mat_01_1, 245); // A0(12-15) A0(12-15) A1(12-15) A1(12-15)
                                // A0(12-15) A0(12-15) A1(12-15) A1(12-15)
          const __m256i lhs_mat_23_1_sp2 = _mm256_shuffle_epi32(
            lhs_mat_23_1, 245); // A2(12-15) A2(12-15) A3(12-15) A3(12-15)
                                // A2(12-15) A2(12-15) A3(12-15) A3(12-15)

          const __m256i lhs_mat_01_2_sp2 = _mm256_shuffle_epi32(
            lhs_mat_01_2, 245); // A0(20-23) A0(20-23) A1(20-23) A1(20-23)
                                // A0(20-23) A0(20-23) A1(20-23) A1(20-23)
          const __m256i lhs_mat_23_2_sp2 = _mm256_shuffle_epi32(
            lhs_mat_23_2, 245); // A2(20-23) A2(20-23) A3(20-23) A3(20-23)
                                // A2(20-23) A2(20-23) A3(20-23) A3(20-23)

          const __m256i lhs_mat_01_3_sp2 = _mm256_shuffle_epi32(
            lhs_mat_01_3, 245); // A0(28-31) A0(28-31) A1(28-31) A1(28-31)
                                // A0(28-31) A0(28-31) A1(28-31) A1(28-31)
          const __m256i lhs_mat_23_3_sp2 = _mm256_shuffle_epi32(
            lhs_mat_23_3, 245); // A2(28-31) A2(28-31) A3(28-31) A3(28-31)
                                // A2(28-31) A2(28-31) A3(28-31) A3(28-31)

          // The values arranged in shuffle patterns are operated with dot
          // product operation within 32 bit lane i.e corresponding bytes and
          // multiplied and added into 32 bit integers within 32 bit lane
          // Resembles MMLAs into 2x2 matrices in ARM Version
          const __m256i zero = _mm256_setzero_si256();
          __m256i iacc_mat_00_sp1 = mul_sum_i8_pairs_acc_int32x8(
            mul_sum_i8_pairs_acc_int32x8(
              mul_sum_i8_pairs_acc_int32x8(
                mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_01_3_sp1,
                                             rhs_mat_0145_3_sp1),
                lhs_mat_01_2_sp1, rhs_mat_0145_2_sp1),
              lhs_mat_01_1_sp1, rhs_mat_0145_1_sp1),
            lhs_mat_01_0_sp1, rhs_mat_0145_0_sp1);
          __m256i iacc_mat_01_sp1 = mul_sum_i8_pairs_acc_int32x8(
            mul_sum_i8_pairs_acc_int32x8(
              mul_sum_i8_pairs_acc_int32x8(
                mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_01_3_sp1,
                                             rhs_mat_2367_3_sp1),
                lhs_mat_01_2_sp1, rhs_mat_2367_2_sp1),
              lhs_mat_01_1_sp1, rhs_mat_2367_1_sp1),
            lhs_mat_01_0_sp1, rhs_mat_2367_0_sp1);
          __m256i iacc_mat_10_sp1 = mul_sum_i8_pairs_acc_int32x8(
            mul_sum_i8_pairs_acc_int32x8(
              mul_sum_i8_pairs_acc_int32x8(
                mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_23_3_sp1,
                                             rhs_mat_0145_3_sp1),
                lhs_mat_23_2_sp1, rhs_mat_0145_2_sp1),
              lhs_mat_23_1_sp1, rhs_mat_0145_1_sp1),
            lhs_mat_23_0_sp1, rhs_mat_0145_0_sp1);
          __m256i iacc_mat_11_sp1 = mul_sum_i8_pairs_acc_int32x8(
            mul_sum_i8_pairs_acc_int32x8(
              mul_sum_i8_pairs_acc_int32x8(
                mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_23_3_sp1,
                                             rhs_mat_2367_3_sp1),
                lhs_mat_23_2_sp1, rhs_mat_2367_2_sp1),
              lhs_mat_23_1_sp1, rhs_mat_2367_1_sp1),
            lhs_mat_23_0_sp1, rhs_mat_2367_0_sp1);
          __m256i iacc_mat_00_sp2 = mul_sum_i8_pairs_acc_int32x8(
            mul_sum_i8_pairs_acc_int32x8(
              mul_sum_i8_pairs_acc_int32x8(
                mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_01_3_sp2,
                                             rhs_mat_0145_3_sp2),
                lhs_mat_01_2_sp2, rhs_mat_0145_2_sp2),
              lhs_mat_01_1_sp2, rhs_mat_0145_1_sp2),
            lhs_mat_01_0_sp2, rhs_mat_0145_0_sp2);
          __m256i iacc_mat_01_sp2 = mul_sum_i8_pairs_acc_int32x8(
            mul_sum_i8_pairs_acc_int32x8(
              mul_sum_i8_pairs_acc_int32x8(
                mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_01_3_sp2,
                                             rhs_mat_2367_3_sp2),
                lhs_mat_01_2_sp2, rhs_mat_2367_2_sp2),
              lhs_mat_01_1_sp2, rhs_mat_2367_1_sp2),
            lhs_mat_01_0_sp2, rhs_mat_2367_0_sp2);
          __m256i iacc_mat_10_sp2 = mul_sum_i8_pairs_acc_int32x8(
            mul_sum_i8_pairs_acc_int32x8(
              mul_sum_i8_pairs_acc_int32x8(
                mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_23_3_sp2,
                                             rhs_mat_0145_3_sp2),
                lhs_mat_23_2_sp2, rhs_mat_0145_2_sp2),
              lhs_mat_23_1_sp2, rhs_mat_0145_1_sp2),
            lhs_mat_23_0_sp2, rhs_mat_0145_0_sp2);
          __m256i iacc_mat_11_sp2 = mul_sum_i8_pairs_acc_int32x8(
            mul_sum_i8_pairs_acc_int32x8(
              mul_sum_i8_pairs_acc_int32x8(
                mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_23_3_sp2,
                                             rhs_mat_2367_3_sp2),
                lhs_mat_23_2_sp2, rhs_mat_2367_2_sp2),
              lhs_mat_23_1_sp2, rhs_mat_2367_1_sp2),
            lhs_mat_23_0_sp2, rhs_mat_2367_0_sp2);

          // Output of both shuffle patterns are added in order to sum dot
          // product outputs of all 32 values in block
          __m256i iacc_mat_00 =
            _mm256_add_epi32(iacc_mat_00_sp1, iacc_mat_00_sp2);
          __m256i iacc_mat_01 =
            _mm256_add_epi32(iacc_mat_01_sp1, iacc_mat_01_sp2);
          __m256i iacc_mat_10 =
            _mm256_add_epi32(iacc_mat_10_sp1, iacc_mat_10_sp2);
          __m256i iacc_mat_11 =
            _mm256_add_epi32(iacc_mat_11_sp1, iacc_mat_11_sp2);

          // Straighten out to make 4 row vectors
          __m256i iacc_row_0 = _mm256_blend_epi32(
            iacc_mat_00, _mm256_shuffle_epi32(iacc_mat_01, 78), 204);
          __m256i iacc_row_1 = _mm256_blend_epi32(
            _mm256_shuffle_epi32(iacc_mat_00, 78), iacc_mat_01, 204);
          __m256i iacc_row_2 = _mm256_blend_epi32(
            iacc_mat_10, _mm256_shuffle_epi32(iacc_mat_11, 78), 204);
          __m256i iacc_row_3 = _mm256_blend_epi32(
            _mm256_shuffle_epi32(iacc_mat_10, 78), iacc_mat_11, 204);

          // Load the scale(d) values for all the 4 Q8_0 blocks and repeat it
          // across lanes
          const __m256 row_scale_f32 =
            GGML_F32Cx8_REPEAT_LOAD(a_ptr[b].d, loadMask);

          // Multiply with appropiate scales and accumulate
          acc_rows[0] = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(iacc_row_0),
            _mm256_mul_ps(col_scale_f32,
                          _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)),
            acc_rows[0]);
          acc_rows[1] = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(iacc_row_1),
            _mm256_mul_ps(col_scale_f32,
                          _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 85)),
            acc_rows[1]);
          acc_rows[2] = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(iacc_row_2),
            _mm256_mul_ps(col_scale_f32,
                          _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)),
            acc_rows[2]);
          acc_rows[3] = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(iacc_row_3),
            _mm256_mul_ps(col_scale_f32,
                          _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 255)),
            acc_rows[3]);
        }

        // Store the accumulated values
        for (int i = 0; i < 4; i++) {
          _mm256_storeu_ps((float *)(s + ((y * 4 + i) * bs + x * 8)),
                           acc_rows[i]);
        }
      }
    }
    return;
  }

  float sumf[4][8];
  int sumi;

  for (int y = 0; y < nr / 4; y++) {
    const block_q8_0x4 *a_ptr = (const block_q8_0x4 *)vy + (y * nb);
    for (int x = 0; x < nc / ncols_interleaved; x++) {
      const block_q4_0x8 *b_ptr = (const block_q4_0x8 *)vx + (x * nb);
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
              sumf[m][j] += sumi * nntr_fp16_to_fp32(b_ptr[l].d[j]) *
                            nntr_fp16_to_fp32(a_ptr[l].d[m]);
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

void nntr_gemm_q4_K_8x8_q8_K(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc) {
  const int qk = QK_K;
  const int nb = n / qk;
  const int ncols_interleaved = 8;
  const int blocklen = 8;
  static const uint32_t kmask1 = 0x3f3f3f3f;
  static const uint32_t kmask2 = 0x0f0f0f0f;
  static const uint32_t kmask3 = 0x03030303;

  assert(n % qk == 0);
  assert(nr % 4 == 0);
  assert(nc % ncols_interleaved == 0);

  const block_q4_Kx8 *b_ptr_start = (const block_q4_Kx8 *)vx;
  const block_q8_Kx4 *a_ptr_start = (const block_q8_Kx4 *)vy;
  int64_t b_nb = n / QK_K;
  int64_t y = 0;

  // Mask to mask out nibbles from packed bytes
  const __m256i m4b = _mm256_set1_epi8(0x0F);
  // Permute mask used for easier vector processing at later stages
  __m256i requiredOrder = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);
  int64_t xstart = 0;
  int anr = nr - nr % 16;
  ; // Used to align nr with boundary of 16

  // Take group of four block_q8_Kx4 structures at each pass of the loop and
  // perform dot product operation
  for (; y < anr / 4; y += 4) {

    const block_q8_Kx4 *a_ptrs[4];

    a_ptrs[0] = a_ptr_start + (y * nb);
    for (int i = 0; i < 3; ++i) {
      a_ptrs[i + 1] = a_ptrs[i] + nb;
    }

    // Take group of eight block_q4_kx8 structures at each pass of the loop and
    // perform dot product operation
    for (int64_t x = xstart; x < nc / 8; x++) {

      const block_q4_Kx8 *b_ptr = b_ptr_start + (x * b_nb);

      // Master FP accumulators
      __m256 acc_rows[16];
      for (int i = 0; i < 16; i++) {
        acc_rows[i] = _mm256_setzero_ps();
      }

      __m256 acc_min_rows[16];
      for (int i = 0; i < 16; i++) {
        acc_min_rows[i] = _mm256_setzero_ps();
      }

      // For super block
      for (int64_t b = 0; b < nb; b++) {

        // Scale values - Load the eight scale values of block_q4_kx8
        const __m256 col_scale_f32 = GGML_F32Cx8_LOAD(b_ptr[b].d);

        // dmin values - Load the eight dmin values of block_q4_kx8
        const __m256 col_dmin_f32 = GGML_F32Cx8_LOAD(b_ptr[b].dmin);

        // Loop to iterate over the eight sub blocks of a super block - two sub
        // blocks are processed per iteration
        for (int sb = 0; sb < QK_K / 64; sb++) {

          // Load the eight block_q4_K for two sub blocks quantized values
          // interleaved with each other in chunks of eight bytes - B0,B1
          // ....B6,B7
          const __m256i rhs_raw_mat_0123_0 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + sb * 256));
          const __m256i rhs_raw_mat_4567_0 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 32 + sb * 256));
          const __m256i rhs_raw_mat_0123_1 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 64 + sb * 256));
          const __m256i rhs_raw_mat_4567_1 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 96 + sb * 256));
          const __m256i rhs_raw_mat_0123_2 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 128 + sb * 256));
          const __m256i rhs_raw_mat_4567_2 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 160 + sb * 256));
          const __m256i rhs_raw_mat_0123_3 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 192 + sb * 256));
          const __m256i rhs_raw_mat_4567_3 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 224 + sb * 256));

          // Save the values in the following vectors in the formats B0B1B4B5,
          // B2B3B6B7 for further processing and storing of values
          const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(
            rhs_raw_mat_0123_0,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder),
            240);
          const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder),
            rhs_raw_mat_4567_0, 240);
          const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(
            rhs_raw_mat_0123_1,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder),
            240);
          const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder),
            rhs_raw_mat_4567_1, 240);
          const __m256i rhs_raw_mat_0145_2 = _mm256_blend_epi32(
            rhs_raw_mat_0123_2,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_2, requiredOrder),
            240);
          const __m256i rhs_raw_mat_2367_2 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_0123_2, requiredOrder),
            rhs_raw_mat_4567_2, 240);
          const __m256i rhs_raw_mat_0145_3 = _mm256_blend_epi32(
            rhs_raw_mat_0123_3,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_3, requiredOrder),
            240);
          const __m256i rhs_raw_mat_2367_3 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_0123_3, requiredOrder),
            rhs_raw_mat_4567_3, 240);

          // 4-bit -> 8-bit
          // First sub block of the two sub blocks processed in the iteration
          const __m256i rhs_mat_0145_00 = _mm256_and_si256(
            rhs_raw_mat_0145_0, m4b); // B00(0-7) B01(0-7) B04(0-7) B05(0-7)
          const __m256i rhs_mat_2367_00 = _mm256_and_si256(
            rhs_raw_mat_2367_0, m4b); // B02(0-7) B03(0-7) B06(0-7) B07(0-7)

          const __m256i rhs_mat_0145_01 = _mm256_and_si256(
            rhs_raw_mat_0145_1, m4b); // B00(8-15) B01(8-15) B04(8-15) B05(8-15)
          const __m256i rhs_mat_2367_01 = _mm256_and_si256(
            rhs_raw_mat_2367_1, m4b); // B02(8-15) B03(8-15) B06(8-15) B07(8-15)

          const __m256i rhs_mat_0145_02 = _mm256_and_si256(
            rhs_raw_mat_0145_2,
            m4b); // B00(16-23) B01(16-23) B04(16-23) B05(16-23)
          const __m256i rhs_mat_2367_02 = _mm256_and_si256(
            rhs_raw_mat_2367_2,
            m4b); // B02(16-23) B03(16-23) B06(16-23) B07(16-23)

          const __m256i rhs_mat_0145_03 = _mm256_and_si256(
            rhs_raw_mat_0145_3,
            m4b); // B00(24-31) B01(24-31) B04(24-31) B05(24-31)
          const __m256i rhs_mat_2367_03 = _mm256_and_si256(
            rhs_raw_mat_2367_3,
            m4b); // B02(24-31) B03(24-31) B06(24-31) B07(24-31)

          // Second sub block of the two sub blocks processed in the iteration
          const __m256i rhs_mat_0145_10 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_0, 4),
                             m4b); // B10(0-7) B11(0-7) B14(0-7) B15(0-7)
          const __m256i rhs_mat_2367_10 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_0, 4),
                             m4b); // B12(0-7) B13(0-7) B16(0-7) B17(0-7)

          const __m256i rhs_mat_0145_11 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_1, 4),
                             m4b); // B10(8-15) B11(8-15) B14(8-15) B15(8-15)
          const __m256i rhs_mat_2367_11 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_1, 4),
                             m4b); // B12(8-15) B13(8-15) B16(8-15) B17(8-15)

          const __m256i rhs_mat_0145_12 = _mm256_and_si256(
            _mm256_srli_epi16(rhs_raw_mat_0145_2, 4),
            m4b); // B10(16-23) B11(16-23) B14(16-23) B15(16-23)
          const __m256i rhs_mat_2367_12 = _mm256_and_si256(
            _mm256_srli_epi16(rhs_raw_mat_2367_2, 4),
            m4b); // B12(16-23) B13(16-23) B16(16-23) B17(16-23)

          const __m256i rhs_mat_0145_13 = _mm256_and_si256(
            _mm256_srli_epi16(rhs_raw_mat_0145_3, 4),
            m4b); // B10(24-31) B11(24-31) B14(24-31) B15(24-31)
          const __m256i rhs_mat_2367_13 = _mm256_and_si256(
            _mm256_srli_epi16(rhs_raw_mat_2367_3, 4),
            m4b); // B12(24-31) B13(24-31) B16(24-31) B17(24-31)

          // Shuffle pattern one - right side input
          const __m256i rhs_mat_0145_00_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_00, 136); // B00(0-3) B01(0-3) B00(0-3) B01(0-3)
                                   // B04(0-3) B05(0-3) B04(0-3) B05(0-3)
          const __m256i rhs_mat_2367_00_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_00, 136); // B02(0-3) B03(0-3) B02(0-3) B03(0-3)
                                   // B06(0-3) B07(0-3) B06(0-3) B07(0-3)

          const __m256i rhs_mat_0145_01_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_01, 136); // B00(8-11) B01(8-11) B00(8-11) B01(8-11)
                                   // B04(8-11) B05(8-11) B04(8-11) B05(8-11)
          const __m256i rhs_mat_2367_01_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_01, 136); // B02(8-11) B03(8-11) B02(8-11) B03(8-11)
                                   // B06(8-11) B07(8-11) B06(8-11) B07(8-11)

          const __m256i rhs_mat_0145_02_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_02,
            136); // B00(16-19) B01(16-19) B00(16-19) B01(16-19) B04(16-19)
                  // B05(16-19) B04(16-19) B05(16-19)
          const __m256i rhs_mat_2367_02_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_02,
            136); // B02(16-19) B03(16-19) B02(16-19) B03(16-19) B06(16-19)
                  // B07(16-19) B06(16-19) B07(16-19)

          const __m256i rhs_mat_0145_03_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_03,
            136); // B00(24-27) B01(24-27) B00(24-27) B01(24-27) B04(24-27)
                  // B05(24-27) B04(24-27) B05(24-27)
          const __m256i rhs_mat_2367_03_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_03,
            136); // B02(24-27) B03(24-27) B02(24-27) B03(24-27) B06(24-27)
                  // B07(24-27) B06(24-27) B07(24-27)

          const __m256i rhs_mat_0145_10_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_10, 136); // B10(0-3) B11(0-3) B10(0-3) B11(0-3)
                                   // B14(0-3) B15(0-3) B14(0-3) B15(0-3)
          const __m256i rhs_mat_2367_10_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_10, 136); // B12(0-3) B13(0-3) B12(0-3) B13(0-3)
                                   // B16(0-3) B17(0-3) B16(0-3) B17(0-3)

          const __m256i rhs_mat_0145_11_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_11, 136); // B10(8-11) B11(8-11) B10(8-11) B11(8-11)
                                   // B14(8-11) B15(8-11) B14(8-11) B15(8-11)
          const __m256i rhs_mat_2367_11_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_11, 136); // B12(8-11) B13(8-11) B12(8-11) B13(8-11)
                                   // B16(8-11) B17(8-11) B16(8-11) B17(8-11)

          const __m256i rhs_mat_0145_12_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_12,
            136); // B10(16-19) B11(16-19) B10(16-19) B11(16-19) B14(16-19)
                  // B15(16-19) B14(16-19) B15(16-19)
          const __m256i rhs_mat_2367_12_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_12,
            136); // B12(16-19) B13(16-19) B12(16-19) B13(16-19) B16(16-19)
                  // B17(16-19) B16(16-19) B17(16-19)

          const __m256i rhs_mat_0145_13_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_13,
            136); // B10(24-27) B11(24-27) B10(24-27) B11(24-27) B14(24-27)
                  // B15(24-27) B14(24-27) B15(24-27)
          const __m256i rhs_mat_2367_13_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_13,
            136); // B12(24-27) B13(24-27) B12(24-27) B13(24-27) B16(24-27)
                  // B17(24-27) B16(24-27) B17(24-27)

          // Shuffle pattern two - right side input
          const __m256i rhs_mat_0145_00_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_00, 221); // B00(4-7) B01(4-7) B00(4-7) B01(4-7)
                                   // B04(4-7) B05(4-7) B04(4-7) B05(4-7)
          const __m256i rhs_mat_2367_00_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_00, 221); // B02(4-7) B03(4-7) B02(4-7) B03(4-7)
                                   // B06(4-7) B07(4-7) B06(4-7) B07(4-7)

          const __m256i rhs_mat_0145_01_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_01,
            221); // B00(12-15) B01(12-15) B00(12-15) B01(12-15) B04(12-15)
                  // B05(12-15) B04(12-15) B05(12-15)
          const __m256i rhs_mat_2367_01_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_01,
            221); // B02(12-15) B03(12-15) B02(12-15) B03(12-15) B06(12-15)
                  // B07(12-15) B06(12-15) B07(12-15)

          const __m256i rhs_mat_0145_02_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_02,
            221); // B00(20-23) B01(20-23) B00(20-23) B01(20-23) B04(20-23)
                  // B05(20-23) B04(20-23) B05(20-23)
          const __m256i rhs_mat_2367_02_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_02,
            221); // B02(20-23) B03(20-23) B02(20-23) B03(20-23) B06(20-23)
                  // B07(20-23) B06(20-23) B07(20-23)

          const __m256i rhs_mat_0145_03_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_03,
            221); // B00(28-31) B01(28-31) B00(28-31) B01(28-31) B04(28-31)
                  // B05(28-31) B04(28-31) B05(28-31)
          const __m256i rhs_mat_2367_03_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_03,
            221); // B02(28-31) B03(28-31) B02(28-31) B03(28-31) B06(28-31)
                  // B07(28-31) B06(28-31) B07(28-31)

          const __m256i rhs_mat_0145_10_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_10, 221); // B10(4-7) B11(4-7) B10(4-7) B11(4-7)
                                   // B14(4-7) B15(4-7) B14(4-7) B15(4-7)
          const __m256i rhs_mat_2367_10_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_10, 221); // B12(4-7) B13(4-7) B12(4-7) B13(4-7)
                                   // B16(4-7) B17(4-7) B16(4-7) B17(4-7)

          const __m256i rhs_mat_0145_11_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_11,
            221); // B10(12-15) B11(12-15) B10(12-15) B11(12-15) B14(12-15)
                  // B15(12-15) B14(12-15) B15(12-15)
          const __m256i rhs_mat_2367_11_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_11,
            221); // B12(12-15) B13(12-15) B12(12-15) B13(12-15) B16(12-15)
                  // B17(12-15) B16(12-15) B17(12-15)

          const __m256i rhs_mat_0145_12_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_12,
            221); // B10(20-23) B11(20-23) B10(20-23) B11(20-23) B14(20-23)
                  // B15(20-23) B14(20-23) B15(20-23)
          const __m256i rhs_mat_2367_12_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_12,
            221); // B12(20-23) B13(20-23) B12(20-23) B13(20-23) B16(20-23)
                  // B17(20-23) B16(20-23) B17(20-23)

          const __m256i rhs_mat_0145_13_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_13,
            221); // B10(28-31) B11(28-31) B10(28-31) B11(28-31) B14(28-31)
                  // B15(28-31) B14(28-31) B15(28-31)
          const __m256i rhs_mat_2367_13_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_13,
            221); // B12(28-31) B13(28-31) B12(28-31) B13(28-31) B16(28-31)
                  // B17(28-31) B16(28-31) B17(28-31)

          uint32_t utmp_0[4], utmp_1[4];

          // Scales and Mins of corresponding sub blocks from different Q4_K
          // structures are stored together The below block is for eg to extract
          // first sub block's scales and mins from different Q4_K structures
          // for the sb loop
          memcpy(utmp_0, b_ptr[b].scales + 24 * sb, 12);
          utmp_0[3] =
            ((utmp_0[2] >> 4) & kmask2) | (((utmp_0[1] >> 6) & kmask3) << 4);
          const uint32_t uaux_0 = utmp_0[1] & kmask1;
          utmp_0[1] = (utmp_0[2] & kmask2) | (((utmp_0[0] >> 6) & kmask3) << 4);
          utmp_0[2] = uaux_0;
          utmp_0[0] &= kmask1;

          // The below block is for eg to extract second sub block's scales and
          // mins from different Q4_K structures for the sb loop
          memcpy(utmp_1, b_ptr[b].scales + 12 + sb * 24, 12);
          utmp_1[3] =
            ((utmp_1[2] >> 4) & kmask2) | (((utmp_1[1] >> 6) & kmask3) << 4);
          const uint32_t uaux_1 = utmp_1[1] & kmask1;
          utmp_1[1] = (utmp_1[2] & kmask2) | (((utmp_1[0] >> 6) & kmask3) << 4);
          utmp_1[2] = uaux_1;
          utmp_1[0] &= kmask1;

          // Scales of first sub block in the sb loop
          const __m128i mins_and_scales_0 =
            _mm_set_epi32(utmp_0[3], utmp_0[2], utmp_0[1], utmp_0[0]);
          const __m256i scales_0 = _mm256_cvtepu8_epi16(
            _mm_unpacklo_epi8(mins_and_scales_0, mins_and_scales_0));

          // Scales of second sub block in the sb loop
          const __m128i mins_and_scales_1 =
            _mm_set_epi32(utmp_1[3], utmp_1[2], utmp_1[1], utmp_1[0]);
          const __m256i scales_1 = _mm256_cvtepu8_epi16(
            _mm_unpacklo_epi8(mins_and_scales_1, mins_and_scales_1));

          // Mins of first and second sub block of Q4_K block are arranged side
          // by side
          const __m256i mins_01 = _mm256_cvtepu8_epi16(
            _mm_unpacklo_epi8(_mm_shuffle_epi32(mins_and_scales_0, 78),
                              _mm_shuffle_epi32(mins_and_scales_1, 78)));

          const __m256i scale_0145_0 = _mm256_shuffle_epi32(scales_0, 68);
          const __m256i scale_2367_0 = _mm256_shuffle_epi32(scales_0, 238);

          const __m256i scale_0145_1 = _mm256_shuffle_epi32(scales_1, 68);
          const __m256i scale_2367_1 = _mm256_shuffle_epi32(scales_1, 238);

          for (int rp = 0; rp < 4; rp++) {

            // Load the four block_q8_k quantized values interleaved with each
            // other in chunks of eight bytes - A0,A1,A2,A3 Loaded as set of 128
            // bit vectors and repeated into a 256 bit vector
            __m256i lhs_mat_0123_00 = _mm256_loadu_si256(
              (const __m256i *)((a_ptrs[rp][b].qs + 256 * sb)));
            __m256i lhs_mat_01_00 =
              _mm256_permute2f128_si256(lhs_mat_0123_00, lhs_mat_0123_00, 0);
            __m256i lhs_mat_23_00 =
              _mm256_permute2f128_si256(lhs_mat_0123_00, lhs_mat_0123_00, 17);
            __m256i lhs_mat_0123_01 = _mm256_loadu_si256(
              (const __m256i *)((a_ptrs[rp][b].qs + 32 + 256 * sb)));
            __m256i lhs_mat_01_01 =
              _mm256_permute2f128_si256(lhs_mat_0123_01, lhs_mat_0123_01, 0);
            __m256i lhs_mat_23_01 =
              _mm256_permute2f128_si256(lhs_mat_0123_01, lhs_mat_0123_01, 17);
            __m256i lhs_mat_0123_02 = _mm256_loadu_si256(
              (const __m256i *)((a_ptrs[rp][b].qs + 64 + 256 * sb)));
            __m256i lhs_mat_01_02 =
              _mm256_permute2f128_si256(lhs_mat_0123_02, lhs_mat_0123_02, 0);
            __m256i lhs_mat_23_02 =
              _mm256_permute2f128_si256(lhs_mat_0123_02, lhs_mat_0123_02, 17);
            __m256i lhs_mat_0123_03 = _mm256_loadu_si256(
              (const __m256i *)((a_ptrs[rp][b].qs + 96 + 256 * sb)));
            __m256i lhs_mat_01_03 =
              _mm256_permute2f128_si256(lhs_mat_0123_03, lhs_mat_0123_03, 0);
            __m256i lhs_mat_23_03 =
              _mm256_permute2f128_si256(lhs_mat_0123_03, lhs_mat_0123_03, 17);
            __m256i lhs_mat_0123_10 = _mm256_loadu_si256(
              (const __m256i *)((a_ptrs[rp][b].qs + 128 + 256 * sb)));
            __m256i lhs_mat_01_10 =
              _mm256_permute2f128_si256(lhs_mat_0123_10, lhs_mat_0123_10, 0);
            __m256i lhs_mat_23_10 =
              _mm256_permute2f128_si256(lhs_mat_0123_10, lhs_mat_0123_10, 17);
            __m256i lhs_mat_0123_11 = _mm256_loadu_si256(
              (const __m256i *)((a_ptrs[rp][b].qs + 160 + 256 * sb)));
            __m256i lhs_mat_01_11 =
              _mm256_permute2f128_si256(lhs_mat_0123_11, lhs_mat_0123_11, 0);
            __m256i lhs_mat_23_11 =
              _mm256_permute2f128_si256(lhs_mat_0123_11, lhs_mat_0123_11, 17);
            __m256i lhs_mat_0123_12 = _mm256_loadu_si256(
              (const __m256i *)((a_ptrs[rp][b].qs + 192 + 256 * sb)));
            __m256i lhs_mat_01_12 =
              _mm256_permute2f128_si256(lhs_mat_0123_12, lhs_mat_0123_12, 0);
            __m256i lhs_mat_23_12 =
              _mm256_permute2f128_si256(lhs_mat_0123_12, lhs_mat_0123_12, 17);
            __m256i lhs_mat_0123_13 = _mm256_loadu_si256(
              (const __m256i *)((a_ptrs[rp][b].qs + 224 + 256 * sb)));
            __m256i lhs_mat_01_13 =
              _mm256_permute2f128_si256(lhs_mat_0123_13, lhs_mat_0123_13, 0);
            __m256i lhs_mat_23_13 =
              _mm256_permute2f128_si256(lhs_mat_0123_13, lhs_mat_0123_13, 17);

            // Bsums are loaded - four bsums are loaded (for two sub blocks) for
            // the different Q8_K blocks
            __m256i lhs_bsums_0123_01 = _mm256_loadu_si256(
              (const __m256i *)((a_ptrs[rp][b].bsums + 16 * sb)));
            __m256i lhs_bsums_hsum_0123_01 = _mm256_castsi128_si256(
              _mm_hadd_epi16(_mm256_castsi256_si128(lhs_bsums_0123_01),
                             _mm256_extractf128_si256(lhs_bsums_0123_01, 1)));
            lhs_bsums_hsum_0123_01 = _mm256_permute2x128_si256(
              lhs_bsums_hsum_0123_01, lhs_bsums_hsum_0123_01, 0);

            // Shuffle pattern one - left side input
            const __m256i lhs_mat_01_00_sp1 = _mm256_shuffle_epi32(
              lhs_mat_01_00, 160); // A00(0-3) A00(0-3) A01(0-3) A01(0-3)
                                   // A00(0-3) A00(0-3) A01(0-3) A01(0-3)
            const __m256i lhs_mat_23_00_sp1 = _mm256_shuffle_epi32(
              lhs_mat_23_00, 160); // A02(0-3) A03(0-3) A02(0-3) A03(0-3)
                                   // A02(0-3) A03(0-3) A02(0-3) A03(0-3)

            const __m256i lhs_mat_01_01_sp1 = _mm256_shuffle_epi32(
              lhs_mat_01_01, 160); // A00(8-11) A00(8-11) A01(8-11) A01(8-11)
                                   // A00(8-11) A00(8-11) A01(8-11) A01(8-11)
            const __m256i lhs_mat_23_01_sp1 = _mm256_shuffle_epi32(
              lhs_mat_23_01, 160); // A02(8-11) A03(8-11) A02(8-11) A03(8-11)
                                   // A02(8-11) A03(8-11) A02(8-11) A03(8-11)

            const __m256i lhs_mat_01_02_sp1 = _mm256_shuffle_epi32(
              lhs_mat_01_02,
              160); // A00(16-19) A00(16-19) A01(16-19) A01(16-19) A00(16-19)
                    // A00(16-19) A01(16-19) A01(16-19)
            const __m256i lhs_mat_23_02_sp1 = _mm256_shuffle_epi32(
              lhs_mat_23_02,
              160); // A02(16-19) A03(16-19) A02(16-19) A03(16-19) A02(16-19)
                    // A03(16-19) A02(16-19) A03(16-19)

            const __m256i lhs_mat_01_03_sp1 = _mm256_shuffle_epi32(
              lhs_mat_01_03,
              160); // A00(24-27) A00(24-27) A01(24-27) A01(24-27) A00(24-27)
                    // A00(24-27) A01(24-27) A01(24-27)
            const __m256i lhs_mat_23_03_sp1 = _mm256_shuffle_epi32(
              lhs_mat_23_03,
              160); // A02(24-27) A03(24-27) A02(24-27) A03(24-27) A02(24-27)
                    // A03(24-27) A02(24-27) A03(24-27)

            const __m256i lhs_mat_01_10_sp1 = _mm256_shuffle_epi32(
              lhs_mat_01_10, 160); // A10(0-3) A10(0-3) A11(0-3) A11(0-3)
                                   // A10(0-3) A10(0-3) A11(0-3) A11(0-3)
            const __m256i lhs_mat_23_10_sp1 = _mm256_shuffle_epi32(
              lhs_mat_23_10, 160); // A12(0-3) A13(0-3) A12(0-3) A13(0-3)
                                   // A12(0-3) A13(0-3) A12(0-3) A13(0-3)

            const __m256i lhs_mat_01_11_sp1 = _mm256_shuffle_epi32(
              lhs_mat_01_11, 160); // A10(8-11) A10(8-11) A11(8-11) A11(8-11)
                                   // A10(8-11) A10(8-11) A11(8-11) A11(8-11)
            const __m256i lhs_mat_23_11_sp1 = _mm256_shuffle_epi32(
              lhs_mat_23_11, 160); // A12(8-11) A13(8-11) A12(8-11) A13(8-11)
                                   // A12(8-11) A13(8-11) A12(8-11) A13(8-11)

            const __m256i lhs_mat_01_12_sp1 = _mm256_shuffle_epi32(
              lhs_mat_01_12,
              160); // A10(16-19) A10(16-19) A11(16-19) A11(16-19) A10(16-19)
                    // A10(16-19) A11(16-19) A11(16-19)
            const __m256i lhs_mat_23_12_sp1 = _mm256_shuffle_epi32(
              lhs_mat_23_12,
              160); // A12(16-19) A13(16-19) A12(16-19) A13(16-19) A12(16-19)
                    // A13(16-19) A12(16-19) A13(16-19)

            const __m256i lhs_mat_01_13_sp1 = _mm256_shuffle_epi32(
              lhs_mat_01_13,
              160); // A10(24-27) A10(24-27) A11(24-27) A11(24-27) A10(24-27)
                    // A10(24-27) A11(24-27) A11(24-27)
            const __m256i lhs_mat_23_13_sp1 = _mm256_shuffle_epi32(
              lhs_mat_23_13,
              160); // A12(24-27) A13(24-27) A12(24-27) A13(24-27) A12(24-27)
                    // A13(24-27) A12(24-27) A13(24-27)

            // Shuffle pattern two- left side input
            const __m256i lhs_mat_01_00_sp2 = _mm256_shuffle_epi32(
              lhs_mat_01_00, 245); // A00(4-7) A00(4-7) A01(4-7) A01(4-7)
                                   // A00(4-7) A00(4-7) A01(4-7) A01(4-7)
            const __m256i lhs_mat_23_00_sp2 = _mm256_shuffle_epi32(
              lhs_mat_23_00, 245); // A02(4-7) A03(4-7) A02(4-7) A03(4-7)
                                   // A02(4-7) A03(4-7) A02(4-7) A03(4-7)

            const __m256i lhs_mat_01_01_sp2 = _mm256_shuffle_epi32(
              lhs_mat_01_01,
              245); // A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15)
                    // A00(12-15) A01(12-15) A01(12-15)
            const __m256i lhs_mat_23_01_sp2 = _mm256_shuffle_epi32(
              lhs_mat_23_01,
              245); // A02(12-15) A03(12-15) A02(12-15) A03(12-15) A02(12-15)
                    // A03(12-15) A02(12-15) A03(12-15)

            const __m256i lhs_mat_01_02_sp2 = _mm256_shuffle_epi32(
              lhs_mat_01_02,
              245); // A00(20-23) A00(20-23) A01(20-23) A01(20-23) A00(20-23)
                    // A00(20-23) A01(20-23) A01(20-23)
            const __m256i lhs_mat_23_02_sp2 = _mm256_shuffle_epi32(
              lhs_mat_23_02,
              245); // A02(20-23) A03(20-23) A02(20-23) A03(20-23) A02(20-23)
                    // A03(20-23) A02(20-23) A03(20-23)

            const __m256i lhs_mat_01_03_sp2 = _mm256_shuffle_epi32(
              lhs_mat_01_03,
              245); // A00(28-31) A00(28-31) A01(28-31) A01(28-31) A00(28-31)
                    // A00(28-31) A01(28-31) A01(28-31)
            const __m256i lhs_mat_23_03_sp2 = _mm256_shuffle_epi32(
              lhs_mat_23_03,
              245); // A02(28-31) A03(28-31) A02(28-31) A03(28-31) A02(28-31)
                    // A03(28-31) A02(28-31) A03(28-31)

            const __m256i lhs_mat_01_10_sp2 = _mm256_shuffle_epi32(
              lhs_mat_01_10, 245); // A10(4-7) A10(4-7) A11(4-7) A11(4-7)
                                   // A10(4-7) A10(4-7) A11(4-7) A11(4-7)
            const __m256i lhs_mat_23_10_sp2 = _mm256_shuffle_epi32(
              lhs_mat_23_10, 245); // A12(4-7) A13(4-7) A12(4-7) A13(4-7)
                                   // A12(4-7) A13(4-7) A12(4-7) A13(4-7)

            const __m256i lhs_mat_01_11_sp2 = _mm256_shuffle_epi32(
              lhs_mat_01_11,
              245); // A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15)
                    // A10(12-15) A11(12-15) A11(12-15)
            const __m256i lhs_mat_23_11_sp2 = _mm256_shuffle_epi32(
              lhs_mat_23_11,
              245); // A12(12-15) A13(12-15) A12(12-15) A13(12-15) A12(12-15)
                    // A13(12-15) A12(12-15) A13(12-15)

            const __m256i lhs_mat_01_12_sp2 = _mm256_shuffle_epi32(
              lhs_mat_01_12,
              245); // A10(20-23) A10(20-23) A11(20-23) A11(20-23) A10(20-23)
                    // A10(20-23) A11(20-23) A11(20-23)
            const __m256i lhs_mat_23_12_sp2 = _mm256_shuffle_epi32(
              lhs_mat_23_12,
              245); // A12(20-23) A13(20-23) A12(20-23) A13(20-23) A12(20-23)
                    // A13(20-23) A12(20-23) A13(20-23)

            const __m256i lhs_mat_01_13_sp2 = _mm256_shuffle_epi32(
              lhs_mat_01_13,
              245); // A10(28-31) A10(28-31) A11(28-31) A11(28-31) A10(28-31)
                    // A10(28-31) A11(28-31) A11(28-31)
            const __m256i lhs_mat_23_13_sp2 = _mm256_shuffle_epi32(
              lhs_mat_23_13,
              245); // A12(28-31) A13(28-31) A12(28-31) A13(28-31) A12(28-31)
                    // A13(28-31) A12(28-31) A13(28-31)

            // The values arranged in shuffle patterns are operated with dot
            // product operation within 32 bit lane i.e corresponding bytes and
            // multiplied and added into 32 bit integers within 32 bit lane
            __m256i iacc_mat_00_0_sp1 = _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_add_epi16(
                  _mm256_maddubs_epi16(rhs_mat_0145_03_sp1, lhs_mat_01_03_sp1),
                  _mm256_maddubs_epi16(rhs_mat_0145_02_sp1, lhs_mat_01_02_sp1)),
                _mm256_maddubs_epi16(rhs_mat_0145_01_sp1, lhs_mat_01_01_sp1)),
              _mm256_maddubs_epi16(rhs_mat_0145_00_sp1, lhs_mat_01_00_sp1));
            __m256i iacc_mat_01_0_sp1 = _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_add_epi16(
                  _mm256_maddubs_epi16(rhs_mat_2367_03_sp1, lhs_mat_01_03_sp1),
                  _mm256_maddubs_epi16(rhs_mat_2367_02_sp1, lhs_mat_01_02_sp1)),
                _mm256_maddubs_epi16(rhs_mat_2367_01_sp1, lhs_mat_01_01_sp1)),
              _mm256_maddubs_epi16(rhs_mat_2367_00_sp1, lhs_mat_01_00_sp1));
            __m256i iacc_mat_10_0_sp1 = _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_add_epi16(
                  _mm256_maddubs_epi16(rhs_mat_0145_03_sp1, lhs_mat_23_03_sp1),
                  _mm256_maddubs_epi16(rhs_mat_0145_02_sp1, lhs_mat_23_02_sp1)),
                _mm256_maddubs_epi16(rhs_mat_0145_01_sp1, lhs_mat_23_01_sp1)),
              _mm256_maddubs_epi16(rhs_mat_0145_00_sp1, lhs_mat_23_00_sp1));
            __m256i iacc_mat_11_0_sp1 = _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_add_epi16(
                  _mm256_maddubs_epi16(rhs_mat_2367_03_sp1, lhs_mat_23_03_sp1),
                  _mm256_maddubs_epi16(rhs_mat_2367_02_sp1, lhs_mat_23_02_sp1)),
                _mm256_maddubs_epi16(rhs_mat_2367_01_sp1, lhs_mat_23_01_sp1)),
              _mm256_maddubs_epi16(rhs_mat_2367_00_sp1, lhs_mat_23_00_sp1));
            __m256i iacc_mat_00_1_sp1 = _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_add_epi16(
                  _mm256_maddubs_epi16(rhs_mat_0145_13_sp1, lhs_mat_01_13_sp1),
                  _mm256_maddubs_epi16(rhs_mat_0145_12_sp1, lhs_mat_01_12_sp1)),
                _mm256_maddubs_epi16(rhs_mat_0145_11_sp1, lhs_mat_01_11_sp1)),
              _mm256_maddubs_epi16(rhs_mat_0145_10_sp1, lhs_mat_01_10_sp1));
            __m256i iacc_mat_01_1_sp1 = _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_add_epi16(
                  _mm256_maddubs_epi16(rhs_mat_2367_13_sp1, lhs_mat_01_13_sp1),
                  _mm256_maddubs_epi16(rhs_mat_2367_12_sp1, lhs_mat_01_12_sp1)),
                _mm256_maddubs_epi16(rhs_mat_2367_11_sp1, lhs_mat_01_11_sp1)),
              _mm256_maddubs_epi16(rhs_mat_2367_10_sp1, lhs_mat_01_10_sp1));
            __m256i iacc_mat_10_1_sp1 = _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_add_epi16(
                  _mm256_maddubs_epi16(rhs_mat_0145_13_sp1, lhs_mat_23_13_sp1),
                  _mm256_maddubs_epi16(rhs_mat_0145_12_sp1, lhs_mat_23_12_sp1)),
                _mm256_maddubs_epi16(rhs_mat_0145_11_sp1, lhs_mat_23_11_sp1)),
              _mm256_maddubs_epi16(rhs_mat_0145_10_sp1, lhs_mat_23_10_sp1));
            __m256i iacc_mat_11_1_sp1 = _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_add_epi16(
                  _mm256_maddubs_epi16(rhs_mat_2367_13_sp1, lhs_mat_23_13_sp1),
                  _mm256_maddubs_epi16(rhs_mat_2367_12_sp1, lhs_mat_23_12_sp1)),
                _mm256_maddubs_epi16(rhs_mat_2367_11_sp1, lhs_mat_23_11_sp1)),
              _mm256_maddubs_epi16(rhs_mat_2367_10_sp1, lhs_mat_23_10_sp1));

            __m256i iacc_mat_00_0_sp2 = _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_add_epi16(
                  _mm256_maddubs_epi16(rhs_mat_0145_03_sp2, lhs_mat_01_03_sp2),
                  _mm256_maddubs_epi16(rhs_mat_0145_02_sp2, lhs_mat_01_02_sp2)),
                _mm256_maddubs_epi16(rhs_mat_0145_01_sp2, lhs_mat_01_01_sp2)),
              _mm256_maddubs_epi16(rhs_mat_0145_00_sp2, lhs_mat_01_00_sp2));
            __m256i iacc_mat_01_0_sp2 = _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_add_epi16(
                  _mm256_maddubs_epi16(rhs_mat_2367_03_sp2, lhs_mat_01_03_sp2),
                  _mm256_maddubs_epi16(rhs_mat_2367_02_sp2, lhs_mat_01_02_sp2)),
                _mm256_maddubs_epi16(rhs_mat_2367_01_sp2, lhs_mat_01_01_sp2)),
              _mm256_maddubs_epi16(rhs_mat_2367_00_sp2, lhs_mat_01_00_sp2));
            __m256i iacc_mat_10_0_sp2 = _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_add_epi16(
                  _mm256_maddubs_epi16(rhs_mat_0145_03_sp2, lhs_mat_23_03_sp2),
                  _mm256_maddubs_epi16(rhs_mat_0145_02_sp2, lhs_mat_23_02_sp2)),
                _mm256_maddubs_epi16(rhs_mat_0145_01_sp2, lhs_mat_23_01_sp2)),
              _mm256_maddubs_epi16(rhs_mat_0145_00_sp2, lhs_mat_23_00_sp2));
            __m256i iacc_mat_11_0_sp2 = _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_add_epi16(
                  _mm256_maddubs_epi16(rhs_mat_2367_03_sp2, lhs_mat_23_03_sp2),
                  _mm256_maddubs_epi16(rhs_mat_2367_02_sp2, lhs_mat_23_02_sp2)),
                _mm256_maddubs_epi16(rhs_mat_2367_01_sp2, lhs_mat_23_01_sp2)),
              _mm256_maddubs_epi16(rhs_mat_2367_00_sp2, lhs_mat_23_00_sp2));
            __m256i iacc_mat_00_1_sp2 = _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_add_epi16(
                  _mm256_maddubs_epi16(rhs_mat_0145_13_sp2, lhs_mat_01_13_sp2),
                  _mm256_maddubs_epi16(rhs_mat_0145_12_sp2, lhs_mat_01_12_sp2)),
                _mm256_maddubs_epi16(rhs_mat_0145_11_sp2, lhs_mat_01_11_sp2)),
              _mm256_maddubs_epi16(rhs_mat_0145_10_sp2, lhs_mat_01_10_sp2));
            __m256i iacc_mat_01_1_sp2 = _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_add_epi16(
                  _mm256_maddubs_epi16(rhs_mat_2367_13_sp2, lhs_mat_01_13_sp2),
                  _mm256_maddubs_epi16(rhs_mat_2367_12_sp2, lhs_mat_01_12_sp2)),
                _mm256_maddubs_epi16(rhs_mat_2367_11_sp2, lhs_mat_01_11_sp2)),
              _mm256_maddubs_epi16(rhs_mat_2367_10_sp2, lhs_mat_01_10_sp2));
            __m256i iacc_mat_10_1_sp2 = _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_add_epi16(
                  _mm256_maddubs_epi16(rhs_mat_0145_13_sp2, lhs_mat_23_13_sp2),
                  _mm256_maddubs_epi16(rhs_mat_0145_12_sp2, lhs_mat_23_12_sp2)),
                _mm256_maddubs_epi16(rhs_mat_0145_11_sp2, lhs_mat_23_11_sp2)),
              _mm256_maddubs_epi16(rhs_mat_0145_10_sp2, lhs_mat_23_10_sp2));
            __m256i iacc_mat_11_1_sp2 = _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_add_epi16(
                  _mm256_maddubs_epi16(rhs_mat_2367_13_sp2, lhs_mat_23_13_sp2),
                  _mm256_maddubs_epi16(rhs_mat_2367_12_sp2, lhs_mat_23_12_sp2)),
                _mm256_maddubs_epi16(rhs_mat_2367_11_sp2, lhs_mat_23_11_sp2)),
              _mm256_maddubs_epi16(rhs_mat_2367_10_sp2, lhs_mat_23_10_sp2));

            // Output of both shuffle patterns are added in order to sum dot
            // product outputs of all 32 values in block
            __m256i iacc_mat_00_0 =
              _mm256_add_epi16(iacc_mat_00_0_sp1, iacc_mat_00_0_sp2);
            __m256i iacc_mat_01_0 =
              _mm256_add_epi16(iacc_mat_01_0_sp1, iacc_mat_01_0_sp2);
            __m256i iacc_mat_10_0 =
              _mm256_add_epi16(iacc_mat_10_0_sp1, iacc_mat_10_0_sp2);
            __m256i iacc_mat_11_0 =
              _mm256_add_epi16(iacc_mat_11_0_sp1, iacc_mat_11_0_sp2);

            __m256i iacc_mat_00_1 =
              _mm256_add_epi16(iacc_mat_00_1_sp1, iacc_mat_00_1_sp2);
            __m256i iacc_mat_01_1 =
              _mm256_add_epi16(iacc_mat_01_1_sp1, iacc_mat_01_1_sp2);
            __m256i iacc_mat_10_1 =
              _mm256_add_epi16(iacc_mat_10_1_sp1, iacc_mat_10_1_sp2);
            __m256i iacc_mat_11_1 =
              _mm256_add_epi16(iacc_mat_11_1_sp1, iacc_mat_11_1_sp2);

            // Output of both shuffle patterns are added in order to sum dot
            // product outputs of all 32 values in block
            iacc_mat_00_0 = _mm256_madd_epi16(iacc_mat_00_0, scale_0145_0);
            iacc_mat_01_0 = _mm256_madd_epi16(iacc_mat_01_0, scale_2367_0);
            iacc_mat_10_0 = _mm256_madd_epi16(iacc_mat_10_0, scale_0145_0);
            iacc_mat_11_0 = _mm256_madd_epi16(iacc_mat_11_0, scale_2367_0);

            iacc_mat_00_1 = _mm256_madd_epi16(iacc_mat_00_1, scale_0145_1);
            iacc_mat_01_1 = _mm256_madd_epi16(iacc_mat_01_1, scale_2367_1);
            iacc_mat_10_1 = _mm256_madd_epi16(iacc_mat_10_1, scale_0145_1);
            iacc_mat_11_1 = _mm256_madd_epi16(iacc_mat_11_1, scale_2367_1);

            // Straighten out to make 4 row vectors (4 for each sub block which
            // are accumulated together in the next step)
            __m256i iacc_row_0_0 = _mm256_blend_epi32(
              iacc_mat_00_0, _mm256_shuffle_epi32(iacc_mat_01_0, 78), 204);
            __m256i iacc_row_1_0 = _mm256_blend_epi32(
              _mm256_shuffle_epi32(iacc_mat_00_0, 78), iacc_mat_01_0, 204);
            __m256i iacc_row_2_0 = _mm256_blend_epi32(
              iacc_mat_10_0, _mm256_shuffle_epi32(iacc_mat_11_0, 78), 204);
            __m256i iacc_row_3_0 = _mm256_blend_epi32(
              _mm256_shuffle_epi32(iacc_mat_10_0, 78), iacc_mat_11_0, 204);
            __m256i iacc_row_0_1 = _mm256_blend_epi32(
              iacc_mat_00_1, _mm256_shuffle_epi32(iacc_mat_01_1, 78), 204);
            __m256i iacc_row_1_1 = _mm256_blend_epi32(
              _mm256_shuffle_epi32(iacc_mat_00_1, 78), iacc_mat_01_1, 204);
            __m256i iacc_row_2_1 = _mm256_blend_epi32(
              iacc_mat_10_1, _mm256_shuffle_epi32(iacc_mat_11_1, 78), 204);
            __m256i iacc_row_3_1 = _mm256_blend_epi32(
              _mm256_shuffle_epi32(iacc_mat_10_1, 78), iacc_mat_11_1, 204);

            __m256i iacc_row_0 = _mm256_add_epi32(iacc_row_0_0, iacc_row_0_1);
            __m256i iacc_row_1 = _mm256_add_epi32(iacc_row_1_0, iacc_row_1_1);
            __m256i iacc_row_2 = _mm256_add_epi32(iacc_row_2_0, iacc_row_2_1);
            __m256i iacc_row_3 = _mm256_add_epi32(iacc_row_3_0, iacc_row_3_1);

            // Load the scale(d) values for all the 4 Q8_k blocks and repeat it
            // across lanes
            const __m128 row_scale_f32_sse = _mm_load_ps(a_ptrs[rp][b].d);
            const __m256 row_scale_f32 = _mm256_set_m128(
              row_scale_f32_sse,
              row_scale_f32_sse); // GGML_F32Cx8_REPEAT_LOAD(a_ptrs[rp][b].d,
                                  // loadMask);

            // Multiply with appropiate scales and accumulate (for both d and
            // dmin) below
            acc_rows[rp * 4] = _mm256_fmadd_ps(
              _mm256_cvtepi32_ps(iacc_row_0),
              _mm256_mul_ps(col_scale_f32,
                            _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)),
              acc_rows[rp * 4]);
            acc_rows[rp * 4 + 1] = _mm256_fmadd_ps(
              _mm256_cvtepi32_ps(iacc_row_1),
              _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(
                                             row_scale_f32, row_scale_f32, 85)),
              acc_rows[rp * 4 + 1]);
            acc_rows[rp * 4 + 2] = _mm256_fmadd_ps(
              _mm256_cvtepi32_ps(iacc_row_2),
              _mm256_mul_ps(
                col_scale_f32,
                _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)),
              acc_rows[rp * 4 + 2]);
            acc_rows[rp * 4 + 3] = _mm256_fmadd_ps(
              _mm256_cvtepi32_ps(iacc_row_3),
              _mm256_mul_ps(
                col_scale_f32,
                _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 255)),
              acc_rows[rp * 4 + 3]);

            __m256i iacc_row_min_0 = _mm256_madd_epi16(
              _mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 0), mins_01);
            __m256i iacc_row_min_1 = _mm256_madd_epi16(
              _mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 85), mins_01);
            __m256i iacc_row_min_2 = _mm256_madd_epi16(
              _mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 170), mins_01);
            __m256i iacc_row_min_3 = _mm256_madd_epi16(
              _mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 255), mins_01);

            acc_min_rows[rp * 4] = _mm256_fmadd_ps(
              _mm256_cvtepi32_ps(iacc_row_min_0),
              _mm256_mul_ps(col_dmin_f32,
                            _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)),
              acc_min_rows[rp * 4]);
            acc_min_rows[rp * 4 + 1] = _mm256_fmadd_ps(
              _mm256_cvtepi32_ps(iacc_row_min_1),
              _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32,
                                                            row_scale_f32, 85)),
              acc_min_rows[rp * 4 + 1]);
            acc_min_rows[rp * 4 + 2] = _mm256_fmadd_ps(
              _mm256_cvtepi32_ps(iacc_row_min_2),
              _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(
                                            row_scale_f32, row_scale_f32, 170)),
              acc_min_rows[rp * 4 + 2]);
            acc_min_rows[rp * 4 + 3] = _mm256_fmadd_ps(
              _mm256_cvtepi32_ps(iacc_row_min_3),
              _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(
                                            row_scale_f32, row_scale_f32, 255)),
              acc_min_rows[rp * 4 + 3]);
          }
        }
      }
      // Store the accumulated values
      for (int i = 0; i < 16; i++) {
        _mm256_storeu_ps((float *)(s + ((y * 4 + i) * bs + x * 8)),
                         _mm256_sub_ps(acc_rows[i], acc_min_rows[i]));
      }
    }
  }
  for (; y < nr / 4; y++) {

    const block_q8_Kx4 *a_ptr = a_ptr_start + (y * nb);

    for (int64_t x = xstart; x < nc / 8; x++) {

      const block_q4_Kx8 *b_ptr = b_ptr_start + (x * b_nb);

      // Master FP accumulators
      __m256 acc_rows[4];
      for (int i = 0; i < 4; i++) {
        acc_rows[i] = _mm256_setzero_ps();
      }

      __m256 acc_min_rows[4];
      for (int i = 0; i < 4; i++) {
        acc_min_rows[i] = _mm256_setzero_ps();
      }

      for (int64_t b = 0; b < nb; b++) {

        // Scale values - Load the eight scale values of block_q4_Kx8
        const __m256 col_scale_f32 = GGML_F32Cx8_LOAD(b_ptr[b].d);

        // dmin values - Load the eight dmin values of block_q4_Kx8
        const __m256 col_dmin_f32 = GGML_F32Cx8_LOAD(b_ptr[b].dmin);

        // Loop to iterate over the eight sub blocks of a super block - two sub
        // blocks are processed per iteration
        for (int sb = 0; sb < QK_K / 64; sb++) {

          // Load the eight block_q4_k for two sub blocks quantized values
          // interleaved with each other in chunks of eight bytes - B0,B1
          // ....B6,B7
          const __m256i rhs_raw_mat_0123_0 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + sb * 256));
          const __m256i rhs_raw_mat_4567_0 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 32 + sb * 256));
          const __m256i rhs_raw_mat_0123_1 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 64 + sb * 256));
          const __m256i rhs_raw_mat_4567_1 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 96 + sb * 256));
          const __m256i rhs_raw_mat_0123_2 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 128 + sb * 256));
          const __m256i rhs_raw_mat_4567_2 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 160 + sb * 256));
          const __m256i rhs_raw_mat_0123_3 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 192 + sb * 256));
          const __m256i rhs_raw_mat_4567_3 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 224 + sb * 256));

          // Save the values in the following vectors in the formats B0B1B4B5,
          // B2B3B6B7 for further processing and storing of values
          const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(
            rhs_raw_mat_0123_0,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder),
            240);
          const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder),
            rhs_raw_mat_4567_0, 240);
          const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(
            rhs_raw_mat_0123_1,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder),
            240);
          const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder),
            rhs_raw_mat_4567_1, 240);
          const __m256i rhs_raw_mat_0145_2 = _mm256_blend_epi32(
            rhs_raw_mat_0123_2,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_2, requiredOrder),
            240);
          const __m256i rhs_raw_mat_2367_2 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_0123_2, requiredOrder),
            rhs_raw_mat_4567_2, 240);
          const __m256i rhs_raw_mat_0145_3 = _mm256_blend_epi32(
            rhs_raw_mat_0123_3,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_3, requiredOrder),
            240);
          const __m256i rhs_raw_mat_2367_3 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_0123_3, requiredOrder),
            rhs_raw_mat_4567_3, 240);

          // 4-bit -> 8-bit
          // First sub block of the two sub blocks processed in the iteration
          const __m256i rhs_mat_0145_00 = _mm256_and_si256(
            rhs_raw_mat_0145_0, m4b); // B00(0-7) B01(0-7) B04(0-7) B05(0-7)
          const __m256i rhs_mat_2367_00 = _mm256_and_si256(
            rhs_raw_mat_2367_0, m4b); // B02(0-7) B03(0-7) B06(0-7) B07(0-7)

          const __m256i rhs_mat_0145_01 = _mm256_and_si256(
            rhs_raw_mat_0145_1, m4b); // B00(8-15) B01(8-15) B04(8-15) B05(8-15)
          const __m256i rhs_mat_2367_01 = _mm256_and_si256(
            rhs_raw_mat_2367_1, m4b); // B02(8-15) B03(8-15) B06(8-15) B07(8-15)

          const __m256i rhs_mat_0145_02 = _mm256_and_si256(
            rhs_raw_mat_0145_2,
            m4b); // B00(16-23) B01(16-23) B04(16-23) B05(16-23)
          const __m256i rhs_mat_2367_02 = _mm256_and_si256(
            rhs_raw_mat_2367_2,
            m4b); // B02(16-23) B03(16-23) B06(16-23) B07(16-23)

          const __m256i rhs_mat_0145_03 = _mm256_and_si256(
            rhs_raw_mat_0145_3,
            m4b); // B00(24-31) B01(24-31) B04(24-31) B05(24-31)
          const __m256i rhs_mat_2367_03 = _mm256_and_si256(
            rhs_raw_mat_2367_3,
            m4b); // B02(24-31) B03(24-31) B06(24-31) B07(24-31)

          // Second sub block of the two sub blocks processed in the iteration
          const __m256i rhs_mat_0145_10 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_0, 4),
                             m4b); // B10(0-7) B11(0-7) B14(0-7) B15(0-7)
          const __m256i rhs_mat_2367_10 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_0, 4),
                             m4b); // B12(0-7) B13(0-7) B16(0-7) B17(0-7)

          const __m256i rhs_mat_0145_11 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_1, 4),
                             m4b); // B10(8-15) B11(8-15) B14(8-15) B15(8-15)
          const __m256i rhs_mat_2367_11 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_1, 4),
                             m4b); // B12(8-15) B13(8-15) B16(8-15) B17(8-15)

          const __m256i rhs_mat_0145_12 = _mm256_and_si256(
            _mm256_srli_epi16(rhs_raw_mat_0145_2, 4),
            m4b); // B10(16-23) B11(16-23) B14(16-23) B15(16-23)
          const __m256i rhs_mat_2367_12 = _mm256_and_si256(
            _mm256_srli_epi16(rhs_raw_mat_2367_2, 4),
            m4b); // B12(16-23) B13(16-23) B16(16-23) B17(16-23)

          const __m256i rhs_mat_0145_13 = _mm256_and_si256(
            _mm256_srli_epi16(rhs_raw_mat_0145_3, 4),
            m4b); // B10(24-31) B11(24-31) B14(24-31) B15(24-31)
          const __m256i rhs_mat_2367_13 = _mm256_and_si256(
            _mm256_srli_epi16(rhs_raw_mat_2367_3, 4),
            m4b); // B12(24-31) B13(24-31) B16(24-31) B17(24-31)

          // Shuffle pattern one - right side input
          const __m256i rhs_mat_0145_00_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_00, 136); // B00(0-3) B01(0-3) B00(0-3) B01(0-3)
                                   // B04(0-3) B05(0-3) B04(0-3) B05(0-3)
          const __m256i rhs_mat_2367_00_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_00, 136); // B02(0-3) B03(0-3) B02(0-3) B03(0-3)
                                   // B06(0-3) B07(0-3) B06(0-3) B07(0-3)

          const __m256i rhs_mat_0145_01_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_01, 136); // B00(8-11) B01(8-11) B00(8-11) B01(8-11)
                                   // B04(8-11) B05(8-11) B04(8-11) B05(8-11)
          const __m256i rhs_mat_2367_01_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_01, 136); // B02(8-11) B03(8-11) B02(8-11) B03(8-11)
                                   // B06(8-11) B07(8-11) B06(8-11) B07(8-11)

          const __m256i rhs_mat_0145_02_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_02,
            136); // B00(16-19) B01(16-19) B00(16-19) B01(16-19) B04(16-19)
                  // B05(16-19) B04(16-19) B05(16-19)
          const __m256i rhs_mat_2367_02_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_02,
            136); // B02(16-19) B03(16-19) B02(16-19) B03(16-19) B06(16-19)
                  // B07(16-19) B06(16-19) B07(16-19)

          const __m256i rhs_mat_0145_03_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_03,
            136); // B00(24-27) B01(24-27) B00(24-27) B01(24-27) B04(24-27)
                  // B05(24-27) B04(24-27) B05(24-27)
          const __m256i rhs_mat_2367_03_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_03,
            136); // B02(24-27) B03(24-27) B02(24-27) B03(24-27) B06(24-27)
                  // B07(24-27) B06(24-27) B07(24-27)

          const __m256i rhs_mat_0145_10_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_10, 136); // B10(0-3) B11(0-3) B10(0-3) B11(0-3)
                                   // B14(0-3) B15(0-3) B14(0-3) B15(0-3)
          const __m256i rhs_mat_2367_10_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_10, 136); // B12(0-3) B13(0-3) B12(0-3) B13(0-3)
                                   // B16(0-3) B17(0-3) B16(0-3) B17(0-3)

          const __m256i rhs_mat_0145_11_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_11, 136); // B10(8-11) B11(8-11) B10(8-11) B11(8-11)
                                   // B14(8-11) B15(8-11) B14(8-11) B15(8-11)
          const __m256i rhs_mat_2367_11_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_11, 136); // B12(8-11) B13(8-11) B12(8-11) B13(8-11)
                                   // B16(8-11) B17(8-11) B16(8-11) B17(8-11)

          const __m256i rhs_mat_0145_12_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_12,
            136); // B10(16-19) B11(16-19) B10(16-19) B11(16-19) B14(16-19)
                  // B15(16-19) B14(16-19) B15(16-19)
          const __m256i rhs_mat_2367_12_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_12,
            136); // B12(16-19) B13(16-19) B12(16-19) B13(16-19) B16(16-19)
                  // B17(16-19) B16(16-19) B17(16-19)

          const __m256i rhs_mat_0145_13_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_13,
            136); // B10(24-27) B11(24-27) B10(24-27) B11(24-27) B14(24-27)
                  // B15(24-27) B14(24-27) B15(24-27)
          const __m256i rhs_mat_2367_13_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_13,
            136); // B12(24-27) B13(24-27) B12(24-27) B13(24-27) B16(24-27)
                  // B17(24-27) B16(24-27) B17(24-27)

          // Shuffle pattern two - right side input
          const __m256i rhs_mat_0145_00_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_00, 221); // B00(4-7) B01(4-7) B00(4-7) B01(4-7)
                                   // B04(4-7) B05(4-7) B04(4-7) B05(4-7)
          const __m256i rhs_mat_2367_00_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_00, 221); // B02(4-7) B03(4-7) B02(4-7) B03(4-7)
                                   // B06(4-7) B07(4-7) B06(4-7) B07(4-7)

          const __m256i rhs_mat_0145_01_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_01,
            221); // B00(12-15) B01(12-15) B00(12-15) B01(12-15) B04(12-15)
                  // B05(12-15) B04(12-15) B05(12-15)
          const __m256i rhs_mat_2367_01_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_01,
            221); // B02(12-15) B03(12-15) B02(12-15) B03(12-15) B06(12-15)
                  // B07(12-15) B06(12-15) B07(12-15)

          const __m256i rhs_mat_0145_02_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_02,
            221); // B00(20-23) B01(20-23) B00(20-23) B01(20-23) B04(20-23)
                  // B05(20-23) B04(20-23) B05(20-23)
          const __m256i rhs_mat_2367_02_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_02,
            221); // B02(20-23) B03(20-23) B02(20-23) B03(20-23) B06(20-23)
                  // B07(20-23) B06(20-23) B07(20-23)

          const __m256i rhs_mat_0145_03_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_03,
            221); // B00(28-31) B01(28-31) B00(28-31) B01(28-31) B04(28-31)
                  // B05(28-31) B04(28-31) B05(28-31)
          const __m256i rhs_mat_2367_03_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_03,
            221); // B02(28-31) B03(28-31) B02(28-31) B03(28-31) B06(28-31)
                  // B07(28-31) B06(28-31) B07(28-31)

          const __m256i rhs_mat_0145_10_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_10, 221); // B10(4-7) B11(4-7) B10(4-7) B11(4-7)
                                   // B14(4-7) B15(4-7) B14(4-7) B15(4-7)
          const __m256i rhs_mat_2367_10_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_10, 221); // B12(4-7) B13(4-7) B12(4-7) B13(4-7)
                                   // B16(4-7) B17(4-7) B16(4-7) B17(4-7)

          const __m256i rhs_mat_0145_11_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_11,
            221); // B10(12-15) B11(12-15) B10(12-15) B11(12-15) B14(12-15)
                  // B15(12-15) B14(12-15) B15(12-15)
          const __m256i rhs_mat_2367_11_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_11,
            221); // B12(12-15) B13(12-15) B12(12-15) B13(12-15) B16(12-15)
                  // B17(12-15) B16(12-15) B17(12-15)

          const __m256i rhs_mat_0145_12_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_12,
            221); // B10(20-23) B11(20-23) B10(20-23) B11(20-23) B14(20-23)
                  // B15(20-23) B14(20-23) B15(20-23)
          const __m256i rhs_mat_2367_12_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_12,
            221); // B12(20-23) B13(20-23) B12(20-23) B13(20-23) B16(20-23)
                  // B17(20-23) B16(20-23) B17(20-23)

          const __m256i rhs_mat_0145_13_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_13,
            221); // B10(28-31) B11(28-31) B10(28-31) B11(28-31) B14(28-31)
                  // B15(28-31) B14(28-31) B15(28-31)
          const __m256i rhs_mat_2367_13_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_13,
            221); // B12(28-31) B13(28-31) B12(28-31) B13(28-31) B16(28-31)
                  // B17(28-31) B16(28-31) B17(28-31)

          uint32_t utmp_0[4], utmp_1[4];

          // Scales and Mins of corresponding sub blocks from different Q4_K
          // structures are stored together The below block is for eg to extract
          // first sub block's scales and mins from different Q4_K structures
          // for the sb loop
          memcpy(utmp_0, b_ptr[b].scales + 24 * sb, 12);
          utmp_0[3] =
            ((utmp_0[2] >> 4) & kmask2) | (((utmp_0[1] >> 6) & kmask3) << 4);
          const uint32_t uaux_0 = utmp_0[1] & kmask1;
          utmp_0[1] = (utmp_0[2] & kmask2) | (((utmp_0[0] >> 6) & kmask3) << 4);
          utmp_0[2] = uaux_0;
          utmp_0[0] &= kmask1;

          // The below block is for eg to extract second sub block's scales and
          // mins from different Q4_K structures when sb = 1
          memcpy(utmp_1, b_ptr[b].scales + 12 + sb * 24, 12);
          utmp_1[3] =
            ((utmp_1[2] >> 4) & kmask2) | (((utmp_1[1] >> 6) & kmask3) << 4);
          const uint32_t uaux_1 = utmp_1[1] & kmask1;
          utmp_1[1] = (utmp_1[2] & kmask2) | (((utmp_1[0] >> 6) & kmask3) << 4);
          utmp_1[2] = uaux_1;
          utmp_1[0] &= kmask1;

          // Scales of first sub block in the sb loop
          const __m128i mins_and_scales_0 =
            _mm_set_epi32(utmp_0[3], utmp_0[2], utmp_0[1], utmp_0[0]);
          const __m256i scales_0 = _mm256_cvtepu8_epi16(
            _mm_unpacklo_epi8(mins_and_scales_0, mins_and_scales_0));

          // Scales of second sub block in the sb loop
          const __m128i mins_and_scales_1 =
            _mm_set_epi32(utmp_1[3], utmp_1[2], utmp_1[1], utmp_1[0]);
          const __m256i scales_1 = _mm256_cvtepu8_epi16(
            _mm_unpacklo_epi8(mins_and_scales_1, mins_and_scales_1));

          // Mins of first and second sub block of Q4_K block are arranged side
          // by side
          const __m256i mins_01 = _mm256_cvtepu8_epi16(
            _mm_unpacklo_epi8(_mm_shuffle_epi32(mins_and_scales_0, 78),
                              _mm_shuffle_epi32(mins_and_scales_1, 78)));

          const __m256i scale_0145_0 = _mm256_shuffle_epi32(scales_0, 68);
          const __m256i scale_2367_0 = _mm256_shuffle_epi32(scales_0, 238);

          const __m256i scale_0145_1 = _mm256_shuffle_epi32(scales_1, 68);
          const __m256i scale_2367_1 = _mm256_shuffle_epi32(scales_1, 238);

          // Load the four block_q8_k quantized values interleaved with each
          // other in chunks of eight bytes - A0,A1,A2,A3 Loaded as set of 128
          // bit vectors and repeated into a 256 bit vector
          __m256i lhs_mat_0123_00 =
            _mm256_loadu_si256((const __m256i *)((a_ptr[b].qs + 256 * sb)));
          __m256i lhs_mat_01_00 =
            _mm256_permute2f128_si256(lhs_mat_0123_00, lhs_mat_0123_00, 0);
          __m256i lhs_mat_23_00 =
            _mm256_permute2f128_si256(lhs_mat_0123_00, lhs_mat_0123_00, 17);
          __m256i lhs_mat_0123_01 = _mm256_loadu_si256(
            (const __m256i *)((a_ptr[b].qs + 32 + 256 * sb)));
          __m256i lhs_mat_01_01 =
            _mm256_permute2f128_si256(lhs_mat_0123_01, lhs_mat_0123_01, 0);
          __m256i lhs_mat_23_01 =
            _mm256_permute2f128_si256(lhs_mat_0123_01, lhs_mat_0123_01, 17);
          __m256i lhs_mat_0123_02 = _mm256_loadu_si256(
            (const __m256i *)((a_ptr[b].qs + 64 + 256 * sb)));
          __m256i lhs_mat_01_02 =
            _mm256_permute2f128_si256(lhs_mat_0123_02, lhs_mat_0123_02, 0);
          __m256i lhs_mat_23_02 =
            _mm256_permute2f128_si256(lhs_mat_0123_02, lhs_mat_0123_02, 17);
          __m256i lhs_mat_0123_03 = _mm256_loadu_si256(
            (const __m256i *)((a_ptr[b].qs + 96 + 256 * sb)));
          __m256i lhs_mat_01_03 =
            _mm256_permute2f128_si256(lhs_mat_0123_03, lhs_mat_0123_03, 0);
          __m256i lhs_mat_23_03 =
            _mm256_permute2f128_si256(lhs_mat_0123_03, lhs_mat_0123_03, 17);
          __m256i lhs_mat_0123_10 = _mm256_loadu_si256(
            (const __m256i *)((a_ptr[b].qs + 128 + 256 * sb)));
          __m256i lhs_mat_01_10 =
            _mm256_permute2f128_si256(lhs_mat_0123_10, lhs_mat_0123_10, 0);
          __m256i lhs_mat_23_10 =
            _mm256_permute2f128_si256(lhs_mat_0123_10, lhs_mat_0123_10, 17);
          __m256i lhs_mat_0123_11 = _mm256_loadu_si256(
            (const __m256i *)((a_ptr[b].qs + 160 + 256 * sb)));
          __m256i lhs_mat_01_11 =
            _mm256_permute2f128_si256(lhs_mat_0123_11, lhs_mat_0123_11, 0);
          __m256i lhs_mat_23_11 =
            _mm256_permute2f128_si256(lhs_mat_0123_11, lhs_mat_0123_11, 17);
          __m256i lhs_mat_0123_12 = _mm256_loadu_si256(
            (const __m256i *)((a_ptr[b].qs + 192 + 256 * sb)));
          __m256i lhs_mat_01_12 =
            _mm256_permute2f128_si256(lhs_mat_0123_12, lhs_mat_0123_12, 0);
          __m256i lhs_mat_23_12 =
            _mm256_permute2f128_si256(lhs_mat_0123_12, lhs_mat_0123_12, 17);
          __m256i lhs_mat_0123_13 = _mm256_loadu_si256(
            (const __m256i *)((a_ptr[b].qs + 224 + 256 * sb)));
          __m256i lhs_mat_01_13 =
            _mm256_permute2f128_si256(lhs_mat_0123_13, lhs_mat_0123_13, 0);
          __m256i lhs_mat_23_13 =
            _mm256_permute2f128_si256(lhs_mat_0123_13, lhs_mat_0123_13, 17);

          // Bsums are loaded - four bsums are loaded (for two sub blocks) for
          // the different Q8_K blocks
          __m256i lhs_bsums_0123_01 =
            _mm256_loadu_si256((const __m256i *)((a_ptr[b].bsums + 16 * sb)));
          __m256i lhs_bsums_hsum_0123_01 = _mm256_castsi128_si256(
            _mm_hadd_epi16(_mm256_castsi256_si128(lhs_bsums_0123_01),
                           _mm256_extractf128_si256(lhs_bsums_0123_01, 1)));
          lhs_bsums_hsum_0123_01 = _mm256_permute2x128_si256(
            lhs_bsums_hsum_0123_01, lhs_bsums_hsum_0123_01, 0);

          // Shuffle pattern one - left side input
          const __m256i lhs_mat_01_00_sp1 = _mm256_shuffle_epi32(
            lhs_mat_01_00, 160); // A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3)
                                 // A00(0-3) A01(0-3) A01(0-3)
          const __m256i lhs_mat_23_00_sp1 = _mm256_shuffle_epi32(
            lhs_mat_23_00, 160); // A02(0-3) A03(0-3) A02(0-3) A03(0-3) A02(0-3)
                                 // A03(0-3) A02(0-3) A03(0-3)

          const __m256i lhs_mat_01_01_sp1 = _mm256_shuffle_epi32(
            lhs_mat_01_01, 160); // A00(8-11) A00(8-11) A01(8-11) A01(8-11)
                                 // A00(8-11) A00(8-11) A01(8-11) A01(8-11)
          const __m256i lhs_mat_23_01_sp1 = _mm256_shuffle_epi32(
            lhs_mat_23_01, 160); // A02(8-11) A03(8-11) A02(8-11) A03(8-11)
                                 // A02(8-11) A03(8-11) A02(8-11) A03(8-11)

          const __m256i lhs_mat_01_02_sp1 = _mm256_shuffle_epi32(
            lhs_mat_01_02, 160); // A00(16-19) A00(16-19) A01(16-19) A01(16-19)
                                 // A00(16-19) A00(16-19) A01(16-19) A01(16-19)
          const __m256i lhs_mat_23_02_sp1 = _mm256_shuffle_epi32(
            lhs_mat_23_02, 160); // A02(16-19) A03(16-19) A02(16-19) A03(16-19)
                                 // A02(16-19) A03(16-19) A02(16-19) A03(16-19)

          const __m256i lhs_mat_01_03_sp1 = _mm256_shuffle_epi32(
            lhs_mat_01_03, 160); // A00(24-27) A00(24-27) A01(24-27) A01(24-27)
                                 // A00(24-27) A00(24-27) A01(24-27) A01(24-27)
          const __m256i lhs_mat_23_03_sp1 = _mm256_shuffle_epi32(
            lhs_mat_23_03, 160); // A02(24-27) A03(24-27) A02(24-27) A03(24-27)
                                 // A02(24-27) A03(24-27) A02(24-27) A03(24-27)

          const __m256i lhs_mat_01_10_sp1 = _mm256_shuffle_epi32(
            lhs_mat_01_10, 160); // A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3)
                                 // A10(0-3) A11(0-3) A11(0-3)
          const __m256i lhs_mat_23_10_sp1 = _mm256_shuffle_epi32(
            lhs_mat_23_10, 160); // A12(0-3) A13(0-3) A12(0-3) A13(0-3) A12(0-3)
                                 // A13(0-3) A12(0-3) A13(0-3)

          const __m256i lhs_mat_01_11_sp1 = _mm256_shuffle_epi32(
            lhs_mat_01_11, 160); // A10(8-11) A10(8-11) A11(8-11) A11(8-11)
                                 // A10(8-11) A10(8-11) A11(8-11) A11(8-11)
          const __m256i lhs_mat_23_11_sp1 = _mm256_shuffle_epi32(
            lhs_mat_23_11, 160); // A12(8-11) A13(8-11) A12(8-11) A13(8-11)
                                 // A12(8-11) A13(8-11) A12(8-11) A13(8-11)

          const __m256i lhs_mat_01_12_sp1 = _mm256_shuffle_epi32(
            lhs_mat_01_12, 160); // A10(16-19) A10(16-19) A11(16-19) A11(16-19)
                                 // A10(16-19) A10(16-19) A11(16-19) A11(16-19)
          const __m256i lhs_mat_23_12_sp1 = _mm256_shuffle_epi32(
            lhs_mat_23_12, 160); // A12(16-19) A13(16-19) A12(16-19) A13(16-19)
                                 // A12(16-19) A13(16-19) A12(16-19) A13(16-19)

          const __m256i lhs_mat_01_13_sp1 = _mm256_shuffle_epi32(
            lhs_mat_01_13, 160); // A10(24-27) A10(24-27) A11(24-27) A11(24-27)
                                 // A10(24-27) A10(24-27) A11(24-27) A11(24-27)
          const __m256i lhs_mat_23_13_sp1 = _mm256_shuffle_epi32(
            lhs_mat_23_13, 160); // A12(24-27) A13(24-27) A12(24-27) A13(24-27)
                                 // A12(24-27) A13(24-27) A12(24-27) A13(24-27)

          // Shuffle pattern two- left side input
          const __m256i lhs_mat_01_00_sp2 = _mm256_shuffle_epi32(
            lhs_mat_01_00, 245); // A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7)
                                 // A00(4-7) A01(4-7) A01(4-7)
          const __m256i lhs_mat_23_00_sp2 = _mm256_shuffle_epi32(
            lhs_mat_23_00, 245); // A02(4-7) A03(4-7) A02(4-7) A03(4-7) A02(4-7)
                                 // A03(4-7) A02(4-7) A03(4-7)

          const __m256i lhs_mat_01_01_sp2 = _mm256_shuffle_epi32(
            lhs_mat_01_01, 245); // A00(12-15) A00(12-15) A01(12-15) A01(12-15)
                                 // A00(12-15) A00(12-15) A01(12-15) A01(12-15)
          const __m256i lhs_mat_23_01_sp2 = _mm256_shuffle_epi32(
            lhs_mat_23_01, 245); // A02(12-15) A03(12-15) A02(12-15) A03(12-15)
                                 // A02(12-15) A03(12-15) A02(12-15) A03(12-15)

          const __m256i lhs_mat_01_02_sp2 = _mm256_shuffle_epi32(
            lhs_mat_01_02, 245); // A00(20-23) A00(20-23) A01(20-23) A01(20-23)
                                 // A00(20-23) A00(20-23) A01(20-23) A01(20-23)
          const __m256i lhs_mat_23_02_sp2 = _mm256_shuffle_epi32(
            lhs_mat_23_02, 245); // A02(20-23) A03(20-23) A02(20-23) A03(20-23)
                                 // A02(20-23) A03(20-23) A02(20-23) A03(20-23)

          const __m256i lhs_mat_01_03_sp2 = _mm256_shuffle_epi32(
            lhs_mat_01_03, 245); // A00(28-31) A00(28-31) A01(28-31) A01(28-31)
                                 // A00(28-31) A00(28-31) A01(28-31) A01(28-31)
          const __m256i lhs_mat_23_03_sp2 = _mm256_shuffle_epi32(
            lhs_mat_23_03, 245); // A02(28-31) A03(28-31) A02(28-31) A03(28-31)
                                 // A02(28-31) A03(28-31) A02(28-31) A03(28-31)

          const __m256i lhs_mat_01_10_sp2 = _mm256_shuffle_epi32(
            lhs_mat_01_10, 245); // A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7)
                                 // A10(4-7) A11(4-7) A11(4-7)
          const __m256i lhs_mat_23_10_sp2 = _mm256_shuffle_epi32(
            lhs_mat_23_10, 245); // A12(4-7) A13(4-7) A12(4-7) A13(4-7) A12(4-7)
                                 // A13(4-7) A12(4-7) A13(4-7)

          const __m256i lhs_mat_01_11_sp2 = _mm256_shuffle_epi32(
            lhs_mat_01_11, 245); // A10(12-15) A10(12-15) A11(12-15) A11(12-15)
                                 // A10(12-15) A10(12-15) A11(12-15) A11(12-15)
          const __m256i lhs_mat_23_11_sp2 = _mm256_shuffle_epi32(
            lhs_mat_23_11, 245); // A12(12-15) A13(12-15) A12(12-15) A13(12-15)
                                 // A12(12-15) A13(12-15) A12(12-15) A13(12-15)

          const __m256i lhs_mat_01_12_sp2 = _mm256_shuffle_epi32(
            lhs_mat_01_12, 245); // A10(20-23) A10(20-23) A11(20-23) A11(20-23)
                                 // A10(20-23) A10(20-23) A11(20-23) A11(20-23)
          const __m256i lhs_mat_23_12_sp2 = _mm256_shuffle_epi32(
            lhs_mat_23_12, 245); // A12(20-23) A13(20-23) A12(20-23) A13(20-23)
                                 // A12(20-23) A13(20-23) A12(20-23) A13(20-23)

          const __m256i lhs_mat_01_13_sp2 = _mm256_shuffle_epi32(
            lhs_mat_01_13, 245); // A10(28-31) A10(28-31) A11(28-31) A11(28-31)
                                 // A10(28-31) A10(28-31) A11(28-31) A11(28-31)
          const __m256i lhs_mat_23_13_sp2 = _mm256_shuffle_epi32(
            lhs_mat_23_13, 245); // A12(28-31) A13(28-31) A12(28-31) A13(28-31)
                                 // A12(28-31) A13(28-31) A12(28-31) A13(28-31)

          // The values arranged in shuffle patterns are operated with dot
          // product operation within 32 bit lane i.e corresponding bytes and
          // multiplied and added into 32 bit integers within 32 bit lane
          __m256i iacc_mat_00_0_sp1 = _mm256_add_epi16(
            _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_maddubs_epi16(rhs_mat_0145_03_sp1, lhs_mat_01_03_sp1),
                _mm256_maddubs_epi16(rhs_mat_0145_02_sp1, lhs_mat_01_02_sp1)),
              _mm256_maddubs_epi16(rhs_mat_0145_01_sp1, lhs_mat_01_01_sp1)),
            _mm256_maddubs_epi16(rhs_mat_0145_00_sp1, lhs_mat_01_00_sp1));
          __m256i iacc_mat_01_0_sp1 = _mm256_add_epi16(
            _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_maddubs_epi16(rhs_mat_2367_03_sp1, lhs_mat_01_03_sp1),
                _mm256_maddubs_epi16(rhs_mat_2367_02_sp1, lhs_mat_01_02_sp1)),
              _mm256_maddubs_epi16(rhs_mat_2367_01_sp1, lhs_mat_01_01_sp1)),
            _mm256_maddubs_epi16(rhs_mat_2367_00_sp1, lhs_mat_01_00_sp1));
          __m256i iacc_mat_10_0_sp1 = _mm256_add_epi16(
            _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_maddubs_epi16(rhs_mat_0145_03_sp1, lhs_mat_23_03_sp1),
                _mm256_maddubs_epi16(rhs_mat_0145_02_sp1, lhs_mat_23_02_sp1)),
              _mm256_maddubs_epi16(rhs_mat_0145_01_sp1, lhs_mat_23_01_sp1)),
            _mm256_maddubs_epi16(rhs_mat_0145_00_sp1, lhs_mat_23_00_sp1));
          __m256i iacc_mat_11_0_sp1 = _mm256_add_epi16(
            _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_maddubs_epi16(rhs_mat_2367_03_sp1, lhs_mat_23_03_sp1),
                _mm256_maddubs_epi16(rhs_mat_2367_02_sp1, lhs_mat_23_02_sp1)),
              _mm256_maddubs_epi16(rhs_mat_2367_01_sp1, lhs_mat_23_01_sp1)),
            _mm256_maddubs_epi16(rhs_mat_2367_00_sp1, lhs_mat_23_00_sp1));
          __m256i iacc_mat_00_1_sp1 = _mm256_add_epi16(
            _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_maddubs_epi16(rhs_mat_0145_13_sp1, lhs_mat_01_13_sp1),
                _mm256_maddubs_epi16(rhs_mat_0145_12_sp1, lhs_mat_01_12_sp1)),
              _mm256_maddubs_epi16(rhs_mat_0145_11_sp1, lhs_mat_01_11_sp1)),
            _mm256_maddubs_epi16(rhs_mat_0145_10_sp1, lhs_mat_01_10_sp1));
          __m256i iacc_mat_01_1_sp1 = _mm256_add_epi16(
            _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_maddubs_epi16(rhs_mat_2367_13_sp1, lhs_mat_01_13_sp1),
                _mm256_maddubs_epi16(rhs_mat_2367_12_sp1, lhs_mat_01_12_sp1)),
              _mm256_maddubs_epi16(rhs_mat_2367_11_sp1, lhs_mat_01_11_sp1)),
            _mm256_maddubs_epi16(rhs_mat_2367_10_sp1, lhs_mat_01_10_sp1));
          __m256i iacc_mat_10_1_sp1 = _mm256_add_epi16(
            _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_maddubs_epi16(rhs_mat_0145_13_sp1, lhs_mat_23_13_sp1),
                _mm256_maddubs_epi16(rhs_mat_0145_12_sp1, lhs_mat_23_12_sp1)),
              _mm256_maddubs_epi16(rhs_mat_0145_11_sp1, lhs_mat_23_11_sp1)),
            _mm256_maddubs_epi16(rhs_mat_0145_10_sp1, lhs_mat_23_10_sp1));
          __m256i iacc_mat_11_1_sp1 = _mm256_add_epi16(
            _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_maddubs_epi16(rhs_mat_2367_13_sp1, lhs_mat_23_13_sp1),
                _mm256_maddubs_epi16(rhs_mat_2367_12_sp1, lhs_mat_23_12_sp1)),
              _mm256_maddubs_epi16(rhs_mat_2367_11_sp1, lhs_mat_23_11_sp1)),
            _mm256_maddubs_epi16(rhs_mat_2367_10_sp1, lhs_mat_23_10_sp1));

          __m256i iacc_mat_00_0_sp2 = _mm256_add_epi16(
            _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_maddubs_epi16(rhs_mat_0145_03_sp2, lhs_mat_01_03_sp2),
                _mm256_maddubs_epi16(rhs_mat_0145_02_sp2, lhs_mat_01_02_sp2)),
              _mm256_maddubs_epi16(rhs_mat_0145_01_sp2, lhs_mat_01_01_sp2)),
            _mm256_maddubs_epi16(rhs_mat_0145_00_sp2, lhs_mat_01_00_sp2));
          __m256i iacc_mat_01_0_sp2 = _mm256_add_epi16(
            _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_maddubs_epi16(rhs_mat_2367_03_sp2, lhs_mat_01_03_sp2),
                _mm256_maddubs_epi16(rhs_mat_2367_02_sp2, lhs_mat_01_02_sp2)),
              _mm256_maddubs_epi16(rhs_mat_2367_01_sp2, lhs_mat_01_01_sp2)),
            _mm256_maddubs_epi16(rhs_mat_2367_00_sp2, lhs_mat_01_00_sp2));
          __m256i iacc_mat_10_0_sp2 = _mm256_add_epi16(
            _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_maddubs_epi16(rhs_mat_0145_03_sp2, lhs_mat_23_03_sp2),
                _mm256_maddubs_epi16(rhs_mat_0145_02_sp2, lhs_mat_23_02_sp2)),
              _mm256_maddubs_epi16(rhs_mat_0145_01_sp2, lhs_mat_23_01_sp2)),
            _mm256_maddubs_epi16(rhs_mat_0145_00_sp2, lhs_mat_23_00_sp2));
          __m256i iacc_mat_11_0_sp2 = _mm256_add_epi16(
            _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_maddubs_epi16(rhs_mat_2367_03_sp2, lhs_mat_23_03_sp2),
                _mm256_maddubs_epi16(rhs_mat_2367_02_sp2, lhs_mat_23_02_sp2)),
              _mm256_maddubs_epi16(rhs_mat_2367_01_sp2, lhs_mat_23_01_sp2)),
            _mm256_maddubs_epi16(rhs_mat_2367_00_sp2, lhs_mat_23_00_sp2));
          __m256i iacc_mat_00_1_sp2 = _mm256_add_epi16(
            _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_maddubs_epi16(rhs_mat_0145_13_sp2, lhs_mat_01_13_sp2),
                _mm256_maddubs_epi16(rhs_mat_0145_12_sp2, lhs_mat_01_12_sp2)),
              _mm256_maddubs_epi16(rhs_mat_0145_11_sp2, lhs_mat_01_11_sp2)),
            _mm256_maddubs_epi16(rhs_mat_0145_10_sp2, lhs_mat_01_10_sp2));
          __m256i iacc_mat_01_1_sp2 = _mm256_add_epi16(
            _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_maddubs_epi16(rhs_mat_2367_13_sp2, lhs_mat_01_13_sp2),
                _mm256_maddubs_epi16(rhs_mat_2367_12_sp2, lhs_mat_01_12_sp2)),
              _mm256_maddubs_epi16(rhs_mat_2367_11_sp2, lhs_mat_01_11_sp2)),
            _mm256_maddubs_epi16(rhs_mat_2367_10_sp2, lhs_mat_01_10_sp2));
          __m256i iacc_mat_10_1_sp2 = _mm256_add_epi16(
            _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_maddubs_epi16(rhs_mat_0145_13_sp2, lhs_mat_23_13_sp2),
                _mm256_maddubs_epi16(rhs_mat_0145_12_sp2, lhs_mat_23_12_sp2)),
              _mm256_maddubs_epi16(rhs_mat_0145_11_sp2, lhs_mat_23_11_sp2)),
            _mm256_maddubs_epi16(rhs_mat_0145_10_sp2, lhs_mat_23_10_sp2));
          __m256i iacc_mat_11_1_sp2 = _mm256_add_epi16(
            _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_maddubs_epi16(rhs_mat_2367_13_sp2, lhs_mat_23_13_sp2),
                _mm256_maddubs_epi16(rhs_mat_2367_12_sp2, lhs_mat_23_12_sp2)),
              _mm256_maddubs_epi16(rhs_mat_2367_11_sp2, lhs_mat_23_11_sp2)),
            _mm256_maddubs_epi16(rhs_mat_2367_10_sp2, lhs_mat_23_10_sp2));

          // Output of both shuffle patterns are added in order to sum dot
          // product outputs of all 32 values in block
          __m256i iacc_mat_00_0 =
            _mm256_add_epi16(iacc_mat_00_0_sp1, iacc_mat_00_0_sp2);
          __m256i iacc_mat_01_0 =
            _mm256_add_epi16(iacc_mat_01_0_sp1, iacc_mat_01_0_sp2);
          __m256i iacc_mat_10_0 =
            _mm256_add_epi16(iacc_mat_10_0_sp1, iacc_mat_10_0_sp2);
          __m256i iacc_mat_11_0 =
            _mm256_add_epi16(iacc_mat_11_0_sp1, iacc_mat_11_0_sp2);

          __m256i iacc_mat_00_1 =
            _mm256_add_epi16(iacc_mat_00_1_sp1, iacc_mat_00_1_sp2);
          __m256i iacc_mat_01_1 =
            _mm256_add_epi16(iacc_mat_01_1_sp1, iacc_mat_01_1_sp2);
          __m256i iacc_mat_10_1 =
            _mm256_add_epi16(iacc_mat_10_1_sp1, iacc_mat_10_1_sp2);
          __m256i iacc_mat_11_1 =
            _mm256_add_epi16(iacc_mat_11_1_sp1, iacc_mat_11_1_sp2);

          // Output of both shuffle patterns are added in order to sum dot
          // product outputs of all 32 values in block
          iacc_mat_00_0 = _mm256_madd_epi16(iacc_mat_00_0, scale_0145_0);
          iacc_mat_01_0 = _mm256_madd_epi16(iacc_mat_01_0, scale_2367_0);
          iacc_mat_10_0 = _mm256_madd_epi16(iacc_mat_10_0, scale_0145_0);
          iacc_mat_11_0 = _mm256_madd_epi16(iacc_mat_11_0, scale_2367_0);

          iacc_mat_00_1 = _mm256_madd_epi16(iacc_mat_00_1, scale_0145_1);
          iacc_mat_01_1 = _mm256_madd_epi16(iacc_mat_01_1, scale_2367_1);
          iacc_mat_10_1 = _mm256_madd_epi16(iacc_mat_10_1, scale_0145_1);
          iacc_mat_11_1 = _mm256_madd_epi16(iacc_mat_11_1, scale_2367_1);

          // Straighten out to make 4 row vectors (4 for each sub block which
          // are accumulated together in the next step)
          __m256i iacc_row_0_0 = _mm256_blend_epi32(
            iacc_mat_00_0, _mm256_shuffle_epi32(iacc_mat_01_0, 78), 204);
          __m256i iacc_row_1_0 = _mm256_blend_epi32(
            _mm256_shuffle_epi32(iacc_mat_00_0, 78), iacc_mat_01_0, 204);
          __m256i iacc_row_2_0 = _mm256_blend_epi32(
            iacc_mat_10_0, _mm256_shuffle_epi32(iacc_mat_11_0, 78), 204);
          __m256i iacc_row_3_0 = _mm256_blend_epi32(
            _mm256_shuffle_epi32(iacc_mat_10_0, 78), iacc_mat_11_0, 204);
          __m256i iacc_row_0_1 = _mm256_blend_epi32(
            iacc_mat_00_1, _mm256_shuffle_epi32(iacc_mat_01_1, 78), 204);
          __m256i iacc_row_1_1 = _mm256_blend_epi32(
            _mm256_shuffle_epi32(iacc_mat_00_1, 78), iacc_mat_01_1, 204);
          __m256i iacc_row_2_1 = _mm256_blend_epi32(
            iacc_mat_10_1, _mm256_shuffle_epi32(iacc_mat_11_1, 78), 204);
          __m256i iacc_row_3_1 = _mm256_blend_epi32(
            _mm256_shuffle_epi32(iacc_mat_10_1, 78), iacc_mat_11_1, 204);

          __m256i iacc_row_0 = _mm256_add_epi32(iacc_row_0_0, iacc_row_0_1);
          __m256i iacc_row_1 = _mm256_add_epi32(iacc_row_1_0, iacc_row_1_1);
          __m256i iacc_row_2 = _mm256_add_epi32(iacc_row_2_0, iacc_row_2_1);
          __m256i iacc_row_3 = _mm256_add_epi32(iacc_row_3_0, iacc_row_3_1);

          // Load the scale(d) values for all the 4 Q8_k blocks and repeat it
          // across lanes
          const __m128 row_scale_f32_sse = _mm_load_ps(a_ptr[b].d);
          const __m256 row_scale_f32 = _mm256_set_m128(
            row_scale_f32_sse,
            row_scale_f32_sse); // GGML_F32Cx8_REPEAT_LOAD(a_ptrs[rp][b].d,
                                // loadMask);

          // Multiply with appropiate scales and accumulate (for both d and
          // dmin) below
          acc_rows[0] = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(iacc_row_0),
            _mm256_mul_ps(col_scale_f32,
                          _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)),
            acc_rows[0]);
          acc_rows[1] = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(iacc_row_1),
            _mm256_mul_ps(col_scale_f32,
                          _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 85)),
            acc_rows[1]);
          acc_rows[2] = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(iacc_row_2),
            _mm256_mul_ps(col_scale_f32,
                          _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)),
            acc_rows[2]);
          acc_rows[3] = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(iacc_row_3),
            _mm256_mul_ps(col_scale_f32,
                          _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 255)),
            acc_rows[3]);

          __m256i iacc_row_min_0 = _mm256_madd_epi16(
            _mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 0), mins_01);
          __m256i iacc_row_min_1 = _mm256_madd_epi16(
            _mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 85), mins_01);
          __m256i iacc_row_min_2 = _mm256_madd_epi16(
            _mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 170), mins_01);
          __m256i iacc_row_min_3 = _mm256_madd_epi16(
            _mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 255), mins_01);

          acc_min_rows[0] = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(iacc_row_min_0),
            _mm256_mul_ps(col_dmin_f32,
                          _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)),
            acc_min_rows[0]);
          acc_min_rows[1] = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(iacc_row_min_1),
            _mm256_mul_ps(col_dmin_f32,
                          _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 85)),
            acc_min_rows[1]);
          acc_min_rows[2] = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(iacc_row_min_2),
            _mm256_mul_ps(col_dmin_f32,
                          _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)),
            acc_min_rows[2]);
          acc_min_rows[3] = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(iacc_row_min_3),
            _mm256_mul_ps(col_dmin_f32,
                          _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 255)),
            acc_min_rows[3]);
        }
      }

      // Store the accumulated values
      for (int i = 0; i < 4; i++) {
        _mm256_storeu_ps((float *)(s + ((y * 4 + i) * bs + x * 8)),
                         _mm256_sub_ps(acc_rows[i], acc_min_rows[i]));
      }
    }
  }
}

void nntr_gemv_q4_K_8x8_q8_K(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc) {
  const int qk = QK_K;
  const int nb = n / qk;
  const int ncols_interleaved = 8;
  const int blocklen = 8;
  static const uint32_t kmask1 = 0x3f3f3f3f;
  static const uint32_t kmask2 = 0x0f0f0f0f;
  static const uint32_t kmask3 = 0x03030303;

  assert(n % qk == 0);
  assert(nc % ncols_interleaved == 0);

  // Lookup table to convert signed nibbles to signed bytes
  __m256i signextendlut = _mm256_castsi128_si256(
    _mm_set_epi8(-1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0));
  signextendlut = _mm256_permute2f128_si256(signextendlut, signextendlut, 0);
  // Shuffle masks to rearrange delta and scale values to multiply with
  // appropriate scales
  __m128i deltamask =
    _mm_set_epi8(15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0);
  __m128i scalemask =
    _mm_set_epi8(7, 7, 3, 3, 6, 6, 2, 2, 5, 5, 1, 1, 4, 4, 0, 0);
  // Permute mask used for easier vector processing at later stages
  __m256i finalpermutemask = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);

  // Mask to extract nibbles from bytes
  const __m256i m4b = _mm256_set1_epi8(0x0F);

  int64_t b_nb = n / QK_K;

  const block_q4_Kx8 *b_ptr_start = (const block_q4_Kx8 *)vx;
  const block_q8_K *a_ptr_start = (const block_q8_K *)vy;

  // Process Q8_K blocks one by one
  for (int64_t y = 0; y < nr; y++) {

    // Pointers to LHS blocks of block_q8_K format
    const block_q8_K *a_ptr = a_ptr_start + (y * nb);

    // Take group of eight interleaved block_q4_K structures at each pass of the
    // loop and perform dot product operation
    for (int64_t x = 0; x < nc / 8; x++) {

      // Pointers to RHS blocks
      const block_q4_Kx8 *b_ptr = b_ptr_start + (x * b_nb);

      // Master FP accumulators
      __m256 acc_row = _mm256_setzero_ps();
      __m256 acc_min_rows = _mm256_setzero_ps();

      for (int64_t b = 0; b < nb; b++) {

        // Load and convert to FP32 scale from block_q8_K
        const __m256 row_scale_f32 = _mm256_set1_ps((a_ptr[b].d));

        // Load the scale values for the 8 blocks interleaved in block_q4_Kx8
        // col_scale_f32 rearranged so as to multiply with appropriate quants
        const __m256 col_scale_f32 =
          GGML_F32Cx8_REARRANGE_LOAD(b_ptr[b].d, deltamask);
        const __m256 col_dmin_f32 = GGML_F32Cx8_LOAD(b_ptr[b].dmin);

        __m256i iacc_b = _mm256_setzero_si256();
        __m256i iacc_min_b = _mm256_setzero_si256();

        const __m256i q8sums =
          _mm256_loadu_si256((const __m256i *)(a_ptr[b].bsums));
        __m256i q8s = _mm256_castsi128_si256(_mm_hadd_epi16(
          _mm256_castsi256_si128(q8sums), _mm256_extracti128_si256(q8sums, 1)));
        q8s = _mm256_permute2f128_si256(q8s, q8s, 0);

        // Processes two sub blocks from each Q4_K in each iteration
        for (int sb = 0; sb < QK_K / 64; sb++) {

          // Load the eight block_q4_K for two sub blocks quantized values
          // interleaved with each other in chunks of eight - B0,B1 ....B6,B7
          const __m256i rhs_raw_vec_0123_0 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + sb * 256));
          const __m256i rhs_raw_vec_4567_0 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 32 + sb * 256));
          const __m256i rhs_raw_vec_0123_1 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 64 + sb * 256));
          const __m256i rhs_raw_vec_4567_1 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 96 + sb * 256));
          const __m256i rhs_raw_vec_0123_2 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 128 + sb * 256));
          const __m256i rhs_raw_vec_4567_2 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 160 + sb * 256));
          const __m256i rhs_raw_vec_0123_3 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 192 + sb * 256));
          const __m256i rhs_raw_vec_4567_3 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 224 + sb * 256));

          // 4-bit -> 8-bit
          // Values of the first sub block of eight block_q4_K structures for
          // the sb loop
          const __m256i rhs_vec_0123_00 =
            _mm256_and_si256(rhs_raw_vec_0123_0, m4b);
          const __m256i rhs_vec_4567_00 =
            _mm256_and_si256(rhs_raw_vec_4567_0, m4b);
          const __m256i rhs_vec_0123_01 =
            _mm256_and_si256(rhs_raw_vec_0123_1, m4b);
          const __m256i rhs_vec_4567_01 =
            _mm256_and_si256(rhs_raw_vec_4567_1, m4b);
          const __m256i rhs_vec_0123_02 =
            _mm256_and_si256(rhs_raw_vec_0123_2, m4b);
          const __m256i rhs_vec_4567_02 =
            _mm256_and_si256(rhs_raw_vec_4567_2, m4b);
          const __m256i rhs_vec_0123_03 =
            _mm256_and_si256(rhs_raw_vec_0123_3, m4b);
          const __m256i rhs_vec_4567_03 =
            _mm256_and_si256(rhs_raw_vec_4567_3, m4b);

          // Values of the second sub block of eight block_q4_K structures when
          // sb = 1
          const __m256i rhs_vec_0123_10 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_0, 4), m4b);
          const __m256i rhs_vec_4567_10 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_0, 4), m4b);
          const __m256i rhs_vec_0123_11 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_1, 4), m4b);
          const __m256i rhs_vec_4567_11 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_1, 4), m4b);
          const __m256i rhs_vec_0123_12 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_2, 4), m4b);
          const __m256i rhs_vec_4567_12 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_2, 4), m4b);
          const __m256i rhs_vec_0123_13 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_3, 4), m4b);
          const __m256i rhs_vec_4567_13 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_3, 4), m4b);

          uint32_t utmp_0[4], utmp_1[4];

          // Scales and Mins of corresponding sub blocks from different Q8_K
          // structures are stored together The below block is for eg to extract
          // first sub block's scales and mins from different Q4_K structures
          // for the sb loop
          memcpy(utmp_0, b_ptr[b].scales + 24 * sb, 12);
          utmp_0[3] =
            ((utmp_0[2] >> 4) & kmask2) | (((utmp_0[1] >> 6) & kmask3) << 4);
          const uint32_t uaux_0 = utmp_0[1] & kmask1;
          utmp_0[1] = (utmp_0[2] & kmask2) | (((utmp_0[0] >> 6) & kmask3) << 4);
          utmp_0[2] = uaux_0;
          utmp_0[0] &= kmask1;

          // The below block is for eg to extract second sub block's scales and
          // mins from different Q4_K structures for the sb loop
          memcpy(utmp_1, b_ptr[b].scales + 12 + sb * 24, 12);
          utmp_1[3] =
            ((utmp_1[2] >> 4) & kmask2) | (((utmp_1[1] >> 6) & kmask3) << 4);
          const uint32_t uaux_1 = utmp_1[1] & kmask1;
          utmp_1[1] = (utmp_1[2] & kmask2) | (((utmp_1[0] >> 6) & kmask3) << 4);
          utmp_1[2] = uaux_1;
          utmp_1[0] &= kmask1;

          // Scales of first sub block in the sb loop
          const __m128i mins_and_scales_0 =
            _mm_set_epi32(utmp_0[3], utmp_0[2], utmp_0[1], utmp_0[0]);
          __m128i scales_rearrange_0 =
            _mm_shuffle_epi8(mins_and_scales_0, scalemask);
          __m256i scales_0 = _mm256_cvtepu8_epi16(scales_rearrange_0);

          // Scales of second sub block in the sb loop
          __m128i mins_and_scales_1 =
            _mm_set_epi32(utmp_1[3], utmp_1[2], utmp_1[1], utmp_1[0]);
          __m128i scales_rearrange_1 =
            _mm_shuffle_epi8(mins_and_scales_1, scalemask);
          __m256i scales_1 = _mm256_cvtepu8_epi16(scales_rearrange_1);

          // Mins of first and second sub block of Q4_K block are arranged side
          // by side
          __m256i mins_01 = _mm256_cvtepu8_epi16(
            _mm_unpacklo_epi8(_mm_shuffle_epi32(mins_and_scales_0, 78),
                              _mm_shuffle_epi32(mins_and_scales_1, 78)));

          // Load the two sub block values corresponding to sb in block_q8_K in
          // batches of 16 bytes and replicate the same across 256 bit vector
          __m256i lhs_vec_00 = _mm256_castsi128_si256(
            _mm_loadu_si128((const __m128i *)(a_ptr[b].qs + sb * 64)));
          __m256i lhs_vec_01 = _mm256_castsi128_si256(
            _mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 16 + sb * 64)));
          __m256i lhs_vec_10 = _mm256_castsi128_si256(
            _mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 32 + sb * 64)));
          __m256i lhs_vec_11 = _mm256_castsi128_si256(
            _mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 48 + sb * 64)));

          lhs_vec_00 = _mm256_permute2f128_si256(lhs_vec_00, lhs_vec_00, 0);
          lhs_vec_01 = _mm256_permute2f128_si256(lhs_vec_01, lhs_vec_01, 0);
          lhs_vec_10 = _mm256_permute2f128_si256(lhs_vec_10, lhs_vec_10, 0);
          lhs_vec_11 = _mm256_permute2f128_si256(lhs_vec_11, lhs_vec_11, 0);

          // Dot product done within 32 bit lanes and accumulated in the same
          // vector First done for first sub block and thenn for second sub
          // block in each sb B0(0-3) B4(0-3) B1(0-3) B5(0-3) B2(0-3) B6(0-3)
          // B3(0-3) B7(0-3) with A0(0-3) B0(4-7) B4(4-7) B1(4-7) B5(4-7)
          // B2(4-7) B6(4-7) B3(4-7) B7(4-7) with A0(4-7)
          // ...........................................................................
          // B0(28-31) B4(28-31) B1(28-31) B5(28-31) B2(28-31) B6(28-31)
          // B3(28-31) B7(28-31) with A0(28-31)

          __m256i iacc_0 = _mm256_setzero_si256();
          __m256i iacc_1 = _mm256_setzero_si256();

          iacc_0 = _mm256_add_epi16(
            iacc_0, _mm256_maddubs_epi16(
                      _mm256_blend_epi32(
                        rhs_vec_0123_00,
                        _mm256_shuffle_epi32(rhs_vec_4567_00, 177), 170),
                      _mm256_shuffle_epi32(lhs_vec_00, 0)));
          iacc_0 = _mm256_add_epi16(
            iacc_0,
            _mm256_maddubs_epi16(
              _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_00, 177),
                                 rhs_vec_4567_00, 170),
              _mm256_shuffle_epi32(lhs_vec_00, 85)));

          iacc_0 = _mm256_add_epi16(
            iacc_0, _mm256_maddubs_epi16(
                      _mm256_blend_epi32(
                        rhs_vec_0123_01,
                        _mm256_shuffle_epi32(rhs_vec_4567_01, 177), 170),
                      _mm256_shuffle_epi32(lhs_vec_00, 170)));
          iacc_0 = _mm256_add_epi16(
            iacc_0,
            _mm256_maddubs_epi16(
              _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_01, 177),
                                 rhs_vec_4567_01, 170),
              _mm256_shuffle_epi32(lhs_vec_00, 255)));

          iacc_0 = _mm256_add_epi16(
            iacc_0, _mm256_maddubs_epi16(
                      _mm256_blend_epi32(
                        rhs_vec_0123_02,
                        _mm256_shuffle_epi32(rhs_vec_4567_02, 177), 170),
                      _mm256_shuffle_epi32(lhs_vec_01, 0)));
          iacc_0 = _mm256_add_epi16(
            iacc_0,
            _mm256_maddubs_epi16(
              _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_02, 177),
                                 rhs_vec_4567_02, 170),
              _mm256_shuffle_epi32(lhs_vec_01, 85)));

          iacc_0 = _mm256_add_epi16(
            iacc_0, _mm256_maddubs_epi16(
                      _mm256_blend_epi32(
                        rhs_vec_0123_03,
                        _mm256_shuffle_epi32(rhs_vec_4567_03, 177), 170),
                      _mm256_shuffle_epi32(lhs_vec_01, 170)));
          iacc_0 = _mm256_add_epi16(
            iacc_0,
            _mm256_maddubs_epi16(
              _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_03, 177),
                                 rhs_vec_4567_03, 170),
              _mm256_shuffle_epi32(lhs_vec_01, 255)));

          iacc_0 = _mm256_madd_epi16(iacc_0, scales_0);

          iacc_1 = _mm256_add_epi16(
            iacc_1, _mm256_maddubs_epi16(
                      _mm256_blend_epi32(
                        rhs_vec_0123_10,
                        _mm256_shuffle_epi32(rhs_vec_4567_10, 177), 170),
                      _mm256_shuffle_epi32(lhs_vec_10, 0)));
          iacc_1 = _mm256_add_epi16(
            iacc_1,
            _mm256_maddubs_epi16(
              _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_10, 177),
                                 rhs_vec_4567_10, 170),
              _mm256_shuffle_epi32(lhs_vec_10, 85)));

          iacc_1 = _mm256_add_epi16(
            iacc_1, _mm256_maddubs_epi16(
                      _mm256_blend_epi32(
                        rhs_vec_0123_11,
                        _mm256_shuffle_epi32(rhs_vec_4567_11, 177), 170),
                      _mm256_shuffle_epi32(lhs_vec_10, 170)));
          iacc_1 = _mm256_add_epi16(
            iacc_1,
            _mm256_maddubs_epi16(
              _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_11, 177),
                                 rhs_vec_4567_11, 170),
              _mm256_shuffle_epi32(lhs_vec_10, 255)));

          iacc_1 = _mm256_add_epi16(
            iacc_1, _mm256_maddubs_epi16(
                      _mm256_blend_epi32(
                        rhs_vec_0123_12,
                        _mm256_shuffle_epi32(rhs_vec_4567_12, 177), 170),
                      _mm256_shuffle_epi32(lhs_vec_11, 0)));
          iacc_1 = _mm256_add_epi16(
            iacc_1,
            _mm256_maddubs_epi16(
              _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_12, 177),
                                 rhs_vec_4567_12, 170),
              _mm256_shuffle_epi32(lhs_vec_11, 85)));

          iacc_1 = _mm256_add_epi16(
            iacc_1, _mm256_maddubs_epi16(
                      _mm256_blend_epi32(
                        rhs_vec_0123_13,
                        _mm256_shuffle_epi32(rhs_vec_4567_13, 177), 170),
                      _mm256_shuffle_epi32(lhs_vec_11, 170)));
          iacc_1 = _mm256_add_epi16(
            iacc_1,
            _mm256_maddubs_epi16(
              _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_13, 177),
                                 rhs_vec_4567_13, 170),
              _mm256_shuffle_epi32(lhs_vec_11, 255)));

          iacc_1 = _mm256_madd_epi16(iacc_1, scales_1);

          // Accumulate the iacc value for one sb
          __m256i iacc_sb = _mm256_add_epi32(iacc_0, iacc_1);

          // Broadcast the bsums of the two sub blocks  of the iteration of Q8_K
          // across the vector Multiply-Add with corresponding mins of Q4_Kx8
          // with bsums
          __m256i q8s_sb = _mm256_shuffle_epi32(q8s, 0);
          __m256i iacc_min_sb = _mm256_madd_epi16(q8s_sb, mins_01);
          q8s = _mm256_bsrli_epi128(q8s, 4);

          // Accumulate for the complete block
          iacc_b = _mm256_add_epi32(iacc_b, iacc_sb);
          iacc_min_b = _mm256_add_epi32(iacc_min_b, iacc_min_sb);
        }

        // Multiply-Add with scale values for the complete super block
        acc_row =
          _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_b),
                          _mm256_mul_ps(col_scale_f32, row_scale_f32), acc_row);
        acc_min_rows = _mm256_fmadd_ps(
          _mm256_cvtepi32_ps(iacc_min_b),
          _mm256_mul_ps(col_dmin_f32, row_scale_f32), acc_min_rows);
      }

      // Accumulated output values permuted so as to be stored in appropriate
      // order post accumulation
      acc_row = _mm256_permutevar8x32_ps(acc_row, finalpermutemask);
      _mm256_storeu_ps(s + (y * nr + x * 8),
                       _mm256_sub_ps(acc_row, acc_min_rows));
    }
  }
}

void nntr_quantize_mat_q8_K_4x8(const float *__restrict x, void *__restrict vy,
                                int64_t k) {
  assert(QK_K == 256);
  assert(k % QK_K == 0);
  const int nb = k / QK_K;

  block_q8_Kx4 *__restrict y = (block_q8_Kx4 *)vy;

  float iscale[4];
  __m256 srcv[4][32];
  __m256 iscale_vec[4];

  for (int i = 0; i < nb; i++) {
    for (int row_iter = 0; row_iter < 4; row_iter++) {
      // Load elements into 4 AVX vectors
      __m256 v0 = _mm256_loadu_ps(x + row_iter * k + i * 256);
      __m256 v1 = _mm256_loadu_ps(x + row_iter * k + i * 256 + 8);
      __m256 v2 = _mm256_loadu_ps(x + row_iter * k + i * 256 + 16);
      __m256 v3 = _mm256_loadu_ps(x + row_iter * k + i * 256 + 24);

      // Compute max(abs(e)) for the block
      const __m256 signBit = _mm256_set1_ps(-0.0f);
      __m256 abs0 = _mm256_andnot_ps(signBit, v0);
      __m256 abs1 = _mm256_andnot_ps(signBit, v1);
      __m256 abs2 = _mm256_andnot_ps(signBit, v2);
      __m256 abs3 = _mm256_andnot_ps(signBit, v3);

      __m256 maxAbs = _mm256_max_ps(abs0, abs1);
      maxAbs = _mm256_max_ps(maxAbs, abs2);
      maxAbs = _mm256_max_ps(maxAbs, abs3);

      __m256 mask0 = _mm256_cmp_ps(maxAbs, v0, _CMP_EQ_OQ);
      __m256 mask1 = _mm256_cmp_ps(maxAbs, v1, _CMP_EQ_OQ);
      __m256 mask2 = _mm256_cmp_ps(maxAbs, v2, _CMP_EQ_OQ);
      __m256 mask3 = _mm256_cmp_ps(maxAbs, v3, _CMP_EQ_OQ);

      __m256 maskAbs =
        _mm256_or_ps(_mm256_or_ps(mask0, mask1), _mm256_or_ps(mask2, mask3));

      srcv[row_iter][0] = v0;
      srcv[row_iter][1] = v1;
      srcv[row_iter][2] = v2;
      srcv[row_iter][3] = v3;

      for (int sb = 1; sb < 8; sb++) {
        // Temporarily stores absolute quant values
        __m256 tempAbs = maxAbs;

        // Load elements into 4 AVX vectors
        __m256 v0 = _mm256_loadu_ps(x + row_iter * k + i * 256 + sb * 32);
        __m256 v1 = _mm256_loadu_ps(x + row_iter * k + i * 256 + sb * 32 + 8);
        __m256 v2 = _mm256_loadu_ps(x + row_iter * k + i * 256 + sb * 32 + 16);
        __m256 v3 = _mm256_loadu_ps(x + row_iter * k + i * 256 + sb * 32 + 24);

        // Compute max(abs(e)) for the block
        __m256 abs0 = _mm256_andnot_ps(signBit, v0);
        __m256 abs1 = _mm256_andnot_ps(signBit, v1);
        __m256 abs2 = _mm256_andnot_ps(signBit, v2);
        __m256 abs3 = _mm256_andnot_ps(signBit, v3);

        maxAbs = _mm256_max_ps(maxAbs, abs0);
        maxAbs = _mm256_max_ps(maxAbs, abs1);
        maxAbs = _mm256_max_ps(maxAbs, abs2);
        maxAbs = _mm256_max_ps(maxAbs, abs3);

        __m256 mask_prev = _mm256_cmp_ps(tempAbs, maxAbs, _CMP_EQ_OQ);
        maskAbs = _mm256_and_ps(maskAbs, mask_prev);

        mask0 = _mm256_cmp_ps(maxAbs, v0, _CMP_EQ_OQ);
        mask1 = _mm256_cmp_ps(maxAbs, v1, _CMP_EQ_OQ);
        mask2 = _mm256_cmp_ps(maxAbs, v2, _CMP_EQ_OQ);
        mask3 = _mm256_cmp_ps(maxAbs, v3, _CMP_EQ_OQ);

        __m256 mask_curr =
          _mm256_or_ps(_mm256_or_ps(mask0, mask1), _mm256_or_ps(mask2, mask3));
        maskAbs = _mm256_or_ps(maskAbs, mask_curr);

        srcv[row_iter][sb * 4] = v0;
        srcv[row_iter][sb * 4 + 1] = v1;
        srcv[row_iter][sb * 4 + 2] = v2;
        srcv[row_iter][sb * 4 + 3] = v3;
      }

      __m128 max4 = _mm_max_ps(_mm256_extractf128_ps(maxAbs, 1),
                               _mm256_castps256_ps128(maxAbs));
      max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
      max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
      const float maxScalar = _mm_cvtss_f32(max4);

      __m256 maxScalarVec = _mm256_set1_ps(maxScalar);

      __m256 mask_next = _mm256_cmp_ps(maxScalarVec, maxAbs, _CMP_EQ_OQ);
      __m256 finalMask = _mm256_and_ps(maskAbs, mask_next);

      const int mask = _mm256_movemask_ps(finalMask);
      iscale[row_iter] = (maxScalar != 0.0f) ? 127.f / maxScalar : 0.0f;

      if (mask) {
        iscale[row_iter] = (maxScalar != 0.0f) ? -127.f / maxScalar : 0.0f;
      }

      y[i].d[row_iter] = maxScalar ? 1 / iscale[row_iter] : 0;
      iscale_vec[row_iter] = _mm256_set1_ps(iscale[row_iter]);
    }

    __m256i quants_interleaved[32];
    for (int j = 0; j < 32; j++) {
      // Apply the multiplier
      __m256 v0 = _mm256_mul_ps(srcv[0][j], iscale_vec[0]);
      __m256 v1 = _mm256_mul_ps(srcv[1][j], iscale_vec[1]);
      __m256 v2 = _mm256_mul_ps(srcv[2][j], iscale_vec[2]);
      __m256 v3 = _mm256_mul_ps(srcv[3][j], iscale_vec[3]);

      // Round to nearest integer
      v0 = _mm256_round_ps(v0, _MM_ROUND_NEAREST);
      v1 = _mm256_round_ps(v1, _MM_ROUND_NEAREST);
      v2 = _mm256_round_ps(v2, _MM_ROUND_NEAREST);
      v3 = _mm256_round_ps(v3, _MM_ROUND_NEAREST);

      // Convert floats to integers
      __m256i i0 = _mm256_cvtps_epi32(v0);
      __m256i i1 = _mm256_cvtps_epi32(v1);
      __m256i i2 = _mm256_cvtps_epi32(v2);
      __m256i i3 = _mm256_cvtps_epi32(v3);

      // Convert int32 to int16
      i0 = _mm256_packs_epi32(i0, i1);
      i2 = _mm256_packs_epi32(i2, i3);
      // Convert int16 to int8
      i0 = _mm256_packs_epi16(i0, i2);

      //  Permute and store the quantized weights in the required order after
      //  the pack instruction
      const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
      i0 = _mm256_permutevar8x32_epi32(i0, perm);

      _mm256_storeu_si256((__m256i *)(y[i].qs + 32 * j), i0);
      quants_interleaved[j] = i0;
    }

    // Masks to shuffle the quants of corresonding sub blocks for rearraning
    // quants for vectorized bsums computation
    __m256i shuffle_mask_sb2 = _mm256_castsi128_si256(
      _mm_setr_epi8(0, 1, 0, 1, 4, 5, 6, 7, 8, 9, 8, 9, 12, 13, 14, 15));
    shuffle_mask_sb2 =
      _mm256_permute2f128_si256(shuffle_mask_sb2, shuffle_mask_sb2, 0);
    __m256i shuffle_mask_sb3 = _mm256_castsi128_si256(
      _mm_setr_epi8(0, 1, 2, 3, 0, 1, 6, 7, 8, 9, 10, 11, 8, 9, 14, 15));
    shuffle_mask_sb3 =
      _mm256_permute2f128_si256(shuffle_mask_sb3, shuffle_mask_sb3, 0);
    __m256i shuffle_mask_sb4 = _mm256_castsi128_si256(
      _mm_setr_epi8(0, 1, 2, 3, 4, 5, 0, 1, 8, 9, 10, 11, 12, 13, 8, 9));
    shuffle_mask_sb4 =
      _mm256_permute2f128_si256(shuffle_mask_sb4, shuffle_mask_sb4, 0);

    for (int k = 0; k < 4; k++) {
      // Quants from four different sub blocks are taken
      __m256i q0 = quants_interleaved[k * 8 + 0];
      __m256i q1 = quants_interleaved[k * 8 + 1];
      __m256i q2 = quants_interleaved[k * 8 + 2];
      __m256i q3 = quants_interleaved[k * 8 + 3];
      __m256i q4 = quants_interleaved[k * 8 + 4];
      __m256i q5 = quants_interleaved[k * 8 + 5];
      __m256i q6 = quants_interleaved[k * 8 + 6];
      __m256i q7 = quants_interleaved[k * 8 + 7];

      // The below code block has the first half of different sub blocks
      // shuffled and blended so as to process 2 values from each sub block at a
      // time
      __m256i sb2_h1_shuffled = _mm256_shuffle_epi8(q2, shuffle_mask_sb2);
      __m256i sb_h1_interleaved = _mm256_blend_epi16(q0, sb2_h1_shuffled, 34);
      __m256i sb3_h1_shuffled = _mm256_shuffle_epi8(q4, shuffle_mask_sb3);
      sb_h1_interleaved =
        _mm256_blend_epi16(sb_h1_interleaved, sb3_h1_shuffled, 68);
      __m256i sb4_h1_shuffled = _mm256_shuffle_epi8(q6, shuffle_mask_sb4);
      sb_h1_interleaved =
        _mm256_blend_epi16(sb_h1_interleaved, sb4_h1_shuffled, 136);

      __m256i one = _mm256_set1_epi8(1);
      __m256i bsums_r1 = _mm256_maddubs_epi16(one, sb_h1_interleaved);

      for (int l = 0; l < 3; l++) {
        // Quants value shifted to process next two values from each sub block
        q0 = _mm256_srli_epi64(q0, 16);
        q2 = _mm256_srli_epi64(q2, 16);
        q4 = _mm256_srli_epi64(q4, 16);
        q6 = _mm256_srli_epi64(q6, 16);

        sb2_h1_shuffled = _mm256_shuffle_epi8(q2, shuffle_mask_sb2);
        sb_h1_interleaved = _mm256_blend_epi16(q0, sb2_h1_shuffled, 34);
        sb3_h1_shuffled = _mm256_shuffle_epi8(q4, shuffle_mask_sb3);
        sb_h1_interleaved =
          _mm256_blend_epi16(sb_h1_interleaved, sb3_h1_shuffled, 68);
        sb4_h1_shuffled = _mm256_shuffle_epi8(q6, shuffle_mask_sb4);
        sb_h1_interleaved =
          _mm256_blend_epi16(sb_h1_interleaved, sb4_h1_shuffled, 136);

        bsums_r1 = _mm256_add_epi16(
          bsums_r1, _mm256_maddubs_epi16(one, sb_h1_interleaved));
      }

      // The below code block has the second half of different sub blocks
      // shuffled and blended so as to process 2 values from each sub block at a
      // time
      __m256i sb2_h2_shuffled = _mm256_shuffle_epi8(q3, shuffle_mask_sb2);
      __m256i sb_h2_interleaved = _mm256_blend_epi16(q1, sb2_h2_shuffled, 34);
      __m256i sb3_h2_shuffled = _mm256_shuffle_epi8(q5, shuffle_mask_sb3);
      sb_h2_interleaved =
        _mm256_blend_epi16(sb_h2_interleaved, sb3_h2_shuffled, 68);
      __m256i sb4_h2_shuffled = _mm256_shuffle_epi8(q7, shuffle_mask_sb4);
      sb_h2_interleaved =
        _mm256_blend_epi16(sb_h2_interleaved, sb4_h2_shuffled, 136);

      __m256i bsums_r2 = _mm256_maddubs_epi16(one, sb_h2_interleaved);

      for (int l = 0; l < 3; l++) {
        // Quants value shifted to process next two values from each sub block
        q1 = _mm256_srli_epi64(q1, 16);
        q3 = _mm256_srli_epi64(q3, 16);
        q5 = _mm256_srli_epi64(q5, 16);
        q7 = _mm256_srli_epi64(q7, 16);

        sb2_h2_shuffled = _mm256_shuffle_epi8(q3, shuffle_mask_sb2);
        sb_h2_interleaved = _mm256_blend_epi16(q1, sb2_h2_shuffled, 34);
        sb3_h2_shuffled = _mm256_shuffle_epi8(q5, shuffle_mask_sb3);
        sb_h2_interleaved =
          _mm256_blend_epi16(sb_h2_interleaved, sb3_h2_shuffled, 68);
        sb4_h2_shuffled = _mm256_shuffle_epi8(q7, shuffle_mask_sb4);
        sb_h2_interleaved =
          _mm256_blend_epi16(sb_h2_interleaved, sb4_h2_shuffled, 136);

        bsums_r2 = _mm256_add_epi16(
          bsums_r2, _mm256_maddubs_epi16(one, sb_h2_interleaved));
      }

      // Overall bsums in interleaved fashion computed by adding results of both
      // halves
      __m256i bsums_r = _mm256_add_epi16(bsums_r1, bsums_r2);
      _mm256_storeu_si256((__m256i *)(y[i].bsums + 16 * k), bsums_r);
    }
  }
}

void nntr_quantize_mat_q8_0_4x8(const float *__restrict x, void *__restrict vy,
                                int64_t k) {
  assert(Q8_0 == 32);
  assert(k % Q8_0 == 0);
  const int nb = k / Q8_0;

  block_q8_0x4 *__restrict y = (block_q8_0x4 *)vy;
  float id[4];
  __m256 srcv[4][4];
  __m256 idvec[4];

  for (int i = 0; i < nb; i++) {
    for (int row_iter = 0; row_iter < 4; row_iter++) {
      // Load elements into 4 AVX vectors
      __m256 v0 = _mm256_loadu_ps(x + row_iter * k + i * 32);
      __m256 v1 = _mm256_loadu_ps(x + row_iter * k + i * 32 + 8);
      __m256 v2 = _mm256_loadu_ps(x + row_iter * k + i * 32 + 16);
      __m256 v3 = _mm256_loadu_ps(x + row_iter * k + i * 32 + 24);

      // Compute max(abs(e)) for the block
      const __m256 signBit = _mm256_set1_ps(-0.0f);
      __m256 maxAbs = _mm256_andnot_ps(signBit, v0);
      maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v1));
      maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v2));
      maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v3));

      __m128 max4 = _mm_max_ps(_mm256_extractf128_ps(maxAbs, 1),
                               _mm256_castps256_ps128(maxAbs));
      max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
      max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
      const float maxScalar = _mm_cvtss_f32(max4);

      // Divided by 127.f to mirror results in quantize_row_q8_0
      const float d = maxScalar / 127.f;
      id[row_iter] =
        (maxScalar != 0.0f) ? 127.f / maxScalar : 0.0f; // d ? 1.0f / d : 0.0f;

      // Store the scale for the individual block
      y[i].d[row_iter] = nntr_compute_fp32_to_fp16(d);

      // Store the values in blocks of eight values - Aim is to use these later
      // for block interleaving
      srcv[row_iter][0] = v0;
      srcv[row_iter][1] = v1;
      srcv[row_iter][2] = v2;
      srcv[row_iter][3] = v3;
      idvec[row_iter] = _mm256_set1_ps(id[row_iter]);
    }

    // The loop iterates four times - The aim is to get 4 corresponding chunks
    // of eight bytes from the original weight blocks that are interleaved
    for (int j = 0; j < 4; j++) {
      // Apply the multiplier
      __m256 v0 = _mm256_mul_ps(srcv[0][j], idvec[0]);
      __m256 v1 = _mm256_mul_ps(srcv[1][j], idvec[1]);
      __m256 v2 = _mm256_mul_ps(srcv[2][j], idvec[2]);
      __m256 v3 = _mm256_mul_ps(srcv[3][j], idvec[3]);

      // Round to nearest integer
      v0 = _mm256_round_ps(v0, _MM_ROUND_NEAREST);
      v1 = _mm256_round_ps(v1, _MM_ROUND_NEAREST);
      v2 = _mm256_round_ps(v2, _MM_ROUND_NEAREST);
      v3 = _mm256_round_ps(v3, _MM_ROUND_NEAREST);

      // Convert floats to integers
      __m256i i0 = _mm256_cvtps_epi32(v0);
      __m256i i1 = _mm256_cvtps_epi32(v1);
      __m256i i2 = _mm256_cvtps_epi32(v2);
      __m256i i3 = _mm256_cvtps_epi32(v3);

      // Convert int32 to int16
      i0 = _mm256_packs_epi32(i0, i1);
      i2 = _mm256_packs_epi32(i2, i3);
      // Convert int16 to int8
      i0 = _mm256_packs_epi16(i0, i2);

      //  Permute and store the quantized weights in the required order after
      //  the pack instruction
      const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
      i0 = _mm256_permutevar8x32_epi32(i0, perm);

      _mm256_storeu_si256((__m256i *)(y[i].qs + 32 * j), i0);
    }
  }
}

static block_q4_0x4 nntr_make_block_q4_0x4(block_q4_0 *in,
                                           unsigned int blck_size_interleave) {
  block_q4_0x4 out;

  for (int i = 0; i < 4; i++) {
    out.d[i] = in[i].d;
  }

  const int end = Q4_0 * 2 / blck_size_interleave;

  if (blck_size_interleave == 8) {
    const uint64_t xor_mask = 0x8888888888888888ULL;
    for (int i = 0; i < end; ++i) {
      int src_id = i % 4;
      int src_offset = (i / 4) * blck_size_interleave;
      int dst_offset = i * blck_size_interleave;

      uint64_t elems;
      // Using memcpy to avoid unaligned memory accesses
      memcpy(&elems, &in[src_id].qs[src_offset], sizeof(uint64_t));
      elems ^= xor_mask;
      memcpy(&out.qs[dst_offset], &elems, sizeof(uint64_t));
    }
  } else if (blck_size_interleave == 4) {
    const uint32_t xor_mask = 0x88888888;
    for (int i = 0; i < end; ++i) {
      int src_id = i % 4;
      int src_offset = (i / 4) * blck_size_interleave;
      int dst_offset = i * blck_size_interleave;

      uint32_t elems;
      memcpy(&elems, &in[src_id].qs[src_offset], sizeof(uint32_t));
      elems ^= xor_mask;
      memcpy(&out.qs[dst_offset], &elems, sizeof(uint32_t));
    }
  } else {
    assert(false);
  }

  return out;
}

static block_q4_0x8 nntr_make_block_q4_0x8(block_q4_0 *in,
                                           unsigned int blck_size_interleave) {
  block_q4_0x8 out;

  for (int i = 0; i < 8; i++) {
    out.d[i] = in[i].d;
  }

  const int end = QK_0<4>() * 4 / blck_size_interleave;
  const uint64_t xor_mask = 0x8888888888888888ULL;

  for (int i = 0; i < end; ++i) {
    int src_id = i % 8;
    int src_offset = (i / 8) * blck_size_interleave;
    int dst_offset = i * blck_size_interleave;

    uint64_t elems;
    memcpy(&elems, &in[src_id].qs[src_offset], sizeof(uint64_t));
    elems ^= xor_mask;
    memcpy(&out.qs[dst_offset], &elems, sizeof(uint64_t));
  }

  return out;
}

static block_q4_Kx8 make_block_q4_Kx8(block_q4_K *in,
                                      unsigned int blck_size_interleave) {
  block_q4_Kx8 out;
  // Delta(scale) and dmin values of the eight Q4_K structures are copied onto
  // the output interleaved structure
  for (int i = 0; i < 8; i++) {
    out.d[i] = in[i].data.data.d;
  }

  for (int i = 0; i < 8; i++) {
    out.dmin[i] = in[i].data.data.dmin;
  }

  const int end = QK_K * 4 / blck_size_interleave;

  // Interleave Q4_K quants by taking 8 bytes at a time
  for (int i = 0; i < end; ++i) {
    int src_id = i % 8;
    int src_offset = (i / 8) * blck_size_interleave;
    int dst_offset = i * blck_size_interleave;

    uint64_t elems;
    memcpy(&elems, &in[src_id].qs[src_offset], sizeof(uint64_t));
    memcpy(&out.qs[dst_offset], &elems, sizeof(uint64_t));
  }

  // The below logic is designed so as to unpack and rearrange scales and mins
  // values in Q4_K Currently the Q4_K structure has 8 scales and 8 mins packed
  // in 12 bytes ( 6 bits for each value) The output Q4_Kx8 structure has 96
  // bytes Every 12 byte is packed such that it contains scales and mins for
  // corresponding sub blocks from Q4_K structure For eg - First 12 bytes
  // contains 8 scales and 8 mins - each of first sub block from different Q4_K
  // structures
  uint8_t s[8], m[8];

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 8; j++) {
      s[j] = in[j].scales[i] & 63;
      m[j] = in[j].scales[i + 4] & 63;
    }

    out.scales[i * 12] = (s[0] & 63) + ((s[4] & 48) << 2);
    out.scales[i * 12 + 1] = (s[1] & 63) + ((s[5] & 48) << 2);
    out.scales[i * 12 + 2] = (s[2] & 63) + ((s[6] & 48) << 2);
    out.scales[i * 12 + 3] = (s[3] & 63) + ((s[7] & 48) << 2);
    out.scales[i * 12 + 4] = (m[0] & 63) + ((m[4] & 48) << 2);
    out.scales[i * 12 + 5] = (m[1] & 63) + ((m[5] & 48) << 2);
    out.scales[i * 12 + 6] = (m[2] & 63) + ((m[6] & 48) << 2);
    out.scales[i * 12 + 7] = (m[3] & 63) + ((m[7] & 48) << 2);
    out.scales[i * 12 + 8] = (s[4] & 15) + ((m[4] & 15) << 4);
    out.scales[i * 12 + 9] = (s[5] & 15) + ((m[5] & 15) << 4);
    out.scales[i * 12 + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
    out.scales[i * 12 + 11] = (s[7] & 15) + ((m[7] & 15) << 4);
  }

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 8; j++) {
      s[j] = ((in[j].scales[i] & 192) >> 2) | (in[j].scales[i + 8] & 15);
      m[j] =
        ((in[j].scales[i + 4] & 192) >> 2) | ((in[j].scales[i + 8] & 240) >> 4);
    }

    out.scales[i * 12 + 48] = (s[0] & 63) + ((s[4] & 48) << 2);
    out.scales[i * 12 + 49] = (s[1] & 63) + ((s[5] & 48) << 2);
    out.scales[i * 12 + 50] = (s[2] & 63) + ((s[6] & 48) << 2);
    out.scales[i * 12 + 51] = (s[3] & 63) + ((s[7] & 48) << 2);
    out.scales[i * 12 + 52] = (m[0] & 63) + ((m[4] & 48) << 2);
    out.scales[i * 12 + 53] = (m[1] & 63) + ((m[5] & 48) << 2);
    out.scales[i * 12 + 54] = (m[2] & 63) + ((m[6] & 48) << 2);
    out.scales[i * 12 + 55] = (m[3] & 63) + ((m[7] & 48) << 2);
    out.scales[i * 12 + 56] = (s[4] & 15) + ((m[4] & 15) << 4);
    out.scales[i * 12 + 57] = (s[5] & 15) + ((m[5] & 15) << 4);
    out.scales[i * 12 + 58] = (s[6] & 15) + ((m[6] & 15) << 4);
    out.scales[i * 12 + 59] = (s[7] & 15) + ((m[7] & 15) << 4);
  }

  return out;
}

int nntr_repack_q4_0_to_q4_0_4_bl(void *__restrict dst, int interleave_block,
                                  const void *__restrict data, size_t data_size,
                                  size_t nrow, size_t k) {
  assert(interleave_block == 4 || interleave_block == 8);
  constexpr int nrows_interleaved = 4;

  block_q4_0x4 *dst_ = (block_q4_0x4 *)dst;
  const block_q4_0 *src = (const block_q4_0 *)data;
  block_q4_0 dst_tmp[4];
  int nblocks = k / Q4_0;

  assert(data_size == nrow * nblocks * sizeof(block_q4_0));

  if (nrow % nrows_interleaved != 0 || k % 8 != 0) {
    return -1;
  }

  for (size_t b = 0; b < nrow; b += nrows_interleaved) {
    for (int64_t x = 0; x < nblocks; x++) {
      for (size_t i = 0; i < nrows_interleaved; i++) {
        dst_tmp[i] = src[x + i * nblocks];
      }
      *dst_++ = nntr_make_block_q4_0x4(dst_tmp, interleave_block);
    }
    src += nrows_interleaved * nblocks;
  }
  return 0;
}

int nntr_repack_q4_0_to_q4_0_8_bl(void *__restrict dst, int interleave_block,
                                  const void *__restrict data, size_t data_size,
                                  size_t nrow, size_t k) {
  assert(interleave_block == 8);
  constexpr size_t nrows_interleaved = 8;

  block_q4_0x8 *dst_ = (block_q4_0x8 *)dst;
  const block_q4_0 *src = (const block_q4_0 *)data;
  block_q4_0 dst_tmp[8];
  int nblocks = k / QK_0<4>();

  assert(data_size == nrow * nblocks * sizeof(block_q4_0));

  if (nrow % nrows_interleaved != 0 || k % 8 != 0) {
    return -1;
  }

  for (size_t b = 0; b < nrow; b += nrows_interleaved) {
    for (int64_t x = 0; x < nblocks; x++) {
      for (size_t i = 0; i < nrows_interleaved; i++) {
        dst_tmp[i] = src[x + i * nblocks];
      }
      *dst_++ = nntr_make_block_q4_0x8(dst_tmp, interleave_block);
    }
    src += nrows_interleaved * nblocks;
  }
  return 0;
}

int nntr_repack_q4_K_to_q4_K_8_bl(void *__restrict dst, int interleave_block,
                                  const void *__restrict data, size_t data_size,
                                  size_t nrow, size_t k) {
  assert(interleave_block == 8);
  constexpr size_t nrows_interleaved = 8;

  block_q4_Kx8 *dst_ = (block_q4_Kx8 *)dst;
  const block_q4_K *src = (const block_q4_K *)data;
  block_q4_K dst_tmp[8];
  int nblocks = k / QK_K;

  assert(data_size == nrow * nblocks * sizeof(block_q4_K));

  if (nrow % nrows_interleaved != 0 || k % 8 != 0) {
    return -1;
  }

  for (size_t b = 0; b < nrow; b += nrows_interleaved) {
    for (int64_t x = 0; x < nblocks; x++) {
      for (size_t i = 0; i < nrows_interleaved; i++) {
        dst_tmp[i] = src[x + i * nblocks];
      }
      *dst_++ = make_block_q4_Kx8(dst_tmp, interleave_block);
    }
    src += nrows_interleaved * nblocks;
  }
  return 0;
}

//===================================== Dot products
//=================================

//
// Helper functions
//

static inline __m128i get_scale_shuffle(int i) {
  static const uint8_t k_shuffle[128] = {
    0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,
    2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  4,
    4,  4,  5,  5,  5,  5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  6,  6,  6,  7,
    7,  7,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,  8,  8,  8,  9,  9,  9,  9,
    9,  9,  9,  9,  10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11,
    11, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14,
    14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15};
  return _mm_loadu_si128((const __m128i *)k_shuffle + i);
}

void nntr_vec_dot_q6_K_q8_K(int n, float *__restrict s, size_t bs,
                            const void *__restrict vx, size_t bx,
                            const void *__restrict vy, size_t by, int nrc) {
  assert(n % QK_K == 0);
  assert(nrc == 1);

  const block_q6_K *__restrict x = (const block_q6_K *)vx;
  const block_q8_K *__restrict y = (const block_q8_K *)vy;

  const int nb = n / QK_K;

  const __m256i m4 = _mm256_set1_epi8(0xF);
  const __m256i m2 = _mm256_set1_epi8(3);
  const __m256i m32s = _mm256_set1_epi8(32);

  __m256 acc = _mm256_setzero_ps();

  for (int i = 0; i < nb; ++i) {

    const float d = y[i].d * nntr_compute_fp16_to_fp32(x[i].d);

    const uint8_t *__restrict q4 = x[i].ql;
    const uint8_t *__restrict qh = x[i].qh;
    const int8_t *__restrict q8 = y[i].qs;

    const __m128i scales = _mm_loadu_si128((const __m128i *)x[i].scales);

    __m256i sumi = _mm256_setzero_si256();

    int is = 0;

    for (int j = 0; j < QK_K / 128; ++j) {

      const __m128i scale_0 =
        _mm_shuffle_epi8(scales, get_scale_shuffle(is + 0));
      const __m128i scale_1 =
        _mm_shuffle_epi8(scales, get_scale_shuffle(is + 1));
      const __m128i scale_2 =
        _mm_shuffle_epi8(scales, get_scale_shuffle(is + 2));
      const __m128i scale_3 =
        _mm_shuffle_epi8(scales, get_scale_shuffle(is + 3));
      is += 4;

      const __m256i q4bits1 = _mm256_loadu_si256((const __m256i *)q4);
      q4 += 32;
      const __m256i q4bits2 = _mm256_loadu_si256((const __m256i *)q4);
      q4 += 32;
      const __m256i q4bitsH = _mm256_loadu_si256((const __m256i *)qh);
      qh += 32;

      const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m2), 4);
      const __m256i q4h_1 = _mm256_slli_epi16(
        _mm256_and_si256(_mm256_srli_epi16(q4bitsH, 2), m2), 4);
      const __m256i q4h_2 = _mm256_slli_epi16(
        _mm256_and_si256(_mm256_srli_epi16(q4bitsH, 4), m2), 4);
      const __m256i q4h_3 = _mm256_slli_epi16(
        _mm256_and_si256(_mm256_srli_epi16(q4bitsH, 6), m2), 4);

      const __m256i q4_0 =
        _mm256_or_si256(_mm256_and_si256(q4bits1, m4), q4h_0);
      const __m256i q4_1 =
        _mm256_or_si256(_mm256_and_si256(q4bits2, m4), q4h_1);
      const __m256i q4_2 = _mm256_or_si256(
        _mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m4), q4h_2);
      const __m256i q4_3 = _mm256_or_si256(
        _mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m4), q4h_3);

      const __m256i q8_0 = _mm256_loadu_si256((const __m256i *)q8);
      q8 += 32;
      const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8);
      q8 += 32;
      const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8);
      q8 += 32;
      const __m256i q8_3 = _mm256_loadu_si256((const __m256i *)q8);
      q8 += 32;

      __m256i q8s_0 = _mm256_maddubs_epi16(m32s, q8_0);
      __m256i q8s_1 = _mm256_maddubs_epi16(m32s, q8_1);
      __m256i q8s_2 = _mm256_maddubs_epi16(m32s, q8_2);
      __m256i q8s_3 = _mm256_maddubs_epi16(m32s, q8_3);

      __m256i p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
      __m256i p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
      __m256i p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
      __m256i p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);

      p16_0 = _mm256_sub_epi16(p16_0, q8s_0);
      p16_1 = _mm256_sub_epi16(p16_1, q8s_1);
      p16_2 = _mm256_sub_epi16(p16_2, q8s_2);
      p16_3 = _mm256_sub_epi16(p16_3, q8s_3);

      p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
      p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);
      p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_2);
      p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_3);

      sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
      sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));
    }

    acc =
      _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
  }

  *s = hsum_float_8(acc);
}

void nntr_gemv_q4_0_8x8_q8_0(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc) {
  const int qk = QK8_0;
  const int nb = n / qk;
  const int ncols_interleaved = 8;
  const int blocklen = 8;

  assert(n % qk == 0);
  assert(nc % ncols_interleaved == 0);

  // Lookup table to convert signed nibbles to signed bytes
  __m256i signextendlut = _mm256_castsi128_si256(
    _mm_set_epi8(-1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0));
  signextendlut = _mm256_permute2f128_si256(signextendlut, signextendlut, 0);
  __m128i changemask =
    _mm_set_epi8(15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0);
  __m256i finalpermutemask = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);

  // Permute mask used for easier vector processing at later stages
  const __m256i m4b = _mm256_set1_epi8(0x0F);

  int64_t b_nb = n / QK4_0;

  const block_q4_0x8 *b_ptr_start = (const block_q4_0x8 *)vx;
  const block_q8_0 *a_ptr_start = (const block_q8_0 *)vy;

  // Process Q8_0 blocks one by one
  for (int64_t y = 0; y < nr; y++) {

    // Pointers to LHS blocks of block_q8_0 format
    const block_q8_0 *a_ptr = a_ptr_start + (y * nb);

    // Take group of eight block_q4_0x8 structures at each pass of the loop and
    // perform dot product operation
    for (int64_t x = 0; x < nc / 8; x++) {

      // Pointers to RHS blocks
      const block_q4_0x8 *b_ptr = b_ptr_start + (x * b_nb);

      // Master FP accumulator
      __m256 acc_row = _mm256_setzero_ps();

      for (int64_t b = 0; b < nb; b++) {
        // Load 8 blocks of Q4_0 interleaved as 8 bytes (B0 - B7)
        const __m256i rhs_raw_vec_0123_0 =
          _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs));
        const __m256i rhs_raw_vec_4567_0 =
          _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs) + 1);
        const __m256i rhs_raw_vec_0123_1 =
          _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs) + 2);
        const __m256i rhs_raw_vec_4567_1 =
          _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs) + 3);

        // 4-bit -> 8-bit - Sign is maintained
        const __m256i rhs_vec_0123_0 = _mm256_shuffle_epi8(
          signextendlut,
          _mm256_and_si256(rhs_raw_vec_0123_0,
                           m4b)); // B0(0-7) B1(0-7) B2(0-7) B3(0-7)
        const __m256i rhs_vec_4567_0 = _mm256_shuffle_epi8(
          signextendlut,
          _mm256_and_si256(rhs_raw_vec_4567_0,
                           m4b)); // B4(0-7) B5(0-7) B6(0-7) B7(0-7)
        const __m256i rhs_vec_0123_1 = _mm256_shuffle_epi8(
          signextendlut,
          _mm256_and_si256(rhs_raw_vec_0123_1,
                           m4b)); // B0(8-15) B1(8-15) B2(8-15) B3(8-15)
        const __m256i rhs_vec_4567_1 = _mm256_shuffle_epi8(
          signextendlut,
          _mm256_and_si256(rhs_raw_vec_4567_1,
                           m4b)); // B0(8-15) B1(8-15) B2(8-15) B3(8-15)

        const __m256i rhs_vec_0123_2 = _mm256_shuffle_epi8(
          signextendlut,
          _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_0, 4),
                           m4b)); // B0(16-23) B1(16-23) B2(16-23) B3(16-23)
        const __m256i rhs_vec_4567_2 = _mm256_shuffle_epi8(
          signextendlut,
          _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_0, 4),
                           m4b)); // B4(16-23) B5(16-23) B6(16-23) B7(16-23)
        const __m256i rhs_vec_0123_3 = _mm256_shuffle_epi8(
          signextendlut,
          _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_1, 4),
                           m4b)); // B0(24-31) B1(24-31) B2(24-31) B3(24-31)
        const __m256i rhs_vec_4567_3 = _mm256_shuffle_epi8(
          signextendlut,
          _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_1, 4),
                           m4b)); // B4(24-31) B5(24-31) B6(24-31) B7(24-31)

        // Load the scale values for the 8 blocks interleaved in block_q4_0x8
        const __m256 col_scale_f32 =
          GGML_F32Cx8_REARRANGE_LOAD(b_ptr[b].d, changemask);

        // Load and convert to FP32 scale from block_q8_0
        const __m256 row_scale_f32 =
          _mm256_set1_ps(nntr_fp16_to_fp32(a_ptr[b].d));

        // Load the block values in block_q8_0 in batches of 16 bytes and
        // replicate the same across 256 bit vector
        __m256i lhs_vec_0 =
          _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)a_ptr[b].qs));
        __m256i lhs_vec_1 = _mm256_castsi128_si256(
          _mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 16)));

        lhs_vec_0 = _mm256_permute2f128_si256(lhs_vec_0, lhs_vec_0,
                                              0); // A0 (0-15) A0(0-15)
        lhs_vec_1 = _mm256_permute2f128_si256(lhs_vec_1, lhs_vec_1,
                                              0); // A0 (16-31) A0(16-31))

        __m256i iacc = _mm256_setzero_si256();

        // Dot product done within 32 bit lanes and accumulated in the same
        // vector B0(0-3) B4(0-3) B1(0-3) B5(0-3) B2(0-3) B6(0-3) B3(0-3)
        // B7(0-3) with A0(0-3) B0(4-7) B4(4-7) B1(4-7) B5(4-7) B2(4-7) B6(4-7)
        // B3(4-7) B7(4-7) with A0(4-7)
        // ...........................................................................
        // B0(28-31) B4(28-31) B1(28-31) B5(28-31) B2(28-31) B6(28-31) B3(28-31)
        // B7(28-31) with A0(28-31)

        iacc = mul_sum_i8_pairs_acc_int32x8(
          iacc,
          _mm256_blend_epi32(rhs_vec_0123_0,
                             _mm256_shuffle_epi32(rhs_vec_4567_0, 177), 170),
          _mm256_shuffle_epi32(lhs_vec_0, 0));
        iacc = mul_sum_i8_pairs_acc_int32x8(
          iacc,
          _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_0, 177),
                             rhs_vec_4567_0, 170),
          _mm256_shuffle_epi32(lhs_vec_0, 85));

        iacc = mul_sum_i8_pairs_acc_int32x8(
          iacc,
          _mm256_blend_epi32(rhs_vec_0123_1,
                             _mm256_shuffle_epi32(rhs_vec_4567_1, 177), 170),
          _mm256_shuffle_epi32(lhs_vec_0, 170));
        iacc = mul_sum_i8_pairs_acc_int32x8(
          iacc,
          _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_1, 177),
                             rhs_vec_4567_1, 170),
          _mm256_shuffle_epi32(lhs_vec_0, 255));

        iacc = mul_sum_i8_pairs_acc_int32x8(
          iacc,
          _mm256_blend_epi32(rhs_vec_0123_2,
                             _mm256_shuffle_epi32(rhs_vec_4567_2, 177), 170),
          _mm256_shuffle_epi32(lhs_vec_1, 0));
        iacc = mul_sum_i8_pairs_acc_int32x8(
          iacc,
          _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_2, 177),
                             rhs_vec_4567_2, 170),
          _mm256_shuffle_epi32(lhs_vec_1, 85));

        iacc = mul_sum_i8_pairs_acc_int32x8(
          iacc,
          _mm256_blend_epi32(rhs_vec_0123_3,
                             _mm256_shuffle_epi32(rhs_vec_4567_3, 177), 170),
          _mm256_shuffle_epi32(lhs_vec_1, 170));
        iacc = mul_sum_i8_pairs_acc_int32x8(
          iacc,
          _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_3, 177),
                             rhs_vec_4567_3, 170),
          _mm256_shuffle_epi32(lhs_vec_1, 255));

        // Accumulated values multipled with appropriate scales
        acc_row =
          _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc),
                          _mm256_mul_ps(col_scale_f32, row_scale_f32), acc_row);
      }

      // Accumulated output values permuted so as to be stored in appropriate
      // order post accumulation
      acc_row = _mm256_permutevar8x32_ps(acc_row, finalpermutemask);
      _mm256_storeu_ps(s + (y * nr + x * 8), acc_row);
    }
  }
  return;
}
