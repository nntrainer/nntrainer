// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   matrix_transpose_neon.cpp
 * @date   09 May 2024
 * @see    https://github.com/nntrainer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is source file for matrix transpose using NEON
 *
 */

#include <arm_neon.h>
#include <matrix_transpose_neon.h>

template <>
void transpose_neon(unsigned int M, unsigned int N, const float *src,
                    unsigned int ld_src, float *dst, unsigned int ld_dst) {
  float *src_ptr = (float *)(src);
  float *dst_ptr = (float *)(dst);

  /// the remainder handling below re-transposes the last full 4x4 tile, which
  /// does not exist when either dimension is smaller than 4
  if (M < 4 || N < 4) {
    for (unsigned int i = 0; i < M; ++i) {
      for (unsigned int j = 0; j < N; ++j) {
        dst_ptr[i + j * ld_dst] = src_ptr[i * ld_src + j];
      }
    }
    return;
  }

  unsigned int M_blocks = M / 4;
  unsigned int N_blocks = N / 4;
  unsigned int M_left = M % 4;
  unsigned int N_left = N % 4;

  if (M > 128) {
    for (unsigned int m = 0; m < M_blocks; ++m) {
      auto M_tile_ptr = src_ptr + m * 4 * ld_src;

      for (unsigned int n = 0; n < N_blocks; ++n) {
        auto dst_tile_ptr = dst_ptr + n * 4 * ld_dst;

        auto tile_0 = vld1q_f32(M_tile_ptr + 4 * n);
        auto tile_1 = vld1q_f32(M_tile_ptr + 4 * n + ld_src);
        auto tile_2 = vld1q_f32(M_tile_ptr + 4 * n + 2 * ld_src);
        auto tile_3 = vld1q_f32(M_tile_ptr + 4 * n + 3 * ld_src);

        float32x4x2_t row01 = vtrnq_f32(tile_0, tile_1);
        float32x4x2_t row23 = vtrnq_f32(tile_2, tile_3);

        vst1q_f32(
          dst_tile_ptr + 4 * m,
          vcombine_f32(vget_low_f32(row01.val[0]), vget_low_f32(row23.val[0])));
        vst1q_f32(
          dst_tile_ptr + 4 * m + ld_dst,
          vcombine_f32(vget_low_f32(row01.val[1]), vget_low_f32(row23.val[1])));
        vst1q_f32(dst_tile_ptr + 4 * m + 2 * ld_dst,
                  vcombine_f32(vget_high_f32(row01.val[0]),
                               vget_high_f32(row23.val[0])));
        vst1q_f32(dst_tile_ptr + 4 * m + 3 * ld_dst,
                  vcombine_f32(vget_high_f32(row01.val[1]),
                               vget_high_f32(row23.val[1])));
      }

      if (N_left) {
        auto dst_tile_ptr = dst_ptr + (N_blocks * 4 - (4 - N_left)) * ld_dst;

        auto tile_0 = vld1q_f32(M_tile_ptr + (N_blocks * 4 - (4 - N_left)));
        auto tile_1 =
          vld1q_f32(M_tile_ptr + (N_blocks * 4 - (4 - N_left)) + ld_src);
        auto tile_2 =
          vld1q_f32(M_tile_ptr + (N_blocks * 4 - (4 - N_left)) + 2 * ld_src);
        auto tile_3 =
          vld1q_f32(M_tile_ptr + (N_blocks * 4 - (4 - N_left)) + 3 * ld_src);

        float32x4x2_t row01 = vtrnq_f32(tile_0, tile_1);
        float32x4x2_t row23 = vtrnq_f32(tile_2, tile_3);

        vst1q_f32(
          dst_tile_ptr + 4 * m,
          vcombine_f32(vget_low_f32(row01.val[0]), vget_low_f32(row23.val[0])));
        vst1q_f32(
          dst_tile_ptr + 4 * m + ld_dst,
          vcombine_f32(vget_low_f32(row01.val[1]), vget_low_f32(row23.val[1])));
        vst1q_f32(dst_tile_ptr + 4 * m + 2 * ld_dst,
                  vcombine_f32(vget_high_f32(row01.val[0]),
                               vget_high_f32(row23.val[0])));
        vst1q_f32(dst_tile_ptr + 4 * m + 3 * ld_dst,
                  vcombine_f32(vget_high_f32(row01.val[1]),
                               vget_high_f32(row23.val[1])));
      }
    }
  } else {
    for (unsigned int m = 0; m < M_blocks; ++m) {
      auto M_tile_ptr = src_ptr + m * 4 * ld_src;

      for (unsigned int n = 0; n < N_blocks; ++n) {
        auto dst_tile_ptr = dst_ptr + n * 4 * ld_dst;

        auto tile_0 = vld1q_f32(M_tile_ptr + 4 * n);
        auto tile_1 = vld1q_f32(M_tile_ptr + 4 * n + ld_src);
        auto tile_2 = vld1q_f32(M_tile_ptr + 4 * n + 2 * ld_src);
        auto tile_3 = vld1q_f32(M_tile_ptr + 4 * n + 3 * ld_src);

        float32x4x2_t row01 = vtrnq_f32(tile_0, tile_1);
        float32x4x2_t row23 = vtrnq_f32(tile_2, tile_3);

        vst1q_f32(
          dst_tile_ptr + 4 * m,
          vcombine_f32(vget_low_f32(row01.val[0]), vget_low_f32(row23.val[0])));
        vst1q_f32(
          dst_tile_ptr + 4 * m + ld_dst,
          vcombine_f32(vget_low_f32(row01.val[1]), vget_low_f32(row23.val[1])));
        vst1q_f32(dst_tile_ptr + 4 * m + 2 * ld_dst,
                  vcombine_f32(vget_high_f32(row01.val[0]),
                               vget_high_f32(row23.val[0])));
        vst1q_f32(dst_tile_ptr + 4 * m + 3 * ld_dst,
                  vcombine_f32(vget_high_f32(row01.val[1]),
                               vget_high_f32(row23.val[1])));
      }

      if (N_left) {
        auto dst_tile_ptr = dst_ptr + (N_blocks * 4 - (4 - N_left)) * ld_dst;

        auto tile_0 = vld1q_f32(M_tile_ptr + (N_blocks * 4 - (4 - N_left)));
        auto tile_1 =
          vld1q_f32(M_tile_ptr + (N_blocks * 4 - (4 - N_left)) + ld_src);
        auto tile_2 =
          vld1q_f32(M_tile_ptr + (N_blocks * 4 - (4 - N_left)) + 2 * ld_src);
        auto tile_3 =
          vld1q_f32(M_tile_ptr + (N_blocks * 4 - (4 - N_left)) + 3 * ld_src);

        float32x4x2_t row01 = vtrnq_f32(tile_0, tile_1);
        float32x4x2_t row23 = vtrnq_f32(tile_2, tile_3);

        vst1q_f32(
          dst_tile_ptr + 4 * m,
          vcombine_f32(vget_low_f32(row01.val[0]), vget_low_f32(row23.val[0])));
        vst1q_f32(
          dst_tile_ptr + 4 * m + ld_dst,
          vcombine_f32(vget_low_f32(row01.val[1]), vget_low_f32(row23.val[1])));
        vst1q_f32(dst_tile_ptr + 4 * m + 2 * ld_dst,
                  vcombine_f32(vget_high_f32(row01.val[0]),
                               vget_high_f32(row23.val[0])));
        vst1q_f32(dst_tile_ptr + 4 * m + 3 * ld_dst,
                  vcombine_f32(vget_high_f32(row01.val[1]),
                               vget_high_f32(row23.val[1])));
      }
    }
  }

  if (M_left) {
    auto M_tile_ptr = src_ptr + (M_blocks * 4 - (4 - M_left)) * ld_src;

    for (unsigned int n = 0; n < N_blocks; ++n) {
      auto dst_tile_ptr = dst_ptr + n * 4 * ld_dst;

      auto tile_0 = vld1q_f32(M_tile_ptr + 4 * n);
      auto tile_1 = vld1q_f32(M_tile_ptr + 4 * n + ld_src);
      auto tile_2 = vld1q_f32(M_tile_ptr + 4 * n + 2 * ld_src);
      auto tile_3 = vld1q_f32(M_tile_ptr + 4 * n + 3 * ld_src);

      float32x4x2_t row01 = vtrnq_f32(tile_0, tile_1);
      float32x4x2_t row23 = vtrnq_f32(tile_2, tile_3);

      vst1q_f32(
        dst_tile_ptr + (M_blocks * 4 - (4 - M_left)),
        vcombine_f32(vget_low_f32(row01.val[0]), vget_low_f32(row23.val[0])));
      vst1q_f32(
        dst_tile_ptr + (M_blocks * 4 - (4 - M_left)) + ld_dst,
        vcombine_f32(vget_low_f32(row01.val[1]), vget_low_f32(row23.val[1])));
      vst1q_f32(
        dst_tile_ptr + (M_blocks * 4 - (4 - M_left)) + 2 * ld_dst,
        vcombine_f32(vget_high_f32(row01.val[0]), vget_high_f32(row23.val[0])));
      vst1q_f32(
        dst_tile_ptr + (M_blocks * 4 - (4 - M_left)) + 3 * ld_dst,
        vcombine_f32(vget_high_f32(row01.val[1]), vget_high_f32(row23.val[1])));
    }

    if (N_left) {
      auto dst_tile_ptr = dst_ptr + (N_blocks * 4 - (4 - N_left)) * ld_dst;

      auto tile_0 = vld1q_f32(M_tile_ptr + (N_blocks * 4 - (4 - N_left)));
      auto tile_1 =
        vld1q_f32(M_tile_ptr + (N_blocks * 4 - (4 - N_left)) + ld_src);
      auto tile_2 =
        vld1q_f32(M_tile_ptr + (N_blocks * 4 - (4 - N_left)) + 2 * ld_src);
      auto tile_3 =
        vld1q_f32(M_tile_ptr + (N_blocks * 4 - (4 - N_left)) + 3 * ld_src);

      float32x4x2_t row01 = vtrnq_f32(tile_0, tile_1);
      float32x4x2_t row23 = vtrnq_f32(tile_2, tile_3);

      vst1q_f32(
        dst_tile_ptr + (M_blocks * 4 - (4 - M_left)),
        vcombine_f32(vget_low_f32(row01.val[0]), vget_low_f32(row23.val[0])));
      vst1q_f32(
        dst_tile_ptr + (M_blocks * 4 - (4 - M_left)) + ld_dst,
        vcombine_f32(vget_low_f32(row01.val[1]), vget_low_f32(row23.val[1])));
      vst1q_f32(
        dst_tile_ptr + (M_blocks * 4 - (4 - M_left)) + 2 * ld_dst,
        vcombine_f32(vget_high_f32(row01.val[0]), vget_high_f32(row23.val[0])));
      vst1q_f32(
        dst_tile_ptr + (M_blocks * 4 - (4 - M_left)) + 3 * ld_dst,
        vcombine_f32(vget_high_f32(row01.val[1]), vget_high_f32(row23.val[1])));
    }
  }
}
