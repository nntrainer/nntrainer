/**
 * @file   kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0_neon.c
 * @date   02 July 2026
 * @see    https://github.com/ARM-software/kleidiai
 * @author Jaemin Shin <jaemin980311@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  Custom NEON-optimized variant of kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0
 *         specialized for (nr, kr, sr) = (8, 16, 2).
 */
//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates
// <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#if !defined(__aarch64__) && !defined(_M_ARM64)
#error This file must be compiled for AArch64.
#else // Architectural features check.

#include "kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0_neon.h"

#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "kai/kai_common.h"

static const size_t kai_num_bytes_sum_rhs = sizeof(int32_t);
static const size_t kai_num_bytes_multiplier_rhs = sizeof(float);
static const size_t kai_num_bytes_bias = sizeof(float);

inline static size_t kai_k_roundedup(size_t k) {
  // Round up k to be a multiple of 32.
  size_t kai_k_multiple_of = 32;
  return kai_roundup(k, kai_k_multiple_of);
}

// With (nr, kr, sr) = (8, 16, 2), the packing of one row block (8 source rows)
// decomposes into independent 32-wide chunks along K. For chunk t, source row
// bytes s[16t .. 16t+15] (holding k = 32t .. 32t+31, two nibbles per byte,
// low nibble first) map to two 8-byte destination groups:
//
//   lo_nib[j] = (s[j]   & 0x0F) | (s[8+j] << 4)    j = 0..7
//   hi_nib[j] = (s[8+j] & 0xF0) | (s[j]   >> 4)    j = 0..7
//   zip(lo_nib, hi_nib) = out[0..15]
//
//   out[0..7]  -> dst + 128*t      + 8*nr_idx   (super block 2t)
//   out[8..15] -> dst + 128*t + 64 + 8*nr_idx   (super block 2t+1)
//
// followed by XOR 0x88 (rhs_zero_point == 8). The per-row reduction sum is
// the plain sum of all source nibbles minus 2 * zero_point per packed byte.
void kai_run_rhs_pack_nxk_qsi4cxp_qs4cxs1s0_neon(
  size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr,
  const uint8_t *rhs, const float *bias, const float *scale, void *rhs_packed,
  size_t extra_bytes,
  const struct kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0_params *params) {
  KAI_ASSERT(num_groups == 1);
  KAI_ASSERT(extra_bytes == 0);
  KAI_ASSERT((kr % sr) == 0);
  KAI_ASSERT(rhs != NULL);
  KAI_ASSERT(scale != NULL);
  KAI_ASSERT(rhs_packed != NULL);
  KAI_ASSERT(params != NULL);
  KAI_ASSERT(params->lhs_zero_point == 1);
  KAI_ASSERT(params->rhs_zero_point == 0 || params->rhs_zero_point == 8);

  // Fast path only for (nr, kr, sr) = (8, 16, 2) with rhs_zero_point == 8 and
  // even k. The uint16x8 nibble-sum accumulators are safe up to 1024 chunks
  // (k_internal <= 32768); larger k is delegated as well.
  if (nr != 8 || kr != 16 || sr != 2 || params->rhs_zero_point != 8 ||
      (k % 2) != 0 || kai_k_roundedup(k) > 32768) {
    kai_run_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(num_groups, n, k, nr, kr, sr, rhs,
                                           bias, scale, rhs_packed, extra_bytes,
                                           params);
    return;
  }

  const size_t k_internal = kai_k_roundedup(k);
  const size_t rhs_packed_stride =
    8 * ((k_internal / 2) + kai_num_bytes_multiplier_rhs +
         kai_num_bytes_sum_rhs + kai_num_bytes_bias);
  const size_t dst_num_rows = kai_roundup(n, 8) / 8;
  const size_t rhs_stride = k / 2;
  const size_t num_chunks = k_internal / 32;
  const size_t num_full_chunks = k / 32;
  const size_t tail_valid_bytes = (k - num_full_chunks * 32) / 2;

  const uint8x16_t vmask_lo = vdupq_n_u8(0x0F);
  const uint8x16_t vxor = vdupq_n_u8(0x88);

  for (size_t dst_row_idx = 0; dst_row_idx < dst_num_rows; ++dst_row_idx) {
    uint8_t *dst_row = (uint8_t *)rhs_packed + dst_row_idx * rhs_packed_stride;

    int32_t *sums = (int32_t *)(dst_row + 8 * (k_internal / 2));
    float *scales_out = (float *)((uint8_t *)sums + 8 * kai_num_bytes_sum_rhs);
    float *bias_out =
      (float *)((uint8_t *)scales_out + 8 * kai_num_bytes_multiplier_rhs);

    // Clamp the row indices to avoid out-of-bound reads
    const uint8_t *src_row[8];
    for (size_t i = 0; i < 8; ++i) {
      src_row[i] = rhs + KAI_MIN(dst_row_idx * 8 + i, n - 1) * rhs_stride;
    }

    uint16x8_t acc[8];
    for (size_t i = 0; i < 8; ++i) {
      acc[i] = vdupq_n_u16(0);
    }

    for (size_t t = 0; t < num_full_chunks; ++t) {
      uint8_t *dst_t = dst_row + 128 * t;

      for (size_t p = 0; p < 4; ++p) {
        const uint8x16_t v0 = vld1q_u8(src_row[2 * p] + 16 * t);
        const uint8x16_t v1 = vld1q_u8(src_row[2 * p + 1] + 16 * t);

        const uint8x8_t v0l = vget_low_u8(v0);
        const uint8x8_t v0h = vget_high_u8(v0);
        const uint8x8_t v1l = vget_low_u8(v1);
        const uint8x8_t v1h = vget_high_u8(v1);

        const uint8x8x2_t z0 =
          vzip_u8(vsli_n_u8(v0l, v0h, 4), vsri_n_u8(v0h, v0l, 4));
        const uint8x8x2_t z1 =
          vzip_u8(vsli_n_u8(v1l, v1h, 4), vsri_n_u8(v1h, v1l, 4));

        // Adjacent rows share contiguous 8-byte slots -> one 16-byte store
        vst1q_u8(dst_t + 16 * p,
                 veorq_u8(vcombine_u8(z0.val[0], z1.val[0]), vxor));
        vst1q_u8(dst_t + 64 + 16 * p,
                 veorq_u8(vcombine_u8(z0.val[1], z1.val[1]), vxor));

        acc[2 * p] = vpadalq_u8(
          acc[2 * p], vaddq_u8(vandq_u8(v0, vmask_lo), vshrq_n_u8(v0, 4)));
        acc[2 * p + 1] = vpadalq_u8(
          acc[2 * p + 1], vaddq_u8(vandq_u8(v1, vmask_lo), vshrq_n_u8(v1, 4)));
      }
    }

    if (num_full_chunks < num_chunks) {
      // Tail chunk: pad the source with 0x88 so padded positions pack to
      // (8 | 8 << 4) ^ 0x88 == 0 and contribute zero to the sums, matching
      // the reference implementation.
      const size_t t = num_full_chunks;
      uint8_t *dst_t = dst_row + 128 * t;

      for (size_t i = 0; i < 8; ++i) {
        uint8_t buf[16];
        memset(buf, 0x88, sizeof(buf));
        memcpy(buf, src_row[i] + 16 * t, tail_valid_bytes);

        const uint8x16_t v = vld1q_u8(buf);
        const uint8x8_t vl = vget_low_u8(v);
        const uint8x8_t vh = vget_high_u8(v);
        const uint8x8x2_t z =
          vzip_u8(vsli_n_u8(vl, vh, 4), vsri_n_u8(vh, vl, 4));

        vst1_u8(dst_t + 8 * i, veor_u8(z.val[0], vget_low_u8(vxor)));
        vst1_u8(dst_t + 64 + 8 * i, veor_u8(z.val[1], vget_low_u8(vxor)));

        acc[i] =
          vpadalq_u8(acc[i], vaddq_u8(vandq_u8(v, vmask_lo), vshrq_n_u8(v, 4)));
      }
    }

    // Reduction sums: subtract 2 * rhs_zero_point per packed byte, then
    // adjust by 16 as in the reference implementation.
    for (size_t i = 0; i < 8; ++i) {
      sums[i] = ((int32_t)vaddlvq_u16(acc[i]) - 8 * (int32_t)k_internal) * 16;
    }

    // Adjust the scales
    for (size_t i = 0; i < 8; ++i) {
      // Clamp the row index to avoid out-of-bound reads
      const size_t src_row_idx = KAI_MIN(dst_row_idx * 8 + i, n - 1);
      scales_out[i] = scale[src_row_idx] * 0.0625F;
    }

    // Set the bias
    if (bias == NULL) {
      memset(bias_out, 0, 8 * kai_num_bytes_bias);
    } else {
      for (size_t i = 0; i < 8; ++i) {
        // Clamp the row index to avoid out-of-bound reads
        const size_t src_row_idx = KAI_MIN(dst_row_idx * 8 + i, n - 1);
        bias_out[i] = bias[src_row_idx];
      }
    }
  }
}

#endif // Architectural features check.
