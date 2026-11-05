// SPDX-License-Identifier: Apache-2.0
/**
 * @file htp_tl2_codec.h
 * @brief Portable native TL2 layout and ternary weight codec.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#ifndef __NNTRAINER_HTP_TL2_CODEC_H__
#define __NNTRAINER_HTP_TL2_CODEC_H__

#include <stddef.h>
#include <stdint.h>

static inline uint32_t htp_tl2_blocks(uint32_t K) {
  return K / 24u + (K % 24u != 0);
}

static inline uint64_t htp_tl2_native_bytes(uint32_t K, uint32_t N) {
  return (uint64_t)htp_tl2_blocks(K) * (N / 128u + (N % 128u != 0)) * 640u;
}

/** Checked size stays inline for the kernel's partition selection. */
static inline size_t htp_tl2_weight_size(uint32_t K, uint32_t N) {
  if (!K || !N || K > INT32_MAX / 255u || N > UINT32_MAX - 127u)
    return 0;
  const uint64_t bytes = htp_tl2_native_bytes(K, N);
  return bytes > SIZE_MAX ? 0 : (size_t)bytes;
}

static inline uint32_t htp_tl2_column(uint32_t panel, uint32_t lane) {
  return panel * 128u + (lane % 4u) * 32u + lane / 4u;
}

/** Add one balanced-base-three triple to an initially zeroed native block. */
static inline void htp_tl2_store_group(uint8_t *block, uint32_t group,
                                       uint32_t lane, int value) {
  const unsigned magnitude = value < 0 ? (unsigned)-value : (unsigned)value;
  block[(group / 2u) * 128u + lane] |= magnitude << (4u * (group % 2u));
  block[512u + lane] |= (value < 0) << group;
}

/** Balanced-base-three digits of an unsigned magnitude (0..13). */
static inline void htp_tl2_digits(unsigned m, int *a, int *b, int *c) {
  *a = (int)((m + 13u) / 9u) - 1;
  *b = (int)(((m + 13u) / 3u) % 3u) - 1;
  *c = (int)((m + 13u) % 3u) - 1;
}

#ifdef __cplusplus
extern "C" {
#endif

/** Native compute layout, distinct from serialized TL2CX_WH/1:
 * [ceil(N/128)][ceil(K/24)][5][128] bytes, all blocks 128-byte aligned.
 * The first four vectors hold two magnitude nibbles per output each;
 * vector four holds their eight sign bits. Each magnitude is abs(9a+3b+c),
 * a,b,c in {-1,0,1}. Byte j addresses column (j%4)*32+j/4 in its panel.
 * Missing rows/columns are zero, including negative-zero rejection.
 * Weights are symmetric ternary; per-column scales are applied separately.
 * No asymmetric palette mapping or activation requantization occurs.
 * K <= INT32_MAX/255 ensures even the worst u8 dot product fits int32.
 * pack/unpack buffers must be disjoint and sized using the helpers below.
 * Based on BitNet TL2: https://arxiv.org/abs/2502.11880 .
 */
size_t hvx_tl2_weight_bytes(uint32_t K, uint32_t N);
int hvx_tl2_pack(const int8_t *kn, uint32_t K, uint32_t N, uint8_t *packed);
int hvx_tl2_unpack(const uint8_t *packed, uint32_t K, uint32_t N, int8_t *kn);

#ifdef __cplusplus
}
#endif
#endif
