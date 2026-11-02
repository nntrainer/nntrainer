// SPDX-License-Identifier: Apache-2.0
/**
 * @file htp_tl2_codec.c
 * @brief Portable packing and validation of native ternary TL2 weights.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#include "htp_tl2_codec.h"
#include <string.h>

size_t hvx_tl2_weight_bytes(uint32_t K, uint32_t N) {
  return htp_tl2_weight_size(K, N);
}

int hvx_tl2_pack(const int8_t *kn, uint32_t K, uint32_t N, uint8_t *packed) {
  const size_t bytes = hvx_tl2_weight_bytes(K, N);
  if (!kn || !packed || !bytes || (size_t)K > SIZE_MAX / N)
    return -1;
  for (size_t i = 0; i < (size_t)K * N; ++i)
    if (kn[i] < -1 || kn[i] > 1)
      return -1;
  memset(packed, 0, bytes);
  for (uint32_t p = 0; p < (N + 127u) / 128u; ++p)
    for (uint32_t kb = 0; kb < htp_tl2_blocks(K); ++kb) {
      uint8_t *dst = packed + ((size_t)p * htp_tl2_blocks(K) + kb) * 640u;
      for (uint32_t j = 0; j < 128u; ++j) {
        const uint32_t n = htp_tl2_column(p, j);
        for (uint32_t g = 0; g < 8u; ++g) {
          int value = 0;
          for (uint32_t t = 0; t < 3u; ++t) {
            const uint32_t k = kb * 24u + g * 3u + t;
            value = 3 * value + (k < K && n < N ? kn[(size_t)k * N + n] : 0);
          }
          htp_tl2_store_group(dst, g, j, value);
        }
      }
    }
  return 0;
}

int hvx_tl2_unpack(const uint8_t *packed, uint32_t K, uint32_t N, int8_t *kn) {
  if (!packed || !kn || !hvx_tl2_weight_bytes(K, N) || (size_t)K > SIZE_MAX / N)
    return -1;
  for (uint32_t p = 0; p < (N + 127u) / 128u; ++p)
    for (uint32_t kb = 0; kb < htp_tl2_blocks(K); ++kb) {
      const uint8_t *src = packed + ((size_t)p * htp_tl2_blocks(K) + kb) * 640u;
      for (uint32_t j = 0; j < 128u; ++j) {
        const uint32_t n = htp_tl2_column(p, j);
        for (uint32_t g = 0; g < 8u; ++g) {
          const unsigned mag =
            (src[(g / 2u) * 128u + j] >> (4u * (g % 2u))) & 15u;
          const int sign = (src[512u + j] >> g) & 1u;
          if (mag > 13u || (!mag && sign))
            return -1;
          int t[3];
          htp_tl2_digits(mag, &t[0], &t[1], &t[2]);
          for (uint32_t i = 0; i < 3u; ++i) {
            const uint32_t k = kb * 24u + g * 3u + i;
            const int w = sign ? -t[i] : t[i];
            if (k < K && n < N)
              kn[(size_t)k * N + n] = (int8_t)w;
            else if (w)
              return -1;
          }
        }
      }
    }
  return 0;
}
