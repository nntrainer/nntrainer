// SPDX-License-Identifier: Apache-2.0
/**
 * @file tl2_quant_native_check.c
 * @brief Compare dense TL2 quantization with the former AH path using HVX.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#include "hvx_quant_u8.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static uint32_t rng = 91237u;
static uint32_t random_u32(void) {
  rng = rng * 1664525u + 1013904223u;
  return rng;
}

int main(void) {
  static float input[8192] __attribute__((aligned(128)));
  static uint8_t ah[64u * 8192u] __attribute__((aligned(128)));
  static uint8_t dense[8192u + 128u] __attribute__((aligned(128)));
  const uint32_t widths[] = {32u, 64u, 704u, 2816u, 8192u};
  unsigned cases = 0;
  for (unsigned ki = 0; ki < sizeof(widths) / sizeof(widths[0]); ++ki)
    for (unsigned pattern = 0; pattern < 8u; ++pattern)
      for (unsigned rep = 0; rep < 16u; ++rep) {
        const uint32_t K = widths[ki];
        float old_scale, new_scale;
        int32_t old_zp, new_zp;
        memset(dense, 0xa5, sizeof(dense));
        memset(ah, 0x96, sizeof(ah));
        for (uint32_t j = 0; j < K; ++j) {
          float value = ((int32_t)(random_u32() % 65536u) - 32768) / 1024.0f;
          switch (pattern) {
          case 0:
            value = 0.0f;
            break;
          case 1:
            value = -0.0f;
            break;
          case 2:
            value = (float)(random_u32() % 256u);
            break;
          case 3:
            value = -(float)(random_u32() % 256u);
            break;
          case 4:
            value *= 0x1p-20f;
            break;
          case 5:
            value *= 0x1p20f;
            break;
          case 6:
            value = (float)((int)(j % 255u) - 127) + 0.5f;
            break;
          default:
            break;
          }
          input[j] = value;
        }
        if (pattern == 6u) {
          input[0] = -127.0f;
          input[1] = 128.0f;
        }
        hvx_quant_rows_u8_params(input, 1u, 1u, K, &old_scale, &old_zp, NULL);
        hvx_quant_pack_u8_ah_m1_repeat(input, K, old_scale, old_zp, ah);
        hvx_quant_rows_u8_params_rm(input, 1u, 1u, K, &new_scale, &new_zp,
                                    dense, NULL, 0.0f, NULL, NULL);
        assert(!memcmp(&old_scale, &new_scale, sizeof(old_scale)));
        assert(old_zp == new_zp);
        for (uint32_t k = 0; k < K; k += 32u)
          assert(!memcmp(dense + k, ah + (size_t)(k / 32u) * 2048u, 32u));
        for (uint32_t j = K; j < sizeof(dense); ++j)
          assert(dense[j] == 0xa5);
        ++cases;
      }
  printf("TL2 dense vs repeated AH quantization exact: %u cases\n", cases);
  return 0;
}
