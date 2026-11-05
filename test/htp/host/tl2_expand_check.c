// SPDX-License-Identifier: Apache-2.0
/** @file tl2_expand_check.c
 * @brief Independent TL2 to WH layout, padding and magnitude checks.
 * Also runs on the Hexagon simulator to exercise the actual HVX branch.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug See docs/gemma4-tl2-affine-handoff.md for validation limits.
 */
#include "hvx_gemv_u8tl2.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

static int8_t weights[96 * 128] __attribute__((aligned(128)));
static uint8_t packed[2560 + 128] __attribute__((aligned(128)));
static uint8_t actual[3 * 4096 + 256] __attribute__((aligned(128)));
static uint8_t expected[sizeof(actual)] __attribute__((aligned(128)));

int main(void) {
  unsigned cases = 0;
  for (uint32_t rows = 32; rows <= 96; rows += 32)
    for (uint32_t columns = 32; columns <= 128; columns += 32)
      for (uint32_t stride = 2048; stride <= 4096; stride += 2048)
        for (int magnitude = 1; magnitude <= 7; magnitude += 6)
          for (uint32_t pattern = 0; pattern < 4; ++pattern)
            for (uint32_t unaligned = 0; unaligned < 2; ++unaligned) {
              uint8_t *source = packed + unaligned;
              const uint32_t dst_offset = 128u + 3u * unaligned;
              for (uint32_t k = 0; k < rows; ++k)
                for (uint32_t n = 0; n < 128; ++n) {
                  /* Cover all 27 triples, both signs and panel boundaries. */
                  unsigned code = (n + k / 3u * 7u) % 27u;
                  for (uint32_t t = 0; t < k % 3u; ++t)
                    code /= 3u;
                  weights[k * 128u + n] = n >= columns ? 0
                                          : pattern    ? (int)pattern - 2
                                                       : (int)(code % 3u) - 1;
                }
              memset(packed, 0xa5, sizeof(packed));
              assert(hvx_tl2_pack(weights, rows, 128, source) == 0);
              memset(actual, 0xa5, sizeof(actual));
              memset(expected, 0xa5, sizeof(expected));
              for (uint32_t kt = 0; kt < rows / 32u; ++kt)
                memset(expected + dst_offset + kt * stride, 0, columns * 16u);
              for (uint32_t k = 0; k < rows; ++k)
                for (uint32_t n = 0; n < columns; ++n) {
                  const size_t pos = dst_offset + k / 32u * stride +
                                     n / 32u * 512u + k % 32u / 8u * 128u +
                                     n % 32u * 4u + k % 4u;
                  const int q = weights[k * 128u + n] * magnitude;
                  expected[pos] |= (q & 15) << (4u * (k / 4u % 2u));
                }
              hvx_tl2_expand_wh96(source, rows, columns, stride, magnitude,
                                  actual + dst_offset);
              if (memcmp(actual, expected, sizeof(actual))) {
                for (size_t i = 0; i < sizeof(actual); ++i)
                  if (actual[i] != expected[i]) {
                    printf("FAIL rows=%u columns=%u stride=%u magnitude=%d "
                           "pattern=%u unaligned=%u byte=%zu actual=%u "
                           "expected=%u\n",
                           (unsigned)rows, (unsigned)columns, (unsigned)stride,
                           magnitude, (unsigned)pattern, (unsigned)unaligned, i,
                           actual[i], expected[i]);
                    return 1;
                  }
              }
              for (size_t i = 0; i < unaligned; ++i)
                assert(packed[i] == 0xa5);
              for (size_t i = unaligned + hvx_tl2_weight_bytes(rows, 128);
                   i < sizeof(packed); ++i)
                assert(packed[i] == 0xa5);
              ++cases;
            }
  printf("TL2_EXPAND_PASS cases=%u exact=1 guards=1 unaligned=1\n", cases);
  return 0;
}
