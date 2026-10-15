// SPDX-License-Identifier: Apache-2.0
/** @file wh_affine_vector_check.c
 * @brief Exact AH row sums and affine accumulator correction on host / HVX.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug See docs/gemma4-tl2-affine-handoff.md for validation limits.
 */
#include "hexkl_wh_affine.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

static uint8_t ah[8192 * 64] __attribute__((aligned(128)));
static int32_t actual[64 * 128 + 64] __attribute__((aligned(128)));
static int32_t expected[64 * 128 + 64];

int main(void) {
  const uint32_t widths[] = {32, 96, 704, 2816, 8192};
  const uint32_t counts[] = {1, 2, 3, 4, 31, 63, 64};
  const uint32_t columns[] = {1, 31, 32, 33, 64, 97};
  unsigned cases = 0;
  for (uint32_t ki = 0; ki < sizeof(widths) / sizeof(*widths); ++ki) {
    const uint32_t K = widths[ki];
    int32_t zp[64], reference_sums[64], sums[66];
    for (uint32_t r = 0; r < 64; ++r) {
      zp[r] = (r * 67u) % 256u;
      reference_sums[r] = 0;
      for (uint32_t k = 0; k < K; ++k) {
        const uint8_t value = (r * 17u + k * 13u + k / 7u) & 255u;
        ah[k / 32u * 2048u + r * 32u + k % 32u] = value;
        reference_sums[r] += (int32_t)value - zp[r];
      }
    }
    for (uint32_t ri = 0; ri < sizeof(counts) / sizeof(*counts); ++ri) {
      const uint32_t rows = counts[ri];
      for (uint32_t j = 0; j < 66; ++j)
        sums[j] = 1234567;
      hexkl_wh_affine_sums(ah, rows, K, zp, sums + 1);
      assert(sums[0] == 1234567);
      assert(!memcmp(sums + 1, reference_sums, rows * sizeof(int32_t)));
      for (uint32_t j = rows + 1; j < 66; ++j)
        assert(sums[j] == 1234567);
      for (uint32_t ci = 0; ci < sizeof(columns) / sizeof(*columns); ++ci) {
        const uint32_t cols = columns[ci], stride = cols + 7;
        int32_t offsets[128];
        for (uint32_t c = 0; c < cols; ++c)
          offsets[c] = (int32_t)(c % 31u) - 15;
        for (uint32_t j = 0; j < sizeof(actual) / sizeof(*actual); ++j)
          expected[j] = actual[j] = (int32_t)(j * 31u) - 123456;
        /* An unaligned base and stride exercise accumulator tile views. */
        for (uint32_t r = 0; r < rows; ++r)
          for (uint32_t c = 0; c < cols; ++c)
            expected[3 + r * stride + c] += reference_sums[r] * offsets[c];
        hexkl_wh_affine_acc(actual + 3, stride, rows, cols, offsets, sums + 1);
        assert(!memcmp(actual, expected, sizeof(actual)));
        hexkl_wh_affine_acc(actual + 3, stride, rows, cols, NULL, sums + 1);
        assert(!memcmp(actual, expected, sizeof(actual)));
        ++cases;
      }
    }
  }
  printf("WH_AFFINE_VECTOR_PASS cases=%u exact=1 guards=1\n", cases);
  return 0;
}
