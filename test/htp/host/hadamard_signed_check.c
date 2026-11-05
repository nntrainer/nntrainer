// SPDX-License-Identifier: Apache-2.0
/** @file hadamard_signed_check.c
 * @brief Exact transform-then-XOR parity, alignment and bounds checks.
 * @author NNTrainer contributors
 * @bug No known bugs.
 */
#include "hvx_hadamard_f32.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int reference_hvx_hadamard_f32(float *, uint32_t, uint32_t);
int reference_hvx_hadamard_rows_f32(float *, uint32_t, uint32_t, uint32_t);
static uint32_t rng = 71;
static uint32_t random_word(void) {
  rng ^= rng << 13;
  rng ^= rng >> 17;
  return rng ^= rng << 5;
}
static uint32_t data(uint32_t pattern, size_t i) {
  static const uint32_t special[] = {
    0,           0x80000000u, 1,           0x80000001u, 0x007fffffu,
    0x807fffffu, 0x7f7fffffu, 0xff7fffffu, 0x7f800000u, 0xff800000u,
    0x7fc12345u, 0xffc54321u, 0x7f800001u, 0xff800001u};
  if (pattern == 0)
    return (random_word() & 0x807fffffu) | 0x3f000000u;
  if (pattern == 1)
    return 0;
  if (pattern == 2)
    return (i & 1u) ? 0x80000000u : 0;
  if (pattern == 3)
    return (i & 1u) ? 0x80000001u : 1;
  if (pattern == 4)
    return (i & 1u) ? 0xff7fffffu : 0x7f7fffffu;
  if (pattern == 5)
    return special[i % (sizeof(special) / sizeof(special[0]))];
  if (pattern == 6)
    return i % 37u ? 0 : 0x3f800000u;
  return 0xbf400000u;
}
static void check(uint32_t n, uint32_t rows, uint32_t mode, uint32_t offset,
                  uint32_t mask_offset, uint32_t pattern,
                  uint32_t mask_pattern) {
  const size_t count = (size_t)n * rows;
  const size_t bytes = (count + 96u) * sizeof(float);
  uint8_t *a, *b, *m;
  assert(!posix_memalign((void **)&a, 128, bytes));
  assert(!posix_memalign((void **)&b, 128, bytes));
  assert(!posix_memalign((void **)&m, 128, (n + 96u) * sizeof(uint32_t)));
  memset(a, 0xa5, bytes);
  memset(b, 0xa5, bytes);
  memset(m, 0x5a, (n + 96u) * sizeof(uint32_t));
  float *ref = (float *)a + 32u + offset;
  float *out = (float *)b + 32u + offset;
  uint32_t *mask = (uint32_t *)m + 32u + mask_offset;
  for (size_t i = 0; i < count; ++i) {
    uint32_t bits = data(pattern, i);
    memcpy(ref + i, &bits, sizeof(bits));
  }
  memcpy(out, ref, count * sizeof(float));
  for (uint32_t i = 0; i < n; ++i)
    mask[i] = mask_pattern == 0   ? 0
              : mask_pattern == 1 ? 0x80000000u
              : mask_pattern == 2 ? ((i & 1u) << 31)
                                  : (random_word() & 0x80000000u);
  assert(reference_hvx_hadamard_rows_f32(ref, rows, n, mode) == 0);
  for (uint32_t r = 0; r < rows; ++r)
    for (uint32_t i = 0; i < n; ++i) {
      uint32_t v;
      memcpy(&v, ref + (size_t)r * n + i, 4);
      v ^= mask[i];
      memcpy(ref + (size_t)r * n + i, &v, 4);
    }
  assert(hvx_hadamard_signed_rows_f32(out, rows, n, mode, mask) == 0);
  if (memcmp(a, b, bytes)) {
    for (size_t i = 0; i < count; ++i) {
      uint32_t av, bv;
      memcpy(&av, ref + i, 4);
      memcpy(&bv, out + i, 4);
      if (av != bv) {
        fprintf(stderr,
                "FAIL n=%u rows=%u mode=%u align=%u maskalign=%u pattern=%u "
                "mask=%u i=%zu %08x != %08x\n",
                n, rows, mode, offset, mask_offset, pattern, mask_pattern, i,
                av, bv);
        break;
      }
    }
    abort();
  }
  for (uint32_t i = 0; i < 32u + mask_offset; ++i)
    assert(((uint32_t *)m)[i] == 0x5a5a5a5au);
  for (uint32_t i = 32u + mask_offset + n; i < n + 96u; ++i)
    assert(((uint32_t *)m)[i] == 0x5a5a5a5au);
  /* Also exercise the single-row API, then its null-mask wrapper. */
  if (rows == 1) {
    for (uint32_t i = 0; i < n; ++i) {
      uint32_t bits = data(pattern, i);
      memcpy(ref + i, &bits, 4);
    }
    memcpy(out, ref, n * sizeof(float));
    assert(reference_hvx_hadamard_f32(ref, n, mode) == 0);
    assert(hvx_hadamard_signed_f32(out, n, mode, NULL) == 0);
    assert(!memcmp(a, b, bytes));
    for (uint32_t i = 0; i < n; ++i) {
      uint32_t bits = data(pattern, i);
      memcpy(ref + i, &bits, 4);
    }
    memcpy(out, ref, n * sizeof(float));
    assert(reference_hvx_hadamard_f32(ref, n, mode) == 0);
    for (uint32_t i = 0; i < n; ++i) {
      uint32_t bits;
      memcpy(&bits, ref + i, 4);
      bits ^= mask[i];
      memcpy(ref + i, &bits, 4);
    }
    assert(hvx_hadamard_signed_f32(out, n, mode, mask) == 0);
    assert(!memcmp(a, b, bytes));
  }
  free(a);
  free(b);
  free(m);
}
int main(void) {
  unsigned cases = 0;
  const uint32_t widths[] = {1536, 2048};
  const uint32_t rows[] = {0, 1, 3, 17};
  for (unsigned w = 0; w < 2; ++w)
    for (unsigned r = 0; r < 4; ++r)
      for (unsigned mode = 1; mode <= 2; ++mode)
        for (unsigned pattern = 0; pattern < 8; ++pattern)
          for (unsigned mask = 0; mask < 4; ++mask) {
            const unsigned offset = (pattern + mask) & 1,
                           mask_offset = (pattern >> 1) & 1;
            check(widths[w], rows[r], mode, offset, mask_offset, pattern, mask);
            ++cases;
          }
  const uint32_t fallback[] = {1, 16, 32, 64, 128, 176, 704, 2816, 8192};
  for (unsigned w = 0; w < 9; ++w)
    for (unsigned mode = 1; mode <= 2; ++mode) {
      /** Native H44 assembly cannot execute on the host: exercise scalar
       * dispatch. */
      check(fallback[w], 3, mode, 1, 1, 0, 3);
      ++cases;
    }
  uint32_t mask = 0x80000000u;
  float x = 91;
  assert(hvx_hadamard_signed_f32(NULL, 1536, 2, &mask) == -1);
  assert(hvx_hadamard_signed_f32(&x, 0, 2, &mask) == -1);
  assert(hvx_hadamard_signed_f32(&x, 2112, 2, &mask) == -1);
  assert(hvx_hadamard_signed_f32(&x, 1, 3, &mask) == -1);
  assert(hvx_hadamard_signed_rows_f32(NULL, 0, 1536, 2, &mask) == -1);
  assert(hvx_hadamard_signed_rows_f32(&x, 1, 2112, 2, &mask) == -1);
  assert(hvx_hadamard_signed_rows_f32(&x, 0, 1536, 2, &mask) == 0);
  assert(x == 91);
  printf("PASS hadamard signed exact/canary cases=%u\n", cases);
}
