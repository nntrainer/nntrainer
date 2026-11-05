// SPDX-License-Identifier: Apache-2.0
/**
 * @file acc_i24_host_check.c
 * @brief Signed 24-bit extraction, guard, layout and scratch-bound checks.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#include "../acc_i24_probe.h"
#include <assert.h>
#include <limits.h>
#include <stdio.h>

int main(void) {
  static uint8_t scratch[HEXKL_ACC_I24_PROBE_SCRATCH_BYTES + 256u]
    __attribute__((aligned(2048)));
  static int32_t out[2048u + 64u] __attribute__((aligned(128)));
  uint8_t *lo = scratch + HEXKL_ACC_I24_PROBE_PLANE_OFF;
  uint8_t *mid = lo + HEXKL_CVT_PLANE_BYTES;
  uint8_t *hi = mid + HEXKL_CVT_PLANE_BYTES;
  const int32_t edge[] = {
    INT32_MIN, -8388609, -8388608, -8388607, -65537,  -65536,  -257,     -256,
    -129,      -128,     -1,       0,        1,       127,     128,      255,
    256,       65535,    65536,    8388606,  8388607, 8388608, INT32_MAX};
  memset(scratch, 0x5a, sizeof scratch);
  hexkl_acc_i24_prepare(scratch);
  const uint16_t scales[] = {0x6000, 0x4000, 0x2000};
  for (uint32_t p = 0; p < 3u; ++p) {
    const uint32_t *words =
      (const uint32_t *)(scratch + p * HEXKL_CVT_BLOCK_BYTES);
    for (uint32_t j = 0; j < HEXKL_CVT_BLOCK_BYTES / 4u; ++j)
      assert(words[j] == (j < 64u ? scales[p] : 0u));
  }
  uint32_t state = 12345u;
  for (uint32_t pass = 0; pass < 128u; ++pass) {
    int32_t expected[2048];
    for (uint32_t i = 0; i < 2048u; ++i) {
      state = state * 1664525u + 1013904223u;
      const int32_t acc = i < sizeof edge / sizeof *edge
                            ? edge[i]
                            : (int32_t)(state & 0xffffffu) - 8388608;
      lo[i] = hexkl_cvt_model(acc, scales[0]);
      mid[i] = hexkl_cvt_model(acc, scales[1]);
      hi[i] = hexkl_cvt_model(acc, scales[2]);
      const uint32_t bits = (uint32_t)acc & 0xffffffu;
      expected[i] = bits < 0x800000u ? (int32_t)bits : (int32_t)bits - 16777216;
    }
    memset(out, 0xa5, sizeof out);
    hexkl_acc_i24_zip(lo, mid, hi, out + 32);
    assert(memcmp(out + 32, expected, sizeof expected) == 0);
    for (uint32_t i = 0; i < 32u; ++i) {
      assert((uint32_t)out[i] == 0xa5a5a5a5u);
      assert((uint32_t)out[2080u + i] == 0xa5a5a5a5u);
    }
    for (uint32_t i = HEXKL_ACC_I24_PROBE_SCRATCH_BYTES; i < sizeof scratch;
         ++i)
      assert(scratch[i] == 0x5au);
  }
  const hexkl_acc_layout good = {1, 1, 0, 32};
  hexkl_acc_layout bad = good;
  assert(hexkl_acc_i24_usable(2816, -7, 7, &good));
  assert(hexkl_acc_i24_usable(704, -7, 7, &good));
  assert(hexkl_acc_i24_usable(8192, -1, 1, &good));
  assert(hexkl_acc_i24_usable(4699, -7, 7, &good));
  assert(!hexkl_acc_i24_usable(4700, -7, 7, &good));
  assert(hexkl_acc_i24_range(4112, -8, 0));
  assert(!hexkl_acc_i24_range(4113, -8, 0));
  assert(!hexkl_acc_i24_range(UINT32_MAX, -7, 7));
  assert(!hexkl_acc_i24_range(0, -7, 7));
  assert(!hexkl_acc_i24_range(1, -9, 7));
  assert(!hexkl_acc_i24_range(1, -7, 8));
  assert(!hexkl_acc_i24_usable(2816, -7, 7, NULL));
  bad.base = 32;
  assert(!hexkl_acc_i24_usable(2816, -7, 7, &bad));
  bad = good;
  bad.row_stride = 64;
  assert(!hexkl_acc_i24_usable(2816, -7, 7, &bad));
  bad = good;
  bad.usable = 0;
  assert(!hexkl_acc_i24_usable(2816, -7, 7, &bad));
  assert(hexkl_acc_i24_scratch_offset(3072u, 0u) == 0u);
  assert(hexkl_acc_i24_scratch_offset(3071u, 0u) == UINT32_MAX);
  assert(hexkl_acc_i24_scratch_offset(3073u, 1u) == UINT32_MAX);
  assert(hexkl_acc_i24_scratch_offset(32768u, 28672u) == 28672u);
  assert(hexkl_acc_i24_scratch_offset(32768u, 28673u) == UINT32_MAX);
  assert(hexkl_acc_i24_scratch_offset(31744u, 28672u) == 28672u);
  assert(hexkl_acc_i24_scratch_offset(31743u, 26624u) == 26624u);
  assert(hexkl_acc_i24_scratch_offset(31743u, 26625u) == UINT32_MAX);
  assert(hexkl_acc_i24_scratch_offset(32768u, UINT32_MAX) == UINT32_MAX);
  {
    uint32_t down = 0u, end = 0u;
    const uint32_t first = 5439488u, native_bytes = 1976320u;
    assert(
      hexkl_acc_i24_banks(8u << 20, first, 32u, native_bytes, &down, &end));
    assert(down == 5832704u && end == 6225920u);
    const uint32_t exact = end + native_bytes + HEXKL_ACC_I24_SCRATCH_BYTES;
    assert(hexkl_acc_i24_banks(exact, first, 32u, native_bytes, &down, &end));
    assert(
      !hexkl_acc_i24_banks(exact - 1u, first, 32u, native_bytes, &down, &end));
    assert(down == 5832704u && end == 6225920u);
    assert(!hexkl_acc_i24_banks(UINT32_MAX, UINT32_MAX - 2047u, 32u,
                                native_bytes, &down, &end));
    assert(!hexkl_acc_i24_banks(8u << 20, first, UINT32_MAX - 1u, native_bytes,
                                &down, &end));
    assert(!hexkl_acc_i24_banks(8u << 20, first + 1u, 32u, native_bytes, &down,
                                &end));
    assert(
      !hexkl_acc_i24_banks(8u << 20, first, 1u, native_bytes, &down, &end));
    assert(
      !hexkl_acc_i24_banks(8u << 20, first, 31u, native_bytes, &down, &end));
    assert(
      !hexkl_acc_i24_banks(8u << 20, first, 32u, native_bytes, NULL, &end));
  }
  puts("ACC_I24_HOST_PASS: 262144 signed lanes, range/layout guards, canaries");
  return 0;
}
