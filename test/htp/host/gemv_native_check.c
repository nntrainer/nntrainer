// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   gemv_native_check.c
 * @date   22 Sep 2026
 * @brief  The real HVX u8 x i4 GEMV (hvx_gemm_u8i4_wh.c) built on x86
 *         against the SDK's HVX emulation, held against a scalar spec
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * moe_layer_host_check replaces the GEMV with a scalar stand-in, so it
 * checks the plumbing around the kernel, not the kernel. This one compiles
 * the kernel itself -- the same source the skel builds -- with the Hexagon
 * tools' libnative, which implements every HVX intrinsic in C, and checks
 * each output int32 against the plain sum over the WH nibbles. That sum is
 * what the HMX produces for the same tiles (hvx_gemm_u8i4_wh.h), so a pass
 * here is the kernel's arithmetic, not a stand-in's. What it cannot see:
 * the l2fetch (compiled out off-target), timing, and the pool's lanes; the
 * device gtest MoeLayerM1GemvMatchesHmx covers those.
 *
 * Every row count 1..16 runs, so both the one-row loop (m = 1, 5, 9, 13)
 * and the four-row loop run, through the self-prefetching column call and
 * the one without its own l2fetch. Rows past m must stay untouched.
 */
#include "hvx_gemm_u8i4_wh.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The device's WH nibble layout (htp_wh_layout.h), the same function the
   moe_layer_host_check stand-ins read tiles through. */
static int wh_value(const uint8_t *tile, uint32_t k, uint32_t c) {
  const uint32_t byte = (k / 8u) * 128u + c * 4u + (k % 4u);
  const int nib = (tile[byte] >> (((k / 4u) % 2u) ? 4 : 0)) & 0xF;
  return nib >= 8 ? nib - 16 : nib;
}

static uint32_t g_rng = 0x105u;
static uint8_t rnd8(void) {
  g_rng = g_rng * 1664525u + 1013904223u;
  return (uint8_t)(g_rng >> 24);
}

#define GUARD 0x5A5A5A5A

typedef void (*col_fn)(const uint8_t *, uint32_t, uint32_t, const uint8_t *,
                       uint32_t, uint32_t, int32_t *);

/* One (m, k_tiles, n_col, nt) case through one entry point: the number of
   int32 that differ from the scalar spec plus the guard words written. */
static uint32_t run_case(col_fn fn, const uint8_t *act, const uint8_t *wh,
                         uint32_t m, uint32_t k_tiles, uint32_t n_col,
                         uint32_t nt) {
  int32_t out[(HVX_GEMM_U8I4_MAX_ROWS + 1u) * 32u];
  for (uint32_t i = 0; i < sizeof out / sizeof out[0]; ++i)
    out[i] = GUARD;
  fn(act, m, k_tiles, wh, n_col, nt, out);
  uint32_t bad = 0;
  for (uint32_t r = 0; r < m; ++r)
    for (uint32_t c = 0; c < 32u; ++c) {
      int32_t s = 0;
      for (uint32_t kt = 0; kt < k_tiles; ++kt) {
        const uint8_t *tile = wh + ((size_t)kt * n_col + nt) * 512u;
        const uint8_t *arow = act + (size_t)kt * 2048u + r * 32u;
        for (uint32_t k = 0; k < 32u; ++k)
          s += (int32_t)arow[k] * wh_value(tile, k, c);
      }
      bad += out[r * 32u + c] != s;
    }
  for (uint32_t i = m * 32u; i < sizeof out / sizeof out[0]; ++i)
    bad += out[i] != GUARD;
  return bad;
}

int main(void) {
  static const uint32_t kts[] = {1u, 56u, 64u};
  static const uint32_t ncs[] = {64u, 112u};
  const size_t act_bytes = 64u * 2048u, wh_max = 64u * 112u * 512u;
  uint8_t *act = (uint8_t *)malloc(act_bytes);
  uint8_t *wh = (uint8_t *)malloc(wh_max);
  if (!act || !wh)
    return 2;
  uint32_t bad = 0, cases = 0;
  /* data 0: random bytes; data 1: the extremes, activation 255 against
     nibble -8 everywhere -- the largest |sum| the header's bound allows. */
  for (int data = 0; data < 2; ++data) {
    for (size_t i = 0; i < act_bytes; ++i)
      act[i] = data ? 0xFFu : rnd8();
    for (size_t i = 0; i < wh_max; ++i)
      wh[i] = data ? 0x88u : rnd8();
    for (size_t a = 0; a < sizeof kts / sizeof kts[0]; ++a)
      for (size_t b = 0; b < sizeof ncs / sizeof ncs[0]; ++b) {
        const uint32_t nts[] = {0u, 1u, ncs[b] - 1u};
        for (size_t t = 0; t < 3; ++t)
          for (uint32_t m = 1; m <= HVX_GEMM_U8I4_MAX_ROWS; ++m)
            for (int f = 0; f < 2; ++f) {
              const uint32_t b1 =
                run_case(f ? hvx_gemm_u8i4_wh_col_nopf : hvx_gemm_u8i4_wh_col,
                         act, wh, m, kts[a], ncs[b], nts[t]);
              bad += b1;
              ++cases;
              if (b1)
                printf("HVX GEMV NATIVE %s m=%u k_tiles=%u n_col=%u nt=%u "
                       "data=%d bad=%u\n",
                       f ? "col_nopf" : "col", m, kts[a], ncs[b], nts[t], data,
                       b1);
            }
      }
  }
  printf("HVX GEMV NATIVE cases=%u bad=%u\n", cases, bad);
  printf(
    bad ? "HVX GEMV NATIVE DIFFERS FROM THE SCALAR SPEC\n"
        : "HVX GEMV NATIVE BIT-IDENTICAL (libnative; m=1..16 rows1+rows4)\n");
  free(act);
  free(wh);
  return bad ? 1 : 0;
}
