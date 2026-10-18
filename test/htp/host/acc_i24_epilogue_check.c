// SPDX-License-Identifier: Apache-2.0
/**
 * @file acc_i24_epilogue_check.c
 * @brief Real HVX signed24 GU/DN epilogues versus full int32 tiles.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#include "hvx_dequant_i32.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum { NP = 3, NT = 2 * NP, ROWS = 64, WIDTH = 160, COLS = 2 * WIDTH };
#ifndef ACC_I24_TILE_BYTES
#define ACC_I24_TILE_BYTES 8192u
#endif
static int32_t tiles[NT][2048] __attribute__((aligned(128)));
static uint8_t planes[NT][ACC_I24_TILE_BYTES] __attribute__((aligned(128)));
static uint8_t saved[sizeof planes];
static float out[ROWS][WIDTH], ref[ROWS][WIDTH];

int main(void) {
  float as[ROWS], ws[COLS], bias[COLS], weights[ROWS];
  int32_t az[ROWS], cs[COLS];
  uint32_t rows[ROWS];
  uint8_t seen[ROWS], seen_ref[ROWS];
  const int32_t edge[] = {
    -8388608, -8388607, -65537, -65536, -257, -256,  -129,  -128,    -1,     0,
    1,        127,      128,    255,    256,  65535, 65536, 8388606, 8388607};
  uint32_t state = 8319u;
  memset(planes, 0x5a, sizeof planes);
  for (uint32_t t = 0; t < NT; ++t)
    for (uint32_t p = 0; p < 2048u; ++p) {
      state = state * 1664525u + 1013904223u;
      const int32_t v = p < sizeof edge / sizeof *edge
                          ? edge[p]
                          : (int32_t)(state & 0xffffffu) - 8388608;
      tiles[t][p] = v;
      for (uint32_t b = 0; b < 3u; ++b)
        planes[t][b * 2048u + p] = (uint8_t)((uint32_t)v >> (b * 8u));
    }
  memcpy(saved, planes, sizeof planes);
  for (uint32_t r = 0; r < ROWS; ++r) {
    as[r] = 0.0001f * (float)(1u + r % 3u);
    az[r] = 127 + (int32_t)r;
    weights[r] = 0.125f * (float)(1u + r % 5u);
    rows[r] = (r * 13u + 17u) % ROWS;
  }
  for (uint32_t c = 0; c < COLS; ++c) {
    ws[c] = 0.001f * (float)(1u + c % 5u);
    bias[c] = ((float)(c % 7u) - 3.f) * 0.01f;
    cs[c] = (int32_t)(c * 317u % 2048u) - 1024;
  }
  const uint32_t counts[] = {1, 2, 3, 4, 7, 15, 16, 31, 63, 64};
  uint32_t cases = 0;
  for (uint32_t act = 0; act < 2u; ++act)
    for (uint32_t ci = 0; ci < sizeof counts / sizeof *counts; ++ci)
      for (uint32_t workers = 1; workers <= 8u; ++workers) {
        memset(out, 0xa5, sizeof out);
        memset(ref, 0xa5, sizeof ref);
        hvx_dq_swiglu_job j = {(const uint8_t *)tiles,
                               sizeof tiles[0],
                               NP,
                               1u,
                               32u,
                               counts[ci],
                               as,
                               az,
                               cs,
                               ws,
                               bias,
                               WIDTH,
                               &ref[0][0],
                               WIDTH,
                               act};
        hvx_dq_swiglu_worker(1, 0, &j);
        j.tiles_base = (const uint8_t *)planes;
        j.tile_stride = sizeof planes[0];
        j.dst = &out[0][0];
        for (uint32_t w = 0; w < workers; ++w)
          hvx_dq_swiglu_i24(workers, w, &j);
        assert(memcmp(out, ref, sizeof out) == 0);
        assert(memcmp(planes, saved, sizeof planes) == 0);
        ++cases;
      }
  for (uint32_t ci = 0; ci < sizeof counts / sizeof *counts; ++ci)
    for (uint32_t workers = 1; workers <= 8u; ++workers)
      for (uint32_t use_seen = 0; use_seen < 2u; ++use_seen) {
        memset(out, 0, sizeof out);
        memset(ref, 0, sizeof ref);
        for (uint32_t r = 0; r < ROWS; ++r) {
          seen[r] = seen_ref[r] = (uint8_t)(r % 3u == 0u);
          for (uint32_t c = 0; c < WIDTH; ++c)
            out[r][c] = ref[r][c] = (float)((r * 17u + c) % 23u) * 0.02f;
        }
        for (uint32_t repeat = 0; repeat < 3u; ++repeat) {
          hvx_dequant_acc_tiles_scatter_rows_f32(
            (const uint8_t *)tiles, sizeof tiles[0], NP, 32u, 0u, counts[ci],
            as, az, cs, ws, bias, &ref[0][0], WIDTH, rows, weights,
            use_seen ? seen_ref : NULL);
          for (uint32_t w = 0; w < workers; ++w)
            hvx_dequant_i24_tiles_scatter_rows_f32(
              (const uint8_t *)planes, sizeof planes[0], NP,
              counts[ci] * w / workers, counts[ci] * (w + 1u) / workers, as, az,
              cs, ws, bias, &out[0][0], WIDTH, rows, weights,
              use_seen ? seen : NULL);
          assert(memcmp(out, ref, sizeof out) == 0);
          assert(memcmp(seen, seen_ref, sizeof seen) == 0);
          assert(memcmp(planes, saved, sizeof planes) == 0);
          ++cases;
        }
      }
  printf("ACC_I24_EPILOGUE_PASS: %u GU/DN cases, partial rows, worker slices, "
         "GLU variants, repeated scatter and read-only planes\n",
         cases);
  return 0;
}
