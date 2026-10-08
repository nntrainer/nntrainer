// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   tile_f16_host_check.c
 * @date   8 October 2026
 * @brief  hvx_tile_f16_rows_to_tile_transposed on the lane emulation,
 *         bit for bit against the word-transpose definition
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The K^T tile of attention: vector i holds word i (fp16 columns 2i,
 * 2i+1) of each of the 32 K rows in row order. The reference is that
 * sentence in scalar code -- the builder it replaced -- at the row
 * strides the model has (head dims 64, 128, 256, 512) and at every
 * 32-column offset inside a row, on random bit patterns.
 */
#include "hvx_tile_f16.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t next_u32(uint32_t *s) {
  *s = *s * 1664525u + 1013904223u;
  return *s;
}

/** @brief The definition: buf[i][r] = word i of row r. */
static void reference(uint8_t *dst, const uint16_t *src, uint32_t stride) {
  uint32_t buf[HVX_TILE_F16_VECS][32];
  for (uint32_t r = 0; r < 32u; ++r) {
    uint32_t words[16];
    memcpy(words, src + r * stride, HVX_TILE_F16_ROW_BYTES);
    for (uint32_t i = 0; i < HVX_TILE_F16_VECS; ++i) {
      buf[i][r] = words[i];
    }
  }
  memcpy(dst, buf, sizeof(buf));
}

int main(void) {
  static const uint32_t hds[] = {64u, 128u, 256u, 512u};
  uint32_t seed = 0x7116F16Au;
  int bad = 0, cases = 0;
  for (size_t h = 0; h < sizeof(hds) / sizeof(hds[0]); ++h) {
    const uint32_t hd = hds[h];
    /* 33 rows: the builder reads a whole vector per row, 64 bytes past the
       32 fp16 it uses, like the V builder; the extra row keeps that read
       inside the buffer, as the landing buffers' following region does. */
    const size_t n = (size_t)33u * hd;
    uint16_t *rows = (uint16_t *)malloc(n * sizeof(uint16_t));
    for (size_t i = 0; i < n; ++i) {
      rows[i] = (uint16_t)next_u32(&seed);
    }
    for (uint32_t col = 0; col < hd; col += HVX_TILE_F16_COLS) {
      uint8_t want[HVX_TILE_F16_VECS * 128u];
      HVX_Vector got[HVX_TILE_F16_VECS];
      reference(want, rows + col, hd);
      hvx_tile_f16_rows_to_tile_transposed(got, rows + col, hd);
      ++cases;
      if (memcmp(want, got, sizeof(want)) != 0) {
        ++bad;
        if (bad <= 4) {
          printf("tile_f16: hd=%u col=%u differs\n", hd, col);
        }
      }
    }
    free(rows);
  }
  printf("tile_f16 transposed: %d cases, %d bad\n", cases, bad);
  if (bad != 0) {
    printf("TILE F16 FAILED\n");
    return 1;
  }
  printf("TILE F16 OK\n");
  return 0;
}
