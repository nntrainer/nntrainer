// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file hexkl_acc_tile.c
 * @date 08 Aug 2026
 * @brief Derives the int32 accumulator tile's VTCM layout at runtime
 * @see https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug No known bugs except for NYI items */

#include "hexkl_acc_tile.h"

#include <AEEStdErr.h>

#include "hexkl_micro.h"

/** @brief Elements in one tile. */
#define TILE_N (HEXKL_ACC_TILE_ROWS * HEXKL_ACC_TILE_COLS)

static hexkl_acc_layout g_layout;
/** @brief Probe destination. Static rather than a 8 KB stack frame, and it
 * outlives nothing the probe runs once and only reads it back. */
static int32_t g_probe_dst[TILE_N];

const hexkl_acc_layout *hexkl_acc_layout_get(uint8_t *vtcm_base,
                                             uint32_t result_off) {
  int32_t *tile;
  int32_t base, stride;
  uint32_t r, c, i;

  if (g_layout.probed) {
    return &g_layout;
  }
  g_layout.probed = 1;
  g_layout.usable = 0;

  /** The ramp is the trick: after the copy, g_probe_dst[r][c] holds the index
   * the vendor pulled that element from. */
  tile = (int32_t *)(vtcm_base + result_off);
  for (i = 0; i < TILE_N; ++i) {
    tile[i] = (int32_t)i;
  }

  if (hexkl_micro_hmx_copy_32b_to_submatrix(
        vtcm_base, result_off, g_probe_dst, 0u, 0u, HEXKL_ACC_TILE_ROWS,
        HEXKL_ACC_TILE_COLS) != AEE_SUCCESS) {
    return &g_layout;
  }

  base = g_probe_dst[0];
  stride = g_probe_dst[HEXKL_ACC_TILE_COLS] - base; /* destination row 1 */
  if (base < 0 || stride <= 0) {
    return &g_layout;
  }

  /** Check EVERY element, not the two the parameters came from: a layout that
   * is affine over the first row and blocked after it would pass a sampled
   * check and produce wrong numbers everywhere else. */
  for (r = 0; r < HEXKL_ACC_TILE_ROWS; ++r) {
    for (c = 0; c < HEXKL_ACC_TILE_COLS; ++c) {
      const int32_t want = base + (int32_t)r * stride + (int32_t)c;
      if (want < 0 || (uint32_t)want >= TILE_N ||
          g_probe_dst[r * HEXKL_ACC_TILE_COLS + c] != want) {
        return &g_layout;
      }
    }
  }

  g_layout.base = (uint32_t)base;
  g_layout.row_stride = (uint32_t)stride;
  g_layout.usable = 1;
  return &g_layout;
}
