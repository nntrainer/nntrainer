// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file   hexkl_kv_tiles_f16.c
 * @date   14 Sep 2026
 * @brief  DSP-resident fp16 KV cache kept as ready HMX tiles ("NativeKV")
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "hexkl_kv_tiles_f16.h"

#include <stdlib.h>
#include <string.h>

#ifdef __hexagon__
#include <AEEStdErr.h>
#else
// Host build (the unit test): the three codes this file returns, with the
// SDK's values (AEEStdErr.h: AEE_EOFFSET + 0x002 / 0x00E).
#define AEE_SUCCESS 0
#define AEE_ENOMEMORY 0x80000402
#define AEE_EBADPARM 0x8000040E
#endif

#define TILE_BYTES 2048u

int hexkl_kv_tiles_f16_register(hexkl_kv_tiles_f16_table *tbl,
                                uint32_t max_rows, uint32_t n_head_kv,
                                uint32_t head_dim, uint32_t *out_handle) {
  if (!tbl || !out_handle || max_rows == 0 || n_head_kv == 0 || head_dim == 0 ||
      (head_dim % 32u) != 0 || head_dim > 256u) {
    return AEE_EBADPARM;
  }
  uint32_t slot = HEXKL_KV_TILES_MAX;
  for (uint32_t i = 0; i < HEXKL_KV_TILES_MAX; ++i) {
    if (!tbl->slots[i].in_use) {
      slot = i;
      break;
    }
  }
  if (slot == HEXKL_KV_TILES_MAX) {
    return AEE_ENOMEMORY;
  }

  hexkl_kv_tiles_f16 *kv = &tbl->slots[slot];
  memset(kv, 0, sizeof(*kv));
  kv->max_rows = ((max_rows + 31u) / 32u) * 32u;
  kv->n_head_kv = n_head_kv;
  kv->head_dim = head_dim;
  kv->n_col_tiles = kv->max_rows / 32u;
  kv->n_dot_tiles = head_dim / 32u;
  const size_t n_tiles = (size_t)n_head_kv * kv->n_col_tiles * kv->n_dot_tiles;
  // calloc: untouched tiles read as zero, which the kernel relies on for
  // rows past cache_to (a masked probability is exactly 0, but only if
  // the value it multiplies is finite).
  kv->kt = (uint8_t *)calloc(n_tiles, TILE_BYTES);
  kv->v = (uint8_t *)calloc(n_tiles, TILE_BYTES);
  if (!kv->kt || !kv->v) {
    free(kv->kt);
    free(kv->v);
    memset(kv, 0, sizeof(*kv));
    return AEE_ENOMEMORY;
  }
  kv->in_use = 1;
  *out_handle = slot;
  return AEE_SUCCESS;
}

int hexkl_kv_tiles_f16_release(hexkl_kv_tiles_f16_table *tbl, uint32_t handle) {
  if (!tbl || handle >= HEXKL_KV_TILES_MAX || !tbl->slots[handle].in_use) {
    return AEE_EBADPARM;
  }
  hexkl_kv_tiles_f16 *kv = &tbl->slots[handle];
  free(kv->kt);
  free(kv->v);
  memset(kv, 0, sizeof(*kv));
  return AEE_SUCCESS;
}

const hexkl_kv_tiles_f16 *
hexkl_kv_tiles_f16_get(const hexkl_kv_tiles_f16_table *tbl, uint32_t handle) {
  if (!tbl || handle >= HEXKL_KV_TILES_MAX || !tbl->slots[handle].in_use) {
    return NULL;
  }
  return &tbl->slots[handle];
}

int hexkl_kv_tiles_f16_append(hexkl_kv_tiles_f16_table *tbl, uint32_t handle,
                              uint32_t row0, uint32_t n_rows,
                              const uint16_t *k_rows, const uint16_t *v_rows) {
  hexkl_kv_tiles_f16 *kv =
    (hexkl_kv_tiles_f16 *)hexkl_kv_tiles_f16_get(tbl, handle);
  if (!kv || !k_rows || !v_rows || n_rows == 0 ||
      row0 + n_rows > kv->max_rows || row0 + n_rows < row0) {
    return AEE_EBADPARM;
  }
  const uint32_t hd = kv->head_dim;
  const uint32_t stride = kv->n_head_kv * hd; // elements per cache row

  for (uint32_t r = 0; r < n_rows; ++r) {
    const uint32_t row = row0 + r;
    const uint32_t c = row / 32u;
    const uint32_t rr = row % 32u;
    const uint16_t *krow = k_rows + (size_t)r * stride;
    const uint16_t *vrow = v_rows + (size_t)r * stride;
    for (uint32_t n = 0; n < kv->n_head_kv; ++n) {
      for (uint32_t d = 0; d < kv->n_dot_tiles; ++d) {
        const uint32_t off = hexkl_kv_tiles_f16_off(kv, n, c, d);
        // K^T tile: vector i holds K^T rows 2i, 2i+1 (dims) interleaved
        // over the 32 cache rows, so cache row rr contributes word rr of
        // every vector, and that word is K[row][2i], K[row][2i+1] -- the
        // same word in the source row (hvx_tile_f16.h, transposed builder).
        uint32_t *kt = (uint32_t *)(kv->kt + off);
        const uint16_t *ksrc = krow + n * hd + 32u * d;
        for (uint32_t i = 0; i < 16u; ++i) {
          uint32_t w;
          memcpy(&w, ksrc + 2u * i, sizeof(w));
          kt[i * 32u + rr] = w;
        }
        // V tile: vector rr/2 holds cache rows 2(rr/2), 2(rr/2)+1
        // interleaved by dim; row rr is the (rr&1) lane of each pair.
        uint16_t *vt = (uint16_t *)(kv->v + off) + (rr >> 1) * 64u + (rr & 1u);
        const uint16_t *vsrc = vrow + n * hd + 32u * d;
        for (uint32_t k = 0; k < 32u; ++k) {
          vt[2u * k] = vsrc[k];
        }
      }
    }
  }
  return AEE_SUCCESS;
}
