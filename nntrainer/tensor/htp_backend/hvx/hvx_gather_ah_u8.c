// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hvx_gather_ah_u8.c
 * @date   10 Sep 2026
 * @brief  Pick rows out of an AH-tiled uint8 activation into a fresh block
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "hvx_gather_ah_u8.h"

#include <string.h>

/** @brief AH tile stride in bytes: 64 rows x 32 inner. */
#define AH_TILE_BYTES 2048u
/** @brief Rows per AH row-block, and inner values per tile. */
#define AH_ROWS 64u
#define AH_INNER 32u

typedef struct {
  uint8_t *dst;
  const uint8_t *src;
  const uint32_t *rows;
  uint32_t n_rows;
  uint32_t kt_n;
} gather_ctx;

static void gather_worker(uint32_t n_threads, uint32_t i, void *vctx) {
  gather_ctx *c = (gather_ctx *)vctx;
  const uint32_t lo = (uint32_t)((uint64_t)c->n_rows * i / n_threads);
  const uint32_t hi = (uint32_t)((uint64_t)c->n_rows * (i + 1) / n_threads);
  for (uint32_t r = lo; r < hi; ++r) {
    const uint32_t t = c->rows[r];
    const uint8_t *s = c->src +
                       (size_t)(t / AH_ROWS) * c->kt_n * AH_TILE_BYTES +
                       (size_t)(t % AH_ROWS) * AH_INNER;
    uint8_t *d = c->dst + (size_t)r * AH_INNER;
    /* One 32-byte run per inner tile. The destination rows for four
       consecutive r are contiguous, which a wider version could exploit;
       it would have to shuffle four unrelated source rows together to do
       it, so it is not written until this shows up in a profile. */
    for (uint32_t kt = 0; kt < c->kt_n; ++kt) {
      memcpy(d + (size_t)kt * AH_TILE_BYTES, s + (size_t)kt * AH_TILE_BYTES,
             AH_INNER);
    }
  }
}

void hvx_gather_ah_u8(uint8_t *dst_ah, const uint8_t *src_ah,
                      const uint32_t *rows, uint32_t n_rows, uint32_t k,
                      hvx_worker_pool *pool) {
  const uint32_t kt_n = k / AH_INNER;
  if (!dst_ah || !src_ah || !rows || kt_n == 0u) {
    return;
  }
  memset(dst_ah, 0, (size_t)kt_n * AH_TILE_BYTES);
  if (n_rows == 0u) {
    return;
  }
  gather_ctx c = {dst_ah, src_ah, rows, n_rows, kt_n};
  hvx_worker_pool_run(pool, gather_worker, &c, n_rows);
}
