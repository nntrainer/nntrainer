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

/**
 * @brief Copies one inner tile's worth of every selected row.
 *
 * Split by inner tile rather than by row, which is the opposite of the
 * obvious reading of "gather rows". Two reasons, both measured
 * (doc 46 section 23.1: 3.6 MB moved in 3271 us, about 1.1 GB/s, for what
 * is a pure copy):
 *
 *  - Per tile the destination is ONE contiguous 2048-byte run, written
 *    front to back. Splitting by row instead makes every worker write 32
 *    bytes every 2048, so the block's 128 KB leaves as 4096 scattered
 *    stores rather than 64 sequential tiles.
 *  - There are always kt_n units (64 for K=2048), where splitting by row
 *    gives n_rows units -- and the tail block of an expert can have as few
 *    as one row, which left three of the four workers with nothing.
 *
 * The source side is scattered either way: the rows a block wants are
 * wherever the router put them.
 */
static void gather_worker(uint32_t n_threads, uint32_t i, void *vctx) {
  gather_ctx *c = (gather_ctx *)vctx;
  const uint32_t lo = (uint32_t)((uint64_t)c->kt_n * i / n_threads);
  const uint32_t hi = (uint32_t)((uint64_t)c->kt_n * (i + 1) / n_threads);

  for (uint32_t kt = lo; kt < hi; ++kt) {
    uint8_t *d = c->dst + (size_t)kt * AH_TILE_BYTES;
    for (uint32_t r = 0; r < c->n_rows; ++r) {
      const uint32_t t = c->rows[r];
      const uint8_t *s =
        c->src + (size_t)(t / AH_ROWS) * c->kt_n * AH_TILE_BYTES +
        (size_t)kt * AH_TILE_BYTES + (size_t)(t % AH_ROWS) * AH_INNER;
      memcpy(d + (size_t)r * AH_INNER, s, AH_INNER);
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
  if (n_rows == 0u) {
    memset(dst_ah, 0, (size_t)kt_n * AH_TILE_BYTES);
    return;
  }
  /* Only the rows past n_rows need clearing -- the ones below are about to
     be overwritten in full. Zeroing all 128 KB first, on the calling thread,
     cost more than the copy it was protecting. */
  if (n_rows < AH_ROWS) {
    for (uint32_t kt = 0; kt < kt_n; ++kt) {
      memset(dst_ah + (size_t)kt * AH_TILE_BYTES + (size_t)n_rows * AH_INNER, 0,
             (size_t)(AH_ROWS - n_rows) * AH_INNER);
    }
  }
  gather_ctx c = {dst_ah, src_ah, rows, n_rows, kt_n};
  /* kt_n units, matching how gather_worker slices. Passing n_rows here
     would leave a one-row tail block running on a single worker. */
  hvx_worker_pool_run(pool, gather_worker, &c, kt_n);
}
