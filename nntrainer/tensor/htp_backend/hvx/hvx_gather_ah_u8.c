// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hvx_gather_ah_u8.c
 * @date   10 Sep 2026
 * @brief  Copy one 64-row AH activation block into VTCM, on the pool
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
  size_t bytes;
} copy_ctx;

static void copy_worker(uint32_t n_threads, uint32_t i, void *vctx) {
  copy_ctx *c = (copy_ctx *)vctx;
  const size_t lo = c->bytes * i / n_threads;
  const size_t hi = c->bytes * (i + 1) / n_threads;
  if (hi > lo) {
    memcpy(c->dst + lo, c->src + lo, hi - lo);
  }
}

void hvx_copy_ah_block(uint8_t *dst_ah, const uint8_t *src_ah, uint32_t k,
                       hvx_worker_pool *pool) {
  const uint32_t kt_n = k / AH_INNER;
  if (!dst_ah || !src_ah || kt_n == 0u) {
    return;
  }
  /* The whole block is one contiguous run now that the pack writes slot
     order directly, so this is a flat copy split by bytes -- nothing to
     gather. The pass it replaces read 32 bytes at a time from addresses
     scattered across the entire activation and cost 3.2 ms a layer; doc 46
     section 26.3 has why tidying its destination did not help. */
  copy_ctx c = {dst_ah, src_ah, (size_t)kt_n * AH_TILE_BYTES};
  hvx_worker_pool_run(pool, copy_worker, &c, kt_n);
}
