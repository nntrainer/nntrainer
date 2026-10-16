// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file   hexkl_kv_tiles_f16.h
 * @date   14 Sep 2026
 * @brief  DSP-resident fp16 KV cache kept as ready HMX tiles ("NativeKV")
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The attention kernel spends part of every block on DMA-ing raw K/V rows
 * (256 B rows, the slow DMA regime) and turning them into K^T and V weight
 * tiles on HVX. Both costs are paid again on every step for the same
 * cache rows. Storing the cache as tiles once, at append time, turns each
 * block's staging into one DMA of contiguous 2 KiB rows and no tile work
 * at all -- the tutorial's "NativeKV" and what the prior branch measured
 * as the long-context win. This module owns that storage.
 *
 * Layout per registered cache, for kv head n, column tile c (cache rows
 * 32c..32c+31) and dot tile d (dims 32d..32d+31), tile index
 * (n*n_col_tiles + c)*n_dot_tiles + d, 2048 bytes each, in two arrays:
 *   kt: the K^T weight tile (K^T rows = dims, columns = cache rows)
 *   v : the V weight tile   (rows = cache rows, columns = dims)
 * so a cache block of bc rows for one head is n_col_tiles*n_dot_tiles
 * consecutive tiles in each. Tiles are zero until written, so rows past
 * cache_to are zero, never garbage.
 */

#ifndef __NNTRAINER_HEXKL_KV_TILES_F16_H__
#define __NNTRAINER_HEXKL_KV_TILES_F16_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Resident caches per session (one per attention layer). */
#define HEXKL_KV_TILES_MAX 64u

typedef struct {
  uint8_t *kt;       /**< K^T tiles, n_head_kv*n_col_tiles*n_dot_tiles */
  uint8_t *v;        /**< V tiles, same count */
  uint32_t max_rows; /**< capacity in cache rows, multiple of 32 */
  uint32_t n_head_kv;
  uint32_t head_dim;
  uint32_t n_col_tiles; /**< max_rows / 32 */
  uint32_t n_dot_tiles; /**< head_dim / 32 */
  int in_use;
} hexkl_kv_tiles_f16;

typedef struct {
  hexkl_kv_tiles_f16 slots[HEXKL_KV_TILES_MAX];
} hexkl_kv_tiles_f16_table;

/**
 * @brief Allocates zeroed tile storage for up to @a max_rows cache rows.
 *
 * @param max_rows  rounded up to a multiple of 32
 * @return AEE_SUCCESS with *out_handle set, AEE_EBADPARM, AEE_ENOMEMORY
 */
int hexkl_kv_tiles_f16_register(hexkl_kv_tiles_f16_table *tbl,
                                uint32_t max_rows, uint32_t n_head_kv,
                                uint32_t head_dim, uint32_t *out_handle);

int hexkl_kv_tiles_f16_release(hexkl_kv_tiles_f16_table *tbl, uint32_t handle);

/**
 * @brief Writes cache rows [row0, row0 + n_rows) from row-major fp16 K and
 *        V ([n_rows][n_head_kv*head_dim], the cache's own layout) into the
 *        tiles. Rows may be rewritten (e.g. a rolled-back step).
 *
 * Scalar scatter: per row, per head, per dot tile, 16 word stores into the
 * K^T tile (one per tile vector -- two adjacent dims of one cache row are
 * one word) and 32 halfword stores into the V tile. Cheap next to the
 * projection that produced the row, and it runs once per token rather
 * than once per attention step.
 */
int hexkl_kv_tiles_f16_append(hexkl_kv_tiles_f16_table *tbl, uint32_t handle,
                              uint32_t row0, uint32_t n_rows,
                              const uint16_t *k_rows, const uint16_t *v_rows);

/** @brief The slot, or NULL if the handle is not in use. */
const hexkl_kv_tiles_f16 *
hexkl_kv_tiles_f16_get(const hexkl_kv_tiles_f16_table *tbl, uint32_t handle);

/** @brief Byte offset of tile (n, c, d) in either array. */
static inline uint32_t hexkl_kv_tiles_f16_off(const hexkl_kv_tiles_f16 *kv,
                                              uint32_t n, uint32_t c,
                                              uint32_t d) {
  return ((n * kv->n_col_tiles + c) * kv->n_dot_tiles + d) * 2048u;
}

#ifdef __cplusplus
}
#endif

#endif /* __NNTRAINER_HEXKL_KV_TILES_F16_H__ */
