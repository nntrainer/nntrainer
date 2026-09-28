// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hexkl_attn_dtype.h
 * @date   07 Aug 2026
 * @brief  Width vtable over the u8i4 and u8i8 weight registries
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * hexkl_mm_u8i4_dma.h and hexkl_mm_u8i8_dma.h expose the same three functions
 * with the same parameters; the only difference is the table type (and the WH
 * tile size that follows from the width). This turns that into one indirection
 * so the attention module is written once instead of mirrored.
 *
 * That REVERSES the decision hexkl_mm_u8i8_dma.c was written under, on
 * purpose: mirroring there protected an already-device-verified u8i4 path from
 * being re-verified. Here neither width is verified yet, so there is nothing to
 * protect, and duplicating the orchestration would mean fixing every later bug
 * twice (MHA_HTP_U8_TASKS.md §0.1).
 */

#ifndef __NNTRAINER_HEXKL_ATTN_DTYPE_H__
#define __NNTRAINER_HEXKL_ATTN_DTYPE_H__

#include <stdint.h>

#include "hexkl_kv_quant.h" /* hexkl_w_width */
#include "hexkl_mm_opts.h"

/**
 * @brief The three registry entry points at one width, plus the constants
 *        that follow from it.
 *
 * RESOLVE THIS ONCE PER layer_run CALL, NEVER PER TILE. An indirect branch in
 * the 32x32 tile loop would be a real regression that no correctness test can
 * see, and grepping the inner loop for `ops->` is the review check for it
 * (MHA_HTP_U8_TASKS.md §4).
 */
typedef struct {
  hexkl_w_width width;
  uint32_t wh_tile_bytes; /**< 512 (i4) or 1024 (i8) */
  void *table;            /**< hexkl_weight_u8i4_table* or _u8i8_table* */

  int (*reg)(void *tbl, uint8_t *vtcm_base, uint32_t vtcm_size, uint32_t K,
             uint32_t N, const int8_t *w_rm, const float *w_scale,
             const int32_t *colsum_w, const float *bias, uint32_t *out_handle);
  int (*rel)(void *tbl, uint32_t handle);
  int (*run)(void *tbl, uint8_t *vtcm_base, uint32_t vtcm_size,
             uint32_t config_off, uint32_t M, uint32_t K,
             const uint32_t *handles, uint32_t n_handles, const float *act_f32,
             float *out_cat, const hexkl_mm_opts *opts);
} hexkl_attn_ops;

/**
 * @brief Points @a out at the entry points for width @a w, over @a table.
 *
 * The caller supplies the table rather than this allocating one -- the session
 * already owns both registries (nntr_hvx_session), so allocating here would
 * make a second, competing registry. That is why there is no matching fini:
 * nothing is owned.
 *
 * @param table  hexkl_weight_u8i4_table* when @a w is HEXKL_W_I4, otherwise
 *               hexkl_weight_u8i8_table*. Not checked; passing the wrong one
 *               is a type error the compiler cannot see through void*.
 * @return 0 on success, -1 if @a out or @a table is NULL.
 */
int hexkl_attn_ops_init(hexkl_attn_ops *out, hexkl_w_width w, void *table);

#endif /* __NNTRAINER_HEXKL_ATTN_DTYPE_H__ */
