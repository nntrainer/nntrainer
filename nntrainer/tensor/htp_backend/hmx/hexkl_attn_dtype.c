// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hexkl_attn_dtype.c
 * @date   07 Aug 2026
 * @brief  Width vtable over the u8i4 and u8i8 weight registries
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Thin casting adapters, no logic. Every function here exists only because the
 * two registries take a differently-typed table pointer.
 */

#include <stddef.h>

#include "hexkl_attn_dtype.h"
#include "hexkl_mm_u8i4_dma.h"
#include "hexkl_mm_u8i8_dma.h"

static int reg_u8i4(void *tbl, uint8_t *vtcm_base, uint32_t vtcm_size,
                    uint32_t K, uint32_t N, const int8_t *w_rm,
                    const float *w_scale, const int32_t *colsum_w,
                    const float *bias, uint32_t *out_handle) {
  return hexkl_weight_u8i4_register((hexkl_weight_u8i4_table *)tbl, vtcm_base,
                                    vtcm_size, K, N, w_rm, w_scale, colsum_w,
                                    bias, out_handle);
}

static int rel_u8i4(void *tbl, uint32_t handle) {
  return hexkl_weight_u8i4_release((hexkl_weight_u8i4_table *)tbl, handle);
}

static int run_u8i4(void *tbl, uint8_t *vtcm_base, uint32_t vtcm_size,
                    uint32_t config_off, uint32_t M, uint32_t K,
                    const uint32_t *handles, uint32_t n_handles,
                    const float *act_f32, float *out_cat,
                    const hexkl_mm_opts *opts) {
  return hexkl_mm_u8i4_layer_run((hexkl_weight_u8i4_table *)tbl, vtcm_base,
                                 vtcm_size, config_off, M, K, handles,
                                 n_handles, act_f32, out_cat, opts);
}

static int reg_u8i8(void *tbl, uint8_t *vtcm_base, uint32_t vtcm_size,
                    uint32_t K, uint32_t N, const int8_t *w_rm,
                    const float *w_scale, const int32_t *colsum_w,
                    const float *bias, uint32_t *out_handle) {
  return hexkl_weight_u8i8_register((hexkl_weight_u8i8_table *)tbl, vtcm_base,
                                    vtcm_size, K, N, w_rm, w_scale, colsum_w,
                                    bias, out_handle);
}

static int rel_u8i8(void *tbl, uint32_t handle) {
  return hexkl_weight_u8i8_release((hexkl_weight_u8i8_table *)tbl, handle);
}

static int run_u8i8(void *tbl, uint8_t *vtcm_base, uint32_t vtcm_size,
                    uint32_t config_off, uint32_t M, uint32_t K,
                    const uint32_t *handles, uint32_t n_handles,
                    const float *act_f32, float *out_cat,
                    const hexkl_mm_opts *opts) {
  return hexkl_mm_u8i8_layer_run((hexkl_weight_u8i8_table *)tbl, vtcm_base,
                                 vtcm_size, config_off, M, K, handles,
                                 n_handles, act_f32, out_cat, opts);
}

int hexkl_attn_ops_init(hexkl_attn_ops *out, hexkl_w_width w, void *table) {
  if (out == NULL || table == NULL) {
    return -1;
  }

  out->width = w;
  out->table = table;
  if (w == HEXKL_W_I4) {
    out->wh_tile_bytes = 512u;
    out->reg = reg_u8i4;
    out->rel = rel_u8i4;
    out->run = run_u8i4;
  } else {
    out->wh_tile_bytes = 1024u;
    out->reg = reg_u8i8;
    out->rel = rel_u8i8;
    out->run = run_u8i8;
  }
  return 0;
}
