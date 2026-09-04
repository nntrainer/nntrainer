// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file hexkl_mm_u8i8_dma.h
 * @date 06 Aug 2026
 * @brief Persistent u8i8 weight registry and the cross-matmul DMA layer path
 * @see https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug No known bugs except for NYI items
 *
 * Mirrors hexkl_mm_u8i4_dma.h exactly, at the wider weight width: u8i8
 * fixes the activation at uint8 (same as u8i4 HMX's activation port is
 * always 8-bit) and widens only the weight slot, so the tile byte count
 * doubles (1024 vs 512) and the bake/matmul primitives are HexKL's _i8
 * pair instead of _i4, but the VTCM layout, registry, and cross-matmul
 * prefetch logic are otherwise identical see hexkl_mm_u8i4_dma.c's
 * comments for the reasoning behind each of them. */

#ifndef __NNTRAINER_HEXKL_MM_U8I8_DMA_H__
#define __NNTRAINER_HEXKL_MM_U8I8_DMA_H__

#include <stdint.h>

#include "hexkl_mm_opts.h"

/** @brief Upper bound on weights resident at once see the u8i4 header
 * for the sizing rationale; identical here. */
#define HEXKL_MM_U8I8_MAX_WEIGHTS 512

typedef struct {
  int in_use;
  uint8_t *wh_bytes; /**< k_tiles*n_tiles*1024 bytes, WH layout */
  float *w_scale;    /**< N entries */
  int32_t *colsum_w; /**< N entries */
  float *bias;       /**< N entries */
  uint32_t K, N;
} hexkl_weight_u8i8;

typedef struct {
  hexkl_weight_u8i8 slots[HEXKL_MM_U8I8_MAX_WEIGHTS];
} hexkl_weight_u8i8_table;

/**
 * @brief Bakes a K x N int8 weight (row-major) to WH layout and keeps the
 * result resident until hexkl_weight_u8i8_release. * hexkl_weight_u8i4_register
 * same contract, wider weight. */
int hexkl_weight_u8i8_register(hexkl_weight_u8i8_table *tbl, uint8_t *vtcm_base,
                               uint32_t vtcm_size, uint32_t K, uint32_t N,
                               const int8_t *w_i8_rm, const float *w_scale,
                               const int32_t *colsum_w, const float *bias,
                               uint32_t *out_handle);

/** @brief Frees a registered weight's resident bytes. */
int hexkl_weight_u8i8_release(hexkl_weight_u8i8_table *tbl, uint32_t handle);

/**
 * @brief Runs matmuls against several registered u8i8 weights that share
 * one quantized activation. hexkl_mm_u8i4_layer_run same
 * contract, wider weight; every handle must share K, and (as with
 * u8i4) all in one call. */
int hexkl_mm_u8i8_layer_run(hexkl_weight_u8i8_table *tbl, uint8_t *vtcm_base,
                            uint32_t vtcm_size, uint32_t config_off, uint32_t M,
                            uint32_t K, const uint32_t *handles,
                            uint32_t n_handles, const float *act_f32,
                            float *out_cat, const hexkl_mm_opts *opts);

#endif /* __NNTRAINER_HEXKL_MM_U8I8_DMA_H__ */
