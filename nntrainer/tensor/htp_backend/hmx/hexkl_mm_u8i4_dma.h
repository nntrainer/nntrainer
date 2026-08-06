// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hexkl_mm_u8i4_dma.h
 * @date   06 Aug 2026
 * @brief  Persistent u8i4 weight registry and the cross-matmul DMA layer path
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Turns hexkl_mm_u8i4's accuracy-harness building blocks (plan / bake /
 * run, one matmul at a time, weight baked fresh every call) into the
 * performance path doc15 §4/§8 item 3 describes: a weight is baked once
 * and kept DSP-resident until released, and a "layer" of matmuls that share
 * one activation (Q/K/V, or gate/up) runs back to back with the next
 * weight prefetched into VTCM while the current one computes -- the
 * cross-matmul win measured in doc13 §3a (1.7-2x over SDKL on V79).
 */

#ifndef __NNTRAINER_HEXKL_MM_U8I4_DMA_H__
#define __NNTRAINER_HEXKL_MM_U8I4_DMA_H__

#include <stdint.h>

/** @brief Upper bound on weights resident at once. A 28-layer qwen3-sized
 *         model registers 7 per layer (q,k,v,o,gate,up,down) = 196; this
 *         leaves headroom without making the table itself large. */
#define HEXKL_MM_U8I4_MAX_WEIGHTS 512

/**
 * @brief One registered weight: WH-baked bytes plus its dequant constants,
 *        resident in DSP heap memory (not VTCM -- VTCM is compute
 *        scratch) until hexkl_weight_u8i4_release.
 */
typedef struct {
  int in_use;
  uint8_t *wh_bytes; /**< k_tiles*n_tiles*512 bytes, WH layout */
  float *w_scale;    /**< N entries */
  int32_t *colsum_w; /**< N entries */
  float *bias;       /**< N entries */
  uint32_t K, N;
} hexkl_weight_u8i4;

typedef struct {
  hexkl_weight_u8i4 slots[HEXKL_MM_U8I4_MAX_WEIGHTS];
} hexkl_weight_u8i4_table;

/**
 * @brief Bakes a K x N int4 weight (int4 values in int8 containers,
 *        row-major) to WH layout and keeps the result resident until
 *        hexkl_weight_u8i4_release.
 *
 * Borrows the caller's VTCM arena as scratch for the bake -- the caller
 * must hold the HMX lock, and no other VTCM use may be in flight for the
 * duration of this call.
 *
 * @param[out] out_handle  index into @a tbl; pass back to
 *                         hexkl_mm_u8i4_layer_run / hexkl_weight_u8i4_release
 * @return AEE_SUCCESS, AEE_EBADPARM on a shape violation, AEE_ENOMEMORY if
 *         the table is full or a host allocation fails, or a HexKL error
 *         code from the bake itself.
 */
int hexkl_weight_u8i4_register(hexkl_weight_u8i4_table *tbl, uint8_t *vtcm_base,
                               uint32_t vtcm_size, uint32_t K, uint32_t N,
                               const int8_t *w_i4_rm, const float *w_scale,
                               const int32_t *colsum_w, const float *bias,
                               uint32_t *out_handle);

/** @brief Frees a registered weight's resident bytes. */
int hexkl_weight_u8i4_release(hexkl_weight_u8i4_table *tbl, uint32_t handle);

/**
 * @brief Runs matmuls against several registered weights that share one
 *        quantized activation, double-buffering each weight into VTCM and
 *        prefetching the next handle's weight while the current one
 *        computes.
 *
 * Every handle must have been registered with K == @a K; this is checked,
 * not assumed. Output is dequantized f32, one contiguous M x handle[i].N
 * block per handle in call order (not interleaved row-by-row).
 *
 * @param config_off  session-constant HMX config region offset (from
 *                     hexkl_mm_u8i4_plan; the same for every M/K/N because
 *                     it depends only on vtcm_size). The caller must have
 *                     called hexkl_micro_hmx_setup_acc_read_int32 for it
 *                     once already -- this function does not repeat that
 *                     because it is session-scoped, not call-scoped.
 * @param[out] out_cat  sum(handles[i].N) * M f32 entries
 */
int hexkl_mm_u8i4_layer_run(hexkl_weight_u8i4_table *tbl, uint8_t *vtcm_base,
                            uint32_t vtcm_size, uint32_t config_off, uint32_t M,
                            uint32_t K, const uint32_t *handles,
                            uint32_t n_handles, const float *act_f32,
                            float *out_cat);

#endif /* __NNTRAINER_HEXKL_MM_U8I4_DMA_H__ */
