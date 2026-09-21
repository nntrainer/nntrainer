// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hexkl_conv_block.h
 * @date   22 Sep 2026
 * @brief  An LFM2 conv block -- in_proj, gate, conv1d, gate, out_proj -- in
 *         one call
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Design: docs/htp_attention/51_block_calls_on_htp.md section 2. The
 * block is
 *
 *   p = x . W_in            [M x 3C], split into a | b | c
 *   y = conv1d(a * c)       depthwise, causal, L = 3
 *   o = (b * y) . W_out     [M x N_out]
 *
 * and the point of doing it in one call is that a, b, c and y -- four
 * M x C f32 intermediates, 14.5 MB at M = 444 -- never cross FastRPC.
 * Doc 50 measured what crossing costs: in_proj alone on the HTP won 13 ms
 * a prefill of the 100 it computes, the rest eaten by the round trip.
 *
 * Weights resident, activation streamed (doc 51 section 2.2's second
 * option): W_in's slices are 2 MiB each and W_out 2 MiB, so two fit in
 * VTCM at a time and the M rows are walked twice, in 64-row blocks:
 *
 *   phase 1  W_a, W_c resident:  block -> HMX a and c tiles in pairs ->
 *            dq(a) * dq(c) straight into g [M x C] on the DSP heap
 *   phase 2  W_b, W_out resident: block -> HMX b -> dq(b) -> * conv1d(g)
 *            (two rows of g before the block are its carry) -> requant ->
 *            HMX out_proj -> dq into out
 *
 * Weight DMA is 8 MiB a layer, once; the activation round trip through
 * the heap is g's 3.6 MB written and read. The only quantization points
 * are the two an FC-by-FC offload has (x and b*y to u8): a * c is formed
 * in f32 from the dequantized tiles and stays f32 in g.
 *
 * Built on hexkl_mm_u8i4_moe.c's parts (its scratch, DMA pushes, pack
 * and the HMX-issue timing) so the per-block cost is that kernel's,
 * which doc 51 section 1.4 measured at 340 us a block.
 */

#ifndef __NNTRAINER_HEXKL_CONV_BLOCK_H__
#define __NNTRAINER_HEXKL_CONV_BLOCK_H__

#include <stddef.h>
#include <stdint.h>

#include "hexkl_mm_u8i4_dma.h"
#include "hexkl_mm_u8i4_moe.h"
#include "hvx_worker_pool.h"

/**
 * @brief VTCM regions, in bytes from vtcm_base. Computed -- and rejected
 *        -- without touching hardware, like hexkl_moe_layout.
 */
typedef struct {
  uint32_t act_off;    /**< one 64-row activation block, AH tiles */
  uint32_t w_a_off;    /**< slot A: W_a (phase 1), then W_b (phase 2) */
  uint32_t w_b_off;    /**< slot B: W_c (phase 1), then W_out (phase 2) */
  uint32_t z_off;      /**< TWO [64 x C] f32: dq(b), gated in place; phase
                            2 keeps two blocks in flight */
  uint32_t mid_off;    /**< TWO blocks of z requantized, AH tiles */
  uint32_t conv_w_off; /**< [3 x C] f32, copied in from the call's buffer */
  uint32_t result_off; /**< two staging buffers of acc_tiles tiles each */
  uint32_t acc_tiles;  /**< tiles per staging buffer; even, >= 2 */
  uint32_t total;
} hexkl_conv_block_layout;

/**
 * @brief Computes the VTCM layout and reports whether it fits.
 *
 * @param K, C, N_out  x's width, the conv width (a, b, c's), out's width
 * @param arena_bytes  usable VTCM (min of vtcm_size and config_off)
 * @return AEE_SUCCESS, AEE_EBADPARM for shapes that do not tile-divide, or
 *         AEE_ENOMEMORY when the layout does not fit
 */
int hexkl_conv_block_layout_get(uint32_t K, uint32_t C, uint32_t N_out,
                                uint32_t arena_bytes,
                                hexkl_conv_block_layout *out);

/**
 * @brief One conv block over M rows.
 *
 * @param h_a, h_b, h_c  in_proj's column slices [0,C), [C,2C), [2C,3C) as
 *                       three registered weights, each K x C
 * @param h_out          out_proj, C x N_out
 * @param conv_w         [3 x C] f32: w0 (row t), w1 (t-1), w2 (t-2)
 * @param act_f32        [M x K], the normed residual
 * @param[out] out_f32   [M x N_out]
 * @param[out] state_f32 [2 x C]: the conv input's rows M-2 and M-1 (zeros
 *                       where M is shorter), the state the CPU decode
 *                       path continues from
 * @param scratch        the session's MoE scratch; grown here as needed
 * @return AEE_SUCCESS, or the first failing stage's code
 */
int hexkl_conv_block_run(hexkl_weight_u8i4_table *tbl, uint8_t *vtcm_base,
                         uint32_t vtcm_size, uint32_t config_off, uint32_t M,
                         uint32_t K, uint32_t C, uint32_t N_out, uint32_t h_a,
                         uint32_t h_b, uint32_t h_c, uint32_t h_out,
                         const float *conv_w, const float *act_f32,
                         float *out_f32, float *state_f32,
                         hvx_worker_pool *pool, hexkl_moe_scratch *scratch);

#endif /* __NNTRAINER_HEXKL_CONV_BLOCK_H__ */
