// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hexkl_mm_u8i4_moe.h
 * @date   10 Sep 2026
 * @brief  A whole MoE FFN layer -- every expert -- in one call
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Replaces the 64 FastRPC calls a MoE layer costs today (32 experts x
 * gate_up_swiglu + layer_u8in) with one. Design and the VTCM arithmetic:
 * docs/htp_attention/46_moe_resident_kernel_design.md.
 *
 * What moves onto the DSP with it: the token gather, the routing-weight
 * multiply and the scatter-add, all of which the ARM side does today at a
 * measured 6.3 ms/layer (doc 44 section 15.1).
 *
 * The activation and output are plain pointers and this file does not care
 * where they live. Today the skel wrapper passes buffers FastRPC moved;
 * when the residual stream stays on the DSP (doc 45 Phase D) another
 * wrapper passes DSP-side addresses and the kernel is unchanged.
 */

#ifndef __NNTRAINER_HEXKL_MM_U8I4_MOE_H__
#define __NNTRAINER_HEXKL_MM_U8I4_MOE_H__

#include <stdint.h>

#include "hexkl_mm_u8i4_dma.h"
#include "hvx_worker_pool.h"

/**
 * @brief VTCM regions, in bytes from vtcm_base.
 *
 * Broken out of the run function so the layout can be computed -- and
 * rejected -- without touching hardware, which is what makes it testable on
 * a host and what doc 46 section 3's table is checked against. L2's first
 * attempt sized VTCM for the whole M and died with AEE_ENOMEMORY on expert
 * 6 of a real run; this is that check, moved to before the work starts.
 */
typedef struct {
  uint32_t act_off;     /**< one 64-row activation block, AH tiles */
  uint32_t w_gu_off;    /**< buffer A: gate_up's WH bytes */
  uint32_t w_dn_off;    /**< buffer B: down's WH bytes */
  uint32_t gate_off;    /**< [64 x inter] f32, then SwiGLU's output in place */
  uint32_t up_off;      /**< [64 x inter] f32 */
  uint32_t mid_off;     /**< requantized SwiGLU output, AH tiles */
  uint32_t result_off;  /**< one HMX accumulator tile */
  uint32_t res_f32_off; /**< [64 x N_out] f32, down's dequantized block.
                             ALIASES gate_off -- see the .c on why that is
                             safe and what would break it. */
  uint32_t total;       /**< bytes needed; compared against the arena */
} hexkl_moe_layout;

/**
 * @brief Computes the VTCM layout and reports whether it fits.
 *
 * @param[in]  K, inter, N_out  the layer's shapes
 * @param[in]  arena_bytes      usable VTCM (min of vtcm_size and config_off)
 * @param[out] out              filled on success; untouched on failure
 * @return AEE_SUCCESS, AEE_EBADPARM for shapes that do not tile-divide, or
 *         AEE_ENOMEMORY when the layout does not fit
 */
int hexkl_mm_u8i4_moe_layout(uint32_t K, uint32_t inter, uint32_t N_out,
                             uint32_t arena_bytes, hexkl_moe_layout *out);

/**
 * @brief One MoE FFN layer: routing, every expert, and the scatter-add.
 *
 * out_f32 is zeroed here and accumulated into, because experts share token
 * rows -- that is the same read-modify-write the ARM side does today with
 * add_i, one pass instead of one Tensor object per token per expert.
 *
 * @param[in] row_index   [n_rows] token row indices, grouped by expert in
 *                        expert order; expert e owns the row_count[e]
 *                        entries after the previous experts' counts
 * @param[in] row_count   [n_experts]; zero is allowed and skipped
 * @param[in] row_weight  [n_rows] routing weight for each entry
 * @param[in] act_f32     [M x K]
 * @param[out] out_f32    [M x N_out]
 * @return AEE_SUCCESS, or the first failing stage's code
 */
int hexkl_mm_u8i4_moe_layer_run(
  hexkl_weight_u8i4_table *tbl, uint8_t *vtcm_base, uint32_t vtcm_size,
  uint32_t config_off, uint32_t M, uint32_t K, uint32_t inter, uint32_t N_out,
  uint32_t n_experts, const uint32_t *h_gate_up, const uint32_t *h_down,
  const uint32_t *row_index, const uint32_t *row_count, const float *row_weight,
  const float *act_f32, float *out_f32, hvx_worker_pool *pool);

#endif /* __NNTRAINER_HEXKL_MM_U8I4_MOE_H__ */
