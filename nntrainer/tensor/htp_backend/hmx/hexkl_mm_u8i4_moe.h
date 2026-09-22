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

#include <stddef.h>
#include <stdint.h>

#include "hexkl_micro.h" /* the block constants the shared macros use */
#include "hexkl_mm_u8i4_dma.h"
#include "hexkl_probe.h"
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
  uint32_t gate_off;    /**< [64 x inter] f32: silu(gate)*up, written by
                             the fused gate_up epilogue */
  uint32_t mid_off;     /**< requantized SwiGLU output, AH tiles */
  uint32_t result_off;  /**< TWO staging buffers of acc_tiles accumulator
                             tiles each, back to back: the HMX issues a
                             batch into one while the pool dequantizes the
                             other */
  /** Accumulator tiles per staging buffer, given the arena. Even, at least
      2 (one gate/up pair); gate_up's n-tile count is the useful ceiling. */
  uint32_t acc_tiles;
  uint32_t res_f32_off; /**< [64 x N_out] f32, down's dequantized block.
                             Its own region: the scatter reads it while the
                             next block's epilogue already writes gate_off */
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
 * @brief The layer call's heap scratch, kept across calls.
 *
 * A call needs about 12.8 MB at prefill (the slot-ordered activation, a
 * cached copy of the activation, the output) and a few KB of tables.
 * Allocated and freed per call, that cost 3.0 ms of a 28 ms prefill call
 * on the DSP heap (HEXKL_PROBE_ALLOC, doc 46 section 50.2 P1) -- not
 * computation, page-in and heap bookkeeping. So the block lives with the
 * session and only grows: whatever prefill needed stays for decode, which
 * needs a twentieth of it. Zero-initialised is empty; free with
 * hexkl_moe_scratch_free.
 */
typedef struct {
  void *raw;     /**< what malloc returned, for free() */
  uint8_t *base; /**< raw rounded up to 128 bytes */
  size_t cap;    /**< usable bytes from base */
} hexkl_moe_scratch;

/** @brief Releases the block. Safe on an empty scratch. */
void hexkl_moe_scratch_free(hexkl_moe_scratch *s);

/* ---- Shared with hexkl_conv_block.c ------------------------------------
 *
 * The conv block kernel (doc 51 section 2) is this kernel's loop with a
 * different block body: same scratch, same DMA ring pushes, same
 * background pack, same HMX-issue timing. These are that shared part,
 * exported rather than copied so a fix to one is a fix to both. */

/** @brief Grows the scratch to @a bytes; a no-op once it is big enough,
 *         which after the first prefill call is every call. */
int hexkl_moe_scratch_reserve(hexkl_moe_scratch *s, size_t bytes);

/** @brief Hands out the next @a bytes of the scratch, 128-byte aligned. The
 *         caller summed the same sizes through the reserve first. */
void *hexkl_moe_carve(uint8_t **cur, size_t bytes);

/**
 * @brief Pushes n-tile columns [nt0, nt0+cn) of a WH weight into VTCM at
 *        @a dst_off, one 2D descriptor; the destination keeps the source's
 *        tile indexing (kt * n_col + nt).
 * @return the ring index to hand hexkl_dma_ring_wait
 */
uint32_t hexkl_moe_push_weight_chunk(uint8_t *vtcm_base, uint32_t dst_off,
                                     const hexkl_weight_u8i4 *h,
                                     uint32_t k_tiles, uint32_t n_col,
                                     uint32_t nt0, uint32_t cn);

/** @brief Queues one 64-row AH activation block (rows [slot, slot+64) of
 *         a packed activation) heap -> VTCM. Push it AHEAD of the weights
 *         it will be computed against -- see the definition for why.
 * @return the ring index to hand hexkl_dma_ring_wait */
uint32_t hexkl_moe_push_act_block(uint8_t *vtcm_base, uint32_t act_off,
                                  const uint8_t *act_ah, uint32_t slot,
                                  uint32_t K, uint32_t k_tiles);

/** @brief Bulk copy through the DMA engine, drained before returning. */
void hexkl_moe_dma_copy(void *dst, const void *src, size_t bytes, int src_vtcm,
                        int dst_vtcm);

/** @brief Background-lane pack of the activation into AH tiles, one unit
 *         per HEXKL_MOE_PACK_UNIT_ROWS rows; slot_row NULL packs rows in
 *         order. */
typedef struct {
  const float *act_c;
  const uint32_t *slot_row;
  const float *slot_scale;
  const int32_t *slot_zp;
  uint8_t *act_ah;
  uint32_t K;
} hexkl_moe_pack_ctx;

/** @brief Rows per background unit: a quarter block. A worker mid-unit
 *         picks the next epilogue up late by one unit, and at 64 rows that
 *         read as +0.32 ms/call in the DEQUANT column (doc 47 section 19.2);
 *         16 rows is ~2.5 us of work. Divides 64, so a block is whole
 *         units and the waits stay block arithmetic. */
#define HEXKL_MOE_PACK_UNIT_ROWS 16u

void hexkl_moe_pack_bg_worker(uint32_t n_units, uint32_t u, void *vctx);

/** @brief The unit count that covers slots [0, slot + 64): what to wait for
 *         before the block at @a slot is queued. */
#define HEXKL_MOE_PACK_UNITS_THROUGH(slot)                                     \
  (((slot) + HEXKL_HMX_INT8_BLOCK_N_ROW) / HEXKL_MOE_PACK_UNIT_ROWS)

/**
 * @brief Times an n-tile loop's HMX issue by difference.
 *
 * hexkl_micro_hmx_acc_read_int32 and the tile dequant sit inside the same
 * loop and are already probed, so their running totals are snapshotted
 * across it and subtracted rather than probed again per tile: two clock
 * reads for a 112-tile loop instead of 224. What is left is acc_clear, the
 * k-tile mm calls and the loop itself -- exactly what the old mm residual
 * was meant to be, except a residual also absorbs everything unnamed, which
 * is how it moved 3.1 ms between two runs at the same block count with
 * nothing in between that touches it.
 *
 * Needs mm_t0 / mm_acc0 / mm_dq0 (uint64_t) in scope; declare them once per
 * call so the loops in a block do not redeclare them.
 */
#define HEXKL_MOE_MM_BEGIN()                                                   \
  do {                                                                         \
    if (hexkl_probe_on) {                                                      \
      mm_acc0 = hexkl_probe_us[HEXKL_PROBE_ACC_READ];                          \
      mm_dq0 = hexkl_probe_us[HEXKL_PROBE_DEQUANT];                            \
      mm_t0 = hexkl_probe_now();                                               \
    }                                                                          \
  } while (0)

#define HEXKL_MOE_MM_END()                                                     \
  do {                                                                         \
    if (hexkl_probe_on) {                                                      \
      hexkl_probe_us[HEXKL_PROBE_MM] +=                                        \
        (hexkl_probe_now() - mm_t0) -                                          \
        (hexkl_probe_us[HEXKL_PROBE_ACC_READ] - mm_acc0) -                     \
        (hexkl_probe_us[HEXKL_PROBE_DEQUANT] - mm_dq0);                        \
    }                                                                          \
  } while (0)

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
 * @param[in,out] scratch session-lifetime heap scratch; grown here as needed
 * @return AEE_SUCCESS, or the first failing stage's code
 */
int hexkl_mm_u8i4_moe_layer_run(
  hexkl_weight_u8i4_table *tbl, uint8_t *vtcm_base, uint32_t vtcm_size,
  uint32_t config_off, uint32_t M, uint32_t K, uint32_t inter, uint32_t N_out,
  uint32_t n_experts, const uint32_t *h_gate_up, const uint32_t *h_down,
  const uint32_t *row_index, const uint32_t *row_count, const float *row_weight,
  const float *act_f32, float *out_f32, hvx_worker_pool *pool,
  hexkl_moe_scratch *scratch);

#endif /* __NNTRAINER_HEXKL_MM_U8I4_MOE_H__ */
