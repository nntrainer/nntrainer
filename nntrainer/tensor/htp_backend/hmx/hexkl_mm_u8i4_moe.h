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

#include "hexkl_dma_trace.h" /* HEXKL_MOE_MM_END samples the #87 trace */
#include "hexkl_micro.h"     /* the block constants the shared macros use */
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
      /* One #87 completion sample per HMX batch: the bracket on every         \
         descriptor is only as tight as the points that read its bit. */       \
      hexkl_dma_trace_sample(hexkl_probe_now_ticks());                         \
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
 * @param[in] flags       HEXKL_MOE_FLAG_* bits; 0 is the HMX block loop
 * @return AEE_SUCCESS, or the first failing stage's code
 */
int hexkl_mm_u8i4_moe_layer_run(
  hexkl_weight_u8i4_table *tbl, uint8_t *vtcm_base, uint32_t vtcm_size,
  uint32_t config_off, uint32_t M, uint32_t K, uint32_t inter, uint32_t N_out,
  uint32_t n_experts, const uint32_t *h_gate_up, const uint32_t *h_down,
  const uint32_t *row_index, const uint32_t *row_count, const float *row_weight,
  const float *act_f32, float *out_f32, hvx_worker_pool *pool,
  hexkl_moe_scratch *scratch, uint32_t flags);

/**
 * @brief Take a call of at most 4 rows (M <= 4, at most 16 active experts)
 *        through the HVX GEMV -- every expert, no 64-row HMX block, no
 *        weight DMA -- instead of the block loop. Bit-identical output:
 *        the GEMV's int32 sums are the HMX's own and the epilogues are the
 *        same functions on the same numbers. Off by default; a call the
 *        bounds exclude falls through to the HMX loop unchanged.
 */
#define HEXKL_MOE_FLAG_M1_GEMV 1u

/**
 * @brief The M=1 GEMV path's l2fetch lead, in KB of weight per lane: a
 *        lane issues the box of its next block of units -- that many KB of
 *        adjacent columns, at least one unit, never across an expert --
 *        before it computes the current one, so each block's weight is on
 *        its way to L2 a block ahead. 0 is the old behaviour: every column
 *        issues its own l2fetch right before its loads. A hint only; no
 *        result depends on it. The compile-time default, overridable per
 *        call by the moe_set_opts bits below; 192 since the #113 sitting
 *        (D192: one-row loop + 192 KB, decode +3.3..4.3 %, the only cell
 *        of the 2 x 5 matrix that beat the four-row loop without a lead).
 *
 * Read it as the *gate* half's lead. The hardware queues three l2fetch
 * per thread and stalls the thread on a fourth, so stage A can only issue
 * block b+1's up box after block b's last gate column (moe_m1_pair_worker):
 * the gate columns of a block get the full lead and its up columns get
 * about two column-computes of it. Half of stage A's weight traffic
 * therefore runs at well under the nominal number, which is why the sweep
 * reads the lead as an ordinal knob and not as a byte count. Giving the up
 * half its own lead needs a fourth outstanding box and is #114's.
 */
#ifndef HVX_GEMV_PF_LEAD_KB
#define HVX_GEMV_PF_LEAD_KB 192u
#endif

/**
 * @brief The M=1 GEMV path's row loop: 1 = a lone last row takes
 *        gemm_row1's single accumulator (PR #107), 0 = every row group
 *        takes gemm_rows4. The compile-time default, overridable per call
 *        by the bits below. 1 since #113's matrix: the lead helps only
 *        the loop that is latency-starved (LEDGER rule 31), and the
 *        four-row loop, with eight vrmpy of loads in flight per
 *        quarter-tile, is not -- the lead regresses it 2.1-2.5x at every
 *        size. The prefill tail keeps gemm_rows4 by passing 0u itself.
 */
#ifndef HVX_GEMV_M1_ROWS1
#define HVX_GEMV_M1_ROWS1 1u
#endif

/**
 * @brief The M=1 GEMV path's weight feed: 1 = each expert's gate_up and
 *        down are staged into VTCM by the DMA engine one expert ahead
 *        (#117: the #100 `f2` shape, whole-matrix descriptors, two 3.5 MiB
 *        slabs double-buffered by expert) and the GEMV reads the VTCM
 *        copy; 0 = the GEMV reads the arena behind its own l2fetch. The
 *        compile-time default, overridable per call by the bits below.
 *        1 since #117's sitting passed its gate twice (M==1 mm 676 us vs
 *        D192's 932, decode +29..33 % at every G, text identical, M>1
 *        dsp within 0.04 %). Under the feed the l2fetch lead is off by
 *        construction (nothing to fetch), so the (loop, lead) pair reads
 *        as (rows1, ignored).
 */
#ifndef HVX_GEMV_M1_FEED
#define HVX_GEMV_M1_FEED 1u
#endif

/**
 * @brief moe_set_opts' tune bits: each says that its own field below is
 *        authoritative and replaces the matching compile-time default above
 *        for every call of the session. Unset, that default stands. The
 *        ARM side (htp_moe_opts.h) always sets all three, sending its own
 *        copy of the defaults when the env names nothing, so every log's
 *        echo names the (loop, lead, feed) cell that ran (LEDGER rule 21).
 *
 * The point of carrying the pair in the word rather than in the build is
 * that one skel and one app then serve the whole (loop x lead) matrix,
 * and moe_set_opts' echo (nntr_hvx_moe_set_opts' *applied, which the ARM
 * side compares and throws on) proves per log which cell ran. A skel that
 * predates these bits masks them off, so the echo differs and the run
 * fails loudly instead of silently measuring the wrong cell (#97's failure
 * mode, LEDGER rule 21).
 */
#define HEXKL_MOE_FLAG_GEMV_LEAD_SET 0x80u
#define HEXKL_MOE_FLAG_GEMV_ROWS1_SET 0x40u
/** @brief Bit 5: bit 17 (the VTCM feed) is authoritative (#117). */
#define HEXKL_MOE_FLAG_GEMV_FEED_SET 0x20u

/** @brief Bits [15:8] of the flags word: the l2fetch lead in units of
 *         HEXKL_MOE_GEMV_LEAD_KB_UNIT KB. 8 bits x 64 KB caps the lead at
 *         16320 KB, and the caller clamps to 127 units anyway (the
 *         l2fetch width field is 16 bits, so a box is at most 127 tiles of
 *         512 B). */
#define HEXKL_MOE_GEMV_LEAD_SHIFT 8u
#define HEXKL_MOE_GEMV_LEAD_BITS 0xFFu
#define HEXKL_MOE_GEMV_LEAD_KB_UNIT 64u

/** @brief Bit 16 of the flags word: the rows1 loop selector. */
#define HEXKL_MOE_FLAG_GEMV_ROWS1 0x10000u

/** @brief Bit 17 of the flags word: the VTCM feed selector (#117). */
#define HEXKL_MOE_FLAG_GEMV_FEED 0x20000u

/** @brief Every bit this build understands; moe_set_opts keeps these and
 *         drops the rest, which is what makes the echo a version check. */
#define HEXKL_MOE_FLAGS_KNOWN                                                  \
  (HEXKL_MOE_FLAG_M1_GEMV | HEXKL_MOE_FLAG_GEMV_LEAD_SET |                     \
   HEXKL_MOE_FLAG_GEMV_ROWS1_SET | HEXKL_MOE_FLAG_GEMV_FEED_SET |              \
   ((uint32_t)HEXKL_MOE_GEMV_LEAD_BITS << HEXKL_MOE_GEMV_LEAD_SHIFT) |         \
   HEXKL_MOE_FLAG_GEMV_ROWS1 | HEXKL_MOE_FLAG_GEMV_FEED)

/** @brief The call's l2fetch lead in KB: the flags word when the lead bit
 *         is set, else the build's default. */
static inline uint32_t hexkl_moe_flags_lead_kb(uint32_t flags) {
  if ((flags & HEXKL_MOE_FLAG_GEMV_LEAD_SET) == 0u) {
    return HVX_GEMV_PF_LEAD_KB;
  }
  return ((flags >> HEXKL_MOE_GEMV_LEAD_SHIFT) & HEXKL_MOE_GEMV_LEAD_BITS) *
         HEXKL_MOE_GEMV_LEAD_KB_UNIT;
}

/** @brief The call's row loop: the flags word when the rows1 bit is set,
 *         else the build's default. */
static inline uint32_t hexkl_moe_flags_rows1(uint32_t flags) {
  if ((flags & HEXKL_MOE_FLAG_GEMV_ROWS1_SET) == 0u) {
    return HVX_GEMV_M1_ROWS1;
  }
  return (flags & HEXKL_MOE_FLAG_GEMV_ROWS1) != 0u ? 1u : 0u;
}

/** @brief The call's weight feed: the flags word when the feed bit is
 *         set, else the build's default. 1 = VTCM by DMA, 0 = arena. */
static inline uint32_t hexkl_moe_flags_feed(uint32_t flags) {
  if ((flags & HEXKL_MOE_FLAG_GEMV_FEED_SET) == 0u) {
    return HVX_GEMV_M1_FEED;
  }
  return (flags & HEXKL_MOE_FLAG_GEMV_FEED) != 0u ? 1u : 0u;
}

#endif /* __NNTRAINER_HEXKL_MM_U8I4_MOE_H__ */
