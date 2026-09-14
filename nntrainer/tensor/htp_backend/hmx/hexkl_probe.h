// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hexkl_probe.h
 * @date   08 Aug 2026
 * @brief  Five counters that split layer_run's time (Tier 0 measurement)
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The two-parameter cost model (31_dataflow_as_built.md 3) attributes 75-96%
 * of attention DSP time to "accumulator readout", but that bucket is really
 * two calls -- acc_read_int32 (HMX acc -> VTCM, vendor) and
 * copy_32b_to_submatrix (VTCM -> DDR, ours) -- and which of the two carries
 * the 52 us decides whether the fix is dequant-from-VTCM or bypassing HMX for
 * small M. These counters exist to answer that one question and go away with
 * the rest of the debug surface (MHA_HTP_PLAN.md 9.5).
 *
 * Globals, not parameters: threading a stats struct through the layer_run
 * signature would touch every caller for scaffolding. The DSP session is
 * single-threaded through these calls, and the counters are read only by
 * hexkl_attn_u8_forward, which resets them on entry.
 */

#ifndef __NNTRAINER_HEXKL_PROBE_H__
#define __NNTRAINER_HEXKL_PROBE_H__

#include <stdint.h>

#include <HAP_perf.h>

/** @brief What each counter times, inside every layer_run call. */
enum {
  HEXKL_PROBE_ACC_READ = 0, /**< hexkl_micro_hmx_acc_read_int32 */
  HEXKL_PROBE_ACC_COPY,     /**< hexkl_micro_hmx_copy_32b_to_submatrix */
  HEXKL_PROBE_DEQUANT,      /**< hvx_dequant_i32_to_f32 */
  HEXKL_PROBE_QUANT,        /**< hvx_quant_rows_u8_params + pack */
  /** The MoE layer's three quantization-adjacent costs, split apart. One
      bucket held all of them and read 5537us both before and after the
      restructure that was supposed to cut it to ~1500 -- which is only
      readable as "the saving landed somewhere else in the same bucket" if
      the bucket is split. QUANT keeps the layer-wide activation quantize
      (now once, not once per expert block), GATHER is the uint8 row pick
      that replaced re-quantizing per block, REQUANT is the SwiGLU output's
      per-block quantize, which the restructure never touched. */
  HEXKL_PROBE_GATHER,
  HEXKL_PROBE_REQUANT,
  /** The HMX issue loop itself: acc_clear plus the k-tile mm calls, with
      the acc_read and dequant that share the n-tile loop subtracted back
      out (both are already timed, so their totals are snapshotted around
      the loop rather than probed again inside it -- two timer reads per
      block instead of 176, which is what makes naming this stage affordable
      at all). Until now mm was a residual, and a residual absorbs whatever
      the probes do not name: it read 10638 us at 42.7 ms and 13722 at
      37.2 ms with the same 43 blocks, which no change in between explains.
      Naming it turns the leftover into a real leftover. */
  HEXKL_PROBE_MM,
  /** NOT a time: kilobytes pushed through the DMA ring this call. Pairs
      with DMA_FIRST below. */
  HEXKL_PROBE_DMA_KB,
  /** Microseconds of the FIRST weight drain, which waits on exactly one
      gate_up transfer (3.5 MiB) with nothing else in flight -- the one
      point in the call where a drain measures a transfer instead of a
      pipeline. 3.5 MiB over this gives the DDR-to-VTCM rate, and that
      number decides whether the rest of the kernel work is worth doing:
      the 176 MB of int4 expert weights per layer need 12.8 GB/s to hide
      behind the matmul, and the staging copies (DDR to DDR) already show
      18.7 GB/s. If the weight path is near 6 GB/s instead, the layer is
      bandwidth-bound at ~30 ms and no amount of gather or dequant work
      moves it. */
  HEXKL_PROBE_DMA_FIRST,
  /** The two drains an expert makes, apart. DRAIN waits on that expert's
      gate_up, DRAIN_DN on its down -- one bucket held both, and a bucket
      holding two things is how section 13.1's quant reading stayed
      unreadable for two runs. The split matters here because the two have
      different amounts of work in front of them to hide behind: gate_up
      was pushed a whole expert earlier, down only one gate_up matmul
      earlier. */
  HEXKL_PROBE_DRAIN_DN,
  /** hexkl_dma_ring_push2d itself. It chains onto an in-flight descriptor
      via dmlink, and whether that ever blocks is the difference between
      "the prefetch is not hiding" and "issuing the prefetch is the cost".
      At 31.6 GB/s measured and 5.6 GB/s averaged, something is serialising
      and the push is one of two candidates. */
  HEXKL_PROBE_PUSH,
  HEXKL_PROBE_DRAIN,        /**< hexkl_dma_ring_drain */
  HEXKL_PROBE_SWIGLU,       /**< hvx_swiglu_inplace_f32 (fused layer only) */
  /** Routing multiply + accumulate into the layer output (MoE layer call
      only). Its own slot rather than sharing ACC_COPY: the first
      measurement of that call had 104.8 ms in a bucket ACC_READ and this
      shared, and which of the two it was could not be read off the
      profile. */
  HEXKL_PROBE_SCATTER,
  /** NOT a time: how many 64-row blocks the MoE layer issued, summed over
      experts. The matmul cost scales with THIS, not with the routed row
      count -- HMX computes all 64 rows of a tile whatever m_blk says. At
      1776 rows over 32 experts the average expert holds 55.5, just under
      the tile, so whether an expert lands at 60 or 70 decides between one
      block and two-with-one-nearly-empty. The spread between the two
      readings is 22.5 vs 33.8 GMAC, and nothing in the profile could tell
      them apart. */
  HEXKL_PROBE_BLOCKS,
  /** NOT a time: hexkl_acc_layout::row_stride when the in-place tile dequant
   *  is running, 0 when layer_run fell back to the vendor copy. One number
   *  that says which path the numbers beside it came from. */
  HEXKL_PROBE_ACC_STRIDE,
  HEXKL_PROBE_N
};

/** @brief Accumulated microseconds per slot since the last reset. */
extern uint64_t hexkl_probe_us[HEXKL_PROBE_N];

/**
 * @brief Nonzero while a caller wants the breakdown.
 *
 * The acc_read and dequant probes sit INSIDE the tile loop, so they run
 * 1,536 times for one kv=1024 prefill layer -- four qtimer reads each. That
 * is instrumentation the production entry point should not be paying, and
 * before this flag it did: the probes were unconditional while the stage
 * timers around them were already gated on stage_us.
 */
extern int hexkl_probe_on;

/** @brief Zeroes every counter and sets whether the probes run at all. */
void hexkl_probe_reset(int enable);

/** @brief Same DSP-internal clock hexkl_attn_u8.c times stages with. */
static inline uint64_t hexkl_probe_now(void) {
  return HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count());
}

/** @brief Reads the clock into @a v, or does nothing when probing is off. */
#define HEXKL_PROBE_T0(v)                                                      \
  do {                                                                         \
    if (hexkl_probe_on) {                                                      \
      (v) = hexkl_probe_now();                                                 \
    }                                                                          \
  } while (0)

/** @brief Folds now - @a t0 into @a slot, or does nothing when probing is
 *         off. Pairs with HEXKL_PROBE_T0 on the same variable. */
#define HEXKL_PROBE_ADD(slot, t0)                                              \
  do {                                                                         \
    if (hexkl_probe_on) {                                                      \
      hexkl_probe_us[slot] += hexkl_probe_now() - (t0);                        \
    }                                                                          \
  } while (0)

/** @brief Folds @a n into a slot that holds a count rather than a time.
 *         Same gate as the timers, so "probing off" means off. */
#define HEXKL_PROBE_COUNT(slot, n)                                             \
  do {                                                                         \
    if (hexkl_probe_on) {                                                      \
      hexkl_probe_us[slot] += (n);                                             \
    }                                                                          \
  } while (0)

#endif /* __NNTRAINER_HEXKL_PROBE_H__ */
