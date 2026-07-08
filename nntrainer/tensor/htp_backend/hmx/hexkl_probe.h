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
  HEXKL_PROBE_DRAIN,        /**< hexkl_dma_ring_drain */
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

#endif /* __NNTRAINER_HEXKL_PROBE_H__ */
