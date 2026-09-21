// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hexkl_dma_trace.h
 * @date   21 Sep 2026
 * @brief  Per-descriptor bookkeeping for one MoE layer call's DMA ring use
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * Answers issue #87 (LEDGER item 6, wall 2): the MoE layer call averages
 * 16-18 GB/s over the call while the same engine on the same arena does
 * 72-117 GB/s in isolation. This records, only while hexkl_probe_on is
 * set, every push (issue time, shape, ring depth) and every wait (entry,
 * exit, whether it blocked) of one call, and brackets each descriptor's
 * completion between two instrumentation points -- the engine does not
 * timestamp completion, so "busy" is a range, never a point. The ring
 * itself (hexkl_dma_ring.c, shared with prefill) is not touched beyond the
 * one pure query hexkl_dma_ring_is_done; every call into this file sits
 * behind if (hexkl_probe_on) in the kernel.
 *
 * Free of Hexagon headers: times arrive as raw qtimer ticks from the
 * caller (hexkl_probe_now_ticks), so test/htp/host compiles this file
 * as-is and scripts the clock. Fixed static tables, no heap: about 13 KiB
 * of .bss in the skel, sized for the 256-slot ring.
 */

#ifndef __NNTRAINER_HEXKL_DMA_TRACE_H__
#define __NNTRAINER_HEXKL_DMA_TRACE_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief One call's tables: the ring holds 256 descriptors, so a call
 *  cannot have more in flight; the MoE call makes 14 waits at M=1 and
 *  about 60 at a 1776-row prefill. */
#define HEXKL_DMA_TRACE_MAX_PUSH 256u
#define HEXKL_DMA_TRACE_MAX_WAIT 64u

/** @brief What a pushed descriptor carries. */
enum {
  HEXKL_DMA_KIND_ACT = 0, /**< one 64-row activation block */
  HEXKL_DMA_KIND_GATE,    /**< gate half of a paired gate_up chunk */
  HEXKL_DMA_KIND_UP,      /**< up half of a paired gate_up chunk */
  HEXKL_DMA_KIND_DOWN,    /**< a down chunk */
  HEXKL_DMA_KIND_COPY     /**< moe_dma_copy piece (act in / out out) */
};

/** @brief Which wait in hexkl_mm_u8i4_moe.c a wait record came from. */
enum {
  HEXKL_DMA_SITE_ACT = 0,  /**< the activation-block wait before a block */
  HEXKL_DMA_SITE_GU,       /**< the gate_up chunk wait (DRAIN) */
  HEXKL_DMA_SITE_DN,       /**< the down chunk wait (DRAIN_DN) */
  HEXKL_DMA_SITE_COPY_IN,  /**< moe_dma_copy's drain, activation in */
  HEXKL_DMA_SITE_COPY_OUT, /**< moe_dma_copy's drain, output out */
  HEXKL_DMA_SITE_N
};

/** @brief One push. Times are ticks since hexkl_dma_trace_reset. depth is
 *  how many descriptors were outstanding BEFORE this one (0 = the ring
 *  was seen empty). Completion lies in [t_done_lo, t_done_hi]: the last
 *  instrumentation point at which the done bit read 0 and the first at
 *  which it read 1. */
typedef struct {
  uint32_t t_issue;
  uint32_t kind;
  uint32_t expert; /**< expert ordinal in the call's active order */
  uint32_t chunk;
  uint32_t bytes;
  uint32_t row_size;
  uint32_t nrows;
  uint32_t src_stride;
  uint32_t ring_idx;
  uint32_t depth;
  uint32_t t_done_lo;
  uint32_t t_done_hi;
} hexkl_dma_trace_push_rec;

/** @brief One wait. blocked = the done bit read 0 at entry. */
typedef struct {
  uint32_t t_in;
  uint32_t t_out;
  uint32_t ring_idx;
  uint32_t site;
  uint32_t blocked;
} hexkl_dma_trace_wait_rec;

/** @brief Word counts of the two records as hexkl_dma_trace_serialize
 *  lays them out; the host decodes with the same constants. */
#define HEXKL_DMA_TRACE_PUSH_WORDS 12u
#define HEXKL_DMA_TRACE_WAIT_WORDS 5u
/** @brief Serialized header: n_push, n_wait, n_blocked, depth_max,
 *  n_dropped, t_end, push words, wait words. */
#define HEXKL_DMA_TRACE_HDR_WORDS 8u
#define HEXKL_DMA_TRACE_MAX_WORDS                                              \
  (HEXKL_DMA_TRACE_HDR_WORDS +                                                 \
   HEXKL_DMA_TRACE_MAX_PUSH * HEXKL_DMA_TRACE_PUSH_WORDS +                     \
   HEXKL_DMA_TRACE_MAX_WAIT * HEXKL_DMA_TRACE_WAIT_WORDS)

typedef struct {
  uint32_t n_push;       /**< records kept (at most MAX_PUSH) */
  uint32_t n_wait;       /**< records kept (at most MAX_WAIT) */
  uint32_t n_blocked;    /**< waits that found the done bit 0 at entry */
  uint32_t depth_max;    /**< most descriptors outstanding at any point */
  uint32_t n_dropped;    /**< pushes past MAX_PUSH: counted, not recorded */
  uint32_t t_end;        /**< time of the finish call */
  uint32_t seen_done;    /**< watermark: records [0, seen_done) are done */
  uint32_t t_last;       /**< time of the previous instrumentation point */
  uint32_t n_wait_total; /**< waits seen, recorded or not */
  uint32_t cur_wait;     /**< 1 + index of the wait between begin and end,
                              0 when none or when it was not recorded */
  uint64_t t0;
  hexkl_dma_trace_push_rec push[HEXKL_DMA_TRACE_MAX_PUSH];
  hexkl_dma_trace_wait_rec wait[HEXKL_DMA_TRACE_MAX_WAIT];
} hexkl_dma_trace;

/** @brief The per-call numbers, all in ticks except the counts. */
typedef struct {
  uint32_t n_desc;      /**< pushes, dropped ones included */
  uint32_t n_wait;      /**< waits, recorded or not */
  uint32_t n_blocked;   /**< of those, blocked at entry */
  uint32_t depth_max;   /**< see hexkl_dma_trace::depth_max */
  uint32_t wait;        /**< sum over every wait's t_out - t_in */
  uint32_t wait_act;    /**< the HEXKL_DMA_SITE_ACT share of that */
  uint32_t busy_lo;     /**< union of [t_issue, t_done_lo] */
  uint32_t busy_hi;     /**< union of [t_issue, t_done_hi] */
  uint32_t first_ready; /**< expert 0's gate_up all done (t_done_hi) */
  uint32_t last_issue;  /**< the last weight push's t_issue */
} hexkl_dma_trace_summary;

/** @brief Empties the tables; @a now becomes t = 0. */
void hexkl_dma_trace_reset(uint64_t now);

/** @brief Records a descriptor just pushed at ring slot @a ring_idx
 *  (hexkl_dma_ring_next_idx before the push). Advances the completion
 *  watermark first, so depth is what the ring held when this went in. */
void hexkl_dma_trace_push(uint64_t now, uint32_t ring_idx, uint32_t kind,
                          uint32_t expert, uint32_t chunk, uint32_t row_size,
                          uint32_t nrows, uint32_t src_stride);

/** @brief Call around hexkl_dma_ring_wait / drain on @a ring_idx. */
void hexkl_dma_trace_wait_begin(uint64_t now, uint32_t ring_idx, uint32_t site);
void hexkl_dma_trace_wait_end(uint64_t now);

/** @brief An instrumentation point with no push or wait of its own (the
 *  kernel samples once per HMX batch): advances the watermark. */
void hexkl_dma_trace_sample(uint64_t now);

/** @brief Closes the call at @a now (descriptors still open are bracketed
 *  to it) and computes the summary. Idempotent for the tables. */
void hexkl_dma_trace_finish(uint64_t now, hexkl_dma_trace_summary *out);

/** @brief The tables of the last call, for the skel's read-back entry. */
const hexkl_dma_trace *hexkl_dma_trace_get(void);

/** @brief Flattens the tables: header, then n_push x PUSH_WORDS, then
 *  n_wait x WAIT_WORDS. @return words written, 0 if @a max_words is too
 *  small for the whole thing. */
uint32_t hexkl_dma_trace_serialize(uint32_t *words, uint32_t max_words);

#ifdef __cplusplus
}
#endif

#endif /* __NNTRAINER_HEXKL_DMA_TRACE_H__ */
