// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hexkl_dma_ring.h
 * @date   06 Aug 2026
 * @brief  Minimal dmlink user-DMA ring for cross-matmul weight prefetch
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __NNTRAINER_HEXKL_DMA_RING_H__
#define __NNTRAINER_HEXKL_DMA_RING_H__

#include <stdint.h>

/** @brief One 2D DMA descriptor. Hardware-defined layout; do not reorder.
 *
 * Shared with test/htp/nntr_hvx_dma_probe.c, which drives one engine per
 * worker thread with its own descriptors instead of this file's single
 * dmlinked chain. The asm helpers below are Hexagon-only; a host build
 * (test/htp/host) sees the struct and nothing else. */
typedef struct __attribute__((aligned(128))) hexkl_dma_desc2d_s {
  void *next;
  uint32_t dst_stride : 24;
  uint32_t desc_size : 2;
  uint32_t dst_comp : 1;
  uint32_t src_comp : 1;
  uint32_t dst_bypass : 1;
  uint32_t src_bypass : 1;
  uint32_t order : 1;
  uint32_t done : 1;
  void *src;
  void *dst;
  uint32_t desc_type : 8;
  uint32_t reserved0 : 24;
  uint32_t row_size : 24;
  uint32_t nrows_lo : 8;
  uint32_t nrows_hi : 8;
  uint32_t src_stride : 24;
  uint32_t offset : 24;
  uint32_t reserved1 : 8;
} hexkl_dma_desc2d;

#if defined(__hexagon__)
/** @brief Chains @a next after the in-flight @a cur (dmlink). */
static inline void hexkl_dma_link(void *cur, void *next) {
  asm volatile(" release(%0):at" : : "r"(next));
  asm volatile(" dmlink(%0, %1)" : : "r"(cur), "r"(next));
}

/** @brief Starts this thread's DMA engine on descriptor @a p (dmstart). */
static inline void hexkl_dma_start(void *p) {
  asm volatile(" release(%0):at" : : "r"(p));
  asm volatile(" dmstart(%0)" : : "r"(p));
}

/** @brief One dmpoll; callers spin on the descriptor's done bit around it. */
static inline void hexkl_dma_poll(void) {
  unsigned r = 0;
  asm volatile(" %0 = dmpoll" : "=r"(r) : : "memory");
  (void)r;
}
#endif /* __hexagon__ */

/**
 * @brief Resets the ring to empty.
 *
 * Call once per logical group of transfers (here: once per mm_u8i4_layer
 * call), not once per transfer -- hexkl_dma_ring_push2d's own wraparound
 * handling is what keeps a long-running ring correct across many pushes.
 */
void hexkl_dma_ring_reset(void);

/**
 * @brief Queues one 2D transfer; starts the DMA engine if idle, otherwise
 *        chains onto the in-flight descriptor via dmlink so many transfers
 *        run on the single DMA engine without a second dmstart.
 *
 * @param src_vtcm  nonzero if @a src is in VTCM (bypasses L2)
 * @param dst_vtcm  nonzero if @a dst is in VTCM (bypasses L2)
 */
void hexkl_dma_ring_push2d(void *dst, const void *src, uint32_t dst_stride,
                           uint32_t src_stride, uint32_t row_size,
                           uint32_t nrows, int src_vtcm, int dst_vtcm);

/**
 * @brief The ring slot the next hexkl_dma_ring_push2d will occupy.
 *
 * Take it before the push, pass it to hexkl_dma_ring_wait after. Lets a
 * caller split one logical transfer into chunks and start computing on the
 * first while the rest is still moving -- which drain cannot express,
 * because it waits for everything and so also for the prefetches a
 * pipelined caller deliberately has in flight.
 *
 * The ring holds 256 descriptors and wraps, so an index is only meaningful
 * until that many further pushes have happened. Callers that queue a
 * bounded number per group (the MoE layer queues about 194) are safe by
 * construction; one that queued more would have to wait sooner.
 */
uint32_t hexkl_dma_ring_next_idx(void);

/** @brief Blocks until the transfer queued at @a idx has completed. */
void hexkl_dma_ring_wait(uint32_t idx);

/** @brief Blocks until every queued transfer has completed. */
void hexkl_dma_ring_drain(void);

/**
 * @brief Whether the transfer queued at @a idx has completed: its done bit,
 *        read once. No dmpoll, no state change -- a pure query for the
 *        #87 trace (hexkl_dma_trace.c), which brackets completions between
 *        the points it already visits rather than adding any of its own.
 */
int hexkl_dma_ring_is_done(uint32_t idx);

#endif /* __NNTRAINER_HEXKL_DMA_RING_H__ */
