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

#endif /* __NNTRAINER_HEXKL_DMA_RING_H__ */
