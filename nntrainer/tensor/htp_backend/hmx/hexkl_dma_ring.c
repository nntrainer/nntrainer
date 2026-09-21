// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hexkl_dma_ring.c
 * @date   06 Aug 2026
 * @brief  Minimal dmlink user-DMA ring for cross-matmul weight prefetch
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Ported from the doc13 §3a measurement bench's ring_push2d/ring_drain,
 * itself lifted from llama.cpp ggml-hexagon's dma-queue.{c,h}. Behaviour is
 * unchanged; only the names are de-abbreviated and the file stands alone so
 * both the u8i4 and (later) u8i8 matmul modules can share one ring.
 */

#include "hexkl_dma_ring.h"

#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <hmx_hexagon_protos.h>
#include <string.h>

/** @brief Power of two, comfortably above the pushes any one call issues,
 *         so a call's transfers never wrap into ones still in flight. */
#define HEXKL_DMA_RING_N 256u

static hexkl_dma_desc2d g_ring[HEXKL_DMA_RING_N] __attribute__((aligned(128)));
static uint32_t g_push, g_pop;
static hexkl_dma_desc2d *g_tail;
static int g_started;

static void hexkl_dma_ring_wait_idx_(uint32_t i) {
  long guard = 0;
  while (!g_ring[i].done && guard++ < 50000000L) {
    hexkl_dma_poll();
  }
}

void hexkl_dma_ring_reset(void) {
  memset(g_ring, 0, sizeof(g_ring));
  g_push = 0;
  g_pop = 0;
  g_tail = &g_ring[HEXKL_DMA_RING_N - 1];
  g_started = 0;
}

void hexkl_dma_ring_push2d(void *dst, const void *src, uint32_t dst_stride,
                           uint32_t src_stride, uint32_t row_size,
                           uint32_t nrows, int src_vtcm, int dst_vtcm) {
  if (((g_push + 1) & (HEXKL_DMA_RING_N - 1)) == g_pop) {
    // Ring full: the oldest transfer must have already finished by the time
    // we have wrapped this far around, so reclaiming it is a formality, not
    // a stall -- but wait for `done` explicitly rather than assume it.
    hexkl_dma_ring_wait_idx_(g_pop);
    g_pop = (g_pop + 1) & (HEXKL_DMA_RING_N - 1);
  }
  hexkl_dma_desc2d *d = &g_ring[g_push];
  d->next = 0;
  d->desc_size = 1;
  d->desc_type = 9;
  d->src_comp = 0;
  d->dst_comp = 0;
  d->src_bypass = src_vtcm ? 1 : 0;
  d->dst_bypass = dst_vtcm ? 1 : 0;
  d->order = 0;
  d->done = 0;
  d->reserved0 = 0;
  d->reserved1 = 0;
  d->offset = 0;
  d->src = (void *)src;
  d->dst = dst;
  d->src_stride = src_stride;
  d->dst_stride = dst_stride;
  d->row_size = row_size;
  d->nrows_lo = nrows & 0xffu;
  d->nrows_hi = (nrows >> 8) & 0xffu;
  Q6_dccleaninva_A(
    (void *)d); // push the descriptor to memory for the DMA engine
  if (!g_started) {
    hexkl_dma_start(d);
    g_started = 1;
  } else {
    hexkl_dma_link(g_tail, d);
  }
  g_tail = d;
  g_push = (g_push + 1) & (HEXKL_DMA_RING_N - 1);
}

uint32_t hexkl_dma_ring_next_idx(void) { return g_push; }

void hexkl_dma_ring_wait(uint32_t idx) {
  /* Descriptors are dmlinked, so they retire in push order and waiting on
     one implies every earlier one. g_pop is deliberately NOT advanced: the
     ring-full path in push2d reclaims, and leaving it alone keeps drain's
     "everything pending" meaning intact for the callers that still use it. */
  hexkl_dma_ring_wait_idx_(idx);
}

void hexkl_dma_ring_drain(void) {
  while (g_pop != g_push) {
    hexkl_dma_ring_wait_idx_(g_pop);
    g_pop = (g_pop + 1) & (HEXKL_DMA_RING_N - 1);
  }
  g_started = 0; // engine idle after drain -- the next push must dmstart
}
