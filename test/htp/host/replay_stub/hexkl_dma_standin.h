// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hexkl_dma_standin.h
 * @date   27 Sep 2026
 * @brief  Host stand-in for the user-DMA instructions hexkl_dma_ring.c
 *         issues: a descriptor lands whole when started or linked
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * Shared by replay_stub/ (dma_replay_host_check) and inproc/ (the
 * in-process HTP build, #84). Transfers retire in issue order, as one
 * dmlinked chain does; the device's timing and interleaving are not
 * modelled, which is the useful direction: a read before its wait
 * computes the right bytes here and stale ones on the device.
 */
#pragma once
#include "hexkl_dma_ring.h"
#include <string.h>

/** @brief The DMA stand-in: a descriptor lands whole the moment it is
 *  started or linked, so transfers retire in issue order (one chain). The
 *  device's timing and interleaving are not modelled. */
static inline void replay_stub_dma_run(void *p) {
  hexkl_dma_desc2d *d = (hexkl_dma_desc2d *)p;
  const uint32_t n = d->nrows_lo | (d->nrows_hi << 8);
  for (uint32_t r = 0; r < n; ++r) {
    memcpy((uint8_t *)d->dst + (size_t)r * d->dst_stride,
           (const uint8_t *)d->src + (size_t)r * d->src_stride, d->row_size);
  }
  d->done = 1;
}
/** @brief dmstart. */
static inline void hexkl_dma_start(void *p) { replay_stub_dma_run(p); }
/** @brief dmlink. */
static inline void hexkl_dma_link(void *cur, void *next) {
  (void)cur;
  replay_stub_dma_run(next);
}
/** @brief dmpoll. */
static inline void hexkl_dma_poll(void) {}
