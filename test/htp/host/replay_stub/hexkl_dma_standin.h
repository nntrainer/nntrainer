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

/** @brief [#90] What the stand-in moved: every landed byte, and when
 *  @a pages is set, one byte per 4 KiB source page of [base, base +
 *  n_pages x 4 KiB) set to 1 once a row reads it (two_reader_host_check's
 *  footprint). Weak, so every translation unit that includes this header
 *  shares the one instance. */
typedef struct {
  uint64_t bytes_landed;
  const uint8_t *base;
  uint8_t *pages;
  size_t n_pages;
} replay_stub_meter_t;
__attribute__((weak)) replay_stub_meter_t replay_stub_meter;

/** @brief The DMA stand-in: a descriptor lands whole the moment it is
 *  started or linked, so transfers retire in issue order (one chain). The
 *  device's timing and interleaving are not modelled. */
static inline void replay_stub_dma_run(void *p) {
  hexkl_dma_desc2d *d = (hexkl_dma_desc2d *)p;
  replay_stub_meter_t *m = &replay_stub_meter;
  const uint32_t n = d->nrows_lo | (d->nrows_hi << 8);
  for (uint32_t r = 0; r < n; ++r) {
    const uint8_t *src = (const uint8_t *)d->src + (size_t)r * d->src_stride;
    memcpy((uint8_t *)d->dst + (size_t)r * d->dst_stride, src, d->row_size);
    if (m->pages && src >= m->base &&
        src + d->row_size <= m->base + (m->n_pages << 12)) {
      const size_t lo = (size_t)(src - m->base) >> 12,
                   hi = (size_t)(src + d->row_size - 1 - m->base) >> 12;
      memset(m->pages + lo, 1, hi - lo + 1);
    }
  }
  m->bytes_landed += (uint64_t)n * d->row_size;
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
