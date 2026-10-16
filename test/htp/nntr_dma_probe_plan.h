// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   nntr_dma_probe_plan.h
 * @date   21 Sep 2026
 * @brief  Pure descriptor planning for the per-worker DMA bandwidth probe
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * Header-only and free of Hexagon headers so test/htp/host can compile the
 * same function the skel runs and check the plan's geometry (disjoint
 * source rows, in-bounds destinations, worker balance) without a device.
 */

#ifndef __NNTR_DMA_PROBE_PLAN_H__
#define __NNTR_DMA_PROBE_PLAN_H__

#include <stdint.h>

/** @brief Upper bound on descriptors one probe call issues. 256 MiB of
 *  8 KiB x 64 @ 56 KiB stride is 511; 1024 leaves room for a 128 MiB
 *  chunk of a smaller row without changing the contract. */
#define NNTR_DMA_PROBE_MAX_DESC 1024u

/** @brief Largest payload one descriptor carries (row_size x nrows). */
#define NNTR_DMA_PROBE_MAX_DESC_BYTES (1u << 20)

/** @brief One planned transfer: rows of @a row_size at @a src_stride from
 *  @a src_off, landing contiguously at @a dst_off inside @a worker's VTCM
 *  slice. */
typedef struct {
  uint32_t src_off;
  uint32_t dst_off;
  uint32_t row_size;
  uint32_t nrows;
  uint32_t src_stride;
  uint32_t worker;
} nntr_dma_probe_desc;

/**
 * @brief Splits [0, bytes) of a source range into descriptors.
 *
 * The range is read as bands of nrows x src_stride bytes; inside a band,
 * column c in [0, src_stride / row_size) is one descriptor reading rows
 * [c * row_size, (c + 1) * row_size) of every row -- the geometry of a
 * column block of a WH weight (hexkl_mm_u8i4_moe.c, row = cn * 512,
 * stride = n_col * 512). src_stride == row_size degenerates to contiguous
 * descriptors. Descriptors go round-robin to @a workers workers, and each
 * worker's destination cycles through the 1-or-more payload-sized slots
 * of its @a vtcm_per_worker slice.
 *
 * @return descriptors written to @a out (at most @a out_max), or 0 when
 *         the shape is unusable: zero sizes, row_size > src_stride,
 *         payload above NNTR_DMA_PROBE_MAX_DESC_BYTES or the slice, or
 *         no whole band fits.
 */
static inline uint32_t nntr_dma_probe_plan(uint32_t bytes, uint32_t row_size,
                                           uint32_t nrows, uint32_t src_stride,
                                           uint32_t workers,
                                           uint32_t vtcm_per_worker,
                                           nntr_dma_probe_desc *out,
                                           uint32_t out_max) {
  if (row_size == 0u || nrows == 0u || src_stride < row_size || workers == 0u ||
      out == 0) {
    return 0u;
  }
  const uint64_t payload = (uint64_t)row_size * nrows;
  if (payload > NNTR_DMA_PROBE_MAX_DESC_BYTES || payload > vtcm_per_worker) {
    return 0u;
  }
  const uint32_t cols = src_stride / row_size;
  const uint64_t band = (uint64_t)nrows * src_stride;
  const uint32_t slots = (uint32_t)(vtcm_per_worker / payload);
  uint32_t n = 0;
  uint32_t per_worker_count[64];
  uint32_t w;
  if (workers > 64u) {
    return 0u;
  }
  for (w = 0; w < workers; ++w) {
    per_worker_count[w] = 0;
  }
  for (uint64_t b0 = 0; b0 + band <= bytes && n < out_max; b0 += band) {
    for (uint32_t c = 0; c < cols && n < out_max; ++c) {
      nntr_dma_probe_desc *d = &out[n];
      d->src_off = (uint32_t)(b0 + (uint64_t)c * row_size);
      d->row_size = row_size;
      d->nrows = nrows;
      d->src_stride = src_stride;
      d->worker = n % workers;
      d->dst_off = (uint32_t)((per_worker_count[d->worker] % slots) * payload);
      per_worker_count[d->worker]++;
      ++n;
    }
  }
  return n;
}

#endif /* __NNTR_DMA_PROBE_PLAN_H__ */
