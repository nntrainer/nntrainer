// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   nntr_hvx_dma_probe.c
 * @date   21 Sep 2026
 * @brief  FastRPC entry: per-worker dmstart bandwidth probe over an arena
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * Answers doc 48 section 5 C (LEDGER item 3): does the arena DMA rate move
 * with descriptor shape, with the number of engines issuing, or with the
 * bus vote? arena_probe cannot say -- one shape, the shared dmlinked ring
 * (one dmstart), the vote compiled in. Here every worker thread runs its
 * own engine on its own descriptors, and the skel it came from is named by
 * vote_state. No arithmetic, no DSP heap: descriptors live in a static
 * array sized for the worst-case plan (128 KiB), the destination is the
 * session's own VTCM below the HMX config block.
 */

#include <string.h>

#include <AEEStdErr.h>
#include <HAP_farf.h>
#include <HAP_perf.h>
#include <hexagon_protos.h>
#include <remote.h>

#include "hexkl_dma_ring.h"
#include "hexkl_probe.h"
#include "hvx_worker_pool.h"
#include "nntr_dma_probe_plan.h"
#include "nntr_hvx.h"
#include "nntr_hvx_session.h"

#ifdef NNTR_HVX_NO_BUS_VOTE
#define DMA_PROBE_VOTE_STATE 0u
#else
#define DMA_PROBE_VOTE_STATE 1u
#endif

static nntr_dma_probe_desc g_plan[NNTR_DMA_PROBE_MAX_DESC];
static hexkl_dma_desc2d g_desc[NNTR_DMA_PROBE_MAX_DESC]
  __attribute__((aligned(128)));

typedef struct {
  const uint8_t *src;
  uint8_t *vtcm;
  uint32_t vtcm_per_worker;
  uint32_t n_desc;
  uint32_t passes;
  uint32_t workers_used; /**< written by i == 0 from n_threads */
} dma_probe_ctx;

/** @brief Same fill as hexkl_dma_ring_push2d, minus the chaining. */
static void dma_probe_fill(hexkl_dma_desc2d *d, const nntr_dma_probe_desc *p,
                           const dma_probe_ctx *c) {
  memset(d, 0, sizeof(*d));
  d->desc_size = 1;
  d->desc_type = 9;
  d->src_bypass = 0;
  d->dst_bypass = 1;
  d->src = (void *)(c->src + p->src_off);
  d->dst = c->vtcm + (size_t)p->worker * c->vtcm_per_worker + p->dst_off;
  d->src_stride = p->src_stride;
  d->dst_stride = p->row_size;
  d->row_size = p->row_size;
  d->nrows_lo = p->nrows & 0xffu;
  d->nrows_hi = (p->nrows >> 8) & 0xffu;
}

/** @brief One worker: its descriptors, one dmstart each, in order, for
 *  every pass. The done spin is hexkl_dma_ring.c's, on this thread's own
 *  engine. */
static void dma_probe_worker(uint32_t n_threads, uint32_t i, void *v) {
  dma_probe_ctx *c = (dma_probe_ctx *)v;
  if (i == 0) {
    c->workers_used = n_threads;
  }
  for (uint32_t pass = 0; pass < c->passes; ++pass) {
    for (uint32_t k = 0; k < c->n_desc; ++k) {
      if (g_plan[k].worker != i) {
        continue;
      }
      hexkl_dma_desc2d *d = &g_desc[k];
      d->done = 0;
      Q6_dccleaninva_A((void *)d);
      hexkl_dma_start(d);
      long guard = 0;
      while (!d->done && guard++ < 50000000L) {
        hexkl_dma_poll();
      }
    }
  }
}

int nntr_hvx_dma_probe(remote_handle64 handle, uint32 arena, uint32 src_off,
                       uint32 bytes, uint32 row_size, uint32 nrows,
                       uint32 src_stride, uint32 workers, uint32 passes,
                       uint32 *res, int resLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s || !res || resLen < 10 || workers == 0u || passes == 0u) {
    return AEE_EBADPARM;
  }
  memset(res, 0, (size_t)resLen * sizeof(*res));
  if (arena >= NNTR_HVX_MAX_ARENAS || s->arenas[arena].va == NULL ||
      (uint64_t)src_off + bytes > s->arenas[arena].bytes) {
    FARF(ERROR, "dma_probe: arena %u not attached or [%u,+%u) out of range",
         (unsigned)arena, (unsigned)src_off, (unsigned)bytes);
    return AEE_EBADPARM;
  }

  /* Workers get equal slices of the VTCM below the HMX config block;
     nothing else of the session lives there between calls. Round-robin
     over the plan's workers, not the pool's, so a request for more
     threads than the pool has simply leaves some slices unused -- the
     n_threads the pool reports is what res[4] carries. */
  dma_probe_ctx c;
  memset(&c, 0, sizeof(c));
  c.src = s->arenas[arena].va + src_off;
  c.vtcm = s->vtcm_base;
  c.vtcm_per_worker = (s->config_off / workers) & ~127u;
  c.passes = passes;
  c.n_desc =
    nntr_dma_probe_plan(bytes, row_size, nrows, src_stride, workers,
                        c.vtcm_per_worker, g_plan, NNTR_DMA_PROBE_MAX_DESC);
  if (c.n_desc == 0u) {
    FARF(ERROR, "dma_probe: no plan for row=%u nrows=%u stride=%u workers=%u",
         (unsigned)row_size, (unsigned)nrows, (unsigned)src_stride,
         (unsigned)workers);
    return AEE_EBADPARM;
  }
  uint32_t max_desc_bytes = 0;
  for (uint32_t k = 0; k < c.n_desc; ++k) {
    dma_probe_fill(&g_desc[k], &g_plan[k], &c);
    const uint32_t b = g_plan[k].row_size * g_plan[k].nrows;
    if (b > max_desc_bytes) {
      max_desc_bytes = b;
    }
  }
  /* Zero what the checksum reads, so a stale slice from the previous cell
     cannot pass for a live transfer. */
  const uint32_t sum_bytes =
    max_desc_bytes < c.vtcm_per_worker ? max_desc_bytes : c.vtcm_per_worker;
  memset(c.vtcm, 0, sum_bytes);

  const uint64_t t0 = hexkl_probe_now();
  hvx_worker_pool_run(s->quant_pool, dma_probe_worker, &c, workers);
  const uint64_t t1 = hexkl_probe_now();

  uint32_t sum = 0;
  for (uint32_t i = 0; i < sum_bytes; i += 64u) {
    sum += c.vtcm[i];
  }
  /* Only descriptors whose worker actually ran moved bytes: the pool caps
     n_threads at its worker count + 1, and a request above that leaves
     the higher-numbered slices idle. Counting them would inflate GB/s. */
  uint32_t n_moved = 0;
  for (uint32_t k = 0; k < c.n_desc; ++k) {
    if (g_plan[k].worker < c.workers_used) {
      ++n_moved;
    }
  }
  const uint64_t total =
    (uint64_t)n_moved * row_size * nrows * (uint64_t)passes;
  res[0] = (uint32)(t1 - t0);
  res[1] = (uint32)(total & 0xFFFFFFFFu);
  res[2] = (uint32)(total >> 32);
  res[3] = sum;
  res[4] = c.workers_used;
  res[5] = DMA_PROBE_VOTE_STATE;
  res[6] = c.n_desc;
  res[7] = max_desc_bytes;
  res[8] = c.vtcm_per_worker;
  res[9] = passes;
  return AEE_SUCCESS;
}
