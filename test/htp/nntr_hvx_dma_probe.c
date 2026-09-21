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
 *
 * dma_replay (#87, below) drives the MoE layer call's own M=1 descriptor
 * list (test/htp/nntr_moe_dma_plan.h) through the production ring or
 * through per-worker chains, with no compute between the pushes.
 */

#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#include <AEEStdErr.h>
#include <HAP_farf.h>
#include <HAP_perf.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <remote.h>

#include "hexkl_dma_ring.h"
#include "hexkl_dma_trace.h"
#include "hexkl_probe.h"
#include "hvx_worker_pool.h"
#include "nntr_dma_probe_plan.h"
#include "nntr_hvx.h"
#include "nntr_hvx_session.h"
#include "nntr_moe_dma_plan.h"

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

/* ======================================================================
 * [#87] dma_replay: the M=1 MoE chunk schedule with no compute between
 * ====================================================================== */

#define REPLAY_MAX_ITEMS NNTR_MOE_DMA_PLAN_MAX
#define REPLAY_WORDS 8u
#define REPLAY_MAX_REGIONS 32u
#define REPLAY_LOAD_DDR_BYTES (4u << 20)
#define REPLAY_LOAD_VTCM_BYTES (1u << 20)
#define REPLAY_LOAD_UNITS 4096u

typedef struct {
  uint32_t op, kind, expert, src_off, dst_off, row_size, nrows, src_stride,
    t_rel;
} replay_item;

static replay_item g_items[REPLAY_MAX_ITEMS];
static uint32_t g_item_ord[REPLAY_MAX_ITEMS]; /**< push ordinal of a push */
static uint8_t g_load_done[REPLAY_LOAD_UNITS];

typedef struct {
  const uint8_t *arena;
  uint32_t region_bytes, n_regions, n_experts;
  uint8_t *vtcm;
  uint32_t n_items, workers, pace, fresh, call;
  uint64_t t_call0; /**< ticks: the pace clock's zero for this call */
  /* per worker, summed by the caller after the call */
  uint32_t wait_ticks[8], n_blocked[8], depth_max[8];
  uint32_t workers_used;
} replay_ctx;

typedef struct {
  uint8_t *buf;
  uint32_t bytes;
  _Atomic int stop;
  _Atomic uint32_t units;
} replay_load_ctx;

static inline void replay_spin_until(uint64_t t0, uint32_t t_rel) {
  while ((uint32_t)(HAP_perf_get_qtimer_count() - t0) < t_rel) {
  }
}

static const uint8_t *replay_src(const replay_ctx *c, const replay_item *it) {
  if (it->kind == NNTR_MOE_DMA_KIND_ACT || it->kind == NNTR_MOE_DMA_KIND_COPY) {
    return c->arena + (size_t)c->n_regions * c->region_bytes + it->src_off;
  }
  const uint32_t region =
    c->fresh ? (c->call * c->n_experts + it->expert) % c->n_regions
             : it->expert;
  return c->arena + (size_t)region * c->region_bytes + it->src_off;
}

/** @brief workers == 1: the production ring and the production trace. */
static void replay_ring(replay_ctx *c, hexkl_dma_trace_summary *sum) {
  static uint32_t ring_idx[REPLAY_MAX_ITEMS];
  hexkl_dma_ring_reset();
  hexkl_dma_trace_reset(HAP_perf_get_qtimer_count());
  for (uint32_t k = 0; k < c->n_items; ++k) {
    const replay_item *it = &g_items[k];
    if (c->pace) {
      replay_spin_until(c->t_call0, it->t_rel);
    }
    if (it->op == NNTR_MOE_DMA_OP_PUSH) {
      ring_idx[k] = hexkl_dma_ring_next_idx();
      hexkl_dma_ring_push2d(c->vtcm + it->dst_off, replay_src(c, it),
                            it->row_size, it->src_stride, it->row_size,
                            it->nrows, 0, 1);
      hexkl_dma_trace_push(HAP_perf_get_qtimer_count(), ring_idx[k], it->kind,
                           it->expert, 0u, it->row_size, it->nrows,
                           it->src_stride);
    } else {
      const uint32_t idx = ring_idx[it->src_off];
      hexkl_dma_trace_wait_begin(HAP_perf_get_qtimer_count(), idx, it->kind);
      hexkl_dma_ring_wait(idx);
      hexkl_dma_trace_wait_end(HAP_perf_get_qtimer_count());
    }
  }
  hexkl_dma_ring_drain();
  hexkl_dma_trace_finish(HAP_perf_get_qtimer_count(), sum);
  c->workers_used = 1u;
}

/** @brief workers > 1: this thread's pushes on its own dmstart chain, its
 *  own waits; depth and blocked bookkeeping per thread. */
static void replay_worker(uint32_t n_threads, uint32_t i, void *v) {
  replay_ctx *c = (replay_ctx *)v;
  if (i == 0u) {
    c->workers_used = n_threads;
  }
  hexkl_dma_desc2d *prev = NULL;
  uint32_t issued = 0u, seen = 0u, own[REPLAY_MAX_ITEMS], depth_max = 0u,
           wait_ticks = 0u, n_blocked = 0u;
  for (uint32_t k = 0; k < c->n_items; ++k) {
    const replay_item *it = &g_items[k];
    /* Owner by push ordinal modulo the threads the pool actually gave,
       not the count asked for: a capped pool must still issue everything. */
    if (it->op == NNTR_MOE_DMA_OP_PUSH) {
      if (g_item_ord[k] % n_threads != i) {
        continue;
      }
      if (c->pace) {
        replay_spin_until(c->t_call0, it->t_rel);
      }
      hexkl_dma_desc2d *d = &g_desc[k];
      memset(d, 0, sizeof(*d));
      d->desc_size = 1;
      d->desc_type = 9;
      d->dst_bypass = 1;
      d->src = (void *)replay_src(c, it);
      d->dst = c->vtcm + it->dst_off;
      d->src_stride = it->src_stride;
      d->dst_stride = it->row_size;
      d->row_size = it->row_size;
      d->nrows_lo = it->nrows & 0xffu;
      d->nrows_hi = (it->nrows >> 8) & 0xffu;
      Q6_dccleaninva_A((void *)d);
      /* Outstanding before this push, own chain only. */
      while (seen < issued && g_desc[own[seen]].done) {
        ++seen;
      }
      if (issued - seen + 1u > depth_max) {
        depth_max = issued - seen + 1u;
      }
      if (prev == NULL) {
        hexkl_dma_start(d);
      } else {
        hexkl_dma_link(prev, d);
      }
      prev = d;
      own[issued++] = k;
    } else {
      if (g_item_ord[it->src_off] % n_threads != i) {
        continue;
      }
      if (c->pace) {
        replay_spin_until(c->t_call0, it->t_rel);
      }
      hexkl_dma_desc2d *d = &g_desc[it->src_off];
      const uint64_t t_in = HAP_perf_get_qtimer_count();
      if (!d->done) {
        ++n_blocked;
      }
      long guard = 0;
      while (!d->done && guard++ < 50000000L) {
        hexkl_dma_poll();
      }
      wait_ticks += (uint32_t)(HAP_perf_get_qtimer_count() - t_in);
    }
  }
  /* Nothing may still be moving into VTCM when the next call rewrites
     the descriptors. */
  if (prev != NULL) {
    long guard = 0;
    while (!prev->done && guard++ < 50000000L) {
      hexkl_dma_poll();
    }
  }
  if (i < 8u) {
    c->wait_ticks[i] = wait_ticks;
    c->n_blocked[i] = n_blocked;
    c->depth_max[i] = depth_max;
  }
}

/** @brief One background unit: one read-modify-write pass over the load
 *  buffer with HVX, unless the replay has finished. */
static void replay_load_unit(uint32_t n_units, uint32_t u, void *v) {
  replay_load_ctx *l = (replay_load_ctx *)v;
  (void)n_units;
  (void)u;
  if (atomic_load(&l->stop)) {
    return;
  }
  HVX_Vector *p = (HVX_Vector *)l->buf;
  const uint32_t n = l->bytes / sizeof(HVX_Vector);
  const HVX_Vector one = Q6_V_vsplat_R(1);
  for (uint32_t k = 0; k < n; ++k) {
    p[k] = Q6_Vw_vadd_VwVw(p[k], one);
  }
  atomic_fetch_add(&l->units, 1u);
}

int nntr_hvx_dma_replay(remote_handle64 handle, uint32 arena,
                        uint32 region_bytes, const uint32 *schedule,
                        int scheduleLen, uint32 workers, uint32 load,
                        uint32 pace, uint32 fresh, uint32 gap_us, uint32 calls,
                        uint32 *res, int resLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s || !res || resLen < 12 || !schedule || scheduleLen <= 0 ||
      (scheduleLen % (int)REPLAY_WORDS) != 0 || workers == 0u || workers > 8u ||
      calls == 0u || region_bytes == 0u || (load != 0u && workers != 1u)) {
    return AEE_EBADPARM;
  }
  memset(res, 0, (size_t)resLen * sizeof(*res));
  if (arena >= NNTR_HVX_MAX_ARENAS || s->arenas[arena].va == NULL) {
    return AEE_EBADPARM;
  }
  replay_ctx c;
  memset(&c, 0, sizeof(c));
  c.arena = s->arenas[arena].va;
  c.region_bytes = region_bytes;
  /* One region-sized slot past the rotation serves the activation and
     copy pieces. */
  const uint32_t slots = s->arenas[arena].bytes / region_bytes;
  if (slots < 2u) {
    return AEE_EBADPARM;
  }
  c.n_regions =
    slots - 1u < REPLAY_MAX_REGIONS ? slots - 1u : REPLAY_MAX_REGIONS;
  c.vtcm = s->vtcm_base;
  c.n_items = (uint32_t)scheduleLen / REPLAY_WORDS;
  if (c.n_items > REPLAY_MAX_ITEMS) {
    return AEE_EBADPARM;
  }
  c.workers = workers;
  c.pace = pace;
  c.fresh = fresh;
  const uint32_t vtcm_limit =
    load == 2u ? s->config_off - REPLAY_LOAD_VTCM_BYTES : s->config_off;
  uint32_t bytes_per_call = 0u, gu_lo = UINT32_MAX, gu_hi = 0u, n_push = 0u;
  for (uint32_t k = 0; k < c.n_items; ++k) {
    const uint32 *w = schedule + k * REPLAY_WORDS;
    replay_item *it = &g_items[k];
    it->op = w[0] >> 8;
    it->kind = w[0] & 0xffu;
    it->expert = w[1];
    it->src_off = w[2];
    it->dst_off = w[3];
    it->row_size = w[4];
    it->nrows = w[5];
    it->src_stride = w[6];
    it->t_rel = w[7];
    if (it->op == NNTR_MOE_DMA_OP_PUSH) {
      const uint32_t bytes = it->row_size * it->nrows;
      const int weight = it->kind == NNTR_MOE_DMA_KIND_GATE ||
                         it->kind == NNTR_MOE_DMA_KIND_UP ||
                         it->kind == NNTR_MOE_DMA_KIND_DOWN;
      const uint64_t src_end = (uint64_t)it->src_off +
                               (uint64_t)(it->nrows - 1u) * it->src_stride +
                               it->row_size;
      if (it->row_size == 0u || it->nrows == 0u ||
          it->src_stride < it->row_size || bytes > (1u << 20) ||
          (uint64_t)it->dst_off + bytes > vtcm_limit ||
          src_end > region_bytes || (weight && it->expert >= c.n_regions) ||
          (weight && it->expert >= 8u)) {
        FARF(ERROR, "dma_replay: item %u out of range", (unsigned)k);
        return AEE_EBADPARM;
      }
      if (weight && it->expert + 1u > c.n_experts) {
        c.n_experts = it->expert + 1u;
      }
      if (it->kind == NNTR_MOE_DMA_KIND_GATE ||
          it->kind == NNTR_MOE_DMA_KIND_UP) {
        if (it->dst_off < gu_lo) {
          gu_lo = it->dst_off;
        }
        if (it->dst_off + bytes > gu_hi) {
          gu_hi = it->dst_off + bytes;
        }
      }
      bytes_per_call += bytes;
      g_item_ord[k] = n_push++;
    } else if (it->op == NNTR_MOE_DMA_OP_WAIT) {
      if (it->src_off >= k || g_items[it->src_off].op != NNTR_MOE_DMA_OP_PUSH) {
        FARF(ERROR, "dma_replay: wait %u on a non-push", (unsigned)k);
        return AEE_EBADPARM;
      }
    } else {
      return AEE_EBADPARM;
    }
  }
  if (n_push == 0u || c.n_experts == 0u) {
    return AEE_EBADPARM;
  }
  if (gu_lo != UINT32_MAX) {
    memset(c.vtcm + gu_lo, 0, gu_hi - gu_lo);
  }

  /* The optional load: the pool's workers claim units while the caller
     runs the replay; the stop flag turns the rest of the units into
     no-ops so wait_bg returns promptly. Heap only for the DDR variant,
     freed before return. */
  replay_load_ctx l;
  hvx_bg_job job;
  memset(&l, 0, sizeof(l));
  memset(&job, 0, sizeof(job));
  if (load == 1u) {
    l.buf = (uint8_t *)memalign(128, REPLAY_LOAD_DDR_BYTES);
    if (l.buf == NULL) {
      return AEE_ENOMEMORY;
    }
    l.bytes = REPLAY_LOAD_DDR_BYTES;
  } else if (load == 2u) {
    l.buf = c.vtcm + vtcm_limit;
    l.bytes = REPLAY_LOAD_VTCM_BYTES;
  }
  if (load != 0u) {
    memset(l.buf, 0, l.bytes);
    memset(g_load_done, 0, sizeof(g_load_done));
    job.func = replay_load_unit;
    job.ctx = &l;
    job.n_units = REPLAY_LOAD_UNITS;
    job.done = g_load_done;
    hvx_worker_pool_submit_bg(s->quant_pool, &job);
  }

  uint64_t wait_ticks = 0, busy_lo = 0, busy_hi = 0;
  uint32_t n_blocked = 0, depth_max = 0;
  const uint64_t t_start = HAP_perf_get_qtimer_count();
  for (c.call = 0; c.call < calls; ++c.call) {
    if (gap_us != 0u && c.call != 0u) {
      const uint64_t g0 = hexkl_probe_now();
      while (hexkl_probe_now() - g0 < gap_us) {
      }
    }
    c.t_call0 = HAP_perf_get_qtimer_count();
    if (workers == 1u) {
      hexkl_dma_trace_summary sum;
      replay_ring(&c, &sum);
      wait_ticks += sum.wait;
      n_blocked += sum.n_blocked;
      busy_lo += sum.busy_lo;
      busy_hi += sum.busy_hi;
      if (sum.depth_max > depth_max) {
        depth_max = sum.depth_max;
      }
    } else {
      hvx_worker_pool_run(s->quant_pool, replay_worker, &c, workers);
      for (uint32_t i = 0; i < 8u; ++i) {
        wait_ticks += c.wait_ticks[i];
        n_blocked += c.n_blocked[i];
        if (c.depth_max[i] > depth_max) {
          depth_max = c.depth_max[i];
        }
      }
    }
  }
  const uint64_t t_end = HAP_perf_get_qtimer_count();
  if (load != 0u) {
    atomic_store(&l.stop, 1);
    hvx_worker_pool_wait_bg(s->quant_pool, &job, UINT32_MAX);
    if (load == 1u) {
      free(l.buf);
    }
  }

  uint32_t sum = 0;
  if (gu_lo != UINT32_MAX) {
    for (uint32_t i = gu_lo; i < gu_hi; i += 64u) {
      sum += c.vtcm[i];
    }
  }
  res[0] = (uint32)HAP_perf_qtimer_count_to_us(t_end - t_start);
  res[1] = calls;
  res[2] = bytes_per_call;
  res[3] = (uint32)HAP_perf_qtimer_count_to_us(wait_ticks);
  res[4] = n_blocked;
  res[5] = depth_max;
  res[6] = sum;
  res[7] = c.n_regions;
  res[8] = c.workers_used;
  res[9] = atomic_load(&l.units);
  res[10] = (uint32)HAP_perf_qtimer_count_to_us(busy_lo);
  res[11] = (uint32)HAP_perf_qtimer_count_to_us(busy_hi);
  return AEE_SUCCESS;
}
