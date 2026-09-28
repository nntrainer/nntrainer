// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hexkl_dma_trace.c
 * @date   21 Sep 2026
 * @brief  Per-descriptor bookkeeping for one MoE layer call's DMA ring use
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * The one fact the arithmetic rests on: the ring's descriptors are
 * dmlinked, so they retire in push order. The completion watermark
 * therefore walks the push table from the oldest unseen record and stops
 * at the first whose done bit still reads 0 -- one is_done load per
 * completed descriptor over the whole call, plus one miss per point.
 */

#include "hexkl_dma_trace.h"

#include <string.h>

#include "hexkl_dma_ring.h"

static hexkl_dma_trace g_trace;

static uint32_t rel_(uint64_t now) { return (uint32_t)(now - g_trace.t0); }

/** @brief Marks every record whose descriptor has completed since the last
 *  point: done between then (lo) and now (hi). */
static void advance_(uint32_t now) {
  const uint32_t total = g_trace.n_push + g_trace.n_dropped;
  while (g_trace.seen_done < g_trace.n_push) {
    hexkl_dma_trace_push_rec *p = &g_trace.push[g_trace.seen_done];
    const int reclaimed = total - g_trace.seen_done >= HEXKL_DMA_TRACE_RING_N;
    if (!reclaimed && !hexkl_dma_ring_is_done(p->ring_idx)) {
      break;
    }
    p->t_done_lo = g_trace.t_last;
    p->t_done_hi = now;
    ++g_trace.seen_done;
  }
  g_trace.t_last = now;
}

void hexkl_dma_trace_reset(uint64_t now) {
  memset(&g_trace, 0, sizeof(g_trace));
  g_trace.t0 = now;
}

void hexkl_dma_trace_push(uint64_t now, uint32_t ring_idx, uint32_t kind,
                          uint32_t expert, uint32_t chunk, uint32_t row_size,
                          uint32_t nrows, uint32_t src_stride) {
  const uint32_t t = rel_(now);
  advance_(t);
  if (g_trace.n_push >= HEXKL_DMA_TRACE_MAX_PUSH) {
    ++g_trace.n_dropped;
    return;
  }
  const uint32_t depth = g_trace.n_push - g_trace.seen_done;
  hexkl_dma_trace_push_rec *p = &g_trace.push[g_trace.n_push++];
  p->t_issue = t;
  p->kind = kind;
  p->expert = expert;
  p->chunk = chunk;
  p->bytes = row_size * nrows;
  p->row_size = row_size;
  p->nrows = nrows;
  p->src_stride = src_stride;
  p->ring_idx = ring_idx;
  p->depth = depth;
  p->t_done_lo = 0;
  p->t_done_hi = 0;
  if (depth + 1u > g_trace.depth_max) {
    g_trace.depth_max = depth + 1u;
  }
}

void hexkl_dma_trace_wait_begin(uint64_t now, uint32_t ring_idx,
                                uint32_t site) {
  const uint32_t t = rel_(now);
  /* Read the bit before the watermark moves so "blocked" is what the wait
     itself is about to see, not what the walk found. */
  const uint32_t blocked = hexkl_dma_ring_is_done(ring_idx) ? 0u : 1u;
  advance_(t);
  g_trace.n_blocked += blocked;
  ++g_trace.n_wait_total;
  if (g_trace.n_wait >= HEXKL_DMA_TRACE_MAX_WAIT) {
    g_trace.cur_wait = 0u;
    return;
  }
  g_trace.cur_wait = g_trace.n_wait + 1u;
  hexkl_dma_trace_wait_rec *w = &g_trace.wait[g_trace.n_wait++];
  w->t_in = t;
  w->t_out = t;
  w->ring_idx = ring_idx;
  w->site = site;
  w->blocked = blocked;
}

void hexkl_dma_trace_wait_end(uint64_t now) {
  const uint32_t t = rel_(now);
  advance_(t);
  if (g_trace.cur_wait != 0u) {
    g_trace.wait[g_trace.cur_wait - 1u].t_out = t;
    g_trace.cur_wait = 0u;
  }
}

void hexkl_dma_trace_sample(uint64_t now) { advance_(rel_(now)); }

/** @brief Length of the union of [t_issue, t_issue + len(i)] over the push
 *  table, which is sorted by t_issue already (pushes are appended in
 *  time order). @a hi selects the t_done_hi bracket edge over t_done_lo. */
static uint32_t union_(int hi) {
  uint32_t total = 0, cur_lo = 0, cur_hi = 0;
  int open = 0;
  for (uint32_t k = 0; k < g_trace.n_push; ++k) {
    const hexkl_dma_trace_push_rec *p = &g_trace.push[k];
    const uint32_t a = p->t_issue;
    uint32_t b = hi ? p->t_done_hi : p->t_done_lo;
    if (b < a) {
      b = a; /* lo edge before the issue: the point before this push */
    }
    if (open && a <= cur_hi) {
      if (b > cur_hi) {
        cur_hi = b;
      }
      continue;
    }
    if (open) {
      total += cur_hi - cur_lo;
    }
    cur_lo = a;
    cur_hi = b;
    open = 1;
  }
  if (open) {
    total += cur_hi - cur_lo;
  }
  return total;
}

void hexkl_dma_trace_finish(uint64_t now, hexkl_dma_trace_summary *out) {
  const uint32_t t = rel_(now);
  advance_(t);
  /* Whatever is still open completed no earlier than the last point and no
     later than... unknown; the call is over, so close it at now. */
  for (uint32_t k = g_trace.seen_done; k < g_trace.n_push; ++k) {
    g_trace.push[k].t_done_lo = g_trace.t_last;
    g_trace.push[k].t_done_hi = t;
  }
  g_trace.t_end = t;
  if (!out) {
    return;
  }
  memset(out, 0, sizeof(*out));
  out->n_desc = g_trace.n_push + g_trace.n_dropped;
  out->n_wait = g_trace.n_wait_total;
  out->n_blocked = g_trace.n_blocked;
  out->depth_max = g_trace.depth_max;
  for (uint32_t k = 0; k < g_trace.n_wait; ++k) {
    const hexkl_dma_trace_wait_rec *w = &g_trace.wait[k];
    out->wait += w->t_out - w->t_in;
    if (w->site == HEXKL_DMA_SITE_ACT) {
      out->wait_act += w->t_out - w->t_in;
    }
  }
  if (g_trace.n_dropped != 0u) {
    /* The union over a truncated table is not the call's busy time and
       dividing the call's bytes by it would overstate the engine. */
    return;
  }
  out->busy_lo = union_(0);
  out->busy_hi = union_(1);
  for (uint32_t k = 0; k < g_trace.n_push; ++k) {
    const hexkl_dma_trace_push_rec *p = &g_trace.push[k];
    const int weight = p->kind == HEXKL_DMA_KIND_GATE ||
                       p->kind == HEXKL_DMA_KIND_UP ||
                       p->kind == HEXKL_DMA_KIND_DOWN;
    if (!weight) {
      continue;
    }
    if (p->t_issue > out->last_issue) {
      out->last_issue = p->t_issue;
    }
    if (p->expert == 0u && p->kind != HEXKL_DMA_KIND_DOWN &&
        p->t_done_hi > out->first_ready) {
      out->first_ready = p->t_done_hi;
    }
  }
}

const hexkl_dma_trace *hexkl_dma_trace_get(void) { return &g_trace; }

uint32_t hexkl_dma_trace_serialize(uint32_t *words, uint32_t max_words) {
  const uint32_t need = HEXKL_DMA_TRACE_HDR_WORDS +
                        g_trace.n_push * HEXKL_DMA_TRACE_PUSH_WORDS +
                        g_trace.n_wait * HEXKL_DMA_TRACE_WAIT_WORDS;
  if (!words || max_words < need) {
    return 0;
  }
  uint32_t *w = words;
  *w++ = g_trace.n_push;
  *w++ = g_trace.n_wait;
  *w++ = g_trace.n_blocked;
  *w++ = g_trace.depth_max;
  *w++ = g_trace.n_dropped;
  *w++ = g_trace.t_end;
  *w++ = HEXKL_DMA_TRACE_PUSH_WORDS;
  *w++ = HEXKL_DMA_TRACE_WAIT_WORDS;
  for (uint32_t k = 0; k < g_trace.n_push; ++k) {
    const hexkl_dma_trace_push_rec *p = &g_trace.push[k];
    const uint32_t rec[HEXKL_DMA_TRACE_PUSH_WORDS] = {
      p->t_issue,  p->kind,     p->expert,    p->chunk,
      p->bytes,    p->row_size, p->nrows,     p->src_stride,
      p->ring_idx, p->depth,    p->t_done_lo, p->t_done_hi};
    memcpy(w, rec, sizeof(rec));
    w += HEXKL_DMA_TRACE_PUSH_WORDS;
  }
  for (uint32_t k = 0; k < g_trace.n_wait; ++k) {
    const hexkl_dma_trace_wait_rec *r = &g_trace.wait[k];
    const uint32_t rec[HEXKL_DMA_TRACE_WAIT_WORDS] = {
      r->t_in, r->t_out, r->ring_idx, r->site, r->blocked};
    memcpy(w, rec, sizeof(rec));
    w += HEXKL_DMA_TRACE_WAIT_WORDS;
  }
  return need;
}
