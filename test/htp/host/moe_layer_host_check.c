/* Host harness for hexkl_mm_u8i4_moe_layer_run: scalar stand-ins for every
   primitive it calls, then the kernel against a straightforward reference.
   The point is the loop structure -- which rows each expert gets, which
   weights it uses, whether the buffer reuse clobbers anything, whether the
   scatter lands on the right output row with the right routing weight --
   not HMX's arithmetic, which the stubs define self-consistently for both
   sides. */
#include "hexkl_acc_tile.h"
#include "hexkl_dma_ring.h"
#include "hexkl_dma_trace.h"
#include "hexkl_mm_u8i4_moe.h"
#include "hexkl_probe.h"
#include "hvx_dequant_i32.h"
#include "hvx_expand_i2i4.h"
#include "hvx_gather_ah_u8.h"
#include "hvx_gemm_u8i4_wh.h"
#include "hvx_scalar.h"
#include "hvx_scale_add_f32.h"
#include <AEEStdErr.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fc_wh_det.h"
#include "nntr_moe_dma_plan.h"
#include "swiglu_det.h"

/* ---- acc layout: the tile stand-in (standin/hvx_scalar.c) is a 64 x 32
   int32 tile at row stride 32, base 0 ---- */
static hexkl_acc_layout g_layout = {1, 1, 0,
                                    32}; /* probed, usable, base, stride */
const hexkl_acc_layout *hexkl_acc_layout_get(uint8_t *b, uint32_t off) {
  (void)b;
  (void)off;
  return &g_layout;
}
/* Every GEMV call's int32 result, logged so the M=1 check can hold the
   tiles the kernel fed its epilogues against a reference computed from the
   token rows the routing named -- which is what the kernel's slot pack and
   row bookkeeping are for. Rows up to the GEMV's 4-row group. */
typedef struct {
  const uint8_t *wh;
  uint32_t nt, m;
  int32_t tile[4 * 32];
} gemv_log_entry;
static gemv_log_entry *g_gemv_log;
static size_t g_gemv_n, g_gemv_cap;
static void gemv_log_reset(void) { g_gemv_n = 0; }

/* The HVX GEMM's stand-in: the same sum, over the tiles the kernel points
   it at, into the row-stride-32 tile the header promises. */
/* ---- #117: the M=1 feed's scoreboard ------------------------------------
   Every push is recorded (destination range, source, ring index) and every
   ring wait marks the pushes it retires (in issue order, as the dmlinked
   chain does). A GEMV column read from VTCM must then find the latest push
   covering its address WAITED and long enough for the column, a push onto
   a slab must find the slab's previous matrix fully read (n_col columns,
   once each: the pool run that read it has joined), and every push must
   land inside the arena. The DMA stand-in still copies at once, so a read
   before its wait computes the right bytes here and stale ones on device;
   the scoreboard is what turns that into a failure without a phone. On
   only inside run_m1_case with the feed set (g_score_on): the HMX path's
   chunked pushes are read by the HMX stand-in, not by columns. */
static uint32_t g_lane; /* set by the pool stand-in below */
/* The device pool's lanes: 5 workers + the caller. */
#define PF_LANES 6u
#define SCORE_MAX 256u
/* #177: a push may also be one lane's slice of a matrix (lane >= 0), issued
   on that lane's own queue inside pool run `run`, with its descriptor. */
typedef struct {
  const uint8_t *dst, *src;
  uint32_t bytes, row_size, nrows, idx, reads, waited;
  int lane; /* -1: the ring */
  uint32_t run;
  const void *desc;
} score_push;
static score_push g_score[SCORE_MAX];
static uint32_t g_score_n, g_score_on, g_score_bad, g_score_waits,
  g_score_vtcm_reads, g_lane_waits;
/* Bumped by the pool stand-in at every run: a slice must be waited by the
   lane that issued it inside the same run, and read only in a later one. */
static uint32_t g_run;
/* Set only by the timeout case in run_m1_cases. */
static int g_lane_timeout;
/* [#225] The FC cells: a lane reads the block it pushed and waited inside
   the same run (its own double buffer); another lane's slice is still a
   race. */
static int g_own_slice_ok;
static const uint8_t *g_vtcm_lo, *g_vtcm_hi;
/* #185: the pool stand-ins' state. g_workers is the pool's worker count
   (the caller is not one): a submitted job runs on min(n, g_workers) lanes,
   a run on min(n, g_workers + 1). A submitted job is pending until the
   wait; g_in_run is set while a run's or a job's lanes execute. */
static uint32_t g_workers = PF_LANES - 1u, g_pending, g_in_run, g_submits;
/* #185: the dataflow scoreboard. The stand-ins report every buffer they
   read or write (hvx_scalar_hook.buf); inside one run or job, a read and a
   write -- or two writes -- of overlapping bytes from two different lanes
   are a race on the device, where the lanes run at once, and a silent pass
   here, where they run one after another. Same-lane accesses of one kind
   are merged, so a run's table stays a few entries per lane. */
typedef struct {
  const uint8_t *lo, *hi;
  uint32_t lane;
  int w;
} df_access;
#define DF_MAX 1024u
static df_access g_df[DF_MAX];
static uint32_t g_df_n, g_df_bad;
static uint64_t g_df_seen;
static void score_reset(int on, const uint8_t *vtcm, size_t vtcm_bytes) {
  g_score_n = g_score_bad = g_score_waits = g_score_vtcm_reads = 0;
  g_lane_waits = 0;
  g_pending = g_in_run = g_df_n = 0;
  g_score_on = on ? 1u : 0u;
  g_vtcm_lo = vtcm;
  g_vtcm_hi = vtcm + vtcm_bytes;
}
static int in_vtcm(const uint8_t *p) { return p >= g_vtcm_lo && p < g_vtcm_hi; }
/* The latest push whose destination range holds @a p, or NULL. */
static score_push *score_find(const uint8_t *p) {
  for (uint32_t k = g_score_n; k-- > 0u;) {
    score_push *sp = &g_score[k];
    if (p >= sp->dst && p < sp->dst + sp->bytes)
      return sp;
  }
  return NULL;
}

/* @a table: NULL for a 4-bit column (512-byte tiles), the expansion table
   of a 2-bit one (256-byte tiles of QS2CX_WH codes, plan 229). */
static void gemv_stand_in(const uint8_t *act_ah, uint32_t m, uint32_t k_tiles,
                          const uint8_t *wh, uint32_t n_col, uint32_t nt,
                          const uint8_t *table, int32_t *out) {
  const uint32_t tb = table ? 256u : 512u;
  /* rows1 picks between two HVX loops that compute the same int32 sums
     (hvx_gemm_u8i4_wh.c); the stand-in is that sum, so it is the same
     function either way and the caller only records which loop was
     asked for. */
  const uint8_t *src_wh = wh;
  if (g_score_on && in_vtcm(wh)) {
    /* Every k-tile row of the column must sit in a waited push of this
       matrix's shape: on the ring one whole matrix (f2), under #177 the
       slices of one, each waited in an earlier pool run than this read.
       Each push the column touches counts one read. */
    const uint32_t row = n_col * tb;
    const uint8_t *origin = NULL;
    const char *why = NULL;
    score_push *prev = NULL;
    for (uint32_t kt = 0; kt < k_tiles && !why; ++kt) {
      const uint8_t *a = wh + (size_t)kt * row + (size_t)nt * tb;
      score_push *sp = score_find(a);
      if (!sp)
        why = "no push";
      else if (!sp->waited)
        why = "push not waited";
      else if (sp->lane >= 0 && sp->run >= g_run &&
               !(g_own_slice_ok && sp->lane == (int)g_lane))
        why = "slice read in the run that issued it";
      else if (sp->row_size != row ||
               (size_t)(a - sp->dst) % row != (size_t)nt * tb ||
               (sp->lane < 0 && (sp->dst != wh || sp->nrows != k_tiles)))
        why = "shape";
      else if (origin && origin != sp->src - (sp->dst - wh))
        why = "rows from two matrices";
      else {
        origin = sp->src - (sp->dst - wh);
        if (sp != prev)
          sp->reads += tb / 256u; /* 256 B units: 2 a 4-bit column, 1 a 2-bit */
        prev = sp;
      }
    }
    if (why) {
      if (g_score_bad++ < 4u)
        printf("FEED read not covered: lane=%u nt=%u n_col=%u %s\n", g_lane, nt,
               n_col, why);
    } else {
      ++g_score_vtcm_reads;
      src_wh = origin; /* the log names the expert by its arena bytes */
    }
  }
  if (table)
    hvx_scalar_gemv_i2(act_ah, m, k_tiles, wh, n_col, nt, table, out);
  else
    hvx_scalar_gemv(act_ah, m, k_tiles, wh, n_col, nt, out);
  if (g_gemv_n == g_gemv_cap) {
    g_gemv_cap = g_gemv_cap ? 2u * g_gemv_cap : 1024u;
    g_gemv_log =
      (gemv_log_entry *)realloc(g_gemv_log, g_gemv_cap * sizeof *g_gemv_log);
  }
  gemv_log_entry *le = &g_gemv_log[g_gemv_n++];
  le->wh = src_wh;
  le->nt = nt;
  le->m = m > 4u ? 4u : m;
  memcpy(le->tile, out, sizeof(int32_t) * 32u * le->m);
}

/* ---- The M=1 path's l2fetch lead (HVX_GEMV_PF_LEAD_KB), per lane -------
   The pool stand-in below sets g_lane. Each lane keeps its last PF_RING
   boxes (two blocks of stage A's gate + up pair) with a bit per column; a
   column computed without its own l2fetch must find its bit in one of them,
   unconsumed, from its own lane -- so it was covered, ahead of it, at most
   two blocks earlier, and fetched once. Every box must also sit inside its
   weight's columns, fit the l2fetch's 16-bit fields, and go out with at
   most two others still unread. */
#define PF_RING 4u
typedef struct {
  const uint8_t *wh;
  uint32_t n_col, k_tiles, nt0, n;
  uint64_t used; /* bit t: column nt0 + t consumed */
} pf_box;
static pf_box g_pf[PF_LANES][PF_RING];
static uint32_t g_pf_head[PF_LANES];
static uint64_t g_pf_cols, g_pf_used, g_pf_bad, g_nopf_n, g_col_n;
/* Bit 0: a column ran with rows1 = 0, bit 1: with rows1 = 1. The two knobs
   are independent only if what the call asked for is what every column
   got, whatever the lead (#113). */
static uint32_t g_rows1_seen;
static void pf_reset(void) {
  memset(g_pf, 0, sizeof g_pf);
  memset(g_pf_head, 0, sizeof g_pf_head);
  g_pf_cols = g_pf_used = g_pf_bad = g_nopf_n = g_col_n = 0;
  g_rows1_seen = 0;
}
static void hook_prefetch(const uint8_t *wh, uint32_t n_col, uint32_t nt,
                          uint32_t n_tiles, uint32_t k_tiles) {
  /* n_tiles > 64 is this stand-in's limit, not the kernel's (it clamps at
     127, the l2fetch width field's bound): the per-column bitmap below is
     one uint64_t. No shape reaches 64 units today -- the deepest swept
     block is 54 -- so widening it would be speculation; the message says
     which bound fired. */
  if (n_tiles == 0u || n_tiles > 64u || nt + n_tiles > n_col ||
      n_col * 512u > 0xFFFFu || n_tiles * 512u > 0xFFFFu || k_tiles > 0xFFFFu) {
    printf("PF box out of range%s: lane=%u nt=%u n=%u n_col=%u k_tiles=%u\n",
           n_tiles > 64u ? " (stand-in's 64-column bitmap, not the kernel)"
                         : "",
           g_lane, nt, n_tiles, n_col, k_tiles);
    ++g_pf_bad;
    return;
  }
  /* The hardware queues three l2fetch per thread and stalls it on a
     fourth. A box counts as outstanding until every column of it was read
     (a fetch finishes no later than the loads that need all of it). */
  uint32_t outstanding = 0;
  for (uint32_t k = 0; k < PF_RING; ++k) {
    const pf_box *o = &g_pf[g_lane][k];
    outstanding +=
      o->wh && o->used != ((o->n == 64u) ? ~0ull : ((1ull << o->n) - 1u));
  }
  if (outstanding >= 3u) {
    printf("PF fourth box outstanding: lane=%u nt=%u\n", g_lane, nt);
    ++g_pf_bad;
  }
  pf_box *b = &g_pf[g_lane][g_pf_head[g_lane]++ % PF_RING];
  /* A box leaving the ring with columns unconsumed was a fetch nothing
     read ahead of time: over-fetch, or a lead longer than two blocks. */
  if (b->wh && b->used != ((b->n == 64u) ? ~0ull : ((1ull << b->n) - 1u)))
    ++g_pf_bad;
  b->wh = wh;
  b->n_col = n_col;
  b->k_tiles = k_tiles;
  b->nt0 = nt;
  b->n = n_tiles;
  b->used = 0;
  g_pf_cols += n_tiles;
}
/* hvx_gemm_u8i4_wh_col (nopf = 0) and _col_nopf (nopf = 1), through the
   stand-in file's hook: the bookkeeping is this check's, the sum is the
   shared one. */
static void hook_gemv_any(const uint8_t *act_ah, uint32_t m, uint32_t k_tiles,
                          const uint8_t *wh, uint32_t n_col, uint32_t nt,
                          uint32_t rows1, int nopf, const uint8_t *table,
                          int32_t *out) {
  int found = 0;
  g_rows1_seen |= 1u << (rows1 != 0u);
  if (!nopf) {
    ++g_col_n;
    gemv_stand_in(act_ah, m, k_tiles, wh, n_col, nt, table, out);
    return;
  }
  if (g_score_on && in_vtcm(wh)) {
    /* Under the feed no box covers a column; the scoreboard in
       gemv_stand_in holds the read instead. */
    ++g_nopf_n;
    gemv_stand_in(act_ah, m, k_tiles, wh, n_col, nt, table, out);
    return;
  }
  for (uint32_t k = 0; k < PF_RING && !found; ++k) {
    pf_box *b = &g_pf[g_lane][k];
    if (b->wh == wh && b->n_col == n_col && b->k_tiles == k_tiles &&
        nt >= b->nt0 && nt < b->nt0 + b->n &&
        !(b->used & (1ull << (nt - b->nt0)))) {
      b->used |= 1ull << (nt - b->nt0);
      ++g_pf_used;
      found = 1;
    }
  }
  if (!found) {
    printf("PF column not covered: lane=%u nt=%u n_col=%u\n", g_lane, nt,
           n_col);
    ++g_pf_bad;
  }
  ++g_nopf_n;
  gemv_stand_in(act_ah, m, k_tiles, wh, n_col, nt, table, out);
}
static void hook_gemv(const uint8_t *act_ah, uint32_t m, uint32_t k_tiles,
                      const uint8_t *wh, uint32_t n_col, uint32_t nt,
                      uint32_t rows1, int nopf, int32_t *out) {
  hook_gemv_any(act_ah, m, k_tiles, wh, n_col, nt, rows1, nopf, NULL, out);
}
/* [plan 229] The u8i2 GEMV columns, through the same audit. */
static void hook_gemv2(const uint8_t *act_ah, uint32_t m, uint32_t k_tiles,
                       const uint8_t *wh, uint32_t n_col, uint32_t nt,
                       uint32_t rows1, int nopf, const uint8_t *table,
                       int32_t *out) {
  hook_gemv_any(act_ah, m, k_tiles, wh, n_col, nt, rows1, nopf, table, out);
}
/* #185: the stand-ins' buffer hook (the dataflow scoreboard above). Only
   on the feed cells and inside a run or job; the caller's own accesses
   between runs are ordered by the joins. */
static void hook_buf(const void *p, size_t bytes, int write) {
  const uint8_t *lo = (const uint8_t *)p, *hi = lo + bytes;
  if (!g_score_on || !g_in_run)
    return;
  ++g_df_seen;
  for (uint32_t k = 0; k < g_df_n; ++k) {
    const df_access *a = &g_df[k];
    if (a->lane != g_lane && (a->w || write) && lo < a->hi && a->lo < hi) {
      ++g_df_bad;
      if (g_score_bad++ < 4u)
        printf("FEED cross-lane RAW/WAR in run %u: lane %u %s +%zu B, lane %u "
               "%s it\n",
               g_run, g_lane, write ? "writes" : "reads", bytes, a->lane,
               a->w ? "writes" : "reads");
      return;
    }
  }
  for (uint32_t k = 0; k < g_df_n; ++k) {
    df_access *a = &g_df[k];
    if (a->lane == g_lane && a->w == write && lo <= a->hi && a->lo <= hi) {
      a->lo = lo < a->lo ? lo : a->lo;
      a->hi = hi > a->hi ? hi : a->hi;
      return;
    }
  }
  if (g_df_n == DF_MAX) {
    ++g_df_bad;
    if (g_score_bad++ < 4u)
      printf("FEED dataflow table full in run %u\n", g_run);
    return;
  }
  g_df[g_df_n++] = (df_access){lo, hi, g_lane, write};
}

/* ---- DMA ring: completes immediately, so a push issued while the
   destination is still live shows up as a wrong result. ---- */
void hexkl_dma_ring_reset(void) {}
/* Indices are handed out and waited on for real shape, but every transfer
   has already completed by the time push2d returns, so a wait is a no-op
   for the bytes. That is the point: a chunk consumed before its push would
   read stale bytes on device and read correct ones here, so what this
   harness checks is that every chunk IS pushed and that the offsets line
   up -- and, under the M=1 feed, the scoreboard above: a wait retires
   every push at or before its index, as the dmlinked chain does. */
static uint32_t g_stub_idx;
uint32_t hexkl_dma_ring_next_idx(void) { return g_stub_idx++; }
static void score_retire(uint32_t idx) {
  for (uint32_t k = 0; k < g_score_n; ++k)
    if (g_score[k].idx <= idx)
      g_score[k].waited = 1u;
}
void hexkl_dma_ring_wait(uint32_t idx) {
  ++g_score_waits;
  score_retire(idx);
}
void hexkl_dma_ring_drain(void) { score_retire(g_stub_idx); }
/* Every transfer is complete by the time push2d returns, so the trace's
   watermark sees each descriptor done at the very next point. */
int hexkl_dma_ring_is_done(uint32_t idx) {
  (void)idx;
  return 1;
}
/* [#158] Bytes pushed with src_bypass set, and pushes that set it on a
   heap-to-heap copy (dst_vtcm = 0), which must never happen: the DSP wrote
   those sources. The checks below hold the byte count against the weight
   bytes of the arena-backed experts, so a bypassed activation block or a
   bypassed heap weight shows up as a surplus. */
static uint64_t g_bypass_bytes, g_bypass_bad;
static void score_push2d(void *dst, const void *src, uint32_t ds, uint32_t ss,
                         uint32_t rs, uint32_t nrows, int sv, int dv, int lane,
                         const void *desc) {
  if (sv) {
    g_bypass_bytes += (uint64_t)rs * nrows;
    g_bypass_bad += dv == 0;
  }
  /* Only the VTCM-bound pushes are slab pushes; moe_dma_copy's heap-to-heap
     pieces (dst_vtcm = 0) are the two other descriptors of a call. */
  if (g_score_on && dv) {
    const uint8_t *d0 = (const uint8_t *)dst;
    const size_t bytes = (size_t)(nrows - 1u) * ds + rs;
    /* (iii) inside the arena; (ii) the slab's previous matrix was read
       whole -- its pool run has joined -- before this lands on it. */
    if (!in_vtcm(d0) || d0 + bytes > g_vtcm_hi) {
      if (g_score_bad++ < 4u)
        printf("FEED push outside the arena: %zu bytes at +%td\n", bytes,
               d0 - g_vtcm_lo);
    }
    for (uint32_t k = 0; k < g_score_n; ++k) {
      const score_push *sp = &g_score[k];
      if (d0 < sp->dst + sp->bytes && sp->dst < d0 + bytes &&
          sp->reads != sp->row_size / 256u) {
        if (g_score_bad++ < 4u)
          printf("FEED push onto a live slab: +%td (previous matrix %u of %u "
                 "256-byte column units read)\n",
                 d0 - g_vtcm_lo, sp->reads, sp->row_size / 256u);
        break;
      }
    }
    if (g_score_n < SCORE_MAX) {
      score_push *sp = &g_score[g_score_n++];
      sp->dst = d0;
      sp->src = (const uint8_t *)src;
      sp->bytes = (uint32_t)bytes;
      sp->row_size = rs;
      sp->nrows = nrows;
      sp->idx = lane < 0 ? g_stub_idx - 1u /* next_idx was taken before */
                         : ~0u;            /* never retired by the ring */
      sp->reads = 0u;
      sp->waited = 0u;
      sp->lane = lane;
      sp->run = g_run;
      sp->desc = desc;
    }
  }
  for (uint32_t r = 0; r < nrows; ++r)
    memcpy((uint8_t *)dst + (size_t)r * ds,
           (const uint8_t *)src + (size_t)r * ss, rs);
}
void hexkl_dma_ring_push2d(void *dst, const void *src, uint32_t ds, uint32_t ss,
                           uint32_t rs, uint32_t nrows, int sv, int dv) {
  score_push2d(dst, src, ds, ss, rs, nrows, sv, dv, -1, NULL);
}
/* #177's lane queue: the slice lands at once, as on the ring, and is
   recorded with the lane and run that issued it. A wait retires the
   descriptor it names and every earlier slice of the same lane and run
   (one dmlinked chain per lane); a wait on anything else is a bug. */
void hexkl_dma_lane_push2d(hexkl_dma_desc2d *d, hexkl_dma_desc2d *prev,
                           void *dst, const void *src, uint32_t ds, uint32_t ss,
                           uint32_t rs, uint32_t nrows, int sv, int dv) {
  (void)prev;
  score_push2d(dst, src, ds, ss, rs, nrows, sv, dv, (int)g_lane, d);
}
int hexkl_dma_lane_wait(hexkl_dma_desc2d *d) {
  uint32_t k = g_score_n;
  ++g_lane_waits;
  while (k-- > 0u) {
    const score_push *sp = &g_score[k];
    if (sp->desc == d && sp->lane == (int)g_lane && sp->run == g_run &&
        !sp->waited)
      break;
  }
  if (k == ~0u) {
    if (g_score_on && g_score_bad++ < 4u)
      printf("FEED lane %u waits on a descriptor it did not issue in run %u\n",
             g_lane, g_run);
    return 0;
  }
  for (uint32_t j = 0; j <= k; ++j)
    if (g_score[j].lane == (int)g_lane && g_score[j].run == g_run)
      g_score[j].waited = 1u;
  /* A wait that ran out of its guard must fail the call: one case injects
     it on lane 1 (g_lane_timeout) and expects AEE_EFAILED. */
  return (g_lane_timeout && g_lane == 1u) ? -1 : 0;
}
/* After a run's join: every slice it issued was waited inside it. */
static void score_run_end(void) {
  for (uint32_t k = 0; k < g_score_n; ++k)
    if (g_score[k].lane >= 0 && g_score[k].run == g_run && !g_score[k].waited) {
      if (g_score_bad++ < 4u)
        printf("FEED slice of lane %d not waited inside its run %u\n",
               g_score[k].lane, g_run);
    }
}

/* The pool runs every lane on the caller, one after another. Doing it here
   rather than passing NULL keeps the kernel's call sites exercised: the
   range arithmetic they hand the worker is part of what this check is
   for. On the feed cells a run or job started inside another, or while a
   submitted job is still pending (#185: the kernel must join it first,
   or the join is lost), is a schedule error. */
static void pool_lanes(hvx_worker_pool_func func, void *ctx, uint32_t n) {
  if (g_score_on && (g_in_run || g_pending) && g_score_bad++ < 4u)
    printf(g_in_run ? "FEED nested pool run\n"
                    : "FEED pool used while a submitted job is in flight\n");
  g_pending = 0; /* the real run and submit wait for it first */
  ++g_run;
  g_df_n = 0;
  g_in_run = 1;
  /* The degenerate branch is func(1, 0): writing it the way the real
     function's used to be -- func(n_units, 0, ctx) -- is what this check
     caught first time out; that form means "worker 0 of n_units" and does
     1/n_units of the work. */
  for (uint32_t i = 0; i < (n > 1u ? n : 1u); ++i) {
    g_lane = i;
    func(n > 1u ? n : 1u, i, ctx);
  }
  g_lane = 0;
  g_in_run = 0;
  score_run_end();
}
/* The device's lane count, n = min(n_units, workers + 1), so each lane's
   slice and its l2fetch boxes are checked as the device splits them. */
void hvx_worker_pool_run(hvx_worker_pool *pool, hvx_worker_pool_func func,
                         void *ctx, uint32_t n_units) {
  (void)pool;
  pool_lanes(func, ctx, n_units < g_workers + 1u ? n_units : g_workers + 1u);
}
/* submit: the workers only, n = min(n_units, workers) -- lane i is worker
   i + 1 on the device -- run to completion on the spot; the job then stays
   pending until the wait. With no workers it runs inline, as the real one
   does, and nothing is pending. The harness cannot exercise the overlap
   with the caller, only that the job is joined before the pool is used
   again and before its buffers are; the device tests check the timing. */
void hvx_worker_pool_submit(hvx_worker_pool *pool, hvx_worker_pool_func func,
                            void *ctx, uint32_t n_units) {
  (void)pool;
  if (n_units == 0u)
    return;
  pool_lanes(func, ctx,
             g_workers ? (n_units < g_workers ? n_units : g_workers) : 1u);
  g_pending = g_workers != 0u;
  ++g_submits;
}
void hvx_worker_pool_wait(hvx_worker_pool *pool) {
  (void)pool;
  g_pending = 0;
}
uint32_t hvx_worker_pool_workers(const hvx_worker_pool *pool) {
  (void)pool;
  return g_workers;
}
/* The background lane, likewise: every unit runs at submit, in order, and
   the waits find them done. What this checks is that the kernel waits for
   the right block before it queues it -- a wait for too few units cannot
   show here, and a wait for too many only as a hang on device. */
void hvx_worker_pool_submit_bg(hvx_worker_pool *pool, hvx_bg_job *job) {
  (void)pool;
  for (uint32_t u = 0; u < job->n_units; ++u) {
    job->func(job->n_units, u, job->ctx);
    job->done[u] = 1;
  }
}
void hvx_worker_pool_wait_bg(hvx_worker_pool *pool, hvx_bg_job *job,
                             uint32_t n) {
  (void)pool;
  (void)job;
  (void)n;
}

void hvx_copy_ah_block(uint8_t *dst, const uint8_t *src, uint32_t k,
                       hvx_worker_pool *pool) {
  (void)pool;
  memcpy(dst, src, (size_t)(k / 32u) * 2048u);
}

void hvx_scale_add_rows_f32(float *dst, const float *src, float scale,
                            uint32_t n) {
  /* Two operations, never one: HVX has no f32 fused multiply-add and the
     ARM path this has to agree with applies the routing weight with
     multiply_i and then accumulates with add_i. The volatile is what stops
     the host compiler contracting them and making this stub disagree with
     the kernel for a reason that is the stub's. */
  for (uint32_t i = 0; i < n; ++i) {
    volatile float p = src[i] * scale;
    dst[i] = dst[i] + p;
  }
}
uint64_t hexkl_probe_us[HEXKL_PROBE_N];
/* On, so the counting probes (blocks, DMA bytes) run here too -- this
   harness is where a miscounted block or a push that never happens shows up
   without a device. The timers read the stub clock and are not checked. */
int hexkl_probe_on = 1;

/* ---------------- reference: one expert, one row, at a time -------------- */
typedef struct {
  uint32_t K, N;
  int8_t *nib;
  float *ws;
  int32_t *cs;
  float *bias;
} W;

static void ref_mm(const W *w, const uint8_t *a_u8, float a_scale, int32_t a_zp,
                   float *out) {
  for (uint32_t col = 0; col < w->N; ++col)
    out[col] =
      fc_wh_col_det(a_u8, a_scale, a_zp, (const uint8_t *)w->nib, w->K / 32u,
                    w->N / 32u, col, w->cs[col], w->ws[col], w->bias[col]);
}
static void quant_row(const float *x, uint32_t k, uint8_t *q, float *scale,
                      int32_t *zp) {
  fc_wh_quant_row_det(x, k, q, scale, zp);
}

/* The layer, one expert and one row at a time: quantize the row, gate_up,
   SwiGLU (GeGLU-tanh under glu), requantize, down, then the routing
   multiply and the add into the token's output row -- in expert order, rows in
   order, like the kernel. */
static void reference_layer(uint32_t M, uint32_t K, uint32_t inter,
                            uint32_t N_out, uint32_t NE, const W *wg,
                            const W *wd, const float *act, const uint32_t *ridx,
                            const uint32_t *rc_, const float *rw, float *want,
                            uint32_t glu) {
  uint8_t *aq = (uint8_t *)malloc(K);
  uint8_t *mq = (uint8_t *)malloc(inter);
  float *gu = (float *)malloc(sizeof(float) * 2 * inter);
  float *dn = (float *)malloc(sizeof(float) * N_out);
  float *mid = (float *)malloc(sizeof(float) * inter);
  uint32_t base = 0;
  memset(want, 0, sizeof(float) * M * N_out);
  for (uint32_t e = 0; e < NE; ++e) {
    for (uint32_t i = 0; i < rc_[e]; ++i) {
      uint32_t row = ridx[base + i];
      float as;
      int32_t az;
      quant_row(act + (size_t)row * K, K, aq, &as, &az);
      ref_mm(&wg[e], aq, as, az, gu);
      /* swiglu_det.h: the spec the HVX SwiGLU matches bit for bit, and
         what standin/hvx_scalar.c runs, so the two agree to the bit. */
      for (uint32_t j = 0; j < inter; ++j)
        mid[j] = glu == HVX_GLU_GELU_TANH
                   ? geglu_det_one(gu[j], gu[inter + j])
                   : swiglu_det_one(gu[j], gu[inter + j]);
      float ms;
      int32_t mz;
      quant_row(mid, inter, mq, &ms, &mz);
      ref_mm(&wd[e], mq, ms, mz, dn);
      for (uint32_t c = 0; c < N_out; ++c) {
        /* Two operations through a volatile, matching what the kernel and
           the ARM path both do -- see hvx_scale_add_rows_f32's stub. */
        volatile float p = dn[c] * rw[base + i];
        want[(size_t)row * N_out + c] = want[(size_t)row * N_out + c] + p;
      }
    }
    base += rc_[e];
  }
  free(aq);
  free(mq);
  free(gu);
  free(dn);
  free(mid);
}

/* Elements of got outside 1e-5 relative of want; worst gets the largest. */
static uint32_t count_mismatches(const float *got, const float *want,
                                 uint32_t n, double *worst) {
  uint32_t bad = 0;
  *worst = 0.0;
  for (uint32_t i = 0; i < n; ++i) {
    double d = fabs((double)got[i] - (double)want[i]);
    double s = fabs((double)want[i]) + 1e-6;
    if (d / s > 1e-5) {
      ++bad;
    }
    if (d / s > *worst)
      *worst = d / s;
  }
  return bad;
}

/* ------------------------------- the test ------------------------------- */
static uint32_t rnd_state = 12345u;
static uint32_t rnd(void) {
  rnd_state = rnd_state * 1664525u + 1013904223u;
  return rnd_state >> 8;
}
static float rndf(void) { return (float)rnd() / 8388608.0f - 1.0f; }

static hexkl_weight_u8i4_table g_tbl;

static void make_weight(uint32_t slot, uint32_t K, uint32_t N, W *w) {
  const uint32_t tiles = (K / 32u) * (N / 32u);
  w->K = K;
  w->N = N;
  w->nib = (int8_t *)malloc(tiles * 512u);
  w->ws = (float *)malloc(sizeof(float) * N);
  w->cs = (int32_t *)malloc(sizeof(int32_t) * N);
  w->bias = (float *)malloc(sizeof(float) * N);
  for (uint32_t i = 0; i < tiles * 512u; ++i)
    w->nib[i] = (int8_t)(rnd() & 0xFF);
  for (uint32_t c = 0; c < N; ++c) {
    w->ws[c] = 0.01f + 0.001f * (float)(rnd() % 100u);
    w->cs[c] = 0; /* colsum folded into the model: kept 0 so both sides agree */
    w->bias[c] = 0.05f * rndf();
  }
  hexkl_weight_u8i4 *s = &g_tbl.slots[slot];
  s->in_use = 1;
  s->borrowed = 1; /* arena-backed, as every QS4CX_WH expert is */
  s->bits = 4u;    /* [#234 P4] a slot a 2-bit cell left must not stay 2 */
  s->K = K;
  s->N = N;
  s->wh_bytes = (uint8_t *)w->nib;
  s->w_scale = w->ws;
  s->colsum_w = w->cs;
  s->bias = w->bias;
}

/* The same weight in both widths: nibbles drawn only from pal, so the
   4-bit slot and the 2-bit slot hold the same values and the kernel's two
   paths have to agree byte for byte. Here rather than in the shared stubs
   because make_weight, which it builds on, is this file's own since the
   stand-in split. */
static void make_weight_pal(uint32_t slot4, uint32_t slot2, uint32_t K,
                            uint32_t N, const int8_t *pal, W *w) {
  const uint32_t tiles = (K / 32u) * (N / 32u);
  const uint32_t nib_bytes = tiles * 512u;
  uint8_t *two = (uint8_t *)malloc(nib_bytes / 2u);
  make_weight(slot4, K, N, w);
  /* Redraw the values from the palette and encode the same choices twice:
     nibble j of the 4-bit stream and code j of the 2-bit stream are the
     same weight, which is exactly what hvx_expand_i2i4 assumes. */
  memset(two, 0, nib_bytes / 2u);
  for (uint32_t j = 0; j < nib_bytes; ++j) {
    /* Nibble slots 2j and 2j+1 of tile j/512. The 2-bit side goes through
       whCodeByte2/whCodeShift2 -- the kernel's own order -- rather than a
       second encoder written here, which is what went wrong when this
       function had one. */
    const uint32_t tile = j / 512u;
    const uint32_t sl0 = (j % 512u) * 2u, sl1 = sl0 + 1u;
    const uint32_t c0 = rnd() & 3u, c1 = rnd() & 3u;
    ((uint8_t *)w->nib)[j] =
      (uint8_t)((pal[c0] & 0x0F) | ((pal[c1] & 0x0F) << 4));
    two[tile * 256u + whCodeByte2(sl0)] |= (uint8_t)(c0 << whCodeShift2(sl0));
    two[tile * 256u + whCodeByte2(sl1)] |= (uint8_t)(c1 << whCodeShift2(sl1));
  }
  hexkl_weight_u8i4 *s = &g_tbl.slots[slot2];
  s->in_use = 1;
  s->K = K;
  s->N = N;
  s->bits = 2u;
  s->pal[0] = pal[0];
  s->pal[1] = pal[1];
  s->pal[2] = pal[2];
  s->pal[3] = pal[3];
  s->wh_bytes = two;
  s->w_scale = w->ws;
  s->colsum_w = w->cs;
  s->bias = w->bias;
}

/* ---- The M=1 GEMV path against the HMX stand-in ------------------------
   One (shape, M) case: the same call with flags 0 (every expert a 64-row
   HMX block) and with HEXKL_MOE_FLAG_M1_GEMV, byte-compared. The HMX stub
   and the GEMV stub sum the same u8 x i4 products, so identical output
   here says the M=1 path packs the right rows into the right slots, runs
   the same epilogues on them and adds them into the output in the same
   order -- the plumbing, which is what a host check can hold. Whether the
   HVX GEMV's int32 equals the HMX's on silicon is the device gtest's.
   The log of GEMV tiles is then held against a reference built from the
   token rows the routing names, independently of the kernel's slot pack. */
static int g_pf_lead_ok = 1;
static int g_feed_sched_ok = 1;
static uint32_t g_bypass_cells; /**< [#158] cells run with the bypass bit */
static uint64_t g_bypass_total; /**< bytes they bypassed, summed */
static uint64_t g_feed_pushes, g_feed_waits;
/* #177: the feed cells rerun at N = 2, 3, 4 queues. */
static int g_q_ok = 1;
static uint32_t g_q_cells;
static uint64_t g_q_slices, g_q_lane_waits;
static int g_last_rc; /* the M=1 call's return code, for the timeout case */
/* #185: the N > 1 cells' run shape -- pool runs and submitted jobs per call
   against the schedule's table (hexkl_mm_u8i4_moe.c, N DMA QUEUES). */
static int g_ov_ok = 1;
static uint32_t g_ov_cells;
static int run_m1_case(const char *shape, uint32_t M, uint32_t K,
                       uint32_t inter, uint32_t N_out, uint32_t NE,
                       const uint32_t *rc_, uint32_t slot0, uint8_t *vtcm,
                       size_t vtcm_bytes, hexkl_moe_scratch *scratch,
                       uint32_t gemv_flags, float *ref, int have_ref) {
  const uint32_t lead_kb = hexkl_moe_flags_lead_kb(gemv_flags);
  const uint32_t rows1 = hexkl_moe_flags_rows1(gemv_flags);
  const uint32_t feed = hexkl_moe_flags_feed(gemv_flags);
  /* #177: the queues the feed splits over; the pool stand-in has PF_LANES
     lanes, more than 4, so every asked-for queue gets one. */
  const uint32_t nq = feed ? hexkl_moe_flags_dma_q(gemv_flags) : 1u;
  /* Same weights and activation for every configuration of a case, so the
     HMX reference above can be computed once and reused. */
  rnd_state = 12345u + 7919u * M + 104729u * (uint32_t)(shape[0] == 'r');
  W *wg = (W *)calloc(NE, sizeof(W));
  W *wd = (W *)calloc(NE, sizeof(W));
  uint32_t *hg = (uint32_t *)calloc(NE, sizeof(uint32_t));
  uint32_t *hd = (uint32_t *)calloc(NE, sizeof(uint32_t));
  uint32_t n_rows = 0, active = 0;
  for (uint32_t e = 0; e < NE; ++e) {
    n_rows += rc_[e];
    if (rc_[e] == 0u)
      continue; /* the kernel validates handles only where rows are */
    make_weight(slot0 + 2u * active, K, 2 * inter, &wg[e]);
    make_weight(slot0 + 2u * active + 1u, inter, N_out, &wd[e]);
    hg[e] = slot0 + 2u * active;
    hd[e] = slot0 + 2u * active + 1u;
    ++active;
  }
  /* Distinct rows inside an expert (the top-k guarantee the kernel's
     scatter needs), a different first row per expert. */
  uint32_t *ridx = (uint32_t *)malloc(sizeof(uint32_t) * n_rows);
  float *rw = (float *)malloc(sizeof(float) * n_rows);
  for (uint32_t e = 0, i = 0; e < NE; ++e)
    for (uint32_t r = 0; r < rc_[e]; ++r, ++i) {
      ridx[i] = (e * 7u + r) % M;
      rw[i] = 0.1f + 0.9f * ((float)(rnd() % 100u) / 100.f);
    }
  float *act = (float *)malloc(sizeof(float) * M * K);
  for (uint32_t i = 0; i < M * K; ++i)
    act[i] = rndf();

  float *out_hmx = (float *)malloc(sizeof(float) * M * N_out);
  float *out_m1 = (float *)malloc(sizeof(float) * M * N_out);
  int fail = 0;

  /* The HMX reference depends only on (shape, M), not on the GEMV's
     (loop, lead) pair, and it is the expensive half of this check -- the
     scalar stand-in runs 64-row blocks. run_m1_cases hands the same
     buffer back for every configuration of a case, so it runs once. */
  int r = 0;
  uint64_t blocks_hmx = active;
  score_reset(0, vtcm, vtcm_bytes); /* the HMX run is not scored */
  if (have_ref) {
    memcpy(out_hmx, ref, sizeof(float) * M * N_out);
  } else {
    memset(hexkl_probe_us, 0, sizeof hexkl_probe_us);
    r = hexkl_mm_u8i4_moe_layer_run(&g_tbl, vtcm, vtcm_bytes, vtcm_bytes, M, K,
                                    inter, N_out, NE, hg, hd, ridx, rc_, rw,
                                    act, out_hmx, NULL, scratch, 0u);
    blocks_hmx = hexkl_probe_us[HEXKL_PROBE_BLOCKS];
    const uint64_t path_hmx = hexkl_probe_us[HEXKL_PROBE_PATH];
    if (r != 0 || blocks_hmx != active || path_hmx != 0u) {
      printf("M1 GEMV shape=%s M=%u: HMX stand-in rc=%d blocks=%llu (want %u) "
             "path=%llu\n",
             shape, M, r, (unsigned long long)blocks_hmx, active,
             (unsigned long long)path_hmx);
      fail = 1;
    }
    memcpy(ref, out_hmx, sizeof(float) * M * N_out);
  }

  memset(hexkl_probe_us, 0, sizeof hexkl_probe_us);
  gemv_log_reset();
  pf_reset();
  score_reset(feed != 0u, vtcm, vtcm_bytes);
  g_bypass_bytes = g_bypass_bad = 0u;
  const uint32_t run0 = g_run, sub0 = g_submits;
  r = hexkl_mm_u8i4_moe_layer_run(&g_tbl, vtcm, vtcm_bytes, vtcm_bytes, M, K,
                                  inter, N_out, NE, hg, hd, ridx, rc_, rw, act,
                                  out_m1, NULL, scratch, gemv_flags);
  g_last_rc = r;
  const uint64_t blocks = hexkl_probe_us[HEXKL_PROBE_BLOCKS];
  const uint64_t dma_kb = hexkl_probe_us[HEXKL_PROBE_DMA_KB];
  const uint64_t path = hexkl_probe_us[HEXKL_PROBE_PATH];
  const uint64_t fed = hexkl_probe_us[HEXKL_PROBE_M1_FEED];
  /* With the feed every active expert's two matrices go through the ring
     (the same count the HMX path's check below wants); without it none. */
  const uint64_t want_kb =
    feed
      ? (uint64_t)active * ((((K / 32u) * ((2u * inter) / 32u) * 512u) >> 10) +
                            (((inter / 32u) * (N_out / 32u) * 512u) >> 10))
      : 0u;
  const int same = memcmp(out_hmx, out_m1, sizeof(float) * M * N_out);
  fail |= (r != 0) || blocks != 0u || dma_kb != want_kb || path != 1u ||
          fed != (feed ? nq : 0u) || same != 0;
  /* [#158] Under the bypass bit every feed push, and nothing else, reads
     around the L2: exactly the active experts' weight bytes. Without the
     feed there is no weight push, so nothing is bypassed either. */
  const int bypass = (gemv_flags & HEXKL_MOE_FLAG_DMA_BYPASS) != 0u;
  const uint64_t want_bypass =
    bypass && feed
      ? (uint64_t)active *
          ((K / 32u) * ((2u * inter) / 32u) + (inter / 32u) * (N_out / 32u)) *
          512u
      : 0u;
  if (g_bypass_bytes != want_bypass || g_bypass_bad != 0u) {
    printf("M1 GEMV shape=%s M=%u bypass=%d feed=%u: bypassed %llu B (want "
           "%llu), %llu on heap copies\n",
           shape, M, bypass, (unsigned)feed, (unsigned long long)g_bypass_bytes,
           (unsigned long long)want_bypass, (unsigned long long)g_bypass_bad);
    fail = 1;
  }
  g_bypass_cells += bypass;
  g_bypass_total += g_bypass_bytes;

  /* The lead: with it, every GEMV column went through the prefetch-free
     call, each covered once by its own lane's box (no double fetch, no
     over-fetch); without it, every column fetched itself as before. Under
     the feed (#117) no box at all: every column took the prefetch-free
     entry on a VTCM address the scoreboard holds. */
  const uint64_t cols = (uint64_t)active * ((2u * inter + N_out) / 32u);
  const int pf_ok =
    g_pf_bad == 0u && g_rows1_seen == (1u << (rows1 != 0u)) &&
    (feed != 0u      ? (g_col_n == 0u && g_pf_cols == 0u && g_nopf_n == cols)
     : lead_kb != 0u ? (g_col_n == 0u && g_nopf_n == cols &&
                        g_pf_cols == cols && g_pf_used == cols)
                     : (g_nopf_n == 0u && g_pf_cols == 0u && g_col_n == cols));
  if (!pf_ok) {
    printf("M1 GEMV shape=%s M=%u PF lead=%u rows1=%u feed=%u: boxes=%llu "
           "cols covered=%llu nopf=%llu self-prefetched=%llu bad=%llu "
           "rows1_seen=%u (want %llu columns)\n",
           shape, M, (unsigned)lead_kb, (unsigned)rows1, (unsigned)feed,
           (unsigned long long)g_pf_cols, (unsigned long long)g_pf_used,
           (unsigned long long)g_nopf_n, (unsigned long long)g_col_n,
           (unsigned long long)g_pf_bad, (unsigned)g_rows1_seen,
           (unsigned long long)cols);
    fail = 1;
  }
  g_pf_lead_ok &= pf_ok;

  /* The feed's schedule (#117): two whole-matrix pushes and two waits per
     active expert, every VTCM read covered by a waited push, every push
     inside the arena and onto a slab whose previous matrix was read whole
     (gemv_stand_in / push2d above), and nothing pushed and left unread. */
  if (feed != 0u && nq == 1u) {
    uint32_t unread = 0;
    for (uint32_t k = 0; k < g_score_n; ++k)
      unread += g_score[k].reads != g_score[k].row_size / 256u;
    const int sched_ok = g_score_bad == 0u && g_score_n == 2u * active &&
                         g_score_waits == 2u * active &&
                         g_score_vtcm_reads == cols && unread == 0u;
    if (!sched_ok) {
      printf("M1 GEMV shape=%s M=%u FEED SCHEDULE: pushes=%u waits=%u "
             "vtcm reads=%u unread=%u bad=%u (want %u/%u/%llu/0/0)\n",
             shape, M, g_score_n, g_score_waits, g_score_vtcm_reads, unread,
             g_score_bad, 2u * active, 2u * active, (unsigned long long)cols);
      fail = 1;
    }
    g_feed_sched_ok &= sched_ok;
    g_feed_pushes += g_score_n;
    g_feed_waits += g_score_waits;
  } else if (feed != 0u) {
    /* #177: every matrix as min(nq, rows) row slices, no ring wait at all,
       each slice waited by its own lane inside the run that issued it and
       read only in a later run (the stand-ins above), every read covered,
       nothing left unread. The output is held against the HMX reference
       above, which the N = 1 cell of the same flags also matched, so the
       f32 memcmp is also the byte-compare against N = 1. */
    uint32_t unread = 0;
    for (uint32_t k = 0; k < g_score_n; ++k)
      unread += g_score[k].reads != g_score[k].row_size / 256u;
    const uint32_t kt = K / 32u, it = inter / 32u;
    const uint32_t want = active * ((nq < kt ? nq : kt) + (nq < it ? nq : it));
    const int q_ok = g_score_bad == 0u && g_score_n == want &&
                     g_score_waits == 0u && g_lane_waits != 0u &&
                     g_score_vtcm_reads == cols && unread == 0u;
    if (!q_ok) {
      printf("M1 GEMV shape=%s M=%u QUEUES=%u: slices=%u ring waits=%u "
             "lane waits=%u vtcm reads=%u unread=%u bad=%u (want "
             "%u/0/>0/%llu/0/0)\n",
             shape, M, nq, g_score_n, g_score_waits, g_lane_waits,
             g_score_vtcm_reads, unread, g_score_bad, want,
             (unsigned long long)cols);
      fail = 1;
    }
    /* #185: GU(0) is one submitted job when the pool has a worker per
       queue, else a run; then n A runs, run B and n C runs. */
    const uint32_t want_sub = g_workers >= nq ? 1u : 0u;
    const uint32_t want_runs = (1u - want_sub) + active + 1u + active;
    const int ov_ok =
      g_submits - sub0 == want_sub && g_run - run0 == want_runs + want_sub;
    if (!ov_ok) {
      printf("M1 GEMV shape=%s M=%u QUEUES=%u n=%u: %u runs, %u jobs (want "
             "%u runs, %u jobs)\n",
             shape, M, nq, active, g_run - run0 - (g_submits - sub0),
             g_submits - sub0, want_runs, want_sub);
      fail = 1;
    }
    g_ov_ok &= ov_ok;
    ++g_ov_cells;
    g_q_ok &= q_ok && !fail;
    ++g_q_cells;
    g_q_slices += g_score_n;
    g_q_lane_waits += g_lane_waits;
  }

  /* The logged gate_up tiles, row by row, against the routing's token row
     quantized by the reference's own quantizer. Down tiles are not held
     this way: their input is the kernel's requantized SwiGLU, which no
     independent int32 spec exists for -- the f32 memcmp above covers them.
     Every (expert, gate/up column) must be there exactly once. */
  uint32_t i32_bad = 0, i32_seen = 0;
  {
    uint8_t *aq = (uint8_t *)malloc(K);
    const uint32_t kt_n = K / 32u, gu_nt = (2u * inter) / 32u;
    for (size_t li = 0; li < g_gemv_n; ++li) {
      const gemv_log_entry *le = &g_gemv_log[li];
      uint32_t e = NE, base = 0;
      for (uint32_t x = 0, b = 0; x < NE; b += rc_[x], ++x)
        if (rc_[x] != 0u && (const uint8_t *)wg[x].nib == le->wh) {
          e = x;
          base = b;
        }
      if (e == NE)
        continue; /* a down GEMV */
      ++i32_seen;
      for (uint32_t rr = 0; rr < le->m; ++rr) {
        float as;
        int32_t az;
        quant_row(act + (size_t)ridx[base + rr] * K, K, aq, &as, &az);
        for (uint32_t c = 0; c < 32; ++c) {
          int32_t sum = 0;
          for (uint32_t kt = 0; kt < kt_n; ++kt)
            for (uint32_t k = 0; k < 32; ++k)
              sum += (int32_t)aq[kt * 32u + k] *
                     wh_value((const uint8_t *)wg[e].nib +
                                (size_t)(kt * gu_nt + le->nt) * 512u,
                              k, c);
          if (sum != le->tile[rr * 32u + c])
            ++i32_bad;
        }
      }
    }
    free(aq);
    if (i32_seen != active * gu_nt)
      fail = 1;
  }
  fail |= (i32_bad != 0u);

  /* And the M=1 output against the f32 reference on its own, as the
     97-row fixture is. */
  float *want = (float *)malloc(sizeof(float) * M * N_out);
  reference_layer(M, K, inter, N_out, NE, wg, wd, act, ridx, rc_, rw, want,
                  HVX_GLU_SILU);
  double worst = 0.0;
  const uint32_t bad = count_mismatches(out_m1, want, M * N_out, &worst);
  fail |= (bad != 0u);

  printf("M1 GEMV shape=%s M=%u lead=%u rows1=%u feed=%u q=%u bypass=%d f32 "
         "memcmp=%d i32 "
         "exact=%s blocks=%llu dma_kb=%llu (path=%llu, hmx blocks=%llu, "
         "gate_up tiles=%u, ref mismatches=%u worst_rel=%g)\n",
         shape, M, (unsigned)lead_kb, (unsigned)rows1, (unsigned)feed,
         (unsigned)nq, bypass, same != 0, i32_bad == 0u ? "yes" : "NO",
         (unsigned long long)blocks, (unsigned long long)dma_kb,
         (unsigned long long)path, (unsigned long long)blocks_hmx, i32_seen,
         bad, worst);

  for (uint32_t e = 0; e < NE; ++e) {
    if (rc_[e] == 0u)
      continue;
    g_tbl.slots[hg[e]].in_use = 0;
    g_tbl.slots[hd[e]].in_use = 0;
    free(wg[e].nib);
    free(wg[e].ws);
    free(wg[e].cs);
    free(wg[e].bias);
    free(wd[e].nib);
    free(wd[e].ws);
    free(wd[e].cs);
    free(wd[e].bias);
  }
  free(wg);
  free(wd);
  free(hg);
  free(hd);
  free(ridx);
  free(rw);
  free(act);
  free(out_hmx);
  free(out_m1);
  free(want);
  return fail;
}

/* M=1: four experts one row each; M=2: both tokens share two experts (two
   rows of a pair to the same expert) and take one more each; M=4: one
   expert holds all four tokens, then 3, 2 and seven singles; then M=1 to
   three and to five experts, the odd counts the feed's slab schedule
   (#117) branches on: the slab whose last gate_up is expert n-2 takes its
   downs first, and a fifth expert's down reuses a slot; then M=1 to one
   and to two experts (#185: n < 4, with and without a run B). The active
   experts are spread over the table with empties between them, at a
   stride coprime to the count so no two land on one slot. */
static int run_m1_cases(uint8_t *vtcm, size_t vtcm_bytes,
                        hexkl_moe_scratch *scratch) {
  static const uint32_t counts[7][10] = {
    {1, 1, 1, 1}, {2, 2, 1, 1, 1, 1}, {4, 3, 2, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1},    {1, 1, 1, 1, 1},    {1},
    {1, 1}};
  static const uint32_t Ms[7] = {1, 2, 4, 1, 1, 1, 1};
  /* #113's matrix, run inside one build: the five leads the device sweep
     measures x the two row loops, plus the build's own defaults (the tune
     bits clear), which is what a run that sets no env var gets; since
     #117 the whole matrix again with the VTCM feed on. The lead changes
     which columns each lane's boxes must cover, the loop changes which
     kernel entry every column takes and the feed changes where the bytes
     are read from; none changes a result bit, so every cell is also held
     against the HMX path. */
  static const uint32_t leads_kb[] = {0u, 192u, 384u, 768u, 1536u};
  const size_t n_pairs = 2u * (sizeof leads_kb / sizeof *leads_kb);
  int fail = 0;
  for (int shape = 0; shape < 2; ++shape) {
    const uint32_t K = shape ? 2048 : 64, inter = shape ? 1792 : 32,
                   N_out = shape ? 2048 : 64, NE = shape ? 32 : 12;
    uint32_t *rc_ = (uint32_t *)calloc(NE, sizeof(uint32_t));
    for (int c = 0; c < 7; ++c) {
      memset(rc_, 0, sizeof(uint32_t) * NE);
      for (uint32_t i = 0; i < 10 && counts[c][i] != 0u; ++i)
        rc_[(i * 5u) % NE] = counts[c][i];
      float *ref = (float *)malloc(sizeof(float) * Ms[c] * N_out);
      for (size_t cfg = 0; cfg <= 2u * n_pairs; ++cfg) {
        uint32_t flags = HEXKL_MOE_FLAG_M1_GEMV;
        if (cfg != 0u) {
          const size_t i = (cfg - 1u) % n_pairs;
          flags |= HEXKL_MOE_FLAG_GEMV_LEAD_SET |
                   HEXKL_MOE_FLAG_GEMV_ROWS1_SET |
                   HEXKL_MOE_FLAG_GEMV_FEED_SET |
                   ((leads_kb[i / 2u] / HEXKL_MOE_GEMV_LEAD_KB_UNIT)
                    << HEXKL_MOE_GEMV_LEAD_SHIFT);
          if (i % 2u)
            flags |= HEXKL_MOE_FLAG_GEMV_ROWS1;
          if (cfg > n_pairs)
            flags |= HEXKL_MOE_FLAG_GEMV_FEED;
          /* [#158] The bypass bit on the 192 and 768 KB cells of both
             halves: both loops with and without it, on the feed and on
             the arena read. The lead means nothing under the feed, so the
             cells it lands on differ only in this bit. */
          if ((i / 2u) % 2u)
            flags |= HEXKL_MOE_FLAG_DMA_BYPASS;
        }
        /* #177: the feed cells of the first two leads -- both loops, bypass
           off and on; the lead means nothing under the feed -- again at
           N = 2, 3, 4 queues. */
        const uint32_t max_q =
          (cfg > n_pairs && ((cfg - 1u) % n_pairs) / 2u < 2u) ? 4u : 1u;
        for (uint32_t q = 1u; q <= max_q; ++q)
          fail |= run_m1_case(shape ? "real" : "tiny", Ms[c], K, inter, N_out,
                              NE, rc_, 64u, vtcm, vtcm_bytes, scratch,
                              flags | ((q - 1u) << HEXKL_MOE_DMA_Q_SHIFT), ref,
                              cfg != 0u);
      }
      free(ref);
    }
    free(rc_);
  }
  if (fail)
    printf("M1 GEMV PATH DIFFERS FROM HMX PATH\n");
  else
    printf(
      "M1 GEMV PATH BIT-IDENTICAL TO HMX PATH (M=1,2,4, 1, 2, 3 and 5 experts; "
      "tiny+real; "
      "%u lead x loop configurations, feed=arena,vtcm)\n",
      (unsigned)(1u + 2u * n_pairs));
  printf(g_pf_lead_ok
           ? "M1 GEMV PREFETCH LEAD COVERS EVERY COLUMN (lanes=%u; leads "
             "0/192/384/768/1536 KB x rows4,rows1; build default %u KB "
             "rows1=%u)\n"
           : "M1 GEMV PREFETCH LEAD WRONG (lanes=%u; leads "
             "0/192/384/768/1536 KB x rows4,rows1; build default %u KB "
             "rows1=%u)\n",
         PF_LANES, (unsigned)HVX_GEMV_PF_LEAD_KB, (unsigned)HVX_GEMV_M1_ROWS1);
  if (!fail)
    printf("M1 GEMV DMA BYPASS OK (%u cells, %llu B bypassed = the feed "
           "cells' weight bytes, 0 on the arena read and on heap copies; "
           "output bit-identical to the HMX path)\n",
           g_bypass_cells, (unsigned long long)g_bypass_total);
  /* A lane wait that runs out of its guard (lane 1 of every run here) must
     fail the call, not return a result: the call returns AEE_EFAILED. Its
     other mismatches are expected, so its fail flag is not counted. */
  {
    static const uint32_t K = 64, inter = 32, N_out = 64, NE = 12;
    uint32_t rc_[12] = {0};
    float ref[64];
    for (uint32_t i = 0; i < 4u; ++i)
      rc_[(i * 5u) % NE] = 1u;
    const int q_ok = g_q_ok, ov_ok = g_ov_ok;
    const uint32_t q_cells = g_q_cells, ov_cells = g_ov_cells;
    const uint64_t q_slices = g_q_slices, q_waits = g_q_lane_waits;
    printf("-- injected lane timeout (the lines below are expected):\n");
    g_lane_timeout = 1;
    (void)run_m1_case(
      "tiny", 1u, K, inter, N_out, NE, rc_, 64u, vtcm, vtcm_bytes, scratch,
      HEXKL_MOE_FLAG_M1_GEMV | HEXKL_MOE_FLAG_GEMV_FEED_SET |
        HEXKL_MOE_FLAG_GEMV_FEED | (3u << HEXKL_MOE_DMA_Q_SHIFT),
      ref, 0);
    g_lane_timeout = 0;
    g_q_ok = q_ok; /* the case's own mismatches are the expected ones */
    g_ov_ok = ov_ok;
    g_ov_cells = ov_cells;
    g_q_cells = q_cells;
    g_q_slices = q_slices;
    g_q_lane_waits = q_waits;
    if (g_last_rc == AEE_EFAILED) {
      printf("M1 FEED QUEUES TIMEOUT OK (a lane wait past its guard fails "
             "the call: AEE_EFAILED)\n");
    } else {
      printf("M1 FEED QUEUES TIMEOUT WRONG (rc=%d)\n", g_last_rc);
      g_q_ok = 0;
    }
  }
  /* #185: N = 4 queues on a pool of 3 workers, fewer than the queues: the
     call must still use all four (a run's lanes include the caller) and
     match the HMX path. */
  {
    static const uint32_t K = 64, inter = 32, N_out = 64, NE = 12;
    uint32_t rc_[12] = {0};
    float ref[64];
    for (uint32_t i = 0; i < 4u; ++i)
      rc_[(i * 5u) % NE] = 1u;
    g_workers = 3u;
    const int f = run_m1_case(
      "tiny", 1u, K, inter, N_out, NE, rc_, 64u, vtcm, vtcm_bytes, scratch,
      HEXKL_MOE_FLAG_M1_GEMV | HEXKL_MOE_FLAG_GEMV_FEED_SET |
        HEXKL_MOE_FLAG_GEMV_FEED | (3u << HEXKL_MOE_DMA_Q_SHIFT),
      ref, 0);
    g_workers = PF_LANES - 1u;
    printf("M1 FEED QUEUES 3-WORKER POOL %s (N=4, fed=4, bit-identical)\n",
           f ? "WRONG" : "OK");
    fail |= f;
  }
  if (g_ov_ok && g_ov_cells != 0u)
    printf("M1 FEED OVERLAP OK (%u cells; GU(0) job beside QUANT; run B "
           "always)\n",
           g_ov_cells);
  else
    printf("M1 FEED OVERLAP WRONG (%u cells)\n", g_ov_cells);
  fail |= !g_ov_ok || g_ov_cells == 0u;
  if (g_df_bad == 0u && g_df_seen != 0u)
    printf("M1 FEED DATAFLOW OK (%llu buffer accesses in feed runs; no "
           "cross-lane RAW/WAR/WAW inside a run)\n",
           (unsigned long long)g_df_seen);
  else
    printf("M1 FEED DATAFLOW WRONG (%u races, %llu accesses)\n", g_df_bad,
           (unsigned long long)g_df_seen);
  fail |= g_df_bad != 0u || g_df_seen == 0u;
  if (g_q_ok && g_q_cells != 0u)
    printf("M1 FEED QUEUES OK (n=2,3,4; %u cells, %llu slices, %llu lane "
           "waits; each slice waited by its lane in its run, read in a later "
           "run; output bit-identical to N=1)\n",
           g_q_cells, (unsigned long long)g_q_slices,
           (unsigned long long)g_q_lane_waits);
  else
    printf("M1 FEED QUEUES WRONG (%u cells)\n", g_q_cells);
  fail |= !g_q_ok || g_q_cells == 0u;
  printf(g_feed_sched_ok
           ? "M1 GEMV VTCM FEED SCHEDULE OK (%llu pushes, %llu waits over the "
             "cases; whole-matrix descriptors, every read after its wait, 2 "
             "slabs <= arena, no l2fetch under the feed; build default "
             "feed=%u)\n"
           : "M1 GEMV VTCM FEED SCHEDULE WRONG (%llu pushes, %llu waits; build "
             "default feed=%u)\n",
         (unsigned long long)g_feed_pushes, (unsigned long long)g_feed_waits,
         (unsigned)HVX_GEMV_M1_FEED);
  return fail;
}

/* ---- [#225] The decode FC on WH weights (hexkl_mm_u8i4_fc_m1_run) -------
   Its parts side by side against fc_wh_det.h, bit for bit, with the feed
   off (the arena behind the GEMV's l2fetch) and on (each lane's own
   double buffer), both row loops, the bypass bit, a VTCM too small for
   two columns a lane (the arena read again), and on the feed cells the
   scoreboard: every block read only after its own lane's wait, no push
   onto a block not read whole, every push read whole by the end. Then a
   lane wait past its guard must fail the call. */
static int run_fc_wh_case(const char *name, uint32_t K, uint32_t n_parts,
                          const uint32_t *Np, uint8_t *vtcm, size_t vtcm_bytes,
                          hexkl_moe_scratch *scratch, uint32_t flags,
                          int *pushed) {
  W w[4];
  uint32_t h[4], N = 0;
  rnd_state = 4242u + K + 31u * n_parts;
  for (uint32_t p = 0; p < n_parts; ++p) {
    make_weight(200u + p, K, Np[p], &w[p]);
    for (uint32_t c = 0; c < Np[p]; ++c)
      w[p].cs[c] = (int32_t)(rnd() % 4001u) - 2000; /* the zp term on */
    h[p] = 200u + p;
    N += Np[p];
  }
  float *x = (float *)malloc(sizeof(float) * K);
  float *got = (float *)malloc(sizeof(float) * N);
  float *want = (float *)malloc(sizeof(float) * N);
  uint8_t *q = (uint8_t *)malloc(K);
  for (uint32_t k = 0; k < K; ++k)
    x[k] = rndf() * (k % 7u == 0u ? 3.0f : 1.0f);
  float as;
  int32_t az;
  fc_wh_quant_row_det(x, K, q, &as, &az);
  for (uint32_t p = 0, o = 0; p < n_parts; o += Np[p++])
    for (uint32_t c = 0; c < Np[p]; ++c)
      want[o + c] =
        fc_wh_col_det(q, as, az, (const uint8_t *)w[p].nib, K / 32u,
                      Np[p] / 32u, c, w[p].cs[c], w[p].ws[c], w[p].bias[c]);
  memset(got, 0xA5, sizeof(float) * N);
  score_reset(1, vtcm, vtcm_bytes);
  g_own_slice_ok = 1;
  const int rc = hexkl_mm_u8i4_fc_m1_run(&g_tbl, vtcm, (uint32_t)vtcm_bytes,
                                         (uint32_t)vtcm_bytes, K, n_parts, h, x,
                                         got, NULL, scratch, flags);
  g_own_slice_ok = 0;
  uint32_t unread = 0;
  for (uint32_t k = 0; k < g_score_n; ++k)
    unread += g_score[k].reads != g_score[k].row_size / 256u; /* 256 B units */
  const int same = memcmp(got, want, sizeof(float) * N) == 0;
  const int fail = rc != AEE_SUCCESS || !same || g_score_bad != 0u || unread;
  *pushed = (int)g_score_n;
  printf("FC WH %s K=%u parts=%u N=%u flags=0x%x: rc=%d bit_identical=%d "
         "pushes=%u lane_waits=%u vtcm_reads=%u unread=%u bad=%u%s\n",
         name, K, n_parts, N, flags, rc, same, g_score_n, g_lane_waits,
         g_score_vtcm_reads, unread, g_score_bad, fail ? "  FAIL" : "");
  g_score_on = 0;
  for (uint32_t p = 0; p < n_parts; ++p) {
    free(w[p].nib);
    free(w[p].ws);
    free(w[p].cs);
    free(w[p].bias);
    g_tbl.slots[200u + p].in_use = 0;
  }
  free(x);
  free(got);
  free(want);
  free(q);
  return fail;
}

static int run_fc_wh_cases(uint8_t *vtcm, size_t vtcm_bytes,
                           hexkl_moe_scratch *scratch) {
  static const uint32_t qkv[3] = {2048, 512, 512},
                        thirds[3] = {2048, 2048, 2048}, one[1] = {2048},
                        tiny[2] = {32, 96};
  const struct {
    const char *name;
    uint32_t K, n;
    const uint32_t *N;
  } shapes[4] = {{"q|k|v", 2048, 3, qkv},
                 {"in_proj", 2048, 3, thirds},
                 {"out_proj", 2048, 1, one},
                 {"tiny", 64, 2, tiny}};
  const uint32_t feed_on =
    HEXKL_MOE_FLAG_GEMV_FEED_SET | HEXKL_MOE_FLAG_GEMV_FEED;
  const uint32_t cfgs[5] = {
    HEXKL_MOE_FLAG_GEMV_FEED_SET, /* arena */
    HEXKL_MOE_FLAG_GEMV_FEED_SET | HEXKL_MOE_FLAG_GEMV_ROWS1_SET |
      HEXKL_MOE_FLAG_GEMV_ROWS1,
    feed_on,
    feed_on | HEXKL_MOE_FLAG_GEMV_ROWS1_SET | HEXKL_MOE_FLAG_GEMV_ROWS1,
    feed_on | HEXKL_MOE_FLAG_DMA_BYPASS};
  int fail = 0, pushed = 0, fed_cells = 0, small_ok = 1, cells = 0;
  for (uint32_t s = 0; s < 4u; ++s)
    for (uint32_t c = 0; c < 5u; ++c) {
      fail |=
        run_fc_wh_case(shapes[s].name, shapes[s].K, shapes[s].n, shapes[s].N,
                       vtcm, vtcm_bytes, scratch, cfgs[c], &pushed);
      ++cells;
      /* the feed cells must feed, the arena cells must not */
      if ((c >= 2u) != (pushed != 0)) {
        printf("FC WH %s cfg %u: pushes=%d, want %s\n", shapes[s].name, c,
               pushed, c >= 2u ? ">0" : "0");
        fail = 1;
      }
      fed_cells += pushed != 0;
    }
  /* VTCM for less than two columns a lane: the arena read, same bytes */
  fail |= run_fc_wh_case("q|k|v small-vtcm", 2048, 3, qkv, vtcm,
                         2u * 6u * 2048u * 16u - 1u, scratch, feed_on, &pushed);
  small_ok = pushed == 0;
  fail |= !small_ok;
  ++cells;
  printf("-- injected lane timeout (the line below is expected to fail):\n");
  g_lane_timeout = 1;
  {
    W w;
    uint32_t hh = 200u;
    float x[64], y[64];
    make_weight(200u, 64, 64, &w);
    for (uint32_t k = 0; k < 64u; ++k)
      x[k] = rndf();
    score_reset(1, vtcm, vtcm_bytes);
    g_own_slice_ok = 1;
    const int rc = hexkl_mm_u8i4_fc_m1_run(&g_tbl, vtcm, (uint32_t)vtcm_bytes,
                                           (uint32_t)vtcm_bytes, 64u, 1u, &hh,
                                           x, y, NULL, scratch, feed_on);
    g_own_slice_ok = 0;
    g_score_on = 0;
    g_tbl.slots[200].in_use = 0;
    free(w.nib);
    free(w.ws);
    free(w.cs);
    free(w.bias);
    printf("FC WH lane timeout: rc=%d (want AEE_EFAILED %d)\n", rc,
           AEE_EFAILED);
    fail |= rc != AEE_EFAILED;
  }
  g_lane_timeout = 0;
  /* a handle of another K and a free one */
  {
    W w;
    uint32_t hh = 200u, free_h = 201u;
    float x[64], y[64] = {0};
    make_weight(200u, 64, 64, &w);
    const int r1 = hexkl_mm_u8i4_fc_m1_run(&g_tbl, vtcm, (uint32_t)vtcm_bytes,
                                           (uint32_t)vtcm_bytes, 128u, 1u, &hh,
                                           x, y, NULL, scratch, 0u);
    const int r2 = hexkl_mm_u8i4_fc_m1_run(&g_tbl, vtcm, (uint32_t)vtcm_bytes,
                                           (uint32_t)vtcm_bytes, 64u, 1u,
                                           &free_h, x, y, NULL, scratch, 0u);
    /* [#234 P4] a 2-bit handle: refused until plan 229 S2 */
    g_tbl.slots[200].bits = 2u;
    const int r3 = hexkl_mm_u8i4_fc_m1_run(&g_tbl, vtcm, (uint32_t)vtcm_bytes,
                                           (uint32_t)vtcm_bytes, 64u, 1u, &hh,
                                           x, y, NULL, scratch, 0u);
    g_tbl.slots[200].bits = 4u;
    g_tbl.slots[200].in_use = 0;
    free(w.nib);
    free(w.ws);
    free(w.cs);
    free(w.bias);
    printf("FC WH refusals: other K rc=%d, free handle rc=%d, 2-bit rc=%d "
           "(want %d)\n",
           r1, r2, r3, AEE_EBADITEM);
    fail |= r1 != AEE_EBADITEM || r2 != AEE_EBADITEM || r3 != AEE_EBADITEM;
  }
  if (!fail)
    printf("FC WH BIT-IDENTICAL: hexkl_mm_u8i4_fc_m1_run vs fc_wh_det.h, %d "
           "cells (q|k|v, in_proj thirds, out_proj, tiny; arena x rows4/rows1, "
           "VTCM feed x rows4/rows1/bypass, a VTCM too small to feed); %d fed "
           "cells, every block read after its lane's wait and read whole; "
           "lane timeout fails the call\n",
           cells, fed_cells);
  else
    printf("FC WH DIFFERS FROM fc_wh_det.h\n");
  return fail;
}

int main(void) {
  const uint32_t M = 97, K = 64, inter = 32, N_out = 64, NE = 5;
  hvx_scalar_hook.prefetch = hook_prefetch;
  hvx_scalar_hook.gemv = hook_gemv;
  hvx_scalar_hook.gemv2 = hook_gemv2;
  hvx_scalar_hook.buf = hook_buf;
  static uint8_t vtcm[8u << 20];
  /* [#225] the FC cells alone: run_host_checks.sh's mutants of
     hexkl_mm_u8i4_fc_m1_run, which need nothing else */
  if (getenv("MOE_CHECK_FC_WH_ONLY") != NULL) {
    hexkl_moe_scratch fs = {NULL, NULL, 0};
    const int f = run_fc_wh_cases(vtcm, sizeof vtcm, &fs);
    hexkl_moe_scratch_free(&fs);
    return f;
  }

  hexkl_moe_layout L;
  int rc = hexkl_mm_u8i4_moe_layout(K, inter, N_out, sizeof vtcm, &L);
  printf("layout rc=%d total=%u (act %u gu %u dn %u gate %u mid %u stage %u "
         "dn stage %u x%u)\n",
         rc, L.total, L.act_off, L.w_gu_off, L.w_dn_off, L.gate_off, L.mid_off,
         L.result_off, L.dn_stage_off, L.dn_ring);
  if (rc)
    return 1;

  W wg[8], wd[8];
  for (uint32_t e = 0; e < NE; ++e) {
    make_weight(e, K, 2 * inter, &wg[e]);
    make_weight(NE + e, inter, N_out, &wd[e]);
  }
  uint32_t hg[8], hd[8];
  for (uint32_t e = 0; e < NE; ++e) {
    hg[e] = e;
    hd[e] = NE + e;
  }

  float *act = (float *)malloc(sizeof(float) * M * K);
  for (uint32_t i = 0; i < M * K; ++i)
    act[i] = rndf();

  /* Routing: expert 0 gets many rows (multi-block, a 6-row tail for the
     HVX path), expert 2 gets none, expert 3 a tail of exactly the
     threshold, expert 4 a second block too big for it (stays on the HMX);
     rows repeat across experts the way top-k does. */
  uint32_t rc_[8] = {70, 12, 0, 64 + HVX_GEMM_U8I4_MAX_ROWS,
                     64 + HVX_GEMM_U8I4_MAX_ROWS + 10};
  uint32_t n_rows = 0;
  for (uint32_t e = 0; e < NE; ++e)
    n_rows += rc_[e];
  uint32_t *ridx = (uint32_t *)malloc(sizeof(uint32_t) * n_rows);
  float *rw = (float *)malloc(sizeof(float) * n_rows);
  /* Distinct within an expert, the kernel's contract (a token picks an
     expert once; the down epilogue's row units run side by side): a random
     start and step over the prime M. */
  for (uint32_t e = 0, i = 0; e < NE; ++e) {
    const uint32_t s0 = rnd() % M, st = 1u + rnd() % (M - 1u);
    for (uint32_t r = 0; r < rc_[e]; ++r, ++i) {
      ridx[i] = (s0 + r * st) % M;
      rw[i] = 0.1f + 0.9f * ((float)(rnd() % 100u) / 100.f);
    }
  }

  float *got = (float *)malloc(sizeof(float) * M * N_out);
  /* One scratch across every call below, the way the session holds it: the
     later, smaller calls must work out of the block the first one grew. */
  hexkl_moe_scratch scratch = {NULL, NULL, 0};
  rc = hexkl_mm_u8i4_moe_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M, K,
                                   inter, N_out, NE, hg, hd, ridx, rc_, rw, act,
                                   got, NULL, &scratch, 0u);
  printf("run rc=%d  n_rows=%u\n", rc, n_rows);
  if (rc)
    return 1;

  /* reference */
  float *want = (float *)calloc(M * N_out, sizeof(float));
  reference_layer(M, K, inter, N_out, NE, wg, wd, act, ridx, rc_, rw, want,
                  HVX_GLU_SILU);

  double worst = 0.0;
  uint32_t bad = count_mismatches(got, want, M * N_out, &worst);
  printf("mismatches=%u of %u   worst_rel=%g\n", bad, M * N_out, worst);
  printf(bad == 0 ? "MOE KERNEL MATCHES REFERENCE\n" : "MOE KERNEL DIFFERS\n");
  int fail = (bad != 0);

  /* Every active expert's gate_up and down must have been pushed. The DMA
     stub here completes immediately, so a missing push cannot show up as a
     wrong result the way a premature one does -- this counter is the only
     thing that catches it, and it is also what the device profile divides
     by to get GB/s, so a wrong count would quietly misreport the bandwidth
     the whole plan is gated on. */
  {
    uint32_t active = 0;
    for (uint32_t e = 0; e < NE; ++e) {
      if (rc_[e] != 0u)
        ++active;
    }
    const uint32_t gu_kb = ((K / 32u) * ((2u * inter) / 32u) * 512u) >> 10;
    const uint32_t dn_kb = ((inter / 32u) * (N_out / 32u) * 512u) >> 10;
    const uint64_t want = (uint64_t)active * (gu_kb + dn_kb);
    const uint64_t got_kb = hexkl_probe_us[HEXKL_PROBE_DMA_KB];
    printf("weight DMA        : %llu KB over %u experts (want %llu)\n",
           (unsigned long long)got_kb, active, (unsigned long long)want);
    if (got_kb != want)
      fail = 1;
    /* HMX blocks only: 70 -> 1 + a 6-row tail on the HVX, 12 -> 1,
       80 -> 1 + a 16-row tail, 90 -> 2 (its 26-row second block is over
       the tail threshold). 6 would mean a tail went to the HMX after all,
       4 that a full block was skipped. */
    printf("HMX blocks        : %llu (want 5: two tails on the HVX)\n",
           (unsigned long long)hexkl_probe_us[HEXKL_PROBE_BLOCKS]);
    if (hexkl_probe_us[HEXKL_PROBE_BLOCKS] != 5u)
      fail = 1;
  }

  /* The switch does not apply at M > 4: the same call with the flag set
     takes the HMX path -- same blocks, same DMA, same bytes. */
  {
    float *got_on = (float *)malloc(sizeof(float) * M * N_out);
    const uint64_t kb_off = hexkl_probe_us[HEXKL_PROBE_DMA_KB];
    memset(hexkl_probe_us, 0, sizeof hexkl_probe_us);
    int r = hexkl_mm_u8i4_moe_layer_run(
      &g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M, K, inter, N_out, NE, hg, hd,
      ridx, rc_, rw, act, got_on, NULL, &scratch, HEXKL_MOE_FLAG_M1_GEMV);
    const int same = memcmp(got, got_on, sizeof(float) * M * N_out);
    printf("M=97 flag on      : rc=%d HMX blocks=%llu dma_kb=%llu (off %llu) "
           "path=%llu memcmp=%d\n",
           r, (unsigned long long)hexkl_probe_us[HEXKL_PROBE_BLOCKS],
           (unsigned long long)hexkl_probe_us[HEXKL_PROBE_DMA_KB],
           (unsigned long long)kb_off,
           (unsigned long long)hexkl_probe_us[HEXKL_PROBE_PATH], same != 0);
    fail |= (r != 0) || hexkl_probe_us[HEXKL_PROBE_BLOCKS] != 5u ||
            hexkl_probe_us[HEXKL_PROBE_DMA_KB] != kb_off ||
            hexkl_probe_us[HEXKL_PROBE_PATH] != 0u || same != 0;
    free(got_on);
  }

  /* [#158] The M > 1 HMX path under the bypass bit: the same bytes, and
     src_bypass on exactly the arena-backed weight chunks -- not on the
     activation blocks, not on the staging copies, and not on a heap slot
     the DSP memcpy'd and did not flush (expert 1's gate_up here, clean 0);
     [#267 L1] a flushed heap image (clean 1) takes it like an arena slot.
     The two runs above sent no bypass bit, so nothing may have been
     bypassed yet. */
  {
    const uint64_t before = g_bypass_bytes;
    float *got_bp = (float *)malloc(sizeof(float) * M * N_out);
    uint64_t want = 0u;
    for (uint32_t e = 0; e < NE; ++e)
      if (rc_[e] != 0u)
        want += (uint64_t)((K / 32u) * ((2u * inter) / 32u) +
                           (inter / 32u) * (N_out / 32u)) *
                512u;
    g_tbl.slots[hg[1]].borrowed = 0;
    want -= (uint64_t)(K / 32u) * ((2u * inter) / 32u) * 512u;
    int r = hexkl_mm_u8i4_moe_layer_run(
      &g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M, K, inter, N_out, NE, hg, hd,
      ridx, rc_, rw, act, got_bp, NULL, &scratch,
      HEXKL_MOE_FLAG_M1_GEMV | HEXKL_MOE_FLAG_DMA_BYPASS);
    /* [#267 L1] the same slot flushed: bypassed again, same bytes */
    const uint64_t after_dirty = g_bypass_bytes;
    const uint64_t one_gu = (uint64_t)(K / 32u) * ((2u * inter) / 32u) * 512u;
    g_tbl.slots[hg[1]].clean = 1;
    float *got_cl = (float *)malloc(sizeof(float) * M * N_out);
    const int r_cl = hexkl_mm_u8i4_moe_layer_run(
      &g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M, K, inter, N_out, NE, hg, hd,
      ridx, rc_, rw, act, got_cl, NULL, &scratch,
      HEXKL_MOE_FLAG_M1_GEMV | HEXKL_MOE_FLAG_DMA_BYPASS);
    const int clean_ok = r_cl == 0 &&
                         g_bypass_bytes - after_dirty == want + one_gu &&
                         memcmp(got, got_cl, sizeof(float) * M * N_out) == 0;
    printf("M=97 clean heap   : rc=%d bypassed %llu B (want %llu) %s\n", r_cl,
           (unsigned long long)(g_bypass_bytes - after_dirty),
           (unsigned long long)(want + one_gu),
           clean_ok ? "bit-identical" : "WRONG");
    fail |= !clean_ok;
    free(got_cl);
    g_tbl.slots[hg[1]].clean = 0;
    g_tbl.slots[hg[1]].borrowed = 1;
    const int same = memcmp(got, got_bp, sizeof(float) * M * N_out);
    printf("M=97 dma bypass   : rc=%d bypassed %llu B (want %llu, before %llu, "
           "heap copies %llu) memcmp=%d\n",
           r, (unsigned long long)after_dirty, (unsigned long long)want,
           (unsigned long long)before, (unsigned long long)g_bypass_bad,
           same != 0);
    const int ok = r == 0 && before == 0u && after_dirty == want &&
                   g_bypass_bad == 0u && same == 0;
    printf(ok ? "MOE HMX DMA BYPASS OK (weights only, arena slots and "
                "flushed heap images only, bit-identical)\n"
              : "MOE HMX DMA BYPASS WRONG\n");
    fail |= !ok;
    free(got_bp);
  }

  /* The identity the GeGLU epilogue rests on, x sigmoid(2y) == 0.5 x (1 +
     tanh y), on a sweep of gate values in f32 and before any quantization:
     a wrong constant shows up here as 1e-3, not as a u8 flip. */
  {
    double worst_id = 0.0;
    for (int i = -2000; i <= 2000; ++i) {
      const float g = (float)i * 0.005f; /* [-10, 10] */
      /* In double: f32's 1 + tanh(y) cancels to a few ulps for y < -5. */
      const double gd = g;
      const double ref =
        0.5 * gd *
        (1.0 + tanh(0.7978845608028654 * (gd + 0.044715 * gd * gd * gd)));
      const double d =
        fabs((double)geglu_det_one(g, 1.f) - ref) / (fabs(ref) + 1e-6);
      if (d > worst_id)
        worst_id = d;
    }
    printf("gelu identity     : worst_rel=%g (want < 1e-5)\n", worst_id);
    fail |= (worst_id > 1e-5);
  }

  /* The same layer under the GeGLU bit (doc 55: gelu_tanh experts) against
     the reference in swiglu_det.h's form: the epilogue is the only
     difference, so it must match to the bit. The M=1 GEMV reaches the same
     hvx_dequant_swiglu_acc_tiles_to_f32 with the same act. */
  {
    float *got_g = (float *)malloc(sizeof(float) * M * N_out);
    float *want_g = (float *)calloc(M * N_out, sizeof(float));
    int r = hexkl_mm_u8i4_moe_layer_run(
      &g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M, K, inter, N_out, NE, hg, hd,
      ridx, rc_, rw, act, got_g, NULL, &scratch, HEXKL_MOE_FLAG_GELU_TANH);
    reference_layer(M, K, inter, N_out, NE, wg, wd, act, ridx, rc_, rw, want_g,
                    HVX_GLU_GELU_TANH);
    double worst_g = 0.0;
    const uint32_t bad_g = count_mismatches(got_g, want_g, M * N_out, &worst_g);
    const int differs = memcmp(got, got_g, sizeof(float) * M * N_out) == 0;
    printf(
      "gelu epilogue     : rc=%d mismatches=%u worst_rel=%g same_as_silu=%d\n",
      r, bad_g, worst_g, differs);
    fail |= (r != 0 || bad_g != 0 || differs);
    free(got_g);
    free(want_g);
  }

  /* [plan 201 S4] HEXKL_MOE_FLAG_GELU_TANH on every path of the call. M=97
     (HMX blocks plus two HVX tails) against the reference run with
     geglu_det_one, and unlike the SwiGLU output above; then one token
     routed to four experts on the HMX loop, the M=1 GEMV with the VTCM
     feed and the GEMV on the arena read: the reference again, and the
     three byte-equal. The epilogue's arithmetic is the stand-in's here;
     the real HVX one is geglu_host_check.c's. */
  {
    float *gg = (float *)malloc(sizeof(float) * M * N_out);
    float *want_g = (float *)calloc(M * N_out, sizeof(float));
    score_reset(0, vtcm, sizeof vtcm);
    int r = hexkl_mm_u8i4_moe_layer_run(
      &g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M, K, inter, N_out, NE, hg, hd,
      ridx, rc_, rw, act, gg, NULL, &scratch, HEXKL_MOE_FLAG_GELU_TANH);
    reference_layer(M, K, inter, N_out, NE, wg, wd, act, ridx, rc_, rw, want_g,
                    HVX_GLU_GELU_TANH);
    double w37 = 0.0;
    const uint32_t bad37 = count_mismatches(gg, want_g, M * N_out, &w37);
    const int differs = memcmp(gg, got, sizeof(float) * M * N_out) != 0;

    const uint32_t rc1[8] = {1, 1, 0, 1, 1};
    const uint32_t ridx1[4] = {0, 0, 0, 0};
    const float rw1[4] = {0.4f, 0.3f, 0.2f, 0.1f};
    static const uint32_t fl1[3] = {
      HEXKL_MOE_FLAG_GELU_TANH,
      HEXKL_MOE_FLAG_GELU_TANH | HEXKL_MOE_FLAG_M1_GEMV,
      HEXKL_MOE_FLAG_GELU_TANH | HEXKL_MOE_FLAG_M1_GEMV |
        HEXKL_MOE_FLAG_GEMV_FEED_SET};
    static const uint64_t path1[3] = {0u, 1u, 1u};
    float o1[3][64], want1[64];
    reference_layer(1, K, inter, N_out, NE, wg, wd, act, ridx1, rc1, rw1, want1,
                    HVX_GLU_GELU_TANH);
    int ok1 = 1;
    for (int f = 0; f < 3; ++f) {
      memset(hexkl_probe_us, 0, sizeof hexkl_probe_us);
      r |= hexkl_mm_u8i4_moe_layer_run(
        &g_tbl, vtcm, sizeof vtcm, sizeof vtcm, 1, K, inter, N_out, NE, hg, hd,
        ridx1, rc1, rw1, act, o1[f], NULL, &scratch, fl1[f]);
      ok1 &= hexkl_probe_us[HEXKL_PROBE_PATH] == path1[f] &&
             memcmp(o1[f], o1[0], sizeof o1[0]) == 0;
    }
    double w1 = 0.0;
    const uint32_t bad1 = count_mismatches(o1[0], want1, N_out, &w1);
    const int ok = r == 0 && bad37 == 0u && differs && bad1 == 0u && ok1;
    printf("geglu M=97 bad=%u worst_rel=%g differs_from_swiglu=%d; M=1 "
           "hmx/gemv-vtcm/gemv-arena bad=%u same_path_bytes=%d rc=%d\n",
           bad37, w37, differs, bad1, ok1, r);
    printf(ok ? "MOE GEGLU FLAG OK (HMX, tail, M=1 GEMV feed+arena)\n"
              : "MOE GEGLU FLAG WRONG\n");
    fail |= !ok;
    free(gg);
    free(want_g);
  }

  /* Split (doc 52 section 10.14): experts {0,1,2} then {3,4}, summed on
     the host, against the whole call. Only the fp32 addition order may
     differ, so the error is taken against the output's largest magnitude:
     per element, a sum that cancels to near zero reads 3.5e-5 here. */
  {
    float *a = (float *)malloc(sizeof(float) * M * N_out);
    float *b = (float *)malloc(sizeof(float) * M * N_out);
    uint32_t n0 = rc_[0] + rc_[1] + rc_[2];
    int r = hexkl_mm_u8i4_moe_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm,
                                        M, K, inter, N_out, 3, hg, hd, ridx,
                                        rc_, rw, act, a, NULL, &scratch, 0u);
    r |= hexkl_mm_u8i4_moe_layer_run(
      &g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M, K, inter, N_out, 2, hg + 3,
      hd + 3, ridx + n0, rc_ + 3, rw + n0, act, b, NULL, &scratch, 0u);
    double w = 0.0, big = 0.0;
    for (uint32_t i = 0; i < M * N_out; ++i) {
      double d = fabs((double)(a[i] + b[i]) - (double)got[i]);
      if (d > w)
        w = d;
      if (fabs((double)got[i]) > big)
        big = fabs((double)got[i]);
    }
    w /= big;
    printf("split 3+2 vs whole: rc=%d worst_rel=%g\n", r, w);
    fail |= (r != 0 || w > 1e-5);
    free(a);
    free(b);
  }

  /* QS2CX_WH: the same weights at two bits, expanded in VTCM after their
     DMA lands. The output must be BIT identical -- the expansion puts the
     int4 lattice back exactly, so every stage after the matmul sees what
     it saw before. A tolerance here would hide the one failure mode that
     matters, an expansion that is nearly right. This also exercises the
     in-place placement: the codes land in the top half of the slot they
     expand into, so a chunk boundary off by one clobbers a live tile and
     shows up as a wrong row. */
  {
    const int8_t pal[4] = {-6, -2, 1, 5};
    W wg2[8], wd2[8];
    uint32_t hg2[8], hd2[8];
    for (uint32_t e = 0; e < NE; ++e) {
      make_weight_pal(2u * NE + e, 4u * NE + e, K, 2 * inter, pal, &wg2[e]);
      make_weight_pal(3u * NE + e, 5u * NE + e, inter, N_out, pal, &wd2[e]);
      hg2[e] = 2u * NE + e;
      hd2[e] = 3u * NE + e;
    }
    float *four = (float *)malloc(sizeof(float) * M * N_out);
    float *two = (float *)malloc(sizeof(float) * M * N_out);
    hexkl_moe_scratch sc2 = {NULL, NULL, 0};
    const unsigned long long kb0 = hexkl_probe_us[HEXKL_PROBE_DMA_KB];
    int r4 = hexkl_mm_u8i4_moe_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm,
                                         M, K, inter, N_out, NE, hg2, hd2, ridx,
                                         rc_, rw, act, four, NULL, &sc2, 0u);
    const unsigned long long kb4 = hexkl_probe_us[HEXKL_PROBE_DMA_KB] - kb0;
    for (uint32_t e = 0; e < NE; ++e) {
      hg2[e] = 4u * NE + e;
      hd2[e] = 5u * NE + e;
    }
    int r2 = hexkl_mm_u8i4_moe_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm,
                                         M, K, inter, N_out, NE, hg2, hd2, ridx,
                                         rc_, rw, act, two, NULL, &sc2, 0u);
    const unsigned long long kb2 =
      hexkl_probe_us[HEXKL_PROBE_DMA_KB] - kb0 - kb4;
    uint32_t nb = 0;
    for (uint32_t i = 0; i < M * N_out; ++i) {
      if (memcmp(&four[i], &two[i], sizeof(float)) != 0)
        ++nb;
    }
    printf("2-bit experts     : rc=%d/%d  bitwise mismatches=%u of %u\n", r4,
           r2, nb, M * N_out);
    /* The point of the whole format: half the bytes cross the bus. Asserted
       rather than inspected, because a push that quietly kept the full
       width would still give the right answer. The probe counts whole
       kilobytes and this shape's chunks are smaller than that, so the
       2-bit side reads 0 -- what the bound catches is the regression that
       matters, a 2-bit weight pushed at full width (which would read the
       same as the 4-bit side, not half of it). */
    printf("2-bit weight DMA  : %llu KB vs %llu (want <= half)\n", kb2, kb4);
    fail |= (r4 != 0) || (r2 != 0) || (nb != 0) || (kb4 == 0ull) ||
            (kb2 * 2ull > kb4);
    free(four);
    free(two);
  }

  /* A shape with more than one chunk a weight, so the expansion's
     background lane actually runs. The shape above has inter_ntiles = 1
     and dn_ntiles = acc_tiles, so every chunk is the first one and the
     lookahead never fires -- which would have shipped the pipeline
     untested. inter 64 and N_out 128 give two chunks each way. */
  {
    /* inter must exceed 512 for gate_up to have more than one chunk:
       half is acc_tiles/2 and acc_tiles is min(32, 2*inter/32), so below
       that every chunk is the first one. The lookahead is two batches, so
       the queued path only runs at three chunks or more: inter 1088 gives
       inter_ntiles 34 against half 16, and N_out 2112 gives dn_ntiles 66
       against acc_tiles 32 -- three each. The real model is 1792 and 2048,
       four and two. */
    const uint32_t K2 = 64, I2 = 1088, N2 = 2112, NE2 = 3;
    const int8_t pal[4] = {-7, -2, 2, 6};
    W wg3[4], wd3[4];
    uint32_t hg3[4], hd3[4], hg3b[4], hd3b[4];
    for (uint32_t e = 0; e < NE2; ++e) {
      make_weight_pal(6u * NE + e, 6u * NE + 8u + e, K2, 2 * I2, pal, &wg3[e]);
      make_weight_pal(6u * NE + 16u + e, 6u * NE + 24u + e, I2, N2, pal,
                      &wd3[e]);
      hg3[e] = 6u * NE + e;
      hd3[e] = 6u * NE + 16u + e;
      hg3b[e] = 6u * NE + 8u + e;
      hd3b[e] = 6u * NE + 24u + e;
    }
    const uint32_t M2 = 97;
    float *a2 = (float *)malloc(sizeof(float) * M2 * K2);
    for (uint32_t i = 0; i < M2 * K2; ++i)
      a2[i] = rndf();
    uint32_t rc2[4] = {70, 1, 64};
    uint32_t nr2 = 0;
    for (uint32_t e = 0; e < NE2; ++e)
      nr2 += rc2[e];
    uint32_t *ri2 = (uint32_t *)malloc(sizeof(uint32_t) * nr2);
    float *rw2 = (float *)malloc(sizeof(float) * nr2);
    for (uint32_t e = 0, i = 0; e < NE2; ++e) { /* distinct within an expert */
      const uint32_t s0 = rnd() % M2, st = 1u + rnd() % (M2 - 1u);
      for (uint32_t r = 0; r < rc2[e]; ++r, ++i) {
        ri2[i] = (s0 + r * st) % M2;
        rw2[i] = 0.2f + 0.7f * ((float)(rnd() % 100u) / 100.f);
      }
    }
    float *o4 = (float *)malloc(sizeof(float) * M2 * N2);
    float *o2 = (float *)malloc(sizeof(float) * M2 * N2);
    hexkl_moe_scratch sc3 = {NULL, NULL, 0};
    int r4 = hexkl_mm_u8i4_moe_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm,
                                         M2, K2, I2, N2, NE2, hg3, hd3, ri2,
                                         rc2, rw2, a2, o4, NULL, &sc3, 0u);
    int r2 = hexkl_mm_u8i4_moe_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm,
                                         M2, K2, I2, N2, NE2, hg3b, hd3b, ri2,
                                         rc2, rw2, a2, o2, NULL, &sc3, 0u);
    uint32_t nb = 0;
    for (uint32_t i = 0; i < M2 * N2; ++i) {
      if (memcmp(&o4[i], &o2[i], sizeof(float)) != 0)
        ++nb;
    }
    printf("2-bit multi-chunk : rc=%d/%d  bitwise mismatches=%u of %u "
           "(chunks gu=%u dn=%u)\n",
           r4, r2, nb, M2 * N2, (I2 / 32u + 15u) / 16u, (N2 / 32u + 31u) / 32u);
    fail |= (r4 != 0) || (r2 != 0) || (nb != 0);
    free(a2);
    free(ri2);
    free(rw2);
    free(o4);
    free(o2);
  }

  /* 2-bit on the M=1 GEMV path with the VTCM feed (#117): the feed pushes
     half the bytes into packed-size slabs and the native LUT GEMV consumes
     them without materializing int4. Bitwise identical or the format is not
     usable on this path. The real
     shape, because the feed only engages when two gate_up slabs and two
     downs fit the arena -- the tiny shape would silently run the arena
     read and prove nothing. */
  {
    const uint32_t K4 = 2048, I4 = 1792, N4 = 2048, NE4 = 4;
    const int8_t pal[4] = {-6, -2, 1, 5};
    const uint32_t S0 = 200u;
    W wg4[4], wd4[4];
    uint32_t hg4[4], hd4[4], hg4b[4], hd4b[4];
    uint32_t rc4[4] = {1, 1, 1, 1};
    uint32_t ri4[4] = {0, 0, 0, 0};
    float rw4[4] = {0.3f, 0.7f, 0.5f, 0.9f};
    for (uint32_t e = 0; e < NE4; ++e) {
      make_weight_pal(S0 + 4u * e, S0 + 4u * e + 1u, K4, 2 * I4, pal, &wg4[e]);
      make_weight_pal(S0 + 4u * e + 2u, S0 + 4u * e + 3u, I4, N4, pal, &wd4[e]);
      hg4[e] = S0 + 4u * e;
      hd4[e] = S0 + 4u * e + 2u;
      hg4b[e] = S0 + 4u * e + 1u;
      hd4b[e] = S0 + 4u * e + 3u;
    }
    float *a4 = (float *)malloc(sizeof(float) * K4);
    for (uint32_t i = 0; i < K4; ++i)
      a4[i] = rndf();
    float *g4 = (float *)malloc(sizeof(float) * N4);
    float *g2 = (float *)malloc(sizeof(float) * N4);
    /* [plan 229] Every M=1 schedule htp_decode has: the arena read, the
       VTCM feed on the ring and on four queues (#177 / #185), and the
       GeGLU epilogue on the four-queue feed (#231). Each against its own
       4-bit run; dma_kb only means something where the feed pushes. */
    const uint32_t feed = HEXKL_MOE_FLAG_M1_GEMV |
                          HEXKL_MOE_FLAG_GEMV_FEED_SET |
                          HEXKL_MOE_FLAG_GEMV_FEED;
    static const char *const name[4] = {"arena", "feed q=1", "feed q=4",
                                        "feed q=4 geglu"};
    const uint32_t fls[4] = {
      HEXKL_MOE_FLAG_M1_GEMV | HEXKL_MOE_FLAG_GEMV_FEED_SET, feed,
      feed | (3u << HEXKL_MOE_DMA_Q_SHIFT),
      feed | (3u << HEXKL_MOE_DMA_Q_SHIFT) | HEXKL_MOE_FLAG_GELU_TANH};
    hexkl_moe_scratch sc4 = {NULL, NULL, 0};
    for (uint32_t c = 0; c < 4u; ++c) {
      const uint32_t fl = fls[c];
      const uint64_t want_feed = c == 0u ? 0ull : c == 1u ? 1ull : 4ull;
      /* Both runs under the audits run_m1_case applies: the lead's boxes,
         and under the feed the scoreboard (every column read from a waited
         push of its shape) and the cross-lane dataflow. */
      const uint64_t cols2 = (uint64_t)NE4 * ((2u * I4 + N4) / 32u);
      const uint32_t df0 = g_df_bad;
      memset(hexkl_probe_us, 0, sizeof hexkl_probe_us);
      pf_reset();
      score_reset(c != 0u, vtcm, sizeof vtcm);
      int r4 = hexkl_mm_u8i4_moe_layer_run(
        &g_tbl, vtcm, sizeof vtcm, sizeof vtcm, 1u, K4, I4, N4, NE4, hg4, hd4,
        ri4, rc4, rw4, a4, g4, NULL, &sc4, fl);
      int audit = g_pf_bad == 0u && g_score_bad == 0u &&
                  (c == 0u || g_score_vtcm_reads == cols2);
      const uint64_t path4 = hexkl_probe_us[HEXKL_PROBE_PATH];
      const uint64_t feed4 = hexkl_probe_us[HEXKL_PROBE_M1_FEED];
      const uint64_t kb4 = hexkl_probe_us[HEXKL_PROBE_DMA_KB];
      memset(hexkl_probe_us, 0, sizeof hexkl_probe_us);
      pf_reset();
      score_reset(c != 0u, vtcm, sizeof vtcm);
      int r2 = hexkl_mm_u8i4_moe_layer_run(
        &g_tbl, vtcm, sizeof vtcm, sizeof vtcm, 1u, K4, I4, N4, NE4, hg4b, hd4b,
        ri4, rc4, rw4, a4, g2, NULL, &sc4, fl);
      audit &= g_pf_bad == 0u && g_score_bad == 0u && g_df_bad == df0 &&
               (c == 0u || g_score_vtcm_reads == cols2);
      score_reset(0, vtcm, sizeof vtcm);
      const uint64_t path2 = hexkl_probe_us[HEXKL_PROBE_PATH];
      const uint64_t feed2 = hexkl_probe_us[HEXKL_PROBE_M1_FEED];
      const uint64_t kb2 = hexkl_probe_us[HEXKL_PROBE_DMA_KB];
      uint32_t nb = 0;
      for (uint32_t i = 0; i < N4; ++i) {
        if (memcmp(&g4[i], &g2[i], sizeof(float)) != 0)
          ++nb;
      }
      printf("2-bit M1 GEMV %-14s: rc=%d/%d path=%llu/%llu feed=%llu/%llu "
             "dma_kb=%llu vs %llu (want %s) audit=%d  bitwise mismatches=%u "
             "of %u\n",
             name[c], r4, r2, (unsigned long long)path4,
             (unsigned long long)path2, (unsigned long long)feed4,
             (unsigned long long)feed2, (unsigned long long)kb2,
             (unsigned long long)kb4, c == 0u ? "0" : "half", audit, nb, N4);
      fail |= (r4 != 0) || (r2 != 0) || (nb != 0) || !audit ||
              (path4 != 1ull) || (path2 != 1ull) || (feed4 != want_feed) ||
              (feed2 != want_feed) ||
              (c == 0u ? (kb4 != 0ull || kb2 != 0ull)
                       : (kb4 == 0ull || kb2 * 2ull != kb4));
    }
    /* [#267 L3] HEXKL_MOE_FLAG_ROWS_OUT on the decode schedule (the feed
       on one queue, the bypass bit): one call over the four 4-bit experts
       writes four rows, row i bit-identical to a call with expert i alone
       (0 + w * res), under the same audits as the plain run (every column
       from a waited push, cross-lane dataflow); at M = 2 refused
       (AEE_EUNSUPPORTED) with nothing written. */
    {
      const uint32_t fl = feed | HEXKL_MOE_FLAG_DMA_BYPASS;
      const uint64_t cols2 = (uint64_t)NE4 * ((2u * I4 + N4) / 32u);
      float *rows = (float *)malloc(sizeof(float) * NE4 * N4);
      float *one = (float *)malloc(sizeof(float) * N4);
      float *a2 = (float *)calloc(2u * K4, sizeof(float));
      uint32_t nb = 0, solo_bad = 0;
      const uint32_t df0 = g_df_bad;
      memset(hexkl_probe_us, 0, sizeof hexkl_probe_us);
      pf_reset();
      score_reset(1, vtcm, sizeof vtcm);
      int rr = hexkl_mm_u8i4_moe_layer_run(
        &g_tbl, vtcm, sizeof vtcm, sizeof vtcm, 1u, K4, I4, N4, NE4, hg4, hd4,
        ri4, rc4, rw4, a4, rows, NULL, &sc4, fl | HEXKL_MOE_FLAG_ROWS_OUT);
      const int audit = g_pf_bad == 0u && g_score_bad == 0u &&
                        g_df_bad == df0 && g_score_vtcm_reads == cols2 &&
                        hexkl_probe_us[HEXKL_PROBE_PATH] == 1u &&
                        hexkl_probe_us[HEXKL_PROBE_M1_FEED] == 1u;
      score_reset(0, vtcm, sizeof vtcm);
      for (uint32_t e = 0; e < NE4; ++e) {
        uint32_t c1[4] = {0, 0, 0, 0};
        c1[e] = 1u;
        solo_bad |= (uint32_t)hexkl_mm_u8i4_moe_layer_run(
          &g_tbl, vtcm, sizeof vtcm, sizeof vtcm, 1u, K4, I4, N4, NE4, hg4, hd4,
          ri4, c1, &rw4[e], a4, one, NULL, &sc4, fl);
        for (uint32_t i = 0; i < N4; ++i)
          nb += memcmp(&one[i], &rows[(size_t)e * N4 + i], sizeof(float)) != 0;
      }
      /* M = 2: two rows, experts 0 and 1 one each */
      uint32_t rc2[4] = {1, 1, 0, 0}, ri2[2] = {0, 1};
      memset(rows, 0xA5, sizeof(float) * NE4 * N4);
      int rneg = hexkl_mm_u8i4_moe_layer_run(
        &g_tbl, vtcm, sizeof vtcm, sizeof vtcm, 2u, K4, I4, N4, NE4, hg4, hd4,
        ri2, rc2, rw4, a2, rows, NULL, &sc4, fl | HEXKL_MOE_FLAG_ROWS_OUT);
      uint32_t touched = 0;
      for (uint32_t i = 0; i < NE4 * N4; ++i) {
        uint32_t u;
        memcpy(&u, &rows[i], 4);
        touched += u != 0xA5A5A5A5u;
      }
      const int ok = rr == 0 && solo_bad == 0u && nb == 0u && audit &&
                     rneg == AEE_EUNSUPPORTED && touched == 0u;
      printf("M1 ROWS_OUT feed q=1 bypass: rc=%d audit=%d rows vs lone "
             "calls: %u of %u bitwise mismatches; M=2 rc=%d (want %d) "
             "touched=%u\n",
             rr, audit, nb, NE4 * N4, rneg, AEE_EUNSUPPORTED, touched);
      printf(ok ? "MOE M1 ROWS_OUT OK (one call, a row per expert, each "
                  "bit-identical to its lone call; M > 1 refused)\n"
                : "MOE M1 ROWS_OUT WRONG\n");
      fail |= !ok;
      free(rows);
      free(one);
      free(a2);
    }
    for (uint32_t e = 0; e < 4u * NE4; ++e)
      g_tbl.slots[S0 + e].in_use = 0;
    free(a4);
    free(g4);
    free(g2);
  }

  /* --- edge cases the routing can actually produce --------------------- */
  {
    uint32_t z[8] = {0, 0, 0, 0, 0};
    memset(got, 0xA5, sizeof(float) * M * N_out);
    int r = hexkl_mm_u8i4_moe_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm,
                                        M, K, inter, N_out, NE, hg, hd, ridx, z,
                                        rw, act, got, NULL, &scratch, 0u);
    int ok = (r == 0);
    for (uint32_t i = 0; i < M * N_out; ++i) {
      if (got[i] != 0.f) {
        ok = 0;
        break;
      }
    }
    printf("all-empty routing : %s\n", ok ? "zeroed, rc=0" : "WRONG");
    fail |= !ok;
  }
  {
    uint32_t c64[8] = {64, 0, 0, 0, 0};
    int r = hexkl_mm_u8i4_moe_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm,
                                        M, K, inter, N_out, NE, hg, hd, ridx,
                                        c64, rw, act, got, NULL, &scratch, 0u);
    printf("exactly 64 rows   : rc=%d\n", r);
    fail |= (r != 0);
  }
  {
    uint32_t c1[8] = {1, 0, 0, 0, 0};
    uint32_t bad_row[1] = {M};
    int r = hexkl_mm_u8i4_moe_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm,
                                        M, K, inter, N_out, NE, hg, hd, bad_row,
                                        c1, rw, act, got, NULL, &scratch, 0u);
    printf("row_index >= M    : rc=%d (want %d)\n", r, AEE_EBADPARM);
    fail |= (r != AEE_EBADPARM);
  }
  {
    hexkl_moe_layout R;
    int r = hexkl_mm_u8i4_moe_layout(2048, 1792, 2048, 8300u * 1024u, &R);
    printf("LFM2 shapes       : rc=%d total=%.2f MB (doc 47 section 11 says "
           "6.92)\n",
           r, R.total / 1048576.0);
    fail |= (r != 0);
    r = hexkl_mm_u8i4_moe_layout(2048, 1792, 2048, 4u << 20, &R);
    printf("4 MB arena        : rc=%d (want %d)\n", r, AEE_ENOMEMORY);
    fail |= (r != AEE_ENOMEMORY);
  }

  /* --- #87: the DMA trace is inert when probing is off, and a no-op for
     the output when it is on ------------------------------------------ */
  {
    float *got_on = (float *)malloc(sizeof(float) * M * N_out);
    hexkl_probe_on = 1;
    int r = hexkl_mm_u8i4_moe_layer_run(
      &g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M, K, inter, N_out, NE, hg, hd,
      ridx, rc_, rw, act, got_on, NULL, &scratch, 0u);
    fail |= (r != 0);
    const uint64_t desc_on = hexkl_probe_us[HEXKL_PROBE_DMA_DESC];
    hexkl_dma_trace_reset(0);
    memset(hexkl_probe_us, 0, sizeof(hexkl_probe_us));
    hexkl_probe_on = 0;
    r = hexkl_mm_u8i4_moe_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M,
                                    K, inter, N_out, NE, hg, hd, ridx, rc_, rw,
                                    act, got, NULL, &scratch, 0u);
    fail |= (r != 0);
    const int same = memcmp(got_on, got, sizeof(float) * M * N_out) == 0;
    printf("profile on vs off : memcmp %s (%llu descriptors traced when on)\n",
           same ? "== 0" : "!= 0", (unsigned long long)desc_on);
    printf(same ? "PROFILE ON/OFF BYTE-IDENTICAL\n"
                : "PROFILE ON/OFF DIFFER\n");
    fail |= !same || desc_on == 0u;
    const hexkl_dma_trace *t = hexkl_dma_trace_get();
    uint64_t slots = 0;
    for (int k = HEXKL_PROBE_DMA_DESC; k <= HEXKL_PROBE_DMA_LAST_ISSUE_US; ++k)
      slots += hexkl_probe_us[k];
    const int untouched = t->n_push == 0u && t->n_wait == 0u &&
                          t->n_blocked == 0u && t->n_dropped == 0u &&
                          slots == 0u;
    printf(untouched ? "PROFILE OFF: TRACE UNTOUCHED\n"
                     : "PROFILE OFF: TRACE WRITTEN (n_push=%u n_wait=%u)\n",
           (unsigned)t->n_push, (unsigned)t->n_wait);
    fail |= !untouched;
    hexkl_probe_on = 1;
    free(got_on);
  }

  /* --- #87: the M=1 push/wait trace at the LFM2 shape is the planner's
     list (test/htp/nntr_moe_dma_plan.h), descriptor for descriptor ------ */
  {
    const uint32_t LK = 2048, LI = 1792, LN = 2048, LNE = 4;
    hexkl_moe_layout R;
    fail |= hexkl_mm_u8i4_moe_layout(LK, LI, LN, sizeof vtcm, &R) != 0;
    W lg[4], ld[4];
    uint32_t lhg[4], lhd[4];
    for (uint32_t e = 0; e < LNE; ++e) {
      make_weight(8u + e, LK, 2 * LI, &lg[e]);
      make_weight(12u + e, LI, LN, &ld[e]);
      lhg[e] = 8u + e;
      lhd[e] = 12u + e;
    }
    float *lact = (float *)malloc(sizeof(float) * LK);
    for (uint32_t i = 0; i < LK; ++i)
      lact[i] = rndf();
    float *lout = (float *)malloc(sizeof(float) * LN);
    uint32_t lrc[4] = {1, 1, 1, 1}, lridx[4] = {0, 0, 0, 0};
    float lrw[4] = {0.4f, 0.3f, 0.2f, 0.1f};
    const uint32_t ring0 = g_stub_idx;
    int r = hexkl_mm_u8i4_moe_layer_run(
      &g_tbl, vtcm, sizeof vtcm, sizeof vtcm, 1, LK, LI, LN, LNE, lhg, lhd,
      lridx, lrc, lrw, lact, lout, NULL, &scratch, 0u);
    fail |= (r != 0);
    static nntr_moe_dma_item plan[NNTR_MOE_DMA_PLAN_MAX];
    const uint32_t n_plan =
      nntr_moe_dma_plan_m1(LK, LI, LN, LNE, R.acc_tiles, R.w_gu_off, R.w_dn_off,
                           R.act_off, 0u, plan, NNTR_MOE_DMA_PLAN_MAX);
    const hexkl_dma_trace *t = hexkl_dma_trace_get();
    uint32_t push_ord[NNTR_MOE_DMA_PLAN_MAX];
    uint32_t np = 0, nw = 0, bad = 0;
    for (uint32_t k = 0; k < n_plan; ++k) {
      const nntr_moe_dma_item *it = &plan[k];
      if (it->op == NNTR_MOE_DMA_OP_PUSH) {
        push_ord[k] = np;
        const hexkl_dma_trace_push_rec *p = &t->push[np];
        if (np >= t->n_push || p->kind != it->kind || p->expert != it->expert ||
            p->chunk != it->chunk || p->row_size != it->row_size ||
            p->nrows != it->nrows || p->src_stride != it->src_stride ||
            p->ring_idx - ring0 != np) {
          if (bad++ < 4)
            printf("  push %u: plan kind=%u e=%u c=%u %ux%u@%u, trace kind=%u "
                   "e=%u c=%u %ux%u@%u\n",
                   np, it->kind, it->expert, it->chunk, it->row_size, it->nrows,
                   it->src_stride, p->kind, p->expert, p->chunk, p->row_size,
                   p->nrows, p->src_stride);
        }
        ++np;
      } else {
        const hexkl_dma_trace_wait_rec *w = &t->wait[nw];
        if (nw >= t->n_wait || w->site != it->kind ||
            w->ring_idx - ring0 != push_ord[it->src_off]) {
          if (bad++ < 4)
            printf("  wait %u: plan site=%u on push %u, trace site=%u on "
                   "push %u\n",
                   nw, it->kind, push_ord[it->src_off], w->site,
                   w->ring_idx - ring0);
        }
        ++nw;
      }
    }
    const int match = bad == 0u && np == t->n_push && nw == t->n_wait &&
                      hexkl_probe_us[HEXKL_PROBE_DMA_DESC] == np;
    printf("LFM2 M=1 trace    : %u pushes %u waits (plan %u/%u, %u items), "
           "depth max %llu\n",
           t->n_push, t->n_wait, np, nw, n_plan,
           (unsigned long long)hexkl_probe_us[HEXKL_PROBE_DMA_DEPTH_MAX]);
    printf(match ? "IN-SITU CHUNK PLAN MATCHES KERNEL (%u descriptors)\n"
                 : "IN-SITU CHUNK PLAN DIFFERS FROM KERNEL (%u descriptors)\n",
           np);
    fail |= !match || np != 46u;
    free(lact);
    free(lout);
  }

  fail |= run_m1_cases(vtcm, sizeof vtcm, &scratch);
  fail |= run_fc_wh_cases(vtcm, sizeof vtcm, &scratch);
  printf(fail ? "\nFAIL\n" : "\nALL CHECKS PASS\n");
  free(g_gemv_log);
  hexkl_moe_scratch_free(&scratch);
  return fail;
}
