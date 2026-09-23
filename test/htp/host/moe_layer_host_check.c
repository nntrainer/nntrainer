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
#include "hvx_gather_ah_u8.h"
#include "hvx_gemm_u8i4_wh.h"
#include "hvx_scale_add_f32.h"
#include <AEEStdErr.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nntr_moe_dma_plan.h"

/* ---- accumulator + acc layout ---- */
static int32_t g_acc[64][32];
static hexkl_acc_layout g_layout = {1, 1, 0,
                                    32}; /* probed, usable, base, stride */
const hexkl_acc_layout *hexkl_acc_layout_get(uint8_t *b, uint32_t off) {
  (void)b;
  (void)off;
  return &g_layout;
}
int hexkl_micro_hmx_acc_clear_int32(void) {
  memset(g_acc, 0, sizeof g_acc);
  return 0;
}

/* Weight tile: 512 bytes = 32k x 32n int4 in the device's WH layout
   (htp_wh_layout.h: byte (k/8)*128 + c*4 + k%4, low nibble for k%8 < 4),
   two's complement nibbles. The HMX stub, the HVX GEMM stand-in and the
   reference all read a tile through this one function, so the layout is
   the real one and the three cannot disagree on it -- the HVX GEMM's
   whole claim is that it reads the same bytes as the HMX.
   Activation tile: 64 rows x 32 bytes u8. */
static int wh_value(const uint8_t *tile, uint32_t k, uint32_t c) {
  const uint32_t byte = (k / 8u) * 128u + c * 4u + (k % 4u);
  const int nib = (tile[byte] >> (((k / 4u) % 2u) ? 4 : 0)) & 0xF;
  return nib >= 8 ? nib - 16 : nib;
}
int hexkl_micro_hmx_mm_u8i4(uint8_t *base, uint32_t act_off, uint32_t w_off) {
  const uint8_t *a = base + act_off;
  const uint8_t *w = base + w_off;
  for (int r = 0; r < 64; ++r)
    for (uint32_t c = 0; c < 32; ++c) {
      int32_t s = 0;
      for (uint32_t k = 0; k < 32; ++k)
        s += (int32_t)a[r * 32 + k] * wh_value(w, k, c);
      g_acc[r][c] += s;
    }
  return 0;
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
#define SCORE_MAX 256u
typedef struct {
  const uint8_t *dst, *src;
  uint32_t bytes, row_size, nrows, idx, reads, waited;
} score_push;
static score_push g_score[SCORE_MAX];
static uint32_t g_score_n, g_score_on, g_score_bad, g_score_waits,
  g_score_vtcm_reads;
static const uint8_t *g_vtcm_lo, *g_vtcm_hi;
static void score_reset(int on, const uint8_t *vtcm, size_t vtcm_bytes) {
  g_score_n = g_score_bad = g_score_waits = g_score_vtcm_reads = 0;
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

static void gemv_stand_in(const uint8_t *act_ah, uint32_t m, uint32_t k_tiles,
                          const uint8_t *wh, uint32_t n_col, uint32_t nt,
                          int32_t *out) {
  /* rows1 picks between two HVX loops that compute the same int32 sums
     (hvx_gemm_u8i4_wh.c); the stand-in is that sum, so it is the same
     function either way and the caller only records which loop was
     asked for. */
  const uint8_t *src_wh = wh;
  if (g_score_on && in_vtcm(wh)) {
    /* One whole matrix per push (f2): the column's last tile must be
       inside it, and it must have been waited for. */
    score_push *sp = score_find(wh);
    const size_t need = ((size_t)(k_tiles - 1u) * n_col + nt + 1u) * 512u;
    if (!sp || sp->dst != wh || !sp->waited || need > sp->bytes ||
        sp->row_size != n_col * 512u || sp->nrows != k_tiles) {
      if (g_score_bad++ < 4u)
        printf("FEED read not covered: lane=%u nt=%u n_col=%u %s\n", g_lane, nt,
               n_col,
               !sp           ? "no push"
               : !sp->waited ? "push not waited"
                             : "shape");
    } else {
      ++sp->reads;
      ++g_score_vtcm_reads;
      src_wh = sp->src; /* the log names the expert by its arena bytes */
    }
  }
  for (uint32_t r = 0; r < m; ++r)
    for (uint32_t c = 0; c < 32; ++c) {
      int32_t s = 0;
      for (uint32_t kt = 0; kt < k_tiles; ++kt) {
        const uint8_t *tile = wh + ((size_t)kt * n_col + nt) * 512u;
        const uint8_t *arow = act_ah + (size_t)kt * 2048u + r * 32u;
        for (uint32_t k = 0; k < 32; ++k)
          s += (int32_t)arow[k] * wh_value(tile, k, c);
      }
      out[r * 32u + c] = s;
    }
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
#define PF_LANES 6u
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
void hvx_gemm_u8i4_wh_prefetch(const uint8_t *wh, uint32_t n_col, uint32_t nt,
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
void hvx_gemm_u8i4_wh_col_nopf(const uint8_t *act_ah, uint32_t m,
                               uint32_t k_tiles, const uint8_t *wh,
                               uint32_t n_col, uint32_t nt, uint32_t rows1,
                               int32_t *out) {
  int found = 0;
  g_rows1_seen |= 1u << (rows1 != 0u);
  if (g_score_on && in_vtcm(wh)) {
    /* Under the feed no box covers a column; the scoreboard in
       gemv_stand_in holds the read instead. */
    ++g_nopf_n;
    gemv_stand_in(act_ah, m, k_tiles, wh, n_col, nt, out);
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
  gemv_stand_in(act_ah, m, k_tiles, wh, n_col, nt, out);
}
void hvx_gemm_u8i4_wh_col(const uint8_t *act_ah, uint32_t m, uint32_t k_tiles,
                          const uint8_t *wh, uint32_t n_col, uint32_t nt,
                          uint32_t rows1, int32_t *out) {
  ++g_col_n;
  g_rows1_seen |= 1u << (rows1 != 0u);
  gemv_stand_in(act_ah, m, k_tiles, wh, n_col, nt, out);
}
int hexkl_micro_hmx_acc_read_int32(uint8_t *base, uint32_t cfg, uint32_t off) {
  (void)cfg;
  memcpy(base + off, g_acc, sizeof g_acc);
  return 0;
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
void hexkl_dma_ring_push2d(void *dst, const void *src, uint32_t ds, uint32_t ss,
                           uint32_t rs, uint32_t nrows, int sv, int dv) {
  (void)sv;
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
          sp->reads != sp->row_size / 512u) {
        if (g_score_bad++ < 4u)
          printf("FEED push onto a live slab: +%td (previous matrix %u of %u "
                 "columns read)\n",
                 d0 - g_vtcm_lo, sp->reads, sp->row_size / 512u);
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
      sp->idx = g_stub_idx - 1u; /* next_idx was taken just before */
      sp->reads = 0u;
      sp->waited = 0u;
    }
  }
  for (uint32_t r = 0; r < nrows; ++r)
    memcpy((uint8_t *)dst + (size_t)r * ds,
           (const uint8_t *)src + (size_t)r * ss, rs);
}

/* ---- quant / dequant / swiglu ---- */
void hvx_quant_rows_u8_params(const float *x, uint32_t m, uint32_t mp,
                              uint32_t k, float *scale, int32_t *zp,
                              hvx_worker_pool *p) {
  (void)p;
  for (uint32_t r = 0; r < mp; ++r) {
    float lo = 0.f, hi = 0.f;
    if (r < m)
      for (uint32_t j = 0; j < k; ++j) {
        float v = x[(size_t)r * k + j];
        if (v < lo)
          lo = v;
        if (v > hi)
          hi = v;
      }
    float s = (hi - lo) / 255.f;
    if (s <= 0.f)
      s = 1e-8f;
    scale[r] = s;
    long z = lrintf(-lo / s);
    if (z < 0)
      z = 0;
    if (z > 255)
      z = 255;
    zp[r] = (int32_t)z;
  }
}
int hvx_quant_pack_u8_ah_mapped(const float *x, const uint32_t *map, uint32_t m,
                                uint32_t mp, uint32_t k, const float *scale,
                                const int32_t *zp, uint8_t *out,
                                hvx_worker_pool *p) {
  (void)p;
  /* Tiles run (row_block, inner_tile) at a 2048-byte stride, so a caller
     passing more than 64 rows writes several row blocks. The kernel now
     packs the whole activation in one call, so this can no longer assume
     one block the way it did. */
  const uint32_t kt_n = k / 32u;
  memset(out, 0, (size_t)mp * k);
  for (uint32_t r = 0; r < m; ++r)
    for (uint32_t kt = 0; kt < kt_n; ++kt)
      for (uint32_t j = 0; j < 32; ++j) {
        const size_t sr = map ? map[r] : r;
        long q = lrintf(x[sr * k + kt * 32 + j] / scale[r]) + zp[r];
        if (q < 0)
          q = 0;
        if (q > 255)
          q = 255;
        out[(size_t)(r / 64u) * kt_n * 2048u + (size_t)kt * 2048u +
            (size_t)(r % 64u) * 32u + j] = (uint8_t)q;
      }
  return 0;
}

int hvx_quant_pack_u8_ah(const float *x, uint32_t m, uint32_t mp, uint32_t k,
                         const float *scale, const int32_t *zp, uint8_t *out,
                         hvx_worker_pool *p) {
  return hvx_quant_pack_u8_ah_mapped(x, NULL, m, mp, k, scale, zp, out, p);
}
/* A row range, same scalar formula as the mapped stand-in above so the
   two cannot drift: the kernel's units are 16-row quarters of a block. */
void hvx_quant_pack_u8_ah_rows(const float *x, const uint32_t *map, uint32_t m0,
                               uint32_t m1, uint32_t k, const float *scale,
                               const int32_t *zp, uint8_t *out) {
  const uint32_t kt_n = k / 32u;
  for (uint32_t r = m0; r < m1; ++r)
    for (uint32_t kt = 0; kt < kt_n; ++kt)
      for (uint32_t j = 0; j < 32; ++j) {
        const size_t sr = map ? map[r] : r;
        long q = lrintf(x[sr * k + kt * 32 + j] / scale[r]) + zp[r];
        if (q < 0)
          q = 0;
        if (q > 255)
          q = 255;
        out[(size_t)(r / 64u) * kt_n * 2048u + (size_t)kt * 2048u +
            (size_t)(r % 64u) * 32u + j] = (uint8_t)q;
      }
}
void hvx_dequant_acc_tile_to_f32(const int32_t *tile, uint32_t stride,
                                 uint32_t m, const float *as, const int32_t *az,
                                 const int32_t *cs, const float *ws,
                                 const float *bias, float *out,
                                 uint32_t ostride, int accumulate) {
  for (uint32_t r = 0; r < m; ++r)
    for (uint32_t c = 0; c < 32; ++c) {
      float v = ((float)(tile[(size_t)r * stride + c] - az[r] * cs[c])) *
                  as[r] * ws[c] +
                bias[c];
      if (accumulate)
        out[(size_t)r * ostride + c] += v;
      else
        out[(size_t)r * ostride + c] = v;
    }
}
/* The pool runs everything on the caller, which is what its own NULL path
   does for n_units <= 1. Doing it here rather than passing NULL keeps the
   kernel's call sites exercised: the range arithmetic they hand the worker
   is part of what this check is for. */
void hvx_worker_pool_run(hvx_worker_pool *pool, hvx_worker_pool_func func,
                         void *ctx, uint32_t n_units) {
  (void)pool;
  /* The device's lane count, n = min(n_units, 6), one lane after another,
     so each lane's slice and its l2fetch boxes are checked as the device
     splits them. The degenerate branch is func(1, 0): writing it the way
     the real function's used to be -- func(n_units, 0, ctx) -- is what this
     check caught first time out; that form means "worker 0 of n_units"
     and does 1/n_units of the work. */
  const uint32_t n = n_units < PF_LANES ? n_units : PF_LANES;
  if (n <= 1u) {
    g_lane = 0;
    func(1u, 0, ctx);
    return;
  }
  for (uint32_t i = 0; i < n; ++i) {
    g_lane = i;
    func(n, i, ctx);
  }
  g_lane = 0;
}
/* submit runs the job to completion on the spot and wait is a no-op: the
   harness cannot exercise the overlap, only that every job is submitted
   with the right buffer and retired before that buffer is reused -- which
   a job that ran late would show as a wrong result on device and cannot
   show here. The device tests are where the ordering is checked. */
void hvx_worker_pool_submit(hvx_worker_pool *pool, hvx_worker_pool_func func,
                            void *ctx, uint32_t n_units) {
  (void)pool;
  if (n_units != 0u)
    func(1u, 0, ctx);
}
void hvx_worker_pool_wait(hvx_worker_pool *pool) { (void)pool; }
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
/* Scalar stand-in for the fused gate/up dequant + SwiGLU job. Same policy
   as the batch dequant stand-in below: a loop over the per-tile stand-in,
   so what is checked is the kernel's pairing arithmetic -- that staged
   slot j is gate column g0+j and slot n_pairs+j the up column opposite it
   -- not the arithmetic of SwiGLU, which the device gates cover bit for
   bit. */
void hvx_dq_swiglu_worker(uint32_t n_threads, uint32_t i, void *vjob) {
  const hvx_dq_swiglu_job *c = (const hvx_dq_swiglu_job *)vjob;
  (void)n_threads;
  (void)i;
  float gt[64 * 32], ut[64 * 32];
  for (uint32_t j = 0; j < c->n_pairs; ++j) {
    const uint32_t cg = (c->g0 + j) * 32u, cu = c->inter + cg;
    hvx_dequant_acc_tile_to_f32(
      (const int32_t *)(c->tiles_base + (size_t)j * c->tile_stride),
      c->row_stride, c->m_count, c->act_scale, c->act_zp, c->colsum_w + cg,
      c->w_scale + cg, c->bias + cg, gt, 32u, 0);
    hvx_dequant_acc_tile_to_f32(
      (const int32_t *)(c->tiles_base +
                        (size_t)(c->n_pairs + j) * c->tile_stride),
      c->row_stride, c->m_count, c->act_scale, c->act_zp, c->colsum_w + cu,
      c->w_scale + cu, c->bias + cu, ut, 32u, 0);
    for (uint32_t r = 0; r < c->m_count; ++r)
      for (uint32_t k = 0; k < 32u; ++k) {
        const float g = gt[r * 32u + k];
        c->dst[(size_t)r * c->dst_stride + cg + k] =
          g / (1.f + expf(-g)) * ut[r * 32u + k];
      }
  }
}
/* The tail path calls the pooled pair function directly (pool NULL, one
   pair); it is the job above run synchronously, as on device. */
void hvx_dequant_swiglu_acc_tiles_to_f32(
  const uint8_t *tiles_base, uint32_t tile_stride, uint32_t n_pairs,
  uint32_t g0, uint32_t row_stride, uint32_t m_count, const float *act_scale,
  const int32_t *act_zp, const int32_t *colsum_w, const float *w_scale,
  const float *bias, uint32_t inter, float *dst, uint32_t dst_stride,
  hvx_worker_pool *pool) {
  (void)pool;
  hvx_dq_swiglu_job jb;
  jb.tiles_base = tiles_base;
  jb.tile_stride = tile_stride;
  jb.n_pairs = n_pairs;
  jb.g0 = g0;
  jb.row_stride = row_stride;
  jb.m_count = m_count;
  jb.act_scale = act_scale;
  jb.act_zp = act_zp;
  jb.colsum_w = colsum_w;
  jb.w_scale = w_scale;
  jb.bias = bias;
  jb.inter = inter;
  jb.dst = dst;
  jb.dst_stride = dst_stride;
  hvx_dq_swiglu_worker(1u, 0u, &jb);
}
/* Scalar stand-in for the pooled batch dequant job. Deliberately a loop
   over the per-tile stand-in above, exactly as the real one is a pooled
   loop over the real per-tile kernel: what this harness can check is the
   kernel's batching arithmetic -- which tile lands at which staged slot,
   which column it carries -- and a stand-in that recomputed the dequant
   itself would check the stand-in. */
void hvx_dq_tiles_worker(uint32_t n_threads, uint32_t i, void *vjob) {
  const hvx_dq_tiles_job *c = (const hvx_dq_tiles_job *)vjob;
  (void)n_threads;
  (void)i;
  for (uint32_t j = 0; j < c->n_tiles; ++j) {
    const uint32_t c0 = (c->nt0 + j) * 32u;
    const int32_t *tile =
      (const int32_t *)(c->tiles_base + (size_t)j * c->tile_stride);
    float *out =
      (c0 < c->split) ? (c->dst_a + c0) : (c->dst_b + (c0 - c->split));
    hvx_dequant_acc_tile_to_f32(tile, c->row_stride, c->m_count, c->act_scale,
                                c->act_zp, c->colsum_w + c0, c->w_scale + c0,
                                c->bias + c0, out, c->dst_stride, 0);
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
  const uint32_t kt_n = w->K / 32u, nt_n = w->N / 32u;
  for (uint32_t nt = 0; nt < nt_n; ++nt)
    for (uint32_t c = 0; c < 32; ++c) {
      int32_t s = 0;
      for (uint32_t kt = 0; kt < kt_n; ++kt)
        for (uint32_t k = 0; k < 32; ++k)
          s +=
            (int32_t)a_u8[kt * 32 + k] *
            wh_value((const uint8_t *)w->nib + (size_t)(kt * nt_n + nt) * 512u,
                     k, c);
      uint32_t col = nt * 32 + c;
      out[col] =
        ((float)(s - a_zp * w->cs[col])) * a_scale * w->ws[col] + w->bias[col];
    }
}
static void quant_row(const float *x, uint32_t k, uint8_t *q, float *scale,
                      int32_t *zp) {
  float lo = 0.f, hi = 0.f;
  for (uint32_t j = 0; j < k; ++j) {
    if (x[j] < lo)
      lo = x[j];
    if (x[j] > hi)
      hi = x[j];
  }
  float s = (hi - lo) / 255.f;
  if (s <= 0.f)
    s = 1e-8f;
  *scale = s;
  long z = lrintf(-lo / s);
  if (z < 0)
    z = 0;
  if (z > 255)
    z = 255;
  *zp = (int32_t)z;
  for (uint32_t j = 0; j < k; ++j) {
    long v = lrintf(x[j] / s) + *zp;
    if (v < 0)
      v = 0;
    if (v > 255)
      v = 255;
    q[j] = (uint8_t)v;
  }
}

/* The layer, one expert and one row at a time: quantize the row, gate_up,
   SwiGLU, requantize, down, then the routing multiply and the add into the
   token's output row -- in expert order, rows in order, like the kernel. */
static void reference_layer(uint32_t M, uint32_t K, uint32_t inter,
                            uint32_t N_out, uint32_t NE, const W *wg,
                            const W *wd, const float *act, const uint32_t *ridx,
                            const uint32_t *rc_, const float *rw, float *want) {
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
      for (uint32_t j = 0; j < inter; ++j)
        mid[j] = gu[j] / (1.f + expf(-gu[j])) * gu[inter + j];
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
  s->K = K;
  s->N = N;
  s->wh_bytes = (uint8_t *)w->nib;
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
static uint64_t g_feed_pushes, g_feed_waits;
static int run_m1_case(const char *shape, uint32_t M, uint32_t K,
                       uint32_t inter, uint32_t N_out, uint32_t NE,
                       const uint32_t *rc_, uint32_t slot0, uint8_t *vtcm,
                       size_t vtcm_bytes, hexkl_moe_scratch *scratch,
                       uint32_t gemv_flags, float *ref, int have_ref) {
  const uint32_t lead_kb = hexkl_moe_flags_lead_kb(gemv_flags);
  const uint32_t rows1 = hexkl_moe_flags_rows1(gemv_flags);
  const uint32_t feed = hexkl_moe_flags_feed(gemv_flags);
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
  r = hexkl_mm_u8i4_moe_layer_run(&g_tbl, vtcm, vtcm_bytes, vtcm_bytes, M, K,
                                  inter, N_out, NE, hg, hd, ridx, rc_, rw, act,
                                  out_m1, NULL, scratch, gemv_flags);
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
          fed != feed || same != 0;

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
  if (feed != 0u) {
    uint32_t unread = 0;
    for (uint32_t k = 0; k < g_score_n; ++k)
      unread += g_score[k].reads != g_score[k].row_size / 512u;
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
     37-row fixture is. */
  float *want = (float *)malloc(sizeof(float) * M * N_out);
  reference_layer(M, K, inter, N_out, NE, wg, wd, act, ridx, rc_, rw, want);
  double worst = 0.0;
  const uint32_t bad = count_mismatches(out_m1, want, M * N_out, &worst);
  fail |= (bad != 0u);

  printf("M1 GEMV shape=%s M=%u lead=%u rows1=%u feed=%u f32 memcmp=%d i32 "
         "exact=%s blocks=%llu dma_kb=%llu (path=%llu, hmx blocks=%llu, "
         "gate_up tiles=%u, ref mismatches=%u worst_rel=%g)\n",
         shape, M, (unsigned)lead_kb, (unsigned)rows1, (unsigned)feed,
         same != 0, i32_bad == 0u ? "yes" : "NO", (unsigned long long)blocks,
         (unsigned long long)dma_kb, (unsigned long long)path,
         (unsigned long long)blocks_hmx, i32_seen, bad, worst);

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
   downs first, and a fifth expert's down reuses a slot. The active
   experts are spread over the table with empties between them, at a
   stride coprime to the count so no two land on one slot. */
static int run_m1_cases(uint8_t *vtcm, size_t vtcm_bytes,
                        hexkl_moe_scratch *scratch) {
  static const uint32_t counts[5][10] = {{1, 1, 1, 1},
                                         {2, 2, 1, 1, 1, 1},
                                         {4, 3, 2, 1, 1, 1, 1, 1, 1, 1},
                                         {1, 1, 1},
                                         {1, 1, 1, 1, 1}};
  static const uint32_t Ms[5] = {1, 2, 4, 1, 1};
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
    for (int c = 0; c < 5; ++c) {
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
        }
        fail |=
          run_m1_case(shape ? "real" : "tiny", Ms[c], K, inter, N_out, NE, rc_,
                      64u, vtcm, vtcm_bytes, scratch, flags, ref, cfg != 0u);
      }
      free(ref);
    }
    free(rc_);
  }
  if (fail)
    printf("M1 GEMV PATH DIFFERS FROM HMX PATH\n");
  else
    printf("M1 GEMV PATH BIT-IDENTICAL TO HMX PATH (M=1,2,4, 3 and 5 experts; "
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

int main(void) {
  const uint32_t M = 37, K = 64, inter = 32, N_out = 64, NE = 5;
  static uint8_t vtcm[8u << 20];

  hexkl_moe_layout L;
  int rc = hexkl_mm_u8i4_moe_layout(K, inter, N_out, sizeof vtcm, &L);
  printf(
    "layout rc=%d total=%u (act %u gu %u dn %u gate %u mid %u stage %u res "
    "%u)\n",
    rc, L.total, L.act_off, L.w_gu_off, L.w_dn_off, L.gate_off, L.mid_off,
    L.result_off, L.res_f32_off);
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
  for (uint32_t i = 0; i < n_rows; ++i) {
    ridx[i] = rnd() % M;
    rw[i] = 0.1f + 0.9f * ((float)(rnd() % 100u) / 100.f);
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
  reference_layer(M, K, inter, N_out, NE, wg, wd, act, ridx, rc_, rw, want);

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
    printf("M=37 flag on      : rc=%d HMX blocks=%llu dma_kb=%llu (off %llu) "
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
  printf(fail ? "\nFAIL\n" : "\nALL CHECKS PASS\n");
  free(g_gemv_log);
  hexkl_moe_scratch_free(&scratch);
  return fail;
}
