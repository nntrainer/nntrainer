// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   dma_trace_host_check.c
 * @date   21 Sep 2026
 * @brief  Scripted push/wait sequence against hexkl_dma_trace's arithmetic
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * The ring is a scripted done[] array and the clock is the script's own
 * numbers, so every expected value below is hand arithmetic on the
 * timeline, not a re-run of the code under test. What this pins: the
 * completion bracket (lo = previous point, hi = the point that saw done),
 * the union of intervals (not their sum), the depth-before-push convention
 * and the blocked bit read at wait entry.
 */

#include "hexkl_dma_ring.h"
#include "hexkl_dma_trace.h"

#include <stdio.h>
#include <string.h>

/* ---- the scripted ring ---- */
static int g_done[8];
int hexkl_dma_ring_is_done(uint32_t idx) { return g_done[idx & 7u]; }

static int g_fail;
#define EXPECT_EQ(what, got, want)                                             \
  do {                                                                         \
    if ((uint32_t)(got) != (uint32_t)(want)) {                                 \
      printf("FAIL %s: got %u want %u\n", what, (unsigned)(got),               \
             (unsigned)(want));                                                \
      g_fail = 1;                                                              \
    }                                                                          \
  } while (0)

int main(void) {
  hexkl_dma_trace_summary s;
  hexkl_dma_trace_reset(1000); /* t0 = 1000 ticks: every time below is
                                  relative to that */

  /* Timeline (ticks since t0):
       0  push d0  gate e0  8 KiB x 64 @ 56 KiB     depth before: 0
      10  push d1  up   e0                         depth before: 1
      20  push d2  down e0                         depth before: 2
      30  sample: d0 reads done                    -> d0 in [20, 30]
      30  wait on d2: not done at entry -> blocked
      50  wait exits; d1, d2 read done             -> d1, d2 in [30, 50]
      60  push d3  act  e1, ring seen empty        depth before: 0
      70  wait on d3: done already -> not blocked  -> d3 in [60, 70]
      70  wait exits
      80  finish */
  hexkl_dma_trace_push(1000, 0, HEXKL_DMA_KIND_GATE, 0, 0, 8192, 64, 57344);
  hexkl_dma_trace_push(1010, 1, HEXKL_DMA_KIND_UP, 0, 0, 8192, 64, 57344);
  hexkl_dma_trace_push(1020, 2, HEXKL_DMA_KIND_DOWN, 0, 0, 16384, 56, 32768);
  g_done[0] = 1;
  hexkl_dma_trace_sample(1030);
  hexkl_dma_trace_wait_begin(1030, 2, HEXKL_DMA_SITE_DN);
  g_done[1] = 1;
  g_done[2] = 1;
  hexkl_dma_trace_wait_end(1050);
  hexkl_dma_trace_push(1060, 3, HEXKL_DMA_KIND_ACT, 1, 0, 16384, 8, 16384);
  g_done[3] = 1;
  hexkl_dma_trace_wait_begin(1070, 3, HEXKL_DMA_SITE_ACT);
  hexkl_dma_trace_wait_end(1070);
  hexkl_dma_trace_finish(1080, &s);

  const hexkl_dma_trace *t = hexkl_dma_trace_get();
  EXPECT_EQ("n_push", t->n_push, 4);
  EXPECT_EQ("d0.depth", t->push[0].depth, 0);
  EXPECT_EQ("d1.depth", t->push[1].depth, 1);
  EXPECT_EQ("d2.depth", t->push[2].depth, 2);
  EXPECT_EQ("d3.depth", t->push[3].depth, 0);
  EXPECT_EQ("d0.lo", t->push[0].t_done_lo, 20);
  EXPECT_EQ("d0.hi", t->push[0].t_done_hi, 30);
  EXPECT_EQ("d1.lo", t->push[1].t_done_lo, 30);
  EXPECT_EQ("d1.hi", t->push[1].t_done_hi, 50);
  EXPECT_EQ("d2.lo", t->push[2].t_done_lo, 30);
  EXPECT_EQ("d2.hi", t->push[2].t_done_hi, 50);
  EXPECT_EQ("d3.lo", t->push[3].t_done_lo, 60);
  EXPECT_EQ("d3.hi", t->push[3].t_done_hi, 70);
  EXPECT_EQ("d0.bytes", t->push[0].bytes, 8192u * 64u);
  EXPECT_EQ("n_wait", t->n_wait, 2);
  EXPECT_EQ("w0.blocked", t->wait[0].blocked, 1);
  EXPECT_EQ("w0.t_out", t->wait[0].t_out, 50);
  EXPECT_EQ("w1.blocked", t->wait[1].blocked, 0);

  EXPECT_EQ("s.n_desc", s.n_desc, 4);
  EXPECT_EQ("s.n_wait", s.n_wait, 2);
  EXPECT_EQ("s.n_blocked", s.n_blocked, 1);
  EXPECT_EQ("s.depth_max", s.depth_max, 3);
  EXPECT_EQ("s.wait", s.wait, 20);
  EXPECT_EQ("s.wait_act", s.wait_act, 0);
  /* lo edges: [0,20] [10,30] [20,30] [60,60] -> [0,30] + [60,60] = 30.
     hi edges: [0,30] [10,50] [20,50] [60,70] -> [0,50] + [60,70] = 60.
     The sum of the hi intervals would be 30+40+30+10 = 110: the union is
     what "engine busy" means. */
  EXPECT_EQ("s.busy_lo", s.busy_lo, 30);
  EXPECT_EQ("s.busy_hi", s.busy_hi, 60);
  EXPECT_EQ("s.first_ready", s.first_ready, 50); /* max hi of d0, d1 */
  EXPECT_EQ("s.last_issue", s.last_issue, 20);   /* d2, the last weight */

  /* Serialize / decode round trip on the same tables. */
  static uint32_t words[HEXKL_DMA_TRACE_MAX_WORDS];
  const uint32_t n =
    hexkl_dma_trace_serialize(words, HEXKL_DMA_TRACE_MAX_WORDS);
  EXPECT_EQ("ser.n", n,
            HEXKL_DMA_TRACE_HDR_WORDS + 4u * HEXKL_DMA_TRACE_PUSH_WORDS +
              2u * HEXKL_DMA_TRACE_WAIT_WORDS);
  EXPECT_EQ("ser.n_push", words[0], 4);
  EXPECT_EQ("ser.t_end", words[5], 80);
  EXPECT_EQ(
    "ser.d2.done_hi",
    words[HEXKL_DMA_TRACE_HDR_WORDS + 2u * HEXKL_DMA_TRACE_PUSH_WORDS + 11u],
    50);
  EXPECT_EQ(
    "ser.w0.blocked",
    words[HEXKL_DMA_TRACE_HDR_WORDS + 4u * HEXKL_DMA_TRACE_PUSH_WORDS + 4u], 1);
  EXPECT_EQ("ser.too_small", hexkl_dma_trace_serialize(words, n - 1u), 0);

  /* A push past the table is counted, never written, and the busy
     numbers are refused. */
  hexkl_dma_trace_reset(0);
  for (uint32_t k = 0; k < HEXKL_DMA_TRACE_MAX_PUSH + 3u; ++k) {
    hexkl_dma_trace_push(k, k, HEXKL_DMA_KIND_GATE, 0, 0, 16, 1, 16);
  }
  hexkl_dma_trace_finish(1000, &s);
  EXPECT_EQ("overflow.n_push", hexkl_dma_trace_get()->n_push,
            HEXKL_DMA_TRACE_MAX_PUSH);
  EXPECT_EQ("overflow.n_desc", s.n_desc, HEXKL_DMA_TRACE_MAX_PUSH + 3u);
  EXPECT_EQ("overflow.busy_hi", s.busy_hi, 0);
  EXPECT_EQ("overflow.last_issue", s.last_issue, 0);

  /* Past the ring size a record's slot is reused, so its done bit is no
     longer its own: the watermark takes the ring's reclaim (push2d waits
     for the oldest before reusing) as completion. 300 pushes at t = k
     with every bit reading 0: record k is done at the push of k + 256. */
  memset(g_done, 0, sizeof(g_done));
  hexkl_dma_trace_reset(0);
  for (uint32_t k = 0; k < 300u; ++k) {
    hexkl_dma_trace_push(k, k, HEXKL_DMA_KIND_ACT, 0, 0, 16, 1, 16);
  }
  hexkl_dma_trace_finish(1000, &s);
  t = hexkl_dma_trace_get();
  EXPECT_EQ("reclaim.d0.hi", t->push[0].t_done_hi, 256);
  EXPECT_EQ("reclaim.d0.lo", t->push[0].t_done_lo, 255);
  EXPECT_EQ("reclaim.d43.hi", t->push[43].t_done_hi, 299);
  EXPECT_EQ("reclaim.d44.hi", t->push[44].t_done_hi, 1000);
  EXPECT_EQ("reclaim.depth_max", s.depth_max, 256);
  EXPECT_EQ("reclaim.busy_hi", s.busy_hi, 1000);

  if (g_fail) {
    return 1;
  }
  printf("DMA TRACE ARITHMETIC OK\n");
  return 0;
}
