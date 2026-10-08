// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   nntr_hvx_fc_q4.c
 * @date   29 Sep 2026
 * @brief  DSP-side test entries for #132 PR 2: the CPU-exact M=1 Q4_0 FC
 *         (hvx_q4_gemv_f32.c) with its rate, and the CPU-order Q8_0
 *         quantizer, SwiGLU and argmax
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * No model path calls these (plan 132 Part A). q4m1_register copies one
 * Q4M1 weight to the DSP heap -- at most one lm_head slice of 18 MiB or
 * one FC of 14 MiB at a time in the tests and the shadow, inside the
 * ~100 MiB the loaded app has (doc 46 section 41, plan 132 section 0.4)
 * -- and cleans it out of the DSP caches, so the rate test's DMA may
 * read it around L2 (src_bypass = 1). fc_q4m1_f32 splits the 32-column
 * groups over @a lanes pool threads; with the VTCM feed each thread
 * double-buffers its next group into its own VTCM slice on its own DMA
 * engine while it computes the current one (weight DMA hidden behind
 * compute). q8_quant_f32 runs the FC's own vector quantizer; SwiGLU and
 * argmax run the scalar specs themselves (m1_ops_det.h compiled for the
 * DSP is the kernel, so G1 checks the DSP's scalar IEEE unit and the
 * compiler, not a second implementation).
 * ponytail: scalar, so a 7168-element SwiGLU pays 7168 integer divides on
 * one thread (test entries only); plan 132 section 3.2's HVX SwiGLU and
 * argmax come with Part B if decision D keeps the dense FFN and lm_head on
 * the NPU.
 *
 * A bad shape or handle is AEE_EINVALIDFORMAT / AEE_EBADITEM, never
 * AEE_EBADPARM, which stays the stale-skel symptom (rule 3); a DMA that
 * never completes is AEE_EEXPIRED (its output and timing are void).
 *
 * ponytail: the quantized activation and the per-lane scratch are static,
 * so two fc_q4m1_f32 calls must not overlap -- true while nntrainer opens
 * one session per process and calls it from one thread (the tests, the
 * shadow, and the graph's forward, which runs its ops one at a time). A
 * second session is a second PD with its own statics.
 *
 * [#132 Part B] nntr_hvx_fc_q4m1_run is the one lane runner: the test
 * entry and the resident graph's FC / DENSE_FFN / LM_HEAD ops
 * (nntr_hvx_graph.c's callback) both call it. Its feeds: VTCM by DMA
 * (dst_bypass 1) below the HMX config block, or -- lifted from the #178
 * probe -- the same double buffer in a 2 MiB DSP heap scratch with
 * dst_bypass 0 (the DMA writes through the L2, where the GEMV reads it),
 * for a session whose VTCM does not hold two groups per lane (#178: the
 * second session gets none). Address space: the Q4M1 weights are DSP heap
 * copies (q4m1_register) or [E3] borrowed from an attached arena
 * (q4m1_attach, no heap), 80 slots; the whole LFM2.5 FC set is 383 MiB,
 * which only a session without the 3840 MiB MoE arena can hold (plan 132
 * Part B section 3.2: S2's arena, attached, so S2's heap stays small) --
 * one session holding both is a host-only configuration. The L2 scratch
 * is 2 MiB of heap on first use, freed in close().
 */

#include <stdlib.h>
#include <string.h>

#include <AEEStdErr.h>
#include <HAP_farf.h>
#include <HAP_perf.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <remote.h>
#if defined(__hexagon__)
#include <qurt_memory.h>
#endif

#include "hexkl_dma_ring.h"
#include "hexkl_probe.h"
#include "hvx_m1_ops_f32.h"
#include "hvx_q4_gemv_f32.h"
#include "hvx_rmsnorm_rows_f32.h"
#include "hvx_softcap_f32.h"
#include "hvx_worker_pool.h"
#include "m1_ops_det.h"
#include "nntr_hvx.h"
#include "nntr_hvx_session.h"
#include "q4_gemv_cpu_det.h"

/** @brief Largest K the static scratch serves (the model's is 7168). */
#define FC_Q4_MAX_K 8192u
/** @brief Pool threads the static per-thread state serves. */
#define FC_Q4_MAX_LANES 8u
/** @brief fc_q4m1_f32's variant word: bit 16 the VTCM feed, bit 17 the L2
 *  feed (not both); every other bit must be 0 (the kernel variants of the
 *  first sitting are gone). */
#define FC_Q4_FEED_VTCM (1u << 16)
#define FC_Q4_FEED_L2 (1u << 17)
/** @brief [#194 L1] fc_q4m1_f32's variant and the runner's feed word: bit
 *  18 the native pair (hvx_q4m1_prep_vec, hvx_q4m1_gemv_groups_native)
 *  beside either feed. */
#define FC_Q4_NATIVE (1u << 18)
/** @brief The L2 feed's scratch: 2 x 129 KiB x 6 lanes at K = 7168 fits. */
#define FC_Q4_L2_BYTES (2u << 20)
/** @brief The graph's lanes per feed (#178 set3, L2-fed: 26 GB/s at K =
 *  7168 and 52 at K = 2048 with 3 lanes, the best of 1..6; VTCM-fed: 6,
 *  45.5 GB/s). ponytail: one count per feed, not per K; the upgrade is a
 *  per-shape table from the E sitting's profile lines. */
#define FC_Q4_GRAPH_LANES_VTCM 6u
#define FC_Q4_GRAPH_LANES_L2 3u
/** @brief [#132 Part B E5f] The L2 feed's lanes for K <= 2048: #178's
 *  ladder read 52 GB/s at 6 lanes there (26 at K = 7168, whose best is
 *  FC_Q4_GRAPH_LANES_L2); E5d's token ran every FC at 3. The op's feed
 *  word may name others (HTP_GRAPH_FEED_LANES_*). */
#define FC_Q4_GRAPH_LANES_L2_SMALL_K 6u
/** @brief fc_q4m1_f32's stats words. */
#define FC_Q4_STATS 8

static int8_t g_q[FC_Q4_MAX_K] __attribute__((aligned(128)));
static int32_t g_s8[FC_Q4_MAX_K / 32u], g_ma[FC_Q4_MAX_K / 32u],
  g_ea[FC_Q4_MAX_K / 32u];
static float g_df[FC_Q4_MAX_K / 32u];
static uint16_t g_d[FC_Q4_MAX_K / 32u];
static hexkl_dma_desc2d g_desc[FC_Q4_MAX_LANES][2]
  __attribute__((aligned(128)));

void nntr_hvx_q4m1_free_all(nntr_hvx_session *s) {
  for (uint32_t i = 0; i < NNTR_HVX_Q4M1_SLOTS; ++i) {
    if (!s->q4m1[i].borrowed) {
      free(s->q4m1[i].w);
    }
    s->q4m1[i].w = NULL;
    s->q4m1[i].borrowed = 0;
  }
  free(s->fc_l2);
  s->fc_l2 = NULL;
}

int nntr_hvx_q4m1_register(remote_handle64 handle, uint32 K, uint32 N,
                           const uint8 *w, int wLen, uint32 *h) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  if (K == 0u || K % 64u != 0u || K > FC_Q4_MAX_K || N == 0u ||
      N % Q4M1_GROUP != 0u || (size_t)wLen != q4m1_bytes(K, N)) {
    FARF(ERROR, "q4m1_register: bad shape (K=%u N=%u bytes=%d)", (unsigned)K,
         (unsigned)N, wLen);
    return AEE_EINVALIDFORMAT;
  }
  for (uint32_t i = 0; i < NNTR_HVX_Q4M1_SLOTS; ++i) {
    if (s->q4m1[i].w == NULL) {
      uint8_t *p = (uint8_t *)memalign(128, (size_t)wLen);
      if (!p) {
        FARF(ERROR, "q4m1_register: no heap for %d bytes", wLen);
        return AEE_ENOMEMORY;
      }
      memcpy(p, w, (size_t)wLen);
#if defined(__hexagon__)
      /* the DMA feed reads around the DSP L2 (src_bypass): DDR must hold
         what this memcpy wrote (the in-process host build has no cache) */
      qurt_mem_cache_clean((qurt_addr_t)p, (qurt_size_t)wLen,
                           QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
#endif
      s->q4m1[i].w = p;
      s->q4m1[i].K = K;
      s->q4m1[i].N = N;
      s->q4m1[i].borrowed = 0;
      *h = i;
      return AEE_SUCCESS;
    }
  }
  return AEE_EBADITEM;
}

int nntr_hvx_q4m1_attach(remote_handle64 handle, uint32 arena, uint32 off,
                         uint32 K, uint32 N, uint32 *h) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s || !h) {
    return AEE_EBADPARM;
  }
  if (K == 0u || K % 64u != 0u || K > FC_Q4_MAX_K || N == 0u ||
      N % Q4M1_GROUP != 0u || off % 128u != 0u ||
      arena >= NNTR_HVX_MAX_ARENAS || s->arenas[arena].va == NULL ||
      (uint64_t)off + q4m1_bytes(K, N) > s->arenas[arena].bytes) {
    FARF(ERROR, "q4m1_attach: bad weight (arena=%u off=%u K=%u N=%u)",
         (unsigned)arena, (unsigned)off, (unsigned)K, (unsigned)N);
    return AEE_EINVALIDFORMAT;
  }
  for (uint32_t i = 0; i < NNTR_HVX_Q4M1_SLOTS; ++i) {
    if (s->q4m1[i].w == NULL) {
      /* no copy and no cache maintenance: the ARM wrote the uncached
         arena before this call and the DSP never writes it, so no line
         of it is dirty here when the feed's DMA reads DDR around the L2 */
      s->q4m1[i].w = s->arenas[arena].va + off;
      s->q4m1[i].K = K;
      s->q4m1[i].N = N;
      s->q4m1[i].borrowed = 1;
      *h = i;
      return AEE_SUCCESS;
    }
  }
  return AEE_EBADITEM;
}

int nntr_hvx_q4m1_borrows(const nntr_hvx_session *s, const uint8_t *va,
                          uint32_t bytes) {
  for (uint32_t i = 0; i < NNTR_HVX_Q4M1_SLOTS; ++i) {
    if (s->q4m1[i].borrowed && s->q4m1[i].w >= va &&
        s->q4m1[i].w < va + bytes) {
      return 1;
    }
  }
  return 0;
}

int nntr_hvx_q4m1_release(remote_handle64 handle, uint32 h) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  if (h >= NNTR_HVX_Q4M1_SLOTS || s->q4m1[h].w == NULL) {
    return AEE_EBADITEM;
  }
  if (hexkl_graph_uses_q4m1(s->graph, h)) {
    return AEE_EBADSTATE; /* a resident op names it (graph_release first) */
  }
  if (!s->q4m1[h].borrowed) {
    free(s->q4m1[h].w);
  }
  s->q4m1[h].w = NULL;
  s->q4m1[h].borrowed = 0;
  return AEE_SUCCESS;
}

typedef struct {
  const uint8_t *w;
  uint32_t K, G;
  size_t gbytes;
  const hvx_q4m1_act *a;
  float *y;
  uint32_t feed_vtcm;  /**< any DMA feed (VTCM or L2) */
  uint32_t native;     /**< the native GEMV (#194 L1) */
  uint32_t dst_bypass; /**< 1 the VTCM feed, 0 the L2 feed */
  uint8_t *vtcm;
  uint32_t vtcm_per_lane;
  uint32_t lanes_used;           /**< written by lane 0 */
  volatile uint32_t dma_expired; /**< any lane whose DMA never completed */
} fc_ctx;

/** @brief One 1-row DMA of @a bytes into VTCM (or, @a dst_bypass 0, the L2
 *  scratch) on this thread's engine, reading around the DSP L2 (the
 *  registration cleaned the source). */
static void fc_dma_start(hexkl_dma_desc2d *d, void *dst, const void *src,
                         uint32_t bytes, uint32_t dst_bypass) {
  memset(d, 0, sizeof(*d));
  d->desc_size = 1;
  d->desc_type = 9;
  d->src_bypass = 1;
  d->dst_bypass = dst_bypass;
  d->src = (void *)src;
  d->dst = dst;
  d->src_stride = bytes;
  d->dst_stride = bytes;
  d->row_size = bytes;
  d->nrows_lo = 1;
  Q6_dccleaninva_A((void *)d);
  hexkl_dma_start(d);
}

/** @brief Spins on the descriptor's done bit; 0 when it never came. */
static int fc_dma_wait(hexkl_dma_desc2d *d) {
  long guard = 0;
  while (!((volatile hexkl_dma_desc2d *)d)->done) {
    if (guard++ >= 50000000L) {
      return 0;
    }
    hexkl_dma_poll();
  }
  return 1;
}

/** @brief Lane i: groups i, i + n, ...; with the VTCM feed, group g + n
 *  moves into the other half of this lane's slice while g computes. */
static void fc_lane(uint32_t n, uint32_t i, void *v) {
  fc_ctx *c = (fc_ctx *)v;
  if (i == 0u) {
    c->lanes_used = n;
  }
  void (*const gemv)(const uint8_t *, uint32_t, uint32_t, const hvx_q4m1_act *,
                     float *) =
    c->native ? hvx_q4m1_gemv_groups_native : hvx_q4m1_gemv_groups;
  if (!c->feed_vtcm) {
    for (uint32_t g = i; g < c->G; g += n) {
      gemv(c->w + g * c->gbytes, c->K, 1u, c->a, c->y + (size_t)g * Q4M1_GROUP);
    }
    return;
  }
  uint8_t *buf[2] = {c->vtcm + (size_t)i * c->vtcm_per_lane,
                     c->vtcm + (size_t)i * c->vtcm_per_lane + c->gbytes};
  uint32_t cur = 0;
  if (i < c->G) {
    fc_dma_start(&g_desc[i][0], buf[0], c->w + i * c->gbytes,
                 (uint32_t)c->gbytes, c->dst_bypass);
  }
  for (uint32_t g = i; g < c->G; g += n) {
    if (!fc_dma_wait(&g_desc[i][cur])) {
      c->dma_expired = 1u; /* the engine may still be busy: stop here */
      return;
    }
    if (g + n < c->G) {
      fc_dma_start(&g_desc[i][cur ^ 1u], buf[cur ^ 1u],
                   c->w + (g + n) * c->gbytes, (uint32_t)c->gbytes,
                   c->dst_bypass);
    }
    gemv(buf[cur], c->K, 1u, c->a, c->y + (size_t)g * Q4M1_GROUP);
    cur ^= 1u;
  }
}

int nntr_hvx_fc_q4m1_run(nntr_hvx_session *s, uint32_t h, const hvx_q4m1_act *a,
                         float *y, uint32_t lanes, uint32_t feed,
                         uint32_t *lanes_used) {
  if (h >= NNTR_HVX_Q4M1_SLOTS || s->q4m1[h].w == NULL) {
    return AEE_EBADITEM;
  }
  const nntr_hvx_q4m1_slot *w = &s->q4m1[h];
  const uint32_t native = feed & FC_Q4_NATIVE;
  feed &= ~FC_Q4_NATIVE;
  if (lanes == 0u || lanes > FC_Q4_MAX_LANES ||
      (feed & ~(FC_Q4_FEED_VTCM | FC_Q4_FEED_L2)) != 0u ||
      feed == (FC_Q4_FEED_VTCM | FC_Q4_FEED_L2)) {
    return AEE_EINVALIDFORMAT;
  }
  fc_ctx c;
  memset(&c, 0, sizeof(c));
  c.native = native != 0u;
  c.w = w->w;
  c.K = w->K;
  c.G = w->N / Q4M1_GROUP;
  c.gbytes = (size_t)(w->K / 64u) * Q4M1_PAIR_BYTES;
  c.a = a;
  c.y = y;
  c.feed_vtcm = feed != 0u;
  c.dst_bypass = feed == FC_Q4_FEED_VTCM;
  if (feed == FC_Q4_FEED_VTCM) {
    /* two groups per lane below the HMX config block; nothing of the
       session lives there between calls */
    c.vtcm = s->vtcm_base;
    c.vtcm_per_lane = (s->config_off / lanes) & ~127u;
  } else if (feed == FC_Q4_FEED_L2) {
    if (!s->fc_l2) {
      s->fc_l2 = (uint8_t *)memalign(128, FC_Q4_L2_BYTES);
      if (!s->fc_l2) {
        return AEE_ENOMEMORY;
      }
    }
    c.vtcm = s->fc_l2;
    c.vtcm_per_lane = (FC_Q4_L2_BYTES / lanes) & ~127u;
  }
  if (c.feed_vtcm && c.vtcm_per_lane < 2u * c.gbytes) {
    FARF(ERROR, "fc_q4m1: feed %u per lane < 2 x %u", c.vtcm_per_lane,
         (unsigned)c.gbytes);
    return AEE_EINVALIDFORMAT;
  }
  hvx_worker_pool_run(s->quant_pool, fc_lane, &c, lanes);
  if (c.dma_expired) {
    FARF(ERROR, "fc_q4m1: a lane's weight DMA never completed");
    return AEE_EEXPIRED;
  }
  if (lanes_used) {
    *lanes_used = c.lanes_used;
  }
  return AEE_SUCCESS;
}

int nntr_hvx_fc_q4m1_graph(void *ctx, uint32_t h, uint32_t feed,
                           const hvx_q4m1_act *a, float *y) {
  nntr_hvx_session *s = (nntr_hvx_session *)ctx;
  if (h >= NNTR_HVX_Q4M1_SLOTS || s->q4m1[h].w == NULL) {
    return AEE_EBADITEM;
  }
  const size_t gbytes = (size_t)(s->q4m1[h].K / 64u) * Q4M1_PAIR_BYTES;
  const uint32_t per_lane = (s->config_off / FC_Q4_GRAPH_LANES_VTCM) & ~127u;
  const uint32_t native =
    (feed & HTP_GRAPH_FEED_NATIVE) != 0u ? FC_Q4_NATIVE : 0u;
  if ((feed & HTP_GRAPH_FEED_L2) == 0u && per_lane >= 2u * gbytes) {
    return nntr_hvx_fc_q4m1_run(s, h, a, y, FC_Q4_GRAPH_LANES_VTCM,
                                FC_Q4_FEED_VTCM | native, NULL);
  }
  const uint32_t small = s->q4m1[h].K <= 2048u;
  uint32_t lanes =
    small ? HTP_GRAPH_FEED_LANES_SMALL(feed) : HTP_GRAPH_FEED_LANES_LARGE(feed);
  if (lanes == 0u) {
    lanes = small ? FC_Q4_GRAPH_LANES_L2_SMALL_K : FC_Q4_GRAPH_LANES_L2;
  }
  return nntr_hvx_fc_q4m1_run(s, h, a, y, lanes, FC_Q4_FEED_L2 | native, NULL);
}

/** @brief The lm_head's normed input row (static like the rest of this
 *  file's scratch: one call at a time). */
static float g_xn[FC_Q4_MAX_K] __attribute__((aligned(128)));

int nntr_hvx_lm_head_q4m1_f32(remote_handle64 handle, float eps,
                              const float *gamma, int gammaLen, float softcap,
                              const uint32 *h, int hLen, const float *x,
                              int xLen, float *y, int yLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  if (hLen <= 0 || hLen > (int)NNTR_HVX_Q4M1_SLOTS || xLen <= 0 ||
      (uint32_t)xLen > FC_Q4_MAX_K || xLen % 64 != 0 ||
      (gammaLen != 0 && gammaLen != xLen) || softcap < 0.0f) {
    FARF(ERROR, "lm_head_q4m1_f32: bad shape (h=%d x=%d gamma=%d)", hLen, xLen,
         gammaLen);
    return AEE_EINVALIDFORMAT;
  }
  uint64_t n_total = 0;
  for (int p = 0; p < hLen; ++p) {
    if (h[p] >= NNTR_HVX_Q4M1_SLOTS || s->q4m1[h[p]].w == NULL) {
      return AEE_EBADITEM;
    }
    if (s->q4m1[h[p]].K != (uint32_t)xLen) {
      FARF(ERROR, "lm_head_q4m1_f32: slice %d has K %u, x is %d", p,
           (unsigned)s->q4m1[h[p]].K, xLen);
      return AEE_EINVALIDFORMAT;
    }
    n_total += s->q4m1[h[p]].N;
  }
  if (n_total != (uint64_t)yLen) {
    FARF(ERROR, "lm_head_q4m1_f32: slices hold %u rows, y is %d",
         (unsigned)n_total, yLen);
    return AEE_EINVALIDFORMAT;
  }
  /* the output norm, the activation quantized once (the CPU's Q4_0 FC
     order, as the graph's LM_HEAD op), each slice's GEMV into its rows,
     then the softcap over the whole row */
  const float *in = x;
  if (gammaLen != 0) {
    if (hvx_rmsnorm_rows_f32(x, g_xn, 1u, (uint32_t)xLen, (uint32_t)xLen, gamma,
                             eps, NULL) != 0) {
      return AEE_EINVALIDFORMAT;
    }
    in = g_xn;
  }
  hvx_q4m1_act a = {g_q, g_s8, g_ma, g_ea, g_df, g_d};
  hvx_q4m1_prep(in, (uint32_t)xLen, &a);
  float *out = y;
  for (int p = 0; p < hLen; ++p) {
    const int rc = nntr_hvx_fc_q4m1_graph(s, h[p], 0u, &a, out);
    if (rc != AEE_SUCCESS) {
      return rc;
    }
    out += s->q4m1[h[p]].N;
  }
  if (softcap > 0.0f) {
    hvx_softcap_f32(y, (uint32_t)yLen, softcap);
  }
  return AEE_SUCCESS;
}

int nntr_hvx_fc_q4m1_f32(remote_handle64 handle, uint32 h, uint32 variant,
                         uint32 lanes, uint32 reps, const float *x, int xLen,
                         float *y, int yLen, uint32 *stats, int statsLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  if (h >= NNTR_HVX_Q4M1_SLOTS || s->q4m1[h].w == NULL) {
    return AEE_EBADITEM;
  }
  const nntr_hvx_q4m1_slot *w = &s->q4m1[h];
  const uint32_t feed = variant & (FC_Q4_FEED_VTCM | FC_Q4_FEED_L2);
  const uint32_t native = variant & FC_Q4_NATIVE;
  if ((uint32_t)xLen != w->K || (uint32_t)yLen != w->N ||
      statsLen != FC_Q4_STATS || variant != (feed | native) ||
      feed == (FC_Q4_FEED_VTCM | FC_Q4_FEED_L2) || lanes == 0u ||
      lanes > FC_Q4_MAX_LANES || reps == 0u) {
    FARF(ERROR, "fc_q4m1_f32: bad call (x=%d y=%d variant=0x%x lanes=%u)", xLen,
         yLen, (unsigned)variant, (unsigned)lanes);
    return AEE_EINVALIDFORMAT;
  }
  hvx_q4m1_act a = {g_q, g_s8, g_ma, g_ea, g_df, g_d};
  uint64_t prep_us = 0, gemv_us = 0, pcyc = 0;
  uint32_t lanes_used = 0;
  for (uint32_t r = 0; r < reps; ++r) {
    const uint64_t t0 = hexkl_probe_now();
    if (native) {
      hvx_q4m1_prep_vec(x, w->K, &a);
    } else {
      hvx_q4m1_prep(x, w->K, &a);
    }
    const uint64_t t1 = hexkl_probe_now();
    const uint64_t p0 = HAP_perf_get_pcycles();
    const int rc =
      nntr_hvx_fc_q4m1_run(s, h, &a, y, lanes, feed | native, &lanes_used);
    pcyc += HAP_perf_get_pcycles() - p0;
    if (rc != AEE_SUCCESS) {
      return rc;
    }
    gemv_us += hexkl_probe_now() - t1;
    prep_us += t1 - t0;
  }
  stats[0] = (uint32)gemv_us;
  stats[1] = (uint32)prep_us;
  stats[2] = (uint32)(pcyc & 0xffffffffu);
  stats[3] = (uint32)(pcyc >> 32);
  stats[4] = lanes_used;
  stats[5] = reps;
  stats[6] = w->K;
  stats[7] = w->N;
  return AEE_SUCCESS;
}

int nntr_hvx_q8_quant_f32(remote_handle64 handle, const float *x, int xLen,
                          int8 *q, int qLen, uint16 *d, int dLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  if (xLen <= 0 || xLen % 64 != 0 || xLen > (int)FC_Q4_MAX_K || qLen != xLen ||
      dLen != xLen / 32) {
    return AEE_EINVALIDFORMAT;
  }
  /* the FC's own quantizer (hvx_q4m1_prep), not the scalar spec */
  hvx_q4m1_act a = {g_q, g_s8, g_ma, g_ea, g_df, g_d};
  hvx_q4m1_prep(x, (uint32_t)xLen, &a);
  memcpy(q, g_q, (size_t)xLen);
  memcpy(d, g_d, (size_t)dLen * sizeof(uint16_t));
  return AEE_SUCCESS;
}

int nntr_hvx_swiglu_cpu_f32(remote_handle64 handle, const float *y, int yLen,
                            const float *z, int zLen, float *out, int outLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  if (yLen <= 0 || zLen != yLen || outLen != yLen) {
    return AEE_EINVALIDFORMAT;
  }
  /* [#132 E5f] the graph's kernel, so G1 (SmallOpsMatchSpec) holds it and
     its scalar divide against the host's spec */
  hvx_swiglu_cpu_f32(y, z, out, (uint32_t)yLen, s->quant_pool);
  return AEE_SUCCESS;
}

int nntr_hvx_argmax_f32(remote_handle64 handle, const float *x, int xLen,
                        uint32 *idx) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  if (xLen <= 0) {
    return AEE_EINVALIDFORMAT;
  }
  *idx = hvx_argmax_first_f32(x, (uint32_t)xLen); /* the graph's kernel */
  return AEE_SUCCESS;
}
