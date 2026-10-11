// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hexkl_mm_u8i4_dma.c
 * @date   06 Aug 2026
 * @brief  Persistent u8i4 weight registry and the cross-matmul DMA layer path
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "hexkl_mm_u8i4_dma.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <AEEStdErr.h>

#include "hexkl_acc_tile.h"
#include "hexkl_dma_ring.h"
#include "hexkl_hmx_mm.h"
#include "hexkl_micro.h"
#include "hexkl_mm_opts.h"
#include "hvx_dequant_i32.h"
#include "hvx_quant_u8.h"
#include "hvx_rmsnorm_rows_f32.h"
#include "hvx_rope_rows_f32.h"
#include "hvx_swiglu_f32.h"

#include "hexkl_probe.h"

#if defined(__hexagon__)
#include <qurt.h>
#include <qurt_memory.h>
#endif

#define ROUND_UP_U32(v, a) ((((v) + ((a)-1)) / (a)) * (a))

/** @brief Bytes in one packed i4 weight tile (32x32 values). Matches
 *         hexkl_mm_u8i4.c's local definition -- both are file-scoped, so
 *         this is not a redefinition, just the same constant kept in sync
 *         by inspection. If HexKL ever exposes it from hexkl_micro.h, both
 *         files should switch to that instead of this comment. */
#define WEIGHT_TILE_BYTES_U8I4 512u

/** @brief One u8 x i4 tile multiply, the packet inline (hexkl_hmx_mm.h):
 *  the library call wraps it in ~50 cycles for ~9 of array time (PR 4343,
 *  a63aeac6), and a 1024-row qkv call issues hundreds of thousands. The two
 *  alignment tests the library makes stay, here as two ANDs; a tile that
 *  fails them takes the library call and its error code. */
static inline int mm_u8i4_tile(uint8_t *vtcm_base, uint32_t act_off,
                               uint32_t w_off) {
  const uint8_t *a = vtcm_base + act_off, *w = vtcm_base + w_off;
  if (hexkl_hmx_mm_aligned(a, w)) {
    hexkl_hmx_mm_u8i4(a, w);
    return AEE_SUCCESS;
  }
  return hexkl_micro_hmx_mm_u8i4(vtcm_base, act_off, w_off);
}
/** @brief Bytes in one int32 accumulator tile (64x32 values). */
#define ACC_TILE_BYTES 8192u
/** @brief Largest single DMA row HexKL/HVX will move correctly; rows above
 *         this size have a documented hardware bug (doc13 §3a, §5). */
#define MAX_DMA_ROW_BYTES 262144u

static uint32_t dma_row_size_dividing(uint32_t total_bytes) {
  uint32_t rs = MAX_DMA_ROW_BYTES;
  while (rs > 1 && (total_bytes % rs) != 0) {
    rs >>= 1;
  }
  return rs;
}

/** @brief hvx_worker_pool_func body for the registration bake. Tile t is
 *         self-contained: it reads w_i4_rm's (t/n_tiles_row, t%n_tiles_row)
 *         region -- disjoint per t -- and writes VTCM at t*512, also
 *         disjoint, so the finished image is the serial bake's regardless of
 *         how t is split across workers. */
typedef struct {
  const int8_t *w_i4_rm;
  uint8_t *vtcm_base;
  uint32_t n_tiles;     /**< total tiles to bake: k_tiles * n_tiles_row */
  uint32_t n_tiles_row; /**< tiles per row band: n_tiles */
  uint32_t N;
  int err; /**< Benign race: the vendor bake's inputs are shape- and
            * alignment-identical across tiles, so it fails for all tiles or
            * none -- which failing worker's store survives does not change
            * the outcome, registration reports an error either way. */
} hexkl_bake_u8i4_ctx;

static void hexkl_bake_u8i4_worker(uint32_t n_threads, uint32_t i, void *vctx) {
  hexkl_bake_u8i4_ctx *c = (hexkl_bake_u8i4_ctx *)vctx;
  const uint32_t lo = (uint32_t)((uint64_t)c->n_tiles * i / n_threads);
  const uint32_t hi = (uint32_t)((uint64_t)c->n_tiles * (i + 1) / n_threads);
  for (uint32_t t = lo; t < hi; ++t) {
    const int res = hexkl_micro_hmx_rm_to_wh_i4(
      c->vtcm_base, t * WEIGHT_TILE_BYTES_U8I4, c->w_i4_rm, t / c->n_tiles_row,
      t % c->n_tiles_row, c->N);
    if (res != AEE_SUCCESS) {
      c->err = res;
    }
  }
}

/**
 * @brief Shape check and WH byte count, shared by the two register paths.
 *
 * @return AEE_SUCCESS with *out_bytes set, or the same rejection either
 *         entry point would have made on its own.
 */
static int hexkl_weight_u8i4_check(uint32_t K, uint32_t N, uint32_t vtcm_size,
                                   uint32_t *out_bytes) {
  if (K == 0 || N == 0 || (K % HEXKL_HMX_INT8_BLOCK_N_INNER) != 0 ||
      (N % HEXKL_HMX_INT8_BLOCK_N_COL) != 0) {
    return AEE_EBADPARM;
  }
  *out_bytes = (K / HEXKL_HMX_INT8_BLOCK_N_INNER) *
               (N / HEXKL_HMX_INT8_BLOCK_N_COL) * WEIGHT_TILE_BYTES_U8I4;
  /* The bake path stages through the VTCM arena, so a weight that does not
     fit it cannot be registered that way. Checked for both paths so the two
     accept exactly the same set of weights -- a cache written by one and
     read by the other must not hit a shape the other rejects. */
  if (*out_bytes > vtcm_size) {
    return AEE_ENOMEMORY;
  }
  return AEE_SUCCESS;
}

/** @brief First free slot, or HEXKL_MM_U8I4_MAX_WEIGHTS when full. */
static uint32_t hexkl_weight_u8i4_free_slot(hexkl_weight_u8i4_table *tbl) {
  for (uint32_t i = 0; i < HEXKL_MM_U8I4_MAX_WEIGHTS; ++i) {
    if (!tbl->slots[i].in_use) {
      return i;
    }
  }
  return HEXKL_MM_U8I4_MAX_WEIGHTS;
}

/**
 * @brief [doc 52 section 10.30] Fills a borrowed slot's scales and column
 *        sums from the arena, where the host lays them right after the WH
 *        bytes: N f32 scales, then N i32 column sums. Bias is zero.
 *
 * Through the DMA engine, not memcpy: the arena is host-uncached DDR,
 * which the scalar core reads at under 1 GB/s (doc 46 section 10) and the
 * DMA at full bandwidth. Both ends are invalidated around the transfer --
 * the source because the engine reads through L2 (src_bypass=0) and a
 * previous swap into this slot may have left its lines there, the
 * destination because the epilogues read these arrays and L1 may still
 * hold them. Drained before returning: the slot is usable at once.
 */
static void hexkl_weight_u8i4_tail_from_arena(hexkl_weight_u8i4 *h,
                                              uint32_t wh_bytes) {
  const uint8_t *tail = h->wh_bytes + wh_bytes;
  const uint32_t n4 = (uint32_t)sizeof(float) * h->N;
  hexkl_dcache_inva(tail, 2u * n4);
  hexkl_dma_ring_push2d(h->w_scale, tail, n4, n4, n4, 1u, 0, 0);
  hexkl_dma_ring_push2d(h->colsum_w, tail + n4, n4, n4, n4, 1u, 0, 0);
  hexkl_dma_ring_drain();
  hexkl_dcache_inva(h->w_scale, 2u * n4);
  memset(h->bias, 0, n4);
}

/** @brief [#267 L1] Writes [p, p + n) back to DDR (L1 and L2), keeping
 *  the lines: a heap image's one write, made visible to a DMA that reads
 *  around the L2 (hexkl_mm_u8i4_moe.h, HEXKL_MOE_FLAG_DMA_BYPASS).
 *  @return 1 if the lines were written back */
static int hexkl_weight_u8i4_flush(const void *p, size_t n) {
#if defined(__hexagon__)
  return qurt_mem_cache_clean((qurt_addr_t)p, (qurt_size_t)n,
                              QURT_MEM_CACHE_FLUSH,
                              QURT_MEM_DCACHE) == QURT_EOK;
#else
  (void)p; /* the host stand-in's transfers are memcpys: no cache */
  (void)n;
  return 1;
#endif
}

/**
 * @brief Allocates a slot's arrays and copies the caller's bytes in.
 *
 * @a wh_src is the finished WH bytes -- the VTCM scratch the bake just
 * wrote, or the host's cached copy. Neither path owns them afterwards.
 * The three N-sized arrays are one 128-aligned block (hexkl_weight_u8i4's
 * arrays): the DMA that fills them from an arena wants aligned rows, and
 * one allocation frees as one.
 */
/* borrow != 0 points the slot at wh_src instead of copying it: the bytes
   live in a host arena the DSP has mapped, and release() must not free
   them. A NULL w_scale with borrow set takes the scales and column sums
   from the arena too (hexkl_weight_u8i4_tail_from_arena). Everything else
   about the slot is the same either way. */
static int hexkl_weight_u8i4_fill_slot(hexkl_weight_u8i4_table *tbl,
                                       uint32_t slot, uint32_t K, uint32_t N,
                                       uint32_t wh_bytes, const uint8_t *wh_src,
                                       const float *w_scale,
                                       const int32_t *colsum_w,
                                       const float *bias, int borrow) {
  hexkl_weight_u8i4 *h = &tbl->slots[slot];
  const size_t n4 = sizeof(float) * N; /* N is a multiple of 32: n4 of 128 */
  uintptr_t a;
  h->wh_bytes = borrow ? (uint8_t *)wh_src : (uint8_t *)malloc(wh_bytes);
  h->arrays = malloc(3u * n4 + 128u);
  if (!h->wh_bytes || !h->arrays) {
    if (!borrow) {
      free(h->wh_bytes);
    }
    free(h->arrays);
    memset(h, 0, sizeof(*h));
    return AEE_ENOMEMORY;
  }
  a = ((uintptr_t)h->arrays + 127u) & ~(uintptr_t)127u;
  h->w_scale = (float *)a;
  h->colsum_w = (int32_t *)(a + n4);
  h->bias = (float *)(a + 2u * n4);
  h->borrowed = borrow;
  h->K = K;
  h->N = N;
  if (!borrow) {
    memcpy(h->wh_bytes, wh_src, wh_bytes);
  }
  if (w_scale) {
    memcpy(h->w_scale, w_scale, n4);
    memcpy(h->colsum_w, colsum_w, n4);
    memcpy(h->bias, bias, n4);
  } else {
    hexkl_weight_u8i4_tail_from_arena(h, wh_bytes);
  }
  if (!borrow) {
    /* [#267 L1] Nothing on the DSP writes the image again (prefill and
       decode only read it, release frees it), so one flush here lets every
       reader's DMA take src_bypass as for an arena slot. Only wh_bytes:
       the arrays are read by the core, never by a bypassing DMA. A failed
       flush leaves the slot on the L2 path. */
    h->clean = hexkl_weight_u8i4_flush(h->wh_bytes, wh_bytes);
  }
  /* Four unless the caller says otherwise (hexkl_weight_u2i4_register_arena
     sets 2 after this returns). */
  h->bits = 4u;
  h->in_use = 1;
  return AEE_SUCCESS;
}

int hexkl_weight_u8i4_register(hexkl_weight_u8i4_table *tbl, uint8_t *vtcm_base,
                               uint32_t vtcm_size, uint32_t K, uint32_t N,
                               const int8_t *w_i4_rm, const float *w_scale,
                               const int32_t *colsum_w, const float *bias,
                               hvx_worker_pool *pool, uint32_t *out_handle) {
  uint32_t wh_bytes = 0;
  uint32_t slot;
  int rc;

  if (!tbl || !vtcm_base || !w_i4_rm || !w_scale || !colsum_w || !bias ||
      !out_handle) {
    return AEE_EBADPARM;
  }
  rc = hexkl_weight_u8i4_check(K, N, vtcm_size, &wh_bytes);
  if (rc != AEE_SUCCESS) {
    return rc;
  }
  slot = hexkl_weight_u8i4_free_slot(tbl);
  if (slot == HEXKL_MM_U8I4_MAX_WEIGHTS) {
    return AEE_ENOMEMORY;
  }

  // Bake every tile into VTCM (borrowed as scratch -- caller holds the HMX
  // lock and has no other VTCM use in flight), then copy the baked bytes
  // out to DSP heap memory where they stay resident across calls. A plain
  // copy, not the DMA ring: registration happens once per weight at model
  // load, off the per-token hot path.
  //
  // This bake is the expensive half of registration -- 7168 independent
  // 512-byte tiles for a gate_up weight, about 25 ms even split ~6 ways
  // across the caller's HVX pool, and 64 weights of it is the 1597 ms the
  // profile reports as "register FastRPC". A run pays it once per weight and
  // exports the bytes; every later run hands them back through
  // hexkl_weight_u8i4_register_arena below, already in a host buffer the
  // DSP has mapped, so neither the bake nor a DSP-heap copy happens again.
  //
  // Verified NOT a race (doc 43 section 7, corrected): an earlier session
  // read a hash mismatch as HMX-lock corruption from running this across
  // worker threads, but the "serial reference" in that comparison was
  // itself produced by hvx_worker_pool_run(NULL, ...) with n_units > 1 --
  // which at the time ran ONLY worker 0's 1/n_units slice. That reference
  // had baked one tile and left the rest as stale VTCM scratch. The REAL
  // pool path was re-verified bit-identical to a true serial loop --
  // FNV-1a of the same M/K/N matmul output matches either way
  // (0x198748e597cf4105).
  {
    const uint32_t n_tiles_row = N / HEXKL_HMX_INT8_BLOCK_N_COL;
    const uint32_t n_tiles = wh_bytes / WEIGHT_TILE_BYTES_U8I4;
    hexkl_bake_u8i4_ctx bc = {w_i4_rm,     vtcm_base, n_tiles,
                              n_tiles_row, N,         AEE_SUCCESS};
    hvx_worker_pool_run(pool, hexkl_bake_u8i4_worker, &bc, n_tiles);
    if (bc.err != AEE_SUCCESS) {
      return bc.err;
    }
  }

  rc = hexkl_weight_u8i4_fill_slot(tbl, slot, K, N, wh_bytes, vtcm_base,
                                   w_scale, colsum_w, bias, /*borrow=*/0);
  if (rc != AEE_SUCCESS) {
    return rc;
  }
  *out_handle = slot;
  return AEE_SUCCESS;
}

int hexkl_weight_u8i4_register_arena(hexkl_weight_u8i4_table *tbl,
                                     uint32_t vtcm_size, uint32_t K, uint32_t N,
                                     const uint8_t *wh, const float *w_scale,
                                     const int32_t *colsum_w, const float *bias,
                                     uint32_t *out_handle) {
  uint32_t wh_bytes = 0;
  uint32_t slot;
  int rc;

  /* Either all three arrays or none: none means they follow wh in the
     arena (doc 52 section 10.30). */
  if (!tbl || !wh || !out_handle || (!w_scale != !colsum_w) ||
      (!w_scale != !bias)) {
    return AEE_EBADPARM;
  }
  if (((uintptr_t)wh % WEIGHT_TILE_BYTES_U8I4) != 0u) {
    return AEE_EBADPARM;
  }
  rc = hexkl_weight_u8i4_check(K, N, vtcm_size, &wh_bytes);
  if (rc != AEE_SUCCESS) {
    return rc;
  }
  slot = hexkl_weight_u8i4_free_slot(tbl);
  if (slot == HEXKL_MM_U8I4_MAX_WEIGHTS) {
    return AEE_ENOMEMORY;
  }
  rc = hexkl_weight_u8i4_fill_slot(tbl, slot, K, N, wh_bytes, wh, w_scale,
                                   colsum_w, bias, /*borrow=*/1);
  if (rc != AEE_SUCCESS) {
    return rc;
  }
  *out_handle = slot;
  return AEE_SUCCESS;
}

int hexkl_weight_u8i4_rebind_arena(hexkl_weight_u8i4_table *tbl, uint32_t h,
                                   uint32_t K, uint32_t N, const uint8_t *wh,
                                   const int8_t *pal, const float *w_scale,
                                   const int32_t *colsum_w) {
  hexkl_weight_u8i4 *w;
  if (!tbl || !wh || (!w_scale != !colsum_w) ||
      h >= HEXKL_MM_U8I4_MAX_WEIGHTS) {
    return AEE_EBADPARM;
  }
  w = &tbl->slots[h];
  if (!w->in_use || !w->borrowed || w->K != K || w->N != N ||
      ((uintptr_t)wh % WEIGHT_TILE_BYTES_U8I4) != 0u) {
    return AEE_EBADPARM;
  }
  w->wh_bytes = (uint8_t *)wh;
  /* The width moves with the bytes: a pool slot holds whatever expert the
     host read into it last. */
  w->bits = pal ? 2u : 4u;
  if (pal) {
    memcpy(w->pal, pal, sizeof(w->pal));
  }
  if (w_scale) {
    memcpy(w->w_scale, w_scale, sizeof(float) * N);
    memcpy(w->colsum_w, colsum_w, sizeof(int32_t) * N);
    memset(w->bias, 0, sizeof(float) * N);
  } else {
    hexkl_weight_u8i4_tail_from_arena(
      w, (K / HEXKL_HMX_INT8_BLOCK_N_INNER) * (N / HEXKL_HMX_INT8_BLOCK_N_COL) *
           WEIGHT_TILE_BYTES_U8I4 / (pal ? 2u : 1u));
  }
  return AEE_SUCCESS;
}

int hexkl_weight_u2i4_register_arena(hexkl_weight_u8i4_table *tbl,
                                     uint32_t vtcm_size, uint32_t K, uint32_t N,
                                     const uint8_t *wh, const int8_t *pal,
                                     const float *w_scale,
                                     const int32_t *colsum_w, const float *bias,
                                     uint32_t *out_handle) {
  uint32_t wh_bytes = 0;
  uint32_t slot;
  int rc;

  /* Either all three arrays or none, as for the 4-bit entry: none means
     they follow the codes in the arena. */
  if (!tbl || !wh || !pal || !out_handle || (!w_scale != !colsum_w) ||
      (!w_scale != !bias)) {
    return AEE_EBADPARM;
  }
  /* A whole tile, as for four bits: hexkl_weight_u8i4_rebind_arena (the
     pool's swap) checks the same, so a slot registered here can always be
     rebound. The host places weights on 4 KiB boundaries. */
  if (((uintptr_t)wh % WEIGHT_TILE_BYTES_U8I4) != 0u) {
    return AEE_EBADPARM;
  }
  /* The VTCM check is the 4-bit one on purpose: the HMX path expands the
     codes in place into a full-width region, so what has to fit is the
     expanded size. */
  rc = hexkl_weight_u8i4_check(K, N, vtcm_size, &wh_bytes);
  if (rc != AEE_SUCCESS) {
    return rc;
  }
  slot = hexkl_weight_u8i4_free_slot(tbl);
  if (slot == HEXKL_MM_U8I4_MAX_WEIGHTS) {
    return AEE_ENOMEMORY;
  }
  rc = hexkl_weight_u8i4_fill_slot(tbl, slot, K, N, wh_bytes / 2u, wh, w_scale,
                                   colsum_w, bias, /*borrow=*/1);
  if (rc != AEE_SUCCESS) {
    return rc;
  }
  tbl->slots[slot].bits = 2u;
  memcpy(tbl->slots[slot].pal, pal, sizeof(tbl->slots[slot].pal));
  *out_handle = slot;
  return AEE_SUCCESS;
}

int hexkl_weight_u8i4_borrows(const hexkl_weight_u8i4_table *tbl,
                              const uint8_t *base, uint32_t bytes) {
  uint32_t i;
  if (!tbl || !base) {
    return 0;
  }
  for (i = 0; i < HEXKL_MM_U8I4_MAX_WEIGHTS; ++i) {
    const hexkl_weight_u8i4 *h = &tbl->slots[i];
    if (h->in_use && h->borrowed && h->wh_bytes >= base &&
        h->wh_bytes < base + bytes) {
      return 1;
    }
  }
  return 0;
}

int hexkl_weight_u8i4_export(const hexkl_weight_u8i4_table *tbl,
                             uint32_t handle, uint8_t *wh_out,
                             uint32_t wh_out_len) {
  const hexkl_weight_u8i4 *h;
  uint32_t wh_bytes;

  if (!tbl || !wh_out || handle >= HEXKL_MM_U8I4_MAX_WEIGHTS) {
    return AEE_EBADPARM;
  }
  h = &tbl->slots[handle];
  /* The bake cache holds int4 WH bytes; a 2-bit slot has none to give. */
  if (!h->in_use || h->bits == 2u) {
    return AEE_EBADPARM;
  }
  wh_bytes = (h->K / HEXKL_HMX_INT8_BLOCK_N_INNER) *
             (h->N / HEXKL_HMX_INT8_BLOCK_N_COL) * WEIGHT_TILE_BYTES_U8I4;
  if (wh_out_len != wh_bytes) {
    return AEE_EBADPARM;
  }
  memcpy(wh_out, h->wh_bytes, wh_bytes);
  return AEE_SUCCESS;
}

int hexkl_weight_u8i4_release(hexkl_weight_u8i4_table *tbl, uint32_t handle) {
  if (!tbl || handle >= HEXKL_MM_U8I4_MAX_WEIGHTS ||
      !tbl->slots[handle].in_use) {
    return AEE_EBADPARM;
  }
  hexkl_weight_u8i4 *h = &tbl->slots[handle];
  if (!h->borrowed) {
    free(h->wh_bytes);
  }
  free(h->arrays);
  memset(h, 0, sizeof(*h));
  return AEE_SUCCESS;
}

/** @brief A staged batch's per-head passes (hexkl_mm_opts.post_done): the
 *         run of columns a row just got, normed and rotated in place. */
typedef struct {
  const float *gamma; /**< the handle's chunk gamma */
  const float *cs;    /**< the table row of the batch's row 0, or NULL */
  uint32_t n, chunk, hd;
  float eps;
} fc_post;

static void fc_row_post(void *v, float *seg, uint32_t row) {
  const fc_post *p = (const fc_post *)v;
  if (p->chunk != 0u) {
    hvx_rmsnorm_row_chunks_f32(seg, seg, p->gamma, p->n, p->chunk, p->eps);
  }
  if (p->cs != NULL) {
    hvx_rope_row_f32(seg, p->n, p->hd, p->cs + (size_t)row * 2u * p->hd);
  }
}

int hexkl_mm_u8i4_layer_run(hexkl_weight_u8i4_table *tbl, uint8_t *vtcm_base,
                            uint32_t vtcm_size, uint32_t config_off, uint32_t M,
                            uint32_t K, const uint32_t *handles,
                            uint32_t n_handles, const float *act_f32,
                            float *out_cat, const hexkl_mm_opts *opts) {
  static const hexkl_mm_opts kDefaults = {0};
  const hexkl_mm_opts *o = opts ? opts : &kDefaults;

  if (!tbl || !vtcm_base || !handles || n_handles == 0 ||
      (!act_f32 && !o->act_ah_prepacked && !o->act_u16) || !out_cat || M == 0 ||
      K == 0) {
    return AEE_EBADPARM;
  }
  if (o->act_u16 != NULL &&
      (o->pre_scratch == NULL || o->pre_gamma != NULL || o->act_scale != NULL ||
       o->act_ah_prepacked != NULL || o->act16_enc == NULL ||
       o->act16_hd == 0 || K % o->act16_hd != 0)) {
    return AEE_EBADPARM; // the u16 rows ride the pre-norm scratch alone
  }
  if ((o->act_scale == NULL) != (o->act_zp == NULL)) {
    return AEE_EBADPARM; // both or neither, per hexkl_mm_opts.h
  }
  if (o->act_ah_prepacked != NULL && o->act_scale == NULL) {
    return AEE_EBADPARM; // prepacked bytes need the caller's own scale/zp
  }
  if (o->pre_scratch != NULL && o->act_scale != NULL) {
    return AEE_EBADPARM; // the pre norm rides on the params scan
  }
  if ((o->out_off == NULL) != (o->out_ld == NULL)) {
    return AEE_EBADPARM;
  }

  const uint32_t m_pad = ROUND_UP_U32(M, HEXKL_HMX_INT8_BLOCK_N_ROW);
  const uint32_t k_tiles = K / HEXKL_HMX_INT8_BLOCK_N_INNER;
  const uint32_t n_rblocks = m_pad / HEXKL_HMX_INT8_BLOCK_N_ROW;
  if ((K % HEXKL_HMX_INT8_BLOCK_N_INNER) != 0) {
    return AEE_EBADPARM;
  }

  // Validate every handle up front -- K must match, and this is also where
  // the widest weight (for the double-buffer size) and the total output
  // width (for out_cat bounds) come from.
  uint32_t n_tiles_max = 0, n_max = 0;
  for (uint32_t i = 0; i < n_handles; ++i) {
    if (handles[i] >= HEXKL_MM_U8I4_MAX_WEIGHTS ||
        !tbl->slots[handles[i]].in_use) {
      return AEE_EBADPARM;
    }
    const hexkl_weight_u8i4 *h = &tbl->slots[handles[i]];
    if (h->K != K) {
      return AEE_EBADPARM;
    }
    const uint32_t nt = h->N / HEXKL_HMX_INT8_BLOCK_N_COL;
    if (nt > n_tiles_max) {
      n_tiles_max = nt;
    }
    if (h->N > n_max) {
      n_max = h->N;
    }
  }

  // VTCM layout: activation (all row-bands, shared across every handle) |
  // weight double-buffer (sized for the widest handle) | TWO staging
  // buffers of acc_tiles result tiles each, in whatever room is left (at
  // least one tile each): the HMX issues a batch of n-tiles into one while
  // the pool dequantizes the other, the same shape as hexkl_mm_u8i4_moe.c's
  // epilogue. Dequantizing each tile on the calling thread between two
  // acc_reads left the HMX idle for it -- 270-400 us of a 1.3-1.8 ms
  // prefill projection call (doc 51 section 2.24).
  const uint32_t act_bytes =
    n_rblocks * k_tiles * HEXKL_HMX_ACTIVATION_ALIGNMENT;
  const uint32_t act_off = 0;
  const uint32_t wb_max = k_tiles * n_tiles_max * WEIGHT_TILE_BYTES_U8I4;
  const uint32_t wbuf[2] = {
    ROUND_UP_U32(act_off + act_bytes, HEXKL_HMX_ACTIVATION_ALIGNMENT),
    ROUND_UP_U32(act_off + act_bytes, HEXKL_HMX_ACTIVATION_ALIGNMENT) + wb_max,
  };
  const uint32_t result_off =
    ROUND_UP_U32(wbuf[1] + wb_max, HEXKL_HMX_ACTIVATION_ALIGNMENT);
  const uint32_t arena = vtcm_size < config_off ? vtcm_size : config_off;
  if (result_off + 2u * ACC_TILE_BYTES > arena) {
    return AEE_ENOMEMORY; // double-buffered widest weight does not fit VTCM
  }
  /* Capped at 32 like the MoE kernel's: past the worker count a bigger
     batch parallelises no better, and a smaller one exposes less at the
     end of each row block. */
  uint32_t acc_tiles = (arena - result_off) / (2u * ACC_TILE_BYTES);
  if (acc_tiles > 32u) {
    acc_tiles = 32u;
  }
  /* With the per-head passes asked for (post_done), a batch of whole
     heads when the room allows one: the widest head's tiles divide it. */
  if (o->post_done != NULL) {
    uint32_t unit = 1u;
    for (uint32_t i = 0; i < n_handles; ++i) {
      const uint32_t c = o->post_chunk ? o->post_chunk[i] : 0u;
      const uint32_t hd = i < o->rope_handles ? o->rope_hd : 0u;
      const uint32_t w = (c > hd ? c : hd) / HEXKL_ACC_TILE_COLS;
      unit = w > unit ? w : unit;
    }
    if (acc_tiles >= unit) {
      acc_tiles -= acc_tiles % unit;
    }
  }

  // Reset once per call, before ANY push2d -- moved here (was just before
  // the weight-loading loop) because the act_ah_prepacked path below now
  // pushes a transfer of its own; nothing pushed before this point in
  // either path, so the move changes nothing for the pre-existing flow.
  hexkl_dma_ring_reset();

  // K1/K2: quantize the shared activation once, straight into its AH tiles
  // in VTCM -- no separate layout pass, matching hvx_quant_pack_u8_ah's
  // contract.
  /** Safe here and not earlier: result_off has just been bounds-checked, and
   * the probe writes a ramp over that tile. */
  const hexkl_acc_layout *acc_layout =
    hexkl_acc_layout_get(vtcm_base, result_off);
  hexkl_probe_us[HEXKL_PROBE_ACC_STRIDE] =
    acc_layout->usable ? acc_layout->row_stride : 0u;
  if (o->out_off && !acc_layout->usable) {
    return AEE_EUNSUPPORTED; // the fallback dequant writes packed blocks
  }

  /** Caller-supplied params (o->act_scale) skip the min/max scan; loc_* are
   * only allocated for the dynamic path. */
  float *loc_scale = NULL;
  int32_t *loc_zp = NULL;
  /* Only the fallback dequant path needs the DDR staging matrix. */
  int32_t *acc_scratch =
    acc_layout->usable
      ? NULL
      : (int32_t *)malloc(sizeof(int32_t) * (size_t)m_pad * n_max);
  if (o->act_scale == NULL) {
    loc_scale = (float *)malloc(sizeof(float) * m_pad);
    loc_zp = (int32_t *)malloc(sizeof(int32_t) * m_pad);
  }
  if ((!acc_layout->usable && !acc_scratch) ||
      (o->act_scale == NULL && (!loc_scale || !loc_zp))) {
    free(loc_scale);
    free(loc_zp);
    free(acc_scratch);
    return AEE_ENOMEMORY;
  }
  uint64_t p0 = 0;
  const float *act_scale = o->act_scale;
  const int32_t *act_zp = o->act_zp;
  int rc0 = AEE_SUCCESS;

  /** Issue handle 0's weight transfer HERE, before the activation is
     quantized, and wait for it after. The quantization needs no weight and
     the DMA engine needs no HVX, so the two overlap for free; the drain
     that used to sit right after this push had nothing in front of it to
     hide behind. Nothing else in the loop can cover it either when
     n_handles is 1, which is every MoE expert call.
     Measured on device 2026-09-10: gate_up's drain 111.9 -> 22.8 us/call. */
  {
    const hexkl_weight_u8i4 *h0 = &tbl->slots[handles[0]];
    const uint32_t nt0 = h0->N / HEXKL_HMX_INT8_BLOCK_N_COL;
    const uint32_t wb0 = k_tiles * nt0 * WEIGHT_TILE_BYTES_U8I4;
    const uint32_t rs0 = dma_row_size_dividing(wb0);
    hexkl_dma_ring_push2d(vtcm_base + wbuf[0], h0->wh_bytes, rs0, rs0, rs0,
                          wb0 / rs0, /*src_vtcm=*/0, /*dst_vtcm=*/1);
  }

  if (o->act_ah_prepacked != NULL) {
    // The caller already quantized and AH-tile-packed the activation (see
    // hexkl_mm_opts.h's doc on this field) -- DMA the bytes into VTCM the
    // same way a weight tile lands there, instead of quantizing act_f32
    // (which may be NULL here; there is nothing left in it to quantize).
    // act_bytes was sized exactly m_pad*K above for this same region.
    // Charged to DRAIN, not QUANT: this is a DMA wait, not compute, and
    // the whole point of this path is that QUANT should read ~0.
    const uint32_t rs = dma_row_size_dividing(act_bytes);
    hexkl_dma_ring_push2d(vtcm_base + act_off, o->act_ah_prepacked, rs, rs, rs,
                          act_bytes / rs, /*src_vtcm=*/0,
                          /*dst_vtcm=*/1);
    HEXKL_PROBE_T0(p0);
    hexkl_dma_ring_drain();
    HEXKL_PROBE_ADD(HEXKL_PROBE_DRAIN, p0);
  } else {
    HEXKL_PROBE_T0(p0);
    if (act_scale == NULL) {
      rc0 = hvx_quant_params_pack_u8_ah(act_f32, o->act_u16, o->act16_enc,
                                        o->act16_hd, M, m_pad, K, o->pre_gamma,
                                        o->pre_eps, o->pre_scratch, loc_scale,
                                        loc_zp, vtcm_base + act_off, o->pool);
      act_scale = loc_scale;
      act_zp = loc_zp;
    } else {
      rc0 = hvx_quant_pack_u8_ah(act_f32, M, m_pad, K, act_scale, act_zp,
                                 vtcm_base + act_off, o->pool);
    }
    HEXKL_PROBE_ADD(HEXKL_PROBE_QUANT, p0);
  }
  if (rc0 != AEE_SUCCESS) {
    free(loc_scale);
    free(loc_zp);
    free(acc_scratch);
    return rc0;
  }

  int rc = AEE_SUCCESS;
  size_t out_off = 0;
  /* The pooled epilogue: one job per staging buffer, each alive from its
     submit to the wait one batch later, and the staging parity runs across
     row blocks and handles. Not with accumulate (the attention paths add
     into out_cat and the job has no flag for it) nor without the in-place
     tile layout; those keep the synchronous tile-at-a-time path. */
  const int pipelined = acc_layout->usable && !o->accumulate;
  hvx_dq_rows_job dq_job[2];
  fc_post post[2];
  uint32_t sb = 0u;
  /* The per-head passes in the epilogue: only when every staged batch of
     every handle that has one holds whole heads. */
  int fuse_post = pipelined && o->post_done != NULL;
  for (uint32_t i = 0; fuse_post && i < n_handles; ++i) {
    const uint32_t c = o->post_chunk ? o->post_chunk[i] : 0u;
    const uint32_t hd = i < o->rope_handles ? o->rope_hd : 0u;
    const uint32_t cols = acc_tiles * HEXKL_ACC_TILE_COLS;
    fuse_post = (c == 0u || cols % c == 0u) && (hd == 0u || cols % hd == 0u);
  }
  if (fuse_post) {
    *o->post_done = 1;
  }
  const float *post_gamma = o->post_gamma;

  // Handle 0's weight was issued before the activation quantization above;
  // this is where the wait for it lands. With one handle -- every MoE
  // expert call -- there is no next weight to prefetch it behind, so
  // overlapping it with the quant is the only cover available.
  HEXKL_PROBE_T0(p0);
  hexkl_dma_ring_drain();
  HEXKL_PROBE_ADD(HEXKL_PROBE_DRAIN, p0);

  for (uint32_t i = 0; i < n_handles; ++i) {
    const hexkl_weight_u8i4 *h = &tbl->slots[handles[i]];
    const uint32_t nt_n = h->N / HEXKL_HMX_INT8_BLOCK_N_COL;
    const uint32_t wcur = wbuf[i & 1u];
    float *const out_h =
      o->out_off ? out_cat + o->out_off[i] : out_cat + out_off;
    const uint32_t ld_h = o->out_ld ? o->out_ld[i] : h->N;

    if (i + 1 < n_handles) {
      // Cross-matmul prefetch (doc13 §3a): while handle i computes below,
      // stream handle i+1's weight into the OTHER buffer in the
      // background. One transfer, chunked under the >512KB-row DMA bug's
      // threshold, not one per tile.
      const hexkl_weight_u8i4 *hn = &tbl->slots[handles[i + 1]];
      const uint32_t nt_next = hn->N / HEXKL_HMX_INT8_BLOCK_N_COL;
      const uint32_t wb_next = k_tiles * nt_next * WEIGHT_TILE_BYTES_U8I4;
      const uint32_t rs = dma_row_size_dividing(wb_next);
      hexkl_dma_ring_push2d(vtcm_base + wbuf[(i + 1) & 1u], hn->wh_bytes, rs,
                            rs, rs, wb_next / rs, /*src_vtcm=*/0,
                            /*dst_vtcm=*/1);
    }

    for (uint32_t rb = 0; rb < n_rblocks; ++rb) {
      /** The rows beyond M are the accumulator's padding: their
       * quantization parameters are synthetic, so emitting them would be
       * wrong, not merely wasted -- the same rule hvx_dequant_i32_to_f32
       * applies via m_valid. At decode that is 62 of 64 rows never touched
       * at all. */
      const uint32_t m0 = rb * HEXKL_ACC_TILE_ROWS;
      const uint32_t cnt =
        (m0 >= M)
          ? 0u
          : ((M - m0 < HEXKL_ACC_TILE_ROWS) ? (M - m0) : HEXKL_ACC_TILE_ROWS);
      for (uint32_t nt0 = 0; nt0 < nt_n; nt0 += acc_tiles) {
        const uint32_t nb = (nt_n - nt0 < acc_tiles) ? (nt_n - nt0) : acc_tiles;
        const uint32_t stage_off =
          result_off + (sb & 1u) * acc_tiles * ACC_TILE_BYTES;
        /* This staging buffer was last read by the job two batches back,
           retired when the last one was submitted. */
        for (uint32_t j = 0; j < nb; ++j) {
          hexkl_micro_hmx_acc_clear_int32();
          for (uint32_t kt = 0; kt < k_tiles; ++kt) {
            const uint32_t act_tile_off =
              act_off + (rb * k_tiles + kt) * HEXKL_HMX_ACTIVATION_ALIGNMENT;
            const uint32_t w_tile_off =
              wcur + (kt * nt_n + nt0 + j) * WEIGHT_TILE_BYTES_U8I4;
            rc = mm_u8i4_tile(vtcm_base, act_tile_off, w_tile_off);
            if (rc != AEE_SUCCESS) {
              goto out;
            }
          }
          HEXKL_PROBE_T0(p0);
          rc = hexkl_micro_hmx_acc_read_int32(vtcm_base, config_off,
                                              stage_off + j * ACC_TILE_BYTES);
          HEXKL_PROBE_ADD(HEXKL_PROBE_ACC_READ, p0);
          if (rc != AEE_SUCCESS) {
            goto out;
          }
        }
        if (pipelined) {
          /* Retire the previous batch's job -- what this wait reads is the
             exposed part -- then hand this batch to the pool and go on
             issuing. */
          HEXKL_PROBE_T0(p0);
          hvx_worker_pool_wait(o->pool);
          HEXKL_PROBE_ADD(HEXKL_PROBE_DEQUANT, p0);
          if (cnt != 0u) {
            hvx_dq_rows_job *rj = &dq_job[sb & 1u];
            hvx_dq_tiles_job *jb = &rj->t;
            fc_post *pp = &post[sb & 1u];
            pp->chunk = fuse_post && o->post_chunk ? o->post_chunk[i] : 0u;
            pp->gamma = post_gamma;
            pp->eps = o->post_eps;
            pp->hd = o->rope_hd;
            pp->cs = fuse_post && i < o->rope_handles
                       ? o->rope_cs + (size_t)m0 * 2u * o->rope_hd
                       : NULL;
            pp->n = nb * HEXKL_ACC_TILE_COLS;
            rj->row_done = (pp->chunk != 0u || pp->cs) ? fc_row_post : NULL;
            rj->ctx = pp;
            jb->tiles_base =
              (const uint8_t *)((const int32_t *)(vtcm_base + stage_off) +
                                acc_layout->base);
            jb->tile_stride = ACC_TILE_BYTES;
            jb->nt0 = nt0;
            jb->row_stride = acc_layout->row_stride;
            jb->m_count = cnt;
            jb->act_scale = act_scale + m0;
            jb->act_zp = act_zp + m0;
            jb->colsum_w = h->colsum_w;
            jb->w_scale = h->w_scale;
            jb->bias = h->bias;
            jb->dst_a = out_h + (size_t)m0 * ld_h;
            jb->dst_b = NULL;
            jb->split = h->N;
            jb->dst_stride = ld_h;
            jb->n_tiles = nb;
            hvx_worker_pool_submit(o->pool, hvx_dq_tiles_rows_worker, rj, cnt);
          }
        } else if (acc_layout->usable) {
          /* Dequantize each tile where it is, on this thread. */
          for (uint32_t j = 0; j < nb && cnt != 0u; ++j) {
            const int32_t *tile =
              (const int32_t *)(vtcm_base + stage_off + j * ACC_TILE_BYTES) +
              acc_layout->base;
            const uint32_t c0 = (nt0 + j) * HEXKL_ACC_TILE_COLS;
            HEXKL_PROBE_T0(p0);
            hvx_dequant_acc_tile_to_f32(
              tile, acc_layout->row_stride, cnt, act_scale + m0, act_zp + m0,
              h->colsum_w + c0, h->w_scale + c0, h->bias + c0,
              out_h + (size_t)m0 * ld_h + c0, ld_h, o->accumulate);
            HEXKL_PROBE_ADD(HEXKL_PROBE_DEQUANT, p0);
          }
        } else {
          for (uint32_t j = 0; j < nb; ++j) {
            HEXKL_PROBE_T0(p0);
            rc = hexkl_micro_hmx_copy_32b_to_submatrix(
              vtcm_base, stage_off + j * ACC_TILE_BYTES, acc_scratch, rb,
              nt0 + j, m_pad, h->N);
            HEXKL_PROBE_ADD(HEXKL_PROBE_ACC_COPY, p0);
            if (rc != AEE_SUCCESS) {
              goto out;
            }
          }
        }
        ++sb;
      }
    }

    // Block until handle i+1's weight has fully landed before moving on to
    // it -- matches the measured bench's "next weight fully in wnxt before
    // matmul i+1" invariant. On the in-place path the dequant has already
    // happened, per tile, above this drain: it reads the VTCM result tile
    // and writes DDR, neither of which the weight prefetch touches, so it
    // overlaps the transfer for free. The fallback below cannot -- it needs
    // the whole staging matrix first.
    HEXKL_PROBE_T0(p0);
    hexkl_dma_ring_drain();
    HEXKL_PROBE_ADD(HEXKL_PROBE_DRAIN, p0);

    if (!acc_layout->usable) {
      HEXKL_PROBE_T0(p0);
      hvx_dequant_i32_to_f32(acc_scratch, M, m_pad, h->N, act_scale, act_zp,
                             h->colsum_w, h->w_scale, h->bias,
                             out_cat + out_off, o->accumulate);
      HEXKL_PROBE_ADD(HEXKL_PROBE_DEQUANT, p0);
    }
    out_off += (size_t)M * h->N;
    if (o->post_chunk) {
      post_gamma += o->post_chunk[i];
    }
  }

out:
  /* The last batch's job, or one left in flight by an error path, reads
     VTCM and the scale arrays freed below and writes out_cat: retire it
     before either goes away. */
  HEXKL_PROBE_T0(p0);
  hvx_worker_pool_wait(o->pool);
  HEXKL_PROBE_ADD(HEXKL_PROBE_DEQUANT, p0);
  free(loc_scale);
  free(loc_zp);
  free(acc_scratch);
  return rc;
}

int hexkl_mm_u8i4_fused_run(hexkl_weight_u8i4_table *tbl, uint8_t *vtcm_base,
                            uint32_t vtcm_size, uint32_t config_off, uint32_t M,
                            uint32_t K, const uint32_t *handles,
                            const float *act_f32, float *out,
                            const hexkl_mm_opts *opts) {
  static const hexkl_mm_opts kDefaults = {0};
  const hexkl_mm_opts *o = opts ? opts : &kDefaults;

  if (!tbl || !vtcm_base || !handles || !act_f32 || !out || M == 0 || K == 0) {
    return AEE_EBADPARM;
  }
  /** The down matmul's quantization is inherently dynamic -- it quantizes
   * the SwiGLU output the caller has never seen -- so caller-supplied act
   * params could only cover stage 1, an accuracy change dressed up as an
   * option. Reject rather than silently ignore (header doc). */
  if (o->act_scale != NULL || o->act_zp != NULL ||
      o->act_ah_prepacked != NULL) {
    return AEE_EBADPARM;
  }
  if ((K % HEXKL_HMX_INT8_BLOCK_N_INNER) != 0) {
    return AEE_EBADPARM;
  }
  if (handles[0] >= HEXKL_MM_U8I4_MAX_WEIGHTS ||
      handles[1] >= HEXKL_MM_U8I4_MAX_WEIGHTS ||
      !tbl->slots[handles[0]].in_use || !tbl->slots[handles[1]].in_use) {
    return AEE_EBADPARM;
  }
  const hexkl_weight_u8i4 *h_gu = &tbl->slots[handles[0]];
  const hexkl_weight_u8i4 *h_dn = &tbl->slots[handles[1]];
  /** gate_up's output row is [gate | up]; that ratio IS the SwiGLU
   * contract. inter a multiple of the 32-column tile also guarantees a
   * result tile never straddles the gate/up boundary below. */
  if (h_gu->K != K || h_gu->N != 2 * h_dn->K ||
      (h_dn->K % HEXKL_HMX_INT8_BLOCK_N_INNER) != 0) {
    return AEE_EBADPARM;
  }

  const uint32_t m_pad = ROUND_UP_U32(M, HEXKL_HMX_INT8_BLOCK_N_ROW);
  const uint32_t inter = h_dn->K;
  const uint32_t n_out = h_dn->N;
  const uint32_t k1_tiles = K / HEXKL_HMX_INT8_BLOCK_N_INNER;
  const uint32_t n1_tiles = h_gu->N / HEXKL_HMX_INT8_BLOCK_N_COL;
  const uint32_t k2_tiles = inter / HEXKL_HMX_INT8_BLOCK_N_INNER;
  const uint32_t n2_tiles = n_out / HEXKL_HMX_INT8_BLOCK_N_COL;

  // VTCM layout: act1 AH tiles | gate_up weight | SwiGLU gate | SwiGLU up |
  // act2 AH tiles | down weight | one result tile -- sized for ONE 64-row
  // block. The pipeline below loops over row blocks through those fixed
  // regions, so an expert with M > 64 rows costs no extra VTCM (the
  // full-M layout blew the 8.3 MB arena at m_pad = 128: AEE_ENOMEMORY).
  // One block at the LFM2-A1B prefill shape (K=2048, I=1792, N_out=2048):
  // 128K + 3.67M + 2x448K + 112K + 1.84M + 8K ~= 6.65 MB.
  const uint32_t act1_bytes = k1_tiles * HEXKL_HMX_ACTIVATION_ALIGNMENT;
  const uint32_t wb1 = k1_tiles * n1_tiles * WEIGHT_TILE_BYTES_U8I4;
  const uint32_t inter_bytes = HEXKL_HMX_INT8_BLOCK_N_ROW * inter * 4u;
  const uint32_t act2_bytes = k2_tiles * HEXKL_HMX_ACTIVATION_ALIGNMENT;
  const uint32_t wb2 = k2_tiles * n2_tiles * WEIGHT_TILE_BYTES_U8I4;

  const uint32_t act1_off = 0;
  const uint32_t w1_off =
    ROUND_UP_U32(act1_bytes, HEXKL_HMX_ACTIVATION_ALIGNMENT);
  const uint32_t inter_gate_off =
    ROUND_UP_U32(w1_off + wb1, HEXKL_HMX_ACTIVATION_ALIGNMENT);
  const uint32_t inter_up_off =
    ROUND_UP_U32(inter_gate_off + inter_bytes, HEXKL_HMX_ACTIVATION_ALIGNMENT);
  const uint32_t act2_off =
    ROUND_UP_U32(inter_up_off + inter_bytes, HEXKL_HMX_ACTIVATION_ALIGNMENT);
  const uint32_t w2_off =
    ROUND_UP_U32(act2_off + act2_bytes, HEXKL_HMX_ACTIVATION_ALIGNMENT);
  const uint32_t result_off =
    ROUND_UP_U32(w2_off + wb2, HEXKL_HMX_ACTIVATION_ALIGNMENT);
  if (result_off + ACC_TILE_BYTES > config_off ||
      result_off + ACC_TILE_BYTES > vtcm_size) {
    return AEE_ENOMEMORY;
  }

  /** The fused pipeline only runs on the in-place accumulator tile layout:
   * the gate/up split of the dequant has no vendor-copy counterpart, and
   * no caller exists where a repack kernel would be worth it. */
  const hexkl_acc_layout *acc_layout =
    hexkl_acc_layout_get(vtcm_base, result_off);
  hexkl_probe_us[HEXKL_PROBE_ACC_STRIDE] =
    acc_layout->usable ? acc_layout->row_stride : 0u;
  if (!acc_layout->usable) {
    return AEE_EUNSUPPORTED;
  }

  float *scale1 = (float *)malloc(sizeof(float) * m_pad);
  int32_t *zp1 = (int32_t *)malloc(sizeof(int32_t) * m_pad);
  float *scale2 = (float *)malloc(sizeof(float) * m_pad);
  int32_t *zp2 = (int32_t *)malloc(sizeof(int32_t) * m_pad);
  if (!scale1 || !zp1 || !scale2 || !zp2) {
    free(scale1);
    free(zp1);
    free(scale2);
    free(zp2);
    return AEE_ENOMEMORY;
  }

  hexkl_dma_ring_reset();

  uint64_t p0 = 0;
  int rc = AEE_SUCCESS;

  // gate_up's weight blocks (nothing to prefetch it behind), then down's
  // weight starts streaming for real -- issued once; every row block below
  // reuses them.
  {
    const uint32_t rs1 = dma_row_size_dividing(wb1);
    hexkl_dma_ring_push2d(vtcm_base + w1_off, h_gu->wh_bytes, rs1, rs1, rs1,
                          wb1 / rs1, /*src_vtcm=*/0, /*dst_vtcm=*/1);
    HEXKL_PROBE_T0(p0);
    hexkl_dma_ring_drain();
    HEXKL_PROBE_ADD(HEXKL_PROBE_DRAIN, p0);

    const uint32_t rs2 = dma_row_size_dividing(wb2);
    hexkl_dma_ring_push2d(vtcm_base + w2_off, h_dn->wh_bytes, rs2, rs2, rs2,
                          wb2 / rs2, /*src_vtcm=*/0, /*dst_vtcm=*/1);
  }

  // The whole pipeline runs per 64-row block through the fixed one-block
  // regions: quant -> gate_up matmul -> dequant to gate|up -> SwiGLU ->
  // requant -> down matmul -> out. Blocks touch disjoint out rows, so the
  // accumulate semantics are unchanged.
  for (uint32_t mb = 0; mb < M; mb += HEXKL_HMX_INT8_BLOCK_N_ROW) {
    const uint32_t m_blk = (M - mb < HEXKL_HMX_INT8_BLOCK_N_ROW)
                             ? (M - mb)
                             : HEXKL_HMX_INT8_BLOCK_N_ROW;
    const float *act_blk = act_f32 + (size_t)mb * K;

    // Stage 1: K1 on this block's activation, straight into AH tiles.
    HEXKL_PROBE_T0(p0);
    hvx_quant_rows_u8_params(act_blk, m_blk, HEXKL_HMX_INT8_BLOCK_N_ROW, K,
                             scale1, zp1, o->pool);
    rc = hvx_quant_pack_u8_ah(act_blk, m_blk, HEXKL_HMX_INT8_BLOCK_N_ROW, K,
                              scale1, zp1, vtcm_base + act1_off, o->pool);
    HEXKL_PROBE_ADD(HEXKL_PROBE_QUANT, p0);
    if (rc != AEE_SUCCESS) {
      goto out;
    }

    // Stage 2a: gate_up matmul. Each 64x32 result tile dequantizes straight
    // into the gate or the up region -- gate occupies gate_up's output
    // columns [0, inter), up [inter, 2*inter) -- and stays in VTCM.
    for (uint32_t nt = 0; nt < n1_tiles; ++nt) {
      hexkl_micro_hmx_acc_clear_int32();
      for (uint32_t kt = 0; kt < k1_tiles; ++kt) {
        const uint32_t act_tile_off =
          act1_off + kt * HEXKL_HMX_ACTIVATION_ALIGNMENT;
        const uint32_t w_tile_off =
          w1_off + (kt * n1_tiles + nt) * WEIGHT_TILE_BYTES_U8I4;
        rc = mm_u8i4_tile(vtcm_base, act_tile_off, w_tile_off);
        if (rc != AEE_SUCCESS) {
          goto out;
        }
      }
      HEXKL_PROBE_T0(p0);
      rc = hexkl_micro_hmx_acc_read_int32(vtcm_base, config_off, result_off);
      HEXKL_PROBE_ADD(HEXKL_PROBE_ACC_READ, p0);
      if (rc != AEE_SUCCESS) {
        goto out;
      }
      const int32_t *tile =
        (const int32_t *)(vtcm_base + result_off) + acc_layout->base;
      const uint32_t c0 = nt * HEXKL_ACC_TILE_COLS;
      float *dst =
        (float *)(vtcm_base + (c0 < inter ? inter_gate_off : inter_up_off)) +
        (c0 < inter ? c0 : c0 - inter);
      HEXKL_PROBE_T0(p0);
      hvx_dequant_acc_tile_to_f32(tile, acc_layout->row_stride, m_blk, scale1,
                                  zp1, h_gu->colsum_w + c0, h_gu->w_scale + c0,
                                  h_gu->bias + c0, dst, inter,
                                  /*accumulate=*/0);
      HEXKL_PROBE_ADD(HEXKL_PROBE_DEQUANT, p0);
    }

    // down's weight has now had this block's gate_up matmul + dequant to
    // land behind -- block 0 drains the whole prefetch; later blocks find
    // the ring already empty.
    HEXKL_PROBE_T0(p0);
    hexkl_dma_ring_drain();
    HEXKL_PROBE_ADD(HEXKL_PROBE_DRAIN, p0);

    // Stage 2b: SwiGLU in place (gate half becomes the SwiGLU output; the up
    // half is dead after this), then requantize it dynamically per row.
    HEXKL_PROBE_T0(p0);
    hvx_swiglu_inplace_f32((float *)(vtcm_base + inter_gate_off),
                           (const float *)(vtcm_base + inter_up_off), m_blk,
                           inter, o->pool);
    HEXKL_PROBE_ADD(HEXKL_PROBE_SWIGLU, p0);

    HEXKL_PROBE_T0(p0);
    hvx_quant_rows_u8_params((const float *)(vtcm_base + inter_gate_off), m_blk,
                             HEXKL_HMX_INT8_BLOCK_N_ROW, inter, scale2, zp2,
                             o->pool);
    rc = hvx_quant_pack_u8_ah((const float *)(vtcm_base + inter_gate_off),
                              m_blk, HEXKL_HMX_INT8_BLOCK_N_ROW, inter, scale2,
                              zp2, vtcm_base + act2_off, o->pool);
    HEXKL_PROBE_ADD(HEXKL_PROBE_QUANT, p0);
    if (rc != AEE_SUCCESS) {
      goto out;
    }

    // Stage 3: down matmul, dequant straight to the caller's out rows.
    for (uint32_t nt = 0; nt < n2_tiles; ++nt) {
      hexkl_micro_hmx_acc_clear_int32();
      for (uint32_t kt = 0; kt < k2_tiles; ++kt) {
        const uint32_t act_tile_off =
          act2_off + kt * HEXKL_HMX_ACTIVATION_ALIGNMENT;
        const uint32_t w_tile_off =
          w2_off + (kt * n2_tiles + nt) * WEIGHT_TILE_BYTES_U8I4;
        rc = mm_u8i4_tile(vtcm_base, act_tile_off, w_tile_off);
        if (rc != AEE_SUCCESS) {
          goto out;
        }
      }
      HEXKL_PROBE_T0(p0);
      rc = hexkl_micro_hmx_acc_read_int32(vtcm_base, config_off, result_off);
      HEXKL_PROBE_ADD(HEXKL_PROBE_ACC_READ, p0);
      if (rc != AEE_SUCCESS) {
        goto out;
      }
      const int32_t *tile =
        (const int32_t *)(vtcm_base + result_off) + acc_layout->base;
      const uint32_t c0 = nt * HEXKL_ACC_TILE_COLS;
      HEXKL_PROBE_T0(p0);
      hvx_dequant_acc_tile_to_f32(
        tile, acc_layout->row_stride, m_blk, scale2, zp2, h_dn->colsum_w + c0,
        h_dn->w_scale + c0, h_dn->bias + c0, out + (size_t)mb * n_out + c0,
        n_out, o->accumulate);
      HEXKL_PROBE_ADD(HEXKL_PROBE_DEQUANT, p0);
    }
  }

out:
  free(scale1);
  free(zp1);
  free(scale2);
  free(zp2);
  return rc;
}

/**
 * @brief [L2, split-call variant] gate_up matmul -> SwiGLU -> requantize to
 *        u8 AH tiles, ONE weight, no down matmul.
 *
 * hexkl_mm_u8i4_fused_run (above) does gate_up AND down in one call, with
 * a six-region VTCM layout carved by hand -- that shipped a real, still
 * unexplained correctness bug on this model's real weights (doc 43 §7's
 * L2 rows), passing its own synthetic-data unit test throughout. This is
 * the smaller-surface alternative: ONE weight, ONE region set (no dual-
 * weight double-buffer, no down-side accumulator reuse to get wrong), and
 * the down matmul is NOT reimplemented here -- the caller feeds this
 * function's out_ah/out_scale/out_zp straight into the ALREADY-DEVICE-
 * VERIFIED hexkl_mm_u8i4_layer_run via its act_ah_prepacked path (the same
 * one mm_u8i4_layer_u8in exposes), so half of the fused kernel's risk
 * surface is "reuse code nobody had to write new."
 *
 * SwiGLU still never leaves the DSP: gate_up's dequantized output lands in
 * VTCM (gate half, up half), hvx_swiglu_inplace_f32 runs on it there, and
 * the result is requantized straight into out_ah without a DDR round trip.
 * What DOES cross FastRPC that a fully-fused call would not: the
 * requantized u8 intermediate (1 byte/value + a scale/zp per row) instead
 * of nothing -- still 4x smaller than the pre-L2 unfused path's f32
 * intermediate, just not zero.
 *
 * @param[in]  handle_gate_up  registered handle, N must be even (gate|up)
 * @param[out] out_ah     m_pad(M) * inter bytes, AH-tiled -- inter =
 *                        registered N / 2. Same layout htp_act_quant.h's
 *                        htp_quant_pack_u8_ah produces, and exactly what
 *                        hexkl_mm_opts.act_ah_prepacked expects.
 * @param[out] out_scale, out_zp  m_pad(M) entries each
 * @return AEE_SUCCESS, AEE_EBADPARM on a shape violation, AEE_ENOMEMORY if
 *         the block layout does not fit VTCM, AEE_EUNSUPPORTED if the
 *         in-place accumulator tile layout is not available (no vendor-
 *         copy fallback exists for the split dequant, same restriction as
 *         hexkl_mm_u8i4_fused_run above).
 */
int hexkl_mm_u8i4_gate_up_swiglu_run(hexkl_weight_u8i4_table *tbl,
                                     uint8_t *vtcm_base, uint32_t vtcm_size,
                                     uint32_t config_off, uint32_t M,
                                     uint32_t K, uint32_t handle_gate_up,
                                     const float *act_f32, uint8_t *out_ah,
                                     float *out_scale, int32_t *out_zp,
                                     hvx_worker_pool *pool) {
  if (!tbl || !vtcm_base || !act_f32 || !out_ah || !out_scale || !out_zp ||
      M == 0 || K == 0) {
    return AEE_EBADPARM;
  }
  if ((K % HEXKL_HMX_INT8_BLOCK_N_INNER) != 0) {
    return AEE_EBADPARM;
  }
  if (handle_gate_up >= HEXKL_MM_U8I4_MAX_WEIGHTS ||
      !tbl->slots[handle_gate_up].in_use) {
    return AEE_EBADPARM;
  }
  const hexkl_weight_u8i4 *h_gu = &tbl->slots[handle_gate_up];
  if (h_gu->K != K || (h_gu->N % 2) != 0) {
    return AEE_EBADPARM;
  }
  const uint32_t inter = h_gu->N / 2;
  // inter must itself tile-divide: it becomes the down matmul's K in the
  // caller's next call, and it is this call's own requant width.
  if ((inter % HEXKL_HMX_INT8_BLOCK_N_INNER) != 0) {
    return AEE_EBADPARM;
  }

  const uint32_t k1_tiles = K / HEXKL_HMX_INT8_BLOCK_N_INNER;
  const uint32_t n1_tiles = h_gu->N / HEXKL_HMX_INT8_BLOCK_N_COL;
  const uint32_t inter_ktiles = inter / HEXKL_HMX_INT8_BLOCK_N_INNER;

  // VTCM layout, sized for ONE 64-row block: activation | gate_up weight
  // (single-buffered -- one weight, nothing to prefetch it behind) |
  // SwiGLU gate | SwiGLU up | one result tile. No down-side regions at
  // all, unlike hexkl_mm_u8i4_fused_run's six.
  const uint32_t act_bytes = k1_tiles * HEXKL_HMX_ACTIVATION_ALIGNMENT;
  const uint32_t wb1 = k1_tiles * n1_tiles * WEIGHT_TILE_BYTES_U8I4;
  const uint32_t inter_row_bytes = HEXKL_HMX_INT8_BLOCK_N_ROW * inter * 4u;

  const uint32_t act_off = 0;
  const uint32_t w_off =
    ROUND_UP_U32(act_off + act_bytes, HEXKL_HMX_ACTIVATION_ALIGNMENT);
  const uint32_t gate_off =
    ROUND_UP_U32(w_off + wb1, HEXKL_HMX_ACTIVATION_ALIGNMENT);
  const uint32_t up_off =
    ROUND_UP_U32(gate_off + inter_row_bytes, HEXKL_HMX_ACTIVATION_ALIGNMENT);
  const uint32_t result_off =
    ROUND_UP_U32(up_off + inter_row_bytes, HEXKL_HMX_ACTIVATION_ALIGNMENT);
  if (result_off + ACC_TILE_BYTES > config_off ||
      result_off + ACC_TILE_BYTES > vtcm_size) {
    return AEE_ENOMEMORY;
  }

  const hexkl_acc_layout *acc_layout =
    hexkl_acc_layout_get(vtcm_base, result_off);
  hexkl_probe_us[HEXKL_PROBE_ACC_STRIDE] =
    acc_layout->usable ? acc_layout->row_stride : 0u;
  if (!acc_layout->usable) {
    return AEE_EUNSUPPORTED;
  }

  float *scale1 = (float *)malloc(sizeof(float) * HEXKL_HMX_INT8_BLOCK_N_ROW);
  int32_t *zp1 =
    (int32_t *)malloc(sizeof(int32_t) * HEXKL_HMX_INT8_BLOCK_N_ROW);
  if (!scale1 || !zp1) {
    free(scale1);
    free(zp1);
    return AEE_ENOMEMORY;
  }

  hexkl_dma_ring_reset();
  int rc = AEE_SUCCESS;
  uint64_t p0 = 0;

  // Issue gate_up's weight once. One weight, so there is no cross-matmul
  // handle to prefetch it behind the way hexkl_mm_u8i4_layer_run's
  // multi-handle case does -- the wait for it sits after the first block's
  // quantization below instead, which is the only work available to cover
  // it.
  {
    const uint32_t rs = dma_row_size_dividing(wb1);
    hexkl_dma_ring_push2d(vtcm_base + w_off, h_gu->wh_bytes, rs, rs, rs,
                          wb1 / rs, /*src_vtcm=*/0, /*dst_vtcm=*/1);
  }

  for (uint32_t mb = 0; mb < M; mb += HEXKL_HMX_INT8_BLOCK_N_ROW) {
    const uint32_t m_blk = (M - mb < HEXKL_HMX_INT8_BLOCK_N_ROW)
                             ? (M - mb)
                             : HEXKL_HMX_INT8_BLOCK_N_ROW;
    const float *act_blk = act_f32 + (size_t)mb * K;

    HEXKL_PROBE_T0(p0);
    hvx_quant_rows_u8_params(act_blk, m_blk, HEXKL_HMX_INT8_BLOCK_N_ROW, K,
                             scale1, zp1, pool);
    rc = hvx_quant_pack_u8_ah(act_blk, m_blk, HEXKL_HMX_INT8_BLOCK_N_ROW, K,
                              scale1, zp1, vtcm_base + act_off, pool);
    HEXKL_PROBE_ADD(HEXKL_PROBE_QUANT, p0);
    if (rc != AEE_SUCCESS) {
      goto out;
    }

    /** The weight transfer was issued before this loop and waited for
       here, after the first block's quantization -- which needs no weight,
       so it covers part of the transfer instead of running after it. Only
       the first block waits: the weight is loaded once and every later
       block reuses it. */
    if (mb == 0) {
      HEXKL_PROBE_T0(p0);
      hexkl_dma_ring_drain();
      HEXKL_PROBE_ADD(HEXKL_PROBE_DRAIN, p0);
    }

    for (uint32_t nt = 0; nt < n1_tiles; ++nt) {
      hexkl_micro_hmx_acc_clear_int32();
      for (uint32_t kt = 0; kt < k1_tiles; ++kt) {
        const uint32_t act_tile_off =
          act_off + kt * HEXKL_HMX_ACTIVATION_ALIGNMENT;
        const uint32_t w_tile_off =
          w_off + (kt * n1_tiles + nt) * WEIGHT_TILE_BYTES_U8I4;
        rc = mm_u8i4_tile(vtcm_base, act_tile_off, w_tile_off);
        if (rc != AEE_SUCCESS) {
          goto out;
        }
      }
      HEXKL_PROBE_T0(p0);
      rc = hexkl_micro_hmx_acc_read_int32(vtcm_base, config_off, result_off);
      HEXKL_PROBE_ADD(HEXKL_PROBE_ACC_READ, p0);
      if (rc != AEE_SUCCESS) {
        goto out;
      }
      const int32_t *tile =
        (const int32_t *)(vtcm_base + result_off) + acc_layout->base;
      const uint32_t c0 = nt * HEXKL_ACC_TILE_COLS;
      // gate occupies gate_up's output columns [0, inter), up [inter,
      // 2*inter) -- same convention hexkl_mm_u8i4_fused_run uses, and the
      // one hvx_swiglu_inplace_f32's contract assumes.
      float *dst = (float *)(vtcm_base + (c0 < inter ? gate_off : up_off)) +
                   (c0 < inter ? c0 : c0 - inter);
      HEXKL_PROBE_T0(p0);
      hvx_dequant_acc_tile_to_f32(tile, acc_layout->row_stride, m_blk, scale1,
                                  zp1, h_gu->colsum_w + c0, h_gu->w_scale + c0,
                                  h_gu->bias + c0, dst, inter,
                                  /*accumulate=*/0);
      HEXKL_PROBE_ADD(HEXKL_PROBE_DEQUANT, p0);
    }

    HEXKL_PROBE_T0(p0);
    hvx_swiglu_inplace_f32((float *)(vtcm_base + gate_off),
                           (const float *)(vtcm_base + up_off), m_blk, inter,
                           pool);
    HEXKL_PROBE_ADD(HEXKL_PROBE_SWIGLU, p0);

    // Requantize this block's SwiGLU output (in gate_off, in place) into
    // the caller's out_ah/out_scale/out_zp. The block's global row-block
    // index is mb/64 -- offsetting the destination pointer by that many
    // whole row-blocks' worth of bytes before calling hvx_quant_pack_u8_ah
    // with a LOCAL (block-relative) m_valid/m_pad makes its internal
    // rb=m/64 (always 0 for m < 64 here) land at the correct GLOBAL tile
    // address, identical to what one m_pad(M)-sized call would produce --
    // AH tiling's row-block byte stride is fixed (inter_ktiles*2048)
    // regardless of how many blocks the caller happens to split M into.
    HEXKL_PROBE_T0(p0);
    hvx_quant_rows_u8_params((const float *)(vtcm_base + gate_off), m_blk,
                             HEXKL_HMX_INT8_BLOCK_N_ROW, inter, scale1, zp1,
                             pool);
    for (uint32_t r = 0; r < m_blk; ++r) {
      out_scale[mb + r] = scale1[r];
      out_zp[mb + r] = zp1[r];
    }
    uint8_t *out_ah_blk =
      out_ah + (size_t)(mb / HEXKL_HMX_INT8_BLOCK_N_ROW) * inter_ktiles * 2048u;
    rc = hvx_quant_pack_u8_ah((const float *)(vtcm_base + gate_off), m_blk,
                              HEXKL_HMX_INT8_BLOCK_N_ROW, inter, scale1, zp1,
                              out_ah_blk, pool);
    HEXKL_PROBE_ADD(HEXKL_PROBE_QUANT, p0);
    if (rc != AEE_SUCCESS) {
      goto out;
    }
  }

out:
  free(scale1);
  free(zp1);
  return rc;
}
