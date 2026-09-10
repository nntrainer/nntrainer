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

#include <stdlib.h>
#include <string.h>

#include <AEEStdErr.h>

#include "hexkl_acc_tile.h"
#include "hexkl_dma_ring.h"
#include "hexkl_micro.h"
#include "hexkl_mm_opts.h"
#include "hvx_dequant_i32.h"
#include "hvx_quant_u8.h"
#include "hvx_swiglu_f32.h"

#include "hexkl_probe.h"

#define ROUND_UP_U32(v, a) ((((v) + ((a)-1)) / (a)) * (a))

/** @brief Bytes in one packed i4 weight tile (32x32 values). Matches
 *         hexkl_mm_u8i4.c's local definition -- both are file-scoped, so
 *         this is not a redefinition, just the same constant kept in sync
 *         by inspection. If HexKL ever exposes it from hexkl_micro.h, both
 *         files should switch to that instead of this comment. */
#define WEIGHT_TILE_BYTES_U8I4 512u
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

int hexkl_weight_u8i4_register(hexkl_weight_u8i4_table *tbl, uint8_t *vtcm_base,
                               uint32_t vtcm_size, uint32_t K, uint32_t N,
                               const int8_t *w_i4_rm, const float *w_scale,
                               const int32_t *colsum_w, const float *bias,
                               hvx_worker_pool *pool, uint32_t *out_handle) {
  if (!tbl || !vtcm_base || !w_i4_rm || !w_scale || !colsum_w || !bias ||
      !out_handle || K == 0 || N == 0) {
    return AEE_EBADPARM;
  }
  if ((K % HEXKL_HMX_INT8_BLOCK_N_INNER) != 0 ||
      (N % HEXKL_HMX_INT8_BLOCK_N_COL) != 0) {
    return AEE_EBADPARM;
  }

  uint32_t slot = HEXKL_MM_U8I4_MAX_WEIGHTS;
  for (uint32_t i = 0; i < HEXKL_MM_U8I4_MAX_WEIGHTS; ++i) {
    if (!tbl->slots[i].in_use) {
      slot = i;
      break;
    }
  }
  if (slot == HEXKL_MM_U8I4_MAX_WEIGHTS) {
    return AEE_ENOMEMORY;
  }

  const uint32_t k_tiles = K / HEXKL_HMX_INT8_BLOCK_N_INNER;
  const uint32_t n_tiles = N / HEXKL_HMX_INT8_BLOCK_N_COL;
  const uint32_t wh_bytes = k_tiles * n_tiles * WEIGHT_TILE_BYTES_U8I4;
  if (wh_bytes > vtcm_size) {
    return AEE_ENOMEMORY; // weight alone does not fit the VTCM scratch arena
  }

  // Bake every tile into VTCM (borrowed as scratch -- caller holds the HMX
  // lock and has no other VTCM use in flight), then copy the baked bytes
  // out to DSP heap memory where they stay resident across calls. A plain
  // copy, not the DMA ring: registration happens once per weight at model
  // load, off the per-token hot path, so its cost is not what this file
  // exists to optimise. The bake itself is the 34-48 ms/weight line item
  // (doc 43 §2): 7168 independent 512-byte tiles for a gate_up weight, so
  // the caller's HVX pool splits it ~6-way here -- NULL keeps the serial
  // order (one worker takes every tile, ascending).
  //
  // Verified NOT a race (doc 43 §7, corrected): an earlier session read a
  // hash mismatch as HMX-lock corruption from running this across worker
  // threads, but the "serial reference" in that comparison was itself
  // produced by hvx_worker_pool_run(NULL, ...) with n_units > 1 -- which
  // does not run the full range on one thread, it runs ONLY worker 0's
  // 1/n_units SLICE under a hypothetical n_units-way split (see that
  // function's NULL branch: func(n_units, 0, ctx), not func(1, 0, ctx)).
  // That "reference" had baked one tile and left the rest as stale VTCM
  // scratch. The REAL pool path (an actual N-worker pool, N < n_units) was
  // re-verified bit-identical to a true manual serial loop -- FNV-1a of
  // the same M/K/N matmul output matches (0x198748e597cf4105) either way.
  hexkl_bake_u8i4_ctx bc = {w_i4_rm, vtcm_base, k_tiles * n_tiles,
                            n_tiles, N,         AEE_SUCCESS};
  hvx_worker_pool_run(pool, hexkl_bake_u8i4_worker, &bc, k_tiles * n_tiles);
  if (bc.err != AEE_SUCCESS) {
    return bc.err;
  }

  hexkl_weight_u8i4 *h = &tbl->slots[slot];
  h->wh_bytes = (uint8_t *)malloc(wh_bytes);
  h->w_scale = (float *)malloc(sizeof(float) * N);
  h->colsum_w = (int32_t *)malloc(sizeof(int32_t) * N);
  h->bias = (float *)malloc(sizeof(float) * N);
  if (!h->wh_bytes || !h->w_scale || !h->colsum_w || !h->bias) {
    free(h->wh_bytes);
    free(h->w_scale);
    free(h->colsum_w);
    free(h->bias);
    memset(h, 0, sizeof(*h));
    return AEE_ENOMEMORY;
  }
  memcpy(h->wh_bytes, vtcm_base, wh_bytes);
  memcpy(h->w_scale, w_scale, sizeof(float) * N);
  memcpy(h->colsum_w, colsum_w, sizeof(int32_t) * N);
  memcpy(h->bias, bias, sizeof(float) * N);
  h->K = K;
  h->N = N;
  h->in_use = 1;

  *out_handle = slot;
  return AEE_SUCCESS;
}

int hexkl_weight_u8i4_release(hexkl_weight_u8i4_table *tbl, uint32_t handle) {
  if (!tbl || handle >= HEXKL_MM_U8I4_MAX_WEIGHTS ||
      !tbl->slots[handle].in_use) {
    return AEE_EBADPARM;
  }
  hexkl_weight_u8i4 *h = &tbl->slots[handle];
  free(h->wh_bytes);
  free(h->w_scale);
  free(h->colsum_w);
  free(h->bias);
  memset(h, 0, sizeof(*h));
  return AEE_SUCCESS;
}

int hexkl_mm_u8i4_layer_run(hexkl_weight_u8i4_table *tbl, uint8_t *vtcm_base,
                            uint32_t vtcm_size, uint32_t config_off, uint32_t M,
                            uint32_t K, const uint32_t *handles,
                            uint32_t n_handles, const float *act_f32,
                            float *out_cat, const hexkl_mm_opts *opts) {
  static const hexkl_mm_opts kDefaults = {0};
  const hexkl_mm_opts *o = opts ? opts : &kDefaults;

  if (!tbl || !vtcm_base || !handles || n_handles == 0 ||
      (!act_f32 && !o->act_ah_prepacked) || !out_cat || M == 0 || K == 0) {
    return AEE_EBADPARM;
  }
  if ((o->act_scale == NULL) != (o->act_zp == NULL)) {
    return AEE_EBADPARM; // both or neither, per hexkl_mm_opts.h
  }
  if (o->act_ah_prepacked != NULL && o->act_scale == NULL) {
    return AEE_EBADPARM; // prepacked bytes need the caller's own scale/zp
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
  // weight double-buffer (sized for the widest handle) | one result tile.
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
  if (result_off + ACC_TILE_BYTES > config_off) {
    return AEE_ENOMEMORY; // double-buffered widest weight does not fit VTCM
  }
  if (result_off + ACC_TILE_BYTES > vtcm_size) {
    return AEE_ENOMEMORY;
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
      hvx_quant_rows_u8_params(act_f32, M, m_pad, K, loc_scale, loc_zp,
                               o->pool);
      act_scale = loc_scale;
      act_zp = loc_zp;
    }
    rc0 = hvx_quant_pack_u8_ah(act_f32, M, m_pad, K, act_scale, act_zp,
                               vtcm_base + act_off, o->pool);
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
      for (uint32_t nt = 0; nt < nt_n; ++nt) {
        hexkl_micro_hmx_acc_clear_int32();
        for (uint32_t kt = 0; kt < k_tiles; ++kt) {
          const uint32_t act_tile_off =
            act_off + (rb * k_tiles + kt) * HEXKL_HMX_ACTIVATION_ALIGNMENT;
          const uint32_t w_tile_off =
            wcur + (kt * nt_n + nt) * WEIGHT_TILE_BYTES_U8I4;
          rc = hexkl_micro_hmx_mm_u8i4(vtcm_base, act_tile_off, w_tile_off);
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
        if (acc_layout->usable) {
          /** Dequantize the tile where it already is. The rows beyond M are
           * the accumulator's padding: their quantization parameters are
           * synthetic, so emitting them would be wrong, not merely wasted --
           * the same rule hvx_dequant_i32_to_f32 applies via m_valid. At
           * decode that is 62 of 64 rows never touched at all. */
          const uint32_t m0 = rb * HEXKL_ACC_TILE_ROWS;
          const uint32_t cnt =
            (m0 >= M) ? 0u
                      : ((M - m0 < HEXKL_ACC_TILE_ROWS) ? (M - m0)
                                                        : HEXKL_ACC_TILE_ROWS);
          if (cnt != 0u) {
            const int32_t *tile =
              (const int32_t *)(vtcm_base + result_off) + acc_layout->base;
            const uint32_t c0 = nt * HEXKL_ACC_TILE_COLS;
            HEXKL_PROBE_T0(p0);
            hvx_dequant_acc_tile_to_f32(
              tile, acc_layout->row_stride, cnt, act_scale + m0, act_zp + m0,
              h->colsum_w + c0, h->w_scale + c0, h->bias + c0,
              out_cat + out_off + (size_t)m0 * h->N + c0, h->N, o->accumulate);
            HEXKL_PROBE_ADD(HEXKL_PROBE_DEQUANT, p0);
          }
        } else {
          HEXKL_PROBE_T0(p0);
          rc = hexkl_micro_hmx_copy_32b_to_submatrix(
            vtcm_base, result_off, acc_scratch, rb, nt, m_pad, h->N);
          HEXKL_PROBE_ADD(HEXKL_PROBE_ACC_COPY, p0);
          if (rc != AEE_SUCCESS) {
            goto out;
          }
        }
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
  }

out:
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
        rc = hexkl_micro_hmx_mm_u8i4(vtcm_base, act_tile_off, w_tile_off);
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
        rc = hexkl_micro_hmx_mm_u8i4(vtcm_base, act_tile_off, w_tile_off);
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
        rc = hexkl_micro_hmx_mm_u8i4(vtcm_base, act_tile_off, w_tile_off);
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
