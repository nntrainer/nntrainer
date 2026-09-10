// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hexkl_mm_u8i4_moe.c
 * @date   10 Sep 2026
 * @brief  A whole MoE FFN layer -- every expert -- in one call
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Design, and the arithmetic every constant here is checked against:
 * docs/htp_attention/46_moe_resident_kernel_design.md.
 *
 * Two things about this file are deliberate and easy to undo by accident.
 *
 * THE WEIGHTS ARE NOT DOUBLE-BUFFERED. Two copies of gate_up and down is
 * 10.5 MB against an 8.3 MB arena, so instead each buffer is reused with a
 * time offset: while expert e's down matmul runs, the gate_up buffer is
 * already dead and e+1's gate_up is pulled into it. That is the whole
 * reason the layout fits, and it is what keeps the weight DMA hidden --
 * doc 44 section 14.3 measured what happens when it is not: the u8in call
 * has no activation quant for the DMA to hide behind and all 5.18 ms/layer
 * of it is exposed.
 *
 * res_f32 ALIASES gate. By the time down's result is dequantized, gate has
 * been SwiGLU'd in place and requantized into mid, so its bytes are dead.
 * Reusing them is what keeps a [64 x N_out] f32 block out of the budget.
 * If a future change reads gate after the requant, this aliasing silently
 * corrupts it -- hexkl_mm_u8i4_moe_layout asserts the sizes but cannot
 * assert the lifetime.
 */

#include "hexkl_mm_u8i4_moe.h"

#include <stdlib.h>
#include <string.h>

#include <AEEStdErr.h>

#include "hexkl_acc_tile.h"
#include "hexkl_dma_ring.h"
#include "hexkl_micro.h"
#include "hexkl_probe.h"
#include "hvx_dequant_i32.h"
#include "hvx_gather_ah_u8.h"
#include "hvx_quant_u8.h"
#include "hvx_scale_add_f32.h"
#include "hvx_swiglu_f32.h"

#define ROUND_UP_U32(v, a) ((((v) + ((a)-1)) / (a)) * (a))

/** @brief Kept in sync with hexkl_mm_u8i4_dma.c's copies by inspection, the
 *         same file-scoped constant practice that file documents. */
#define WEIGHT_TILE_BYTES_U8I4 512u
#define ACC_TILE_BYTES 8192u
#define MAX_DMA_ROW_BYTES 16384u

/** @brief Largest power-of-two row size that divides the transfer, so a
 *         2D descriptor covers it exactly. Same rule as
 *         hexkl_mm_u8i4_dma.c's dma_row_size_dividing. */
static uint32_t moe_dma_row_size(uint32_t total_bytes) {
  uint32_t rs = MAX_DMA_ROW_BYTES;
  while (rs > 1u && (total_bytes % rs) != 0u) {
    rs >>= 1;
  }
  return rs;
}

/** @brief Pushes a registered weight's WH bytes into a VTCM buffer. */
static void moe_push_weight(uint8_t *vtcm_base, uint32_t dst_off,
                            const hexkl_weight_u8i4 *h, uint32_t K) {
  const uint32_t bytes = (K / HEXKL_HMX_INT8_BLOCK_N_INNER) *
                         (h->N / HEXKL_HMX_INT8_BLOCK_N_COL) *
                         WEIGHT_TILE_BYTES_U8I4;
  const uint32_t rs = moe_dma_row_size(bytes);
  hexkl_dma_ring_push2d(vtcm_base + dst_off, h->wh_bytes, rs, rs, rs,
                        bytes / rs, /*src_vtcm=*/0, /*dst_vtcm=*/1);
}

/**
 * @brief hvx_worker_pool_func body for the routing multiply and scatter-add.
 *
 * Safe to split by row because a token picks k DISTINCT experts, so within
 * one expert's block every row_index is different and no two workers touch
 * the same output row. Across blocks and experts they do collide, which is
 * why the split is inside a block and the blocks stay sequential.
 */
typedef struct {
  float *out;
  const float *res;
  const uint32_t *rows;
  const float *weights;
  uint32_t n_rows;
  uint32_t N_out;
} moe_scatter_ctx;

static void moe_scatter_worker(uint32_t n_threads, uint32_t i, void *vctx) {
  moe_scatter_ctx *c = (moe_scatter_ctx *)vctx;
  const uint32_t lo = (uint32_t)((uint64_t)c->n_rows * i / n_threads);
  const uint32_t hi = (uint32_t)((uint64_t)c->n_rows * (i + 1) / n_threads);
  for (uint32_t r = lo; r < hi; ++r) {
    hvx_scale_add_rows_f32(c->out + (size_t)c->rows[r] * c->N_out,
                           c->res + (size_t)r * c->N_out, c->weights[r],
                           c->N_out);
  }
}

int hexkl_mm_u8i4_moe_layout(uint32_t K, uint32_t inter, uint32_t N_out,
                             uint32_t arena_bytes, hexkl_moe_layout *out) {
  if (!out || K == 0u || inter == 0u || N_out == 0u) {
    return AEE_EBADPARM;
  }
  if ((K % HEXKL_HMX_INT8_BLOCK_N_INNER) != 0u ||
      (inter % HEXKL_HMX_INT8_BLOCK_N_INNER) != 0u ||
      (N_out % HEXKL_HMX_INT8_BLOCK_N_COL) != 0u ||
      ((2u * inter) % HEXKL_HMX_INT8_BLOCK_N_COL) != 0u) {
    return AEE_EBADPARM;
  }

  const uint32_t k_tiles = K / HEXKL_HMX_INT8_BLOCK_N_INNER;
  const uint32_t gu_tiles =
    k_tiles * ((2u * inter) / HEXKL_HMX_INT8_BLOCK_N_COL);
  const uint32_t dn_tiles = (inter / HEXKL_HMX_INT8_BLOCK_N_INNER) *
                            (N_out / HEXKL_HMX_INT8_BLOCK_N_COL);
  const uint32_t inter_ktiles = inter / HEXKL_HMX_INT8_BLOCK_N_INNER;

  const uint32_t act_bytes = k_tiles * HEXKL_HMX_ACTIVATION_ALIGNMENT;
  const uint32_t gu_bytes = gu_tiles * WEIGHT_TILE_BYTES_U8I4;
  const uint32_t dn_bytes = dn_tiles * WEIGHT_TILE_BYTES_U8I4;
  const uint32_t inter_rows = HEXKL_HMX_INT8_BLOCK_N_ROW * inter * 4u;
  const uint32_t mid_bytes = inter_ktiles * HEXKL_HMX_ACTIVATION_ALIGNMENT;
  const uint32_t res_bytes = HEXKL_HMX_INT8_BLOCK_N_ROW * N_out * 4u;

  hexkl_moe_layout L;
  L.act_off = 0u;
  L.w_gu_off =
    ROUND_UP_U32(L.act_off + act_bytes, HEXKL_HMX_ACTIVATION_ALIGNMENT);
  L.w_dn_off =
    ROUND_UP_U32(L.w_gu_off + gu_bytes, HEXKL_HMX_ACTIVATION_ALIGNMENT);
  L.gate_off =
    ROUND_UP_U32(L.w_dn_off + dn_bytes, HEXKL_HMX_ACTIVATION_ALIGNMENT);
  L.up_off =
    ROUND_UP_U32(L.gate_off + inter_rows, HEXKL_HMX_ACTIVATION_ALIGNMENT);
  L.mid_off =
    ROUND_UP_U32(L.up_off + inter_rows, HEXKL_HMX_ACTIVATION_ALIGNMENT);
  L.result_off =
    ROUND_UP_U32(L.mid_off + mid_bytes, HEXKL_HMX_ACTIVATION_ALIGNMENT);
  L.total = L.result_off + ACC_TILE_BYTES;

  /* down's dequantized block lands on top of gate, whose bytes are dead by
     then. The two regions the alias covers -- gate and up -- are adjacent,
     so the check is against both together. */
  L.res_f32_off = L.gate_off;
  if (res_bytes > (L.mid_off - L.gate_off)) {
    return AEE_ENOMEMORY;
  }
  if (L.total > arena_bytes) {
    return AEE_ENOMEMORY;
  }

  *out = L;
  return AEE_SUCCESS;
}

int hexkl_mm_u8i4_moe_layer_run(
  hexkl_weight_u8i4_table *tbl, uint8_t *vtcm_base, uint32_t vtcm_size,
  uint32_t config_off, uint32_t M, uint32_t K, uint32_t inter, uint32_t N_out,
  uint32_t n_experts, const uint32_t *h_gate_up, const uint32_t *h_down,
  const uint32_t *row_index, const uint32_t *row_count, const float *row_weight,
  const float *act_f32, float *out_f32, hvx_worker_pool *pool) {

  if (!tbl || !vtcm_base || !h_gate_up || !h_down || !row_index || !row_count ||
      !row_weight || !act_f32 || !out_f32 || M == 0u || n_experts == 0u) {
    return AEE_EBADPARM;
  }

  const uint32_t arena = vtcm_size < config_off ? vtcm_size : config_off;
  hexkl_moe_layout L;
  int rc = hexkl_mm_u8i4_moe_layout(K, inter, N_out, arena, &L);
  if (rc != AEE_SUCCESS) {
    return rc;
  }

  /* Validate every handle and its shape before any work: a bad handle
     found halfway through would leave out_f32 partly written, and the
     caller cannot tell that from a correct result. */
  uint32_t n_rows = 0u;
  for (uint32_t e = 0; e < n_experts; ++e) {
    if (row_count[e] == 0u) {
      continue;
    }
    if (h_gate_up[e] >= HEXKL_MM_U8I4_MAX_WEIGHTS ||
        h_down[e] >= HEXKL_MM_U8I4_MAX_WEIGHTS) {
      return AEE_EBADPARM;
    }
    const hexkl_weight_u8i4 *g = &tbl->slots[h_gate_up[e]];
    const hexkl_weight_u8i4 *d = &tbl->slots[h_down[e]];
    if (!g->in_use || !d->in_use || g->K != K || g->N != 2u * inter ||
        d->K != inter || d->N != N_out) {
      return AEE_EBADPARM;
    }
    n_rows += row_count[e];
  }
  for (uint32_t i = 0; i < n_rows; ++i) {
    if (row_index[i] >= M) {
      return AEE_EBADPARM;
    }
  }

  const hexkl_acc_layout *acc = hexkl_acc_layout_get(vtcm_base, L.result_off);
  if (!acc->usable) {
    return AEE_EUNSUPPORTED;
  }
  hexkl_probe_us[HEXKL_PROBE_ACC_STRIDE] = acc->row_stride;

  const uint32_t k_tiles = K / HEXKL_HMX_INT8_BLOCK_N_INNER;
  const uint32_t gu_ntiles = (2u * inter) / HEXKL_HMX_INT8_BLOCK_N_COL;
  const uint32_t inter_ktiles = inter / HEXKL_HMX_INT8_BLOCK_N_INNER;
  const uint32_t dn_ntiles = N_out / HEXKL_HMX_INT8_BLOCK_N_COL;
  const uint32_t BR = HEXKL_HMX_INT8_BLOCK_N_ROW;

  /* Scratch on the DSP heap, not VTCM: these are per-block staging areas
     read and written once each, so VTCM buys them nothing and the arena is
     the scarce resource. */
  float *scale = (float *)malloc(sizeof(float) * BR);
  int32_t *zp = (int32_t *)malloc(sizeof(int32_t) * BR);
  /* The whole activation, quantized once. hvx_quant_pack_u8_ah writes it,
     so a row's bytes are exactly what quantizing that row inside any expert
     block would have produced -- row quantization is independent of how
     rows are grouped -- and the per-block work becomes a uint8 move rather
     than a scan and a pack of the same rows four times over at top-4. */
  const uint32_t m_pad = ROUND_UP_U32(M, BR);
  uint8_t *act_ah = (uint8_t *)malloc((size_t)m_pad * K);
  float *scale_all = (float *)malloc(sizeof(float) * m_pad);
  int32_t *zp_all = (int32_t *)malloc(sizeof(int32_t) * m_pad);
  /* act_f32 and out_f32 are the host's FastRPC buffers, which are rpcmem
     and therefore UNCACHED. The gather reads M*K*4 bytes of act four times
     over at top-4 routing, and the scatter is a read-modify-write of
     M*N_out*4 one float at a time -- both catastrophic against uncached
     DDR, and measured so: 17.3 ms of gather+quant and 104.8 ms of scatter
     against 3.6 and 2.8 for the path this replaces (doc 46 section 10).
     Same axis Q1 broke on, doc 44 section 10.1.
     So each is touched exactly once, sequentially: act is copied in at the
     start and out is copied out at the end, and everything in between
     works on cached heap. */
  float *act_c = (float *)malloc(sizeof(float) * (size_t)M * K);
  float *out_c = (float *)malloc(sizeof(float) * (size_t)M * N_out);
  uint32_t *order = (uint32_t *)malloc(sizeof(uint32_t) * n_experts);
  /* row_index is grouped by expert, so each active expert needs the offset
     its group starts at. Heap, not a stack array sized by
     HEXKL_MM_U8I4_MAX_WEIGHTS: that constant is 2048 and has nothing to do
     with how many experts this layer has. */
  uint32_t *base_of = (uint32_t *)malloc(sizeof(uint32_t) * n_experts);
  uint32_t n_active = 0u;
  uint64_t p0 = 0;
  if (!scale || !zp || !act_ah || !scale_all || !zp_all || !order || !base_of ||
      !act_c || !out_c) {
    rc = AEE_ENOMEMORY;
    goto out;
  }

  /* The experts that actually have rows, in order. The DMA pipeline below
     hands each expert's gate_up transfer to its predecessor, so a skipped
     expert in the middle would break the chain; compacting first means the
     pipeline never has to think about empties. */
  {
    uint32_t base = 0u;
    for (uint32_t e = 0; e < n_experts; ++e) {
      if (row_count[e] != 0u) {
        base_of[n_active] = base;
        order[n_active++] = e;
      }
      base += row_count[e];
    }
  }
  HEXKL_PROBE_T0(p0);
  memcpy(act_c, act_f32, sizeof(float) * (size_t)M * K);
  memset(out_c, 0, sizeof(float) * (size_t)M * N_out);
  HEXKL_PROBE_ADD(HEXKL_PROBE_ACC_COPY, p0);

  /* Once for the layer, from the cached copy. 444 rows here against 1776
     scanned and packed a block at a time before. */
  HEXKL_PROBE_T0(p0);
  hvx_quant_rows_u8_params(act_c, M, m_pad, K, scale_all, zp_all, pool);
  rc =
    hvx_quant_pack_u8_ah(act_c, M, m_pad, K, scale_all, zp_all, act_ah, pool);
  HEXKL_PROBE_ADD(HEXKL_PROBE_QUANT, p0);
  if (rc != AEE_SUCCESS) {
    goto out;
  }
  if (n_active == 0u) {
    memcpy(out_f32, out_c, sizeof(float) * (size_t)M * N_out);
    rc = AEE_SUCCESS;
    goto out;
  }

  hexkl_dma_ring_reset();
  moe_push_weight(vtcm_base, L.w_gu_off, &tbl->slots[h_gate_up[order[0]]], K);

  for (uint32_t i = 0; i < n_active; ++i) {
    const uint32_t e = order[i];
    const hexkl_weight_u8i4 *g = &tbl->slots[h_gate_up[e]];
    const hexkl_weight_u8i4 *d = &tbl->slots[h_down[e]];
    const uint32_t *rows = row_index + base_of[i];
    const float *weights = row_weight + base_of[i];
    const uint32_t n_e = row_count[e];

    /* gate_up[e] was pushed either before this loop or by iteration i-1,
       where it had this expert's predecessor's down matmul to hide behind. */
    HEXKL_PROBE_T0(p0);
    hexkl_dma_ring_drain();
    HEXKL_PROBE_ADD(HEXKL_PROBE_DRAIN, p0);

    /* down[e] goes out now so the first block's gate_up matmul covers it. */
    moe_push_weight(vtcm_base, L.w_dn_off, d, inter);

    for (uint32_t mb = 0; mb < n_e; mb += BR) {
      const uint32_t m_blk = (n_e - mb < BR) ? (n_e - mb) : BR;
      const int last_block = (mb + BR >= n_e);

      /* Gather this block's token rows into a contiguous staging buffer and
         quantize them. Gathering f32 and quantizing per block -- rather
         than quantizing all M rows once and gathering uint8 -- is what
         makes this bit-identical to the path it replaces: the same rows go
         through the same scan and the same pack, in the same grouping. The
         uint8 version reads 4x fewer bytes and is the obvious follow-up,
         but it changes what is compared, so it does not belong in the
         commit that has to prove equivalence. */
      HEXKL_PROBE_T0(p0);
      hvx_gather_ah_u8(vtcm_base + L.act_off, act_ah, rows + mb, m_blk, K,
                       pool);
      for (uint32_t r = 0; r < m_blk; ++r) {
        scale[r] = scale_all[rows[mb + r]];
        zp[r] = zp_all[rows[mb + r]];
      }
      HEXKL_PROBE_ADD(HEXKL_PROBE_QUANT, p0);

      /* --- gate_up ------------------------------------------------- */
      for (uint32_t nt = 0; nt < gu_ntiles; ++nt) {
        hexkl_micro_hmx_acc_clear_int32();
        for (uint32_t kt = 0; kt < k_tiles; ++kt) {
          rc = hexkl_micro_hmx_mm_u8i4(
            vtcm_base, L.act_off + kt * HEXKL_HMX_ACTIVATION_ALIGNMENT,
            L.w_gu_off + (kt * gu_ntiles + nt) * WEIGHT_TILE_BYTES_U8I4);
          if (rc != AEE_SUCCESS) {
            goto out;
          }
        }
        HEXKL_PROBE_T0(p0);
        rc =
          hexkl_micro_hmx_acc_read_int32(vtcm_base, config_off, L.result_off);
        HEXKL_PROBE_ADD(HEXKL_PROBE_ACC_READ, p0);
        if (rc != AEE_SUCCESS) {
          goto out;
        }
        const int32_t *tile =
          (const int32_t *)(vtcm_base + L.result_off) + acc->base;
        const uint32_t c0 = nt * HEXKL_ACC_TILE_COLS;
        /* gate is gate_up's columns [0, inter), up is [inter, 2*inter) --
           the convention hvx_swiglu_inplace_f32's contract assumes. */
        float *dst =
          (float *)(vtcm_base + (c0 < inter ? L.gate_off : L.up_off)) +
          (c0 < inter ? c0 : c0 - inter);
        HEXKL_PROBE_T0(p0);
        hvx_dequant_acc_tile_to_f32(tile, acc->row_stride, m_blk, scale, zp,
                                    g->colsum_w + c0, g->w_scale + c0,
                                    g->bias + c0, dst, inter,
                                    /*accumulate=*/0);
        HEXKL_PROBE_ADD(HEXKL_PROBE_DEQUANT, p0);
      }

      /* down[e]'s transfer is waited for here, after a whole gate_up
         matmul has run over it. Draining before the next push, not after,
         is what keeps this wait off the gate_up prefetch below -- the ring
         drains everything pending, so a push issued first would be waited
         for too and the pipeline would collapse into a serial chain. */
      if (mb == 0u) {
        HEXKL_PROBE_T0(p0);
        hexkl_dma_ring_drain();
        HEXKL_PROBE_ADD(HEXKL_PROBE_DRAIN, p0);
      }

      /* A is dead once the last block's gate_up matmul is done, so the next
         expert's gate_up goes out now and rides under this block's SwiGLU,
         requantization and down matmul. */
      if (last_block && i + 1u < n_active) {
        moe_push_weight(vtcm_base, L.w_gu_off,
                        &tbl->slots[h_gate_up[order[i + 1u]]], K);
      }

      HEXKL_PROBE_T0(p0);
      hvx_swiglu_inplace_f32((float *)(vtcm_base + L.gate_off),
                             (const float *)(vtcm_base + L.up_off), m_blk,
                             inter, pool);
      HEXKL_PROBE_ADD(HEXKL_PROBE_SWIGLU, p0);

      HEXKL_PROBE_T0(p0);
      hvx_quant_rows_u8_params((const float *)(vtcm_base + L.gate_off), m_blk,
                               BR, inter, scale, zp, pool);
      rc =
        hvx_quant_pack_u8_ah((const float *)(vtcm_base + L.gate_off), m_blk, BR,
                             inter, scale, zp, vtcm_base + L.mid_off, pool);
      HEXKL_PROBE_ADD(HEXKL_PROBE_QUANT, p0);
      if (rc != AEE_SUCCESS) {
        goto out;
      }

      /* --- down ---------------------------------------------------- */
      for (uint32_t nt = 0; nt < dn_ntiles; ++nt) {
        hexkl_micro_hmx_acc_clear_int32();
        for (uint32_t kt = 0; kt < inter_ktiles; ++kt) {
          rc = hexkl_micro_hmx_mm_u8i4(
            vtcm_base, L.mid_off + kt * HEXKL_HMX_ACTIVATION_ALIGNMENT,
            L.w_dn_off + (kt * dn_ntiles + nt) * WEIGHT_TILE_BYTES_U8I4);
          if (rc != AEE_SUCCESS) {
            goto out;
          }
        }
        HEXKL_PROBE_T0(p0);
        rc =
          hexkl_micro_hmx_acc_read_int32(vtcm_base, config_off, L.result_off);
        HEXKL_PROBE_ADD(HEXKL_PROBE_ACC_READ, p0);
        if (rc != AEE_SUCCESS) {
          goto out;
        }
        const int32_t *tile =
          (const int32_t *)(vtcm_base + L.result_off) + acc->base;
        const uint32_t c0 = nt * HEXKL_ACC_TILE_COLS;
        HEXKL_PROBE_T0(p0);
        hvx_dequant_acc_tile_to_f32(
          tile, acc->row_stride, m_blk, scale, zp, d->colsum_w + c0,
          d->w_scale + c0, d->bias + c0,
          (float *)(vtcm_base + L.res_f32_off) + c0, N_out, /*accumulate=*/0);
        HEXKL_PROBE_ADD(HEXKL_PROBE_DEQUANT, p0);
      }

      /* Routing multiply and scatter-add in one pass. The ARM side does
         these as two loops building a Tensor per token per expert (doc 44
         section 16.3); here the block is already in VTCM and the
         destination row is one index away. */
      HEXKL_PROBE_T0(p0);
      {
        moe_scatter_ctx sc = {
          out_c,     (const float *)(vtcm_base + L.res_f32_off),
          rows + mb, weights + mb,
          m_blk,     N_out};
        hvx_worker_pool_run(pool, moe_scatter_worker, &sc, m_blk);
      }
      HEXKL_PROBE_ADD(HEXKL_PROBE_SCATTER, p0);
    }
  }

  HEXKL_PROBE_T0(p0);
  memcpy(out_f32, out_c, sizeof(float) * (size_t)M * N_out);
  HEXKL_PROBE_ADD(HEXKL_PROBE_ACC_COPY, p0);

out:
  free(scale);
  free(zp);
  free(act_ah);
  free(scale_all);
  free(zp_all);
  free(order);
  free(base_of);
  free(act_c);
  free(out_c);
  return rc;
}
