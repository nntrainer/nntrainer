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
 * THE POOL RUNS ONE JOB BEHIND THE HMX, AND THE DOWN MATMUL ONE BLOCK
 * BEHIND THE GATE_UP. Each staged batch's epilogue (dequant+SwiGLU, or
 * dequant+scatter) is submitted to the worker pool and retired only when
 * the buffer it reads or writes is about to be reused: two accumulator
 * staging buffers alternate under the HMX issue, two mid buffers under the
 * requantization, res_f32 has its own region rather than aliasing gate.
 * Block n's down is issued after block n+1's gate_up, so that block n's
 * requantization and last epilogues have a matmul to hide behind (the
 * comment above the block loop derives it). Every wait is placed where
 * the dependency actually is, and each is timed, so the profile's
 * DEQUANT, REQUANT and SCATTER columns read the exposed part, not the
 * work. Moving a submit or a wait without re-deriving who reads what is
 * how this breaks -- silently, as a plausible wrong output.
 */

#include "hexkl_mm_u8i4_moe.h"

#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#include <AEEStdErr.h>

#include "hexkl_acc_tile.h"
#include "hexkl_dma_ring.h"
#include "hexkl_dma_trace.h"
#include "hexkl_micro.h"
#include "hexkl_probe.h"
#include "hvx_dequant_i32.h"
#include "hvx_gather_ah_u8.h"
#include "hvx_gemm_u8i4_wh.h"
#include "hvx_quant_u8.h"
#include "hvx_scale_add_f32.h"

#define ROUND_UP_U32(v, a) ((((v) + ((a)-1)) / (a)) * (a))
#define ROUND_UP_SZ(v, a) ((((v) + ((a)-1)) / (a)) * (a))

/** @brief Every carve starts on a 128-byte boundary: the slot-ordered
 *         activation is a DMA source and the HVX passes read the rest. */
#define MOE_SCRATCH_ALIGN 128u

void hexkl_moe_scratch_free(hexkl_moe_scratch *s) {
  if (s) {
    free(s->raw);
    s->raw = NULL;
    s->base = NULL;
    s->cap = 0;
  }
}

/** @brief Grows the scratch to @a bytes; a no-op once it is big enough,
 *         which after the first prefill call is every call. */
int hexkl_moe_scratch_reserve(hexkl_moe_scratch *s, size_t bytes) {
  if (s->cap >= bytes) {
    return AEE_SUCCESS;
  }
  hexkl_moe_scratch_free(s);
  s->raw = malloc(bytes + MOE_SCRATCH_ALIGN);
  if (!s->raw) {
    return AEE_ENOMEMORY;
  }
  s->base = (uint8_t *)(((uintptr_t)s->raw + (MOE_SCRATCH_ALIGN - 1u)) &
                        ~(uintptr_t)(MOE_SCRATCH_ALIGN - 1u));
  s->cap = bytes;
  return AEE_SUCCESS;
}

/** @brief Hands out the next @a bytes of the scratch, aligned. The caller
 *         summed the same sizes through moe_scratch_reserve first. */
void *hexkl_moe_carve(uint8_t **cur, size_t bytes) {
  void *p = *cur;
  *cur += ROUND_UP_SZ(bytes, MOE_SCRATCH_ALIGN);
  return p;
}

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

/**
 * @brief Pushes n-tile columns [nt0, nt0+cn) of a weight, one descriptor.
 *
 * WH tiles are indexed kt*n_col + nt, so a run of n-tile columns is cn*512
 * contiguous bytes repeated k_tiles times at a n_col*512 stride -- exactly
 * a 2D transfer, and the destination keeps the same layout so the matmul's
 * indexing is unchanged. The bake order does not have to move.
 *
 * Splitting the weight this way is what lets the matmul start on column
 * nt0 while the rest is still arriving. Waiting for all 3.5 MB of a
 * gate_up first cost 110 us an expert against a 136 us transfer -- the
 * prefetch was hiding almost nothing (doc 46 section 26.4).
 *
 * @return the ring index to hand hexkl_dma_ring_wait
 */
uint32_t hexkl_moe_push_weight_chunk(uint8_t *vtcm_base, uint32_t dst_off,
                                     const hexkl_weight_u8i4 *h,
                                     uint32_t k_tiles, uint32_t n_col,
                                     uint32_t nt0, uint32_t cn) {
  const uint32_t row = cn * WEIGHT_TILE_BYTES_U8I4;
  const uint32_t stride = n_col * WEIGHT_TILE_BYTES_U8I4;
  const uint32_t off = nt0 * WEIGHT_TILE_BYTES_U8I4;
  const uint32_t idx = hexkl_dma_ring_next_idx();
  uint64_t pt = 0;
  HEXKL_PROBE_COUNT(HEXKL_PROBE_DMA_KB, (row * k_tiles) >> 10);
  HEXKL_PROBE_T0(pt);
  hexkl_dma_ring_push2d(vtcm_base + dst_off + off, h->wh_bytes + off, stride,
                        stride, row, k_tiles, /*src_vtcm=*/0, /*dst_vtcm=*/1);
  HEXKL_PROBE_ADD(HEXKL_PROBE_PUSH, pt);
  return idx;
}

/** @brief hexkl_moe_push_weight_chunk with a #87 trace record. kind /
 *         expert / chunk only label the record and cost nothing when
 *         probing is off; the exported entry (the conv block's) is not
 *         traced -- the trace is per MoE layer call (hexkl_dma_trace.h). */
static uint32_t moe_push_weight_chunk(uint8_t *vtcm_base, uint32_t dst_off,
                                      const hexkl_weight_u8i4 *h,
                                      uint32_t k_tiles, uint32_t n_col,
                                      uint32_t nt0, uint32_t cn, uint32_t kind,
                                      uint32_t expert, uint32_t chunk) {
  const uint32_t idx =
    hexkl_moe_push_weight_chunk(vtcm_base, dst_off, h, k_tiles, n_col, nt0, cn);
  if (hexkl_probe_on) {
    hexkl_dma_trace_push(hexkl_probe_now_ticks(), idx, kind, expert, chunk,
                         cn * WEIGHT_TILE_BYTES_U8I4, k_tiles,
                         n_col * WEIGHT_TILE_BYTES_U8I4);
  }
  return idx;
}

/**
 * @brief Pushes a gate_up weight as PAIRED chunks: chunk c carries gate
 *        n-tile columns [c*half, +cn) and the up columns opposite them
 *        (inter_ntiles further along), two descriptors, one index -- the
 *        second's, since the ring retires in order.
 *
 * The epilogue consumes gate tile j and up tile j together
 * (hvx_dequant_swiglu_acc_tiles_to_f32), so this is the order the columns
 * have to arrive in: chunked by consecutive column the way down still is,
 * the first batch would wait for three quarters of the weight.
 *
 * @return how many chunks were pushed; idx_out gets one index each
 */
static uint32_t moe_push_gate_up_chunks(uint8_t *vtcm_base, uint32_t dst_off,
                                        const hexkl_weight_u8i4 *h,
                                        uint32_t k_tiles, uint32_t gu_ntiles,
                                        uint32_t inter_ntiles, uint32_t half,
                                        uint32_t expert, uint32_t *idx_out) {
  uint32_t n = 0u;
  for (uint32_t g0 = 0; g0 < inter_ntiles; g0 += half) {
    const uint32_t cn = (inter_ntiles - g0 < half) ? (inter_ntiles - g0) : half;
    (void)moe_push_weight_chunk(vtcm_base, dst_off, h, k_tiles, gu_ntiles, g0,
                                cn, HEXKL_DMA_KIND_GATE, expert, n);
    idx_out[n] = moe_push_weight_chunk(vtcm_base, dst_off, h, k_tiles,
                                       gu_ntiles, inter_ntiles + g0, cn,
                                       HEXKL_DMA_KIND_UP, expert, n);
    ++n;
  }
  return n;
}

/**
 * @brief Queues one 64-row activation block, slot-ordered heap -> VTCM.
 *
 * Issued AHEAD of the weight chunks it will be computed against, never
 * behind them. The ring retires in push order and hexkl_dma_ring_wait(idx)
 * therefore waits for everything queued before idx: with this transfer
 * queued after an expert's gate_up and down, its wait covered all 5.25 MB
 * of them, the chunked gate_up wait below then found nothing left to wait
 * for, and the profile filed the whole weight transfer under GATHER --
 * 795 us at decode for a 128 KB copy (doc 46 section 49). Pushed first,
 * the wait covers 128 KB and the weight waits time the weights.
 *
 * @return the ring index to hand hexkl_dma_ring_wait
 */
uint32_t hexkl_moe_push_act_block(uint8_t *vtcm_base, uint32_t act_off,
                                  const uint8_t *act_ah, uint32_t slot,
                                  uint32_t K, uint32_t k_tiles) {
  const uint32_t blk_bytes = k_tiles * HEXKL_HMX_ACTIVATION_ALIGNMENT;
  const uint32_t rs = moe_dma_row_size(blk_bytes);
  const uint32_t idx = hexkl_dma_ring_next_idx();
  hexkl_dma_ring_push2d(vtcm_base + act_off, act_ah + (size_t)slot * K, rs, rs,
                        rs, blk_bytes / rs, /*src_vtcm=*/0, /*dst_vtcm=*/1);
  return idx;
}

/** @brief hexkl_moe_push_act_block with a #87 trace record (expert ordinal
 *         and block within it); see moe_push_weight_chunk. */
static uint32_t moe_push_act_block(uint8_t *vtcm_base, uint32_t act_off,
                                   const uint8_t *act_ah, uint32_t slot,
                                   uint32_t K, uint32_t k_tiles,
                                   uint32_t expert, uint32_t block) {
  const uint32_t blk_bytes = k_tiles * HEXKL_HMX_ACTIVATION_ALIGNMENT;
  const uint32_t rs = moe_dma_row_size(blk_bytes);
  const uint32_t idx =
    hexkl_moe_push_act_block(vtcm_base, act_off, act_ah, slot, K, k_tiles);
  if (hexkl_probe_on) {
    hexkl_dma_trace_push(hexkl_probe_now_ticks(), idx, HEXKL_DMA_KIND_ACT,
                         expert, block, rs, blk_bytes / rs, rs);
  }
  return idx;
}

/**
 * @brief A down batch's epilogue as ONE pool job: each staged tile
 *        dequantized into res_f32 and its 32 columns scatter-added into
 *        the output rows with the routing weight applied -- plus, in a
 *        block's first down batch, the NEXT block's requantization of
 *        gate_off into the other mid buffer, in MOE_RQ_UNIT_ROWS-row
 *        units.
 *
 * Why not a scatter job after the last dequant and a requantization of
 * its own, as it was: the foreground lane holds one job at a time, and a
 * job hides only when an HMX batch is issued between its submit and the
 * wait that retires it. Neither had one -- REQUANT read 0.7-1.0 ms of a
 * 15 ms prefill call and the last dequant and scatter 0.3 more (doc 51
 * section 2.21). Folded in, a block submits exactly as many jobs as it
 * issues batches.
 *
 * Why the fold is exact: a tile's 32 columns of every output row are its
 * own, so no two units write one address, and each output element still
 * receives one multiply-add per expert, in block order, since blocks'
 * jobs are sequential -- the bytes are the row split's. The requant
 * units read gate_off, which every gate_up epilogue has finished writing
 * by the wait before this job's submit, and write the mid buffer the
 * next down will read, not the one this down's HMX is reading. Work is
 * dealt round-robin over the workers so the few heavy requant units do
 * not all land on one.
 *
 * ponytail: the scatter is 64 one-vector calls per tile where the row
 * split made 64 calls of 64 vectors; a tile-shaped scale-add would trim
 * the call overhead if the DN epilogue ever shows above its batch.
 */
typedef struct {
  /* the batch's tiles */
  const uint8_t *tiles_base;
  uint32_t tile_stride, nt0, nb, row_stride, m_count;
  const float *act_scale; /**< mid's row params for this block */
  const int32_t *act_zp;
  const int32_t *colsum_w;
  const float *w_scale;
  const float *bias;
  float *res; /**< [64 x N_out] f32, VTCM */
  /* the scatter */
  float *out; /**< out_c, heap */
  const uint32_t *rows;
  const float *weights;
  uint32_t N_out;
  /* A second run of rows of the SAME expert: its tail block, computed on
     the HVX (moe_tail_* below) and scattered with the expert's last HMX
     block so a token's four contributions still add in expert order.
     n_rows_b == 0 when the expert has no tail. */
  const float *res_b;
  const uint32_t *rows_b;
  const float *weights_b;
  uint32_t n_rows_b;
  /* the next block's requantization; gate NULL when there is none */
  float *gate; /**< [64 x inter] f32, VTCM */
  uint32_t rq_m_blk, inter;
  float *rq_scale; /**< 64 each, the params the next down dequantizes with */
  int32_t *rq_zp;
  uint8_t *mid; /**< the mid buffer it packs into */
} moe_dn_ctx;

#define MOE_RQ_UNIT_ROWS 16u
#define MOE_RQ_UNITS (HEXKL_HMX_INT8_BLOCK_N_ROW / MOE_RQ_UNIT_ROWS)

static inline void moe_worker_probe_add(uint64_t t0);

static void moe_dn_worker(uint32_t n_threads, uint32_t i, void *vctx) {
  moe_dn_ctx *c = (moe_dn_ctx *)vctx;
  uint64_t t0 = 0;
  HEXKL_PROBE_T0(t0);
  if (c->gate) {
    /* Rows [m_blk, m4) are padding the pack takes whole, zeroed so the
       bytes are deterministic; rows past m4 are not packed and never
       read. Row params are per row, so a unit's are its own. */
    const uint32_t m4 = ROUND_UP_U32(c->rq_m_blk, 4u);
    for (uint32_t u = i; u < MOE_RQ_UNITS; u += n_threads) {
      const uint32_t r0 = u * MOE_RQ_UNIT_ROWS;
      if (r0 >= m4) {
        break;
      }
      const uint32_t r1 =
        (m4 - r0 < MOE_RQ_UNIT_ROWS) ? m4 : r0 + MOE_RQ_UNIT_ROWS;
      const uint32_t valid =
        (c->rq_m_blk - r0 < r1 - r0) ? c->rq_m_blk - r0 : r1 - r0;
      float *x = c->gate + (size_t)r0 * c->inter;
      if (valid < r1 - r0) {
        memset(x + (size_t)valid * c->inter, 0,
               sizeof(float) * (size_t)(r1 - r0 - valid) * c->inter);
      }
      hvx_quant_rows_u8_params(x, valid, r1 - r0, c->inter, c->rq_scale + r0,
                               c->rq_zp + r0, NULL);
      hvx_quant_pack_u8_ah_rows(c->gate, NULL, r0, r1, c->inter, c->rq_scale,
                                c->rq_zp, c->mid);
    }
  }
  for (uint32_t j = i; j < c->nb; j += n_threads) {
    const uint32_t c0 = (c->nt0 + j) * HEXKL_ACC_TILE_COLS;
    const int32_t *tile =
      (const int32_t *)(c->tiles_base + (size_t)j * c->tile_stride);
    float *res = c->res + c0;
    hvx_dequant_acc_tile_to_f32(tile, c->row_stride, c->m_count, c->act_scale,
                                c->act_zp, c->colsum_w + c0, c->w_scale + c0,
                                c->bias + c0, res, c->N_out, 0);
    for (uint32_t r = 0; r < c->m_count; ++r) {
      hvx_scale_add_rows_f32(c->out + (size_t)c->rows[r] * c->N_out + c0,
                             res + (size_t)r * c->N_out, c->weights[r],
                             HEXKL_ACC_TILE_COLS);
    }
    for (uint32_t q = 0; q < c->n_rows_b; ++q) {
      hvx_scale_add_rows_f32(c->out + (size_t)c->rows_b[q] * c->N_out + c0,
                             c->res_b + (size_t)q * c->N_out + c0,
                             c->weights_b[q], HEXKL_ACC_TILE_COLS);
    }
  }
  moe_worker_probe_add(t0);
}

/** @brief The gate_up epilogue job, timed: hvx_dq_swiglu_worker's slice
 *         with its worker time filed like moe_dn_worker's. */
static void moe_gu_worker(uint32_t n_threads, uint32_t i, void *vctx) {
  uint64_t t0 = 0;
  HEXKL_PROBE_T0(t0);
  hvx_dq_swiglu_worker(n_threads, i, vctx);
  moe_worker_probe_add(t0);
}

/* ---- O1: an expert's tail block on the HVX ------------------------------
 *
 * The HMX computes 64 rows whatever the count, 252 us a block; 13.6 of a
 * prefill call's 45.6 blocks are an expert's second block of 10-20 rows
 * (doc 47 section 14). A block of at most MOE_TAIL_MAX_ROWS rows after a
 * full one is taken off the HMX and computed on the pool's background lane
 * as three gated jobs -- gate/up column pairs with the fused SwiGLU
 * epilogue, the requantization, down columns with their dequant -- reading
 * the weights straight from the arena (DDR) and the activation from the
 * packed slots. Every job of every tail is queued at the start of the call,
 * in expert order, so the lane grinds through them under the whole HMX
 * loop, and each tail's rows are scattered by the job that scatters its
 * expert's last HMX block. The int32 sums are the HMX's own (see
 * hvx_gemm_u8i4_wh.h) and the epilogues are the same functions on the same
 * numbers, so the output is byte for byte what the all-HMX path produced.
 *
 * ponytail: one tail at a time -- the gating serializes them, so the
 * staging below is shared. If tails ever outrun the shadow, two staging
 * sets and pairwise gating is the upgrade. And the threshold is a guess
 * until measured: HVX_GEMM_U8I4_MAX_ROWS (16) is where a row-group unit is
 * ~20 us; the average tail here is 10-20 rows.
 */
/* OFF by default (0 rows): measured on device (doc 47 section 21.1) the path
   took 3.5 tails a call off the HMX (-0.59 ms) and cost 1.07 ms in return --
   the tails were not done when their expert's scatter needed them (SCATTER
   +0.45) and the units held workers the requant and epilogue runs were
   waiting for (REQUANT +0.48, DEQUANT +0.15). Net -0.5 ms a call. The host
   check builds with -DMOE_TAIL_MAX_ROWS=16u so the path stays exercised;
   HVX_GEMM_U8I4_MAX_ROWS is the ceiling. */
#ifndef MOE_TAIL_MAX_ROWS
#define MOE_TAIL_MAX_ROWS 0u
#endif
#define MOE_TAIL_TILE_I32 (MOE_TAIL_MAX_ROWS * HEXKL_HMX_INT8_BLOCK_N_COL)
#define MOE_TAIL_TILE_BYTES (MOE_TAIL_TILE_I32 * 4u)

typedef struct {
  int32_t *acc_gu; /**< inter_ntiles pairs x 2 tiles, MOE_TAIL_TILE_I32 each */
  int32_t *acc_dn; /**< dn_ntiles tiles */
  float *gate_f32; /**< MOE_TAIL_MAX_ROWS x inter, silu(gate)*up */
  uint8_t *mid_ah; /**< one 64-row AH block, inter_ktiles tiles */
  float *rq_scale; /**< 64: the requantization's row params */
  int32_t *rq_zp;
} moe_tail_shared;

typedef struct {
  const moe_tail_shared *sh;
  const uint8_t *act_ah; /**< the block's AH tiles, slot order */
  const hexkl_weight_u8i4 *g;
  const hexkl_weight_u8i4 *d;
  const float *act_scale; /**< slot tables at the block's first slot */
  const int32_t *act_zp;
  float *res; /**< this tail's m x N_out, read by its expert's scatter */
  uint32_t m, k_tiles, inter, inter_ktiles, inter_ntiles, gu_ntiles, dn_ntiles,
    N_out;
} moe_tail_ctx;

/** @brief Worker time spent inside this kernel's pool jobs -- the gate_up
 *         epilogue (dequant + SwiGLU), the down epilogue (dequant, scatter,
 *         the requant units it carries) and the tail units -- summed
 *         across workers and filed under the SWIGLU column, which this
 *         kernel has no synchronous pass for. The same reading as the conv
 *         block's cb_stage_probe_add: HIDDEN work, not wall time, so the
 *         host leaves it out of the mm residual. What it answers is how
 *         much of the HMX shadow (3 workers x the issue time) the
 *         epilogues consume -- the number that decides whether making the
 *         epilogue arithmetic cheaper can move anything (doc 53 section
 *         8.4). Atomic because the slices run concurrently; HEXKL_PROBE_ADD
 *         is not. */
static inline void moe_worker_probe_add(uint64_t t0) {
  if (hexkl_probe_on) {
    atomic_fetch_add_explicit(
      (_Atomic uint64_t *)&hexkl_probe_us[HEXKL_PROBE_SWIGLU],
      hexkl_probe_now() - t0, memory_order_relaxed);
  }
}

/** @brief Unit j: gate column j and up column inter_ntiles+j, then the
 *         fused dequant + SwiGLU of that pair into gate_f32. */
static void moe_tail_pair_unit(uint32_t n_units, uint32_t j, void *v) {
  (void)n_units;
  uint64_t t0 = 0;
  HEXKL_PROBE_T0(t0);
  const moe_tail_ctx *t = (const moe_tail_ctx *)v;
  int32_t *tiles = t->sh->acc_gu + (size_t)j * 2u * MOE_TAIL_TILE_I32;
  /* rows1 = 0: the tail is a prefill shape, so a lone last row stays on
     gemm_rows4; HVX_GEMV_M1_ROWS1 is the M=1 path's knob (moe_m1_*). */
  hvx_gemm_u8i4_wh_col(t->act_ah, t->m, t->k_tiles, t->g->wh_bytes,
                       t->gu_ntiles, j, 0u, tiles);
  hvx_gemm_u8i4_wh_col(t->act_ah, t->m, t->k_tiles, t->g->wh_bytes,
                       t->gu_ntiles, t->inter_ntiles + j, 0u,
                       tiles + MOE_TAIL_TILE_I32);
  hvx_dequant_swiglu_acc_tiles_to_f32(
    (const uint8_t *)tiles, MOE_TAIL_TILE_BYTES, 1u, j,
    HEXKL_HMX_INT8_BLOCK_N_COL, t->m, t->act_scale, t->act_zp, t->g->colsum_w,
    t->g->w_scale, t->g->bias, t->inter, t->sh->gate_f32, t->inter, NULL);
  moe_worker_probe_add(t0);
}

/** @brief The one requantization unit: gate_f32 -> mid_ah, the same two
 *         calls the HMX path makes for its block. */
static void moe_tail_requant_unit(uint32_t n_units, uint32_t u, void *v) {
  (void)n_units;
  (void)u;
  uint64_t t0 = 0;
  HEXKL_PROBE_T0(t0);
  const moe_tail_ctx *t = (const moe_tail_ctx *)v;
  const moe_tail_shared *sh = t->sh;
  const uint32_t m4 = ROUND_UP_U32(t->m, 4u);
  /* The pack takes whole row groups; rows [m, m4) are padding it packs and
     nothing reads, zeroed so the bytes are deterministic. */
  if (m4 > t->m) {
    memset(sh->gate_f32 + (size_t)t->m * t->inter, 0,
           sizeof(float) * (size_t)(m4 - t->m) * t->inter);
  }
  hvx_quant_rows_u8_params(sh->gate_f32, t->m, HEXKL_HMX_INT8_BLOCK_N_ROW,
                           t->inter, sh->rq_scale, sh->rq_zp, NULL);
  hvx_quant_pack_u8_ah_rows(sh->gate_f32, NULL, 0u, m4, t->inter, sh->rq_scale,
                            sh->rq_zp, sh->mid_ah);
  moe_worker_probe_add(t0);
}

/** @brief Unit nt: down column nt, dequantized into res. */
static void moe_tail_down_unit(uint32_t n_units, uint32_t nt, void *v) {
  (void)n_units;
  uint64_t t0 = 0;
  HEXKL_PROBE_T0(t0);
  const moe_tail_ctx *t = (const moe_tail_ctx *)v;
  int32_t *tile = t->sh->acc_dn + (size_t)nt * MOE_TAIL_TILE_I32;
  const uint32_t c0 = nt * HEXKL_HMX_INT8_BLOCK_N_COL;
  hvx_gemm_u8i4_wh_col(t->sh->mid_ah, t->m, t->inter_ktiles, t->d->wh_bytes,
                       t->dn_ntiles, nt, 0u, tile);
  hvx_dequant_acc_tile_to_f32(tile, HEXKL_HMX_INT8_BLOCK_N_COL, t->m,
                              t->sh->rq_scale, t->sh->rq_zp,
                              t->d->colsum_w + c0, t->d->w_scale + c0,
                              t->d->bias + c0, t->res + c0, t->N_out, 0);
  moe_worker_probe_add(t0);
}

/** @brief Rows of expert e's tail, 0 when it has none: a block of at most
 *         MOE_TAIL_MAX_ROWS after at least one full one. */
static uint32_t moe_tail_rows(uint32_t n_e) {
  const uint32_t r = n_e % HEXKL_HMX_INT8_BLOCK_N_ROW;
  return (n_e > HEXKL_HMX_INT8_BLOCK_N_ROW && r != 0u && r <= MOE_TAIL_MAX_ROWS)
           ? r
           : 0u;
}

/* ---- The M=1 path: every expert on the HVX GEMV, no 64-row block ---------
 *
 * Decode routes one token (M=1; a few at a small batch) to four experts,
 * and the block loop above computes a 64-row HMX block for each of them
 * plus the 5.25 MB weight DMA into VTCM that one block of one live row
 * cannot hide (LEDGER wall 1). This path takes the whole call off the HMX:
 * it is the tail path's three stages (moe_tail_* above) run for every
 * active expert, the weights read straight from the arena behind the
 * GEMV's own l2fetch, the activation packed inline -- no weight DMA, so
 * BLOCKS and DMA_KB read 0 and PATH reads 1.
 *
 * Foreground lane only. The tail path lost on device because its
 * background units held workers the HMX epilogues then waited for (doc 47
 * section 21.1); here there is no HMX loop to hide under, so each stage is
 * one hvx_worker_pool_run across the pool's lanes (n_hvx - 1 workers plus
 * the caller, 6 on v79) and nothing is left in flight: A = gate/up column
 * pairs with the fused SwiGLU epilogue, B =
 * one requantization per expert, C = down columns with their dequant, then
 * the scatter inline on the caller in expert order and row order -- the add
 * sequence moe_scatter_worker produces (blocks sequential across experts,
 * rows distinct inside one), so the f32 bytes are the HMX path's. The int32
 * sums are the HMX's own (hvx_gemm_u8i4_wh.h). A lane takes a contiguous,
 * expert-major slice of the units so its l2fetch stream stays in one weight:
 * a lane's consecutive units are adjacent 512-byte tiles, so one 2D l2fetch
 * covers a block of them, and the lane issues block b+1's box before it
 * computes block b (HVX_GEMV_PF_LEAD_KB, hexkl_mm_u8i4_moe.h).
 *
 * Taken only with HEXKL_MOE_FLAG_M1_GEMV set and M <= MOE_M1_MAX_ROWS with
 * at most MOE_M1_MAX_EXPERTS active experts (the top-4 bound at M = 4), so a
 * prefill call never sees it and a wider routing falls back to the HMX
 * loop rather than growing scratch. The scratch is reserved for the expert
 * bound, not the call's routing, the n_slots_cap rule: ~4.2 MB at the
 * LFM2 shape, only when a decode call comes before any prefill call (a
 * prefill call's 12.8 MB already covers it).
 *
 * THE VTCM FEED (#117, hexkl_moe_flags_feed). The arena read above tops
 * out at 21-27 GB/s on this SoC (LEDGER rule 27) while the DMA engine
 * moves the same bytes at 31-32 (#100's f2 cell, rule 32), so with the
 * feed on each expert's gate_up and down are staged into VTCM by the
 * engine one expert ahead and the GEMV reads the VTCM copy. The M=1 path
 * uses no VTCM otherwise, so the whole arena is free: two gate_up-sized
 * slabs G0 / G1 at offsets 0 and gu_bytes double-buffer stage A by expert
 * (gate_up of expert i in slab i & 1), and once a slab has no gate_up
 * left to hold, two down matrices land in it (down of expert j in slab
 * (j >> 1) & 1, half j & 1). Every push is a whole matrix in one
 * descriptor -- f2's shape, 8 per call at top-4 -- and every reuse of a
 * slab follows the join of the pool run that read it, which is what lets
 * the host scoreboard (test/htp/host/moe_layer_host_check.c) check the
 * schedule exactly. The caller alone touches the ring; the waits sit
 * inside the MM bracket so mm reads the feed + compute wall, and they are
 * NOT added to DRAIN / DRAIN_DN, whose subtraction would double-count
 * them. The l2fetch lead is off under the feed: an l2fetch of a VTCM
 * address is meaningless and one of the arena bytes would pull the same
 * DDR bytes twice. A shape whose two slabs do not fit the arena runs the
 * arena read as before and the call's feed count says so.
 *
 * ponytail: one pool run per expert and stage (6 extra fork/joins a call)
 * is the price of a schedule a host check can prove. If the vtcm
 * microbench cell shows mm - bytes / engine rate well above the ~20 us
 * compute tail, the upgrade is one run per stage with lanes polling their
 * slab's descriptor (hexkl_dma_ring_is_done) -- plan 117 section 3.2.
 */
#define MOE_M1_MAX_ROWS 4u
#define MOE_M1_MAX_EXPERTS 16u
#define MOE_M1_TILE_I32 (MOE_M1_MAX_ROWS * HEXKL_HMX_INT8_BLOCK_N_COL)
#define MOE_M1_TILE_BYTES (MOE_M1_TILE_I32 * 4u)

typedef struct {
  const uint8_t *act_ah; /**< the expert's slot block, AH tiles */
  const hexkl_weight_u8i4 *g;
  const hexkl_weight_u8i4 *d;
  const uint8_t *gu_wh;   /**< the WH bytes the GEMV reads: g->wh_bytes (the
                               arena) or the expert's VTCM slab (#117) */
  const uint8_t *dn_wh;   /**< likewise for the down matrix */
  const float *act_scale; /**< slot tables at the block's first slot */
  const int32_t *act_zp;
  int32_t *acc_gu; /**< inter_ntiles pairs x 2 tiles, MOE_M1_TILE_I32 each */
  int32_t *acc_dn; /**< dn_ntiles tiles */
  float *gate_f32; /**< MOE_M1_MAX_ROWS x inter, silu(gate)*up */
  uint8_t *mid_ah; /**< one AH block, inter_ktiles tiles */
  float *rq_scale; /**< 64: the requantization's row params */
  int32_t *rq_zp;
  float *res; /**< m x N_out, read by the scatter */
  uint32_t m;
} moe_m1_expert;

typedef struct {
  moe_m1_expert ex[MOE_M1_MAX_EXPERTS];
  uint32_t n_active, k_tiles, inter, inter_ktiles, inter_ntiles, gu_ntiles,
    dn_ntiles, N_out;
  uint32_t lead_kb; /**< this call's l2fetch lead, KB per lane; 0 = each
                         column issues its own fetch (hexkl_moe_flags_*) */
  uint32_t rows1;   /**< this call's row loop: 1 = gemm_row1 for a lone row */
  uint32_t feed;    /**< 1: the workers read VTCM slabs, take the
                         prefetch-free entry and issue no box (#117) */
  /** The unit range [u_lo, u_hi) one pool run of stage A or C covers:
      every unit of the call when the feed is off (today's three runs), one
      expert's units per run when it is on, so each slab reuse follows a
      join. Unit numbering stays global (expert = u / per). */
  uint32_t u_lo, u_hi;
} moe_m1_ctx;

/** @brief Lane @a i's contiguous share [lo, hi) of the units [u_lo, u_hi). */
static inline void moe_m1_slice(uint32_t u_lo, uint32_t u_hi, uint32_t n_lanes,
                                uint32_t i, uint32_t *lo, uint32_t *hi) {
  const uint32_t n_units = u_hi - u_lo;
  *lo = u_lo + (uint32_t)((uint64_t)n_units * i / n_lanes);
  *hi = u_lo + (uint32_t)((uint64_t)n_units * (i + 1u) / n_lanes);
}

/** @brief End of the lead block that starts at unit @a u: at most @a d
 *         units, inside the lane's slice (@a hi), one expert (@a per units
 *         each) -- so the block is adjacent columns of one weight. */
static inline uint32_t moe_m1_block_end(uint32_t u, uint32_t d, uint32_t per,
                                        uint32_t hi) {
  const uint32_t e_end = (u / per + 1u) * per;
  uint32_t end = u + d;
  if (end > e_end) {
    end = e_end;
  }
  return end < hi ? end : hi;
}

/** @brief Units per lead block for a unit of @a unit_bytes weight, for a
 *         lead of @a lead_kb KB. Clamped to 127 units: hvx_gemm_u8i4_wh's
 *         l2fetch width field is 16 bits over tiles of 512 B, so a box of
 *         128 tiles or more would not fit. The block-end clamp below cuts
 *         it further to the lane's slice and to one expert. */
static inline uint32_t moe_m1_lead_units(uint32_t lead_kb,
                                         uint32_t unit_bytes) {
  const uint32_t d = (uint32_t)((uint64_t)lead_kb * 1024u / unit_bytes);
  if (d > 127u) {
    return 127u;
  }
  return d ? d : 1u;
}

/** @brief Stage A's box for units [b0, b1): gate (@a half 0) or up (1). */
static inline void moe_m1_pf_gu(const moe_m1_ctx *c, uint32_t b0, uint32_t b1,
                                uint32_t half) {
  hvx_gemm_u8i4_wh_prefetch(
    c->ex[b0 / c->inter_ntiles].g->wh_bytes, c->gu_ntiles,
    half * c->inter_ntiles + b0 % c->inter_ntiles, b1 - b0, c->k_tiles);
}

/** @brief Stage A. Unit u = (expert u / inter_ntiles, pair u % inter_ntiles):
 *         moe_tail_pair_unit's body on that expert's buffers.
 *
 * With a lead, block b+1's gate box goes out before block b's first gate
 * column and its up box after block b's last gate column. The hardware
 * queues three l2fetch per thread and stalls on a fourth, so this keeps
 * at most three outstanding: by the time the fourth box goes out, every
 * column of block b's gate box has been read. */
static void moe_m1_pair_worker(uint32_t n_lanes, uint32_t i, void *v) {
  const moe_m1_ctx *c = (const moe_m1_ctx *)v;
  const int lead = c->lead_kb != 0u;
  /* Under the feed the bytes are in VTCM: no box, no self-prefetch. */
  const int nopf = lead || c->feed;
  const uint32_t d = moe_m1_lead_units(c->lead_kb, 2u * c->k_tiles * 512u);
  uint32_t lo, hi, b1, n1;
  uint64_t t0 = 0;
  HEXKL_PROBE_T0(t0);
  moe_m1_slice(c->u_lo, c->u_hi, n_lanes, i, &lo, &hi);
  b1 = lo < hi ? moe_m1_block_end(lo, d, c->inter_ntiles, hi) : lo;
  if (lead && lo < b1) {
    moe_m1_pf_gu(c, lo, b1, 0u);
    moe_m1_pf_gu(c, lo, b1, 1u);
  }
  for (uint32_t b0 = lo; b0 < hi; b0 = b1, b1 = n1) {
    n1 = b1 < hi ? moe_m1_block_end(b1, d, c->inter_ntiles, hi) : b1;
    for (uint32_t u = b0; u < b1; ++u) {
      const moe_m1_expert *e = &c->ex[u / c->inter_ntiles];
      const uint32_t j = u % c->inter_ntiles;
      int32_t *tiles = e->acc_gu + (size_t)j * 2u * MOE_M1_TILE_I32;
      if (lead && u == b0 && b1 < n1) {
        moe_m1_pf_gu(c, b1, n1, 0u);
      }
      (nopf ? hvx_gemm_u8i4_wh_col_nopf
            : hvx_gemm_u8i4_wh_col)(e->act_ah, e->m, c->k_tiles, e->gu_wh,
                                    c->gu_ntiles, j, c->rows1, tiles);
      if (lead && u + 1u == b1 && b1 < n1) {
        moe_m1_pf_gu(c, b1, n1, 1u);
      }
      (nopf ? hvx_gemm_u8i4_wh_col_nopf : hvx_gemm_u8i4_wh_col)(
        e->act_ah, e->m, c->k_tiles, e->gu_wh, c->gu_ntiles,
        c->inter_ntiles + j, c->rows1, tiles + MOE_M1_TILE_I32);
      hvx_dequant_swiglu_acc_tiles_to_f32(
        (const uint8_t *)tiles, MOE_M1_TILE_BYTES, 1u, j,
        HEXKL_HMX_INT8_BLOCK_N_COL, e->m, e->act_scale, e->act_zp,
        e->g->colsum_w, e->g->w_scale, e->g->bias, c->inter, e->gate_f32,
        c->inter, NULL);
    }
  }
  moe_worker_probe_add(t0);
}

/** @brief Stage B. One expert per unit: moe_tail_requant_unit's body. */
static void moe_m1_requant_worker(uint32_t n_lanes, uint32_t i, void *v) {
  const moe_m1_ctx *c = (const moe_m1_ctx *)v;
  uint32_t lo, hi;
  moe_m1_slice(0u, c->n_active, n_lanes, i, &lo, &hi);
  for (uint32_t u = lo; u < hi; ++u) {
    const moe_m1_expert *e = &c->ex[u];
    const uint32_t m4 = ROUND_UP_U32(e->m, 4u);
    if (m4 > e->m) {
      memset(e->gate_f32 + (size_t)e->m * c->inter, 0,
             sizeof(float) * (size_t)(m4 - e->m) * c->inter);
    }
    hvx_quant_rows_u8_params(e->gate_f32, e->m, HEXKL_HMX_INT8_BLOCK_N_ROW,
                             c->inter, e->rq_scale, e->rq_zp, NULL);
    hvx_quant_pack_u8_ah_rows(e->gate_f32, NULL, 0u, m4, c->inter, e->rq_scale,
                              e->rq_zp, e->mid_ah);
  }
}

/** @brief Stage C's box for units [b0, b1): adjacent down columns. */
static inline void moe_m1_pf_dn(const moe_m1_ctx *c, uint32_t b0, uint32_t b1) {
  hvx_gemm_u8i4_wh_prefetch(c->ex[b0 / c->dn_ntiles].d->wh_bytes, c->dn_ntiles,
                            b0 % c->dn_ntiles, b1 - b0, c->inter_ktiles);
}

/** @brief Stage C. Unit u = (expert u / dn_ntiles, column u % dn_ntiles):
 *         moe_tail_down_unit's body. With a lead, block b+1's box goes out
 *         before block b is computed: two outstanding at most. */
static void moe_m1_down_worker(uint32_t n_lanes, uint32_t i, void *v) {
  const moe_m1_ctx *c = (const moe_m1_ctx *)v;
  const int lead = c->lead_kb != 0u;
  const int nopf = lead || c->feed;
  const uint32_t d = moe_m1_lead_units(c->lead_kb, c->inter_ktiles * 512u);
  uint32_t lo, hi, b1, n1;
  uint64_t t0 = 0;
  HEXKL_PROBE_T0(t0);
  moe_m1_slice(c->u_lo, c->u_hi, n_lanes, i, &lo, &hi);
  b1 = lo < hi ? moe_m1_block_end(lo, d, c->dn_ntiles, hi) : lo;
  if (lead && lo < b1) {
    moe_m1_pf_dn(c, lo, b1);
  }
  for (uint32_t b0 = lo; b0 < hi; b0 = b1, b1 = n1) {
    n1 = b1 < hi ? moe_m1_block_end(b1, d, c->dn_ntiles, hi) : b1;
    if (lead && b1 < n1) {
      moe_m1_pf_dn(c, b1, n1);
    }
    for (uint32_t u = b0; u < b1; ++u) {
      const moe_m1_expert *e = &c->ex[u / c->dn_ntiles];
      const uint32_t nt = u % c->dn_ntiles;
      int32_t *tile = e->acc_dn + (size_t)nt * MOE_M1_TILE_I32;
      const uint32_t c0 = nt * HEXKL_HMX_INT8_BLOCK_N_COL;
      (nopf ? hvx_gemm_u8i4_wh_col_nopf
            : hvx_gemm_u8i4_wh_col)(e->mid_ah, e->m, c->inter_ktiles, e->dn_wh,
                                    c->dn_ntiles, nt, c->rows1, tile);
      hvx_dequant_acc_tile_to_f32(tile, HEXKL_HMX_INT8_BLOCK_N_COL, e->m,
                                  e->rq_scale, e->rq_zp, e->d->colsum_w + c0,
                                  e->d->w_scale + c0, e->d->bias + c0,
                                  e->res + c0, c->N_out, 0);
    }
  }
  moe_worker_probe_add(t0);
}

/** @brief One 64-row HMX block of the sequence the layer call walks. */
typedef struct {
  uint32_t i;     /**< active-expert index (into order / slot_of / base_of) */
  uint32_t e;     /**< the expert */
  uint32_t mb;    /**< first row within the expert's HMX rows */
  uint32_t m_blk; /**< rows carrying data, <= 64 */
  uint32_t slot;  /**< first slot: slot_of[i] + mb */
  int first;      /**< the expert's first block: its weights are new */
  int last;       /**< the expert's last HMX block: its tail goes with it */
} moe_blk;

static void moe_blk_set(moe_blk *b, uint32_t i, uint32_t mb,
                        const uint32_t *order, const uint32_t *row_count,
                        const uint32_t *slot_of) {
  const uint32_t n_e = row_count[order[i]];
  const uint32_t hmx_rows = n_e - moe_tail_rows(n_e);
  b->i = i;
  b->e = order[i];
  b->mb = mb;
  b->m_blk = (hmx_rows - mb < HEXKL_HMX_INT8_BLOCK_N_ROW)
               ? hmx_rows - mb
               : HEXKL_HMX_INT8_BLOCK_N_ROW;
  b->slot = slot_of[i] + mb;
  b->first = (mb == 0u);
  b->last = (mb + HEXKL_HMX_INT8_BLOCK_N_ROW >= hmx_rows);
}

/** @brief Steps @a b to the block after it; 0 when it was the last. */
static int moe_blk_next(moe_blk *b, uint32_t n_active, const uint32_t *order,
                        const uint32_t *row_count, const uint32_t *slot_of) {
  if (!b->last) {
    moe_blk_set(b, b->i, b->mb + HEXKL_HMX_INT8_BLOCK_N_ROW, order, row_count,
                slot_of);
    return 1;
  }
  if (b->i + 1u < n_active) {
    moe_blk_set(b, b->i + 1u, 0u, order, row_count, slot_of);
    return 1;
  }
  return 0;
}

/* ---- The M=1 feed's slabs and ring calls (#117; the schedule is in the
   M=1 block of hexkl_mm_u8i4_moe_layer_run) ------------------------------ */

/** @brief VTCM offset of expert @a i's gate_up: slab i & 1. */
static inline uint32_t moe_m1_gu_off(uint32_t i, uint32_t gu_bytes) {
  return (i & 1u) * gu_bytes;
}

/** @brief VTCM offset of expert @a j's down: slab (j >> 1) & 1, half
 *         j & 1 -- so downs j and j + 4 share a slot, and j + 4 is pushed
 *         only after C(j)'s join. */
static inline uint32_t moe_m1_dn_off(uint32_t j, uint32_t gu_bytes,
                                     uint32_t dn_bytes) {
  return ((j >> 1u) & 1u) * gu_bytes + (j & 1u) * dn_bytes;
}

/** @brief One whole-matrix descriptor (f2's shape: moe_push_weight_chunk
 *         with nt0 = 0, cn = n_col), counted in DMA_KB and traced but NOT
 *         timed into PUSH: it is issued inside the MM bracket, and the
 *         profile row subtracts both, so charging PUSH too would take the
 *         push time off the rest twice.
 *  @return the ring index to hand moe_m1_wait */
static inline uint32_t moe_m1_push(uint8_t *vtcm_base, uint32_t dst_off,
                                   const hexkl_weight_u8i4 *h, uint32_t k_tiles,
                                   uint32_t n_col, uint32_t kind,
                                   uint32_t expert) {
  const uint32_t row = n_col * WEIGHT_TILE_BYTES_U8I4;
  const uint32_t idx = hexkl_dma_ring_next_idx();
  HEXKL_PROBE_COUNT(HEXKL_PROBE_DMA_KB, (row * k_tiles) >> 10);
  hexkl_dma_ring_push2d(vtcm_base + dst_off, h->wh_bytes, row, row, row,
                        k_tiles, /*src_vtcm=*/0, /*dst_vtcm=*/1);
  if (hexkl_probe_on) {
    hexkl_dma_trace_push(hexkl_probe_now_ticks(), idx, kind, expert, 0u, row,
                         k_tiles, row);
  }
  return idx;
}

/** @brief A traced ring wait. Not timed into DRAIN / DRAIN_DN: it sits
 *         inside the MM bracket (the comment atop the M=1 section). */
static inline void moe_m1_wait(uint32_t idx, uint32_t site) {
  if (hexkl_probe_on) {
    hexkl_dma_trace_wait_begin(hexkl_probe_now_ticks(), idx, site);
  }
  hexkl_dma_ring_wait(idx);
  if (hexkl_probe_on) {
    hexkl_dma_trace_wait_end(hexkl_probe_now_ticks());
  }
}

/**
 * @brief Background-lane unit: pack one 64-row slot block of the activation.
 *
 * The pack of a prefill call's ~2900 slots ran on the pool synchronously
 * before the expert loop, with the HMX idle for all of it (1.0 of the 1.44
 * ms QUANT column, doc 47 section 14). Every block is a unit here, in slot
 * order, and the expert loop waits for a block only just before it queues
 * that block's DMA -- so expert 0's block is packed at once and the rest
 * land under the first experts' HMX issue, in the workers' idle time
 * between epilogues.
 */
void hexkl_moe_pack_bg_worker(uint32_t n_units, uint32_t u, void *vctx) {
  (void)n_units;
  const hexkl_moe_pack_ctx *c = (const hexkl_moe_pack_ctx *)vctx;
  hvx_quant_pack_u8_ah_rows(c->act_c, c->slot_row, u * HEXKL_MOE_PACK_UNIT_ROWS,
                            (u + 1u) * HEXKL_MOE_PACK_UNIT_ROWS, c->K,
                            c->slot_scale, c->slot_zp, c->act_ah);
}

/**
 * @brief Copies through the DMA engine instead of the core.
 *
 * The two staging copies move 3.6 MB each between the host's uncached
 * rpcmem buffers and cached heap, and doing that with memcpy costs about
 * 3.6 ms of the scatter column -- a scalar core reading uncached DDR. The
 * DMA engine is what this hardware has for bulk moves, and it is free at
 * both points: the activation copy runs before the expert loop, when only
 * expert 0's gate_up is in flight, and the output copy runs after the loop,
 * when nothing is.
 *
 * Split into ring-sized pieces because a 2D descriptor's row count is
 * bounded; row_size stays the largest power of two that divides the
 * transfer, the same rule the weight pushes use.
 *
 * @param site the #87 trace's HEXKL_DMA_SITE_* for the drain, or -1 for a
 *             copy outside the MoE layer call (no record)
 */
static void moe_dma_copy(void *dst, const void *src, size_t bytes, int src_vtcm,
                         int dst_vtcm, int site) {
  const uint32_t CHUNK = 1u << 20;
  const int traced = hexkl_probe_on && site >= 0;
  size_t off = 0;
  uint32_t idx = 0, piece = 0;
  while (off < bytes) {
    const uint32_t n = (bytes - off) > CHUNK ? CHUNK : (uint32_t)(bytes - off);
    const uint32_t rs = moe_dma_row_size(n);
    idx = hexkl_dma_ring_next_idx();
    hexkl_dma_ring_push2d((uint8_t *)dst + off, (const uint8_t *)src + off, rs,
                          rs, rs, n / rs, src_vtcm, dst_vtcm);
    if (traced) {
      hexkl_dma_trace_push(hexkl_probe_now_ticks(), idx, HEXKL_DMA_KIND_COPY, 0,
                           piece++, rs, n / rs, rs);
    }
    off += n;
  }
  if (traced) {
    hexkl_dma_trace_wait_begin(hexkl_probe_now_ticks(), idx, (uint32_t)site);
  }
  hexkl_dma_ring_drain();
  if (traced) {
    hexkl_dma_trace_wait_end(hexkl_probe_now_ticks());
  }
}

void hexkl_moe_dma_copy(void *dst, const void *src, size_t bytes, int src_vtcm,
                        int dst_vtcm) {
  moe_dma_copy(dst, src, bytes, src_vtcm, dst_vtcm, -1);
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
  /* No up region: the fused epilogue writes silu(gate)*up straight into
     gate_off and up is never materialised. */
  L.mid_off =
    ROUND_UP_U32(L.gate_off + inter_rows, HEXKL_HMX_ACTIVATION_ALIGNMENT);
  /* Two mid buffers: block n's down reads one while block n+1 is
     requantized into the other (see the block loop). */
  L.result_off =
    ROUND_UP_U32(L.mid_off + 2u * mid_bytes, HEXKL_HMX_ACTIVATION_ALIGNMENT);

  /* Two staging buffers of acc_tiles accumulator tiles: the HMX fills one
     while the pool dequantizes the other, which is what lets the epilogue
     hide behind the next batch's issue instead of following it. res_f32
     comes after them and no longer aliases gate -- the scatter reads it
     while the next block's epilogue is already writing gate -- so how many
     tiles fit is decided with res_f32 already taken out. */
  {
    const uint32_t fixed =
      L.result_off + ROUND_UP_U32(res_bytes, HEXKL_HMX_ACTIVATION_ALIGNMENT);
    uint32_t want = (2u * inter) / HEXKL_HMX_INT8_BLOCK_N_COL;
    uint32_t fits = 0u;
    if (arena_bytes > fixed) {
      fits = (arena_bytes - fixed) / (2u * ACC_TILE_BYTES);
    }
    /* Capped, not maximised. This count is also the weight chunk size (see
       moe_push_weight_chunk): a bigger batch parallelises the dequant no
       better once it is well past the worker count, and a smaller chunk is
       what makes the matmul start sooner. 32 leaves gate_up in 4 paired
       chunks and down in 2. */
    if (want > 32u) {
      want = 32u;
    }
    L.acc_tiles = (fits < want) ? fits : want;
    /* Pairs: the gate_up epilogue dequantizes gate tile j and up tile j
       together (hvx_dq_swiglu_worker), so a staged batch is an even count
       with room for at least one pair. */
    L.acc_tiles &= ~1u;
    if (L.acc_tiles < 2u) {
      return AEE_ENOMEMORY;
    }
  }
  L.res_f32_off = ROUND_UP_U32(L.result_off + 2u * L.acc_tiles * ACC_TILE_BYTES,
                               HEXKL_HMX_ACTIVATION_ALIGNMENT);
  L.total = L.res_f32_off + res_bytes;
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
  const float *act_f32, float *out_f32, hvx_worker_pool *pool,
  hexkl_moe_scratch *scratch, uint32_t flags) {

  if (!tbl || !vtcm_base || !h_gate_up || !h_down || !row_index || !row_count ||
      !row_weight || !act_f32 || !out_f32 || !scratch || M == 0u ||
      n_experts == 0u) {
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
  uint32_t n_rows = 0u, n_with_rows = 0u, widest = 0u;
  for (uint32_t e = 0; e < n_experts; ++e) {
    if (row_count[e] == 0u) {
      continue;
    }
    ++n_with_rows;
    widest = (row_count[e] > widest) ? row_count[e] : widest;
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
  /* Bounds are checked because out_f32 is indexed by these; distinctness
     within one expert's slice is NOT, and moe_scatter_worker needs it (see
     its comment) or two workers read-modify-write one output row. Top-k
     routing gives it for free -- a token picks k distinct experts -- and
     checking it here would cost a per-expert bitmap over M on every call to
     restate what the caller's routing already guarantees.
     ponytail: a caller that repeats a row inside one expert gets a racy
     sum, not a wrong shape. If a future router can do that, the fix is to
     sort each slice and merge duplicates on the ARM side before the call,
     not a scan here. */
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
  /* The M=1 GEMV path (moe_m1_* above), decided here so the sizing below
     and the first gate_up push know. `widest` restates M <= 4 for a caller
     that repeats a row inside an expert: the GEMV tiles hold 4 rows. */
  const int use_m1 = (flags & HEXKL_MOE_FLAG_M1_GEMV) != 0u &&
                     M <= MOE_M1_MAX_ROWS && widest <= MOE_M1_MAX_ROWS &&
                     n_with_rows != 0u && n_with_rows <= MOE_M1_MAX_EXPERTS;
  hexkl_probe_us[HEXKL_PROBE_PATH] = use_m1 ? 1u : 0u;

  const uint32_t k_tiles = K / HEXKL_HMX_INT8_BLOCK_N_INNER;
  const uint32_t gu_ntiles = (2u * inter) / HEXKL_HMX_INT8_BLOCK_N_COL;
  /* gate_up's n-tiles are the gate's [0, inter_ntiles) then the up's; a
     gate tile and the up tile opposite it are inter_ntiles apart. */
  const uint32_t inter_ntiles = gu_ntiles / 2u;
  const uint32_t inter_ktiles = inter / HEXKL_HMX_INT8_BLOCK_N_INNER;
  const uint32_t dn_ntiles = N_out / HEXKL_HMX_INT8_BLOCK_N_COL;
  const uint32_t BR = HEXKL_HMX_INT8_BLOCK_N_ROW;
  /* The M=1 feed (#117): asked for, and the two gate_up slabs fit the
     arena with two downs fitting one slab. A shape that fails runs the
     arena read and the call's M1_FEED count says so. 7 MiB of the ~8.3 MB
     arena at the LFM2 shape; no heap, no address-space growth. */
  const uint32_t gu_bytes = k_tiles * gu_ntiles * WEIGHT_TILE_BYTES_U8I4;
  const uint32_t dn_bytes = inter_ktiles * dn_ntiles * WEIGHT_TILE_BYTES_U8I4;
  const int m1_feed = use_m1 && hexkl_moe_flags_feed(flags) != 0u &&
                      2u * (uint64_t)gu_bytes <= arena &&
                      2u * (uint64_t)dn_bytes <= gu_bytes;
  /* Pairs per staged batch; the chunk size the gate_up pushes use too. */
  const uint32_t half = L.acc_tiles / 2u;
  /* Bounded arrays below; a shape that needs more chunks than they hold is
     refused up front rather than overrun. 4 and 2 for this model. */
#define MOE_MAX_CHUNKS 16u
  if ((inter_ntiles + half - 1u) / half > MOE_MAX_CHUNKS ||
      (dn_ntiles + L.acc_tiles - 1u) / L.acc_tiles > MOE_MAX_CHUNKS) {
    return AEE_EUNSUPPORTED;
  }

  /* Scratch on the DSP heap, not VTCM: these are per-block staging areas
     read and written once each, so VTCM buys them nothing and the arena is
     the scarce resource. All of it is carved from one session-lifetime
     block (hexkl_moe_scratch): allocating and freeing the ~12.8 MB a
     prefill call needs cost 3.0 ms of the call, and the block only grows. */
  /* The whole activation, quantized once. hvx_quant_pack_u8_ah writes it,
     so a row's bytes are exactly what quantizing that row inside any expert
     block would have produced -- row quantization is independent of how
     rows are grouped -- and the per-block work becomes a uint8 move rather
     than a scan and a pack of the same rows four times over at top-4. */
  const uint32_t m_pad = ROUND_UP_U32(M, BR);
  /* One 64-row slot per row every expert asked for, padded up per block:
     the pack writes straight into this order so no gather is needed. A
     token picked by four experts occupies four slots. */
  uint32_t n_slots = 0u;
  for (uint32_t e = 0; e < n_experts; ++e) {
    n_slots += ROUND_UP_U32(row_count[e], HEXKL_HMX_INT8_BLOCK_N_ROW);
  }
  /* act_f32 and out_f32 are the host's FastRPC buffers, which are rpcmem
     and therefore UNCACHED. The gather reads M*K*4 bytes of act four times
     over at top-4 routing, and the scatter is a read-modify-write of
     M*N_out*4 one float at a time -- both catastrophic against uncached
     DDR, and measured so: 17.3 ms of gather+quant and 104.8 ms of scatter
     against 3.6 and 2.8 for the path this replaces (doc 46 section 10).
     Same axis Q1 broke on, doc 44 section 10.1.
     So each is touched exactly once, sequentially: act is copied in at the
     start (act_c) and out is copied out at the end (out_c), and everything
     in between works on cached heap. */
  /* Sizes once, in carve order; the sum is what the scratch must hold.
     order / base_of / slot_of are per-expert tables on the heap rather
     than stack arrays sized by HEXKL_MM_U8I4_MAX_WEIGHTS: that constant is
     2048 and has nothing to do with how many experts this layer has. */
  /* Reserved for the most slots THIS many rows can ever need -- every
     expert padded to a whole block -- not for this call's routing. The 22
     prefill calls of a token batch have the same n_rows but a different
     n_slots each, and reserving for the actual count grew the block on
     every layer that routed a little wider than the last: 323 us a call
     left of the 3064 us this scratch was meant to remove (doc 46 section
     50.7). With the bound, the first prefill call grows it once and no
     later call of the same batch can exceed it. */
  const size_t n_slots_cap = (size_t)n_rows + (size_t)n_experts * (BR - 1u);
  /* mid's row params, one set per mid buffer. */
  const size_t sz_scale = sizeof(float) * 2u * BR;
  const size_t sz_zp = sizeof(int32_t) * 2u * BR;
  const size_t sz_act_ah = n_slots_cap * K;
  const size_t sz_slot_u32 = sizeof(uint32_t) * n_slots_cap;
  const size_t sz_expert_u32 = sizeof(uint32_t) * n_experts;
  const size_t sz_mpad_u32 = sizeof(uint32_t) * m_pad;
  const size_t sz_act_c = sizeof(float) * (size_t)M * K;
  const size_t sz_out_c = sizeof(float) * (size_t)M * N_out;
  /* One done byte per slot block for the background pack. */
  const size_t sz_pack_done = n_slots_cap / HEXKL_MOE_PACK_UNIT_ROWS + 1u;
  /* The tail path (moe_tail_*): jobs, done bytes and contexts for as many
     tails as there are experts -- a bound, like n_slots_cap, so the block
     never regrows -- one shared staging set, and each tail's own result. */
  const size_t sz_jobs = sizeof(hvx_bg_job) * (1u + 3u * (size_t)n_experts);
  const size_t sz_tail_done =
    (size_t)n_experts * (inter_ntiles + 1u + dn_ntiles);
  const size_t sz_tail_ctx = sizeof(moe_tail_ctx) * (size_t)n_experts;
  const size_t sz_tail_acc_gu = (size_t)inter_ntiles * 2u * MOE_TAIL_TILE_BYTES;
  const size_t sz_tail_acc_dn = (size_t)dn_ntiles * MOE_TAIL_TILE_BYTES;
  const size_t sz_tail_gate = sizeof(float) * MOE_TAIL_MAX_ROWS * inter;
  const size_t sz_tail_mid =
    (size_t)inter_ktiles * HEXKL_HMX_ACTIVATION_ALIGNMENT;
  const size_t sz_tail_rq = sizeof(float) * HEXKL_HMX_INT8_BLOCK_N_ROW;
  const size_t sz_tail_res =
    sizeof(float) * MOE_TAIL_MAX_ROWS * (size_t)N_out * n_experts;
  /* The M=1 path's per-expert staging, reserved for MOE_M1_MAX_EXPERTS
     whenever the path is taken (the bound, so no later routing regrows). */
  const size_t sz_m1_expert =
    ROUND_UP_SZ((size_t)inter_ntiles * 2u * MOE_M1_TILE_BYTES,
                MOE_SCRATCH_ALIGN) +
    ROUND_UP_SZ((size_t)dn_ntiles * MOE_M1_TILE_BYTES, MOE_SCRATCH_ALIGN) +
    ROUND_UP_SZ(sizeof(float) * MOE_M1_MAX_ROWS * inter, MOE_SCRATCH_ALIGN) +
    ROUND_UP_SZ(sz_tail_mid, MOE_SCRATCH_ALIGN) +
    2u * ROUND_UP_SZ(sz_tail_rq, MOE_SCRATCH_ALIGN) +
    ROUND_UP_SZ(sizeof(float) * MOE_M1_MAX_ROWS * N_out, MOE_SCRATCH_ALIGN);
  const size_t sz_m1 = use_m1 ? MOE_M1_MAX_EXPERTS * sz_m1_expert : 0u;
  const size_t need = ROUND_UP_SZ(sz_scale, MOE_SCRATCH_ALIGN) +
                      ROUND_UP_SZ(sz_zp, MOE_SCRATCH_ALIGN) +
                      ROUND_UP_SZ(sz_act_ah, MOE_SCRATCH_ALIGN) +
                      3u * ROUND_UP_SZ(sz_slot_u32, MOE_SCRATCH_ALIGN) +
                      3u * ROUND_UP_SZ(sz_expert_u32, MOE_SCRATCH_ALIGN) +
                      2u * ROUND_UP_SZ(sz_mpad_u32, MOE_SCRATCH_ALIGN) +
                      ROUND_UP_SZ(sz_act_c, MOE_SCRATCH_ALIGN) +
                      ROUND_UP_SZ(sz_out_c, MOE_SCRATCH_ALIGN) +
                      ROUND_UP_SZ(sz_pack_done, MOE_SCRATCH_ALIGN) +
                      ROUND_UP_SZ(sz_jobs, MOE_SCRATCH_ALIGN) +
                      ROUND_UP_SZ(sz_tail_done, MOE_SCRATCH_ALIGN) +
                      ROUND_UP_SZ(sz_tail_ctx, MOE_SCRATCH_ALIGN) +
                      ROUND_UP_SZ(sz_tail_acc_gu, MOE_SCRATCH_ALIGN) +
                      ROUND_UP_SZ(sz_tail_acc_dn, MOE_SCRATCH_ALIGN) +
                      ROUND_UP_SZ(sz_tail_gate, MOE_SCRATCH_ALIGN) +
                      ROUND_UP_SZ(sz_tail_mid, MOE_SCRATCH_ALIGN) +
                      2u * ROUND_UP_SZ(sz_tail_rq, MOE_SCRATCH_ALIGN) +
                      ROUND_UP_SZ(sz_tail_res, MOE_SCRATCH_ALIGN) +
                      ROUND_UP_SZ(sz_expert_u32, MOE_SCRATCH_ALIGN) + sz_m1;
  uint64_t p_alloc = 0;
  HEXKL_PROBE_T0(p_alloc);
  rc = hexkl_moe_scratch_reserve(scratch, need);
  HEXKL_PROBE_ADD(HEXKL_PROBE_ALLOC, p_alloc);
  if (rc != AEE_SUCCESS) {
    return rc;
  }
  uint8_t *cur = scratch->base;
  float *rq_scale = (float *)hexkl_moe_carve(&cur, sz_scale);
  int32_t *rq_zp = (int32_t *)hexkl_moe_carve(&cur, sz_zp);
  uint8_t *act_ah = (uint8_t *)hexkl_moe_carve(&cur, sz_act_ah);
  uint32_t *slot_row = (uint32_t *)hexkl_moe_carve(&cur, sz_slot_u32);
  float *slot_scale = (float *)hexkl_moe_carve(&cur, sz_slot_u32);
  int32_t *slot_zp = (int32_t *)hexkl_moe_carve(&cur, sz_slot_u32);
  uint32_t *slot_of = (uint32_t *)hexkl_moe_carve(&cur, sz_expert_u32);
  uint32_t *order = (uint32_t *)hexkl_moe_carve(&cur, sz_expert_u32);
  uint32_t *base_of = (uint32_t *)hexkl_moe_carve(&cur, sz_expert_u32);
  float *scale_all = (float *)hexkl_moe_carve(&cur, sz_mpad_u32);
  int32_t *zp_all = (int32_t *)hexkl_moe_carve(&cur, sz_mpad_u32);
  float *act_c = (float *)hexkl_moe_carve(&cur, sz_act_c);
  float *out_c = (float *)hexkl_moe_carve(&cur, sz_out_c);
  uint8_t *pack_done = (uint8_t *)hexkl_moe_carve(&cur, sz_pack_done);
  hvx_bg_job *jobs = (hvx_bg_job *)hexkl_moe_carve(&cur, sz_jobs);
  uint8_t *tail_done = (uint8_t *)hexkl_moe_carve(&cur, sz_tail_done);
  moe_tail_ctx *tails = (moe_tail_ctx *)hexkl_moe_carve(&cur, sz_tail_ctx);
  moe_tail_shared tail_sh;
  tail_sh.acc_gu = (int32_t *)hexkl_moe_carve(&cur, sz_tail_acc_gu);
  tail_sh.acc_dn = (int32_t *)hexkl_moe_carve(&cur, sz_tail_acc_dn);
  tail_sh.gate_f32 = (float *)hexkl_moe_carve(&cur, sz_tail_gate);
  tail_sh.mid_ah = (uint8_t *)hexkl_moe_carve(&cur, sz_tail_mid);
  tail_sh.rq_scale = (float *)hexkl_moe_carve(&cur, sz_tail_rq);
  tail_sh.rq_zp = (int32_t *)hexkl_moe_carve(&cur, sz_tail_rq);
  float *tail_res = (float *)hexkl_moe_carve(&cur, sz_tail_res);
  /* Per active expert: its tail's index, or UINT32_MAX. */
  uint32_t *tail_of = (uint32_t *)hexkl_moe_carve(&cur, sz_expert_u32);
  uint8_t *m1_base = use_m1 ? (uint8_t *)hexkl_moe_carve(&cur, sz_m1) : NULL;
  moe_m1_ctx m1;
  hvx_bg_job *pack_job = &jobs[0];
  hvx_bg_job *last_job = NULL;
  uint32_t n_tails = 0u;
  hexkl_moe_pack_ctx pack;
  uint32_t n_active = 0u;
  uint64_t p0 = 0;
  /* The pool's in-flight jobs. Each outlives its submit until the wait that
     retires it, so they live here, not in the loop body: one per staging
     buffer for each epilogue, one for the first block's requantization. */
  hvx_dq_swiglu_job gu_job[2];
  moe_dn_ctx dn_ctx[2], rq0;
  /* HEXKL_MOE_MM_BEGIN/END's state. See the macros in hexkl_mm_u8i4_moe.h
     for why the HMX issue loop is timed by difference. */
  uint64_t mm_t0 = 0, mm_acc0 = 0, mm_dq0 = 0;

  /* The experts that actually have rows, in order. The DMA pipeline below
     hands each expert's gate_up transfer to its predecessor, so a skipped
     expert in the middle would break the chain; compacting first means the
     pipeline never has to think about empties. */
  {
    uint32_t base = 0u, slot = 0u;
    for (uint32_t e = 0; e < n_experts; ++e) {
      if (row_count[e] != 0u) {
        base_of[n_active] = base;
        slot_of[n_active] = slot;
        order[n_active++] = e;
        slot += ROUND_UP_U32(row_count[e], HEXKL_HMX_INT8_BLOCK_N_ROW);
      }
      base += row_count[e];
    }
  }
  HEXKL_PROBE_T0(p0);
  hexkl_dma_ring_reset();
  if (hexkl_probe_on) {
    hexkl_dma_trace_reset(hexkl_probe_now_ticks());
  }
  moe_dma_copy(act_c, act_f32, sizeof(float) * (size_t)M * K, 0, 0,
               HEXKL_DMA_SITE_COPY_IN);
  memset(out_c, 0, sizeof(float) * (size_t)M * N_out);
  HEXKL_PROBE_ADD(HEXKL_PROBE_ACC_COPY, p0);
  if (n_active == 0u) {
    moe_dma_copy(out_f32, out_c, sizeof(float) * (size_t)M * N_out, 0, 0,
                 HEXKL_DMA_SITE_COPY_OUT);
    rc = AEE_SUCCESS;
    goto out;
  }

  /* The first expert's gate_up goes out before the quantization, not after
     it: nothing in the scan or the pack needs VTCM's weight region, and a
     3.5 MB transfer that used to be the one wait with nothing to hide
     behind (DMA_FIRST) now has the whole scan in front of it. In paired
     chunks, so the first gate/up pair is usable long before the last --
     see moe_push_gate_up_chunks. */
  uint32_t gu_idx[MOE_MAX_CHUNKS];
  uint32_t gu_nchunk = 0u;
  /* The M=1 feed's ring indices: gate_up of expert i, down of expert j. */
  uint32_t m1_gu_idx[MOE_M1_MAX_EXPERTS], m1_dn_idx[MOE_M1_MAX_EXPERTS];
  if (!use_m1) {
    gu_nchunk = moe_push_gate_up_chunks(
      vtcm_base, L.w_gu_off, &tbl->slots[h_gate_up[order[0]]], k_tiles,
      gu_ntiles, inter_ntiles, half, 0u, gu_idx);
  } else if (m1_feed) {
    /* One gate_up per slab, at the same point for the same reason: the
       scan and the pack are in front of them. Whole matrices (f2). */
    for (uint32_t i = 0; i < n_active && i < 2u; ++i) {
      m1_gu_idx[i] = moe_m1_push(vtcm_base, moe_m1_gu_off(i, gu_bytes),
                                 &tbl->slots[h_gate_up[order[i]]], k_tiles,
                                 gu_ntiles, HEXKL_DMA_KIND_GATE, i);
    }
  }
  (void)gu_nchunk;

  /* The scan is per source row and independent of where a row ends up, so
     it still runs once over M rows. */
  HEXKL_PROBE_T0(p0);
  hvx_quant_rows_u8_params(act_c, M, m_pad, K, scale_all, zp_all, pool);

  /* Slot order: every active expert's rows, each expert padded up to a
     whole 64-row block. Padding slots repeat row 0 -- their accumulator
     output is never dequantized (m_blk bounds that) so the content does not
     matter, only that the read is in range. */
  {
    uint32_t d = 0u;
    for (uint32_t i = 0; i < n_active; ++i) {
      const uint32_t e = order[i];
      const uint32_t *rows_e = row_index + base_of[i];
      const uint32_t padded =
        ROUND_UP_U32(row_count[e], HEXKL_HMX_INT8_BLOCK_N_ROW);
      for (uint32_t r = 0; r < padded; ++r, ++d) {
        const uint32_t src = (r < row_count[e]) ? rows_e[r] : 0u;
        slot_row[d] = src;
        slot_scale[d] = scale_all[src];
        slot_zp[d] = zp_all[src];
      }
    }
    n_slots = d; /* inactive experts contribute nothing */
  }

  if (use_m1) {
    /* Each expert's rows packed into its slot block inline -- at most
       MOE_M1_MAX_ROWS x MOE_M1_MAX_EXPERTS rows, no background job; QUANT
       (opened before the scan) closes after it. The pack reads whole row
       groups, and the block's padding slots repeat row 0 (above), so the
       reads past m are in range and the rows they write are never read
       (gemm_rows4 clamps to the count). */
    for (uint32_t i = 0; i < n_active; ++i) {
      const uint32_t e = order[i];
      const uint32_t sb = slot_of[i];
      uint8_t *xb = m1_base + (size_t)i * sz_m1_expert;
      moe_m1_expert *x = &m1.ex[i];
      hvx_quant_pack_u8_ah_rows(act_c, slot_row + sb, 0u, row_count[e], K,
                                slot_scale + sb, slot_zp + sb,
                                act_ah + (size_t)sb * K);
      x->act_ah = act_ah + (size_t)sb * K;
      x->g = &tbl->slots[h_gate_up[e]];
      x->d = &tbl->slots[h_down[e]];
      x->gu_wh =
        m1_feed ? vtcm_base + moe_m1_gu_off(i, gu_bytes) : x->g->wh_bytes;
      x->dn_wh = m1_feed ? vtcm_base + moe_m1_dn_off(i, gu_bytes, dn_bytes)
                         : x->d->wh_bytes;
      x->act_scale = slot_scale + sb;
      x->act_zp = slot_zp + sb;
      x->acc_gu = (int32_t *)hexkl_moe_carve(&xb, (size_t)inter_ntiles * 2u *
                                                    MOE_M1_TILE_BYTES);
      x->acc_dn =
        (int32_t *)hexkl_moe_carve(&xb, (size_t)dn_ntiles * MOE_M1_TILE_BYTES);
      x->gate_f32 =
        (float *)hexkl_moe_carve(&xb, sizeof(float) * MOE_M1_MAX_ROWS * inter);
      x->mid_ah = (uint8_t *)hexkl_moe_carve(&xb, sz_tail_mid);
      x->rq_scale = (float *)hexkl_moe_carve(&xb, sz_tail_rq);
      x->rq_zp = (int32_t *)hexkl_moe_carve(&xb, sz_tail_rq);
      x->res =
        (float *)hexkl_moe_carve(&xb, sizeof(float) * MOE_M1_MAX_ROWS * N_out);
      x->m = row_count[e];
    }
    HEXKL_PROBE_ADD(HEXKL_PROBE_QUANT, p0);
    m1.n_active = n_active;
    m1.k_tiles = k_tiles;
    m1.inter = inter;
    m1.inter_ktiles = inter_ktiles;
    m1.inter_ntiles = inter_ntiles;
    m1.gu_ntiles = gu_ntiles;
    m1.dn_ntiles = dn_ntiles;
    m1.N_out = N_out;
    /* The (loop, lead) pair, resolved once per call: the flags word when
       the caller set the tune bit, else the build's defaults (#113). */
    /* Under the feed the lead is off by construction (the section
       comment); the banner still prints the requested lead beside
       feed=vtcm so the log names the cell. */
    m1.lead_kb = m1_feed ? 0u : hexkl_moe_flags_lead_kb(flags);
    m1.rows1 = hexkl_moe_flags_rows1(flags);
    m1.feed = m1_feed ? 1u : 0u;
    HEXKL_PROBE_COUNT(HEXKL_PROBE_M1_FEED, m1_feed ? 1u : 0u);

    /* MM is the wall of the two GEMV stages on the caller; SWIGLU their
       summed worker-time (moe_worker_probe_add), so SWIGLU / lanes ~ MM says
       the lanes were balanced, and the bytes over MM is the arena read
       rate. REQUANT and SCATTER are their stages' wall; BLOCKS, ACC_READ,
       DEQUANT and DRAIN stay 0. With the feed, DMA_KB counts the eight
       pushes, the waits sit inside MM, and the first wait sets DMA_FIRST
       with the HMX path's caveat (the push went out before the scan, so
       it times the exposed remainder). */
    HEXKL_PROBE_T0(p0);
    if (!m1_feed) {
      m1.u_lo = 0u;
      m1.u_hi = n_active * inter_ntiles;
      hvx_worker_pool_run(pool, moe_m1_pair_worker, &m1,
                          n_active * inter_ntiles);
    } else {
      for (uint32_t i = 0; i < n_active; ++i) {
        uint64_t pw = 0;
        HEXKL_PROBE_T0(pw);
        moe_m1_wait(m1_gu_idx[i], HEXKL_DMA_SITE_GU);
        if (i == 0u) {
          HEXKL_PROBE_ADD(HEXKL_PROBE_DMA_FIRST, pw);
          HEXKL_PROBE_COUNT(HEXKL_PROBE_DMA_FIRST_KB, gu_bytes >> 10);
        }
        m1.u_lo = i * inter_ntiles;
        m1.u_hi = (i + 1u) * inter_ntiles;
        hvx_worker_pool_run(pool, moe_m1_pair_worker, &m1, inter_ntiles);
        /* Slab i & 1 is free: the join above is the barrier. It takes the
           gate_up two experts on, or -- once no gate_up is left for it --
           its two downs (experts 2s and 2s + 1 of slab s). */
        if (i + 2u < n_active) {
          m1_gu_idx[i + 2u] = moe_m1_push(
            vtcm_base, moe_m1_gu_off(i + 2u, gu_bytes), m1.ex[i + 2u].g,
            k_tiles, gu_ntiles, HEXKL_DMA_KIND_GATE, i + 2u);
        } else {
          for (uint32_t j = 2u * (i & 1u); j < 2u * (i & 1u) + 2u; ++j) {
            if (j < n_active) {
              m1_dn_idx[j] = moe_m1_push(
                vtcm_base, moe_m1_dn_off(j, gu_bytes, dn_bytes), m1.ex[j].d,
                inter_ktiles, dn_ntiles, HEXKL_DMA_KIND_DOWN, j);
            }
          }
        }
      }
    }
    HEXKL_PROBE_ADD(HEXKL_PROBE_MM, p0);
    HEXKL_PROBE_T0(p0);
    hvx_worker_pool_run(pool, moe_m1_requant_worker, &m1, n_active);
    HEXKL_PROBE_ADD(HEXKL_PROBE_REQUANT, p0);
    HEXKL_PROBE_T0(p0);
    if (!m1_feed) {
      m1.u_lo = 0u;
      m1.u_hi = n_active * dn_ntiles;
      hvx_worker_pool_run(pool, moe_m1_down_worker, &m1, n_active * dn_ntiles);
    } else {
      for (uint32_t j = 0; j < n_active; ++j) {
        moe_m1_wait(m1_dn_idx[j], HEXKL_DMA_SITE_DN);
        m1.u_lo = j * dn_ntiles;
        m1.u_hi = (j + 1u) * dn_ntiles;
        hvx_worker_pool_run(pool, moe_m1_down_worker, &m1, dn_ntiles);
        /* Down slot j & 3 is free after the join; a fifth or later expert
           (M <= 4 routing to more than four) takes it. */
        if (j + 4u < n_active) {
          m1_dn_idx[j + 4u] =
            moe_m1_push(vtcm_base, moe_m1_dn_off(j + 4u, gu_bytes, dn_bytes),
                        m1.ex[j + 4u].d, inter_ktiles, dn_ntiles,
                        HEXKL_DMA_KIND_DOWN, j + 4u);
        }
      }
    }
    HEXKL_PROBE_ADD(HEXKL_PROBE_MM, p0);

    HEXKL_PROBE_T0(p0);
    for (uint32_t i = 0; i < n_active; ++i) {
      const moe_m1_expert *x = &m1.ex[i];
      const uint32_t *rows = row_index + base_of[i];
      const float *weights = row_weight + base_of[i];
      for (uint32_t r = 0; r < x->m; ++r) {
        hvx_scale_add_rows_f32(out_c + (size_t)rows[r] * N_out,
                               x->res + (size_t)r * N_out, weights[r], N_out);
      }
    }
    HEXKL_PROBE_ADD(HEXKL_PROBE_SCATTER, p0);

    HEXKL_PROBE_T0(p0);
    moe_dma_copy(out_f32, out_c, sizeof(float) * (size_t)M * N_out, 0, 0,
                 HEXKL_DMA_SITE_COPY_OUT);
    HEXKL_PROBE_ADD(HEXKL_PROBE_ACC_COPY, p0);
    rc = AEE_SUCCESS;
    goto out;
  }

  /* Packed straight into slot order, so a block's 64 rows are already
     contiguous and the gather pass that used to follow is gone. On the
     pool's background lane, one unit per block (hexkl_moe_pack_bg_worker): only
     the block about to be queued is waited for, so from here on the pack
     runs under the HMX. QUANT times the scan and those waits -- what stays
     exposed -- not the pack. */
  pack.act_c = act_c;
  pack.slot_row = slot_row;
  pack.slot_scale = slot_scale;
  pack.slot_zp = slot_zp;
  pack.act_ah = act_ah;
  pack.K = K;
  pack_job->func = hexkl_moe_pack_bg_worker;
  pack_job->ctx = &pack;
  pack_job->n_units = n_slots / HEXKL_MOE_PACK_UNIT_ROWS;
  pack_job->done = pack_done;
  hvx_worker_pool_submit_bg(pool, pack_job);
  last_job = pack_job;

  /* Every tail's three jobs, queued now in expert order behind the pack --
     see the moe_tail_* comment. Nothing they read is produced by the HMX
     loop, so the earlier they queue the more of the loop they hide under. */
  for (uint32_t i = 0; i < n_active; ++i) {
    const uint32_t e = order[i];
    const uint32_t tr = moe_tail_rows(row_count[e]);
    tail_of[i] = UINT32_MAX;
    if (tr == 0u) {
      continue;
    }
    const uint32_t t = n_tails++;
    moe_tail_ctx *tc = &tails[t];
    const uint32_t sb = slot_of[i] + (row_count[e] - tr);
    tc->sh = &tail_sh;
    tc->act_ah = act_ah + (size_t)sb * K;
    tc->g = &tbl->slots[h_gate_up[e]];
    tc->d = &tbl->slots[h_down[e]];
    tc->act_scale = slot_scale + sb;
    tc->act_zp = slot_zp + sb;
    tc->res = tail_res + (size_t)t * MOE_TAIL_MAX_ROWS * N_out;
    tc->m = tr;
    tc->k_tiles = k_tiles;
    tc->inter = inter;
    tc->inter_ktiles = inter_ktiles;
    tc->inter_ntiles = inter_ntiles;
    tc->gu_ntiles = gu_ntiles;
    tc->dn_ntiles = dn_ntiles;
    tc->N_out = N_out;
    hvx_bg_job *jb = &jobs[1u + 3u * t];
    uint8_t *dn = tail_done + (size_t)t * (inter_ntiles + 1u + dn_ntiles);
    jb[0].func = moe_tail_pair_unit;
    jb[0].ctx = tc;
    jb[0].n_units = inter_ntiles;
    jb[0].done = dn;
    jb[1].func = moe_tail_requant_unit;
    jb[1].ctx = tc;
    jb[1].n_units = 1u;
    jb[1].done = dn + inter_ntiles;
    jb[2].func = moe_tail_down_unit;
    jb[2].ctx = tc;
    jb[2].n_units = dn_ntiles;
    jb[2].done = dn + inter_ntiles + 1u;
    hvx_worker_pool_submit_bg(pool, &jb[0]);
    hvx_worker_pool_submit_bg(pool, &jb[1]);
    hvx_worker_pool_submit_bg(pool, &jb[2]);
    last_job = &jb[2];
    tail_of[i] = t;
  }
  hvx_worker_pool_wait_bg(pool, pack_job,
                          HEXKL_MOE_PACK_UNITS_THROUGH(slot_of[0]));
  HEXKL_PROBE_ADD(HEXKL_PROBE_QUANT, p0);

  /* ---- The block sequence ---------------------------------------------
   *
   * Blocks are numbered n = 0, 1, ... in expert order. GU(n) is block n's
   * gate_up matmul with its fused epilogues, rq(n) the requantization of
   * its gate_off into mid, DN(n) its down matmul with its epilogues. They
   * are issued one block behind:
   *
   *   GU(0) rq(0) | GU(1) DN(0) | GU(2) DN(1) | ... | GU(N-1) DN(N-2) | DN(N-1)
   *
   * The foreground lane runs one job at a time, and a job hides only when
   * an HMX batch is issued between its submit and the wait that retires
   * it (doc 47 section 14). In block order -- GU(n) then DN(n) -- three
   * jobs a block had nothing to hide behind: the last gate_up epilogue
   * (rq needs all of gate_off), rq itself, and the last down epilogue
   * with the scatter after it; 1.5-2 ms of a 15 ms prefill call (doc 51
   * section 2.21). One block behind, DN(n-1)'s batches cover GU(n)'s last
   * epilogue and rq(n), and GU(n+1)'s first covers DN(n-1)'s last. rq(n)
   * rides inside DN(n-1)'s first epilogue job and the scatter inside every
   * down epilogue (moe_dn_worker), so a block submits exactly as many jobs
   * as it issues batches; what stays exposed is the two ends of the call.
   *
   * What the reorder needs. Two mid buffers, DN(n-1) reading one while
   * rq(n) writes the other, with two sets of row params. down[e] pushed
   * after DN(prev) has issued, not before GU(e)'s first block: a new
   * expert's down would otherwise land on the one DN(prev) still reads.
   * The staging parity (sb) carried across gate_up and down batches: a
   * staging buffer is free once the epilogue that read it is retired,
   * which is the wait one issue back whichever phase that was. gate_off
   * stays single, because rq(n) is retired before GU(n+1)'s first
   * epilogue is submitted. gate_up[e+1] and the next activation block go
   * out as before, once GU(n) has freed their slots, and arrive under
   * DN(n-1). */
  moe_blk blk, pblk;
  moe_blk_set(&blk, 0u, 0u, order, row_count, slot_of);
  /* The first expert's first activation block. Queued behind its gate_up
     here (the one place the order is reversed: that gate_up has the scan
     to hide behind and is long done). Every later block is queued when
     the block before it has freed A -- see moe_push_act_block for why the
     order matters. */
  uint32_t act_idx = moe_push_act_block(vtcm_base, L.act_off, act_ah, blk.slot,
                                        K, k_tiles, 0u, 0u);
  uint32_t dn_idx[MOE_MAX_CHUNKS];
  uint32_t dn_nchunk = 0u;
  /* The active expert whose down is in buffer B, or on its way there. */
  uint32_t dn_resident = UINT32_MAX;
  uint32_t sb = 0u; /* staging buffer parity, across every batch */
  const uint32_t mid_bytes = inter_ktiles * HEXKL_HMX_ACTIVATION_ALIGNMENT;
  float *const gate = (float *)(vtcm_base + L.gate_off);
  int have_cur = 1;

  for (uint32_t n = 0;; ++n) {
    moe_blk nblk = blk;
    int have_next = 0;
    if (have_cur) {
      const hexkl_weight_u8i4 *g = &tbl->slots[h_gate_up[blk.e]];
      have_next = moe_blk_next(&nblk, n_active, order, row_count, slot_of);
      /* The block's rows' quantization parameters: contiguous in slot
         order, since every expert's slots are padded to whole blocks. */
      const float *act_scale = slot_scale + blk.slot;
      const int32_t *act_zp = slot_zp + blk.slot;
      /* Counted, not timed -- see HEXKL_PROBE_BLOCKS. */
      HEXKL_PROBE_COUNT(HEXKL_PROBE_BLOCKS, 1);

      /* The pack already wrote this block in slot order, so its 64 rows are
         one contiguous run -- a single DMA, not a copy. Doing it on the
         core moved 5.5 MB a layer at 3.1 GB/s, against the 33 the engine
         measures, because the destination is VTCM and the core is the wrong
         thing to write it with. Rows past m_blk are padding whose
         accumulator output is never read. By now the block has had a down
         matmul to arrive under. */
      HEXKL_PROBE_T0(p0);
      if (hexkl_probe_on) {
        hexkl_dma_trace_wait_begin(hexkl_probe_now_ticks(), act_idx,
                                   HEXKL_DMA_SITE_ACT);
      }
      hexkl_dma_ring_wait(act_idx);
      if (hexkl_probe_on) {
        hexkl_dma_trace_wait_end(hexkl_probe_now_ticks());
      }
      HEXKL_PROBE_ADD(HEXKL_PROBE_GATHER, p0);

      /* --- gate_up ------------------------------------------------- */
      /* Matmul a batch of gate/up PAIRS -- gate tiles g0.. into staged
         slots [0, np) of one staging buffer, the up tiles opposite them
         into [np, 2np) -- then hand the batch to the pool, which
         dequantizes and SwiGLUs it straight into gate_off (the same
         hvx_swiglu_det_sf on the same two vectors the store-and-reread
         sequence had, so no byte changes) while THIS thread issues the
         nblk batch into the other staging buffer. A batch's HMX issue
         (47 us at prefill) covers the previous batch's epilogue (18 us).
         DEQUANT times the waits -- what stays exposed -- not the work. */
      for (uint32_t g0 = 0, gb = 0; g0 < inter_ntiles; g0 += half, ++gb) {
        const uint32_t np =
          (inter_ntiles - g0 < half) ? (inter_ntiles - g0) : half;
        const uint32_t stage_off =
          L.result_off + (sb & 1u) * L.acc_tiles * ACC_TILE_BYTES;
        /* Only the first block of an expert waits: by the second the whole
           weight is resident. n == 0 && gb == 0 is the one wait that cannot
           hide behind anything, which is what DMA_FIRST records. */
        if (blk.first) {
          HEXKL_PROBE_T0(p0);
          if (hexkl_probe_on) {
            hexkl_dma_trace_wait_begin(hexkl_probe_now_ticks(), gu_idx[gb],
                                       HEXKL_DMA_SITE_GU);
          }
          hexkl_dma_ring_wait(gu_idx[gb]);
          if (hexkl_probe_on) {
            hexkl_dma_trace_wait_end(hexkl_probe_now_ticks());
          }
          HEXKL_PROBE_ADD(HEXKL_PROBE_DRAIN, p0);
          if (n == 0u && gb == 0u) {
            hexkl_probe_us[HEXKL_PROBE_DMA_FIRST] =
              hexkl_probe_us[HEXKL_PROBE_DRAIN];
            HEXKL_PROBE_COUNT(HEXKL_PROBE_DMA_FIRST_KB,
                              (2u * np * k_tiles * WEIGHT_TILE_BYTES_U8I4) >>
                                10);
          }
        }
        /* This staging buffer was last read by the epilogue two issues
           back, retired when the last one was submitted. */
        HEXKL_MOE_MM_BEGIN();
        for (uint32_t j = 0; j < 2u * np; ++j) {
          const uint32_t col =
            (j < np) ? (g0 + j) : (inter_ntiles + g0 + (j - np));
          hexkl_micro_hmx_acc_clear_int32();
          for (uint32_t kt = 0; kt < k_tiles; ++kt) {
            rc = hexkl_micro_hmx_mm_u8i4(
              vtcm_base, L.act_off + kt * HEXKL_HMX_ACTIVATION_ALIGNMENT,
              L.w_gu_off + (kt * gu_ntiles + col) * WEIGHT_TILE_BYTES_U8I4);
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
        HEXKL_MOE_MM_END();

        /* Retire what the pool is running -- DN(n-1)'s last epilogue at
           gb == 0, epilogue gb-1 otherwise (its buffer is the one batch
           gb+1 will fill) -- then launch this batch's. Each is timed in
           its own column, so both read what is exposed after a batch's
           issue has covered them. */
        HEXKL_PROBE_T0(p0);
        hvx_worker_pool_wait(pool);
        HEXKL_PROBE_ADD(gb == 0u ? HEXKL_PROBE_SCATTER : HEXKL_PROBE_DEQUANT,
                        p0);
        {
          hvx_dq_swiglu_job *jb = &gu_job[sb & 1u];
          jb->tiles_base =
            (const uint8_t *)((const int32_t *)(vtcm_base + stage_off) +
                              acc->base);
          jb->tile_stride = ACC_TILE_BYTES;
          jb->n_pairs = np;
          jb->g0 = g0;
          jb->row_stride = acc->row_stride;
          jb->m_count = blk.m_blk;
          jb->act_scale = act_scale;
          jb->act_zp = act_zp;
          jb->colsum_w = g->colsum_w;
          jb->w_scale = g->w_scale;
          jb->bias = g->bias;
          jb->inter = inter;
          jb->dst = gate;
          jb->dst_stride = inter;
          hvx_worker_pool_submit(pool, moe_gu_worker, jb, np);
        }
        ++sb;
      }

      /* A and the gate_up slot are free once the last batch is issued: the
         next block's activation, and its expert's gate_up when it is a
         new expert, go out now and arrive under DN(n-1). Activation
         first: it is what GU(n+1) waits on before anything else, and
         queued behind 3.5 MB of gate_up that wait would cover the gate_up
         too. Measured, not guessed: pushing the next expert's chunks early,
         one after each batch that freed its columns, put the activation
         behind them in the ring and GATHER went 84 -> 747 us a call (388
         at decode) against DRAIN's 251 -> 18 -- the DMA moves ~12 GB/s
         beside the HMX, not the idle 33, so 3.5 MB does not land in the
         two down batches it had (doc 51 section 2.26). */
      if (have_next) {
        HEXKL_PROBE_T0(p0);
        hvx_worker_pool_wait_bg(pool, pack_job,
                                HEXKL_MOE_PACK_UNITS_THROUGH(nblk.slot));
        HEXKL_PROBE_ADD(HEXKL_PROBE_QUANT, p0);
        act_idx = moe_push_act_block(vtcm_base, L.act_off, act_ah, nblk.slot, K,
                                     k_tiles, nblk.i,
                                     nblk.mb / HEXKL_HMX_INT8_BLOCK_N_ROW);
        if (nblk.first) {
          gu_nchunk = moe_push_gate_up_chunks(
            vtcm_base, L.w_gu_off, &tbl->slots[h_gate_up[nblk.e]], k_tiles,
            gu_ntiles, inter_ntiles, half, nblk.i, gu_idx);
        }
      }
    }

    if (n == 0u) {
      /* rq(0) has no down to ride in: its own job, behind the one wait for
         a last gate_up epilogue the call makes. GU(1)'s first batch covers
         it -- or DN(0)'s wait below retires it, when block 0 is the only
         block. */
      HEXKL_PROBE_T0(p0);
      hvx_worker_pool_wait(pool);
      HEXKL_PROBE_ADD(HEXKL_PROBE_DEQUANT, p0);
      memset(&rq0, 0, sizeof rq0);
      rq0.gate = gate;
      rq0.rq_m_blk = blk.m_blk;
      rq0.inter = inter;
      rq0.rq_scale = rq_scale;
      rq0.rq_zp = rq_zp;
      rq0.mid = vtcm_base + L.mid_off;
      hvx_worker_pool_submit(pool, moe_dn_worker, &rq0, MOE_RQ_UNITS);
    } else {
      /* --- down of block n-1 ------------------------------------------ */
      const hexkl_weight_u8i4 *d = &tbl->slots[h_down[pblk.e]];
      const uint32_t pp = (n - 1u) & 1u; /* its mid buffer and row params */
      const uint32_t *rows = row_index + base_of[pblk.i] + pblk.mb;
      const float *weights = row_weight + base_of[pblk.i] + pblk.mb;
      const moe_tail_ctx *tail = (pblk.last && tail_of[pblk.i] != UINT32_MAX)
                                   ? &tails[tail_of[pblk.i]]
                                   : NULL;
      if (!have_cur) {
        /* The last block: no gate_up has been issued since the job that
           carries rq(n-1), so retire it before its mid is read. */
        HEXKL_PROBE_T0(p0);
        hvx_worker_pool_wait(pool);
        HEXKL_PROBE_ADD(HEXKL_PROBE_REQUANT, p0);
      }
      if (tail) {
        /* The tail's rows go out with this block's. Its down job is
           usually long complete; what the wait reads is the exposure when
           it is not, filed under SCATTER. */
        HEXKL_PROBE_T0(p0);
        hvx_worker_pool_wait_bg(pool, &jobs[1u + 3u * tail_of[pblk.i] + 2u],
                                UINT32_MAX);
        HEXKL_PROBE_ADD(HEXKL_PROBE_SCATTER, p0);
      }
      for (uint32_t nt0 = 0, db = 0; nt0 < dn_ntiles;
           nt0 += L.acc_tiles, ++db) {
        const uint32_t nb =
          (dn_ntiles - nt0 < L.acc_tiles) ? (dn_ntiles - nt0) : L.acc_tiles;
        const uint32_t stage_off =
          L.result_off + (sb & 1u) * L.acc_tiles * ACC_TILE_BYTES;
        if (pblk.first) {
          HEXKL_PROBE_T0(p0);
          if (hexkl_probe_on) {
            hexkl_dma_trace_wait_begin(hexkl_probe_now_ticks(), dn_idx[db],
                                       HEXKL_DMA_SITE_DN);
          }
          hexkl_dma_ring_wait(dn_idx[db]);
          if (hexkl_probe_on) {
            hexkl_dma_trace_wait_end(hexkl_probe_now_ticks());
          }
          HEXKL_PROBE_ADD(HEXKL_PROBE_DRAIN_DN, p0);
        }
        HEXKL_MOE_MM_BEGIN();
        for (uint32_t j = 0; j < nb; ++j) {
          hexkl_micro_hmx_acc_clear_int32();
          for (uint32_t kt = 0; kt < inter_ktiles; ++kt) {
            rc = hexkl_micro_hmx_mm_u8i4(
              vtcm_base,
              L.mid_off + pp * mid_bytes + kt * HEXKL_HMX_ACTIVATION_ALIGNMENT,
              L.w_dn_off + (kt * dn_ntiles + nt0 + j) * WEIGHT_TILE_BYTES_U8I4);
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
        HEXKL_MOE_MM_END();

        /* db == 0 retires GU(n)'s last epilogue -- gate_off is complete,
           which the requantization in this job needs -- and every later
           wait the epilogue before it, the one carrying rq(n) included. */
        HEXKL_PROBE_T0(p0);
        hvx_worker_pool_wait(pool);
        HEXKL_PROBE_ADD(db == 0u ? HEXKL_PROBE_DEQUANT : HEXKL_PROBE_REQUANT,
                        p0);
        {
          moe_dn_ctx *c = &dn_ctx[sb & 1u];
          c->tiles_base =
            (const uint8_t *)((const int32_t *)(vtcm_base + stage_off) +
                              acc->base);
          c->tile_stride = ACC_TILE_BYTES;
          c->nt0 = nt0;
          c->nb = nb;
          c->row_stride = acc->row_stride;
          c->m_count = pblk.m_blk;
          c->act_scale = rq_scale + pp * BR;
          c->act_zp = rq_zp + pp * BR;
          c->colsum_w = d->colsum_w;
          c->w_scale = d->w_scale;
          c->bias = d->bias;
          c->res = (float *)(vtcm_base + L.res_f32_off);
          c->out = out_c;
          c->rows = rows;
          c->weights = weights;
          c->N_out = N_out;
          if (tail) {
            const uint32_t hmx_rows = row_count[pblk.e] - tail->m;
            c->res_b = tail->res;
            c->rows_b = row_index + base_of[pblk.i] + hmx_rows;
            c->weights_b = row_weight + base_of[pblk.i] + hmx_rows;
            c->n_rows_b = tail->m;
          } else {
            c->res_b = NULL;
            c->rows_b = NULL;
            c->weights_b = NULL;
            c->n_rows_b = 0u;
          }
          if (db == 0u && have_cur) {
            c->gate = gate;
            c->rq_m_blk = blk.m_blk;
            c->inter = inter;
            c->rq_scale = rq_scale + (n & 1u) * BR;
            c->rq_zp = rq_zp + (n & 1u) * BR;
            c->mid = vtcm_base + L.mid_off + (n & 1u) * mid_bytes;
          } else {
            c->gate = NULL;
          }
          hvx_worker_pool_submit(pool, moe_dn_worker, c,
                                 nb + (c->gate ? MOE_RQ_UNITS : 0u));
        }
        ++sb;
      }
    }

    if (!have_cur) {
      break;
    }
    /* This block's down, for DN(n) after GU(n+1): pushed now that DN(n-1)
       has issued its last read of buffer B, in chunks for the same reason
       gate_up is (waiting for all 1.75 MB before the first n-tile column
       cost 55 us an expert). Already there when the expert spans blocks. */
    if (dn_resident != blk.i) {
      const hexkl_weight_u8i4 *d = &tbl->slots[h_down[blk.e]];
      dn_nchunk = 0u;
      for (uint32_t nt0 = 0; nt0 < dn_ntiles; nt0 += L.acc_tiles) {
        const uint32_t cn =
          (dn_ntiles - nt0 < L.acc_tiles) ? (dn_ntiles - nt0) : L.acc_tiles;
        dn_idx[dn_nchunk] = moe_push_weight_chunk(
          vtcm_base, L.w_dn_off, d, inter_ktiles, dn_ntiles, nt0, cn,
          HEXKL_DMA_KIND_DOWN, blk.i, dn_nchunk);
        ++dn_nchunk;
      }
      dn_resident = blk.i;
    }
    pblk = blk;
    blk = nblk;
    have_cur = have_next;
  }

  /* The last down epilogue. */
  HEXKL_PROBE_T0(p0);
  hvx_worker_pool_wait(pool);
  HEXKL_PROBE_ADD(HEXKL_PROBE_SCATTER, p0);

  HEXKL_PROBE_T0(p0);
  moe_dma_copy(out_f32, out_c, sizeof(float) * (size_t)M * N_out, 0, 0,
               HEXKL_DMA_SITE_COPY_OUT);
  HEXKL_PROBE_ADD(HEXKL_PROBE_ACC_COPY, p0);

out:
  /* An error path may leave a job in flight over VTCM this call owns, and
     the background pack over this call's scratch. */
  hvx_worker_pool_wait(pool);
  if (last_job) {
    hvx_worker_pool_wait_bg(pool, last_job, UINT32_MAX);
  }
  if (hexkl_probe_on) {
    /* #87: close the trace and fold its per-call numbers into the probe
       slots, ticks to microseconds through the same HAP conversion the
       stage timers use. Counts stay counts. */
    hexkl_dma_trace_summary ds;
    hexkl_dma_trace_finish(hexkl_probe_now_ticks(), &ds);
    hexkl_probe_us[HEXKL_PROBE_DMA_DESC] = ds.n_desc;
    hexkl_probe_us[HEXKL_PROBE_DMA_WAITS] = ds.n_wait;
    hexkl_probe_us[HEXKL_PROBE_DMA_WAITS_BLOCKED] = ds.n_blocked;
    hexkl_probe_us[HEXKL_PROBE_DMA_DEPTH_MAX] = ds.depth_max;
    hexkl_probe_us[HEXKL_PROBE_DMA_WAIT_US] =
      HAP_perf_qtimer_count_to_us(ds.wait);
    hexkl_probe_us[HEXKL_PROBE_DMA_WAIT_ACT_US] =
      HAP_perf_qtimer_count_to_us(ds.wait_act);
    hexkl_probe_us[HEXKL_PROBE_DMA_BUSY_LO_US] =
      HAP_perf_qtimer_count_to_us(ds.busy_lo);
    hexkl_probe_us[HEXKL_PROBE_DMA_BUSY_HI_US] =
      HAP_perf_qtimer_count_to_us(ds.busy_hi);
    hexkl_probe_us[HEXKL_PROBE_DMA_FIRST_READY_US] =
      HAP_perf_qtimer_count_to_us(ds.first_ready);
    hexkl_probe_us[HEXKL_PROBE_DMA_LAST_ISSUE_US] =
      HAP_perf_qtimer_count_to_us(ds.last_issue);
  }
  /* Nothing to free: the scratch stays with the session. */
  return rc;
}
