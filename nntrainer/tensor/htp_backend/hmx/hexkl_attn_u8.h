// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hexkl_attn_u8.h
 * @date   07 Aug 2026
 * @brief  KV block registry and the S = Q.Kt half of the fused attention path
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * ONE module for both widths, via hexkl_attn_ops, and per operand: Kt and V
 * each carry their own ops, which is what makes the (w_k, w_v) knob of
 * MHA_HTP_U8_TASKS.md §0.2 free.
 *
 * There is no matmul in here. hexkl_mm_u8iX_layer_run already takes one shared
 * activation plus a list of registered weights and returns dequantized f32 --
 * which IS S = Q.Kt with different arguments, cross-block weight prefetch
 * included. Reaching for hexkl_micro_hmx_mm_u8iX directly would be a wrong
 * turn.
 *
 * The KV cache is a SEQUENCE OF REGISTERED BLOCKS, not one big weight, and
 * that is structural rather than a convenience: V's per-column scale is taken
 * over its own block's rows only, so a block registered once never has to be
 * revisited as later tokens arrive (§1.5). Only the partially filled tail
 * block is re-registered on append.
 */

#ifndef __NNTRAINER_HEXKL_ATTN_U8_H__
#define __NNTRAINER_HEXKL_ATTN_U8_H__

#include <stdint.h>

#include "hexkl_attn_dtype.h"

struct hvx_worker_pool_s; /* hvx_worker_pool.h; only a pointer is held here */

/** @brief Upper bound on KV blocks per head. max_kv 8192 at T=64 needs 128. */
#define HEXKL_ATTN_MAX_BLOCKS 128u
/** @brief Upper bound on cache heads, so the handle table is a flat array. */
#define HEXKL_ATTN_MAX_HEADS 32u

/** @brief Marks a block slot that holds no registered weight. */
#define HEXKL_ATTN_NO_HANDLE 0xFFFFFFFFu

/**
 * @brief One attention layer's DSP-side state.
 *
 * The fp16 shadows exist because a block must be re-quantized whenever a new
 * token lands inside it, and the quantizer reads whole rows -- so the rows have
 * to still be here. They are the DSP-side KV cache of MHA_HTP_PLAN.md §2.2,
 * held in DSP heap rather than VTCM (VTCM is compute scratch; the DMA ring
 * streams blocks in from here).
 */
typedef struct {
  uint8_t *vtcm_base;
  uint32_t vtcm_size;
  uint32_t config_off;

  uint32_t nch, gqa, head_dim, max_kv, T, M_band;
  uint32_t max_blocks; /**< ceil(max_kv / T) */
  uint32_t kv_len;     /**< positions appended so far */

  hexkl_attn_ops ops_k, ops_v;

  uint16_t *k_shadow; /**< [max_kv][nch][head_dim] fp16, CPU cache layout */
  uint16_t *v_shadow;

  /** Per (head, block) weight handles; HEXKL_ATTN_NO_HANDLE when unregistered.
      Indexed head * max_blocks + block. */
  uint32_t *h_kt;
  uint32_t *h_v;

  /** Scratch for quantizing one block. Sized for the larger of the two
     orientations, allocated once so append is not a malloc storm. */
  int8_t *q_rm;
  float *q_scale;
  int32_t *q_colsum;
  /** Zeros. Attention registers no bias, but hexkl_weight_u8iX_register
      rejects a NULL one, so a zero vector is what "no bias" looks like. */
  float *q_bias;

  /** Forward scratch, allocated once. layer_run already mallocs per call
     (§1.8); adding a second storm per band would be worse. */
  float *s_band;   /**< max_blocks * M_band * T, block-major */
  float *o_band;   /**< M_band * head_dim; layer_run accumulates into it
                        directly (hexkl_mm_opts accumulate), so the staged
                        o_part buffer and its add loop no longer exist */
  float *l_row;    /**< M_band */
  float *sink_row; /**< M_band */
  float *q_gather; /**< M_band * head_dim, this band's Q rows */
  uint32_t *begin; /**< M_band */
  uint32_t *end;   /**< M_band */
  float **seg;     /**< max_blocks pointers into s_band */

  /** P's quant params come from bm below, NOT from a (1/255, 0) constant:
     max(p) == 1.0f holds PER ROW ACROSS THE BAND, but P is quantized PER
     BLOCK, and a block that does not hold its row's maximum has a local max
     below 1.0f (measured: 0x1.ff2d18p-9 on real softmax output). The scan
     adapted to that; bm reproduces the scan's answer exactly -- the softmax
     already took the running max of every value it stored -- so the scan
     itself is what gets skipped, not the adaptation. */
  float *bm;      /**< [max_blocks][M_band] per-(block, row) stored maxima */
  float *p_scale; /**< ROUND_UP(M_band, 64) entries, refilled per block */
  int32_t *p_zp;  /**< same length, permanently zero: p >= 0 pins rmin at 0 */

  /** Softmax row split. NULL runs PHASE B inline (decode-sized bands are
      cheaper than a fork/join). Not owned; the session owns the pool. */
  struct hvx_worker_pool_s *pool;
} hexkl_attn_u8_ctx;

/**
 * @brief Allocates the shadows and handle tables. Registers nothing yet.
 *
 * @param T  kv positions per block. Must be a multiple of 32
 *           (hexkl_mm_u8iX_plan enforces N % 32 == 0) and is recommended at
 *           256 so a (head, block) DMA slab stays out of the slow row-size
 *           regime (§1.8).
 * @return AEE_SUCCESS, or AEE_EBADPARM / AEE_ENOMEMORY.
 */
int hexkl_attn_u8_ctx_init(hexkl_attn_u8_ctx *ctx, uint8_t *vtcm_base,
                           uint32_t vtcm_size, uint32_t config_off,
                           uint32_t nch, uint32_t gqa, uint32_t head_dim,
                           uint32_t max_kv, uint32_t T, uint32_t M_band,
                           hexkl_w_width w_k, hexkl_w_width w_v, void *table_k,
                           void *table_v);

/** @brief Releases every registered block and frees the shadows. */
void hexkl_attn_u8_ctx_fini(hexkl_attn_u8_ctx *ctx);

/**
 * @brief Appends @a n_rows KV positions at @a kv_from and re-registers every
 *        block they touch.
 *
 * A block that a new token lands in is released and registered again, because
 * its per-N scales change. Blocks entirely before @a kv_from keep the handles
 * they already have -- that is the whole point of blocking the cache.
 *
 * @param k_rows_f16 [n_rows][nch][head_dim] fp16, post-RoPE
 * @param v_rows_f16 [n_rows][nch][head_dim] fp16
 */
int hexkl_attn_u8_kv_append(hexkl_attn_u8_ctx *ctx, uint32_t kv_from,
                            uint32_t n_rows, const uint16_t *k_rows_f16,
                            const uint16_t *v_rows_f16);

/**
 * @brief PHASE A: the S band for one kv_head, in ONE ops->run call.
 *
 * handles = that head's Kt blocks in kv order, so the cross-block weight
 * prefetch inside layer_run applies without anything being asked of it.
 *
 * @param M      rows in the band, n_query * gqa folded (doc08 §7)
 * @param q_band [M][head_dim] f32; layer_run quantizes it internally, once,
 *               and shares it across every block -- which is exactly "the gqa
 *               query heads of this kv_head share one K"
 * @param[out] out_s  BLOCK-MAJOR [n_blocks][M][T] f32, dequantized. Row m of
 *               block j is at out_s + j*M*T + m*T. Do not flatten it: that
 *               layout IS the KV tiling.
 *
 * 1/sqrt(head_dim) is deliberately NOT folded in here; it belongs to the
 * softmax's scale parameter, not to quantization metadata (§1.8).
 */
int hexkl_attn_u8_scores(hexkl_attn_u8_ctx *ctx, uint32_t head, uint32_t M,
                         const float *q_band, float *out_s);

/** @brief Slots in the stage_us array hexkl_attn_u8_forward can fill. */
enum {
  HEXKL_ATTN_T_QK = 0,  /**< PHASE A, the ops_k->run call */
  HEXKL_ATTN_T_SOFTMAX, /**< PHASE B */
  HEXKL_ATTN_T_PV,      /**< PHASE C, the ops_v->run calls */
  HEXKL_ATTN_T_ACCUM,   /**< PHASE C's f32 accumulate across blocks */
  HEXKL_ATTN_T_GATHER,  /**< Q row gather + the 1/l writeback */
  HEXKL_ATTN_T_TOTAL,   /**< whole call, DSP side only */
  HEXKL_ATTN_T_CALLS,   /**< layer_run calls made, not microseconds */
  /** The next five split the INSIDE of the layer_run calls (hexkl_probe.h).
   * They overlap T_QK/T_PV rather than adding to them: acc_read + acc_copy
   * is the "accumulator readout" the cost model charges 75-96% of the total
   * to, and which of the two dominates decides the Tier 1 fix. Scaffolding;
   * leaves with the rest of the debug surface (MHA_HTP_PLAN.md 9.5). */
  HEXKL_ATTN_T_ACC_READ,   /**< HMX acc -> VTCM result tile (vendor call) */
  HEXKL_ATTN_T_ACC_COPY,   /**< VTCM result tile -> DDR scratch (our call) */
  HEXKL_ATTN_T_DEQUANT,    /**< i32 -> f32, DDR -> DDR */
  HEXKL_ATTN_T_QUANT,      /**< activation f32 -> u8 AH tiles in VTCM */
  HEXKL_ATTN_T_DRAIN,      /**< DMA drains inside layer_run */
  HEXKL_ATTN_T_ACC_STRIDE, /**< not a time; see HEXKL_PROBE_ACC_STRIDE */
  HEXKL_ATTN_N_STAGES
};

/**
 * @brief The whole fused forward: PHASE A, B and C for every kv_head and band.
 *
 * ONE FastRPC entry point covers a layer. An unfused seam would pay the ~404 us
 * fixed FastRPC cost three times instead of once, which is why softmax runs on
 * the DSP at all (MHA_HTP_PLAN.md §2).
 *
 * Masking, for query row i at absolute position p = kv_from + i:
 *   end   = is_causal ? p + 1 : kv_len
 *   begin = (window && window < end) ? end - window : 0
 *
 * @param scale  folded into the softmax, 1/sqrt(head_dim); NOT into the
 *               quantization metadata
 * @param sink   NULL, or one logit per query head, [nch*gqa]
 * @param q      [n_query][nch*gqa*head_dim] f32, post-RoPE
 * @param[out] out same shape as @a q
 * @param[out] dbg_p  NULL, or max_blocks*M_band*T floats receiving the LAST
 *               band's unnormalized P -- debug only, stripped before the PR
 * @param[out] stage_us  NULL, or HEXKL_ATTN_N_STAGES entries receiving the
 *               per-stage microseconds. Reading the timer costs a few cycles
 *               per band, so passing NULL skips it entirely -- the production
 *               path pays nothing. MHA_HTP_PLAN.md §9.7 asks for this
 *               breakdown from day one because the last three times this
 *               project acted on a performance hypothesis without one, the
 *               hypothesis was wrong.
 */
int hexkl_attn_u8_forward(hexkl_attn_u8_ctx *ctx, uint32_t kv_from,
                          uint32_t n_query, float scale, int is_causal,
                          uint32_t window, const float *sink, const float *q,
                          float *out, float *dbg_p, float *dbg_l,
                          uint32_t *stage_us);

/** @brief Blocks currently carrying registered KV, i.e. ceil(kv_len / T). */
uint32_t hexkl_attn_u8_n_blocks(const hexkl_attn_u8_ctx *ctx);

#endif /* __NNTRAINER_HEXKL_ATTN_U8_H__ */
