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
static int moe_scratch_reserve(hexkl_moe_scratch *s, size_t bytes) {
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
static void *moe_carve(uint8_t **cur, size_t bytes) {
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
static uint32_t moe_push_weight_chunk(uint8_t *vtcm_base, uint32_t dst_off,
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
static uint32_t moe_push_act_block(uint8_t *vtcm_base, uint32_t act_off,
                                   const uint8_t *act_ah, uint32_t slot,
                                   uint32_t K, uint32_t k_tiles) {
  const uint32_t blk_bytes = k_tiles * HEXKL_HMX_ACTIVATION_ALIGNMENT;
  const uint32_t rs = moe_dma_row_size(blk_bytes);
  const uint32_t idx = hexkl_dma_ring_next_idx();
  hexkl_dma_ring_push2d(vtcm_base + act_off, act_ah + (size_t)slot * K, rs, rs,
                        rs, blk_bytes / rs, /*src_vtcm=*/0, /*dst_vtcm=*/1);
  return idx;
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
 */
static void moe_dma_copy(void *dst, const void *src, size_t bytes, int src_vtcm,
                         int dst_vtcm) {
  const uint32_t CHUNK = 1u << 20;
  size_t off = 0;
  while (off < bytes) {
    const uint32_t n = (bytes - off) > CHUNK ? CHUNK : (uint32_t)(bytes - off);
    const uint32_t rs = moe_dma_row_size(n);
    hexkl_dma_ring_push2d((uint8_t *)dst + off, (const uint8_t *)src + off, rs,
                          rs, rs, n / rs, src_vtcm, dst_vtcm);
    off += n;
  }
  hexkl_dma_ring_drain();
}

/**
 * @brief Times an n-tile loop's HMX issue by difference.
 *
 * hexkl_micro_hmx_acc_read_int32 and the tile dequant sit inside the same
 * loop and are already probed, so their running totals are snapshotted
 * across it and subtracted rather than probed again per tile: two clock
 * reads for a 112-tile loop instead of 224. What is left is acc_clear, the
 * k-tile mm calls and the loop itself -- exactly what the old mm residual
 * was meant to be, except a residual also absorbs everything unnamed, which
 * is how it moved 3.1 ms between two runs at the same block count with
 * nothing in between that touches it.
 *
 * Needs mm_t0 / mm_acc0 / mm_dq0 in scope; they are declared once per call
 * beside p0 so the two loops in a block do not redeclare them.
 */
#define MOE_MM_BEGIN()                                                         \
  do {                                                                         \
    if (hexkl_probe_on) {                                                      \
      mm_acc0 = hexkl_probe_us[HEXKL_PROBE_ACC_READ];                          \
      mm_dq0 = hexkl_probe_us[HEXKL_PROBE_DEQUANT];                            \
      mm_t0 = hexkl_probe_now();                                               \
    }                                                                          \
  } while (0)

#define MOE_MM_END()                                                           \
  do {                                                                         \
    if (hexkl_probe_on) {                                                      \
      hexkl_probe_us[HEXKL_PROBE_MM] +=                                        \
        (hexkl_probe_now() - mm_t0) -                                          \
        (hexkl_probe_us[HEXKL_PROBE_ACC_READ] - mm_acc0) -                     \
        (hexkl_probe_us[HEXKL_PROBE_DEQUANT] - mm_dq0);                        \
    }                                                                          \
  } while (0)

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

  /* Accumulator tiles are staged so the dequant can run over a run of them
     on the whole worker pool instead of one tile at a time on the calling
     thread (hvx_dequant_acc_tiles_to_f32 explains why that is worth doing).
     Whatever the arena has left after everything else decides how many fit;
     one still works, and is exactly the old behaviour. gate_up has the most
     n-tiles, so it sets the useful ceiling. */
  {
    uint32_t want = (2u * inter) / HEXKL_HMX_INT8_BLOCK_N_COL;
    uint32_t fits = 0u;
    if (arena_bytes > L.result_off) {
      fits = (arena_bytes - L.result_off) / ACC_TILE_BYTES;
    }
    if (fits == 0u) {
      return AEE_ENOMEMORY;
    }
    /* Capped, not maximised. This count is also the weight chunk size (see
       moe_push_weight_chunk): a bigger batch parallelises the dequant no
       better once it is well past the worker count, and a smaller chunk is
       what makes the matmul start sooner. 32 leaves gate_up in 4 chunks and
       down in 2, which is 194 ring descriptors a layer against the ring's
       256. */
    if (want > 32u) {
      want = 32u;
    }
    L.acc_tiles = (fits < want) ? fits : want;
  }
  L.total = L.result_off + L.acc_tiles * ACC_TILE_BYTES;

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
  const float *act_f32, float *out_f32, hvx_worker_pool *pool,
  hexkl_moe_scratch *scratch) {

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

  const uint32_t k_tiles = K / HEXKL_HMX_INT8_BLOCK_N_INNER;
  const uint32_t gu_ntiles = (2u * inter) / HEXKL_HMX_INT8_BLOCK_N_COL;
  const uint32_t inter_ktiles = inter / HEXKL_HMX_INT8_BLOCK_N_INNER;
  const uint32_t dn_ntiles = N_out / HEXKL_HMX_INT8_BLOCK_N_COL;
  const uint32_t BR = HEXKL_HMX_INT8_BLOCK_N_ROW;

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
  const size_t sz_scale = sizeof(float) * BR;
  const size_t sz_zp = sizeof(int32_t) * BR;
  const size_t sz_act_ah = n_slots_cap * K;
  const size_t sz_slot_u32 = sizeof(uint32_t) * n_slots_cap;
  const size_t sz_expert_u32 = sizeof(uint32_t) * n_experts;
  const size_t sz_mpad_u32 = sizeof(uint32_t) * m_pad;
  const size_t sz_act_c = sizeof(float) * (size_t)M * K;
  const size_t sz_out_c = sizeof(float) * (size_t)M * N_out;
  const size_t need = ROUND_UP_SZ(sz_scale, MOE_SCRATCH_ALIGN) +
                      ROUND_UP_SZ(sz_zp, MOE_SCRATCH_ALIGN) +
                      ROUND_UP_SZ(sz_act_ah, MOE_SCRATCH_ALIGN) +
                      3u * ROUND_UP_SZ(sz_slot_u32, MOE_SCRATCH_ALIGN) +
                      3u * ROUND_UP_SZ(sz_expert_u32, MOE_SCRATCH_ALIGN) +
                      2u * ROUND_UP_SZ(sz_mpad_u32, MOE_SCRATCH_ALIGN) +
                      ROUND_UP_SZ(sz_act_c, MOE_SCRATCH_ALIGN) +
                      ROUND_UP_SZ(sz_out_c, MOE_SCRATCH_ALIGN);
  uint64_t p_alloc = 0;
  HEXKL_PROBE_T0(p_alloc);
  rc = moe_scratch_reserve(scratch, need);
  HEXKL_PROBE_ADD(HEXKL_PROBE_ALLOC, p_alloc);
  if (rc != AEE_SUCCESS) {
    return rc;
  }
  uint8_t *cur = scratch->base;
  float *scale = (float *)moe_carve(&cur, sz_scale);
  int32_t *zp = (int32_t *)moe_carve(&cur, sz_zp);
  uint8_t *act_ah = (uint8_t *)moe_carve(&cur, sz_act_ah);
  uint32_t *slot_row = (uint32_t *)moe_carve(&cur, sz_slot_u32);
  float *slot_scale = (float *)moe_carve(&cur, sz_slot_u32);
  int32_t *slot_zp = (int32_t *)moe_carve(&cur, sz_slot_u32);
  uint32_t *slot_of = (uint32_t *)moe_carve(&cur, sz_expert_u32);
  uint32_t *order = (uint32_t *)moe_carve(&cur, sz_expert_u32);
  uint32_t *base_of = (uint32_t *)moe_carve(&cur, sz_expert_u32);
  float *scale_all = (float *)moe_carve(&cur, sz_mpad_u32);
  int32_t *zp_all = (int32_t *)moe_carve(&cur, sz_mpad_u32);
  float *act_c = (float *)moe_carve(&cur, sz_act_c);
  float *out_c = (float *)moe_carve(&cur, sz_out_c);
  uint32_t n_active = 0u;
  uint64_t p0 = 0;
  /* MOE_MM_BEGIN/END's state. See the macros above hexkl_mm_u8i4_moe_layout
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
  moe_dma_copy(act_c, act_f32, sizeof(float) * (size_t)M * K, 0, 0);
  memset(out_c, 0, sizeof(float) * (size_t)M * N_out);
  HEXKL_PROBE_ADD(HEXKL_PROBE_ACC_COPY, p0);

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

  /* Packed straight into slot order, so a block's 64 rows are already
     contiguous and the gather pass that used to follow is gone. */
  HEXKL_PROBE_ADD(HEXKL_PROBE_QUANT, p0);
  HEXKL_PROBE_T0(p0);
  rc = hvx_quant_pack_u8_ah_mapped(act_c, slot_row, n_slots, n_slots, K,
                                   slot_scale, slot_zp, act_ah, pool);
  HEXKL_PROBE_ADD(HEXKL_PROBE_QUANT, p0);
  if (rc != AEE_SUCCESS) {
    goto out;
  }
  if (n_active == 0u) {
    moe_dma_copy(out_f32, out_c, sizeof(float) * (size_t)M * N_out, 0, 0);
    rc = AEE_SUCCESS;
    goto out;
  }

  /* The first expert's first activation block goes out before its weights
     -- see moe_push_act_block for why the order matters. Every later
     block-0 is queued the same way at the point its predecessor's gate_up
     matmul finishes with the activation slot (below, next to the gate_up
     prefetch); blocks after the first of an expert are queued in place. */
  uint32_t act_idx =
    moe_push_act_block(vtcm_base, L.act_off, act_ah, slot_of[0], K, k_tiles);

  /* gate_up in chunks, so the first n-tile column is usable long before
     the last. MOE_MAX_CHUNKS bounds the array; acc_tiles caps at 32 and
     gate_up has 112 columns, so 4 is the real count. */
#define MOE_MAX_CHUNKS 16u
  uint32_t gu_idx[MOE_MAX_CHUNKS];
  uint32_t gu_nchunk = 0u;
  for (uint32_t nt0 = 0; nt0 < gu_ntiles; nt0 += L.acc_tiles) {
    const uint32_t cn =
      (gu_ntiles - nt0 < L.acc_tiles) ? (gu_ntiles - nt0) : L.acc_tiles;
    gu_idx[gu_nchunk++] = moe_push_weight_chunk(
      vtcm_base, L.w_gu_off, &tbl->slots[h_gate_up[order[0]]], k_tiles,
      gu_ntiles, nt0, cn);
  }

  for (uint32_t i = 0; i < n_active; ++i) {
    const uint32_t e = order[i];
    /* i == 0 waits on gate_up[order[0]] alone -- pushed just above with an
       empty ring -- so it times a 3.5 MiB DDR->VTCM transfer rather than a
       pipeline bubble. Recorded separately for that reason; it still counts
       toward DRAIN so the stage columns keep summing to the DSP total. */
    const int first_drain = (i == 0u);
    const hexkl_weight_u8i4 *g = &tbl->slots[h_gate_up[e]];
    const hexkl_weight_u8i4 *d = &tbl->slots[h_down[e]];
    const uint32_t *rows = row_index + base_of[i];
    const float *weights = row_weight + base_of[i];
    const uint32_t n_e = row_count[e];

    /* gate_up[e] was pushed either before this loop or by iteration i-1,
       where it had this expert's predecessor's down matmul to hide behind. */
    uint32_t dn_idx[MOE_MAX_CHUNKS];
    uint32_t dn_nchunk = 0u;

    for (uint32_t mb = 0; mb < n_e; mb += BR) {
      const uint32_t m_blk = (n_e - mb < BR) ? (n_e - mb) : BR;
      const int last_block = (mb + BR >= n_e);
      /* Counted, not timed -- see HEXKL_PROBE_BLOCKS. */
      HEXKL_PROBE_COUNT(HEXKL_PROBE_BLOCKS, 1);

      /* The pack already wrote this block in slot order, so its 64 rows are
         one contiguous run -- a single DMA, not a copy. Doing it on the
         core moved 5.5 MB a layer at 3.1 GB/s, against the 33 the engine
         measures, because the destination is VTCM and the core is the wrong
         thing to write it with. Rows past m_blk are padding whose
         accumulator output is never read.
         Block 0 is already in flight (queued ahead of this expert's
         gate_up); only the blocks after it are queued here, and by then
         nothing but this expert's own down is ahead of them in the ring. */
      HEXKL_PROBE_T0(p0);
      if (mb != 0u) {
        act_idx = moe_push_act_block(vtcm_base, L.act_off, act_ah,
                                     slot_of[i] + mb, K, k_tiles);
      }
      hexkl_dma_ring_wait(act_idx);
      for (uint32_t r = 0; r < m_blk; ++r) {
        scale[r] = slot_scale[slot_of[i] + mb + r];
        zp[r] = slot_zp[slot_of[i] + mb + r];
      }
      HEXKL_PROBE_ADD(HEXKL_PROBE_GATHER, p0);

      /* down[e] goes out now, behind the activation it must not delay and
         ahead of the whole gate_up matmul that covers it; in chunks for the
         same reason gate_up is: waiting for all 1.75 MB before the first
         n-tile column cost 55 us an expert. */
      if (mb == 0u) {
        for (uint32_t nt0 = 0; nt0 < dn_ntiles; nt0 += L.acc_tiles) {
          const uint32_t cn =
            (dn_ntiles - nt0 < L.acc_tiles) ? (dn_ntiles - nt0) : L.acc_tiles;
          dn_idx[dn_nchunk++] = moe_push_weight_chunk(
            vtcm_base, L.w_dn_off, d, inter_ktiles, dn_ntiles, nt0, cn);
        }
      }

      /* --- gate_up ------------------------------------------------- */
      /* Matmul a batch of n-tiles, staging each accumulator read into its
         own slot, then dequantize the whole batch on the pool. gate is
         gate_up's columns [0, inter) and up is [inter, 2*inter) -- the
         convention hvx_swiglu_inplace_f32's contract assumes -- and a
         batch can straddle that boundary, which is why the split is a
         parameter rather than two calls. */
      for (uint32_t nt0 = 0, ci = 0; nt0 < gu_ntiles;
           nt0 += L.acc_tiles, ++ci) {
        const uint32_t nb =
          (gu_ntiles - nt0 < L.acc_tiles) ? (gu_ntiles - nt0) : L.acc_tiles;
        /* Only the first block of an expert waits: by the second the whole
           weight is resident. i == 0 && ci == 0 is the one wait that cannot
           hide behind anything, which is what DMA_FIRST records. */
        if (mb == 0u) {
          HEXKL_PROBE_T0(p0);
          hexkl_dma_ring_wait(gu_idx[ci]);
          HEXKL_PROBE_ADD(HEXKL_PROBE_DRAIN, p0);
          if (first_drain && ci == 0u) {
            hexkl_probe_us[HEXKL_PROBE_DMA_FIRST] =
              hexkl_probe_us[HEXKL_PROBE_DRAIN];
            HEXKL_PROBE_COUNT(HEXKL_PROBE_DMA_FIRST_KB,
                              (nb * k_tiles * WEIGHT_TILE_BYTES_U8I4) >> 10);
          }
        }
        MOE_MM_BEGIN();
        for (uint32_t j = 0; j < nb; ++j) {
          hexkl_micro_hmx_acc_clear_int32();
          for (uint32_t kt = 0; kt < k_tiles; ++kt) {
            rc = hexkl_micro_hmx_mm_u8i4(
              vtcm_base, L.act_off + kt * HEXKL_HMX_ACTIVATION_ALIGNMENT,
              L.w_gu_off + (kt * gu_ntiles + nt0 + j) * WEIGHT_TILE_BYTES_U8I4);
            if (rc != AEE_SUCCESS) {
              goto out;
            }
          }
          HEXKL_PROBE_T0(p0);
          rc = hexkl_micro_hmx_acc_read_int32(
            vtcm_base, config_off, L.result_off + j * ACC_TILE_BYTES);
          HEXKL_PROBE_ADD(HEXKL_PROBE_ACC_READ, p0);
          if (rc != AEE_SUCCESS) {
            goto out;
          }
        }
        MOE_MM_END();

        HEXKL_PROBE_T0(p0);
        hvx_dequant_acc_tiles_to_f32(
          (const uint8_t *)((const int32_t *)(vtcm_base + L.result_off) +
                            acc->base),
          ACC_TILE_BYTES, nb, nt0, acc->row_stride, m_blk, scale, zp,
          g->colsum_w, g->w_scale, g->bias, (float *)(vtcm_base + L.gate_off),
          (float *)(vtcm_base + L.up_off), inter, inter, pool);
        HEXKL_PROBE_ADD(HEXKL_PROBE_DEQUANT, p0);
      }

      /* down[e]'s transfer is waited for here, after a whole gate_up
         matmul has run over it. Draining before the next push, not after,
         is what keeps this wait off the gate_up prefetch below -- the ring
         drains everything pending, so a push issued first would be waited
         for too and the pipeline would collapse into a serial chain. */

      /* A is dead once the last block's gate_up matmul is done, so the next
         expert's first activation block and then its gate_up go out now
         and ride under this block's SwiGLU, requantization and down matmul.
         Activation first: it is what the next expert waits on before
         anything else, and queued behind 3.5 MB of gate_up that wait would
         cover the gate_up too. */
      if (last_block && i + 1u < n_active) {
        act_idx = moe_push_act_block(vtcm_base, L.act_off, act_ah,
                                     slot_of[i + 1u], K, k_tiles);
        gu_nchunk = 0u;
        for (uint32_t nt0 = 0; nt0 < gu_ntiles; nt0 += L.acc_tiles) {
          const uint32_t cn =
            (gu_ntiles - nt0 < L.acc_tiles) ? (gu_ntiles - nt0) : L.acc_tiles;
          gu_idx[gu_nchunk++] = moe_push_weight_chunk(
            vtcm_base, L.w_gu_off, &tbl->slots[h_gate_up[order[i + 1u]]],
            k_tiles, gu_ntiles, nt0, cn);
        }
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
      HEXKL_PROBE_ADD(HEXKL_PROBE_REQUANT, p0);
      if (rc != AEE_SUCCESS) {
        goto out;
      }

      /* --- down ---------------------------------------------------- */
      /* Same batching as gate_up. One destination here, so the split is set
         past the last column and dst_b is never reached. */
      for (uint32_t nt0 = 0, ci = 0; nt0 < dn_ntiles;
           nt0 += L.acc_tiles, ++ci) {
        const uint32_t nb =
          (dn_ntiles - nt0 < L.acc_tiles) ? (dn_ntiles - nt0) : L.acc_tiles;
        if (mb == 0u) {
          HEXKL_PROBE_T0(p0);
          hexkl_dma_ring_wait(dn_idx[ci]);
          HEXKL_PROBE_ADD(HEXKL_PROBE_DRAIN_DN, p0);
        }
        MOE_MM_BEGIN();
        for (uint32_t j = 0; j < nb; ++j) {
          hexkl_micro_hmx_acc_clear_int32();
          for (uint32_t kt = 0; kt < inter_ktiles; ++kt) {
            rc = hexkl_micro_hmx_mm_u8i4(
              vtcm_base, L.mid_off + kt * HEXKL_HMX_ACTIVATION_ALIGNMENT,
              L.w_dn_off + (kt * dn_ntiles + nt0 + j) * WEIGHT_TILE_BYTES_U8I4);
            if (rc != AEE_SUCCESS) {
              goto out;
            }
          }
          HEXKL_PROBE_T0(p0);
          rc = hexkl_micro_hmx_acc_read_int32(
            vtcm_base, config_off, L.result_off + j * ACC_TILE_BYTES);
          HEXKL_PROBE_ADD(HEXKL_PROBE_ACC_READ, p0);
          if (rc != AEE_SUCCESS) {
            goto out;
          }
        }
        MOE_MM_END();

        HEXKL_PROBE_T0(p0);
        hvx_dequant_acc_tiles_to_f32(
          (const uint8_t *)((const int32_t *)(vtcm_base + L.result_off) +
                            acc->base),
          ACC_TILE_BYTES, nb, nt0, acc->row_stride, m_blk, scale, zp,
          d->colsum_w, d->w_scale, d->bias,
          (float *)(vtcm_base + L.res_f32_off), NULL, N_out, N_out, pool);
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
  moe_dma_copy(out_f32, out_c, sizeof(float) * (size_t)M * N_out, 0, 0);
  HEXKL_PROBE_ADD(HEXKL_PROBE_ACC_COPY, p0);

out:
  /* Nothing to free: the scratch stays with the session. */
  return rc;
}
