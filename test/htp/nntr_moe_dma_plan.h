// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   nntr_moe_dma_plan.h
 * @date   21 Sep 2026
 * @brief  The MoE layer call's M=1 DMA descriptor list, restated on its own
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * hexkl_mm_u8i4_moe.c pushes its weights and activations through the ring
 * in an order fixed by the code (gate_up[0] ahead of the quantize, the
 * activation block behind it, the next expert's activation and gate_up
 * after the gate_up matmul, then down[i] once down[i-1]'s matmul has
 * issued its last read of the down buffer -- the down matmul runs one
 * block behind the gate_up, upstream PR #4327 5731b6e5). This header
 * writes that order down as data for one 64-row block per expert (M=1 is
 * the decode shape), so the host check can hold the kernel's push trace
 * against it and the dma_replay entry can drive the same list with no
 * compute in between. Header-only and free of Hexagon headers.
 *
 * Offsets: a push's src_off is relative to its expert's weight region
 * (gate_up at 0, down at NNTR_MOE_DMA_DOWN_OFF) or, for the activation and
 * copy pieces, to a small DDR scratch; dst_off is relative to a VTCM base
 * laid out gate_up, down, activation, copy scratch.
 */

#ifndef __NNTR_MOE_DMA_PLAN_H__
#define __NNTR_MOE_DMA_PLAN_H__

#include <stdint.h>

/** @brief 4 experts x 11 + 2 copies at the LFM2 shape; room for more. */
#define NNTR_MOE_DMA_PLAN_MAX 128u

/** @brief What one line of the plan is. Kinds and sites are
 *  hexkl_dma_trace.h's enums, restated so this header stands alone. */
enum {
  NNTR_MOE_DMA_OP_PUSH = 0,
  NNTR_MOE_DMA_OP_WAIT = 1,
  NNTR_MOE_DMA_OP_DRAIN = 2, /**< replay only (#100): wait for every push so
                                the next one dmstarts an idle engine */
};
enum {
  NNTR_MOE_DMA_KIND_ACT = 0,
  NNTR_MOE_DMA_KIND_GATE = 1,
  NNTR_MOE_DMA_KIND_UP = 2,
  NNTR_MOE_DMA_KIND_DOWN = 3,
  NNTR_MOE_DMA_KIND_COPY = 4,
};
enum {
  NNTR_MOE_DMA_SITE_ACT = 0,
  NNTR_MOE_DMA_SITE_GU = 1,
  NNTR_MOE_DMA_SITE_DN = 2,
  NNTR_MOE_DMA_SITE_COPY_IN = 3,
  NNTR_MOE_DMA_SITE_COPY_OUT = 4,
};

typedef struct {
  uint32_t op;     /**< NNTR_MOE_DMA_OP_* */
  uint32_t kind;   /**< push: KIND_*; wait: SITE_* */
  uint32_t expert; /**< expert ordinal (0 for the copies) */
  uint32_t chunk;
  uint32_t row_size;   /**< push only */
  uint32_t nrows;      /**< push only */
  uint32_t src_stride; /**< push only */
  uint32_t src_off;    /**< push: region-relative; wait: the push's index */
  uint32_t dst_off;    /**< push only, VTCM-base-relative */
  uint32_t flags;      /**< replay only: NNTR_MOE_DMA_DST_* (0 = default) */
} nntr_moe_dma_item;

#define NNTR_MOE_DMA_TILE_BYTES 512u
#define NNTR_MOE_DMA_ACT_ALIGN 2048u
#define NNTR_MOE_DMA_MAX_ROW 16384u

/** @brief Bytes of one expert's gate_up WH weight. */
static inline uint32_t nntr_moe_dma_gu_bytes(uint32_t K, uint32_t inter) {
  return (K / 32u) * (2u * inter / 32u) * NNTR_MOE_DMA_TILE_BYTES;
}
/** @brief Bytes of one expert's down WH weight. */
static inline uint32_t nntr_moe_dma_dn_bytes(uint32_t inter, uint32_t N_out) {
  return (inter / 32u) * (N_out / 32u) * NNTR_MOE_DMA_TILE_BYTES;
}
/** @brief Where down sits in an expert region: gate_up rounded to 4 KiB,
 *  the spacing HtpComputeOps::place uses. */
static inline uint32_t nntr_moe_dma_down_off(uint32_t K, uint32_t inter) {
  return (nntr_moe_dma_gu_bytes(K, inter) + 4095u) & ~4095u;
}
/** @brief Bytes of one expert region (gate_up + down, each 4 KiB-rounded). */
static inline uint32_t nntr_moe_dma_region_bytes(uint32_t K, uint32_t inter,
                                                 uint32_t N_out) {
  return nntr_moe_dma_down_off(K, inter) +
         ((nntr_moe_dma_dn_bytes(inter, N_out) + 4095u) & ~4095u);
}

/** @brief hexkl_mm_u8i4_moe.c's moe_dma_row_size. */
static inline uint32_t nntr_moe_dma_row_size(uint32_t total) {
  uint32_t rs = NNTR_MOE_DMA_MAX_ROW;
  while (rs > 1u && (total % rs) != 0u) {
    rs >>= 1;
  }
  return rs;
}

static inline uint32_t nntr_moe_dma_put_(nntr_moe_dma_item *out, uint32_t *n,
                                         uint32_t max, uint32_t op,
                                         uint32_t kind, uint32_t expert,
                                         uint32_t chunk, uint32_t row_size,
                                         uint32_t nrows, uint32_t src_stride,
                                         uint32_t src_off, uint32_t dst_off) {
  if (*n >= max) {
    return 0u;
  }
  nntr_moe_dma_item *it = &out[(*n)++];
  it->op = op;
  it->kind = kind;
  it->expert = expert;
  it->chunk = chunk;
  it->row_size = row_size;
  it->nrows = nrows;
  it->src_stride = src_stride;
  it->src_off = src_off;
  it->dst_off = dst_off;
  it->flags = 0u;
  return 1u;
}

/**
 * @brief The M=1 push/wait list for @a n_active experts.
 *
 * @param acc_tiles  hexkl_moe_layout::acc_tiles (32 at the LFM2 shape);
 *                   gate_up chunks carry acc_tiles / 2 pairs, down chunks
 *                   acc_tiles columns.
 * @param vtcm_gu / vtcm_dn / vtcm_act / vtcm_copy  destination offsets.
 * @return items written, 0 if @a max is too small or a size is not a
 *         tile multiple.
 */
static inline uint32_t
nntr_moe_dma_plan_m1(uint32_t K, uint32_t inter, uint32_t N_out,
                     uint32_t n_active, uint32_t acc_tiles, uint32_t vtcm_gu,
                     uint32_t vtcm_dn, uint32_t vtcm_act, uint32_t vtcm_copy,
                     nntr_moe_dma_item *out, uint32_t max) {
  if ((K % 32u) != 0u || (inter % 32u) != 0u || (N_out % 32u) != 0u ||
      n_active == 0u || acc_tiles < 2u ||
      (inter / 32u + acc_tiles / 2u - 1u) / (acc_tiles / 2u) > 16u ||
      (N_out / 32u + acc_tiles - 1u) / acc_tiles > 16u) {
    return 0u; /* the kernel's MOE_MAX_CHUNKS bound, restated */
  }
  const uint32_t T = NNTR_MOE_DMA_TILE_BYTES;
  const uint32_t k_tiles = K / 32u;
  const uint32_t gu_ntiles = 2u * inter / 32u;
  const uint32_t inter_ntiles = gu_ntiles / 2u;
  const uint32_t inter_ktiles = inter / 32u;
  const uint32_t dn_ntiles = N_out / 32u;
  const uint32_t half = acc_tiles / 2u;
  const uint32_t dn_off = nntr_moe_dma_down_off(K, inter);
  const uint32_t act_bytes = k_tiles * NNTR_MOE_DMA_ACT_ALIGN;
  const uint32_t act_rs = nntr_moe_dma_row_size(act_bytes);
  const uint32_t copy_in = 4u * K, copy_out = 4u * N_out; /* M = 1 floats */
  uint32_t n = 0u, ok = 1u;
  uint32_t gu_last[16], dn_last[16], act_push = 0u, gu_n = 0u;

#define PUT(...) ok &= nntr_moe_dma_put_(out, &n, max, __VA_ARGS__)
  /* moe_dma_copy(act_c, act_f32): pieces of at most 1 MiB, then drain. */
  {
    const uint32_t rs = nntr_moe_dma_row_size(copy_in);
    PUT(NNTR_MOE_DMA_OP_PUSH, NNTR_MOE_DMA_KIND_COPY, 0u, 0u, rs, copy_in / rs,
        rs, 0u, vtcm_copy);
    PUT(NNTR_MOE_DMA_OP_WAIT, NNTR_MOE_DMA_SITE_COPY_IN, 0u, 0u, 0u, 0u, 0u,
        n - 1u, 0u);
  }
  /* gate_up[0] in paired chunks, then the first activation block. */
  for (uint32_t g0 = 0, c = 0; g0 < inter_ntiles; g0 += half, ++c) {
    const uint32_t cn = (inter_ntiles - g0 < half) ? (inter_ntiles - g0) : half;
    PUT(NNTR_MOE_DMA_OP_PUSH, NNTR_MOE_DMA_KIND_GATE, 0u, c, cn * T, k_tiles,
        gu_ntiles * T, g0 * T, vtcm_gu + g0 * T);
    PUT(NNTR_MOE_DMA_OP_PUSH, NNTR_MOE_DMA_KIND_UP, 0u, c, cn * T, k_tiles,
        gu_ntiles * T, (inter_ntiles + g0) * T,
        vtcm_gu + (inter_ntiles + g0) * T);
    gu_last[c] = n - 1u;
    gu_n = c + 1u;
  }
  PUT(NNTR_MOE_DMA_OP_PUSH, NNTR_MOE_DMA_KIND_ACT, 0u, 0u, act_rs,
      act_bytes / act_rs, act_rs, 0u, vtcm_act);
  act_push = n - 1u;

  /* Block i (one per expert at M=1): the activation wait, GU(i)'s chunk
     waits, the next block's activation and gate_up, DN(i-1)'s chunk waits,
     then down[i] into the buffer DN(i-1) has just finished reading. */
  uint32_t dn_n = 0u;
  for (uint32_t i = 0; i < n_active; ++i) {
    PUT(NNTR_MOE_DMA_OP_WAIT, NNTR_MOE_DMA_SITE_ACT, i, 0u, 0u, 0u, 0u,
        act_push, 0u);
    for (uint32_t c = 0; c < gu_n; ++c) {
      PUT(NNTR_MOE_DMA_OP_WAIT, NNTR_MOE_DMA_SITE_GU, i, c, 0u, 0u, 0u,
          gu_last[c], 0u);
    }
    if (i + 1u < n_active) {
      PUT(NNTR_MOE_DMA_OP_PUSH, NNTR_MOE_DMA_KIND_ACT, i + 1u, 0u, act_rs,
          act_bytes / act_rs, act_rs, 0u, vtcm_act);
      act_push = n - 1u;
      for (uint32_t g0 = 0, c = 0; g0 < inter_ntiles; g0 += half, ++c) {
        const uint32_t cn =
          (inter_ntiles - g0 < half) ? (inter_ntiles - g0) : half;
        PUT(NNTR_MOE_DMA_OP_PUSH, NNTR_MOE_DMA_KIND_GATE, i + 1u, c, cn * T,
            k_tiles, gu_ntiles * T, g0 * T, vtcm_gu + g0 * T);
        PUT(NNTR_MOE_DMA_OP_PUSH, NNTR_MOE_DMA_KIND_UP, i + 1u, c, cn * T,
            k_tiles, gu_ntiles * T, (inter_ntiles + g0) * T,
            vtcm_gu + (inter_ntiles + g0) * T);
        gu_last[c] = n - 1u;
      }
    }
    for (uint32_t c = 0; i != 0u && c < dn_n; ++c) {
      PUT(NNTR_MOE_DMA_OP_WAIT, NNTR_MOE_DMA_SITE_DN, i - 1u, c, 0u, 0u, 0u,
          dn_last[c], 0u);
    }
    dn_n = 0u;
    for (uint32_t nt0 = 0, c = 0; nt0 < dn_ntiles; nt0 += acc_tiles, ++c) {
      const uint32_t cn =
        (dn_ntiles - nt0 < acc_tiles) ? (dn_ntiles - nt0) : acc_tiles;
      PUT(NNTR_MOE_DMA_OP_PUSH, NNTR_MOE_DMA_KIND_DOWN, i, c, cn * T,
          inter_ktiles, dn_ntiles * T, dn_off + nt0 * T, vtcm_dn + nt0 * T);
      dn_last[c] = n - 1u;
      dn_n = c + 1u;
    }
  }
  /* The last block's down matmul, after the loop. */
  for (uint32_t c = 0; c < dn_n; ++c) {
    PUT(NNTR_MOE_DMA_OP_WAIT, NNTR_MOE_DMA_SITE_DN, n_active - 1u, c, 0u, 0u,
        0u, dn_last[c], 0u);
  }
  {
    const uint32_t rs = nntr_moe_dma_row_size(copy_out);
    PUT(NNTR_MOE_DMA_OP_PUSH, NNTR_MOE_DMA_KIND_COPY, 0u, 0u, rs, copy_out / rs,
        rs, 0u, vtcm_copy);
    PUT(NNTR_MOE_DMA_OP_WAIT, NNTR_MOE_DMA_SITE_COPY_OUT, 0u, 0u, 0u, 0u, 0u,
        n - 1u, 0u);
  }
#undef PUT
  return ok ? n : 0u;
}

/** ======================================================================
 * [#100] Replay cells: synthetic lists that separate the DMA probe's rate
 * from the chunk list's, and the GEMV feed candidates. Replay only; the
 * kernel never sees these (docs/plans/100-dma-chunk-list.md section 3.2).
 * ====================================================================== */

/** @brief Destination mode of one replay push (item flags). DST_PACKED
 *  lands rows back to back (dst_stride = row_size), DST_STRIDED keeps the
 *  source stride (the kernel's VTCM layout). Neither set = the replay's
 *  default, which is packed in this tree; #99 makes it strided, and it
 *  changes nntr_moe_dma_dst_stride with the skel. */
#define NNTR_MOE_DMA_DST_PACKED 1u
#define NNTR_MOE_DMA_DST_STRIDED 2u

/** @brief Largest payload of one replay descriptor: a whole gate_up matrix
 *  (57344 x 64 = 3.5 MiB, cell f2) fits; the struct's row_size is 24 bits
 *  and nrows 16 (hexkl_dma_ring.h). */
#define NNTR_MOE_DMA_REPLAY_MAX_DESC_BYTES (4u << 20)

/** @brief Schedule word 0 of an item: flags << 16 | op << 8 | kind. */
static inline uint32_t nntr_moe_dma_word0(const nntr_moe_dma_item *it) {
  return (it->flags << 16) | (it->op << 8) | it->kind;
}

/** @brief VTCM row pitch of a push under its flags. */
static inline uint32_t nntr_moe_dma_dst_stride(uint32_t flags,
                                               uint32_t row_size,
                                               uint32_t src_stride) {
  return (flags & NNTR_MOE_DMA_DST_STRIDED) ? src_stride : row_size;
}

/** @brief Bytes of VTCM a push spans from its dst_off. */
static inline uint32_t nntr_moe_dma_extent(uint32_t flags, uint32_t row_size,
                                           uint32_t nrows,
                                           uint32_t src_stride) {
  return (nrows - 1u) * nntr_moe_dma_dst_stride(flags, row_size, src_stride) +
         row_size;
}

/**
 * @brief Rewrites a list: drops every wait and drain, keeps the pushes whose
 *        kind bit is in @a kind_mask, ORs @a or_flags into them and, when
 *        @a after is NNTR_MOE_DMA_OP_WAIT or _DRAIN, puts one of those
 *        right behind every push (NNTR_MOE_DMA_OP_PUSH = nothing behind).
 * @return items written, 0 if @a max is too small.
 */
static inline uint32_t nntr_moe_dma_rewrite(const nntr_moe_dma_item *in,
                                            uint32_t n_in, uint32_t kind_mask,
                                            uint32_t after, uint32_t or_flags,
                                            nntr_moe_dma_item *out,
                                            uint32_t max) {
  uint32_t n = 0u;
  for (uint32_t k = 0; k < n_in; ++k) {
    if (in[k].op != NNTR_MOE_DMA_OP_PUSH || !((kind_mask >> in[k].kind) & 1u)) {
      continue;
    }
    if (n + (after != NNTR_MOE_DMA_OP_PUSH ? 2u : 1u) > max) {
      return 0u;
    }
    out[n] = in[k];
    out[n].flags |= or_flags;
    ++n;
    if (after != NNTR_MOE_DMA_OP_PUSH) {
      nntr_moe_dma_put_(out, &n, max, after, NNTR_MOE_DMA_SITE_GU, in[k].expert,
                        0u, 0u, 0u, 0u, n - 1u, 0u);
    }
  }
  return n;
}

/**
 * @brief Depth-2 issue order over a push list: push 0, then for every k
 *        push k + 1 and wait k. At most two descriptors are outstanding,
 *        so a destination is free again one push after its wait -- the
 *        discipline of hvx_impl's mm_pf_kick.
 * @return items written (2 x @a n_push), 0 if @a max is too small.
 */
static inline uint32_t nntr_moe_dma_depth2(const nntr_moe_dma_item *push,
                                           uint32_t n_push,
                                           nntr_moe_dma_item *out,
                                           uint32_t max) {
  uint32_t n = 0u, pos_prev = 0u;
  if (n_push == 0u || 2u * n_push > max) {
    return 0u;
  }
  out[n++] = push[0];
  for (uint32_t k = 0; k < n_push; ++k) {
    const uint32_t pos_k = (k == 0u) ? 0u : pos_prev;
    if (k + 1u < n_push) {
      pos_prev = n;
      out[n++] = push[k + 1u];
    }
    nntr_moe_dma_put_(out, &n, max, NNTR_MOE_DMA_OP_WAIT, NNTR_MOE_DMA_SITE_GU,
                      push[k].expert, 0u, 0u, 0u, 0u, pos_k, 0u);
  }
  return n;
}

/**
 * @brief Pushes only: per expert, the gate_up matrix as columns of
 *        @a gu_row bytes x K/32 rows at the matrix stride and, if @a dn_row
 *        is not 0, the down matrix as columns of @a dn_row bytes x inter/32
 *        rows. DST_STRIDED lands each column at its own offset inside a
 *        gate_up buffer at 0 and a down buffer at nntr_moe_dma_gu_bytes
 *        (the kernel's layout; a column as wide as the matrix makes the
 *        push one contiguous copy). DST_PACKED lands the gate_up columns in
 *        @a slots rotating payload-sized slots from 0 (probe iii's plan);
 *        down is not allowed then.
 * @return items written, 0 on a bad argument or a full @a out.
 */
static inline uint32_t nntr_moe_dma_columns(uint32_t K, uint32_t inter,
                                            uint32_t N_out, uint32_t n_experts,
                                            uint32_t gu_row, uint32_t dn_row,
                                            uint32_t flags, uint32_t slots,
                                            nntr_moe_dma_item *out,
                                            uint32_t max) {
  const uint32_t T = NNTR_MOE_DMA_TILE_BYTES;
  const uint32_t gu_stride = (2u * inter / 32u) * T,
                 dn_stride = (N_out / 32u) * T;
  const uint32_t k_tiles = K / 32u, inter_ktiles = inter / 32u;
  const uint32_t dn_off = nntr_moe_dma_down_off(K, inter);
  const uint32_t v_dn = nntr_moe_dma_gu_bytes(K, inter);
  const int packed = flags == NNTR_MOE_DMA_DST_PACKED;
  uint32_t n = 0u, ok = 1u;
  if (gu_row == 0u || (gu_stride % gu_row) != 0u ||
      (dn_row != 0u && (packed || (dn_stride % dn_row) != 0u)) ||
      (!packed && flags != NNTR_MOE_DMA_DST_STRIDED) ||
      (packed && slots == 0u)) {
    return 0u;
  }
  for (uint32_t e = 0; e < n_experts; ++e) {
    for (uint32_t c = 0; c < gu_stride / gu_row; ++c) {
      const uint32_t dst = packed ? (n % slots) * gu_row * k_tiles : c * gu_row;
      ok &= nntr_moe_dma_put_(out, &n, max, NNTR_MOE_DMA_OP_PUSH,
                              NNTR_MOE_DMA_KIND_GATE, e, c, gu_row, k_tiles,
                              gu_stride, c * gu_row, dst);
    }
    for (uint32_t c = 0; dn_row != 0u && c < dn_stride / dn_row; ++c) {
      ok &=
        nntr_moe_dma_put_(out, &n, max, NNTR_MOE_DMA_OP_PUSH,
                          NNTR_MOE_DMA_KIND_DOWN, e, c, dn_row, inter_ktiles,
                          dn_stride, dn_off + c * dn_row, v_dn + c * dn_row);
    }
  }
  for (uint32_t k = 0; ok && k < n; ++k) {
    out[k].flags = flags;
  }
  return ok ? n : 0u;
}

/**
 * @brief Pushes only: each expert region read front to back (gate_up, then
 *        down) in contiguous pieces of @a piece bytes (the last one what is
 *        left), rows of @a row bytes, alternating between two piece-sized
 *        VTCM slots at 0 -- hvx_impl's mm_pf_kick shape.
 */
static inline uint32_t nntr_moe_dma_linear(uint32_t K, uint32_t inter,
                                           uint32_t N_out, uint32_t n_experts,
                                           uint32_t piece, uint32_t row,
                                           nntr_moe_dma_item *out,
                                           uint32_t max) {
  const uint32_t region = nntr_moe_dma_region_bytes(K, inter, N_out);
  uint32_t n = 0u, ok = 1u;
  if (row == 0u || (piece % row) != 0u || (region % row) != 0u) {
    return 0u;
  }
  for (uint32_t e = 0; e < n_experts; ++e) {
    for (uint32_t off = 0; off < region; off += piece) {
      const uint32_t b = region - off < piece ? region - off : piece;
      /* kind only picks the weight region in the replay */
      ok &= nntr_moe_dma_put_(out, &n, max, NNTR_MOE_DMA_OP_PUSH,
                              NNTR_MOE_DMA_KIND_GATE, e, off / piece, row,
                              b / row, row, off, (n % 2u) * piece);
    }
  }
  for (uint32_t k = 0; ok && k < n; ++k) {
    out[k].flags = NNTR_MOE_DMA_DST_PACKED;
  }
  return ok ? n : 0u;
}

/** @brief Number of #100 replay cells nntr_moe_dma_cell builds. */
#define NNTR_MOE_DMA_N_CELLS 16u

/**
 * @brief Replay cell @a id of plan 100 section 3.2, in its fixed order.
 *
 * @param traced   the kernel's M=1 list (nntr_moe_dma_plan_m1) at the
 *                 gtest's VTCM layout; the traced* cells are derived from it
 * @param name / fresh / load   the cell's id and the replay arguments it
 *                 runs with (workers 1, pace 0, gap 0 for every cell)
 * @return items written, 0 for an unknown id or a list that does not fit.
 */
static inline uint32_t
nntr_moe_dma_cell(uint32_t id, const nntr_moe_dma_item *traced,
                  uint32_t n_traced, uint32_t K, uint32_t inter, uint32_t N_out,
                  uint32_t n_experts, nntr_moe_dma_item *out, uint32_t max,
                  const char **name, uint32_t *fresh, uint32_t *load) {
  const uint32_t all = 0x1fu,
                 gu = (1u << NNTR_MOE_DMA_KIND_GATE) |
                      (1u << NNTR_MOE_DMA_KIND_UP),
                 dn = 1u << NNTR_MOE_DMA_KIND_DOWN;
  const uint32_t T = NNTR_MOE_DMA_TILE_BYTES;
  const uint32_t iii_row = 8192u, dn_col = 16384u, iii_slots = 8u;
  nntr_moe_dma_item tmp[NNTR_MOE_DMA_PLAN_MAX];
  uint32_t n_tmp = 0u;
  *fresh = 0u;
  *load = 0u;
  switch (id) {
  case 0:
  case 1:
    *name = id == 0u ? "traced" : "traced_f";
    *fresh = id;
    if (n_traced > max) {
      return 0u;
    }
    for (uint32_t k = 0; k < n_traced; ++k) {
      out[k] = traced[k];
    }
    return n_traced;
  case 2:
    *name = "traced_nowait";
    return nntr_moe_dma_rewrite(traced, n_traced, all, NNTR_MOE_DMA_OP_PUSH, 0u,
                                out, max);
  case 3:
    *name = "traced_gu";
    return nntr_moe_dma_rewrite(traced, n_traced, gu, NNTR_MOE_DMA_OP_PUSH, 0u,
                                out, max);
  case 4:
    *name = "traced_dn";
    return nntr_moe_dma_rewrite(traced, n_traced, dn, NNTR_MOE_DMA_OP_PUSH, 0u,
                                out, max);
  case 5:
  case 6:
  case 7:
  case 9: {
    static const char *const nm[] = {"iii_chain", "iii_dmstart", "iii_link1",
                                     "", "c_star"};
    static const uint32_t after[] = {
      NNTR_MOE_DMA_OP_PUSH, NNTR_MOE_DMA_OP_DRAIN, NNTR_MOE_DMA_OP_WAIT, 0u,
      NNTR_MOE_DMA_OP_DRAIN};
    *name = nm[id - 5u];
    *fresh = id == 9u;
    n_tmp = nntr_moe_dma_columns(K, inter, N_out, n_experts, iii_row, 0u,
                                 NNTR_MOE_DMA_DST_PACKED, iii_slots, tmp,
                                 NNTR_MOE_DMA_PLAN_MAX);
    return n_tmp ? nntr_moe_dma_rewrite(tmp, n_tmp, all, after[id - 5u], 0u,
                                        out, max)
                 : 0u;
  }
  case 8:
    *name = "iii_strided";
    return nntr_moe_dma_columns(K, inter, N_out, n_experts, iii_row, 0u,
                                NNTR_MOE_DMA_DST_STRIDED, 0u, out, max);
  case 10:
  case 11:
    *name = id == 10u ? "f1" : "f1_load";
    n_tmp = nntr_moe_dma_linear(K, inter, N_out, n_experts, 1u << 20, 16384u,
                                tmp, NNTR_MOE_DMA_PLAN_MAX);
    break;
  case 12:
  case 13:
    *name = id == 12u ? "f2" : "f2_load";
    n_tmp = nntr_moe_dma_columns(
      K, inter, N_out, n_experts, (2u * inter / 32u) * T, (N_out / 32u) * T,
      NNTR_MOE_DMA_DST_STRIDED, 0u, tmp, NNTR_MOE_DMA_PLAN_MAX);
    break;
  case 14:
  case 15:
    *name = id == 14u ? "f3" : "f3_load";
    n_tmp = nntr_moe_dma_columns(K, inter, N_out, n_experts, iii_row, dn_col,
                                 NNTR_MOE_DMA_DST_STRIDED, 0u, tmp,
                                 NNTR_MOE_DMA_PLAN_MAX);
    break;
  default:
    return 0u;
  }
  /* the feed cells: load 0 and 2 (HVX streaming VTCM), depth 2 */
  *load = (id & 1u) ? 2u : 0u;
  return n_tmp ? nntr_moe_dma_depth2(tmp, n_tmp, out, max) : 0u;
}

/**
 * @brief The fill of the gtest's arena chunks at chunk offset @a off. At
 *        every 64-byte sample the byte is 0xA5 (so the old coverage sums,
 *        dma_probe's and the replay's res[6], stay shape-independent); 32
 *        bytes further on it is a tag hashed from the 4 KiB page, so two
 *        regions' pages differ and the replay's res[12] names which source
 *        the last transfer into each sample came from.
 */
static inline uint8_t nntr_dma_pattern(uint32_t off) {
  if ((off & 63u) == 32u) {
    return (uint8_t)(((off >> 12) * 2654435761u) >> 24);
  }
  return (uint8_t)(0xA5u ^ (off & 63u));
}

/** @brief [lo, hi): the VTCM every push of the list spans; lo = hi = 0 if
 *  there is no push. The replay zeroes it once and sums its tag bytes. */
static inline void nntr_moe_dma_window(const nntr_moe_dma_item *it, uint32_t n,
                                       uint32_t *lo, uint32_t *hi) {
  uint32_t l = UINT32_MAX, h = 0u;
  for (uint32_t k = 0; k < n; ++k) {
    if (it[k].op != NNTR_MOE_DMA_OP_PUSH) {
      continue;
    }
    const uint32_t e =
      it[k].dst_off + nntr_moe_dma_extent(it[k].flags, it[k].row_size,
                                          it[k].nrows, it[k].src_stride);
    l = it[k].dst_off < l ? it[k].dst_off : l;
    h = e > h ? e : h;
  }
  *lo = h ? l : 0u;
  *hi = h;
}

/**
 * @brief What the replay's res[12] must read: the sum of the VTCM bytes at
 *        lo + 32 + 64 j < hi after @a calls calls of the list on a window
 *        zeroed once, every transfer landing in list order (one chain,
 *        workers = 1). Sources as the skel's replay_src: weights from region
 *        (fresh ? call x E + expert : expert) mod @a n_regions, activation
 *        and copy pieces from the scratch slot after the regions.
 *
 * @param samples  scratch of at least (hi - lo) / 64 + 1 bytes
 */
static inline uint32_t nntr_moe_dma_tag_sum(const nntr_moe_dma_item *it,
                                            uint32_t n, uint32_t calls,
                                            uint32_t fresh, uint32_t n_regions,
                                            uint32_t region_bytes,
                                            uint8_t *samples) {
  uint32_t lo, hi, n_experts = 0u, sum = 0u;
  nntr_moe_dma_window(it, n, &lo, &hi);
  if (hi == 0u || n_regions == 0u) {
    return 0u;
  }
  const uint32_t n_samples = (hi - lo + 31u) / 64u;
  for (uint32_t j = 0; j < n_samples; ++j) {
    samples[j] = 0u;
  }
  for (uint32_t k = 0; k < n; ++k) {
    if (it[k].op == NNTR_MOE_DMA_OP_PUSH &&
        it[k].kind != NNTR_MOE_DMA_KIND_ACT &&
        it[k].kind != NNTR_MOE_DMA_KIND_COPY && it[k].expert + 1u > n_experts) {
      n_experts = it[k].expert + 1u;
    }
  }
  for (uint32_t call = 0; call < calls; ++call) {
    for (uint32_t k = 0; k < n; ++k) {
      const nntr_moe_dma_item *p = &it[k];
      if (p->op != NNTR_MOE_DMA_OP_PUSH) {
        continue;
      }
      const int scratch =
        p->kind == NNTR_MOE_DMA_KIND_ACT || p->kind == NNTR_MOE_DMA_KIND_COPY;
      const uint32_t region =
        scratch
          ? n_regions
          : (fresh ? (call * n_experts + p->expert) % n_regions : p->expert);
      const uint32_t src = region * region_bytes + p->src_off;
      const uint32_t ds =
        nntr_moe_dma_dst_stride(p->flags, p->row_size, p->src_stride);
      for (uint32_t r = 0; r < p->nrows; ++r) {
        const uint32_t d0 = p->dst_off + r * ds, s0 = src + r * p->src_stride;
        const uint32_t q = d0 - lo;
        for (uint32_t j = q <= 32u ? 0u : (q - 32u + 63u) / 64u;
             j < n_samples && lo + 32u + 64u * j < d0 + p->row_size; ++j) {
          samples[j] = nntr_dma_pattern(s0 + (lo + 32u + 64u * j - d0));
        }
      }
    }
  }
  for (uint32_t j = 0; j < n_samples; ++j) {
    sum += samples[j];
  }
  return sum;
}

#endif /* __NNTR_MOE_DMA_PLAN_H__ */
