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
 * activation block behind it, down[i] behind the activation wait, the next
 * expert's activation and gate_up after the gate_up matmul). This header
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

  for (uint32_t i = 0; i < n_active; ++i) {
    PUT(NNTR_MOE_DMA_OP_WAIT, NNTR_MOE_DMA_SITE_ACT, i, 0u, 0u, 0u, 0u,
        act_push, 0u);
    uint32_t dn_n = 0u;
    for (uint32_t nt0 = 0, c = 0; nt0 < dn_ntiles; nt0 += acc_tiles, ++c) {
      const uint32_t cn =
        (dn_ntiles - nt0 < acc_tiles) ? (dn_ntiles - nt0) : acc_tiles;
      PUT(NNTR_MOE_DMA_OP_PUSH, NNTR_MOE_DMA_KIND_DOWN, i, c, cn * T,
          inter_ktiles, dn_ntiles * T, dn_off + nt0 * T, vtcm_dn + nt0 * T);
      dn_last[c] = n - 1u;
      dn_n = c + 1u;
    }
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
    for (uint32_t c = 0; c < dn_n; ++c) {
      PUT(NNTR_MOE_DMA_OP_WAIT, NNTR_MOE_DMA_SITE_DN, i, c, 0u, 0u, 0u,
          dn_last[c], 0u);
    }
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

#endif /* __NNTR_MOE_DMA_PLAN_H__ */
