// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file   hexkl_attn_f16_plan.h
 * @date   14 Sep 2026
 * @brief  VTCM layout and causal block geometry for fp16 flash attention
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Everything in this header is plain arithmetic with no Hexagon dependency,
 * so the host unit test (test/unittest/unittest_hexkl_attn_f16_plan.cpp)
 * can pin it down before a device is involved. The kernel in
 * hexkl_attn_f16.c consumes the same functions, so what the host proves
 * about region overlap and mask geometry is what runs on the DSP.
 *
 * Terminology follows llama.cpp's flash-attn-ops: Br query rows and Bc
 * cache rows per block; G = n_head_q / n_head_kv query heads share one KV
 * head, and they are packed into the tile row dimension as row = q*G + g,
 * so one Q.K^T per KV head covers all of its query heads and no K or V is
 * ever replicated. g_br = align_up(G*Br, 32) is the row count HMX sees.
 */

#ifndef __NNTRAINER_HEXKL_ATTN_F16_PLAN_H__
#define __NNTRAINER_HEXKL_ATTN_F16_PLAN_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Return codes; the Hexagon caller maps them onto AEE_* values so
 *         this header stays buildable on the host. */
enum {
  HEXKL_ATTN_OK = 0,
  HEXKL_ATTN_EBADPARM = -1,
  HEXKL_ATTN_ENOMEM = -2,
};

/** @brief One 32x32 fp16 tile in bytes; also the activation alignment. */
#define HEXKL_ATTN_TILE_BYTES 2048u
/** @brief Rows or columns per tile. */
#define HEXKL_ATTN_TILE 32u
/** @brief HMX config region alignment (HEXKL_HMX_CONFIG_ALIGNMENT). */
#define HEXKL_ATTN_CFG_ALIGN 256u

/**
 * @brief The attention problem for one (batch, layer) step.
 *
 * Query row q sits at absolute cache position cache_from + q and may see
 * cache rows [max(0, pos + 1 - window), pos]. cache_to is the number of
 * valid cache rows and must be at least cache_from + n_q, because the
 * caller (MHACoreLayer) has already appended this step's K/V before the
 * attention runs.
 */
typedef struct {
  uint32_t n_q;        /**< query rows this step (1 for decode) */
  uint32_t cache_from; /**< absolute position of query row 0 */
  uint32_t cache_to;   /**< valid cache rows, > every query position */
  uint32_t n_head_q;
  uint32_t n_head_kv;
  uint32_t head_dim; /**< multiple of 32, at most 256 */
  uint32_t window;   /**< sliding window length; 0 means unlimited */
  float softcap;     /**< > 0: logits become tanh(s/softcap)*softcap
                          (MHACoreLayer's attn_logit_softcapping); 0 off */
} hexkl_attn_f16_shape;

/** @brief Block sizes. br is in query rows; bc in cache rows. */
typedef struct {
  uint32_t br;   /**< query rows per block; G*br is padded up to 32 */
  uint32_t bc;   /**< cache rows per block, multiple of 32 */
  uint32_t g;    /**< n_head_q / n_head_kv, derived */
  uint32_t g_br; /**< align_up(g*br, 32): tile rows HMX actually sees */
} hexkl_attn_f16_tiling;

/**
 * @brief Byte offsets from vtcm_base for every region one call uses.
 *
 * Regions are all multiples of HEXKL_ATTN_TILE_BYTES and start on that
 * alignment, which satisfies the activation (2048) and weight (128) rules
 * at once. Double-buffered regions are indexed [0]/[1]. The f16 HMX config
 * sits at the top, 256-aligned, and is written by setup_acc_read_f16 on
 * every call (see the probe entry for why it cannot be permanent).
 */
typedef struct {
  uint32_t q_ah;        /**< [g_br/32][hd/32] Q tiles, scaled f16 */
  uint32_t kt_wh[2];    /**< [hd/32][bc/32] K^T weight tiles */
  uint32_t v_wh[2];     /**< [bc/32][hd/32] V weight tiles */
  uint32_t s_ah[2];     /**< [g_br/32][bc/32] scores, then P in place;
                             two so HMX can write block k+1's scores while
                             HVX turns block k's into probabilities */
  uint32_t d_ah;        /**< [g_br/32] diagonal rescale tiles */
  uint32_t o_wh[2];     /**< [g_br/32][hd/32] O ping-pong; read back as AH,
                             fed back as the weight of D.O_prev (AH == WH
                             for fp16). 1/l is applied in the HVX store, so
                             there is no diag(1/l) pass or region. */
  uint32_t k_land[2];   /**< [bc][hd] f16 raw K rows landed by DMA */
  uint32_t v_land[2];   /**< [bc][hd] f16 raw V rows landed by DMA */
  uint32_t ml;          /**< 2*g_br f32: running max, then running sum */
  uint32_t cfg_f16;     /**< HMX fp16 accumulator config */
  uint32_t total;       /**< first byte past the last region */
  uint32_t n_row_tiles; /**< g_br/32 */
  uint32_t n_col_tiles; /**< bc/32 */
  uint32_t n_dot_tiles; /**< hd/32 */
} hexkl_attn_f16_layout;

static inline uint32_t hexkl_attn_round_up(uint32_t v, uint32_t a) {
  return ((v + a - 1u) / a) * a;
}

/**
 * @brief Validates the shape and derives G and g_br into @a t.
 *
 * br and bc must already be set by the caller (the Phase 3 chooser or a
 * test); this only checks them and fills the derived fields.
 */
static inline int hexkl_attn_f16_tiling_init(const hexkl_attn_f16_shape *s,
                                             hexkl_attn_f16_tiling *t) {
  if (!s || !t || s->n_q == 0 || s->n_head_q == 0 || s->n_head_kv == 0 ||
      s->head_dim == 0 || s->head_dim > 256u ||
      (s->head_dim % HEXKL_ATTN_TILE) != 0 ||
      (s->n_head_q % s->n_head_kv) != 0 ||
      s->cache_to < s->cache_from + s->n_q) {
    return HEXKL_ATTN_EBADPARM;
  }
  if (t->br == 0 || t->bc == 0 || (t->bc % HEXKL_ATTN_TILE) != 0) {
    return HEXKL_ATTN_EBADPARM;
  }
  t->g = s->n_head_q / s->n_head_kv;
  t->g_br = hexkl_attn_round_up(t->g * t->br, HEXKL_ATTN_TILE);
  return HEXKL_ATTN_OK;
}

/**
 * @brief Lays the regions out and checks them against the arena.
 *
 * @param arena_top   first byte NOT available (the session's int32 config
 *                    offset on the DSP; anything on the host)
 * @param cfg_size    hexkl_micro_hmx_config_size() on the DSP
 */
static inline int hexkl_attn_f16_plan(const hexkl_attn_f16_shape *s,
                                      const hexkl_attn_f16_tiling *t,
                                      uint32_t arena_top, uint32_t cfg_size,
                                      hexkl_attn_f16_layout *L) {
  if (!s || !t || !L || t->g_br == 0 || (t->g_br % HEXKL_ATTN_TILE) != 0) {
    return HEXKL_ATTN_EBADPARM;
  }
  const uint32_t TB = HEXKL_ATTN_TILE_BYTES;
  const uint32_t rt = t->g_br / HEXKL_ATTN_TILE;
  const uint32_t ct = t->bc / HEXKL_ATTN_TILE;
  const uint32_t dt = s->head_dim / HEXKL_ATTN_TILE;
  L->n_row_tiles = rt;
  L->n_col_tiles = ct;
  L->n_dot_tiles = dt;

  uint32_t off = 0;
  L->q_ah = off;
  off += rt * dt * TB;
  for (int i = 0; i < 2; ++i) {
    L->kt_wh[i] = off;
    off += dt * ct * TB;
  }
  for (int i = 0; i < 2; ++i) {
    L->v_wh[i] = off;
    off += ct * dt * TB;
  }
  for (int i = 0; i < 2; ++i) {
    L->s_ah[i] = off;
    off += rt * ct * TB;
  }
  L->d_ah = off;
  off += rt * TB;
  for (int i = 0; i < 2; ++i) {
    L->o_wh[i] = off;
    off += dt * rt * TB;
  }
  // Landing buffers are raw rows: bc rows of head_dim f16 = bc*hd*2 bytes,
  // which is exactly ct*dt tiles' worth, so they stay tile-aligned too.
  for (int i = 0; i < 2; ++i) {
    L->k_land[i] = off;
    off += t->bc * s->head_dim * 2u;
  }
  for (int i = 0; i < 2; ++i) {
    L->v_land[i] = off;
    off += t->bc * s->head_dim * 2u;
  }
  L->ml = off;
  off += hexkl_attn_round_up(2u * t->g_br * 4u, TB);

  if (arena_top < cfg_size) {
    return HEXKL_ATTN_ENOMEM;
  }
  L->cfg_f16 = (arena_top - cfg_size) & ~(HEXKL_ATTN_CFG_ALIGN - 1u);
  L->total = off;
  if (off > L->cfg_f16) {
    return HEXKL_ATTN_ENOMEM;
  }
  return HEXKL_ATTN_OK;
}

/**
 * @brief Picks br and bc for a shape when the caller has no opinion.
 *
 * Rows: aim for 64 tile rows (two row tiles) so a G*br block is big enough
 * to amortize K/V staging but small enough that the diagonal region stays
 * a couple of cache blocks. Columns: llama.cpp's rule -- at least three
 * cache blocks so a prefetch pipeline has something to overlap, capped so
 * S (g_br x bc, doubled) stays a few tens of KB. Both are multiples of 32
 * by construction; bc is also a multiple of 64 when the cache allows, for
 * the 2 KiB DMA rows resident tiles will use.
 */
static inline void hexkl_attn_f16_choose_tiling(const hexkl_attn_f16_shape *s,
                                                hexkl_attn_f16_tiling *t) {
  const uint32_t g = s->n_head_kv ? s->n_head_q / s->n_head_kv : 1u;
  uint32_t br = 64u / (g ? g : 1u);
  if (br == 0) {
    br = 1u;
  }
  if (br > s->n_q) {
    br = s->n_q;
  }
  uint32_t bc = (s->cache_to / 3u) & ~63u;
  if (bc < 64u) {
    bc = (s->cache_to >= 64u) ? 64u : 32u;
  }
  if (bc > 256u) {
    bc = 256u;
  }
  t->br = br;
  t->bc = bc;
}

/** @brief Number of query blocks. */
static inline uint32_t
hexkl_attn_f16_n_q_blocks(const hexkl_attn_f16_shape *s,
                          const hexkl_attn_f16_tiling *t) {
  return (s->n_q + t->br - 1u) / t->br;
}

/**
 * @brief Cache rows [lo, hi) that query row @a q may attend to.
 *
 * hi is min(pos + 1, cache_to); lo is max(0, pos + 1 - window). This is
 * MHACoreLayer's rule (compute_kcaches' start_row clamp plus causality by
 * construction), expressed as a range so the kernel never materializes a
 * mask.
 */
static inline void hexkl_attn_f16_row_range(const hexkl_attn_f16_shape *s,
                                            uint32_t q, uint32_t *lo,
                                            uint32_t *hi) {
  const uint32_t pos = s->cache_from + q;
  uint32_t h = pos + 1u;
  if (h > s->cache_to) {
    h = s->cache_to;
  }
  uint32_t l = 0;
  if (s->window != 0 && h > s->window) {
    l = h - s->window;
  }
  *lo = l;
  *hi = h;
}

/**
 * @brief Cache BLOCKS [blk_lo, blk_hi) that query block @a qb touches.
 *
 * The union over the block's rows: the first row sets the lowest visible
 * cache row, the last row the highest. Blocks entirely in the future are
 * simply not visited -- that, not a -inf mask, is how causality is enforced
 * at block granularity.
 */
static inline void hexkl_attn_f16_kv_block_range(const hexkl_attn_f16_shape *s,
                                                 const hexkl_attn_f16_tiling *t,
                                                 uint32_t qb, uint32_t *blk_lo,
                                                 uint32_t *blk_hi) {
  const uint32_t q0 = qb * t->br;
  uint32_t q1 = q0 + t->br;
  if (q1 > s->n_q) {
    q1 = s->n_q;
  }
  uint32_t lo0, hi0, lo1, hi1;
  hexkl_attn_f16_row_range(s, q0, &lo0, &hi0);
  hexkl_attn_f16_row_range(s, q1 - 1u, &lo1, &hi1);
  *blk_lo = lo0 / t->bc;
  *blk_hi = (hi1 + t->bc - 1u) / t->bc;
}

/**
 * @brief Whether cache block @a kb is fully visible to every real row of
 *        query block @a qb, so the softmax can skip the lane mask.
 *
 * Fully visible means the block lies inside every row's [lo, hi): at or
 * above the LAST row's lo (the largest lo) and at or below the FIRST
 * row's hi (the smallest hi).
 */
static inline int
hexkl_attn_f16_kv_block_unmasked(const hexkl_attn_f16_shape *s,
                                 const hexkl_attn_f16_tiling *t, uint32_t qb,
                                 uint32_t kb) {
  const uint32_t q0 = qb * t->br;
  uint32_t q1 = q0 + t->br;
  if (q1 > s->n_q) {
    q1 = s->n_q;
  }
  uint32_t lo0, hi0, lo1, hi1;
  hexkl_attn_f16_row_range(s, q0, &lo0, &hi0);
  hexkl_attn_f16_row_range(s, q1 - 1u, &lo1, &hi1);
  const uint32_t k0 = kb * t->bc;
  const uint32_t k1 = k0 + t->bc;
  return (k0 >= lo1 && k1 <= hi0) ? 1 : 0;
}

/**
 * @brief Maps a tile row r in [0, g_br) of query block @a qb back to its
 *        (query row, query head within the KV group). Returns 0 for pad
 *        rows, which exist only because G*br was rounded up to 32.
 */
static inline int hexkl_attn_f16_tile_row_to_qg(const hexkl_attn_f16_shape *s,
                                                const hexkl_attn_f16_tiling *t,
                                                uint32_t qb, uint32_t r,
                                                uint32_t *q, uint32_t *g) {
  const uint32_t qq = qb * t->br + r / t->g;
  if (r >= t->g * t->br || qq >= s->n_q) {
    return 0;
  }
  *q = qq;
  *g = r % t->g;
  return 1;
}

#ifdef __cplusplus
}
#endif

#endif /* __NNTRAINER_HEXKL_ATTN_F16_PLAN_H__ */
