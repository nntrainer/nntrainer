// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file   hexkl_kv_q.h
 * @date   23 Sep 2026
 * @brief  DSP-resident int8 / int4 KV cache with per-token scales
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The quantized counterpart of hexkl_kv_tiles_f16: one registered cache per
 * attention layer, rows appended once as fp16 and quantized here, on the
 * DSP, never revisited. Two consumers read it:
 *
 * - The HMX prefill kernel wants K^T and V as WH weight tiles (32x32,
 *   1024 B at int8 / 512 B packed at int4) so a cache block is one DMA of
 *   ready tiles. Baked by HexKL's rm_to_wh_i8 / _i4 from a row-major
 *   staging of the column tile being written -- HexKL's extraction helpers
 *   read DDR fast and VTCM slowly, so the staging is DDR.
 * - The HVX decode kernel wants one vrmpy per (32 cache rows x 4 dims) for
 *   the scores and per (4 rows x 32 dims) for the values, which fixes the
 *   master layouts: K^T as [hd/4][rows][4] and V as [rows/4][hd][4]. They
 *   are stored offset-binary (value + 128, one byte each; int4 values in
 *   [-7, 7] unpacked, packing is a follow-up) because vrmpy's vector
 *   operand is unsigned bytes against a scalar register of 4 unsigned
 *   bytes -- Q or P' -- and the +128 comes out as one integer correction
 *   per block (128 * sum of the scalar side's bytes).
 *
 * Scales are symmetric, per token: K one scale per (row, kv head) over
 * head_dim plus colsum (the int sum over head_dim, for the uint8 activation
 * zero-point term); V one scale per (row, kv head, 32-dim group), which the
 * attention kernel folds into P per group before quantizing P. The axes are
 * exactly the ones an int32 HMX accumulator can be dequantized on -- see
 * docs/backend_guide/htp_backend/21_quantized_attention_plan.md.
 *
 * The scalar quantizer and every index helper are plain C so the host unit
 * test runs the same code the DSP does; only the tile bake needs HexKL.
 */

#ifndef __NNTRAINER_HEXKL_KV_Q_H__
#define __NNTRAINER_HEXKL_KV_Q_H__

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Resident caches per session: one per (attention layer, batch). */
#define HEXKL_KV_Q_MAX 128u
/** @brief Extra s_v rows after the last head (the largest block size). */
#define HEXKL_KV_Q_SV_PAD 256u
/** @brief Largest head_dim: Gemma-4's global layers use 512. */
#define HEXKL_KV_Q_MAX_HEAD_DIM 512u

/** @brief Weight width of the cache: A8W8 or A8W4 in the FC layers' terms. */
typedef enum {
  HEXKL_KV_Q8 = 0, /**< int8 values in [-127, 127], 1024 B tiles */
  HEXKL_KV_Q4 = 1, /**< int4 values in [-8, 7], 512 B packed tiles */
} hexkl_kv_q_kind;

typedef struct {
  int in_use;
  hexkl_kv_q_kind kind;
  int32_t qmax;        /**< 127 or 7: the symmetric range */
  uint32_t tile_bytes; /**< WH tile size for this kind */
  uint32_t max_rows;   /**< capacity in cache rows, multiple of 32 */
  uint32_t n_head_kv;
  uint32_t head_dim;    /**< multiple of 32, at most HEXKL_KV_Q_MAX_HEAD_DIM */
  uint32_t n_col_tiles; /**< max_rows / 32 */
  uint32_t n_dot_tiles; /**< head_dim / 32 */
  uint8_t *kt4;         /**< [n_head_kv][head_dim/4][max_rows][4], q+128 */
  uint8_t *v4;          /**< [n_head_kv][max_rows/4][head_dim][4], q+128 */
  float *s_k;           /**< [n_head_kv][max_rows] */
  int32_t *colsum_k;    /**< [n_head_kv][max_rows] */
  float *s_v;           /**< [n_head_kv][max_rows + pad]: one scale per
                             (row, head), covering every 32-dim group of
                             that row. A block's scales are contiguous, so
                             the attention kernel reads them from DDR as
                             vectors; HEXKL_KV_Q_SV_PAD rows of 1.0 past
                             the end keep a block that overruns the last
                             head in bounds. Being independent of the
                             group is what lets the kernel fold the scale
                             into P and quantize P once rather than once
                             per group */
  uint8_t *kt;          /**< K^T WH tiles, [n_head_kv][n_col][n_dot] */
  uint8_t *v;           /**< V WH tiles, same order */
  int8_t *stage_kt;     /**< [head_dim][32] row-major bake source */
  int8_t *stage_v;      /**< [32][head_dim] row-major bake source */
  int fixed;            /**< 1: appends quantize with fs_k / fs_v instead
                             of per-row maxima (hexkl_kv_q_set_fixed_scales).
                             The row-blocked kernel needs this: K's scale
                             then sits in the softmax's residual and V's in
                             the convert unit, neither per cache row */
  int plain_masters;    /**< 1 (fixed-scale int8): kt4 / v4 hold the masters
                             row-major, [n_head_kv][max_rows][head_dim], so
                             an appended row is one contiguous store and a
                             full column tile is baked from 32 contiguous
                             rows. The 4-interleaved layouts exist for the
                             HVX decode kernel, which a fixed-scale cache
                             never feeds (the row-blocked kernel takes its
                             decode steps too) */
  float *fs_k;          /**< [n_head_kv] fixed K scale per head */
  float *fs_v;          /**< [n_head_kv][head_dim] fixed V scale per dim */
  float *fs_v_inv;      /**< [n_head_kv][head_dim] 1 / fs_v, for the append */
} hexkl_kv_q;

typedef struct {
  hexkl_kv_q slots[HEXKL_KV_Q_MAX];
} hexkl_kv_q_table;

/** @brief IEEE binary16 bits -> f32, bit-exact, no fp16 support needed. */
static inline float hexkl_kv_q_hf_to_f32(uint16_t h) {
  const uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t mant = h & 0x3FFu;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      exp = 127u - 15u + 1u;
      while ((mant & 0x400u) == 0) {
        mant <<= 1;
        --exp;
      }
      mant &= 0x3FFu;
      bits = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000u | (mant << 13);
  } else {
    bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
  }
  float f;
  memcpy(&f, &bits, sizeof(f));
  return f;
}

/** @brief Symmetric range of a kind. */
static inline int32_t hexkl_kv_q_qmax(hexkl_kv_q_kind kind) {
  return kind == HEXKL_KV_Q4 ? 7 : 127;
}

/** @brief WH tile bytes of a kind. */
static inline uint32_t hexkl_kv_q_tile_bytes(hexkl_kv_q_kind kind) {
  return kind == HEXKL_KV_Q4 ? 512u : 1024u;
}

/** @brief Offset-binary encoding of the masters. */
#define HEXKL_KV_Q_BIAS 128

/** @brief Master index of K[row][dim] for kv head n: 4-dim interleaved,
 *         or row-major when plain_masters. */
static inline size_t hexkl_kv_q_kt4_index(const hexkl_kv_q *kv, uint32_t n,
                                          uint32_t row, uint32_t dim) {
  if (kv->plain_masters) {
    return ((size_t)n * kv->max_rows + row) * kv->head_dim + dim;
  }
  return (((size_t)n * (kv->head_dim / 4u) + dim / 4u) * kv->max_rows + row) *
           4u +
         (dim & 3u);
}

/** @brief Master index of V[row][dim] for kv head n: 4-row interleaved, or
 *         row-major when plain_masters. */
static inline size_t hexkl_kv_q_v4_index(const hexkl_kv_q *kv, uint32_t n,
                                         uint32_t row, uint32_t dim) {
  if (kv->plain_masters) {
    return ((size_t)n * kv->max_rows + row) * kv->head_dim + dim;
  }
  return (((size_t)n * (kv->max_rows / 4u) + row / 4u) * kv->head_dim + dim) *
           4u +
         (row & 3u);
}

/** @brief Index into s_k / colsum_k. */
static inline size_t hexkl_kv_q_sk_index(const hexkl_kv_q *kv, uint32_t n,
                                         uint32_t row) {
  return (size_t)n * kv->max_rows + row;
}

/** @brief Index into s_v: the rows of one head are contiguous. */
static inline size_t hexkl_kv_q_sv_index(const hexkl_kv_q *kv, uint32_t n,
                                         uint32_t row) {
  return (size_t)n * kv->max_rows + row;
}

/** @brief Byte offset of WH tile (n, column tile c, dot tile d) in kt or v. */
static inline size_t hexkl_kv_q_tile_off(const hexkl_kv_q *kv, uint32_t n,
                                         uint32_t c, uint32_t d) {
  return (((size_t)n * kv->n_col_tiles + c) * kv->n_dot_tiles + d) *
         kv->tile_bytes;
}

/**
 * @brief Quantizes one K row of one head: symmetric over @a hd values.
 *
 * scale = amax / qmax (1 when the row is all zero), q = rint(x * qmax /
 * amax) clamped to [-qmax, qmax], colsum = sum of q. Round to nearest even
 * through rintf, the same on host and DSP.
 */
void hexkl_kv_q_quant_k_row(const float *x, uint32_t hd, int32_t qmax,
                            int8_t *q, float *scale, int32_t *colsum);

/**
 * @brief Quantizes one V row of one head: symmetric over all @a hd values.
 *
 * One scale for the whole row rather than one per 32-dim group. Measured
 * on Qwen3-0.6B, V magnitudes vary 1.7x (median) across the groups of a
 * row but 5x to 99x across the rows of a 256-row block, so the row is
 * where the adaptivity is worth keeping; dropping the group refinement
 * costs 0.2-0.4 dB and makes the scale independent of the group, which is
 * what lets the attention kernel quantize P once instead of once per
 * group.
 *
 * @param[out] scale  one entry
 */
void hexkl_kv_q_quant_v_row(const float *x, uint32_t hd, int32_t qmax,
                            int8_t *q, float *scale);

/**
 * @brief K row with a given scale: q = rint(x / scale) clamped, colsum.
 *
 * @param inv  1 / scale
 */
void hexkl_kv_q_quant_k_row_fixed(const float *x, uint32_t hd, int32_t qmax,
                                  float inv, int8_t *q, int32_t *colsum);

/**
 * @brief V row with a given scale per dim: q[d] = rint(x[d] * inv[d])
 *        clamped.
 */
void hexkl_kv_q_quant_v_row_fixed(const float *x, uint32_t hd, int32_t qmax,
                                  const float *inv, int8_t *q);

/**
 * @brief Switches a cache to fixed scales: @a s_k [n_head_kv] for K and
 *        @a s_v [n_head_kv][head_dim] for V, as a quantized model's
 *        encodings give them (K per head after RoPE, V per head dim). From
 *        then on appends quantize with them; s_k[row] is set to the head's
 *        scale so colsum-based readers keep working, s_v[row] stays 1 and
 *        is not meaningful. An int8 cache also switches to row-major
 *        masters (plain_masters). Call before the first append.
 */
int hexkl_kv_q_set_fixed_scales(hexkl_kv_q_table *tbl, uint32_t handle,
                                const float *s_k, const float *s_v);

/**
 * @brief The int8 WH tile permutation the DSP derived from rm_to_wh_i8:
 *        element (r, k) of a 32x32 row-major tile lands at byte pos[r*32+k].
 *        DSP only (probes it on first use, clobbering 1 KiB at @a vtcm);
 *        NULL when the part's layout is not usable.
 */
const uint16_t *hexkl_kv_q_wh_i8_pos(uint8_t *vtcm_base);

/**
 * @brief Allocates a zeroed cache for up to @a max_rows rows.
 *
 * @param max_rows  rounded up to a multiple of 32
 * @return AEE_SUCCESS with *out_handle set, AEE_EBADPARM, AEE_ENOMEMORY
 */
int hexkl_kv_q_register(hexkl_kv_q_table *tbl, hexkl_kv_q_kind kind,
                        uint32_t max_rows, uint32_t n_head_kv,
                        uint32_t head_dim, uint32_t *out_handle);

int hexkl_kv_q_release(hexkl_kv_q_table *tbl, uint32_t handle);

/** @brief The slot, or NULL if the handle is not in use. */
const hexkl_kv_q *hexkl_kv_q_get(const hexkl_kv_q_table *tbl, uint32_t handle);

/** @brief Where an append spends its time, microseconds (DSP only; the host
 *         build leaves it zero). */
typedef struct {
  uint32_t quant_us; /**< fp16 -> int8/int4 masters, scales, colsums */
  uint32_t stage_us; /**< masters -> row-major staging of the column tiles */
  uint32_t bake_us;  /**< HexKL rm_to_wh of the staged tiles */
} hexkl_kv_q_append_stats;

/**
 * @brief Quantizes and stores cache rows [row0, row0 + n_rows) from the
 *        cache's own row-major fp16 layout ([n_rows][n_head_kv*head_dim]),
 *        then re-bakes the WH tiles of every column tile touched. Rows may
 *        be rewritten.
 *
 * @param vtcm_base  scratch for the bake: the first 2*tile_bytes bytes are
 *                   clobbered. The caller holds the HMX lock. NULL skips
 *                   the bake (host builds), leaving only the masters,
 *                   scales and staging updated.
 */
int hexkl_kv_q_append(hexkl_kv_q_table *tbl, uint32_t handle, uint32_t row0,
                      uint32_t n_rows, const uint16_t *k_rows,
                      const uint16_t *v_rows, uint8_t *vtcm_base,
                      hexkl_kv_q_append_stats *st);

/**
 * @brief Fills the row-major staging buffers with column tile @a c of head
 *        @a n from the masters (stage_kt as [head_dim][32], stage_v as
 *        [32][head_dim]). What the bake reads; exposed for the host test.
 */
void hexkl_kv_q_stage(hexkl_kv_q *kv, uint32_t n, uint32_t c);

/**
 * @brief Gathers rows [row0, row0 + n_rows) back into the cache's row-major
 *        order: k_q / v_q [n_rows][n_head_kv*head_dim] int8, s_k / colsum_k
 *        [n_rows][n_head_kv], s_v [n_rows][n_head_kv][head_dim/32]. Any
 *        output may be NULL. For tests and the dump entry.
 */
int hexkl_kv_q_dump(const hexkl_kv_q *kv, uint32_t row0, uint32_t n_rows,
                    int8_t *k_q, int8_t *v_q, float *s_k, int32_t *colsum_k,
                    float *s_v);

#ifdef __cplusplus
}
#endif

#endif /* __NNTRAINER_HEXKL_KV_Q_H__ */
