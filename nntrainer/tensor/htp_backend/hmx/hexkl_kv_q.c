// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file   hexkl_kv_q.c
 * @date   23 Sep 2026
 * @brief  DSP-resident int8 / int4 KV cache with per-token scales
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "hexkl_kv_q.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef __hexagon__
#include "hexkl_micro.h"
#include "hvx_kv_quant.h"
#include "hvx_tile_f16.h"
#include <AEEStdErr.h>
#include <HAP_perf.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
static inline uint64_t kvq_now_us(void) { return HAP_perf_get_time_us(); }
#else
static inline uint64_t kvq_now_us(void) { return 0; }
// Host build (the unit test): the codes this file returns, with the SDK's
// values (AEEStdErr.h: AEE_EOFFSET + 0x002 / 0x00E / 0x001).
#define AEE_SUCCESS 0
#define AEE_EFAILED 0x80000401
#define AEE_ENOMEMORY 0x80000402
#define AEE_EBADPARM 0x8000040E
#endif

void hexkl_kv_q_quant_k_row(const float *x, uint32_t hd, int32_t qmax,
                            int8_t *q, float *scale, int32_t *colsum) {
  float amax = 0.0f;
  for (uint32_t d = 0; d < hd; ++d) {
    const float a = fabsf(x[d]);
    if (a > amax) {
      amax = a;
    }
  }
  const float fq = (float)qmax;
  int32_t sum = 0;
  if (amax == 0.0f) {
    memset(q, 0, hd);
    *scale = 1.0f;
    *colsum = 0;
    return;
  }
  const float inv = fq / amax;
  for (uint32_t d = 0; d < hd; ++d) {
    float r = rintf(x[d] * inv);
    if (r > fq) {
      r = fq;
    } else if (r < -fq) {
      r = -fq;
    }
    q[d] = (int8_t)r;
    sum += (int32_t)r;
  }
  *scale = amax / fq;
  *colsum = sum;
}

void hexkl_kv_q_quant_v_row(const float *x, uint32_t hd, int32_t qmax,
                            int8_t *q, float *scale) {
  const float fq = (float)qmax;
  float amax = 0.0f;
  for (uint32_t d = 0; d < hd; ++d) {
    const float a = fabsf(x[d]);
    if (a > amax) {
      amax = a;
    }
  }
  if (amax == 0.0f) {
    memset(q, 0, hd);
    *scale = 1.0f;
    return;
  }
  const float inv = fq / amax;
  for (uint32_t d = 0; d < hd; ++d) {
    float r = rintf(x[d] * inv);
    if (r > fq) {
      r = fq;
    } else if (r < -fq) {
      r = -fq;
    }
    q[d] = (int8_t)r;
  }
  *scale = amax / fq;
}

void hexkl_kv_q_quant_k_row_fixed(const float *x, uint32_t hd, int32_t qmax,
                                  float inv, int8_t *q, int32_t *colsum) {
  const float fq = (float)qmax;
  int32_t sum = 0;
  for (uint32_t d = 0; d < hd; ++d) {
    float r = rintf(x[d] * inv);
    if (r > fq) {
      r = fq;
    } else if (r < -fq) {
      r = -fq;
    }
    q[d] = (int8_t)r;
    sum += (int32_t)r;
  }
  *colsum = sum;
}

void hexkl_kv_q_quant_v_row_fixed(const float *x, uint32_t hd, int32_t qmax,
                                  const float *inv, int8_t *q) {
  const float fq = (float)qmax;
  for (uint32_t d = 0; d < hd; ++d) {
    float r = rintf(x[d] * inv[d]);
    if (r > fq) {
      r = fq;
    } else if (r < -fq) {
      r = -fq;
    }
    q[d] = (int8_t)r;
  }
}

int hexkl_kv_q_set_fixed_scales(hexkl_kv_q_table *tbl, uint32_t handle,
                                const float *s_k, const float *s_v) {
  hexkl_kv_q *kv = (hexkl_kv_q *)hexkl_kv_q_get(tbl, handle);
  if (!kv || !s_k || !s_v) {
    return AEE_EBADPARM;
  }
  for (uint32_t n = 0; n < kv->n_head_kv; ++n) {
    if (!(s_k[n] > 0.0f)) {
      return AEE_EBADPARM;
    }
    for (uint32_t d = 0; d < kv->head_dim; ++d) {
      if (!(s_v[(size_t)n * kv->head_dim + d] > 0.0f)) {
        return AEE_EBADPARM;
      }
    }
  }
  memcpy(kv->fs_k, s_k, kv->n_head_kv * sizeof(float));
  memcpy(kv->fs_v, s_v, (size_t)kv->n_head_kv * kv->head_dim * sizeof(float));
  for (size_t i = 0; i < (size_t)kv->n_head_kv * kv->head_dim; ++i) {
    kv->fs_v_inv[i] = 1.0f / s_v[i];
  }
  for (uint32_t n = 0; n < kv->n_head_kv; ++n) {
    for (uint32_t row = 0; row < kv->max_rows; ++row) {
      kv->s_k[hexkl_kv_q_sk_index(kv, n, row)] = s_k[n];
    }
  }
  kv->fixed = 1;
  kv->plain_masters = kv->kind == HEXKL_KV_Q8;
  return AEE_SUCCESS;
}

static void free_slot(hexkl_kv_q *kv) {
  free(kv->fs_k);
  free(kv->fs_v);
  free(kv->fs_v_inv);
  free(kv->kt4);
  free(kv->v4);
  free(kv->s_k);
  free(kv->colsum_k);
  free(kv->s_v);
  free(kv->kt);
  free(kv->v);
  free(kv->stage_kt);
  free(kv->stage_v);
  memset(kv, 0, sizeof(*kv));
}

int hexkl_kv_q_register(hexkl_kv_q_table *tbl, hexkl_kv_q_kind kind,
                        uint32_t max_rows, uint32_t n_head_kv,
                        uint32_t head_dim, uint32_t *out_handle) {
  if (!tbl || !out_handle || max_rows == 0 || n_head_kv == 0 || head_dim == 0 ||
      (head_dim % 32u) != 0 || head_dim > HEXKL_KV_Q_MAX_HEAD_DIM ||
      (kind != HEXKL_KV_Q8 && kind != HEXKL_KV_Q4)) {
    return AEE_EBADPARM;
  }
  uint32_t slot = HEXKL_KV_Q_MAX;
  for (uint32_t i = 0; i < HEXKL_KV_Q_MAX; ++i) {
    if (!tbl->slots[i].in_use) {
      slot = i;
      break;
    }
  }
  if (slot == HEXKL_KV_Q_MAX) {
    return AEE_ENOMEMORY;
  }

  hexkl_kv_q *kv = &tbl->slots[slot];
  memset(kv, 0, sizeof(*kv));
  kv->kind = kind;
  kv->qmax = hexkl_kv_q_qmax(kind);
  kv->tile_bytes = hexkl_kv_q_tile_bytes(kind);
  kv->max_rows = ((max_rows + 31u) / 32u) * 32u;
  kv->n_head_kv = n_head_kv;
  kv->head_dim = head_dim;
  kv->n_col_tiles = kv->max_rows / 32u;
  kv->n_dot_tiles = head_dim / 32u;

  const size_t values = (size_t)n_head_kv * kv->max_rows * head_dim;
  const size_t rows = (size_t)n_head_kv * kv->max_rows;
  const size_t n_tiles = rows / 32u * kv->n_dot_tiles;
  // calloc throughout: rows past cache_to read as zero with scale 1, so a
  // block that runs past the cache multiplies finite zeros, never garbage.
  // Offset-binary masters: an unwritten row must read as 0, i.e. 128.
  kv->kt4 = (uint8_t *)malloc(values);
  kv->v4 = (uint8_t *)malloc(values);
  kv->s_k = (float *)calloc(rows, sizeof(float));
  kv->colsum_k = (int32_t *)calloc(rows, sizeof(int32_t));
  kv->s_v = (float *)calloc(rows + HEXKL_KV_Q_SV_PAD, sizeof(float));
  kv->kt = (uint8_t *)calloc(n_tiles, kv->tile_bytes);
  kv->v = (uint8_t *)calloc(n_tiles, kv->tile_bytes);
  kv->stage_kt = (int8_t *)calloc((size_t)head_dim * 32u, 1u);
  kv->stage_v = (int8_t *)calloc((size_t)head_dim * 32u, 1u);
  kv->fs_k = (float *)calloc(n_head_kv, sizeof(float));
  kv->fs_v = (float *)calloc((size_t)n_head_kv * head_dim, sizeof(float));
  kv->fs_v_inv = (float *)calloc((size_t)n_head_kv * head_dim, sizeof(float));
  if (!kv->kt4 || !kv->v4 || !kv->s_k || !kv->colsum_k || !kv->s_v || !kv->kt ||
      !kv->v || !kv->stage_kt || !kv->stage_v || !kv->fs_k || !kv->fs_v ||
      !kv->fs_v_inv) {
    free_slot(kv);
    return AEE_ENOMEMORY;
  }
  memset(kv->kt4, HEXKL_KV_Q_BIAS, values);
  memset(kv->v4, HEXKL_KV_Q_BIAS, values);
  for (size_t i = 0; i < rows; ++i) {
    kv->s_k[i] = 1.0f;
  }
  for (size_t i = 0; i < rows + HEXKL_KV_Q_SV_PAD; ++i) {
    kv->s_v[i] = 1.0f;
  }
  kv->in_use = 1;
  *out_handle = slot;
  return AEE_SUCCESS;
}

int hexkl_kv_q_release(hexkl_kv_q_table *tbl, uint32_t handle) {
  if (!tbl || handle >= HEXKL_KV_Q_MAX || !tbl->slots[handle].in_use) {
    return AEE_EBADPARM;
  }
  free_slot(&tbl->slots[handle]);
  return AEE_SUCCESS;
}

const hexkl_kv_q *hexkl_kv_q_get(const hexkl_kv_q_table *tbl, uint32_t handle) {
  if (!tbl || handle >= HEXKL_KV_Q_MAX || !tbl->slots[handle].in_use) {
    return NULL;
  }
  return &tbl->slots[handle];
}

void hexkl_kv_q_stage(hexkl_kv_q *kv, uint32_t n, uint32_t c) {
  const uint32_t hd = kv->head_dim;
  const uint32_t row0 = 32u * c;
  for (uint32_t rr = 0; rr < 32u; ++rr) {
    const uint32_t row = row0 + rr;
    for (uint32_t d = 0; d < hd; ++d) {
      kv->stage_kt[(size_t)d * 32u + rr] =
        (int8_t)((int)kv->kt4[hexkl_kv_q_kt4_index(kv, n, row, d)] -
                 HEXKL_KV_Q_BIAS);
      kv->stage_v[(size_t)rr * hd + d] =
        (int8_t)((int)kv->v4[hexkl_kv_q_v4_index(kv, n, row, d)] -
                 HEXKL_KV_Q_BIAS);
    }
  }
}

/**
 * @brief Where element (row r, col k) of a 32x32 int8 weight tile lands in
 *        the 1024-byte WH layout, derived at runtime from HexKL's own
 *        rm_to_wh_i8 the way hexkl_acc_tile derives the accumulator layout:
 *        bake a tile whose bytes are the low 8 bits of their own index,
 *        then one whose bytes are the high 2 bits, and read the permutation
 *        off the two results. With it an appended row is 2 * head_dim
 *        byte stores per head straight into the tiles -- no 32-row
 *        re-staging, no HexKL call per token. int4 tiles are nibble-packed
 *        and keep the re-bake path.
 */
#ifdef __hexagon__
static uint16_t g_wh_i8_pos[1024];
static int g_wh_i8_state; /* 0 unprobed, 1 usable, -1 not */

static int wh_i8_probe(uint8_t *vtcm_base) {
  if (g_wh_i8_state != 0) {
    return g_wh_i8_state == 1;
  }
  g_wh_i8_state = -1;
  int8_t src[1024];
  uint8_t out[2][1024];
  for (int pass = 0; pass < 2; ++pass) {
    for (int i = 0; i < 1024; ++i) {
      src[i] = (int8_t)(pass ? (i >> 8) : (i & 0xFF));
    }
    if (hexkl_micro_hmx_rm_to_wh_i8(vtcm_base, 0u, src, 0u, 0u, 32u) !=
        AEE_SUCCESS) {
      return 0;
    }
    memcpy(out[pass], vtcm_base, 1024u);
  }
  uint8_t seen[1024];
  memset(seen, 0, sizeof(seen));
  for (int p = 0; p < 1024; ++p) {
    const unsigned idx = (unsigned)out[0][p] | ((unsigned)out[1][p] << 8);
    if (idx >= 1024u || seen[idx]) {
      return 0;
    }
    seen[idx] = 1;
    g_wh_i8_pos[idx] = (uint16_t)p;
  }
  g_wh_i8_state = 1;
  return 1;
}

const uint16_t *hexkl_kv_q_wh_i8_pos(uint8_t *vtcm_base) {
  return wh_i8_probe(vtcm_base) ? g_wh_i8_pos : NULL;
}

/** @brief Writes one quantized row of head n straight into its int8 WH
 *         tiles: cache row @a row is column row%32 of the K^T tiles and
 *         row row%32 of the V tiles of column tile row/32. */
static void write_row_i8_tiles(hexkl_kv_q *kv, uint32_t n, uint32_t row,
                               const int8_t *qk, const int8_t *qv) {
  const uint32_t c = row / 32u, rr = row % 32u;
  for (uint32_t d = 0; d < kv->n_dot_tiles; ++d) {
    uint8_t *kt = kv->kt + hexkl_kv_q_tile_off(kv, n, c, d);
    uint8_t *vt = kv->v + hexkl_kv_q_tile_off(kv, n, c, d);
    for (uint32_t j = 0; j < 32u; ++j) {
      kt[g_wh_i8_pos[j * 32u + rr]] = (uint8_t)qk[32u * d + j];
      vt[g_wh_i8_pos[rr * 32u + j]] = (uint8_t)qv[32u * d + j];
    }
  }
}
#endif

#ifdef __hexagon__
/**
 * @brief Fixed-scale int8 append on the DSP: rows quantized on HVX straight
 *        into the row-major masters (one unaligned vector store per 128
 *        values), then every column tile the rows cover is baked: a tile
 *        the rows fill completely is staged from its 32 contiguous master
 *        rows and run through rm_to_wh_i8 once per dot tile; a tile they
 *        only partly cover takes the per-row scatter through the position
 *        table, which is what rows appended one at a time (decode) pay.
 */
static inline void xor80_copy(uint8_t *dst, const uint8_t *src, uint32_t n) {
  const HVX_Vector x = Q6_V_vsplat_R(0x80808080);
  for (uint32_t i = 0; i < n; i += 128u) {
    hvx_tile_store_u(dst + i, Q6_V_vxor_VV(hvx_tile_load_u(src + i), x));
  }
}

/**
 * @brief Bakes the K^T and V WH tiles of one full column tile of head n on
 *        HVX from its 32 contiguous master rows.
 *
 * The int8 WH tile puts row-major element (r, k) at byte
 * (r/4)*128 + 4k + r%4 (hexkl_kv_q_wh_i8_pos, checked for all 1024). For
 * the V tile r is the cache row and k the dim: each 128-byte segment is
 * four consecutive rows interleaved bytewise, two vshuff levels. For the
 * K^T tile r is the dim and k the cache row: segment j/4 holds, for every
 * row in order, the 4-byte word of dims 4(j/4)..+3, i.e. the tile set is
 * the 32 x (hd/4) word transpose of the rows, five vshuff stages at word
 * width pairing rows 16, 8, 4, 2, 1 apart (the unique sequence, derived
 * and checked against the byte model). Masters are offset binary, the
 * tiles int8: xor 0x80 on the way.
 */
static void bake_tile_hvx(hexkl_kv_q *kv, uint32_t n, uint32_t c) {
  const uint32_t hd = kv->head_dim;
  const uint32_t row0 = 32u * c;
  const uint8_t *km0 = kv->kt4 + hexkl_kv_q_kt4_index(kv, n, row0, 0);
  const uint8_t *vm0 = kv->v4 + hexkl_kv_q_v4_index(kv, n, row0, 0);
  const HVX_Vector x80 = Q6_V_vsplat_R(0x80808080);

  /* V: rows 4s..4s+3, 128 bytes (four dim tiles) at a time. */
  for (uint32_t s = 0; s < 8u; ++s) {
    for (uint32_t m = 0; m < hd / 128u; ++m) {
      const uint8_t *r = vm0 + (size_t)(4u * s) * hd + 128u * m;
      const HVX_Vector a = Q6_V_vxor_VV(hvx_tile_load_u(r), x80);
      const HVX_Vector b = Q6_V_vxor_VV(hvx_tile_load_u(r + hd), x80);
      const HVX_Vector cc = Q6_V_vxor_VV(hvx_tile_load_u(r + 2u * hd), x80);
      const HVX_Vector d = Q6_V_vxor_VV(hvx_tile_load_u(r + 3u * hd), x80);
      const HVX_VectorPair ab = Q6_W_vshuff_VVR(b, a, -1);
      const HVX_VectorPair cd = Q6_W_vshuff_VVR(d, cc, -1);
      const HVX_VectorPair t0 =
        Q6_W_vshuff_VVR(Q6_V_lo_W(cd), Q6_V_lo_W(ab), -2);
      const HVX_VectorPair t1 =
        Q6_W_vshuff_VVR(Q6_V_hi_W(cd), Q6_V_hi_W(ab), -2);
      const HVX_Vector seg[4] = {Q6_V_lo_W(t0), Q6_V_hi_W(t0), Q6_V_lo_W(t1),
                                 Q6_V_hi_W(t1)};
      for (uint32_t t = 0; t < 4u; ++t) {
        uint8_t *tile = kv->v + hexkl_kv_q_tile_off(kv, n, c, 4u * m + t);
        hvx_tile_store_u(tile + 128u * s, seg[t]);
      }
    }
  }

  /* K^T: one 32 x 32 word block per 128-byte column of the rows. */
  for (uint32_t m = 0; m < hd / 128u; ++m) {
    HVX_Vector v[32];
    for (uint32_t r = 0; r < 32u; ++r) {
      v[r] =
        Q6_V_vxor_VV(hvx_tile_load_u(km0 + (size_t)r * hd + 128u * m), x80);
    }
    for (uint32_t D = 16u; D >= 1u; D >>= 1) {
      for (uint32_t i = 0; i < 32u; ++i) {
        if (i & D) {
          continue;
        }
        const HVX_VectorPair w = Q6_W_vshuff_VVR(v[i + D], v[i], -4);
        v[i] = Q6_V_lo_W(w);
        v[i + D] = Q6_V_hi_W(w);
      }
    }
    /** v[s] is word column 32m + s of the rows: dim tile (32m+s)/8,
     * segment (32m+s)%8. */
    for (uint32_t s = 0; s < 32u; ++s) {
      const uint32_t wq = 32u * m + s;
      uint8_t *tile = kv->kt + hexkl_kv_q_tile_off(kv, n, c, wq / 8u);
      hvx_tile_store_u(tile + 128u * (wq % 8u), v[s]);
    }
  }
}

static int append_fixed_i8(hexkl_kv_q *kv, uint32_t row0, uint32_t n_rows,
                           const uint16_t *k_rows, const uint16_t *v_rows,
                           uint8_t *vtcm_base, hexkl_kv_q_append_stats *st) {
  const uint64_t tq0 = kvq_now_us();
  const uint32_t hd = kv->head_dim;
  const uint32_t stride = kv->n_head_kv * hd;
  int8_t q[HEXKL_KV_Q_MAX_HEAD_DIM] __attribute__((aligned(128)));
  for (uint32_t r = 0; r < n_rows; ++r) {
    const uint32_t row = row0 + r;
    const uint16_t *krow = k_rows + (size_t)r * stride;
    const uint16_t *vrow = v_rows + (size_t)r * stride;
    for (uint32_t n = 0; n < kv->n_head_kv; ++n) {
      int32_t cs;
      hvx_kv_quant_k_row_fixed(krow + (size_t)n * hd, hd, kv->qmax,
                               1.0f / kv->fs_k[n], q, &cs);
      kv->s_k[hexkl_kv_q_sk_index(kv, n, row)] = kv->fs_k[n];
      kv->colsum_k[hexkl_kv_q_sk_index(kv, n, row)] = cs;
      xor80_copy(kv->kt4 + hexkl_kv_q_kt4_index(kv, n, row, 0),
                 (const uint8_t *)q, hd);
      hvx_kv_quant_v_row_fixed(vrow + (size_t)n * hd, hd, kv->qmax,
                               kv->fs_v_inv + (size_t)n * hd, q);
      kv->s_v[hexkl_kv_q_sv_index(kv, n, row)] = 1.0f;
      xor80_copy(kv->v4 + hexkl_kv_q_v4_index(kv, n, row, 0),
                 (const uint8_t *)q, hd);
    }
  }
  const uint64_t tq1 = kvq_now_us();
  if (st) {
    st->quant_us = (uint32_t)(tq1 - tq0);
  }
  if (!vtcm_base) {
    return AEE_SUCCESS;
  }
  const int direct = wh_i8_probe(vtcm_base);
  const uint32_t c_lo = row0 / 32u;
  const uint32_t c_hi = (row0 + n_rows - 1u) / 32u;
  const uint32_t tb = kv->tile_bytes;
  int8_t qk[HEXKL_KV_Q_MAX_HEAD_DIM], qv[HEXKL_KV_Q_MAX_HEAD_DIM];
  for (uint32_t n = 0; n < kv->n_head_kv; ++n) {
    for (uint32_t c = c_lo; c <= c_hi; ++c) {
      const uint32_t t_lo = 32u * c, t_hi = t_lo + 32u;
      const int full = row0 <= t_lo && row0 + n_rows >= t_hi;
      // A tile the rows cover completely is built on HVX from its 32
      // master rows (bake_tile_hvx; whole 128-byte row chunks, so head_dim
      // a multiple of 128); a partial one, or a narrower head, takes the
      // per-row scatter through the position table (~5 us per tile pair, what
      // single-row decode appends pay). rm_to_wh_i8 with its staging and
      // copies measured 84 us and is the fallback for a part without a
      // usable position table.
      if (full && direct && (hd % 128u) == 0u) {
        const uint64_t tb0 = kvq_now_us();
        bake_tile_hvx(kv, n, c);
        if (st) {
          st->bake_us += (uint32_t)(kvq_now_us() - tb0);
        }
        continue;
      }
      if (direct) {
        const uint32_t r_lo = row0 > t_lo ? row0 : t_lo;
        const uint32_t r_hi = row0 + n_rows < t_hi ? row0 + n_rows : t_hi;
        for (uint32_t row = r_lo; row < r_hi; ++row) {
          const uint8_t *km = kv->kt4 + hexkl_kv_q_kt4_index(kv, n, row, 0);
          const uint8_t *vm = kv->v4 + hexkl_kv_q_v4_index(kv, n, row, 0);
          for (uint32_t d = 0; d < hd; ++d) {
            qk[d] = (int8_t)(km[d] ^ 0x80u);
            qv[d] = (int8_t)(vm[d] ^ 0x80u);
          }
          write_row_i8_tiles(kv, n, row, qk, qv);
        }
        continue;
      }
      // No position table: stage the full tile and bake. V's staging is
      // the 32 master rows as they are; K^T's is their transpose.
      const uint64_t ts0 = kvq_now_us();
      const uint8_t *km0 = kv->kt4 + hexkl_kv_q_kt4_index(kv, n, t_lo, 0);
      const uint8_t *vm0 = kv->v4 + hexkl_kv_q_v4_index(kv, n, t_lo, 0);
      xor80_copy((uint8_t *)kv->stage_v, vm0, 32u * hd);
      for (uint32_t rr = 0; rr < 32u; ++rr) {
        const uint8_t *km = km0 + (size_t)rr * hd;
        for (uint32_t d = 0; d < hd; ++d) {
          kv->stage_kt[(size_t)d * 32u + rr] = (int8_t)(km[d] ^ 0x80u);
        }
      }
      const uint64_t ts1 = kvq_now_us();
      for (uint32_t d = 0; d < kv->n_dot_tiles; ++d) {
        int rc =
          hexkl_micro_hmx_rm_to_wh_i8(vtcm_base, 0u, kv->stage_kt, d, 0u, 32u);
        if (rc != AEE_SUCCESS) {
          return rc;
        }
        memcpy(kv->kt + hexkl_kv_q_tile_off(kv, n, c, d), vtcm_base, tb);
        rc = hexkl_micro_hmx_rm_to_wh_i8(vtcm_base, tb, kv->stage_v, 0u, d, hd);
        if (rc != AEE_SUCCESS) {
          return rc;
        }
        memcpy(kv->v + hexkl_kv_q_tile_off(kv, n, c, d), vtcm_base + tb, tb);
      }
      if (st) {
        st->stage_us += (uint32_t)(ts1 - ts0);
        st->bake_us += (uint32_t)(kvq_now_us() - ts1);
      }
    }
  }
  return AEE_SUCCESS;
}
#else
const uint16_t *hexkl_kv_q_wh_i8_pos(uint8_t *vtcm_base) {
  (void)vtcm_base;
  return NULL;
}
#endif

/**
 * @brief Re-bakes the K^T and V tiles of column tile c of head n from the
 *        staging through a VTCM scratch tile. DSP only: needs HexKL.
 */
static int bake_col_tile(hexkl_kv_q *kv, uint32_t n, uint32_t c,
                         uint8_t *vtcm_base, hexkl_kv_q_append_stats *st) {
#ifdef __hexagon__
  const uint64_t t0 = kvq_now_us();
  hexkl_kv_q_stage(kv, n, c);
  const uint64_t t1 = kvq_now_us();
  if (st) {
    st->stage_us += (uint32_t)(t1 - t0);
  }
  const uint32_t tb = kv->tile_bytes;
  for (uint32_t d = 0; d < kv->n_dot_tiles; ++d) {
    int rc;
    // K^T is [head_dim][32]: reduction rows are dims, columns cache rows;
    // tile (d, 0) of a 32-column matrix.
    if (kv->kind == HEXKL_KV_Q4) {
      rc = hexkl_micro_hmx_rm_to_wh_i4(vtcm_base, 0u, kv->stage_kt, d, 0u, 32u);
    } else {
      rc = hexkl_micro_hmx_rm_to_wh_i8(vtcm_base, 0u, kv->stage_kt, d, 0u, 32u);
    }
    if (rc != AEE_SUCCESS) {
      return rc;
    }
    memcpy(kv->kt + hexkl_kv_q_tile_off(kv, n, c, d), vtcm_base, tb);
    // V is [32][head_dim]: reduction rows are cache rows; tile (0, d).
    if (kv->kind == HEXKL_KV_Q4) {
      rc = hexkl_micro_hmx_rm_to_wh_i4(vtcm_base, tb, kv->stage_v, 0u, d,
                                       kv->head_dim);
    } else {
      rc = hexkl_micro_hmx_rm_to_wh_i8(vtcm_base, tb, kv->stage_v, 0u, d,
                                       kv->head_dim);
    }
    if (rc != AEE_SUCCESS) {
      return rc;
    }
    memcpy(kv->v + hexkl_kv_q_tile_off(kv, n, c, d), vtcm_base + tb, tb);
  }
  if (st) {
    st->bake_us += (uint32_t)(kvq_now_us() - t1);
  }
  return AEE_SUCCESS;
#else
  (void)kv;
  (void)n;
  (void)c;
  (void)vtcm_base;
  (void)st;
  return AEE_EFAILED;
#endif
}

int hexkl_kv_q_append(hexkl_kv_q_table *tbl, uint32_t handle, uint32_t row0,
                      uint32_t n_rows, const uint16_t *k_rows,
                      const uint16_t *v_rows, uint8_t *vtcm_base,
                      hexkl_kv_q_append_stats *st) {
  hexkl_kv_q *kv = (hexkl_kv_q *)hexkl_kv_q_get(tbl, handle);
  if (!kv || !k_rows || !v_rows || n_rows == 0 ||
      row0 + n_rows > kv->max_rows || row0 + n_rows < row0) {
    return AEE_EBADPARM;
  }
  if (st) {
    memset(st, 0, sizeof(*st));
  }
#ifdef __hexagon__
  if (kv->plain_masters) {
    return append_fixed_i8(kv, row0, n_rows, k_rows, v_rows, vtcm_base, st);
  }
#endif
  const uint64_t tq0 = kvq_now_us();
  const uint32_t hd = kv->head_dim;
  const uint32_t stride = kv->n_head_kv * hd;
#ifndef __hexagon__
  float x[HEXKL_KV_Q_MAX_HEAD_DIM];
#endif
  int8_t q[HEXKL_KV_Q_MAX_HEAD_DIM];
  int8_t qk_row[HEXKL_KV_Q_MAX_HEAD_DIM];
  float sv[8];
  int direct = 0;
#ifdef __hexagon__
  direct = vtcm_base && kv->kind == HEXKL_KV_Q8 && wh_i8_probe(vtcm_base);
#endif

  for (uint32_t r = 0; r < n_rows; ++r) {
    const uint32_t row = row0 + r;
    const uint16_t *krow = k_rows + (size_t)r * stride;
    const uint16_t *vrow = v_rows + (size_t)r * stride;
    for (uint32_t n = 0; n < kv->n_head_kv; ++n) {
      float sk;
      int32_t cs;
      const float *inv_v = kv->fs_v_inv + (size_t)n * hd;
      if (kv->fixed) {
        sk = kv->fs_k[n];
      }
#ifdef __hexagon__
      if (kv->fixed) {
        hvx_kv_quant_k_row_fixed(krow + (size_t)n * hd, hd, kv->qmax, 1.0f / sk,
                                 qk_row, &cs);
      } else {
        hvx_kv_quant_k_row(krow + (size_t)n * hd, hd, kv->qmax, qk_row, &sk,
                           &cs);
      }
#else
      for (uint32_t d = 0; d < hd; ++d) {
        x[d] = hexkl_kv_q_hf_to_f32(krow[(size_t)n * hd + d]);
      }
      if (kv->fixed) {
        hexkl_kv_q_quant_k_row_fixed(x, hd, kv->qmax, 1.0f / sk, qk_row, &cs);
      } else {
        hexkl_kv_q_quant_k_row(x, hd, kv->qmax, qk_row, &sk, &cs);
      }
#endif
      kv->s_k[hexkl_kv_q_sk_index(kv, n, row)] = sk;
      kv->colsum_k[hexkl_kv_q_sk_index(kv, n, row)] = cs;
      for (uint32_t d = 0; d < hd; ++d) {
        kv->kt4[hexkl_kv_q_kt4_index(kv, n, row, d)] =
          (uint8_t)(qk_row[d] + HEXKL_KV_Q_BIAS);
      }

#ifdef __hexagon__
      if (kv->fixed) {
        hvx_kv_quant_v_row_fixed(vrow + (size_t)n * hd, hd, kv->qmax, inv_v, q);
        sv[0] = 1.0f;
      } else {
        hvx_kv_quant_v_row(vrow + (size_t)n * hd, hd, kv->qmax, q, sv);
      }
#else
      for (uint32_t d = 0; d < hd; ++d) {
        x[d] = hexkl_kv_q_hf_to_f32(vrow[(size_t)n * hd + d]);
      }
      if (kv->fixed) {
        hexkl_kv_q_quant_v_row_fixed(x, hd, kv->qmax, inv_v, q);
        sv[0] = 1.0f;
      } else {
        hexkl_kv_q_quant_v_row(x, hd, kv->qmax, q, sv);
      }
#endif
      kv->s_v[hexkl_kv_q_sv_index(kv, n, row)] = sv[0];
      for (uint32_t d = 0; d < hd; ++d) {
        kv->v4[hexkl_kv_q_v4_index(kv, n, row, d)] =
          (uint8_t)(q[d] + HEXKL_KV_Q_BIAS);
      }
#ifdef __hexagon__
      if (direct) {
        write_row_i8_tiles(kv, n, row, qk_row, q);
      }
#endif
    }
  }

  if (st) {
    st->quant_us = (uint32_t)(kvq_now_us() - tq0);
  }
  if (!vtcm_base || direct) {
    return AEE_SUCCESS;
  }
  const uint32_t c_lo = row0 / 32u;
  const uint32_t c_hi = (row0 + n_rows - 1u) / 32u;
  for (uint32_t n = 0; n < kv->n_head_kv; ++n) {
    for (uint32_t c = c_lo; c <= c_hi; ++c) {
      const int rc = bake_col_tile(kv, n, c, vtcm_base, st);
      if (rc != AEE_SUCCESS) {
        return rc;
      }
    }
  }
  return AEE_SUCCESS;
}

int hexkl_kv_q_dump(const hexkl_kv_q *kv, uint32_t row0, uint32_t n_rows,
                    int8_t *k_q, int8_t *v_q, float *s_k, int32_t *colsum_k,
                    float *s_v) {
  if (!kv || n_rows == 0 || row0 + n_rows > kv->max_rows ||
      row0 + n_rows < row0) {
    return AEE_EBADPARM;
  }
  const uint32_t hd = kv->head_dim;
  const uint32_t stride = kv->n_head_kv * hd;
  for (uint32_t r = 0; r < n_rows; ++r) {
    const uint32_t row = row0 + r;
    for (uint32_t n = 0; n < kv->n_head_kv; ++n) {
      if (k_q) {
        for (uint32_t d = 0; d < hd; ++d) {
          k_q[(size_t)r * stride + (size_t)n * hd + d] =
            (int8_t)((int)kv->kt4[hexkl_kv_q_kt4_index(kv, n, row, d)] -
                     HEXKL_KV_Q_BIAS);
        }
      }
      if (v_q) {
        for (uint32_t d = 0; d < hd; ++d) {
          v_q[(size_t)r * stride + (size_t)n * hd + d] =
            (int8_t)((int)kv->v4[hexkl_kv_q_v4_index(kv, n, row, d)] -
                     HEXKL_KV_Q_BIAS);
        }
      }
      if (s_k) {
        s_k[(size_t)r * kv->n_head_kv + n] =
          kv->s_k[hexkl_kv_q_sk_index(kv, n, row)];
      }
      if (colsum_k) {
        colsum_k[(size_t)r * kv->n_head_kv + n] =
          kv->colsum_k[hexkl_kv_q_sk_index(kv, n, row)];
      }
      if (s_v) {
        s_v[(size_t)r * kv->n_head_kv + n] =
          kv->s_v[hexkl_kv_q_sv_index(kv, n, row)];
      }
    }
  }
  return AEE_SUCCESS;
}
