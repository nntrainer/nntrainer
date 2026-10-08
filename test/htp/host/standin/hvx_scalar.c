// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hvx_scalar.c
 * @date   27 Sep 2026
 * @brief  Scalar stand-ins for the HMX tile and the intrinsic-heavy HVX
 *         kernels, shared by the host checks and the in-process HTP build
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 */

#include "hvx_scalar.h"

#include "hexkl_micro.h"
#include "hvx_dequant_i32.h"
#include "hvx_expand_i2i4.h"
#include "hvx_gemm_u8i4_wh.h"
#include "hvx_quant_u8.h"
#include "hvx_swiglu_f32.h"
#include "swiglu_det.h"
#include <math.h>
#include <stddef.h>
#include <string.h>

hvx_scalar_hooks hvx_scalar_hook = {NULL, NULL, NULL, NULL};

/** @brief Reports a buffer access to a check's hook, if any. */
static void buf(const void *p, size_t bytes, int write) {
  if (hvx_scalar_hook.buf && bytes != 0u)
    hvx_scalar_hook.buf(p, bytes, write);
}

/* ---- the HMX accumulator: one 64 x 32 int32 tile ---- */
static int32_t g_acc[64][32];

int hexkl_micro_hmx_acc_clear_int32(void) {
  memset(g_acc, 0, sizeof g_acc);
  return 0;
}

int wh_value(const uint8_t *tile, uint32_t k, uint32_t c) {
  const uint32_t byte = (k / 8u) * 128u + c * 4u + (k % 4u);
  const int nib = (tile[byte] >> (((k / 4u) % 2u) ? 4 : 0)) & 0xF;
  return nib >= 8 ? nib - 16 : nib;
}

/* Weight tile: 512 bytes = 32k x 32n int4 in the device's WH layout,
   two's complement nibbles. Activation tile: 64 rows x 32 bytes u8. */
int hexkl_micro_hmx_mm_u8i4(uint8_t *base, uint32_t act_off, uint32_t w_off) {
  const uint8_t *a = base + act_off;
  const uint8_t *w = base + w_off;
  for (int r = 0; r < 64; ++r)
    for (uint32_t c = 0; c < 32; ++c) {
      int32_t s = 0;
      for (uint32_t k = 0; k < 32; ++k)
        s += (int32_t)a[r * 32 + k] * wh_value(w, k, c);
      g_acc[r][c] += s;
    }
  return 0;
}

int hexkl_micro_hmx_acc_read_int32(uint8_t *base, uint32_t cfg, uint32_t off) {
  (void)cfg;
  memcpy(base + off, g_acc, sizeof g_acc);
  return 0;
}

/* ---- the u8i4 GEMV: the same sum as the tile above, over the tiles the
   kernel points it at. rows1 picks between two HVX loops that compute
   the same int32 sums (hvx_gemm_u8i4_wh.c), so it is one function here;
   the l2fetch entry point is a no-op unless a check hooks it. ---- */
void hvx_scalar_gemv(const uint8_t *act_ah, uint32_t m, uint32_t k_tiles,
                     const uint8_t *wh, uint32_t n_col, uint32_t nt,
                     int32_t *out) {
  if (m != 0u && k_tiles != 0u)
    buf(act_ah, (size_t)(k_tiles - 1u) * 2048u + (size_t)m * 32u, 0);
  for (uint32_t r = 0; r < m; ++r)
    for (uint32_t c = 0; c < 32; ++c) {
      int32_t s = 0;
      for (uint32_t kt = 0; kt < k_tiles; ++kt) {
        const uint8_t *tile = wh + ((size_t)kt * n_col + nt) * 512u;
        const uint8_t *arow = act_ah + (size_t)kt * 2048u + r * 32u;
        for (uint32_t k = 0; k < 32; ++k)
          s += (int32_t)arow[k] * wh_value(tile, k, c);
      }
      out[r * 32u + c] = s;
    }
}
void hvx_gemm_u8i4_wh_prefetch(const uint8_t *wh, uint32_t n_col, uint32_t nt,
                               uint32_t n_tiles, uint32_t k_tiles) {
  if (hvx_scalar_hook.prefetch)
    hvx_scalar_hook.prefetch(wh, n_col, nt, n_tiles, k_tiles);
}
void hvx_gemm_u8i4_wh_col_nopf(const uint8_t *act_ah, uint32_t m,
                               uint32_t k_tiles, const uint8_t *wh,
                               uint32_t n_col, uint32_t nt, uint32_t rows1,
                               int32_t *out) {
  if (hvx_scalar_hook.gemv)
    hvx_scalar_hook.gemv(act_ah, m, k_tiles, wh, n_col, nt, rows1, 1, out);
  else
    hvx_scalar_gemv(act_ah, m, k_tiles, wh, n_col, nt, out);
}
void hvx_gemm_u8i4_wh_col(const uint8_t *act_ah, uint32_t m, uint32_t k_tiles,
                          const uint8_t *wh, uint32_t n_col, uint32_t nt,
                          uint32_t rows1, int32_t *out) {
  if (hvx_scalar_hook.gemv)
    hvx_scalar_hook.gemv(act_ah, m, k_tiles, wh, n_col, nt, rows1, 0, out);
  else
    hvx_scalar_gemv(act_ah, m, k_tiles, wh, n_col, nt, out);
}

static int wh2_value(const uint8_t *tile, uint32_t k, uint32_t c,
                     const uint8_t *table) {
  const uint32_t slot =
    (k / 8u) * 256u + c * 8u + (k % 4u) * 2u + ((k / 4u) % 2u);
  const uint32_t code = (tile[whCodeByte2(slot)] >> whCodeShift2(slot)) & 0x3u;
  const int nibble = table[2u * code] & 0x0f;
  return nibble >= 8 ? nibble - 16 : nibble;
}

void hvx_scalar_gemv_i2(const uint8_t *act_ah, uint32_t m, uint32_t k_tiles,
                        const uint8_t *wh, uint32_t n_col, uint32_t nt,
                        const uint8_t *table, int32_t *out) {
  if (m != 0u && k_tiles != 0u)
    buf(act_ah, (size_t)(k_tiles - 1u) * 2048u + (size_t)m * 32u, 0);
  for (uint32_t r = 0; r < m; ++r)
    for (uint32_t c = 0; c < 32; ++c) {
      int32_t s = 0;
      for (uint32_t kt = 0; kt < k_tiles; ++kt) {
        const uint8_t *tile = wh + ((size_t)kt * n_col + nt) * 256u;
        const uint8_t *arow = act_ah + (size_t)kt * 2048u + r * 32u;
        for (uint32_t k = 0; k < 32; ++k)
          s += (int32_t)arow[k] * wh2_value(tile, k, c, table);
      }
      out[r * 32u + c] = s;
    }
}

/** @brief One u8i2 column, through the check's hook when there is one. */
static void gemv_i2_col(const uint8_t *act_ah, uint32_t m, uint32_t k_tiles,
                        const uint8_t *wh, uint32_t n_col, uint32_t nt,
                        uint32_t rows1, int nopf, const uint8_t *table,
                        int32_t *out) {
  if (hvx_scalar_hook.gemv2)
    hvx_scalar_hook.gemv2(act_ah, m, k_tiles, wh, n_col, nt, rows1, nopf, table,
                          out);
  else
    hvx_scalar_gemv_i2(act_ah, m, k_tiles, wh, n_col, nt, table, out);
}

void hvx_gemm_u8i2_wh_prefetch(const uint8_t *wh, uint32_t n_col, uint32_t nt,
                               uint32_t n_tiles, uint32_t k_tiles) {
  if (hvx_scalar_hook.prefetch)
    hvx_scalar_hook.prefetch(wh, n_col, nt, n_tiles, k_tiles);
}

void hvx_gemm_u8i2_wh_col_nopf(const uint8_t *act_ah, uint32_t m,
                               uint32_t k_tiles, const uint8_t *wh,
                               uint32_t n_col, uint32_t nt, uint32_t rows1,
                               const uint8_t *table, int32_t *out) {
  gemv_i2_col(act_ah, m, k_tiles, wh, n_col, nt, rows1, 1, table, out);
}

void hvx_gemm_u8i2_wh_cols2_nopf(const uint8_t *act_ah, uint32_t m,
                                 uint32_t k_tiles, const uint8_t *wh,
                                 uint32_t n_col, uint32_t nt0, uint32_t nt1,
                                 uint32_t rows1, const uint8_t *table,
                                 int32_t *out0, int32_t *out1) {
  gemv_i2_col(act_ah, m, k_tiles, wh, n_col, nt0, rows1, 1, table, out0);
  gemv_i2_col(act_ah, m, k_tiles, wh, n_col, nt1, rows1, 1, table, out1);
}

void hvx_gemm_u8i2_wh_col(const uint8_t *act_ah, uint32_t m, uint32_t k_tiles,
                          const uint8_t *wh, uint32_t n_col, uint32_t nt,
                          uint32_t rows1, const uint8_t *table, int32_t *out) {
  gemv_i2_col(act_ah, m, k_tiles, wh, n_col, nt, rows1, 0, table, out);
}

/* ---- quant / dequant / swiglu ---- */
void hvx_quant_rows_u8_params(const float *x, uint32_t m, uint32_t mp,
                              uint32_t k, float *scale, int32_t *zp,
                              hvx_worker_pool *p) {
  (void)p;
  buf(x, sizeof(float) * (size_t)m * k, 0);
  buf(scale, sizeof(float) * mp, 1);
  buf(zp, sizeof(int32_t) * mp, 1);
  for (uint32_t r = 0; r < mp; ++r) {
    float lo = 0.f, hi = 0.f;
    if (r < m)
      for (uint32_t j = 0; j < k; ++j) {
        float v = x[(size_t)r * k + j];
        if (v < lo)
          lo = v;
        if (v > hi)
          hi = v;
      }
    float s = (hi - lo) / 255.f;
    if (s <= 0.f)
      s = 1e-8f;
    scale[r] = s;
    long z = lrintf(-lo / s);
    if (z < 0)
      z = 0;
    if (z > 255)
      z = 255;
    zp[r] = (int32_t)z;
  }
}
/* A row range, the one scalar formula the two entry points below share so
   they cannot drift: the kernel's units are 16-row quarters of a block.
   Tiles run (row_block, inner_tile) at a 2048-byte stride, so a caller
   passing more than 64 rows writes several row blocks. */
void hvx_quant_pack_u8_ah_rows(const float *x, const uint32_t *map, uint32_t m0,
                               uint32_t m1, uint32_t k, const float *scale,
                               const int32_t *zp, uint8_t *out) {
  const uint32_t kt_n = k / 32u;
  if (m1 > m0) {
    for (uint32_t r = m0; r < m1; ++r)
      buf(x + (map ? map[r] : r) * (size_t)k, sizeof(float) * k, 0);
    buf(scale + m0, sizeof(float) * (m1 - m0), 0);
    buf(zp + m0, sizeof(int32_t) * (m1 - m0), 0);
    /* The whole 64-row blocks the rows fall in: over-covers, never under. */
    buf(out + (size_t)(m0 / 64u) * kt_n * 2048u,
        (size_t)((m1 + 63u) / 64u - m0 / 64u) * kt_n * 2048u, 1);
  }
  for (uint32_t r = m0; r < m1; ++r)
    for (uint32_t kt = 0; kt < kt_n; ++kt)
      for (uint32_t j = 0; j < 32; ++j) {
        const size_t sr = map ? map[r] : r;
        long q = lrintf(x[sr * k + kt * 32 + j] / scale[r]) + zp[r];
        if (q < 0)
          q = 0;
        if (q > 255)
          q = 255;
        out[(size_t)(r / 64u) * kt_n * 2048u + (size_t)kt * 2048u +
            (size_t)(r % 64u) * 32u + j] = (uint8_t)q;
      }
}
int hvx_quant_pack_u8_ah_mapped(const float *x, const uint32_t *map, uint32_t m,
                                uint32_t mp, uint32_t k, const float *scale,
                                const int32_t *zp, uint8_t *out,
                                hvx_worker_pool *p) {
  (void)p;
  memset(out, 0, (size_t)mp * k);
  hvx_quant_pack_u8_ah_rows(x, map, 0u, m, k, scale, zp, out);
  return 0;
}
int hvx_quant_pack_u8_ah(const float *x, uint32_t m, uint32_t mp, uint32_t k,
                         const float *scale, const int32_t *zp, uint8_t *out,
                         hvx_worker_pool *p) {
  return hvx_quant_pack_u8_ah_mapped(x, NULL, m, mp, k, scale, zp, out, p);
}

/* The tile dequant without the buffer hook: the pooled workers below call
   it on their own stack tiles, which lanes run one after another on the
   host share, so reporting those would be a false cross-lane race. */
static void dq_tile(const int32_t *tile, uint32_t stride, uint32_t m,
                    const float *as, const int32_t *az, const int32_t *cs,
                    const float *ws, const float *bias, float *out,
                    uint32_t ostride, int accumulate) {
  for (uint32_t r = 0; r < m; ++r)
    for (uint32_t c = 0; c < 32; ++c) {
      float v = ((float)(tile[(size_t)r * stride + c] - az[r] * cs[c])) *
                  as[r] * ws[c] +
                bias[c];
      if (accumulate)
        out[(size_t)r * ostride + c] += v;
      else
        out[(size_t)r * ostride + c] = v;
    }
}
void hvx_dequant_acc_tile_to_f32(const int32_t *tile, uint32_t stride,
                                 uint32_t m, const float *as, const int32_t *az,
                                 const int32_t *cs, const float *ws,
                                 const float *bias, float *out,
                                 uint32_t ostride, int accumulate) {
  buf(as, sizeof(float) * m, 0);
  buf(az, sizeof(int32_t) * m, 0);
  for (uint32_t r = 0; r < m; ++r)
    buf(out + (size_t)r * ostride, sizeof(float) * 32u, 1);
  dq_tile(tile, stride, m, as, az, cs, ws, bias, out, ostride, accumulate);
}
/* The DDR fallback's whole-matrix dequant (accumulator layout unusable):
   the same formula per element, row stride n. */
void hvx_dequant_i32_to_f32(const int32_t *acc, uint32_t m_valid,
                            uint32_t m_pad, uint32_t n, const float *act_scale,
                            const int32_t *act_zp, const int32_t *colsum_w,
                            const float *w_scale, const float *bias, float *out,
                            int accumulate) {
  (void)m_pad;
  for (uint32_t r = 0; r < m_valid; ++r)
    for (uint32_t c = 0; c < n; ++c) {
      float v = ((float)(acc[(size_t)r * n + c] - act_zp[r] * colsum_w[c])) *
                  act_scale[r] * w_scale[c] +
                bias[c];
      if (accumulate)
        out[(size_t)r * n + c] += v;
      else
        out[(size_t)r * n + c] = v;
    }
}

/* The pooled workers: each a loop over the per-tile stand-in above,
   exactly as the real ones are pooled loops over the real per-tile
   kernel, so what a check sees is the kernel's batching arithmetic --
   which tile lands at which staged slot, which column it carries -- and
   not a second dequant. Unit i of n_threads takes its slice, as the real
   workers do; a check's serial pool stand-in hands every slice over. */
static void slice(uint32_t n, uint32_t n_threads, uint32_t i, uint32_t *lo,
                  uint32_t *hi) {
  *lo = (uint32_t)((uint64_t)n * i / n_threads);
  *hi = (uint32_t)((uint64_t)n * (i + 1u) / n_threads);
}
void hvx_dq_tiles_worker(uint32_t n_threads, uint32_t i, void *vjob) {
  const hvx_dq_tiles_job *c = (const hvx_dq_tiles_job *)vjob;
  uint32_t lo, hi;
  slice(c->n_tiles, n_threads, i, &lo, &hi);
  for (uint32_t j = lo; j < hi; ++j) {
    const uint32_t c0 = (c->nt0 + j) * 32u;
    const int32_t *tile =
      (const int32_t *)(c->tiles_base + (size_t)j * c->tile_stride);
    float *out =
      (c0 < c->split) ? (c->dst_a + c0) : (c->dst_b + (c0 - c->split));
    hvx_dequant_acc_tile_to_f32(tile, c->row_stride, c->m_count, c->act_scale,
                                c->act_zp, c->colsum_w + c0, c->w_scale + c0,
                                c->bias + c0, out, c->dst_stride, 0);
  }
}
void hvx_dequant_acc_tiles_to_f32(
  const uint8_t *tiles_base, uint32_t tile_stride, uint32_t n_tiles,
  uint32_t nt0, uint32_t row_stride, uint32_t m_count, const float *act_scale,
  const int32_t *act_zp, const int32_t *colsum_w, const float *w_scale,
  const float *bias, float *dst_a, float *dst_b, uint32_t split,
  uint32_t dst_stride, hvx_worker_pool *pool) {
  (void)pool;
  if (!tiles_base || n_tiles == 0u || m_count == 0u)
    return;
  hvx_dq_tiles_job jb = {tiles_base, tile_stride, nt0,    row_stride,
                         m_count,    act_scale,   act_zp, colsum_w,
                         w_scale,    bias,        dst_a,  dst_b,
                         split,      dst_stride,  n_tiles};
  hvx_dq_tiles_worker(1u, 0u, &jb);
}
/* Fused gate/up dequant + SwiGLU: staged slot j is gate column g0 + j and
   slot n_pairs + j the up column opposite it. The SwiGLU itself is
   swiglu_det.h, which the HVX kernel matches bit for bit (rule 24); with
   the job's geglu set, geglu_det_one (plan 201 S4). */
void hvx_dq_swiglu_worker(uint32_t n_threads, uint32_t i, void *vjob) {
  const hvx_dq_swiglu_job *c = (const hvx_dq_swiglu_job *)vjob;
  float gt[64 * 32], ut[64 * 32];
  uint32_t lo, hi;
  slice(c->n_pairs, n_threads, i, &lo, &hi);
  if (lo < hi) {
    buf(c->act_scale, sizeof(float) * c->m_count, 0);
    buf(c->act_zp, sizeof(int32_t) * c->m_count, 0);
    for (uint32_t r = 0; r < c->m_count; ++r)
      buf(c->dst + (size_t)r * c->dst_stride + (c->g0 + lo) * 32u,
          sizeof(float) * (hi - lo) * 32u, 1);
  }
  for (uint32_t j = lo; j < hi; ++j) {
    const uint32_t cg = (c->g0 + j) * 32u, cu = c->inter + cg;
    dq_tile((const int32_t *)(c->tiles_base + (size_t)j * c->tile_stride),
            c->row_stride, c->m_count, c->act_scale, c->act_zp,
            c->colsum_w + cg, c->w_scale + cg, c->bias + cg, gt, 32u, 0);
    dq_tile((const int32_t *)(c->tiles_base +
                              (size_t)(c->n_pairs + j) * c->tile_stride),
            c->row_stride, c->m_count, c->act_scale, c->act_zp,
            c->colsum_w + cu, c->w_scale + cu, c->bias + cu, ut, 32u, 0);
    for (uint32_t r = 0; r < c->m_count; ++r)
      for (uint32_t k = 0; k < 32u; ++k)
        c->dst[(size_t)r * c->dst_stride + cg + k] =
          c->act == HVX_GLU_GELU_TANH
            ? geglu_det_one(gt[r * 32u + k], ut[r * 32u + k])
            : swiglu_det_one(gt[r * 32u + k], ut[r * 32u + k]);
  }
}
/* The tail path calls the pooled pair function directly (pool NULL, one
   pair); it is the job above run synchronously, as on device. */
void hvx_dequant_swiglu_acc_tiles_to_f32(
  const uint8_t *tiles_base, uint32_t tile_stride, uint32_t n_pairs,
  uint32_t g0, uint32_t row_stride, uint32_t m_count, const float *act_scale,
  const int32_t *act_zp, const int32_t *colsum_w, const float *w_scale,
  const float *bias, uint32_t inter, uint32_t act, float *dst,
  uint32_t dst_stride, hvx_worker_pool *pool) {
  (void)pool;
  hvx_dq_swiglu_job jb;
  jb.tiles_base = tiles_base;
  jb.tile_stride = tile_stride;
  jb.n_pairs = n_pairs;
  jb.g0 = g0;
  jb.row_stride = row_stride;
  jb.m_count = m_count;
  jb.act_scale = act_scale;
  jb.act_zp = act_zp;
  jb.colsum_w = colsum_w;
  jb.w_scale = w_scale;
  jb.bias = bias;
  jb.inter = inter;
  jb.dst = dst;
  jb.dst_stride = dst_stride;
  jb.act = act;
  hvx_dq_swiglu_worker(1u, 0u, &jb);
}
/* Dequant + product pair job (the conv block's pre-conv gate): slot j is
   the first weight's column c0 + 32 j, slot n_pairs + j the second's. */
void hvx_dq_mul_worker(uint32_t n_threads, uint32_t i, void *vjob) {
  const hvx_dq_mul_job *c = (const hvx_dq_mul_job *)vjob;
  float at[64 * 32], bt[64 * 32];
  uint32_t lo, hi;
  slice(c->n_pairs, n_threads, i, &lo, &hi);
  for (uint32_t j = lo; j < hi; ++j) {
    const uint32_t col = c->c0 + j * 32u;
    dq_tile((const int32_t *)(c->tiles_base + (size_t)j * c->tile_stride),
            c->row_stride, c->m_count, c->act_scale, c->act_zp,
            c->colsum_a + col, c->w_scale_a + col, c->bias_a + col, at, 32u, 0);
    dq_tile((const int32_t *)(c->tiles_base +
                              (size_t)(c->n_pairs + j) * c->tile_stride),
            c->row_stride, c->m_count, c->act_scale, c->act_zp,
            c->colsum_b + col, c->w_scale_b + col, c->bias_b + col, bt, 32u, 0);
    for (uint32_t r = 0; r < c->m_count; ++r)
      for (uint32_t k = 0; k < 32u; ++k)
        c->dst[(size_t)r * c->dst_stride + col + k] =
          at[r * 32u + k] * bt[r * 32u + k];
  }
}
/* The unfused SwiGLU (the FC fused path): the spec, element by element. */
void hvx_swiglu_inplace_f32(float *gate, const float *up, uint32_t m_valid,
                            uint32_t n_out, hvx_worker_pool *pool) {
  (void)pool;
  for (size_t j = 0, n = (size_t)m_valid * n_out; j < n; ++j)
    gate[j] = swiglu_det_one(gate[j], up[j]);
}
