// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd.
 *
 * @file hexkl_integer.h
 * @brief Integer FC accumulator assembly and fixed-point requantization
 * @bug No known bugs except for NYI items
 */
#ifndef __NNTRAINER_HEXKL_INTEGER_H__
#define __NNTRAINER_HEXKL_INTEGER_H__

#include "hvx_worker_pool.h"
#include <AEEStdErr.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/** @brief Output code = clamp(round_away(acc * multiplier / 2^shift) + zp).
 * The caller prepares multiplier/shift offline from s_a * s_w / s_out.
 * Ties round away from zero; multiplier is in [0, INT32_MAX], shift in
 * [0, 62]. All runtime arithmetic is integer. */
typedef struct hexkl_requant_u16 {
  uint32_t multiplier;
  uint32_t shift;
  uint16_t zero_point;
} hexkl_requant_u16;

/** @brief One staged FC batch; each worker owns disjoint output rows.
 * Bias is in accumulator units, round(bias_real / (s_a * s_w[c])). The
 * weight registry's float scale/bias are deliberately not read. */
typedef struct {
  const uint8_t *tiles;
  // plane_count: 3/4 byte planes, or 0 for row-major INT32 tiles.
  uint32_t tile_stride, row_stride, lo_off, plane_count;
  uint32_t n_tiles, nt0, rows, cols;
  uint32_t sum_fits_i32; // Proven uncorrected UINT16 dot bound, before bias.
  uint16_t zero_point;
  const int32_t *colsum, *bias;
  int32_t *out;
  const hexkl_requant_u16 *requant;
  uint32_t n_requant;
  uint16_t *out_u16;
  uint8_t *ah_lo, *ah_hi;
  uint32_t ah_cols, ah_col, ah_row;
  uint8_t overflow[64];
} hexkl_i32_rows_job;

/** @brief Scalar specification, also used by short tails. */
static inline uint16_t hexkl_requant_one(int32_t x,
                                         const hexkl_requant_u16 *p) {
  const int64_t v = (int64_t)x * p->multiplier;
  const uint64_t mag = (uint64_t)(v < 0 ? -v : v);
  const uint64_t rounded =
    p->shift ? (mag + (UINT64_C(1) << (p->shift - 1u))) >> p->shift : mag;
  const int64_t q =
    (v < 0 ? -(int64_t)rounded : (int64_t)rounded) + p->zero_point;
  return (uint16_t)(q < 0 ? 0 : q > UINT16_MAX ? UINT16_MAX : q);
}

#if defined(__hexagon__) && !defined(HEXKL_INTEGER_SCALAR)
#include "hexkl_integer_hvx.h"
#endif

/** @brief Validate before any worker writes an output. */
static inline int hexkl_requant_validate(const int32_t *src, uint16_t *dst,
                                         size_t rows, uint32_t cols,
                                         const hexkl_requant_u16 *params,
                                         uint32_t n_params) {
  if (!src || !dst || !params || !rows || !cols ||
      (n_params != 1u && n_params != cols) ||
      rows > SIZE_MAX / sizeof(int32_t) / cols)
    return AEE_EBADPARM;
  for (uint32_t c = 0; c < n_params; ++c)
    if (params[c].multiplier > INT32_MAX || params[c].shift > 62u)
      return AEE_EBADPARM;
  return AEE_SUCCESS;
}

/** @brief Requantizes a row-major INT32 matrix to asymmetric UINT16.
 * params has either one entry (per tensor) or cols entries (per output
 * channel). Buffers must not overlap. Invalid params leave dst untouched.
 * Returns AEE_SUCCESS or AEE_EBADPARM. Saturation here is intentional;
 * overflow in the FC accumulator itself is reported instead. */
static inline int hexkl_requantize_i32_u16(const int32_t *src, uint16_t *dst,
                                           size_t rows, uint32_t cols,
                                           const hexkl_requant_u16 *params,
                                           uint32_t n_params) {
  if (hexkl_requant_validate(src, dst, rows, cols, params, n_params) !=
      AEE_SUCCESS)
    return AEE_EBADPARM;
  for (size_t r = 0; r < rows; ++r) {
    uint32_t c = 0;
#if defined(__hexagon__) && !defined(HEXKL_INTEGER_SCALAR)
    c = hexkl_requant_row_hvx(src + r * cols, dst + r * cols, cols, params,
                              n_params);
#endif
    for (; c < cols; ++c) {
      const hexkl_requant_u16 *p = params + (n_params == 1u ? 0u : c);
      dst[r * cols + c] = hexkl_requant_one(src[r * cols + c], p);
    }
  }
  return AEE_SUCCESS;
}

typedef struct {
  const int32_t *src;
  uint16_t *dst;
  size_t rows;
  uint32_t cols;
  const hexkl_requant_u16 *params;
  uint32_t n_params;
} hexkl_requant_job;

static inline void hexkl_requant_worker(uint32_t threads, uint32_t lane,
                                        void *ctx) {
  const hexkl_requant_job *j = (const hexkl_requant_job *)ctx;
  const size_t chunk = j->rows / threads, extra = j->rows % threads;
  const size_t begin = chunk * lane + (lane < extra ? lane : extra);
  const size_t count = chunk + (lane < extra);
  // All parameters and the full address range were checked before dispatch.
  (void)hexkl_requantize_i32_u16(j->src + begin * j->cols,
                                 j->dst + begin * j->cols, count, j->cols,
                                 j->params, j->n_params);
}

/** @brief Same exact requantization distributed over independent rows.
 * NULL pool runs inline. Invalid parameters leave the entire dst untouched.
 * The pool must not be in use by another caller. */
static inline int
hexkl_requantize_i32_u16_pooled(const int32_t *src, uint16_t *dst, size_t rows,
                                uint32_t cols, const hexkl_requant_u16 *params,
                                uint32_t n_params, hvx_worker_pool *pool) {
  if (hexkl_requant_validate(src, dst, rows, cols, params, n_params) !=
      AEE_SUCCESS)
    return AEE_EBADPARM;
  hexkl_requant_job j = {src, dst, rows, cols, params, n_params};
  hvx_worker_pool_run(pool, hexkl_requant_worker, &j,
                      rows > UINT32_MAX ? UINT32_MAX : (uint32_t)rows);
  return AEE_SUCCESS;
}

/** @brief Read one integer from the HMX readout, with no float conversion. */
static inline int32_t hexkl_i32_tile_value(const uint8_t *p, uint32_t row,
                                           uint32_t col, uint32_t stride,
                                           uint32_t planes) {
  const size_t i = (size_t)row * (planes ? 32u : stride) + col;
  if (planes == 4u) {
    const uint32_t bits = p[i] | ((uint32_t)p[i + 2048u] << 8) |
                          ((uint32_t)p[i + 4096u] << 16) |
                          ((uint32_t)p[i + 6144u] << 24);
    int32_t v;
    memcpy(&v, &bits, sizeof v);
    return v;
  }
  if (planes == 3u) {
    const int32_t hi =
      p[i + 4096u] < 128u ? p[i + 4096u] : (int32_t)p[i + 4096u] - 256;
    return p[i] + 256 * (int32_t)p[i + 2048u] + 65536 * hi;
  }
  int32_t v;
  memcpy(&v, p + i * sizeof v, sizeof v);
  return v;
}

/** @brief Exact integer epilogue distributed over the worker pool.
 * INT64 intermediates allow cancellation with the zero-point correction
 * even when the uncorrected sum exceeds INT32. Never wrap or silently
 * saturate the output: an overflowing row is flagged and the caller fails.
 * Hexagon uses widened HVX words; hosts retain the scalar reference. */
static inline void hexkl_i32_rows_worker(uint32_t threads, uint32_t lane,
                                         void *ctx) {
#if defined(__hexagon__) && !defined(HEXKL_INTEGER_SCALAR)
  hexkl_i32_rows_hvx(threads, lane, ctx);
#else
  hexkl_i32_rows_job *j = (hexkl_i32_rows_job *)ctx;
  const uint32_t begin = (uint32_t)((uint64_t)j->rows * lane / threads);
  const uint32_t end = (uint32_t)((uint64_t)j->rows * (lane + 1u) / threads);
  for (uint32_t r = begin; r < end; ++r) {
    for (uint32_t t = 0; t < j->n_tiles; ++t) {
      const uint8_t *hi = j->tiles + (size_t)t * j->tile_stride;
      const uint8_t *lo = hi + j->lo_off;
      for (uint32_t c = 0; c < 32u; ++c) {
        const uint32_t col = (j->nt0 + t) * 32u + c;
        const int64_t v =
          256LL *
            hexkl_i32_tile_value(hi, r, c, j->row_stride, j->plane_count) +
          hexkl_i32_tile_value(lo, r, c, j->row_stride, j->plane_count) -
          (int64_t)j->zero_point * j->colsum[col] +
          (j->bias ? j->bias[col] : 0);
        if (v < INT32_MIN || v > INT32_MAX) {
          j->overflow[r] = 1;
        } else {
          j->out[(size_t)r * j->cols + col] = (int32_t)v;
          if (j->out_u16)
            j->out_u16[(size_t)r * j->cols + col] = hexkl_requant_one(
              (int32_t)v, j->requant + (j->n_requant == 1u ? 0u : col));
          if (j->ah_lo) {
            const uint32_t row = j->ah_row + r, c0 = j->ah_col + col;
            const size_t d = (size_t)(row / 64u) * 64u * j->ah_cols +
                             (size_t)(c0 / 32u) * 2048u + (row % 64u) * 32u +
                             c0 % 32u;
            const uint16_t q = hexkl_requant_one(
              (int32_t)v, j->requant + (j->n_requant == 1u ? 0u : col));
            j->ah_lo[d] = (uint8_t)q;
            j->ah_hi[d] = (uint8_t)(q >> 8);
          }
        }
      }
    }
  }
#endif
}

/** @brief Several row/handle segments share one accumulator staging bank. */
#define HEXKL_I32_BATCH_JOBS 32u
typedef struct {
  hexkl_i32_rows_job jobs[HEXKL_I32_BATCH_JOBS];
  uint32_t n_jobs;
  uint32_t n_lanes;
  uint8_t failed[64];
} hexkl_i32_batch;

static inline void hexkl_i32_batch_worker(uint32_t threads, uint32_t lane,
                                          void *ctx) {
  hexkl_i32_batch *b = (hexkl_i32_batch *)ctx;
  if (lane == 0u)
    b->n_lanes = threads;
  uint8_t failed = 0;
  uint32_t groups = 0;
  for (uint32_t i = 0; i < b->n_jobs; ++i)
    groups += (b->jobs[i].rows + 3u) / 4u;
  const uint32_t begin = groups * lane / threads;
  const uint32_t end = groups * (lane + 1u) / threads;
  uint32_t base = 0;
  for (uint32_t i = 0; i < b->n_jobs; ++i) {
    hexkl_i32_rows_job *original = &b->jobs[i];
    const uint32_t count = (original->rows + 3u) / 4u;
    const uint32_t lo = begin > base ? begin : base;
    const uint32_t hi = end < base + count ? end : base + count;
    if (lo < hi) {
      hexkl_i32_rows_job view;
      // Copy only the immutable descriptor; flags stay private to this lane.
      memcpy(&view, original, offsetof(hexkl_i32_rows_job, overflow));
      memset(view.overflow, 0, sizeof view.overflow);
      const uint32_t first = 4u * (lo - base);
      view.rows = original->rows - first < 4u * (hi - lo)
                    ? original->rows - first
                    : 4u * (hi - lo);
      view.tiles += original->plane_count
                      ? (size_t)(lo - base) * 128u
                      : (size_t)first * original->row_stride * sizeof(int32_t);
      view.out += (size_t)first * original->cols;
      view.ah_row += first;
      if (view.out_u16)
        view.out_u16 += (size_t)first * original->cols;
      hexkl_i32_rows_worker(1, 0, &view);
      for (uint32_t r = 0; r < view.rows; ++r)
        if (view.overflow[r]) {
          failed = 1;
          break;
        }
    }
    base += count;
  }
  b->failed[lane] = failed;
}

/** @brief Inspect only after the pool has retired this job. */
static inline int hexkl_i32_rows_status(const hexkl_i32_rows_job *j) {
  for (uint32_t r = 0; r < j->rows; ++r)
    if (j->overflow[r])
      return AEE_EFAILED;
  return AEE_SUCCESS;
}

static inline int hexkl_i32_batch_status(const hexkl_i32_batch *b) {
  for (uint32_t i = 0; i < b->n_lanes; ++i)
    if (b->failed[i])
      return AEE_EFAILED;
  return AEE_SUCCESS;
}

#endif
