// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file   quant_u16_host_check.c
 * @date   22 Oct 2026
 * @brief  The REAL a16 activation pack (hvx_quant_params_pack_u16_ah) and
 *         the a16 dequant combine (hvx_dq_tiles_rows_worker with lo_off) on
 *         hvx_emu/, bit for bit against their scalar spec
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The pack: 67 rows (16 groups of four and a 3-row scalar tail, a short
 * last 64-row block) of K = 128 from each source -- f32 rows, u16 rows
 * with one encoding, u16 rows with a head each -- against per-row codes
 * rne(x / s) + zp over [0, 65535], low bytes to one AH plane and high
 * bytes to the other. The combine: (256 f(hi) + f(lo) - f(zp) f(cs)) s ws
 * + b, the kernel's operation order, per element.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hexkl_acc_tile.h"
#include "hvx_dequant_i32.h"
#include "hvx_quant_u8.h"

#define M 67u
#define MP 128u
#define KK 128u
#define HD 64u

static uint32_t rng = 7u;
static uint32_t rnd(void) {
  rng = rng * 1664525u + 1013904223u;
  return rng >> 8;
}

/** The spec: row @a x's parameters and codes, as the kernel's formula. */
static void spec_row(const float *x, float *s, int32_t *z, uint16_t *q) {
  float lo = x[0], hi = x[0];
  for (uint32_t j = 1; j < KK; ++j) {
    lo = x[j] < lo ? x[j] : lo;
    hi = x[j] > hi ? x[j] : hi;
  }
  lo = lo < 0.f ? lo : 0.f;
  hi = hi > 0.f ? hi : 0.f;
  *s = 1.0f;
  *z = 0;
  if (lo != hi) {
    *s = (hi - lo) / 65535.0f;
    long t = (long)nearbyintf(-lo / *s);
    *z = (int32_t)(t < 0 ? 0 : (t > 65535 ? 65535 : t));
  }
  const float inv = 1.0f / *s;
  for (uint32_t j = 0; j < KK; ++j) {
    long v = (long)nearbyintf(x[j] * inv) + *z;
    q[j] = (uint16_t)(v < 0 ? 0 : (v > 65535 ? 65535 : v));
  }
}

static size_t ah(uint32_t m, uint32_t j) {
  return (size_t)(m / 64u) * (KK / 32u) * 2048u + (size_t)(j / 32u) * 2048u +
         (size_t)(m % 64u) * 32u + j % 32u;
}

/** Compares the planes and parameters with the spec's codes @a q. */
static int check(const char *what, const uint8_t *lo, const uint8_t *hi,
                 const float *s, const int32_t *z, uint16_t (*q)[KK],
                 const float *ws, const int32_t *wz) {
  size_t bad = 0, pbad = 0;
  for (uint32_t m = 0; m < MP; ++m) {
    const float es = m < M ? ws[m] : 1.0f;
    const int32_t ez = m < M ? wz[m] : 0;
    pbad += s[m] != es || z[m] != ez;
    for (uint32_t j = 0; j < KK; ++j) {
      const uint16_t e = m < M ? q[m][j] : 0u;
      bad += lo[ah(m, j)] != (e & 0xffu) || hi[ah(m, j)] != (e >> 8);
    }
  }
  printf("%-22s: bytes bad %zu of %u, params bad %zu\n", what, bad,
         2u * MP * KK, pbad);
  return bad != 0 || pbad != 0;
}

int main(void) {
  static float x[M * KK], dq[KK], scratch[4u * KK];
  static uint16_t x16[M * KK], q[MP][KK];
  static uint8_t lo[MP * KK] __attribute__((aligned(128)));
  static uint8_t hi[MP * KK] __attribute__((aligned(128)));
  static float s[MP], ws[M];
  static int32_t z[MP], wz[M];
  int fail = 0;
  for (uint32_t i = 0; i < M * KK; ++i) {
    x[i] = ((float)(rnd() % 20001u) - 9000.0f) / 3000.0f;
    x16[i] = (uint16_t)(rnd() & 0xffffu);
  }

  /* The production integer-only pack, including tails and real pool lanes. */
  hvx_worker_pool *pool = hvx_worker_pool_create(3);
  if (!pool)
    return 1;
  const uint32_t sizes[] = {1, 3, 4, 63, 64, 65, M};
  for (uint32_t threaded = 0; threaded < 2; ++threaded) {
    for (uint32_t shape = 0; shape < sizeof sizes / sizeof sizes[0]; ++shape) {
      const uint32_t rows = sizes[shape];
      memset(lo, 0xa5, sizeof lo);
      memset(hi, 0x5a, sizeof hi);
      int prc =
        hvx_pack_u16_ah(x16, rows, MP, KK, lo, hi, threaded ? pool : NULL);
      size_t bad = 0;
      for (uint32_t m = 0; m < MP; ++m)
        for (uint32_t j = 0; j < KK; ++j) {
          const uint16_t qv = m < rows ? x16[(size_t)m * KK + j] : 0;
          bad += lo[ah(m, j)] != (qv & 255u) || hi[ah(m, j)] != (qv >> 8);
        }
      fail |= prc != 0 || bad != 0;
      // Background units may finish out of order; no group may clear
      // another group's valid rows, including the final partial group.
      memset(lo, 0xa5, sizeof lo);
      memset(hi, 0x5a, sizeof hi);
      for (uint32_t g = MP / 4u; g > 0; --g)
        hvx_pack_u16_ah_group(x16, g - 1u, rows, KK, lo, hi);
      for (uint32_t m = 0; m < MP; ++m)
        for (uint32_t j = 0; j < KK; ++j) {
          const uint16_t qv = m < rows ? x16[(size_t)m * KK + j] : 0;
          fail |= lo[ah(m, j)] != (qv & 255u) || hi[ah(m, j)] != (qv >> 8);
        }
    }
  }
  fail |= hvx_pack_u16_ah(x16, M, 64, KK, lo, hi, NULL) == 0;
  fail |= hvx_pack_u16_ah(x16, M, MP, 32, lo, hi, NULL) == 0;
  hvx_worker_pool_destroy(pool);
  printf("integer-only pack: %s (tails, padding, 1 and 4 lanes)\n",
         fail ? "FAIL" : "OK");

  /* f32 rows */
  memset(lo, 0xa5, sizeof lo);
  memset(hi, 0x5a, sizeof hi);
  int rc = hvx_quant_params_pack_u16_ah(x, NULL, NULL, 0u, M, MP, KK, NULL, s,
                                        z, lo, hi, NULL);
  for (uint32_t m = 0; m < M; ++m)
    spec_row(x + (size_t)m * KK, &ws[m], &wz[m], q[m]);
  fail |= rc != 0 || check("f32 rows", lo, hi, s, z, q, ws, wz);

  /* u16 rows, one encoding: the codes as they are, zp rounded */
  const float e1[2] = {0.01f, 1234.6f};
  rc = hvx_quant_params_pack_u16_ah(NULL, x16, e1, KK, M, MP, KK, NULL, s, z,
                                    lo, hi, NULL);
  for (uint32_t m = 0; m < M; ++m) {
    memcpy(q[m], x16 + (size_t)m * KK, sizeof q[m]);
    ws[m] = e1[0];
    wz[m] = 1235;
  }
  fail |= rc != 0 || check("u16 one encoding", lo, hi, s, z, q, ws, wz);

  /* u16 rows, a head each: dequantized, then requantized per row */
  const float e2[4] = {0.002f, 31000.f, 0.0007f, 12000.f};
  rc = hvx_quant_params_pack_u16_ah(NULL, x16, e2, HD, M, MP, KK, scratch, s, z,
                                    lo, hi, NULL);
  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t j = 0; j < KK; ++j)
      dq[j] = ((float)x16[(size_t)m * KK + j] - e2[2u * (j / HD) + 1u]) *
              e2[2u * (j / HD)];
    spec_row(dq, &ws[m], &wz[m], q[m]);
  }
  fail |= rc != 0 || check("u16 per head", lo, hi, s, z, q, ws, wz);

  /* the a16 combine: two staged int32 tiles a result tile, 3 tiles */
  {
    enum { NT = 3, RS = HEXKL_ACC_TILE_COLS };
    static int32_t tiles[2 * NT][64 * RS];
    static float got[64 * NT * 32], cs_f[NT * 32];
    static int32_t cs[NT * 32];
    static float wsc[NT * 32], bias[NT * 32];
    for (uint32_t t = 0; t < 2u * NT; ++t)
      for (uint32_t i = 0; i < 64u * RS; ++i)
        tiles[t][i] = (int32_t)(rnd() % 4000001u) - 2000000;
    for (uint32_t c = 0; c < NT * 32u; ++c) {
      cs[c] = (int32_t)(rnd() % 40001u) - 20000;
      cs_f[c] = (float)cs[c];
      wsc[c] = 0.001f * (float)(1u + rnd() % 50u);
      bias[c] = 0.01f * (float)(rnd() % 100u);
    }
    for (uint32_t m = 0; m < 64u; ++m) {
      s[m] = 1e-4f * (float)(1u + rnd() % 30u);
      z[m] = (int32_t)(rnd() % 65536u);
    }
    hvx_dq_rows_job rj;
    memset(&rj, 0, sizeof rj);
    rj.t.tiles_base = (const uint8_t *)tiles[0];
    rj.t.tile_stride = sizeof tiles[0];
    rj.t.row_stride = RS;
    rj.t.m_count = 64u;
    rj.t.act_scale = s;
    rj.t.act_zp = z;
    rj.t.colsum_w = cs;
    rj.t.w_scale = wsc;
    rj.t.bias = bias;
    rj.t.dst_a = got;
    rj.t.dst_stride = NT * 32u;
    rj.t.n_tiles = NT;
    rj.t.lo_off = NT * sizeof tiles[0];
    hvx_dq_tiles_rows_worker(1u, 0u, &rj);
    size_t bad = 0;
    for (uint32_t m = 0; m < 64u; ++m)
      for (uint32_t c = 0; c < NT * 32u; ++c) {
        volatile float af = (float)tiles[c / 32u][m * RS + c % 32u] * 256.0f;
        af = af + (float)tiles[NT + c / 32u][m * RS + c % 32u];
        volatile float zc = (float)z[m] * cs_f[c];
        volatile float v = af - zc;
        v = v * s[m];
        v = v * wsc[c];
        v = v + bias[c];
        bad += memcmp((const void *)&v, &got[m * NT * 32u + c], 4) != 0;
      }
    printf("%-22s: values bad %zu of %u\n", "a16 combine", bad, 64u * NT * 32u);
    fail |= bad != 0;
  }

  /** the byte-plane readout (planes24): the same accumulators as three
      planes a tile, padded to 8 KiB slots as the kernel stages them, must
      dequantize to the bytes the int32 tiles give -- a8 (no lo) and a16 */
  {
    enum { NT = 3, RS = HEXKL_ACC_TILE_COLS, SLOT = 8192 };
    static int32_t ti[2 * NT][64 * RS];
    static uint8_t tp[2 * NT][SLOT];
    static float g_int[64 * NT * 32], g_pl[64 * NT * 32];
    static int32_t cs[NT * 32];
    static float wsc[NT * 32], bias[NT * 32];
    for (uint32_t t = 0; t < 2u * NT; ++t)
      for (uint32_t i = 0; i < 64u * 32u; ++i) {
        const int32_t v = (int32_t)(rnd() % 16777215u) - 8388607;
        ti[t][(i / 32u) * RS + i % 32u] = v;
        tp[t][i] = (uint8_t)v;
        tp[t][2048u + i] = (uint8_t)(v >> 8);
        tp[t][4096u + i] = (uint8_t)(v >> 16);
      }
    for (uint32_t c = 0; c < NT * 32u; ++c) {
      cs[c] = (int32_t)(rnd() % 40001u) - 20000;
      wsc[c] = 0.001f * (float)(1u + rnd() % 50u);
      bias[c] = 0.01f * (float)(rnd() % 100u);
    }
    for (uint32_t m = 0; m < 64u; ++m) {
      s[m] = 1e-4f * (float)(1u + rnd() % 30u);
      z[m] = (int32_t)(rnd() % 65536u);
    }
    for (int a16 = 0; a16 < 2; ++a16) {
      hvx_dq_rows_job rj;
      memset(&rj, 0, sizeof rj);
      rj.t.row_stride = RS;
      rj.t.m_count = 64u;
      rj.t.act_scale = s;
      rj.t.act_zp = z;
      rj.t.colsum_w = cs;
      rj.t.w_scale = wsc;
      rj.t.bias = bias;
      rj.t.dst_stride = NT * 32u;
      rj.t.n_tiles = NT;
      rj.t.tiles_base = (const uint8_t *)ti[0];
      rj.t.tile_stride = sizeof ti[0];
      rj.t.lo_off = a16 ? NT * sizeof ti[0] : 0u;
      rj.t.dst_a = g_int;
      hvx_dq_tiles_rows_worker(1u, 0u, &rj);
      rj.t.tiles_base = tp[0];
      rj.t.tile_stride = SLOT;
      rj.t.lo_off = a16 ? NT * SLOT : 0u;
      rj.t.planes24 = 1u;
      rj.t.dst_a = g_pl;
      hvx_dq_tiles_rows_worker(1u, 0u, &rj);
      const int same = !memcmp(g_int, g_pl, sizeof g_int);
      printf("%-22s: %s\n", a16 ? "planes24 a16" : "planes24 a8",
             same ? "bit-identical to int32 tiles" : "DIFFERS");
      fail |= !same;
    }
  }

  printf(fail ? "\nQUANT U16 FAIL\n" : "\nQUANT U16 OK\n");
  return fail;
}
