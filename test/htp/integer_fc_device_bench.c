// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd.
 * @file integer_fc_device_bench.c
 * @brief DSP correctness checks and matched FP32/integer FC timings
 * @bug No known bugs except for NYI items
 */
#include "hexkl_cvt.h"
#include "hexkl_integer.h"
#include "hexkl_probe.h"
#include "nntr_hvx.h"
#include "nntr_hvx_session.h"
#include <HAP_perf.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static FILE *report;
static void bench_printf(const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vfprintf(report, fmt, args);
  va_end(args);
  fflush(report);
}
#define printf bench_printf
static uint32_t state = 17;
static uint32_t rnd(void) {
  state = 1664525u * state + 1013904223u;
  return state;
}
#define REQUIRE(x)                                                             \
  do {                                                                         \
    if (!(x)) {                                                                \
      printf("FAIL line=%d expr=%s\n", __LINE__, #x);                          \
      return 1;                                                                \
    }                                                                          \
  } while (0)
static uint16_t rqref(int32_t x, const hexkl_requant_u16 *p) {
  int64_t v = (int64_t)x * p->multiplier, d = INT64_C(1) << p->shift;
  int64_t q = v / d, r = v % d;
  if (r && (r < 0 ? -r : r) >= (d + 1) / 2)
    q += v < 0 ? -1 : 1;
  q += p->zero_point;
  return q < 0 ? 0 : q > 65535 ? 65535 : (uint16_t)q;
}
static int arithmetic_checks(void) {
  enum { N = 97 };
  int32_t x[N];
  uint16_t out[N];
  hexkl_requant_u16 p[N];
  for (uint32_t trial = 0; trial < 256; ++trial) {
    for (uint32_t c = 0; c < N; ++c) {
      x[c] = (int32_t)rnd();
      if (c == 0)
        x[c] = INT32_MIN;
      if (c == 1)
        x[c] = INT32_MAX;
      p[c].multiplier = trial % 3 == 0   ? INT32_MAX
                        : trial % 3 == 1 ? 1
                                         : rnd() & INT32_MAX;
      p[c].shift = (trial + c) % 63;
      p[c].zero_point = (uint16_t)rnd();
    }
    REQUIRE(hexkl_requantize_i32_u16(x, out, 1, N, p, N) == 0);
    for (uint32_t c = 0; c < N; ++c) {
      if (out[c] != rqref(x[c], p + c))
        printf("rq c=%u x=%d m=%u s=%u z=%u got=%u want=%u\n", (unsigned)c,
               (int)x[c], (unsigned)p[c].multiplier, (unsigned)p[c].shift,
               p[c].zero_point, out[c], rqref(x[c], p + c));
      REQUIRE(out[c] == rqref(x[c], p + c));
    }
    REQUIRE(hexkl_requantize_i32_u16(x, out, 1, N, p, 1) == 0);
    for (uint32_t c = 0; c < N; ++c)
      REQUIRE(out[c] == rqref(x[c], p));
  }
#if defined(__hexagon__) && !defined(HEXKL_INTEGER_SCALAR)
  uint32_t ma[32] __attribute__((aligned(128))),
    mb[32] __attribute__((aligned(128)));
  uint32_t ml[32] __attribute__((aligned(128))),
    mh[32] __attribute__((aligned(128)));
  for (uint32_t trial = 0; trial < 128; ++trial) {
    for (uint32_t c = 0; c < 32; ++c) {
      ma[c] = c ? rnd() & INT32_MAX : 0x80000000u;
      mb[c] = trial % 2 ? rnd() & INT32_MAX : INT32_MAX;
    }
    hexkl_wide product = hexkl_umul(*(HVX_Vector *)ma, *(HVX_Vector *)mb);
    *(HVX_Vector *)ml = product.lo;
    *(HVX_Vector *)mh = product.hi;
    for (uint32_t c = 0; c < 32; ++c)
      REQUIRE((((uint64_t)mh[c] << 32) | ml[c]) == (uint64_t)ma[c] * mb[c]);
  }
  const uint32_t fast_m[] = {0, 1, 3, 65535, 65536, INT32_MAX};
  for (uint32_t m = 0; m < sizeof fast_m / sizeof fast_m[0]; ++m)
    for (uint32_t shift = 0; shift <= 16u; ++shift)
      for (uint32_t z = 0; z < 3u; ++z) {
        const uint32_t round = shift ? 1u << (shift - 1u) : 0;
        const uint32_t safe =
          fast_m[m] ? (UINT32_MAX - round) / fast_m[m] : UINT32_MAX;
        const uint16_t zp = z == 0 ? 0 : z == 1 ? 32768 : 65535;
        const hexkl_requant_u16 p = {fast_m[m], shift, zp};
        for (uint32_t c = 0; c < 32; ++c)
          ma[c] = rnd();
        ma[0] = (uint32_t)INT32_MIN;
        ma[1] = INT32_MAX;
        ma[2] = safe;
        ma[3] = safe + 1u;
        *(HVX_Vector *)ml =
          hexkl_requant_small_shift(*(HVX_Vector *)ma, Q6_V_vsplat_R(fast_m[m]),
                                    Q6_V_vsplat_R(zp), shift, fast_m[m], safe);
        for (uint32_t c = 0; c < 32; ++c)
          REQUIRE(ml[c] == rqref((int32_t)ma[c], &p));
      }
#endif
  // Three tiles exercise paired full-vector stores and a trailing half
  // vector, with odd nt0, unaligned output, row tails and all readouts.
  enum { TILES = 3, COLS = 128, CAPACITY = 64 * COLS + 128 };
  int32_t *tiles = memalign(128, 2u * TILES * 8192u);
  uint8_t *planes = memalign(128, 2u * TILES * 8192u);
  int32_t cs[COLS], bias[COLS], got[64 * COLS];
  uint16_t *qbuf = memalign(128, CAPACITY * sizeof(uint16_t));
  hexkl_requant_u16 qp[COLS];
  REQUIRE(tiles && planes && qbuf);
  for (uint32_t trial = 0; trial < 64; ++trial) {
    uint16_t zp = (uint16_t)rnd();
    for (uint32_t c = 0; c < COLS; ++c) {
      qp[c] = (hexkl_requant_u16){rnd() & INT32_MAX, (trial + c) % 63u,
                                  (uint16_t)rnd()};
      cs[c] = (int32_t)rnd();
      bias[c] = (int32_t)rnd();
    }
    for (uint32_t t = 0; t < 2u * TILES; ++t)
      for (uint32_t i = 0; i < 2048; ++i) {
        int32_t v = (int32_t)(rnd() & 0xffffffu) - 0x800000;
        if (trial & 2u)
          v /= 4; // Prove the uncorrected dot fits for the narrow epilogue.
        if (trial % 2 && i % 3)
          v = 0; // mix representable and overflowing results
        tiles[t * 2048 + i] = v;
        planes[t * 8192 + i] = (uint8_t)v;
        planes[t * 8192 + 2048 + i] = (uint8_t)((uint32_t)v >> 8);
        planes[t * 8192 + 4096 + i] = (uint8_t)((uint32_t)v >> 16);
        planes[t * 8192 + 6144 + i] = (uint8_t)((uint32_t)v >> 24);
      }
    if (trial % 2)
      memset(cs, 0, sizeof cs);
    for (uint32_t layout = 0; layout < 3; ++layout) {
      hexkl_i32_rows_job j = {0};
      j.tiles = layout ? planes : (uint8_t *)tiles;
      j.tile_stride = 8192;
      j.row_stride = 32;
      j.lo_off = TILES * 8192;
      j.plane_count = layout == 0 ? 0u : layout == 1 ? 3u : 4u;
      j.sum_fits_i32 = (trial & 2u) != 0u;
      j.n_tiles = TILES;
      j.nt0 = trial % 2u;
      j.rows = trial % 64 + 1;
      j.cols = COLS;
      j.zero_point = zp;
      j.colsum = cs;
      j.bias = bias;
      j.out = got;
      const uint32_t qoff = trial % 3u == 0 ? 0u : trial % 3u == 1 ? 1u : 32u;
      memset(qbuf, 0xa5, CAPACITY * sizeof(uint16_t));
      j.out_u16 = qbuf + qoff;
      j.requant = qp;
      j.n_requant = trial % 3u ? 1u : COLS;
      for (uint32_t lane = 0; lane < 4; ++lane)
        hexkl_i32_rows_worker(4, lane, &j);
      int ov = 0;
      for (uint32_t r = 0; r < j.rows; ++r)
        for (uint32_t c = 0; c < COLS; ++c) {
          size_t i = r * COLS + c;
          if (c < j.nt0 * 32u || c >= (j.nt0 + TILES) * 32u) {
            REQUIRE(j.out_u16[i] == 0xa5a5);
            continue;
          }
          const size_t ti = (c / 32u - j.nt0) * 2048u + r * 32u + c % 32u;
          int64_t v = 256LL * tiles[ti] + tiles[TILES * 2048u + ti] -
                      (int64_t)zp * cs[c] + bias[c];
          if (v < INT32_MIN || v > INT32_MAX)
            ov = 1;
          else {
            REQUIRE(got[i] == v);
            REQUIRE(j.out_u16[i] ==
                    rqref((int32_t)v, qp + (j.n_requant == 1u ? 0u : c)));
          }
        }
      REQUIRE((hexkl_i32_rows_status(&j) != 0) == ov);
      for (uint32_t i = 0; i < qoff; ++i)
        REQUIRE(qbuf[i] == 0xa5a5);
      for (uint32_t i = qoff + j.rows * COLS; i < CAPACITY; ++i)
        REQUIRE(qbuf[i] == 0xa5a5);
    }
  }
  free(qbuf);
  free(tiles);
  free(planes);
  printf(
    "ARITHMETIC OK: exact requant; three readouts; paired stores and tails\n");
  return 0;
}
static double us(uint64_t start) {
  return (double)(HAP_perf_get_qtimer_count() - start) / 19.2;
}
static int pooled_checks(hvx_worker_pool *pool) {
  enum { ROWS = 37, COLS = 97 };
  int32_t *x = memalign(128, ROWS * COLS * sizeof(*x));
  uint16_t *q = memalign(128, ROWS * COLS * sizeof(*q));
  hexkl_requant_u16 p[COLS];
  REQUIRE(x && q);
  for (size_t i = 0; i < ROWS * COLS; ++i)
    x[i] = (int32_t)rnd();
  for (uint32_t c = 0; c < COLS; ++c)
    p[c] = (hexkl_requant_u16){rnd() & INT32_MAX, c % 63, (uint16_t)rnd()};
  REQUIRE(hexkl_requantize_i32_u16_pooled(x, q, ROWS, COLS, p, COLS, pool) ==
          0);
  for (size_t i = 0; i < ROWS * COLS; ++i)
    REQUIRE(q[i] == rqref(x[i], p + i % COLS));
  memset(q, 0x5a, ROWS * COLS * sizeof(*q));
  p[COLS - 1].shift = 63;
  REQUIRE(hexkl_requantize_i32_u16_pooled(x, q, ROWS, COLS, p, COLS, pool) ==
          AEE_EBADPARM);
  for (size_t i = 0; i < ROWS * COLS; ++i)
    REQUIRE(q[i] == 0x5a5a);
  free(x);
  free(q);
  printf("POOLED OK: uneven rows, per-channel params, invalid params leave "
         "output intact\n");
  return 0;
}
static int extreme_fc_checks(nntr_hvx_session *s) {
  const uint32_t ks[] = {4096, 4160, 8192, 8192, 32768, 32768};
  const uint16_t zps[] = {0, 32768, 32768, 0, 65535, 65535};
  for (uint32_t trial = 0; trial < 6u; ++trial) {
    const uint32_t k = ks[trial];
    const int32_t weight = trial == 5u ? 7 : -8;
    uint16_t *a = memalign(128, k * sizeof(*a));
    int8_t *w = malloc((size_t)k * 32u);
    float scale[32], bias[32];
    int32_t cs[32], out[32];
    uint16_t q[32];
    uint32_t h;
    REQUIRE(a && w);
    for (uint32_t i = 0; i < k; ++i)
      a[i] = 65535;
    memset(w, weight, (size_t)k * 32u);
    for (uint32_t c = 0; c < 32u; ++c) {
      scale[c] = 1;
      bias[c] = 0;
      cs[c] = weight * (int32_t)k;
    }
    REQUIRE(hexkl_weight_u8i4_register(&s->weights_u8i4, s->vtcm_base,
                                       s->vtcm_size, k, 32, w, scale, cs, bias,
                                       s->quant_pool, &h) == 0);
    hexkl_requant_u16 p = {INT32_MAX, 47, 32768};
    const hexkl_requant_u16 *rp[] = {&p};
    const uint32_t nr[] = {1};
    int64_t v = ((int64_t)65535 - zps[trial]) * weight * (int64_t)k;
    int expected = v < INT32_MIN || v > INT32_MAX ? AEE_EFAILED : 0;
    int rc = hexkl_mm_u16i4_layer_run_i32_requant(
      &s->weights_u8i4, s->vtcm_base, s->vtcm_size, s->config_off, 1, k, &h, 1,
      a, zps[trial], NULL, out, s->quant_pool, rp, nr, q);
    REQUIRE(rc == expected);
    if (!rc)
      for (uint32_t c = 0; c < 32; ++c) {
        REQUIRE(out[c] == v);
        REQUIRE(q[c] == rqref((int32_t)v, &p));
      }
#ifndef HEXKL_BENCH_ROW_ONLY
    uint8_t *guarded = memalign(128, 4096u + 256u);
    REQUIRE(guarded);
    memset(guarded, 0xa5, 4096u + 256u);
    hexkl_u16_ah_io ah = {NULL, 0, guarded + 128u, 4096u};
    rc = hexkl_mm_u16i4_layer_run_i32_ah(
      &s->weights_u8i4, s->vtcm_base, s->vtcm_size, s->config_off, 1, k, &h, 1,
      a, zps[trial], NULL, out, s->quant_pool, rp, nr, &ah);
    REQUIRE(rc == expected);
    for (uint32_t i = 0; i < 128u; ++i) {
      REQUIRE(guarded[i] == 0xa5);
      REQUIRE(guarded[128u + 4096u + i] == 0xa5);
    }
    if (!rc)
      for (uint32_t r = 0; r < 64u; ++r)
        for (uint32_t c = 0; c < 32u; ++c) {
          const uint32_t d = r * 32u + c;
          const uint16_t code =
            ah.output[d] | ((uint16_t)ah.output[2048u + d] << 8);
          REQUIRE(code == (r ? 0 : rqref((int32_t)v, &p)));
          if (!r)
            REQUIRE(out[c] == v);
        }
    free(guarded);
#endif
#if defined(__hexagon__)
    // Probe the retained final (low-byte) accumulator directly. These are
    // precisely the four conversion encodings used by the vendor readout.
    const uint16_t enc[4] = {0x6000, 0x4000, 0x2000, 0};
    for (uint32_t plane = 0; plane < 4; ++plane) {
      hexkl_cvt_block_set_uniform(s->vtcm_base + plane * 1024u, enc[plane]);
      hexkl_cvt_issue(s->vtcm_base + plane * 1024u,
                      s->vtcm_base + 8192u + plane * 2048u);
    }
    (void)*(const volatile HVX_Vector *)(s->vtcm_base + 16384u - 128u);
    for (uint32_t c = 0; c < 32; ++c) {
      const uint8_t *b = s->vtcm_base + 8192u + c;
      uint32_t bits = b[0] | ((uint32_t)b[2048] << 8) |
                      ((uint32_t)b[4096] << 16) | ((uint32_t)b[6144] << 24);
      REQUIRE((int32_t)bits == weight * (int32_t)(k * 255u));
    }
#endif
    REQUIRE(hexkl_weight_u8i4_release(&s->weights_u8i4, h) == 0);
    free(a);
    free(w);
  }
  printf(
    "EXTREME FC OK: K=4096/4160/8192/32768; cancellation and true overflow\n");
  return 0;
}

static int late_overflow_check(nntr_hvx_session *s) {
  enum { M = 67, K = 64, N = 1056 };
  uint16_t *a = memalign(128, 2u * M * K);
  int8_t *w = malloc(K * N);
  float *scale = malloc(4u * N), *fb = calloc(N, sizeof(float));
  int32_t *cs = malloc(4u * N), *bias = calloc(N, sizeof(int32_t));
  int32_t *out = memalign(128, 4u * M * N);
  REQUIRE(a && w && scale && fb && cs && bias && out);
  memset(w, 1, K * N);
  for (uint32_t i = 0; i < M * K; ++i)
    a[i] = 30000;
  a[(M - 1u) * K] = 30001;
  for (uint32_t c = 0; c < N; ++c) {
    cs[c] = K;
    scale[c] = 1;
  }
  bias[N - 1u] = INT32_MAX;
  const int32_t *bp[] = {bias};
  uint32_t h;
  REQUIRE(hexkl_weight_u8i4_register(&s->weights_u8i4, s->vtcm_base,
                                     s->vtcm_size, K, N, w, scale, cs, fb,
                                     s->quant_pool, &h) == 0);
  REQUIRE(hexkl_mm_u16i4_layer_run_i32(
            &s->weights_u8i4, s->vtcm_base, s->vtcm_size, s->config_off, M, K,
            &h, 1, a, 30000, bp, out, s->quant_pool) == AEE_EFAILED);
#ifndef HEXKL_BENCH_ROW_ONLY
  uint8_t *q = memalign(128, 2u * 128u * N);
  REQUIRE(q);
  hexkl_u16_ah_io io = {NULL, 0, q, 2u * 128u * N};
  hexkl_requant_u16 p = {3, 8, 30000};
  const hexkl_requant_u16 *rp[] = {&p};
  const uint32_t nr[] = {1};
  REQUIRE(hexkl_mm_u16i4_layer_run_i32_ah(&s->weights_u8i4, s->vtcm_base,
                                          s->vtcm_size, s->config_off, M, K, &h,
                                          1, a, 30000, bp, out, s->quant_pool,
                                          rp, nr, &io) == AEE_EFAILED);
  free(q);
#endif
  REQUIRE(hexkl_weight_u8i4_release(&s->weights_u8i4, h) == 0);
  free(a);
  free(w);
  free(scale);
  free(fb);
  free(cs);
  free(bias);
  free(out);
  printf("LATE OVERFLOW OK: last row/column of final partial batch\n");
  return 0;
}

static int bench(nntr_hvx_session *s, uint32_t M, uint32_t K, uint32_t N,
                 uint32_t reps) {
  const uint32_t slice = 256, nh = N / slice;
  uint32_t handles[32];
  int32_t expected[32][9];
  uint16_t *a = memalign(128, (size_t)M * K * 2),
           *q = memalign(128, (size_t)M * N * 2);
  int32_t *y = memalign(128, (size_t)M * N * 4);
  float *yf = memalign(128, (size_t)M * N * 4);
  int8_t *w = malloc((size_t)K * slice);
  float scales[256], bias[256];
  int32_t cs[256];
  REQUIRE(a && q && y && yf && w && nh <= 32);
  for (size_t i = 0; i < (size_t)M * K; ++i)
    a[i] = (uint16_t)(rnd() >> 8);
  for (uint32_t h = 0; h < nh; ++h) {
    memset(cs, 0, sizeof cs);
    for (uint32_t k = 0; k < K; ++k)
      for (uint32_t c = 0; c < slice; ++c) {
        int v = (int)((rnd() >> 16) & 15) - 8;
        w[(size_t)k * slice + c] = (int8_t)v;
        cs[c] += v;
      }
    const uint32_t rs[3] = {0, M / 2, M - 1}, csample[3] = {0, 31, 255};
    for (uint32_t ri = 0; ri < 3; ++ri)
      for (uint32_t ci = 0; ci < 3; ++ci) {
        int64_t v = 0;
        for (uint32_t k = 0; k < K; ++k)
          v += ((int64_t)a[(size_t)rs[ri] * K + k] - 30000) *
               w[(size_t)k * slice + csample[ci]];
        REQUIRE(v >= INT32_MIN && v <= INT32_MAX);
        expected[h][ri * 3 + ci] = (int32_t)v;
      }
    for (uint32_t c = 0; c < slice; ++c) {
      scales[c] = 1.0f / 1024.0f;
      bias[c] = 0;
    }
    REQUIRE(hexkl_weight_u8i4_register(&s->weights_u8i4, s->vtcm_base,
                                       s->vtcm_size, K, slice, w, scales, cs,
                                       bias, s->quant_pool, handles + h) == 0);
  }
  free(w);
  const float enc[2] = {1.0f / 256.0f, 30000.f};
  hexkl_mm_opts o = {0};
  o.act16 = 1;
  o.act_u16 = a;
  o.act16_enc = enc;
  o.act16_hd = K;
  o.pool = s->quant_pool;
  hexkl_requant_u16 p = {3, 8, 30000};
  const hexkl_requant_u16 *rpp[32];
  uint32_t nr[32];
  for (uint32_t h = 0; h < nh; ++h) {
    rpp[h] = &p;
    nr[h] = 1;
  }
  const uint32_t batch = 3;
  for (uint32_t it = 0; it < reps + 2; ++it) {
    double tm[3] = {0};
    // Rotate the three modes. Warm each mode once, then time three calls
    // to compare steady-state kernels with equally warm output buffers.
    for (uint32_t step = 0; step < 3; ++step) {
      uint32_t mode = (step + it) % 3;
      uint64_t t = 0;
      for (uint32_t b = 0; b <= batch; ++b) {
        if (b == 1)
          t = HAP_perf_get_qtimer_count();
        int rc;
        if (mode == 0)
          rc = hexkl_mm_u8i4_layer_run(&s->weights_u8i4, s->vtcm_base,
                                       s->vtcm_size, s->config_off, M, K,
                                       handles, nh, NULL, yf, &o);
        else if (mode == 2)
          rc = hexkl_mm_u16i4_layer_run_i32_requant(
            &s->weights_u8i4, s->vtcm_base, s->vtcm_size, s->config_off, M, K,
            handles, nh, a, 30000, NULL, y, s->quant_pool, rpp, nr, q);
        else
          rc = hexkl_mm_u16i4_layer_run_i32(
            &s->weights_u8i4, s->vtcm_base, s->vtcm_size, s->config_off, M, K,
            handles, nh, a, 30000, NULL, y, s->quant_pool);
        REQUIRE(rc == 0);
      }
      tm[mode] = us(t) / batch;
    }
    for (size_t i = 0; i < (size_t)M * N; ++i)
      REQUIRE(q[i] == rqref(y[i], &p));
    uint64_t t = HAP_perf_get_qtimer_count();
    for (uint32_t b = 0; b < batch; ++b)
      REQUIRE(
        hexkl_requantize_i32_u16_pooled(y, q, M, N, &p, 1, s->quant_pool) == 0);
    double rq = us(t) / batch;
    t = HAP_perf_get_qtimer_count();
    for (uint32_t b = 0; b < batch; ++b)
      REQUIRE(hexkl_requantize_i32_u16(y, q, M, N, &p, 1) == 0);
    double single = us(t) / batch;
    if (it >= 2)
      printf("TIME M=%u K=%u N=%u rep=%u fp32_us=%.3f i32_us=%.3f "
             "chain_us=%.3f rq_us=%.3f rq_single_us=%.3f\n",
             (unsigned)M, (unsigned)K, (unsigned)N, (unsigned)(it - 2), tm[0],
             tm[1], tm[2], rq, single);
  }
  hexkl_probe_reset(1);
  uint64_t profile_start = HAP_perf_get_qtimer_count();
  REQUIRE(hexkl_mm_u16i4_layer_run_i32_requant(
            &s->weights_u8i4, s->vtcm_base, s->vtcm_size, s->config_off, M, K,
            handles, nh, a, 30000, NULL, y, s->quant_pool, rpp, nr, q) == 0);
  double profile_us = us(profile_start);
  hexkl_probe_on = 0;
  printf("PROFILE total_us=%.3f hmx_readout_us=%llu pack_wait_us=%llu "
         "epilogue_wait_us=%llu dma_wait_us=%llu\n",
         profile_us, (unsigned long long)hexkl_probe_us[HEXKL_PROBE_MM],
         (unsigned long long)hexkl_probe_us[HEXKL_PROBE_QUANT],
         (unsigned long long)hexkl_probe_us[HEXKL_PROBE_DEQUANT],
         (unsigned long long)hexkl_probe_us[HEXKL_PROBE_DRAIN]);
  const uint32_t rs[3] = {0, M / 2, M - 1}, csample[3] = {0, 31, 255};
  for (uint32_t h = 0; h < nh; ++h)
    for (uint32_t ri = 0; ri < 3; ++ri)
      for (uint32_t ci = 0; ci < 3; ++ci)
        REQUIRE(
          y[(size_t)h * M * slice + (size_t)rs[ri] * slice + csample[ci]] ==
          expected[h][ri * 3 + ci]);
  // Direct INT64 reference samples were computed before weight packing.
  // Here compare all outputs with the legacy float result at its scale;
  // integer bit-exactness is additionally checked by cross-build hashes.
  double maxerr = 0;
  uint32_t hash = 2166136261u;
  for (size_t i = 0; i < (size_t)M * N; ++i) {
    double e = (double)yf[i] - (double)y[i] / 262144.;
    if (e < 0)
      e = -e;
    if (e > maxerr)
      maxerr = e;
    hash = (hash ^ (uint32_t)y[i]) * 16777619u;
    REQUIRE(q[i] == rqref(y[i], &p));
  }
  printf("CHECK M=%u K=%u N=%u hash=%08x fp32_max_abs=%.9g\n", (unsigned)M,
         (unsigned)K, (unsigned)N, (unsigned)hash, maxerr);
  for (uint32_t h = 0; h < nh; ++h)
    REQUIRE(hexkl_weight_u8i4_release(&s->weights_u8i4, handles[h]) == 0);
  free(a);
  free(q);
  free(y);
  free(yf);
  return 0;
}
#include "integer_fc_chain_bench.h"

int main(int argc, char **argv) {
  if (argc < 2)
    return 1;
  report = fopen(argv[1], "w");
  if (!report)
    return 1;
  REQUIRE(arithmetic_checks() == 0);
  if (argc > 2 && !strcmp(argv[2], "check")) {
    fclose(report);
    return 0;
  }
  remote_handle64 handle = 0;
  REQUIRE(nntr_hvx_open("", &handle) == 0);
  nntr_hvx_session *s = (nntr_hvx_session *)(uintptr_t)handle;
  printf("SESSION VTCM=%u workers=%u hmx=%d\n", (unsigned)s->vtcm_size,
         (unsigned)hvx_worker_pool_workers(s->quant_pool), s->hmx_locked);
  REQUIRE(s->hmx_locked);
  REQUIRE(pooled_checks(s->quant_pool) == 0);
  REQUIRE(extreme_fc_checks(s) == 0);
  REQUIRE(late_overflow_check(s) == 0);
  const uint32_t shapes[][3] = {{128, 2816, 4096}, {512, 2816, 4096},
                                {768, 2816, 4096}, {128, 4096, 4096},
                                {512, 4096, 4096}, {67, 4160, 256}};
  if (argc > 2 && !strcmp(argv[2], "chain")) {
    const uint32_t shape = argc > 3 ? (uint32_t)atoi(argv[3]) : 1u;
    REQUIRE(shape < sizeof shapes / sizeof shapes[0]);
    state = 17u + 101u * shape;
    REQUIRE(chain_bench(s, shapes[shape][0], shapes[shape][1],
                        shapes[shape][2]) == 0);
    REQUIRE(nntr_hvx_close(handle) == 0);
    printf("BENCH OK\n");
    fclose(report);
    return 0;
  }
  for (uint32_t i = 0; i < sizeof shapes / sizeof shapes[0]; ++i) {
    if (argc > 2 && i != (uint32_t)atoi(argv[2]))
      continue;
    state = 17u + 101u * i; // Identical inputs in isolated and full runs.
    int rc = bench(s, shapes[i][0], shapes[i][1], shapes[i][2], 6);
    if (rc) {
      nntr_hvx_close(handle);
      return rc;
    }
  }
  REQUIRE(nntr_hvx_close(handle) == 0);
  printf("BENCH OK\n");
  fclose(report);
  return 0;
}
