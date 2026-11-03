// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd.
 * @file pr4471_fc_device_bench.c
 * @brief Original PR #4471 FC timing; build against its unchanged skel
 * @bug No known bugs except for NYI items
 */
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
static int bench(nntr_hvx_session *s, uint32_t M, uint32_t K, uint32_t N,
                 uint32_t reps) {
  const uint32_t slice = 256, nh = N / slice;
  uint32_t handles[32];
  int32_t expected[32][9];
  uint16_t *a = memalign(128, (size_t)M * K * 2);
  float *yf = memalign(128, (size_t)M * N * 4);
  int8_t *w = malloc((size_t)K * slice);
  float scales[256], bias[256];
  int32_t cs[256];
  REQUIRE(a && yf && w && nh <= 32);
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
  // Same inputs, offline weight setup, warmup and three-call samples as
  // integer_fc_device_bench.c. Only the original FP32 API is linked here.
  const uint32_t batch = 3;
  for (uint32_t it = 0; it < reps + 2; ++it) {
    uint64_t start = 0;
    for (uint32_t b = 0; b <= batch; ++b) {
      if (b == 1)
        start = HAP_perf_get_qtimer_count();
      REQUIRE(hexkl_mm_u8i4_layer_run(&s->weights_u8i4, s->vtcm_base,
                                      s->vtcm_size, s->config_off, M, K,
                                      handles, nh, NULL, yf, &o) == 0);
    }
    const double elapsed =
      (double)(HAP_perf_get_qtimer_count() - start) / 19.2 / batch;
    if (it >= 2)
      printf("TIME M=%u K=%u N=%u rep=%u fp32_us=%.3f\n", (unsigned)M,
             (unsigned)K, (unsigned)N, (unsigned)(it - 2), elapsed);
  }
  double maxerr = 0;
  const uint32_t rs[3] = {0, M / 2, M - 1}, csample[3] = {0, 31, 255};
  for (uint32_t h = 0; h < nh; ++h)
    for (uint32_t ri = 0; ri < 3; ++ri)
      for (uint32_t ci = 0; ci < 3; ++ci) {
        double error =
          yf[(size_t)h * M * slice + (size_t)rs[ri] * slice + csample[ci]] -
          (double)expected[h][ri * 3 + ci] / 262144.;
        if (error < 0)
          error = -error;
        REQUIRE(error <= 0.001);
        if (error > maxerr)
          maxerr = error;
      }
  uint32_t hash = 2166136261u;
  for (size_t i = 0; i < (size_t)M * N; ++i) {
    uint32_t bits;
    memcpy(&bits, yf + i, sizeof bits);
    hash = (hash ^ bits) * 16777619u;
  }
  printf("CHECK M=%u K=%u N=%u hash=%08x fp32_sample_max_abs=%.9g\n",
         (unsigned)M, (unsigned)K, (unsigned)N, (unsigned)hash, maxerr);
  for (uint32_t h = 0; h < nh; ++h)
    REQUIRE(hexkl_weight_u8i4_release(&s->weights_u8i4, handles[h]) == 0);
  free(a);
  free(yf);
  return 0;
}
int main(int argc, char **argv) {
  if (argc < 2)
    return 1;
  report = fopen(argv[1], "w");
  if (!report)
    return 1;
  remote_handle64 handle = 0;
  REQUIRE(nntr_hvx_open("", &handle) == 0);
  nntr_hvx_session *s = (nntr_hvx_session *)(uintptr_t)handle;
  printf("SESSION VTCM=%u workers=%u hmx=%d\n", (unsigned)s->vtcm_size,
         (unsigned)hvx_worker_pool_workers(s->quant_pool), s->hmx_locked);
  REQUIRE(s->hmx_locked);
  const uint32_t shapes[][3] = {{128, 2816, 4096}, {512, 2816, 4096},
                                {768, 2816, 4096}, {128, 4096, 4096},
                                {512, 4096, 4096}, {67, 4160, 256}};
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
