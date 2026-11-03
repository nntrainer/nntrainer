// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd.
 * @file fc_integer_host_check.c
 * @brief Integer FC against direct INT64 dots, including a two-FC chain
 * @bug No known bugs except for NYI items
 */
#include "hexkl_integer.h"
#include "hvx_scalar_stubs.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint8_t vtcm[8u << 20];
static int fail;
#define CHECK(x)                                                               \
  do {                                                                         \
    if (!(x)) {                                                                \
      printf("FAIL line %d: %s\n", __LINE__, #x);                              \
      fail = 1;                                                                \
    }                                                                          \
  } while (0)

static int weight_at(const W *w, uint32_t k, uint32_t c) {
  return wh_value((uint8_t *)w->nib +
                    ((size_t)(k / 32u) * (w->N / 32u) + c / 32u) * 512u,
                  k % 32u, c % 32u);
}

static void prepare(uint32_t slot, uint32_t k, uint32_t n, W *w, int constant) {
  make_weight(slot, k, n, w);
  g_tbl.slots[slot].bits = 4;
  if (constant)
    memset(w->nib, 0x88, (size_t)k * n / 2u); // every weight = -8
  for (uint32_t c = 0; c < n; ++c) {
    w->cs[c] = 0;
    for (uint32_t j = 0; j < k; ++j)
      w->cs[c] += weight_at(w, j, c);
  }
  // A float dequant/bias access in the integer path must not work.
  g_tbl.slots[slot].w_scale = NULL;
  g_tbl.slots[slot].bias = NULL;
}

static void release_weight(W *w) {
  free(w->nib);
  free(w->ws);
  free(w->cs);
  free(w->bias);
}

static int reference(const W *w, const uint16_t *a, uint32_t m, uint16_t zp,
                     const int32_t *bias, int32_t *out) {
  int overflow = 0;
  for (uint32_t r = 0; r < m; ++r)
    for (uint32_t c = 0; c < w->N; ++c) {
      int64_t sum = bias ? bias[c] : 0;
      for (uint32_t k = 0; k < w->K; ++k)
        sum += ((int64_t)a[(size_t)r * w->K + k] - zp) * weight_at(w, k, c);
      if (sum < INT32_MIN || sum > INT32_MAX)
        overflow = 1;
      else
        out[(size_t)r * w->N + c] = (int32_t)sum;
    }
  return overflow ? AEE_EFAILED : AEE_SUCCESS;
}

// Independent quotient/remainder rounding specification, including negatives.
static uint16_t requant_ref(int32_t x, const hexkl_requant_u16 *p) {
  int64_t prod = (int64_t)x * p->multiplier;
  const int64_t divisor = INT64_C(1) << p->shift;
  int64_t q = prod / divisor, rem = prod % divisor;
  if (rem && (rem < 0 ? -rem : rem) >= (divisor + 1) / 2)
    q += prod < 0 ? -1 : 1;
  q += p->zero_point;
  return (uint16_t)(q < 0 ? 0 : q > 65535 ? 65535 : q);
}

static void check_requant(void) {
  const int32_t x[] = {INT32_MIN, INT32_MAX, -65536, -3, -1, 0, 1, 3, 65536};
  uint16_t got[9];
  const uint32_t mults[] = {0, 1, 123456789, INT32_MAX};
  for (uint32_t a = 0; a < 4; ++a)
    for (uint32_t shift = 0; shift <= 62; ++shift) {
      hexkl_requant_u16 p = {mults[a], shift, 32768};
      CHECK(hexkl_requantize_i32_u16(x, got, 1, 9, &p, 1) == 0);
      for (uint32_t c = 0; c < 9; ++c)
        CHECK(got[c] == requant_ref(x[c], &p));
    }
  hexkl_requant_u16 ps[9];
  for (uint32_t c = 0; c < 9; ++c) {
    ps[c].multiplier = c + 1;
    ps[c].shift = c;
    ps[c].zero_point = c * 7000;
  }
  CHECK(hexkl_requantize_i32_u16(x, got, 1, 9, ps, 9) == 0);
  for (uint32_t c = 0; c < 9; ++c)
    CHECK(got[c] == requant_ref(x[c], &ps[c]));
  memset(got, 0xa5, sizeof got);
  ps[8].shift = 63;
  CHECK(hexkl_requantize_i32_u16(x, got, 1, 9, ps, 9) == AEE_EBADPARM);
  for (uint32_t c = 0; c < 9; ++c)
    CHECK(got[c] == 0xa5a5);
  CHECK(hexkl_requantize_i32_u16_pooled(x, got, 1, 9, ps, 9, NULL) ==
        AEE_EBADPARM);
  for (uint32_t c = 0; c < 9; ++c)
    CHECK(got[c] == 0xa5a5);
  ps[8].shift = 0;
  ps[8].multiplier = UINT32_MAX;
  CHECK(hexkl_requantize_i32_u16(x, got, 1, 9, ps, 9) == AEE_EBADPARM);
  CHECK(hexkl_requantize_i32_u16(x, got, 1, 9, ps, 2) == AEE_EBADPARM);
}

static void check_fc(uint32_t m, uint32_t k, int constant, uint16_t zp) {
  const uint32_t ns[2] = {constant ? 32u : 1056u, 64u}, handles[2] = {0, 1};
  W w[2];
  size_t count = (size_t)m * (ns[0] + ns[1]);
  uint16_t *x = malloc(sizeof(*x) * m * k);
  int32_t *out = malloc(sizeof(*out) * (count + 2));
  int32_t *want = calloc(count, sizeof(*want));
  int32_t *bias[2];
  for (uint32_t i = 0; i < 2; ++i) {
    prepare(i, k, ns[i], w + i, constant);
    bias[i] = malloc(sizeof(int32_t) * ns[i]);
    for (uint32_t c = 0; c < ns[i]; ++c)
      bias[i][c] = (int32_t)c - 100;
  }
  const int32_t *bp[2] = {bias[0], bias[1]};
  for (size_t j = 0; j < (size_t)m * k; ++j)
    x[j] = constant     ? 65535
           : j % 7 == 0 ? 0
           : j % 7 == 1 ? 65535
                        : (uint16_t)rnd();
  out[0] = out[count + 1] = 0x12345678;
  const int rc =
    hexkl_mm_u16i4_layer_run_i32(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm, m, k,
                                 handles, 2, x, zp, bp, out + 1, NULL);
  int expected = 0;
  size_t off = 0;
  for (uint32_t i = 0; i < 2; ++i) {
    expected |= reference(w + i, x, m, zp, bias[i], want + off);
    off += (size_t)m * ns[i];
  }
  CHECK(rc == expected);
  if (!expected)
    CHECK(memcmp(out + 1, want, sizeof(*want) * count) == 0);
  CHECK(out[0] == 0x12345678 && out[count + 1] == 0x12345678);
  uint16_t *q = malloc(sizeof(*q) * (count + 2));
  hexkl_requant_u16 *rp[2];
  for (uint32_t h = 0; h < 2; ++h) {
    rp[h] = malloc(sizeof(**rp) * ns[h]);
    for (uint32_t c = 0; c < ns[h]; ++c)
      rp[h][c] = (hexkl_requant_u16){c % 2 ? INT32_MAX : c + 1u, c % 63u,
                                     (uint16_t)(c * 71u)};
  }
  const hexkl_requant_u16 *rpp[2] = {rp[0], rp[1]};
  uint32_t nr[2] = {1, ns[1]};
  q[0] = q[count + 1] = 0xa55a;
  CHECK(hexkl_mm_u16i4_layer_run_i32_requant(
          &g_tbl, vtcm, sizeof vtcm, sizeof vtcm, m, k, handles, 2, x, zp, bp,
          out + 1, NULL, rpp, nr, q + 1) == expected);
  if (!expected) {
    CHECK(memcmp(out + 1, want, sizeof(*want) * count) == 0);
    off = 0;
    for (uint32_t h = 0; h < 2; ++h) {
      for (size_t i = 0; i < (size_t)m * ns[h]; ++i)
        CHECK(
          q[1 + off + i] ==
          requant_ref(want[off + i], rp[h] + (nr[h] == 1u ? 0u : i % ns[h])));
      off += (size_t)m * ns[h];
    }
  }
  CHECK(q[0] == 0xa55a && q[count + 1] == 0xa55a);
  memset(q, 0x5a, (count + 2) * sizeof(*q));
  memset(out, 0x5a, (count + 2) * sizeof(*out));
  rp[1][ns[1] - 1].shift = 63;
  CHECK(hexkl_mm_u16i4_layer_run_i32_requant(
          &g_tbl, vtcm, sizeof vtcm, sizeof vtcm, m, k, handles, 2, x, zp, bp,
          out + 1, NULL, rpp, nr, q + 1) == AEE_EBADPARM);
  for (size_t i = 0; i < count + 2; ++i) {
    CHECK(q[i] == 0x5a5a);
    CHECK(out[i] == 0x5a5a5a5a);
  }
  free(q);
  free(rp[0]);
  free(rp[1]);
  printf("integer FC M=%u K=%u zp=%u rc=%d expected=%d\n", m, k, zp, rc,
         expected);
  CHECK(hexkl_mm_u16i4_layer_run_i32(&g_tbl, vtcm, 16384, 16384, m, k, handles,
                                     2, x, zp, bp, out + 1,
                                     NULL) == AEE_ENOMEMORY);
  g_tbl.slots[0].bits = 2;
  CHECK(hexkl_mm_u16i4_layer_run_i32(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm, m,
                                     k, handles, 2, x, zp, bp, out + 1,
                                     NULL) == AEE_EBADPARM);
  for (uint32_t i = 0; i < 2; ++i) {
    release_weight(w + i);
    free(bias[i]);
  }
  free(x);
  free(out);
  free(want);
}

static void check_chain(void) {
  enum { M = 5, K = 64, N = 64 };
  W w[2];
  uint16_t x[M * K], mid[M * N], refmid[M * N];
  int32_t acc[M * N], refacc[M * N], out[M * N], want[M * N];
  const uint32_t h0 = 0, h1 = 1;
  for (uint32_t i = 0; i < M * K; ++i)
    x[i] = (uint16_t)rnd();
  prepare(0, K, N, w, 0);
  prepare(1, N, N, w + 1, 0);
  hexkl_requant_u16 p = {3, 8, 30000};
  CHECK(hexkl_mm_u16i4_layer_run_i32(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M,
                                     K, &h0, 1, x, 29000, NULL, acc,
                                     NULL) == 0);
  CHECK(reference(w, x, M, 29000, NULL, refacc) == 0);
  CHECK(hexkl_requantize_i32_u16_pooled(acc, mid, M, N, &p, 1, NULL) == 0);
  for (uint32_t i = 0; i < M * N; ++i)
    refmid[i] = requant_ref(refacc[i], &p);
  CHECK(memcmp(mid, refmid, sizeof mid) == 0);
  CHECK(hexkl_mm_u16i4_layer_run_i32(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M,
                                     N, &h1, 1, mid, p.zero_point, NULL, out,
                                     NULL) == 0);
  CHECK(reference(w + 1, refmid, M, p.zero_point, NULL, want) == 0);
  CHECK(memcmp(out, want, sizeof out) == 0);
  // Exact INT32 endpoints with bias; then bias-induced overflow.
  for (uint32_t i = 0; i < M * K; ++i)
    x[i] = 29000;
  int32_t bias[N];
  const int32_t *bp[] = {bias};
  for (uint32_t i = 0; i < N; ++i)
    bias[i] = i % 2 ? INT32_MIN : INT32_MAX;
  CHECK(hexkl_mm_u16i4_layer_run_i32(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M,
                                     K, &h0, 1, x, 29000, bp, acc, NULL) == 0);
  for (uint32_t i = 0; i < M * N; ++i)
    CHECK(acc[i] == bias[i % N]);
  memset(w[0].nib, 0x88, K * N / 2);
  for (uint32_t i = 0; i < N; ++i)
    w[0].cs[i] = -8 * K;
  x[0] = 29001; // -8 plus INT32_MIN must fail, not wrap
  CHECK(hexkl_mm_u16i4_layer_run_i32(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M,
                                     K, &h0, 1, x, 29000, bp, acc,
                                     NULL) == AEE_EFAILED);
  for (uint32_t i = 0; i < 2; ++i)
    release_weight(w + i);
}

static void check_packed_chain(void) {
  enum { M = 67, MP = 128, K = 128, N = 256, O = 96 };
  const uint32_t handles[2] = {0, 1}, ns[2] = {96, 160}, consumer = 2;
  W w[3];
  uint16_t *x = malloc(2u * M * K), *mid = malloc(2u * M * N);
  int32_t *acc = malloc(4u * M * N), *ref = malloc(4u * M * N);
  int32_t *out = malloc(4u * M * O), *want = malloc(4u * M * O);
  void *pi = malloc(2u * MP * K + 256u), *po = malloc(2u * MP * N + 256u);
  uint8_t *in = (uint8_t *)(((uintptr_t)pi + 127u) & ~(uintptr_t)127u);
  uint8_t *packed = (uint8_t *)(((uintptr_t)po + 127u) & ~(uintptr_t)127u);
  for (uint32_t i = 0; i < M * K; ++i)
    x[i] = (uint16_t)rnd();
  // Build the input independently by walking full tiles and padded rows.
  for (uint32_t rb = 0; rb < MP / 64u; ++rb)
    for (uint32_t kt = 0; kt < K / 32u; ++kt)
      for (uint32_t r = 0; r < 64u; ++r)
        for (uint32_t c = 0; c < 32u; ++c) {
          const uint32_t row = rb * 64u + r;
          const uint16_t q = row < M ? x[row * K + kt * 32u + c] : 0;
          const size_t d = ((size_t)rb * (K / 32u) + kt) * 2048u + r * 32u + c;
          in[d] = (uint8_t)q;
          in[MP * K + d] = (uint8_t)(q >> 8);
        }
  hexkl_requant_u16 params[2][160];
  const hexkl_requant_u16 *rp[2] = {params[0], params[1]};
  uint32_t nr[2];
  for (uint32_t h = 0; h < 2u; ++h) {
    prepare(h, K, ns[h], w + h, 0);
    CHECK(reference(w + h, x, M, 30000, NULL, ref + (h ? M * ns[0] : 0)) == 0);
  }
  prepare(consumer, N, O, w + 2, 0);
  for (uint32_t per_channel = 0; per_channel < 2u; ++per_channel) {
    for (uint32_t h = 0; h < 2u; ++h) {
      nr[h] = per_channel ? ns[h] : 1u;
      for (uint32_t c = 0; c < ns[h]; ++c)
        params[h][c] = (hexkl_requant_u16){per_channel ? 123456789u : 3u,
                                           per_channel ? c % 63u : 8u, 30000};
    }
    for (uint32_t packed_input = 0; packed_input < 2u; ++packed_input) {
      memset(packed, 0xa5, 2u * MP * N);
      hexkl_u16_ah_io io = {packed_input ? in : NULL, 2u * MP * K, packed,
                            2u * MP * N};
      CHECK(hexkl_mm_u16i4_layer_run_i32_ah(&g_tbl, vtcm, sizeof vtcm,
                                            sizeof vtcm, M, K, handles, 2,
                                            packed_input ? NULL : x, 30000,
                                            NULL, acc, NULL, rp, nr, &io) == 0);
      CHECK(memcmp(acc, ref, 4u * M * N) == 0);
      for (uint32_t r = 0; r < MP; ++r)
        for (uint32_t c = 0; c < N; ++c) {
          uint16_t q = 0;
          if (r < M) {
            const uint32_t h = c < ns[0] ? 0u : 1u;
            const uint32_t ch = c - (h ? ns[0] : 0u);
            q = requant_ref(ref[(h ? M * ns[0] : 0u) + r * ns[h] + ch],
                            params[h] + (per_channel ? ch : 0u));
            mid[r * N + c] = q;
          }
          const size_t d = ((size_t)(r / 64u) * (N / 32u) + c / 32u) * 2048u +
                           (r % 64u) * 32u + c % 32u;
          CHECK((uint16_t)(packed[d] | ((uint16_t)packed[MP * N + d] << 8)) ==
                q);
        }
      hexkl_u16_ah_io next = {packed, 2u * MP * N, NULL, 0};
      CHECK(hexkl_mm_u16i4_layer_run_i32_ah(
              &g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M, N, &consumer, 1, NULL,
              30000, NULL, out, NULL, NULL, NULL, &next) == 0);
      CHECK(reference(w + 2, mid, M, 30000, NULL, want) == 0);
      CHECK(memcmp(out, want, 4u * M * O) == 0);
    }
  }
  // Invalid capacities/alignment/aliasing must fail before either output
  // changes.
  for (uint32_t invalid = 0; invalid < 6u; ++invalid) {
    hexkl_u16_ah_io io = {in, 2u * MP * K, packed, 2u * MP * N};
    if (invalid == 0)
      --io.input_bytes;
    if (invalid == 1)
      --io.output_bytes;
    if (invalid == 2)
      ++io.input;
    if (invalid == 3)
      io.output = vtcm;
    if (invalid == 4)
      io.output = in;
    memset(acc, 0x5a, 4u * M * N);
    memset(packed, 0xa5, 2u * MP * N);
    CHECK(hexkl_mm_u16i4_layer_run_i32_ah(
            &g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M, K, handles, 2,
            invalid == 5 ? x : NULL, 30000, NULL, acc, NULL, rp, nr,
            &io) == AEE_EBADPARM);
    for (uint32_t i = 0; i < M * N; ++i)
      CHECK(acc[i] == 0x5a5a5a5a);
    for (uint32_t i = 0; i < 2u * MP * N; ++i)
      CHECK(packed[i] == 0xa5);
  }
  for (uint32_t i = 0; i < 3u; ++i)
    release_weight(w + i);
  free(x);
  free(mid);
  free(acc);
  free(ref);
  free(out);
  free(want);
  free(pi);
  free(po);
  puts("PACKED INTEGER FC CHAIN OK");
}

static void check_late_overflow(void) {
  enum { M = 67, K = 64, N = 1056 };
  W w;
  const uint32_t h = 0;
  prepare(0, K, N, &w, 0);
  memset(w.nib, 0x11, K * N / 2u);
  for (uint32_t c = 0; c < N; ++c)
    w.cs[c] = K;
  uint16_t *x = malloc(2u * M * K);
  int32_t *y = malloc(4u * M * N), *b = calloc(N, sizeof(int32_t));
  for (uint32_t i = 0; i < M * K; ++i)
    x[i] = 30000;
  x[(M - 1u) * K] = 30001;
  b[N - 1u] = INT32_MAX;
  const int32_t *bias[] = {b};
  // Only the last row/column overflows, in the final, partially filled bank.
  CHECK(hexkl_mm_u16i4_layer_run_i32(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M,
                                     K, &h, 1, x, 30000, bias, y,
                                     NULL) == AEE_EFAILED);
  free(x);
  free(y);
  free(b);
  release_weight(&w);
}

int main(void) {
  check_requant();
  check_fc(67, 128, 0, 0);
  check_fc(67, 128, 0, 32768);
  check_fc(67, 128, 0, 65535);
  check_fc(1, 4096, 1, 0);     // int24 readout, INT32 boundary
  check_fc(1, 4160, 1, 32768); // int32 readout, raw overflow cancels after zp
  check_fc(1, 8192, 1, 0);     // true overflow
  check_chain();
  check_packed_chain();
  check_late_overflow();
  puts(fail ? "INTEGER FC FAIL" : "INTEGER FC OK");
  return fail;
}
