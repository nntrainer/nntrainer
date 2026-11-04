// SPDX-License-Identifier: Apache-2.0
/** @file attn_snapshot_check.c
 * @brief INT8 snapshot round trip, rejected corruption and exact continuation.
 */
#include "hvx_attn_m1_f32.h"
#include <AEEStdErr.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void check(uint32_t flags, uint32_t drop) {
  int err;
  const uint32_t hd = 256, nk = 2, gqa = 2, layers = 2, rows = 127;
  hvx_attn_m1_ctx *src = hvx_attn_m1_create(
    layers, nk, gqa, hd, 256u | HVX_ATTN_M1_Q8 | flags, NULL, &err);
  assert(src && err == AEE_SUCCESS);
  float *k = malloc((size_t)rows * nk * hd * sizeof(float));
  float *v = malloc((size_t)rows * nk * hd * sizeof(float));
  for (size_t i = 0; i < (size_t)rows * nk * hd; ++i) {
    k[i] = ((int)(i % 113) - 56) * 0.0125f;
    v[i] = ((int)(i % 71) - 35) * 0.025f;
  }
  for (uint32_t l = 0; l < layers; ++l)
    assert(hvx_attn_m1_kv_append(src, l, 0, rows, k, v) == AEE_SUCCESS);
  if (drop) {
    hvx_attn_m1_ctx *next = hvx_attn_m1_reserve(src, 192, drop, &err);
    assert(next && !err);
    hvx_attn_m1_free(src);
    src = next;
  }
  hvx_attn_m1_ctx *dst = hvx_attn_m1_create(
    layers, nk, gqa, hd,
    src->max_seq | HVX_ATTN_M1_Q8 | HVX_ATTN_M1_LAZY | flags, NULL, &err);
  assert(dst && !err);
  for (uint32_t l = 0; l < layers; ++l) {
    size_t bytes = hvx_attn_m1_snapshot_bytes(src, l);
    uint8_t *data = malloc(bytes), *copy = malloc(bytes);
    assert(!hvx_attn_m1_snapshot_export(src, l, data, bytes));
    assert(hvx_attn_m1_snapshot_import(dst, l, data, bytes - 1) ==
           AEE_EINVALIDFORMAT);
    uint32_t bad = 0x7f800000u, saved;
    memcpy(&saved, data + 8, 4);
    memcpy(data + 8, &bad, 4);
    assert(hvx_attn_m1_snapshot_import(dst, l, data, bytes) ==
           AEE_EINVALIDFORMAT);
    assert(dst->kv_len[l] == 0 && !dst->kt[l]);
    memcpy(data + 8, &saved, 4);
    assert(!hvx_attn_m1_snapshot_import(dst, l, data, bytes));
    assert(!hvx_attn_m1_snapshot_export(dst, l, copy, bytes));
    assert(!memcmp(data, copy, bytes));
    free(data);
    free(copy);
  }
  float *q = calloc((size_t)nk * gqa * hd, sizeof(float));
  float *a = malloc((size_t)nk * gqa * hd * sizeof(float));
  float *b = malloc((size_t)nk * gqa * hd * sizeof(float));
  for (size_t i = 0; i < (size_t)nk * gqa * hd; ++i)
    q[i] = ((int)(i % 37) - 18) * 0.02f;
  for (uint32_t l = 0; l < layers; ++l) {
    // A fresh row proves both the historical bytes and fixed quantizer state.
    assert(!hvx_attn_m1_kv_append(src, l, src->kv_len[l], 1, k, v));
    assert(!hvx_attn_m1_kv_append(dst, l, dst->kv_len[l], 1, k, v));
    const uint32_t pos = src->kv_len[l] - 1u;
    assert(!hvx_attn_m1_forward(src, l, pos, 0, 0.0625f, q, k, v, a, NULL));
    assert(!hvx_attn_m1_forward(dst, l, pos, 0, 0.0625f, q, k, v, b, NULL));
    assert(!memcmp(a, b, (size_t)nk * gqa * hd * sizeof(float)));
  }
  hvx_attn_m1_park(dst);
  for (uint32_t l = 0; l < layers; ++l)
    assert(!dst->kt[l] && !dst->v[l] && !dst->kv_len[l]);
  hvx_attn_m1_free(src);
  hvx_attn_m1_free(dst);
  free(k);
  free(v);
  free(q);
  free(a);
  free(b);
}
int main(void) {
  check(0, 0);
  check(0, 64);
  check(HVX_ATTN_M1_FAST, 0);
  check(HVX_ATTN_M1_FAST | HVX_ATTN_M1_QK_I8, 64);
  puts("INT8 snapshot round trips, corruption rejection, park and exact "
       "decode: PASS");
  return 0;
}
