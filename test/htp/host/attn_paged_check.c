// SPDX-License-Identifier: Apache-2.0
/** @file attn_paged_check.c
 * @brief Paged/linear exact output, COW branches, window release and snapshots.
 */
#include "hvx_attn_m1_f32.h"
#include <AEEStdErr.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void equal_layer(hvx_attn_m1_ctx *a, hvx_attn_m1_ctx *b) {
  const size_t n = hvx_attn_m1_snapshot_bytes(a, 0);
  assert(n == hvx_attn_m1_snapshot_bytes(b, 0));
  uint8_t *x = malloc(n), *y = malloc(n);
  assert(!hvx_attn_m1_snapshot_export(a, 0, x, n));
  assert(!hvx_attn_m1_snapshot_export(b, 0, y, n));
  assert(!memcmp(x, y, n));
  free(x);
  free(y);
}

static void check(uint32_t nk, uint32_t gqa, uint32_t hd, uint32_t flags,
                  uint32_t workers) {
  int err;
  hvx_worker_pool *pool = workers ? hvx_worker_pool_create(workers) : NULL;
  assert(!workers || pool);
  hvx_attn_m1_ctx *a = hvx_attn_m1_create(
    1, nk, gqa, hd, 384 | HVX_ATTN_M1_Q8 | flags, pool, &err);
  hvx_attn_m1_ctx *b = hvx_attn_m1_create(
    1, nk, gqa, hd, 384 | HVX_ATTN_M1_Q8 | HVX_ATTN_M1_PAGED | flags, pool,
    &err);
  assert(a && b && !err);
  assert(!hvx_attn_m1_payload_bytes(b));
  const size_t row = (size_t)nk * hd, out_bytes = row * gqa * sizeof(float);
  float *k = malloc(131 * row * sizeof(float)),
        *v = malloc(131 * row * sizeof(float));
  float *q = malloc(out_bytes), *x = malloc(out_bytes), *y = malloc(out_bytes);
  for (size_t i = 0; i < 131 * row; ++i) {
    k[i] = ((int)(i % 113) - 56) * 0.0125f;
    v[i] = ((int)(i % 71) - 35) * 0.025f;
  }
  for (size_t i = 0; i < row * gqa; ++i)
    q[i] = ((int)(i % 37) - 18) * 0.02f;
  assert(!hvx_attn_m1_kv_append(a, 0, 0, 131, k, v));
  assert(!hvx_attn_m1_kv_append(b, 0, 0, 131, k, v));
  equal_layer(a, b);
  assert(hvx_attn_m1_payload_bytes(b) == 2 * 192 * row);
  hvx_attn_m1_ctx *fork = hvx_attn_m1_clone(b, &err);
  assert(fork && !err && fork->pages[0] == b->pages[0]);
  const size_t frozen_bytes = hvx_attn_m1_snapshot_bytes(fork, 0);
  uint8_t *frozen = malloc(frozen_bytes), *copy = malloc(frozen_bytes);
  assert(!hvx_attn_m1_snapshot_export(fork, 0, frozen, frozen_bytes));
  for (uint32_t pos = 131; pos < 322; ++pos) {
    assert(!hvx_attn_m1_forward(a, 0, pos, pos & 1 ? 129 : 0, 0.0625f, q, k, v,
                                x, NULL));
    assert(!hvx_attn_m1_forward(b, 0, pos, pos & 1 ? 129 : 0, 0.0625f, q, k, v,
                                y, NULL));
    assert(!memcmp(x, y, out_bytes));
  }
  assert(fork->pages[0] == b->pages[0]);
  assert(fork->pages[2] != b->pages[2]); // shared partial tile detached
  assert(!hvx_attn_m1_snapshot_export(fork, 0, copy, frozen_bytes));
  assert(!memcmp(copy, frozen, frozen_bytes));
  equal_layer(a, b);
  hvx_attn_m1_ctx *window = hvx_attn_m1_reserve(b, 256, 128, &err);
  assert(window && !err && window->kv_len[0] == 194);
  assert(window->pages[0] == b->pages[2]);
  hvx_attn_m1_ctx *linear_window = hvx_attn_m1_reserve(a, 256, 128, &err);
  assert(linear_window && !err);
  equal_layer(linear_window, window);
  assert(!hvx_attn_m1_forward(window, 0, 194, 129, 0.0625f, q, k, v, x, NULL));
  assert(!hvx_attn_m1_forward(linear_window, 0, 194, 129, 0.0625f, q, k, v, y,
                              NULL));
  assert(!memcmp(x, y, out_bytes));
  // Import into a shared branch must detach rather than overwrite another
  // session.
  assert(!hvx_attn_m1_snapshot_import(fork, 0, frozen, frozen_bytes));
  hvx_attn_m1_park(b);
  assert(!hvx_attn_m1_payload_bytes(b));
  assert(!hvx_attn_m1_snapshot_export(fork, 0, copy, frozen_bytes));
  assert(!memcmp(copy, frozen, frozen_bytes));
  hvx_attn_m1_free(a);
  hvx_attn_m1_free(b);
  hvx_attn_m1_free(fork);
  hvx_attn_m1_free(window);
  hvx_attn_m1_free(linear_window);
  if (pool)
    hvx_worker_pool_destroy(pool);
  free(k);
  free(v);
  free(q);
  free(x);
  free(y);
  free(frozen);
  free(copy);
}
int main(void) {
  check(2, 2, 64, 0, 0);
  check(8, 2, 256, 0, 3);
  check(2, 8, 512, 0, 3);
  check(2, 8, 512, HVX_ATTN_M1_FAST, 3);
  check(2, 8, 512, HVX_ATTN_M1_FAST | HVX_ATTN_M1_QK_I8, 3);
  puts("DSP paged/linear exact attention, COW, window, snapshot: PASS");
}
