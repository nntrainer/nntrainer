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
static void check_prefix(uint32_t flags) {
  int err;
  const uint32_t hd = 256, nk = 2, gqa = 2;
  const size_t row = (size_t)nk * hd, output_bytes = row * gqa * sizeof(float);
  float *k = malloc(1024 * row * sizeof(float));
  float *v = malloc(1024 * row * sizeof(float));
  float *q = malloc(output_bytes), *x = malloc(output_bytes),
        *y = malloc(output_bytes);
  for (size_t i = 0; i < 1024 * row; ++i) {
    k[i] = ((int)(i % 113) - 56) * 0.0125f;
    v[i] = ((int)(i % 71) - 35) * 0.025f;
  }
  for (size_t i = 0; i < row * gqa; ++i)
    q[i] = ((int)(i % 37) - 18) * 0.02f;
  hvx_attn_m1_ctx *source = hvx_attn_m1_create(
    1, nk, gqa, hd, 1088 | HVX_ATTN_M1_Q8 | HVX_ATTN_M1_PAGED | flags, NULL,
    &err);
  hvx_attn_m1_ctx *reference = hvx_attn_m1_create(
    1, nk, gqa, hd, 1088 | HVX_ATTN_M1_Q8 | flags, NULL, &err);
  assert(source && reference);
  assert(!hvx_attn_m1_kv_append(source, 0, 0, 1024, k, v));
  assert(!hvx_attn_m1_kv_append(reference, 0, 0, 1024, k, v));
  equal_layer(source, reference);
  // A global boundary can end inside a sliding cache's local tile. Reusing
  // that partial tile must not expose its stale suffix or overwrite source.
  hvx_attn_m1_ctx *partial = hvx_attn_m1_clone(source, &err);
  assert(partial && !err && !hvx_attn_m1_truncate(partial, 0, 769));
  reference->kv_len[0] = 769;
  equal_layer(partial, reference);
  assert(partial->pages[12] == source->pages[12]);
  assert(!hvx_attn_m1_forward(partial, 0, 769, 0, 0.0625f, q, k, v, x, NULL));
  assert(!hvx_attn_m1_forward(reference, 0, 769, 0, 0.0625f, q, k, v, y, NULL));
  assert(!memcmp(x, y, output_bytes));
  assert(partial->pages[12] != source->pages[12] && source->kv_len[0] == 1024);
  hvx_attn_m1_free(partial);
  hvx_attn_m1_ctx *prefix = hvx_attn_m1_clone(source, &err);
  assert(prefix && !err);
  assert(hvx_attn_m1_truncate(prefix, 0, 1025) == AEE_EINVALIDFORMAT);
  assert(prefix->kv_len[0] == 1024);
  assert(!hvx_attn_m1_truncate(prefix, 0, 768));
  reference->kv_len[0] = 768; // independent linear view of the encoded prefix
  equal_layer(prefix, reference);
  assert(source->kv_len[0] == 1024 && prefix->pages[0] == source->pages[0]);
  assert(hvx_attn_m1_payload_bytes(prefix) == 2u * 768u * row);
  for (uint32_t h = 0; h < nk; ++h) {
    const size_t base = (size_t)h * (prefix->seq / 64u);
    for (uint32_t t = 12; t < prefix->seq / 64u; ++t)
      assert(!prefix->pages[base + t]);
  }
  for (uint32_t pos = 768; pos < 771; ++pos) {
    assert(!hvx_attn_m1_forward(prefix, 0, pos, 0, 0.0625f, q, k, v, x, NULL));
    assert(
      !hvx_attn_m1_forward(reference, 0, pos, 0, 0.0625f, q, k, v, y, NULL));
    assert(!memcmp(x, y, output_bytes));
  }
  // Recomputed suffix owns new pages; the original source remains usable.
  assert(prefix->pages[12] != source->pages[12]);
  assert(source->kv_len[0] == 1024);
  hvx_attn_m1_free(source);
  hvx_attn_m1_free(prefix);
  hvx_attn_m1_free(reference);
  free(k);
  free(v);
  free(q);
  free(x);
  free(y);
}

int main(int argc, char **argv) {
  check_prefix(0);
  check_prefix(HVX_ATTN_M1_FAST);
  check_prefix(HVX_ATTN_M1_FAST | HVX_ATTN_M1_QK_I8);
  if (argc == 2 && !strcmp(argv[1], "--prefix-only")) {
    puts("1024 -> 768 page prefix, exact encoded attention, source "
         "preservation: PASS");
    return 0;
  }

  check(2, 2, 64, 0, 0);
  check(8, 2, 256, 0, 3);
  check(2, 8, 512, 0, 3);
  check(2, 8, 512, HVX_ATTN_M1_FAST, 3);
  check(2, 8, 512, HVX_ATTN_M1_FAST | HVX_ATTN_M1_QK_I8, 3);
  puts("DSP paged/linear exact attention, COW, window, snapshot: PASS");
}
