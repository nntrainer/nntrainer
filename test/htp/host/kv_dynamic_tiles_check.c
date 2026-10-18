// SPDX-License-Identifier: Apache-2.0
/**
 * @file kv_dynamic_tiles_check.c
 * @brief Exact dynamic KV masters, scales and WH tile ownership under workers.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug No known bugs.
 */
#include "kv_dynamic_under_test.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void compare(const hexkl_kv_q *a, const hexkl_kv_q *b) {
  const size_t rows = (size_t)a->max_rows * a->n_head_kv;
  const size_t values = rows * a->head_dim;
  const size_t tiles = rows / 32u * a->n_dot_tiles * a->tile_bytes;
  assert(memcmp(a->kt4, b->kt4, values) == 0);
  assert(memcmp(a->v4, b->v4, values) == 0);
  assert(memcmp(a->s_k, b->s_k, rows * sizeof(float)) == 0);
  assert(memcmp(a->s_v, b->s_v, rows * sizeof(float)) == 0);
  assert(memcmp(a->colsum_k, b->colsum_k, rows * sizeof(int32_t)) == 0);
  assert(memcmp(a->kt, b->kt, tiles) == 0);
  assert(memcmp(a->v, b->v, tiles) == 0);
}
static void check(unsigned row0, unsigned rows, unsigned lanes) {
  hexkl_kv_q_table *table = calloc(1, sizeof(*table));
  uint32_t ah, bh;
  assert(table);
  assert(hexkl_kv_q_register(table, HEXKL_KV_Q8, row0 + rows + 32u, 8u, 64u,
                             &ah) == 0);
  assert(hexkl_kv_q_register(table, HEXKL_KV_Q8, row0 + rows + 32u, 8u, 64u,
                             &bh) == 0);
  hexkl_kv_q *a = (hexkl_kv_q *)hexkl_kv_q_get(table, ah);
  hexkl_kv_q *b = (hexkl_kv_q *)hexkl_kv_q_get(table, bh);
  const size_t count = (size_t)rows * 8u * 64u;
  uint16_t *k = malloc(count * sizeof(uint16_t)),
           *v = malloc(count * sizeof(uint16_t));
  hvx_worker_pool *pool = hvx_worker_pool_create(lanes);
  assert(k && v && pool);
  for (unsigned pass = 0; pass < 3; ++pass) {
    for (size_t i = 0; i < count; ++i) {
      const unsigned seed = pass == 1 ? 17u : 5u;
      k[i] = (uint16_t)(((i * 73u + seed) % 0x4400u) | ((i & 1u) << 15u));
      v[i] = (uint16_t)(((i * 137u + seed) % 0x4800u) | ((i & 2u) << 14u));
    }
    append_dynamic_heads(a, row0, rows, k, v, 1, 0u, 8u);
    dynamic_heads_ctx c = {b, row0, rows, k, v, 1};
    hvx_worker_pool_run(pool, dynamic_heads_worker, &c, 8u * rows);
    compare(a, b);
  }
  hvx_worker_pool_destroy(pool);
  assert(hexkl_kv_q_release(table, ah) == 0);
  assert(hexkl_kv_q_release(table, bh) == 0);
  free(table);
  free(k);
  free(v);
}
int main(void) {
  for (unsigned r = 0; r < 32u; ++r)
    for (unsigned c = 0; c < 32u; ++c)
      g_wh_i8_pos[32u * r + c] = (r / 4u) * 128u + 4u * c + r % 4u;
  const unsigned lanes[] = {1u, 2u, 3u, 4u, 6u, 8u};
  unsigned cases = 0;
  for (unsigned i = 0; i < sizeof(lanes) / sizeof(lanes[0]); ++i) {
    check(0, 64, lanes[i]);
    check(32, 128, lanes[i]);
    check(3, 65, lanes[i]);
    check(32, 127, lanes[i]);
    cases += 4;
  }
  check(0, 1024, 6);
  ++cases;
  printf("Dynamic KV: %u exact A/B/A serial-HVX/worker cases, full "
         "masters/metadata/WH and untouched rows PASS\n",
         cases);
}
