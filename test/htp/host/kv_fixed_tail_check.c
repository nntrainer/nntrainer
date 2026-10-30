// SPDX-License-Identifier: Apache-2.0
/**
 * @file kv_fixed_tail_check.c
 * @brief kv fixed tail check validation.
 * @author NNTrainer contributors
 * @bug No known bugs.
 */
/** Actual DSP copy/row worker + actual libnative fixed quantizers.
 * Registration and fixed-scale setup use the unchanged host production TU.
 * The pool is the production pool with the existing pthread QuRT stand-in.
 * This proves byte ownership; no HMX, DMA or device timing is emulated.
 */
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__SANITIZE_ADDRESS__)
#include <sanitizer/asan_interface.h>
#define POISON(p, n) __asan_poison_memory_region((p), (n))
#define UNPOISON(p, n) __asan_unpoison_memory_region((p), (n))
#else
#define POISON(p, n) ((void)0)
#define UNPOISON(p, n) ((void)0)
#endif

#include "hvx_worker_pool.h"
#include "kv_fixed_tail_under_test.h"

static void copy_case(uint32_t n, uint32_t offset) {
  const size_t cap = (size_t)n + offset + 256u;
  uint8_t *src = malloc(cap), *dst = malloc(cap);
  assert(src && dst);
  for (size_t i = 0; i < cap; ++i) {
    src[i] = (uint8_t)(i * 73u + n * 11u);
    dst[i] = 0x5au;
  }
  // Keep the row's following bytes inaccessible even inside an allocation.
  POISON(src + offset + n, cap - offset - n);
  POISON(dst + offset + n, cap - offset - n);
  xor80_copy(dst + offset, src + offset, n);
  UNPOISON(src, cap);
  UNPOISON(dst, cap);
  for (size_t i = 0; i < cap; ++i) {
    const uint8_t want =
      i >= offset && i < offset + n ? (uint8_t)(src[i] ^ 0x80u) : 0x5au;
    assert(dst[i] == want);
  }
  free(src);
  free(dst);
}

static const uint16_t samples[] = {
  0x0000u, 0x8000u, 0x3800u, 0xb800u, 0x3c00u, 0xbc00u,
  0x4000u, 0xc000u, 0x4c00u, 0xcc00u, 0x0001u, 0x8001u,
};

static void fill_rows(uint16_t *k, uint16_t *v, size_t count, unsigned seed) {
  for (size_t i = 0; i < count; ++i) {
    k[i] = samples[(i * 7u + seed) % (sizeof(samples) / sizeof(samples[0]))];
    v[i] = samples[(i * 11u + 2u * seed + 1u) %
                   (sizeof(samples) / sizeof(samples[0]))];
  }
}

static void reference_rows(const quant_rows_ctx *c, uint8_t *k, uint8_t *v,
                           int32_t *sums, float *sk, float *sv) {
  const hexkl_kv_q *kv = c->kv;
  const uint32_t hd = kv->head_dim, stride = hd * kv->n_head_kv;
  int8_t q[HEXKL_KV_Q_MAX_HEAD_DIM] __attribute__((aligned(128)));
  for (uint32_t r = 0; r < c->n_rows; ++r) {
    for (uint32_t h = 0; h < kv->n_head_kv; ++h) {
      const size_t in = (size_t)r * stride + (size_t)h * hd;
      const size_t out = hexkl_kv_q_kt4_index(kv, h, c->row0 + r, 0);
      const size_t meta = hexkl_kv_q_sk_index(kv, h, c->row0 + r);
      hvx_kv_quant_k_row_fixed(c->k_rows + in, hd, kv->qmax, 1.0f / kv->fs_k[h],
                               q, sums + meta);
      for (uint32_t d = 0; d < hd; ++d)
        k[out + d] = (uint8_t)q[d] ^ 0x80u;
      hvx_kv_quant_v_row_fixed(c->v_rows + in, hd, kv->qmax,
                               kv->fs_v_inv + (size_t)h * hd, q);
      for (uint32_t d = 0; d < hd; ++d)
        v[out + d] = (uint8_t)q[d] ^ 0x80u;
      sk[meta] = kv->fs_k[h];
      sv[meta] = 1.0f;
    }
  }
}

static void worker_case(uint32_t hd, uint32_t heads, uint32_t row0,
                        uint32_t rows, uint32_t lanes, int reverse) {
  hexkl_kv_q_table *tbl = calloc(1u, sizeof(*tbl));
  uint32_t handle;
  assert(tbl &&
         hexkl_kv_q_register(tbl, HEXKL_KV_Q8, 64u, heads, hd, &handle) == 0);
  hexkl_kv_q *kv = (hexkl_kv_q *)hexkl_kv_q_get(tbl, handle);
  const size_t meta = (size_t)heads * kv->max_rows;
  const size_t values = meta * hd;
  const size_t inputs = (size_t)rows * heads * hd;
  float *fixed_k = malloc(heads * sizeof(float));
  float *fixed_v = malloc((size_t)heads * hd * sizeof(float));
  for (uint32_t h = 0; h < heads; ++h) {
    fixed_k[h] = (h & 1u) ? 0.125f : 0.0625f;
    for (uint32_t d = 0; d < hd; ++d)
      fixed_v[(size_t)h * hd + d] = (d & 1u) ? 0.03125f : 0.125f;
  }
  assert(hexkl_kv_q_set_fixed_scales(tbl, handle, fixed_k, fixed_v) == 0);
  assert(kv->plain_masters && kv->fixed);
  uint8_t *want_k = malloc(values), *want_v = malloc(values);
  int32_t *want_sum = malloc(meta * sizeof(int32_t));
  float *want_sk = malloc(meta * sizeof(float));
  float *want_sv = malloc(meta * sizeof(float));
  uint8_t *first_k = malloc(values), *first_v = malloc(values);
  uint16_t *kin = malloc(inputs * sizeof(uint16_t));
  uint16_t *vin = malloc(inputs * sizeof(uint16_t));
  assert(want_k && want_v && want_sum && want_sk && want_sv && first_k &&
         first_v && kin && vin);
  memcpy(want_k, kv->kt4, values);
  memcpy(want_v, kv->v4, values);
  memcpy(want_sum, kv->colsum_k, meta * sizeof(int32_t));
  memcpy(want_sk, kv->s_k, meta * sizeof(float));
  memcpy(want_sv, kv->s_v, meta * sizeof(float));
  hvx_worker_pool *pool = reverse ? NULL : hvx_worker_pool_create(lanes - 1u);
  assert(reverse || pool);
  quant_rows_ctx c = {kv, row0, rows, kin, vin};
  for (unsigned rep = 0; rep < 3u; ++rep) {
    fill_rows(kin, vin, inputs, rep == 1u ? 18u : 17u);
    reference_rows(&c, want_k, want_v, want_sum, want_sk, want_sv);
    if (reverse) {
      // Deterministic worker boundary stress: a later row finishes first.
      for (uint32_t lane = lanes; lane > 0u; --lane)
        quant_rows_worker(lanes, lane - 1u, &c);
    } else {
      hvx_worker_pool_run(pool, quant_rows_worker, &c, rows);
    }
    assert(memcmp(kv->kt4, want_k, values) == 0);
    assert(memcmp(kv->v4, want_v, values) == 0);
    assert(memcmp(kv->colsum_k, want_sum, meta * sizeof(int32_t)) == 0);
    assert(memcmp(kv->s_k, want_sk, meta * sizeof(float)) == 0);
    assert(memcmp(kv->s_v, want_sv, meta * sizeof(float)) == 0);
    if (rep == 0u) {
      memcpy(first_k, kv->kt4, values);
      memcpy(first_v, kv->v4, values);
    } else if (rep == 1u) {
      assert(memcmp(first_k, kv->kt4, values) != 0);
    } else {
      assert(memcmp(first_k, kv->kt4, values) == 0);
      assert(memcmp(first_v, kv->v4, values) == 0);
    }
  }
  hvx_worker_pool_destroy(pool);
  assert(hexkl_kv_q_release(tbl, handle) == 0);
  free(tbl);
  free(fixed_k);
  free(fixed_v);
  free(want_k);
  free(want_v);
  free(want_sum);
  free(want_sk);
  free(want_sv);
  free(first_k);
  free(first_v);
  free(kin);
  free(vin);
}

int main(int argc, char **argv) {
  if (argc == 2 && strcmp(argv[1], "--copy-overflow") == 0) {
    uint8_t *src = calloc(512u, 1u), *dst = malloc(64u);
    assert(src && dst);
    xor80_copy(dst, src, 64u);
    free(src);
    free(dst);
    return 0;
  }
  if (argc == 2 && strcmp(argv[1], "--worker-order") == 0) {
    worker_case(64u, 2u, 3u, 16u, 4u, 1);
    return 0;
  }
  assert(argc == 1);
  unsigned copies = 0u, workers = 0u;
  const uint32_t offsets[] = {0u, 1u, 4u, 31u, 63u, 127u};
  for (uint32_t n = 0u; n <= 512u; ++n) {
    for (unsigned o = 0; o < sizeof(offsets) / sizeof(offsets[0]); ++o) {
      copy_case(n, offsets[o]);
      ++copies;
    }
  }
  for (uint32_t hd = 32u; hd <= 512u; hd += 32u) {
    // The other production call copies a whole stage: always 128B-aligned size.
    copy_case(32u * hd, 0u);
    ++copies;
    for (unsigned reverse = 0u; reverse < 2u; ++reverse) {
      for (uint32_t lanes = 1u; lanes <= 8u; lanes *= 2u) {
        worker_case(hd, 3u, 0u, 64u, lanes, reverse);
        worker_case(hd, 2u, 31u, 33u, lanes, reverse);
        worker_case(hd, 2u, 3u, 5u, lanes, reverse);
        workers += 3u;
      }
    }
  }
  printf("PASS actual copy %u cases; actual fixed quant worker %u A/B/A "
         "cases (real HVX/libnative + pthread pool, no HMX).\n",
         copies, workers);
  return 0;
}
