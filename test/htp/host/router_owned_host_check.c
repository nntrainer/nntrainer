// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file router_owned_host_check.c
 * @brief Lossless router storage against the existing FP32 kernel and spec.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hvx_m1_ops_f32.h"
#include "m1_ops_det.h"

#ifndef NNTR_ROUTER_BF16_IF_EXACT
#define NNTR_ROUTER_BF16_IF_EXACT 0
#endif

static uint32_t rng = 0x419bc272u;
static int failed;
static unsigned checked;

#define CHECK(c, ...)                                                          \
  do {                                                                         \
    if (!(c)) {                                                                \
      fprintf(stderr, "FAIL: " __VA_ARGS__);                                   \
      fputc('\n', stderr);                                                     \
      failed = 1;                                                              \
    }                                                                          \
  } while (0)

static float random_value(void) {
  rng = rng * 1664525u + 1013904223u;
  return ((float)(rng >> 8) / 16777216.0f - 0.5f) * 0.125f;
}

static void check_storage(const float *w, uint32_t K, uint32_t E,
                          const hvx_router_sm_weights *p, int expected) {
  CHECK(p && p->K == K && p->E == E && p->bf16 == (uint32_t)expected,
        "storage K=%u E=%u expected bf16=%d", K, E, expected);
  if (!p)
    return;
  CHECK(((uintptr_t)p & 127u) == 0u, "weight alignment");
  const uint16_t *h = (const uint16_t *)(p + 1);
  const float *f = (const float *)(p + 1);
  const uint32_t rows = p->bf16 ? p->Kp : K;
  for (uint32_t b = 0; b < p->Ep / 32u; ++b) {
    for (uint32_t k = 0; k < rows; ++k) {
      for (uint32_t j = 0; j < 32u; ++j) {
        const size_t at = ((size_t)b * rows + k) * 32u + j;
        const uint32_t e = b * 32u + j;
        uint32_t ref = 0u, got;
        if (k < K && e < E)
          memcpy(&ref, w + (size_t)k * E + e, sizeof(ref));
        if (p->bf16)
          got = (uint32_t)h[at] << 16;
        else
          memcpy(&got, f + at, sizeof(got));
        CHECK(got == ref, "payload K=%u E=%u k=%u e=%u", K, E, k, e);
      }
    }
  }
}

static void check_forward(const float *w, uint32_t K, uint32_t E,
                          hvx_worker_pool *const pools[3], int exact) {
  hvx_router_sm_weights *p = hvx_router_sm_weights_create(w, K, E);
  check_storage(w, K, E, p, exact && NNTR_ROUTER_BF16_IF_EXACT);
  if (!p)
    return;
  const uint32_t Ep = (E + 31u) / 32u * 32u, top = E < 8u ? E : 8u;
  float *packed = malloc((size_t)K * Ep * sizeof(float));
  float *xbuf = malloc((K + 2u) * sizeof(float)), *x = xbuf + 1u;
  float pes[128], ref[128], old[128], out[130], rw[128], ow[128], ww[128];
  uint32_t rs[128], os[128], ss[128];
  hvx_router_sm_pack_f32(packed, w, K, E);
  for (uint32_t e = 0; e < E; ++e)
    pes[e] = 0.75f + (float)e / 256.0f;
  for (uint32_t t = 0; t < 3u; ++t) {
    for (uint32_t k = 0; k < K; ++k)
      x[k] = t == 1u ? 0.0f : random_value() * (t == 2u ? 100.0f : 1.0f);
    m1_router_softmax_det(x, w, pes, K, E, top, ref, rs, rw);
    for (uint32_t pi = 0; pi < 3u; ++pi) {
      out[0] = 123.5f;
      out[E + 1u] = -987.25f;
      hvx_router_softmax_topk_f32(x, packed, pes, K, E, top, old, os, ow,
                                  pools[pi]);
      hvx_router_softmax_topk_owned_f32(x, p, pes, top, out + 1u, ss, ww,
                                        pools[pi]);
      CHECK(memcmp(ref, old, E * sizeof(float)) == 0 &&
              memcmp(ref, out + 1u, E * sizeof(float)) == 0 &&
              memcmp(rs, os, top * sizeof(uint32_t)) == 0 &&
              memcmp(rs, ss, top * sizeof(uint32_t)) == 0 &&
              memcmp(rw, ow, top * sizeof(float)) == 0 &&
              memcmp(rw, ww, top * sizeof(float)) == 0,
            "forward K=%u E=%u t=%u workers=%u", K, E, t, pi);
      CHECK(out[0] == 123.5f && out[E + 1u] == -987.25f, "output canary");
      ++checked;
    }
  }
  free(xbuf);
  free(packed);
  free(p);
}

static void check_synthetic(hvx_worker_pool *const pools[3]) {
  const uint32_t ks[] = {1u, 2u, 31u, 255u, 256u, 257u, 2816u};
  const uint32_t es[] = {1u, 31u, 32u, 33u, 127u, 128u};
  for (size_t ki = 0; ki < sizeof(ks) / sizeof(ks[0]); ++ki) {
    for (size_t ei = 0; ei < sizeof(es) / sizeof(es[0]); ++ei) {
      const uint32_t K = ks[ki], E = es[ei];
      const size_t count = (size_t)K * E;
      float *w = malloc(count * sizeof(float));
      for (size_t j = 0; j < count; ++j) {
        const float f = random_value();
        uint32_t bits;
        memcpy(&bits, &f, sizeof(bits));
        bits &= 0xFFFF0000u;
        memcpy(w + j, &bits, sizeof(bits));
      }
      /* Preserve both zero signs and exact BF16 subnormals. */
      const uint32_t special[] = {0x80000000u, 0x00010000u, 0x80010000u};
      for (size_t j = 0; j < count && j < 3u; ++j)
        memcpy(w + j, special + j, sizeof(float));
      check_forward(w, K, E, pools, 1);
      uint32_t bits;
      memcpy(&bits, w + count / 2u, sizeof(bits));
      bits |= 1u; /* one inexact source must retain the whole FP32 weight */
      memcpy(w + count / 2u, &bits, sizeof(bits));
      check_forward(w, K, E, pools, 0);
      free(w);
    }
  }
  const uint32_t nonfinite[] = {0x7F800000u, 0xFF800000u, 0x7FC00000u,
                                0x7FC01234u};
  for (size_t i = 0; i < sizeof(nonfinite) / sizeof(nonfinite[0]); ++i) {
    float w[6] = {1.0f, -1.0f, 0.5f, 0.0f, 2.0f, 3.0f};
    memcpy(w + 4u, nonfinite + i, sizeof(float));
    hvx_router_sm_weights *p = hvx_router_sm_weights_create(w, 3u, 2u);
    check_storage(w, 3u, 2u, p, 0);
    /* Nonfinite logits are outside the existing selection contract. */
    free(p);
  }
}

/** Optional actual-model replay: TSV has 256 expert offsets per layer;
   the router and per-expert scales immediately precede the first expert. */
static void check_model(const char *model, const char *index,
                        hvx_worker_pool *const pools[3]) {
  FILE *f = fopen(model, "rb"), *tsv = fopen(index, "r");
  CHECK(f && tsv, "model replay open");
  if (!f || !tsv)
    goto done;
  const uint32_t K = 2816u, E = 128u;
  float *w = malloc((size_t)K * E * sizeof(float));
  unsigned long long off;
  unsigned k, n, row = 0u, layers = 0u;
  while (fscanf(tsv, "%llu %u %u", &off, &k, &n) == 3) {
    if (row++ % 256u)
      continue;
    const unsigned long long roff = off - (unsigned long long)(K + 1u) * E * 4u;
    CHECK(k == K && n == 1408u && fseek(f, (long)roff, SEEK_SET) == 0 &&
            fread(w, sizeof(float), (size_t)K * E, f) == (size_t)K * E,
          "model layer %u read", layers);
    check_forward(w, K, E, pools, 1);
    ++layers;
  }
  CHECK(layers == 30u, "model layer count %u", layers);
  free(w);
done:
  if (f)
    fclose(f);
  if (tsv)
    fclose(tsv);
}

int main(int argc, char **argv) {
  hvx_worker_pool *pools[3] = {NULL, hvx_worker_pool_create(3u),
                               hvx_worker_pool_create(7u)};
  CHECK(pools[1] && pools[2], "worker pools");
  check_synthetic(pools);
  if (argc == 3)
    check_model(argv[1], argv[2], pools);
  hvx_worker_pool_destroy(pools[1]);
  hvx_worker_pool_destroy(pools[2]);
  printf("ROUTER OWNED bf16_enabled=%u checks=%u %s\n",
         (unsigned)NNTR_ROUTER_BF16_IF_EXACT, checked,
         failed ? "FAIL" : "PASS");
  return failed;
}
