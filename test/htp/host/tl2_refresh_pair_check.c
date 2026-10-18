// SPDX-License-Identifier: Apache-2.0
/**
 * @file tl2_refresh_pair_check.c
 * @brief Real native registry refresh with deferred DMA and cache checks.
 * @author NNTrainer contributors
 * @bug See the project issue tracker.
 */
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hexkl_mm_u8i4_dma.h"

static unsigned allocations, frees, refreshing;
static hexkl_weight_u8i4 *gu, *dn;
static void *checked_malloc(size_t bytes) {
  ++allocations;
  void *p = malloc(bytes);
  if (p)
    memset(p, 0xa5, bytes);
  return p;
}
static void checked_free(void *p) {
  frees += p != NULL;
  free(p);
}
static void *checked_memset(void *p, int value, size_t bytes) {
  /* Bias is immutable during refresh. Register/rebind must initialize it. */
  assert(!refreshing || (p != gu->bias && p != dn->bias));
  return memset(p, value, bytes);
}

/** Keep the production register/rebind/release and refresh bodies intact.
 * Unused compute entry points are discarded by --gc-sections. */
#define malloc checked_malloc
#define free checked_free
#define memset checked_memset
#include "hexkl_mm_u8i4_dma.c"
#undef malloc
#undef free
#undef memset

typedef struct {
  void *dst;
  const void *src;
  uint32_t ds, ss, rs, nr;
} queued_copy;
static queued_copy queue[4];
static unsigned queued, pushes, drains, invalidations, source_invalidated;
static uint32_t seen_before;
static const uint8_t *gu_tail, *dn_tail;
static uint32_t *generation_word;

static void reset_trace(void) {
  assert(queued == 0u);
  pushes = drains = invalidations = source_invalidated = 0u;
}

void hexkl_dma_ring_push2d(void *dst, const void *src, uint32_t ds, uint32_t ss,
                           uint32_t rs, uint32_t nr, int sv, int dv) {
  assert(queued < 4u);
  assert(sv == 0 && dv == 0);
  if (refreshing) {
    assert(gu->tl2_generation_seen == seen_before);
    assert(source_invalidated == 3u);
    const hexkl_weight_u8i4 *w = pushes == 0u ? gu : dn;
    assert(pushes < 2u && nr == 2u && rs == 4u * w->N);
    assert(ds == rs && ss == rs && dst == w->w_scale);
    assert(src == (pushes == 0u ? gu_tail : dn_tail));
  }
  queue[queued++] = (queued_copy){dst, src, ds, ss, rs, nr};
  ++pushes;
}

void hexkl_dma_ring_drain(void) {
  if (refreshing) {
    assert(queued == 2u && pushes == 2u && drains == 0u);
    assert(gu->tl2_generation_seen == seen_before);
  }
  /* Deferred completion makes an early invalidation or publication visible. */
  for (unsigned i = 0; i < queued; ++i) {
    const queued_copy *q = queue + i;
    for (uint32_t r = 0; r < q->nr; ++r)
      memcpy((uint8_t *)q->dst + (size_t)r * q->ds,
             (const uint8_t *)q->src + (size_t)r * q->ss, q->rs);
  }
  queued = 0u;
  ++drains;
}

void hexkl_dcache_inva(const void *p, uint32_t bytes) {
  if (refreshing) {
    assert(gu->tl2_generation_seen == seen_before);
    if (p == generation_word) {
      assert(bytes == 128u && invalidations == 0u);
    } else if (p == gu_tail) {
      assert(bytes == 8u * gu->N && pushes == 0u);
      source_invalidated |= 1u;
    } else if (p == dn_tail) {
      assert(bytes == 8u * dn->N && pushes == 0u);
      source_invalidated |= 2u;
    } else {
      assert(queued == 0u && drains == 1u && pushes == 2u);
      if (p == gu->w_scale) {
        assert(bytes == 8u * gu->N);
        assert(memcmp(p, gu_tail, bytes) == 0);
      } else {
        assert(p == dn->w_scale && bytes == 8u * dn->N);
        assert(memcmp(p, dn_tail, bytes) == 0);
      }
    }
  }
  ++invalidations;
}

static void check_zero(const float *p, uint32_t n) {
  for (uint32_t i = 0; i < n; ++i) {
    uint32_t bits;
    memcpy(&bits, p + i, sizeof(bits));
    assert(bits == 0u);
  }
}

static void fill_tail(uint8_t *p, uint32_t n, unsigned salt) {
  for (uint32_t i = 0; i < n; ++i) {
    float scale = 0.125f + (float)(i + 17u * salt) / 16.0f;
    int32_t sum = (int32_t)i - 35 * (int32_t)salt;
    memcpy(p + 4u * i, &scale, 4u);
    memcpy(p + 4u * (n + i), &sum, 4u);
  }
}

int main(void) {
  enum { K = 2048, I = 1536, N = 2048 };
  const uint32_t gb = (uint32_t)htp_tl2_native_bytes(K, 2u * I);
  const uint32_t db = (uint32_t)htp_tl2_native_bytes(I, N);
  const uint32_t tail_end = gb + 16u * I;
  const uint32_t down = (tail_end + 4095u) & ~4095u;
  const uint32_t cap = down + db + 8u * N;
  uint8_t *arena_allocation = malloc((size_t)cap + 4096u + 128u);
  assert(arena_allocation);
  uint8_t *arena =
    (uint8_t *)(((uintptr_t)arena_allocation + 4095u) & ~(uintptr_t)4095u);
  memset(arena, 0x6d, cap + 128u);
  gu_tail = arena + gb;
  dn_tail = arena + down + db;
  generation_word = (uint32_t *)(arena + tail_end);
  *generation_word = 0u;
  fill_tail((uint8_t *)gu_tail, 2u * I, 1u);
  fill_tail((uint8_t *)dn_tail, N, 2u);
  hexkl_weight_u8i4_table *table = calloc(1, sizeof(*table));
  assert(table);
  uint32_t gh, dh, rg, rd;
  assert(hexkl_weight_tl2_compact_register_arena(table, 8u << 20, K, 2u * I,
                                                 arena, &gh) == AEE_SUCCESS);
  assert(hexkl_weight_tl2_compact_register_arena(
           table, 8u << 20, I, N, arena + down, &dh) == AEE_SUCCESS);
  assert(hexkl_weight_tl2_compact_register_arena(table, 8u << 20, K, 2u * I,
                                                 arena, &rg) == AEE_SUCCESS);
  assert(hexkl_weight_tl2_compact_register_arena(
           table, 8u << 20, I, N, arena + down, &rd) == AEE_SUCCESS);
  gu = table->slots + gh;
  dn = table->slots + dh;
  assert((uint8_t *)gu->colsum_w == (uint8_t *)gu->w_scale + 4u * gu->N);
  assert((uint8_t *)dn->colsum_w == (uint8_t *)dn->w_scale + 4u * dn->N);
  check_zero(gu->bias, gu->N);
  check_zero(dn->bias, dn->N);
  hexkl_weight_tl2_set_generation(gu, generation_word);
  void *ga = gu->arrays, *da = dn->arrays;

  const uint32_t gens[] = {1u, 2u, 9u, UINT32_MAX, 0u, 3u};
  for (unsigned j = 0; j < sizeof(gens) / sizeof(*gens); ++j) {
    fill_tail((uint8_t *)gu_tail, gu->N, j + 10u);
    fill_tail((uint8_t *)dn_tail, dn->N, j + 20u);
    seen_before = gu->tl2_generation_seen;
    const unsigned old_allocations = allocations, old_frees = frees;
    reset_trace();
    refreshing = 1u;
    assert(hexkl_weight_tl2_refresh_pair(gu, dn) == AEE_SUCCESS);
    assert(pushes == 0u && drains == 0u && invalidations == 1u);
    assert(gu->tl2_generation_seen == seen_before);
    *generation_word = gens[j];
    reset_trace();
    assert(hexkl_weight_tl2_refresh_pair(gu, dn) == AEE_SUCCESS);
    refreshing = 0u;
    assert(pushes == 2u && drains == 1u && invalidations == 5u);
    assert(gu->tl2_generation_seen == gens[j]);
    assert(allocations == old_allocations && frees == old_frees);
    assert(gu->arrays == ga && dn->arrays == da);
    check_zero(gu->bias, gu->N);
    check_zero(dn->bias, dn->N);
    /* The existing serial helper is an independent byte-copy reference. */
    hexkl_weight_u8i4_tail_from_arena(table->slots + rg, gb);
    hexkl_weight_u8i4_tail_from_arena(table->slots + rd, db);
    assert(memcmp(gu->w_scale, table->slots[rg].w_scale, 12u * gu->N) == 0);
    assert(memcmp(dn->w_scale, table->slots[rd].w_scale, 12u * dn->N) == 0);
  }

  /* Every current validation rejection precedes DMA and generation commit. */
  for (unsigned i = 0; i < 10u; ++i) {
    hexkl_weight_u8i4 g = *gu, d = *dn;
    switch (i) {
    case 0:
      g.in_use = 0;
      break;
    case 1:
      d.in_use = 0;
      break;
    case 2:
      g.borrowed = 0;
      break;
    case 3:
      d.borrowed = 0;
      break;
    case 4:
      g.tl2_compact = 0;
      break;
    case 5:
      d.tl2_compact = 0;
      break;
    case 6:
      g.N -= 128u;
      break;
    case 7:
      d.K -= 128u;
      break;
    case 8:
      g.K -= 128u;
      break;
    case 9:
      d.N -= 128u;
      break;
    }
    reset_trace();
    assert(hexkl_weight_tl2_refresh_pair(&g, &d) == AEE_EBADPARM);
    assert(pushes == 0u && drains == 0u && invalidations == 0u);
    assert(g.tl2_generation_seen == gu->tl2_generation_seen);
  }
  /* Gemma/non-stable handles retain the original no-op. */
  hexkl_weight_u8i4 gemma_g = *gu, gemma_d = *dn;
  gemma_g.tl2_generation = NULL;
  gemma_g.K = 2816u;
  gemma_g.N = 1408u;
  gemma_d.K = 704u;
  gemma_d.N = 2816u;
  reset_trace();
  assert(hexkl_weight_tl2_refresh_pair(&gemma_g, &gemma_d) == AEE_SUCCESS);
  assert(pushes == 0u && drains == 0u && invalidations == 0u);

  /* Even an explicitly rebound compact slot re-establishes zero bias. */
  memset(gu->bias, 0xa5, 4u * gu->N);
  memset(dn->bias, 0xa5, 4u * dn->N);
  assert(hexkl_weight_tl2_compact_rebind_arena(table, gh, K, 2u * I, arena) ==
         AEE_SUCCESS);
  assert(hexkl_weight_tl2_compact_rebind_arena(table, dh, I, N, arena + down) ==
         AEE_SUCCESS);
  check_zero(gu->bias, gu->N);
  check_zero(dn->bias, dn->N);
  assert(gu->tl2_generation_seen == *generation_word);
  /** Legacy callers may supply nonzero bias, but entering compact native
   * format reinitializes it before the stable refresh can see the handle. */
  float *legacy_bias = malloc(4u * gu->N);
  assert(legacy_bias);
  for (uint32_t i = 0; i < gu->N; ++i)
    legacy_bias[i] = 11.75f;
  const int8_t legacy_pal[4] = {-1, 0, 1, 7};
  uint32_t legacy_h;
  assert(hexkl_weight_u2i4_register_arena(
           table, 8u << 20, K, 2u * I, arena, legacy_pal, gu->w_scale,
           gu->colsum_w, legacy_bias, &legacy_h) == AEE_SUCCESS);
  hexkl_weight_u8i4 *legacy = table->slots + legacy_h;
  assert(memcmp(legacy->bias, legacy_bias, 4u * gu->N) == 0);
  assert(!legacy->tl2_compact && !legacy->tl2_generation);
  assert(hexkl_weight_tl2_compact_rebind_arena(table, legacy_h, K, 2u * I,
                                               arena) == AEE_SUCCESS);
  check_zero(legacy->bias, legacy->N);
  assert(legacy->tl2_compact);
  hexkl_weight_tl2_set_generation(legacy, generation_word);
  assert(hexkl_weight_u8i4_rebind_arena(table, legacy_h, K, 2u * I, arena, NULL,
                                        gu->w_scale,
                                        gu->colsum_w) == AEE_SUCCESS);
  assert(!legacy->tl2_compact && !legacy->tl2_generation);
  assert(legacy->tl2_generation_seen == 0u);
  hexkl_weight_u8i4_release(table, legacy_h);
  free(legacy_bias);

  /** Moving a compact pair to a different arena clears the old generation
   * pointer. The newly published generation then refreshes only its own
   * scales and sums, with no new metadata allocation. */
  uint8_t *arena2_allocation = malloc((size_t)cap + 4096u + 128u);
  assert(arena2_allocation);
  uint8_t *arena2 =
    (uint8_t *)(((uintptr_t)arena2_allocation + 4095u) & ~(uintptr_t)4095u);
  memcpy(arena2, arena, cap + 128u);
  assert(hexkl_weight_tl2_compact_rebind_arena(table, gh, K, 2u * I, arena2) ==
         AEE_SUCCESS);
  assert(hexkl_weight_tl2_compact_rebind_arena(table, dh, I, N,
                                               arena2 + down) == AEE_SUCCESS);
  assert(!gu->tl2_generation && gu->tl2_generation_seen == 0u);
  assert(!dn->tl2_generation && dn->tl2_generation_seen == 0u);
  assert(gu->arrays == ga && dn->arrays == da);
  gu_tail = arena2 + gb;
  dn_tail = arena2 + down + db;
  generation_word = (uint32_t *)(arena2 + tail_end);
  hexkl_weight_tl2_set_generation(gu, generation_word);
  fill_tail((uint8_t *)gu_tail, gu->N, 300u);
  fill_tail((uint8_t *)dn_tail, dn->N, 400u);
  seen_before = gu->tl2_generation_seen;
  *generation_word = 99u;
  const unsigned old_allocations = allocations, old_frees = frees;
  reset_trace();
  refreshing = 1u;
  assert(hexkl_weight_tl2_refresh_pair(gu, dn) == AEE_SUCCESS);
  refreshing = 0u;
  assert(pushes == 2u && drains == 1u && invalidations == 5u);
  assert(gu->tl2_generation_seen == 99u);
  assert(allocations == old_allocations && frees == old_frees);
  assert(memcmp(gu->w_scale, gu_tail, 8u * gu->N) == 0);
  assert(memcmp(dn->w_scale, dn_tail, 8u * dn->N) == 0);
  check_zero(gu->bias, gu->N);
  check_zero(dn->bias, dn->N);
  for (uint32_t i = 0; i < gb; ++i)
    assert(arena2[i] == 0x6d);
  for (uint32_t i = 0; i < db; ++i)
    assert(arena2[down + i] == 0x6d);
  for (uint32_t i = cap; i < cap + 128u; ++i)
    assert(arena2[i] == 0x6d);

  for (uint32_t i = 0; i < gb; ++i)
    assert(arena[i] == 0x6d);
  for (uint32_t i = 0; i < db; ++i)
    assert(arena[down + i] == 0x6d);
  for (uint32_t i = cap; i < cap + 128u; ++i)
    assert(arena[i] == 0x6d);
  hexkl_weight_u8i4_release(table, gh);
  hexkl_weight_u8i4_release(table, dh);
  hexkl_weight_u8i4_release(table, rg);
  hexkl_weight_u8i4_release(table, rd);
  assert(allocations == frees);
  free(table);
  free(arena_allocation);
  free(arena2_allocation);
  puts("TL2 REFRESH PAIR OK: 2 descriptors/1 drain, exact tails, generation "
       "publication, zero bias, no allocation, rejection/no-op guards");
  return 0;
}
