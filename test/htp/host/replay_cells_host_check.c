// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   replay_cells_host_check.c
 * @date   22 Sep 2026
 * @brief  Host check of the #100 replay cells and their tag expectation
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * docs/plans/100-dma-chunk-list.md section 3.2. The gtest builds its 16
 * cells with nntr_moe_dma_cell and holds the skel's tag sum (res[12])
 * against nntr_moe_dma_tag_sum; both are the header-only functions checked
 * here, at the gtest's shape and VTCM layout. Per cell: item, push, wait and
 * drain counts and bytes (hand arithmetic), every push inside the skel's
 * parse rules and under 6 MiB of VTCM (config_off - 1 MiB for load = 2 on
 * the 8 MiB v79 VTCM), the outstanding depth, the tag simulator against a
 * byte-by-byte copy into a host VTCM, and the negative control: the sum a
 * stale window would give (the previous call's for fresh = 1, the zeroed
 * window for fresh = 0) differs from the live one. Rates are the device's.
 */
#include "nntr_moe_dma_plan.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, ...)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      failures++;                                                              \
      printf("FAIL %s:%d: ", __FILE__, __LINE__);                              \
      printf(__VA_ARGS__);                                                     \
      printf("\n");                                                            \
    }                                                                          \
  } while (0)

#define VTCM_BOUND (6u << 20)
#define CALLS 20u

static uint8_t vtcm[VTCM_BOUND];
static uint8_t samples[VTCM_BOUND / 64u + 1u];

/** @brief The same sum the brute way: every byte copied, in list order. */
static uint32_t brute_tag_sum(const nntr_moe_dma_item *it, uint32_t n,
                              uint32_t calls, uint32_t fresh,
                              uint32_t n_regions, uint32_t region_bytes) {
  uint32_t lo, hi, n_experts = 0u, sum = 0u;
  nntr_moe_dma_window(it, n, &lo, &hi);
  memset(vtcm, 0, sizeof(vtcm));
  for (uint32_t k = 0; k < n; ++k) {
    if (it[k].op == NNTR_MOE_DMA_OP_PUSH &&
        it[k].kind != NNTR_MOE_DMA_KIND_ACT &&
        it[k].kind != NNTR_MOE_DMA_KIND_COPY && it[k].expert + 1u > n_experts) {
      n_experts = it[k].expert + 1u;
    }
  }
  for (uint32_t call = 0; call < calls; ++call) {
    for (uint32_t k = 0; k < n; ++k) {
      const nntr_moe_dma_item *p = &it[k];
      if (p->op != NNTR_MOE_DMA_OP_PUSH) {
        continue;
      }
      const int scratch =
        p->kind == NNTR_MOE_DMA_KIND_ACT || p->kind == NNTR_MOE_DMA_KIND_COPY;
      const uint32_t region =
        scratch
          ? n_regions
          : (fresh ? (call * n_experts + p->expert) % n_regions : p->expert);
      const uint32_t ds =
        (p->flags & NNTR_MOE_DMA_DST_STRIDED) ? p->src_stride : p->row_size;
      for (uint32_t r = 0; r < p->nrows; ++r) {
        for (uint32_t b = 0; b < p->row_size; ++b) {
          vtcm[p->dst_off + r * ds + b] = nntr_dma_pattern(
            region * region_bytes + p->src_off + r * p->src_stride + b);
        }
      }
    }
  }
  for (uint32_t i = lo; i + 32u < hi; i += 64u) {
    sum += vtcm[i + 32u];
  }
  return sum;
}

typedef struct {
  const char *name;
  uint32_t items, push, wait, drain, fresh, load, depth, hi;
  uint64_t bytes;
} expect_t;

int main(void) {
  const uint32_t K = 2048, I = 1792, N = 2048, E = 4;
  const uint32_t gu = nntr_moe_dma_gu_bytes(K, I),
                 dn = nntr_moe_dma_dn_bytes(I, N);
  const uint32_t region = nntr_moe_dma_region_bytes(K, I, N);
  const uint32_t v_act = gu + dn, v_copy = v_act + (K / 32u) * 2048u;
  static nntr_moe_dma_item traced[NNTR_MOE_DMA_PLAN_MAX],
    cell[NNTR_MOE_DMA_PLAN_MAX];
  const uint32_t n_traced = nntr_moe_dma_plan_m1(
    K, I, N, E, 32u, 0u, gu, v_act, v_copy, traced, NNTR_MOE_DMA_PLAN_MAX);

  CHECK(gu == 3670016u && dn == 1835008u && region == 5505024u,
        "shape: gu %u dn %u region %u", gu, dn, region);
  CHECK(n_traced == 76u, "traced list has %u items, want 76", n_traced);

  /** Hand arithmetic: 4 experts x (3.5 MiB gate_up + 1.75 MiB down) =
     22 020 096 weight bytes; + 4 x 128 KiB activation + 2 x 8 KiB copies =
     22 560 768 for the traced list. Probe iii's geometry: 4 x 7 columns of
     8 KiB x 64 = 28 x 512 KiB = 14 680 064. traced_gu is gate_up only (the
     same 14 680 064), traced_dn down only (7 340 032). hi is the top of the
     VTCM window (from 0; traced_dn from the down buffer). The traced
     list's packed gate_up chunks overlap in the first 556 KiB (#99), and
     its layout ends at v_copy + 8 KiB = 5 644 288; f2 and f3 fill gate_up
     + down exactly, 5 505 024; iii packs 8 x 512 KiB slots, 4 MiB;
     iii_strided tiles one gate_up, 3.5 MiB; f1 uses two 1 MiB slots. depth is
     the most descriptors outstanding (0 = unbounded by the list: the ring runs
     it as one chain). */
  static const expect_t want[NNTR_MOE_DMA_N_CELLS] = {
    {"traced", 76, 46, 30, 0, 0, 0, 0, 5644288u, 22560768u},
    {"traced_f", 76, 46, 30, 0, 1, 0, 0, 5644288u, 22560768u},
    {"traced_nowait", 46, 46, 0, 0, 0, 0, 0, 5644288u, 22560768u},
    {"traced_gu", 32, 32, 0, 0, 0, 0, 0, 569344u, 14680064u},
    {"traced_dn", 8, 8, 0, 0, 0, 0, 0, 3670016u + 16384u + 917504u, 7340032u},
    {"iii_chain", 28, 28, 0, 0, 0, 0, 0, 4194304u, 14680064u},
    {"iii_dmstart", 56, 28, 0, 28, 0, 0, 1, 4194304u, 14680064u},
    {"iii_link1", 56, 28, 28, 0, 0, 0, 1, 4194304u, 14680064u},
    {"iii_strided", 28, 28, 0, 0, 0, 0, 0, 3670016u, 14680064u},
    {"c_star", 56, 28, 0, 28, 1, 0, 1, 4194304u, 14680064u},
    {"f1", 48, 24, 24, 0, 0, 0, 2, 2097152u, 22020096u},
    {"f1_load", 48, 24, 24, 0, 0, 2, 2, 2097152u, 22020096u},
    {"f2", 16, 8, 8, 0, 0, 0, 2, 5505024u, 22020096u},
    {"f2_load", 16, 8, 8, 0, 0, 2, 2, 5505024u, 22020096u},
    {"f3", 72, 36, 36, 0, 0, 0, 2, 5505024u, 22020096u},
    {"f3_load", 72, 36, 36, 0, 0, 2, 2, 5505024u, 22020096u},
  };

  for (uint32_t id = 0; id < NNTR_MOE_DMA_N_CELLS; ++id) {
    const expect_t *w = &want[id];
    const char *name = "?";
    uint32_t fresh = 9u, load = 9u;
    const uint32_t n =
      nntr_moe_dma_cell(id, traced, n_traced, K, I, N, E, cell,
                        NNTR_MOE_DMA_PLAN_MAX, &name, &fresh, &load);
    uint32_t push = 0, wait = 0, drain = 0, depth = 0, out = 0, lo, hi;
    uint64_t bytes = 0;
    CHECK(n > 0u && n <= NNTR_MOE_DMA_PLAN_MAX, "%s: %u items", w->name, n);
    CHECK(strcmp(name, w->name) == 0, "cell %u is %s, want %s", id, name,
          w->name);
    CHECK(fresh == w->fresh && load == w->load, "%s: fresh %u load %u", w->name,
          fresh, load);
    for (uint32_t k = 0; k < n; ++k) {
      const nntr_moe_dma_item *p = &cell[k];
      if (p->op == NNTR_MOE_DMA_OP_PUSH) {
        const uint32_t ext =
          nntr_moe_dma_extent(p->flags, p->row_size, p->nrows, p->src_stride);
        const int weight =
          p->kind != NNTR_MOE_DMA_KIND_ACT && p->kind != NNTR_MOE_DMA_KIND_COPY;
        /** the skel's parse rules (nntr_hvx_dma_probe.c) and the struct's
           bit widths (hexkl_dma_ring.h) */
        CHECK(p->row_size > 0u && p->row_size < (1u << 24) && p->nrows > 0u &&
                p->nrows < (1u << 16) && p->src_stride >= p->row_size &&
                p->src_stride < (1u << 24) && p->flags <= 2u &&
                (uint64_t)p->row_size * p->nrows <=
                  NNTR_MOE_DMA_REPLAY_MAX_DESC_BYTES &&
                (uint64_t)p->src_off +
                    (uint64_t)(p->nrows - 1u) * p->src_stride + p->row_size <=
                  region &&
                (!weight || p->expert < 8u) &&
                (uint64_t)p->dst_off + ext <= VTCM_BOUND,
              "%s: push %u out of the replay's range", w->name, k);
        bytes += (uint64_t)p->row_size * p->nrows;
        ++push;
        if (++out > depth) {
          depth = out;
        }
      } else if (p->op == NNTR_MOE_DMA_OP_WAIT) {
        CHECK(p->src_off < k && cell[p->src_off].op == NNTR_MOE_DMA_OP_PUSH,
              "%s: wait %u on a non-push", w->name, k);
        /* in order: waiting on a push retires it and every earlier one */
        uint32_t later = 0;
        for (uint32_t j = p->src_off + 1u; j < k; ++j) {
          later += cell[j].op == NNTR_MOE_DMA_OP_PUSH;
        }
        out = later < out ? later : out;
        ++wait;
      } else {
        CHECK(p->op == NNTR_MOE_DMA_OP_DRAIN && push > 0u, "%s: item %u op %u",
              w->name, k, p->op);
        out = 0;
        ++drain;
      }
    }
    nntr_moe_dma_window(cell, n, &lo, &hi);
    CHECK(n == w->items && push == w->push && wait == w->wait &&
            drain == w->drain && bytes == w->bytes,
          "%s: items %u push %u wait %u drain %u bytes %llu", w->name, n, push,
          wait, drain, (unsigned long long)bytes);
    CHECK(hi == w->hi && lo == (id == 4u ? gu : 0u), "%s: window [%u, %u)",
          w->name, lo, hi);
    CHECK(w->depth == 0u || depth == w->depth, "%s: depth %u, want %u", w->name,
          depth, w->depth);

    /** The simulator is the byte copy, sampled. Two calls exercise both the
       fresh rotation and the overwrite of the first call's window. */
    const uint32_t sim2 =
      nntr_moe_dma_tag_sum(cell, n, 2u, fresh, 32u, region, samples);
    const uint32_t ref2 = brute_tag_sum(cell, n, 2u, fresh, 32u, region);
    CHECK(sim2 == ref2, "%s: simulated tag sum %u, byte copy %u", w->name, sim2,
          ref2);
    /* 32 regions (256 MiB chunks) and 23 (the 128 MiB fallback) */
    for (uint32_t n_regions = 32u; n_regions >= 23u; n_regions -= 9u) {
      const uint32_t live =
        nntr_moe_dma_tag_sum(cell, n, CALLS, fresh, n_regions, region, samples);
      const uint32_t stale =
        fresh ? nntr_moe_dma_tag_sum(cell, n, CALLS - 1u, fresh, n_regions,
                                     region, samples)
              : 0u;
      CHECK(live != stale && live != 0u,
            "%s regions=%u: live %u = stale %u, a dropped call passes", w->name,
            n_regions, live, stale);
      if (n_regions == 32u) {
        printf("cell %-13s items=%2u push=%2u wait=%2u drain=%2u bytes=%8llu "
               "vtcm=[%u,%u) depth=%u fresh=%u load=%u tag_sum=%u stale=%u\n",
               w->name, n, push, wait, drain, (unsigned long long)bytes, lo, hi,
               depth, fresh, load, live, stale);
      }
    }
  }
  /* An id past the table builds nothing. */
  {
    const char *name = "";
    uint32_t fresh, load;
    CHECK(nntr_moe_dma_cell(NNTR_MOE_DMA_N_CELLS, traced, n_traced, K, I, N, E,
                            cell, NNTR_MOE_DMA_PLAN_MAX, &name, &fresh,
                            &load) == 0u,
          "id %u built a list", NNTR_MOE_DMA_N_CELLS);
  }

  if (failures) {
    printf("REPLAY CELLS PLAN FAILED (%d)\n", failures);
    return 1;
  }
  printf("REPLAY CELLS PLAN OK (%u cells)\n", NNTR_MOE_DMA_N_CELLS);
  return 0;
}
