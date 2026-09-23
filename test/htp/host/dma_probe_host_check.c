// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   dma_probe_host_check.c
 * @date   21 Sep 2026
 * @brief  Host check of nntr_dma_probe_plan for the shapes the gtest drives
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * For every shape of docs/plans/77-first-handoff.md section 3.4 and worker
 * count 1..4 over a 256 MiB (and a 128 MiB fallback) source: descriptors
 * read disjoint source bytes, stay inside the range, the payload total is
 * n x row_size x nrows, every destination fits its worker's slice, the
 * descriptors are balanced across workers (max - min <= 1), and the
 * count fits NNTR_DMA_PROBE_MAX_DESC. Timing is the device's business.
 */
#include "nntr_dma_probe_plan.h"

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

static nntr_dma_probe_desc plan[NNTR_DMA_PROBE_MAX_DESC];

/** @brief One 4 KiB page per byte of the 256 MiB source. Every shape's
 *  row_size is a multiple of 4 KiB, so page granularity is exact. */
static uint8_t seen[(256u << 20) / 4096u];

static void check_shape(const char *name, uint32_t bytes, uint32_t row_size,
                        uint32_t nrows, uint32_t src_stride, uint32_t workers) {
  const uint32_t vtcm_usable = (8u << 20) - (64u << 10); /* v79 minus cfg */
  const uint32_t per_worker = (vtcm_usable / workers) & ~127u;
  const uint32_t n =
    nntr_dma_probe_plan(bytes, row_size, nrows, src_stride, workers, per_worker,
                        plan, NNTR_DMA_PROBE_MAX_DESC);
  const uint64_t payload = (uint64_t)row_size * nrows;
  uint32_t per_w[8] = {0};
  uint64_t total = 0;
  uint32_t k, r;

  CHECK(n > 0, "%s w=%u: empty plan", name, workers);
  CHECK(n < NNTR_DMA_PROBE_MAX_DESC, "%s w=%u: plan hit the cap (%u)", name,
        workers, n);
  memset(seen, 0, sizeof(seen));
  for (k = 0; k < n; ++k) {
    const nntr_dma_probe_desc *d = &plan[k];
    CHECK(d->row_size == row_size && d->nrows == nrows &&
            d->src_stride == src_stride,
          "%s w=%u: desc %u geometry changed", name, workers, k);
    CHECK(d->worker < workers, "%s w=%u: desc %u worker %u", name, workers, k,
          d->worker);
    CHECK((uint64_t)d->dst_off + payload <= per_worker,
          "%s w=%u: desc %u dst_off %u + %llu > slice %u", name, workers, k,
          d->dst_off, (unsigned long long)payload, per_worker);
    CHECK(d->dst_off % 128u == 0u, "%s w=%u: desc %u dst unaligned", name,
          workers, k);
    per_w[d->worker]++;
    total += payload;
    for (r = 0; r < nrows; ++r) {
      const uint64_t row0 = (uint64_t)d->src_off + (uint64_t)r * src_stride;
      const uint64_t row1 = row0 + row_size;
      CHECK(row1 <= bytes, "%s w=%u: desc %u row %u ends at %llu > %u", name,
            workers, k, r, (unsigned long long)row1, bytes);
      if (row1 > bytes) {
        break;
      }
      for (uint64_t pg = row0 / 4096u; pg < row1 / 4096u; ++pg) {
        CHECK(seen[pg] == 0, "%s w=%u: desc %u row %u overlaps page %llu", name,
              workers, k, r, (unsigned long long)pg);
        seen[pg] = 1;
      }
    }
  }
  CHECK(total == (uint64_t)n * payload, "%s w=%u: total %llu", name, workers,
        (unsigned long long)total);
  {
    uint32_t mn = per_w[0], mx = per_w[0];
    for (r = 1; r < workers; ++r) {
      if (per_w[r] < mn)
        mn = per_w[r];
      if (per_w[r] > mx)
        mx = per_w[r];
    }
    CHECK(mx - mn <= 1u, "%s w=%u: unbalanced %u..%u", name, workers, mn, mx);
  }
  /* Coverage: the payload fraction of the bands used is
     cols*row_size/src_stride; every whole band must have been used. */
  {
    const uint64_t band = (uint64_t)nrows * src_stride;
    const uint32_t cols = src_stride / row_size;
    const uint64_t bands = bytes / band;
    CHECK(n == bands * cols, "%s w=%u: %u descriptors, want %llu bands x %u",
          name, workers, n, (unsigned long long)bands, cols);
  }
  printf("  %-4s bytes=%u workers=%u n=%u payload=%llu MiB/pass=%llu\n", name,
         bytes, workers, n, (unsigned long long)payload,
         (unsigned long long)(total >> 20));
}

int main(void) {
  static const struct {
    const char *name;
    uint32_t row_size, nrows, src_stride;
  } shapes[] = {
    {"i", 4096u, 256u, 4096u},
    {"i1", 1048576u, 1u, 1048576u},
    {"ii", 16384u, 64u, 57344u},
    {"iii", 8192u, 64u, 57344u},
  };
  const uint32_t sizes[] = {256u << 20, 128u << 20};
  for (uint32_t si = 0; si < 2; ++si) {
    for (uint32_t sh = 0; sh < 4; ++sh) {
      for (uint32_t w = 1; w <= 4; ++w) {
        check_shape(shapes[sh].name, sizes[si], shapes[sh].row_size,
                    shapes[sh].nrows, shapes[sh].src_stride, w);
      }
    }
  }
  /* Rejections the skel relies on to answer EBADPARM rather than run. */
  CHECK(nntr_dma_probe_plan(1u << 20, 4096u, 512u, 4096u, 1u, 1u << 20, plan,
                            8u) == 0u,
        "2 MiB payload must be refused");
  CHECK(nntr_dma_probe_plan(1u << 20, 8192u, 64u, 4096u, 1u, 1u << 20, plan,
                            8u) == 0u,
        "row_size > src_stride must be refused");
  CHECK(nntr_dma_probe_plan(1u << 20, 4096u, 256u, 4096u, 1u, 512u << 10, plan,
                            8u) == 0u,
        "payload above the slice must be refused");
  CHECK(nntr_dma_probe_plan(1u << 19, 4096u, 256u, 4096u, 1u, 1u << 20, plan,
                            8u) == 0u,
        "no whole band must give an empty plan");
  if (failures) {
    printf("DMA PROBE PLAN: %d failures\n", failures);
    return 1;
  }
  printf("DMA PROBE PLAN OK\n");
  return 0;
}
