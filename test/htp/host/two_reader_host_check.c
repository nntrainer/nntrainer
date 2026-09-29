// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   two_reader_host_check.c
 * @date   29 Sep 2026
 * @brief  The #90 ring reader's cell on the skel's dma_replay, and the
 *         probe's bounds and projection arithmetic
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * test/htp/nntr_hvx_dma_probe.c compiled as-is against replay_stub/, as
 * dma_replay_host_check does, driven with nntr_two_reader_cell (f2,
 * fresh = 1). Checks that the bytes the stand-in landed are res[2] x
 * res[1], that one rotation reads n_regions x region_bytes of distinct
 * source pages (256 MiB arena: 168 MiB; 128 MiB: 23 regions), that res[12]
 * equals the tag simulator for 1, 20 and 500 calls and that call 500's
 * tag differs from call 499's (a transfer landing a call late is caught).
 * Then the pure functions: #77's 3265.6 and 85.4 GB/s are refused, 67.9
 * and 37.3 accepted, the ring verdict's +-20 % edge, the net per token.
 * The device's DMA timing, caches and contention are not modelled.
 */
#include "hexkl_dma_standin.h"
#include "nntr_hvx.h"
#include "nntr_hvx_session.h"
#include "nntr_moe_dma_plan.h"

#include <AEEStdErr.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint8_t samples[(8u << 20) / 64u + 1u];
static uint32_t sched[NNTR_MOE_DMA_PLAN_MAX * 8u];

/** @brief The gtest's 8 words per item, t_rel 0. */
static int serialize(const nntr_moe_dma_item *it, uint32_t n) {
  for (uint32_t k = 0; k < n; ++k) {
    const uint32_t w[8] = {nntr_moe_dma_word0(&it[k]),
                           it[k].expert,
                           it[k].src_off,
                           it[k].dst_off,
                           it[k].row_size,
                           it[k].nrows,
                           it[k].src_stride,
                           0u};
    memcpy(sched + 8u * k, w, sizeof(w));
  }
  return (int)(8u * n);
}

/** @brief Distinct 4 KiB pages the meter has marked. */
static uint64_t pages_read(void) {
  uint64_t n = 0;
  for (size_t i = 0; i < replay_stub_meter.n_pages; ++i) {
    n += replay_stub_meter.pages[i];
  }
  return n << 12;
}

int main(void) {
  const uint32_t region = nntr_moe_dma_region_bytes(2048u, 1792u, 2048u);
  static nntr_hvx_session s;
  static nntr_moe_dma_item items[NNTR_MOE_DMA_PLAN_MAX];
  nntr_two_reader_spec spec;
  int fail = 0;
  uint64_t footprint_256 = 0;

  const uint32_t n = nntr_two_reader_cell(items, NNTR_MOE_DMA_PLAN_MAX, &spec);
  const int words = serialize(items, n);
  if (n == 0u || strcmp(spec.name, "f2") != 0 || spec.fresh != 1u ||
      spec.calls_per_chunk != 500u || spec.min_us != 1200000u) {
    printf("FAIL: nntr_two_reader_cell n=%u name=%s\n", n, spec.name);
    return 1;
  }
  s.vtcm_size = 8u << 20;
  s.vtcm_base = (uint8_t *)aligned_alloc(128, s.vtcm_size);
  s.config_off = s.vtcm_size - (64u << 10);
  s.quant_pool = hvx_worker_pool_create(3);
  if (!s.vtcm_base || !s.quant_pool) {
    printf("FAIL: allocation\n");
    return 1;
  }
  const remote_handle64 h = (remote_handle64)(uintptr_t)&s;

  /* The gtest's two chunk sizes (plan section 5's fallback). */
  for (uint32_t mib = 256u; mib >= 128u; mib /= 2u) {
    const uint32_t arena_bytes = mib << 20;
    const uint32_t want_regions = mib == 256u ? 32u : 23u;
    s.arenas[0].va = (uint8_t *)aligned_alloc(4096, arena_bytes);
    s.arenas[0].bytes = arena_bytes;
    uint8_t *pages = (uint8_t *)calloc(arena_bytes >> 12, 1);
    if (!s.arenas[0].va || !pages) {
      printf("FAIL: allocation\n");
      return 1;
    }
    for (uint32_t i = 0; i < arena_bytes; ++i) {
      s.arenas[0].va[i] = nntr_dma_pattern(i);
    }
    replay_stub_meter.base = s.arenas[0].va;
    replay_stub_meter.pages = pages;
    replay_stub_meter.n_pages = arena_bytes >> 12;

    /* 1 + 2: bytes landed and the distinct source pages of 24 calls (a
       full rotation is n_regions / 4 calls). */
    uint32_t res[13] = {0};
    const uint32_t calls = 24u;
    replay_stub_meter.bytes_landed = 0;
    int err = nntr_hvx_dma_replay(h, 0, region, sched, words, 1, 0, 0,
                                  spec.fresh, 0, calls, res, 13);
    const uint64_t want_fp = nntr_two_reader_footprint_bytes(res[7], region);
    const uint64_t fp = pages_read();
    if (err != AEE_SUCCESS || res[1] != calls || res[2] != 22020096u ||
        replay_stub_meter.bytes_landed != (uint64_t)res[2] * res[1] ||
        res[7] != want_regions || fp != want_fp) {
      printf("FAIL %u MiB: err=%d bytes=%llu res=%u x %u regions=%u "
             "footprint=%llu want=%llu\n",
             mib, err, (unsigned long long)replay_stub_meter.bytes_landed,
             res[2], res[1], res[7], (unsigned long long)fp,
             (unsigned long long)want_fp);
      fail = 1;
    }
    if (mib == 256u) {
      footprint_256 = fp;
    }

    /* 3: the tag for 1, 20 and 500 calls, and 500 != 499. */
    uint32_t tag_prev = 0;
    static const uint32_t kCalls[] = {1u, 20u, 499u, 500u};
    for (uint32_t i = 0; i < 4u; ++i) {
      const uint32_t c = kCalls[i];
      const uint32_t want =
        nntr_moe_dma_tag_sum(items, n, c, spec.fresh, res[7], region, samples);
      if (c == 499u) {
        tag_prev = want;
        continue;
      }
      memset(s.vtcm_base, 0x5a, s.vtcm_size);
      err = nntr_hvx_dma_replay(h, 0, region, sched, words, 1, 0, 0, spec.fresh,
                                0, c, res, 13);
      if (err != AEE_SUCCESS || res[12] != want ||
          (c == 500u && want == tag_prev)) {
        printf("FAIL %u MiB calls=%u: err=%d tag=%u want=%u tag499=%u\n", mib,
               c, err, res[12], want, tag_prev);
        fail = 1;
      }
    }
    replay_stub_meter.pages = NULL;
    free(pages);
    free(s.arenas[0].va);
  }
  hvx_worker_pool_destroy(s.quant_pool);
  free(s.vtcm_base);

  /* 4: the bounds (rule 12). #77's DSP pair, and one just past the peak. */
  const double c = NNTR_DDR_CEILING_GBS, ref = 37.3;
  const uint64_t b77 = 85362475008ull, us77 = 26140u;
  if (nntr_ddr_rate_valid(b77, us77, c) ||
      nntr_ddr_rate_valid(85400000000ull, 1000000u, c) ||
      !nntr_ddr_rate_valid(67900000000ull, 1000000u, c) ||
      !nntr_ddr_rate_valid(37300000000ull, 1000000u, c) ||
      nntr_ddr_rate_valid(1u, 0u, c) ||
      nntr_two_reader_verdict(ref * 1.21, ref, 1) ||
      !nntr_two_reader_verdict(ref * 1.19, ref, 1) ||
      !nntr_two_reader_verdict(ref * 0.81, ref, 1) ||
      nntr_two_reader_verdict(ref * 0.79, ref, 1) ||
      nntr_two_reader_verdict(ref, ref, 0) ||
      nntr_two_reader_verdict(86.0, 80.0, 1)) {
    printf("FAIL: bounds\n");
    fail = 1;
  }
  /* 5: the projection. */
  const double net = nntr_prefetch_net_us(250, 150, 640, 600);
  const double per_token = 22.0 * net / 1000.0;
  if (net != 60.0 || fabs(per_token - 1.32) > 1e-9) {
    printf("FAIL: net_us=%.3f per_token=%.3f\n", net, per_token);
    fail = 1;
  }
  if (fail) {
    return 1;
  }
  printf("TWO READER CELL SOUND (bytes=%llu footprint=%llu MiB)\n",
         24ull * 22020096ull, (unsigned long long)(footprint_256 >> 20));
  printf("TWO READER BOUNDS OK (%.1f INVALID, 85.4 INVALID, 67.9 valid, "
         "net=%.2f ms/token)\n",
         (double)b77 / us77 / 1e3, per_token);
  return 0;
}
