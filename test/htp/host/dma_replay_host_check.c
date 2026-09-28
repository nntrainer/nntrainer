// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   dma_replay_host_check.c
 * @date   22 Sep 2026
 * @brief  Runs the skel's dma_replay entry on the host for the #100 cells
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * test/htp/nntr_hvx_dma_probe.c compiled as-is against replay_stub/, where
 * a descriptor lands whole the moment it is started or linked. Feeds each
 * of the 16 cells (nntr_moe_dma_cell) through nntr_hvx_dma_replay exactly
 * as the gtest serializes them and holds res[12] against
 * nntr_moe_dma_tag_sum, so the skel's parse, destination modes, drain,
 * window and tag sum are checked against the simulator the gtest trusts.
 * Also: the old traced cells still read the res[6] #99 reported
 * (1 467 840), and bad flags or a drain with two workers are refused. The
 * device's timing and its interleaving of transfers are not modelled.
 */
#include "nntr_hvx.h"
#include "nntr_hvx_session.h"
#include "nntr_moe_dma_plan.h"

#include <AEEStdErr.h>
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

int main(void) {
  const uint32_t K = 2048, I = 1792, N = 2048, E = 4, calls = 20;
  const uint32_t gu = nntr_moe_dma_gu_bytes(K, I),
                 dn = nntr_moe_dma_dn_bytes(I, N),
                 region = nntr_moe_dma_region_bytes(K, I, N);
  const uint32_t v_act = gu + dn, v_copy = v_act + (K / 32u) * 2048u;
  const uint32_t arena_bytes = 256u << 20;
  static nntr_hvx_session s;
  static nntr_moe_dma_item traced[NNTR_MOE_DMA_PLAN_MAX],
    items[NNTR_MOE_DMA_PLAN_MAX];
  int fail = 0;

  s.vtcm_size = 8u << 20;
  s.vtcm_base = (uint8_t *)aligned_alloc(128, s.vtcm_size);
  s.config_off = s.vtcm_size - (64u << 10);
  s.quant_pool = hvx_worker_pool_create(3);
  s.arenas[0].va = (uint8_t *)aligned_alloc(4096, arena_bytes);
  s.arenas[0].bytes = arena_bytes;
  if (!s.vtcm_base || !s.quant_pool || !s.arenas[0].va) {
    printf("FAIL: allocation\n");
    return 1;
  }
  for (uint32_t i = 0; i < arena_bytes; ++i) {
    s.arenas[0].va[i] = nntr_dma_pattern(i);
  }
  const remote_handle64 h = (remote_handle64)(uintptr_t)&s;
  const uint32_t nt = nntr_moe_dma_plan_m1(
    K, I, N, E, 32u, 0u, gu, v_act, v_copy, traced, NNTR_MOE_DMA_PLAN_MAX);

  for (uint32_t id = 0; id < NNTR_MOE_DMA_N_CELLS; ++id) {
    const char *name = "?";
    uint32_t fresh, load, res[13] = {0};
    const uint32_t n =
      nntr_moe_dma_cell(id, traced, nt, K, I, N, E, items,
                        NNTR_MOE_DMA_PLAN_MAX, &name, &fresh, &load);
    /* what a previous cell or the HMX left behind */
    memset(s.vtcm_base, 0x5a, s.vtcm_size);
    const int err =
      nntr_hvx_dma_replay(h, 0, region, sched, serialize(items, n), 1, load, 0,
                          fresh, 0, calls, res, 13);
    const uint32_t want =
      nntr_moe_dma_tag_sum(items, n, calls, fresh, res[7], region, samples);
    if (err != AEE_SUCCESS || res[12] != want || res[7] != 32u) {
      printf("FAIL %s: err=%d regions=%u tag=%u want=%u\n", name, err, res[7],
             res[12], want);
      fail = 1;
    }
  }
  /* The old path: default dst, one and two workers, the #99 number. */
  for (uint32_t w = 1; w <= 2; ++w) {
    uint32_t res[12] = {0};
    const int err =
      nntr_hvx_dma_replay(h, 0, region, sched, serialize(traced, nt), w, 0, 0,
                          0, 0, calls, res, 12);
    if (err != AEE_SUCCESS || res[6] != 1467840u) {
      printf("FAIL old cell workers=%u: err=%d res[6]=%u\n", w, err, res[6]);
      fail = 1;
    }
  }
  /* Refused: both destination modes at once; a drain on two workers. */
  {
    const uint32_t both[8] = {3u << 16, 0, 0, 0, 4096, 1, 4096, 0};
    const uint32_t drain[16] = {
      NNTR_MOE_DMA_KIND_GATE,     0, 0, 0, 4096, 1, 4096, 0,
      NNTR_MOE_DMA_OP_DRAIN << 8, 0, 0, 0, 0,    0, 0,    0};
    uint32_t res[13];
    if (nntr_hvx_dma_replay(h, 0, region, both, 8, 1, 0, 0, 0, 0, 1, res, 13) !=
          AEE_EBADPARM ||
        nntr_hvx_dma_replay(h, 0, region, drain, 16, 2, 0, 0, 0, 0, 1, res,
                            13) != AEE_EBADPARM ||
        nntr_hvx_dma_replay(h, 0, region, drain, 16, 1, 0, 0, 0, 0, 1, res,
                            13) != AEE_SUCCESS) {
      printf("FAIL: flags / drain parse\n");
      fail = 1;
    }
  }
  hvx_worker_pool_destroy(s.quant_pool);
  free(s.arenas[0].va);
  free(s.vtcm_base);
  if (fail) {
    return 1;
  }
  printf("SKEL REPLAY MATCHES TAG SIMULATOR (%u cells)\n",
         NNTR_MOE_DMA_N_CELLS);
  return 0;
}
