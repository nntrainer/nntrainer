// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   moe_opts_host_check.c
 * @date   22 Sep 2026
 * @brief  The M=1 GEMV default word (D192 since #113) and the MoE
 *         profile row's rest (#101, #102)
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * The two rows are #94 sitting 2's level-2 M==1 rows
 * (docs/htp_moe/BENCHMARK.md "C proof run"): A on the HMX loop printed
 * rest<=22.7; C on the GEMV path printed rest<=-5588.2 because its swiglu
 * slot (5601.2) is lane-time, not a stage. The expected values are hand
 * arithmetic from those printed numbers.
 */

#include "htp_moe_opts.h"
#include "htp_profile_wire.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

_Static_assert(HTP_N_STAGES == 7 && HTP_T_ACC_STRIDE == 6, "FC wire layout");
_Static_assert(HTP_FU_N_STAGES == 8 && HTP_FU_T_ACC_COPY == 5,
               "fused wire layout");
_Static_assert(HTP_GU_N_STAGES == 7 && HTP_GU_T_DRAIN == 5,
               "gate/up wire layout");
_Static_assert(HTP_MOE_N_STAGES == 31 && HTP_MOE_T_PATH == 29 &&
                 HTP_MOE_T_M1_FEED == 30 && HTP_MOE_T_DMA_DESC == 19,
               "MoE wire layout");

static int g_fail;

static void expect(int ok, const char *what) {
  if (!ok) {
    printf("FAIL: %s\n", what);
    g_fail = 1;
  }
}

int main(void) {
  {
    uint32_t scope = 0u;
    expect(htp_moe_opts_hadamard_scope(NULL, &scope) == 0 &&
             scope == 0x40000000u &&
             (htp_moe_opts_must_match(scope) & scope) == scope,
           "default Hadamard scope is 2-bit and must be echoed");
    expect(htp_moe_opts_hadamard_scope("2bit", &scope) == 0 &&
             scope == 0x40000000u,
           "explicit 2-bit scope");
    expect(htp_moe_opts_hadamard_scope("all", &scope) == 0 && scope == 0u,
           "all-weight scope");
    expect(htp_moe_opts_hadamard_scope("", &scope) == -1 && scope == 0u,
           "invalid scope is rejected without changing output");
    const char *names[] = {NULL,       "off",         "up",     "down",
                           "both",     "both_scalar", "up_hmx", "down_hmx",
                           "both_hmx", "both_paley"};
    const uint32_t words[] = {
      0u,          0u,          0x04000000u, 0x08000000u, 0x0c000000u,
      0x1c000000u, 0x24000000u, 0x28000000u, 0x2c000000u, 0x8c000000u};
    for (size_t i = 0; i < sizeof words / sizeof *words; ++i) {
      uint32_t bits = 0xdeadu;
      expect(htp_moe_opts_hadamard(names[i], &bits) == 0 && bits == words[i],
             "Hadamard option maps to DSP flags");
      expect((htp_moe_opts_must_match(bits) & bits) == bits,
             "Hadamard must be echoed even without GEMV");
    }
    const char *invalid[] = {"", "0", "1", "Both", "both ", "both-scalar"};
    for (size_t i = 0; i < sizeof invalid / sizeof *invalid; ++i) {
      uint32_t bits = 0xdeadu;
      expect(htp_moe_opts_hadamard(invalid[i], &bits) == -1 && bits == 0xdeadu,
             "invalid Hadamard option fails without changing flags");
    }
  }
  // Expected wire words are independent of the option encoder.
  const struct {
    unsigned enabled, lead_kb, rows1, feed;
    uint32_t word;
  } tunes[] = {
    {1, 192, 1, 1, 0x303e1u}, {0, 192, 1, 1, 0x303e0u},
    {1, 0, 0, 0, 0x000e1u},   {1, 1536, 1, 1, 0x318e1u},
    {1, 192, 0, 1, 0x203e1u}, {1, 192, 1, 0, 0x103e1u},
    {1, 0, 0, 1, 0x200e1u},   {1, 8128, 1, 1, 0x37fe1u},
  };
  for (size_t i = 0; i < sizeof tunes / sizeof *tunes; ++i)
    expect(htp_moe_opts_flags(tunes[i].enabled, tunes[i].lead_kb,
                              tunes[i].rows1, tunes[i].feed) == tunes[i].word,
           "GEMV options preserve the DSP wire word");
  expect(htp_moe_gemv_lead_units(0) == 0u, "zero lead");
  expect(htp_moe_gemv_lead_units(31) == 0u, "31 KB rounds down");
  expect(htp_moe_gemv_lead_units(32) == 1u, "32 KB rounds up");
  expect(htp_moe_gemv_lead_units(64) == 1u, "64 KB is one unit");
  expect(htp_moe_gemv_lead_units(100) == 2u, "100 KB rounds to two units");
  expect(htp_moe_gemv_lead_units(999999) == 127u, "lead field clamps");
  expect(!strcmp(htp_moe_opts_feed_name(0x303e1u), "vtcm") &&
           !strcmp(htp_moe_opts_feed_name(0x103e1u), "arena"),
         "feed names match their bits");
  expect(htp_moe_opts_dma_bypass(1) == 0x40000u &&
           htp_moe_opts_dma_bypass(0) == 0u,
         "DMA source bypass bit");
  expect(htp_moe_opts_must_match(0x703e1u) == ~0u,
         "GEMV requires the complete echoed word");
  expect(htp_moe_opts_must_match(0x303e0u) == HTP_MOE_FLAG_M1_GEMV,
         "HMX without bypass requires only the dispatch bit");
  expect(htp_moe_opts_must_match(0x703e0u) ==
           (HTP_MOE_FLAG_M1_GEMV | HTP_MOE_FLAG_DMA_BYPASS),
         "HMX still requires DMA bypass");
  expect(htp_moe_opts_must_match(0x803e0u) ==
           (HTP_MOE_FLAG_M1_GEMV | HTP_MOE_FLAG_GEGLU),
         "HMX still requires the activation bit");
  const uint32_t base =
    htp_moe_opts_flags(1, 192, 1, 1) | htp_moe_opts_dma_bypass(1);
  const uint32_t queue_words[] = {0x703e1u, 0x4703e1u, 0x8703e1u, 0xc703e1u};
  for (unsigned queues = 1; queues <= 4; ++queues) {
    uint32_t bits = 0xdeadu;
    expect(htp_moe_opts_dma_queues(queues, &bits) == 0 &&
             (base | bits) == queue_words[queues - 1] &&
             htp_moe_opts_dma_q(base | bits) == queues,
           "DMA queue count round trip and wire word");
  }
  const unsigned invalid_queues[] = {0, 5, 12, UINT32_MAX};
  for (size_t i = 0; i < sizeof invalid_queues / sizeof *invalid_queues; ++i) {
    uint32_t bits = 0xdeadu;
    expect(htp_moe_opts_dma_queues(invalid_queues[i], &bits) == -1 &&
             bits == 0xdeadu,
           "invalid queue count leaves the output unchanged");
  }
  if (!g_fail)
    puts("MOE GEMV OPTIONS: DSP wire words, lead rounding, feed and queue "
         "counts OK");

  /** C: dsp 1044.0, swiglu 5601.2, printed rest -5588.2 -> the other named
     stages (mm 974.7 among them) sum to 1044.0 + 5588.2 - 5601.2 = 1031.0;
     host 1792.2. */
  const double c_rest = htp_moe_row_rest_us(1044.0, 1031.0, 5601.2, 1);
  expect(c_rest >= 0.0 && c_rest <= 0.05 * 1792.2, "gemv rest in [0, 5%]");
  expect(fabs(c_rest - 13.0) < 0.1, "gemv rest = 13.0");
  /** A: dsp 1414.0, swiglu 0.0, named 14.8+146.5+41.9+13.0+259.9+113.1+8.9
     +2.5+0.9+0.1+6.3+783.4 = 1391.3 -> rest 22.7, as printed. */
  const double a_rest = htp_moe_row_rest_us(1414.0, 1391.3, 0.0, 0);
  expect(fabs(a_rest - 22.7) < 0.1, "hmx rest unchanged");
  /** An HMX row with a non-zero swiglu (the fused FFN buckets) still
     subtracts it. */
  expect(fabs(htp_moe_row_rest_us(100.0, 50.0, 30.0, 0) - 20.0) < 1e-9,
         "hmx row subtracts swiglu");
  if (g_fail)
    return 1;
  printf("MOE PROFILE ROW: gemv rest>=0 hmx unchanged\n");
  return 0;
}
