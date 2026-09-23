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

#include <math.h>
#include <stdio.h>
#include <string.h>

static int g_fail;

static void expect(int ok, const char *what) {
  if (!ok) {
    printf("FAIL: %s\n", what);
    g_fail = 1;
  }
}

int main(void) {
  /* The D192 word, #113's landed default: bit 0 on, both tune bits, lead
     3 x 64 KB in [15:8], rows1 in bit 16 = 0x103c1. */
  const uint32_t d192 = HTP_MOE_FLAG_M1_GEMV | HTP_MOE_FLAG_GEMV_LEAD_SET |
                        (3u << HTP_MOE_GEMV_LEAD_SHIFT) |
                        HTP_MOE_FLAG_GEMV_ROWS1_SET | HTP_MOE_FLAG_GEMV_ROWS1;
  expect(d192 == 0x103c1u, "D192 is 0x103c1");
  expect(htp_moe_opts_flags(NULL, NULL, NULL, NULL) == d192,
         "unset is on, D192");
  expect(htp_moe_opts_flags("0", NULL, NULL, NULL) ==
           (d192 & ~HTP_MOE_FLAG_M1_GEMV),
         "0 is off");
  expect(htp_moe_opts_flags("1", NULL, NULL, NULL) == d192, "1 is on");
  if (!g_fail)
    printf("MOE M1 GEMV OPTS: unset=on(0x103c1) 0=off 1=on\n");

  /* #113's (loop, lead) fields. Each knob is overridden only by its own
     variable, so naming one leaves the other at the default instead of
     resetting it to zero; both tune bits are always sent so the echo
     names the cell. */
  expect(htp_moe_opts_flags(NULL, "0", "0", NULL) ==
           (HTP_MOE_FLAG_M1_GEMV | HTP_MOE_FLAG_GEMV_LEAD_SET |
            HTP_MOE_FLAG_GEMV_ROWS1_SET),
         "LEAD_KB=0 ROWS1=0 is the old four-row, no-lead cell 0xc1");
  expect(htp_moe_opts_flags(NULL, "0", "0", NULL) == 0xc1u, "A0 word is 0xc1");
  expect(htp_moe_opts_flags(NULL, "192", NULL, NULL) == d192,
         "192 KB is 3 units of 64 KB, and the loop stays at the default");
  expect(htp_moe_opts_flags(NULL, "1536", "1", NULL) ==
           (HTP_MOE_FLAG_M1_GEMV | HTP_MOE_FLAG_GEMV_LEAD_SET |
            (24u << HTP_MOE_GEMV_LEAD_SHIFT) | HTP_MOE_FLAG_GEMV_ROWS1_SET |
            HTP_MOE_FLAG_GEMV_ROWS1),
         "1536 KB + rows1");
  expect(htp_moe_opts_flags(NULL, NULL, "1", NULL) == d192,
         "rows1 alone does not force the lead to 0");
  expect(htp_moe_opts_flags(NULL, NULL, "0", NULL) ==
           (d192 & ~HTP_MOE_FLAG_GEMV_ROWS1),
         "rows1=0 is an explicit four-row loop at the default lead");
  expect(htp_moe_opts_flags("0", NULL, "1", NULL) ==
           (d192 & ~HTP_MOE_FLAG_M1_GEMV),
         "the opt-out keeps bit 0 clear and still carries the pair");
  /* Rounding to the nearest unit, the >= 0 floor and the 127-unit clamp
     the l2fetch width field imposes. */
  expect(htp_moe_gemv_lead_units(NULL) == 0u, "unset lead is 0");
  expect(htp_moe_gemv_lead_units("0") == 0u, "0 KB is 0 units");
  expect(htp_moe_gemv_lead_units("-64") == 0u, "a negative lead is 0");
  expect(htp_moe_gemv_lead_units("31") == 0u, "31 KB rounds down to 0");
  expect(htp_moe_gemv_lead_units("32") == 1u, "32 KB rounds up to 1");
  expect(htp_moe_gemv_lead_units("64") == 1u, "64 KB is 1 unit");
  expect(htp_moe_gemv_lead_units("100") == 2u, "100 KB rounds to 2 units");
  expect(htp_moe_gemv_lead_units("999999") == HTP_MOE_GEMV_LEAD_MAX_UNITS,
         "clamped to 127 units");
  expect((htp_moe_gemv_lead_units("999999") & ~HTP_MOE_GEMV_LEAD_BITS) == 0u,
         "the clamped field still fits bits [15:8]");
  /* #117's feed knob. Sent only when its variable is set, so an unset run
     keeps A's word (0x103c1, feed=default = the skel's build value);
     FEED=1 on D192 is 0x303e1 (feed=vtcm), FEED=0 is 0x103e1 (feed=arena).
     It touches neither of the other two knobs. */
  expect(htp_moe_opts_flags(NULL, NULL, NULL, NULL) == 0x103c1u,
         "unset feed leaves A's word 0x103c1");
  expect(htp_moe_opts_flags(NULL, NULL, NULL, "1") == 0x303e1u,
         "FEED=1 on D192 is 0x303e1");
  expect(htp_moe_opts_flags(NULL, NULL, NULL, "0") == 0x103e1u,
         "FEED=0 on D192 is 0x103e1");
  expect(htp_moe_opts_flags(NULL, "0", "0", "1") == 0x200e1u,
         "FEED=1 on the four-row, no-lead cell is 0x200e1");
  expect(htp_moe_opts_flags("0", NULL, NULL, "1") == 0x303e0u,
         "the opt-out keeps bit 0 clear and still carries the feed");
  expect(!strcmp(htp_moe_opts_feed_name(0x103c1u), "default") &&
           !strcmp(htp_moe_opts_feed_name(0x303e1u), "vtcm") &&
           !strcmp(htp_moe_opts_feed_name(0x103e1u), "arena"),
         "feed names: default / vtcm / arena");
  if (!g_fail)
    printf("MOE GEMV TUNE OPTS: per-knob tune bits, 64 KB units, "
           "round+clamp 127, feed unset/0/1\n");

  /* C: dsp 1044.0, swiglu 5601.2, printed rest -5588.2 -> the other named
     stages (mm 974.7 among them) sum to 1044.0 + 5588.2 - 5601.2 = 1031.0;
     host 1792.2. */
  const double c_rest = htp_moe_row_rest_us(1044.0, 1031.0, 5601.2, 1);
  expect(c_rest >= 0.0 && c_rest <= 0.05 * 1792.2, "gemv rest in [0, 5%]");
  expect(fabs(c_rest - 13.0) < 0.1, "gemv rest = 13.0");
  /* A: dsp 1414.0, swiglu 0.0, named 14.8+146.5+41.9+13.0+259.9+113.1+8.9
     +2.5+0.9+0.1+6.3+783.4 = 1391.3 -> rest 22.7, as printed. */
  const double a_rest = htp_moe_row_rest_us(1414.0, 1391.3, 0.0, 0);
  expect(fabs(a_rest - 22.7) < 0.1, "hmx rest unchanged");
  /* An HMX row with a non-zero swiglu (the fused FFN buckets) still
     subtracts it. */
  expect(fabs(htp_moe_row_rest_us(100.0, 50.0, 30.0, 0) - 20.0) < 1e-9,
         "hmx row subtracts swiglu");
  if (g_fail)
    return 1;
  printf("MOE PROFILE ROW: gemv rest>=0 hmx unchanged\n");
  return 0;
}
