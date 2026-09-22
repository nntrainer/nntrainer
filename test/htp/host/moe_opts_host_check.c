// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   moe_opts_host_check.c
 * @date   22 Sep 2026
 * @brief  The M=1 GEMV default and the MoE profile row's rest (#101, #102)
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

static int g_fail;

static void expect(int ok, const char *what) {
  if (!ok) {
    printf("FAIL: %s\n", what);
    g_fail = 1;
  }
}

int main(void) {
  expect(htp_moe_opts_flags(NULL) == HTP_MOE_FLAG_M1_GEMV, "unset is on");
  expect(htp_moe_opts_flags("0") == 0u, "0 is off");
  expect(htp_moe_opts_flags("1") == HTP_MOE_FLAG_M1_GEMV, "1 is on");
  if (!g_fail)
    printf("MOE M1 GEMV OPTS: unset=on 0=off 1=on\n");

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
