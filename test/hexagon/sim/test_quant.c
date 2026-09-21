// SPDX-License-Identifier: Apache-2.0
/**
 * @file	test_quant.c
 * @date	18 August 2026
 * @brief	Hexagon-sim test for per-token dynamic quantization
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "hvx-quant.h"
#include "ref_ops.h"
#include "sim_test_util.h"

#define KQ 3072

static __fp16 x_row[KQ] __attribute__((aligned(128)));
static int8_t q_got[KQ] __attribute__((aligned(128)));
static int8_t q_ref[KQ] __attribute__((aligned(128)));
static int16_t q16_got[KQ] __attribute__((aligned(128)));
static int16_t q16_ref[KQ] __attribute__((aligned(128)));

/** Quantize x_row with the kernel and with the scalar reference (esz==1: int8
 * into q_got/q_ref; esz==2: int16 into q16_got/q16_ref). scale must always be
 * equal; q must be byte-identical unless pm1_out is given, in which case
 * +/-1 differences are counted there instead (the caller bounds the rate).
 * Prints the first mismatch on failure. */
static int check_row(const char *tag, uint32_t k, uint32_t esz,
                     uint32_t *pm1_out) {
  float scale_got = esz == 1 ? htp_quant_row_fp16(x_row, q_got, k)
                             : htp_quant_row_fp16_i16(x_row, q16_got, k);
  float scale_ref = esz == 1 ? ref_quant_row(x_row, q_ref, k)
                             : ref_quant_row_i16(x_row, q16_ref, k);

  if (scale_got != scale_ref) {
    printf("SIM_TEST quant FAIL %s scale got=%f ref=%f\n", tag,
           (double)scale_got, (double)scale_ref);
    return 1;
  }
  uint32_t n = 0, first = 0;
  int pm1 = 1;
  for (uint32_t i = 0; i < k; ++i) {
    int got = esz == 1 ? q_got[i] : q16_got[i];
    int ref = esz == 1 ? q_ref[i] : q16_ref[i];
    if (got != ref) {
      if (n == 0)
        first = i;
      ++n;
      if (got - ref != 1 && got - ref != -1)
        pm1 = 0;
    }
  }
  if (n == 0)
    return 0;
  if (pm1_out && pm1) {
    *pm1_out += n;
    return 0;
  }
  int got = esz == 1 ? q_got[first] : q16_got[first];
  int ref = esz == 1 ? q_ref[first] : q16_ref[first];
  printf("SIM_TEST quant FAIL %s n=%u/%u all_pm1=%d i=%u got=%d ref=%d "
         "x=%.9g x/scale=%.9g\n",
         tag, (unsigned)n, (unsigned)k, pm1, (unsigned)first, got, ref,
         (double)x_row[first], (double)x_row[first] / (double)scale_ref);
  return 1;
}

static int test_quant_row(void) {
  char tag[32];
  int fails = 0;

  /* (a) 16 random rows, k=1024, amplitude cycling over the exponent range. */
  static const float amp[4] = {8.f, 0.01f, 1000.f, 1e-4f};
  for (int r = 0; r < 16; ++r) {
    for (uint32_t i = 0; i < 1024; ++i)
      x_row[i] = (__fp16)(frand() * amp[r & 3]);
    snprintf(tag, sizeof(tag), "rand%d", r);
    fails += check_row(tag, 1024, 1u, NULL);
  }

  /* (b) all-zero row: inv = 0 branch. */
  memset(x_row, 0, sizeof(x_row));
  fails += check_row("zero", 1024, 1u, NULL);

  /** (c) tie row: x[0] = 127 makes inv exactly 1.0, the rest are +/-(n + 0.5)
   * so every element lands on a rounding tie (ties-to-even like lrintf). */
  x_row[0] = (__fp16)127.f;
  for (uint32_t i = 1; i < 1024; ++i)
    x_row[i] = (__fp16)((i & 1u ? -1.f : 1.f) * ((float)(i % 127u) + 0.5f));
  fails += check_row("tie", 1024, 1u, NULL);
  /** tie2: the sign changes every two elements, independently of the parity
   * of n, so each of the four classes +(even).5, +(odd).5, -(even).5,
   * -(odd).5 holds a quarter of the row. The tie row above also reaches all
   * four (its sign and the parity of n drift out of step every 127
   * elements) but unevenly; tie2 makes the coverage explicit. */
  for (uint32_t i = 1; i < 1024; ++i)
    x_row[i] =
      (__fp16)(((i >> 1) & 1u ? -1.f : 1.f) * ((float)(i % 127u) + 0.5f));
  fails += check_row("tie2", 1024, 1u, NULL);

  /* (d) widest and narrowest row widths. */
  for (uint32_t i = 0; i < KQ; ++i)
    x_row[i] = (__fp16)(frand() * 8.f);
  fails += check_row("k3072", KQ, 1u, NULL);
  fails += check_row("k128", 128, 1u, NULL);

  /* (e) negative-only row, and a row whose absmax is the last element. */
  for (uint32_t i = 0; i < 1024; ++i)
    x_row[i] = (__fp16)(-1.f - fabsf(frand()) * 7.f);
  fails += check_row("neg", 1024, 1u, NULL);
  for (uint32_t i = 0; i < 1024; ++i)
    x_row[i] = (__fp16)(frand() * 0.5f);
  x_row[1023] = (__fp16)9.5f;
  fails += check_row("last", 1024, 1u, NULL);

  /** (f) generic (non-dyadic) rows: the kernel rounds at a 2^-14 resolution, so
   * a true product just above a .5 tie may round to even instead of up. Only
   * the +/-1 rate is bounded here. */
  static const float gamp[4] = {3.7f, 0.013f, 731.f, 2.9e-4f};
  uint32_t pm1 = 0;
  for (int r = 0; r < 64; ++r) {
    for (uint32_t i = 0; i < 1024; ++i)
      x_row[i] = (__fp16)(frand() * gamp[r & 3]);
    x_row[0] = (__fp16)(gamp[r & 3] * 0.973f);
    snprintf(tag, sizeof(tag), "generic%d", r);
    fails += check_row(tag, 1024, 1u, &pm1);
  }
  printf("SIM_TEST quant_generic STAT pm1=%u/%u\n", (unsigned)pm1,
         (unsigned)(64u * 1024u));
  if (pm1 > 13u) {
    printf("SIM_TEST quant FAIL generic rate %u > 13\n", (unsigned)pm1);
    ++fails;
  }

  /** (g) int16: zero row and tie row (x[0] = 32767 -> inv = 1.0, rest n + 0.5)
   * must be identical; (h) k=128/k=3072; (i) 16 generic rows, +/-1 rate only
   * (the kernel rounds at 2^-6 resolution). */
  memset(x_row, 0, sizeof(x_row));
  fails += check_row("i16_zero", 1024, 2u, NULL);
  x_row[0] = (__fp16)32767.f;
  for (uint32_t i = 1; i < 1024; ++i)
    x_row[i] = (__fp16)((i & 1u ? -1.f : 1.f) * ((float)(i % 2047u) + 0.5f));
  fails += check_row("i16_tie", 1024, 2u, NULL);
  for (uint32_t i = 1; i < 1024; ++i)
    x_row[i] =
      (__fp16)(((i >> 1) & 1u ? -1.f : 1.f) * ((float)(i % 2047u) + 0.5f));
  fails += check_row("i16_tie2", 1024, 2u, NULL);
  uint32_t pm16 = 0;
  for (int r = 0; r < 16; ++r) {
    for (uint32_t i = 0; i < KQ; ++i)
      x_row[i] = (__fp16)(frand() * gamp[r & 3]);
    x_row[0] = (__fp16)(gamp[r & 3] * 0.973f);
    snprintf(tag, sizeof(tag), "i16_generic%d", r);
    fails += check_row(tag, r & 1 ? KQ : 128u, 2u, &pm16);
  }
  printf("SIM_TEST quant16_generic STAT pm1=%u/%u\n", (unsigned)pm16,
         (unsigned)(8u * (KQ + 128u)));
  if (pm16 > (8u * (KQ + 128u)) / 50u) {
    printf("SIM_TEST quant FAIL i16 generic rate %u\n", (unsigned)pm16);
    ++fails;
  }

  return fails;
}

int test_quant(void) {
  if (test_quant_row())
    return 1;

  printf("SIM_TEST quant PASS\n");
  return 0;
}
