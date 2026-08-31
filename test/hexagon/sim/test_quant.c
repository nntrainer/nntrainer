// SPDX-License-Identifier: Apache-2.0
/**
 * @file	test_quant.c
 * @date	18 August 2026
 * @brief	Hexagon-sim test for per-token dynamic quantization and the int8 dot
 * primitive
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include <stdio.h>
#include <string.h>

#include "hvx-quant.h"
#include "ref_ops.h"
#include "sim_test_util.h"

#define KQ 1024

static __fp16 x_row[KQ] __attribute__((aligned(128)));
static int8_t q_got[KQ] __attribute__((aligned(128)));
static int8_t q_ref[KQ] __attribute__((aligned(128)));

static int8_t w_i8[KQ] __attribute__((aligned(128)));
static int8_t x_i8[KQ] __attribute__((aligned(128)));

static int test_quant_row(void) {
  /* (a) random fp16 row: all int8 values identical and scale exactly equal. */
  for (uint32_t i = 0; i < KQ; ++i)
    x_row[i] = (__fp16)(frand() * 8.f);

  float scale_got = htp_quant_row_fp16(x_row, q_got, KQ);
  float scale_ref = ref_quant_row(x_row, q_ref, KQ);

  if (scale_got != scale_ref) {
    printf("SIM_TEST quant FAIL scale mismatch got=%f ref=%f\n", scale_got,
           scale_ref);
    return 1;
  }
  for (uint32_t i = 0; i < KQ; ++i) {
    if (q_got[i] != q_ref[i]) {
      printf("SIM_TEST quant FAIL random row i=%u got=%d ref=%d\n", (unsigned)i,
             q_got[i], q_ref[i]);
      return 1;
    }
  }

  /* (b) all-zero row: q all zero, scale == 0. */
  memset(x_row, 0, sizeof(x_row));
  scale_got = htp_quant_row_fp16(x_row, q_got, KQ);
  if (scale_got != 0.f) {
    printf("SIM_TEST quant FAIL zero-row scale=%f\n", scale_got);
    return 1;
  }
  for (uint32_t i = 0; i < KQ; ++i) {
    if (q_got[i] != 0) {
      printf("SIM_TEST quant FAIL zero-row i=%u got=%d\n", (unsigned)i,
             q_got[i]);
      return 1;
    }
  }
  return 0;
}

static int test_dot_i8_k(uint32_t k) {
  for (uint32_t i = 0; i < k; ++i) {
    w_i8[i] = (int8_t)(frand() * 127.f);
    x_i8[i] = (int8_t)(frand() * 127.f);
  }
  int32_t got = hvx_dot_i8(w_i8, x_i8, k);
  int32_t ref = ref_dot_i8(w_i8, x_i8, k);
  if (got != ref) {
    printf("SIM_TEST quant FAIL dot_i8 k=%u got=%d ref=%d\n", (unsigned)k,
           (int)got, (int)ref);
    return 1;
  }
  return 0;
}

int test_quant(void) {
  if (test_quant_row())
    return 1;
  if (test_dot_i8_k(128))
    return 1;
  if (test_dot_i8_k(1024))
    return 1;

  printf("SIM_TEST quant PASS\n");
  return 0;
}
