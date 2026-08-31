// SPDX-License-Identifier: Apache-2.0
/**
 * @file	test_exp.c
 * @date	18 August 2026
 * @brief	Accuracy test for the imported HVX exp primitive vs expf
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include <stdio.h>

#include "hvx-exp.h"
#include "sim_test_util.h"

#define N 4096

/* 128-byte aligned buffers: hvx_exp_f32 takes the fast (vector-load) path
 * only when both src and dst are VLEN(128)-aligned; unaligned inputs still
 * work via its HVX_UVector fallback path, but alignment is exercised here. */
static float src[N] __attribute__((aligned(128)));
static float got[N] __attribute__((aligned(128)));
static float ref[N];

int test_exp(void) {
  for (int i = 0; i < N; ++i) {
    /* frand() in [-1, 1) -> x in [-20, 4] */
    float x = -8.f + 12.f * frand();
    src[i] = x;
    ref[i] = expf(x);
  }

  hvx_exp_f32((uint8_t *)got, (uint8_t *)src, N, false);

  if (cmp_f("exp", ref, got, N, 1e-3f, 1e-6f))
    return 1;

  printf("SIM_TEST exp PASS\n");
  return 0;
}
