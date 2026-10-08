// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   geglu_host_check.c
 * @date   01 Oct 2026
 * @brief  Host check: the MoE gate_up epilogue's GeGLU-tanh (plan 201 S4) --
 *         geglu_det_one against f64, and the real HVX epilogue on hvx_emu/
 *         against geglu_det_one / swiglu_det_one bit for bit
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * 1. SPEC. geglu_det_one (swiglu_det.h) against gelu_pytorch_tanh in double:
 *    the SNR over g in [-12, 12], the tanh saturation ends (|g| up to
 *    3e38, where g^3 overflows) and subnormal g, no NaN anywhere.
 * 2. KERNEL. hvx_geglu_det_sf over the same values, and the skel's own
 *    hvx_dequant_swiglu_acc_tiles_to_f32 (hvx_dequant_i32.c, both values
 *    of the geglu field) over random int32 tile pairs, memcmp'd against
 *    the scalar dequant + spec. It rests on one Vsf op = one IEEE op
 *    (rule 24); whether the device's Vsf keeps subnormals is not checked
 *    here. run_host_checks.sh then swaps gelu for silu in the kernel and
 *    this check must fail.
 */

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hvx_dequant_i32.h"
#include "hvx_swiglu_det.h"
#include "swiglu_det.h"

static int g_fail;

#define CHECK(c, ...)                                                          \
  do {                                                                         \
    if (!(c)) {                                                                \
      printf("FAIL: " __VA_ARGS__);                                            \
      printf("\n");                                                            \
      g_fail = 1;                                                              \
    }                                                                          \
  } while (0)

/** @brief gelu_pytorch_tanh(g) * u in double. */
static double geglu_f64(double g, double u) {
  return 0.5 * g *
         (1.0 + tanh(0.7978845608028654 * (g + 0.044715 * g * g * g))) * u;
}

#define N_SWEEP 4096u
#define N_EDGE 32u
#define N_ALL (N_SWEEP + N_EDGE)

int main(void) {
  static float g[N_ALL], u[N_ALL], spec[N_ALL], hvx[N_ALL];
  /* Each with both signs. 1e30, not FLT_MAX: exp_det's clamp at 85 floors
     the sigmoid at exp(-85) = 1.2e-37, so g = -FLT_MAX gives -41 where the
     answer is 0 -- swiglu_det_one's same floor, unreachable from a
     dequantized matmul output. */
  static const float edge[N_EDGE / 2u] = {
    20.f,    50.f,   1e3f,   1e6f,   7e12f,    1e13f, 1e20f, 1e30f,
    FLT_MIN, 1e-38f, 1e-40f, 1e-44f, 1.4e-45f, 0.f,   10.3f, 9.0f};
  for (uint32_t i = 0; i < N_SWEEP; ++i) {
    g[i] = -12.f + 24.f * (float)i / (float)(N_SWEEP - 1u);
    u[i] = 0.5f + (float)(i % 7u) * 0.25f;
  }
  for (uint32_t i = 0; i < N_EDGE; ++i) {
    g[N_SWEEP + i] = (i & 1u) ? -edge[i / 2u] : edge[i / 2u];
    u[N_SWEEP + i] = 1.0f;
  }

  /* 1. the spec against f64 */
  double sig = 0.0, err = 0.0, edge_rel = 0.0, sub_abs = 0.0;
  uint32_t nan = 0;
  for (uint32_t i = 0; i < N_ALL; ++i) {
    spec[i] = geglu_det_one(g[i], u[i]);
    const double r = geglu_f64(g[i], u[i]), d = (double)spec[i] - r;
    nan += isnan(spec[i]) != 0;
    if (i < N_SWEEP) {
      sig += r * r;
      err += d * d;
    } else if (fabsf(g[N_SWEEP + (i - N_SWEEP)]) < FLT_MIN) {
      sub_abs = fmax(sub_abs, fabs(d));
    } else {
      /* relative where the answer is large, absolute near zero */
      edge_rel = fmax(edge_rel, fabs(d) / fmax(fabs(r), 1.0));
    }
  }
  const double snr = 10.0 * log10(sig / err);
  printf("GEGLU SPEC vs f64: snr=%.1f dB over [-12,12] n=%u; saturation ends "
         "max_err=%.2e; subnormal g max_abs=%.2e; nan=%u\n",
         snr, N_SWEEP, edge_rel, sub_abs, nan);
  CHECK(snr >= 130.0, "geglu spec SNR %.1f dB < 130", snr);
  CHECK(edge_rel <= 1e-6, "geglu spec saturation error %.2e", edge_rel);
  CHECK(sub_abs <= 2.0 * 1.4e-45, "geglu spec subnormal error %.2e", sub_abs);
  CHECK(nan == 0u, "geglu spec produced NaN");

  /* 2a. hvx_geglu_det_sf, lane for lane */
  uint32_t same = 0;
  for (uint32_t i = 0; i < N_ALL; i += 32u) {
    const HVX_Vector vg = *(const HVX_UVector *)(g + i);
    const HVX_Vector vu = *(const HVX_UVector *)(u + i);
    *(HVX_UVector *)(hvx + i) = hvx_geglu_det_sf(vg, vu);
  }
  for (uint32_t i = 0; i < N_ALL; ++i)
    same += memcmp(&hvx[i], &spec[i], sizeof(float)) == 0;
  printf("GEGLU HVX == SPEC bit-exact %u/%u\n", same, N_ALL);
  CHECK(same == N_ALL, "hvx_geglu_det_sf differs from geglu_det_one");

  /* 2b. the epilogue the MoE kernel runs: n_pairs gate/up tile pairs of
     m_count rows, dequantized and fed to the activation the job names */
  enum { NP = 6, ROWS = 4, RS = 32, INTER = NP * 32 };
  static int32_t tiles[2 * NP][ROWS * RS];
  static float out[ROWS * INTER];
  static int32_t colsum[2 * INTER];
  static float wsc[2 * INTER], bias[2 * INTER];
  float as[ROWS];
  int32_t az[ROWS];
  uint32_t rs = 777u;
#define RND() (rs = rs * 1664525u + 1013904223u, rs >> 8)
  for (uint32_t t = 0; t < 2u * NP; ++t)
    for (uint32_t i = 0; i < ROWS * RS; ++i)
      tiles[t][i] = (int32_t)(RND() % 200001u) - 100000;
  for (uint32_t c = 0; c < 2u * INTER; ++c) {
    colsum[c] = (int32_t)(RND() % 4001u) - 2000;
    wsc[c] = 0.001f + 0.0001f * (float)(RND() % 50u);
    bias[c] = 0.01f * ((float)(RND() % 201u) - 100.f);
  }
  for (uint32_t m = 0; m < ROWS; ++m) {
    as[m] = 0.002f + 0.001f * (float)m;
    az[m] = 100 + (int32_t)m;
  }
  for (uint32_t geglu = 0; geglu < 2u; ++geglu) {
    hvx_dequant_swiglu_acc_tiles_to_f32((const uint8_t *)tiles, sizeof tiles[0],
                                        NP, 0u, RS, ROWS, as, az, colsum, wsc,
                                        bias, INTER, out, INTER, geglu, NULL);
    uint32_t ok = 0;
    for (uint32_t m = 0; m < ROWS; ++m)
      for (uint32_t c = 0; c < INTER; ++c) {
        const uint32_t j = c / 32u, k = c % 32u, cu = INTER + c;
        /* dq_row_sf's order: ((acc - zp * colsum) * scale) * w + bias */
        const float gq =
          (((float)tiles[j][m * RS + k] - (float)az[m] * (float)colsum[c]) *
           as[m]) *
            wsc[c] +
          bias[c];
        const float uq = (((float)tiles[NP + j][m * RS + k] -
                           (float)az[m] * (float)colsum[cu]) *
                          as[m]) *
                           wsc[cu] +
                         bias[cu];
        const float want =
          geglu ? geglu_det_one(gq, uq) : swiglu_det_one(gq, uq);
        ok += memcmp(&out[m * INTER + c], &want, sizeof(float)) == 0;
      }
    printf("GEGLU EPILOGUE hvx_dequant_i32.c %s bit-exact %u/%u\n",
           geglu ? "geglu" : "swiglu", ok, ROWS * INTER);
    CHECK(ok == ROWS * INTER, "epilogue (geglu=%u) differs from the spec",
          geglu);
  }

  printf(g_fail ? "GEGLU CHECK FAILED\n" : "GEGLU OK\n");
  return g_fail;
}
