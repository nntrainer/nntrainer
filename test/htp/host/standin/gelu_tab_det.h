// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 haehun.yang <haehun.yang@samsung.com>
 *
 * @file   gelu_tab_det.h
 * @date   5 Nov 2026
 * @brief  Scalar spec of the GLU epilogue's gelu table (hvx_gelu_tab_build
 *         and its lookup in hvx_dequant_i32.c), op for op
 * @see    https://github.com/nntrainer/nntrainer
 * @author haehun.yang <haehun.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The nodes k / 256 - 6 through geglu_det_one (hvx_geglu_det_sf's twin),
 * the slopes T[k + 1] - T[k], and for one gate g: x = 256 g + 1536 clamped
 * to [0, 3072], k = rne(x - 1/2), T[k] + S[k] (x - k), g itself past 6,
 * times u -- one rounding an op, as the HVX code does them.
 */
#ifndef __GELU_TAB_DET_H__
#define __GELU_TAB_DET_H__

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "hvx_dequant_i32.h" /* HVX_GELU_TAB_* */
#include "swiglu_det.h"

/** @brief hvx_gelu_tab_build's values and slopes for the nodes. */
static inline void gelu_tab_det_build(float *tab) {
  float *s = tab + HVX_GELU_TAB_STRIDE;
  for (uint32_t k = 0; k < HVX_GELU_TAB_NODES; ++k)
    tab[k] = geglu_det_one((float)((int32_t)k - 1536) * (1.0f / 256.0f), 1.0f);
  for (uint32_t k = 0; k + 1u < HVX_GELU_TAB_NODES; ++k)
    s[k] = tab[k + 1] - tab[k];
  s[HVX_GELU_TAB_NODES - 1u] = 0.0f; /* times f = 0: any value does */
}

/** @brief gelu(g) * u off the tables. */
static inline float gelu_tab_det_one(const float *tab, float g, float u) {
  volatile float x = g * 256.0f;
  x = x + 1536.0f;
  x = fminf(fmaxf(x, 0.0f), 3072.0f);
  volatile float xh = x - 0.5f;
  const int32_t k = (int32_t)nearbyintf(xh);
  volatile float f = x - (float)k;
  volatile float sf = tab[HVX_GELU_TAB_STRIDE + k] * f;
  volatile float t = tab[k] + sf;
  int32_t gb;
  memcpy(&gb, &g, sizeof gb);
  return (gb > 0x40c00000 ? g : t) * u;
}

#endif /* __GELU_TAB_DET_H__ */
