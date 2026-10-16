// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hexagon_protos.h
 * @date   22 Sep 2026
 * @brief  Host stand-in: the HVX, cache and user-DMA intrinsics the replay uses
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 */
#pragma once
#include "hexagon_types.h"
#include "hexkl_dma_ring.h"
#include <string.h>

/** @brief Every lane of @a x. */
static inline HVX_Vector Q6_V_vsplat_R(int32_t x) {
  HVX_Vector v;
  for (int i = 0; i < 32; ++i) {
    v.w[i] = x;
  }
  return v;
}
/** @brief Lane-wise add. */
static inline HVX_Vector Q6_Vw_vadd_VwVw(HVX_Vector a, HVX_Vector b) {
  for (int i = 0; i < 32; ++i) {
    a.w[i] += b.w[i];
  }
  return a;
}
/** @brief No cache to clean on the host. */
static inline void Q6_dccleaninva_A(void *p) { (void)p; }

#include "hexkl_dma_standin.h"
