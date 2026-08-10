// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hvx_softmax_util.h
 * @date   07 Aug 2026
 * @brief  Tail-staging and in-vector reduction helpers shared by the HVX
 *         softmax kernels
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * Moved verbatim out of hvx_softmax_f32.c so hvx_softmax_blocked_f32.c can
 * reuse them instead of copying. No behaviour change.
 */

#ifndef __NNTRAINER_HVX_SOFTMAX_UTIL_H__
#define __NNTRAINER_HVX_SOFTMAX_UTIL_H__

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

/** @brief HVX vector width in bytes (128B mode). */
#define VLEN 128u
/** @brief f32 lanes per HVX vector. */
#define LANES (VLEN / sizeof(float))

/**
 * @brief Loads n < LANES floats into lanes 0..n-1, filling the rest with pad.
 *
 * Staging through an aligned stack buffer rather than reading a vector
 * straight off the row end: that read would run up to 124 bytes past the
 * buffer, which rpcmem's page granularity usually hides and occasionally
 * does not. It also keeps whatever lives past the row out of the
 * reductions. Same idiom as hvx_quant_u8.c.
 */
static inline HVX_Vector load_tail_sf(const float *p, uint32_t n, float pad) {
  float buf[LANES] __attribute__((aligned(VLEN)));

  for (uint32_t i = 0; i < LANES; ++i) {
    buf[i] = pad;
  }
  memcpy(buf, p, (size_t)n * sizeof(float));
  return *(const HVX_Vector *)buf;
}

/** @brief Stores lanes 0..n-1, leaving everything past them untouched. */
static inline void store_tail_sf(float *p, uint32_t n, HVX_Vector v) {
  float buf[LANES] __attribute__((aligned(VLEN)));

  *(HVX_Vector *)buf = v;
  memcpy(p, buf, (size_t)n * sizeof(float));
}

/**
 * @brief Folds 32 lanes into one value, replicated across every lane.
 *
 * Rotate by half the remaining width each step: 64, 32, 16, 8, 4 bytes.
 * Same shape as the reduction in hvx_quant_u8.c.
 */
static inline HVX_Vector reduce_max_sf(HVX_Vector v) {
  for (uint32_t rot = VLEN / 2u; rot >= 4u; rot >>= 1) {
    v = Q6_Vsf_vmax_VsfVsf(v, Q6_V_vror_VR(v, (int)rot));
  }
  return v;
}

/** @copydoc reduce_max_sf */
static inline HVX_Vector reduce_sum_sf(HVX_Vector v) {
  for (uint32_t rot = VLEN / 2u; rot >= 4u; rot >>= 1) {
    v = Q6_Vsf_vadd_VsfVsf(v, Q6_V_vror_VR(v, (int)rot));
  }
  return v;
}

/** @brief Pulls lane 0 out as a scalar. */
static inline float lane0_sf(HVX_Vector v) {
  float x;
  store_tail_sf(&x, 1u, v);
  return x;
}

#endif /* __NNTRAINER_HVX_SOFTMAX_UTIL_H__ */
