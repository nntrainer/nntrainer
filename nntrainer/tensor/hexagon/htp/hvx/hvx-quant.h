// SPDX-License-Identifier: Apache-2.0
/**
 * @file	hvx-quant.h
 * @date	18 August 2026
 * @brief	Per-token dynamic int8 quantization and the vrmpy int8 dot primitive
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#ifndef HVX_QUANT_H
#define HVX_QUANT_H

#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <math.h>
#include <stdint.h>

/* Scalar quantization: cost is O(M*K)~=0.1%, so scalar is fine for M2.
 * Returns scale (absmax / 127). */
static inline float htp_quant_row_fp16(const __fp16 *x, int8_t *q, uint32_t k) {
  float amax = 0.f;
  for (uint32_t i = 0; i < k; ++i) {
    float v = fabsf((float)x[i]);
    if (v > amax)
      amax = v;
  }
  float inv = amax > 0.f ? 127.f / amax : 0.f;
  for (uint32_t i = 0; i < k; ++i)
    q[i] = (int8_t)lrintf((float)x[i] * inv);
  return amax / 127.f;
}

static inline int32_t hvx_reduce_vw(HVX_Vector v) {
  for (int s = 4; s < 128; s <<= 1)
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, s));
  int32_t out[32] __attribute__((aligned(128)));
  *(HVX_Vector *)out = v;
  return out[0];
}

/* HVX int8 dot: k%128==0, w/x 128B aligned. */
static inline int32_t hvx_dot_i8(const int8_t *w, const int8_t *x, uint32_t k) {
  const HVX_Vector *wv = (const HVX_Vector *)w, *xv = (const HVX_Vector *)x;
  HVX_Vector acc = Q6_V_vzero();
  for (uint32_t i = 0; i < k / 128; ++i)
    acc = Q6_Vw_vrmpyacc_VwVbVb(acc, wv[i],
                                xv[i]); /* signed*signed, 128 MAC/instruction */
  return hvx_reduce_vw(acc);
}

#endif /* HVX_QUANT_H */
