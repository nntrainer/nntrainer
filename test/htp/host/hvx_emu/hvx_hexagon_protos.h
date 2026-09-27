// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hvx_hexagon_protos.h
 * @date   27 Sep 2026
 * @brief  Host emulation of the HVX intrinsics the M=1 small-op kernels
 *         use, one IEEE f32 operation per lane
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * THE PREMISE: one Vsf multiply / add / subtract is one IEEE-754 f32
 * operation, round to nearest even, subnormals kept. That is what
 * HvxSwigluDet.MatchesScalarBitExact confirmed on the device (rule 24), and
 * what HvxM1Ops.* re-checks for these kernels. Each lane below stores
 * through a volatile so the host compiler cannot contract or reassociate.
 * The integer ops and the byte rotate move bits exactly. What this
 * emulation cannot see: an aligned-load fault (the kernels use
 * HVX_UVector everywhere; a review item), inf/NaN encodings, and timing.
 *
 * vror follows the PRM: Vd.ub[i] = Vu.ub[(i + Rt) mod 128]. The reduction
 * that uses it gives the same bits in either direction (IEEE add
 * commutes), which m1_ops_det.h's comment spells out.
 */

#ifndef __NNTRAINER_HVX_EMU_HVX_HEXAGON_PROTOS_H__
#define __NNTRAINER_HVX_EMU_HVX_HEXAGON_PROTOS_H__

#include <stdint.h>
#include <string.h>

#include "hexagon_types.h"

static inline float hvx_emu_f(int32_t w) {
  float f;
  memcpy(&f, &w, sizeof(f));
  return f;
}
static inline int32_t hvx_emu_w(float f) {
  int32_t w;
  memcpy(&w, &f, sizeof(w));
  return w;
}

static inline HVX_Vector Q6_V_vzero(void) {
  HVX_Vector r;
  memset(&r, 0, sizeof(r));
  return r;
}

static inline HVX_Vector Q6_V_vsplat_R(int32_t x) {
  HVX_Vector r;
  for (int i = 0; i < HVX_EMU_LANES; ++i) {
    r.w[i] = x;
  }
  return r;
}

#define HVX_EMU_SF_BINOP(name, op)                                             \
  static inline HVX_Vector name(HVX_Vector a, HVX_Vector b) {                  \
    HVX_Vector r;                                                              \
    for (int i = 0; i < HVX_EMU_LANES; ++i) {                                  \
      volatile float t = hvx_emu_f(a.w[i]) op hvx_emu_f(b.w[i]);               \
      r.w[i] = hvx_emu_w(t);                                                   \
    }                                                                          \
    return r;                                                                  \
  }
HVX_EMU_SF_BINOP(Q6_Vsf_vadd_VsfVsf, +)
HVX_EMU_SF_BINOP(Q6_Vsf_vsub_VsfVsf, -)
HVX_EMU_SF_BINOP(Q6_Vsf_vmpy_VsfVsf, *)
#undef HVX_EMU_SF_BINOP

#define HVX_EMU_W_BINOP(name, op)                                              \
  static inline HVX_Vector name(HVX_Vector a, HVX_Vector b) {                  \
    HVX_Vector r;                                                              \
    for (int i = 0; i < HVX_EMU_LANES; ++i) {                                  \
      r.w[i] = (int32_t)((uint32_t)a.w[i] op(uint32_t) b.w[i]);                \
    }                                                                          \
    return r;                                                                  \
  }
HVX_EMU_W_BINOP(Q6_Vw_vadd_VwVw, +)
HVX_EMU_W_BINOP(Q6_Vw_vsub_VwVw, -)
#undef HVX_EMU_W_BINOP

static inline HVX_Vector Q6_Vuw_vlsr_VuwR(HVX_Vector a, int32_t n) {
  HVX_Vector r;
  for (int i = 0; i < HVX_EMU_LANES; ++i) {
    r.w[i] = (int32_t)((uint32_t)a.w[i] >> (n & 31));
  }
  return r;
}

static inline HVX_Vector Q6_V_vror_VR(HVX_Vector a, int32_t bytes) {
  HVX_Vector r;
  const uint8_t *src = (const uint8_t *)a.w;
  uint8_t *dst = (uint8_t *)r.w;
  for (int i = 0; i < 4 * HVX_EMU_LANES; ++i) {
    dst[i] = src[(i + bytes) & (4 * HVX_EMU_LANES - 1)];
  }
  return r;
}

static inline int32_t Q6_R_vextract_VR(HVX_Vector a, int32_t byte_off) {
  int32_t w;
  memcpy(&w, (const uint8_t *)a.w + (byte_off & (4 * HVX_EMU_LANES - 1)),
         sizeof(w));
  return w;
}

#endif /* __NNTRAINER_HVX_EMU_HVX_HEXAGON_PROTOS_H__ */
