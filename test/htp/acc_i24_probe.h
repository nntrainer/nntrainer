// SPDX-License-Identifier: Apache-2.0
/**
 * @file acc_i24_probe.h
 * @brief Diagnostic full-int32 reconstruction from the signed24 planes.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#ifndef __NNTRAINER_ACC_I24_PROBE_H__
#define __NNTRAINER_ACC_I24_PROBE_H__
#include "hexkl_acc_i24.h"

#define HEXKL_ACC_I24_PROBE_PLANE_OFF 4096u
#define HEXKL_ACC_I24_PROBE_SCRATCH_BYTES 10240u

#if defined(__hexagon__) || defined(HEXKL_ACC_I24_HVX_EMU)
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

/** Sign-extend the high byte before interleaving halfwords: the resulting
 * words are the complete signed int32 values, in the vendor tile order. */
static inline void hexkl_acc_i24_zip(const uint8_t *lo, const uint8_t *mid,
                                     const uint8_t *hi, int32_t *dst) {
  const HVX_Vector *vl = (const HVX_Vector *)lo;
  const HVX_Vector *vm = (const HVX_Vector *)mid;
  const HVX_Vector *vh = (const HVX_Vector *)hi;
  HVX_Vector *vd = (HVX_Vector *)dst;
  for (uint32_t i = 0; i < HEXKL_CVT_PLANE_BYTES / 128u; ++i) {
    const HVX_VectorPair low16 = Q6_W_vshuff_VVR(vm[i], vl[i], -1);
    const HVX_VectorPair high16 = Q6_Wh_vunpack_Vb(vh[i]);
    const HVX_VectorPair a =
      Q6_W_vshuff_VVR(Q6_V_lo_W(high16), Q6_V_lo_W(low16), -2);
    const HVX_VectorPair b =
      Q6_W_vshuff_VVR(Q6_V_hi_W(high16), Q6_V_hi_W(low16), -2);
    vd[4u * i] = Q6_V_lo_W(a);
    vd[4u * i + 1u] = Q6_V_hi_W(a);
    vd[4u * i + 2u] = Q6_V_lo_W(b);
    vd[4u * i + 3u] = Q6_V_hi_W(b);
  }
}
#else
static inline void hexkl_acc_i24_zip(const uint8_t *lo, const uint8_t *mid,
                                     const uint8_t *hi, int32_t *dst) {
  for (uint32_t i = 0; i < HEXKL_CVT_PLANE_BYTES; ++i)
    dst[i] = (int32_t)lo[i] + ((int32_t)mid[i] << 8) +
             ((hi[i] < 128u ? (int32_t)hi[i] : (int32_t)hi[i] - 256) * 65536);
}
#endif

#if defined(__hexagon__)
/** Full int32 extraction for the direct hardware comparison probe. */
static inline void hexkl_acc_i24_read(uint8_t *scratch, int32_t *dst) {
  uint8_t *lo = scratch + HEXKL_ACC_I24_PROBE_PLANE_OFF;
  uint8_t *mid = lo + HEXKL_CVT_PLANE_BYTES;
  uint8_t *hi = mid + HEXKL_CVT_PLANE_BYTES;
  hexkl_acc_i24_read_planes(scratch, lo);
  hexkl_acc_i24_zip(lo, mid, hi, dst);
}
#endif
#endif
