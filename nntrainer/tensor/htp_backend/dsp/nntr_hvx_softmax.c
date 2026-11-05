// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   nntr_hvx_softmax.c
 * @date   05 Aug 2026
 * @brief  DSP-side entries that expose the HVX softmax kernel to the host test
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 */

#include <AEEStdErr.h>
#include <HAP_farf.h>
#include <remote.h>
#include <string.h>

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include "nntr_hvx.h"
#include "nntr_hvx_session.h"

#include "hvx_exp_f32.h"
#include "hvx_expand_i2i4.h"
#include "hvx_softmax_f32.h"
#include "hvx_swiglu_det.h"

/** @brief HVX vector width in bytes (128B mode). */
#define VLEN 128u
/** @brief f32 lanes per HVX vector. */
#define LANES (VLEN / sizeof(float))

int nntr_hvx_exp_f32(remote_handle64 handle, const float *x, int xLen, float *y,
                     int yLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }

  if (xLen != yLen) {
    FARF(ERROR, "exp_f32: bad lengths (xLen=%d yLen=%d)", xLen, yLen);
    return AEE_EBADPARM;
  }
  if (xLen <= 0 || (unsigned)xLen % LANES != 0u) {
    FARF(ERROR, "exp_f32: xLen not a multiple of %u (xLen=%d)", LANES, xLen);
    return AEE_EBADPARM;
  }

  // FastRPC buffers carry no vector alignment guarantee, so the unaligned
  // vector type is what keeps this from faulting.
  const HVX_UVector *vx = (const HVX_UVector *)x;
  HVX_UVector *vy = (HVX_UVector *)y;

  const int nvec = xLen / (int)LANES;
  for (int i = 0; i < nvec; ++i) {
    vy[i] = hvx_exp_sf(vx[i]);
  }
  return AEE_SUCCESS;
}

int nntr_hvx_swiglu_det_f32(remote_handle64 handle, uint32 act,
                            const float *gate, int gateLen, const float *up,
                            int upLen, float *out, int outLen, float *exp_out,
                            int expLen, float *recip_out, int recipLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s || act > 1u) {
    return AEE_EBADPARM;
  }
  if (gateLen != upLen || gateLen != outLen || gateLen != expLen ||
      gateLen != recipLen) {
    FARF(ERROR, "swiglu_det_f32: length mismatch (%d %d %d %d %d)", gateLen,
         upLen, outLen, expLen, recipLen);
    return AEE_EBADPARM;
  }
  if (gateLen <= 0 || (unsigned)gateLen % LANES != 0u) {
    FARF(ERROR, "swiglu_det_f32: len not a multiple of %u (len=%d)",
         (unsigned)LANES, gateLen);
    return AEE_EBADPARM;
  }

  // FastRPC buffers carry no vector alignment guarantee, so the unaligned
  // vector type is what keeps this from faulting.
  const HVX_UVector *vg = (const HVX_UVector *)gate;
  const HVX_UVector *vu = (const HVX_UVector *)up;
  HVX_UVector *vo = (HVX_UVector *)out;
  HVX_UVector *ve = (HVX_UVector *)exp_out;
  HVX_UVector *vr = (HVX_UVector *)recip_out;

  const int nvec = gateLen / (int)LANES;
  for (int i = 0; i < nvec; ++i) {
    // Recomputed rather than threaded out of hvx_swiglu_det_sf: the point
    // of this entry is to report exactly what that function computes, and
    // a variant of it that returns its own intermediates would be a second
    // implementation to keep in step with the first.
    const HVX_Vector g = vg[i];
    HVX_Vector t = g;
    if (act == 1u) { /* GeGLU: the sigmoid's argument is g (C0 + C1 g^2) */
      const HVX_Vector g2 = Q6_Vsf_vmpy_VsfVsf(g, g);
      t = Q6_Vsf_vmpy_VsfVsf(
        g,
        Q6_Vsf_vadd_VsfVsf(hvx_splat_sf(GEGLU_DET_C0),
                           Q6_Vsf_vmpy_VsfVsf(hvx_splat_sf(GEGLU_DET_C1), g2)));
    }
    const HVX_Vector e = hvx_exp_det_sf(Q6_Vsf_vsub_VsfVsf(Q6_V_vzero(), t));
    ve[i] = e;
    vr[i] = hvx_recip_det_sf(Q6_Vsf_vadd_VsfVsf(hvx_splat_sf(1.0f), e));
    vo[i] =
      act == 1u ? hvx_geglu_det_sf(g, vu[i]) : hvx_swiglu_det_sf(g, vu[i]);
  }
  return AEE_SUCCESS;
}

int nntr_hvx_expand_i2i4(remote_handle64 handle, const uint8 *codes,
                         int codesLen, const int8 *pal, int palLen, uint8 *hvx,
                         int hvxLen, uint8 *ref, int refLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  if (palLen != 4) {
    FARF(ERROR, "expand_i2i4: palLen=%d, want 4", palLen);
    return AEE_EBADPARM;
  }
  if (codesLen <= 0 || (unsigned)codesLen % 128u != 0u) {
    FARF(ERROR, "expand_i2i4: codesLen=%d, want a positive multiple of 128",
         codesLen);
    return AEE_EBADPARM;
  }
  if (hvxLen != 2 * codesLen || refLen != 2 * codesLen) {
    FARF(ERROR, "expand_i2i4: hvxLen=%d refLen=%d, want %d", hvxLen, refLen,
         2 * codesLen);
    return AEE_EBADPARM;
  }

  // The table is built once here rather than inside each call so the two
  // paths below are given byte-identical input: if they disagree it is the
  // lookup, not the table.
  uint8_t table[HVX_EXPAND_TABLE_BYTES] __attribute__((aligned(128)));
  hvx_expand_i2i4_table(pal, table);

  hvx_expand_i2i4(codes, (uint32_t)codesLen, table, hvx);
  hvx_expand_i2i4_scalar(codes, (uint32_t)codesLen, table, ref);
  return AEE_SUCCESS;
}

int nntr_hvx_softmax_f32(remote_handle64 handle, uint32 M, uint32 K,
                         uint32 m_first, float scale, const float *x, int xLen,
                         float *y, int yLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }

  if (M == 0u || K == 0u || m_first > M) {
    FARF(ERROR, "softmax_f32: bad shape (M=%u K=%u m_first=%u)", (unsigned)M,
         (unsigned)K, (unsigned)m_first);
    return AEE_EBADPARM;
  }
  if ((uint32)xLen != M * K || xLen != yLen) {
    FARF(ERROR, "softmax_f32: bad lengths (M=%u K=%u xLen=%d yLen=%d)",
         (unsigned)M, (unsigned)K, xLen, yLen);
    return AEE_EBADPARM;
  }

  hvx_softmax_rows_f32(x, y, m_first, M, K, scale);
  return AEE_SUCCESS;
}
