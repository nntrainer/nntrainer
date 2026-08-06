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

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <qurt.h>

#include "nntr_hvx.h"
#include "nntr_hvx_session.h"

#include "hvx_exp_f32.h"
#include "hvx_softmax_f32.h"

/** @brief f32 lanes per HVX vector in 128B mode. */
#define LANES 32u

/** @brief Fails unless the DSP actually has a 128-byte HVX context. */
static int have_hvx(void) {
  /* Bits 15:8 hold the number of 128-byte HVX contexts. */
  return ((qurt_hvx_get_units() >> 8) & 0xFF) != 0;
}

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
  if (!have_hvx()) {
    return AEE_EUNSUPPORTED;
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
  if (!have_hvx()) {
    return AEE_EUNSUPPORTED;
  }

  hvx_softmax_rows_f32(x, y, m_first, M, K, scale);
  return AEE_SUCCESS;
}
