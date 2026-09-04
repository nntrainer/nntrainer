// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file nntr_hvx_softmax.c
 * @date 05 Aug 2026
 * @brief DSP-side entries that expose the HVX softmax kernel to the host test
 * @see https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug No known bugs except for NYI items */

#include <AEEStdErr.h>
#include <HAP_farf.h>
#include <remote.h>
#include <string.h>

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include "nntr_hvx.h"
#include "nntr_hvx_session.h"

#include "hvx_exp_f32.h"
#include "hvx_softmax_blocked_f32.h"
#include "hvx_softmax_f32.h"

/** @brief HVX vector width in bytes (128B mode). */
#define VLEN 128u
/** @brief f32 lanes per HVX vector. */
#define LANES (VLEN / sizeof(float))

/** @brief Bound on KV blocks per band, so the segment pointer array is a
 * fixed-size local rather than a DSP-heap allocation. kv_len 8192 at
 * the smallest useful T (64) needs 128. */
#define NNTR_HVX_MAX_KV_BLOCKS 128u

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

int nntr_hvx_softmax_blocked_f32(remote_handle64 handle, uint32 n_seg, uint32 T,
                                 uint32 M, float scale, const float *band_in,
                                 int band_inLen, const uint32 *begin,
                                 int beginLen, const uint32 *end, int endLen,
                                 const float *sink, int sinkLen,
                                 float *band_out, int band_outLen, float *l_out,
                                 int l_outLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  float *seg[NNTR_HVX_MAX_KV_BLOCKS];

  if (!s) {
    return AEE_EBADPARM;
  }

  if (n_seg == 0u || n_seg > NNTR_HVX_MAX_KV_BLOCKS || T == 0u || M == 0u) {
    FARF(ERROR, "softmax_blocked: bad shape (n_seg=%u T=%u M=%u)",
         (unsigned)n_seg, (unsigned)T, (unsigned)M);
    return AEE_EBADPARM;
  }
  if ((uint32)band_inLen != n_seg * M * T || band_inLen != band_outLen) {
    FARF(ERROR, "softmax_blocked: bad band length (%d, expected %u)",
         band_inLen, (unsigned)(n_seg * M * T));
    return AEE_EBADPARM;
  }
  if ((uint32)beginLen != M || (uint32)endLen != M || (uint32)l_outLen != M) {
    FARF(ERROR, "softmax_blocked: begin/end/l must be M=%u", (unsigned)M);
    return AEE_EBADPARM;
  }
  if (sinkLen != 0 && (uint32)sinkLen != M) {
    FARF(ERROR, "softmax_blocked: sink must be empty or M=%u", (unsigned)M);
    return AEE_EBADPARM;
  }

  /** The kernel is in place, and band_in is an `in` buffer FastRPC may map
   * read-only, so copy across first and run on band_out. */
  memcpy(band_out, band_in, (size_t)band_inLen * sizeof(float));
  for (uint32 j = 0; j < n_seg; ++j) {
    seg[j] = band_out + (size_t)j * M * T;
  }

  hvx_softmax_blocked_f32(seg, n_seg, T, 0u, M, M, scale, begin, end,
                          sinkLen ? sink : NULL, l_out, NULL);
  return AEE_SUCCESS;
}
