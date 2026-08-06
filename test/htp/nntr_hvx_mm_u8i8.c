// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   nntr_hvx_mm_u8i8.c
 * @date   06 Aug 2026
 * @brief  FastRPC entry points for the u8i8 performance path
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Mirrors nntr_hvx_mm_u8i4.c's weight_register_u8i4/weight_release_u8i4/
 * mm_u8i4_layer trio, at the wider weight width. There is no u8i8
 * accuracy-harness endpoint to mirror nntr_hvx_mm_u8i4_from_f32: u8i8 was
 * never given one in #4236, and none is added here -- the layer endpoint
 * is checked directly against the same per-weight integer reference the
 * u8i4 tests use (unittest_hvx_mm_u8i4.cpp's HmxMmU8I8Layer fixture).
 */

#include <AEEStdErr.h>
#include <HAP_farf.h>
#include <remote.h>

#include "hexkl_mm_u8i8_dma.h"
#include "nntr_hvx.h"
#include "nntr_hvx_session.h"

int nntr_hvx_weight_register_u8i8(remote_handle64 handle, uint32 K, uint32 N,
                                  const int8 *w_i8_rm, int w_i8_rmLen,
                                  const float *w_scale, int w_scaleLen,
                                  const int32 *colsum_w, int colsum_wLen,
                                  const float *bias, int biasLen,
                                  uint32 *w_handle) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  if ((uint32_t)w_i8_rmLen != K * N || (uint32_t)w_scaleLen != N ||
      (uint32_t)colsum_wLen != N || (uint32_t)biasLen != N) {
    FARF(ERROR, "weight_register_u8i8: bad lengths (K=%u N=%u)", (unsigned)K,
         (unsigned)N);
    return AEE_EBADPARM;
  }
  return hexkl_weight_u8i8_register(&s->weights_u8i8, s->vtcm_base,
                                    s->vtcm_size, K, N, w_i8_rm, w_scale,
                                    colsum_w, bias, w_handle);
}

int nntr_hvx_weight_release_u8i8(remote_handle64 handle, uint32 w_handle) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  return hexkl_weight_u8i8_release(&s->weights_u8i8, w_handle);
}

int nntr_hvx_mm_u8i8_layer(remote_handle64 handle, uint32 M, uint32 K,
                           const uint32 *w_handles, int w_handlesLen,
                           const float *act_f32, int act_f32Len,
                           float *out_cat, int out_catLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s || w_handlesLen <= 0) {
    return AEE_EBADPARM;
  }
  if ((uint32_t)act_f32Len != M * K) {
    FARF(ERROR, "mm_u8i8_layer: bad act_f32Len (M=%u K=%u)", (unsigned)M,
         (unsigned)K);
    return AEE_EBADPARM;
  }
  uint32_t n_total = 0;
  for (int i = 0; i < w_handlesLen; ++i) {
    if (w_handles[i] >= HEXKL_MM_U8I8_MAX_WEIGHTS ||
        !s->weights_u8i8.slots[w_handles[i]].in_use) {
      return AEE_EBADPARM;
    }
    n_total += s->weights_u8i8.slots[w_handles[i]].N;
  }
  if ((uint32_t)out_catLen != M * n_total) {
    FARF(ERROR, "mm_u8i8_layer: bad out_catLen (M=%u n_total=%u)",
         (unsigned)M, (unsigned)n_total);
    return AEE_EBADPARM;
  }
  return hexkl_mm_u8i8_layer_run(&s->weights_u8i8, s->vtcm_base,
                                 s->vtcm_size, s->config_off, M, K,
                                 w_handles, (uint32_t)w_handlesLen, act_f32,
                                 out_cat);
}
