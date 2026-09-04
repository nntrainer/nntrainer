// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   nntr_hvx_mm_u8i4.c
 * @date   03 Aug 2026
 * @brief  FastRPC entry points for the HMX u8i4 accuracy harness
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 */

#include <string.h>

#include <AEEStdErr.h>
#include <HAP_farf.h>
#include <remote.h>

#include "hexkl_hmx_geom.h"
#include "hexkl_micro.h"
#include "hexkl_mm_u8i4.h"
#include "hexkl_mm_u8i4_dma.h"
#include "hvx_dequant_i32.h"
#include "hvx_quant_u8.h"
#include "nntr_hvx.h"
#include "nntr_hvx_session.h"

#define ROUND_UP(v, a) HEXKL_ROUND_UP_U32(v, a)

/**
 * @brief Accuracy harness: the whole flow, quantization and dequantization
 *        on the DSP, with every intermediate buffer returned so each stage
 *        is checkable. Used by unittest_hvx_mm_u8i4's fixed shapes.
 *
 * hw_init and the HMX lock are session-scoped (nntr_hvx_open), not per
 * call, since the layer endpoint below turned this from a call that stood
 * alone into one of several entry points sharing one session. The weight is
 * still baked fresh every call -- that is the point of this harness, checking
 * the bake -- unlike the resident-weight path in mm_u8i4_layer below.
 */
int nntr_hvx_mm_u8i4_from_f32(
  remote_handle64 handle, uint32 M, uint32 K, uint32 N, const float *act_f32,
  int act_f32Len, const int8 *w_i4_rm, int w_i4_rmLen, const float *w_scale,
  int w_scaleLen, const int32 *colsum_w, int colsum_wLen, const float *bias,
  int biasLen, uint8 *act_u8_ah, int act_u8_ahLen, float *act_scale,
  int act_scaleLen, int32 *act_zp, int act_zpLen, int32 *acc_i32,
  int acc_i32Len, float *out_f32, int out_f32Len) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }

  const uint32_t m_pad = ROUND_UP(M, HEXKL_HMX_INT8_BLOCK_N_ROW);

  if ((uint32_t)act_f32Len != M * K || (uint32_t)w_i4_rmLen != K * N ||
      (uint32_t)act_u8_ahLen != m_pad * K || (uint32_t)act_scaleLen != m_pad ||
      (uint32_t)act_zpLen != m_pad || (uint32_t)acc_i32Len != m_pad * N ||
      (uint32_t)w_scaleLen != N || (uint32_t)colsum_wLen != N ||
      (uint32_t)biasLen != N || (uint32_t)out_f32Len != M * N) {
    FARF(ERROR, "bad lengths (M=%u K=%u N=%u m_pad=%u)", (unsigned)M,
         (unsigned)K, (unsigned)N, (unsigned)m_pad);
    return AEE_EBADPARM;
  }

  hexkl_mm_u8i4_layout L;
  int res = hexkl_mm_u8i4_plan(s->vtcm_base, s->vtcm_size, m_pad, K, N, &L);
  if (res != AEE_SUCCESS) {
    FARF(ERROR, "plan failed: 0x%08x", res);
    return res;
  }

  // K1 then K2, writing the AH tiles straight into VTCM.
  hvx_quant_rows_u8_params(act_f32, M, m_pad, K, act_scale, act_zp,
                           s->quant_pool);
  res = hvx_quant_pack_u8_ah(act_f32, M, m_pad, K, act_scale, act_zp,
                             s->vtcm_base + L.act_base, s->quant_pool);
  if (res != AEE_SUCCESS) {
    return res;
  }
  memcpy(act_u8_ah, s->vtcm_base + L.act_base, (size_t)m_pad * K);

  // setup_acc_read_int32 already ran once in nntr_hvx_open for
  // s->config_off, which hexkl_mm_u8i4_plan recomputes identically here
  // (it is a pure function of vtcm_size) -- no need to call it again.
  res = hexkl_mm_u8i4_bake_weights(&L, w_i4_rm, K, N);
  if (res == AEE_SUCCESS) {
    res = hexkl_mm_u8i4_run(&L, m_pad, K, N, acc_i32);
  }

  if (res == AEE_SUCCESS) {
    hvx_dequant_i32_to_f32(acc_i32, M, m_pad, N, act_scale, act_zp, colsum_w,
                           w_scale, bias, out_f32);
  }
  return res;
}

/**
 * @brief Bakes a K x N int4 weight once and keeps it resident until
 *        weight_release_u8i4 -- see hexkl_mm_u8i4_dma.h.
 */
int nntr_hvx_weight_register_u8i4(remote_handle64 handle, uint32 K, uint32 N,
                                  const int8 *w_i4_rm, int w_i4_rmLen,
                                  const float *w_scale, int w_scaleLen,
                                  const int32 *colsum_w, int colsum_wLen,
                                  const float *bias, int biasLen,
                                  uint32 *w_handle) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  if ((uint32_t)w_i4_rmLen != K * N || (uint32_t)w_scaleLen != N ||
      (uint32_t)colsum_wLen != N || (uint32_t)biasLen != N) {
    FARF(ERROR, "weight_register_u8i4: bad lengths (K=%u N=%u)", (unsigned)K,
         (unsigned)N);
    return AEE_EBADPARM;
  }
  return hexkl_weight_u8i4_register(&s->weights_u8i4, s->vtcm_base,
                                    s->vtcm_size, K, N, w_i4_rm, w_scale,
                                    colsum_w, bias, w_handle);
}

int nntr_hvx_weight_release_u8i4(remote_handle64 handle, uint32 w_handle) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  return hexkl_weight_u8i4_release(&s->weights_u8i4, w_handle);
}

/**
 * @brief Runs a layer's worth of matmuls (Q/K/V, or gate/up) against one
 *        shared activation -- see hexkl_mm_u8i4_dma.h. This is the entry
 *        point nntrainer's ComputeOps seam will call once it is wired up;
 *        nntr_hvx_mm_u8i4_from_f32 above stays the accuracy harness.
 */
int nntr_hvx_mm_u8i4_layer(remote_handle64 handle, uint32 M, uint32 K,
                           const uint32 *w_handles, int w_handlesLen,
                           const float *act_f32, int act_f32Len, float *out_cat,
                           int out_catLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s || w_handlesLen <= 0) {
    return AEE_EBADPARM;
  }
  if ((uint32_t)act_f32Len != M * K) {
    FARF(ERROR, "mm_u8i4_layer: bad act_f32Len (M=%u K=%u)", (unsigned)M,
         (unsigned)K);
    return AEE_EBADPARM;
  }
  uint32_t n_total = 0;
  for (int i = 0; i < w_handlesLen; ++i) {
    if (w_handles[i] >= HEXKL_MM_U8I4_MAX_WEIGHTS ||
        !s->weights_u8i4.slots[w_handles[i]].in_use) {
      return AEE_EBADPARM;
    }
    n_total += s->weights_u8i4.slots[w_handles[i]].N;
  }
  if ((uint32_t)out_catLen != M * n_total) {
    FARF(ERROR, "mm_u8i4_layer: bad out_catLen (M=%u n_total=%u)", (unsigned)M,
         (unsigned)n_total);
    return AEE_EBADPARM;
  }
  return hexkl_mm_u8i4_layer_run(
    &s->weights_u8i4, s->vtcm_base, s->vtcm_size, s->config_off, M, K,
    w_handles, (uint32_t)w_handlesLen, act_f32, out_cat, s->quant_pool);
}
