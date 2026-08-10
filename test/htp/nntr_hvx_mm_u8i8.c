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
#include "hexkl_probe.h"
#include "nntr_hvx.h"
#include "nntr_hvx_session.h"

/**
 * @brief Slots mm_u8i8_layer_timed fills, in order.
 *
 * unittest_hvx_fc.cpp mirrors this list; the ARM side cannot include this
 * header, so the entry point rejects a stale count with AEE_EBADPARM rather
 * than silently truncating -- the same contract attn_forward_timed uses.
 */
enum {
  FC_T_DSP_TOTAL = 0, /**< the whole layer_run call, DSP clock */
  FC_T_QUANT,         /**< act f32 -> u8 AH in VTCM */
  FC_T_DEQUANT,       /**< i32 -> f32, in place on the VTCM tile */
  FC_T_ACC_READ,      /**< HMX accumulator -> VTCM, vendor */
  FC_T_ACC_COPY,      /**< VTCM -> DDR staging; 0 on the in-place path */
  FC_T_DRAIN,         /**< DMA waits */
  FC_T_ACC_STRIDE,    /**< not a time: the derived row stride, 0 if fallback */
  FC_N_STAGES
};

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

/** @brief Shape checks both layer entry points need, before either runs. */
static int check_layer_args(const nntr_hvx_session *s, uint32 M, uint32 K,
                            const uint32 *w_handles, int w_handlesLen,
                            int act_f32Len, int out_catLen) {
  uint32_t n_total = 0;
  int i;
  if (!s || w_handlesLen <= 0) {
    return AEE_EBADPARM;
  }
  if ((uint32_t)act_f32Len != M * K) {
    FARF(ERROR, "mm_u8i8_layer: bad act_f32Len (M=%u K=%u)", (unsigned)M,
         (unsigned)K);
    return AEE_EBADPARM;
  }
  for (i = 0; i < w_handlesLen; ++i) {
    if (w_handles[i] >= HEXKL_MM_U8I8_MAX_WEIGHTS ||
        !s->weights_u8i8.slots[w_handles[i]].in_use) {
      return AEE_EBADPARM;
    }
    n_total += s->weights_u8i8.slots[w_handles[i]].N;
  }
  if ((uint32_t)out_catLen != M * n_total) {
    FARF(ERROR, "mm_u8i8_layer: bad out_catLen (M=%u n_total=%u)", (unsigned)M,
         (unsigned)n_total);
    return AEE_EBADPARM;
  }
  return AEE_SUCCESS;
}

int nntr_hvx_mm_u8i8_layer(remote_handle64 handle, uint32 M, uint32 K,
                           const uint32 *w_handles, int w_handlesLen,
                           const float *act_f32, int act_f32Len, float *out_cat,
                           int out_catLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  int rc =
    check_layer_args(s, M, K, w_handles, w_handlesLen, act_f32Len, out_catLen);
  if (rc != AEE_SUCCESS) {
    return rc;
  }
  /** No probe reset here: this is the production path, and hexkl_probe_on stays
   * wherever the last timed call left it -- off, unless one ran. */
  return hexkl_mm_u8i8_layer_run(&s->weights_u8i8, s->vtcm_base, s->vtcm_size,
                                 s->config_off, M, K, w_handles,
                                 (uint32_t)w_handlesLen, act_f32, out_cat,
                                 &(const hexkl_mm_opts){.pool = s->quant_pool});
}

int nntr_hvx_mm_u8i8_layer_timed(remote_handle64 handle, uint32 M, uint32 K,
                                 const uint32 *w_handles, int w_handlesLen,
                                 const float *act_f32, int act_f32Len,
                                 float *out_cat, int out_catLen,
                                 uint32 *stage_us, int stage_usLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  uint64_t t0, t1;
  int rc =
    check_layer_args(s, M, K, w_handles, w_handlesLen, act_f32Len, out_catLen);
  if (rc != AEE_SUCCESS) {
    return rc;
  }
  if (!stage_us || stage_usLen != FC_N_STAGES) {
    FARF(ERROR, "mm_u8i8_layer_timed: stage_usLen %d, expected %d", stage_usLen,
         (int)FC_N_STAGES);
    return AEE_EBADPARM;
  }

  hexkl_probe_reset(1);
  t0 = hexkl_probe_now();
  rc = hexkl_mm_u8i8_layer_run(&s->weights_u8i8, s->vtcm_base, s->vtcm_size,
                               s->config_off, M, K, w_handles,
                               (uint32_t)w_handlesLen, act_f32, out_cat,
                               &(const hexkl_mm_opts){.pool = s->quant_pool});
  t1 = hexkl_probe_now();
  /** Left off again on the way out: the production entry point above shares
   * these globals, and an instrumented run must not make the next
   * uninstrumented one pay for the probes. */
  hexkl_probe_on = 0;

  stage_us[FC_T_DSP_TOTAL] = (uint32)(t1 - t0);
  stage_us[FC_T_QUANT] = (uint32)hexkl_probe_us[HEXKL_PROBE_QUANT];
  stage_us[FC_T_DEQUANT] = (uint32)hexkl_probe_us[HEXKL_PROBE_DEQUANT];
  stage_us[FC_T_ACC_READ] = (uint32)hexkl_probe_us[HEXKL_PROBE_ACC_READ];
  stage_us[FC_T_ACC_COPY] = (uint32)hexkl_probe_us[HEXKL_PROBE_ACC_COPY];
  stage_us[FC_T_DRAIN] = (uint32)hexkl_probe_us[HEXKL_PROBE_DRAIN];
  stage_us[FC_T_ACC_STRIDE] = (uint32)hexkl_probe_us[HEXKL_PROBE_ACC_STRIDE];
  return rc;
}
