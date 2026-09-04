// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file nntr_hvx_mm_u8i4.c
 * @date 03 Aug 2026
 * @brief FastRPC entry points for the HMX u8i4 accuracy harness
 * @see https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug No known bugs except for NYI items */

#include <string.h>

#include <AEEStdErr.h>
#include <HAP_farf.h>
#include <remote.h>

#include "hexkl_micro.h"
#include "hexkl_mm_u8i4.h"
#include "hexkl_mm_u8i4_dma.h"
#include "hexkl_probe.h"
#include "hvx_dequant_i32.h"
#include "hvx_quant_u8.h"
#include "nntr_hvx.h"
#include "nntr_hvx_session.h"

/** @brief Rounds @a v up to a multiple of @a a. */
#define ROUND_UP(v, a) ((((v) + ((a)-1)) / (a)) * (a))

/**
 * @brief Accuracy harness: the whole flow, quantization and dequantization
 * on the DSP, with every intermediate buffer returned so each stage
 * is checkable. Used by unittest_hvx_mm_u8i4's fixed shapes.
 *
 * hw_init and the HMX lock are session-scoped (nntr_hvx_open), not per
 * call, since item 3 turned this from a call that stood alone into
 * one of several entry points sharing one session. The weight is still
 * baked fresh every call that is the point of this harness, checking the
 * bake unlike the resident-weight path in mm_u8i4_layer below. */
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
  // (it is a pure function of vtcm_size) no need to call it again.
  res = hexkl_mm_u8i4_bake_weights(&L, w_i4_rm, K, N);
  if (res == AEE_SUCCESS) {
    res = hexkl_mm_u8i4_run(&L, m_pad, K, N, acc_i32);
  }

  if (res == AEE_SUCCESS) {
    hvx_dequant_i32_to_f32(acc_i32, M, m_pad, N, act_scale, act_zp, colsum_w,
                           w_scale, bias, out_f32, /*accumulate=*/0);
  }
  return res;
}

/**
 * @brief Bakes a K x N int4 weight once and keeps it resident until
 * weight_release_u8i4 see hexkl_mm_u8i4_dma.h. */
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
 * shared activation see hexkl_mm_u8i4_dma.h. This is the entry
 * point PR③'s ComputeOps seam will call; nntr_hvx_mm_u8i4_from_f32
 * above stays the accuracy harness. */
/**
 * @brief Slots mm_u8i4_layer_timed fills, in order.
 *
 * MUST match the same enum in nntr_hvx_mm_u8i8.c and the kStage list in
 * unittest_hvx_fc.cpp: the ARM side cannot include this header, so the entry
 * point rejects a stale count with AEE_EBADPARM rather than silently
 * truncating. Restated per width rather than shared, the same call this tree
 * already made for the two dma modules. */
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

/** @brief Shared by both entry points below, so they cannot drift apart on
 * what they accept. */
static int check_layer_args(const nntr_hvx_session *s, uint32 M, uint32 K,
                            const uint32 *w_handles, int w_handlesLen,
                            int act_f32Len, int out_catLen) {
  uint32_t n_total = 0;
  int i;
  if (!s || w_handlesLen <= 0) {
    return AEE_EBADPARM;
  }
  if ((uint32_t)act_f32Len != M * K) {
    FARF(ERROR, "mm_u8i4_layer: bad act_f32Len (M=%u K=%u)", (unsigned)M,
         (unsigned)K);
    return AEE_EBADPARM;
  }
  for (i = 0; i < w_handlesLen; ++i) {
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
  return AEE_SUCCESS;
}

int nntr_hvx_mm_u8i4_layer(remote_handle64 handle, uint32 M, uint32 K,
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
   * wherever the last timed call left it off, unless one ran. */
  return hexkl_mm_u8i4_layer_run(&s->weights_u8i4, s->vtcm_base, s->vtcm_size,
                                 s->config_off, M, K, w_handles,
                                 (uint32_t)w_handlesLen, act_f32, out_cat,
                                 &(const hexkl_mm_opts){.pool = s->quant_pool});
}

int nntr_hvx_mm_u8i4_layer_timed(remote_handle64 handle, uint32 M, uint32 K,
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
    FARF(ERROR, "mm_u8i4_layer_timed: stage_usLen %d, expected %d", stage_usLen,
         (int)FC_N_STAGES);
    return AEE_EBADPARM;
  }

  hexkl_probe_reset(1);
  t0 = hexkl_probe_now;
  rc = hexkl_mm_u8i4_layer_run(&s->weights_u8i4, s->vtcm_base, s->vtcm_size,
                               s->config_off, M, K, w_handles,
                               (uint32_t)w_handlesLen, act_f32, out_cat,
                               &(const hexkl_mm_opts){.pool = s->quant_pool});
  t1 = hexkl_probe_now;
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
