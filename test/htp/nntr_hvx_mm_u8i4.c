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

#include "hexkl_micro.h"
#include "hexkl_mm_u8i4.h"
#include "hvx_dequant_i32.h"
#include "hvx_quant_u8.h"
#include "nntr_hvx.h"

/** @brief Rounds @a v up to a multiple of @a a. */
#define ROUND_UP(v, a) ((((v) + ((a)-1)) / (a)) * (a))

/**
 * @brief Brings up HMX and reports the VTCM budget.
 *
 * Split out because both entry points need the same preamble, and because
 * the VTCM size it reports is itself a measurement we do not have yet.
 */
static int hmx_bringup(uint8_t **vtcm_base, uint32_t *vtcm_size) {
  uint32_t hmx_fp16_rate = 0;
  int res = hexkl_micro_hw_init(vtcm_base, vtcm_size, &hmx_fp16_rate);
  if (res != AEE_SUCCESS) {
    FARF(ERROR, "hexkl_micro_hw_init failed: 0x%08x", res);
    return res;
  }
  FARF(ALWAYS, "vtcm_base=%p vtcm_size=%u config_size=%u hmx_fp16_rate=%u",
       (void *)*vtcm_base, (unsigned)*vtcm_size,
       (unsigned)hexkl_micro_hmx_config_size(), (unsigned)hmx_fp16_rate);
  return AEE_SUCCESS;
}

int nntr_hvx_mm_u8i4_from_f32(
  remote_handle64 handle, uint32 M, uint32 K, uint32 N, const float *act_f32,
  int act_f32Len, const int8 *w_i4_rm, int w_i4_rmLen, const float *w_scale,
  int w_scaleLen, const int32 *colsum_w, int colsum_wLen, const float *bias,
  int biasLen, uint8 *act_u8_ah, int act_u8_ahLen, float *act_scale,
  int act_scaleLen, int32 *act_zp, int act_zpLen, int32 *acc_i32,
  int acc_i32Len, float *out_f32, int out_f32Len) {
  (void)handle;

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

  uint8_t *vtcm_base = NULL;
  uint32_t vtcm_size = 0;
  int res = hmx_bringup(&vtcm_base, &vtcm_size);
  if (res != AEE_SUCCESS) {
    return res;
  }

  hexkl_mm_u8i4_layout L;
  res = hexkl_mm_u8i4_plan(vtcm_base, vtcm_size, m_pad, K, N, &L);
  if (res != AEE_SUCCESS) {
    FARF(ERROR, "plan failed: 0x%08x", res);
    return res;
  }

  // K1 then K2, writing the AH tiles straight into VTCM.
  hvx_quant_rows_u8_params(act_f32, M, m_pad, K, act_scale, act_zp);
  hvx_quant_pack_u8_ah(act_f32, M, m_pad, K, act_scale, act_zp,
                       vtcm_base + L.act_base);
  memcpy(act_u8_ah, vtcm_base + L.act_base, (size_t)m_pad * K);

  res = hexkl_micro_hmx_lock();
  if (res != AEE_SUCCESS) {
    FARF(ERROR, "hexkl_micro_hmx_lock failed: 0x%08x", res);
    return res;
  }

  res = hexkl_micro_hmx_setup_acc_read_int32(vtcm_base, L.config_off);
  if (res == AEE_SUCCESS) {
    res = hexkl_mm_u8i4_bake_weights(&L, w_i4_rm, K, N);
  }
  if (res == AEE_SUCCESS) {
    res = hexkl_mm_u8i4_run(&L, m_pad, K, N, acc_i32);
  }

  if (res == AEE_SUCCESS) {
    hvx_dequant_i32_to_f32(acc_i32, M, m_pad, N, act_scale, act_zp, colsum_w,
                           w_scale, bias, out_f32);
  }

  int res2 = hexkl_micro_hmx_unlock();
  if (res2 != AEE_SUCCESS) {
    FARF(ERROR, "hexkl_micro_hmx_unlock failed: 0x%08x", res2);
    if (res == AEE_SUCCESS) {
      res = res2;
    }
  }
  return res;
}
