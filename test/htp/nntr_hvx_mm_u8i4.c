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
#include "nntr_hvx.h"

/** @brief Rounds @a v up to a multiple of @a a (a must be a power of two). */
#define ROUND_UP(v, a) (((v) + ((a)-1)) & ~((a)-1))

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

int nntr_hvx_mm_u8i4_from_u8(remote_handle64 handle, uint32 M, uint32 K,
                             uint32 N, const uint8 *act_u8_ah, int act_u8_ahLen,
                             const int8 *w_i4_rm, int w_i4_rmLen,
                             int32 *acc_i32, int acc_i32Len, uint8 *w_wh_dump,
                             int w_wh_dumpLen) {
  (void)handle;

  const uint32_t m_pad = ROUND_UP(M, HEXKL_HMX_INT8_BLOCK_N_ROW);

  if ((uint32_t)act_u8_ahLen != m_pad * K || (uint32_t)w_i4_rmLen != K * N ||
      (uint32_t)acc_i32Len != m_pad * N) {
    FARF(ERROR, "bad lengths: act=%d w=%d acc=%d (M=%u K=%u N=%u m_pad=%u)",
         act_u8_ahLen, w_i4_rmLen, acc_i32Len, (unsigned)M, (unsigned)K,
         (unsigned)N, (unsigned)m_pad);
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
    FARF(ERROR, "plan failed: 0x%08x (need w=%u act=%u, vtcm=%u)", res,
         (unsigned)L.w_size, (unsigned)L.act_size, (unsigned)vtcm_size);
    return res;
  }

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
    // Activation arrives already in AH layout, so a flat copy places it.
    memcpy(vtcm_base + L.act_base, act_u8_ah, (size_t)m_pad * K);
    res = hexkl_mm_u8i4_run(&L, m_pad, K, N, acc_i32);
  }
  if (res == AEE_SUCCESS && w_wh_dumpLen > 0) {
    const uint32_t n_copy =
      (uint32_t)w_wh_dumpLen < L.w_size ? (uint32_t)w_wh_dumpLen : L.w_size;
    memcpy(w_wh_dump, vtcm_base + L.w_base, n_copy);
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

int nntr_hvx_mm_u8i4_from_f32(
  remote_handle64 handle, uint32 M, uint32 K, uint32 N, const float *act_f32,
  int act_f32Len, const int8 *w_i4_rm, int w_i4_rmLen, const float *w_scale,
  int w_scaleLen, const int32 *colsum_w, int colsum_wLen, const float *bias,
  int biasLen, uint8 *act_u8_ah, int act_u8_ahLen, float *act_scale,
  int act_scaleLen, int32 *act_zp, int act_zpLen, int32 *acc_i32,
  int acc_i32Len, float *out_f32, int out_f32Len) {
  (void)handle;
  (void)M;
  (void)K;
  (void)N;
  (void)act_f32;
  (void)act_f32Len;
  (void)w_i4_rm;
  (void)w_i4_rmLen;
  (void)w_scale;
  (void)w_scaleLen;
  (void)colsum_w;
  (void)colsum_wLen;
  (void)bias;
  (void)biasLen;
  (void)act_u8_ah;
  (void)act_u8_ahLen;
  (void)act_scale;
  (void)act_scaleLen;
  (void)act_zp;
  (void)act_zpLen;
  (void)acc_i32;
  (void)acc_i32Len;
  (void)out_f32;
  (void)out_f32Len;
  return AEE_EUNSUPPORTED;
}
