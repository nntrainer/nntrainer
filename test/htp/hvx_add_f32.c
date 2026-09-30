// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hvx_add_f32.c
 * @date   03 Aug 2026
 * @brief  DSP-side implementation of the nntr_hvx FastRPC interface
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 */

#include <stdlib.h>
#include <string.h>

#include <AEEStdErr.h>
#include <HAP_farf.h>
#include <HAP_power.h>
#include <remote.h>

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <qurt.h>

#include "hexkl_micro.h"
#include "hvx_worker_pool.h"
#include "nntr_hvx.h"
#include "nntr_hvx_session.h"

/** @brief HVX vector width in bytes (128B mode). */
#define VLEN 128
/** @brief float lanes per HVX vector. */
#define LANES ((int)(VLEN / sizeof(float)))

/**
 * @brief Votes the DSP to its top clocks for the life of the session:
 *        compute client class, core and bus at the maximum corner with
 *        DCVS and sleep off, HVX on, HMX on at its maximum clock -- the
 *        sequence llama.cpp's HTP backend uses. Without a vote the core
 *        runs at whatever DCVS picked for a bursty client, and every
 *        kernel time here (and their ratios to each other) moves with it.
 *        The session pointer is the vote's context; the vote ends with the
 *        process.
 */
static int nntr_hvx_power_vote(void *ctx) {
  HAP_power_request_t req;
  memset(&req, 0, sizeof(req));
  req.type = HAP_power_set_apptype;
  req.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;
  int res = HAP_power_set(ctx, &req);
  if (res != AEE_SUCCESS) {
    return res;
  }

  memset(&req, 0, sizeof(req));
  req.type = HAP_power_set_DCVS_v3;
  req.dcvs_v3.set_dcvs_enable = TRUE;
  req.dcvs_v3.dcvs_enable = FALSE;
  req.dcvs_v3.set_core_params = TRUE;
  req.dcvs_v3.core_params.min_corner = HAP_DCVS_VCORNER_MAX;
  req.dcvs_v3.core_params.max_corner = HAP_DCVS_VCORNER_MAX;
  req.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_MAX;
  req.dcvs_v3.set_bus_params = TRUE;
  req.dcvs_v3.bus_params.min_corner = HAP_DCVS_VCORNER_MAX;
  req.dcvs_v3.bus_params.max_corner = HAP_DCVS_VCORNER_MAX;
  req.dcvs_v3.bus_params.target_corner = HAP_DCVS_VCORNER_MAX;
  req.dcvs_v3.set_sleep_disable = TRUE;
  req.dcvs_v3.sleep_disable = TRUE;
  res = HAP_power_set(ctx, &req);
  if (res != AEE_SUCCESS) {
    return res;
  }

  memset(&req, 0, sizeof(req));
  req.type = HAP_power_set_HVX;
  req.hvx.power_up = TRUE;
  res = HAP_power_set(ctx, &req);
  if (res != AEE_SUCCESS) {
    return res;
  }

  memset(&req, 0, sizeof(req));
  req.type = HAP_power_set_HMX_v2;
  req.hmx_v2.set_power = TRUE;
  req.hmx_v2.power_up = TRUE;
  req.hmx_v2.set_clock = TRUE;
  req.hmx_v2.target_corner = HAP_DCVS_EXP_VCORNER_MAX;
  req.hmx_v2.min_corner = HAP_DCVS_EXP_VCORNER_MAX;
  req.hmx_v2.max_corner = HAP_DCVS_EXP_VCORNER_MAX;
  req.hmx_v2.perf_mode = HAP_CLK_PERF_HIGH;
  return HAP_power_set(ctx, &req);
}

int nntr_hvx_open(const char *uri, remote_handle64 *handle) {
  (void)uri;

  nntr_hvx_session *s = (nntr_hvx_session *)calloc(1, sizeof(nntr_hvx_session));
  if (!s) {
    return AEE_ENOMEMORY;
  }

  // Before hw_init: HexKL's own setup does not vote, and the HMX lock is
  // easier to get with the unit powered.
  int pres = nntr_hvx_power_vote(s);
  if (pres != AEE_SUCCESS) {
    // A part without a separate HMX clock rejects the HMX_v2 request; the
    // core/bus votes above already went through. Report, keep going.
    FARF(ALWAYS, "nntr_hvx_open: power vote returned 0x%08x", pres);
  }

  // hw_init and the HMX lock happen once here, for the session's whole
  // lifetime, instead of per call -- every other entry point
  // in this skel reaches vtcm_base/vtcm_size/config_off through the
  // session rather than re-acquiring either.
  int res =
    hexkl_micro_hw_init(&s->vtcm_base, &s->vtcm_size, &s->hmx_fp16_rate);
  if (res != AEE_SUCCESS) {
    FARF(ERROR, "nntr_hvx_open: hexkl_micro_hw_init failed: 0x%08x", res);
    free(s);
    return res;
  }
  // Kept on the session rather than dropped: the f16 attention entries gate
  // on it, and HexKL's own f16 example treats 0 as "no fp16 HMX here".
  FARF(ALWAYS, "nntr_hvx_open: vtcm_size=%u hmx_fp16_rate=%u",
       (unsigned)s->vtcm_size, (unsigned)s->hmx_fp16_rate);
  // config_off depends only on vtcm_size (see hexkl_mm_u8i4_plan), so it is
  // computed once here rather than at every mm_u8i4_layer call.
  const uint32_t config_size = hexkl_micro_hmx_config_size();
  if (s->vtcm_size < config_size) {
    free(s);
    return AEE_ENOMEMORY;
  }
  s->config_off =
    (s->vtcm_size - config_size) & ~(HEXKL_HMX_CONFIG_ALIGNMENT - 1u);

  res = hexkl_micro_hmx_lock();
  if (res != AEE_SUCCESS) {
    FARF(ERROR, "nntr_hvx_open: hexkl_micro_hmx_lock failed: 0x%08x", res);
    free(s);
    return res;
  }
  s->hmx_locked = 1;

  res = hexkl_micro_hmx_setup_acc_read_int32(s->vtcm_base, s->config_off);
  if (res != AEE_SUCCESS) {
    FARF(ERROR, "nntr_hvx_open: setup_acc_read_int32 failed: 0x%08x", res);
    hexkl_micro_hmx_unlock();
    free(s);
    return res;
  }

  // Sized from the real HVX context count (bits 15:8, same decode
  // nntr_hvx_add_f32 uses to check HVX is present at all) minus one, since
  // this session's own FastRPC thread uses an HVX context too when it runs
  // quant's own share of the work.
  const uint32_t n_hvx = (qurt_hvx_get_units() >> 8) & 0xFFu;
  s->quant_pool = hvx_worker_pool_create(n_hvx > 1 ? n_hvx - 1 : 0);
  if (!s->quant_pool) {
    FARF(ERROR, "nntr_hvx_open: hvx_worker_pool_create failed");
    hexkl_micro_hmx_unlock();
    free(s);
    return AEE_ENOMEMORY;
  }

  *handle = (remote_handle64)s;
  return AEE_SUCCESS;
}

int nntr_hvx_close(remote_handle64 handle) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_SUCCESS;
  }
  for (uint32_t i = 0; i < HEXKL_MM_U8I4_MAX_WEIGHTS; ++i) {
    if (s->weights_u8i4.slots[i].in_use) {
      hexkl_weight_u8i4_release(&s->weights_u8i4, i);
    }
  }
  for (uint32_t i = 0; i < HEXKL_MM_U8I8_MAX_WEIGHTS; ++i) {
    if (s->weights_u8i8.slots[i].in_use) {
      hexkl_weight_u8i8_release(&s->weights_u8i8, i);
    }
  }
  for (uint32_t i = 0; i < HEXKL_KV_TILES_MAX; ++i) {
    if (s->kv_tiles.slots[i].in_use) {
      hexkl_kv_tiles_f16_release(&s->kv_tiles, i);
    }
  }
  for (uint32_t i = 0; i < HEXKL_KV_Q_MAX; ++i) {
    if (s->kv_q.slots[i].in_use) {
      hexkl_kv_q_release(&s->kv_q, i);
    }
  }
  hvx_worker_pool_destroy(s->quant_pool);
  int res = AEE_SUCCESS;
  if (s->hmx_locked) {
    res = hexkl_micro_hmx_unlock();
    if (res != AEE_SUCCESS) {
      FARF(ERROR, "nntr_hvx_close: hexkl_micro_hmx_unlock failed: 0x%08x", res);
    }
  }
  free(s);
  return res;
}

int nntr_hvx_add_f32(remote_handle64 handle, const float *a, int aLen,
                     const float *b, int bLen, float *c, int cLen) {
  (void)handle;

  if (aLen != bLen || aLen != cLen) {
    return AEE_EBADPARM;
  }

  /* Bits 15:8 hold the number of 128-byte HVX contexts. */
  if (((qurt_hvx_get_units() >> 8) & 0xFF) == 0) {
    return AEE_EUNSUPPORTED;
  }

  // FastRPC buffers carry no vector alignment guarantee, so the unaligned
  // vector type is what keeps this from faulting on a misaligned input.
  const HVX_UVector *va = (const HVX_UVector *)a;
  const HVX_UVector *vb = (const HVX_UVector *)b;
  HVX_UVector *vc = (HVX_UVector *)c;

  const int n_vec = aLen / LANES;
  for (int i = 0; i < n_vec; ++i) {
    vc[i] = Q6_Vsf_vadd_VsfVsf(va[i], vb[i]);
  }

  // ponytail: scalar tail; a masked vector store would fold it in, add
  // that only if the tail shows up in a profile.
  for (int i = n_vec * LANES; i < aLen; ++i) {
    c[i] = a[i] + b[i];
  }

  return AEE_SUCCESS;
}
