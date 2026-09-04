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
#include <remote.h>

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <qurt.h>

#include "hexkl_micro.h"
#include "nntr_hvx.h"
#include "nntr_hvx_session.h"

/** @brief HVX vector width in bytes (128B mode). */
#define VLEN 128
/** @brief float lanes per HVX vector. */
#define LANES ((int)(VLEN / sizeof(float)))

int nntr_hvx_open(const char *uri, remote_handle64 *handle) {
  (void)uri;

  nntr_hvx_session *s = (nntr_hvx_session *)calloc(1, sizeof(nntr_hvx_session));
  if (!s) {
    return AEE_ENOMEMORY;
  }

  // hw_init and the HMX lock happen once here, for the session's whole
  // lifetime, instead of per call -- every other entry point
  // in this skel reaches vtcm_base/vtcm_size/config_off through the
  // session rather than re-acquiring either.
  uint32_t hmx_fp16_rate = 0;
  int res = hexkl_micro_hw_init(&s->vtcm_base, &s->vtcm_size, &hmx_fp16_rate);
  if (res != AEE_SUCCESS) {
    FARF(ERROR, "nntr_hvx_open: hexkl_micro_hw_init failed: 0x%08x", res);
    free(s);
    return res;
  }
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

  *handle = (remote_handle64)s;
  return AEE_SUCCESS;
}

int nntr_hvx_close(remote_handle64 handle) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_SUCCESS;
  }
  for (uint32_t i = 0; i < HEXKL_MM_U8I4_MAX_WEIGHTS; ++i) {
    if (s->weights.slots[i].in_use) {
      hexkl_weight_u8i4_release(&s->weights, i);
    }
  }
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
