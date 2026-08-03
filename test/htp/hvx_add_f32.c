// SPDX-License-Identifier: Apache-2.0
/**
 * @file   hvx_add_f32.c
 * @brief  DSP-side implementation of the nntr_hvx FastRPC interface.
 *
 * Built into libnntr_hvx_skel.so and loaded into the CDSP unsigned PD.
 */

#include <stdlib.h>

#include <AEEStdErr.h>
#include <remote.h>

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <qurt.h>

#include "nntr_hvx.h"

/** @brief HVX vector width in bytes (128B mode). */
#define VLEN 128
/** @brief float lanes per HVX vector. */
#define LANES ((int)(VLEN / sizeof(float)))

int nntr_hvx_open(const char *uri, remote_handle64 *handle) {
  (void)uri;
  /* No per-session state; a unique non-NULL handle is all FastRPC needs. */
  void *ctx = malloc(1);
  if (!ctx) {
    return AEE_ENOMEMORY;
  }
  *handle = (remote_handle64)ctx;
  return AEE_SUCCESS;
}

int nntr_hvx_close(remote_handle64 handle) {
  free((void *)handle);
  return AEE_SUCCESS;
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

  /* FastRPC buffers carry no vector alignment guarantee, so the unaligned
     vector type is what keeps this from faulting on a misaligned input. */
  const HVX_UVector *va = (const HVX_UVector *)a;
  const HVX_UVector *vb = (const HVX_UVector *)b;
  HVX_UVector *vc = (HVX_UVector *)c;

  const int n_vec = aLen / LANES;
  for (int i = 0; i < n_vec; ++i) {
    vc[i] = Q6_Vsf_vadd_VsfVsf(va[i], vb[i]);
  }

  /* ponytail: scalar tail; a masked vector store would fold it in, add
     that only if the tail shows up in a profile. */
  for (int i = n_vec * LANES; i < aLen; ++i) {
    c[i] = a[i] + b[i];
  }

  return AEE_SUCCESS;
}
