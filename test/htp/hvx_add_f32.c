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

#include "nntr_hvx.h"

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
  (void)a;
  (void)b;
  (void)c;

  if (aLen != bLen || aLen != cLen) {
    return AEE_EBADPARM;
  }

  /* Not implemented yet -- Task 3 replaces this with the HVX kernel.
     Reaching this return proves the FastRPC path is up. */
  return AEE_EUNSUPPORTED;
}
