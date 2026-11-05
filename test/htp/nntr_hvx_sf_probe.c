// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   nntr_hvx_sf_probe.c
 * @date   29 Sep 2026
 * @brief  DSP-side test entry sf_probe (#132 PR 2): one f32 operation per
 *         element on the DSP, so the device gtest can name the operation a
 *         kernel variant disagrees on
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * Tests the operations used by the kernels on host-chosen operands:
 *   op 3 / 4 / 5  Q6_Vsf_vadd / vsub / vmpy
 *   op 6          scalar a / b used by the vector quantizer
 *   op 7          scalar sffma a * b + 1.0 used by the router
 * Retired inline-assembly experiments 0 / 1 / 2 return AEE_EUNSUPPORTED.
 * No model path calls this entry.
 */

#include <string.h>

#include <AEEStdErr.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <remote.h>

#include "nntr_hvx.h"
#include "nntr_hvx_session.h"

int nntr_hvx_sf_probe(remote_handle64 handle, uint32 op, const float *a,
                      int aLen, const float *b, int bLen, float *o, int oLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  if (op <= 2u) {
    return AEE_EUNSUPPORTED;
  }
  if (op > 7u || aLen <= 0 || bLen != aLen || oLen != aLen ||
      (op <= 5u && aLen % 32 != 0)) {
    return AEE_EINVALIDFORMAT;
  }
  if (op >= 6u) {
    for (int i = 0; i < aLen; ++i) {
      volatile float x = a[i], y = b[i];
      volatile float r =
        op == 6u ? x / y : Q6_R_sfmpyacc_RR(1.0f, (float)x, (float)y);
      o[i] = r;
    }
    return AEE_SUCCESS;
  }
  for (int i = 0; i < aLen; i += 32) {
    HVX_Vector x, y, r;
    memcpy(&x, a + i, sizeof(x));
    memcpy(&y, b + i, sizeof(y));
    switch (op) {
    case 3:
      r = Q6_Vsf_vadd_VsfVsf(x, y);
      break;
    case 4:
      r = Q6_Vsf_vsub_VsfVsf(x, y);
      break;
    default:
      r = Q6_Vsf_vmpy_VsfVsf(x, y);
      break;
    }
    memcpy(o + i, &r, sizeof(r));
  }
  return AEE_SUCCESS;
}
