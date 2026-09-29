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
 * The 2026-09-29 sitting read hvx_native (the inline-asm IEEE vadd / vsub /
 * vmpy .sf instructions) and the sffma tail (whose terms used the same
 * vmpy .sf) as 0 in every output on the phone, while the v79 ISS matched
 * the spec and the Q6_Vsf_* intrinsics (qf32 op + conversion) matched on
 * silicon. This entry runs each candidate on host-chosen operands:
 *   op 0 / 1 / 2  asm vadd / vsub / vmpy (.sf = .sf op .sf), IEEE form
 *   op 3 / 4 / 5  Q6_Vsf_vadd / vsub / vmpy (the kernel's form)
 *   op 6          the scalar a / b the vector quantizer divides with
 *   op 7          the scalar sffma a * b + 1.0 (the router's chains)
 * Built on its own with -mhvx-ieee-fp (test/htp/build.sh), which the asm
 * form requires. No model path calls it.
 */

#include <string.h>

#include <AEEStdErr.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <remote.h>

#include "nntr_hvx.h"
#include "nntr_hvx_session.h"

#if defined(__hexagon__)
static HVX_Vector asm_vadd(HVX_Vector a, HVX_Vector b) {
  HVX_Vector r;
  __asm__("%0.sf = vadd(%1.sf,%2.sf)" : "=v"(r) : "v"(a), "v"(b));
  return r;
}
static HVX_Vector asm_vsub(HVX_Vector a, HVX_Vector b) {
  HVX_Vector r;
  __asm__("%0.sf = vsub(%1.sf,%2.sf)" : "=v"(r) : "v"(a), "v"(b));
  return r;
}
static HVX_Vector asm_vmpy(HVX_Vector a, HVX_Vector b) {
  HVX_Vector r;
  __asm__("%0.sf = vmpy(%1.sf,%2.sf)" : "=v"(r) : "v"(a), "v"(b));
  return r;
}
#else /* the in-process host build: the emulation's IEEE op */
#define asm_vadd Q6_Vsf_vadd_VsfVsf
#define asm_vsub Q6_Vsf_vsub_VsfVsf
#define asm_vmpy Q6_Vsf_vmpy_VsfVsf
#endif

int nntr_hvx_sf_probe(remote_handle64 handle, uint32 op, const float *a,
                      int aLen, const float *b, int bLen, float *o, int oLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
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
    case 0:
      r = asm_vadd(x, y);
      break;
    case 1:
      r = asm_vsub(x, y);
      break;
    case 2:
      r = asm_vmpy(x, y);
      break;
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
