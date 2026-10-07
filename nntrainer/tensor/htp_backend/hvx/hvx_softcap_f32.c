// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hvx_softcap_f32.c
 * @date   7 October 2026
 * @brief  Logit softcap over a row of f32 on HVX, in place
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "hvx_softcap_f32.h"

#include <math.h>

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include "hvx_convert.h"
#include "hvx_swiglu_det.h"

/** @brief f32 lanes per HVX vector at 128B. */
#define LANES 32u

int hvx_softcap_f32(float *y, uint32_t n, float cap) {
  if (!y || !(cap > 0.0f)) {
    return -1;
  }
  /* tanh(|u|) = (1 - e) / (1 + e), e = exp(-2|u|), u = y / cap, the sign
     put back after: the exp and the reciprocal are the GLU epilogue's own
     (hvx_swiglu_det.h), e in (0, 1] keeps the reciprocal's operand in
     (1, 2]. |u| is clamped at 20, where tanh is 1 in f32 already. */
  const HVX_Vector k = hvx_splat_sf(-2.0f / cap);
  const HVX_Vector lo = hvx_splat_sf(-40.0f);
  const HVX_Vector one = hvx_splat_sf(1.0f);
  const HVX_Vector vcap = hvx_splat_sf(cap);
  const HVX_Vector sign_bit = Q6_V_vsplat_R((int)0x80000000u);
  HVX_UVector *v = (HVX_UVector *)y;
  const uint32_t nvec = n / LANES;
  for (uint32_t i = 0; i < nvec; ++i) {
    const HVX_Vector x = v[i];
    const HVX_Vector sign = Q6_V_vand_VV(x, sign_bit);
    const HVX_Vector mag = Q6_V_vand_VV(x, Q6_V_vsplat_R(0x7FFFFFFF));
    const HVX_Vector a = Q6_Vsf_vmax_VsfVsf(Q6_Vsf_vmpy_VsfVsf(mag, k), lo);
    const HVX_Vector e = hvx_exp_det_sf(a);
    const HVX_Vector t =
      Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vsub_VsfVsf(one, e),
                         hvx_recip_det_sf(Q6_Vsf_vadd_VsfVsf(one, e)));
    v[i] = Q6_V_vor_VV(Q6_Vsf_vmpy_VsfVsf(t, vcap), sign);
  }
  /* ponytail: a scalar tail; the vocabularies here are multiples of 32 */
  for (uint32_t i = nvec * LANES; i < n; ++i) {
    y[i] = cap * tanhf(y[i] / cap);
  }
  return 0;
}
