// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hvx_softmax_f32.c
 * @date   05 Aug 2026
 * @brief  Row-wise f32 softmax on HVX
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 */

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include "hvx_convert.h"
#include "hvx_exp_f32.h"
#include "hvx_softmax_f32.h"
#include "hvx_softmax_util.h"

void hvx_softmax_rows_f32(const float *x, float *y, uint32_t m_first,
                          uint32_t m_last, uint32_t k, float scale) {
  const uint32_t nvec = k / LANES;
  const uint32_t nloe = k % LANES;
  const HVX_Vector vscale = hvx_splat_sf(scale);

  for (uint32_t m = m_first; m < m_last; ++m) {
    const float *xr = x + (size_t)m * k;
    float *yr = y + (size_t)m * k;

    // FastRPC buffers carry no vector alignment guarantee, so the
    // unaligned vector type is what keeps this from faulting.
    const HVX_UVector *vx = (const HVX_UVector *)xr;
    HVX_UVector *vy = (HVX_UVector *)yr;

    /** Pass 1: max of x*scale. Scaling here instead of using
       scale*max(x) is what keeps a negative scale correct -- a negative
       scale flips which element is the maximum. */
    HVX_Vector vmax = hvx_splat_sf(xr[0] * scale);
    for (uint32_t v = 0; v < nvec; ++v) {
      vmax = Q6_Vsf_vmax_VsfVsf(vmax, Q6_Vsf_vmpy_VsfVsf(vx[v], vscale));
    }
    if (nloe) {
      /* Pad with a real element of the row: it can never beat the max. */
      const HVX_Vector t = load_tail_sf(xr + nvec * LANES, nloe, xr[0]);
      vmax = Q6_Vsf_vmax_VsfVsf(vmax, Q6_Vsf_vmpy_VsfVsf(t, vscale));
    }
    const HVX_Vector vm = reduce_max_sf(vmax);

    /** Pass 2: y = exp(x*scale - max), summing as we go. The exp results
       go straight to y, so no scratch buffer is needed; pass 3 scales
       them in place. Reading vx[v] before writing vy[v] at the same index
       is what makes y == x safe.

       The sum accumulates in qf32 for the extra mantissa headroom: there
       are k/32 accumulation steps and the output tolerance is 1e-6.
       Adding two zeros is how a canonical qf32 zero is obtained. */
    HVX_Vector acc = Q6_Vqf32_vadd_VsfVsf(Q6_V_vzero(), Q6_V_vzero());
    for (uint32_t v = 0; v < nvec; ++v) {
      const HVX_Vector e =
        hvx_exp_sf(Q6_Vsf_vsub_VsfVsf(Q6_Vsf_vmpy_VsfVsf(vx[v], vscale), vm));
      vy[v] = e;
      acc = Q6_Vqf32_vadd_Vqf32Vsf(acc, e);
    }
    if (nloe) {
      const HVX_Vector t = load_tail_sf(xr + nvec * LANES, nloe, 0.0f);
      HVX_Vector e =
        hvx_exp_sf(Q6_Vsf_vsub_VsfVsf(Q6_Vsf_vmpy_VsfVsf(t, vscale), vm));
      /** The pad lanes hold exp(0*scale - max), not zero, so they have to
         be masked off before they reach the sum. Padding the input cannot
         fix this: no input value maps to an exact zero after exp. */
      e = Q6_V_vmux_QVV(Q6_Q_vsetq2_R((int)(nloe * 4u)), e, Q6_V_vzero());
      store_tail_sf(yr + nvec * LANES, nloe, e);
      acc = Q6_Vqf32_vadd_Vqf32Vsf(acc, e);
    }
    const float sum = lane0_sf(reduce_sum_sf(Q6_Vsf_equals_Vqf32(acc)));

    /** Pass 3: normalize in place. sum is one scalar replicated across the
       vector, so a scalar reciprocal is both exact and cheaper than a
       vector Newton iteration. sum is at least 1 in exact arithmetic --
       the maximum element contributes exp(0) -- so the guard is only
       there for a row of NaNs. */
    const HVX_Vector vr = hvx_splat_sf(sum > 0.0f ? 1.0f / sum : 1.0f);
    for (uint32_t v = 0; v < nvec; ++v) {
      vy[v] = Q6_Vsf_vmpy_VsfVsf(vy[v], vr);
    }

    if (nloe) {
      const HVX_Vector t = load_tail_sf(yr + nvec * LANES, nloe, 0.0f);
      store_tail_sf(yr + nvec * LANES, nloe, Q6_Vsf_vmpy_VsfVsf(t, vr));
    }
  }
}
