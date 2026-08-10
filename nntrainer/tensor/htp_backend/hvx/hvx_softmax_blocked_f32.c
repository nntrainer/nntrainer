// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hvx_softmax_blocked_f32.c
 * @date   07 Aug 2026
 * @brief  Blocked masked softmax on HVX for the fused attention kernel
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * See hvx_softmax_blocked_f32.h for the contract and for why each of the four
 * differences from hvx_softmax_rows_f32 exists. The host-side executable
 * specification this must agree with is mha_softmax_blocked_ref in
 * test/unittest/mha_htp_host_model.cpp.
 */

#include <stddef.h>
#include <string.h>

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include "hvx_convert.h"
#include "hvx_exp_f32.h"
#include "hvx_softmax_blocked_f32.h"
#include "hvx_softmax_util.h"

/**
 * @brief Folds one contiguous slice into the running max of x*scale.
 *
 * Scaling before the max, rather than scaling max(x), is what keeps a
 * negative scale correct -- it flips which element is the maximum. Carried
 * over from hvx_softmax_rows_f32 deliberately.
 *
 * @param pad a real element of the slice: it can never beat the max, so the
 *            tail lanes it fills are harmless.
 */
static inline HVX_Vector slice_max_sf(const float *p, uint32_t len,
                                      HVX_Vector vscale, HVX_Vector vmax,
                                      float pad) {
  const uint32_t nvec = len / LANES;
  const uint32_t nloe = len % LANES;
  /** FastRPC and VTCM buffers carry no vector alignment guarantee, and a
   * slice start is offset by lo anyway, so the unaligned vector type is what
   * keeps this from faulting. */
  const HVX_UVector *v = (const HVX_UVector *)p;

  for (uint32_t i = 0; i < nvec; ++i) {
    vmax = Q6_Vsf_vmax_VsfVsf(vmax, Q6_Vsf_vmpy_VsfVsf(v[i], vscale));
  }
  if (nloe) {
    const HVX_Vector t = load_tail_sf(p + nvec * LANES, nloe, pad);
    vmax = Q6_Vsf_vmax_VsfVsf(vmax, Q6_Vsf_vmpy_VsfVsf(t, vscale));
  }
  return vmax;
}

/**
 * @brief Writes exp(x*scale - vm) over one contiguous slice, in place, and
 *        folds it into the running qf32 sum.
 *
 * The sum accumulates in qf32 for the extra mantissa headroom: a band can be
 * kv_len/LANES accumulation steps long and the tolerance against the host
 * model is 1e-6. Same reasoning, and same idiom, as hvx_softmax_rows_f32.
 */
static inline HVX_Vector slice_exp_sum_sf(float *p, uint32_t len,
                                          HVX_Vector vscale, HVX_Vector vm,
                                          HVX_Vector acc, HVX_Vector *emax) {
  const uint32_t nvec = len / LANES;
  const uint32_t nloe = len % LANES;
  HVX_UVector *v = (HVX_UVector *)p;

  for (uint32_t i = 0; i < nvec; ++i) {
    const HVX_Vector e =
      hvx_exp_sf(Q6_Vsf_vsub_VsfVsf(Q6_Vsf_vmpy_VsfVsf(v[i], vscale), vm));
    v[i] = e;
    acc = Q6_Vqf32_vadd_Vqf32Vsf(acc, e);
    *emax = Q6_Vsf_vmax_VsfVsf(*emax, e);
  }
  if (nloe) {
    const HVX_Vector t = load_tail_sf(p + nvec * LANES, nloe, 0.0f);
    HVX_Vector e =
      hvx_exp_sf(Q6_Vsf_vsub_VsfVsf(Q6_Vsf_vmpy_VsfVsf(t, vscale), vm));
    /** The pad lanes hold exp(0*scale - vm), not zero, so they have to be
     * masked off before they reach the sum. Padding the input cannot fix
     * this: no input value maps to an exact zero after exp. The same mask
     * keeps them out of emax: a pad lane's exp could exceed every real
     * one. */
    e = Q6_V_vmux_QVV(Q6_Q_vsetq2_R((int)(nloe * 4u)), e, Q6_V_vzero());
    store_tail_sf(p + nvec * LANES, nloe, e);
    acc = Q6_Vqf32_vadd_Vqf32Vsf(acc, e);
    *emax = Q6_Vsf_vmax_VsfVsf(*emax, e);
  }
  return acc;
}

void hvx_softmax_blocked_f32(float *const *seg, uint32_t n_seg, uint32_t T,
                             uint32_t m_first, uint32_t m_last, uint32_t M,
                             float scale, const uint32_t *begin,
                             const uint32_t *end, const float *sink,
                             float *l_out, float *bm_out) {
  const HVX_Vector vscale = hvx_splat_sf(scale);
  const uint32_t kv_total = n_seg * T;

  /* M is the bm_out row stride; within a segment the row stride is T. */

  for (uint32_t m = m_first; m < m_last; ++m) {
    uint32_t b = begin[m];
    uint32_t e = end[m];

    if (b > kv_total) {
      b = kv_total;
    }
    if (e > kv_total) {
      e = kv_total;
    }

    /** Zero every masked position first, so a fully masked row and the
     * head/tail of a partially masked block are both already correct if the
     * passes below skip them. PHASE C reads these lanes as P, so they must be
     * exactly 0.0f rather than merely ignored -- there is no second mask
     * downstream. */
    for (uint32_t j = 0; j < n_seg; ++j) {
      uint32_t lo, hi;
      hvx_softmax_seg_range(j, T, b, e, &lo, &hi);
      float *row = seg[j] + (size_t)m * T;
      if (lo) {
        memset(row, 0, (size_t)lo * sizeof(float));
      }
      if (hi < T) {
        memset(row + hi, 0, (size_t)(T - hi) * sizeof(float));
      }
    }

    /** PASS 1 -- maximum over the valid lanes only. The first valid element
     * seeds the running max and doubles as the tail pad, exactly as the dense
     * kernel uses xr[0]: a real element of the set being reduced can never
     * beat its own maximum. */
    int any = 0;
    float first = 0.0f;
    for (uint32_t j = 0; j < n_seg && !any; ++j) {
      uint32_t lo, hi;
      hvx_softmax_seg_range(j, T, b, e, &lo, &hi);
      if (hi > lo) {
        first = seg[j][(size_t)m * T + lo];
        any = 1;
      }
    }

    if (!any) {
      /** No valid position: everything is already zeroed, and the denominator
       * is the sink alone (or nothing, which PHASE C turns into a zero row
       * rather than a divide by zero). */
      l_out[m] = (sink != NULL) ? 1.0f : 0.0f;
      if (bm_out != NULL) {
        for (uint32_t j = 0; j < n_seg; ++j) {
          bm_out[(size_t)j * M + m] = 0.0f;
        }
      }
      continue;
    }

    HVX_Vector vmax = hvx_splat_sf(first * scale);
    for (uint32_t j = 0; j < n_seg; ++j) {
      uint32_t lo, hi;
      hvx_softmax_seg_range(j, T, b, e, &lo, &hi);
      if (hi > lo) {
        vmax = slice_max_sf(seg[j] + (size_t)m * T + lo, hi - lo, vscale, vmax,
                            first);
      }
    }
    const HVX_Vector vm = reduce_max_sf(vmax);

    /** PASS 2 -- unnormalized exp over the same lane set, ascending in kv
     * position. Walking segments in order and slices left to right is what
     * makes the block split invisible: the same addends in the same order
     * whatever T is. */
    HVX_Vector acc = Q6_Vqf32_vadd_VsfVsf(Q6_V_vzero(), Q6_V_vzero());
    for (uint32_t j = 0; j < n_seg; ++j) {
      uint32_t lo, hi;
      hvx_softmax_seg_range(j, T, b, e, &lo, &hi);
      /** Seeded with +0.0f, so the reduced value is max(stored exps, 0) --
       * which is EXACTLY the rmax hvx_quant_rows_u8_params would find
       * scanning this block row afterwards: the positions outside [lo, hi)
       * hold literal zeros, and max is order-independent. That equality is
       * the whole contract of bm_out; see the header. */
      HVX_Vector emax = Q6_V_vzero();
      if (hi > lo) {
        acc = slice_exp_sum_sf(seg[j] + (size_t)m * T + lo, hi - lo, vscale, vm,
                               acc, &emax);
      }
      if (bm_out != NULL) {
        bm_out[(size_t)j * M + m] = lane0_sf(reduce_max_sf(emax));
      }
    }
    float l = lane0_sf(reduce_sum_sf(Q6_Vsf_equals_Vqf32(acc)));

    /** The sink is one scalar logit, and it enters the denominator only. Run
     * it through hvx_exp_sf rather than a scalar expf so it cannot drift from
     * the lanes: same polynomial, same rounding. It is outside the max, which
     * is what keeps the row maximum of seg exactly 1.0f -- so this exp can be
     * larger than 1, and that is correct. */
    if (sink != NULL) {
      l += lane0_sf(hvx_exp_sf(Q6_Vsf_vsub_VsfVsf(hvx_splat_sf(sink[m]), vm)));
    }

    l_out[m] = l;
  }
}
