// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file   hvx_kv_quant.c
 * @date   28 Sep 2026
 * @brief  HVX quantization of one appended KV row (per head) for hexkl_kv_q
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "hvx_kv_quant.h"

#include <string.h>

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include "hvx_attn_softmax_f16.h"
#include "hvx_convert.h"
#include "hvx_tile_f16.h"

/** @brief Largest head_dim: 8 f32 vectors. */
#define MAX_VECS 8u

/**
 * @brief 64 fp16 (natural order) -> two f32 vectors in natural order: the
 *        widening multiply by 1 splits by parity, the word shuffle puts the
 *        halves back in order.
 */
static inline void widen64(HVX_Vector hf, HVX_Vector *lo, HVX_Vector *hi) {
  const HVX_Vector one_hf = Q6_Vh_vsplat_R((int)HVX_ATTN_HF_ONE);
  const HVX_VectorPair w = Q6_Wqf32_vmpy_VhfVhf(hf, one_hf);
  const HVX_VectorPair nat = Q6_W_vshuff_VVR(
    Q6_Vsf_equals_Vqf32(Q6_V_hi_W(w)), Q6_Vsf_equals_Vqf32(Q6_V_lo_W(w)), -4);
  *lo = Q6_V_lo_W(nat);
  *hi = Q6_V_hi_W(nat);
}

/** @brief hd fp16 -> hd/32 f32 vectors, natural order. */
static inline void load_row_f32(const uint16_t *x_hf, uint32_t hd,
                                HVX_Vector *v) {
  for (uint32_t i = 0; i < hd / 64u; ++i) {
    widen64(hvx_tile_load_u(x_hf + 64u * i), &v[2u * i], &v[2u * i + 1u]);
  }
  if (hd % 64u) {
    // 32 trailing values: the half vector's upper lanes are zero.
    HVX_Vector h = Q6_V_vzero();
    memcpy(&h, x_hf + (hd & ~63u), 64u);
    HVX_Vector hi;
    widen64(h, &v[hd / 32u - 1u], &hi);
  }
}

/** @brief Lane 0 of max over the given vectors of |x|. */
static inline float absmax_of(const HVX_Vector *v, uint32_t n) {
  const HVX_Vector mask = Q6_V_vsplat_R(0x7FFFFFFF);
  HVX_Vector m = Q6_V_vand_VV(v[0], mask);
  for (uint32_t i = 1; i < n; ++i) {
    m = Q6_Vsf_vmax_VsfVsf(m, Q6_V_vand_VV(v[i], mask));
  }
  return hvx_attn_lane0_f32(hvx_attn_max32_sf(m));
}

/** @brief rne(x * inv) clamped to [-fq, fq], as int32 lanes. */
static inline HVX_Vector quant_vec(HVX_Vector x, HVX_Vector inv, HVX_Vector lo,
                                   HVX_Vector hi) {
  HVX_Vector q = hvx_sf_to_w_rne(Q6_Vsf_vmpy_VsfVsf(x, inv));
  q = Q6_Vw_vmax_VwVw(q, lo);
  return Q6_Vw_vmin_VwVw(q, hi);
}

/** @brief hd int32 lanes (hd/32 vectors) -> hd bytes at @a q. */
static inline void pack_bytes(const HVX_Vector *w, uint32_t hd, int8_t *q) {
  uint32_t i = 0;
  for (; i + 4u <= hd / 32u; i += 4u) {
    const HVX_Vector h01 = Q6_Vh_vpack_VwVw_sat(w[i + 1u], w[i]);
    const HVX_Vector h23 = Q6_Vh_vpack_VwVw_sat(w[i + 3u], w[i + 2u]);
    hvx_tile_store_u(q + 32u * i, Q6_Vb_vpack_VhVh_sat(h23, h01));
  }
  for (; i < hd / 32u; ++i) {
    const HVX_Vector h = Q6_Vh_vpack_VwVw_sat(w[i], w[i]);
    const HVX_Vector b = Q6_Vb_vpack_VhVh_sat(h, h);
    uint8_t tmp[128];
    hvx_tile_store_u(tmp, b);
    memcpy(q + 32u * i, tmp, 32u);
  }
}

void hvx_kv_quant_k_row(const uint16_t *x_hf, uint32_t hd, int32_t qmax,
                        int8_t *q, float *scale, int32_t *colsum) {
  HVX_Vector v[MAX_VECS];
  const uint32_t n = hd / 32u;
  load_row_f32(x_hf, hd, v);
  const float amax = absmax_of(v, n);
  if (amax == 0.0f) {
    memset(q, 0, hd);
    *scale = 1.0f;
    *colsum = 0;
    return;
  }
  const float fq = (float)qmax;
  const HVX_Vector inv = hvx_splat_sf(fq / amax);
  const HVX_Vector lo = Q6_V_vsplat_R(-qmax), hi = Q6_V_vsplat_R(qmax);
  HVX_Vector w[MAX_VECS];
  HVX_Vector sum = Q6_V_vzero();
  for (uint32_t i = 0; i < n; ++i) {
    w[i] = quant_vec(v[i], inv, lo, hi);
    sum = Q6_Vw_vadd_VwVw(sum, w[i]);
  }
  for (int rot = 4; rot <= 64; rot <<= 1) {
    sum = Q6_Vw_vadd_VwVw(sum, Q6_V_vror_VR(sum, rot));
  }
  pack_bytes(w, hd, q);
  *scale = amax / fq;
  *colsum = (int32_t)hvx_attn_word0(sum);
}

void hvx_kv_quant_v_row(const uint16_t *x_hf, uint32_t hd, int32_t qmax,
                        int8_t *q, float *scales) {
  HVX_Vector v[MAX_VECS];
  const uint32_t n = hd / 32u;
  load_row_f32(x_hf, hd, v);
  const float fq = (float)qmax;
  const HVX_Vector lo = Q6_V_vsplat_R(-qmax), hi = Q6_V_vsplat_R(qmax);
  HVX_Vector w[MAX_VECS];
  for (uint32_t g = 0; g < n; ++g) {
    const float amax = absmax_of(&v[g], 1);
    if (amax == 0.0f) {
      w[g] = Q6_V_vzero();
      scales[g] = 1.0f;
      continue;
    }
    w[g] = quant_vec(v[g], hvx_splat_sf(fq / amax), lo, hi);
    scales[g] = amax / fq;
  }
  pack_bytes(w, hd, q);
}
