// SPDX-License-Identifier: Apache-2.0
/**
 * @file	hvx-f16-math.h
 * @date	18 August 2026
 * @brief	fp16 widening dot / sum-of-squares and scale-multiply HVX
 *		helpers, shared by RMSNORM (whole-row and QK-Norm) and ATTN
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#ifndef HVX_F16_MATH_H
#define HVX_F16_MATH_H

#include <stdint.h>

#include "hvx-base.h"
#include "hvx-types.h"

/* Horizontal sum of an IEEE fp32 vector pair. */
static inline float hvx_sum_sf_pair(HVX_VectorPair acc) {
  float __attribute__((aligned(VLEN))) buf[VLEN_FP32];
  float sum = 0.f;
  hvx_vec_store_a(buf, VLEN, Q6_V_lo_W(acc));
  for (uint32_t i = 0; i < VLEN_FP32; ++i)
    sum += buf[i];
  hvx_vec_store_a(buf, VLEN, Q6_V_hi_W(acc));
  for (uint32_t i = 0; i < VLEN_FP32; ++i)
    sum += buf[i];
  return sum;
}

/* Dot product sum(a_i * b_i) in fp32 (one Vhf vector is VLEN_FP16=64
 * halves; the widened product pair holds 32+32 fp32 lanes). n must be a
 * multiple of 64 (validator guarantees hidden%64==0 and head_dim=128) and
 * a/b must be 128B aligned. Accumulates in IEEE sf through
 * hvx_vec_mpyacc_f32_f16 (native Wsf_vmpyacc on v79+): chaining
 * Q6_Vqf32_vadd_Vqf32Vqf32 across vectors returned garbage on v79 (silicon
 * and simulator) while passing on v75. */
static inline float hvx_dot_fp16(const __fp16 *a, const __fp16 *b, uint32_t n) {
  HVX_VectorPair acc = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vzero());
  for (uint32_t i = 0; i < n; i += VLEN_FP16)
    acc = hvx_vec_mpyacc_f32_f16(acc, hvx_vmem(a + i), hvx_vmem(b + i));
  return hvx_sum_sf_pair(acc);
}

/* Sum of x_i^2 in fp32: dot of x with itself. Same alignment/n rules. */
static inline float hvx_sumsq_fp16(const __fp16 *x, uint32_t n) {
  return hvx_dot_fp16(x, x, n);
}

/* y_i = (fp16)(x_i * r * g_i) computed in fp32: x*g exactly via
 * Wqf32_vmpy_VhfVhf, then * r (fp32 splat), narrowed to hf once. The
 * earlier two-step qf16 version (with r rounded to fp16) drifted ~1% on
 * qwen3's large gammas / massive activations versus the fp32 reference.
 * n must be a multiple of 64; x/g/y must be 128B aligned. */
static inline void hvx_scale_mul_fp16(const __fp16 *x, const __fp16 *g, float r,
                                      __fp16 *y, uint32_t n) {
  const HVX_Vector rv = hvx_vec_splat_f32(r);
  for (uint32_t i = 0; i < n; i += VLEN_FP16) {
    HVX_VectorPair xg = Q6_Wqf32_vmpy_VhfVhf(hvx_vmem(x + i), hvx_vmem(g + i));
    HVX_Vector lo =
      Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(Q6_V_lo_W(xg)), rv);
    HVX_Vector hi =
      Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(Q6_V_hi_W(xg)), rv);
    hvx_vmem(y + i) = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(hi, lo));
  }
}

#endif /* HVX_F16_MATH_H */
