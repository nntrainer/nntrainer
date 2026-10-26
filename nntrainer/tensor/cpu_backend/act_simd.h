// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   act_simd.h
 * @date   22 July 2026
 * @brief  SIMD helpers for activation math and int8 activation quantization.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NaN or Inf
 *
 * @details Shared vectorized primitives used by the W8A8 inference path:
 * an exact-precision NEON expf (Cephes polynomial, ~1e-6 rel error, NOT a
 * LUT approximation), an in-place SiLU built on it, the affine int8
 * activation quantizer q = round((x + off) * inv) - 128, and an |x| max
 * reduction. Every NEON path is bit-identical to its scalar tail (FCVTAS ==
 * std::round, FRINTA rounding), so results do not depend on vector width.
 * Factored here (cpu_backend) so compute layers (conv2d) and application
 * custom layers (rtmcc_head) share one implementation instead of local
 * copies.
 */

#ifndef __NNTR_ACT_SIMD_H__
#define __NNTR_ACT_SIMD_H__

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace nntrainer {

#if defined(__ARM_NEON)
/**
 * @brief Vectorized expf (Cephes polynomial, ~1e-6 rel error).
 *
 * Used to vectorize FP32 sigmoid/SiLU epilogues, which otherwise run a
 * scalar std::exp per element.
 */
static inline float32x4_t nntr_vexpq_f32(float32x4_t x) {
  // Just below 127.5 / log2(e): at the exact bound the exponent rounds up
  // to 128 and 2^n becomes Inf although exp(x) is still finite.
  const float32x4_t hi = vdupq_n_f32(88.376f);
  const float32x4_t lo = vdupq_n_f32(-88.3762626647949f);
  x = vminq_f32(vmaxq_f32(x, lo), hi);
  const float32x4_t log2e = vdupq_n_f32(1.44269504088896341f);
  const float32x4_t c0 = vdupq_n_f32(0.693359375f);
  const float32x4_t c1 = vdupq_n_f32(-2.12194440e-4f);
  const float32x4_t one = vdupq_n_f32(1.0f);
  float32x4_t fx = vmlaq_f32(vdupq_n_f32(0.5f), x, log2e);
  int32x4_t emm0 = vcvtq_s32_f32(fx);
  float32x4_t tmp = vcvtq_f32_s32(emm0);
  uint32x4_t mask = vcgtq_f32(tmp, fx);
  fx = vsubq_f32(
    tmp, vreinterpretq_f32_u32(vandq_u32(mask, vreinterpretq_u32_f32(one))));
  emm0 = vcvtq_s32_f32(fx);
  x = vmlsq_f32(x, fx, c0);
  x = vmlsq_f32(x, fx, c1);
  const float32x4_t p0 = vdupq_n_f32(1.9875691500E-4f);
  const float32x4_t p1 = vdupq_n_f32(1.3981999507E-3f);
  const float32x4_t p2 = vdupq_n_f32(8.3334519073E-3f);
  const float32x4_t p3 = vdupq_n_f32(4.1665795894E-2f);
  const float32x4_t p4 = vdupq_n_f32(1.6666665459E-1f);
  const float32x4_t p5 = vdupq_n_f32(5.0000001201E-1f);
  float32x4_t z = vmulq_f32(x, x);
  float32x4_t y = vmlaq_f32(p1, p0, x);
  y = vmlaq_f32(p2, y, x);
  y = vmlaq_f32(p3, y, x);
  y = vmlaq_f32(p4, y, x);
  y = vmlaq_f32(p5, y, x);
  y = vmlaq_f32(x, y, z);
  y = vaddq_f32(y, one);
  int32x4_t pow2n = vshlq_n_s32(vaddq_s32(emm0, vdupq_n_s32(0x7f)), 23);
  return vmulq_f32(y, vreinterpretq_f32_s32(pow2n));
}
#endif

/**
 * @brief SiLU(x) = x / (1 + exp(-x)) in place over n contiguous floats.
 *
 * NEON path: nntr_vexpq_f32 + two Newton-Raphson reciprocal refinements
 * (validated 1.9e-6 max abs error vs the double-exp scalar reference over
 * [-30, 30]); scalar tail / non-NEON path uses the exact float std::exp.
 */
static inline void nntr_silu_inplace_f32(float *p, size_t n) {
  size_t i = 0;
#if defined(__ARM_NEON)
  const float32x4_t one = vdupq_n_f32(1.0f);
  for (; i + 4 <= n; i += 4) {
    float32x4_t v = vld1q_f32(p + i);
    float32x4_t d = vaddq_f32(one, nntr_vexpq_f32(vnegq_f32(v)));
    float32x4_t r = vrecpeq_f32(d);
    r = vmulq_f32(r, vrecpsq_f32(d, r));
    r = vmulq_f32(r, vrecpsq_f32(d, r));
    vst1q_f32(p + i, vmulq_f32(v, r));
  }
#endif
  for (; i < n; ++i)
    p[i] = p[i] / (1.0f + std::exp(-p[i]));
}

/**
 * @brief Affine int8 quantize of a contiguous FP32 run:
 * q = round((x + off) * inv) - 128, saturated to [-128, 127].
 *
 * The NEON path uses FCVTAS (round-to-nearest, ties away from zero --
 * exactly std::round), a saturating -128 and saturating narrows, so it is
 * bit-identical to the scalar tail. Callers parallelize by chunking.
 */
static inline void nntr_quantize_affine_i8(const float *src, int8_t *dst,
                                           size_t n, float inv, float off) {
  size_t i = 0;
  // FCVTAS (vcvtaq) is AArch64-only; armv7 NEON takes the scalar loop.
#if defined(__ARM_NEON) && defined(__aarch64__)
  const float32x4_t vinv = vdupq_n_f32(inv);
  const float32x4_t voff = vdupq_n_f32(off);
  const int32x4_t vm128 = vdupq_n_s32(-128);
  for (; i + 15 < n; i += 16) {
    const int32x4_t q0 = vqaddq_s32(
      vcvtaq_s32_f32(vmulq_f32(vaddq_f32(vld1q_f32(src + i + 0), voff), vinv)),
      vm128);
    const int32x4_t q1 = vqaddq_s32(
      vcvtaq_s32_f32(vmulq_f32(vaddq_f32(vld1q_f32(src + i + 4), voff), vinv)),
      vm128);
    const int32x4_t q2 = vqaddq_s32(
      vcvtaq_s32_f32(vmulq_f32(vaddq_f32(vld1q_f32(src + i + 8), voff), vinv)),
      vm128);
    const int32x4_t q3 = vqaddq_s32(
      vcvtaq_s32_f32(vmulq_f32(vaddq_f32(vld1q_f32(src + i + 12), voff), vinv)),
      vm128);
    const int16x8_t p0 = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
    const int16x8_t p1 = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
    vst1q_s8(dst + i, vcombine_s8(vqmovn_s16(p0), vqmovn_s16(p1)));
  }
#endif
  for (; i < n; ++i) {
    const float q = std::round((src[i] + off) * inv) - 128.f;
    dst[i] = (int8_t)std::max(-128.f, std::min(127.f, q));
  }
}

/**
 * @brief max(|x|) over a contiguous FP32 run (NEON + scalar tail); callers
 * chunk-parallelize.
 */
static inline float nntr_absmax_f32(const float *src, size_t n) {
  float am = 0.f;
  size_t i = 0;
  // The across-vector max (vmaxvq) is AArch64-only.
#if defined(__ARM_NEON) && defined(__aarch64__)
  float32x4_t vm = vdupq_n_f32(0.f);
  for (; i + 3 < n; i += 4)
    vm = vmaxq_f32(vm, vabsq_f32(vld1q_f32(src + i)));
  am = vmaxvq_f32(vm);
#endif
  for (; i < n; ++i)
    am = std::max(am, std::fabs(src[i]));
  return am;
}

} // namespace nntrainer

#endif // __NNTR_ACT_SIMD_H__
