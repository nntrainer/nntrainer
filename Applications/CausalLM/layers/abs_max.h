// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   abs_max.h
 * @date   8 October 2026
 * @brief  max |x| over fp16 bit patterns and f32, min / max over f32, and
 *         the f32 <-> u16 conversions of the int8 attention path's
 *         fixed-scale calibration and its a16 Q / output
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The calibration ran these as scalar loops: 6.7-7.4 ms a 1024-row layer
 * on device (doc 57 section 9.17). The NEON path gives the same answer bit
 * for bit: max is order-free, and each step is std::max(m, x)'s own
 * "x if m < x else m" as a compare and select, so a NaN x keeps m. (vmaxnm
 * does not: a signaling NaN turns it into NaN.)
 */
#ifndef __CAUSALLM_ABS_MAX_H__
#define __CAUSALLM_ABS_MAX_H__

#include <algorithm>
#include <cmath>
#include <cstdint>

#include <fp16.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace causallm {

#if defined(__aarch64__)
/** @brief std::max(m, x) per lane: x where m < x, else m. */
inline float32x4_t max_keep(float32x4_t m, float32x4_t x) {
  return vbslq_f32(vcltq_f32(m, x), x, m);
}
#endif

/** @brief max(m, max_i |x[i]|) over @a n fp16 bit patterns. */
inline float abs_max_f16(const uint16_t *x, unsigned int n, float m) {
  unsigned int i = 0;
#if defined(__aarch64__)
  float32x4_t acc = vdupq_n_f32(m);
  for (; i + 8 <= n; i += 8) {
    const float16x8_t h = vreinterpretq_f16_u16(vld1q_u16(x + i));
    acc = max_keep(acc, vabsq_f32(vcvt_f32_f16(vget_low_f16(h))));
    acc = max_keep(acc, vabsq_f32(vcvt_high_f32_f16(h)));
  }
  m = vmaxnmvq_f32(acc);
#endif
  for (; i < n; ++i)
    m = std::max(m, std::fabs(nntrainer::compute_fp16_to_fp32(x[i])));
  return m;
}

/** @brief m[i] = max(m[i], |x[i]|) over @a n fp16 bit patterns. */
inline void abs_max_f16_lanes(const uint16_t *x, unsigned int n, float *m) {
  unsigned int i = 0;
#if defined(__aarch64__)
  for (; i + 8 <= n; i += 8) {
    const float16x8_t h = vreinterpretq_f16_u16(vld1q_u16(x + i));
    vst1q_f32(m + i, max_keep(vld1q_f32(m + i),
                              vabsq_f32(vcvt_f32_f16(vget_low_f16(h)))));
    vst1q_f32(m + i + 4,
              max_keep(vld1q_f32(m + i + 4), vabsq_f32(vcvt_high_f32_f16(h))));
  }
#endif
  for (; i < n; ++i)
    m[i] = std::max(m[i], std::fabs(nntrainer::compute_fp16_to_fp32(x[i])));
}

/** @brief max(m, max_i |x[i]|) over @a n floats. */
inline float abs_max_f32(const float *x, unsigned int n, float m) {
  unsigned int i = 0;
#if defined(__aarch64__)
  float32x4_t acc = vdupq_n_f32(m);
  for (; i + 4 <= n; i += 4)
    acc = max_keep(acc, vabsq_f32(vld1q_f32(x + i)));
  m = vmaxnmvq_f32(acc);
#endif
  for (; i < n; ++i)
    m = std::max(m, std::fabs(x[i]));
  return m;
}

#if defined(__aarch64__)
/** @brief std::min(m, x) per lane: x where x < m, else m. */
inline float32x4_t min_keep(float32x4_t m, float32x4_t x) {
  return vbslq_f32(vcltq_f32(x, m), x, m);
}
#endif

/** @brief lo = min(lo, x[i]), hi = max(hi, x[i]) over @a n floats, as
 *  std::min / std::max do (order-free for finite x). */
inline void min_max_f32(const float *x, unsigned int n, float &lo, float &hi) {
  unsigned int i = 0;
#if defined(__aarch64__)
  float32x4_t vlo = vdupq_n_f32(lo), vhi = vdupq_n_f32(hi);
  for (; i + 4 <= n; i += 4) {
    const float32x4_t v = vld1q_f32(x + i);
    vlo = min_keep(vlo, v);
    vhi = max_keep(vhi, v);
  }
  lo = vminnmvq_f32(vlo);
  hi = vmaxnmvq_f32(vhi);
#endif
  for (; i < n; ++i) {
    lo = std::min(lo, x[i]);
    hi = std::max(hi, x[i]);
  }
}

/** @brief dst[i] = u16(clamp(src[i] * inv + zp + 0.5, 0, 65535)), the a16
 *  path's Q quantization (truncation after the + 0.5 rounds to nearest). */
inline void quant_u16_f32(const float *src, unsigned int n, float inv, float zp,
                          uint16_t *dst) {
  unsigned int i = 0;
#if defined(__aarch64__)
  const float32x4_t vinv = vdupq_n_f32(inv), vzp = vdupq_n_f32(zp + 0.5f);
  const float32x4_t lo = vdupq_n_f32(0.0f), hi = vdupq_n_f32(65535.0f);
  for (; i + 8 <= n; i += 8) {
    float32x4_t a = vaddq_f32(vmulq_f32(vld1q_f32(src + i), vinv), vzp);
    float32x4_t b = vaddq_f32(vmulq_f32(vld1q_f32(src + i + 4), vinv), vzp);
    a = vbslq_f32(vcltq_f32(a, lo), lo, vbslq_f32(vcgtq_f32(a, hi), hi, a));
    b = vbslq_f32(vcltq_f32(b, lo), lo, vbslq_f32(vcgtq_f32(b, hi), hi, b));
    vst1q_u16(dst + i, vcombine_u16(vmovn_u32(vcvtq_u32_f32(a)),
                                    vmovn_u32(vcvtq_u32_f32(b))));
  }
#endif
  for (; i < n; ++i) {
    float v = src[i] * inv + (zp + 0.5f);
    v = v < 0.0f ? 0.0f : v > 65535.0f ? 65535.0f : v;
    dst[i] = static_cast<uint16_t>(v);
  }
}

/** @brief dst[i] = (src[i] - zp) * scale, the a16 output's dequantization. */
inline void dequant_u16_f32(const uint16_t *src, unsigned int n, float scale,
                            float zp, float *dst) {
  unsigned int i = 0;
#if defined(__aarch64__)
  const float32x4_t vs = vdupq_n_f32(scale), vzp = vdupq_n_f32(zp);
  for (; i + 8 <= n; i += 8) {
    const uint16x8_t u = vld1q_u16(src + i);
    vst1q_f32(
      dst + i,
      vmulq_f32(vsubq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(u))), vzp), vs));
    vst1q_f32(dst + i + 4,
              vmulq_f32(vsubq_f32(vcvtq_f32_u32(vmovl_high_u16(u)), vzp), vs));
  }
#endif
  for (; i < n; ++i)
    dst[i] = (static_cast<float>(src[i]) - zp) * scale;
}

} // namespace causallm

#endif // __CAUSALLM_ABS_MAX_H__
