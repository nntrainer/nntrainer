// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   abs_max.h
 * @date   8 October 2026
 * @brief  max |x| over fp16 bit patterns and f32, for the int8 attention
 *         path's fixed-scale calibration
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

} // namespace causallm

#endif // __CAUSALLM_ABS_MAX_H__
