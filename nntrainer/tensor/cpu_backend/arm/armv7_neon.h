// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   cpu_backend.h
 * @date   16 August 2024
 * @see    https://github.com/nntrainer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Conditional header file to support unsupported intrinsics on armv7l
 *
 */
#ifndef ARMV7_NEON_
#define ARMV7_NEON_

#include <arm_neon.h>
#include <cmath>

/**
 * @brief macro for vfmaq_n_f32
 *
 */
#define vfmaq_n_f32(a, b, n) vaddq_f32(a, vmulq_f32(b, vmovq_n_f32(n)))

/**
 * @brief vdivq_f32 macro
 *
 * @param a a for a / b
 * @param b b for a / b
 * @return float32x4_t
 */
static inline float32x4_t vdivq_f32(float32x4_t a, float32x4_t b) {
  float32x4_t ret;
  for (unsigned int i = 0; i < 4; ++i) {
    ret[i] = a[i] / b[i];
  }
  return ret;
}

/**
 * @brief vsqrtq_f32 macro
 *
 * @param a input vector
 * @return float32x4_t
 */
static inline float32x4_t vsqrtq_f32(float32x4_t a) {
  float32x4_t ret;
  for (unsigned int i = 0; i < 4; ++i) {
    ret[i] = std::sqrt(a[i]);
  }
  return ret;
}

/**
 * @brief vmaxvq_f32 macro
 *
 * @param a input vector
 * @return float
 */
static inline float vmaxvq_f32(float32x4_t a) {
  float ret = a[0];
  for (unsigned int i = 1; i < 4; ++i) {
    if (ret > a[i])
      ret = a[i];
  }
  return ret;
}

/**
 * @brief vaddvq_f32
 *
 * @param a input vector
 * @return float32_t
 */
static inline float32_t vaddvq_f32(float32x4_t a) {
  float32_t ret = a[0];
  for (unsigned int i = 1; i < 4; ++i) {
    ret += a[i];
  }
  return ret;
}

/**
 * @brief vaddvq_f32
 *
 * @param a input vector
 * @return uint32_t
 */
static inline uint32_t vaddvq_u32(uint32x4_t a) {
  uint32_t ret = a[0];
  for (unsigned int i = 1; i < 4; ++i) {
    ret += a[i];
  }
  return ret;
}

static inline int32x4_t vcvtnq_s32_f32(float32x4_t a) {
  int32x4_t ret;
  for (unsigned int i = 0; i < 4; ++i) {
    ret[i] = std::lround(a[i]);
  }
  return ret;
}

#ifdef ENABLE_FP16
/**
 * @brief macro for vfmaq_n_f16
 *
 */
#define vfmaq_n_f16(a, b, c, n) vaddq_f16(a, vmulq_f16(b, vmovq_n_f16(c[n])))

/**
 * @brief vmaxvq_f16 macro
 *
 * @param a input vector
 * @return float16_t
 */
static inline float16_t vmaxvq_f16(float16x8_t a) {
  float16_t ret = a[0];
  for (unsigned int i = 1; i < 8; ++i) {
    if (ret > a[i])
      ret = a[i];
  }
  return ret;
}
#endif
#endif
