// SPDX-License-Identifier: Apache-2.0
/**
 * @author NNTrainer contributors
 * @bug No known bugs.
 */
/**
 * @file lfm2_router_arm.cpp
 * @brief Exact FP32 router arithmetic, isolated from the app's fast math.
 */
#include "lfm2_router_arm.h"

#if defined(__aarch64__) && defined(ENABLE_HEXKL)
#if defined(__FAST_MATH__)
#error "Compile lfm2_router_arm.cpp with -fno-fast-math -ffp-contract=off"
#endif
#include <arm_neon.h>

#if defined(__clang__)
#define LFM_ROUTER_UNROLL _Pragma("unroll")
#elif defined(__GNUC__)
#define LFM_ROUTER_UNROLL _Pragma("GCC unroll 4")
#else
#define LFM_ROUTER_UNROLL
#endif

namespace causallm {
namespace {
/** HVX and ARM differ for some subnormal operands. This integer-only scan
 * accepts signed zero and does not quiet NaNs. The output check below catches
 * non-finite results, including overflow from otherwise finite operands. */
static bool hasSubnormal(const float *data, unsigned int count) {
  const uint32x4_t abs_mask = vdupq_n_u32(0x7fffffffu);
  const uint32x4_t one = vdupq_n_u32(1);
  const uint32x4_t last = vdupq_n_u32(0x007fffffu);
  uint32x4_t found = vdupq_n_u32(0);
  for (unsigned int i = 0; i < count; i += 4) {
    const uint32x4_t magnitude =
      vandq_u32(vreinterpretq_u32_f32(vld1q_f32(data + i)), abs_mask);
    found = vorrq_u32(found, vcltq_u32(vsubq_u32(magnitude, one), last));
  }
  return vmaxvq_u32(found) != 0;
}
} // namespace

bool lfm2RouterArmWeightSupported(const float *w) {
  return !hasSubnormal(w, 2048 * 64);
}

// Keep the strict arithmetic out of callers compiled with fast math, including
// builds that enable link-time optimization.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
bool lfm2RouterArmBlock(const float *x, const float *w, float *out) {
  constexpr unsigned int hidden = 2048, experts = 64;
  if (hasSubnormal(x, 4 * hidden))
    return false;
  uint32x4_t nonfinite = vdupq_n_u32(0);
  const uint32x4_t exponent = vdupq_n_u32(0x7f800000u);
  for (unsigned int col = 0; col < experts; col += 16) {
    float32x4_t acc[4][4];
    LFM_ROUTER_UNROLL
    for (unsigned int r = 0; r < 4; ++r)
      LFM_ROUTER_UNROLL
    for (unsigned int e = 0; e < 4; ++e)
      acc[r][e] = vdupq_n_f32(0.0f);
    for (unsigned int k = 0; k < hidden; ++k) {
      float32x4_t weight[4];
      LFM_ROUTER_UNROLL
      for (unsigned int e = 0; e < 4; ++e)
        weight[e] = vld1q_f32(w + k * experts + col + e * 4);
      LFM_ROUTER_UNROLL
      for (unsigned int r = 0; r < 4; ++r) {
        const float32x4_t input = vdupq_n_f32(x[r * hidden + k]);
        LFM_ROUTER_UNROLL
        for (unsigned int e = 0; e < 4; ++e)
          acc[r][e] = vaddq_f32(acc[r][e], vmulq_f32(input, weight[e]));
      }
    }
    LFM_ROUTER_UNROLL
    for (unsigned int r = 0; r < 4; ++r)
      LFM_ROUTER_UNROLL
    for (unsigned int e = 0; e < 4; ++e) {
      nonfinite = vorrq_u32(
        nonfinite,
        vceqq_u32(vandq_u32(vreinterpretq_u32_f32(acc[r][e]), exponent),
                  exponent));
      vst1q_f32(out + r * experts + col + e * 4, acc[r][e]);
    }
  }
  return vmaxvq_u32(nonfinite) == 0;
}
} // namespace causallm
#undef LFM_ROUTER_UNROLL
#endif
