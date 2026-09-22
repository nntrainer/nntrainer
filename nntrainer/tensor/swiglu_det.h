// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   swiglu_det.h
 * @date   10 Sep 2026
 * @brief  The host half of the deterministic SwiGLU -- same bits as HVX
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * hvx/hvx_swiglu_det.h computes SwiGLU on the DSP; this computes the same
 * specification on the host, operation for operation, so the two agree bit
 * for bit. Why that matters rather than mere accuracy is in that file's
 * header and doc 44 section 3: the SwiGLU result is requantized to uint8,
 * and two independently-rounded pipelines eventually disagree by one level
 * on some element, which the down matmul then spreads across a whole row.
 * Bit identity is the only property that drives that flip count to zero.
 *
 * The specification lives in hvx_swiglu_det.h. Do not change one side of
 * it. HvxSwigluDet.MatchesScalarBitExact (DSP vs this header's scalar) and
 * SwigluDetNeonMatchesScalar (this header's NEON vs this header's scalar)
 * are what keep the three implementations from drifting -- and drift is not
 * hypothetical: the HVX side shipped with `< 0` where the spec and the
 * scalar say `<= 0`, and the device found it at 5 values out of 8192.
 *
 * FUSED MULTIPLY-ADD IS THE HAZARD HERE. Every `p*r + C` below must round
 * twice, because HVX has no f32 FMA and rounds twice. A compiler that
 * contracts one multiply and one add into an fmla rounds once and produces
 * different bits -- silently, with no warning, in a kernel that otherwise
 * looks right. The scalar path stores every intermediate through a
 * volatile, which no contraction can cross.
 *
 * `#pragma clang fp contract(off)` does NOT protect the NEON path, which is
 * worth writing down because it is the obvious thing to reach for and it
 * looks like it works. vmulq_f32 and vaddq_f32 expand to a multiply and an
 * add written inside arm_neon.h, so the pragma's lexical scope never covers
 * them: built for aarch64 at -O3 -ffp-contract=fast, this kernel emitted
 * the same 12 fmla instructions with the pragma as without it. What works
 * is an empty asm tying each result to a vector register --
 * swiglu_det_vmul/vadd/vsub below.
 *
 * Checked rather than assumed, since this is a property of the compiler
 * rather than of the source: aarch64 gcc 13 and clang 18, at -O3,
 * -ffp-contract=fast, -ffast-math and -Ofast, emit zero fmla and agree with
 * the scalar path on all 8192 values of the gate test's spread (run under
 * qemu-aarch64, so: emulated, not this phone). -ffast-math is not a
 * hypothetical setting here -- Applications/CausalLM/jni/Android.mk builds
 * with it. SwigluDetNeon.MatchesScalar is the same check on the device.
 *
 * OUT OF SCOPE: denormal inputs. -ffast-math links a startup that enables
 * flush-to-zero, and whether HVX flushes identically is not something this
 * header can control. Nothing in the MoE path reaches there -- the SwiGLU
 * inputs are dequantized matmul outputs of order 1, and exp_det's own
 * smallest output is 1.6e-38, a normal.
 */

#ifndef __NNTRAINER_SWIGLU_DET_H__
#define __NNTRAINER_SWIGLU_DET_H__

#include <math.h>
#include <stdint.h>
#include <string.h>

/** @brief Upper clamp on the exp argument. See hvx_swiglu_det.h for why 85
 *         and not 88.7: past ~87.98 the reciprocal's magic seed goes
 *         negative and Newton-Raphson diverges to NaN, and its accuracy is
 *         already gone well before that. */
#define SWIGLU_DET_EXP_MAX 85.0f
/** @brief Lower clamp. exp(-88) is below f32's smallest normal, so nothing
 *         representable is lost. */
#define SWIGLU_DET_EXP_MIN (-88.0f)

/**
 * @brief One f32 operation, forced to round on its own.
 *
 * Storing through a volatile is what makes contraction impossible
 * regardless of -ffp-contract. It is not free, but the scalar path only
 * ever runs on a tail of at most three elements (or on a platform with no
 * NEON, where correctness is what matters and speed is not).
 */
static inline float swiglu_det_mul(float a, float b) {
  volatile float r = a * b;
  return r;
}
static inline float swiglu_det_add(float a, float b) {
  volatile float r = a + b;
  return r;
}
static inline float swiglu_det_sub(float a, float b) {
  volatile float r = a - b;
  return r;
}

static inline int32_t swiglu_det_bits(float f) {
  int32_t i;
  memcpy(&i, &f, sizeof(i));
  return i;
}
static inline float swiglu_det_float(int32_t i) {
  float f;
  memcpy(&f, &i, sizeof(f));
  return f;
}

/**
 * @brief exp(x), the normative scalar form of hvx_exp_det_sf.
 *
 * Range reduction to |r| <= ln2/2 with ln2 split hi/lo so that kl*LN2_HI is
 * exact (the Cephes/fdlibm technique), then Horner on sum r^n/n! for
 * n = 0..7, then scaling by 2^k as an integer add into the exponent field.
 */
static inline float swiglu_det_exp(float x) {
  if (x > SWIGLU_DET_EXP_MAX) {
    x = SWIGLU_DET_EXP_MAX;
  }
  if (x < SWIGLU_DET_EXP_MIN) {
    x = SWIGLU_DET_EXP_MIN;
  }

  /* nearbyintf under the default rounding mode is round-to-nearest-even,
     which is what the DSP's hvx_sf_to_w_rne and NEON's vcvtnq_s32_f32 both
     do. Nothing in nntrainer changes the mode. Rolling this by hand as a
     bias-and-truncate is a trap -- the obvious version sends -2.5 to -4. */
  const int32_t k = (int32_t)nearbyintf(swiglu_det_mul(x, 1.44269504f));
  const float kl = (float)k;

  float r = swiglu_det_sub(x, swiglu_det_mul(kl, 0.693359375f));
  r = swiglu_det_sub(r, swiglu_det_mul(kl, -2.12194440e-4f));

  float p = 1.0f / 5040.0f;
  p = swiglu_det_add(swiglu_det_mul(p, r), 1.0f / 720.0f);
  p = swiglu_det_add(swiglu_det_mul(p, r), 1.0f / 120.0f);
  p = swiglu_det_add(swiglu_det_mul(p, r), 1.0f / 24.0f);
  p = swiglu_det_add(swiglu_det_mul(p, r), 1.0f / 6.0f);
  p = swiglu_det_add(swiglu_det_mul(p, r), 0.5f);
  p = swiglu_det_add(swiglu_det_mul(p, r), 1.0f);
  p = swiglu_det_add(swiglu_det_mul(p, r), 1.0f);

  const int32_t pb = swiglu_det_bits(p);
  /* p is in [0.707, 1.414], so this is 126 or 127. Sign off, exponent
     down. */
  const int32_t p_exp = (int32_t)(((uint32_t)pb << 1) >> 24);
  /* At or below zero the true result is subnormal or smaller and the bit
     add produced garbage rather than a small number: at x = -88 it yields
     3.5e-40 where the answer is 6.1e-39. Flush to zero, which is also
     where the host reference lands. `<=`, not `<` -- equality is the case
     that actually occurs. */
  if (k + p_exp <= 0) {
    return 0.0f;
  }
  return swiglu_det_float(pb + (k << 23));
}

/**
 * @brief 1/d, the normative scalar form of hvx_recip_det_sf.
 *
 * DOMAIN: bits(d) <= 0x7EF311C2 (d <= 1.6154730e38). Past that the seed
 * subtraction goes negative and three Newton-Raphson steps overflow to NaN
 * rather than to a wrong-but-finite reciprocal. swiglu_det's clamp on the
 * exp argument is what keeps d inside it.
 *
 * Note for anyone comparing against a true divide: the fixed point of this
 * iteration is not always the correctly-rounded reciprocal. 1/1 comes out
 * as 0x3F7FFFFF, one ulp below 1.0f, so silu(g) for a large positive g is
 * g*(1-2^-24) rather than g. That is a property of the specification, it is
 * identical on both machines, and bit identity -- not agreement with a
 * divide -- is what this is for.
 */
static inline float swiglu_det_recip(float d) {
  float y = swiglu_det_float((int32_t)0x7EF311C2u - swiglu_det_bits(d));
  for (int it = 0; it < 3; ++it) {
    y = swiglu_det_mul(y, swiglu_det_sub(2.0f, swiglu_det_mul(d, y)));
  }
  return y;
}

/** @brief silu(g)*u = (g * (1/(1 + exp(-g)))) * u, one element. */
static inline float swiglu_det_one(float g, float u) {
  const float e = swiglu_det_exp(swiglu_det_sub(0.0f, g));
  const float s = swiglu_det_recip(swiglu_det_add(1.0f, e));
  return swiglu_det_mul(swiglu_det_mul(g, s), u);
}

#if defined(__ARM_NEON)
#include <arm_neon.h>
#define SWIGLU_DET_HAS_NEON 1

/**
 * @brief The three vector operations, each made opaque to the optimizer.
 *
 * The empty asm ties the result to a vector register through an instruction
 * the compiler cannot see through, so there is no fmul left for the backend
 * to fuse into an fmla and no add left for it to reassociate. It emits
 * nothing, is not volatile and is not a memory barrier; the cost is
 * scheduling freedom, not instructions.
 *
 * Both halves matter, and the second was found by counting: barriers on the
 * multiplies alone still left one fmla under -ffast-math, because
 * `y * (2 - d*y)` distributes into `2y - d*y*y`. Applications/CausalLM is
 * built with -ffast-math, so that is not a hypothetical setting -- it is the
 * one the model actually compiles under.
 */
static inline float32x4_t swiglu_det_vmul(float32x4_t a, float32x4_t b) {
  float32x4_t r = vmulq_f32(a, b);
  __asm__("" : "+w"(r));
  return r;
}
static inline float32x4_t swiglu_det_vadd(float32x4_t a, float32x4_t b) {
  float32x4_t r = vaddq_f32(a, b);
  __asm__("" : "+w"(r));
  return r;
}
static inline float32x4_t swiglu_det_vsub(float32x4_t a, float32x4_t b) {
  float32x4_t r = vsubq_f32(a, b);
  __asm__("" : "+w"(r));
  return r;
}

/** @brief exp(x) for four lanes, matching swiglu_det_exp bit for bit. */
static inline float32x4_t swiglu_det_exp_neon(float32x4_t x) {
  x = vminq_f32(x, vdupq_n_f32(SWIGLU_DET_EXP_MAX));
  x = vmaxq_f32(x, vdupq_n_f32(SWIGLU_DET_EXP_MIN));

  const int32x4_t k =
    vcvtnq_s32_f32(swiglu_det_vmul(x, vdupq_n_f32(1.44269504f)));
  const float32x4_t kl = vcvtq_f32_s32(k);

  float32x4_t r =
    swiglu_det_vsub(x, swiglu_det_vmul(kl, vdupq_n_f32(0.693359375f)));
  r = swiglu_det_vsub(r, swiglu_det_vmul(kl, vdupq_n_f32(-2.12194440e-4f)));

  float32x4_t p = vdupq_n_f32(1.0f / 5040.0f);
  p = swiglu_det_vadd(swiglu_det_vmul(p, r), vdupq_n_f32(1.0f / 720.0f));
  p = swiglu_det_vadd(swiglu_det_vmul(p, r), vdupq_n_f32(1.0f / 120.0f));
  p = swiglu_det_vadd(swiglu_det_vmul(p, r), vdupq_n_f32(1.0f / 24.0f));
  p = swiglu_det_vadd(swiglu_det_vmul(p, r), vdupq_n_f32(1.0f / 6.0f));
  p = swiglu_det_vadd(swiglu_det_vmul(p, r), vdupq_n_f32(0.5f));
  p = swiglu_det_vadd(swiglu_det_vmul(p, r), vdupq_n_f32(1.0f));
  p = swiglu_det_vadd(swiglu_det_vmul(p, r), vdupq_n_f32(1.0f));

  const int32x4_t pb = vreinterpretq_s32_f32(p);
  const float32x4_t y =
    vreinterpretq_f32_s32(vaddq_s32(pb, vshlq_n_s32(k, 23)));
  const int32x4_t p_exp = vreinterpretq_s32_u32(
    vshrq_n_u32(vreinterpretq_u32_s32(vshlq_n_s32(pb, 1)), 24));
  const uint32x4_t underflow = vcleq_s32(vaddq_s32(k, p_exp), vdupq_n_s32(0));
  return vbslq_f32(underflow, vdupq_n_f32(0.0f), y);
}

/** @brief 1/d for four lanes, matching swiglu_det_recip bit for bit. */
static inline float32x4_t swiglu_det_recip_neon(float32x4_t d) {
  float32x4_t y = vreinterpretq_f32_s32(
    vsubq_s32(vdupq_n_s32((int32_t)0x7EF311C2u), vreinterpretq_s32_f32(d)));
  for (int it = 0; it < 3; ++it) {
    y = swiglu_det_vmul(
      y, swiglu_det_vsub(vdupq_n_f32(2.0f), swiglu_det_vmul(d, y)));
  }
  return y;
}

/** @brief silu(g)*u for four lanes. `0 - g`, not vnegq_f32: negation and
 *         subtract-from-zero disagree on signed zero, and the DSP side
 *         subtracts. */
static inline float32x4_t swiglu_det_neon(float32x4_t g, float32x4_t u) {
  const float32x4_t e =
    swiglu_det_exp_neon(swiglu_det_vsub(vdupq_n_f32(0.0f), g));
  const float32x4_t s =
    swiglu_det_recip_neon(swiglu_det_vadd(vdupq_n_f32(1.0f), e));
  return swiglu_det_vmul(swiglu_det_vmul(g, s), u);
}
#endif /* __ARM_NEON */

/**
 * @brief X[i] = silu(Y[i]) * Z[i], the drop-in shape of nntrainer::swiglu.
 *
 * NEON where the target has it, the volatile scalar form everywhere else.
 * Both produce the same bits, so which one runs is only a speed question.
 */
static inline void swiglu_det(unsigned int N, float *X, const float *Y,
                              const float *Z) {
  unsigned int i = 0;
#ifdef SWIGLU_DET_HAS_NEON
  for (; i + 4u <= N; i += 4u) {
    vst1q_f32(X + i, swiglu_det_neon(vld1q_f32(Y + i), vld1q_f32(Z + i)));
  }
#endif
  for (; i < N; ++i) {
    X[i] = swiglu_det_one(Y[i], Z[i]);
  }
}

#endif /* __NNTRAINER_SWIGLU_DET_H__ */
