// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hvx_convert.h
 * @date   03 Aug 2026
 * @brief  int32 <-> f32 vector conversion for HVX, which has no such
 *         instruction
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __NNTRAINER_HVX_CONVERT_H__
#define __NNTRAINER_HVX_CONVERT_H__

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

/**
 * @brief Bit pattern of 1.5 * 2^23 (12582912.0f).
 *
 * HVX has no numeric conversion between int32 and f32. Q6_Vw_equals_Vsf
 * and Q6_Vsf_equals_Vw are bit reinterpretations, and the vcvt family
 * only covers hf<->h/uh/b/ub and sf<->hf/bf. The magic-number identity
 * below is the way across.
 *
 * Adding an integer x to the bit pattern of 1.5 * 2^23 lands x in the low
 * mantissa bits, because that float's exponent puts the mantissa's least
 * significant bit at value 1. Subtracting the same constant as a float
 * then leaves exactly (float)x.
 *
 * Valid for |x| <= 2^22 = 4194304. Our accumulators are bounded by
 * 255 * 8 * K, which is 2088960 at K=1024, so the margin is about 2x.
 * Nothing here is safe for larger K without revisiting this bound.
 */
#define HVX_CONVERT_MAGIC_BITS 0x4B400000u

/**
 * @brief Converts int32 lanes to f32 lanes. Exact for |x| <= 2^22.
 *
 * HVX_Vector is type-agnostic: passing it directly to an f32 intrinsic
 * reinterprets the bits without conversion. Q6_Vsf_equals_Vw /
 * Q6_Vw_equals_Vsf look like reinterpret but are in fact NUMERIC
 * conversions (V79 HVX Programmer's Reference Manual: "Convert IEEE
 * floating point to integer... round to zero"), which would destroy the
 * magic-number identity.
 */
static inline HVX_Vector hvx_w_to_sf(HVX_Vector v) {
  const HVX_Vector magic = Q6_V_vsplat_R(HVX_CONVERT_MAGIC_BITS);
  const HVX_Vector bits = Q6_Vw_vadd_VwVw(v, magic);
  return Q6_Vsf_vsub_VsfVsf(bits, magic);
}

/**
 * @brief Converts f32 lanes to int32 lanes, rounding to nearest even.
 *
 * The rounding comes from the float add, so it is round-to-nearest-even.
 * Host references that compare against this must use nearbyint, not round.
 * Valid for results in |x| <= 2^22.
 *
 * Same reinterpret note as hvx_w_to_sf: the cross-type handoff is implicit.
 */
static inline HVX_Vector hvx_sf_to_w_rne(HVX_Vector v) {
  const HVX_Vector magic = Q6_V_vsplat_R(HVX_CONVERT_MAGIC_BITS);
  const HVX_Vector biased = Q6_Vsf_vadd_VsfVsf(v, magic);
  return Q6_Vw_vsub_VwVw(biased, magic);
}

/**
 * @brief Broadcasts a float to every lane.
 *
 * Q6_V_vsplat_R takes a word, so the float's bits have to get there
 * somehow. memcpy is the way that does not punt on strict aliasing --
 * casting through int* is what -Wstrict-aliasing objects to, and this
 * build uses -Wall -Werror. The compiler folds the memcpy away.
 */
static inline HVX_Vector hvx_splat_sf(float f) {
  int bits;
  __builtin_memcpy(&bits, &f, sizeof(bits));
  return Q6_V_vsplat_R(bits);
}

#endif /* __NNTRAINER_HVX_CONVERT_H__ */
