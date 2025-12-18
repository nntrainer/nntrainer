// SPDX-License-Identifier: The MIT License (MIT)
/**
 * Copyright (c) 2017 Facebook Inc.
 * Copyright (c) 2017 Georgia Institute of Technology
 * Copyright 2019 Google LLC
 */
/**
 * @file   fp16.cpp
 * @date   03 Nov 2023
 * @brief  This is collection of FP16 and FP32 conversion
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Marat Dukhan <maratek@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <fp16.h>

namespace nntrainer {

float fp32_from_bits(uint32_t w) {
#if defined(__OPENCL_VERSION__)
  return as_float(w);
#elif defined(__CUDA_ARCH__)
  return __uint_as_float((unsigned int)w);
#elif defined(__INTEL_COMPILER)
  return _castu32_f32(w);
/// @todo resolve unknown identifier error
// #elif defined(_MSC_VER) && (defined(_M_ARM) || defined(_M_ARM64))
//   return _CopyFloatFromInt32((__int32)w);
#else
  union {
    uint32_t as_bits;
    float as_value;
  } fp32 = {w};
  return fp32.as_value;
#endif
}

uint32_t fp32_to_bits(float f) {
#if defined(__OPENCL_VERSION__)
  return as_uint(f);
#elif defined(__CUDA_ARCH__)
  return (uint32_t)__float_as_uint(f);
#elif defined(__INTEL_COMPILER)
  return _castf32_u32(f);
/// @todo resolve unknown identifier error
// #elif defined(_MSC_VER) && (defined(_M_ARM) || defined(_M_ARM64))
//   return (uint32_t)_CopyInt32FromFloat(f);
#else
  union {
    float as_value;
    uint32_t as_bits;
  } fp32 = {f};
  return fp32.as_bits;
#endif
}

uint16_t compute_fp32_to_fp16(float f) {
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) ||              \
  defined(__GNUC__) && !defined(__STRICT_ANSI__)
  const float scale_to_inf = 0x1.0p+112f;
  const float scale_to_zero = 0x1.0p-110f;
#else
  const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
  const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
  float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

  const uint32_t w = fp32_to_bits(f);
  const uint32_t shl1_w = w + w;
  const uint32_t sign = w & UINT32_C(0x80000000);
  uint32_t bias = shl1_w & UINT32_C(0xFF000000);
  if (bias < UINT32_C(0x71000000)) {
    bias = UINT32_C(0x71000000);
  }

  base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
  const uint32_t bits = fp32_to_bits(base);
  const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
  const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
  const uint32_t nonsign = exp_bits + mantissa_bits;
  return (sign >> 16) |
         (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

float compute_fp16_to_fp32(uint16_t h) {
  const uint32_t w = (uint32_t)h << 16;
  const uint32_t sign = w & UINT32_C(0x80000000);
  const uint32_t two_w = w + w;
  const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) ||              \
  defined(__GNUC__) && !defined(__STRICT_ANSI__)
  const float exp_scale = 0x1.0p-112f;
#else
  const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
  const float normalized_value =
    fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

  const uint32_t magic_mask = UINT32_C(126) << 23;
  const float magic_bias = 0.5f;
  const float denormalized_value =
    fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

  const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
  const uint32_t result =
    sign | (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value)
                                        : fp32_to_bits(normalized_value));
  return fp32_from_bits(result);
}

} /* namespace nntrainer */
