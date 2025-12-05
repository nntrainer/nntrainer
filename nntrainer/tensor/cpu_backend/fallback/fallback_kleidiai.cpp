// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Arm Limited and/or its affiliates
 *
 * @file   fallback_kleidiai.cpp
 * @date   15 September 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @brief  Modified computational backend components of kleidiai. Portions of
 * this file are derived from Arm Limited code licensed under the Apache
 * License, Version 2.0, with modifications
 * @bug    No known bugs except for NYI items
 * @note   Licensed under the Apache License, Version 2.0 (the "License");
 *         you may not use this file except in compliance with the License.
 *         You may obtain a copy of the License at
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * @modifications
 *   - [2025-09-15] Integrated and adapted Arm-provided code into
 *     nntrainer CPU backend
 *
 */

#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include <fallback_kleidiai.h>

#define INT4_MIN (-8)
#define INT4_MAX (7)

static size_t roundup(size_t a, size_t b) { return ((a + b - 1) / b) * b; }

void ref_quant_qa8dx_f32(size_t m, size_t k, const float *lhs_f32,
                         int8_t *lhs_qa8dx) {
  const size_t dst_stride =
    (k * sizeof(int8_t) + sizeof(float) + sizeof(int32_t));

  const size_t lhs_qa8dx_stride = k;

  for (size_t m_idx = 0; m_idx < m; ++m_idx) {
    const float *src_ptr = lhs_f32 + m_idx * lhs_qa8dx_stride;

    float max0 = -FLT_MAX;
    float min0 = FLT_MAX;

    // Find min/max for each channel
    for (size_t k_idx = 0; k_idx < k; ++k_idx) {
      const float src0_0 = src_ptr[k_idx];

      max0 = std::max(src0_0, max0);
      min0 = std::min(src0_0, min0);
    }

    // Maximum/minimum int8 values
    const float qmin = (float)INT8_MIN;
    const float qmax = (float)INT8_MAX;

    const float rmin0 = std::min(0.0f, min0);
    const float rmax0 = std::max(0.0f, max0);

    const float scale0 = rmin0 == rmax0 ? 1.f : (qmax - qmin) / (rmax0 - rmin0);

    // Reciprocal to quantize
    const float recip_scale0 = scale0 ? 1.0f / scale0 : 0.0f;

    const float descaled_min0 = rmin0 * scale0;
    const float descaled_max0 = rmax0 * scale0;

    const float zero_point_from_min_error0 = qmin + descaled_min0;
    const float zero_point_from_max_error0 = qmax + descaled_max0;

    float zero_point0 =
      zero_point_from_min_error0 + zero_point_from_max_error0 > 0
        ? qmin - descaled_min0
        : qmax - descaled_max0;

    zero_point0 = std::max(zero_point0, qmin);
    zero_point0 = std::min(zero_point0, qmax);

    // Round to nearest integer
    const int32_t nudged_zero_point0 = lrintf(zero_point0);

    int8_t *dst_ptr = (int8_t *)lhs_qa8dx + m_idx * dst_stride;

    // LHS offset at the beginning of the row
    *((float *)(dst_ptr)) = recip_scale0;
    dst_ptr += sizeof(float);
    *((int32_t *)(dst_ptr)) = -nudged_zero_point0;
    dst_ptr += sizeof(int32_t);

    // Quantize the channels
    for (size_t k_idx = 0; k_idx < k; ++k_idx) {
      const float src0_0 = src_ptr[k_idx];

      // Scale the values
      int32_t v0_s32 = (int32_t)(std::round(src0_0 * scale0));

      v0_s32 = v0_s32 + nudged_zero_point0;
      v0_s32 = std::max(v0_s32, static_cast<int32_t>(INT8_MIN));
      v0_s32 = std::min(v0_s32, static_cast<int32_t>(INT8_MAX));
      dst_ptr[0] = (int8_t)v0_s32;
      dst_ptr += sizeof(int8_t);
    }
  }
};

static void quant_nxk_qs4cx_f32(size_t n, size_t k, const float *rhs_f32,
                                uint8_t *rhs_qs4cx, float *rhs_scales_f32) {
  const size_t rhs_qs4cx_stride = (roundup(k, 2) / 2);

  // Make sure the output is filled with zeros
  std::memset(rhs_qs4cx, 0, n * rhs_qs4cx_stride);

  for (size_t n_idx = 0; n_idx < n; ++n_idx) {
    const float *src_ptr = rhs_f32 + n_idx * k;

    float max0 = -FLT_MAX;
    float min0 = FLT_MAX;

    // Find min/max for each channel
    for (size_t k_idx = 0; k_idx < k; ++k_idx) {
      const float src0_0 = src_ptr[k_idx];

      max0 = std::max(src0_0, max0);
      min0 = std::min(src0_0, min0);
    }

    // Maximum/minimum int8 values
    const float qmin = (float)INT4_MIN;
    const float qmax = (float)INT4_MAX;

    const float rmin0 = std::min(0.0f, min0);
    const float rmax0 = std::max(0.0f, max0);

    const float scale0 = rmin0 == rmax0 ? 1.f : (qmax - qmin) / (rmax0 - rmin0);

    // Reciprocal to quantize
    const float recip_scale0 = scale0 ? 1.0f / scale0 : 0.0f;

    // Quantize the channels
    for (size_t k_idx = 0; k_idx < k; ++k_idx) {
      const float src0_0 = src_ptr[k_idx];

      // Scale the values
      int32_t v0_s32 = (int32_t)(std::round(src0_0 * scale0));

      // Maximum/minimum int4 values
      v0_s32 = std::max(v0_s32, static_cast<int32_t>(INT4_MIN));
      v0_s32 = std::min(v0_s32, static_cast<int32_t>(INT4_MAX));

      const uint8_t v0_u8 = (uint8_t)(v0_s32 + 8);

      const size_t dst_addr = (k_idx / 2) + n_idx * rhs_qs4cx_stride;
      uint8_t rhs_v0 = rhs_qs4cx[dst_addr];

      if ((k_idx % 2) == 0) {
        rhs_v0 |= v0_u8;
      } else {
        rhs_v0 |= (v0_u8 << 4);
      }
      rhs_qs4cx[dst_addr] = rhs_v0;
    }

    rhs_scales_f32[n_idx] = recip_scale0;
  }
};

static void quant_kxn_qs4cx_f32(size_t n, size_t k, const float *rhs_f32,
                                uint8_t *rhs_qs4cx, float *rhs_scales_f32) {
  const size_t rhs_qs4cx_stride = (roundup(n, 2) / 2);

  // Make sure the output is filled with zeros
  std::memset(rhs_qs4cx, 0, k * rhs_qs4cx_stride);

  for (size_t n_idx = 0; n_idx < n; ++n_idx) {
    const float *src_ptr = rhs_f32 + n_idx * k;

    float max0 = -FLT_MAX;
    float min0 = FLT_MAX;

    // Find min/max for each channel
    for (size_t k_idx = 0; k_idx < k; ++k_idx) {
      const float src0_0 = src_ptr[k_idx];

      max0 = std::max(src0_0, max0);
      min0 = std::min(src0_0, min0);
    }

    // Maximum/minimum int8 values
    const float qmin = (float)INT4_MIN;
    const float qmax = (float)INT4_MAX;

    const float rmin0 = std::min(0.0f, min0);
    const float rmax0 = std::max(0.0f, max0);

    const float scale0 = rmin0 == rmax0 ? 1.f : (qmax - qmin) / (rmax0 - rmin0);

    // Reciprocal to quantize
    const float recip_scale0 = scale0 ? 1.0f / scale0 : 0.0f;

    // Quantize the channels
    for (size_t k_idx = 0; k_idx < k; ++k_idx) {
      const float src0_0 = src_ptr[k_idx];

      // Scale the values
      int32_t v0_s32 = (int32_t)(std::round(src0_0 * scale0));

      // Maximum/minimum int4 values
      v0_s32 = std::max(v0_s32, static_cast<int32_t>(INT4_MIN));
      v0_s32 = std::min(v0_s32, static_cast<int32_t>(INT4_MAX));

      const uint8_t v0_u8 = (uint8_t)(v0_s32 + 8);

      const size_t dst_addr = (n_idx / 2) + k_idx * rhs_qs4cx_stride;
      uint8_t rhs_v0 = rhs_qs4cx[dst_addr];

      if ((n_idx % 2) == 0) {
        rhs_v0 |= v0_u8;
      } else {
        rhs_v0 |= (v0_u8 << 4);
      }
      rhs_qs4cx[dst_addr] = rhs_v0;
    }

    rhs_scales_f32[n_idx] = recip_scale0;
  }
};

void quant_qs4cx_f32(size_t n, size_t k, rhs_format format,
                     const float *rhs_f32, uint8_t *rhs_qs4cx,
                     float *rhs_scales_f32) {
  if (rhs_format::nxk == format) {
    quant_nxk_qs4cx_f32(n, k, rhs_f32, rhs_qs4cx, rhs_scales_f32);
  } else {
    quant_kxn_qs4cx_f32(n, k, rhs_f32, rhs_qs4cx, rhs_scales_f32);
  }
};

static void ref_matmul_mxn_mxk_nxk_f32_qa8dx_qs4cx( // transB
  size_t m, size_t n, size_t k, const int8_t *lhs_qa8dx,
  const uint8_t *rhs_qs4cx, const float *rhs_scales_f32, float *dst_f32,
  float scalar_min, float scalar_max) {
  const size_t lhs_stride =
    k * sizeof(int8_t) + sizeof(float) + sizeof(int32_t);

  const size_t rhs_qs4cx_stride = (roundup(k, 2) / 2);

  for (size_t m_idx = 0; m_idx < m; ++m_idx) {
    const int8_t *lhs_ptr_start = lhs_qa8dx + m_idx * lhs_stride;

    for (size_t n_idx = 0; n_idx < n; ++n_idx) {
      // Main f32 accumulator
      int32_t iacc = 0;

      const int8_t *lhs_ptr = lhs_ptr_start;
      const uint8_t *rhs_ptr = rhs_qs4cx + n_idx * rhs_qs4cx_stride;

      // Get the LHS quantization parameters stored at the
      // beginning of each row
      const float lhs_scale = *(const float *)lhs_ptr;
      lhs_ptr += sizeof(float);

      const int32_t lhs_offset = *(const int32_t *)lhs_ptr;
      lhs_ptr += sizeof(int32_t);

      for (size_t k_idx = 0; k_idx < k; ++k_idx) {
        // Get the LHS values
        const int32_t lhs_v0 = (int32_t)lhs_ptr[0];

        // Get the RHS values
        const uint8_t rhs_byte = rhs_ptr[0];

        // Unpack the RHS values
        int32_t rhs_v0 = 0;
        if ((k_idx % 2) == 0) {
          rhs_v0 = (((int32_t)(rhs_byte & 0x0F)) - 8);
        } else {
          rhs_v0 = (((int32_t)(rhs_byte >> 4)) - 8);
        }

        iacc += lhs_v0 * rhs_v0;
        iacc += lhs_offset * rhs_v0;

        lhs_ptr += 1;

        // Increment only when k_idx is not a multiple of 2
        rhs_ptr += k_idx % 2;
      }

      // Get the RHS scale
      const float rhs_scale = rhs_scales_f32[n_idx];

      float main_acc = iacc * rhs_scale;

      main_acc = main_acc * lhs_scale;

      // Clamp (min-max) operation
      main_acc = std::max(main_acc, scalar_min);
      main_acc = std::min(main_acc, scalar_max);

      dst_f32[0] = main_acc;
      dst_f32 += 1;
    }
  }
};

static void ref_matmul_mxn_mxk_kxn_f32_qa8dx_qs4cx( // noTrans
  size_t m, size_t n, size_t k, const int8_t *lhs_qa8dx,
  const uint8_t *rhs_qs4cx, const float *rhs_scales_f32, float *dst_f32,
  float scalar_min, float scalar_max) {
  const size_t lhs_stride =
    k * sizeof(int8_t) + sizeof(float) + sizeof(int32_t);

  const size_t rhs_qs4cx_stride = (roundup(n, 2) / 2);

  for (size_t m_idx = 0; m_idx < m; ++m_idx) {
    const int8_t *lhs_ptr_start = lhs_qa8dx + m_idx * lhs_stride;

    for (size_t n_idx = 0; n_idx < n; ++n_idx) {
      // Main f32 accumulator
      int32_t iacc = 0;

      const int8_t *lhs_ptr = lhs_ptr_start;
      const uint8_t *rhs_ptr = rhs_qs4cx + (n_idx / 2);

      // Get the LHS quantization parameters stored at the
      // beginning of each row
      const float lhs_scale = *(const float *)lhs_ptr;
      lhs_ptr += sizeof(float);

      const int32_t lhs_offset = *(const int32_t *)lhs_ptr;
      lhs_ptr += sizeof(int32_t);

      for (size_t k_idx = 0; k_idx < k; ++k_idx) {
        // Get the LHS values
        const int32_t lhs_v0 = (int32_t)lhs_ptr[0];

        // Get the RHS values
        const uint8_t rhs_byte = rhs_ptr[0];

        // Unpack the RHS values
        int32_t rhs_v0 = 0;
        if ((n_idx % 2) == 0) {
          rhs_v0 = (((int32_t)(rhs_byte & 0x0F)) - 8);
        } else {
          rhs_v0 = (((int32_t)(rhs_byte >> 4)) - 8);
        }

        iacc += lhs_v0 * rhs_v0;
        iacc += lhs_offset * rhs_v0;

        lhs_ptr += 1;

        // Increment only when k_idx is not a multiple of 2
        rhs_ptr += rhs_qs4cx_stride;
      }

      // Get the RHS scale
      const float rhs_scale = rhs_scales_f32[n_idx];

      float main_acc = iacc * rhs_scale;

      main_acc = main_acc * lhs_scale;

      // Clamp (min-max) operation
      main_acc = std::max(main_acc, scalar_min);
      main_acc = std::min(main_acc, scalar_max);

      dst_f32[0] = main_acc;
      dst_f32 += 1;
    }
  }
};

void ref_matmul_f32_qa8dx_qs4cx(size_t m, size_t n, size_t k, rhs_format format,
                                const int8_t *lhs_qa8dx,
                                const uint8_t *rhs_qs4cx,
                                const float *rhs_scales_f32, float *dst_f32,
                                float scalar_min, float scalar_max) {
  const size_t lhs_stride =
    k * sizeof(int8_t) + sizeof(float) + sizeof(int32_t);

  if (rhs_format::nxk == format) {
    ref_matmul_mxn_mxk_nxk_f32_qa8dx_qs4cx(m, n, k, lhs_qa8dx, rhs_qs4cx,
                                           rhs_scales_f32, dst_f32, scalar_min,
                                           scalar_max);
  } else {
    ref_matmul_mxn_mxk_kxn_f32_qa8dx_qs4cx(m, n, k, lhs_qa8dx, rhs_qs4cx,
                                           rhs_scales_f32, dst_f32, scalar_min,
                                           scalar_max);
  }
};

void quant_qs4c32_f32(size_t n, size_t k, size_t bl, const float *rhs_f32,
                      uint8_t *rhs_qs4c32) {
  throw std::runtime_error("NYI : quant_qs4c32_f32 (fallback)");
}

void ref_quant_qs8d32_f32(size_t n, size_t k, size_t bl, const float *rhs_f32,
                          uint8_t *rhs_qs8c32) {
  throw std::runtime_error("NYI : ref_quant_qs8d32_f32 (fallback)");
}

void ref_matmul_f32_qs8d32_qs4c32(size_t m, size_t n, size_t k, size_t bl,
                                  const int8_t *lhs_qa8d32,
                                  const uint8_t *rhs_qs4c32, float *dst_f32,
                                  float scalar_min, float scalar_max) {
  throw std::runtime_error("NYI : ref_matmul_f32_qs8d32_qs4c32 (fallback)");
}
