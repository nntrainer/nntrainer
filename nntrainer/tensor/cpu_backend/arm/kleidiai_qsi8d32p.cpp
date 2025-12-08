// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Arm Limited and/or its affiliates
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   kleidiai_qsi8d32p.cpp
 * @date   5 December 2025
 * @see    https://github.com/ARM-software/kleidiai
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong
 *
 * @brief  Modified computational backend components of
 * kleidiai. Portions of this file are derived from Arm
 * Limited code licensed under the Apache License, Version 2.0, with
 * modifications
 *
 * @note   Licensed under the Apache License, Version 2.0 (the "License");
 *         you may not use this file except in compliance with the License.
 *         You may obtain a copy of the License at
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * @modifications
 *   - [2025-12-05] Integrated and adapted Arm-provided code into
 *     nntrainer CPU backend
 *
 * @bug    No known bugs except for NYI items
 */
//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates
// <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <assert.h>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <neon_kleidiai.h>
#include <string>
#include <thread>
#include <vector>

#include "kai/kai_common.h"

#include <chrono>
#include <iostream>
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::microseconds; // or microseconds
using std::chrono::milliseconds; // or microseconds
using std::chrono::nanoseconds;  // or microseconds
using std::chrono::seconds;      // or microseconds

// Micro-kernel variants
#include "kai/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod.h"
#include "kai/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h"
#include "kai/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod.h"
#include "kai/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm.h"
#include "kai/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm.h"
#ifdef __ARM_FEATURE_SVE
#include "kai/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p8x4_1x8_sve_dotprod.h"
#include "kai/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod.h"
#include "kai/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm.h"
#endif
#include "kai/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p_qsi4c32p_interface.h"

// Packing functions for qsi8d32p_qsi4c32p
#include "kai/pack/kai_lhs_quant_pack_qsi8d32p_f32.h"
#include "kai/pack/kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.h"

/**
 * @brief rhs_format
 *
 */
enum class rhs_format {
  nxk,
  kxn,
};

// Micro-kernel interface
/**
 * @brief kai_matmul_ukernel_f32_qsi8d32p_qsi4c32p
 *
 */
struct kai_matmul_ukernel_f32_qsi8d32p_qsi4c32p {
  kai_matmul_clamp_f32_qsi8d32p_qsi4c32p_ukernel ukernel;
  std::string name = {};
};

kai_matmul_ukernel_f32_qsi8d32p_qsi4c32p ukernel_variants_qsi8d32p[] = {
  {kai_get_m_step_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
   kai_get_n_step_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
   kai_get_mr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
   kai_get_nr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
   kai_get_kr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
   kai_get_sr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
   kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
   kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
   kai_get_dst_offset_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
   kai_get_dst_size_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
   kai_run_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
   "matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod"},
#ifdef __ARM_FEATURE_SVE
  {kai_get_m_step_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p8x4_1x8_sve_dotprod,
   kai_get_n_step_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p8x4_1x8_sve_dotprod,
   kai_get_mr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p8x4_1x8_sve_dotprod,
   kai_get_nr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p8x4_1x8_sve_dotprod,
   kai_get_kr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p8x4_1x8_sve_dotprod,
   kai_get_sr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p8x4_1x8_sve_dotprod,
   kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p8x4_1x8_sve_dotprod,
   kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p8x4_1x8_sve_dotprod,
   kai_get_dst_offset_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p8x4_1x8_sve_dotprod,
   kai_get_dst_size_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p8x4_1x8_sve_dotprod,
   kai_run_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p8x4_1x8_sve_dotprod,
   "matmul_clamp_f32_qsi8d32p1x4_qsi4c32p8x4_1x8_sve_dotprod"},
#endif
  {kai_get_m_step_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
   kai_get_n_step_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
   kai_get_mr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
   kai_get_nr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
   kai_get_kr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
   kai_get_sr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
   kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
   kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
   kai_get_dst_offset_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
   kai_get_dst_size_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
   kai_run_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
   "matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod"},
#ifdef __ARM_FEATURE_SVE
  {kai_get_m_step_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
   kai_get_n_step_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
   kai_get_mr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
   kai_get_nr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
   kai_get_kr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
   kai_get_sr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
   kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
   kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
   kai_get_dst_offset_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
   kai_get_dst_size_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
   kai_run_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
   "matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod"},
#endif
  {kai_get_m_step_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
   kai_get_n_step_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
   kai_get_mr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
   kai_get_nr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
   kai_get_kr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
   kai_get_sr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
   kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
   kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
   kai_get_dst_offset_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
   kai_get_dst_size_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
   kai_run_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
   "matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod"},
  {kai_get_m_step_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
   kai_get_n_step_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
   kai_get_mr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
   kai_get_nr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
   kai_get_kr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
   kai_get_sr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
   kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
   kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
   kai_get_dst_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
   kai_get_dst_size_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
   kai_run_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
   "matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm"},
  {kai_get_m_step_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
   kai_get_n_step_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
   kai_get_mr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
   kai_get_nr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
   kai_get_kr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
   kai_get_sr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
   kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
   kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
   kai_get_dst_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
   kai_get_dst_size_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
   kai_run_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
   "matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm"},
#ifdef __ARM_FEATURE_SVE
  {kai_get_m_step_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
   kai_get_n_step_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
   kai_get_mr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
   kai_get_nr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
   kai_get_kr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
   kai_get_sr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
   kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
   kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
   kai_get_dst_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
   kai_get_dst_size_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
   kai_run_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
   "matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm"},
#endif
};

static size_t roundup(size_t a, size_t b) { return ((a + b - 1) / b) * b; }

static inline size_t num_blocks_per_row(size_t k, size_t bl) {
  return k / bl;
}

static inline size_t num_bytes_per_block_qs4c32(size_t bl) {
  return (bl / 2) + sizeof(int16_t);
}

void nntr_kai_quant_qs4c32_f32(size_t n, size_t k, size_t bl, const float *rhs_f32,
                               uint8_t *rhs_qs4c32) {
  const size_t num_blocks_row = num_blocks_per_row(k, bl);
  const size_t num_bytes_block = num_bytes_per_block_qs4c32(bl);
  const size_t dst_stride = num_blocks_row * num_bytes_block;

  for (size_t row_idx = 0; row_idx < n; ++row_idx) {
    const float *src_ptr = rhs_f32 + row_idx * k;

    uint8_t *dst_ptr = (uint8_t *)rhs_qs4c32 + row_idx * dst_stride;

    for (size_t block_idx = 0; block_idx < num_blocks_row; ++block_idx) {
      float amax = 0.0f;
      float max = 0.0f;

      for (size_t b = 0; b < bl; ++b) {
        const float src0_0 = src_ptr[block_idx * bl + b];
        const float asrc0_0 = fabsf(src0_0);

        if (amax < asrc0_0) {
          amax = asrc0_0;
          max = src0_0;
        }
      }

      const float scale = max / -8.0f;
      const float recip_scale = scale ? 1.0f / scale : 0.0f;

      // Store the scale at the beginning of the block
      *((uint16_t *)dst_ptr) = kai_cast_f16_f32(scale);
      dst_ptr += sizeof(uint16_t);

      const size_t block_size = 32;
      const size_t num_subblocks = bl / 32;

      for (size_t subblock_idx = 0; subblock_idx < num_subblocks; ++subblock_idx) {
        for (size_t i = 0; i < block_size / 2; ++i) {
          const size_t src_base_addr =
            block_idx * bl + i + subblock_idx * block_size;
          float v0_f32 = src_ptr[src_base_addr];
          float v1_f32 = src_ptr[src_base_addr + block_size / 2];

          v0_f32 *= recip_scale;
          v1_f32 *= recip_scale;

          const uint8_t v0_u8 =
            (uint8_t)std::min((int8_t)15, (int8_t)(v0_f32 + 8.5f));
          const uint8_t v1_u8 =
            (uint8_t)std::min((int8_t)15, (int8_t)(v1_f32 + 8.5f));

          const uint8_t rhs_v0 = (v1_u8 << 4) | v0_u8;

          dst_ptr[0] = rhs_v0;
          dst_ptr += sizeof(uint8_t);
        }
      }
    }
  }
}

uint32_t nntr_kai_gemm_qsi8d32p_qsi4c32p_rtp(
  size_t m, size_t n, size_t k, void *lhs_native_mtx_f32,
  void *rhs_native_mtx_qs4c32, void *rhs_scales_f32, float *dst_act_mtx_f32,
  bool transB, float lower_bound, float upper_bound) {
  // Note: rhs_scales_f32 is unused - scales are embedded in rhs_native_mtx_qs4c32 (qsu4c32 format)
  (void)rhs_scales_f32;
  
  uint32_t ret_idx = 0;
  uint64_t min_latency = INT64_MAX;
  size_t num_variants =
    sizeof(ukernel_variants_qsi8d32p) / sizeof(ukernel_variants_qsi8d32p[0]);

  const size_t bl = 32;  // Block length for qsi8d32p_qsi4c32p

  // Currently only supporting transB=true (NxK format)
  if (!transB) {
    return 0;
  }

  ///@todo check for optimal variant, or check for optimal variant config for
  /// specific M-N-K combination
  for (size_t idx_variant = 0; idx_variant < num_variants; idx_variant++) {
    const size_t mr = ukernel_variants_qsi8d32p[idx_variant].ukernel.get_mr();
    const size_t nr = ukernel_variants_qsi8d32p[idx_variant].ukernel.get_nr();
    const size_t kr = ukernel_variants_qsi8d32p[idx_variant].ukernel.get_kr();
    const size_t sr = ukernel_variants_qsi8d32p[idx_variant].ukernel.get_sr();

    // Get the size in bytes for the packed matrices using the correct packing functions
    const size_t lhs_packed_size =
      kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32(m, k, bl, mr, kr, sr);
    const size_t rhs_packed_size =
      kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0(n, k, nr, kr, bl);

    // Allocate the matrices
    uint8_t *lhs_packed_mtx = new uint8_t[lhs_packed_size];
    uint8_t *rhs_packed_mtx = new uint8_t[rhs_packed_size];

    auto t2 = high_resolution_clock::now();

    // LHS packing: quantize f32 to qsi8d32p and pack
    kai_run_lhs_quant_pack_qsi8d32p_f32(
      m, k, bl, mr, kr, sr,               // Dimensions and packing args
      0,                                  // m_idx_start
      (const float *)lhs_native_mtx_f32,  // LHS (f32)
      k * sizeof(float),                  // LHS stride
      lhs_packed_mtx);                    // LHS packed output

    // RHS packing: pack qsu4c32 (with embedded scales) to qsi4c32p
    struct kai_rhs_pack_qs4cxs1s0_param params;
    params.lhs_zero_point = 1;  // LHS asymmetric zero point (per reference)
    params.rhs_zero_point = 8;  // 4-bit zero point (unsigned 0-15)
    
    kai_run_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0(
      1,                                    // num_groups
      n, k,                                 // Dimensions
      nr, kr, sr,                           // Packing args
      bl,                                   // Block length
      (const uint8_t *)rhs_native_mtx_qs4c32,  // RHS (qsu4c32 with embedded scales)
      NULL,                                 // Bias (not used)
      rhs_packed_mtx,                       // RHS packed output
      0,                                    // Extra bytes
      &params);

    {
      const size_t dst_stride = n * sizeof(float);
      const size_t lhs_offset =
        ukernel_variants_qsi8d32p[idx_variant].ukernel.get_lhs_packed_offset(0, k, bl);
      const size_t rhs_offset =
        ukernel_variants_qsi8d32p[idx_variant].ukernel.get_rhs_packed_offset(0, k, bl);
      const size_t dst_offset =
        ukernel_variants_qsi8d32p[idx_variant].ukernel.get_dst_offset(0, 0, dst_stride);

      const void *lhs_ptr =
        (const void *)((const char *)lhs_packed_mtx + lhs_offset);
      const void *rhs_ptr =
        (const void *)((const char *)rhs_packed_mtx + rhs_offset);
      float *dst_ptr = (float *)((uint8_t *)dst_act_mtx_f32 + dst_offset);

      ukernel_variants_qsi8d32p[idx_variant].ukernel.run_matmul(
        m, n, k, bl,             // Dimensions
        lhs_ptr,                 // LHS packed
        rhs_ptr,                 // RHS packed
        dst_ptr,                 // DST
        dst_stride,              // DST stride (row)
        sizeof(float),           // DST stride (col)
        lower_bound, upper_bound // Min and max for the clamp operation
      );
    }

    auto t3 = high_resolution_clock::now();
    auto dt2 = duration_cast<nanoseconds>(t3 - t2);

    uint64_t casted_time = static_cast<uint64_t>(dt2.count());
    ret_idx = (min_latency > casted_time) ? idx_variant : ret_idx;
    min_latency = (min_latency > casted_time) ? casted_time : min_latency;

    delete[] lhs_packed_mtx;
    delete[] rhs_packed_mtx;

  }

  return ret_idx;
}


size_t nntr_kai_get_rhs_packed_size_qsi8d32p_qsi4c32p(size_t n, size_t k,
                                                      uint32_t idx_variant,
                                                      bool transB) {
  ///@note Packing arguments are identical among all ukernel idx_variants
  const size_t nr = ukernel_variants_qsi8d32p[idx_variant].ukernel.get_nr();
  const size_t kr = ukernel_variants_qsi8d32p[idx_variant].ukernel.get_kr();
  const size_t bl = 32;
  
  if (transB) {
    return kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0(n, k, nr, kr, bl);
  } else {
    // KxN format not yet implemented
    fprintf(stderr, "ERROR: KxN format not yet supported for qsi8d32p_qsi4c32p\n");
    return 0;
  }
}


void nntr_kai_qsi8d32p_qsi4c32p_rhs_pack(size_t n, size_t k,
                                         void *rhs_packed_mtx,
                                         void *rhs_native_mtx_qs4c32,
                                         void *rhs_scales_f32,
                                         uint32_t idx_variant, bool transB) {
  // Note: rhs_scales_f32 is unused - scales are embedded in rhs_native_mtx_qs4c32
  (void)rhs_scales_f32;
  
  ///@note Packing arguments are identical among all ukernel idx_variants
  const size_t nr = ukernel_variants_qsi8d32p[idx_variant].ukernel.get_nr();
  const size_t kr = ukernel_variants_qsi8d32p[idx_variant].ukernel.get_kr();
  const size_t sr = ukernel_variants_qsi8d32p[idx_variant].ukernel.get_sr();
  const size_t bl = 32;

  if (transB) {
    struct kai_rhs_pack_qs4cxs1s0_param params;
    params.lhs_zero_point = 1;  // LHS asymmetric zero point (per reference)
    params.rhs_zero_point = 8;  // 4-bit zero point (unsigned 0-15)
    
    kai_run_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0(
      1,                                       // num_groups
      n, k,                                    // Dimensions
      nr, kr, sr,                              // Packing args
      bl,                                      // Block length
      (const uint8_t *)rhs_native_mtx_qs4c32,  // RHS (qsu4c32 with embedded scales)
      NULL,                                    // Bias (not used)
      rhs_packed_mtx,                          // RHS packed output
      0,                                       // Extra bytes
      &params);
  } else {
    fprintf(stderr, "ERROR: KxN format not yet supported for qsi8d32p_qsi4c32p\n");
  }
}


void nntr_kai_gemm_qsi8d32p_qsi4c32p_olp_single_thread(
  size_t m, size_t n, size_t k, void *lhs_native_mtx_f32,
  void *rhs_packed_mtx, float *dst_act_mtx_f32, uint32_t idx_variant,
  bool transB, float lower_bound, float upper_bound) {
  (void)transB;  // Currently only NxK format is supported
  
  const size_t mr = ukernel_variants_qsi8d32p[idx_variant].ukernel.get_mr();
  const size_t kr = ukernel_variants_qsi8d32p[idx_variant].ukernel.get_kr();
  const size_t sr = ukernel_variants_qsi8d32p[idx_variant].ukernel.get_sr();
  const size_t bl = 32;

  // LHS packing: quantize f32 to qsi8d32p and pack
  const size_t lhs_packed_size =
    kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32(m, k, bl, mr, kr, sr);
  uint8_t *lhs_packed_mtx = new uint8_t[lhs_packed_size];
  kai_run_lhs_quant_pack_qsi8d32p_f32(
    m, k, bl, mr, kr, sr,                 // Dimensions and packing args
    0,                                    // m_idx_start
    (const float *)lhs_native_mtx_f32,    // LHS (f32)
    k * sizeof(float),                    // LHS stride
    lhs_packed_mtx);                      // LHS packed output
    
  {
    const size_t dst_stride = n * sizeof(float);
    const size_t lhs_offset =
      ukernel_variants_qsi8d32p[idx_variant].ukernel.get_lhs_packed_offset(0, k, bl);
    const size_t rhs_offset =
      ukernel_variants_qsi8d32p[idx_variant].ukernel.get_rhs_packed_offset(0, k, bl);
    const size_t dst_offset =
      ukernel_variants_qsi8d32p[idx_variant].ukernel.get_dst_offset(0, 0, dst_stride);

    const void *lhs_ptr =
      (const void *)((const char *)lhs_packed_mtx + lhs_offset);
    const void *rhs_ptr =
      (const void *)((const char *)rhs_packed_mtx + rhs_offset);
    float *dst_ptr = (float *)((uint8_t *)dst_act_mtx_f32 + dst_offset);

    ukernel_variants_qsi8d32p[idx_variant].ukernel.run_matmul(
      m, n, k, bl,             // Dimensions
      lhs_ptr,                 // LHS packed
      rhs_ptr,                 // RHS packed
      dst_ptr,                 // DST
      dst_stride,              // DST stride (row)
      sizeof(float),           // DST stride (col)
      lower_bound, upper_bound // Min and max for the clamp operation
    );
  }

  delete[] lhs_packed_mtx;
}

// External function defined in kleidiai_qsi8d32p_omp.cpp or kleidiai_qsi8d32p_stdthread.cpp
// depending on the thread-backend meson option
extern void nntr_kai_gemm_qsi8d32p_qsi4c32p_olp_parallel(
  size_t m, size_t n, size_t k, void *lhs_native_mtx_f32,
  void *rhs_packed_mtx, float *dst_act_mtx_f32, uint32_t idx_variant,
  bool transB, float lower_bound, float upper_bound);

void nntr_kai_gemm_qsi8d32p_qsi4c32p_olp(size_t m, size_t n, size_t k,
                                         void *lhs_native_mtx_f32,
                                         void *rhs_packed_mtx,
                                         float *dst_act_mtx_f32,
                                         uint32_t idx_variant, bool transB,
                                         float lower_bound, float upper_bound) {
  if (m == 1) {
    return nntr_kai_gemm_qsi8d32p_qsi4c32p_olp_single_thread(
      m, n, k, lhs_native_mtx_f32, rhs_packed_mtx, dst_act_mtx_f32,
      idx_variant, transB, lower_bound, upper_bound);
  } else {
    return nntr_kai_gemm_qsi8d32p_qsi4c32p_olp_parallel(
      m, n, k, lhs_native_mtx_f32, rhs_packed_mtx, dst_act_mtx_f32,
      idx_variant, transB, lower_bound, upper_bound);
  }
}

