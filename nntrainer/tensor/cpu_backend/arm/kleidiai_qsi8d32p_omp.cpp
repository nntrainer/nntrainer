// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Arm Limited and/or its affiliates
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   kleidiai_qsi8d32p_omp.cpp
 * @date   8 December 2025
 * @see    https://github.com/ARM-software/kleidiai
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong
 *
 * @brief  OpenMP-based parallel GEMM implementation for qsi8d32p_qsi4c32p
 *
 * @note   Licensed under the Apache License, Version 2.0 (the "License");
 *         you may not use this file except in compliance with the License.
 *         You may obtain a copy of the License at
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * @bug    No known bugs except for NYI items
 */

#include <cassert>
#include <cfloat>
#include <cstdint>
#include <string>
#include <neon_kleidiai.h>

#include "kai/kai_common.h"

// Micro-kernel interface
#include "kai/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p_qsi4c32p_interface.h"

// Packing functions for qsi8d32p_qsi4c32p
#include "kai/pack/kai_lhs_quant_pack_qsi8d32p_f32.h"

// Micro-kernel struct definition (must match kleidiai_qsi8d32p.cpp)
struct kai_matmul_ukernel_f32_qsi8d32p_qsi4c32p {
  kai_matmul_clamp_f32_qsi8d32p_qsi4c32p_ukernel ukernel;
  std::string name = {};
};

// Forward declaration of ukernel variants from kleidiai_qsi8d32p.cpp
extern kai_matmul_ukernel_f32_qsi8d32p_qsi4c32p ukernel_variants_qsi8d32p[];

/**
 * @brief OpenMP-based parallel GEMM implementation
 * 
 * This function uses OpenMP for parallelization over the N dimension.
 * LHS is packed once before the parallel region and shared across all threads.
 */
void nntr_kai_gemm_qsi8d32p_qsi4c32p_olp_parallel(
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

  int n_threads = 4;
  assert(n % n_threads == 0);
  size_t n_ukernel = n / n_threads;
#pragma omp parallel for num_threads(n_threads)
  for (int current_thread = 0; current_thread < n_threads; ++current_thread) {
    const size_t dst_stride = n * sizeof(float);
    const size_t lhs_offset =
      ukernel_variants_qsi8d32p[idx_variant].ukernel.get_lhs_packed_offset(0, k, bl);
    const size_t rhs_offset =
      ukernel_variants_qsi8d32p[idx_variant].ukernel.get_rhs_packed_offset(
        n_ukernel * current_thread, k, bl);
    const size_t dst_offset =
      ukernel_variants_qsi8d32p[idx_variant].ukernel.get_dst_offset(
        0, n_ukernel * current_thread, dst_stride);

    const void *lhs_ptr =
      (const void *)((const char *)lhs_packed_mtx + lhs_offset);
    const void *rhs_ptr =
      (const void *)((const char *)rhs_packed_mtx + rhs_offset);
    float *dst_ptr = (float *)((uint8_t *)dst_act_mtx_f32 + dst_offset);

    ukernel_variants_qsi8d32p[idx_variant].ukernel.run_matmul(
      m, n / n_threads, k, bl, // Dimensions
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
