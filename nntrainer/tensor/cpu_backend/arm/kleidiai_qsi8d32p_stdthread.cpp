// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Arm Limited and/or its affiliates
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   kleidiai_qsi8d32p_stdthread.cpp
 * @date   8 December 2025
 * @see    https://github.com/ARM-software/kleidiai
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong
 *
 * @brief  std::thread-based parallel GEMM implementation for qsi8d32p_qsi4c32p
 *
 * @note   Licensed under the Apache License, Version 2.0 (the "License");
 *         you may not use this file except in compliance with the License.
 *         You may obtain a copy of the License at
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * @bug    No known bugs except for NYI items
 */

#include <cfloat>
#include <cstdint>
#include <string>
#include <neon_kleidiai.h>
#include <thread>
#include <vector>

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
 * @brief std::thread-based parallel GEMM implementation
 * 
 * This function uses std::thread for parallelization over the M dimension.
 * Each thread handles its own portion of LHS packing and matmul execution.
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
  const int num_threads = 4;

  // Pre-allocate LHS packed buffer (shared across threads, each thread writes to its own portion)
  const size_t lhs_packed_size =
    kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32(m, k, bl, mr, kr, sr);
  uint8_t *lhs_packed_mtx = new uint8_t[lhs_packed_size];

  // Thread worker lambda - parallelizes over M dimension
  auto thread_worker = [&](int thread_index) {
    // Each thread processes m_to_process number of rows
    const size_t m_step = ukernel_variants_qsi8d32p[idx_variant].ukernel.get_m_step();
    const size_t num_m_per_thread = kai_roundup(m, m_step * num_threads) / num_threads;
    const size_t m_start = thread_index * num_m_per_thread;

    // For small shapes and m_step > 1, there may not be enough parallelism
    if (m_start < m) {
      size_t m_to_process = num_m_per_thread;
      if (m_start + m_to_process > m) {
        m_to_process = m - m_start;
      }

      // LHS packing: each thread packs its own portion
      const float *src_ptr = (const float *)lhs_native_mtx_f32 +
        kai_get_lhs_offset_lhs_quant_pack_qsi8d32p_f32(m_start, k * sizeof(float)) / sizeof(float);
      const size_t lhs_packed_offset =
        ukernel_variants_qsi8d32p[idx_variant].ukernel.get_lhs_packed_offset(m_start, k, bl);
      void *lhs_packed_ptr = lhs_packed_mtx + lhs_packed_offset;

      kai_run_lhs_quant_pack_qsi8d32p_f32(
        m_to_process, k, bl, mr, kr, sr,  // Dimensions and packing args
        0,                                 // m_idx_start (relative to this chunk)
        src_ptr,                           // LHS (f32)
        k * sizeof(float),                 // LHS stride
        lhs_packed_ptr);                   // LHS packed

      // Matmul micro-kernel
      const size_t dst_stride = n * sizeof(float);
      const size_t rhs_offset =
        ukernel_variants_qsi8d32p[idx_variant].ukernel.get_rhs_packed_offset(0, k, bl);
      const void *rhs_ptr = (const void *)((const char *)rhs_packed_mtx + rhs_offset);
      const size_t dst_offset =
        ukernel_variants_qsi8d32p[idx_variant].ukernel.get_dst_offset(m_start, 0, dst_stride);
      float *dst_ptr = (float *)((uint8_t *)dst_act_mtx_f32 + dst_offset);

      ukernel_variants_qsi8d32p[idx_variant].ukernel.run_matmul(
        m_to_process, n, k, bl,  // Dimensions
        lhs_packed_ptr,          // LHS packed
        rhs_ptr,                 // RHS packed
        dst_ptr,                 // DST
        dst_stride,              // DST stride (row)
        sizeof(float),           // DST stride (col)
        lower_bound, upper_bound // Min and max for the clamp operation
      );
    }
  };

  // Create and launch worker threads
  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(thread_worker, i);
  }

  // Wait for all threads to complete
  for (auto &t : threads) {
    t.join();
  }

  delete[] lhs_packed_mtx;
}
