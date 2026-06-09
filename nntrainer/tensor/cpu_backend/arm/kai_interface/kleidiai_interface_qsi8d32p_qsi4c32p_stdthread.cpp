// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Arm Limited and/or its affiliates
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   kleidiai_interface_qsi8d32p_qsi4c32p_stdthread.cpp
 * @date   8 December 2025
 * @see    https://github.com/ARM-software/kleidiai
 * @see    https://github.com/nntrainer/nntrainer
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
#include <kleidiai_interface.h>
#include <string>
#include <thread>
#include <vector>

#include "kai/kai_common.h"

// Micro-kernel interface
#include "kai/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p_qsi4c32p_interface.h"

// Packing functions for qsi8d32p_qsi4c32p
#include "kai/pack/kai_lhs_quant_pack_qsi8d32p_f32.h"

/**
 * @brief Micro-kernel struct definition for qsi8d32p_qsi4c32p
 *
 * @note This struct must match the struct definition in kleidiai_qsi8d32p.cpp
 */
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
  size_t m, size_t n, size_t k, void *lhs_native_mtx_f32, void *rhs_packed_mtx,
  float *dst_act_mtx_f32, uint32_t idx_variant, bool transB, float lower_bound,
  float upper_bound) {
  (void)transB; // Currently only NxK format is supported

  const size_t mr = ukernel_variants_qsi8d32p[idx_variant].ukernel.get_mr();
  const size_t kr = ukernel_variants_qsi8d32p[idx_variant].ukernel.get_kr();
  const size_t sr = ukernel_variants_qsi8d32p[idx_variant].ukernel.get_sr();
  const size_t bl = 32;
  const int num_threads = 4;

  // Pre-allocate LHS packed buffer (shared across threads, each thread writes
  // to its own portion)
  const size_t lhs_packed_size =
    kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32(m, k, bl, mr, kr, sr);
  uint8_t *lhs_packed_mtx = new uint8_t[lhs_packed_size];

  // Thread worker lambda - parallelizes over M dimension
  auto thread_worker = [&](int thread_index) {
    // Each thread processes m_to_process number of rows
    const size_t m_step =
      ukernel_variants_qsi8d32p[idx_variant].ukernel.get_m_step();
    const size_t num_m_per_thread =
      kai_roundup(m, m_step * num_threads) / num_threads;
    const size_t m_start = thread_index * num_m_per_thread;

    // For small shapes and m_step > 1, there may not be enough parallelism
    if (m_start < m) {
      size_t m_to_process = num_m_per_thread;
      if (m_start + m_to_process > m) {
        m_to_process = m - m_start;
      }

      // LHS packing: each thread packs its own portion
      const float *src_ptr = (const float *)lhs_native_mtx_f32 +
                             kai_get_lhs_offset_lhs_quant_pack_qsi8d32p_f32(
                               m_start, k * sizeof(float)) /
                               sizeof(float);
      const size_t lhs_packed_offset =
        ukernel_variants_qsi8d32p[idx_variant].ukernel.get_lhs_packed_offset(
          m_start, k, bl);
      void *lhs_packed_ptr = lhs_packed_mtx + lhs_packed_offset;

      kai_run_lhs_quant_pack_qsi8d32p_f32(
        m_to_process, k, bl, mr, kr, sr, // Dimensions and packing args
        0,                               // m_idx_start (relative to this chunk)
        src_ptr,                         // LHS (f32)
        k * sizeof(float),               // LHS stride
        lhs_packed_ptr);                 // LHS packed

      // Matmul micro-kernel
      const size_t dst_stride = n * sizeof(float);
      const size_t rhs_offset =
        ukernel_variants_qsi8d32p[idx_variant].ukernel.get_rhs_packed_offset(
          0, k, bl);
      const void *rhs_ptr =
        (const void *)((const char *)rhs_packed_mtx + rhs_offset);
      const size_t dst_offset =
        ukernel_variants_qsi8d32p[idx_variant].ukernel.get_dst_offset(
          m_start, 0, dst_stride);
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

/**
 * @brief std::thread-based parallel GEMM implementation optimized for m=1
 * (GEMV)
 *
 * When m=1, the LHS is a single row vector. Parallelizing over M is useless,
 * so instead we parallelize over the N dimension, splitting the RHS columns
 * and output vector across threads.
 *
 * Each thread:
 * 1. Uses the same LHS packed data (single row, packed once)
 * 2. Processes a portion of the RHS (n/num_threads columns)
 * 3. Writes to its portion of the output vector
 */
void nntr_kai_gemm_qsi8d32p_qsi4c32p_olp_parallel_m1(
  size_t m, size_t n, size_t k, void *lhs_native_mtx_f32, void *rhs_packed_mtx,
  float *dst_act_mtx_f32, uint32_t idx_variant, bool transB, float lower_bound,
  float upper_bound) {
  (void)transB; // Currently only NxK format is supported
  (void)m;      // m is always 1 for this function

  const size_t mr = ukernel_variants_qsi8d32p[idx_variant].ukernel.get_mr();
  const size_t kr = ukernel_variants_qsi8d32p[idx_variant].ukernel.get_kr();
  const size_t sr = ukernel_variants_qsi8d32p[idx_variant].ukernel.get_sr();
  const size_t nr = ukernel_variants_qsi8d32p[idx_variant].ukernel.get_nr();
  const size_t bl = 32;
  const int num_threads = 4;

  // LHS packing: quantize f32 to qsi8d32p and pack (single row, done once)
  const size_t lhs_packed_size =
    kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32(1, k, bl, mr, kr, sr);
  uint8_t *lhs_packed_mtx = new uint8_t[lhs_packed_size];
  kai_run_lhs_quant_pack_qsi8d32p_f32(
    1, k, bl, mr, kr, sr,              // m=1
    0,                                 // m_idx_start
    (const float *)lhs_native_mtx_f32, // LHS (f32)
    k * sizeof(float),                 // LHS stride
    lhs_packed_mtx);                   // LHS packed output

  // Calculate n_per_thread, ensuring it's a multiple of nr for proper RHS
  // tiling Round up n/num_threads to the nearest multiple of nr
  const size_t n_per_thread_raw = (n + num_threads - 1) / num_threads;
  const size_t n_per_thread = ((n_per_thread_raw + nr - 1) / nr) * nr;

  // Thread worker lambda - parallelizes over N dimension
  auto thread_worker = [&](int thread_index) {
    const size_t n_start = thread_index * n_per_thread;

    // Check if this thread has work to do
    if (n_start >= n) {
      return;
    }

    // Calculate actual n for this thread (may be less for last thread)
    size_t n_chunk = n_per_thread;
    if (n_start + n_chunk > n) {
      n_chunk = n - n_start;
    }

    // Get offsets for this thread's portion of RHS and DST
    const size_t dst_stride = n * sizeof(float);
    const size_t lhs_offset =
      ukernel_variants_qsi8d32p[idx_variant].ukernel.get_lhs_packed_offset(0, k,
                                                                           bl);
    const size_t rhs_offset =
      ukernel_variants_qsi8d32p[idx_variant].ukernel.get_rhs_packed_offset(
        n_start, k, bl);

    const void *lhs_ptr =
      (const void *)((const char *)lhs_packed_mtx + lhs_offset);
    const void *rhs_ptr =
      (const void *)((const char *)rhs_packed_mtx + rhs_offset);
    // For m=1, dst is a vector, so offset by n_start floats
    float *dst_ptr = dst_act_mtx_f32 + n_start;

    ukernel_variants_qsi8d32p[idx_variant].ukernel.run_matmul(
      1, n_chunk, k, bl, // m=1, this thread's n chunk
      lhs_ptr,           // LHS packed (shared)
      rhs_ptr,           // RHS packed (this thread's portion)
      dst_ptr,           // DST (this thread's portion)
      dst_stride,        // DST stride (row)
      sizeof(float),     // DST stride (col)
      lower_bound, upper_bound);
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
