// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Arm Limited and/or its affiliates
 *
 * @file   kleidiai_interface_qai8dxp_qsi8cxp.cpp
 * @date   23 July 2026
 * @see    https://github.com/ARM-software/kleidiai
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jaemin Shin <jaemin980311@gmail.com>
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
 * @bug    No known bugs except for NYI items
 */
//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates
// <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>

#include <kleidiai_interface.h>
#include <thread_manager.h>

#include "kai/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp_qsi8cxp_interface.h"

// Include micro-kernel variants
#include "kai/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod.h"
#include "kai/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod.h"
#include "kai/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod.h"

#if defined(__ARM_FEATURE_MATMUL_INT8)
#include "kai/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.h"
#endif

// Include packing kernels
#include "kai/pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai/pack/kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.h"

namespace {

// Micro-kernel interface
/**
 * @brief kai_matmul_ukernel_f32_qa8dxp_qs8cxp
 */
struct kai_matmul_ukernel_f32_qa8dxp_qs8cxp {
  kai_matmul_clamp_f32_qai8dxp_qsi8cxp_ukernel ukernel;
  std::string name = {};
};

kai_matmul_ukernel_f32_qa8dxp_qs8cxp ukernel_variants[] = {
  {kai_get_m_step_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
   kai_get_n_step_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
   kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
   kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
   kai_get_kr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
   kai_get_sr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
   kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
   kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
   kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
   kai_get_dst_size_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
   kai_run_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
   "matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod"},
  {kai_get_m_step_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
   kai_get_n_step_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
   kai_get_mr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
   kai_get_nr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
   kai_get_kr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
   kai_get_sr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
   kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
   kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
   kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
   kai_get_dst_size_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
   kai_run_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
   "matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod"},
  {kai_get_m_step_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
   kai_get_n_step_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
   kai_get_mr_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
   kai_get_nr_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
   kai_get_kr_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
   kai_get_sr_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
   kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
   kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
   kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
   kai_get_dst_size_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
   kai_run_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
   "matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod"},
#if defined(__ARM_FEATURE_MATMUL_INT8)
  {kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
   kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
   kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
   kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
   kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
   kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
   kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
   kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
   kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
   kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
   kai_run_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
   "matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm"},
#endif

};

const size_t num_ukernel_variants =
  sizeof(ukernel_variants) / sizeof(ukernel_variants[0]);
} // namespace

namespace nntrainer {
std::string __kai_get_num_ukernel_name_qai8dxp_qsi8cxp(size_t idx_variant) {
  return ukernel_variants[idx_variant].name;
}

size_t __kai_get_num_ukernel_variants_qai8dxp_qsi8cxp() {
  return num_ukernel_variants;
}

size_t __kai_get_rhs_packed_size_qsi8cxp_qsi8cx(size_t n, size_t k,
                                                size_t idx_variant) {
  const size_t nr = ukernel_variants[idx_variant].ukernel.get_nr();
  const size_t kr = ukernel_variants[idx_variant].ukernel.get_kr();
  const size_t sr = ukernel_variants[idx_variant].ukernel.get_sr();

  return kai_get_rhs_packed_size_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(n, k, nr, kr,
                                                                  sr);
}

void __kai_rhs_pack_qsi8cxp_qsi8cx(size_t n, size_t k,
                                   void *rhs_packed_mtx_qs8cx,
                                   void *rhs_native_mtx_qs8cx,
                                   void *rhs_scales_f32, size_t idx_variant) {
  const size_t nr = ukernel_variants[idx_variant].ukernel.get_nr();
  const size_t kr = ukernel_variants[idx_variant].ukernel.get_kr();
  const size_t sr = ukernel_variants[idx_variant].ukernel.get_sr();

  struct kai_rhs_pack_qsi8cx_params params;
  params.lhs_zero_point = 1;
  params.scale_multiplier = 1.0f;

  kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(
    1, n, k, nr, kr, sr,                    // Packing arguments
    (const int8_t *)(rhs_native_mtx_qs8cx), // RHS
    NULL,                                   // Bias
    (const float *)(rhs_scales_f32),        // Scale
    rhs_packed_mtx_qs8cx,                   // RHS packed
    0, &params);
}

void __kai_gemm_qai8dxp_qsi8cxp_rhs_unpacked(
  size_t m, size_t n, size_t k, void *lhs_native_mtx_f32,
  void *rhs_native_mtx_qs8cx, void *rhs_scales_f32, float *dst_act_mtx_f32,
  size_t idx_variant, float lower_bound, float upper_bound) {
  // Get the size in bytes for the packed matrices
  const size_t rhs_packed_size =
    __kai_get_rhs_packed_size_qsi8cxp_qsi8cx(n, k, idx_variant);

  // Allocate the matrices
  uint8_t *rhs_packed_mtx_qs8cx = new uint8_t[rhs_packed_size];

  // RHS packing
  __kai_rhs_pack_qsi8cxp_qsi8cx(n, k, rhs_packed_mtx_qs8cx,
                                rhs_native_mtx_qs8cx, rhs_scales_f32,
                                idx_variant);

  // call packed gemm
  __kai_gemm_qai8dxp_qsi8cxp(m, n, k, lhs_native_mtx_f32, rhs_packed_mtx_qs8cx,
                             dst_act_mtx_f32, idx_variant, lower_bound,
                             upper_bound);

  delete[] rhs_packed_mtx_qs8cx;
}

void __kai_gemm_qai8dxp_qsi8cxp(size_t m, size_t n, size_t k,
                                void *lhs_native_mtx_f32,
                                void *rhs_packed_mtx_qs8cx,
                                float *dst_act_mtx_f32, size_t idx_variant,
                                float lower_bound, float upper_bound) {
  const size_t mr = ukernel_variants[idx_variant].ukernel.get_mr();
  const size_t kr = ukernel_variants[idx_variant].ukernel.get_kr();
  const size_t sr = ukernel_variants[idx_variant].ukernel.get_sr();

  // LHS packing
  const size_t lhs_packed_size =
    kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(m, k, mr, kr, sr);
  uint8_t *lhs_packed_mtx_qa8dx = new uint8_t[lhs_packed_size];

  size_t m_step = ukernel_variants[idx_variant].ukernel.get_m_step();
  size_t m_loop = (m + m_step - 1) / m_step;

  size_t n_step = ukernel_variants[idx_variant].ukernel.get_n_step();
  size_t n_loop = (n + n_step - 1) / n_step;

  auto &tm = nntrainer::ThreadManager::Global();

  const size_t dst_stride = n * sizeof(float);

  // LHS quantize + pack, parallel over rows
  tm.parallel_for(0, m_loop, [&](size_t i) {
    size_t m_start = i * m_step;
    size_t m_to_process = std::min(m_step, m - m_start);

    size_t lhs_offset = m_start * k;
    float *lhs_ptr = (float *)lhs_native_mtx_f32 + lhs_offset;

    const size_t lhs_packed_offset =
      ukernel_variants[idx_variant].ukernel.get_lhs_packed_offset(m_start, k);
    uint8_t *lhs_packed_ptr = lhs_packed_mtx_qa8dx + lhs_packed_offset;

    // LHS packing
    kai_run_lhs_quant_pack_qai8dxp_f32(m_to_process, k,   // Dimensions
                                       mr, kr, sr, 0,     // Packing arguments
                                       lhs_ptr,           // LHS
                                       k * sizeof(float), // LHS stride
                                       lhs_packed_ptr);   // LHS packed
  });

  if (m <= m_step) {
    // GEMV-like shape: LHS stays cached, parallelize over n only
    tm.parallel_for(0, n_loop, [&](size_t i) {
      size_t n_start = i * n_step;
      size_t n_to_process = std::min(n_step, n - n_start);

      const size_t rhs_packed_offset =
        ukernel_variants[idx_variant].ukernel.get_rhs_packed_offset(n_start, k);
      const void *rhs_packed_ptr =
        (const void *)((const char *)rhs_packed_mtx_qs8cx + rhs_packed_offset);

      const size_t dst_offset =
        ukernel_variants[idx_variant].ukernel.get_dst_offset(0, n_start,
                                                             dst_stride);
      float *dst_ptr = (float *)((uint8_t *)dst_act_mtx_f32 + dst_offset);

      ukernel_variants[idx_variant].ukernel.run_matmul(
        m, n_to_process, k,      // Dimensions
        lhs_packed_mtx_qa8dx,    // LHS packed
        rhs_packed_ptr,          // RHS packed
        dst_ptr,                 // DST
        dst_stride,              // DST stride (row)
        sizeof(float),           // DST stride (col)
        lower_bound, upper_bound // Min and max for the clamp operation
      );
    });

    delete[] lhs_packed_mtx_qa8dx;
    return;
  }

  // Cache-aware 2D tiling
  // - The kernel re-reads the whole RHS panel every m_step rows, so the panel
  //   must stay L2-resident.
  auto ceil_div = [](size_t a, size_t b) { return (a + b - 1) / b; };

  const size_t n_block_bytes =
    ukernel_variants[idx_variant].ukernel.get_rhs_packed_offset(n_step, k);
  const size_t m_block_bytes =
    ukernel_variants[idx_variant].ukernel.get_lhs_packed_offset(m_step, k);

  // Assume that per-core L2 size is 1MB
  constexpr size_t l2_cache_bytes = 1024 * 1024;

  // n: fewest tiles whose RHS panel fits ~2/3 of L2, split evenly
  constexpr size_t rhs_l2_budget = l2_cache_bytes * 2 / 3;
  size_t n_tiles = std::clamp<size_t>(
    ceil_div(n_loop * n_block_bytes, rhs_l2_budget), 1, n_loop);
  size_t n_blocks = ceil_div(n_loop, n_tiles);
  n_tiles = ceil_div(n_loop, n_blocks);

  // m: enough tiles for work stealing to balance uneven cores
  constexpr size_t tasks_per_thread = 16;
  const size_t num_threads = tm.getComputeThreadCount();
  const size_t target_tasks = tasks_per_thread * num_threads;
  size_t m_tiles =
    std::clamp<size_t>(ceil_div(target_tasks, n_tiles), 1, m_loop);
  size_t m_blocks = ceil_div(m_loop, m_tiles);

  // reduce m block so the LHS panel also fits in L2
  const size_t rhs_panel_bytes = n_blocks * n_block_bytes;
  if (rhs_panel_bytes < l2_cache_bytes) {
    m_blocks = std::min(
      m_blocks,
      std::max<size_t>((l2_cache_bytes - rhs_panel_bytes) / m_block_bytes, 1));
  }
  m_tiles = ceil_div(m_loop, m_blocks);

  // increase n block to reach the task target for small m
  if (m_tiles * n_tiles < target_tasks && n_tiles < n_loop) {
    n_tiles = std::min(ceil_div(target_tasks, m_tiles), n_loop);
    n_blocks = ceil_div(n_loop, n_tiles);
    n_tiles = ceil_div(n_loop, n_blocks);
  }

  const size_t m_chunk = m_blocks * m_step;
  const size_t n_chunk = n_blocks * n_step;

  // row-major: consecutive tasks share the LHS panel
  tm.parallel_for(0, m_tiles * n_tiles, [&](size_t t) {
    const size_t ti = t / n_tiles;
    const size_t tj = t % n_tiles;

    const size_t m_start = ti * m_chunk;
    const size_t m_to_process = std::min(m_chunk, m - m_start);
    const size_t n_start = tj * n_chunk;
    const size_t n_to_process = std::min(n_chunk, n - n_start);

    const size_t lhs_packed_offset =
      ukernel_variants[idx_variant].ukernel.get_lhs_packed_offset(m_start, k);
    const uint8_t *lhs_packed_ptr = lhs_packed_mtx_qa8dx + lhs_packed_offset;

    const size_t rhs_packed_offset =
      ukernel_variants[idx_variant].ukernel.get_rhs_packed_offset(n_start, k);
    const void *rhs_packed_ptr =
      (const void *)((const char *)rhs_packed_mtx_qs8cx + rhs_packed_offset);

    const size_t dst_offset =
      ukernel_variants[idx_variant].ukernel.get_dst_offset(m_start, n_start,
                                                           dst_stride);
    float *dst_ptr = (float *)((uint8_t *)dst_act_mtx_f32 + dst_offset);

    ukernel_variants[idx_variant].ukernel.run_matmul(
      m_to_process, n_to_process, k, // Dimensions
      lhs_packed_ptr,                // LHS packed
      rhs_packed_ptr,                // RHS packed
      dst_ptr,                       // DST
      dst_stride,                    // DST stride (row)
      sizeof(float),                 // DST stride (col)
      lower_bound, upper_bound       // Min and max for the clamp operation
    );
  });

  delete[] lhs_packed_mtx_qa8dx;
}

} // namespace nntrainer
