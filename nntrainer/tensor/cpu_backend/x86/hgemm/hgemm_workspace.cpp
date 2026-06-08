// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Yonghyeon Cho <dyddyd8574@gmail.com>
 *
 * @file   hgemm_workspace.cpp
 * @date   01 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Yonghyeon Cho <dyddyd8574@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  Thread-local scratch workspace for x86 FP16 GEMM
 */

#include "hgemm_workspace.h"

#ifdef ENABLE_TEST
#include "hgemm_test.h"
#endif
#include "hgemm_util.h"

namespace nntrainer::avx2::internal {

HgemmWorkspace::~HgemmWorkspace() { release_buffers(); }

float *HgemmWorkspace::ensure_c32(std::size_t required) {
  return ensure_buffer(c32, c32_capacity, c32_realloc_count, required);
}

float *HgemmWorkspace::ensure_pack_a(std::size_t required) {
  return ensure_buffer(pack_a, pack_a_capacity, pack_a_realloc_count, required);
}

float *HgemmWorkspace::ensure_pack_b(std::size_t required) {
  return ensure_buffer(pack_b, pack_b_capacity, pack_b_realloc_count, required);
}

float *HgemmWorkspace::ensure_scratch(std::size_t required) {
  return ensure_buffer(scratch, scratch_capacity, scratch_realloc_count,
                       required);
}

void HgemmWorkspace::reset_realloc_counts() {
  c32_realloc_count = 0;
  pack_a_realloc_count = 0;
  pack_b_realloc_count = 0;
  scratch_realloc_count = 0;
}

void HgemmWorkspace::release_buffers() {
  aligned_free(pack_b);
  aligned_free(pack_a);
  aligned_free(scratch);
  aligned_free(c32);
  pack_b = nullptr;
  pack_a = nullptr;
  scratch = nullptr;
  c32 = nullptr;
  c32_capacity = 0;
  pack_a_capacity = 0;
  pack_b_capacity = 0;
  scratch_capacity = 0;
  reset_realloc_counts();
}

float *HgemmWorkspace::ensure_buffer(float *&buffer, std::size_t &capacity,
                                     std::size_t &realloc_count,
                                     std::size_t required) {
  if (required <= capacity) {
    return buffer;
  }

  float *next = aligned_alloc_f32(required);
  aligned_free(buffer);
  buffer = next;
  capacity = required;
  ++realloc_count;
  return buffer;
}

HgemmWorkspace &get_hgemm_workspace() {
  thread_local HgemmWorkspace workspace;
  return workspace;
}

#ifdef ENABLE_TEST
namespace testing {

HgemmWorkspaceStats get_hgemm_workspace_stats() {
  const HgemmWorkspace &workspace = get_hgemm_workspace();
  HgemmWorkspaceStats stats;
  stats.c32_capacity = workspace.c32_capacity;
  stats.pack_a_capacity = workspace.pack_a_capacity;
  stats.pack_b_capacity = workspace.pack_b_capacity;
  stats.scratch_capacity = workspace.scratch_capacity;
  stats.c32_realloc_count = workspace.c32_realloc_count;
  stats.pack_a_realloc_count = workspace.pack_a_realloc_count;
  stats.pack_b_realloc_count = workspace.pack_b_realloc_count;
  stats.scratch_realloc_count = workspace.scratch_realloc_count;
  stats.total_realloc_count =
    stats.c32_realloc_count + stats.pack_a_realloc_count +
    stats.pack_b_realloc_count + stats.scratch_realloc_count;
  stats.total_capacity_bytes =
    (stats.c32_capacity + stats.pack_a_capacity + stats.pack_b_capacity +
     stats.scratch_capacity) *
    sizeof(float);
  return stats;
}

void reset_hgemm_workspace_stats() {
  get_hgemm_workspace().reset_realloc_counts();
}

void clear_hgemm_workspace() { get_hgemm_workspace().release_buffers(); }

} // namespace testing
#endif

} /* namespace nntrainer::avx2::internal */
