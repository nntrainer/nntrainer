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

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace nntrainer::hgemm::internal {

namespace {

/**
 * @brief Process-wide registry of every thread-local HgemmWorkspace.
 *
 * The workspaces are reused across GEMM calls (that reuse is the whole point of
 * the thread-local cache), so their buffers intentionally outlive a single
 * call. In a thread pool the owning worker threads never terminate, so without
 * a registry the buffers would only be freed at process teardown via
 * thread_local destruction and look like a leak to external tooling.
 * Registering every workspace keeps the live set reachable and lets
 * release_all_hgemm_workspaces() drop the buffers deterministically at a safe
 * teardown point.
 */
class WorkspaceRegistry {
public:
  void add(HgemmWorkspace *ws) {
    std::lock_guard<std::mutex> lock(mutex_);
    workspaces_.push_back(ws);
  }

  void remove(HgemmWorkspace *ws) {
    std::lock_guard<std::mutex> lock(mutex_);
    workspaces_.erase(std::remove(workspaces_.begin(), workspaces_.end(), ws),
                      workspaces_.end());
  }

  void release_all() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (HgemmWorkspace *ws : workspaces_) {
      ws->release_buffers();
    }
  }

private:
  std::mutex mutex_;
  std::vector<HgemmWorkspace *> workspaces_;
};

/// Lazily-constructed registry. Built on the first workspace registration, so
/// it is guaranteed to outlive every workspace it tracks (each workspace
/// destructor unregisters before the registry itself is destroyed at process
/// exit). The first construction also registers an atexit hook so the buffers
/// are released deterministically at teardown, independent of thread_local
/// destruction order.
WorkspaceRegistry &workspace_registry() {
  static WorkspaceRegistry instance;
  static const bool atexit_registered = [] {
    std::atexit([] { workspace_registry().release_all(); });
    return true;
  }();
  (void)atexit_registered;
  return instance;
}

} // namespace

HgemmWorkspace::HgemmWorkspace() { workspace_registry().add(this); }

HgemmWorkspace::~HgemmWorkspace() {
  workspace_registry().remove(this);
  release_buffers();
}

float *HgemmWorkspace::ensure_c32(std::size_t required) {
#ifdef ENABLE_TEST
  if (required > c32_capacity) {
    ++c32_realloc_count;
  }
#endif
  return ensure_buffer(c32, c32_capacity, required);
}

float *HgemmWorkspace::ensure_pack_a(std::size_t required) {
#ifdef ENABLE_TEST
  if (required > pack_a_capacity) {
    ++pack_a_realloc_count;
  }
#endif
  return ensure_buffer(pack_a, pack_a_capacity, required);
}

float *HgemmWorkspace::ensure_pack_b(std::size_t required) {
#ifdef ENABLE_TEST
  if (required > pack_b_capacity) {
    ++pack_b_realloc_count;
  }
#endif
  return ensure_buffer(pack_b, pack_b_capacity, required);
}

float *HgemmWorkspace::ensure_scratch(std::size_t required) {
#ifdef ENABLE_TEST
  if (required > scratch_capacity) {
    ++scratch_realloc_count;
  }
#endif
  return ensure_buffer(scratch, scratch_capacity, required);
}

#ifdef ENABLE_TEST
void HgemmWorkspace::reset_realloc_counts() {
  c32_realloc_count = 0;
  pack_a_realloc_count = 0;
  pack_b_realloc_count = 0;
  scratch_realloc_count = 0;
}
#endif

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
#ifdef ENABLE_TEST
  reset_realloc_counts();
#endif
}

float *HgemmWorkspace::ensure_buffer(float *&buffer, std::size_t &capacity,
                                     std::size_t required) {
  if (required <= capacity) {
    return buffer;
  }

  float *next = aligned_alloc_f32(required);
  aligned_free(buffer);
  buffer = next;
  capacity = required;
  return buffer;
}

HgemmWorkspace &get_hgemm_workspace() {
  thread_local HgemmWorkspace workspace;
  return workspace;
}

void release_all_hgemm_workspaces() { workspace_registry().release_all(); }

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

} /* namespace nntrainer::hgemm::internal */
