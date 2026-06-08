// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Yonghyeon Cho <dyddyd8574@gmail.com>
 *
 * @file   hgemm_test.h
 * @date   18 May 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Yonghyeon Cho <dyddyd8574@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  Test-only instrumentation for x86 FP16 hgemm
 */

#ifndef __X86_HGEMM_TEST_H_
#define __X86_HGEMM_TEST_H_

#ifdef ENABLE_TEST

#include <cstddef>

namespace nntrainer::avx2::internal::testing {

/**
 * @brief Test/benchmark-only snapshot of the internal hgemm workspace.
 *
 * This is intentionally not part of the public hgemm API. It exists only for
 * unit tests and local benchmarks to verify workspace reuse without relying on
 * noisy process-wide heap allocation counters.
 */
struct HgemmWorkspaceStats {
  std::size_t c32_capacity = 0;
  std::size_t pack_a_capacity = 0;
  std::size_t pack_b_capacity = 0;
  std::size_t scratch_capacity = 0;
  std::size_t c32_realloc_count = 0;
  std::size_t pack_a_realloc_count = 0;
  std::size_t pack_b_realloc_count = 0;
  std::size_t scratch_realloc_count = 0;
  std::size_t total_realloc_count = 0;
  std::size_t total_capacity_bytes = 0;
};

HgemmWorkspaceStats get_hgemm_workspace_stats();
void reset_hgemm_workspace_stats();
void clear_hgemm_workspace();

} // namespace nntrainer::avx2::internal::testing

#endif // ENABLE_TEST

#endif /* __X86_HGEMM_TEST_H_ */
