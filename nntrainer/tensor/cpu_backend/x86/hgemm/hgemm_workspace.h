// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Yonghyeon Cho <dyddyd8574@gmail.com>
 *
 * @file   hgemm_workspace.h
 * @date   01 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Yonghyeon Cho <dyddyd8574@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  Thread-local scratch workspace for x86 FP16 GEMM
 */

#ifndef __X86_HGEMM_WORKSPACE_H_
#define __X86_HGEMM_WORKSPACE_H_

#include <cstddef>

namespace nntrainer::avx2::internal {

/** @brief Thread-local scratch buffers reused across x86 FP16 GEMM calls */
struct HgemmWorkspace {
  ~HgemmWorkspace();

  HgemmWorkspace() = default;
  HgemmWorkspace(const HgemmWorkspace &) = delete;
  HgemmWorkspace &operator=(const HgemmWorkspace &) = delete;

  float *ensure_c32(std::size_t required);
  float *ensure_pack_a(std::size_t required);
  float *ensure_pack_b(std::size_t required);
  float *ensure_scratch(std::size_t required);

  void reset_realloc_counts();
  void release_buffers();

  float *c32 = nullptr;
  float *pack_a = nullptr;
  float *pack_b = nullptr;
  float *scratch = nullptr;
  std::size_t c32_capacity = 0;
  std::size_t pack_a_capacity = 0;
  std::size_t pack_b_capacity = 0;
  std::size_t scratch_capacity = 0;
  std::size_t c32_realloc_count = 0;
  std::size_t pack_a_realloc_count = 0;
  std::size_t pack_b_realloc_count = 0;
  std::size_t scratch_realloc_count = 0;

private:
  static float *ensure_buffer(float *&buffer, std::size_t &capacity,
                              std::size_t &realloc_count, std::size_t required);
};

HgemmWorkspace &get_hgemm_workspace();

} /* namespace nntrainer::avx2::internal */

#endif /* __X86_HGEMM_WORKSPACE_H_ */
