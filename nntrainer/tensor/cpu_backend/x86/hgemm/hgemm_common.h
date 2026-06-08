// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Yonghyeon Cho <dyddyd8574@gmail.com>
 *
 * @file   hgemm_common.h
 * @date   15 May 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Yonghyeon Cho <dyddyd8574@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  Block-size constants for x86 FP16 GEMM
 */

#ifndef __X86_HGEMM_COMMON_H_
#define __X86_HGEMM_COMMON_H_

/// Default outer blocking sizes. Runtime CPU detection can override these for
/// older Intel cores or AMD Zen cores with smaller private L2 caches.
#define X86_HGEMM_M_BLOCKING 1024
#define X86_HGEMM_N_BLOCKING 512
#define X86_HGEMM_K_BLOCKING 256

/// Default C-panel tile sizes used inside the outer blocking panels. These are
/// intentionally smaller than the outer M/N blocks so the FP32 accumulator
/// panel stays cache-resident while K blocks are accumulated.
#define X86_HGEMM_C_M_BLOCKING 192
#define X86_HGEMM_C_N_BLOCKING 128

/// Micro-kernel tile dimensions.
/// 6 x 16 = 12 YMM accumulators + 2 YMM B + 1 YMM A broadcast = 15 of 16 YMMs.
#define X86_HGEMM_MR 6
#define X86_HGEMM_NR 16

namespace nntrainer::avx2::internal {

/** @brief Resolved blocking sizes for the x86 FP16 GEMM panel/tile loops */
struct HgemmBlockSizes {
  /// Outer panel size. The entry point copies/writes one MxN panel of C32 at a
  /// time, so C32 never scales with the full output matrix.
  unsigned int m;
  unsigned int n;

  /// K blocking depth used by the pack/compute loop.
  unsigned int k;

  /// Inner C tile size. K blocks are accumulated while this C tile is hot.
  unsigned int c_m;
  unsigned int c_n;
};

const HgemmBlockSizes &get_hgemm_block_sizes();

} /* namespace nntrainer::avx2::internal */

#endif /* __X86_HGEMM_COMMON_H_ */
