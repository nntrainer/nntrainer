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

/// Default outer blocking sizes (in elements). They bound the per-thread panel
/// and the packed-operand working set that has to stay resident in the shared
/// L2/L3 while the micro-kernel streams over it. Runtime CPU detection
/// (tune_block_sizes) overrides these for cores with smaller private L2 caches.
///
/// One outer panel packs B once into N_BLOCKING * K_BLOCKING FP32 =
/// 512 * 256 * 4 B = 512 KiB and every M tile in the panel re-reads it, so this
/// buffer is sized for L2-tier reuse. M_BLOCKING bounds the FP32 C accumulator
/// height (the panel is also the unit of work handed to the thread pool).
#define X86_HGEMM_M_BLOCKING 1024
#define X86_HGEMM_N_BLOCKING 512
/// K depth packed/accumulated per pass. Sized so each packed micro-stripe stays
/// inside a 32 KiB L1D: packed B stripe = NR * K * 4 B = 16 * 256 * 4 = 16 KiB,
/// packed A stripe = MR * K * 4 B = 6 * 256 * 4 = 6 KiB.
#define X86_HGEMM_K_BLOCKING 256

/// Inner C accumulator tile, kept smaller than the outer M/N blocks so the FP32
/// accumulator (C_M * C_N * 4 B = 192 * 128 * 4 = 96 KiB) stays cache-resident
/// while successive K blocks are accumulated into it.
#define X86_HGEMM_C_M_BLOCKING 192
#define X86_HGEMM_C_N_BLOCKING 128

/// Micro-kernel tile dimensions: rows (MR) x cols (NR) of the FP32 C tile
/// produced per kernel call. NR = 16 cols = 2 YMM registers (8 FP32 each),
/// MR = 6 rows. AVX2 has 16 YMM registers; the kernel uses MR * 2 = 12 of them
/// as C accumulators + 2 for the loaded B columns + 1 for the broadcast A
/// scalar = 15 of 16, the largest register-blocked tile that still avoids
/// spilling the accumulators back to memory.
#define X86_HGEMM_MR 6
#define X86_HGEMM_NR 16

namespace nntrainer::hgemm::internal {

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

} /* namespace nntrainer::hgemm::internal */

#endif /* __X86_HGEMM_COMMON_H_ */
