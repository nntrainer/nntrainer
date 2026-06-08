// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Yonghyeon Cho <dyddyd8574@gmail.com>
 *
 * @file   hgemm_kernel.h
 * @date   15 May 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Yonghyeon Cho <dyddyd8574@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  Micro-kernel declarations for x86 FP16 GEMM
 */

#ifndef __X86_HGEMM_KERNEL_H_
#define __X86_HGEMM_KERNEL_H_

namespace nntrainer::avx2::internal {

/**
 * @brief Primary 6x16 micro-kernel.
 *
 * Computes C[0..5][0..15] += packed_A * packed_B over @p K iterations.
 * packed_A layout: dst[k * 6 + m].
 * packed_B layout: dst[k * 16 + n].
 *
 * @param K          inner-dimension count
 * @param packed_A   FP32 packed A stripe (size K * 6)
 * @param packed_B   FP32 packed B stripe (size K * 16)
 * @param C          FP32 accumulator origin (M_tile = 6 rows, N_tile = 16 cols)
 * @param c_stride   row stride of C (in elements)
 */
void hgemm_kernel_6x16(unsigned int K, const float *packed_A,
                       const float *packed_B, float *C, unsigned int c_stride);

void hgemm_kernel_6x8(unsigned int K, const float *packed_A,
                      const float *packed_B, float *C, unsigned int c_stride);

void hgemm_kernel_4x16(unsigned int K, const float *packed_A,
                       const float *packed_B, float *C, unsigned int c_stride);

void hgemm_kernel_4x8(unsigned int K, const float *packed_A,
                      const float *packed_B, float *C, unsigned int c_stride);

void hgemm_kernel_2x16(unsigned int K, const float *packed_A,
                       const float *packed_B, float *C, unsigned int c_stride);

void hgemm_kernel_2x8(unsigned int K, const float *packed_A,
                      const float *packed_B, float *C, unsigned int c_stride);

void hgemm_kernel_1x16(unsigned int K, const float *packed_A,
                       const float *packed_B, float *C, unsigned int c_stride);

void hgemm_kernel_1x8(unsigned int K, const float *packed_A,
                      const float *packed_B, float *C, unsigned int c_stride);

/**
 * @brief Dispatch to a row/column-tail-aware AVX2 micro-kernel.
 *
 * @p M and @p N are the logical tile sizes at the edge of a 6x16 tile. The
 * dispatcher chooses 1/2/4/6 rows and 8/16 columns. The selected kernel may
 * write padded rows or columns inside the already padded C32 buffer, but avoids
 * unnecessary padded traffic for common 1/2/4-row and <=8-column skinny tiles.
 */
void hgemm_kernel_mxn(unsigned int M, unsigned int N, unsigned int K,
                      const float *packed_A, const float *packed_B, float *C,
                      unsigned int c_stride);

} /* namespace nntrainer::avx2::internal */

#endif /* __X86_HGEMM_KERNEL_H_ */
