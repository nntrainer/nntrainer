// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Yonghyeon Cho <dyddyd8574@gmail.com>
 *
 * @file   hgemm_pack.h
 * @date   15 May 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Yonghyeon Cho <dyddyd8574@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  Packing routines for x86 FP16 GEMM (source->FP32 during pack)
 */

#ifndef __X86_HGEMM_PACK_H_
#define __X86_HGEMM_PACK_H_

#include <tensor_dim.h>

namespace nntrainer::hgemm::internal {

/**
 * @brief Pack an MR-row stripe of A (row-major source) into FP32.
 *
 * @c SrcT may be @c _FP16 (converted via F16C while packing) or @c float
 * (re-laid out without conversion), so the same blocking loop drives both the
 * pure-FP16 and the mixed-precision GEMM paths.
 *
 * Source shape: m_actual rows x k_min cols at stride @p src_stride. Output is
 * stored in k-major MR-tile layout:
 * dst[k * MR + m] = alpha * (float) src[m * src_stride + k] for valid rows.
 * Remaining rows (m_actual..MR) are filled with 0.
 *
 * @tparam SrcT source element type (_FP16 or float)
 * @param m_actual number of valid rows (1..MR)
 * @param k_min K dimension of the stripe
 * @param src   source pointer
 * @param src_stride source row stride in elements
 * @param alpha scalar applied while packing A
 * @param dst   FP32 packed buffer, capacity >= k_min * MR
 */
template <typename SrcT>
void packing_A_M6(unsigned int m_actual, unsigned int k_min, const SrcT *src,
                  unsigned int src_stride, float alpha, float *dst);

/**
 * @brief Pack an MR-row stripe of transposed A into FP32.
 *
 * Source is A in transposed storage for op(A): src[k * src_stride + m].
 */
template <typename SrcT>
void packing_A_M6_trans(unsigned int m_actual, unsigned int k_min,
                        const SrcT *src, unsigned int src_stride, float alpha,
                        float *dst);

/**
 * @brief Pack an NR-col stripe of B (row-major source) into FP32.
 *
 * @c SrcT may be @c _FP16 or @c float (see packing_A_M6).
 *
 * Source shape: k_min rows x n_actual cols at stride @p src_stride. Output is
 * stored in k-major NR-tile layout:
 * dst[k * NR + n] = (float) src[k * src_stride + n] for valid cols.
 * Remaining cols (n_actual..NR) are filled with 0.
 *
 * @tparam SrcT source element type (_FP16 or float)
 * @param k_min    K dimension of the stripe
 * @param n_actual number of valid cols (1..NR)
 * @param src      source pointer
 * @param src_stride source row stride in elements
 * @param dst      FP32 packed buffer, capacity >= k_min * NR
 */
template <typename SrcT>
void packing_B_N16(unsigned int k_min, unsigned int n_actual, const SrcT *src,
                   unsigned int src_stride, float *dst);

/**
 * @brief Pack an NR-col stripe of transposed B into FP32.
 *
 * Source is B in transposed storage for op(B): src[n * src_stride + k].
 */
template <typename SrcT>
void packing_B_N16_trans(unsigned int k_min, unsigned int n_actual,
                         const SrcT *src, unsigned int src_stride, float *dst);

} /* namespace nntrainer::hgemm::internal */

#endif /* __X86_HGEMM_PACK_H_ */
