// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd.
 *
 * @file   x86_qsi4cxp.h
 * @brief  AVX2 GEMV/GEMM for channel-wise int4 (qsi4cxp kxn) weights.
 *         Reads Int4QTensor canonical layout directly — no transpose,
 *         no repack, no extra memory.
 */
#ifndef __X86_QSI4CXP_H__
#define __X86_QSI4CXP_H__

#include <cstdint>

namespace nntrainer {

/**
 * @brief  GEMM/GEMV for fp32 activation × channel-wise int4 weight.
 *
 *         C[M,N] = A[M,K] × dequant(B_int4[K,N]) where
 *         dequant(B[k,n]) = (nibble[k,n] - 8) * scale[n]
 *
 * @param M       number of activation rows (batch * seq_len)
 * @param N       number of output columns (out_features)
 * @param K       reduction axis length (in_features)
 * @param activation  fp32 [M, K] row-major
 * @param packed_data qsi4cxp kxn: K rows × ceil(N/2) bytes per row.
 *                    Even n_idx = low nibble, odd = high nibble,
 *                    offset-binary (stored = real + 8).
 * @param scales  fp32 [N] per-output-column scales
 * @param output  fp32 [M, N] row-major, overwritten (not accumulated)
 */
void gemm_qsi4cxp_kxn_fp32(unsigned int M, unsigned int N, unsigned int K,
                            const float *activation,
                            const uint8_t *packed_data,
                            const float *scales, float *output);

} // namespace nntrainer

#endif // __X86_QSI4CXP_H__
