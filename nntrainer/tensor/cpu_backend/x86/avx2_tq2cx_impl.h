// SPDX-License-Identifier: Apache-2.0
/**
 * @file	avx2_tq2cx_impl.h
 * @date	02 October 2026
 * @brief	AVX2 tile kernel for the TQ2CX (ternary) batch GEMM.
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs
 */

#ifndef __NNTRAINER_AVX2_TQ2CX_IMPL_H__
#define __NNTRAINER_AVX2_TQ2CX_IMPL_H__
#ifdef __cplusplus

#include <tq2cx_kernels.h>

namespace nntrainer::avx2 {

/**
 * @brief AVX2 TQ2CX tile, bit-identical to tq2cx_tile_ref for finite
 *        activations
 * @see tq2cx_tile_fn
 */
void tq2cx_tile(const float *A, unsigned int lda, unsigned int m0,
                unsigned int m1, const void *W, unsigned int N, unsigned int K,
                unsigned int n0, unsigned int n1, float *C, unsigned int ldc,
                const Tq2cxLayout &lay);

} // namespace nntrainer::avx2

#endif /* __cplusplus */
#endif /* __NNTRAINER_AVX2_TQ2CX_IMPL_H__ */
