// SPDX-License-Identifier: Apache-2.0
/**
 * @file	tq2cx_kernels.h
 * @date	02 October 2026
 * @brief	Arch-neutral pieces of the TQ2CX (ternary) batch GEMM: the scalar
 *          reference tile that defines the expected bits and the threaded
 *          batch driver every architecture plugs its tile kernel into.
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs
 *
 * Weight layout (see ternary_tensor.h): N rows of ceil(K / 32) interleaved
 * { uint32 pos; uint32 neg; } units, then N fp32 scales. Bit b of unit u of
 * row n is input tap 32 u + b; pos set means +1, neg set means -1. Unit (n, u)
 * is stored at tq2cxUnitIndex(lay, n, u, units) and the scales start at
 * tq2cxMaskBytes(lay, N, K) (tq2cx_layout.h); the stored layout changes no
 * value and no summation order.
 *
 * Arithmetic contract (every tile implementation must reproduce it exactly):
 *
 *  - The activation is used in fp32 as is (no quantization).
 *  - For output (m, n) one fp32 accumulator starts at +0.0f and visits the
 *    input taps in ascending k:
 *        pos bit set: acc = acc + x[m][k]
 *        neg bit set: acc = acc - x[m][k]
 *        neither    : acc unchanged
 *    and finally C[m][n] = acc * scale[n].
 *  - A SIMD tile may evaluate every tap as fma(x[m][k], w, acc) with
 *    w in {-1, 0, +1}: for w = +-1 the product is exact so the single rounding
 *    of the fma equals the rounding of the add / subtract, and for w = 0 it
 *    adds a signed zero, which leaves acc unchanged because acc can never be
 *    -0.0 (it starts at +0.0 and x - x rounds to +0.0). This holds for finite
 *    activations; an inf / NaN activation on a zero tap would propagate in the
 *    SIMD form only.
 */

#ifndef __NNTRAINER_TQ2CX_KERNELS_H__
#define __NNTRAINER_TQ2CX_KERNELS_H__
#ifdef __cplusplus

#include <cstddef>
#include <cstdint>
#include <vector>

#include <tq2cx_layout.h>

namespace nntrainer {

/**
 * @brief tile kernel: C[m * ldc + n] for m in [m0, m1) and n in [n0, n1)
 *
 * @param A fp32 activation, row m at A + m * lda
 * @param lda activation row stride
 * @param m0 first activation row
 * @param m1 one past the last activation row
 * @param W the whole TQ2CX record (N x K)
 * @param N output channels of @a W (locates the scale tail)
 * @param K input size of @a W
 * @param n0 first output channel
 * @param n1 one past the last output channel
 * @param C output base of this weight
 * @param ldc output row stride
 * @param lay stored layout of @a W
 */
typedef void (*tq2cx_tile_fn)(const float *A, unsigned int lda, unsigned int m0,
                              unsigned int m1, const void *W, unsigned int N,
                              unsigned int K, unsigned int n0, unsigned int n1,
                              float *C, unsigned int ldc,
                              const Tq2cxLayout &lay);

/**
 * @brief scalar reference tile for TQ2CX, defines the expected bits
 */
void tq2cx_tile_ref(const float *A, unsigned int lda, unsigned int m0,
                    unsigned int m1, const void *W, unsigned int N,
                    unsigned int K, unsigned int n0, unsigned int n1, float *C,
                    unsigned int ldc, const Tq2cxLayout &lay);

/**
 * @brief Batched TQ2CX GEMM driver: C_i = A * W_i^T for every weight i
 *
 * Splits (weight, row chunk, column chunk) tasks with tq2cx_for_each_tile() and
 * runs @a tile on each. The result does not depend on the thread count.
 *
 * @param M activation rows
 * @param Ns output channels of each weight
 * @param K shared input size
 * @param A activation, M x K fp32 with row stride lda
 * @param lda row stride of A
 * @param Ws TQ2CX records (N_i x K)
 * @param Cs outputs, M x Ns[i] with row stride ldcs[i]
 * @param ldcs output row strides
 * @param tile tile kernel
 * @param threaded false runs every task on the calling thread
 * @param lay stored layout of every weight in @a Ws
 */
void tq2cx_gemm_batch(unsigned int M, const std::vector<unsigned int> &Ns,
                      unsigned int K, const float *A, unsigned int lda,
                      const std::vector<void *> &Ws,
                      const std::vector<float *> &Cs,
                      const std::vector<unsigned int> &ldcs, tq2cx_tile_fn tile,
                      bool threaded = true,
                      Tq2cxLayout lay = tq2cxStoredLayout());

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __NNTRAINER_TQ2CX_KERNELS_H__ */
