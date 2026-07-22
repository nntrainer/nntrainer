// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Michal Wlasiuk <testmailsmtp12345@gmail.com>
 * Copyright (C) 2025 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   ggml_interface.h
 * @date   15 April 2025
 * @see    https://github.com/nntrainer/nntrainer
 * @author Michal Wlasiuk <testmailsmtp12345@gmail.com>
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Function interface to use ggml lib from cpu_backend
 */

#ifndef __GGML_INTERFACE_H__
#define __GGML_INTERFACE_H__
#ifdef __cplusplus

#include <conv_indirect.h>
#include <stdint.h>
#include <stdlib.h>
#include <tensor_dim.h>

namespace nntrainer {

/**
 * @brief Initialization of ggml backend
 */
void __ggml_init();

/**
 * @brief Quantize float to q4_0 Quantization format
 *
 * @param src input src to be quantized
 * @param dst output destination for quantized data
 * @param nrow number of row
 * @param n_per_row number of elements per row
 * @param quant_weights additional information for quantization. Currently in no
 * use.
 * @return size_t total size of quantized data
 */
size_t __ggml_quantize_q4_0(const float *src, void *dst, int64_t nrow,
                            int64_t n_per_row, const float *quant_weights);

/**
 * @brief Quantize float to q8_0 Quantization format
 *
 * @param src input src to be quantized
 * @param dst output destination for quantized data
 * @param nrow number of row
 * @param n_per_row number of elements per row
 * @param quant_weights additional information for quantization. Currently in no
 * use.
 * @return size_t total size of quantized data
 */
size_t __ggml_quantize_q8_0(const float *src, void *dst, int64_t nrow,
                            int64_t n_per_row, const float *quant_weights);

/**
 * @brief Quantize float to q4_K Quantization format
 *
 * @param src input src to be quantized
 * @param dst output destination for quantized data
 * @param nrow number of row
 * @param n_per_row number of elements per row
 * @param quant_weights additional information for quantization. Currently in no
 * use.
 * @return size_t total size of quantized data
 */
size_t __ggml_quantize_q4_K(const float *src, void *dst, int64_t nrow,
                            int64_t n_per_row, const float *quant_weights);
/**
 * @brief Quantize float to q6_K Quantization format
 *
 * @param src input src to be quantized
 * @param dst output destination for quantized data
 * @param nrow number of row
 * @param n_per_row number of elements per row
 * @param quant_weights additional information for quantization. Currently in no
 * use.
 * @return size_t total size of quantized data
 */
size_t __ggml_quantize_q6_K(const float *src, void *dst, int64_t nrow,
                            int64_t n_per_row, const float *quant_weights);

/**
 * @brief Quantize float to q6_K Quantization format
 *
 * @param src float* src to be quantized
 * @param dst void* dst to store quantized data
 * @param k number of elements in src
 */
void __ggml_quantize_row_q6_K(const float *src, void *dst, int64_t k);

/**
 * @brief Quantize float to q6_K Quantization format
 *
 * @param src float* src to be quantized
 * @param dst void* dst to store quantized data
 * @param k number of elements in src
 */
template <typename T = float>
void __ggml_quantize_row_q8_K(const T *src, void *dst, int64_t k);

/**
 * @brief A(M, K) * W.T(N, K) = (M, N)
 *
 * @param M as descripted above
 * @param N as descripted above
 * @param K as descripted above
 * @param A Activation
 * @param lda leading dimension of A
 * @param B offline quantized and packed q4_0x8 Weight
 * @param ldb leading dimension of B
 * @param C dst matrix
 * @param ldc leading dimension of C
 */
void __ggml_q4_0_8x8_q8_0_GEMM(const unsigned int M, const unsigned int N,
                               const unsigned int K, const float *A,
                               const unsigned int lda, const void *B,
                               const unsigned int ldb, float *C,
                               const unsigned int ldc);

/**
 * @brief Q8_0 weight x Q8_0 activation GEMM dispatcher.
 */
void __ggml_q8_0_q8_0_GEMM(const unsigned int M, const unsigned int N,
                           const unsigned int K, const float *A,
                           const unsigned int lda, const void *B,
                           const unsigned int ldb, float *C,
                           const unsigned int ldc);

/**
 * @brief A(M, K) * W.T(N, K) = (M, N)
 *
 * @param M as descripted above
 * @param N as descripted above
 * @param K as descripted above
 * @param A Activation
 * @param lda leading dimension of A
 * @param Bs vector of offline quantized and packed q4_0x8 Weights
 * @param ldbs vector of leading dimension of B
 * @param C vector of dst matrices
 * @param ldcs vector of leading dimension of C
 */
template <typename T = float>
void __ggml_q4_0_8x8_q8_0_GEMM(const unsigned int M,
                               std::vector<unsigned int> Ns,
                               const unsigned int K, const T *A,
                               const unsigned int lda, std::vector<void *> Bs,
                               std::vector<unsigned int> ldbs,
                               std::vector<T *> C,
                               std::vector<unsigned int> ldcs);

/**
 * @brief A(M, K) * W.T(N, K) = (M, N)
 *
 * @param M as descripted above
 * @param N as descripted above
 * @param K as descripted above
 * @param A Activation
 * @param lda leading dimension of A
 * @param B offline quantized and packed q4_0x4 Weight
 * @param ldb leading dimension of B
 * @param C dst matrix
 * @param ldc leading dimension of C
 */
template <typename T = float>
void __ggml_q4_0_4x8_q8_0_GEMM(const unsigned int M, const unsigned int N,
                               const unsigned int K, const T *A,
                               const unsigned int lda, const void *B,
                               const unsigned int ldb, T *C,
                               const unsigned int ldc);

/**
 * @brief Indirect (im2col-fused) Q4_0 x Q8_0 convolution GEMM.
 *
 * Computes out[M, N] = act[M, K] . W[K, N] where the activation matrix act is
 * the im2col expansion of the NCHW input `in` (M = OH*OW, K = in_ch*kh*kw) but
 * is never materialized: each Q8_0 activation tile is gathered from `in` via
 * `geom` and quantized on the fly, then fed to the same q4_0_4x8_q8_0
 * micro-kernels. Result is bit-identical to materializing im2col and calling
 * __ggml_q4_0_4x8_q8_0_GEMM. ARM (i8mm) path only.
 *
 * @param M  output spatial size (OH*OW)
 * @param N  output channels
 * @param K  in_ch * kernel_h * kernel_w
 * @param in batch-sliced contiguous NCHW input
 * @param geom im2col gather geometry
 * @param B  offline quantized + packed q4_0 weight [K, N]
 * @param ldb leading dimension of B (unused; kept for signature symmetry)
 * @param C  dst matrix [M, N]
 * @param ldc leading dimension of C
 */
void __ggml_q4_0_4x8_q8_0_indirect_GEMM(const unsigned int M,
                                        const unsigned int N,
                                        const unsigned int K, const float *in,
                                        const ConvGatherParams &geom,
                                        const void *B, const unsigned int ldb,
                                        float *C, const unsigned int ldc);

#ifdef ENABLE_FP16
/**
 * @brief FP16-activation indirect (im2col-fused) Q4_0 conv GEMM.
 *
 * Mirror of the FP32 __ggml_q4_0_4x8_q8_0_indirect_GEMM for W4A16 / FP16
 * activations: the activation A = [M=OH*OW, K=CRS] is never materialized; each
 * Q8_0 activation tile is gathered directly from the NCHW _FP16 input via
 * gather_conv_act_rows_fp16 (byte-identical im2col rows, FP16 typed so no FP32
 * staging copy) and quantized on the fly with the FP16-input quantizers, then
 * fed to the same FP16-output micro-kernels the non-indirect FP16 GEMM uses
 * (nntr_gemm_q4_0_4x8_q8_0_fp16 / nntr_gemv_q4_0_4x8_q8_0 + __copy_f16_from_f32
 * tail). Output C is FP16. Used by HalfTensor::convQ4_0Indirect.
 *
 * @param in batch-sliced contiguous NCHW _FP16 input
 * @param C  dst matrix [M, N] of _FP16
 */
void __ggml_q4_0_4x8_q8_0_indirect_GEMM_fp16(
  const unsigned int M, const unsigned int N, const unsigned int K,
  const _FP16 *in, const ConvGatherParams &geom, const void *B,
  const unsigned int ldb, _FP16 *C, const unsigned int ldc);

/**
 * @brief Pre-quantized Q8_0 activation indirect Q4_0 conv GEMM (Task B2).
 *
 * Direct Q8_0-activation indirect-conv GEMM. Input `in` is a contiguous
 * pre-quantized block_q8_0 array of shape [IH_in * IW_in, in_ch_blocks]. We
 * gather blocks directly and pack them into block_q8_0x4 interleaved format,
 * bypassing all dequantization and re-quantization steps. Output C is FP16.
 */
void __ggml_q4_0_4x8_q8_0_indirect_GEMM_q8_0(
  const unsigned int M, const unsigned int N, const unsigned int K,
  const void *in, const ConvGatherParams &geom, const void *B,
  const unsigned int ldb, _FP16 *C, const unsigned int ldc);

/**
 * @brief Q8_0-weight indirect-conv GEMM. Same as the Q4_0 variant above, but
 * the weight operand is a plain block_q8_0 array [N, K/32] (8-bit weights).
 * Both operands are Q8_0; output C is FP16.
 */
void __ggml_q8_0_q8_0_indirect_GEMM_q8_0(const unsigned int M,
                                         const unsigned int N,
                                         const unsigned int K, const void *in,
                                         const ConvGatherParams &geom,
                                         const void *B, const unsigned int ldb,
                                         _FP16 *C, const unsigned int ldc);

/**
 * @brief FP16-activation Q8_0-weight indirect-conv GEMM (W8A16).
 *
 * FP16 mirror of __ggml_q8_0_q8_0_indirect_GEMM_q8_0: the activation is a
 * contiguous NCHW _FP16 input gathered on the fly (gather_conv_act_rows_fp16)
 * and quantized to plain block_q8_0 with the same FP16-input quantizer the
 * working Q4_0 FP16 path uses, then fed to the plain Q8_0xQ8_0 primitive
 * (nntr_gemm_q8_0_q8_0). Weight B is a plain block_q8_0 array [N, K/32]; output
 * C is FP16. Used by HalfTensor::convQ4_0Indirect for Q8_0 weights.
 */
void __ggml_q8_0_q8_0_indirect_GEMM_fp16(const unsigned int M,
                                         const unsigned int N,
                                         const unsigned int K, const _FP16 *in,
                                         const ConvGatherParams &geom,
                                         const void *B, const unsigned int ldb,
                                         _FP16 *C, const unsigned int ldc);
#endif

/**
 * @brief A(M, K) * W.T(N, K) = (M, N)
 *
 * @param M as descripted above
 * @param N as descripted above
 * @param K as descripted above
 * @param A Activation
 * @param lda leading dimension of A
 * @param Bs vector of offline quantized and packed q4_0x4 Weights
 * @param ldbs vector of leading dimension of B
 * @param C vector of dst matrices
 * @param ldcs vector of leading dimension of C
 */
template <typename T = float>
void __ggml_q4_0_4x8_q8_0_GEMM(const unsigned int M,
                               std::vector<unsigned int> Ns,
                               const unsigned int K, const T *A,
                               const unsigned int lda, std::vector<void *> Bs,
                               std::vector<unsigned int> ldbs,
                               std::vector<T *> C,
                               std::vector<unsigned int> ldcs);
/**
 * @brief A(M, K) * W.T(N, K) = (M, N)
 *
 * @param M as descripted above
 * @param N as descripted above
 * @param K as descripted above
 * @param A Activation
 * @param lda leading dimension of A
 * @param B offline quantized and packed q4_kx8 Weight
 * @param ldb leading dimension of B
 * @param C dst matrix
 * @param ldc leading dimension of C
 */
void __ggml_q4_K_8x8_q8_K_GEMM(const unsigned int M, const unsigned int N,
                               const unsigned int K, const float *A,
                               const unsigned int lda, const void *B,
                               const unsigned int ldb, float *C,
                               const unsigned int ldc);

/**
 * @brief A(M, K) * W.T(N, K) = (M, N)
 *
 * @param M as descripted above
 * @param N as descripted above
 * @param K as descripted above
 * @param A Activation
 * @param lda leading dimension of A
 * @param Bs vector of offline quantized and packed q4_kx8 Weights
 * @param ldbs vector of leading dimension of B
 * @param C vector of dst matrices
 * @param ldcs vector of leading dimension of C
 */
void __ggml_q4_K_8x8_q8_K_GEMM(const unsigned int M,
                               std::vector<unsigned int> Ns,
                               const unsigned int K, const float *A,
                               const unsigned int lda, std::vector<void *> Bs,
                               std::vector<unsigned int> ldbs,
                               std::vector<float *> C,
                               std::vector<unsigned int> ldcs);
/**
 * @brief A(M, K) * W.T(N, K) = (M, N)
 *
 * @param M as descripted above
 * @param N as descripted above
 * @param K as descripted above
 * @param A Activation
 * @param lda leading dimension of A
 * @param B offline quantized and packed q4_kx8 Weight
 * @param ldb leading dimension of B
 * @param C dst matrix
 * @param ldc leading dimension of C
 */
template <typename T = float>
void __ggml_gemm_q6_K(const unsigned int M, const unsigned int N,
                      const unsigned int K, const T *A, const unsigned int lda,
                      const void *B, const unsigned int ldb, T *C,
                      const unsigned int ldc);
/**
 * @brief (1xK)*(Kx1) dot product for q6_K and q8_K vectors
 *
 * @param K Length of vectors
 * @param v_q6_K lhs vector - data stored in Q6_K format
 * @param v_q8_K rhs vector - data stored in Q8_K format
 * @return float Result of performing dot operation on v_q6_K and v_q8_K
 */
float __ggml_vec_dot_q6_K_q8_K(const unsigned int K, const void *v_q6_K,
                               const void *v_q8_K);
/**
 * @brief (1xK)*(Kx1) dot product for q6_K and q8_K vectors
 *
 * @param K Length of vectors
 * @param v_q6_K lhs vector - data stored in Q6_K format
 * @param f rhs vector - data stored in float
 * @return float Result of performing dot operation on v_q6_K and f
 */
float __ggml_vec_dot_q6_K_f32(const unsigned int K, const void *v_q6_K,
                              const float *f);

/**
 * @brief q4_0 to float dequantize
 *
 * @param x_raw input src to be dequantized
 * @param y output destination for dequantized data
 * @param k data length
 */
void __ggml_dequantize_row_q4_0(const void *x_raw, float *y, int64_t k);

/**
 * @brief q8_0 to float dequantize
 *
 * @param x_raw input src to be dequantized
 * @param y output destination for dequantized data
 * @param k data length
 */
void __ggml_dequantize_row_q8_0(const void *x_raw, float *y, int64_t k);

/**
 * @brief q4K to float dequantize
 *
 * @param x_raw input src to be dequantized
 * @param y output destination for dequantized data
 * @param k data length
 */
void __ggml_dequantize_row_q4_K(const void *x_raw, float *y, int64_t k);

/**
 * @brief dequantize row of q6_K data to float
 *
 * @param x input to be dequantized from q6_K to float
 * @param y dequantized data output
 * @param k number of elements in x
 */
void __ggml_dequantize_row_q6_K(const void *x, float *y, int64_t k);

/**
 * @brief q8K to float dequantize
 *
 * @param x_raw input src to be dequantized
 * @param y output destination for dequantized data
 * @param k data length
 */
template <typename T = float>
void __ggml_dequantize_row_q8_K(const void *x, T *y, int64_t k);

/**
 * @brief repack q40 to q40x4
 *
 * @param dst output repacked q40x4
 * @param src input q40
 * @param data_size total weight size
 * @param M number of rows
 * @param N number of columns
 */
void __ggml_repack_q4_0_to_q4_0_4(void *dst, void *src, size_t data_size,
                                  const unsigned int M, const unsigned int N);
/**
 * @brief repack q40 to q40x8
 *
 * @param dst output repacked q40x8
 * @param src input q40
 * @param data_size total weight size
 * @param M number of rows
 * @param N number of columns
 */
void __ggml_repack_q4_0_to_q4_0_8(void *dst, void *src, size_t data_size,
                                  const unsigned int M, const unsigned int N);
/**
 * @brief repack q4K to q4Kx8
 *
 * @param dst output repacked q4Kx8
 * @param src input q4K
 * @param data_size total weight size
 * @param M number of rows
 * @param N number of columns
 */
void __ggml_repack_q4_K_to_q4_K_8(void *dst, void *src, size_t data_size,
                                  const unsigned int M, const unsigned int N);

#ifdef ENABLE_FP16
/**
 * @brief Quantize float to q6_K Quantization format
 *
 * @param src float* src to be quantized
 * @param dst void* dst to store quantized data
 * @param k number of elements in src
 */
void __ggml_quantize_row_q8_0(const _FP16 *__restrict src, void *__restrict dst,
                              int64_t k);

/**
 * @brief Quantize _FP16 to q8_0 Quantization format
 *
 * @param src input src to be quantized
 * @param dst output destination for quantized data
 * @param nrow number of row
 * @param n_per_row number of elements per row
 * @param quant_weights additional information for quantization. Currently in no
 * use.
 * @return size_t total size of quantized data
 */
size_t __ggml_quantize_q8_0(const _FP16 *src, void *dst, int64_t nrow,
                            int64_t n_per_row, const float *quant_weights);
/**
 * @brief q8_0 to _FP16 dequantize
 *
 * @param x_raw input src to be dequantized
 * @param y output destination for dequantized data
 * @param k data length
 */
void __ggml_dequantize_row_q8_0(const void *x_raw, _FP16 *y, int64_t k);
#endif
} // namespace nntrainer

#endif
#endif
