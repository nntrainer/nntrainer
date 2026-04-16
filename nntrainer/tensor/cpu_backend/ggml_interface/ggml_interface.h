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
 * @param B offline quantized and packed q4_0x8 Weight
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
 * @brief Quantize float to Q1_0 (1-bit, group size 128) format
 *
 * @param src input src to be quantized
 * @param dst output destination for quantized data
 * @param nrow number of rows
 * @param n_per_row number of elements per row
 * @param quant_weights additional information for quantization (unused)
 * @return size_t total size of quantized data
 */
size_t __ggml_quantize_q1_0(const float *src, void *dst, int64_t nrow,
                            int64_t n_per_row, const float *quant_weights);

/**
 * @brief Q1_0 to float dequantize
 *
 * @param x_raw input src to be dequantized
 * @param y output destination for dequantized data
 * @param k data length
 */
void __ggml_dequantize_row_q1_0(const void *x_raw, float *y, int64_t k);

/**
 * @brief (1xK)*(Kx1) dot product for Q1_0 weights and Q8_0 activations
 *
 * @param K Length of vectors
 * @param v_q1_0 lhs vector - data stored in Q1_0 format
 * @param v_q8_0 rhs vector - data stored in Q8_0 format
 * @return float Result of dot product
 */
float __ggml_vec_dot_q1_0_q8_0(const unsigned int K, const void *v_q1_0,
                               const void *v_q8_0);

/**
 * @brief (1xK)*(Kx1) dot product for Q1_0 weights and float activations
 *
 * @param K Length of vectors
 * @param v_q1_0 lhs vector - data stored in Q1_0 format
 * @param f rhs vector - data stored in float
 * @return float Result of dot product
 */
float __ggml_vec_dot_q1_0_f32(const unsigned int K, const void *v_q1_0,
                              const float *f);

/**
 * @brief A(M, K) * W.T(N, K) = (M, N) for Q1_0 weights
 *
 * @param M rows of A
 * @param N rows of W (columns of output)
 * @param K columns of A / W
 * @param A Activation (float)
 * @param lda leading dimension of A
 * @param B Q1_0 quantized weight
 * @param ldb leading dimension of B
 * @param C output matrix (float)
 * @param ldc leading dimension of C
 */
void __ggml_q1_0_GEMM(const unsigned int M, const unsigned int N,
                      const unsigned int K, const float *A,
                      const unsigned int lda, const void *B,
                      const unsigned int ldb, float *C,
                      const unsigned int ldc);

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
