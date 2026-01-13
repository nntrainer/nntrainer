// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file	gemm_int4_cuda.h
 * @date	28 Nov 2025
 * @brief	CUDA implementation of int4 GEMM operation
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	[Your Name] <[your.email@samsung.com]>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __GEMM_INT4_CUDA_H__
#define __GEMM_INT4_CUDA_H__

/**
 * @brief Optimized CUDA implementation (Packed Block 32x32)
 */
void gemm_a8_w4_b32x32(const void *input, const void *weights,
                       const void *scales, const void *input_scales,
                       float *output, unsigned int M, unsigned int N,
                       unsigned int K, unsigned int quantization_group_size);

/**
 * @brief Optimized CUDA implementation (Packed Block 32x32 Pre-Dequant)
 */
void gemm_a8_w4_b32x32_pre_dequant(const void *input, const void *weights,
                                   const void *scales, const void *input_scales,
                                   float *output, unsigned int M,
                                   unsigned int N, unsigned int K,
                                   unsigned int quantization_group_size);

/**
 * @brief CUDA GEMM using WMMA (Block 32x32)
 */
void gemm_a8_w4_b32x32_wmma(const void *input, const void *weights,
                            const void *scales, const void *input_scales,
                            float *output, unsigned int M, unsigned int N,
                            unsigned int K,
                            unsigned int quantization_group_size);

/**
 * @brief Optimized CUDA implementation (Block 16x16 DP4A)
 */
void gemm_a8_w4_b16x16_dp4a(const void *input, const void *weights,
                            const void *scales, const void *input_scales,
                            float *output, unsigned int M, unsigned int N,
                            unsigned int K,
                            unsigned int quantization_group_size);

/**
 * @brief Naive CUDA GEMM Kernel Wrapper (FP32 Input)
 */
void gemm_a32_w4_b16x16_naive(const float *input, const void *weights,
                              const void *scales, float *output, unsigned int M,
                              unsigned int N, unsigned int K,
                              unsigned int quantization_group_size);

/**
 * @brief CUDA GEMM Kernel Wrapper (Block 32x32 Shared 32x32, FP32 Input)
 */
void gemm_a32_w4_b32x32_s32x32(const float *input, const void *weights,
                               const void *scales, float *output,
                               unsigned int M, unsigned int N, unsigned int K,
                               unsigned int quantization_group_size);

/**
 * @brief CUDA GEMM Kernel Wrapper (Block 32x32 Shared 32x32 Dequant, FP32
 * Input)
 */
void gemm_a32_w4_b32x32_s32x32_dequant(const float *input, const void *weights,
                                       const void *scales, float *output,
                                       unsigned int M, unsigned int N,
                                       unsigned int K,
                                       unsigned int quantization_group_size);

/**
 * @brief CUDA GEMM Kernel Wrapper (Block 16x16 Shared 32x32, FP32 Input)
 */
void gemm_a32_w4_b16x16_s32x32(const float *input, const void *weights,
                               const void *scales, float *output,
                               unsigned int M, unsigned int N, unsigned int K,
                               unsigned int quantization_group_size);

/**
 * @brief Optimized CUDA implementation (Block 16x16)
 */
void gemm_a8_w4_b16x16(const void *input, const void *weights,
                       const void *scales, const void *input_scales,
                       float *output, unsigned int M, unsigned int N,
                       unsigned int K, unsigned int quantization_group_size);

/**
 * @brief CUDA GEMM using WMMA (Block 16x16 Shared 32x32)
 */
void gemm_a8_w4_b16x16_s32x32_wmma(const void *input, const void *weights,
                                   const void *scales, const void *input_scales,
                                   float *output, unsigned int M,
                                   unsigned int N, unsigned int K,
                                   unsigned int quantization_group_size);

/**
 * @brief CUDA GEMM using WMMA (Block 16x16 Shared 64x64)
 */
void gemm_a8_w4_b16x16_s64x64_wmma(const void *input, const void *weights,
                                   const void *scales, const void *input_scales,
                                   float *output, unsigned int M,
                                   unsigned int N, unsigned int K,
                                   unsigned int quantization_group_size);

/**
 * @brief CUDA GEMM using WMMA (Block 16x16 Shared 32x32 VL)
 */
void gemm_a8_w4_b16x16_s32x32_wmma_vl(const void *input, const void *weights,
                                      const void *scales,
                                      const void *input_scales, float *output,
                                      unsigned int M, unsigned int N,
                                      unsigned int K,
                                      unsigned int quantization_group_size);

/**
 * @brief CUDA GEMM using WMMA (Block 16x16 Shared 32x32 CPASYNC)
 */
void gemm_a8_w4_b16x16_s32x32_wmma_cpasync(
  const void *input, const void *weights, const void *scales,
  const void *input_scales, float *output, unsigned int M, unsigned int N,
  unsigned int K, unsigned int quantization_group_size);

/**
 * @brief CUDA GEMM using WMMA (Block 8x16 Shared 32x32 VL)
 */
void gemm_a8_w4_b8x16_s32x32_wmma_vl(const void *input, const void *weights,
                                     const void *scales,
                                     const void *input_scales, float *output,
                                     unsigned int M, unsigned int N,
                                     unsigned int K,
                                     unsigned int quantization_group_size);

/**
 * @brief CUDA GEMM using WMMA (Block 16x16 Packed)
 */
void gemm_a8_w4_b16x16_wmma(const void *input, const void *weights,
                            const void *scales, const void *input_scales,
                            float *output, unsigned int M, unsigned int N,
                            unsigned int K,
                            unsigned int quantization_group_size);

/**
 * @brief CUDA GEMM using WMMA (Block 8x16 Shared 32x32 CPASYNC)
 */
void gemm_a8_w4_b8x16_s32x32_wmma_cpasync(
  const void *input, const void *weights, const void *scales,
  const void *input_scales, float *output, unsigned int M, unsigned int N,
  unsigned int K, unsigned int quantization_group_size);

/**
 * @brief CUDA GEMM for INT8/INT4 using WMMA and Split-K (Block 16x16 Shared
 * 32x32)
 */
void gemm_a8_w4_b16x16_s32x32_wmma_cpasync_splitk(
  const void *input, const void *weights, const void *scales,
  const void *input_scales, float *output, unsigned int M, unsigned int N,
  unsigned int K, unsigned int quantization_group_size);

/**
 * @brief CUDA GEMM for INT8/INT4 using WMMA, Double Buffering, Loop Unrolling
 * (Block 8x16 Shared 32x32)
 */
void gemm_a8_w4_b8x16_s32x32_wmma_cpasync_lu(
  const void *input, const void *weights, const void *scales,
  const void *input_scales, float *output, unsigned int M, unsigned int N,
  unsigned int K, unsigned int quantization_group_size);

#endif // __GEMM_INT4_CUDA_H__
