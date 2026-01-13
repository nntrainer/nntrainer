// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   custom_dot_wrapper_cuda.h
 * @date   23 December 2024
 * @brief  Custom dot product wrapper for CUDA
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Daekyoung Jung <dk11.jung@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __CUSTOM_DOT_WRAPPER_CUDA_H__
#define __CUSTOM_DOT_WRAPPER_CUDA_H__

#include <tensor.h>

namespace custom {

/**
 * @brief GEMV for INT4 quantized weights
 */
void gemv_int4_cuda(const float *data, const char *mdata,
                    const unsigned short *scales,
                    const unsigned short *input_scales, void *d_quantized_input,
                    void *d_input_scales_temp, float *rdata, unsigned int M,
                    unsigned int N, unsigned int K,
                    unsigned int quantization_group_size);

/**
 * @brief Top-level wrapper for s32x32_wmma_cpasync GEMM implementation.
 * Quantizes FP32 input to INT8 and calls the kernel wrapper.
 */
void gemm_a32_w4_b16x16_s32x32_wmma_cpasync_cuda(
  const float *input, const void *weights, const void *scales,
  void *d_quantized_input, void *d_input_scales_temp, float *output,
  unsigned int M, unsigned int N, unsigned int K,
  unsigned int quantization_group_size);

/**
 * @brief Top-level wrapper for s32x32_wmma_vl GEMM implementation (16x8 block).
 */
void gemm_a32_w4_b8x16_s32x32_wmma_vl_cuda(
  const float *input, const void *weights, const void *scales,
  void *d_quantized_input, void *d_input_scales_temp, float *output,
  unsigned int M, unsigned int N, unsigned int K,
  unsigned int quantization_group_size);

/**
 * @brief Top-level wrapper for s32x32_wmma_cpasync GEMM implementation (16x8
 * block).
 */
void gemm_a32_w4_b8x16_s32x32_wmma_cpasync_cuda(
  const float *input, const void *weights, const void *scales,
  void *d_quantized_input, void *d_input_scales_temp, float *output,
  unsigned int M, unsigned int N, unsigned int K,
  unsigned int quantization_group_size);

/**
 * @brief Top-level wrapper for s32x32_wmma_cpasync GEMM implementation with
 * Split-K.
 */
void gemm_a32_w4_b16x16_s32x32_wmma_cpasync_splitk_cuda(
  const float *input, const void *weights, const void *scales,
  void *d_quantized_input, void *d_input_scales_temp, float *output,
  unsigned int M, unsigned int N, unsigned int K,
  unsigned int quantization_group_size);

/**
 * @brief Top-level wrapper for s32x32_wmma_cpasync_lu GEMM implementation (16x8
 * block).
 */
void gemm_a32_w4_b8x16_s32x32_wmma_cpasync_lu_cuda(
  const float *input, const void *weights, const void *scales,
  void *d_quantized_input, void *d_input_scales_temp, float *output,
  unsigned int M, unsigned int N, unsigned int K,
  unsigned int quantization_group_size);

/**
 * @brief Custom dot product using CUDA
 */
void custom_dot_cuda(nntrainer::Tensor &output, nntrainer::Tensor weight,
                     nntrainer::Tensor input, unsigned int from,
                     unsigned int to);

/**
 * @brief CUDA GEMM for FP32 input and QINT4 weights with INT8 quantization
 * (Default: V14)
 */
void gemm_a32_w4_default_cuda(const float *data, const char *mdata,
                              const unsigned short *scales,
                              const unsigned short *input_scales,
                              void *d_quantized_input,
                              void *d_input_scales_temp, float *rdata,
                              unsigned int M, unsigned int N, unsigned int K,
                              unsigned int quantization_group_size);

/**
 * @brief Naive CUDA GEMM (v2)
 */
void gemm_a32_w4_b16x16_naive_cuda(const float *input, const void *weights,
                                   const void *scales, float *output,
                                   unsigned int M, unsigned int N,
                                   unsigned int K,
                                   unsigned int quantization_group_size);

/**
 * @brief CUDA GEMM (v3: b32x32_s32x32)
 */
void gemm_a32_w4_b32x32_s32x32_cuda(const float *input, const void *weights,
                                    const void *scales, float *output,
                                    unsigned int M, unsigned int N,
                                    unsigned int K,
                                    unsigned int quantization_group_size);

/**
 * @brief CUDA GEMM (v4: b32x32_s32x32_dequant)
 */
void gemm_a32_w4_b32x32_s32x32_dequant_cuda(
  const float *input, const void *weights, const void *scales, float *output,
  unsigned int M, unsigned int N, unsigned int K,
  unsigned int quantization_group_size);

/**
 * @brief CUDA GEMM (v5: b32x32_pre_dequant)
 */
void gemm_a32_w4_b32x32_pre_dequant_cuda(
  const float *input, const void *weights, const void *scales,
  void *d_quantized_input, void *d_input_scales_temp, float *output,
  unsigned int M, unsigned int N, unsigned int K,
  unsigned int quantization_group_size);

/**
 * @brief CUDA GEMM (v7: b16x16_s32x32)
 */
void gemm_a32_w4_b16x16_s32x32_cuda(const float *input, const void *weights,
                                    const void *scales, float *output,
                                    unsigned int M, unsigned int N,
                                    unsigned int K,
                                    unsigned int quantization_group_size);

/**
 * @brief CUDA GEMM (v8: b32x32_wmma)
 */
void gemm_a32_w4_b32x32_wmma_cuda(const float *input, const void *weights,
                                  const void *scales, void *d_quantized_input,
                                  void *d_input_scales_temp, float *output,
                                  unsigned int M, unsigned int N,
                                  unsigned int K,
                                  unsigned int quantization_group_size);

/**
 * @brief CUDA GEMM (Packed Block 16x16)
 */
void gemm_a32_w4_b16x16_cuda(const float *input, const void *weights,
                             const void *scales, void *d_quantized_input,
                             void *d_input_scales_temp, float *output,
                             unsigned int M, unsigned int N, unsigned int K,
                             unsigned int quantization_group_size);

/**
 * @brief CUDA GEMM using WMMA (b16x16 s32x32)
 */
void gemm_a32_w4_b16x16_s32x32_wmma_cuda(
  const float *input, const void *weights, const void *scales,
  void *d_quantized_input, void *d_input_scales_temp, float *output,
  unsigned int M, unsigned int N, unsigned int K,
  unsigned int quantization_group_size);

/**
 * @brief CUDA GEMM using WMMA (b16x16 s64x64)
 */
void gemm_a32_w4_b16x16_s64x64_wmma_cuda(
  const float *input, const void *weights, const void *scales,
  void *d_quantized_input, void *d_input_scales_temp, float *output,
  unsigned int M, unsigned int N, unsigned int K,
  unsigned int quantization_group_size);

/**
 * @brief CUDA GEMM using WMMA (b16x16 s32x32 vector load)
 */
void gemm_a32_w4_b16x16_s32x32_wmma_vl_cuda(
  const float *input, const void *weights, const void *scales,
  void *d_quantized_input, void *d_input_scales_temp, float *output,
  unsigned int M, unsigned int N, unsigned int K,
  unsigned int quantization_group_size);

/**
 * @brief CUDA GEMM using WMMA (b16x16 packed)
 */
void gemm_a32_w4_b16x16_wmma_cuda(const float *input, const void *weights,
                                  const void *scales, void *d_quantized_input,
                                  void *d_input_scales_temp, float *output,
                                  unsigned int M, unsigned int N,
                                  unsigned int K,
                                  unsigned int quantization_group_size);

} // namespace custom

#endif // __CUSTOM_DOT_WRAPPER_CUDA_H__
