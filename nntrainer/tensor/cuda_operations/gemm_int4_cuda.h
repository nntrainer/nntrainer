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
 * @brief Optimized CUDA implementation of int4 GEMM operation using packed
 * blocks
 *
 * @param input Quantized input data pointer (device memory, int8_t*)
 * @param weights Quantized weight data pointer (device memory, packed int4)
 * @param scales Weight scale data pointer (device memory, fp16)
 * @param input_scales Input scale data pointer (device memory, fp16)
 * @param output Output data pointer (device memory, float*)
 * @param M Number of rows in the matrix
 * @param N Number of columns in the matrix
 * @param K Inner dimension of the matrix multiplication
 * @param quantization_group_size Quantization group size
 */
void gemm_int4_cuda_packed_block(const void *input, const void *weights,
                                 const void *scales, const void *input_scales,
                                 float *output, unsigned int M, unsigned int N,
                                 unsigned int K,
                                 unsigned int quantization_group_size);

/**
 * @brief Optimized CUDA implementation of int4 GEMM operation using packed
 * blocks (16x16)
 *
 * @param input Input data pointer (device)
 * @param weights Weight data pointer (device)
 * @param scales Scale data pointer (device)
 * @param input_scales Input scale data pointer (device)
 * @param output Output data pointer (device)
 * @param M Number of rows in the matrix
 * @param N Number of columns in the matrix
 * @param K Inner dimension of the matrix multiplication
 * @param quantization_group_size Quantization group size
 */
void gemm_int4_cuda_packed_block_16(const void *input, const void *weights,
                                    const void *scales,
                                    const void *input_scales, float *output,
                                    unsigned int M, unsigned int N,
                                    unsigned int K,
                                    unsigned int quantization_group_size);

#endif // __GEMM_INT4_CUDA_H__
