// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   gemm_int4_cpu.h
 * @brief  CPU implementation of int4 GEMM operation
 * @author Samsung Electronics Co., Ltd.
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __GEMM_INT4_CPU_H__
#define __GEMM_INT4_CPU_H__

#include <cstdint>

namespace nntrainer {

/**
 * @brief CPU implementation of int4 GEMM operation
 *
 * @param input Quantized input data pointer (int8_t*)
 * @param weights Quantized weight data pointer (packed int4)
 * @param scales Weight scale data pointer (fp16)
 * @param input_scales Input scale data pointer (fp16)
 * @param output Output data pointer (float*)
 * @param M Number of rows in the matrix
 * @param N Number of columns in the matrix
 * @param K Inner dimension of the matrix multiplication
 * @param quantization_group_size Quantization group size
 */
void gemm_int4_cpu(const void *input, const void *weights, const void *scales,
                   const void *input_scales, float *output, unsigned int M,
                   unsigned int N, unsigned int K,
                   unsigned int quantization_group_size);

/**
 * @brief Optimized CPU implementation of int4 GEMM operation using packed
 * blocks
 *
 * @param input Quantized input data pointer (int8_t*)
 * @param weights Quantized weight data pointer (packed int4)
 * @param scales Weight scale data pointer (fp16)
 * @param input_scales Input scale data pointer (fp16)
 * @param output Output data pointer (float*)
 * @param M Number of rows in the matrix
 * @param N Number of columns in the matrix
 * @param K Inner dimension of the matrix multiplication
 * @param quantization_group_size Quantization group size
 */
void gemm_int4_cpu_packed_block(const void *input, const void *weights,
                                const void *scales, const void *input_scales,
                                float *output, unsigned int M, unsigned int N,
                                unsigned int K,
                                unsigned int quantization_group_size);

} // namespace nntrainer

#endif // __GEMM_INT4_CPU_H__
