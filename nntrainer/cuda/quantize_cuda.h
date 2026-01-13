// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   quantize_cuda.h
 * @date   13 Jan 2026
 * @brief  Header file for CUDA quantization functions
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Samsung R&D Institute
 * @bug    No known bugs
 */
#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

/**
 * @brief Quantizes FP32 input to INT8 format with padding support (CUDA)
 *
 * This function quantizes FP32 input data to INT8 format in groups.
 * Each group is quantized independently with its own scale factor.
 *
 * @param input Pointer to the input FP32 data array on the device
 * @param quantized_input Pointer to the output INT8 buffer on the device
 * @param scales Pointer to the output UINT16 (FP16) scales buffer on the device
 * @param M Number of rows in the input matrix
 * @param K Number of columns in the input matrix
 * @param quantization_group_size Size of each quantization group (typically 32)
 * @param stream The CUDA stream to execute the kernel on (default: 0)
 */
void quantize_input_int8_pad_cuda(const void *input, void *quantized_input,
                                  void *scales, unsigned int M, unsigned int K,
                                  unsigned int quantization_group_size,
                                  cudaStream_t stream = 0);

/**
 * @brief Quantizes FP32 activation to Q8_1 format (CUDA)
 *
 * This function converts a contiguous array of 32-bit floating point values
 * into the Q8_1 quantization format using CUDA.
 *
 * @param input Pointer to the input FP32 data array.
 * @param output Pointer to the output buffer where Q8_1 blocks will be stored.
 * @param k The number of elements in the input array. Must be a multiple of 32.
 * @param stream The CUDA stream to execute the kernel on (default: 0)
 */
void quantize_activation_q8_1_cuda(const float *input, void *output, int64_t k,
                                   cudaStream_t stream = 0);

/**
 * @brief Quantizes FP32 input to Q8_1 format (CPU Host)
 *
 * This function converts a contiguous array of 32-bit floating point values
 * into the Q8_1 quantization format on the CPU. It is typically used for
 * verification purposes in unit tests.
 *
 * @param x Pointer to the input FP32 data array.
 * @param vy Pointer to the output buffer where Q8_1 blocks will be stored.
 * @param k The number of elements in the input array. Must be a multiple of 32.
 */
void quantize_row_q8_1_host(const float *x, void *vy, int64_t k);

/**
 * @brief Quantizes FP32 input to Q4_0 format (CPU Host)
 *
 * This function converts a contiguous array of 32-bit floating point values
 * into the Q4_0 quantization format on the CPU. It is typically used for
 * verification purposes using GEMM kernels.
 *
 * @param x Pointer to the input FP32 data array.
 * @param vy Pointer to the output buffer where Q4_0 blocks will be stored.
 * @param k The number of elements in the input array. Must be a multiple of 32.
 */
void quantize_row_q4_0_host(const float *x, void *vy, int64_t k);
