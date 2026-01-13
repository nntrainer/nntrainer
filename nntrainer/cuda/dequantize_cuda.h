// SPDX-License-Identifier: Apache-2.0
/**
 * @file	dequantize_cuda.h
 * @date	19 December 2025
 * @brief	Header for CUDA implementation of dequantization utilities
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Daekyoung Jung <dk11.jung@samsung.com>
 * @bug		No known bugs
 */
#ifndef __DEQUANTIZE_CUDA_H__
#define __DEQUANTIZE_CUDA_H__

#include <cstdint>
#include <cuda_runtime.h>

namespace nntrainer {
namespace Int4UtilsCuda {

/**
 * @brief Dequantize multiple rows from an INT4 packed weight tensor into a
 * float output tensor using CUDA.
 *
 * @param weights Pointer to the packed weights on CUDA device.
 * @param scales Pointer to the FP16 scales on CUDA device.
 * @param rows_count Total number of rows in the weight tensor (padded).
 * @param columns_count Total number of columns in the weight tensor (embedding
 * dimension).
 * @param group_size Quantization group size (e.g., 32).
 * @param indices Pointer to the row indices on CUDA device.
 * @param num_indices Number of indices to dequantize.
 * @param output Pointer to the output float buffer on CUDA device.
 * @param stream CUDA stream to use.
 */
void dequantize_rows_cuda(const uint8_t *weights, const uint16_t *scales,
                          unsigned int rows_count, unsigned int columns_count,
                          unsigned int group_size, const float *indices,
                          unsigned int num_indices, float *output,
                          cudaStream_t stream = 0);

} // namespace Int4UtilsCuda
} // namespace nntrainer

#endif // __DEQUANTIZE_CUDA_H__
