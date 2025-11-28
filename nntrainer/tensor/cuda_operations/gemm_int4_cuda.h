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

namespace nntrainer {

/**
 * @brief CUDA implementation of int4 GEMM operation equivalent to openvino_gemm_cl
 * 
 * @param input Input data pointer
 * @param weights Weight data pointer
 * @param scales Scale data pointer
 * @param output Output data pointer
 * @param M Number of rows in the matrix
 * @param N Number of columns in the matrix
 * @param K Inner dimension of the matrix multiplication
 * @param quantization_group_size Quantization group size
 */
void gemm_int4_cuda(void *input, void *weights, void *scales, void *output,
                    unsigned int M, unsigned int N, unsigned int K,
                    unsigned int quantization_group_size);

} // namespace nntrainer

#endif // __GEMM_INT4_CUDA_H__
