// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file	addition_cuda.cu
 * @date	20 Nov 2025
 * @brief	Common blas CUDA kernels for addition
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Samsung Electronics Co., Ltd.
 * @bug		No known bugs except for NYI items
 *
 */

#include "addition_cuda.h"
#include <cuda_runtime.h>

namespace nntrainer {

__global__ void addition_cuda_kernel(const float *input, float *output,
                                     unsigned int size_input,
                                     unsigned int size_res) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size_res) {
    output[idx] = output[idx] + input[idx % size_input];
  }
}

void addition_cuda(const float *input, float *res, unsigned int size_input,
                   unsigned int size_res) {
  const int blockSize = 256;
  const int gridSize = (size_res + blockSize - 1) / blockSize;

  addition_cuda_kernel<<<gridSize, blockSize>>>(input, res, size_input,
                                                size_res);
}

} // namespace nntrainer
