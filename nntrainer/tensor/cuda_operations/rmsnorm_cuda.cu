// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file	rmsnorm_cuda.cpp
 * @date	14 Nov 2025
 * @brief	Common blas CUDA kernels
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Samsung Electronics Co., Ltd.
 * @bug		No known bugs except for NYI items
 *
 */

#include "rmsnorm_cuda.h"
#include <cuda_runtime.h>

 __global__ void rmsnorm_cuda_kernel(const float *input, float *output,
                                    const float *alpha, float epsilon,
                                    int H, int W) {
  // Each block processes one row (height index)
  int h = blockIdx.x;
  int index = h * W;
  
  // Shared memory for reduction
  extern __shared__ float sdata[];
  
  // Thread index within block
  int tid = threadIdx.x;
  const int blockSize = blockDim.x;
  
  // Load input data and compute sum of squares
  const float *in = input + index;
  float sum_squares = 0.0f;
  
  // Each thread processes multiple elements if W > blockSize
  for (int i = tid; i < W; i += blockSize) {
    float val = in[i];
    sum_squares += val * val;
  }
  
  // Store partial sum in shared memory
  sdata[tid] = sum_squares;
  __syncthreads();
  
  // Reduction in shared memory
  for (int s = blockSize / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  
  // First thread in block computes the final result
  if (tid == 0) {
    float mean = sdata[0] / W;
    float scale = 1.0f / sqrtf(mean + epsilon);
    
    // Store the scale value in shared memory for reuse
    sdata[0] = scale;
  }
  __syncthreads();
  
  // Load the computed scale
  float scale = sdata[0];
  
  // Compute output values
  float *out = output + index;
  for (int i = tid; i < W; i += blockSize) {
    out[i] = in[i] * scale * alpha[i];
  }
}

namespace nntrainer {

void rmsnorm_cuda(const float *input, const float *gamma, float *result,
                  const float epsilon, unsigned int height, unsigned int width) {
  // Define block size
  const int blockSize = 256;
  
  // Calculate grid size (one block per row)
  const int gridSize = height;
  
  // Shared memory size for reduction
  const int sharedMemSize = blockSize * sizeof(float);
  
  // Launch the CUDA kernel
  rmsnorm_cuda_kernel<<<gridSize, blockSize, sharedMemSize>>>(
    input, result, gamma, epsilon, height, width);
}

void sscal_cuda(float *X, const unsigned int N, const float alpha) {
  // TODO: Implement CUDA kernel for sscal
}

} // namespace nntrainer
