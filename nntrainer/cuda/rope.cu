// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   rope.cu
 * @date   27 January 2025
 * @brief  RoPE (Rotary Positional Embedding) CUDA implementation
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jung, dk11.jung@samsung.com
 * @bug    No known bugs except for NYI items
 *
 */

#include "rope.h"

#include <cuda_fp16.h>

namespace nntrainer {

__global__ void rotary_embedding_kernel(uint16_t *output, unsigned int width,
                                        unsigned int dim, unsigned int half_,
                                        float *inout, const float *cos_,
                                        const float *sin_,
                                        bool only_convert_to_fp16) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int num_pairs = (width / dim) * half_;

  if (idx >= num_pairs)
    return;

  unsigned int row = idx / half_;
  unsigned int k = idx % half_;
  unsigned int w = row * dim;

  unsigned int i0 = w + k;
  unsigned int i1 = w + k + half_;

  float a = inout[i0];
  float b = inout[i1];

  if (only_convert_to_fp16) {
    if (output != nullptr) {
      output[i0] = __half_as_ushort(__float2half(a));
      output[i1] = __half_as_ushort(__float2half(b));
    }
  } else {
    float c = cos_[k];
    float s = sin_[k];

    float out0 = a * c - b * s;
    float out1 = a * s + b * c;

    if (output != nullptr) {
      output[i0] = __half_as_ushort(__float2half(out0));
      output[i1] = __half_as_ushort(__float2half(out1));
    } else {
      inout[i0] = out0;
      inout[i1] = out1;
    }
  }
}

void rotary_embedding_cuda(void *output, unsigned int width, unsigned int dim,
                           unsigned int half_, float *inout, const float *cos_,
                           const float *sin_, bool only_convert_to_fp16,
                           cudaStream_t stream) {
  unsigned int num_pairs = (width / dim) * half_;
  unsigned int block_size = 256;
  unsigned int grid_size = (num_pairs + block_size - 1) / block_size;

  rotary_embedding_kernel<<<grid_size, block_size, 0, stream>>>(
    static_cast<uint16_t *>(output), width, dim, half_, inout, cos_, sin_,
    only_convert_to_fp16);
}

} // namespace nntrainer
