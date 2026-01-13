// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   quantize_cuda.cu
 * @date   13 Jan 2026
 * @brief  CUDA implementation for quantization functions
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Samsung R&D Institute
 * @bug    No known bugs
 */
#include "quantize_cuda.h"
#include <cstdio>
#include <cuda_fp16.h>

#define CUDA_QUANTIZE_BLOCK_SIZE 256

// CUDA kernel for INT8 quantization with padding
static __global__ void quantize_input_int8_pad_kernel(
  const float *__restrict__ input, int8_t *__restrict__ quantized_input,
  half *__restrict__ scales, unsigned int M, unsigned int K,
  unsigned int quantization_group_size) {

  const unsigned int group_id = blockIdx.x;
  const unsigned int tid = threadIdx.x;

  const unsigned int align_k =
    ((K + quantization_group_size - 1) / quantization_group_size) *
    quantization_group_size;
  const unsigned int groups_in_row = align_k / quantization_group_size;
  const unsigned int row_id = group_id / groups_in_row;
  const unsigned int group_id_in_row = group_id % groups_in_row;
  const unsigned int input_offset =
    (row_id * K) + (group_id_in_row * quantization_group_size);
  const unsigned int output_offset = group_id * quantization_group_size;
  const unsigned int max_quantize_block = quantization_group_size;

  unsigned int quantize_block;
  if (group_id_in_row == groups_in_row - 1) {
    quantize_block = quantization_group_size - (align_k - K);
  } else {
    quantize_block = quantization_group_size;
  }

  // Shared memory for reduction
  __shared__ float shared_max[32];

  // Find maximum absolute value
  float local_max = 0.0f;
  for (unsigned int i = tid; i < quantize_block; i += blockDim.x) {
    unsigned int idx = input_offset + i;
    float val = (idx < row_id * K + K)
                  ? fabsf(__half2float(__float2half(input[idx])))
                  : 0.0f;
    local_max = fmaxf(local_max, val);
  }

  shared_max[tid] = local_max;

  // Reduction in shared memory
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
    }
  }

  float max_value = fmaxf(shared_max[0], 0.001f);

  // Calculate quantization scale
  float quan_scale = max_value / 127.0f;
  float quan_scale_1 = 1.0f / quan_scale;

  // Quantize the data
  for (unsigned int i = tid; i < quantize_block; i += blockDim.x) {
    unsigned int input_idx = input_offset + i;
    unsigned int output_idx = output_offset + i;
    float val = (input_idx < row_id * K + K)
                  ? __half2float(__float2half(input[input_idx]))
                  : 0.0f;
    float quantized_val = val * quan_scale_1;
    quantized_input[output_idx] = (int8_t)__float2int_rn(quantized_val);
  }

  // Pad with zeros if necessary
  for (unsigned int i = quantize_block + tid; i < max_quantize_block;
       i += blockDim.x) {
    unsigned int output_idx = output_offset + i;
    quantized_input[output_idx] = 0;
  }

  // Store the scale (thread 0 only)
  if (tid == 0) {
    scales[group_id * 2] = __float2half(quan_scale);
    scales[group_id * 2 + 1] = (__half)0.0f; // Placeholder for activation sum
  }
}

void quantize_input_int8_pad_cuda(const void *input, void *quantized_input,
                                  void *scales, unsigned int M, unsigned int K,
                                  unsigned int quantization_group_size,
                                  cudaStream_t stream) {
  const unsigned int align_k =
    ((K + quantization_group_size - 1) / quantization_group_size) *
    quantization_group_size;
  const unsigned int groups_in_row = align_k / quantization_group_size;
  const unsigned int total_groups = M * groups_in_row;

  const dim3 grid(total_groups);
  const dim3 block(32);

  quantize_input_int8_pad_kernel<<<grid, block, 0, stream>>>(
    (const float *)input, (int8_t *)quantized_input, (half *)scales, M, K,
    quantization_group_size);
}

// Q8_1 Block Structure
#define QK8_1 32
struct __align__(4) block_q8_1 {
  half d;
  half s;
  int8_t qs[QK8_1];
};

static_assert(sizeof(block_q8_1) == 36, "block_q8_1 size mismatch");

static __global__ void quantize_q8_1_kernel(const float *__restrict__ x,
                                            void *__restrict__ vy, int64_t k) {
  const int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int lane_id = threadIdx.x % 32;
  const int64_t offset = global_warp_id * 32;

  if (offset >= k)
    return;

  // Load data
  float val = 0.0f;
  if (offset + lane_id < k) {
    val = x[offset + lane_id];
  }

  // Find abs max
  float abs_val = fabsf(val);
  float max_val = abs_val;
  // Warp-level parallel reduction: all 32 threads compute the same max value
  // using shuffle XOR
#pragma unroll
  for (int mask = 16; mask > 0; mask /= 2) {
    max_val = fmaxf(max_val, __shfl_xor_sync(0xffffffff, max_val, mask));
  }

  const float d = max_val / 127.0f;
  const float id = d ? 1.0f / d : 0.0f;

  // Quantize
  int8_t q = (int8_t)roundf(val * id);

  // Calculate sum
  int sum = q;
  // Warp-level parallel reduction: all 32 threads compute the same sum using
  // shuffle XOR
#pragma unroll
  for (int mask = 16; mask > 0; mask /= 2) {
    sum += __shfl_xor_sync(0xffffffff, sum, mask);
  }

  // Store result
  block_q8_1 *out_ptr = (block_q8_1 *)vy + global_warp_id;

  if (lane_id == 0) {
    out_ptr->d = __float2half(d);
    out_ptr->s = __float2half(sum * d);
  }
  out_ptr->qs[lane_id] = q;
}

void quantize_activation_q8_1_cuda(const float *input, void *output, int64_t k,
                                   cudaStream_t stream) {
  const int block_size = 256;
  const int num_warps_per_block = block_size / 32;
  const int num_blocks =
    (k + 32 * num_warps_per_block - 1) / (32 * num_warps_per_block);

  quantize_q8_1_kernel<<<num_blocks, block_size, 0, stream>>>(input, output, k);
}

// -------------------------------------------------------------------------
// CPU Host Implementation for verification
// -------------------------------------------------------------------------

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>

// Helper for float to half conversion on CPU
static inline half fp32_to_fp16(float x) {
  uint16_t rh;
  // Fallback C implementation for FP32 to FP16 conversion
  uint32_t x_u;
  std::memcpy(&x_u, &x, sizeof(float));

  const uint32_t sign = (x_u >> 16) & 0x8000;
  const uint32_t exp = (x_u >> 23) & 0xFF;
  const uint32_t mant = x_u & 0x7FFFFF;

  if (exp == 0) {
    rh = sign; // Denormal or zero -> zero
  } else if (exp == 255) {
    rh = sign | 0x7C00 | (mant ? 0x200 : 0); // Inf or NaN
  } else {
    int new_exp = (int)exp - 127 + 15;
    if (new_exp < 0) {
      rh = sign; // Underflow -> zero
    } else if (new_exp >= 31) {
      rh = sign | 0x7C00; // Overflow -> Inf
    } else {
      rh = sign | (new_exp << 10) | (mant >> 13);
    }
  }

  // Reinterpret uint16_t as half
  half h;
  std::memcpy(&h, &rh, sizeof(half));
  return h;
}

// #define GGML_FP32_TO_FP16(x) fp32_to_fp16(x)

void quantize_row_q8_1_host(const float *x, void *vy, int64_t k) {
  assert(QK8_1 == 32);
  assert(k % QK8_1 == 0);
  const int nb = k / QK8_1;

  block_q8_1 *y = (block_q8_1 *)vy;

  for (int i = 0; i < nb; i++) {
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK8_1; j++) {
      const float v = x[i * QK8_1 + j];
      amax = std::max(amax, std::abs(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f / d : 0.0f;

    y[i].d = fp32_to_fp16(d);

    int sum = 0;

    for (int j = 0; j < QK8_1 / 2; ++j) {
      const float v0 = x[i * QK8_1 + j];
      const float v1 = x[i * QK8_1 + j + QK8_1 / 2];

      const int8_t q0 = roundf(v0 * id);
      const int8_t q1 = roundf(v1 * id);

      y[i].qs[j] = q0;
      y[i].qs[j + QK8_1 / 2] = q1;

      sum += q0 + q1;
    }

    // Need to cast d back to float for multiplication, then convert result to
    // half float d_val = 0.0f; half d_half = y[i].d; Simple way to get float
    // value from half on host if cuda_fp16.h is available
    // __half2float is a device function, so we might need manual conversion or
    // just use d Since we computed d just above, we can use it directly.

    y[i].s = fp32_to_fp16(sum * d);
  }
}

// Q4_0 Block Structure
#define QK4_0 32
struct block_q4_0 {
  half d;
  uint8_t qs[QK4_0 / 2];
};
static_assert(sizeof(block_q4_0) == 18, "block_q4_0 size mismatch");

void quantize_row_q4_0_host(const float *x, void *vy, int64_t k) {
  assert(QK4_0 == 32);
  assert(k % QK4_0 == 0);
  const int nb = k / QK4_0;

  block_q4_0 *y = (block_q4_0 *)vy;

  for (int i = 0; i < nb; i++) {
    float amax = 0.0f;
    float max_val = 0.0f;

    for (int j = 0; j < QK4_0; j++) {
      const float v = x[i * QK4_0 + j];
      if (amax < std::abs(v)) {
        amax = std::abs(v);
        max_val = v;
      }
    }

    const float d = max_val / -8.0f;
    const float id = d ? 1.0f / d : 0.0f;

    y[i].d = fp32_to_fp16(d);

    for (int j = 0; j < QK4_0 / 2; ++j) {
      const float x0 = x[i * QK4_0 + 2 * j + 0] * id;
      const float x1 = x[i * QK4_0 + 2 * j + 1] * id;

      const uint8_t xi0 = (uint8_t)(x0 + 8.5f);
      const uint8_t xi1 = (uint8_t)(x1 + 8.5f);

      y[i].qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
    }
  }
}
