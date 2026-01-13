// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   gemm_int4_cuda.cu
 * @date   13 Jan 2026
 * @brief  CUDA implementation of INT4 GEMM kernels
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Samsung R&D Institute
 * @bug    No known bugs
 */
#include <cooperative_groups.h>
#include <cuda/pipeline>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <mma.h>

#include "gemm_int4_cuda.h"
#include "quantize_cuda.h"

// Helper for ceil division
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#define ALIGN(a, b) (CEIL_DIV(a, b) * (b))

/**
 * @brief Optimized CUDA implementation of int4 GEMM operation using packed
 * blocks (16x16 true tiling)
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
void gemm_a8_w4_b16x16(const void *input, const void *weights,
                       const void *scales, const void *input_scales,
                       float *output, unsigned int M, unsigned int N,
                       unsigned int K, unsigned int quantization_group_size);

__global__ void __gemm_a8_w4_b32x32(const int8_t *input, const uint8_t *weights,
                                    const __half *scales,
                                    const __half *input_scales, float *output,
                                    unsigned int M, unsigned int N,
                                    unsigned int K,
                                    unsigned int quantization_group_size) {
  // Block dimensions: 32x32
  // Grid dimensions: (N+31)/32, (M+31)/32

  unsigned int tx = threadIdx.x; // 0..31 (N direction within block)
  unsigned int ty = threadIdx.y; // 0..31 (M direction within block)

  unsigned int bx = blockIdx.x;
  unsigned int by = blockIdx.y;

  unsigned int row = by * 32 + ty; // Global row index (M)
  unsigned int col = bx * 32 + tx; // Global col index (N)

  // Shared memory
  // Input block: 32x32 int8_t
  __shared__ int8_t s_input[32][32];

  // Weight block: 32x32 int8_t (unpacked)
  // [N][K] layout to allow contiguous access along K
  __shared__ int8_t s_weights[32][32];

  float sum = 0.0f;

  // Loop over K in chunks of 32
  for (unsigned int k_chunk = 0; k_chunk < (K + 31) / 32; ++k_chunk) {
    unsigned int k_start = k_chunk * 32;

    // 1. Load Input to Shared Memory
    if (row < M && (k_start + tx) < K) {
      // Input is stored as [M, K] (quantized)
      // We assume row-major linear layout for input_quantized
      // We need alignK for correct indexing if padding was used
      unsigned int alignK = (K + quantization_group_size - 1) /
                            quantization_group_size * quantization_group_size;

      // Simplified: input_idx = row * alignK + (k_start + tx)
      unsigned int input_idx = row * alignK + k_start + tx;

      s_input[ty][tx] = input[input_idx];
    } else {
      s_input[ty][tx] = 0;
    }

    // 2. Load Weights to Shared Memory and Unpack
    // Each thread loads one int8 weight
    // tid covers 0..1023, mapping to 32x32 s_weights
    // n = tx, k = ty
    unsigned int n = tx;
    unsigned int k = ty; // 0..31 relative to k_start

    unsigned int global_n = bx * 32 + n;
    unsigned int global_k = k_start + k;

    if (global_n < N && global_k < K) {
      unsigned int k_parity = k % 2;

      // Address calculation
      // Block index: n_blk * (K/2) + k_blk
      // n_blk = bx
      // k_blk = global_k / 2
      unsigned int block_idx = bx * (K / 2) + (global_k / 2);
      unsigned int byte_offset = n; // n is offset within 32-byte block

      unsigned int weight_idx = block_idx * 32 + byte_offset;
      uint8_t packed_w = weights[weight_idx];

      int8_t w_val = k_parity == 0 ? packed_w & 0x0F : packed_w >> 4;

      if (w_val >= 8)
        w_val -= 16;
      s_weights[n][k] = w_val;
    }

    __syncthreads();

    // 3. Compute
    int chunk_acc[8] = {0};

#pragma unroll
    for (int i = 0; i < 8; ++i) {
      int input_packed = *reinterpret_cast<int *>(&s_input[ty][i * 4]);
      int weight_packed = *reinterpret_cast<int *>(&s_weights[tx][i * 4]);
      chunk_acc[i] = __dp4a(input_packed, weight_packed, chunk_acc[i]);
    }

    {
      int total_acc = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        total_acc += chunk_acc[i];
      }

      // Apply scales
      // Input scale
      unsigned int alignK = (K + quantization_group_size - 1) /
                            quantization_group_size * quantization_group_size;
      unsigned int groups_in_row = alignK / quantization_group_size;
      float i_scale = __half2float(
        input_scales[(row * groups_in_row + k_start / quantization_group_size) *
                     2]);

      // Weight scale
      unsigned int scale_idx =
        col * (K / quantization_group_size) + k_start / quantization_group_size;
      float w_scale = __half2float(scales[scale_idx]);

      sum += total_acc * i_scale * w_scale;
    }
    __syncthreads();
  }

  if (row < M && col < N) {
    output[row * N + col] = sum;
  }
}

void gemm_a8_w4_b32x32(const void *input, const void *weights,
                       const void *scales, const void *input_scales,
                       float *output, unsigned int M, unsigned int N,
                       unsigned int K, unsigned int quantization_group_size) {

  // Launch Kernel
  dim3 blockDim(32, 32);
  dim3 gridDim((N + 31) / 32, (M + 31) / 32);

  __gemm_a8_w4_b32x32<<<gridDim, blockDim>>>(
    static_cast<const int8_t *>(input), static_cast<const uint8_t *>(weights),
    static_cast<const __half *>(scales),
    static_cast<const __half *>(input_scales), output, M, N, K,
    quantization_group_size);
}

__global__ void __gemm_a8_w4_b32x32_pre_dequant(
  const int8_t *input, const uint8_t *weights, const __half *scales,
  const __half *input_scales, float *output, unsigned int M, unsigned int N,
  unsigned int K, unsigned int quantization_group_size) {
  // Block dimensions: 32x32
  // Grid dimensions: (N+31)/32, (M+31)/32

  unsigned int tx = threadIdx.x; // 0..31 (N direction within block)
  unsigned int ty = threadIdx.y; // 0..31 (M direction within block)

  unsigned int bx = blockIdx.x;
  unsigned int by = blockIdx.y;

  unsigned int row = by * 32 + ty; // Global row index (M)
  unsigned int col = bx * 32 + tx; // Global col index (N)

  // Shared memory
  // Input block: 32x32 float (dequantized with input_scale)
  __shared__ float s_input[32][32];

  // Weight block: 32x32 int8_t (unpacked)
  // [N][K] layout to allow contiguous access along K
  __shared__ int8_t s_weights[32][32];

  float sum = 0.0f;

  // Loop over K in chunks of 32
  for (unsigned int k_chunk = 0; k_chunk < (K + 31) / 32; ++k_chunk) {
    unsigned int k_start = k_chunk * 32;

    // 1. Load Input to Shared Memory and dequantize with input_scale
    if (row < M && (k_start + tx) < K) {
      // Input is stored as [M, K] (quantized)
      unsigned int alignK = (K + quantization_group_size - 1) /
                            quantization_group_size * quantization_group_size;
      unsigned int groups_in_row = alignK / quantization_group_size;

      unsigned int current_k = k_start + tx;
      unsigned int group_id_in_row = current_k / quantization_group_size;
      unsigned int global_group_id = row * groups_in_row + group_id_in_row;
      unsigned int offset_in_group = current_k % quantization_group_size;

      unsigned int input_idx =
        global_group_id * quantization_group_size + offset_in_group;

      // Load INT8 input and dequantize with input_scale
      int8_t input_int8 = input[input_idx];
      float i_scale = __half2float(input_scales[global_group_id * 2]);
      s_input[ty][tx] = static_cast<float>(input_int8) * i_scale;
    } else {
      s_input[ty][tx] = 0.0f;
    }

    // 2. Load Weights to Shared Memory and Unpack
    // Each thread loads one int8 weight
    unsigned int n = tx;
    unsigned int k = ty; // 0..31 relative to k_start

    unsigned int global_n = bx * 32 + n;
    unsigned int global_k = k_start + k;

    if (global_n < N && global_k < K) {
      unsigned int k_parity = k % 2;

      // Address calculation
      unsigned int block_idx = bx * (K / 2) + (global_k / 2);
      unsigned int byte_offset = n; // n is offset within 32-byte block

      unsigned int weight_idx = block_idx * 32 + byte_offset;
      uint8_t packed_w = weights[weight_idx];

      int8_t w_val = k_parity == 0 ? packed_w & 0x0F : packed_w >> 4;

      if (w_val >= 8)
        w_val -= 16;
      s_weights[n][k] = w_val;
    } else {
      s_weights[n][k] = 0;
    }

    __syncthreads();

    // 3. Compute (FP32 × INT8)
    // s_input is already dequantized FP32
    // s_weights is INT8, need to convert and apply weight scale

    // Get weight scale for this K chunk
    unsigned int group_id_in_row = k_start / quantization_group_size;
    unsigned int scale_idx =
      col * (K / quantization_group_size) + group_id_in_row;
    float w_scale = __half2float(scales[scale_idx]);

    if (row < M && col < N) {
#pragma unroll
      for (int k = 0; k < 32; k++) {
        if (k_start + k >= K)
          break;
        float w_val_f = static_cast<float>(s_weights[tx][k]);
        sum += s_input[ty][k] * w_val_f * w_scale;
      }
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    output[row * N + col] = sum;
  }
}

void gemm_a8_w4_b32x32_pre_dequant(const void *input, const void *weights,
                                   const void *scales, const void *input_scales,
                                   float *output, unsigned int M,
                                   unsigned int N, unsigned int K,
                                   unsigned int quantization_group_size) {

  // Launch Kernel
  dim3 blockDim(32, 32);
  dim3 gridDim((N + 31) / 32, (M + 31) / 32);

  __gemm_a8_w4_b32x32_pre_dequant<<<gridDim, blockDim>>>(
    static_cast<const int8_t *>(input), static_cast<const uint8_t *>(weights),
    static_cast<const __half *>(scales),
    static_cast<const __half *>(input_scales), output, M, N, K,
    quantization_group_size);
}

__global__ void
__gemm_a8_w4_b16x16_dp4a(const int8_t *input, const uint8_t *weights,
                         const __half *scales, const __half *input_scales,
                         float *output, unsigned int M, unsigned int N,
                         unsigned int K, unsigned int quantization_group_size) {
  // Block dimensions: 16x16 threads
  // Grid dimensions: (N+31)/32, (M+31)/32
  // Each block computes 32x32 output tile, each thread computes 4 outputs (2x2)

  unsigned int tx = threadIdx.x;   // 0..15
  unsigned int ty = threadIdx.y;   // 0..15
  unsigned int tid = ty * 16 + tx; // 0..255

  unsigned int bx = blockIdx.x;
  unsigned int by = blockIdx.y;

  unsigned int row_start = by * 32; // Global row start
  unsigned int col_start = bx * 32; // Global col start

  // Shared memory: 32x32 blocks
  __shared__ int8_t s_input[32][32];
  __shared__ int8_t s_weights[32][32];

  // Accumulators for 4 outputs: [ty][tx], [ty][tx+16], [ty+16][tx],
  // [ty+16][tx+16]
  float sum[4] = {0.0f};

  // Loop over K in chunks of 32
  for (unsigned int k_chunk = 0; k_chunk < (K + 31) / 32; ++k_chunk) {
    unsigned int k_start = k_chunk * 32;

    // 1. Load Input to Shared Memory (32x32)
    // 256 threads load 1024 elements, each thread loads 4 elements
    {
      unsigned int r = tid / 8;            // 0..31
      unsigned int c_base = (tid % 8) * 4; // 0, 4, 8, ..., 28

      unsigned int global_r = row_start + r;

#pragma unroll
      for (int i = 0; i < 4; ++i) {
        unsigned int c = c_base + i;
        unsigned int global_c = k_start + c;

        if (global_r < M && global_c < K) {
          unsigned int alignK = (K + quantization_group_size - 1) /
                                quantization_group_size *
                                quantization_group_size;
          unsigned int groups_in_row = alignK / quantization_group_size;
          unsigned int group_id = global_c / quantization_group_size;
          unsigned int offset = global_c % quantization_group_size;
          unsigned int global_group = global_r * groups_in_row + group_id;
          unsigned int idx = global_group * quantization_group_size + offset;

          s_input[r][c] = input[idx];
        } else {
          s_input[r][c] = 0;
        }
      }
    }

    // 2. Load Weights to Shared Memory and Unpack (32x32)
    // 256 threads load 1024 elements, each thread loads 4 weights
    {
      unsigned int r = tid / 8;            // 0..31 (N within block)
      unsigned int c_base = (tid % 8) * 4; // 0, 4, 8, ..., 28 (K within block)

      unsigned int global_n = col_start + r; // Global N

// Load 4 weights for this thread
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        unsigned int c = c_base + i;
        unsigned int global_k = k_start + c;

        int8_t w_val = 0;

        if (global_n < N && global_k < K) {
          // Use block-based packed layout matching
          // gemm_int4_kernel_packed_block
          unsigned int k_parity = global_k % 2;

          // Block index: n_blk * (K/2) + k_blk
          unsigned int n_blk = global_n / 32;
          unsigned int k_blk = global_k / 2;
          unsigned int block_idx = n_blk * (K / 2) + k_blk;

          // Offset within 32-byte block
          unsigned int byte_offset = global_n % 32;

          unsigned int weight_idx = block_idx * 32 + byte_offset;
          uint8_t packed_w = weights[weight_idx];

          w_val = k_parity == 0 ? (packed_w & 0x0F) : (packed_w >> 4);
          if (w_val >= 8)
            w_val -= 16;
        }

        s_weights[r][c] = w_val;
      }
    }

    __syncthreads();

    // 3. Compute - each thread computes 4 outputs
    int acc[4] = {0};

#pragma unroll
    for (int ki = 0; ki < 8; ++ki) {
      int k_idx = ki * 4;

      // Load inputs for 2 rows
      int i0 = *reinterpret_cast<int *>(&s_input[ty][k_idx]);
      int i1 = *reinterpret_cast<int *>(&s_input[ty + 16][k_idx]);

      // Load weights for 2 cols
      int w0 = *reinterpret_cast<int *>(&s_weights[tx][k_idx]);
      int w1 = *reinterpret_cast<int *>(&s_weights[tx + 16][k_idx]);

      // Accumulate 4 outputs
      acc[0] = __dp4a(i0, w0, acc[0]); // [ty][tx]
      acc[1] = __dp4a(i0, w1, acc[1]); // [ty][tx+16]
      acc[2] = __dp4a(i1, w0, acc[2]); // [ty+16][tx]
      acc[3] = __dp4a(i1, w1, acc[3]); // [ty+16][tx+16]
    }

    // Apply scales for each of the 4 outputs
    unsigned int alignK = (K + quantization_group_size - 1) /
                          quantization_group_size * quantization_group_size;
    unsigned int groups_in_row = alignK / quantization_group_size;
    unsigned int group_id_in_row = k_start / quantization_group_size;

    // For each of the 4 outputs
    unsigned int rows[2] = {row_start + ty, row_start + ty + 16};
    unsigned int cols[2] = {col_start + tx, col_start + tx + 16};

    for (int r_idx = 0; r_idx < 2; ++r_idx) {
      unsigned int r = rows[r_idx];
      if (r >= M)
        continue;

      unsigned int global_group_id = r * groups_in_row + group_id_in_row;
      float i_scale = __half2float(input_scales[global_group_id * 2]);

      for (int c_idx = 0; c_idx < 2; ++c_idx) {
        unsigned int c = cols[c_idx];
        if (c >= N)
          continue;

        unsigned int scale_idx =
          c * (K / quantization_group_size) + group_id_in_row;
        float w_scale = __half2float(scales[scale_idx]);

        int acc_idx = r_idx * 2 + c_idx;
        sum[acc_idx] += acc[acc_idx] * i_scale * w_scale;
      }
    }

    __syncthreads();
  }

  // Store 4 outputs
  unsigned int rows[2] = {row_start + ty, row_start + ty + 16};
  unsigned int cols[2] = {col_start + tx, col_start + tx + 16};

  for (int r_idx = 0; r_idx < 2; ++r_idx) {
    unsigned int r = rows[r_idx];
    if (r >= M)
      continue;

    for (int c_idx = 0; c_idx < 2; ++c_idx) {
      unsigned int c = cols[c_idx];
      if (c >= N)
        continue;

      int sum_idx = r_idx * 2 + c_idx;
      output[r * N + c] = sum[sum_idx];
    }
  }
}

void gemm_a8_w4_b16x16_dp4a(const void *input, const void *weights,
                            const void *scales, const void *input_scales,
                            float *output, unsigned int M, unsigned int N,
                            unsigned int K,
                            unsigned int quantization_group_size) {

  // Launch Kernel - each block computes 32x32 outputs
  dim3 blockDim(16, 16);
  dim3 gridDim((N + 31) / 32, (M + 31) / 32);

  __gemm_a8_w4_b16x16_dp4a<<<gridDim, blockDim>>>(
    static_cast<const int8_t *>(input), static_cast<const uint8_t *>(weights),
    static_cast<const __half *>(scales),
    static_cast<const __half *>(input_scales), output, M, N, K,
    quantization_group_size);
}

/**
 * @brief GEMM kernel for FP32 input and QINT4 weights without INT8 quantization
 *
 * This kernel directly multiplies FP32 input with dequantized QINT4 weights.
 * It avoids the overhead of quantizing FP32 input to INT8.
 */
__global__ void
__gemm_a32_w4_b16x16_naive(const float *input, const uint8_t *weights,
                           const __half *scales, float *output, unsigned int M,
                           unsigned int N, unsigned int K,
                           unsigned int quantization_group_size) {

  // Block dimensions: 16x16 threads
  // Each thread computes one output element

  unsigned int tx = threadIdx.x; // 0..15
  unsigned int ty = threadIdx.y; // 0..15

  unsigned int bx = blockIdx.x;
  unsigned int by = blockIdx.y;

  unsigned int row = by * 16 + ty; // Global row index (M)
  unsigned int col = bx * 16 + tx; // Global col index (N)

  if (row >= M || col >= N)
    return;

  float sum = 0.0f;

  // Loop over K
  for (unsigned int k = 0; k < K; k += 2) {
    // Load 2 FP32 input values
    float in0 = (k < K) ? input[row * K + k] : 0.0f;
    float in1 = (k + 1 < K) ? input[row * K + k + 1] : 0.0f;

    // Load and unpack 2 INT4 weights
    // Weight layout: packed blocks of 32 bytes
    // Block index: n_blk * (K/2) + k_blk
    unsigned int n_blk = col / 32;
    unsigned int k_blk = k / 2;
    unsigned int block_idx = n_blk * (K / 2) + k_blk;
    unsigned int byte_offset = col % 32;
    unsigned int weight_idx = block_idx * 32 + byte_offset;

    uint8_t packed_w = weights[weight_idx];

    // Unpack two 4-bit weights
    int8_t w0 = packed_w & 0x0F;
    int8_t w1 = packed_w >> 4;

    // Convert from unsigned to signed
    if (w0 >= 8)
      w0 -= 16;
    if (w1 >= 8)
      w1 -= 16;

    // Get weight scale for this group
    unsigned int group_id = k / quantization_group_size;
    unsigned int scale_idx = col * (K / quantization_group_size) + group_id;
    float w_scale = __half2float(scales[scale_idx]);

    // Dequantize weights and accumulate
    float dequant_w0 = w0 * w_scale;
    float dequant_w1 = w1 * w_scale;

    sum += in0 * dequant_w0;
    if (k + 1 < K) {
      sum += in1 * dequant_w1;
    }
  }

  output[row * N + col] = sum;
}

void gemm_a32_w4_b16x16_naive(const float *input, const void *weights,
                              const void *scales, float *output, unsigned int M,
                              unsigned int N, unsigned int K,
                              unsigned int quantization_group_size) {

  // Launch kernel with 16x16 thread blocks
  dim3 blockDim(16, 16);
  dim3 gridDim((N + 15) / 16, (M + 15) / 16);

  __gemm_a32_w4_b16x16_naive<<<gridDim, blockDim>>>(
    input, static_cast<const uint8_t *>(weights),
    static_cast<const __half *>(scales), output, M, N, K,
    quantization_group_size);
}

__global__ void
__gemm_a32_w4_b32x32_s32x32(const float *input, const uint8_t *weights,
                            const __half *scales, float *output, unsigned int M,
                            unsigned int N, unsigned int K,
                            unsigned int quantization_group_size) {
  // Block dimensions: 32x32
  // Grid dimensions: (N+31)/32, (M+31)/32

  unsigned int tx = threadIdx.x; // 0..31 (N direction within block)
  unsigned int ty = threadIdx.y; // 0..31 (M direction within block)

  unsigned int bx = blockIdx.x;
  unsigned int by = blockIdx.y;

  unsigned int row = by * 32 + ty; // Global row index (M)
  unsigned int col = bx * 32 + tx; // Global col index (N)

  // Shared memory
  // Input block: 32x32 float (FP32 input)
  __shared__ float s_input[32][32];

  // Weight block: 32x32 int8_t (unpacked)
  // [N][K] layout to allow contiguous access along K
  __shared__ int8_t s_weights[32][32];

  float sum = 0.0f;

  // Loop over K in chunks of 32
  for (unsigned int k_chunk = 0; k_chunk < (K + 31) / 32; ++k_chunk) {
    unsigned int k_start = k_chunk * 32;

    // 1. Load Input to Shared Memory (FP32, row-major)
    if (row < M && (k_start + tx) < K) {
      s_input[ty][tx] = input[row * K + (k_start + tx)];
    } else {
      s_input[ty][tx] = 0.0f;
    }

    // 2. Load Weights to Shared Memory and Unpack
    // Each thread loads one int8 weight
    // n = tx, k = ty
    unsigned int n = tx;
    unsigned int k = ty; // 0..31 relative to k_start

    unsigned int global_n = bx * 32 + n;
    unsigned int global_k = k_start + k;

    if (global_n < N && global_k < K) {
      unsigned int k_parity = k % 2;

      // Address calculation
      // Block index: n_blk * (K/2) + k_blk
      unsigned int block_idx = bx * (K / 2) + (global_k / 2);
      unsigned int byte_offset = n; // n is offset within 32-byte block

      unsigned int weight_idx = block_idx * 32 + byte_offset;
      uint8_t packed_w = weights[weight_idx];

      int8_t w_val = k_parity == 0 ? packed_w & 0x0F : packed_w >> 4;

      if (w_val >= 8)
        w_val -= 16;
      s_weights[n][k] = w_val;
    } else {
      s_weights[n][k] = 0;
    }

    __syncthreads();

    // 3. Compute (FP32 × INT8, read scale per k like V2)
    if (row < M && col < N) {
#pragma unroll
      for (int k = 0; k < 32; k++) {
        unsigned int global_k = k_start + k;
        if (global_k >= K)
          break;

        // Get weight scale for this specific k (like V2)
        unsigned int group_id = global_k / quantization_group_size;
        unsigned int scale_idx = col * (K / quantization_group_size) + group_id;
        float w_scale = __half2float(scales[scale_idx]);

        // FP32 input × INT8 weight (converted to float) × scale
        float w_val_f = static_cast<float>(s_weights[tx][k]);
        float product = s_input[ty][k] * w_val_f * w_scale;
        sum += product;
      }
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    output[row * N + col] = sum;
  }
}

void gemm_a32_w4_b32x32_s32x32(const float *input, const void *weights,
                               const void *scales, float *output,
                               unsigned int M, unsigned int N, unsigned int K,
                               unsigned int quantization_group_size) {

  // Launch Kernel
  dim3 blockDim(32, 32);
  dim3 gridDim((N + 31) / 32, (M + 31) / 32);

  __gemm_a32_w4_b32x32_s32x32<<<gridDim, blockDim>>>(
    input, static_cast<const uint8_t *>(weights),
    static_cast<const __half *>(scales), output, M, N, K,
    quantization_group_size);
}

/**
 * @brief Optimized GEMM kernel (v4) using shared memory tiling
 *
 * Thread block: 32×32 (1024 threads)
 * Tile size: 32×32 for both input and weights
 * Each thread computes 1 output
 * Weights are kept as int8 in shared memory (to match V1 logic)
 * Scales are applied during computation
 */
__global__ void __gemm_a32_w4_b32x32_s32x32_dequant(
  const float *input, const uint8_t *weights, const __half *scales,
  float *output, unsigned int M, unsigned int N, unsigned int K,
  unsigned int quantization_group_size) {

  unsigned int tx = threadIdx.x; // 0..31
  unsigned int ty = threadIdx.y; // 0..31

  unsigned int bx = blockIdx.x; // N direction
  unsigned int by = blockIdx.y; // M direction

  // Shared memory for 32×32 tiles
  __shared__ float s_input[32][32];    // [M_local][K_local]
  __shared__ int8_t s_weights[32][32]; // [N_local][K_local]

  // Each thread computes one output element
  unsigned int row = by * 32 + ty; // Global row index (M)
  unsigned int col = bx * 32 + tx; // Global col index (N)

  float sum = 0.0f;

  // Loop over K in chunks of 32
  for (unsigned int k_start = 0; k_start < K; k_start += 32) {

    // 1. Load Input to Shared Memory
    unsigned int g_row = by * 32 + ty;
    unsigned int g_col = k_start + tx;

    if (g_row < M && g_col < K) {
      s_input[ty][tx] = input[g_row * K + g_col];
    } else {
      s_input[ty][tx] = 0.0f;
    }

    // 2. Load Weights to Shared Memory (Packed INT4 -> Unpacked INT8)
    // Same logic as gemm_int4_kernel_packed_block (V1)
    // tx = N direction (n), ty = K direction (k)
    unsigned int n = tx;
    unsigned int k = ty;

    unsigned int global_n = bx * 32 + n;
    unsigned int global_k = k_start + k;

    if (global_n < N && global_k < K) {
      // Calculate packed weight index (simplified like V3)
      unsigned int k_parity = k % 2;

      // Simplified block index calculation (like V3)
      unsigned int block_idx = bx * (K / 2) + (global_k / 2);
      unsigned int byte_offset = n; // n is offset within 32-byte block

      unsigned int weight_idx = block_idx * 32 + byte_offset;
      uint8_t packed_w = weights[weight_idx];

      // Unpack INT4 value
      int8_t w_val = k_parity == 0 ? (packed_w & 0x0F) : (packed_w >> 4);
      w_val = w_val >= 8 ? w_val - 16 : w_val;
      s_weights[n][k] = w_val;
    } else {
      s_weights[n][k] = 0;
    }

    __syncthreads();

    // 3. Compute matrix multiplication for this tile
    // s_input[row][k]: row in M direction, k in K direction
    // s_weights[col][k]: col in N direction, k in K direction
    // output[row][col] = sum over k of input[row][k] * weights[col][k]

    if (row < M && col < N) {
#pragma unroll
      for (int k = 0; k < 32; k++) {
        unsigned int global_k = k_start + k;
        if (global_k >= K)
          break;

        // Get scale for this specific k (like V2)
        unsigned int group_id = global_k / quantization_group_size;
        unsigned int scale_idx = col * (K / quantization_group_size) + group_id;
        float w_scale = __half2float(scales[scale_idx]);

        // s_weights is int8, convert to float and apply scale
        float w_val_f = static_cast<float>(s_weights[tx][k]);
        sum += s_input[ty][k] * w_val_f * w_scale;
      }
    }

    __syncthreads();
  }

  // Write output
  if (row < M && col < N) {
    output[row * N + col] = sum;
  }
}

void gemm_a32_w4_b32x32_s32x32_dequant(const float *input, const void *weights,
                                       const void *scales, float *output,
                                       unsigned int M, unsigned int N,
                                       unsigned int K,
                                       unsigned int quantization_group_size) {

  // Launch kernel with 32×32 thread blocks
  // Each block computes 32×32 output tile
  dim3 blockDim(32, 32);
  dim3 gridDim((N + 31) / 32, (M + 31) / 32);

  __gemm_a32_w4_b32x32_s32x32_dequant<<<gridDim, blockDim>>>(
    input, static_cast<const uint8_t *>(weights),
    static_cast<const __half *>(scales), output, M, N, K,
    quantization_group_size);
}

/**
 * @brief CUDA GEMM kernel V7: 16x16 threads, each thread loads/computes 4
 * elements
 *
 * Uses same 32x32 shared memory as V4, but with 16x16 thread block.
 * Each thread loads 4 input elements and 4 weight elements to shared memory.
 * Each thread computes 4 output elements (2x2 tile per thread).
 */
__global__ void
__gemm_a32_w4_b16x16_s32x32(const float *input, const uint8_t *weights,
                            const __half *scales, float *output, unsigned int M,
                            unsigned int N, unsigned int K,
                            unsigned int quantization_group_size) {

  unsigned int tx = threadIdx.x; // 0..15
  unsigned int ty = threadIdx.y; // 0..15

  unsigned int bx = blockIdx.x; // N direction
  unsigned int by = blockIdx.y; // M direction

  // Shared memory for 32×32 tiles (same as V4)
  __shared__ float s_input[32][32];    // [M_local][K_local]
  __shared__ int8_t s_weights[32][32]; // [N_local][K_local]

  // Each thread computes 2x2 output elements
  // Thread (tx, ty) computes outputs at:
  // - (2*ty, 2*tx), (2*ty, 2*tx+1), (2*ty+1, 2*tx), (2*ty+1, 2*tx+1)
  float sum[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};

  // Loop over K dimension in chunks of 32
  for (unsigned int k_start = 0; k_start < K; k_start += 32) {

// 1. Load Input to Shared Memory (each thread loads 4 elements)
// Thread (tx, ty) loads input at positions:
// - (2*ty, 2*tx), (2*ty, 2*tx+1), (2*ty+1, 2*tx), (2*ty+1, 2*tx+1)
#pragma unroll
    for (int i = 0; i < 2; i++) {
#pragma unroll
      for (int j = 0; j < 2; j++) {
        unsigned int local_m = 2 * ty + i;
        unsigned int local_k = 2 * tx + j;
        unsigned int global_m = by * 32 + local_m;
        unsigned int global_k = k_start + local_k;

        if (global_m < M && global_k < K) {
          s_input[local_m][local_k] = input[global_m * K + global_k];
        } else {
          s_input[local_m][local_k] = 0.0f;
        }
      }
    }

// 2. Load Weights to Shared Memory (each thread loads 4 elements)
// Thread (tx, ty) loads weights at positions:
// - (2*tx, 2*ty), (2*tx, 2*ty+1), (2*tx+1, 2*ty), (2*tx+1, 2*ty+1)
#pragma unroll
    for (int i = 0; i < 2; i++) {
#pragma unroll
      for (int j = 0; j < 2; j++) {
        unsigned int local_n = 2 * tx + i;
        unsigned int local_k = 2 * ty + j;
        unsigned int global_n = bx * 32 + local_n;
        unsigned int global_k = k_start + local_k;

        if (global_n < N && global_k < K) {
          unsigned int k_parity = local_k % 2;

          unsigned int block_idx = bx * (K / 2) + (global_k / 2);
          unsigned int byte_offset = local_n;
          unsigned int weight_idx = block_idx * 32 + byte_offset;

          uint8_t packed_w = weights[weight_idx];
          int8_t w_val = k_parity == 0 ? (packed_w & 0x0F) : (packed_w >> 4);
          w_val = w_val >= 8 ? w_val - 16 : w_val;

          s_weights[local_n][local_k] = w_val;
        } else {
          s_weights[local_n][local_k] = 0;
        }
      }
    }

    __syncthreads();

// 3. Compute matrix multiplication for 2x2 output tile per thread
#pragma unroll
    for (int i = 0; i < 2; i++) {
#pragma unroll
      for (int j = 0; j < 2; j++) {
        unsigned int local_m = 2 * ty + i;
        unsigned int local_n = 2 * tx + j;
        unsigned int global_m = by * 32 + local_m;
        unsigned int global_n = bx * 32 + local_n;

        if (global_m < M && global_n < N) {
#pragma unroll
          for (int k = 0; k < 32; k++) {
            unsigned int global_k = k_start + k;
            if (global_k >= K)
              break;

            unsigned int group_id = global_k / quantization_group_size;
            unsigned int scale_idx =
              global_n * (K / quantization_group_size) + group_id;
            float w_scale = __half2float(scales[scale_idx]);

            float w_val_f = static_cast<float>(s_weights[local_n][k]);
            float product = s_input[local_m][k] * w_val_f * w_scale;
            sum[i][j] += product;
          }
        }
      }
    }

    __syncthreads();
  }

// 4. Write results (each thread writes 2x2 outputs)
#pragma unroll
  for (int i = 0; i < 2; i++) {
#pragma unroll
    for (int j = 0; j < 2; j++) {
      unsigned int global_m = by * 32 + 2 * ty + i;
      unsigned int global_n = bx * 32 + 2 * tx + j;

      if (global_m < M && global_n < N) {
        output[global_m * N + global_n] = sum[i][j];
      }
    }
  }
}

void gemm_a32_w4_b16x16_s32x32(const float *input, const void *weights,
                               const void *scales, float *output,
                               unsigned int M, unsigned int N, unsigned int K,
                               unsigned int quantization_group_size) {

  // Launch kernel with 16×16 thread blocks
  // Each block still computes 32×32 output tile (each thread computes 2x2)
  dim3 blockDim(16, 16);
  dim3 gridDim((N + 31) / 32, (M + 31) / 32);

  __gemm_a32_w4_b16x16_s32x32<<<gridDim, blockDim>>>(
    input, static_cast<const uint8_t *>(weights),
    static_cast<const __half *>(scales), output, M, N, K,
    quantization_group_size);
}
/**
 * @brief GEMM kernel using WMMA (Tensor Core) for INT8 computation
 *
 * Uses WMMA to accelerate INT8 matrix multiplication.
 * Each warp processes 16x16 output tile using 16x16x16 WMMA operations.
 * Block size: 32x32 threads (4 warps)
 * Each warp handles one 16x16 tile of the 32x32 block output.
 */
__global__ void
__gemm_a8_w4_b32x32_wmma(const int8_t *input, const uint8_t *weights,
                         const __half *scales, const __half *input_scales,
                         float *output, unsigned int M, unsigned int N,
                         unsigned int K, unsigned int quantization_group_size) {

  using namespace nvcuda::wmma;

  // Block dimensions: 32x32 = 1024 threads (32 warps)
  // Grid dimensions: (N+31)/32, (M+31)/32

  unsigned int tx = threadIdx.x; // 0..31
  unsigned int ty = threadIdx.y; // 0..31

  unsigned int bx = blockIdx.x;
  unsigned int by = blockIdx.y;

  // Calculate logical tile mapping
  // We use 32 physical warps (ty = 0..31), but only 4 are needed for the 32x32
  // output block.
  unsigned int pwarp_id = ty; // Each row (ty) is a warp

  // Define which 16x16 tile each of the first 4 warps will handle
  unsigned int warp_row = (pwarp_id / 2) * 16;
  unsigned int warp_col = (pwarp_id % 2) * 16;

  // Shared memory for 32x32 tiles
  __shared__ int8_t s_input[32][32];
  __shared__ int8_t s_weights[32][32];

  // WMMA fragments
  fragment<matrix_a, 16, 16, 16, signed char, row_major> a_frag;
  fragment<matrix_b, 16, 16, 16, signed char, row_major> b_frag;
  fragment<accumulator, 16, 16, 16, int> c_frag;

  // Initialize accumulator (Only for the computing warps)
  if (pwarp_id < 4) {
    fill_fragment(c_frag, 0);
  }

  // Loop over K in chunks of 32
  for (unsigned int k_chunk = 0; k_chunk < (K + 31) / 32; ++k_chunk) {
    unsigned int k_start = k_chunk * 32;

    // 1. Load Input to Shared Memory (Only 4 warps = 128 threads)
    // 32x32 = 1024 elements, 128 threads → 8 elements per thread
    if (pwarp_id < 4) {
      unsigned int thread_id = pwarp_id * 32 + tx; // 0..127

#pragma unroll
      for (int i = 0; i < 8; ++i) {
        unsigned int elem_idx = thread_id * 8 + i; // 0..1023
        unsigned int s_row = elem_idx / 32;
        unsigned int s_col = elem_idx % 32;

        unsigned int g_row = by * 32 + s_row;
        unsigned int g_col = k_start + s_col;

        if (g_row < M && g_col < K) {
          unsigned int alignK = (K + quantization_group_size - 1) /
                                quantization_group_size *
                                quantization_group_size;
          s_input[s_row][s_col] = input[g_row * alignK + g_col];
        } else {
          s_input[s_row][s_col] = 0;
        }
      }
    }

    // 2. Load Weights to Shared Memory (Only 4 warps = 128 threads)
    // 32x32 = 1024 elements, 128 threads → 8 elements per thread
    if (pwarp_id < 4) {
      unsigned int thread_id = pwarp_id * 32 + tx; // 0..127

#pragma unroll
      for (int i = 0; i < 8; ++i) {
        unsigned int elem_idx = thread_id * 8 + i; // 0..1023
        unsigned int s_n = elem_idx / 32; // N dimension in shared memory
        unsigned int s_k = elem_idx % 32; // K dimension in shared memory

        unsigned int g_n = bx * 32 + s_n;
        unsigned int g_k = k_start + s_k;

        if (g_n < N && g_k < K) {
          uint8_t packed_w = weights[(bx * (K / 2) + (g_k / 2)) * 32 + s_n];
          int8_t w_val = (g_k % 2 == 0) ? packed_w & 0x0F : packed_w >> 4;
          if (w_val >= 8)
            w_val -= 16;
          s_weights[s_n][s_k] = w_val;
        } else {
          s_weights[s_n][s_k] = 0;
        }
      }
    }

    __syncthreads();

    // 3. Compute (Only 4 physical warps perform WMMA to avoid 8x redundancy)
    if (pwarp_id < 4) {
#pragma unroll
      for (int k_tile = 0; k_tile < 2; ++k_tile) {
        load_matrix_sync(a_frag, &s_input[warp_row][k_tile * 16], 32);
        load_matrix_sync(b_frag, &s_weights[warp_col][k_tile * 16], 32);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
      }
    }
    __syncthreads();
  }

  // 4. Result Processing (Only for computing warps)
  if (pwarp_id < 4) {
    __shared__ int s_output[32][32];
    store_matrix_sync(&s_output[warp_row][warp_col], c_frag, 32, mem_row_major);

// Each thread of these 4 warps writes back to global memory
// With 4 warps of 32 threads, we have 128 threads covering 32x32=1024 elements.
// Each thread writes 8 elements.
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      unsigned int local_idx = (pwarp_id * 32 + tx) * 8 + i; // 0..1023
      unsigned int l_row = local_idx / 32;
      unsigned int l_col = local_idx % 32;

      unsigned int g_row = by * 32 + l_row;
      unsigned int g_col = bx * 32 + l_col;

      if (g_row < M && g_col < N) {
        int acc_value = s_output[l_row][l_col];

        // Scale calculation (simplified average for now)
        unsigned int alignK = (K + quantization_group_size - 1) /
                              quantization_group_size * quantization_group_size;
        unsigned int groups_in_row = alignK / quantization_group_size;
        unsigned int num_groups =
          (K + quantization_group_size - 1) / quantization_group_size;

        float i_scale_avg = 0.0f;
        for (unsigned int g = 0; g < num_groups; ++g) {
          i_scale_avg +=
            __half2float(input_scales[(g_row * groups_in_row + g) * 2]);
        }
        i_scale_avg /= num_groups;

        float w_scale_avg = 0.0f;
        for (unsigned int g = 0; g < num_groups; ++g) {
          w_scale_avg += __half2float(scales[g_col * num_groups + g]);
        }
        w_scale_avg /= num_groups;

        output[g_row * N + g_col] = acc_value * i_scale_avg * w_scale_avg;
      }
    }
  }
}

void gemm_a8_w4_b32x32_wmma(const void *input, const void *weights,
                            const void *scales, const void *input_scales,
                            float *output, unsigned int M, unsigned int N,
                            unsigned int K,
                            unsigned int quantization_group_size) {
  // blockDim(32,32)=1024 threads is 20-60x faster than blockDim(128)=128
  // threads Both configurations use only 128 threads (4 warps) for actual
  // computation. Performance difference:
  // - (32,32): 0.02ms, 21-69x speedup vs baseline
  // - (128):   0.40-0.94ms, only 1.2-1.6x speedup vs baseline
  // Root cause is currently unknown. Possible factors:
  // - Warp scheduling flexibility (32 warps vs 4 warps)
  // - SM occupancy and instruction cache behavior
  // - Hardware-specific optimizations for larger block sizes
  dim3 blockDim(32, 32);
  dim3 gridDim((N + 31) / 32, (M + 31) / 32);

  __gemm_a8_w4_b32x32_wmma<<<gridDim, blockDim>>>(
    static_cast<const int8_t *>(input), static_cast<const uint8_t *>(weights),
    static_cast<const __half *>(scales),
    static_cast<const __half *>(input_scales), output, M, N, K,
    quantization_group_size);
}

__global__ void __gemm_a8_w4_b16x16(const int8_t *input, const uint8_t *weights,
                                    const __half *scales,
                                    const __half *input_scales, float *output,
                                    unsigned int M, unsigned int N,
                                    unsigned int K,
                                    unsigned int quantization_group_size) {
  // Block dimensions: 16x16
  // Grid dimensions: (N+15)/16, (M+15)/16

  unsigned int tx = threadIdx.x; // 0..15 (N direction within block)
  unsigned int ty = threadIdx.y; // 0..15 (M direction within block)

  unsigned int bx = blockIdx.x;
  unsigned int by = blockIdx.y;

  unsigned int row = by * 16 + ty; // Global row index (M)
  unsigned int col = bx * 16 + tx; // Global col index (N)

  // Shared memory
  // Input block: 16x16 int8_t
  __shared__ int8_t s_input[16][16];

  // Weight block: 16x16 int8_t (unpacked)
  // [N][K] layout to allow contiguous access along K
  __shared__ int8_t s_weights[16][16];

  float sum = 0.0f;

  // Loop over K in chunks of 16
  for (unsigned int k_chunk = 0; k_chunk < (K + 15) / 16; ++k_chunk) {
    unsigned int k_start = k_chunk * 16;

    // 1. Load Input to Shared Memory
    if (row < M && (k_start + tx) < K) {
      unsigned int alignK = (K + quantization_group_size - 1) /
                            quantization_group_size * quantization_group_size;
      unsigned int input_idx = row * alignK + k_start + tx;
      s_input[ty][tx] = input[input_idx];
    } else {
      s_input[ty][tx] = 0;
    }

    // 2. Load Weights to Shared Memory and Unpack
    // Each thread loads one int8 weight
    // n = tx, k = ty (0..15)
    unsigned int n = tx;
    unsigned int k = ty; // 0..15 relative to k_start

    unsigned int global_n = bx * 16 + n;
    unsigned int global_k = k_start + k;

    if (global_n < N && global_k < K) {
      // Reconstruct index from original 32-block layout logic
      unsigned int original_bx = global_n / 32;
      unsigned int offset_in_32 = global_n % 32;
      unsigned int block_idx = original_bx * (K / 2) + (global_k / 2);
      unsigned int weight_idx = block_idx * 32 + offset_in_32;

      uint8_t packed_w = weights[weight_idx];
      int8_t w_val = (global_k % 2 == 0) ? packed_w & 0x0F : packed_w >> 4;
      if (w_val >= 8)
        w_val -= 16;
      s_weights[n][k] = w_val;
    } else {
      s_weights[n][k] = 0;
    }

    __syncthreads();

    // 3. Compute
    int chunk_acc[4] = {0};

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int input_packed = *reinterpret_cast<int *>(&s_input[ty][i * 4]);
      int weight_packed = *reinterpret_cast<int *>(&s_weights[tx][i * 4]);
      chunk_acc[i] = __dp4a(input_packed, weight_packed, chunk_acc[i]);
    }

    {
      int total_acc = 0;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        total_acc += chunk_acc[i];
      }

      // Apply scales
      unsigned int alignK = (K + quantization_group_size - 1) /
                            quantization_group_size * quantization_group_size;
      unsigned int groups_in_row = alignK / quantization_group_size;
      unsigned int group_id =
        (row * groups_in_row + k_start / quantization_group_size);
      float i_scale = __half2float(input_scales[group_id * 2]);

      unsigned int scale_idx =
        col * (K / quantization_group_size) + k_start / quantization_group_size;
      float w_scale = __half2float(scales[scale_idx]);

      sum += total_acc * i_scale * w_scale;
    }
    __syncthreads();
  }

  if (row < M && col < N) {
    output[row * N + col] = sum;
  }
}

void gemm_a8_w4_b16x16(const void *input, const void *weights,
                       const void *scales, const void *input_scales,
                       float *output, unsigned int M, unsigned int N,
                       unsigned int K, unsigned int quantization_group_size) {

  dim3 blockDim(16, 16);
  dim3 gridDim((N + 15) / 16, (M + 15) / 16);

  __gemm_a8_w4_b16x16<<<gridDim, blockDim>>>(
    static_cast<const int8_t *>(input), static_cast<const uint8_t *>(weights),
    static_cast<const __half *>(scales),
    static_cast<const __half *>(input_scales), output, M, N, K,
    quantization_group_size);
}
// New WMMA based kernel
// using namespace nvcuda; // Assuming it might be needed, but usually mma.h
// puts things in nvcuda

__global__ void __gemm_a8_w4_b16x16_s32x32_wmma(
  const int8_t *input, const uint8_t *weights, const __half *scales,
  const __half *input_scales, float *output, unsigned int M, unsigned int N,
  unsigned int K, unsigned int quantization_group_size) {

  unsigned int tx = threadIdx.x;
  unsigned int ty = threadIdx.y;
  unsigned int tid = ty * 16 + tx;

  unsigned int bx = blockIdx.x;
  unsigned int by = blockIdx.y;

  unsigned int row_start = by * 32;
  unsigned int col_start = bx * 32;

  __shared__ int8_t s_input[32][32];
  __shared__ int8_t s_weights[32][32];
  __shared__ int32_t s_output[32][32];

  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, int8_t,
                         nvcuda::wmma::row_major>
    a_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, int8_t,
                         nvcuda::wmma::col_major>
    b_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, int32_t> c_frag;

  float sum[4] = {0.0f};

  for (unsigned int k_chunk = 0; k_chunk < (K + 31) / 32; ++k_chunk) {
    unsigned int k_start = k_chunk * 32;

    // 1. Load Input
    {
      unsigned int r = tid / 8;
      unsigned int c_base = (tid % 8) * 4;
      unsigned int global_r = row_start + r;

#pragma unroll
      for (int i = 0; i < 4; ++i) {
        unsigned int c = c_base + i;
        unsigned int global_c = k_start + c;

        if (global_r < M && global_c < K) {
          unsigned int alignK = (K + quantization_group_size - 1) /
                                quantization_group_size *
                                quantization_group_size;
          unsigned int groups_in_row = alignK / quantization_group_size;
          unsigned int group_id = global_c / quantization_group_size;
          unsigned int offset = global_c % quantization_group_size;
          unsigned int global_group = global_r * groups_in_row + group_id;
          unsigned int idx = global_group * quantization_group_size + offset;
          s_input[r][c] = input[idx];
        } else {
          s_input[r][c] = 0;
        }
      }
    }

    // 2. Load Weights
    {
      unsigned int r = tid / 8;
      unsigned int c_base = (tid % 8) * 4;
      unsigned int global_n = col_start + r;

#pragma unroll
      for (int i = 0; i < 4; ++i) {
        unsigned int c = c_base + i;
        unsigned int global_k = k_start + c;

        int8_t w_val = 0;
        if (global_n < N && global_k < K) {
          unsigned int k_parity = global_k % 2;
          unsigned int n_blk = global_n / 32;
          unsigned int k_blk = global_k / 2;
          unsigned int block_idx = n_blk * (K / 2) + k_blk;
          unsigned int byte_offset = global_n % 32;
          unsigned int weight_idx = block_idx * 32 + byte_offset;
          uint8_t packed_w = weights[weight_idx];
          w_val = k_parity == 0 ? (packed_w & 0x0F) : (packed_w >> 4);
          if (w_val >= 8)
            w_val -= 16;
        }
        s_weights[r][c] = w_val;
      }
    }
    __syncthreads();

    // 3. WMMA Compute
    int warp_id = tid / 32;
    if (warp_id < 4) {
      nvcuda::wmma::fill_fragment(c_frag, 0);
      int wm = (warp_id / 2) * 16;
      int wn = (warp_id % 2) * 16;

      for (int k_step = 0; k_step < 32; k_step += 16) {
        nvcuda::wmma::load_matrix_sync(a_frag, &s_input[wm][k_step], 32);
        nvcuda::wmma::load_matrix_sync(b_frag, &s_weights[wn][k_step], 32);
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      }

      nvcuda::wmma::store_matrix_sync(&s_output[wm][wn], c_frag, 32,
                                      nvcuda::wmma::mem_row_major);
    }
    __syncthreads();

    // 4. Accumulate
    unsigned int rows[2] = {ty, ty + 16};
    unsigned int cols[2] = {tx, tx + 16};

    unsigned int alignK = (K + quantization_group_size - 1) /
                          quantization_group_size * quantization_group_size;
    unsigned int groups_in_row = alignK / quantization_group_size;
    unsigned int group_id_in_row = k_start / quantization_group_size;

    for (int r_idx = 0; r_idx < 2; ++r_idx) {
      unsigned int r = rows[r_idx];
      unsigned int global_r = row_start + r;
      if (global_r >= M)
        continue;

      unsigned int global_group_id = global_r * groups_in_row + group_id_in_row;
      float i_scale = __half2float(input_scales[global_group_id * 2]);

      for (int c_idx = 0; c_idx < 2; ++c_idx) {
        unsigned int c = cols[c_idx];
        unsigned int global_c = col_start + c;
        if (global_c >= N)
          continue;

        unsigned int scale_idx =
          global_c * (K / quantization_group_size) + group_id_in_row;
        float w_scale = __half2float(scales[scale_idx]);

        int32_t val = s_output[r][c];
        sum[r_idx * 2 + c_idx] += (float)val * i_scale * w_scale;
      }
    }
    __syncthreads();
  }

  // 5. Store
  unsigned int rows[2] = {row_start + ty, row_start + ty + 16};
  unsigned int cols[2] = {col_start + tx, col_start + tx + 16};

  for (int r_idx = 0; r_idx < 2; ++r_idx) {
    unsigned int r = rows[r_idx];
    if (r >= M)
      continue;
    for (int c_idx = 0; c_idx < 2; ++c_idx) {
      unsigned int c = cols[c_idx];
      if (c >= N)
        continue;
      output[r * N + c] = sum[r_idx * 2 + c_idx];
    }
  }
}

void gemm_a8_w4_b16x16_s32x32_wmma(const void *input, const void *weights,
                                   const void *scales, const void *input_scales,
                                   float *output, unsigned int M,
                                   unsigned int N, unsigned int K,
                                   unsigned int quantization_group_size) {

  // Launch Kernel
  dim3 blockDim(16, 16);
  dim3 gridDim((N + 31) / 32, (M + 31) / 32);

  __gemm_a8_w4_b16x16_s32x32_wmma<<<gridDim, blockDim>>>(
    static_cast<const int8_t *>(input), static_cast<const uint8_t *>(weights),
    static_cast<const __half *>(scales),
    static_cast<const __half *>(input_scales), output, M, N, K,
    quantization_group_size);
}

__global__ void
__gemm_a8_w4_b16x16_wmma(const int8_t *input, const uint8_t *weights,
                         const __half *scales, const __half *input_scales,
                         float *output, unsigned int M, unsigned int N,
                         unsigned int K, unsigned int quantization_group_size) {

  unsigned int tx = threadIdx.x;
  unsigned int ty = threadIdx.y;
  unsigned int tid = ty * 16 + tx;

  unsigned int bx = blockIdx.x;
  unsigned int by = blockIdx.y;

  unsigned int row_start = by * 32;
  unsigned int col_start = bx * 32;

  __shared__ int8_t s_input[32][32];
  __shared__ int8_t s_weights[32][32];
  __shared__ int32_t s_output[32][32];

  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, int8_t,
                         nvcuda::wmma::row_major>
    a_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, int8_t,
                         nvcuda::wmma::col_major>
    b_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, int32_t> c_frag;

  float sum[4] = {0.0f};

  for (unsigned int k_chunk = 0; k_chunk < (K + 31) / 32; ++k_chunk) {
    unsigned int k_start = k_chunk * 32;

    // 1. Load Input
    {
      unsigned int r = tid / 8;
      unsigned int c_base = (tid % 8) * 4;
      unsigned int global_r = row_start + r;

#pragma unroll
      for (int i = 0; i < 4; ++i) {
        unsigned int c = c_base + i;
        unsigned int global_c = k_start + c;

        if (global_r < M && global_c < K) {
          unsigned int alignK = (K + quantization_group_size - 1) /
                                quantization_group_size *
                                quantization_group_size;
          unsigned int groups_in_row = alignK / quantization_group_size;
          unsigned int group_id = global_c / quantization_group_size;
          unsigned int offset = global_c % quantization_group_size;
          unsigned int global_group = global_r * groups_in_row + group_id;
          unsigned int idx = global_group * quantization_group_size + offset;
          s_input[r][c] = input[idx];
        } else {
          s_input[r][c] = 0;
        }
      }
    }

    // 2. Load Weights
    {
      unsigned int r = tid / 8;
      unsigned int c_base = (tid % 8) * 4;
      unsigned int global_n = col_start + r;

#pragma unroll
      for (int i = 0; i < 4; ++i) {
        unsigned int c = c_base + i;
        unsigned int global_k = k_start + c;

        int8_t w_val = 0;
        if (global_n < N && global_k < K) {
          unsigned int k_parity = global_k % 2;
          unsigned int n_blk = global_n / 32;
          unsigned int k_blk = global_k / 2;
          unsigned int block_idx = n_blk * (K / 2) + k_blk;
          unsigned int byte_offset = global_n % 32;
          unsigned int weight_idx = block_idx * 32 + byte_offset;
          uint8_t packed_w = weights[weight_idx];
          w_val = k_parity == 0 ? (packed_w & 0x0F) : (packed_w >> 4);
          if (w_val >= 8)
            w_val -= 16;
        }
        s_weights[r][c] = w_val;
      }
    }
    __syncthreads();

    // 3. WMMA Compute
    int warp_id = tid / 32;
    if (warp_id < 4) {
      nvcuda::wmma::fill_fragment(c_frag, 0);
      int wm = (warp_id / 2) * 16;
      int wn = (warp_id % 2) * 16;

      for (int k_step = 0; k_step < 32; k_step += 16) {
        nvcuda::wmma::load_matrix_sync(a_frag, &s_input[wm][k_step], 32);
        nvcuda::wmma::load_matrix_sync(b_frag, &s_weights[wn][k_step], 32);
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      }

      nvcuda::wmma::store_matrix_sync(&s_output[wm][wn], c_frag, 32,
                                      nvcuda::wmma::mem_row_major);
    }
    __syncthreads();

    // 4. Accumulate
    unsigned int rows[2] = {ty, ty + 16};
    unsigned int cols[2] = {tx, tx + 16};

    unsigned int alignK = (K + quantization_group_size - 1) /
                          quantization_group_size * quantization_group_size;
    unsigned int groups_in_row = alignK / quantization_group_size;
    unsigned int group_id_in_row = k_start / quantization_group_size;

    for (int r_idx = 0; r_idx < 2; ++r_idx) {
      unsigned int r = rows[r_idx];
      unsigned int global_r = row_start + r;
      if (global_r >= M)
        continue;

      unsigned int global_group_id = global_r * groups_in_row + group_id_in_row;
      float i_scale = __half2float(input_scales[global_group_id * 2]);

      for (int c_idx = 0; c_idx < 2; ++c_idx) {
        unsigned int c = cols[c_idx];
        unsigned int global_c = col_start + c;
        if (global_c >= N)
          continue;

        unsigned int scale_idx =
          global_c * (K / quantization_group_size) + group_id_in_row;
        float w_scale = __half2float(scales[scale_idx]);

        int32_t val = s_output[r][c];
        sum[r_idx * 2 + c_idx] += (float)val * i_scale * w_scale;
      }
    }
    __syncthreads();
  }

  // 5. Store
  unsigned int rows[2] = {row_start + ty, row_start + ty + 16};
  unsigned int cols[2] = {col_start + tx, col_start + tx + 16};

  for (int r_idx = 0; r_idx < 2; ++r_idx) {
    unsigned int r = rows[r_idx];
    if (r >= M)
      continue;
    for (int c_idx = 0; c_idx < 2; ++c_idx) {
      unsigned int c = cols[c_idx];
      if (c >= N)
        continue;
      output[r * N + c] = sum[r_idx * 2 + c_idx];
    }
  }
}

void gemm_a8_w4_b16x16_wmma(const void *input, const void *weights,
                            const void *scales, const void *input_scales,
                            float *output, unsigned int M, unsigned int N,
                            unsigned int K,
                            unsigned int quantization_group_size) {

  // Launch Kernel
  dim3 blockDim(16, 16);
  dim3 gridDim((N + 31) / 32, (M + 31) / 32);

  __gemm_a8_w4_b16x16_wmma<<<gridDim, blockDim>>>(
    static_cast<const int8_t *>(input), static_cast<const uint8_t *>(weights),
    static_cast<const __half *>(scales),
    static_cast<const __half *>(input_scales), output, M, N, K,
    quantization_group_size);
}

__global__ void __gemm_a8_w4_b16x16_s64x64_wmma(
  const int8_t *input, const uint8_t *weights, const __half *scales,
  const __half *input_scales, float *output, unsigned int M, unsigned int N,
  unsigned int K, unsigned int quantization_group_size) {

  unsigned int tx = threadIdx.x;
  unsigned int ty = threadIdx.y;
  unsigned int tid = ty * 16 + tx;

  unsigned int bx = blockIdx.x;
  unsigned int by = blockIdx.y;

  unsigned int row_start = by * 64;
  unsigned int col_start = bx * 64;

  __shared__ int8_t s_input[64][64];
  __shared__ int8_t s_weights[64][64];
  __shared__ int32_t s_output[64][64];

  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, int8_t,
                         nvcuda::wmma::row_major>
    a_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, int8_t,
                         nvcuda::wmma::col_major>
    b_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, int32_t> c_frag;

  float sum[16] = {0.0f};

  for (unsigned int k_chunk = 0; k_chunk < (K + 63) / 64; ++k_chunk) {
    unsigned int k_start = k_chunk * 64;

    // 1. Load Input (64x64)
    {
      unsigned int r_base = tid / 4;
      unsigned int c_base = (tid % 4) * 16;

      unsigned int r = r_base; // 0..63
      unsigned int global_r = row_start + r;

      if (r < 64) {
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          unsigned int c = c_base + i;
          unsigned int global_c = k_start + c;
          int8_t val = 0;
          if (global_r < M && global_c < K) {
            unsigned int alignK = (K + quantization_group_size - 1) /
                                  quantization_group_size *
                                  quantization_group_size;
            unsigned int groups_in_row = alignK / quantization_group_size;
            unsigned int group_id = global_c / quantization_group_size;
            unsigned int offset = global_c % quantization_group_size;
            unsigned int global_group = global_r * groups_in_row + group_id;
            unsigned int idx = global_group * quantization_group_size + offset;
            val = input[idx];
          }
          s_input[r][c] = val;
        }
      }
    }

    // 2. Load Weights (64x64)
    {
      unsigned int r_base = tid / 4;
      unsigned int c_base = (tid % 4) * 16;
      unsigned int r = r_base;
      unsigned int global_n = col_start + r;

      if (r < 64) {
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          unsigned int c = c_base + i;
          unsigned int global_k = k_start + c;
          int8_t w_val = 0;
          if (global_n < N && global_k < K) {
            unsigned int k_parity = global_k % 2;
            unsigned int n_blk = global_n / 32;
            unsigned int k_blk = global_k / 2;
            unsigned int block_idx = n_blk * (K / 2) + k_blk;
            unsigned int byte_offset = global_n % 32;
            unsigned int weight_idx = block_idx * 32 + byte_offset;
            uint8_t packed_w = weights[weight_idx];
            w_val = k_parity == 0 ? (packed_w & 0x0F) : (packed_w >> 4);
            if (w_val >= 8)
              w_val -= 16;
          }
          s_weights[r][c] = w_val;
        }
      }
    }
    __syncthreads();

    // 3. WMMA Compute & Accumulate
    for (int sub_k = 0; sub_k < 64; sub_k += quantization_group_size) {

      int warp_id = tid / 32;

      // 8 Warps -> 16 Tiles (64x64 output = 4x4 tiles).
      // Each Warp computes 2 tiles.
      int w_idx = warp_id; // 0..7

      // Tiles linear index 0..15.
      // Warp 0: 0, 1. Warp 1: 2, 3...
      for (int t = 0; t < 2; ++t) {
        nvcuda::wmma::fill_fragment(c_frag, 0);
        int tile_idx = w_idx * 2 + t; // 0..15
        int tile_r = tile_idx / 4;    // 0..3
        int tile_c = tile_idx % 4;    // 0..3

        int wm = tile_r * 16;
        int wn = tile_c * 16;

        if (wm < 64 && wn < 64) {
          for (int k_step = 0; k_step < quantization_group_size; k_step += 16) {
            int current_k = sub_k + k_step;
            nvcuda::wmma::load_matrix_sync(a_frag, &s_input[wm][current_k], 64);
            nvcuda::wmma::load_matrix_sync(b_frag, &s_weights[wn][current_k],
                                           64);
            nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
          }
          nvcuda::wmma::store_matrix_sync(&s_output[wm][wn], c_frag, 64,
                                          nvcuda::wmma::mem_row_major);
        }
      }

      __syncthreads();

      unsigned int r_base = tid / 4;
      unsigned int c_base = (tid % 4) * 16;

      if (r_base < 64) {
        unsigned int global_r = row_start + r_base;

        unsigned int current_global_k = k_start + sub_k;
        unsigned int alignK = (K + quantization_group_size - 1) /
                              quantization_group_size * quantization_group_size;
        unsigned int groups_in_row = alignK / quantization_group_size;
        unsigned int group_id_in_row =
          current_global_k / quantization_group_size;

        float i_scale = 0.0f;
        if (global_r < M) {
          unsigned int global_group_id =
            global_r * groups_in_row + group_id_in_row;
          i_scale = __half2float(input_scales[global_group_id * 2]);
        }

        for (int i = 0; i < 16; ++i) {
          unsigned int c = c_base + i;
          unsigned int global_c = col_start + c;

          if (global_r < M && global_c < N) {
            unsigned int scale_idx =
              global_c * (K / quantization_group_size) + group_id_in_row;
            float w_scale = __half2float(scales[scale_idx]);

            int32_t val = s_output[r_base][c];
            sum[i] += (float)val * i_scale * w_scale;
          }
        }
      }
      __syncthreads();
    }
  }

  // Store Global
  unsigned int r_base = tid / 4;
  unsigned int c_base = (tid % 4) * 16;

  if (r_base < 64) {
    unsigned int global_r = row_start + r_base;
    for (int i = 0; i < 16; ++i) {
      unsigned int c = c_base + i;
      unsigned int global_c = col_start + c;
      if (global_r < M && global_c < N) {
        output[global_r * N + global_c] = sum[i];
      }
    }
  }
}

void gemm_a8_w4_b16x16_s64x64_wmma(const void *input, const void *weights,
                                   const void *scales, const void *input_scales,
                                   float *output, unsigned int M,
                                   unsigned int N, unsigned int K,
                                   unsigned int quantization_group_size) {

  dim3 blockDim(16, 16);
  dim3 gridDim((N + 63) / 64, (M + 63) / 64);

  __gemm_a8_w4_b16x16_s64x64_wmma<<<gridDim, blockDim>>>(
    static_cast<const int8_t *>(input), static_cast<const uint8_t *>(weights),
    static_cast<const __half *>(scales),
    static_cast<const __half *>(input_scales), output, M, N, K,
    quantization_group_size);
}

__global__ void __gemm_a8_w4_b16x16_s32x32_wmma_vl(
  const int8_t *input, const uint8_t *weights, const __half *scales,
  const __half *input_scales, float *output, unsigned int M, unsigned int N,
  unsigned int K, unsigned int quantization_group_size) {

  unsigned int tid = threadIdx.y * 16 + threadIdx.x;
  unsigned int row_start = blockIdx.y * 32;
  unsigned int col_start = blockIdx.x * 32;

  __shared__ int8_t s_input[32][32];
  __shared__ int8_t s_weights[32][32];
  __shared__ int32_t s_output[32][32];

  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, int8_t,
                         nvcuda::wmma::row_major>
    a_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, int8_t,
                         nvcuda::wmma::col_major>
    b_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, int32_t> c_frag;

  float sum[4] = {0.0f};

  // Shared Constants for Indexing
  const unsigned int groups_in_row =
    (K + quantization_group_size - 1) / quantization_group_size;
  const unsigned int r = tid / 8;
  const unsigned int c_base = (tid % 8) * 4;

  for (unsigned int k_chunk = 0; k_chunk < (K + 31) / 32; ++k_chunk) {
    unsigned int k_start = k_chunk * 32;

    // 1. Load Input (Vectorized)
    {
      unsigned int gr = row_start + r;
      unsigned int gc_base = k_start + c_base;

      if (gr < M && gc_base + 4 <= K) {
        unsigned int idx =
          gr * groups_in_row * quantization_group_size + gc_base;
        // Vectorized load: Reading 4 bytes (32-bit int) at a time.
        *reinterpret_cast<int *>(&s_input[r][c_base]) =
          *reinterpret_cast<const int *>(&input[idx]);
      } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          unsigned int gc = gc_base + i;
          if (gr < M && gc < K) {
            unsigned int idx =
              gr * groups_in_row * quantization_group_size + gc;
            s_input[r][c_base + i] = input[idx];
          } else {
            s_input[r][c_base + i] = 0;
          }
        }
      }
    }

    // 2. Load Weights (Optimized Vectorized Dequantize)
    {
      unsigned int gn = col_start + r;
      unsigned int gk_base = k_start + c_base;
      uint32_t val_packed = 0;

      if (gn < N && gk_base < K) {
        // Calculate Base Index for w0, w1
        // Index formula: ((gn / 32) * (K / 2) + (gk / 2)) * 32 + (gn % 32)
        unsigned int w_idx_base =
          ((gn >> 5) * (K >> 1) + (gk_base >> 1)) * 32 + (gn & 31);

        uint8_t p0 = weights[w_idx_base];
        uint8_t p1 = 0;

        // Check if we can read second byte (w2, w3)
        if (gk_base + 2 < K) {
          p1 = weights[w_idx_base + 32];
        }

        // Unpack p0 (w0, w1) with sign extension
        int8_t v0 = (int8_t)(p0 << 4) >> 4;
        int8_t v1 = (int8_t)p0 >> 4;

        // Unpack p1 (w2, w3) with sign extension
        int8_t v2 = (int8_t)(p1 << 4) >> 4;
        int8_t v3 = (int8_t)p1 >> 4;

        // Boundary Masking (for odd-sized K which is rare for int4)
        if (gk_base + 1 >= K)
          v1 = 0;

        // Pack into 32-bit integer for vectorized store
        val_packed = ((uint8_t)v0) | (((uint8_t)v1) << 8) |
                     (((uint8_t)v2) << 16) | (((uint8_t)v3) << 24);
      }
      *(int *)&s_weights[r][c_base] = val_packed;
    }
    __syncthreads();

    // 3. WMMA Compute
    int warp_id = tid / 32;
    if (warp_id < 4) {
      nvcuda::wmma::fill_fragment(c_frag, 0);
      int wm = (warp_id / 2) * 16;
      int wn = (warp_id % 2) * 16;

      for (int k_step = 0; k_step < 32; k_step += 16) {
        nvcuda::wmma::load_matrix_sync(a_frag, &s_input[wm][k_step], 32);
        nvcuda::wmma::load_matrix_sync(b_frag, &s_weights[wn][k_step], 32);
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      }
      nvcuda::wmma::store_matrix_sync(&s_output[wm][wn], c_frag, 32,
                                      nvcuda::wmma::mem_row_major);
    }
    __syncthreads();

    // 4. Accumulate
    unsigned int rows[2] = {threadIdx.y, threadIdx.y + 16};
    unsigned int cols[2] = {threadIdx.x, threadIdx.x + 16};
    unsigned int gid_row = k_start / quantization_group_size;

    for (int r_idx = 0; r_idx < 2; ++r_idx) {
      unsigned int tr = rows[r_idx];
      unsigned int gr = row_start + tr;
      if (gr >= M)
        continue;

      float i_scale =
        __half2float(input_scales[(gr * groups_in_row + gid_row) * 2]);

      for (int c_idx = 0; c_idx < 2; ++c_idx) {
        unsigned int tc = cols[c_idx];
        unsigned int gc = col_start + tc;
        if (gc >= N)
          continue;

        float w_scale =
          __half2float(scales[gc * (K / quantization_group_size) + gid_row]);
        sum[r_idx * 2 + c_idx] += (float)s_output[tr][tc] * i_scale * w_scale;
      }
    }
    __syncthreads();
  }

  // 5. Store
  unsigned int rows[2] = {row_start + threadIdx.y,
                          row_start + threadIdx.y + 16};
  unsigned int cols[2] = {col_start + threadIdx.x,
                          col_start + threadIdx.x + 16};

  for (int r_idx = 0; r_idx < 2; ++r_idx) {
    if (rows[r_idx] >= M)
      continue;
    for (int c_idx = 0; c_idx < 2; ++c_idx) {
      if (cols[c_idx] >= N)
        continue;
      output[rows[r_idx] * N + cols[c_idx]] = sum[r_idx * 2 + c_idx];
    }
  }
}

void gemm_a8_w4_b16x16_s32x32_wmma_vl(const void *input, const void *weights,
                                      const void *scales,
                                      const void *input_scales, float *output,
                                      unsigned int M, unsigned int N,
                                      unsigned int K,
                                      unsigned int quantization_group_size) {

  // Launch Kernel
  dim3 blockDim(16, 16);
  dim3 gridDim((N + 31) / 32, (M + 31) / 32);

  __gemm_a8_w4_b16x16_s32x32_wmma_vl<<<gridDim, blockDim>>>(
    static_cast<const int8_t *>(input), static_cast<const uint8_t *>(weights),
    static_cast<const __half *>(scales),
    static_cast<const __half *>(input_scales), output, M, N, K,
    quantization_group_size);
}

__global__ void __gemm_a8_w4_b16x16_s32x32_wmma_cpasync(
  const int8_t *input, const uint8_t *weights, const __half *scales,
  const __half *input_scales, float *output, unsigned int M, unsigned int N,
  unsigned int K, unsigned int quantization_group_size) {

  unsigned int tid = threadIdx.y * 16 + threadIdx.x;
  unsigned int row_start = blockIdx.y * 32;
  unsigned int col_start = blockIdx.x * 32;

  __shared__ int8_t s_input[32][32];
  __shared__ int8_t s_weights[32][32];
  __shared__ int32_t s_output[32][32];

  // Use thread-scope pipeline to avoid deadlock with partial block
  // participation.
  // __syncthreads() after consumer_wait() ensures block-wide visibility.
  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, int8_t,
                         nvcuda::wmma::row_major>
    a_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, int8_t,
                         nvcuda::wmma::col_major>
    b_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, int32_t> c_frag;

  float sum[4] = {0.0f};

  const unsigned int groups_in_row =
    (K + quantization_group_size - 1) / quantization_group_size;
  const unsigned int r = tid / 8;
  const unsigned int c_base = (tid % 8) * 4;

  for (unsigned int k_chunk = 0; k_chunk < (K + 31) / 32; ++k_chunk) {
    unsigned int k_start = k_chunk * 32;

    // 1. Load Input (Async using cuda::memcpy_async)
    {
      unsigned int gr = row_start + r;
      unsigned int gc_base = k_start + c_base;

      if (gr < M && gc_base + 4 <= K) {
        unsigned int idx =
          gr * groups_in_row * quantization_group_size + gc_base;
        cuda::memcpy_async(&s_input[r][c_base], &input[idx], sizeof(int), pipe);
      } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          unsigned int gc = gc_base + i;
          if (gr < M && gc < K) {
            unsigned int idx =
              gr * groups_in_row * quantization_group_size + gc;
            s_input[r][c_base + i] = input[idx];
          } else {
            s_input[r][c_base + i] = 0;
          }
        }
      }
    }
    pipe.producer_commit();

    // 2. Load Weights (Sync)
    {
      unsigned int gn = col_start + r;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        unsigned int gk = k_start + c_base + i;
        int8_t w_val = 0;
        if (gn < N && gk < K) {
          unsigned int w_idx =
            ((gn / 32) * (K / 2) + (gk / 2)) * 32 + (gn % 32);
          uint8_t packed = weights[w_idx];
          w_val = (gk % 2) ? (packed >> 4) : (packed & 0x0F);
          if (w_val >= 8)
            w_val -= 16;
        }
        s_weights[r][c_base + i] = w_val;
      }
    }

    pipe.consumer_wait();
    __syncthreads();

    // 3. WMMA Compute
    int warp_id = tid / 32;
    if (warp_id < 4) {
      nvcuda::wmma::fill_fragment(c_frag, 0);
      int wm = (warp_id / 2) * 16;
      int wn = (warp_id % 2) * 16;

      for (int k_step = 0; k_step < 32; k_step += 16) {
        nvcuda::wmma::load_matrix_sync(a_frag, &s_input[wm][k_step], 32);
        nvcuda::wmma::load_matrix_sync(b_frag, &s_weights[wn][k_step], 32);
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      }
      nvcuda::wmma::store_matrix_sync(&s_output[wm][wn], c_frag, 32,
                                      nvcuda::wmma::mem_row_major);
    }
    __syncthreads();

    // 4. Accumulate
    unsigned int rows[2] = {threadIdx.y, threadIdx.y + 16};
    unsigned int cols[2] = {threadIdx.x, threadIdx.x + 16};
    unsigned int gid_row = k_start / quantization_group_size;

    for (int r_idx = 0; r_idx < 2; ++r_idx) {
      unsigned int tr = rows[r_idx];
      unsigned int gr = row_start + tr;
      if (gr >= M)
        continue;

      float i_scale =
        __half2float(input_scales[(gr * groups_in_row + gid_row) * 2]);

      for (int c_idx = 0; c_idx < 2; ++c_idx) {
        unsigned int tc = cols[c_idx];
        unsigned int gc = col_start + tc;
        if (gc >= N)
          continue;

        float w_scale =
          __half2float(scales[gc * (K / quantization_group_size) + gid_row]);
        sum[r_idx * 2 + c_idx] += (float)s_output[tr][tc] * i_scale * w_scale;
      }
    }
    __syncthreads();
  }

  // 5. Store
  unsigned int rows[2] = {row_start + threadIdx.y,
                          row_start + threadIdx.y + 16};
  unsigned int cols[2] = {col_start + threadIdx.x,
                          col_start + threadIdx.x + 16};

  for (int r_idx = 0; r_idx < 2; ++r_idx) {
    if (rows[r_idx] >= M)
      continue;
    for (int c_idx = 0; c_idx < 2; ++c_idx) {
      if (cols[c_idx] >= N)
        continue;
      output[rows[r_idx] * N + cols[c_idx]] = sum[r_idx * 2 + c_idx];
    }
  }
}

void gemm_a8_w4_b16x16_s32x32_wmma_cpasync(
  const void *input, const void *weights, const void *scales,
  const void *input_scales, float *output, unsigned int M, unsigned int N,
  unsigned int K, unsigned int quantization_group_size) {
  dim3 blockDim(16, 16);
  dim3 gridDim((N + 31) / 32, (M + 31) / 32);
  __gemm_a8_w4_b16x16_s32x32_wmma_cpasync<<<gridDim, blockDim>>>(
    static_cast<const int8_t *>(input), static_cast<const uint8_t *>(weights),
    static_cast<const __half *>(scales),
    static_cast<const __half *>(input_scales), output, M, N, K,
    quantization_group_size);
}

__global__ void __gemm_a8_w4_b8x16_s32x32_wmma_vl(
  const int8_t *input, const uint8_t *weights, const __half *scales,
  const __half *input_scales, float *output, unsigned int M, unsigned int N,
  unsigned int K, unsigned int quantization_group_size) {

  unsigned int tid = threadIdx.y * 16 + threadIdx.x;
  unsigned int row_start = blockIdx.y * 32;
  unsigned int col_start = blockIdx.x * 32;

  __shared__ int8_t s_input[32][32];
  __shared__ int8_t s_weights[32][32];
  __shared__ int32_t s_output[32][32];

  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, int8_t,
                         nvcuda::wmma::row_major>
    a_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, int8_t,
                         nvcuda::wmma::col_major>
    b_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, int32_t> c_frag;

  float sum[8] = {0.0f};

  const unsigned int groups_in_row =
    (K + quantization_group_size - 1) / quantization_group_size;

  for (unsigned int k_chunk = 0; k_chunk < (K + 31) / 32; ++k_chunk) {
    unsigned int k_start = k_chunk * 32;

// 1. Load Input (32x32 int8, 128 threads -> 2 ints per thread)
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      unsigned int load_idx = tid + i * 128;
      unsigned int r = load_idx / 8;
      unsigned int c_int = load_idx % 8;
      unsigned int c_base = c_int * 4;

      unsigned int gr = row_start + r;
      unsigned int gc_base = k_start + c_base;

      if (gr < M && gc_base + 4 <= K) {
        unsigned int idx =
          gr * groups_in_row * quantization_group_size + gc_base;
        *(int *)&s_input[r][c_base] = *(const int *)&input[idx];
      } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          unsigned int gc = gc_base + j;
          if (gr < M && gc < K) {
            unsigned int idx =
              gr * groups_in_row * quantization_group_size + gc;
            s_input[r][c_base + j] = input[idx];
          } else {
            s_input[r][c_base + j] = 0;
          }
        }
      }
    }

// 2. Load Weights (32x32 int4 -> int8, 128 threads -> 8 elems per thread)
// Optimized vectorized dequantize
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      unsigned int load_idx = tid + i * 128;
      unsigned int r = load_idx / 8;
      unsigned int c_base = (load_idx % 8) * 4;

      unsigned int gn = col_start + r;
      unsigned int gk_base = k_start + c_base;
      uint32_t val_packed = 0;

      if (gn < N && gk_base < K) {
        unsigned int w_idx_base =
          ((gn >> 5) * (K >> 1) + (gk_base >> 1)) * 32 + (gn & 31);
        uint8_t p0 = weights[w_idx_base];
        uint8_t p1 = (gk_base + 2 < K) ? weights[w_idx_base + 32] : 0;

        int8_t v0 = (int8_t)(p0 << 4) >> 4;
        int8_t v1 = (int8_t)p0 >> 4;
        int8_t v2 = (int8_t)(p1 << 4) >> 4;
        int8_t v3 = (int8_t)p1 >> 4;

        if (gk_base + 1 >= K)
          v1 = 0;
        val_packed = ((uint8_t)v0) | (((uint8_t)v1) << 8) |
                     (((uint8_t)v2) << 16) | (((uint8_t)v3) << 24);
      }
      *(int *)&s_weights[r][c_base] = val_packed;
    }
    __syncthreads();

    // 3. WMMA Compute
    int warp_id = tid / 32;
    nvcuda::wmma::fill_fragment(c_frag, 0);
    int wm = (warp_id / 2) * 16;
    int wn = (warp_id % 2) * 16;

    for (int k_step = 0; k_step < 32; k_step += 16) {
      nvcuda::wmma::load_matrix_sync(a_frag, &s_input[wm][k_step], 32);
      nvcuda::wmma::load_matrix_sync(b_frag, &s_weights[wn][k_step], 32);
      nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    nvcuda::wmma::store_matrix_sync(&s_output[wm][wn], c_frag, 32,
                                    nvcuda::wmma::mem_row_major);
    __syncthreads();

    // 4. Accumulate
    // Thread covers 8 elements.
    unsigned int rows[4] = {threadIdx.y, threadIdx.y + 8, threadIdx.y + 16,
                            threadIdx.y + 24};
    unsigned int cols[2] = {threadIdx.x, threadIdx.x + 16};
    unsigned int gid_row = k_start / quantization_group_size;

    for (int r_idx = 0; r_idx < 4; ++r_idx) {
      unsigned int tr = rows[r_idx];
      unsigned int gr = row_start + tr;
      if (gr >= M)
        continue;

      float i_scale =
        __half2float(input_scales[(gr * groups_in_row + gid_row) * 2]);

      for (int c_idx = 0; c_idx < 2; ++c_idx) {
        unsigned int tc = cols[c_idx];
        unsigned int gc = col_start + tc;
        if (gc >= N)
          continue;

        float w_scale =
          __half2float(scales[gc * (K / quantization_group_size) + gid_row]);
        sum[r_idx * 2 + c_idx] += (float)s_output[tr][tc] * i_scale * w_scale;
      }
    }
    __syncthreads();
  }

  // 5. Store
  unsigned int rows[4] = {row_start + threadIdx.y, row_start + threadIdx.y + 8,
                          row_start + threadIdx.y + 16,
                          row_start + threadIdx.y + 24};
  unsigned int cols[2] = {col_start + threadIdx.x,
                          col_start + threadIdx.x + 16};

  for (int r_idx = 0; r_idx < 4; ++r_idx) {
    if (rows[r_idx] >= M)
      continue;
    for (int c_idx = 0; c_idx < 2; ++c_idx) {
      if (cols[c_idx] >= N)
        continue;
      output[rows[r_idx] * N + cols[c_idx]] = sum[r_idx * 2 + c_idx];
    }
  }
}

void gemm_a8_w4_b8x16_s32x32_wmma_vl(const void *input, const void *weights,
                                     const void *scales,
                                     const void *input_scales, float *output,
                                     unsigned int M, unsigned int N,
                                     unsigned int K,
                                     unsigned int quantization_group_size) {
  dim3 blockDim(16, 8);
  dim3 gridDim((N + 31) / 32, (M + 31) / 32);
  __gemm_a8_w4_b8x16_s32x32_wmma_vl<<<gridDim, blockDim>>>(
    static_cast<const int8_t *>(input), static_cast<const uint8_t *>(weights),
    static_cast<const __half *>(scales),
    static_cast<const __half *>(input_scales), output, M, N, K,
    quantization_group_size);
}

__global__ void __gemm_a8_w4_b16x16_s32x32_wmma_cpasync_splitk(
  const int8_t *input, const uint8_t *weights, const __half *scales,
  const __half *input_scales, float *output, unsigned int M, unsigned int N,
  unsigned int K, unsigned int quantization_group_size,
  unsigned int split_k_slices) {

  unsigned int tid = threadIdx.y * 16 + threadIdx.x;
  unsigned int row_start = blockIdx.y * 32;
  unsigned int col_start = blockIdx.x * 32;
  unsigned int split_idx = blockIdx.z;
  __shared__ int8_t s_input[32][32];
  __shared__ int8_t s_weights[32][32];
  __shared__ int32_t s_output[32][32];
  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, int8_t,
                         nvcuda::wmma::row_major>
    a_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, int8_t,
                         nvcuda::wmma::col_major>
    b_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, int32_t> c_frag;
  float sum[4] = {0.0f};

  const unsigned int groups_in_row =
    (K + quantization_group_size - 1) / quantization_group_size;
  const unsigned int r = tid / 8;
  const unsigned int c_base = (tid % 8) * 4;
  // Split-K Logic: Calculate K range for this block
  unsigned int total_k_chunks = (K + 31) / 32;
  unsigned int chunks_per_split =
    (total_k_chunks + split_k_slices - 1) / split_k_slices;
  unsigned int k_chunk_start = split_idx * chunks_per_split;
  unsigned int k_chunk_end =
    min(k_chunk_start + chunks_per_split, total_k_chunks);
  for (unsigned int k_chunk = k_chunk_start; k_chunk < k_chunk_end; ++k_chunk) {
    unsigned int k_start = k_chunk * 32;
    // 1. Load Input (Async)
    {
      unsigned int gr = row_start + r;
      unsigned int gc_base = k_start + c_base;
      if (gr < M && gc_base + 4 <= K) {
        unsigned int idx =
          gr * groups_in_row * quantization_group_size + gc_base;
        cuda::memcpy_async(&s_input[r][c_base], &input[idx], sizeof(int), pipe);
      } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          unsigned int gc = gc_base + i;
          if (gr < M && gc < K) {
            unsigned int idx =
              gr * groups_in_row * quantization_group_size + gc;
            s_input[r][c_base + i] = input[idx];
          } else {
            s_input[r][c_base + i] = 0;
          }
        }
      }
    }
    pipe.producer_commit();
    // 2. Load Weights (Sync)
    {
      unsigned int gn = col_start + r;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        unsigned int gk = k_start + c_base + i;
        int8_t w_val = 0;
        if (gn < N && gk < K) {
          unsigned int w_idx =
            ((gn / 32) * (K / 2) + (gk / 2)) * 32 + (gn % 32);
          uint8_t packed = weights[w_idx];
          w_val = (gk % 2) ? (packed >> 4) : (packed & 0x0F);
          if (w_val >= 8)
            w_val -= 16;
        }
        s_weights[r][c_base + i] = w_val;
      }
    }

    pipe.consumer_wait();
    __syncthreads();
    // 3. WMMA Compute
    int warp_id = tid / 32;
    if (warp_id < 4) {
      nvcuda::wmma::fill_fragment(c_frag, 0);
      int wm = (warp_id / 2) * 16;
      int wn = (warp_id % 2) * 16;

      for (int k_step = 0; k_step < 32; k_step += 16) {
        nvcuda::wmma::load_matrix_sync(a_frag, &s_input[wm][k_step], 32);
        nvcuda::wmma::load_matrix_sync(b_frag, &s_weights[wn][k_step], 32);
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      }
      nvcuda::wmma::store_matrix_sync(&s_output[wm][wn], c_frag, 32,
                                      nvcuda::wmma::mem_row_major);
    }
    __syncthreads();
    // 4. Accumulate
    unsigned int rows[2] = {threadIdx.y, threadIdx.y + 16};
    unsigned int cols[2] = {threadIdx.x, threadIdx.x + 16};
    unsigned int gid_row = k_start / quantization_group_size;
    for (int r_idx = 0; r_idx < 2; ++r_idx) {
      unsigned int tr = rows[r_idx];
      unsigned int gr = row_start + tr;
      if (gr >= M)
        continue;
      float i_scale =
        __half2float(input_scales[(gr * groups_in_row + gid_row) * 2]);

      for (int c_idx = 0; c_idx < 2; ++c_idx) {
        unsigned int tc = cols[c_idx];
        unsigned int gc = col_start + tc;
        if (gc >= N)
          continue;

        float w_scale =
          __half2float(scales[gc * (K / quantization_group_size) + gid_row]);
        sum[r_idx * 2 + c_idx] += (float)s_output[tr][tc] * i_scale * w_scale;
      }
    }
    __syncthreads();
  }
  // 5. Store (Atomic Add used for Split-K)
  unsigned int rows[2] = {row_start + threadIdx.y,
                          row_start + threadIdx.y + 16};
  unsigned int cols[2] = {col_start + threadIdx.x,
                          col_start + threadIdx.x + 16};
  for (int r_idx = 0; r_idx < 2; ++r_idx) {
    if (rows[r_idx] >= M)
      continue;
    for (int c_idx = 0; c_idx < 2; ++c_idx) {
      if (cols[c_idx] >= N)
        continue;
      // Atomic Add for Split-K accumulation
      atomicAdd(&output[rows[r_idx] * N + cols[c_idx]], sum[r_idx * 2 + c_idx]);
    }
  }
}

void gemm_a8_w4_b16x16_s32x32_wmma_cpasync_splitk(
  const void *input, const void *weights, const void *scales,
  const void *input_scales, float *output, unsigned int M, unsigned int N,
  unsigned int K, unsigned int quantization_group_size) {

  // Example heuristic: usage of Split-K
  unsigned int split_k_slices = 1;
  // If Total Threads (Grid * Block) is small, use more splits to occupy GPU
  // Or if K is very large (latency hiding issue), use split-k.
  if (K >= 4096) {
    split_k_slices = 4;
  }
  if (K >= 8192) {
    split_k_slices = 8;
  }
  // Adjust split slices based on N size (if N is small, we need more splits)
  if (N <= 1024 && split_k_slices < 4)
    split_k_slices = 4;

  dim3 blockDim(16, 16);
  dim3 gridDim((N + 31) / 32, (M + 31) / 32, split_k_slices);

  // IMPORTANT: Output buffer MUST be zero-initialized because we use atomicAdd!
  // cudaMemsetAsync(output, 0, M * N * sizeof(float), stream);
  // For now, assuming caller handles initialization or we add it here if stream
  // available.

  __gemm_a8_w4_b16x16_s32x32_wmma_cpasync_splitk<<<gridDim, blockDim>>>(
    static_cast<const int8_t *>(input), static_cast<const uint8_t *>(weights),
    static_cast<const __half *>(scales),
    static_cast<const __half *>(input_scales), output, M, N, K,
    quantization_group_size, split_k_slices);
}

__global__ void __gemm_a8_w4_b8x16_s32x32_wmma_cpasync(
  const int8_t *input, const uint8_t *weights, const __half *scales,
  const __half *input_scales, float *output, unsigned int M, unsigned int N,
  unsigned int K, unsigned int quantization_group_size) {

  unsigned int tid = threadIdx.y * 16 + threadIdx.x;
  unsigned int row_start = blockIdx.y * 32;
  unsigned int col_start = blockIdx.x * 32;

  __shared__ int8_t s_input[32][32];
  __shared__ int8_t s_weights[32][32];
  __shared__ int32_t s_output[32][32];

  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, int8_t,
                         nvcuda::wmma::row_major>
    a_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, int8_t,
                         nvcuda::wmma::col_major>
    b_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, int32_t> c_frag;

  float sum[8] = {0.0f};

  const unsigned int groups_in_row =
    (K + quantization_group_size - 1) / quantization_group_size;

  for (unsigned int k_chunk = 0; k_chunk < (K + 31) / 32; ++k_chunk) {
    unsigned int k_start = k_chunk * 32;

// 1. Load Input (32x32 int8, 128 threads -> 2 ints per thread)
// Using cpasync
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      unsigned int load_idx = tid + i * 128;
      unsigned int r = load_idx / 8;
      unsigned int c_int = load_idx % 8;
      unsigned int c_base = c_int * 4;

      unsigned int gr = row_start + r;
      unsigned int gc_base = k_start + c_base;

      if (gr < M && gc_base + 4 <= K) {
        unsigned int idx =
          gr * groups_in_row * quantization_group_size + gc_base;
        cuda::memcpy_async(&s_input[r][c_base], &input[idx], sizeof(int), pipe);
      } else {
// Fallback or padding
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          unsigned int gc = gc_base + j;
          if (gr < M && gc < K) {
            unsigned int idx =
              gr * groups_in_row * quantization_group_size + gc;
            s_input[r][c_base + j] = input[idx];
          } else {
            s_input[r][c_base + j] = 0;
          }
        }
      }
    }
    pipe.producer_commit();

// 2. Load Weights (Same as VL)
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      unsigned int load_idx = tid + i * 128;
      unsigned int r = load_idx / 8;
      unsigned int c_base = (load_idx % 8) * 4;

      unsigned int gn = col_start + r;
      unsigned int gk_base = k_start + c_base;
      uint32_t val_packed = 0;

      if (gn < N && gk_base < K) {
        unsigned int w_idx_base =
          ((gn >> 5) * (K >> 1) + (gk_base >> 1)) * 32 + (gn & 31);
        uint8_t p0 = weights[w_idx_base];
        uint8_t p1 = 0;
        if (gk_base + 2 < K)
          p1 = weights[w_idx_base + 32];

        int8_t v0 = (int8_t)(p0 << 4) >> 4;
        int8_t v1 = (int8_t)p0 >> 4;
        int8_t v2 = (int8_t)(p1 << 4) >> 4;
        int8_t v3 = (int8_t)p1 >> 4;

        if (gk_base + 1 >= K)
          v1 = 0;
        val_packed = ((uint8_t)v0) | (((uint8_t)v1) << 8) |
                     (((uint8_t)v2) << 16) | (((uint8_t)v3) << 24);
      }
      *(int *)&s_weights[r][c_base] = val_packed;
    }

    // Wait for cpasync
    pipe.consumer_wait();
    __syncthreads();

    // 3. WMMA Compute
    int warp_id = tid / 32;
    nvcuda::wmma::fill_fragment(c_frag, 0);
    int wm = (warp_id / 2) * 16;
    int wn = (warp_id % 2) * 16;

    for (int k_step = 0; k_step < 32; k_step += 16) {
      nvcuda::wmma::load_matrix_sync(a_frag, &s_input[wm][k_step], 32);
      nvcuda::wmma::load_matrix_sync(b_frag, &s_weights[wn][k_step], 32);
      nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    nvcuda::wmma::store_matrix_sync(&s_output[wm][wn], c_frag, 32,
                                    nvcuda::wmma::mem_row_major);
    __syncthreads();

    // 4. Accumulate
    unsigned int rows[4] = {threadIdx.y, threadIdx.y + 8, threadIdx.y + 16,
                            threadIdx.y + 24};
    unsigned int cols[2] = {threadIdx.x, threadIdx.x + 16};
    unsigned int gid_row = k_start / quantization_group_size;

    for (int r_idx = 0; r_idx < 4; ++r_idx) {
      unsigned int tr = rows[r_idx];
      unsigned int gr = row_start + tr;
      if (gr >= M)
        continue;

      float i_scale =
        __half2float(input_scales[(gr * groups_in_row + gid_row) * 2]);

      for (int c_idx = 0; c_idx < 2; ++c_idx) {
        unsigned int tc = cols[c_idx];
        unsigned int gc = col_start + tc;
        if (gc >= N)
          continue;

        float w_scale =
          __half2float(scales[gc * (K / quantization_group_size) + gid_row]);
        sum[r_idx * 2 + c_idx] += (float)s_output[tr][tc] * i_scale * w_scale;
      }
    }
    __syncthreads();
  }

  // 5. Store
  unsigned int rows[4] = {row_start + threadIdx.y, row_start + threadIdx.y + 8,
                          row_start + threadIdx.y + 16,
                          row_start + threadIdx.y + 24};
  unsigned int cols[2] = {col_start + threadIdx.x,
                          col_start + threadIdx.x + 16};

  for (int r_idx = 0; r_idx < 4; ++r_idx) {
    if (rows[r_idx] >= M)
      continue;
    for (int c_idx = 0; c_idx < 2; ++c_idx) {
      if (cols[c_idx] >= N)
        continue;
      output[rows[r_idx] * N + cols[c_idx]] = sum[r_idx * 2 + c_idx];
    }
  }
}

void gemm_a8_w4_b8x16_s32x32_wmma_cpasync(
  const void *input, const void *weights, const void *scales,
  const void *input_scales, float *output, unsigned int M, unsigned int N,
  unsigned int K, unsigned int quantization_group_size) {
  dim3 blockDim(16, 8);
  dim3 gridDim((N + 31) / 32, (M + 31) / 32);
  __gemm_a8_w4_b8x16_s32x32_wmma_cpasync<<<gridDim, blockDim>>>(
    static_cast<const int8_t *>(input), static_cast<const uint8_t *>(weights),
    static_cast<const __half *>(scales),
    static_cast<const __half *>(input_scales), output, M, N, K,
    quantization_group_size);
}

__global__ void __gemm_a8_w4_b8x16_s32x32_wmma_cpasync_lu(
  const int8_t *input, const uint8_t *weights, const __half *scales,
  const __half *input_scales, float *output, unsigned int M, unsigned int N,
  unsigned int K, unsigned int quantization_group_size) {

  unsigned int tid = threadIdx.y * threadIdx.y * 16 + threadIdx.x;
  unsigned int row_start = blockIdx.y * 32;
  unsigned int col_start = blockIdx.x * 32;

  __shared__ int8_t s_input[32][32];
  __shared__ int8_t s_weights[32][32];
  __shared__ int32_t s_output[32][32];

  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, int8_t,
                         nvcuda::wmma::row_major>
    a_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, int8_t,
                         nvcuda::wmma::col_major>
    b_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, int32_t> c_frag;

  float sum[8] = {0.0f};

  const unsigned int groups_in_row =
    (K + quantization_group_size - 1) / quantization_group_size;

#pragma unroll
  for (unsigned int k_chunk = 0; k_chunk < (K + 31) / 32; ++k_chunk) {
    unsigned int k_start = k_chunk * 32;

// 1. Load Input (32x32 int8, 128 threads -> 2 ints per thread)
// Using cpasync
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      unsigned int load_idx = tid + i * 128;
      unsigned int r = load_idx / 8;
      unsigned int c_int = load_idx % 8;
      unsigned int c_base = c_int * 4;

      unsigned int gr = row_start + r;
      unsigned int gc_base = k_start + c_base;

      if (gr < M && gc_base + 4 <= K) {
        unsigned int idx =
          gr * groups_in_row * quantization_group_size + gc_base;
        cuda::memcpy_async(&s_input[r][c_base], &input[idx], sizeof(int), pipe);
      } else {
// Fallback or padding
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          unsigned int gc = gc_base + j;
          if (gr < M && gc < K) {
            unsigned int idx =
              gr * groups_in_row * quantization_group_size + gc;
            s_input[r][c_base + j] = input[idx];
          } else {
            s_input[r][c_base + j] = 0;
          }
        }
      }
    }
    pipe.producer_commit();

// 2. Load Weights (Same as VL)
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      unsigned int load_idx = tid + i * 128;
      unsigned int r = load_idx / 8;
      unsigned int c_base = (load_idx % 8) * 4;

      unsigned int gn = col_start + r;
      unsigned int gk_base = k_start + c_base;
      uint32_t val_packed = 0;

      if (gn < N && gk_base < K) {
        unsigned int w_idx_base =
          ((gn >> 5) * (K >> 1) + (gk_base >> 1)) * 32 + (gn & 31);
        uint8_t p0 = weights[w_idx_base];
        uint8_t p1 = 0;
        if (gk_base + 2 < K)
          p1 = weights[w_idx_base + 32];

        int8_t v0 = (int8_t)(p0 << 4) >> 4;
        int8_t v1 = (int8_t)p0 >> 4;
        int8_t v2 = (int8_t)(p1 << 4) >> 4;
        int8_t v3 = (int8_t)p1 >> 4;

        if (gk_base + 1 >= K)
          v1 = 0;
        val_packed = ((uint8_t)v0) | (((uint8_t)v1) << 8) |
                     (((uint8_t)v2) << 16) | (((uint8_t)v3) << 24);
      }
      *(int *)&s_weights[r][c_base] = val_packed;
    }

    // Wait for cpasync
    pipe.consumer_wait();
    __syncthreads();

    // 3. WMMA Compute
    int warp_id = tid / 32;
    nvcuda::wmma::fill_fragment(c_frag, 0);
    int wm = (warp_id / 2) * 16;
    int wn = (warp_id % 2) * 16;

    for (int k_step = 0; k_step < 32; k_step += 16) {
      nvcuda::wmma::load_matrix_sync(a_frag, &s_input[wm][k_step], 32);
      nvcuda::wmma::load_matrix_sync(b_frag, &s_weights[wn][k_step], 32);
      nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    nvcuda::wmma::store_matrix_sync(&s_output[wm][wn], c_frag, 32,
                                    nvcuda::wmma::mem_row_major);
    __syncthreads();

    // 4. Accumulate
    unsigned int rows[4] = {threadIdx.y, threadIdx.y + 8, threadIdx.y + 16,
                            threadIdx.y + 24};
    unsigned int cols[2] = {threadIdx.x, threadIdx.x + 16};
    unsigned int gid_row = k_start / quantization_group_size;

    for (int r_idx = 0; r_idx < 4; ++r_idx) {
      unsigned int tr = rows[r_idx];
      unsigned int gr = row_start + tr;
      if (gr >= M)
        continue;

      float i_scale =
        __half2float(input_scales[(gr * groups_in_row + gid_row) * 2]);

      for (int c_idx = 0; c_idx < 2; ++c_idx) {
        unsigned int tc = cols[c_idx];
        unsigned int gc = col_start + tc;
        if (gc >= N)
          continue;

        float w_scale =
          __half2float(scales[gc * (K / quantization_group_size) + gid_row]);
        sum[r_idx * 2 + c_idx] += (float)s_output[tr][tc] * i_scale * w_scale;
      }
    }
    __syncthreads();
  }

  // 5. Store
  unsigned int rows[4] = {row_start + threadIdx.y, row_start + threadIdx.y + 8,
                          row_start + threadIdx.y + 16,
                          row_start + threadIdx.y + 24};
  unsigned int cols[2] = {col_start + threadIdx.x,
                          col_start + threadIdx.x + 16};

  for (int r_idx = 0; r_idx < 4; ++r_idx) {
    if (rows[r_idx] >= M)
      continue;
    for (int c_idx = 0; c_idx < 2; ++c_idx) {
      if (cols[c_idx] >= N)
        continue;
      output[rows[r_idx] * N + cols[c_idx]] = sum[r_idx * 2 + c_idx];
    }
  }
}

void gemm_a8_w4_b8x16_s32x32_wmma_cpasync_lu(
  const void *input, const void *weights, const void *scales,
  const void *input_scales, float *output, unsigned int M, unsigned int N,
  unsigned int K, unsigned int quantization_group_size) {
  dim3 blockDim(16, 8);
  dim3 gridDim((N + 31) / 32, (M + 31) / 32);
  __gemm_a8_w4_b8x16_s32x32_wmma_cpasync_lu<<<gridDim, blockDim>>>(
    static_cast<const int8_t *>(input), static_cast<const uint8_t *>(weights),
    static_cast<const __half *>(scales),
    static_cast<const __half *>(input_scales), output, M, N, K,
    quantization_group_size);
}
