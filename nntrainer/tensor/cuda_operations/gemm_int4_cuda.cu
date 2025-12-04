#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>

#include "gemm_int4_cuda.h"
#include "ggml_quantize_cuda.h"

// Helper for ceil division
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#define ALIGN(a, b) (CEIL_DIV(a, b) * (b))

__global__ void gemm_int4_kernel_packed_block(
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
      unsigned int groups_in_row = alignK / quantization_group_size;

      unsigned int current_k = k_start + tx;
      unsigned int group_id_in_row = current_k / quantization_group_size;
      unsigned int global_group_id = row * groups_in_row + group_id_in_row;
      unsigned int offset_in_group = current_k % quantization_group_size;

      unsigned int input_idx =
        global_group_id * quantization_group_size + offset_in_group;

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
      unsigned int k_pair = k / 2;
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
    unsigned int group_id_in_row = k_start / quantization_group_size;
    unsigned int global_group_id = row * groups_in_row + group_id_in_row;
    float i_scale = __half2float(input_scales[global_group_id * 2]);

    // Weight scale
    unsigned int scale_idx =
      col * (K / quantization_group_size) + group_id_in_row;
    float w_scale = __half2float(scales[scale_idx]);

    sum += total_acc * i_scale * w_scale;

    __syncthreads();
  }

  if (row < M && col < N) {
    output[row * N + col] = sum;
  }
}

void gemm_int4_cuda_packed_block(const void *input, const void *weights,
                                 const void *scales, const void *input_scales,
                                 float *output, unsigned int M, unsigned int N,
                                 unsigned int K,
                                 unsigned int quantization_group_size) {

  // Launch Kernel
  dim3 blockDim(32, 32);
  dim3 gridDim((N + 31) / 32, (M + 31) / 32);

  gemm_int4_kernel_packed_block<<<gridDim, blockDim>>>(
    static_cast<const int8_t *>(input), static_cast<const uint8_t *>(weights),
    static_cast<const __half *>(scales),
    static_cast<const __half *>(input_scales), output, M, N, K,
    quantization_group_size);
}

__global__ void gemm_int4_kernel_packed_block_16(
  const int8_t *input, const uint8_t *weights, const __half *scales,
  const __half *input_scales, float *output, unsigned int M, unsigned int N,
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

void gemm_int4_cuda_packed_block_16(const void *input, const void *weights,
                                    const void *scales,
                                    const void *input_scales, float *output,
                                    unsigned int M, unsigned int N,
                                    unsigned int K,
                                    unsigned int quantization_group_size) {

  // Launch Kernel - each block computes 32x32 outputs
  dim3 blockDim(16, 16);
  dim3 gridDim((N + 31) / 32, (M + 31) / 32);

  gemm_int4_kernel_packed_block_16<<<gridDim, blockDim>>>(
    static_cast<const int8_t *>(input), static_cast<const uint8_t *>(weights),
    static_cast<const __half *>(scales),
    static_cast<const __half *>(input_scales), output, M, N, K,
    quantization_group_size);
}
